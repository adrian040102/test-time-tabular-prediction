# Portions of this file adapt preprocessing choices from Yandex Research's
# TabReD project. This repository translates and integrates those choices into
# its pandas/parquet pipeline. TabReD is licensed under Apache-2.0. See
# THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""
Shared helpers for the eight TabReD preprocessing scripts.

Each ``tabred_<name>.py`` script in this directory consumes raw Kaggle files
from ``data/tabred/raw/<dataset>/`` and emits:

    data/datasets/tabred_<name>/train.parquet
    data/datasets/tabred_<name>/test.parquet
    data/datasets/tabred_<name>/meta.json

The split convention matches the upstream Yandex repo's "default" split
(see github.com/yandex-research/tabred/tree/main/preprocessing): the
time-sorted data is partitioned by two temporal boundaries
``[val_start, test_start]``. The preprocessing excludes the validation window so the
remaining train and test halves carry the natural temporal gap. The
gap-widening sampler in ``src/data/tabred_sampling.py`` slices these
further at sweep time.

Categoricals get ordinal-encoded with ``min_frequency=1/100`` to match the
upstream output shape (this collapses the long high-cardinality tail into
a single "infrequent" bucket per column). Encoded codes are stored as
``int64`` and listed under ``meta["cat_features"]`` so LightGBM's native
categorical splits pick them up via the wrapper's
``categorical_feature=`` forward.

The time column is kept on disk under the canonical name ``__time__``
(``time_col`` in meta.json) so the sampler can re-sort at slice time
without re-running preprocessing.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Reserved column names that the universal loader expects.
TARGET_COL = "__target__"
TIME_COL = "__time__"

# Repo-relative roots.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "tabred" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "datasets"


@dataclass
class SplitSpec:
    """Two-boundary temporal split. Both boundaries are pandas-parseable.

    Convention (matches upstream):
        train = rows with __time__ < val_start
        val   = rows in [val_start, test_start)  (excluded from output)
        test  = rows with __time__ >= test_start
    """
    val_start: str
    test_start: str


def parse_args(
    dataset_short_name: str,
    default_raw_subdir: str,
) -> argparse.Namespace:
    """Common CLI for every per-dataset preprocessing script.

    Args:
        dataset_short_name: emitted as ``tabred_<short_name>`` under
            ``data/datasets/``.
        default_raw_subdir: relative subdir under ``data/tabred/raw/`` where
            the dataset's Kaggle files live (e.g. ``"sberbank"`` →
            ``data/tabred/raw/sberbank``).
    """
    p = argparse.ArgumentParser(description=f"TabReD preprocessor: {dataset_short_name}")
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_ROOT / default_raw_subdir,
        help=f"Raw Kaggle files (default: {DEFAULT_RAW_ROOT / default_raw_subdir}).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / f"tabred_{dataset_short_name}",
        help=f"Output folder (default: {DEFAULT_OUTPUT_ROOT / f'tabred_{dataset_short_name}'}).",
    )
    p.add_argument(
        "--subsample-frac",
        type=float,
        default=None,
        help=(
            "Override the upstream subsample fraction (each script ships "
            "with the per-dataset default from the upstream paper). "
            "Use this option only for diagnostic runs."
        ),
    )
    p.add_argument("--seed", type=int, default=0, help="Subsample / shuffle seed.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-emit output even if train.parquet already exists.",
    )
    return p.parse_args()


def require_raw_files(raw_dir: Path, expected: Iterable[str]) -> None:
    """Raise an informative error if any required raw file is missing.

    ``expected`` items are paths relative to ``raw_dir``. Use this at the
    top of every script so users get one consistent error.
    """
    raw_dir = Path(raw_dir)
    missing = [name for name in expected if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "TabReD raw inputs missing under "
            f"{raw_dir}:\n  " + "\n  ".join(missing)
            + "\n\nSee thesis/docs/tabred_kaggle_setup.md for download instructions."
        )


def unzip_if_needed(raw_dir: Path, archive_name: str) -> None:
    """Unzip ``archive_name`` (zip) into ``raw_dir`` if not already unpacked.

    Idempotent: skips when at least one non-archive file already exists in
    ``raw_dir``.

    Raises a clearer error than ``BadZipFile`` when the archive is partial,
    such as while a download is still in progress. This prevents a partially
    downloaded 4 GB zip from being treated as complete.
    """
    raw_dir = Path(raw_dir)
    archive = raw_dir / archive_name
    if not archive.exists():
        return
    # Heuristic for "already unzipped": at least one file besides zips/excels.
    sibling_data = [
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix not in {".zip", ".gz", ".xlsx"}
    ]
    if sibling_data:
        return
    if not zipfile.is_zipfile(archive):
        raise RuntimeError(
            f"{archive} is not a valid zip file: most likely a Kaggle "
            f"download is still in progress. Wait for it to finish (or "
            f"delete and re-download) before re-running this script."
        )
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)


def read_parquet_subsampled(
    path: Path,
    frac: float,
    seed: int,
    columns: list[str] | None = None,
    row_group_batch_rows: int = 200_000,
) -> pd.DataFrame:
    """Stream a parquet file via row-group batches and per-batch random subsample.

    Reading TabReD's largest parquets (delivery-eta is 11 GB, 17M rows × 223
    cols) into pandas via ``pd.read_parquet`` OOMs on a 32 GB machine because
    pyarrow loads the full column block before any subsampling. This
    helper iterates row groups in batches, samples within each batch, then
    concatenates only the kept rows. Memory peak is ``batch_rows × n_cols ×
    4 bytes`` rather than ``n_rows × n_cols × 4 bytes``.

    Args:
        path: parquet file to read.
        frac: independent per-row inclusion probability. Retained row count is
            approximate.
        seed: per-call RNG seed.
        columns: optional column subset (reduces both IO and memory).
        row_group_batch_rows: batch size for the underlying pyarrow iterator.

    Returns:
        A pandas DataFrame containing roughly ``frac * n_rows`` rows.
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    rng = np.random.default_rng(seed)
    kept: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=row_group_batch_rows, columns=columns):
        # Bernoulli mask reproducible from `seed`.
        n = batch.num_rows
        mask = rng.random(n) < frac
        if not mask.any():
            continue
        # Converting only the retained rows avoids loading rows that are then dropped.
        sub_batch = batch.filter(mask)
        kept.append(sub_batch.to_pandas())
    if not kept:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(kept, ignore_index=True)


def ordinal_encode(
    df: pd.DataFrame,
    cat_cols: list[str],
    min_frequency: float = 1 / 100,
) -> pd.DataFrame:
    """Replace ``cat_cols`` in ``df`` with sklearn-ordinal-encoded int64 codes.

    Uses the upstream call: ``OrdinalEncoder(min_frequency=1/100)``.
    Returns a copy of ``df`` with the same column order. NaN is encoded
    via ``encoded_missing_value=-1`` and bumped to a real code at the end
    so downstream LightGBM does not interpret -1 as a sentinel.
    """
    if not cat_cols:
        return df.copy()
    out = df.copy()
    arr = out[cat_cols].to_numpy()
    enc = OrdinalEncoder(
        min_frequency=min_frequency,
        encoded_missing_value=-1,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoded = enc.fit_transform(arr)
    # Lift -1 (missing/unknown) to max+1 per column so codes are non-negative.
    encoded = encoded.copy()
    for c in range(encoded.shape[1]):
        col = encoded[:, c]
        col_max = col[col != -1].max() if (col != -1).any() else -1
        col[col == -1] = col_max + 1
        encoded[:, c] = col
    out[cat_cols] = encoded.astype(np.int64)
    return out


def write_tabred_dataset(
    *,
    out_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    task_type: str,
    cat_features: list[str],
    num_features: list[str],
    shift_description: str,
    extra: dict | None = None,
) -> dict:
    """Write the three artifacts and return the meta dict (for tests / logs).

    Both ``train`` and ``test`` must already contain ``TARGET_COL`` and
    ``TIME_COL`` plus all ``cat_features`` + ``num_features`` columns.

    The split is assumed to be:
        * ``train[TIME_COL].max()`` < ``test[TIME_COL].min()``
    (i.e. the val window has already been dropped).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for required in (TARGET_COL, TIME_COL):
        for label, df in (("train", train), ("test", test)):
            if required not in df.columns:
                raise ValueError(f"{label}.parquet missing column {required!r}")

    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Refusing to write empty split: |train|={len(train)} |test|={len(test)}"
        )

    if train[TIME_COL].max() >= test[TIME_COL].min():
        raise ValueError(
            "Train/test time windows overlap because the validation gap was not removed. "
            f"Max(train) = {train[TIME_COL].max()} >= min(test) = {test[TIME_COL].min()}"
        )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    name = out_dir.name  # canonical
    meta = {
        "name": name,
        "task_type": task_type,
        "cat_features": cat_features,
        "num_features": num_features,
        "time_col": TIME_COL,
        "shift_description": shift_description,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": len(cat_features) + len(num_features),
        "extra": (extra or {}) | {
            "source": "tabred",
            "tabred_split": "default_natural_gap",
            "train_time_start": _ts_to_iso(train[TIME_COL].min()),
            "train_time_end": _ts_to_iso(train[TIME_COL].max()),
            "test_time_start": _ts_to_iso(test[TIME_COL].min()),
            "test_time_end": _ts_to_iso(test[TIME_COL].max()),
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _ts_to_iso(v) -> str:
    """Coerce numpy datetime / pandas Timestamp / numeric to a JSON-safe string."""
    if pd.isna(v):
        return ""
    try:
        return pd.Timestamp(v).isoformat()
    except (ValueError, TypeError):
        return str(v)


def default_temporal_split(
    df: pd.DataFrame,
    spec: SplitSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice ``df`` (already sorted by TIME_COL) into (train, test) per ``spec``.

    Val rows in ``[val_start, test_start)`` are dropped entirely.
    """
    val_start = pd.Timestamp(spec.val_start)
    test_start = pd.Timestamp(spec.test_start)
    if not df[TIME_COL].is_monotonic_increasing:
        df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    train_mask = df[TIME_COL] < val_start
    test_mask = df[TIME_COL] >= test_start
    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def short_summary(name: str, train: pd.DataFrame, test: pd.DataFrame) -> str:
    return (
        f"  {name}: train={len(train):,}  test={len(test):,}  "
        f"feats={train.shape[1] - 2}  "
        f"gap=[{train[TIME_COL].max()}, {test[TIME_COL].min()}]"
    )

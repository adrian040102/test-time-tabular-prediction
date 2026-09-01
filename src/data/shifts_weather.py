"""
Yandex Shifts Dataset: Weather Prediction (Tabular Track).

This dataset was created for the NeurIPS Shifts Challenge and contains ~10M
weather station records from September 2018 to September 2019. The tabular
features include meteorological measurements and forecast model outputs.

Distribution shift:
  - Temporal: train on Sep 2018 – Apr 2019, evaluate on Jul – Aug 2019.
  - Geographic: different climate zones and station locations.

Target: air temperature at 2m (regression).

Source: https://github.com/yandex-research/shifts
License: CC BY-NC-SA 4.0 (research use only)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.base import TabularDataset


# Canonical Shifts dataset archive URL (individual CSV URLs are no longer available).
# The tar contains train.csv, dev_in.csv, dev_out.csv, eval_in.csv, eval_out.csv.
_SHIFTS_WEATHER_TAR_URL = (
    "https://storage.yandexcloud.net/yandex-research/shifts/weather/canonical-partitioned-dataset.tar"
)

_VALID_SPLITS = {"train", "dev_in", "dev_out", "eval_in", "eval_out"}

# Metadata columns that should not be used as features.
_META_COLS = {"time", "latitude", "longitude", "climate_type"}
_TARGET_COL = "fact_temperature"

# Columns that are categorical by nature
_CAT_COLS = {"climate_type"}


def _download_and_extract(cache_dir: Path) -> None:
    """Download the canonical tar archive and extract all CSV splits."""
    import tarfile
    import urllib.request

    tar_path = cache_dir / "canonical-partitioned-dataset.tar"
    if not tar_path.exists():
        print(f"Downloading Shifts Weather archive to {tar_path} ...")
        print("  (This may take a while: the archive is ~4 GB.)")
        urllib.request.urlretrieve(_SHIFTS_WEATHER_TAR_URL, str(tar_path))

    print("Extracting CSV splits from archive ...")
    with tarfile.open(str(tar_path), "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".csv"):
                # Strip any directory prefix so CSVs land directly in cache_dir
                member.name = Path(member.name).name
                tf.extract(member, path=str(cache_dir))


def _get_split_path(split: str, cache_dir: Path) -> Path:
    """Return the path to a CSV split, downloading if necessary."""
    if split not in _VALID_SPLITS:
        raise ValueError(f"Unknown split '{split}'. Available: {sorted(_VALID_SPLITS)}")

    # The archive may use either "{split}.csv" or "shifts_canonical_{split}.csv"
    csv_path = cache_dir / f"{split}.csv"
    csv_path_alt = cache_dir / f"shifts_canonical_{split}.csv"

    if csv_path.exists():
        return csv_path
    if csv_path_alt.exists():
        return csv_path_alt

    # Need to download and extract the archive
    _download_and_extract(cache_dir)

    if csv_path.exists():
        return csv_path
    if csv_path_alt.exists():
        return csv_path_alt

    raise FileNotFoundError(
        f"Split '{split}' not found after extraction. "
        f"Files in cache: {sorted(p.name for p in cache_dir.glob('*.csv'))}"
    )


def load_shifts_weather(
    train_split: str = "dev_in",
    test_split: str = "dev_out",
    max_samples: int | None = 50_000,
    seed: int = 42,
    cache_dir: str | None = None,
    drop_meta: bool = True,
) -> TabularDataset:
    """
    Load the Yandex Shifts Weather dataset with temporal/geographic shift.

    By default uses the smaller dev splits (dev_in / dev_out, ~2M rows each)
    which are manageable without a cluster. For full-scale experiments, use
    ``train_split="train"`` and ``test_split="eval_out"``.

    Available splits:
        - ``train``:     Full training set (~9M rows, in-distribution)
        - ``dev_in``:    Development in-distribution (~2M rows)
        - ``dev_out``:   Development OOD: temporal & geographic shift (~2M rows)
        - ``eval_in``:   Evaluation in-distribution
        - ``eval_out``:  Evaluation OOD

    Args:
        train_split: Which split to use for training.
        test_split: Which split to use for testing (should be an OOD split).
        max_samples: Cap on samples per split. Defaults to 50,000 to bound memory use.
            Set to None for full data.
        seed: Random seed for subsampling.
        cache_dir: Optional cache directory.
        drop_meta: Whether to drop metadata columns (time, lat, lon, climate_type)
            from features. Set to False to keep them.

    Returns:
        TabularDataset (task_type="regression", target=fact_temperature).
    """
    if cache_dir is None:
        base = os.environ.get("THESIS_DATA_CACHE",
                              os.path.join(os.path.expanduser("~"), ".cache", "thesis_data"))
        cache_dir = os.path.join(base, "shifts_weather")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    train_path = _get_split_path(train_split, cache_path)
    test_path = _get_split_path(test_split, cache_path)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Separate target
    if _TARGET_COL not in df_train.columns:
        raise KeyError(
            f"Target column '{_TARGET_COL}' not found. "
            f"Available columns: {list(df_train.columns[:20])}..."
        )

    y_train = df_train.pop(_TARGET_COL).values.astype(float)
    y_test = df_test.pop(_TARGET_COL).values.astype(float)

    # Drop metadata columns if requested
    drop_cols = set()
    if drop_meta:
        drop_cols = _META_COLS & set(df_train.columns)
        df_train = df_train.drop(columns=list(drop_cols), errors="ignore")
        df_test = df_test.drop(columns=list(drop_cols), errors="ignore")

    # Subsample (these datasets are huge)
    if max_samples:
        df_train, y_train = _subsample(df_train, y_train, max_samples, seed)
        df_test, y_test = _subsample(df_test, y_test, max_samples, seed + 1)

    # Auto-detect feature types
    cat_features = df_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_features = df_train.select_dtypes(include=["number"]).columns.tolist()

    shift_desc = f"temporal+geographic: {train_split}→{test_split}"

    return TabularDataset(
        name=f"shifts_weather_{train_split}_{test_split}",
        X_train=df_train.reset_index(drop=True),
        X_test=df_test.reset_index(drop=True),
        y_train=y_train,
        y_test=y_test,
        cat_features=cat_features,
        num_features=num_features,
        task_type="regression",
        shift_description=shift_desc,
        metadata={
            "source": "yandex_shifts",
            "task": "weather_temperature_prediction",
            "train_split": train_split,
            "test_split": test_split,
            "target": _TARGET_COL,
            "license": "CC BY-NC-SA 4.0",
        },
    )


def _subsample(
    X: pd.DataFrame, y: np.ndarray, n: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(X) <= n:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X.iloc[idx].reset_index(drop=True), y[idx]

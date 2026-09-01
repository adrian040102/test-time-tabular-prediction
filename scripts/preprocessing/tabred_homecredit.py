# Portions of this file adapt preprocessing choices from Yandex Research's
# TabReD project. This repository translates and integrates those choices into
# its pandas/parquet pipeline. TabReD is licensed under Apache-2.0. See
# THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""
TabReD HomeCredit Default Stability: binary classification.

Upstream reference: github.com/yandex-research/tabred/blob/main/preprocessing/homecredit.py
Kaggle source:      home-credit-credit-risk-model-stability  (competition)

Raw files expected at ``--raw-dir`` (default: ``data/tabred/raw/homecredit/``):
    parquet_files/train/train_base.parquet
    parquet_files/train/train_static_0_*.parquet
    parquet_files/train/train_static_cb_0.parquet
    parquet_files/train/train_applprev_1_*.parquet
    parquet_files/train/train_tax_registry_a_1.parquet
    parquet_files/train/train_tax_registry_b_1.parquet
    parquet_files/train/train_tax_registry_c_1.parquet
    parquet_files/train/train_credit_bureau_a_1_*.parquet
    parquet_files/train/train_credit_bureau_b_1.parquet
    parquet_files/train/train_other_1.parquet
    parquet_files/train/train_person_1.parquet
    parquet_files/train/train_deposit_1.parquet
    parquet_files/train/train_debitcard_1.parquet
    parquet_files/train/train_credit_bureau_a_2_*.parquet
    parquet_files/train/train_credit_bureau_b_2.parquet

Upstream's default split (preserved):
    train < 2020-01-01
    val   [2020-01-01, 2020-05-01)
    test  >= 2020-05-01

Subsample fraction: 0.25 (matches upstream).

The implementation uses the upstream aggregation families: max, min, mean and
standard deviation for numeric/date columns and last, unique count and first
for string columns. Dates are converted to day offsets from ``date_decision``.
After joining all aggregates onto ``train_base`` by ``case_id``, columns with
>95% nulls, single-unique-value or string dtype with >50 uniques are dropped.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    SplitSpec,
    TARGET_COL,
    TIME_COL,
    default_temporal_split,
    ordinal_encode,
    parse_args,
    require_raw_files,
    short_summary,
    write_tabred_dataset,
)


SUBSAMPLE_FRAC = 0.25
SPLIT = SplitSpec(val_start="2020-01-01", test_start="2020-05-01")


def _read_concat(raw_dir: Path, pattern: str) -> pd.DataFrame:
    """Concatenate the parquet files selected by ``pattern`` relative to raw_dir."""
    paths = sorted(Path(raw_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files found at {raw_dir / pattern}. "
            f"See thesis/docs/tabred_kaggle_setup.md."
        )
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def _aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group by case_id and compute the upstream-canonical agg suite.

    The upstream uses polars expressions:
        * numeric P/A/T/L cols: max, min, mean, std
        * string M cols: last, n_unique, first
        * date D cols: max, min, mean, std (after converting to day-deltas
          vs date_decision: done at the caller side, since this helper
          operates on already-day-delta columns)
    """
    if "case_id" not in df.columns:
        return pd.DataFrame()

    df = df.sort_values("num_group1" if "num_group1" in df.columns else "case_id", kind="stable")

    num_cols = [c for c in df.columns if c[-1] in ("P", "A", "T", "L")]
    str_cols = [c for c in df.columns if c[-1] == "M"]
    date_cols = [c for c in df.columns if c[-1] == "D"]

    grouped = df.groupby("case_id")
    pieces: list[pd.DataFrame] = []
    if num_cols:
        agg = grouped[num_cols].agg(["max", "min", "mean", "std"])
        agg.columns = [f"{stat}_{col}" for col, stat in agg.columns]
        pieces.append(agg)
    if str_cols:
        # last / first / n_unique on strings
        d_last = grouped[str_cols].last().add_prefix("last_")
        d_first = grouped[str_cols].first().add_prefix("first_")
        d_nu = grouped[str_cols].nunique().add_prefix("n_unique_")
        pieces.extend([d_last, d_first, d_nu])
    if date_cols:
        agg_d = grouped[date_cols].agg(["max", "min", "mean", "std"])
        agg_d.columns = [f"{stat}_{col}" for col, stat in agg_d.columns]
        pieces.append(agg_d)

    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, axis=1).reset_index()
    return out


def main() -> None:
    args = parse_args("homecredit", "homecredit")
    if (args.output_dir / "train.parquet").exists() and not args.overwrite:
        print(f"SKIP: {args.output_dir / 'train.parquet'} exists (pass --overwrite to redo).")
        return

    base_pat = "parquet_files/train/train_base.parquet"
    require_raw_files(args.raw_dir, [base_pat])

    base = _read_concat(args.raw_dir, base_pat)
    static = _read_concat(args.raw_dir, "parquet_files/train/train_static_0_*.parquet")
    static_cb = _read_concat(args.raw_dir, "parquet_files/train/train_static_cb_0.parquet")

    # All side tables that need aggregation.
    side_patterns = [
        "parquet_files/train/train_applprev_1_*.parquet",
        "parquet_files/train/train_tax_registry_a_1.parquet",
        "parquet_files/train/train_tax_registry_b_1.parquet",
        "parquet_files/train/train_tax_registry_c_1.parquet",
        "parquet_files/train/train_credit_bureau_a_1_*.parquet",
        "parquet_files/train/train_credit_bureau_b_1.parquet",
        "parquet_files/train/train_other_1.parquet",
        "parquet_files/train/train_person_1.parquet",
        "parquet_files/train/train_deposit_1.parquet",
        "parquet_files/train/train_debitcard_1.parquet",
        "parquet_files/train/train_credit_bureau_a_2_*.parquet",
        "parquet_files/train/train_credit_bureau_b_2.parquet",
    ]
    aggregated: list[pd.DataFrame] = []
    for pat in side_patterns:
        files = sorted(Path(args.raw_dir).glob(pat))
        if not files:
            print(f"  WARN: no files for {pat}: skipping")
            continue
        side = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True, sort=False)
        aggregated.append(_aggregate_features(side))

    data = base.copy()
    for i, side in enumerate([static, static_cb] + aggregated):
        data = data.merge(side, on="case_id", how="left", suffixes=("", f"_{i}"))

    # Convert any D-suffixed date col to a day-delta vs date_decision.
    data["date_decision"] = pd.to_datetime(data["date_decision"], errors="coerce")
    for c in data.columns:
        if c.endswith("D") and c != "date_decision":
            data[c] = pd.to_datetime(data[c], errors="coerce")
            data[c] = (data[c] - data["date_decision"]).dt.days.astype("float32")

    # Drop columns with >95% nulls, single-unique or string with >50 uniques.
    null_frac = data.isna().mean()
    n_unique = data.nunique(dropna=True)
    drop = []
    for c in data.columns:
        if null_frac[c] > 0.95:
            drop.append(c)
        elif n_unique[c] == 1:
            drop.append(c)
        elif data[c].dtype == object and n_unique[c] > 50:
            drop.append(c)
    data = data.drop(columns=drop)
    for hardcoded in ("min_sex_738L", "MONTH", "WEEK_NUM"):
        if hardcoded in data.columns:
            data = data.drop(columns=[hardcoded])

    # Upstream-derived bin cols.
    bin_cols: list[str] = []
    if "isbidproduct_1095L" in data.columns:
        bin_cols.append("isbidproduct_1095L")
    if "max_sex_738L" in data.columns:
        data["max_sex_738L"] = (data["max_sex_738L"] == "F").astype("float32")
        bin_cols.append("max_sex_738L")

    str_cat_cols = [
        c for c in data.columns
        if data[c].dtype == object
        and c not in {"target", "case_id", "date_decision"} | set(bin_cols)
    ]
    data = ordinal_encode(data, str_cat_cols)

    num_cols = [
        c for c in data.columns
        if pd.api.types.is_numeric_dtype(data[c])
        and c not in {"target", "case_id"} | set(bin_cols) | set(str_cat_cols)
    ]
    for c in num_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce").astype("float32")

    # Time features.
    data["day_of_week"] = data["date_decision"].dt.weekday
    data["day_of_month"] = data["date_decision"].dt.day
    data["day_of_year"] = data["date_decision"].dt.dayofyear
    num_cols = num_cols + ["day_of_week", "day_of_month", "day_of_year"]

    subsample_frac = args.subsample_frac if args.subsample_frac is not None else SUBSAMPLE_FRAC
    if subsample_frac < 1.0:
        data = data.sample(frac=subsample_frac, random_state=args.seed).reset_index(drop=True)

    data = data.sort_values("date_decision", kind="stable").reset_index(drop=True)
    data[TIME_COL] = data["date_decision"]
    data[TARGET_COL] = data["target"].astype("int64")

    num_features = num_cols + bin_cols
    cat_features = str_cat_cols

    keep = num_features + cat_features + [TARGET_COL, TIME_COL]
    train, test = default_temporal_split(data[keep], SPLIT)

    write_tabred_dataset(
        out_dir=args.output_dir,
        train=train,
        test=test,
        task_type="classification",
        cat_features=cat_features,
        num_features=num_features,
        shift_description=f"TabReD homecredit natural gap (val window {SPLIT.val_start}..{SPLIT.test_start} dropped)",
        extra={
            "target_raw": "target",
            "n_classes": 2,
            "subsample_frac": subsample_frac,
            "subsample_seed": args.seed,
            "upstream_split": "default",
            "side_patterns": side_patterns,
        },
    )
    print("Wrote tabred_homecredit:")
    print(short_summary("homecredit", train, test))


if __name__ == "__main__":
    main()

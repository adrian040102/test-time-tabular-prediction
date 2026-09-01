# Portions of this file adapt preprocessing choices from Yandex Research's
# TabReD project. This repository translates and integrates those choices into
# its pandas/parquet pipeline. TabReD is licensed under Apache-2.0. See
# THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""
TabReD Homesite Insurance: binary classification.

Upstream reference: github.com/yandex-research/tabred/blob/main/preprocessing/homesite.py
Kaggle source:      homesite-quote-conversion  (competition)

Raw file expected at ``--raw-dir`` (default: ``data/tabred/raw/homesite/``):
    train.csv.zip  (Kaggle's competition train.csv archive)
or the unzipped equivalent (the script unzips automatically).

Upstream's default split (preserved):
    train < 2015-02-01
    val   2015-02-01..2015-03-31  (months 2,3 of 2015)
    test  2015-04-01..2015-05-31  (months 4,5 of 2015)

Schema changes consistent with the upstream implementation:
    * Drop ``QuoteNumber`` and ``Original_Quote_Date`` (replaced with derived
      timestamp features).
    * Replace -1 with NaN in ordinal cols ending with A/B.
    * Drop all-null columns.
    * Infer bin/cat/num from n_unique + dtype.
    * N/Y string-bin cols are recoded to 0/1.
    * String categoricals are ordinal-encoded with ``min_frequency=1/100``.

Target: ``QuoteConversion_Flag``.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
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
    unzip_if_needed,
    write_tabred_dataset,
)


SPLIT = SplitSpec(val_start="2015-02-01", test_start="2015-04-01")
TEST_END = "2015-06-01"  # upstream filters by months 4,5 of 2015
TARGET_RAW = "QuoteConversion_Flag"


def _read_homesite_csv(raw_dir: Path) -> pd.DataFrame:
    csv_path = raw_dir / "train.csv"
    if csv_path.is_file():
        return pd.read_csv(csv_path, low_memory=False)
    zip_path = raw_dir / "train.csv.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Neither {csv_path} nor {zip_path} present under {raw_dir}. "
            f"See thesis/docs/tabred_kaggle_setup.md."
        )
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("train.csv") as f:
            return pd.read_csv(f, low_memory=False)


def main() -> None:
    args = parse_args("homesite", "homesite")
    if (args.output_dir / "train.parquet").exists() and not args.overwrite:
        print(f"SKIP: {args.output_dir / 'train.parquet'} exists (pass --overwrite to redo).")
        return

    # Kaggle's `competitions download -c homesite-quote-conversion` produces a
    # single outer archive `homesite-quote-conversion.zip` containing
    # train.csv.zip + test.csv.zip + sample_submission.csv. Unzip the outer
    # once so train.csv.zip is directly under raw_dir.
    unzip_if_needed(args.raw_dir, "homesite-quote-conversion.zip")
    require_raw_files(args.raw_dir, ["train.csv.zip"])
    df = _read_homesite_csv(args.raw_dir)

    df["timestamp"] = pd.to_datetime(df["Original_Quote_Date"], errors="coerce")
    df = df.drop(columns=["QuoteNumber", "Original_Quote_Date"])

    # -1 → NaN in ordinal A/B columns.
    ab_cols = [c for c in df.columns if c.endswith(("A", "B"))]
    for c in ab_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            df.loc[df[c] == -1, c] = np.nan

    # Time features.
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["day_of_month"] = df["timestamp"].dt.day
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month
    derived_time_cols = ["day_of_week", "day_of_month", "day_of_year", "month"]

    # Drop entirely-null columns.
    all_null = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=all_null)

    target_columns = [TARGET_RAW]
    meta_columns = ["timestamp"]

    # n_unique per column (excluding nulls)
    nunq = df.apply(lambda s: s.dropna().nunique())

    bin_cols = [
        c for c in df.columns
        if c not in target_columns + meta_columns
        and nunq[c] == 2
    ]
    # String bin cols N/Y → 0/1. Numeric stays as-is.
    for c in bin_cols:
        if df[c].dtype == object:
            df[c] = df[c].map({"N": 0, "Y": 1}).astype("float32")
        else:
            df[c] = df[c].astype("float32")

    str_cat_cols = [
        c for c in df.columns
        if c not in target_columns + meta_columns + bin_cols
        and df[c].dtype == object
    ]
    # Ordinal-encode string categoricals directly. This port omits the upstream
    # intermediate dense-ranking step.
    df = ordinal_encode(df, str_cat_cols)

    num_cols = [
        c for c in df.columns
        if c not in target_columns + meta_columns + bin_cols + str_cat_cols + [TIME_COL]
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    df[TIME_COL] = df["timestamp"]
    df[TARGET_COL] = df[TARGET_RAW].astype("int64")

    num_features = num_cols + bin_cols  # The upstream implementation stores bins separately.
    cat_features = str_cat_cols

    keep = num_features + cat_features + [TARGET_COL, TIME_COL]
    train, test = default_temporal_split(df[keep], SPLIT)
    test = test[test[TIME_COL] < pd.Timestamp(TEST_END)].copy()

    write_tabred_dataset(
        out_dir=args.output_dir,
        train=train,
        test=test,
        task_type="classification",
        cat_features=cat_features,
        num_features=num_features,
        shift_description=f"TabReD homesite natural gap (val months {SPLIT.val_start}..{SPLIT.test_start} dropped. Test = months 4-5 of 2015)",
        extra={
            "target_raw": TARGET_RAW,
            "n_classes": 2,
            "upstream_split": "default",
            "all_null_cols_dropped": all_null,
            "test_end": TEST_END,
        },
    )
    print("Wrote tabred_homesite:")
    print(short_summary("homesite", train, test))


if __name__ == "__main__":
    main()

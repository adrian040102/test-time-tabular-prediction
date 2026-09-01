# Portions of this file adapt preprocessing choices from Yandex Research's
# TabReD project. This repository translates and integrates those choices into
# its pandas/parquet pipeline. TabReD is licensed under Apache-2.0. See
# THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""
TabReD Sberbank Housing: Russian residential property regression.

Upstream reference: github.com/yandex-research/tabred/blob/main/preprocessing/sberbank_housing.py
Kaggle source:      sberbank-russian-housing-market  (competition)

Raw files expected at ``--raw-dir`` (default: ``data/tabred/raw/sberbank/``):
    train.csv.zip
    macro.csv.zip
    BAD_ADDRESS_FIX.xlsx

Upstream's default split (preserved):
    train < 2014-06-30
    val   [2014-06-30, 2014-12-01)
    test  >= 2014-12-01

Cleaning (matches upstream):
    * Drop ``provision_retail_space_modern_sqm`` from macro (constant + nulls).
    * Apply the BAD_ADDRESS_FIX.xlsx Tverskoe overwrite (~600 rows have a
      corrupted ``kremlin_km==min`` cluster).
    * Drop the round-number anomalies excluded by the upstream preprocessing.
    * Join macro on ``timestamp``.
    * Time features: weekday, day, doy, month, year.

Schema:
    Cats = hard-coded list (railroad IDs, road IDs, ecology, material, state,
        sub_area), ordinal-encoded with ``min_frequency=1/100``.
    Bins = any col with exactly 2 unique values, dense-ranked to 0/1.
    Nums = everything else (excluding price_doc / id / timestamp / cats / bins).
    Target = ``log(price_doc / full_sq)``  (price per sqm, logged).
"""

from __future__ import annotations

import zipfile
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


SPLIT = SplitSpec(val_start="2014-06-30", test_start="2014-12-01")
CAT_COLS = [
    "ID_railroad_station_walk",
    "ID_railroad_station_avto",
    "ID_big_road1",
    "ID_big_road2",
    "ID_railroad_terminal",
    "ID_bus_terminal",
    "ecology",
    "material",
    "state",
    "sub_area",
]


def _read_csv_zip(raw_dir: Path, archive: str, member: str) -> pd.DataFrame:
    path = raw_dir / archive
    with zipfile.ZipFile(path) as zf:
        with zf.open(member) as f:
            return pd.read_csv(f, na_values=["NA"], low_memory=False)


def main() -> None:
    args = parse_args("sberbank", "sberbank")
    if (args.output_dir / "train.parquet").exists() and not args.overwrite:
        print(f"SKIP: {args.output_dir / 'train.parquet'} exists (pass --overwrite to redo).")
        return

    # Kaggle's `competitions download -c sberbank-russian-housing-market` produces
    # a single outer archive `sberbank-russian-housing-market.zip` that contains
    # train.csv.zip, macro.csv.zip, etc. Unzip the outer once so the inner zips
    # are siblings of BAD_ADDRESS_FIX.xlsx.
    unzip_if_needed(args.raw_dir, "sberbank-russian-housing-market.zip")
    require_raw_files(args.raw_dir, ["train.csv.zip", "macro.csv.zip", "BAD_ADDRESS_FIX.xlsx"])

    train_df = _read_csv_zip(args.raw_dir, "train.csv.zip", "train.csv")
    macro_df = _read_csv_zip(args.raw_dir, "macro.csv.zip", "macro.csv")
    fix_df = pd.read_excel(args.raw_dir / "BAD_ADDRESS_FIX.xlsx", na_values=["NA"])

    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], errors="coerce")
    macro_df["timestamp"] = pd.to_datetime(macro_df["timestamp"], errors="coerce")

    # Sanitize macro columns that ship as strings with comma decimal sep.
    for c in ("child_on_acc_pre_school", "modern_education_share", "old_education_build_share"):
        if c in macro_df.columns and macro_df[c].dtype == object:
            macro_df[c] = (
                macro_df[c].astype(str).str.replace(",", ".", regex=False)
            )
            macro_df[c] = pd.to_numeric(macro_df[c], errors="coerce").astype("float32")

    if "provision_retail_space_modern_sqm" in macro_df.columns:
        macro_df = macro_df.drop(columns=["provision_retail_space_modern_sqm"])

    # Apply the Tverskoe address correction. Rows where kremlin_km equals its
    # minimum are corrupt.
    # The upstream filter excludes the corrupt cluster, then corrects
    # the few that have an explicit fixup row.
    if "kremlin_km" in train_df.columns:
        anomalous_km_value = train_df["kremlin_km"].min()
        keep = (train_df["kremlin_km"] != anomalous_km_value) | train_df["id"].isin(fix_df["id"])
        train_df = train_df.loc[keep].copy()
    # Update via fix_df on id where columns overlap.
    overlap = [c for c in fix_df.columns if c in train_df.columns and c != "id"]
    if overlap:
        train_df = train_df.set_index("id")
        fix_indexed = fix_df.set_index("id")[overlap]
        train_df.update(fix_indexed)
        train_df = train_df.reset_index()

    # Outlier filtering (matches upstream).
    train_df = train_df[
        (train_df["full_sq"] > 5.0)
        & (train_df["full_sq"] != 5326.0)
        & (train_df["price_doc"] > 1_000_000)
        & (~train_df["price_doc"].isin([2_000_000, 3_000_000]))
    ].copy()

    df = train_df.merge(macro_df, on="timestamp", how="inner")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    df["day_of_week"] = df["timestamp"].dt.weekday
    df["day_of_month"] = df["timestamp"].dt.day
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year

    # Encode cats.
    cat_cols = [c for c in CAT_COLS if c in df.columns]
    df = ordinal_encode(df, cat_cols)

    # Binary columns = anything with exactly 2 unique non-null values (excl. cats).
    bin_cols: list[str] = []
    for c in df.columns:
        if c in cat_cols + ["price_doc", "id", "timestamp"]:
            continue
        nu = df[c].dropna().nunique()
        if nu == 2:
            bin_cols.append(c)
    # Dense-rank bin cols to 0/1 (upstream subtracts 1 from rank('dense')).
    for c in bin_cols:
        vals = df[c].astype("category").cat.codes
        df[c] = vals.astype("float32")
        df.loc[df[c] == -1, c] = np.nan  # preserve missingness

    num_cols = [
        c for c in df.columns
        if c not in {"price_doc", "id", "timestamp"} | set(cat_cols) | set(bin_cols)
    ]
    # Cast num cols to float32. Keep NaN.
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    df[TIME_COL] = df["timestamp"]
    df[TARGET_COL] = np.log(
        df["price_doc"].astype("float64") / df["full_sq"].astype("float64")
    ).astype("float32")

    num_features = num_cols + bin_cols
    cat_features = cat_cols

    keep = num_features + cat_features + [TARGET_COL, TIME_COL]
    train, test = default_temporal_split(df[keep], SPLIT)

    write_tabred_dataset(
        out_dir=args.output_dir,
        train=train,
        test=test,
        task_type="regression",
        cat_features=cat_features,
        num_features=num_features,
        shift_description=f"TabReD sberbank natural gap (val window {SPLIT.val_start}..{SPLIT.test_start} dropped)",
        extra={
            "target_raw": "log(price_doc / full_sq)",
            "upstream_split": "default",
            "address_fixup_applied": True,
            "outlier_filter": (
                "full_sq > 5, full_sq != 5326, price_doc > 1M and "
                "price_doc not in {2M, 3M}"
            ),
        },
    )
    print("Wrote tabred_sberbank:")
    print(short_summary("sberbank", train, test))


if __name__ == "__main__":
    main()

# Portions of this file adapt preprocessing choices from Yandex Research's
# TabReD project. This repository translates and integrates those choices into
# its pandas/parquet pipeline. TabReD is licensed under Apache-2.0. See
# THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""
TabReD Delivery ETA: courier-delivery time prediction (regression).

Upstream reference: github.com/yandex-research/tabred/blob/main/preprocessing/delivery_eta.py
Kaggle source:      pcovkrd84mejm/delivery-eta  (CC BY-NC-SA)

Raw file expected at ``--raw-dir`` (default: ``data/tabred/raw/delivery-eta/``):
    delivery_eta.parquet

Upstream's default split (preserved):
    train < 2023-12-11
    val   [2023-12-11, 2023-12-18)
    test  [2023-12-18, 2023-12-25)

Target is ``log(delivery_eta_minutes)``. Rows with ``delivery_eta_minutes < 1``
dropped. Categorical cols ``cat_1`` and ``cat_2`` are excluded (upstream).

The upstream split caps the test window at 2023-12-25. Later rows are excluded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _common import (
    SplitSpec,
    TARGET_COL,
    TIME_COL,
    default_temporal_split,
    ordinal_encode,
    parse_args,
    read_parquet_subsampled,
    require_raw_files,
    short_summary,
    unzip_if_needed,
    write_tabred_dataset,
)


SUBSAMPLE_FRAC = 0.025
SPLIT = SplitSpec(val_start="2023-12-11", test_start="2023-12-18")
TEST_END = "2023-12-25"
TARGET_RAW = "delivery_eta_minutes"
CAT_EXCLUDED = ["cat_1", "cat_2"]


def main() -> None:
    args = parse_args("delivery_eta", "delivery-eta")
    if (args.output_dir / "train.parquet").exists() and not args.overwrite:
        print(f"SKIP: {args.output_dir / 'train.parquet'} exists (pass --overwrite to redo).")
        return

    unzip_if_needed(args.raw_dir, "delivery-eta.zip")
    require_raw_files(args.raw_dir, ["delivery_eta.parquet"])

    # Stream-subsample at parquet-read time. Full file is 11 GB / 17M rows and
    # loading the complete data exceeds 32 GB of memory before subsampling.
    subsample_frac = args.subsample_frac if args.subsample_frac is not None else SUBSAMPLE_FRAC
    if subsample_frac < 1.0:
        df = read_parquet_subsampled(
            args.raw_dir / "delivery_eta.parquet",
            frac=subsample_frac,
            seed=args.seed,
        )
    else:
        df = pd.read_parquet(args.raw_dir / "delivery_eta.parquet")
    df = df.reset_index(drop=False).rename(columns={"index": "index_in_full"})

    df = df[df[TARGET_RAW] >= 1.0].copy()
    bin_cols = [c for c in df.columns if c.startswith("bin")]
    cat_cols = [c for c in df.columns if c.startswith("cat") and c not in CAT_EXCLUDED]
    num_cols = [c for c in df.columns if c.startswith("num")]
    df = ordinal_encode(df, cat_cols)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)

    df["day_of_week"] = df["timestamp"].dt.weekday
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["hour_of_day"] = df["timestamp"].dt.hour

    df[TIME_COL] = df["timestamp"]
    df[TARGET_COL] = np.log(df[TARGET_RAW].astype("float64")).astype("float32")

    num_features = num_cols + bin_cols + ["day_of_week", "minute_of_day", "hour_of_day"]
    cat_features = cat_cols

    keep = num_features + cat_features + [TARGET_COL, TIME_COL]
    train, test = default_temporal_split(df[keep], SPLIT)
    # Upstream bounds test at TEST_END.
    test = test[test[TIME_COL] < pd.Timestamp(TEST_END)].copy()

    write_tabred_dataset(
        out_dir=args.output_dir,
        train=train,
        test=test,
        task_type="regression",
        cat_features=cat_features,
        num_features=num_features,
        shift_description=f"TabReD delivery-eta natural gap (val window {SPLIT.val_start}..{SPLIT.test_start} dropped. Test capped at {TEST_END})",
        extra={
            "subsample_frac": subsample_frac,
            "subsample_seed": args.seed,
            "target_raw": TARGET_RAW,
            "target_transform": "log",
            "upstream_split": "default",
            "cat_excluded": CAT_EXCLUDED,
            "test_end": TEST_END,
        },
    )
    print("Wrote tabred_delivery_eta:")
    print(short_summary("delivery_eta", train, test))


if __name__ == "__main__":
    main()

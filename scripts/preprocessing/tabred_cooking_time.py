# Portions of this file adapt preprocessing choices from Yandex Research's
# TabReD project. This repository translates and integrates those choices into
# its pandas/parquet pipeline. TabReD is licensed under Apache-2.0. See
# THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""
TabReD Cooking Time: restaurant order cook-time prediction (regression).

Upstream reference: github.com/yandex-research/tabred/blob/main/preprocessing/cooking_time.py
Kaggle source:      pcovkrd84mejm/cooking-time  (CC BY-NC-SA)

Raw file expected at ``--raw-dir`` (default: ``data/tabred/raw/cooking-time/``):
    cooking_time.parquet

Upstream's default split (preserved):
    train < 2023-12-21  |  val [2023-12-21, 2023-12-28)  |  test >= 2023-12-28

Target is ``log(cooking_time_minutes)``. Rows with cooking_time_minutes < 1 are
dropped (matches upstream's filter). Categorical cols are ordinal-encoded with
``min_frequency=1/100``. Cat cols ``cat_0``, ``cat_2``, ``cat_3`` are excluded
(upstream excludes them: they are non-feature metadata).
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
    require_raw_files,
    short_summary,
    unzip_if_needed,
    write_tabred_dataset,
)


SUBSAMPLE_FRAC = 0.025
SPLIT = SplitSpec(val_start="2023-12-21", test_start="2023-12-28")
TARGET_RAW = "cooking_time_minutes"
CAT_EXCLUDED = ["cat_0", "cat_2", "cat_3"]  # upstream excludes these


def main() -> None:
    args = parse_args("cooking_time", "cooking-time")
    if (args.output_dir / "train.parquet").exists() and not args.overwrite:
        print(f"SKIP: {args.output_dir / 'train.parquet'} exists (pass --overwrite to redo).")
        return

    unzip_if_needed(args.raw_dir, "cooking-time.zip")
    require_raw_files(args.raw_dir, ["cooking_time.parquet"])

    df = pd.read_parquet(args.raw_dir / "cooking_time.parquet")
    df = df.reset_index(drop=False).rename(columns={"index": "index_in_full"})

    df = df[df[TARGET_RAW] >= 1.0].copy()
    bin_cols = [c for c in df.columns if c.startswith("bin")]
    cat_cols = [c for c in df.columns if c.startswith("cat") and c not in CAT_EXCLUDED]
    num_cols = [c for c in df.columns if c.startswith("num")]

    df = ordinal_encode(df, cat_cols)

    subsample_frac = args.subsample_frac if args.subsample_frac is not None else SUBSAMPLE_FRAC
    if subsample_frac < 1.0:
        df = df.sample(frac=subsample_frac, random_state=args.seed).reset_index(drop=True)

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

    write_tabred_dataset(
        out_dir=args.output_dir,
        train=train,
        test=test,
        task_type="regression",
        cat_features=cat_features,
        num_features=num_features,
        shift_description=f"TabReD cooking-time natural gap (val window {SPLIT.val_start}..{SPLIT.test_start} dropped)",
        extra={
            "subsample_frac": subsample_frac,
            "subsample_seed": args.seed,
            "target_raw": TARGET_RAW,
            "target_transform": "log",
            "upstream_split": "default",
            "cat_excluded": CAT_EXCLUDED,
        },
    )
    print("Wrote tabred_cooking_time:")
    print(short_summary("cooking_time", train, test))


if __name__ == "__main__":
    main()

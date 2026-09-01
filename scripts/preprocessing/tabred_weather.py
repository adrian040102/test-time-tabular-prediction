# Portions of this file adapt preprocessing choices from Yandex Research's
# TabReD project. This repository translates and integrates those choices into
# its pandas/parquet pipeline. TabReD is licensed under Apache-2.0. See
# THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""
TabReD Weather: temperature forecasting (regression).

Upstream reference: github.com/yandex-research/tabred/blob/main/preprocessing/weather.py
Kaggle source:      pcovkrd84mejm/tabred-weather  (CC BY-NC-SA)

Raw file expected at ``--raw-dir`` (default: ``data/tabred/raw/weather/``):
    weather.parquet

Upstream's default split (preserved):
    train < 2023-06-01  |  val [2023-06-01, 2023-07-01)  |  test >= 2023-07-01

Subsample fraction: 0.025 (matches upstream).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from _common import (
    SplitSpec,
    TARGET_COL,
    TIME_COL,
    default_temporal_split,
    parse_args,
    require_raw_files,
    short_summary,
    unzip_if_needed,
    write_tabred_dataset,
)


SUBSAMPLE_FRAC = 0.025
SPLIT = SplitSpec(val_start="2023-06-01", test_start="2023-07-01")

TARGET_RAW = "fact_temperature"
META_DROP = [
    # Upstream excludes from features.
    "fact_time",
    "apply_time_rl",
    "fact_latitude",
    "fact_longitude",
    "fact_station_id",
    "index_in_full",
]


def main() -> None:
    args = parse_args("weather", "weather")
    if (args.output_dir / "train.parquet").exists() and not args.overwrite:
        print(f"SKIP: {args.output_dir / 'train.parquet'} exists (pass --overwrite to redo).")
        return

    unzip_if_needed(args.raw_dir, "tabred-weather.zip")
    require_raw_files(args.raw_dir, ["weather.parquet"])

    df = pd.read_parquet(args.raw_dir / "weather.parquet")
    df = df.reset_index(drop=False).rename(columns={"index": "index_in_full"})

    subsample_frac = args.subsample_frac if args.subsample_frac is not None else SUBSAMPLE_FRAC
    if subsample_frac < 1.0:
        df = df.sample(frac=subsample_frac, random_state=args.seed).reset_index(drop=True)

    # Upstream stores fact_time as Unix seconds and casts to polars Datetime
    # (microseconds) via `* 10**6`. The pandas equivalent parses the values as seconds.
    df["fact_time"] = pd.to_datetime(df["fact_time"], unit="s", errors="coerce")
    df = df.dropna(subset=["fact_time"]).sort_values("fact_time", kind="stable").reset_index(drop=True)

    df["day_of_week"] = df["fact_time"].dt.weekday
    df["day_of_month"] = df["fact_time"].dt.day
    df["minute_of_day"] = df["fact_time"].dt.hour * 60 + df["fact_time"].dt.minute
    df["hour_of_day"] = df["fact_time"].dt.hour
    df["month"] = df["fact_time"].dt.month

    # Carry the canonical time col through to disk.
    df[TIME_COL] = df["fact_time"]

    feat_cols = [c for c in df.columns if c not in META_DROP + [TARGET_RAW, TIME_COL]]
    # Upstream calls anything with "available" in its name a binary indicator.
    bin_cols = [c for c in feat_cols if "available" in c]
    num_cols = [c for c in feat_cols if c not in bin_cols]
    # Cast bin to float32 (0/1).
    for c in bin_cols:
        df[c] = df[c].astype("float32")
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    df[TARGET_COL] = df[TARGET_RAW].astype("float32")

    cat_features: list[str] = []
    num_features = bin_cols + num_cols

    keep_cols = num_features + cat_features + [TARGET_COL, TIME_COL]
    train, test = default_temporal_split(df[keep_cols], SPLIT)

    meta = write_tabred_dataset(
        out_dir=args.output_dir,
        train=train,
        test=test,
        task_type="regression",
        cat_features=cat_features,
        num_features=num_features,
        shift_description=f"TabReD weather natural gap (val window {SPLIT.val_start}..{SPLIT.test_start} dropped)",
        extra={
            "subsample_frac": subsample_frac,
            "subsample_seed": args.seed,
            "target_raw": TARGET_RAW,
            "upstream_split": "default",
        },
    )
    print("Wrote tabred_weather:")
    print(short_summary("weather", train, test))


if __name__ == "__main__":
    main()

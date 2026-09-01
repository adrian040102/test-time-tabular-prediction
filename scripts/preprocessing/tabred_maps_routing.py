# Portions of this file adapt preprocessing choices from Yandex Research's
# TabReD project. This repository translates and integrates those choices into
# its pandas/parquet pipeline. TabReD is licensed under Apache-2.0. See
# THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""
TabReD Maps Routing: travel-time regression on routing requests.

Upstream reference: github.com/yandex-research/tabred/blob/main/preprocessing/maps_routing.py
Kaggle source:      pcovkrd84mejm/maps-routing  (CC BY-NC-SA)

Raw file expected at ``--raw-dir`` (default: ``data/tabred/raw/maps-routing/``):
    maps_routing.parquet

Upstream's default split (preserved):
    train < 2023-11-20
    val   [2023-11-20, 2023-11-27)
    test  [2023-11-27, 2023-12-04)

Target: ``target_log_spkm`` (already log-transformed and forwarded unchanged).
All ``cat_*`` columns are ordinal-encoded with ``min_frequency=1/100``.
``track_length`` is excluded from the emitted feature set.
"""

from __future__ import annotations

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
SPLIT = SplitSpec(val_start="2023-11-20", test_start="2023-11-27")
TEST_END = "2023-12-04"
TARGET_RAW = "target_log_spkm"


def main() -> None:
    args = parse_args("maps_routing", "maps-routing")
    if (args.output_dir / "train.parquet").exists() and not args.overwrite:
        print(f"SKIP: {args.output_dir / 'train.parquet'} exists (pass --overwrite to redo).")
        return

    unzip_if_needed(args.raw_dir, "maps-routing.zip")
    require_raw_files(args.raw_dir, ["maps_routing.parquet"])

    df = pd.read_parquet(args.raw_dir / "maps_routing.parquet")
    df = df.reset_index(drop=False).rename(columns={"index": "index_in_full"})

    cat_cols = [c for c in df.columns if c.startswith("cat")]
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
    df[TARGET_COL] = df[TARGET_RAW].astype("float32")

    num_features = num_cols + ["day_of_week", "minute_of_day", "hour_of_day"]
    cat_features = cat_cols

    keep = num_features + cat_features + [TARGET_COL, TIME_COL]
    train, test = default_temporal_split(df[keep], SPLIT)
    test = test[test[TIME_COL] < pd.Timestamp(TEST_END)].copy()

    write_tabred_dataset(
        out_dir=args.output_dir,
        train=train,
        test=test,
        task_type="regression",
        cat_features=cat_features,
        num_features=num_features,
        shift_description=f"TabReD maps-routing natural gap (val window {SPLIT.val_start}..{SPLIT.test_start} dropped. Test capped at {TEST_END})",
        extra={
            "subsample_frac": subsample_frac,
            "subsample_seed": args.seed,
            "target_raw": TARGET_RAW,
            "target_transform": "log_spkm_pretransformed",
            "upstream_split": "default",
            "test_end": TEST_END,
        },
    )
    print("Wrote tabred_maps_routing:")
    print(short_summary("maps_routing", train, test))


if __name__ == "__main__":
    main()

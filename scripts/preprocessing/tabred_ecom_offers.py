# Portions of this file adapt preprocessing choices from Yandex Research's
# TabReD project. This repository translates and integrates those choices into
# its pandas/parquet pipeline. TabReD is licensed under Apache-2.0. See
# THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""
TabReD Ecom Offers (Acquire Valued Shoppers Challenge): binary classification.

Upstream reference: github.com/yandex-research/tabred/blob/main/preprocessing/ecom_offers.py
Kaggle source:      acquire-valued-shoppers-challenge  (competition)

Raw files expected at ``--raw-dir`` (default: ``data/tabred/raw/ecom-offers/``):
    offers.csv.gz
    trainHistory.csv.gz
    transactions.csv.gz

Upstream's default split (preserved):
    train < 2013-04-20
    val   [2013-04-20, 2013-04-25)
    test  [2013-04-25, ...]

Feature engineering: aggregation of per-shopper transactions joined against
offers, with three filter flags (bought_company / category / brand), each
crossed with eleven date-diff windows (1, 3, 7, 14, 21, 28, 60, 90, 120, 150,
180 days). The pandas implementation follows the corresponding upstream aggregation
scheme. Column ordering may differ.

Target: ``repeater == 't'`` → 1 / 0.
"""

from __future__ import annotations

import gzip
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


SPLIT = SplitSpec(val_start="2013-04-20", test_start="2013-04-25")

DATE_DIFFS = [1, 3, 7, 14, 21, 28, 60, 90, 120, 150, 180]
FILTERS = ["company", "category", "brand"]


def _read_gz_csv(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        return pd.read_csv(f, low_memory=False)


def _iter_gz_csv_chunks(path: Path, chunksize: int = 5_000_000):
    """Yield DataFrame chunks from a gzipped CSV.

    Used for the 7 GB+ transactions.csv.gz file (350M rows): reading it
    whole OOMs on a 32 GB machine even after filtering.
    """
    with gzip.open(path, "rt") as f:
        for chunk in pd.read_csv(f, low_memory=False, chunksize=chunksize):
            yield chunk


def _per_chunk_aggregate(
    chunk: pd.DataFrame,
    train_offer: pd.DataFrame,
) -> pd.DataFrame | None:
    """Join chunk with train_offer (small), compute per-shopper partial aggs.

    Returns a DataFrame indexed by ``id`` with the per-chunk sums/counts for
    every (filter, window) cell. ``None`` if no rows meet the filter.
    """
    # Filter: only keep rows for shoppers in train_offer.
    chunk = chunk[chunk["id"].isin(train_offer["id"])]
    if len(chunk) == 0:
        return None

    chunk = chunk.merge(train_offer, on="id", how="inner", suffixes=("", "_offer"))
    chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
    chunk["date_diff"] = (chunk["offerdate"] - chunk["date"]).dt.days

    chunk["__same_company"] = chunk["company"] == chunk["company_offer"]
    chunk["__same_category"] = chunk["category"] == chunk["category_offer"]
    chunk["__same_brand"] = chunk["brand"] == chunk["brand_offer"]

    # total_spend (per id). Sum of all purchaseamount in this chunk.
    pieces: dict[str, pd.Series] = {
        "total_spend": chunk.groupby("id")["purchaseamount"].sum().astype("float32"),
    }
    for fn in FILTERS:
        m = chunk[f"__same_{fn}"]
        pieces[f"has_bought_{fn}"] = (
            chunk.loc[m].groupby("id")["purchaseamount"].count().astype("float32")
        )
        pieces[f"has_bought_{fn}_q"] = (
            chunk.loc[m].groupby("id")["purchasequantity"].sum().astype("float32")
        )
        pieces[f"has_bought_{fn}_a"] = (
            chunk.loc[m].groupby("id")["purchaseamount"].sum().astype("float32")
        )
        for d in DATE_DIFFS:
            md = m & (chunk["date_diff"] < d)
            pieces[f"has_bought_{fn}_{d}"] = (
                chunk.loc[md].groupby("id")["purchaseamount"].count().astype("float32")
            )
            pieces[f"has_bought_{fn}_q_{d}"] = (
                chunk.loc[md].groupby("id")["purchasequantity"].sum().astype("float32")
            )
            pieces[f"has_bought_{fn}_a_{d}"] = (
                chunk.loc[md].groupby("id")["purchaseamount"].sum().astype("float32")
            )

    # Combine the per-piece series into one frame keyed by id. Missing → 0.
    return pd.DataFrame(pieces).fillna(0.0)


def main() -> None:
    args = parse_args("ecom_offers", "ecom-offers")
    if (args.output_dir / "train.parquet").exists() and not args.overwrite:
        print(f"SKIP: {args.output_dir / 'train.parquet'} exists (pass --overwrite to redo).")
        return

    # Kaggle's `competitions download -c acquire-valued-shoppers-challenge`
    # produces a single outer archive containing offers.csv.gz, trainHistory.csv.gz,
    # transactions.csv.gz, sampleSubmission.csv.gz and reduced.csv.gz. Unzip the
    # outer once so the three needed gzipped CSVs are siblings under raw_dir.
    unzip_if_needed(args.raw_dir, "acquire-valued-shoppers-challenge.zip")
    require_raw_files(args.raw_dir, ["offers.csv.gz", "trainHistory.csv.gz", "transactions.csv.gz"])
    offers = _read_gz_csv(args.raw_dir / "offers.csv.gz")
    history = _read_gz_csv(args.raw_dir / "trainHistory.csv.gz")

    history["offerdate"] = pd.to_datetime(history["offerdate"], errors="coerce")

    # Join history to offers. Build target.
    train_offer = history.merge(offers, on="offer", how="inner", suffixes=("", "_offer"))
    train_offer["target"] = (train_offer["repeater"] == "t").astype("int32")
    train_offer = train_offer.drop(columns=["repeater"])

    # Stream transactions.csv.gz in chunks (full file is 7+ GB uncompressed,
    # 350M rows). Per-chunk filter to the ~160K shoppers in train_offer, then
    # per-chunk aggregate. At the end, sum partial aggregates across chunks.
    chunk_aggs: list[pd.DataFrame] = []
    print("  streaming transactions.csv.gz in chunks ...", flush=True)
    for i, chunk in enumerate(_iter_gz_csv_chunks(args.raw_dir / "transactions.csv.gz")):
        partial = _per_chunk_aggregate(chunk, train_offer)
        if partial is not None:
            chunk_aggs.append(partial)
        if (i + 1) % 10 == 0:
            print(f"    processed {i + 1} chunks", flush=True)

    if not chunk_aggs:
        raise RuntimeError("No transaction records correspond to shoppers in train_offer.")

    # Sum partial aggregates across chunks: groupby(level=0) sums same-id rows.
    combined = pd.concat(chunk_aggs).groupby(level=0).sum()
    # Add static info from train_offer (one row per id).
    train_offer_indexed = (
        train_offer
        .drop_duplicates(subset="id", keep="first")
        .set_index("id")[["target", "offervalue", "offerdate"]]
    )
    data = combined.join(train_offer_indexed, how="inner").sort_values("offerdate", kind="stable").reset_index()

    data["day_of_week"] = data["offerdate"].dt.weekday
    data["day_of_month"] = data["offerdate"].dt.day
    data["day_of_year"] = data["offerdate"].dt.dayofyear

    # Derived bin features: never_bought_X = has_bought_X == 0. Combined existence flags.
    for fn in FILTERS:
        data[f"never_bought_{fn}"] = (data[f"has_bought_{fn}"] == 0).astype("float32")
    data["has_bought_brand_company_category"] = (
        (data["has_bought_brand"] > 0)
        & (data["has_bought_category"] > 0)
        & (data["has_bought_company"] > 0)
    ).astype("float32")
    data["has_bought_brand_category"] = (
        (data["has_bought_brand"] > 0) & (data["has_bought_category"] > 0)
    ).astype("float32")
    data["has_bought_brand_company"] = (
        (data["has_bought_brand"] > 0) & (data["has_bought_company"] > 0)
    ).astype("float32")

    data[TIME_COL] = data["offerdate"]
    data[TARGET_COL] = data["target"].astype("int64")

    bin_cols = [
        f"never_bought_{fn}" for fn in FILTERS
    ] + [
        "has_bought_brand_company_category",
        "has_bought_brand_category",
        "has_bought_brand_company",
    ]
    drop = {"id", "target", "offerdate"}
    num_cols = [
        c for c in data.columns
        if c not in drop | {TIME_COL, TARGET_COL} | set(bin_cols)
    ]

    num_features = num_cols + bin_cols
    cat_features: list[str] = []
    keep = num_features + cat_features + [TARGET_COL, TIME_COL]
    train, test = default_temporal_split(data[keep], SPLIT)

    write_tabred_dataset(
        out_dir=args.output_dir,
        train=train,
        test=test,
        task_type="classification",
        cat_features=cat_features,
        num_features=num_features,
        shift_description=f"TabReD ecom-offers natural gap (val window {SPLIT.val_start}..{SPLIT.test_start} dropped)",
        extra={
            "target_raw": "repeater",
            "n_classes": 2,
            "upstream_split": "default",
            "date_diffs": DATE_DIFFS,
            "filters": FILTERS,
        },
    )
    print("Wrote tabred_ecom_offers:")
    print(short_summary("ecom_offers", train, test))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Dataset cleaning pipeline.

Reads source exports from ``data/datasets_original/`` or fetches configured
source APIs, applies dataset-specific transformations and writes parquet files
and metadata under ``data/datasets/``.

Usage:
    # Build all cleaned datasets:
    python scripts/build_clean_datasets.py

    # Build a single dataset:
    python scripts/build_clean_datasets.py --datasets folktables_coverage

    # Re-fetch datasets that need it (requires internet):
    python scripts/build_clean_datasets.py --allow-refetch

    # Overwrite existing cleaned datasets:
    python scripts/build_clean_datasets.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ORIGINAL = PROJECT_ROOT / "data" / "datasets_original"
DATA_CLEAN = PROJECT_ROOT / "data" / "datasets"
TARGET_COL = "__target__"


# Helpers

def read_original(name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read train.parquet, test.parquet, meta.json from datasets_original/."""
    d = DATA_ORIGINAL / name
    train = pd.read_parquet(d / "train.parquet")
    test = pd.read_parquet(d / "test.parquet")
    with open(d / "meta.json") as f:
        meta = json.load(f)
    return train, test, meta


def write_clean(name: str, train: pd.DataFrame, test: pd.DataFrame, meta: dict):
    """Write cleaned train.parquet, test.parquet, meta.json to datasets/."""
    d = DATA_CLEAN / name
    d.mkdir(parents=True, exist_ok=True)
    meta["name"] = name  # canonical name = folder name
    train.to_parquet(d / "train.parquet", index=False)
    test.to_parquet(d / "test.parquet", index=False)
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def retype_to_str_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert a float/int column to string category dtype."""
    df[col] = df[col].astype(int).astype(str).astype("category")
    return df


def move_to_cat(meta: dict, cols: list[str]) -> dict:
    """Move columns from num_features to cat_features in meta dict."""
    meta["num_features"] = [f for f in meta["num_features"] if f not in cols]
    meta["cat_features"] = meta["cat_features"] + [c for c in cols if c not in meta["cat_features"]]
    return meta


def drop_column(train: pd.DataFrame, test: pd.DataFrame, meta: dict, col: str):
    """Drop a column from train, test and meta feature lists."""
    train.drop(columns=[col], inplace=True, errors="ignore")
    test.drop(columns=[col], inplace=True, errors="ignore")
    meta["num_features"] = [f for f in meta["num_features"] if f != col]
    meta["cat_features"] = [f for f in meta["cat_features"] if f != col]
    meta["n_features"] = len(meta["num_features"]) + len(meta["cat_features"])
    return train, test, meta


# Per-dataset cleaning functions
#
# Each function returns a description of what it did (for logging).
# Convention: function name = clean_{dataset_name}
# If a dataset needs no cleaning: copy_as_is() handles it.


def clean_folktables_coverage(allow_refetch: bool = False) -> str:
    """
    Fixes:
    1. Retype ESR (float64 → string category): 7 unordered employment codes
    2. Retype ST (float64 → string category): 5 FIPS state codes
    3. Move ESR, ST from num_features to cat_features
    """
    train, test, meta = read_original("folktables_coverage")

    for df in [train, test]:
        retype_to_str_category(df, "ESR")
        retype_to_str_category(df, "ST")

    meta = move_to_cat(meta, ["ESR", "ST"])

    write_clean("folktables_coverage", train, test, meta)
    return "Retyped ESR, ST to categorical. Moved to cat_features"


def clean_folktables_coverage_covid(allow_refetch: bool = False) -> str:
    """Fetch the configured ACS cohorts, treat ESR and ST as categorical and
    regenerate dataset metadata."""
    if not allow_refetch:
        raise RuntimeError(
            "folktables_coverage_covid needs re-fetching (original was CA-only). "
            "Run with --allow-refetch to download from ACS API."
        )

    from src.data.folktables import load_folktables_coverage

    ds = load_folktables_coverage(
        train_year="2019",
        test_year="2022",
        train_states=["CA", "TX", "NY", "FL", "IL"],
        seed=42,
        download=True,
    )

    train = ds.X_train.copy()
    test = ds.X_test.copy()

    # Retype ESR and ST
    for df in [train, test]:
        retype_to_str_category(df, "ESR")
        retype_to_str_category(df, "ST")

    # Add target
    train[TARGET_COL] = ds.y_train
    test[TARGET_COL] = ds.y_test

    meta = {
        "name": "folktables_coverage_2019_2022",
        "task_type": "classification",
        "cat_features": [
            "MAR", "SEX", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC",
            "NATIVITY", "DEAR", "DEYE", "DREM", "RAC1P", "FER", "ESR", "ST",
        ],
        "num_features": ["AGEP", "SCHL", "PINCP"],
        "shift_description": "temporal: 2019\u21922022",
        "n_train": len(train),
        "n_test": len(test),
        "n_features": 19,
        "seed_used": 42,
        "max_samples_used": None,
        "extra": {
            "task": "coverage",
            "train_year": "2019",
            "test_year": "2022",
            "train_states": ["CA", "TX", "NY", "FL", "IL"],
            "test_states": ["CA", "TX", "NY", "FL", "IL"],
        },
    }

    write_clean("folktables_coverage_covid", train, test, meta)
    return (
        f"Re-fetched from ACS API (all 5 states, no max_samples). "
        f"{len(train)} train / {len(test)} test. "
        f"Retyped ESR, ST to categorical."
    )


def clean_folktables_employment(allow_refetch: bool = False) -> str:
    """
    Fixes:
    1. Retype RELP (float64 → string category): 18 relationship-to-householder codes (0-17)
    2. Move RELP from num_features to cat_features
    """
    train, test, meta = read_original("folktables_employment")

    for df in [train, test]:
        retype_to_str_category(df, "RELP")

    meta = move_to_cat(meta, ["RELP"])

    write_clean("folktables_employment", train, test, meta)
    return "Retyped RELP to categorical. Moved to cat_features"


# Registry
# Maps supported dataset names to cleaning functions.

def clean_folktables_travel_time_covid(allow_refetch: bool = False) -> str:
    """Fetch the configured ACS cohorts, treat ST and JWTR as categorical, retain
    PUMA, POWPUMA and RAC1P as stress-test features and regenerate metadata."""
    if not allow_refetch:
        raise RuntimeError(
            "folktables_travel_time_covid needs re-fetching (original was CA-only). "
            "Run with --allow-refetch to download from ACS API."
        )

    from src.data.folktables import load_folktables_travel_time

    ds = load_folktables_travel_time(
        train_year="2019",
        test_year="2022",
        train_states=["CA", "TX", "NY", "FL", "IL"],
        seed=42,
        download=True,
    )

    train = ds.X_train.copy()
    test = ds.X_test.copy()

    # Retype ST and JWTR
    for df in [train, test]:
        retype_to_str_category(df, "ST")
        retype_to_str_category(df, "JWTR")

    # Add target
    train[TARGET_COL] = ds.y_train
    test[TARGET_COL] = ds.y_test

    meta = {
        "name": "folktables_travel_time_2019_2022",
        "task_type": "classification",
        "cat_features": [
            "MAR", "SEX", "DIS", "ESP", "MIG", "RELP", "RAC1P",
            "PUMA", "CIT", "OCCP", "POWPUMA", "POVPIP", "ST", "JWTR",
        ],
        "num_features": ["AGEP", "SCHL"],
        "shift_description": "temporal: 2019\u21922022",
        "n_train": len(train),
        "n_test": len(test),
        "n_features": 16,
        "seed_used": 42,
        "max_samples_used": None,
        "extra": {
            "task": "travel_time",
            "train_year": "2019",
            "test_year": "2022",
            "train_states": ["CA", "TX", "NY", "FL", "IL"],
            "test_states": ["CA", "TX", "NY", "FL", "IL"],
        },
    }

    write_clean("folktables_travel_time_covid", train, test, meta)
    return (
        f"Re-fetched from ACS API (all 5 states, no max_samples). "
        f"{len(train)} train / {len(test)} test. "
        f"Retyped ST and JWTR as categorical. Retained PUMA and RAC1P features."
    )


def clean_folktables_mobility(allow_refetch: bool = False) -> str:
    """
    Fixes:
    1. Retype RELP (float64 → string category): 17 relationship-to-householder codes (0-17)
    2. Move RELP from num_features to cat_features
    3. Drop ESP (constant feature: all values 0.0)
    """
    train, test, meta = read_original("folktables_mobility")

    for df in [train, test]:
        retype_to_str_category(df, "RELP")

    meta = move_to_cat(meta, ["RELP"])
    train, test, meta = drop_column(train, test, meta, "ESP")

    write_clean("folktables_mobility", train, test, meta)
    return "Retyped RELP to categorical. Dropped ESP (constant)"


def clean_folktables_travel_time(allow_refetch: bool = False) -> str:
    """
    Fixes:
    1. Retype ST (float64 → string category): 5 FIPS state codes
    2. Retype JWTR (float64 → string category): 12 transportation mode codes
    3. Move ST, JWTR from num_features to cat_features
    """
    train, test, meta = read_original("folktables_travel_time")

    for df in [train, test]:
        retype_to_str_category(df, "ST")
        retype_to_str_category(df, "JWTR")

    meta = move_to_cat(meta, ["ST", "JWTR"])

    write_clean("folktables_travel_time", train, test, meta)
    return "Retyped ST, JWTR to categorical. Moved to cat_features"


def clean_folktables_income(allow_refetch: bool = False) -> str:
    """No fixes needed. Copy as-is."""
    train, test, meta = read_original("folktables_income")
    write_clean("folktables_income", train, test, meta)
    return "No fixes needed. Copied as-is"


def clean_brfss_diabetes(allow_refetch: bool = False) -> str:
    """Fetch the full BRFSS dataset without a sample cap. No feature-type changes
    are applied."""
    if not allow_refetch:
        raise RuntimeError(
            "brfss_diabetes needs re-fetching (original was 50K subsample). "
            "Run with --allow-refetch to load full BRFSS data."
        )

    from src.data.brfss import load_brfss

    ds = load_brfss(task="diabetes", max_samples=None, seed=42)

    train = ds.X_train.copy()
    test = ds.X_test.copy()
    train[TARGET_COL] = ds.y_train
    test[TARGET_COL] = ds.y_test

    meta = {
        "name": "brfss_diabetes",
        "task_type": "classification",
        "cat_features": ds.cat_features,
        "num_features": ds.num_features,
        "shift_description": "geographic: south+midwest\u2192northeast+west",
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(ds.cat_features) + len(ds.num_features),
        "seed_used": 42,
        "max_samples_used": None,
        "extra": {
            "source": "cdc_brfss",
            "year": 2022,
            "task": "diabetes",
            "n_classes": 2,
            "train_regions": ["south", "midwest"],
            "test_regions": ["northeast", "west"],
        },
    }

    write_clean("brfss_diabetes", train, test, meta)
    return (
        f"Re-fetched with full data (no max_samples). "
        f"{len(train)} train / {len(test)} test."
    )


def clean_tschalzev_sctp(allow_refetch: bool = False) -> str:
    """No fixes needed. IID control dataset, copy as-is."""
    train, test, meta = read_original("tschalzev_sctp")
    write_clean("tschalzev_sctp", train, test, meta)
    return "No fixes needed. Copied as-is (IID control)"


def clean_tschalzev_ogpcc(allow_refetch: bool = False) -> str:
    """No fixes needed. IID control dataset, copy as-is."""
    train, test, meta = read_original("tschalzev_ogpcc")
    write_clean("tschalzev_ogpcc", train, test, meta)
    return "No fixes needed. Copied as-is (IID control)"


def clean_tschalzev_mbgm(allow_refetch: bool = False) -> str:
    """No fixes needed. IID control dataset, copy as-is."""
    train, test, meta = read_original("tschalzev_mbgm")
    write_clean("tschalzev_mbgm", train, test, meta)
    return "No fixes needed. Copied as-is (IID control)"


def clean_tschalzev_aeac(allow_refetch: bool = False) -> str:
    """No fixes needed. IID control dataset, copy as-is."""
    train, test, meta = read_original("tschalzev_aeac")
    write_clean("tschalzev_aeac", train, test, meta)
    return "No fixes needed. Copied as-is (IID control)"


def clean_shifts_weather(allow_refetch: bool = False) -> str:
    """
    Fixes:
    1. Drop fact_time (domain identifier, KS=1.0, zero time overlap between splits)
    2. Drop climate (zero category overlap: train has dry/mild/tropical, test has only snow)
    3. Drop wrf_hail (constant: all 0 or NaN)
    4. Drop cmc_available, gfs_available (near-constant, >99.9% dominant)
    5. Drop cmc_0_1_68_0_grad (constant in test: only 1 unique value)
    """
    train, test, meta = read_original("shifts_weather")

    drop_cols = [
        "fact_time", "climate", "wrf_hail",
        "cmc_available", "gfs_available", "cmc_0_1_68_0_grad",
    ]
    for col in drop_cols:
        train, test, meta = drop_column(train, test, meta, col)

    write_clean("shifts_weather", train, test, meta)
    return (
        "Dropped fact_time (domain ID), climate (zero overlap), "
        "wrf_hail (constant), 3 near-constant features"
    )


def clean_lending_club(allow_refetch: bool = False) -> str:
    """
    Fixes:
    1. Drop application_type (constant in train: 100% 'Individual', no predictive signal)
    """
    train, test, meta = read_original("lending_club")
    train, test, meta = drop_column(train, test, meta, "application_type")
    write_clean("lending_club", train, test, meta)
    return "Dropped application_type (constant in train)"


def clean_gas_sensor_drift(allow_refetch: bool = False) -> str:
    """No fixes needed. Copy as-is."""
    train, test, meta = read_original("gas_sensor_drift")
    write_clean("gas_sensor_drift", train, test, meta)
    return "No fixes needed. Copied as-is"


def clean_diabetes_hospital(allow_refetch: bool = False) -> str:
    """
    Fixes:
    1. Drop admission_source_id (TV=1.0, defines the train/test split, zero category overlap)
    2. Drop nateglinide, chlorpropamide, acarbose (near-constant, >99% dominant class)
    """
    train, test, meta = read_original("diabetes_hospital")

    drop_cols = ["admission_source_id", "nateglinide", "chlorpropamide", "acarbose"]
    for col in drop_cols:
        train, test, meta = drop_column(train, test, meta, col)

    write_clean("diabetes_hospital", train, test, meta)
    return "Dropped admission_source_id (defines split) + 3 near-constant medication features"


def _copy_synthetic(name: str) -> str:
    """Helper: copy a synthetic dataset as-is."""
    train, test, meta = read_original(name)
    write_clean(name, train, test, meta)
    return "No fixes needed. Copied as-is (synthetic)"


def _build_controlled_synthetic(name: str) -> str:
    """Recreate one 10k/10k controlled covariate-shift dataset with its configured
    seed."""
    from src.data.synthetic import (
        CONTROLLED_SYNTHETIC_N_TEST,
        CONTROLLED_SYNTHETIC_N_TRAIN,
        CONTROLLED_SYNTHETIC_SEED,
        controlled_shift_strength,
        load_synthetic,
    )

    shift = controlled_shift_strength(name)
    if shift is None:
        raise ValueError(f"Not a controlled synthetic dataset: {name}")

    ds = load_synthetic(
        n_train=CONTROLLED_SYNTHETIC_N_TRAIN,
        n_test=CONTROLLED_SYNTHETIC_N_TEST,
        shift_strength=shift,
        seed=CONTROLLED_SYNTHETIC_SEED,
    )
    train = ds.X_train.copy()
    test = ds.X_test.copy()
    train[TARGET_COL] = ds.y_train
    test[TARGET_COL] = ds.y_test
    meta = {
        "name": name,
        "task_type": ds.task_type,
        "cat_features": ds.cat_features,
        "num_features": ds.num_features,
        "shift_description": ds.shift_description,
        "n_train": ds.n_train,
        "n_test": ds.n_test,
        "n_features": ds.n_features,
        "seed_used": CONTROLLED_SYNTHETIC_SEED,
        "extra": dict(ds.metadata),
    }
    write_clean(name, train, test, meta)
    return (
        "Recreated controlled synthetic series at "
        f"{CONTROLLED_SYNTHETIC_N_TRAIN}/{CONTROLLED_SYNTHETIC_N_TEST}, "
        f"shift={shift}, seed={CONTROLLED_SYNTHETIC_SEED}"
    )


SYNTHETIC_DATASETS = [
    "synthetic_shift_0.0",
    "synthetic_shift_0.5",
    "synthetic_shift_1.0",
    "synthetic_shift_3.0",
    "synthetic_shift_5.0",
    "synthetic_label_shift",
    "synthetic_partial_shift",
    "synthetic_small_test",
    "synthetic_large_test",
]

CONTROLLED_SYNTHETIC_DATASETS = {
    "synthetic_shift_0.0",
    "synthetic_shift_0.5",
    "synthetic_shift_1.0",
    "synthetic_shift_3.0",
    "synthetic_shift_5.0",
}


def clean_bike_sharing(allow_refetch: bool = False) -> str:
    """No fixes needed. Copy as-is."""
    train, test, meta = read_original("bike_sharing")
    write_clean("bike_sharing", train, test, meta)
    return "No fixes needed. Copied as-is"


CLEANERS: dict[str, callable] = {
    **{
        name: (
            (lambda n=name: lambda allow_refetch=False: _build_controlled_synthetic(n))()
            if name in CONTROLLED_SYNTHETIC_DATASETS
            else (lambda n=name: lambda allow_refetch=False: _copy_synthetic(n))()
        )
        for name in SYNTHETIC_DATASETS
    },
    "bike_sharing": clean_bike_sharing,
    "brfss_diabetes": clean_brfss_diabetes,
    "diabetes_hospital": clean_diabetes_hospital,
    "gas_sensor_drift": clean_gas_sensor_drift,
    "lending_club": clean_lending_club,
    "shifts_weather": clean_shifts_weather,
    "tschalzev_aeac": clean_tschalzev_aeac,
    "tschalzev_mbgm": clean_tschalzev_mbgm,
    "tschalzev_ogpcc": clean_tschalzev_ogpcc,
    "tschalzev_sctp": clean_tschalzev_sctp,
    "folktables_coverage": clean_folktables_coverage,
    "folktables_coverage_covid": clean_folktables_coverage_covid,
    "folktables_employment": clean_folktables_employment,
    "folktables_income": clean_folktables_income,
    "folktables_mobility": clean_folktables_mobility,
    "folktables_travel_time": clean_folktables_travel_time,
    "folktables_travel_time_covid": clean_folktables_travel_time_covid,
}


# Main

def main():
    p = argparse.ArgumentParser(description="Build cleaned datasets from originals.")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Specific datasets to clean (default: all registered)")
    p.add_argument("--allow-refetch", action="store_true",
                   help="Allow re-fetching from source APIs (requires internet)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing cleaned datasets")
    args = p.parse_args()

    targets = args.datasets or list(CLEANERS.keys())
    unknown = [t for t in targets if t not in CLEANERS]
    if unknown:
        print(f"ERROR: Unknown datasets: {unknown}")
        print(f"Registered: {list(CLEANERS.keys())}")
        sys.exit(1)

    print(f"Building {len(targets)} cleaned dataset(s)...\n")

    ok, fail, skip = 0, 0, 0
    for name in targets:
        out_dir = DATA_CLEAN / name / "meta.json"
        if out_dir.exists() and not args.overwrite:
            print(f"  SKIP {name} (already exists, use --overwrite)")
            skip += 1
            continue

        print(f"  {name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            desc = CLEANERS[name](allow_refetch=args.allow_refetch)
            elapsed = time.time() - t0
            print(f"OK ({elapsed:.1f}s): {desc}")
            ok += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {e}")
            fail += 1

    print(f"\nDone: {ok} built, {skip} skipped, {fail} failed.")


if __name__ == "__main__":
    main()

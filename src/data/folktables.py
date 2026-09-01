"""
Folktables dataset loaders for all five ACS prediction tasks.

Uses the folktables library to load US Census ACS data with
temporal and/or geographic distribution shift.

Available tasks:
  - ACSIncome: Predict whether income > $50k
  - ACSPublicCoverage: Predict public health insurance coverage
  - ACSEmployment: Predict employment status
  - ACSTravelTime: Predict whether commute > 20 minutes
  - ACSMobility: Predict whether a young adult moved addresses
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.base import TabularDataset

# Pre-downloaded ACS CSVs live under <project_root>/data/
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# Feature lists by task

# ACSIncome
_ACSINCOME_CAT_FEATURES = ["COW", "MAR", "OCCP", "POBP", "RELP", "RAC1P", "SEX"]
_ACSINCOME_NUM_FEATURES = ["AGEP", "SCHL", "WKHP"]

# ACSPublicCoverage
_ACSCOVERAGE_CAT_FEATURES = [
    "MAR", "SEX", "DIS", "ESP", "CIT", "MIG", "MIL",
    "ANC", "NATIVITY", "DEAR", "DEYE", "DREM", "RAC1P", "FER",
]
_ACSCOVERAGE_NUM_FEATURES = ["AGEP", "SCHL", "PINCP"]

# ACSEmployment
_ACSEMPLOYMENT_CAT_FEATURES = [
    "MAR", "SEX", "DIS", "ESP", "MIG", "CIT", "MIL",
    "ANC", "NATIVITY", "DEAR", "DEYE", "DREM", "RAC1P",
]
_ACSEMPLOYMENT_NUM_FEATURES = ["AGEP", "SCHL"]

# ACSTravelTime
_ACSTRAVELTIME_CAT_FEATURES = [
    "MAR", "SEX", "DIS", "ESP", "MIG", "RELP", "RAC1P",
    "PUMA", "CIT", "OCCP", "POWPUMA", "POVPIP",
]
_ACSTRAVELTIME_NUM_FEATURES = ["AGEP", "SCHL"]

# ACSMobility
_ACSMOBILITY_CAT_FEATURES = [
    "MAR", "SEX", "DIS", "ESP", "CIT", "MIL",
    "ANC", "NATIVITY", "DEAR", "DEYE", "DREM", "RAC1P",
    "GCL", "COW", "ESR",
]
_ACSMOBILITY_NUM_FEATURES = ["AGEP", "SCHL", "PINCP", "WKHP", "JWMNP"]

# Task definitions for the generic loader

_TASK_DEFS: dict[str, dict] = {}  # populated after imports in _get_task_def


def _get_task_def(task_key: str):
    """Return (task_object, cat_features, num_features) for a task key."""
    from folktables import ACSIncome, ACSPublicCoverage, ACSEmployment, ACSTravelTime, ACSMobility

    defs = {
        "income": (ACSIncome, _ACSINCOME_CAT_FEATURES, _ACSINCOME_NUM_FEATURES),
        "coverage": (ACSPublicCoverage, _ACSCOVERAGE_CAT_FEATURES, _ACSCOVERAGE_NUM_FEATURES),
        "employment": (ACSEmployment, _ACSEMPLOYMENT_CAT_FEATURES, _ACSEMPLOYMENT_NUM_FEATURES),
        "travel_time": (ACSTravelTime, _ACSTRAVELTIME_CAT_FEATURES, _ACSTRAVELTIME_NUM_FEATURES),
        "mobility": (ACSMobility, _ACSMOBILITY_CAT_FEATURES, _ACSMOBILITY_NUM_FEATURES),
    }
    return defs[task_key]


# Memory-efficient ACS loader

def _load_acs_split(
    task_obj,
    year: str,
    states: list[str],
    max_samples: int | None,
    seed: int,
    download: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load one split (train or test) state-by-state to keep peak memory low.

    Instead of loading all 5 states at once (~3M rows × 280 raw columns),
    this loads each state individually, immediately extracts only the
    task-specific columns (~10-15), deletes the raw data and concatenates
    only the small extracted DataFrames.
    """
    from folktables import ACSDataSource

    X_parts: list[pd.DataFrame] = []
    y_parts: list[np.ndarray] = []
    collected = 0

    for state in states:
        src = ACSDataSource(
            survey_year=year, horizon="1-Year", survey="person",
            root_dir=str(_DATA_DIR),
        )
        raw = src.get_data(states=[state], download=download)
        raw = _normalize_acs_columns(raw)
        X_st, y_st, _ = task_obj.df_to_pandas(raw)
        del raw, src
        gc.collect()

        X_parts.append(X_st)
        y_parts.append(np.asarray(y_st))
        collected += len(X_st)
        del X_st, y_st

        # Stop loading states once there are
        # enough rows (with 2× margin so subsampling has variety).
        if max_samples and collected >= max_samples * 2:
            break

    X = pd.concat(X_parts, ignore_index=True)
    y = np.concatenate(y_parts)
    del X_parts, y_parts
    gc.collect()

    if max_samples:
        X, y = _subsample(X, y, max_samples, seed)
        gc.collect()

    return X, y


def _load_folktables_task(
    task_key: str,
    train_year: str = "2014",
    test_year: str = "2018",
    train_states: list[str] | None = None,
    test_states: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    download: bool = False,
) -> TabularDataset:
    """Generic loader for any ACS/folktables task."""
    if train_states is None:
        train_states = ["CA", "TX", "NY", "FL", "IL"]
    if test_states is None:
        test_states = train_states

    task_obj, known_cat, known_num = _get_task_def(task_key)

    X_train, y_train = _load_acs_split(task_obj, train_year, train_states, max_samples, seed, download=download)
    X_test, y_test = _load_acs_split(task_obj, test_year, test_states, max_samples, seed + 1, download=download)

    cat_features, num_features = _detect_features(X_train, known_cat, known_num)

    shift_desc = f"temporal: {train_year}→{test_year}"
    if train_states != test_states:
        shift_desc += f", geographic: {train_states}→{test_states}"

    return TabularDataset(
        name=f"folktables_{task_key}_{train_year}_{test_year}",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.astype(int),
        y_test=y_test.astype(int),
        cat_features=cat_features,
        num_features=num_features,
        shift_description=shift_desc,
        metadata={
            "task": task_key,
            "train_year": train_year,
            "test_year": test_year,
            "train_states": train_states,
            "test_states": test_states,
        },
    )


# Public API

def load_folktables_income(
    train_year: str = "2014",
    test_year: str = "2018",
    train_states: list[str] | None = None,
    test_states: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    download: bool = False,
) -> TabularDataset:
    """
    Load ACSIncome task from Folktables with temporal/geographic shift.

    Args:
        train_year: Survey year for training data.
        test_year: Survey year for test data.
        train_states: US states for training (default: CA, TX, NY, FL, IL).
        test_states: US states for test (default: same as train_states).
        max_samples: Optional cap on samples per split for memory-constrained runs.
        seed: Random seed for subsampling.
        download: If True, download missing ACS survey files automatically.

    Returns:
        TabularDataset with ACSIncome data.
    """
    return _load_folktables_task(
        "income", train_year, test_year, train_states, test_states,
        max_samples, seed, download=download,
    )


def load_folktables_coverage(
    train_year: str = "2014",
    test_year: str = "2018",
    train_states: list[str] | None = None,
    test_states: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    download: bool = False,
) -> TabularDataset:
    """
    Load ACSPublicCoverage task from Folktables with temporal/geographic shift.

    Args:
        train_year: Survey year for training data.
        test_year: Survey year for test data.
        train_states: US states for training (default: CA, TX, NY, FL, IL).
        test_states: US states for test (default: same as train_states).
        max_samples: Optional cap on samples per split.
        seed: Random seed for subsampling.
        download: If True, download missing ACS survey files automatically.

    Returns:
        TabularDataset with ACSPublicCoverage data.
    """
    return _load_folktables_task(
        "coverage", train_year, test_year, train_states, test_states,
        max_samples, seed, download=download,
    )


def load_folktables_employment(
    train_year: str = "2014",
    test_year: str = "2018",
    train_states: list[str] | None = None,
    test_states: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    download: bool = False,
) -> TabularDataset:
    """
    Load ACSEmployment task from Folktables with temporal/geographic shift.

    Predicts employment status (ESR) for individuals aged 16–90.
    """
    return _load_folktables_task(
        "employment", train_year, test_year, train_states, test_states,
        max_samples, seed, download=download,
    )


def load_folktables_travel_time(
    train_year: str = "2014",
    test_year: str = "2018",
    train_states: list[str] | None = None,
    test_states: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    download: bool = False,
) -> TabularDataset:
    """
    Load ACSTravelTime task from Folktables with temporal/geographic shift.

    Predicts whether commute time exceeds 20 minutes for employed individuals.
    """
    return _load_folktables_task(
        "travel_time", train_year, test_year, train_states, test_states,
        max_samples, seed, download=download,
    )


def load_folktables_mobility(
    train_year: str = "2014",
    test_year: str = "2018",
    train_states: list[str] | None = None,
    test_states: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    download: bool = False,
) -> TabularDataset:
    """
    Load ACSMobility task from Folktables with temporal/geographic shift.

    Predicts whether a young adult changed addresses within the last year.
    """
    return _load_folktables_task(
        "mobility", train_year, test_year, train_states, test_states,
        max_samples, seed, download=download,
    )


# Helpers

# ACS PUMS column renames introduced in the 2019 redesign.
# The folktables task objects (ACSIncome, ACSEmployment, etc.) still reference
# the pre-2019 names, so 2019 and later data use the pre-2019 schema.
_ACS_COLUMN_RENAMES = {
    "RELSHIPP": "RELP",   # relationship to householder (expanded categories in 2019+)
    "JWTRNS":   "JWTR",   # means of transportation to work (renamed in 2019+)
}


def _normalize_acs_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename 2019+ ACS PUMS columns to the pre-2019 names expected by folktables."""
    renames = {old: new for old, new in _ACS_COLUMN_RENAMES.items() if old in df.columns}
    return df.rename(columns=renames) if renames else df


def _detect_features(
    X: pd.DataFrame,
    known_cat: list[str],
    known_num: list[str],
) -> tuple[list[str], list[str]]:
    """Classify columns into categorical / numerical, using known lists as hints."""
    cat_features = [c for c in known_cat if c in X.columns]
    num_features = [c for c in known_num if c in X.columns]
    for c in X.columns:
        if c not in cat_features and c not in num_features:
            if X[c].dtype == object:
                cat_features.append(c)
            else:
                num_features.append(c)
    return cat_features, num_features


def _subsample(
    X: pd.DataFrame, y: pd.Series | np.ndarray, n: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Randomly subsample to n rows."""
    if len(X) <= n:
        return X, np.asarray(y)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n, replace=False)
    X_sub = X.iloc[idx].reset_index(drop=True)
    y_sub = np.asarray(y)[idx]
    return X_sub, y_sub

"""
Lending Club loan dataset loader with natural temporal / financial shift.

Contains ~2.2M accepted consumer loans issued between 2007 and 2018. Training
on earlier years and testing on later years captures a strong natural temporal
shift driven by changing economic conditions, credit policies, interest-rate
environments and borrower demographics.

Task: binary classification (loan fully paid vs. default/charged-off).

Data source: https://www.kaggle.com/datasets/wordsforthewise/lending-club
Alternative: https://zenodo.org/records/11295916

The loader searches for the CSV in this order:
    1. Explicit ``csv_path`` argument
    2. ``<project_root>/data/accepted_2007_to_2018Q4.csv``
    3. ``<THESIS_DATA_CACHE>/lending_club/`` directory
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.base import TabularDataset


# Selected features (cleaned subset of the ~150 columns)
_CAT_FEATURES = [
    "term",                # 36 or 60 months
    "grade",               # LC assigned grade (A–G)
    "sub_grade",           # LC assigned subgrade (A1–G5)
    "emp_length",          # Employment length
    "home_ownership",      # RENT, OWN, MORTGAGE, OTHER
    "verification_status", # Verified, Source Verified, Not Verified
    "purpose",             # Loan purpose (debt_consolidation, credit_card, etc.)
    "addr_state",          # US state (2-letter code)
    "application_type",    # Individual or Joint
    "initial_list_status", # W (whole) or F (fractional)
]

_NUM_FEATURES = [
    "loan_amnt",           # Loan amount
    "int_rate",            # Interest rate (%)
    "installment",         # Monthly installment ($)
    "annual_inc",          # Annual income
    "dti",                 # Debt-to-income ratio
    "open_acc",            # Number of open credit lines
    "pub_rec",             # Number of derogatory public records
    "revol_bal",           # Total revolving balance
    "revol_util",          # Revolving utilization rate (%)
    "total_acc",           # Total credit lines
    "mort_acc",            # Number of mortgage accounts
    "pub_rec_bankruptcies", # Number of public record bankruptcies
]


def _find_csv(cache_dir: str | None = None) -> str:
    """Locate the Lending Club CSV file.

    Search order:
        1. Explicit ``cache_dir``
        2. ``<project_root>/data/accepted_2007_to_2018Q4.csv``
        3. ``<THESIS_DATA_CACHE>/lending_club/``
    """
    # Check explicit cache dir
    if cache_dir:
        candidates = list(Path(cache_dir).glob("*accepted*.csv"))
        if candidates:
            return str(candidates[0])

    # Check project-local data/ directory
    project_data = Path(__file__).resolve().parent.parent.parent / "data"
    local_csv = project_data / "accepted_2007_to_2018Q4.csv"
    if local_csv.exists():
        return str(local_csv)
    # Also check for any accepted*.csv in data/
    if project_data.is_dir():
        candidates = list(project_data.glob("*accepted*.csv"))
        if candidates:
            return str(candidates[0])

    # Check default cache location
    base = os.environ.get(
        "THESIS_DATA_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "thesis_data"),
    )
    lc_dir = os.path.join(base, "lending_club")

    if os.path.isdir(lc_dir):
        candidates = list(Path(lc_dir).glob("*accepted*.csv"))
        if candidates:
            return str(candidates[0])

    raise FileNotFoundError(
        "Lending Club CSV not found. Please download from:\n"
        "  https://www.kaggle.com/datasets/wordsforthewise/lending-club\n"
        "  or: https://zenodo.org/records/11295916\n"
        "and place the CSV in one of:\n"
        f"  {project_data}/\n"
        f"  {lc_dir}/\n"
        "Expected filename: accepted_2007_to_2018Q4.csv"
    )


def load_lending_club(
    train_years: tuple[int, int] | None = None,
    test_years: tuple[int, int] | None = None,
    max_samples: int | None = 50000,
    seed: int = 42,
    csv_path: str | None = None,
    cache_dir: str | None = None,
) -> TabularDataset:
    """
    Load the Lending Club dataset with temporal shift.

    Default split: train on 2007–2014, test on 2015–2018. This captures
    significant temporal drift in interest rates, credit policies and
    default rates.

    Args:
        train_years: (start, end) inclusive year range for training.
                     Default: (2007, 2014).
        test_years:  (start, end) inclusive year range for testing.
                     Default: (2015, 2018).
        max_samples: Cap on samples per split (default 50k).
        seed: Random seed for subsampling.
        csv_path: Direct path to the CSV file (overrides cache_dir search).
        cache_dir: Directory containing the downloaded CSV.

    Returns:
        TabularDataset (task_type="classification", binary default prediction).
    """
    if train_years is None:
        train_years = (2007, 2014)
    if test_years is None:
        test_years = (2015, 2018)

    if csv_path is None:
        csv_path = _find_csv(cache_dir)

    print(f"Loading Lending Club data from {csv_path} ...")
    # Read only needed columns to save memory
    needed_cols = (
        _CAT_FEATURES + _NUM_FEATURES +
        ["loan_status", "issue_d"]
    )

    # First pass: get column names
    sample = pd.read_csv(csv_path, nrows=0)
    available = [c for c in needed_cols if c in sample.columns]
    missing = [c for c in needed_cols if c not in sample.columns]
    if missing:
        print(f"  Warning: missing columns (will skip): {missing}")

    df = pd.read_csv(csv_path, usecols=available, low_memory=False)

    # Parse the issue date, which uses the format "Dec-2015".
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_d"])
    df["issue_year"] = df["issue_d"].dt.year

    # Build the binary target.
    # Keep only terminal loan statuses
    good_status = ["Fully Paid", "Current"]
    default_status = ["Charged Off", "Default", "Late (31-120 days)"]
    keep_status = good_status + default_status
    df = df[df["loan_status"].isin(keep_status)].copy()
    df["default"] = df["loan_status"].isin(default_status).astype(int)

    # Clean the features.
    cat_available = [c for c in _CAT_FEATURES if c in df.columns]
    num_available = [c for c in _NUM_FEATURES if c in df.columns]

    for col in cat_available:
        df[col] = df[col].fillna("missing").astype(str).str.strip()

    # Clean percentage columns (may have '%' suffix in some versions)
    for col in ["int_rate", "revol_util"]:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.replace("%", "").astype(float)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean emp_length: keep as categorical (ordinal groups)
    if "emp_length" in df.columns:
        df["emp_length"] = (
            df["emp_length"]
            .fillna("missing")
            .astype(str)
            .str.strip()
            .str.replace("< 1 year", "0", regex=False)
            .str.replace("10+ years", "10", regex=False)
            .str.replace(r"\s*years?", "", regex=True)
        )

    features = cat_available + num_available

    # Apply the temporal split.
    train_mask = (df["issue_year"] >= train_years[0]) & (df["issue_year"] <= train_years[1])
    test_mask = (df["issue_year"] >= test_years[0]) & (df["issue_year"] <= test_years[1])

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Empty split. train_years={train_years} ({train_mask.sum()} rows), "
            f"test_years={test_years} ({test_mask.sum()} rows)."
        )

    X_train = df.loc[train_mask, features].reset_index(drop=True)
    y_train = df.loc[train_mask, "default"].values.astype(int)
    X_test = df.loc[test_mask, features].reset_index(drop=True)
    y_test = df.loc[test_mask, "default"].values.astype(int)

    if max_samples:
        X_train, y_train = _subsample(X_train, y_train, max_samples, seed)
        X_test, y_test = _subsample(X_test, y_test, max_samples, seed + 1)

    shift_desc = f"temporal: {train_years[0]}–{train_years[1]}→{test_years[0]}–{test_years[1]}"

    return TabularDataset(
        name="lending_club",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cat_features=cat_available,
        num_features=num_available,
        task_type="classification",
        shift_description=shift_desc,
        metadata={
            "source": "lending_club",
            "task": "loan_default_prediction",
            "n_classes": 2,
            "train_years": list(train_years),
            "test_years": list(test_years),
        },
    )


def _subsample(
    X: pd.DataFrame, y: np.ndarray, n: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(X) <= n:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X.iloc[idx].reset_index(drop=True), y[idx]

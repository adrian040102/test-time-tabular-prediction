"""
CDC BRFSS dataset loader with geographic domain shift.

The Behavioral Risk Factor Surveillance System (BRFSS) is an annual telephone
health survey collecting data from ~400k+ respondents across all US states.
Training on one set of US census regions and testing on another produces a
natural geographic domain shift, as health outcomes, demographics and
risk-factor distributions vary substantially by region.

Two prediction tasks are supported:
- **diabetes**: Predict whether a respondent has been diagnosed with diabetes.
- **blood_pressure**: Predict whether a respondent has been diagnosed with
  high blood pressure.

Source: https://www.cdc.gov/brfss/annual_data/annual_data.htm
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.base import TabularDataset


# BRFSS 2022 SAS transport file used by this loader.
_DOWNLOAD_URL = (
    "https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip"
)

# US Census regions (mapped from BRFSS _STATE FIPS codes)
_REGION_MAP = {
    # Northeast
    9: "northeast", 23: "northeast", 25: "northeast", 33: "northeast",
    34: "northeast", 36: "northeast", 42: "northeast", 44: "northeast",
    50: "northeast",
    # Midwest
    17: "midwest", 18: "midwest", 19: "midwest", 20: "midwest",
    26: "midwest", 27: "midwest", 29: "midwest", 31: "midwest",
    38: "midwest", 39: "midwest", 46: "midwest", 55: "midwest",
    # South
    1: "south", 5: "south", 10: "south", 11: "south", 12: "south",
    13: "south", 21: "south", 22: "south", 24: "south", 28: "south",
    37: "south", 40: "south", 45: "south", 47: "south", 48: "south",
    51: "south", 54: "south",
    # West
    2: "west", 4: "west", 6: "west", 8: "west", 15: "west",
    16: "west", 30: "west", 32: "west", 35: "west", 41: "west",
    49: "west", 53: "west", 56: "west",
}

# Selected features that are consistently available and useful
# Using BRFSS 2022 variable names
_BRFSS_FEATURES = {
    # Categorical
    "cat": [
        "_SEX",          # Sex (1=Male, 2=Female)
        "_AGEG5YR",      # Age group (14 levels)
        "_RACE1",        # Race/ethnicity (computed, 8 levels)
        "_EDUCAG",       # Education level (4 levels)
        "_INCOMG1",      # Income level (7 levels)
        "_BMI5CAT",      # BMI category (4 levels)
        "_SMOKER3",      # Smoking status (4 levels)
        "_RFBING6",      # Binge drinking (1=No, 2=Yes)
        "EXERANY2",      # Any exercise in past 30 days
        "GENHLTH",       # General health (1=Excellent..5=Poor)
        "HLTHPLN1",      # Health plan coverage
        "MARITAL",       # Marital status
    ],
    # Numerical
    "num": [
        "PHYSHLTH",      # Days physical health not good (0-30)
        "MENTHLTH",      # Days mental health not good (0-30)
        "POORHLTH",      # Days poor health prevented activities
        "_BMI5",         # BMI * 100 (continuous)
        "SLEPTIM1",      # Hours of sleep
        "ALCDAY4",       # Days per month drinking alcohol
        "FRUITJU2",      # Times per month fruit juice
        "VEGETAB2",      # Times per month vegetables
    ],
}

# Target variable mappings
# Blood pressure: BPHIGH6 is the raw question. _RFHYPE6 is the computed
# hypertension variable. Multiple names are checked because the available columns
# vary across BRFSS years and LLCP vs. core files.
_TARGETS = {
    "diabetes": {
        "col": ["DIABETE4", "DIABETE3"],
        "positive": [1],       # 1 = Yes
        "negative": [3],       # 3 = No
        "drop": [2, 4, 7, 9],  # pre-diabetes, gestational, dk, refused
    },
    "blood_pressure": {
        "col": ["BPHIGH6", "BPHIGH4", "_RFHYPE6", "_RFHYPE5"],
        "positive": [1],       # 1 = Yes (or 1 = "No" for _RFHYPE, handled below)
        "negative": [3, 4],    # 3 = No, 4 = borderline (for BPHIGH variants)
        "drop": [7, 9],        # dk, refused
        # _RFHYPE6/_RFHYPE5 use 1=No and 2=Yes, so their values are remapped.
    },
}


def _download_and_cache(cache_dir: str | None = None) -> pd.DataFrame:
    """Download the BRFSS 2022 XPT file and return a DataFrame."""
    import urllib.request

    if cache_dir is None:
        base = os.environ.get(
            "THESIS_DATA_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "thesis_data"),
        )
        cache_dir = os.path.join(base, "brfss")
    os.makedirs(cache_dir, exist_ok=True)

    parquet_cache = os.path.join(cache_dir, "brfss_2022.parquet")
    if os.path.exists(parquet_cache):
        return pd.read_parquet(parquet_cache)

    xpt_path = os.path.join(cache_dir, "LLCP2022.XPT")
    if not os.path.exists(xpt_path):
        zip_path = os.path.join(cache_dir, "LLCP2022XPT.zip")
        if not os.path.exists(zip_path):
            print(f"Downloading BRFSS 2022 dataset to {cache_dir} ...")
            print("(This is a ~100MB file and may take a few minutes)")
            urllib.request.urlretrieve(_DOWNLOAD_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(cache_dir)

        # Find the .XPT file
        candidates = list(Path(cache_dir).rglob("*.XPT"))
        if candidates:
            xpt_path = str(candidates[0])

    print("Reading BRFSS SAS transport file (may take a moment) ...")
    df = pd.read_sas(xpt_path, format="xport")

    # Cache as parquet for faster future loads
    df.to_parquet(parquet_cache, index=False)
    return df


def load_brfss(
    task: str = "diabetes",
    train_regions: list[str] | None = None,
    test_regions: list[str] | None = None,
    max_samples: int | None = 50000,
    seed: int = 42,
    cache_dir: str | None = None,
) -> TabularDataset:
    """
    Load the CDC BRFSS dataset with geographic (US census region) shift.

    Default split: train on South + Midwest, test on Northeast + West.
    These region pairs have contrasting demographics and health outcomes.

    Args:
        task: Prediction task: "diabetes" or "blood_pressure".
        train_regions: Census regions for training.
                       Default: ["south", "midwest"].
        test_regions: Census regions for testing.
                      Default: ["northeast", "west"].
        max_samples: Cap on samples per split (default 50k to keep manageable).
        seed: Random seed for subsampling.
        cache_dir: Optional cache directory.

    Returns:
        TabularDataset (task_type="classification", binary).
    """
    if task not in _TARGETS:
        raise ValueError(f"Unknown task '{task}'. Available: {list(_TARGETS.keys())}")
    if train_regions is None:
        train_regions = ["south", "midwest"]
    if test_regions is None:
        test_regions = ["northeast", "west"]

    df = _download_and_cache(cache_dir)

    # Map FIPS state codes to census regions.
    df["region"] = df["_STATE"].map(_REGION_MAP)
    df = df.dropna(subset=["region"])

    # Build the target.
    target_cfg = _TARGETS[task]
    col_candidates = target_cfg["col"]

    # Find first available target column
    target_col = None
    for candidate in col_candidates:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col is None:
        raise ValueError(
            f"No target column found for task '{task}'. "
            f"Tried: {col_candidates}. "
            f"Available columns (first 30): {sorted(df.columns)[:30]}..."
        )

    print(f"  Using target column: {target_col}")

    # _RFHYPE variables use inverted coding: 1=No, 2=Yes
    if target_col.startswith("_RFHYPE"):
        df = df[df[target_col].isin([1, 2])].copy()
        df["target"] = (df[target_col] == 2).astype(int)
    else:
        # Standard BPHIGH / DIABETE coding
        keep_vals = target_cfg["positive"] + target_cfg["negative"]
        df = df[df[target_col].isin(keep_vals)].copy()
        df["target"] = df[target_col].isin(target_cfg["positive"]).astype(int)

    # Select the features.
    cat_cols = [c for c in _BRFSS_FEATURES["cat"] if c in df.columns]
    num_cols = [c for c in _BRFSS_FEATURES["num"] if c in df.columns]

    for col in cat_cols:
        df[col] = df[col].fillna(-1).astype(int).astype(str)

    for col in num_cols:
        # BRFSS uses 77/88/99 for dk/none/refused. Treat as NaN
        df[col] = df[col].replace({77: np.nan, 88: 0, 99: np.nan})

    features = cat_cols + num_cols

    # Split by region.
    train_mask = df["region"].isin(train_regions)
    test_mask = df["region"].isin(test_regions)

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Empty split. train_regions={train_regions} ({train_mask.sum()}), "
            f"test_regions={test_regions} ({test_mask.sum()})."
        )

    X_train = df.loc[train_mask, features].reset_index(drop=True)
    y_train = df.loc[train_mask, "target"].values.astype(int)
    X_test = df.loc[test_mask, features].reset_index(drop=True)
    y_test = df.loc[test_mask, "target"].values.astype(int)

    if max_samples:
        X_train, y_train = _subsample(X_train, y_train, max_samples, seed)
        X_test, y_test = _subsample(X_test, y_test, max_samples, seed + 1)

    shift_desc = f"geographic: {'+'.join(train_regions)}→{'+'.join(test_regions)}"

    return TabularDataset(
        name=f"brfss_{task}",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cat_features=cat_cols,
        num_features=num_cols,
        task_type="classification",
        shift_description=shift_desc,
        metadata={
            "source": "cdc_brfss",
            "year": 2022,
            "task": task,
            "n_classes": 2,
            "train_regions": train_regions,
            "test_regions": test_regions,
        },
    )


def load_brfss_diabetes(**kwargs) -> TabularDataset:
    """Convenience wrapper for BRFSS diabetes task."""
    return load_brfss(task="diabetes", **kwargs)


def load_brfss_blood_pressure(**kwargs) -> TabularDataset:
    """Convenience wrapper for BRFSS blood pressure task."""
    return load_brfss(task="blood_pressure", **kwargs)


def _subsample(
    X: pd.DataFrame, y: np.ndarray, n: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(X) <= n:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X.iloc[idx].reset_index(drop=True), y[idx]

"""
UCI Bike Sharing dataset loader with natural temporal covariate shift.

The dataset contains hourly bike rental counts for 2011–2012 in Washington D.C.
Training on Year 1 (2011) and testing on Year 2 (2012) produces a natural
covariate shift as ridership patterns, volume and seasonal distributions change
with service growth.

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.base import TabularDataset


_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
)

# Exclude instant, dteday, casual and registered. The target is cnt.
_CAT_FEATURES = ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
_NUM_FEATURES = ["temp", "atemp", "hum", "windspeed"]


def _download_and_cache(cache_dir: str | None = None) -> pd.DataFrame:
    """Download the Bike Sharing dataset and return the hourly DataFrame."""
    import urllib.request

    if cache_dir is None:
        # Use THESIS_DATA_CACHE env var if set (redirected on cluster),
        # otherwise fall back to ~/.cache/thesis_data
        base = os.environ.get("THESIS_DATA_CACHE",
                              os.path.join(os.path.expanduser("~"), ".cache", "thesis_data"))
        cache_dir = os.path.join(base, "bike_sharing")
    os.makedirs(cache_dir, exist_ok=True)

    csv_path = os.path.join(cache_dir, "hour.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    zip_path = os.path.join(cache_dir, "bike_sharing.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading Bike Sharing dataset to {cache_dir} ...")
        urllib.request.urlretrieve(_DOWNLOAD_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # The CSV may be nested in a subdirectory inside the zip
        csv_names = [n for n in zf.namelist() if n.endswith("hour.csv")]
        if not csv_names:
            raise FileNotFoundError(
                f"hour.csv not found in zip. Contents: {zf.namelist()}"
            )
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f)
    df.to_csv(csv_path, index=False)
    return df


def load_bike_sharing(
    train_year: int = 0,
    test_year: int = 1,
    max_samples: int | None = None,
    seed: int = 42,
    cache_dir: str | None = None,
) -> TabularDataset:
    """
    Load the UCI Bike Sharing (hourly) dataset with temporal shift.

    The ``yr`` column encodes 0 = 2011, 1 = 2012. The default split uses 2011
    for training and 2012 for testing, which captures the natural growth-driven
    covariate shift.

    Args:
        train_year: Year code for training (0 = 2011, 1 = 2012).
        test_year: Year code for testing.
        max_samples: Optional cap on samples per split.
        seed: Random seed for subsampling.
        cache_dir: Optional cache directory for downloaded data.

    Returns:
        TabularDataset (task_type="regression", target=cnt).
    """
    df = _download_and_cache(cache_dir)

    # Cast categoricals to string for uniform treatment
    for col in _CAT_FEATURES:
        df[col] = df[col].astype(str)

    features = _CAT_FEATURES + _NUM_FEATURES
    target = "cnt"

    train_mask = df["yr"] == train_year
    test_mask = df["yr"] == test_year

    X_train = df.loc[train_mask, features].reset_index(drop=True)
    y_train = df.loc[train_mask, target].values.astype(float)
    X_test = df.loc[test_mask, features].reset_index(drop=True)
    y_test = df.loc[test_mask, target].values.astype(float)

    if max_samples:
        X_train, y_train = _subsample(X_train, y_train, max_samples, seed)
        X_test, y_test = _subsample(X_test, y_test, max_samples, seed + 1)

    year_map = {0: "2011", 1: "2012"}
    shift_desc = f"temporal: {year_map.get(train_year, train_year)}→{year_map.get(test_year, test_year)}"

    return TabularDataset(
        name=f"bike_sharing_{train_year}_{test_year}",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cat_features=list(_CAT_FEATURES),
        num_features=list(_NUM_FEATURES),
        task_type="regression",
        shift_description=shift_desc,
        metadata={
            "source": "uci",
            "task": "bike_sharing_hourly",
            "train_year": train_year,
            "test_year": test_year,
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

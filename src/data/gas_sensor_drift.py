"""
UCI Gas Sensor Array Drift dataset loader with natural temporal concept drift.

The dataset contains measurements from 16 chemical sensors exposed to 6 different
gases over 36 months (Jan 2008 – Feb 2011). Sensor drift over time provides a
hardware-level temporal covariate shift. Data is organized into 10 sequential
batches.

Task: 6-class gas identification (classification).

Source: https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.base import TabularDataset


_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
)

# 6 gases (class labels 1–6)
GAS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}

# 16 sensors × 8 features = 128 features
_N_SENSORS = 16
_N_FEATURES_PER_SENSOR = 8
_FEATURE_NAMES = [
    f"sensor{s}_f{f}"
    for s in range(1, _N_SENSORS + 1)
    for f in range(1, _N_FEATURES_PER_SENSOR + 1)
]


def _parse_dat_file(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a .dat file in LIBSVM-like format.

    Each line:  class_label concentration feat_idx:value feat_idx:value ...

    Returns:
        classes: shape (n,): gas class (1–6)
        concentrations: shape (n,): gas concentration
        features: shape (n, 128)
    """
    classes = []
    concentrations = []
    all_features = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # The first token may combine the class and concentration with a
            # semicolon. Other formats separate them or list feature-value entries.
            if ";" in parts[0]:
                cls_str, conc_str = parts[0].split(";")
                feat_parts = parts[1:]
                concentrations.append(float(conc_str))
            elif len(parts) > 1 and ":" not in parts[1]:
                # Second token is a plain number → concentration
                cls_str = parts[0]
                concentrations.append(float(parts[1]))
                feat_parts = parts[2:]
            else:
                # No concentration field. Features start at parts[1]
                cls_str = parts[0]
                concentrations.append(0.0)
                feat_parts = parts[1:]

            classes.append(int(cls_str))

            # Parse index:value pairs
            feat_vec = np.zeros(128)
            for token in feat_parts:
                if ":" in token:
                    idx_str, val_str = token.split(":")
                    idx = int(idx_str) - 1  # 1-indexed → 0-indexed
                    feat_vec[idx] = float(val_str)
            all_features.append(feat_vec)

    return (
        np.array(classes),
        np.array(concentrations),
        np.array(all_features),
    )


def _download_and_cache(cache_dir: str | None = None) -> Path:
    """Download and extract the Gas Sensor Array Drift dataset into its cache directory."""
    import urllib.request

    if cache_dir is None:
        base = os.environ.get("THESIS_DATA_CACHE",
                              os.path.join(os.path.expanduser("~"), ".cache", "thesis_data"))
        cache_dir = os.path.join(base, "gas_sensor_drift")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    dat_files = sorted(cache_path.rglob("batch*.dat"))
    if len(dat_files) >= 10:
        return cache_path

    zip_path = cache_path / "gas_sensor.zip"
    if not zip_path.exists():
        print(f"Downloading Gas Sensor Array Drift dataset to {cache_dir} ...")
        urllib.request.urlretrieve(_DOWNLOAD_URL, str(zip_path))

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(cache_path))

    return cache_path


def load_gas_sensor_drift(
    train_batches: list[int] | None = None,
    test_batches: list[int] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    cache_dir: str | None = None,
) -> TabularDataset:
    """
    Load the UCI Gas Sensor Array Drift dataset with temporal batch splits.

    Default split: train on batches 1–6, test on batches 7–10 (early vs. late
    measurements, maximizing the sensor degradation / drift effect).

    Args:
        train_batches: Batch numbers for training (1–10). Default: [1,...,6].
        test_batches: Batch numbers for testing (1–10). Default: [7,...,10].
        max_samples: Optional cap on samples per split.
        seed: Random seed for subsampling.
        cache_dir: Optional cache directory.

    Returns:
        TabularDataset (task_type="classification", 6-class gas identification).
    """
    if train_batches is None:
        train_batches = list(range(1, 7))
    if test_batches is None:
        test_batches = list(range(7, 11))

    cache_path = _download_and_cache(cache_dir)

    def _load_batches(batch_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
        all_X, all_y = [], []
        for bid in batch_ids:
            # Find the .dat file (may be in a subdirectory)
            candidates = sorted(cache_path.rglob(f"batch{bid}.dat"))
            if not candidates:
                raise FileNotFoundError(
                    f"batch{bid}.dat not found under {cache_path}. "
                    f"Available: {sorted(cache_path.rglob('batch*.dat'))}"
                )
            classes, _conc, features = _parse_dat_file(str(candidates[0]))
            all_X.append(features)
            all_y.append(classes)
        return np.vstack(all_X), np.concatenate(all_y)

    X_train_np, y_train = _load_batches(train_batches)
    X_test_np, y_test = _load_batches(test_batches)

    # Convert to 0-indexed labels (0–5 instead of 1–6)
    y_train = y_train - 1
    y_test = y_test - 1

    X_train = pd.DataFrame(X_train_np, columns=_FEATURE_NAMES)
    X_test = pd.DataFrame(X_test_np, columns=_FEATURE_NAMES)

    if max_samples:
        X_train, y_train = _subsample(X_train, y_train, max_samples, seed)
        X_test, y_test = _subsample(X_test, y_test, max_samples, seed + 1)

    shift_desc = f"temporal sensor drift: batches {train_batches}→{test_batches}"

    return TabularDataset(
        name=f"gas_sensor_drift_{'_'.join(map(str, train_batches))}_vs_{'_'.join(map(str, test_batches))}",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.astype(int),
        y_test=y_test.astype(int),
        cat_features=[],
        num_features=list(_FEATURE_NAMES),
        task_type="classification",
        shift_description=shift_desc,
        metadata={
            "source": "uci",
            "task": "gas_sensor_drift",
            "n_classes": 6,
            "gas_names": GAS_NAMES,
            "train_batches": train_batches,
            "test_batches": test_batches,
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

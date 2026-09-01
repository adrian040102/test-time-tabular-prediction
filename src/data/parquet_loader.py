"""
Unified parquet-based dataset loader.

Each prepared dataset lives in its own folder with ``train.parquet``,
``test.parquet`` and ``meta.json``. Loading these files does not require an
API call or internet access.

The data root is resolved in this order:
    1. Explicit ``data_root`` argument
    2. ``THESIS_PARQUET_ROOT`` environment variable
    3. ``SCRATCH`` env var + ``/thesis_data/datasets``
    4. ``<project_root>/data/datasets``
    5. ``<project_root>/data/tabarena/datasets``

The TabArena fallback (5) is checked automatically so that dataset names from
``data/tabarena/datasets/`` work with ``load_dataset()``
without any config changes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.base import TabularDataset

TARGET_COL = "__target__"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Determine where parquet datasets live (primary root only)."""
    if data_root:
        return Path(data_root)

    env = os.environ.get("THESIS_PARQUET_ROOT", "")
    if env:
        return Path(env)

    scratch = os.environ.get("SCRATCH", "")
    if scratch:
        return Path(scratch) / "thesis_data" / "datasets"

    return _PROJECT_ROOT / "data" / "datasets"


def _find_dataset_dir(
    name: str,
    data_root: str | Path | None = None,
) -> Path | None:
    """
    Find the folder containing ``meta.json`` for *name*.

    Search order:
      1. Primary root (explicit arg / env var / SCRATCH / data/datasets)
      2. ``data/tabarena/datasets/`` (auto-fallback for TabArena datasets)

    Returns the Path to the dataset directory or None if not found.
    """
    # Primary root
    primary = _resolve_data_root(data_root) / name
    if (primary / "meta.json").is_file():
        return primary

    # TabArena fallback (only when no explicit data_root was given)
    if data_root is None:
        tabarena = _PROJECT_ROOT / "data" / "tabarena" / "datasets" / name
        if (tabarena / "meta.json").is_file():
            return tabarena

    return None


def parquet_dataset_exists(
    name: str,
    data_root: str | Path | None = None,
) -> bool:
    """Check whether a parquet-exported dataset is available."""
    return _find_dataset_dir(name, data_root) is not None


def _validate_feature_type_contract(
    dataset_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_features: list[str],
    num_features: list[str],
) -> None:
    """Fail before modelling if stored columns and metadata disagree."""
    train_columns = list(X_train.columns)
    test_columns = list(X_test.columns)
    if train_columns != test_columns:
        raise ValueError(
            f"Dataset '{dataset_name}' has different train and test columns "
            "or column order."
        )

    if len(cat_features) != len(set(cat_features)):
        raise ValueError(
            f"Dataset '{dataset_name}' has duplicate categorical declarations."
        )
    if len(num_features) != len(set(num_features)):
        raise ValueError(
            f"Dataset '{dataset_name}' has duplicate numerical declarations."
        )

    cat_set = set(cat_features)
    num_set = set(num_features)
    overlap = cat_set & num_set
    declared = cat_set | num_set
    columns = set(train_columns)
    if overlap or declared != columns:
        raise ValueError(
            f"Dataset '{dataset_name}' has an invalid feature-type partition "
            f"(overlap={sorted(overlap)}, "
            f"missing={sorted(columns - declared)}, "
            f"extra={sorted(declared - columns)})."
        )

    invalid_numeric = []
    for column in num_features:
        train_numeric = pd.api.types.is_numeric_dtype(X_train[column])
        test_numeric = pd.api.types.is_numeric_dtype(X_test[column])
        if not train_numeric or not test_numeric:
            invalid_numeric.append(
                f"{column} (train={X_train[column].dtype}, "
                f"test={X_test[column].dtype})"
            )
    if invalid_numeric:
        raise ValueError(
            f"Dataset '{dataset_name}' declares non-numeric stored columns as "
            f"numerical: {invalid_numeric}. Repair meta.json or apply an "
            "explicit value conversion before running experiments."
        )


def _load_fold_split(
    ds_dir: Path,
    meta: dict,
    fold_idx: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """
    Create one fold's train/test split from data.parquet + folds.npz.

    Returns (X_train, X_test, y_train, y_test, fold_spec) where fold_spec is
    the entry from meta['fold_specs'] for this fold_idx.
    """
    data_path = ds_dir / "data.parquet"
    folds_path = ds_dir / "folds.npz"
    if not data_path.is_file() or not folds_path.is_file():
        raise FileNotFoundError(
            f"Dataset '{meta.get('name', ds_dir.name)}' has no fold layer "
            f"(missing data.parquet or folds.npz under {ds_dir}). "
            f"Run scripts/augment_tabarena_folds.py first."
        )

    fold_specs = meta.get("fold_specs") or []
    n_total = len(fold_specs)
    if n_total == 0:
        raise ValueError(
            f"Dataset '{meta.get('name', ds_dir.name)}' has no fold_specs in "
            f"meta.json. Run scripts/augment_tabarena_folds.py to populate."
        )
    if not (0 <= fold_idx < n_total):
        raise IndexError(
            f"fold_idx={fold_idx} out of range for dataset "
            f"'{meta.get('name', ds_dir.name)}' (n_total_folds={n_total})."
        )

    full_df = pd.read_parquet(data_path)
    with np.load(folds_path) as npz:
        train_idx = npz[f"train_idx_{fold_idx}"]
        test_idx = npz[f"test_idx_{fold_idx}"]

    train_df = full_df.iloc[train_idx].reset_index(drop=True)
    test_df = full_df.iloc[test_idx].reset_index(drop=True)

    y_train = train_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()
    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])

    return X_train, X_test, y_train, y_test, fold_specs[fold_idx]


def load_parquet_dataset(
    name: str,
    data_root: str | Path | None = None,
    seed: int = 42,
    max_samples: int | None = None,
    fold_idx: int | None = None,
) -> TabularDataset:
    """
    Load a dataset from pre-exported parquet files.

    Searches the primary data root first, then ``data/tabarena/datasets/``
    automatically so TabArena datasets work without any extra configuration.

    Args:
        name: Dataset name (must match the folder name under data_root).
        data_root: Override the default data root directory.
        seed: Random seed (used for subsampling if max_samples is set).
        max_samples: If set, subsample train and test to at most this many rows.
        fold_idx: If set, load this OpenML CV fold (requires ``data.parquet`` +
            ``folds.npz`` produced by ``scripts/augment_tabarena_folds.py``).
            None (default) keeps the stored single-split behavior.

    Returns:
        A TabularDataset ready for use in the pipeline.
    """
    ds_dir = _find_dataset_dir(name, data_root)

    if ds_dir is None:
        raise FileNotFoundError(
            f"No parquet dataset '{name}' found in data/datasets/ or "
            f"data/tabarena/datasets/. See DATASETS.md for the relevant "
            f"acquisition or construction command."
        )

    with open(ds_dir / "meta.json") as f:
        meta = json.load(f)

    fold_spec: dict | None = None
    time_train: np.ndarray | None = None
    time_test: np.ndarray | None = None
    time_col_name = meta.get("time_col")
    if fold_idx is not None:
        # Fold-aware path: split data.parquet using folds.npz indices
        X_train, X_test, y_train, y_test, fold_spec = _load_fold_split(
            ds_dir, meta, fold_idx,
        )
    else:
        # Default path for pre-split train.parquet and test.parquet files.
        train_df = pd.read_parquet(ds_dir / "train.parquet")
        test_df = pd.read_parquet(ds_dir / "test.parquet")
        y_train = train_df[TARGET_COL].to_numpy()
        y_test = test_df[TARGET_COL].to_numpy()
        X_train = train_df.drop(columns=[TARGET_COL])
        X_test = test_df.drop(columns=[TARGET_COL])

    # If meta.json declares a time_col (TabReD convention), strip it from the
    # feature matrix and stash the values in metadata for the gap-widening
    # sampler to consume. This applies to the fold-aware and default paths so a
    # future temporal-CV dataset cannot silently leak a raw timestamp feature.
    # Pipelines that do not sample never see the time column.
    if time_col_name and time_col_name in X_train.columns:
        time_train = X_train[time_col_name].to_numpy()
        time_test = X_test[time_col_name].to_numpy()
        X_train = X_train.drop(columns=[time_col_name])
        X_test = X_test.drop(columns=[time_col_name])

    cat_features = [
        column
        for column in meta.get("cat_features", [])
        if column != time_col_name
    ]
    num_features = [
        column
        for column in meta.get("num_features", [])
        if column != time_col_name
    ]
    _validate_feature_type_contract(
        meta.get("name", name),
        X_train,
        X_test,
        cat_features,
        num_features,
    )

    # Optional subsampling
    if max_samples:
        rng = np.random.RandomState(seed)
        if len(X_train) > max_samples:
            idx = rng.choice(len(X_train), max_samples, replace=False)
            X_train = X_train.iloc[idx].reset_index(drop=True)
            y_train = y_train[idx]
            # TabReD datasets carry a time_col side-array. Subsample in lockstep
            # so the gap-widening sampler's length checks do not trip.
            if time_train is not None:
                time_train = time_train[idx]
        if len(X_test) > max_samples:
            idx = rng.choice(len(X_test), max_samples, replace=False)
            X_test = X_test.iloc[idx].reset_index(drop=True)
            y_test = y_test[idx]
            if time_test is not None:
                time_test = time_test[idx]

    # Ensure integer targets for classification
    task_type = meta["task_type"]
    if task_type == "regression":
        # A binary or Boolean target labeled as regression would be scored with
        # RMSE instead of AUC. Detect the metadata error while loading the data.
        y_kind = np.asarray(y_train).dtype.kind
        n_unique_train = len(np.unique(y_train))
        if y_kind == "b" or n_unique_train <= 2:
            raise ValueError(
                f"Dataset '{meta.get('name', name)}' is tagged "
                f"task_type='regression' but its target has {n_unique_train} "
                f"unique value(s) (dtype kind '{y_kind}'). This looks like a "
                f"classification target mislabeled as regression. Fix meta.json "
                f"(task_type='classification') or the auto-detect in "
                f"src/data/openml_datasets.py."
            )
    if task_type == "classification":
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

    # Per-fold metadata (additive to dataset.metadata so downstream code
    # can record which fold a result came from).
    extra_meta = dict(meta.get("extra", {}))
    if fold_spec is not None:
        extra_meta["fold_idx"] = int(fold_spec["fold_idx"])
        extra_meta["fold_repeat"] = int(fold_spec["repeat"])
        extra_meta["fold_fold"] = int(fold_spec["fold"])
        extra_meta["fold_sample"] = int(fold_spec["sample"])
        extra_meta["fold_train_size"] = int(fold_spec["train_size"])
        extra_meta["fold_test_size"] = int(fold_spec["test_size"])

    if time_col_name and time_train is not None and time_test is not None:
        extra_meta["time_col"] = time_col_name
        extra_meta["time_train"] = time_train
        extra_meta["time_test"] = time_test

    return TabularDataset(
        name=meta.get("name", name),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cat_features=cat_features,
        num_features=num_features,
        task_type=task_type,
        shift_description=meta.get("shift_description", ""),
        metadata=extra_meta,
    )

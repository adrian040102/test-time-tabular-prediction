"""
OpenML dataset loader for TabArena (Study 457) and standalone OpenML datasets.

Provides loading by OpenML ID plus convenience loaders for TabArena and selected
named datasets. Downloads are cached by the ``openml`` package.

These datasets use standard IID random splits and serve as a broader evaluation
complement to the shift-specific datasets (Folktables, TableShift, etc.).

Sources:
  - TabArena: https://github.com/autogluon/tabarena (OpenML Study 457)
  - Tschalzev et al., "A Data-Centric Perspective on Evaluating ML Models
    for Tabular Data" (NeurIPS 2024 D&B)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from src.data.base import TabularDataset


# ---------------------------------------------------------------------------
# TabArena study ID
# ---------------------------------------------------------------------------
TABARENA_STUDY_ID = 457

# OpenML's nominal-feature indicator is wrong or incomplete for these six
# TabArena datasets. Keep the corrections keyed by stable OpenML dataset ID so
# both the initial downloader and the fold augmenter apply the same rules.
#
# The first six entries are type corrections only: stored values stay
# unchanged and the named columns are treated as categorical. Dt_Customer is a
# date, so it is converted to an integer day offset. This keeps the order,
# keeps one feature and avoids treating 663 dates as unrelated categories.
TABARENA_FEATURE_TYPE_REPAIRS: dict[int, dict[str, Any]] = {
    46910: {"categorical": ("poutcome",)},
    46916: {
        "categorical": (
            "avgAge",
            "contributionPrivateThirdPartyInsurance",
        ),
    },
    46935: {"categorical": ("experience",)},
    46937: {"categorical": ("coupon",)},
    46940: {
        "date_as_days": {
            "Dt_Customer": "2012-07-30",
        },
    },
    46956: {"categorical": ("ghazard",)},
}


def _apply_openml_feature_value_repairs(
    X: pd.DataFrame,
    openml_id: int,
) -> pd.DataFrame:
    """Apply the value part of a configured OpenML feature-type repair.

    Categorical corrections do not change stored values. Date corrections are
    explicit and idempotent: an already numerical day-offset column is
    validated and left unchanged.
    """
    repair = TABARENA_FEATURE_TYPE_REPAIRS.get(int(openml_id), {})
    date_rules = repair.get("date_as_days", {})
    if not date_rules:
        return X

    result = X.copy()
    for column, origin_text in date_rules.items():
        if column not in result.columns:
            raise ValueError(
                f"OpenML dataset {openml_id} is missing repaired date column "
                f"'{column}'."
            )
        series = result[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="raise")
            if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
                raise ValueError(
                    f"OpenML dataset {openml_id} column '{column}' contains "
                    "invalid numerical day offsets."
                )
            result[column] = numeric.astype("int32")
            continue

        parsed = pd.to_datetime(series, format="%Y-%m-%d", errors="raise")
        if parsed.isna().any():
            raise ValueError(
                f"OpenML dataset {openml_id} column '{column}' contains "
                "missing dates."
            )
        origin = pd.Timestamp(origin_text)
        result[column] = (parsed - origin).dt.days.astype("int32")

    return result


def _resolve_openml_feature_types(
    X: pd.DataFrame,
    cat_indicator,
    feature_names,
    openml_id: int,
) -> tuple[list[str], list[str]]:
    """Resolve and validate feature groups after applying configured repairs."""
    cat_features: list[str] = []
    num_features: list[str] = []

    if cat_indicator is not None and len(cat_indicator) == len(X.columns):
        for col, is_cat in zip(X.columns, cat_indicator):
            (cat_features if is_cat else num_features).append(col)
    else:
        for col in X.columns:
            if (
                X[col].dtype == object
                or pd.api.types.is_categorical_dtype(X[col])
                or pd.api.types.is_string_dtype(X[col])
            ):
                cat_features.append(col)
            else:
                num_features.append(col)

    repair = TABARENA_FEATURE_TYPE_REPAIRS.get(int(openml_id), {})
    for column in repair.get("categorical", ()):
        if column not in X.columns:
            raise ValueError(
                f"OpenML dataset {openml_id} is missing repaired categorical "
                f"column '{column}'."
            )
        if column in num_features:
            num_features.remove(column)
        if column not in cat_features:
            cat_features.append(column)

    # A column may be numerically stored and intentionally categorical, but a
    # column declared numerical must be numerical in the actual frame.
    invalid_numeric = [
        column
        for column in num_features
        if not pd.api.types.is_numeric_dtype(X[column])
    ]
    if invalid_numeric:
        raise ValueError(
            f"OpenML dataset {openml_id} has non-numeric columns declared "
            f"numerical after feature-type repair: {invalid_numeric}."
        )

    overlap = set(cat_features) & set(num_features)
    covered = set(cat_features) | set(num_features)
    if overlap or covered != set(X.columns):
        raise ValueError(
            f"OpenML dataset {openml_id} has an invalid feature-type "
            f"partition (overlap={sorted(overlap)}, "
            f"missing={sorted(set(X.columns) - covered)}, "
            f"extra={sorted(covered - set(X.columns))})."
        )
    return cat_features, num_features


def _openml_feature_repair_notes(openml_id: int) -> list[str]:
    """Return plain provenance notes for exported metadata."""
    repair = TABARENA_FEATURE_TYPE_REPAIRS.get(int(openml_id), {})
    notes = [
        f"Declared '{column}' categorical to preserve its feature type."
        for column in repair.get("categorical", ())
    ]
    notes.extend(
        f"Converted '{column}' from YYYY-MM-DD text to integer days since {origin}."
        for column, origin in repair.get("date_as_days", {}).items()
    )
    return notes

# ---------------------------------------------------------------------------
# Tschalzev et al. datasets with known OpenML IDs.
# SCS, BPCCM, HQC and IFD have no OpenML ID. They require Kaggle and are excluded.
# ---------------------------------------------------------------------------
TSCHALZEV_DATASETS: dict[str, dict[str, Any]] = {
    "mbgm": {
        "openml_id": 42570,
        "full_name": "Mercedes-Benz Greener Manufacturing",
        "task_type": "regression",
    },
    "svpc": {
        "openml_id": 42572,
        "full_name": "Santander Value Prediction Challenge",
        "task_type": "regression",
    },
    "aeac": {
        "openml_id": 4135,
        "full_name": "Amazon Employee Access Challenge",
        "task_type": "classification",
    },
    "ogpcc": {
        "openml_id": 45548,
        "full_name": "Otto Group Product Classification Challenge",
        "task_type": "classification",
    },
    "sctp": {
        "openml_id": 42395,
        "full_name": "Santander Customer Transaction Prediction",
        "task_type": "classification",
    },
    "pssdp": {
        "openml_id": 43121,
        "full_name": "Porto Seguro Safe Driver Prediction",
        "task_type": "classification",
    },
}


def _ensure_openml():
    """Import openml, configure cache directory or raise a helpful error."""
    try:
        import openml
    except ImportError:
        raise ImportError(
            "The 'openml' package is required for OpenML datasets.\n"
            "Install with:  pip install openml"
        )

    # Redirect OpenML cache to scratch/tmp storage to avoid filling up the
    # home directory quota on HPC clusters.  Priority order:
    #   1. OPENML_CACHE_DIRECTORY env var (user-set, always respected)
    #   2. $SCRATCH/.cache/openml   (common HPC scratch partition)
    #   3. $TMPDIR/.cache/openml    (job-local temp directory)
    #   4. Default (~/.cache/openml), unchanged
    if "OPENML_CACHE_DIRECTORY" not in os.environ:
        for env_var in ("SCRATCH", "TMPDIR"):
            base = os.environ.get(env_var)
            if base and os.path.isdir(base):
                cache_dir = os.path.join(base, ".cache", "openml")
                os.makedirs(cache_dir, exist_ok=True)
                openml.config.set_root_cache_directory(cache_dir)
                break

    return openml


def _check_disk_space(path: str, required_mb: int = 500) -> None:
    """Raise an informative error if disk space is too low for dataset download."""
    import shutil
    usage = shutil.disk_usage(path)
    free_mb = usage.free // (1024 * 1024)
    if free_mb < required_mb:
        raise OSError(
            f"Insufficient disk space for OpenML dataset download. "
            f"Only {free_mb} MB free at {path}, need ~{required_mb} MB. "
            f"Set OPENML_CACHE_DIRECTORY to a path with more space or "
            f"set SCRATCH or TMPDIR to redirect the cache automatically."
        )


def load_openml_dataset(
    openml_id: int,
    *,
    name: str | None = None,
    task_type: str | None = None,
    test_size: float = 0.2,
    max_samples: int | None = None,
    seed: int = 42,
) -> TabularDataset:
    """
    Load any OpenML dataset by its numeric ID.

    The dataset is split into train/test with a random stratified split
    (stratified for classification, random for regression).

    Args:
        openml_id: OpenML dataset ID (e.g. 42570).
        name: Human-readable name (defaults to OpenML dataset name).
        task_type: ``"classification"`` or ``"regression"``. Auto-detected
            from the OpenML metadata if not provided.
        test_size: Fraction of data for the test split.
        max_samples: Optional cap on total samples before splitting.
        seed: Random seed for splitting and subsampling.

    Returns:
        TabularDataset with IID random split.
    """
    openml = _ensure_openml()

    # Check disk space before attempting download (avoids cryptic Errno 122)
    cache_dir = openml.config.get_cache_directory()
    _check_disk_space(os.path.dirname(cache_dir) if os.path.exists(cache_dir) else os.path.expanduser("~"))

    dataset = openml.datasets.get_dataset(
        openml_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=True,
    )

    X, y, cat_indicator, feature_names = dataset.get_data(
        target=dataset.default_target_attribute,
    )

    if X is None or y is None:
        raise ValueError(
            f"OpenML dataset {openml_id} returned empty data. "
            f"Check the dataset page: https://www.openml.org/d/{openml_id}"
        )

    # Convert to DataFrame if not already
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)
    X = _apply_openml_feature_value_repairs(X, openml_id)

    # Convert target to numpy
    y_arr = np.asarray(y)

    # Auto-detect task type from target dtype
    if task_type is None:
        if y_arr.dtype.kind == "b" or pd.api.types.is_bool_dtype(y):
            # Boolean targets (e.g. True/False defect or purchase labels) are
            # always binary classification. Without this branch, dtype kind
            # ``b`` does not enter the object, string or integer branches and
            # would be classified as regression.
            task_type = "classification"
        elif y_arr.dtype.kind in ("O", "U", "S") or pd.api.types.is_categorical_dtype(y):
            task_type = "classification"
        elif len(np.unique(y_arr)) <= 20 and y_arr.dtype.kind in ("i", "u"):
            task_type = "classification"
        else:
            task_type = "regression"

    # Encode string labels for classification
    if task_type == "classification" and y_arr.dtype.kind in ("O", "U", "S"):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_arr = le.fit_transform(y_arr)

    # Cap total samples
    if max_samples and len(X) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(X), max_samples, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y_arr = y_arr[idx]

    # Train/test split
    from sklearn.model_selection import train_test_split

    stratify = y_arr if task_type == "classification" else None
    # Guard against too few samples per class for stratification
    if stratify is not None:
        _, counts = np.unique(stratify, return_counts=True)
        if counts.min() < 2:
            stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_arr, test_size=test_size, random_state=seed, stratify=stratify,
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    # Drop likely ID columns (all-unique string columns)
    id_cols = []
    for col in X_train.columns:
        if X_train[col].dtype == object or hasattr(X_train[col], "cat"):
            if X_train[col].nunique() >= len(X_train) * 0.9:
                id_cols.append(col)
    if id_cols:
        X_train = X_train.drop(columns=id_cols)
        X_test = X_test.drop(columns=id_cols)
        if cat_indicator is not None:
            col_mask = [c not in id_cols for c in feature_names]
            cat_indicator = [ci for ci, keep in zip(cat_indicator, col_mask) if keep]
            feature_names = [fn for fn, keep in zip(feature_names, col_mask) if keep]

    # Identify and validate categorical and numerical features. Configured
    # TabArena overrides correct known OpenML metadata errors.
    cat_features, num_features = _resolve_openml_feature_types(
        X_train,
        cat_indicator,
        feature_names,
        openml_id,
    )

    ds_name = name or dataset.name or f"openml_{openml_id}"

    return TabularDataset(
        name=ds_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cat_features=cat_features,
        num_features=num_features,
        task_type=task_type,
        shift_description="IID random split (no natural shift)",
        metadata={
            "source": "openml",
            "openml_id": openml_id,
            "openml_name": dataset.name,
            "n_total": len(X_train) + len(X_test),
            "test_size": test_size,
            "feature_type_repairs": _openml_feature_repair_notes(openml_id),
        },
    )


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------

def load_tschalzev_dataset(
    short_name: str,
    **kwargs,
) -> TabularDataset:
    """
    Load one of the Tschalzev et al. (2024) datasets by short name.

    Available short names: mbgm, svpc, aeac, ogpcc, sctp, pssdp

    All keyword arguments are forwarded to :func:`load_openml_dataset`.
    """
    if short_name not in TSCHALZEV_DATASETS:
        raise ValueError(
            f"Unknown Tschalzev dataset '{short_name}'. "
            f"Available: {list(TSCHALZEV_DATASETS.keys())}"
        )
    info = TSCHALZEV_DATASETS[short_name]
    kwargs.setdefault("task_type", info["task_type"])
    kwargs.setdefault("name", f"tschalzev_{short_name}")
    return load_openml_dataset(openml_id=info["openml_id"], **kwargs)


def load_tabarena_dataset(
    openml_id: int,
    **kwargs,
) -> TabularDataset:
    """
    Load a single TabArena dataset by its OpenML ID.

    All keyword arguments are forwarded to :func:`load_openml_dataset`.
    """
    kwargs.setdefault("name", f"tabarena_{openml_id}")
    return load_openml_dataset(openml_id=openml_id, **kwargs)


def list_tabarena_task_ids() -> list[int]:
    """
    Fetch the list of all OpenML task IDs in the TabArena study (Study 457).

    Requires the ``openml`` package and network access.

    Returns:
        List of OpenML task IDs.
    """
    openml = _ensure_openml()
    study = openml.study.get_suite(TABARENA_STUDY_ID)
    return list(study.data)


def list_tschalzev_datasets() -> list[str]:
    """Return available Tschalzev dataset short names."""
    return list(TSCHALZEV_DATASETS.keys())

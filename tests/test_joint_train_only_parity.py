"""Parity tests for joint versions and train-only versions.

The train-only versions in ``src/methods/baselines.py`` must mirror the column
structure of their joint versions in ``src/methods/tier1.py``.
Otherwise, differences in retained categorical columns would confound the
decomposition of encoding value and test-time value.

For every evaluated train-only version, the original categorical columns must
remain in ``result.feature_names`` and ``result.categorical_features``. Each
joint version and train-only version must also declare the same categorical
columns. The comparison then differs only in the statistics used to construct
the features.

On an all-numerical dataset, the frequency and target train-only versions
declare ``categorical_features is None`` and pass the input through unchanged.

The train-only versions stay strictly inductive. The preserved categorical
columns are label-encoded with a training-fitted encoder
(``_label_encode_cats_train_only``). No test labels or joint statistics are
involved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.baselines import get_baseline_method
from src.methods.tier1 import TIER1_METHODS


# ---------------------------------------------------------------------------
# Train-only method key to joint method key.
# ---------------------------------------------------------------------------

TRAIN_ONLY_TO_JOINT: dict[str, str] = {
    "freq_enc_train_only": "joint_freq_combined",
    "target_enc_standard_m10": "target_enc_ratio",
    "target_enc_standard_m20": "target_enc_ratio",
    "groupby_train_only": "joint_groupby_combined",
    "pca_train_only_10": "joint_pca_10",
    "pca_train_only_var90": "joint_pca_var90",
}

TRAIN_ONLY_KEYS = list(TRAIN_ONLY_TO_JOINT.keys())


@pytest.fixture
def tiny_mixed():
    """Small mixed categorical and numeric frame with categories first for column-order checks."""
    X_train = pd.DataFrame({
        "cat_a": ["a", "b", "a", "c", "b", "a", "c", "b"],
        "cat_b": ["x", "y", "x", "z", "y", "z", "x", "y"],
        "num0": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "num1": [1.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
    })
    X_test = pd.DataFrame({
        "cat_a": ["a", "c", "b", "b"],
        "cat_b": ["y", "z", "x", "x"],
        "num0": [0.5, 1.5, 2.5, 3.5],
        "num1": [1.2, 2.2, 3.2, 4.2],
    })
    y_train = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    return X_train, X_test, y_train


@pytest.fixture
def all_numeric():
    """All-numeric frame for the no-op guard."""
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame({
        "num0": rng.normal(size=10),
        "num1": rng.normal(size=10),
        "num2": rng.normal(size=10),
    })
    X_test = pd.DataFrame({
        "num0": rng.normal(size=5),
        "num1": rng.normal(size=5),
        "num2": rng.normal(size=5),
    })
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return X_train, X_test, y_train


# ---------------------------------------------------------------------------
# Parity invariants on the mixed frame.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("train_only_key", TRAIN_ONLY_KEYS)
def test_original_cat_names_in_feature_names(tiny_mixed, train_only_key):
    """Every original categorical column must remain in feature_names."""
    X_train, X_test, y_train = tiny_mixed
    result = get_baseline_method(train_only_key).fit_transform(X_train, X_test, y_train)
    for cat_col in ("cat_a", "cat_b"):
        assert cat_col in result.feature_names, (
            f"{train_only_key}: original cat '{cat_col}' missing from feature_names "
            f"{result.feature_names}"
        )


@pytest.mark.parametrize("train_only_key", TRAIN_ONLY_KEYS)
def test_categorical_features_equals_original_cats(tiny_mixed, train_only_key):
    """The train-only version must declare the original categorical columns in order."""
    X_train, X_test, y_train = tiny_mixed
    result = get_baseline_method(train_only_key).fit_transform(X_train, X_test, y_train)
    assert result.categorical_features == ["cat_a", "cat_b"], (
        f"{train_only_key}: declared cats {result.categorical_features} != "
        f"['cat_a', 'cat_b']"
    )


@pytest.mark.parametrize("train_only_key", TRAIN_ONLY_KEYS)
def test_train_only_declares_same_cats_as_joint_version(tiny_mixed, train_only_key):
    """The train-only and joint versions must declare the same categorical columns.

    The pair may differ only in train-only versus joint statistics. It must not
    differ in which columns reach the model as categorical inputs.
    """
    X_train, X_test, y_train = tiny_mixed
    joint_key = TRAIN_ONLY_TO_JOINT[train_only_key]

    train_only_result = get_baseline_method(train_only_key).fit_transform(
        X_train, X_test, y_train
    )
    joint_result = TIER1_METHODS[joint_key]().fit_transform(
        X_train, X_test, y_train
    )

    assert train_only_result.categorical_features is not None, (
        f"{train_only_key}: train-only version declares no categorical columns"
    )
    assert joint_result.categorical_features is not None, (
        f"{joint_key}: joint version declares no categorical columns"
    )
    assert set(train_only_result.categorical_features) == set(
        joint_result.categorical_features
    ), (
        f"{train_only_key} declares {sorted(train_only_result.categorical_features)} but "
        f"joint version {joint_key} declares "
        f"{sorted(joint_result.categorical_features)}"
    )


# ---------------------------------------------------------------------------
# All-numerical input leaves the frequency and target train-only versions
# unchanged. Group-by aggregation uses low-cardinality numerical groups, and
# PCA appends components, so those methods are not included here.
# ---------------------------------------------------------------------------

NOOP_TRAIN_ONLY_KEYS = [
    "freq_enc_train_only",
    "target_enc_standard_m10",
    "target_enc_standard_m20",
]


@pytest.mark.parametrize("train_only_key", NOOP_TRAIN_ONLY_KEYS)
def test_all_numeric_is_noop(all_numeric, train_only_key):
    """With no categorical columns, train-only encoders must pass inputs through."""
    X_train, X_test, y_train = all_numeric
    result = get_baseline_method(train_only_key).fit_transform(X_train, X_test, y_train)

    assert result.categorical_features is None, (
        f"{train_only_key}: expected None on all-numeric data, got "
        f"{result.categorical_features}"
    )
    assert result.feature_names == list(X_train.columns), (
        f"{train_only_key}: feature_names changed on all-numeric data"
    )
    np.testing.assert_array_equal(
        result.X_train, X_train.values.astype(float),
        err_msg=f"{train_only_key}: X_train values changed on all-numeric data",
    )
    np.testing.assert_array_equal(
        result.X_test, X_test.values.astype(float),
        err_msg=f"{train_only_key}: X_test values changed on all-numeric data",
    )

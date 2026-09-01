"""Preservation invariants for categorical feature engineering.

Categorical feature-engineering methods add columns while retaining the original
categorical features. This preserves information for models with native
categorical handling.

For each of the eight covered Tier 1 classes (M1, M2v2, M4, M4v2, M14, M14b,
M14bv2, M14v2) this module asserts the same three invariants:

1. ``result.categorical_features`` equals the original cat list (so cat-aware
   models get the native-split hint).
2. Every original categorical column appears in ``result.feature_names`` (so
   downstream pipelines and ``CatPreservingMethodPipeline`` see them).
3. ``result.X_train.shape[1] >= input.X_train.shape[1]`` (engineered columns
   appear alongside the originals, with no net feature loss).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic import load_synthetic
from src.methods.column_selection import ColumnSelection, ColumnSelector
from src.methods.selective import SelectiveMethod
from src.methods.tier1 import (
    FrequencyRatioImproved,
    GroupByShiftCohenD,
    GroupByShiftRatio,
    JointFrequencyEncoding,
    JointGroupByFeatures,
    ShiftRegularizedTargetEncoding,
    ShiftRegularizedTargetEncodingOOF,
    TargetedGroupByFeatures,
)


@pytest.fixture
def ds():
    return load_synthetic(n_train=300, n_test=200, shift_strength=1.0, seed=42)


# ---------------------------------------------------------------------------
# One parametrized run checks the shared invariant for each fixed class.
# ---------------------------------------------------------------------------

PRESERVING_METHODS: list[tuple[str, callable]] = [
    ("M1_JointFrequencyEncoding", lambda: JointFrequencyEncoding(variant="combined")),
    ("M2v2_FrequencyRatioImproved", lambda: FrequencyRatioImproved(mode="ratio")),
    ("M4_ShiftRegularizedTargetEncoding", lambda: ShiftRegularizedTargetEncoding(regularization_mode="ratio")),
    ("M4v2_ShiftRegularizedTargetEncodingOOF", lambda: ShiftRegularizedTargetEncodingOOF(regularization_mode="ratio")),
    ("M14_JointGroupByFeatures", lambda: JointGroupByFeatures(variant="combined")),
    ("M14b_GroupByShiftRatio", lambda: GroupByShiftRatio(use_log=True)),
    ("M14bv2_GroupByShiftCohenD", lambda: GroupByShiftCohenD()),
    ("M14v2_TargetedGroupByFeatures", lambda: TargetedGroupByFeatures(top_k_groups=3)),
]


@pytest.mark.parametrize("name,factory", PRESERVING_METHODS, ids=[n for n, _ in PRESERVING_METHODS])
def test_categorical_features_equals_original_list(ds, name, factory):
    """``result.categorical_features`` must equal the original cat list."""
    method = factory()
    result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
    assert result.categorical_features is not None, \
        f"{name}: categorical_features must be declared (got None)"
    assert set(result.categorical_features) == set(ds.cat_features), \
        f"{name}: declared cats {result.categorical_features} != original {ds.cat_features}"


@pytest.mark.parametrize("name,factory", PRESERVING_METHODS, ids=[n for n, _ in PRESERVING_METHODS])
def test_original_cat_names_in_feature_names(ds, name, factory):
    """Original categorical columns must appear in ``result.feature_names``."""
    method = factory()
    result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
    for cat_col in ds.cat_features:
        assert cat_col in result.feature_names, \
            f"{name}: original cat '{cat_col}' missing from feature_names"


@pytest.mark.parametrize("name,factory", PRESERVING_METHODS, ids=[n for n, _ in PRESERVING_METHODS])
def test_no_net_feature_loss(ds, name, factory):
    """Output must contain at least as many columns as the input."""
    method = factory()
    result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
    assert result.X_train.shape[1] >= ds.X_train.shape[1], \
        f"{name}: output {result.X_train.shape[1]} cols < input {ds.X_train.shape[1]} cols"


# ---------------------------------------------------------------------------
# Exact column-count checks for methods whose output shape is determined by
# n_num and n_cat.
# ---------------------------------------------------------------------------

class TestExactShapeM1:
    def test_combined_adds_one_freq_per_cat(self, ds):
        result = JointFrequencyEncoding(variant="combined").fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        n_cat = len(ds.cat_features)
        n_num = len(ds.num_features)
        assert result.X_train.shape[1] == n_num + 2 * n_cat

    def test_separate_adds_two_freq_per_cat(self, ds):
        result = JointFrequencyEncoding(variant="separate").fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        n_cat = len(ds.cat_features)
        n_num = len(ds.num_features)
        assert result.X_train.shape[1] == n_num + 3 * n_cat


class TestExactShapeM4:
    def test_ratio_adds_one_enc_per_cat(self, ds):
        result = ShiftRegularizedTargetEncoding(regularization_mode="ratio").fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        n_cat = len(ds.cat_features)
        n_num = len(ds.num_features)
        assert result.X_train.shape[1] == n_num + 2 * n_cat


class TestExactShapeM4v2:
    def test_oof_adds_one_enc_per_cat(self, ds):
        result = ShiftRegularizedTargetEncodingOOF(regularization_mode="ratio").fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        n_cat = len(ds.cat_features)
        n_num = len(ds.num_features)
        assert result.X_train.shape[1] == n_num + 2 * n_cat


# ---------------------------------------------------------------------------
# Minimal mixed categorical and numerical input
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_mixed():
    X_train = pd.DataFrame({
        "cat_a": ["a", "b", "a", "c", "b"],
        "cat_b": ["x", "y", "x", "z", "y"],
        "num0": [0.0, 1.0, 2.0, 3.0, 4.0],
        "num1": [1.0, 1.5, 2.5, 3.5, 4.5],
    })
    X_test = pd.DataFrame({
        "cat_a": ["a", "c", "b"],
        "cat_b": ["y", "z", "x"],
        "num0": [0.5, 1.5, 2.5],
        "num1": [1.2, 2.2, 3.2],
    })
    y_train = np.array([0, 1, 0, 1, 0])
    return X_train, X_test, y_train


@pytest.mark.parametrize("name,factory", PRESERVING_METHODS, ids=[n for n, _ in PRESERVING_METHODS])
def test_tiny_mixed_preserves_cats(tiny_mixed, name, factory):
    """Both categorical columns must remain in a two-by-two mixed DataFrame."""
    X_train, X_test, y_train = tiny_mixed
    result = factory().fit_transform(X_train, X_test, y_train)
    assert "cat_a" in result.feature_names, f"{name}: 'cat_a' missing"
    assert "cat_b" in result.feature_names, f"{name}: 'cat_b' missing"
    assert result.categorical_features is not None
    assert set(result.categorical_features) == {"cat_a", "cat_b"}, \
        f"{name}: declared cats {result.categorical_features}"


# ---------------------------------------------------------------------------
# SelectiveMethod must declare categorical passthrough columns.
#
# SelectiveMethod must declare both inner-method and passthrough categorical
# columns in the final TransformResult. The wrapper retains unselected
# categoricals as label-encoded columns, and raw-feature models rely on the
# declaration to restore their native categorical representation.
# ---------------------------------------------------------------------------


class _ExplicitSelector(ColumnSelector):
    """Deterministic selector that returns the specified columns."""

    def __init__(self, cat_cols: list[str], num_cols: list[str]):
        self._cat = list(cat_cols)
        self._num = list(num_cols)

    @property
    def name(self) -> str:
        return f"explicit({'+'.join(self._cat + self._num)})"

    def select(self, X_train, X_test, y_train) -> ColumnSelection:
        return ColumnSelection(cat_cols=self._cat, num_cols=self._num, scores={})


@pytest.mark.parametrize("name,factory", PRESERVING_METHODS, ids=[n for n, _ in PRESERVING_METHODS])
def test_selective_wrapper_declares_passthrough_cats_tiny(tiny_mixed, name, factory):
    """Selector picks {cat_a, num0} → cat_b is passthrough. Both cats must be declared."""
    X_train, X_test, y_train = tiny_mixed
    selector = _ExplicitSelector(cat_cols=["cat_a"], num_cols=["num0"])
    wrapped = SelectiveMethod(method=factory(), selector=selector)
    result = wrapped.fit_transform(X_train, X_test, y_train)

    # Both categorical columns must remain in feature_names. The inner method
    # preserves cat_a. The wrapper passes cat_b through.
    assert "cat_a" in result.feature_names, f"{name}: selected cat 'cat_a' missing"
    assert "cat_b" in result.feature_names, f"{name}: passthrough cat 'cat_b' missing"

    # Declare selected cat_a from the inner method and passthrough cat_b.
    assert result.categorical_features is not None, \
        f"{name}: categorical_features must be declared (got None)"
    assert set(result.categorical_features) == {"cat_a", "cat_b"}, \
        f"{name}: declared cats {result.categorical_features} != " \
        f"expected {{'cat_a', 'cat_b'}} (selected + passthrough)"


def test_selective_wrapper_declares_all_passthrough_cats_synthetic(ds):
    """Synthetic ds has 3 cats. Select 1 + 1 num → 2 passthrough cats must be declared."""
    selected_cat = ds.cat_features[0]  # e.g. OCCP
    selected_num = ds.num_features[0]  # e.g. AGEP
    expected_passthrough = set(ds.cat_features) - {selected_cat}  # MAR, ST

    selector = _ExplicitSelector(cat_cols=[selected_cat], num_cols=[selected_num])
    wrapped = SelectiveMethod(
        method=JointFrequencyEncoding(variant="combined"),
        selector=selector,
    )
    result = wrapped.fit_transform(ds.X_train, ds.X_test, ds.y_train)

    # All 3 cats must be declared (1 selected + 2 passthrough).
    assert result.categorical_features is not None
    assert set(result.categorical_features) == set(ds.cat_features), \
        f"declared cats {result.categorical_features} != original {ds.cat_features} " \
        f"(selected={selected_cat}, expected passthrough={expected_passthrough})"

    # All three categorical columns must appear in feature_names.
    for cat_col in ds.cat_features:
        assert cat_col in result.feature_names, \
            f"cat '{cat_col}' missing from feature_names"


def test_selective_wrapper_stable_order_in_declared_cats(tiny_mixed):
    """Declared cats follow dict.fromkeys order: inner-method cats first, then passthrough."""
    X_train, X_test, y_train = tiny_mixed
    selector = _ExplicitSelector(cat_cols=["cat_a"], num_cols=["num0"])
    wrapped = SelectiveMethod(
        method=JointFrequencyEncoding(variant="combined"),
        selector=selector,
    )
    result = wrapped.fit_transform(X_train, X_test, y_train)

    # Inner method's cats (cat_a) come before passthrough cats (cat_b).
    # Reproducibility / diff-friendliness depends on this order being stable.
    assert result.categorical_features == ["cat_a", "cat_b"], \
        f"order not stable: {result.categorical_features}"

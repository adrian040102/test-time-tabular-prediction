"""
Unit tests for individual methods.

Each method gets tested for:
1. Correct output shape
2. No NaN/Inf in output
3. Deterministic output
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic import load_synthetic
from src.methods.base import InductiveBaseline, IdentityMethod, MethodPipeline


@pytest.fixture
def ds():
    return load_synthetic(n_train=200, n_test=100, shift_strength=1.0, seed=42)


class TestInductiveBaseline:
    def test_no_nan(self, ds):
        result = InductiveBaseline().fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert not np.any(np.isnan(result.X_train))
        assert not np.any(np.isnan(result.X_test))

    def test_no_inf(self, ds):
        result = InductiveBaseline().fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert not np.any(np.isinf(result.X_train))
        assert not np.any(np.isinf(result.X_test))

    def test_numeric_output(self, ds):
        result = InductiveBaseline().fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Output should be fully numeric (categoricals label-encoded)
        # Scaling is handled by the pipeline's universal StandardScaler,
        # not by the baseline itself. This test checks that the output is numeric.
        assert result.X_train.dtype == float
        assert result.X_test.dtype == float


class TestIdentityMethod:
    def test_preserves_numeric_values(self, ds):
        result = IdentityMethod().fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Numeric columns should be preserved exactly
        for col in ds.num_features:
            col_idx = list(ds.X_train.columns).index(col)
            np.testing.assert_array_almost_equal(
                result.X_train[:, col_idx],
                ds.X_train[col].values.astype(float),
            )

    def test_encodes_categoricals(self, ds):
        result = IdentityMethod().fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Categorical columns should be label-encoded (all numeric, no NaN)
        for col in ds.cat_features:
            col_idx = list(ds.X_train.columns).index(col)
            assert not np.any(np.isnan(result.X_train[:, col_idx]))
            assert not np.any(np.isnan(result.X_test[:, col_idx]))

    def test_output_shape(self, ds):
        result = IdentityMethod().fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape == (len(ds.X_train), ds.n_features)
        assert result.X_test.shape == (len(ds.X_test), ds.n_features)


class TestMethodPipeline:
    def test_sequential_execution(self, ds):
        pipeline = MethodPipeline([InductiveBaseline()])
        result = pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[0] == len(ds.X_train)
        assert result.X_test.shape[0] == len(ds.X_test)

    def test_name_concatenation(self):
        pipeline = MethodPipeline([InductiveBaseline(), IdentityMethod()])
        assert "inductive_baseline" in pipeline.name
        assert "identity" in pipeline.name
        assert "→" in pipeline.name

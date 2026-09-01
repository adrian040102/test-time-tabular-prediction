"""Tests for the M18 GBDT leaf co-occurrence configurations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import TestTimeMethod as _TestTimeMethod, TransformResult
from src.methods.tier2 import LeafBasedTestDensityFE, TIER2_METHODS


# --- Fixtures ---

@pytest.fixture
def dummy_clf_data():
    """Binary classification data with mild shift."""
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(rng.randn(200, 5), columns=[f"f{i}" for i in range(5)])
    X_test = pd.DataFrame(rng.randn(100, 5) + 0.3, columns=[f"f{i}" for i in range(5)])
    y_train = rng.randint(0, 2, 200)
    return X_train, X_test, y_train


@pytest.fixture
def dummy_reg_data():
    """Regression data."""
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(rng.randn(200, 5), columns=[f"f{i}" for i in range(5)])
    X_test = pd.DataFrame(rng.randn(100, 5) + 0.3, columns=[f"f{i}" for i in range(5)])
    y_train = rng.randn(200)
    return X_train, X_test, y_train


@pytest.fixture
def small_data():
    """Small dataset (n=30) to test adaptive tree reduction."""
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(rng.randn(30, 3), columns=["a", "b", "c"])
    X_test = pd.DataFrame(rng.randn(20, 3), columns=["a", "b", "c"])
    y_train = rng.randint(0, 2, 30)
    return X_train, X_test, y_train


# --- TestLeafBasedTestDensityFE: confidence_feature mode ---

class TestConfidenceFeature:
    """Tests for mode='confidence_feature'."""

    def test_output_shapes(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = LeafBasedTestDensityFE(mode="confidence_feature")
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.X_train.shape[0] == 200
        assert result.X_test.shape[0] == 100
        assert result.X_train.shape[1] == result.X_test.shape[1]
        assert result.X_train.shape[1] == len(result.feature_names)

    def test_adds_3_features(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = LeafBasedTestDensityFE(mode="confidence_feature")
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.X_train.shape[1] == 5 + 3
        assert "leaf_minority_ratio" in result.feature_names
        assert "leaf_avg_train_frac" in result.feature_names
        assert "leaf_avg_size" in result.feature_names

    def test_no_nan_inf(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = LeafBasedTestDensityFE(mode="confidence_feature")
        result = m.fit_transform(X_train, X_test, y_train)

        assert not np.any(np.isnan(result.X_train))
        assert not np.any(np.isnan(result.X_test))
        assert not np.any(np.isinf(result.X_train))
        assert not np.any(np.isinf(result.X_test))

    def test_feature_ranges(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = LeafBasedTestDensityFE(mode="confidence_feature")
        result = m.fit_transform(X_train, X_test, y_train)

        # Last 3 columns are the new features
        minority_ratio_tr = result.X_train[:, -3]
        avg_train_frac_tr = result.X_train[:, -2]
        avg_size_tr = result.X_train[:, -1]

        assert np.all(minority_ratio_tr >= 0) and np.all(minority_ratio_tr <= 1)
        assert np.all(avg_train_frac_tr >= 0) and np.all(avg_train_frac_tr <= 1)
        assert np.all(avg_size_tr > 0)

        minority_ratio_te = result.X_test[:, -3]
        avg_train_frac_te = result.X_test[:, -2]
        assert np.all(minority_ratio_te >= 0) and np.all(minority_ratio_te <= 1)
        assert np.all(avg_train_frac_te >= 0) and np.all(avg_train_frac_te <= 1)

    def test_deterministic(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m1 = LeafBasedTestDensityFE(mode="confidence_feature", seed=42)
        m2 = LeafBasedTestDensityFE(mode="confidence_feature", seed=42)
        r1 = m1.fit_transform(X_train, X_test, y_train)
        r2 = m2.fit_transform(X_train, X_test, y_train)

        np.testing.assert_array_equal(r1.X_train, r2.X_train)
        np.testing.assert_array_equal(r1.X_test, r2.X_test)

    def test_regression(self, dummy_reg_data):
        X_train, X_test, y_train = dummy_reg_data
        m = LeafBasedTestDensityFE(mode="confidence_feature")
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.X_train.shape[0] == 200
        assert result.X_train.shape[1] == 5 + 3
        assert not np.any(np.isnan(result.X_train))

    def test_metadata(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = LeafBasedTestDensityFE(mode="confidence_feature")
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.metadata["m18_mode"] == "confidence_feature"
        assert result.metadata["m18_n_features_added"] == 3
        assert result.metadata["m18_is_classification"] is True
        assert result.metadata["m18_n_trees"] > 0


# --- TestLeafEmbeddingKNN ---

class TestLeafEmbeddingKNN:
    """Tests for mode='leaf_embedding_knn'."""

    def test_output_shapes(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=10)
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.X_train.shape[0] == 200
        assert result.X_test.shape[0] == 100
        assert result.X_train.shape[1] == result.X_test.shape[1]
        assert result.X_train.shape[1] == len(result.feature_names)

    def test_adds_2_features(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=10)
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.X_train.shape[1] == 5 + 2
        assert "leaf_hamming_dist_to_other" in result.feature_names
        assert "leaf_cross_split_frac" in result.feature_names

    def test_no_nan_inf(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=10)
        result = m.fit_transform(X_train, X_test, y_train)

        assert not np.any(np.isnan(result.X_train))
        assert not np.any(np.isnan(result.X_test))
        assert not np.any(np.isinf(result.X_train))
        assert not np.any(np.isinf(result.X_test))

    def test_feature_ranges(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=10)
        result = m.fit_transform(X_train, X_test, y_train)

        hamming_tr = result.X_train[:, -2]
        cross_frac_tr = result.X_train[:, -1]

        assert np.all(hamming_tr >= 0) and np.all(hamming_tr <= 1)
        assert np.all(cross_frac_tr >= 0) and np.all(cross_frac_tr <= 1)

    def test_small_dataset_k_capped(self, small_data):
        X_train, X_test, y_train = small_data
        m = LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=50)
        result = m.fit_transform(X_train, X_test, y_train)

        # The method caps k internally for small datasets.
        assert result.X_train.shape[0] == 30
        assert result.X_test.shape[0] == 20

    def test_metadata(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=10)
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.metadata["m18_mode"] == "leaf_embedding_knn"
        assert result.metadata["m18_n_features_added"] == 2


# --- TestEdgeCases ---

class TestEdgeCases:
    """Edge case tests."""

    def test_small_dataset_reduces_trees(self, small_data):
        X_train, X_test, y_train = small_data
        m = LeafBasedTestDensityFE(mode="confidence_feature", n_estimators=100)
        result = m.fit_transform(X_train, X_test, y_train)

        # n_train=30 < 50 → effective_n_estimators capped at 20
        assert result.metadata["m18_effective_n_estimators"] == 20
        assert result.metadata["m18_n_trees"] <= 20

    def test_uses_labels_property(self):
        m = LeafBasedTestDensityFE()
        assert m.uses_labels is True

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            LeafBasedTestDensityFE(mode="invalid")

    def test_name_property(self):
        m1 = LeafBasedTestDensityFE(mode="confidence_feature", max_depth=6, n_estimators=100)
        assert m1.name == "leaf_density_conf_d6_t100"

        m2 = LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=10, max_depth=6)
        assert m2.name == "leaf_density_embed_k10_d6_t100"

    def test_isinstance_test_time_method(self):
        m = LeafBasedTestDensityFE()
        assert isinstance(m, _TestTimeMethod)

    def test_data_with_nans(self):
        """Ensure NaN values in input do not crash the method."""
        rng = np.random.RandomState(42)
        X_train = pd.DataFrame(rng.randn(100, 3), columns=["a", "b", "c"])
        X_test = pd.DataFrame(rng.randn(50, 3), columns=["a", "b", "c"])
        X_train.iloc[0, 0] = np.nan
        X_train.iloc[5, 2] = np.nan
        X_test.iloc[3, 1] = np.nan
        y_train = rng.randint(0, 2, 100)

        m = LeafBasedTestDensityFE(mode="confidence_feature")
        result = m.fit_transform(X_train, X_test, y_train)

        assert not np.any(np.isnan(result.X_train[:, -3:]))
        assert not np.any(np.isnan(result.X_test[:, -3:]))


# --- Method lookup tests ---

class TestMethodLookup:
    """Test that all configured M18 methods instantiate and run."""

    M18_KEYS = [k for k in TIER2_METHODS if k.startswith("leaf_density")]

    def test_all_configurations_are_available(self):
        assert len(self.M18_KEYS) == 8

    @pytest.mark.parametrize("key", [k for k in TIER2_METHODS if k.startswith("leaf_density")])
    def test_variant_runs(self, key, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        method = TIER2_METHODS[key]()
        result = method.fit_transform(X_train, X_test, y_train)

        assert isinstance(result, TransformResult)
        assert result.X_train.shape[0] == 200
        assert result.X_test.shape[0] == 100
        assert result.X_train.shape[1] == result.X_test.shape[1]
        assert not np.any(np.isnan(result.X_train))
        assert not np.any(np.isnan(result.X_test))

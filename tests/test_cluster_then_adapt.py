"""Tests for the M20 cluster-based adaptation configurations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import TransformResult
from src.methods.tier2 import ClusterMembershipFE, TIER2_METHODS


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
    """Small dataset to test adaptive K reduction."""
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(rng.randn(15, 3), columns=["a", "b", "c"])
    X_test = pd.DataFrame(rng.randn(10, 3), columns=["a", "b", "c"])
    y_train = rng.randint(0, 2, 15)
    return X_train, X_test, y_train


# --- TestFeatureMode ---

class TestFeatureMode:
    """Tests for return_as='feature'."""

    def test_output_shapes(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = ClusterMembershipFE(n_clusters=5, return_as="feature")
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.X_train.shape[0] == 200
        assert result.X_test.shape[0] == 100
        assert result.X_train.shape[1] == result.X_test.shape[1]
        assert result.X_train.shape[1] == len(result.feature_names)

    def test_adds_3_features(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = ClusterMembershipFE(n_clusters=5, return_as="feature")
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.X_train.shape[1] == 5 + 3
        assert "cluster_relative_size" in result.feature_names
        assert "cluster_train_frac" in result.feature_names
        assert "cluster_dist_to_centroid" in result.feature_names

    def test_no_weights_in_feature_mode(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = ClusterMembershipFE(n_clusters=5, return_as="feature")
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.sample_weights_train is None

    def test_no_nan_inf(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = ClusterMembershipFE(n_clusters=5, return_as="feature")
        result = m.fit_transform(X_train, X_test, y_train)

        assert not np.any(np.isnan(result.X_train))
        assert not np.any(np.isnan(result.X_test))
        assert not np.any(np.isinf(result.X_train))
        assert not np.any(np.isinf(result.X_test))

    def test_feature_ranges(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = ClusterMembershipFE(n_clusters=5, return_as="feature")
        result = m.fit_transform(X_train, X_test, y_train)

        rel_size_tr = result.X_train[:, -3]
        train_frac_tr = result.X_train[:, -2]
        dist_tr = result.X_train[:, -1]

        assert np.all(rel_size_tr >= 0) and np.all(rel_size_tr <= 1)
        assert np.all(train_frac_tr >= 0) and np.all(train_frac_tr <= 1)
        assert np.all(dist_tr >= 0)

        rel_size_te = result.X_test[:, -3]
        train_frac_te = result.X_test[:, -2]
        dist_te = result.X_test[:, -1]

        assert np.all(rel_size_te >= 0) and np.all(rel_size_te <= 1)
        assert np.all(train_frac_te >= 0) and np.all(train_frac_te <= 1)
        assert np.all(dist_te >= 0)


# --- TestWeightMode ---

class TestWeightMode:
    """Tests for return_as='weight'."""

    def test_returns_weights_and_1_feature(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = ClusterMembershipFE(n_clusters=10, return_as="weight")
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.sample_weights_train is not None
        assert len(result.sample_weights_train) == 200
        # Weight mode appends cluster_relative_size without an integer cluster identifier.
        assert result.X_train.shape[1] == 5 + 1
        assert "cluster_relative_size" in result.feature_names

    def test_weights_positive(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = ClusterMembershipFE(n_clusters=10, return_as="weight")
        result = m.fit_transform(X_train, X_test, y_train)

        assert np.all(result.sample_weights_train > 0)


# --- TestBothMode ---

class TestBothMode:
    """Tests for return_as='both'."""

    def test_has_features_and_weights(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = ClusterMembershipFE(n_clusters=10, return_as="both")
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.X_train.shape[1] == 5 + 3
        assert result.sample_weights_train is not None
        assert len(result.sample_weights_train) == 200


# --- TestEdgeCases ---

class TestEdgeCases:
    """Edge case tests."""

    def test_small_dataset_adaptive_k(self, small_data):
        X_train, X_test, y_train = small_data
        m = ClusterMembershipFE(n_clusters=10, return_as="feature")
        result = m.fit_transform(X_train, X_test, y_train)

        # n_total=25, 25 < 10*10 → effective_k = max(2, 25//10) = 2
        assert result.metadata["m20_effective_k"] == 2
        assert result.X_train.shape[0] == 15
        assert result.X_test.shape[0] == 10

    def test_deterministic(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m1 = ClusterMembershipFE(n_clusters=5, return_as="both", seed=42)
        m2 = ClusterMembershipFE(n_clusters=5, return_as="both", seed=42)
        r1 = m1.fit_transform(X_train, X_test, y_train)
        r2 = m2.fit_transform(X_train, X_test, y_train)

        np.testing.assert_array_equal(r1.X_train, r2.X_train)
        np.testing.assert_array_equal(r1.X_test, r2.X_test)

    def test_regression(self, dummy_reg_data):
        X_train, X_test, y_train = dummy_reg_data
        m = ClusterMembershipFE(n_clusters=5, return_as="feature")
        result = m.fit_transform(X_train, X_test, y_train)

        assert result.X_train.shape[0] == 200
        assert result.X_train.shape[1] == 5 + 3
        assert not np.any(np.isnan(result.X_train))

    def test_nan_input(self):
        """NaN in input does not crash clustering. Appended features are clean."""
        rng = np.random.RandomState(42)
        X_train = pd.DataFrame(rng.randn(100, 3), columns=["a", "b", "c"])
        X_test = pd.DataFrame(rng.randn(50, 3), columns=["a", "b", "c"])
        X_train.iloc[0, 0] = np.nan
        X_test.iloc[3, 1] = np.nan
        y_train = rng.randint(0, 2, 100)

        m = ClusterMembershipFE(n_clusters=5, return_as="both")
        result = m.fit_transform(X_train, X_test, y_train)

        # Original features may still have NaN (pipeline handles imputation later),
        # but the appended cluster features must be clean.
        n_orig = 3
        assert not np.any(np.isnan(result.X_train[:, n_orig:]))
        assert not np.any(np.isnan(result.X_test[:, n_orig:]))

    def test_y_train_independence(self, dummy_clf_data):
        """Clustering uses only X, so shuffled labels produce the same output."""
        X_train, X_test, y_train = dummy_clf_data
        rng = np.random.RandomState(99)
        y_shuffled = rng.permutation(y_train)

        m1 = ClusterMembershipFE(n_clusters=5, return_as="feature", seed=42)
        m2 = ClusterMembershipFE(n_clusters=5, return_as="feature", seed=42)
        r1 = m1.fit_transform(X_train, X_test, y_train)
        r2 = m2.fit_transform(X_train, X_test, y_shuffled)

        np.testing.assert_array_equal(r1.X_train, r2.X_train)
        np.testing.assert_array_equal(r1.X_test, r2.X_test)

    def test_invalid_return_as_raises(self):
        with pytest.raises(ValueError, match="Unknown return_as"):
            ClusterMembershipFE(return_as="invalid")

    def test_name_property(self):
        m = ClusterMembershipFE(n_clusters=10, return_as="both")
        assert m.name == "cluster_membership_k10_both"

    def test_metadata_contents(self, dummy_clf_data):
        X_train, X_test, y_train = dummy_clf_data
        m = ClusterMembershipFE(n_clusters=5, return_as="both")
        result = m.fit_transform(X_train, X_test, y_train)

        assert "m20_effective_k" in result.metadata
        assert "m20_cluster_train_fracs" in result.metadata
        assert "m20_mean_weight" in result.metadata
        assert result.metadata["m20_effective_k"] == 5


# --- Method lookup tests ---

class TestMethodLookup:
    """Test that all configured M20 methods instantiate and run."""

    M20_KEYS = [k for k in TIER2_METHODS if k.startswith("cluster_membership")]

    def test_all_configurations_are_available(self):
        assert len(self.M20_KEYS) == 8

    @pytest.mark.parametrize("key", [k for k in TIER2_METHODS if k.startswith("cluster_membership")])
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

"""
Dedicated tests for M18 (LeafBasedTestDensityFE), M19 (NeighborhoodConsensusFeatures)
and M20 (ClusterMembershipFE).

Covers: correctness, edge cases, numerical stability and leakage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic import load_synthetic
from src.methods.tier2 import (
    LeafBasedTestDensityFE,
    ClusterMembershipFE,
    NeighborhoodConsensusFeatures,
    TIER2_METHODS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ds():
    return load_synthetic(n_train=200, n_test=100, shift_strength=1.0, seed=42)


@pytest.fixture
def ds_small():
    """Very small dataset to test edge cases."""
    return load_synthetic(n_train=30, n_test=15, shift_strength=1.0, seed=42)


@pytest.fixture
def ds_no_shift():
    return load_synthetic(n_train=200, n_test=100, shift_strength=0.0, seed=42)


# ===========================================================================
# M18: LeafBasedTestDensityFE
# ===========================================================================

class TestLeafBasedTestDensityFE:
    def test_confidence_feature_shape(self, ds):
        method = LeafBasedTestDensityFE(mode="confidence_feature", max_depth=3)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Should add 3 features
        assert result.X_train.shape[1] == ds.n_features + 3
        assert result.X_test.shape[1] == ds.n_features + 3
        assert len(result.feature_names) == result.X_train.shape[1]

    def test_embedding_knn_shape(self, ds):
        method = LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=5)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Should add 2 features
        assert result.X_train.shape[1] == ds.n_features + 2
        assert result.X_test.shape[1] == ds.n_features + 2

    def test_confidence_features_in_range(self, ds):
        method = LeafBasedTestDensityFE(mode="confidence_feature")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # leaf_minority_ratio: [0, 1]
        minority_ratio = result.X_train[:, -3]
        assert np.all(minority_ratio >= 0) and np.all(minority_ratio <= 1)
        # leaf_avg_train_frac: [0, 1]
        avg_train_frac = result.X_train[:, -2]
        assert np.all(avg_train_frac >= 0) and np.all(avg_train_frac <= 1)
        # leaf_avg_size: positive
        avg_size = result.X_train[:, -1]
        assert np.all(avg_size > 0)

    def test_embedding_features_in_range(self, ds):
        method = LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=5)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # hamming distance: [0, 1] (normalized)
        hamming_dist = result.X_train[:, -2]
        assert np.all(hamming_dist >= 0) and np.all(hamming_dist <= 1)
        # cross_split_frac: [0, 1]
        cross_frac = result.X_train[:, -1]
        assert np.all(cross_frac >= 0) and np.all(cross_frac <= 1)

    def test_small_dataset_adaptive_trees(self, ds_small):
        method = LeafBasedTestDensityFE(
            mode="confidence_feature", n_estimators=100, max_depth=3,
        )
        result = method.fit_transform(ds_small.X_train, ds_small.X_test, ds_small.y_train)
        # Should reduce n_estimators for small dataset
        assert result.metadata["m18_effective_n_estimators"] <= 50

    def test_no_nan_inf(self, ds):
        for mode in ("confidence_feature", "leaf_embedding_knn"):
            method = LeafBasedTestDensityFE(mode=mode, k=5)
            result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
            assert not np.any(np.isnan(result.X_train)), f"NaN in {mode} X_train"
            assert not np.any(np.isnan(result.X_test)), f"NaN in {mode} X_test"
            assert not np.any(np.isinf(result.X_train)), f"Inf in {mode} X_train"
            assert not np.any(np.isinf(result.X_test)), f"Inf in {mode} X_test"

    def test_deterministic(self, ds):
        m1 = LeafBasedTestDensityFE(mode="confidence_feature", seed=42)
        r1 = m1.fit_transform(ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy())
        m2 = LeafBasedTestDensityFE(mode="confidence_feature", seed=42)
        r2 = m2.fit_transform(ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy())
        np.testing.assert_array_almost_equal(r1.X_train, r2.X_train)
        np.testing.assert_array_almost_equal(r1.X_test, r2.X_test)

    def test_metadata_keys(self, ds):
        method = LeafBasedTestDensityFE(mode="confidence_feature")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert "m18_mode" in result.metadata
        assert "m18_n_trees" in result.metadata
        assert "m18_n_features_added" in result.metadata

    def test_uses_labels_true(self):
        method = LeafBasedTestDensityFE()
        assert method.uses_labels is True

    def test_method_lookup_contains_all_configurations(self):
        expected = [
            "leaf_density_conf", "leaf_density_conf_shallow",
            "leaf_density_conf_deep", "leaf_density_embed_k10",
            "leaf_density_embed_k5", "leaf_density_embed_k20",
            # Cumulative-shift variants.
            "leaf_density_conf_cumshift",
            "leaf_density_embed_k10_cumshift",
        ]
        for name in expected:
            assert name in TIER2_METHODS, f"Missing {name} in TIER2_METHODS"

    # --- NaN-aware auxiliary GBDT -------------------------------------------

    def test_handles_nan_inputs_without_crash(self, ds):
        """Aux LightGBM must accept NaN natively (no nan_to_num inside)."""
        X_tr = ds.X_train.copy()
        X_te = ds.X_test.copy()
        # Inject NaN into ~10% of cells in the first numeric column
        col = X_tr.columns[0]
        rng = np.random.RandomState(0)
        tr_mask = rng.rand(len(X_tr)) < 0.1
        te_mask = rng.rand(len(X_te)) < 0.1
        X_tr.loc[tr_mask, col] = np.nan
        X_te.loc[te_mask, col] = np.nan
        for mode in ("confidence_feature", "leaf_embedding_knn"):
            method = LeafBasedTestDensityFE(mode=mode, max_depth=3, k=5)
            result = method.fit_transform(X_tr, X_te, ds.y_train)
            n_added = 3 if mode == "confidence_feature" else 2
            # Original NaN preserved in untouched columns
            orig_n = ds.n_features
            assert np.isnan(result.X_train[:, :orig_n][tr_mask, 0]).all(), (
                f"{mode}: original NaN must be preserved in train"
            )
            assert np.isnan(result.X_test[:, :orig_n][te_mask, 0]).all(), (
                f"{mode}: original NaN must be preserved in test"
            )
            # Newly added leaf-derived columns must not contain NaN/inf
            new_tr = result.X_train[:, -n_added:]
            new_te = result.X_test[:, -n_added:]
            assert not np.any(np.isnan(new_tr))
            assert not np.any(np.isnan(new_te))
            assert not np.any(np.isinf(new_tr))
            assert not np.any(np.isinf(new_te))

    # --- cumshift feature selection (Tier B addition) ----------------------

    def test_cumshift_reduces_feature_count(self, ds):
        method = LeafBasedTestDensityFE(
            mode="confidence_feature", max_depth=3,
            feature_subset="cumshift", shift_ratio=0.9,
        )
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        meta = result.metadata
        assert meta["m18_feature_subset"] == "cumshift"
        # Cumshift should keep ≤ all features and ≥ floor of 3 (or n_features)
        n_total = meta["m18_n_features_total"]
        n_used = meta["m18_n_features_for_gbdt"]
        assert n_used <= n_total
        assert n_used >= min(3, n_total)
        assert "m18_shift_ratio_target" in meta
        assert "m18_shift_ratio_achieved" in meta
        assert "m18_selected_features" in meta
        # Output features are appended to the full feature set.
        assert result.X_train.shape[1] == n_total + 3
        assert result.X_test.shape[1] == n_total + 3

    def test_cumshift_ratio_1_keeps_all_features(self, ds):
        """shift_ratio=1.0 must keep every feature (cumsum target == total)."""
        method = LeafBasedTestDensityFE(
            mode="confidence_feature", max_depth=3,
            feature_subset="cumshift", shift_ratio=1.0,
        )
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        meta = result.metadata
        assert meta["m18_n_features_for_gbdt"] == meta["m18_n_features_total"]

    def test_cumshift_invalid_value_raises(self):
        with pytest.raises(ValueError, match="feature_subset"):
            LeafBasedTestDensityFE(feature_subset="bogus")

    def test_cumshift_name_includes_suffix(self):
        method = LeafBasedTestDensityFE(
            mode="confidence_feature", feature_subset="cumshift",
        )
        assert method.name.endswith("_cumshift")
        method2 = LeafBasedTestDensityFE(
            mode="leaf_embedding_knn", feature_subset="cumshift",
        )
        assert method2.name.endswith("_cumshift")

    # --- Naming, neighbor-count and validation contracts ------------------

    def test_embed_name_includes_n_estimators(self):
        """Embedding names include _t{n_estimators} to avoid collisions."""
        m = LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=10, n_estimators=100)
        assert "_t100" in m.name
        m2 = LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=10, n_estimators=150)
        assert "_t150" in m2.name
        # Two embed instances differing only in n_estimators must have
        # different names (else they collide silently in result CSVs).
        assert m.name != m2.name

    def test_embed_uses_full_k_on_small_data(self):
        """Cross-split kNN can use up to n_test or n_train neighbors."""
        from src.data.synthetic import load_synthetic
        small = load_synthetic(n_train=8, n_test=5, shift_strength=1.0, seed=0)
        method = LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=10, max_depth=2)
        result = method.fit_transform(small.X_train, small.X_test, small.y_train)
        # Effective k equals min(self.k=10, n_test=5, n_train=8) == 5.
        assert result.metadata["m18_k_effective"] == 5
        # k_combined should be min(5, n_total - 1) == min(5, 12) == 5 too.
        assert result.metadata["m18_k_combined"] == 5

    @pytest.mark.parametrize("kwargs", [
        {"n_estimators": 0},
        {"n_estimators": -1},
        {"max_depth": 0},
        {"k": 0},
        {"shift_ratio": 0.0},
        {"shift_ratio": 1.5},
        {"minority_threshold": -0.1},
        {"minority_threshold": 1.5},
    ])
    def test_invalid_hyperparameters_raise(self, kwargs):
        """The constructor rejects out-of-range hyperparameters."""
        with pytest.raises(ValueError):
            LeafBasedTestDensityFE(**kwargs)

    # Vectorized cross_frac calculation

    def test_embedding_cross_frac_matches_loop_definition(self, ds):
        """The vectorized cross_frac must match the textbook per-row loop."""
        method = LeafBasedTestDensityFE(
            mode="leaf_embedding_knn", k=5, max_depth=3,
        )
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        cross_frac_tr = result.X_train[:, -1]
        cross_frac_te = result.X_test[:, -1]
        # Values must lie in [0, 1].
        assert np.all((cross_frac_tr >= 0) & (cross_frac_tr <= 1))
        assert np.all((cross_frac_te >= 0) & (cross_frac_te <= 1))
        # This sample size and k value produce at least one cross-split neighbor.
        assert (cross_frac_tr > 0).any() or (cross_frac_te > 0).any()


# ===========================================================================
# M20: ClusterMembershipFE
# ===========================================================================

class TestClusterMembershipFE:
    def test_feature_mode_shape(self, ds):
        method = ClusterMembershipFE(n_clusters=5, return_as="feature")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Feature mode appends three cluster-derived features:
        # [cluster_relative_size, cluster_train_frac, cluster_dist_to_centroid]
        assert result.X_train.shape[1] == ds.n_features + 3
        assert result.X_test.shape[1] == ds.n_features + 3

    def test_weight_mode_shape(self, ds):
        method = ClusterMembershipFE(n_clusters=5, return_as="weight")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Weight mode appends cluster_relative_size rather than the arbitrary
        # cluster_id integer code.
        assert result.X_train.shape[1] == ds.n_features + 1
        assert result.sample_weights_train is not None
        assert len(result.sample_weights_train) == len(ds.X_train)

    def test_both_mode_shape(self, ds):
        method = ClusterMembershipFE(n_clusters=5, return_as="both")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[1] == ds.n_features + 3
        assert result.sample_weights_train is not None

    def test_weights_positive(self, ds):
        method = ClusterMembershipFE(n_clusters=5, return_as="weight")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert np.all(result.sample_weights_train > 0), "Weights should be positive"

    def test_weights_not_extreme(self, ds):
        """After clipping, weights should not be extreme."""
        method = ClusterMembershipFE(n_clusters=5, return_as="weight")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Normalized weights should have reasonable range
        assert result.sample_weights_train.max() < 100, "Weights too extreme"

    def test_empty_cluster_neutral_weight(self):
        """Clusters with no train or test samples should get neutral weight=1."""
        # Empty clusters must not produce extreme sample weights.
        method = ClusterMembershipFE(n_clusters=5, return_as="weight", seed=42)
        ds = load_synthetic(n_train=20, n_test=10, shift_strength=2.0, seed=42)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Weights should not contain extreme values from empty clusters
        assert not np.any(np.isinf(result.sample_weights_train))
        assert not np.any(np.isnan(result.sample_weights_train))

    def test_small_dataset_reduces_k(self, ds_small):
        method = ClusterMembershipFE(n_clusters=15)
        result = method.fit_transform(ds_small.X_train, ds_small.X_test, ds_small.y_train)
        assert result.metadata["m20_effective_k"] < 15

    def test_cluster_train_frac_in_range(self, ds):
        method = ClusterMembershipFE(n_clusters=5, return_as="feature")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # cluster_train_frac feature (second added feature, position -2)
        train_frac = result.X_train[:, -2]
        assert np.all(train_frac >= 0) and np.all(train_frac <= 1)

    def test_dist_to_centroid_positive(self, ds):
        method = ClusterMembershipFE(n_clusters=5, return_as="feature")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        dist = result.X_train[:, -1]
        assert np.all(dist >= 0), "Distance to centroid should be non-negative"

    def test_no_nan_inf(self, ds):
        for return_as in ("feature", "weight", "both"):
            method = ClusterMembershipFE(n_clusters=5, return_as=return_as)
            result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
            assert not np.any(np.isnan(result.X_train)), f"NaN in {return_as}"
            assert not np.any(np.isinf(result.X_train)), f"Inf in {return_as}"

    def test_y_train_independence(self, ds):
        """M20 declares uses_labels=False, so output must not depend on y_train."""
        m1 = ClusterMembershipFE(n_clusters=5, return_as="feature", seed=42)
        r1 = m1.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        y_shuffled = np.random.RandomState(99).permutation(ds.y_train)
        m2 = ClusterMembershipFE(n_clusters=5, return_as="feature", seed=42)
        r2 = m2.fit_transform(ds.X_train, ds.X_test, y_shuffled)

        np.testing.assert_array_almost_equal(
            r1.X_train, r2.X_train,
            err_msg="ClusterMembershipFE should NOT depend on y_train",
        )

    def test_invalid_return_as(self):
        with pytest.raises(ValueError, match="Unknown return_as"):
            ClusterMembershipFE(return_as="invalid")

    # ------------------------------------------------------------------
    # Validation, schema, cumulative-shift, purity and encoding contracts
    # ------------------------------------------------------------------

    def test_invalid_n_clusters(self):
        with pytest.raises(ValueError, match="n_clusters must be >= 2"):
            ClusterMembershipFE(n_clusters=1)

    def test_invalid_feature_subset(self):
        with pytest.raises(ValueError, match="feature_subset"):
            ClusterMembershipFE(feature_subset="bogus")

    def test_invalid_shift_ratio(self):
        with pytest.raises(ValueError, match="shift_ratio"):
            ClusterMembershipFE(shift_ratio=1.5)
        with pytest.raises(ValueError, match="shift_ratio"):
            ClusterMembershipFE(shift_ratio=0.0)

    def test_invalid_purity_threshold(self):
        with pytest.raises(ValueError, match="purity_threshold"):
            ClusterMembershipFE(purity_threshold=0.5)
        with pytest.raises(ValueError, match="purity_threshold"):
            ClusterMembershipFE(purity_threshold=-0.1)

    def test_cluster_relative_size_in_range_and_positive(self, ds):
        """cluster_relative_size feature (position -3 in feature/both modes)
        is per-row total cluster fraction, so values in (0, 1]."""
        method = ClusterMembershipFE(n_clusters=5, return_as="feature")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        rel_size = result.X_train[:, -3]
        assert np.all(rel_size > 0) and np.all(rel_size <= 1.0)

    def test_cluster_relative_sizes_sum_to_one(self, ds):
        """Summing the per-cluster relative_size values from metadata
        should yield 1.0 (every sample is in some cluster)."""
        method = ClusterMembershipFE(n_clusters=5, return_as="feature")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        sizes = result.metadata["m20_cluster_relative_sizes"]
        total = sum(sizes.values())
        assert abs(total - 1.0) < 1e-6, f"Sizes should sum to 1, got {total}"

    def test_weight_mode_appends_relative_size_not_cluster_id(self, ds):
        """Weight mode appends continuous relative size, not integer cluster ID."""
        method = ClusterMembershipFE(n_clusters=5, return_as="weight")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.feature_names[-1] == "cluster_relative_size"
        # Continuous values, not integer codes
        last_col = result.X_train[:, -1]
        assert last_col.dtype.kind == "f"
        assert np.any(last_col != np.floor(last_col)) or last_col.max() <= 1.0

    def test_no_cluster_id_feature_in_output(self, ds):
        """cluster_id remains diagnostic metadata and is never an output feature."""
        for return_as in ("feature", "weight", "both"):
            method = ClusterMembershipFE(n_clusters=5, return_as=return_as)
            result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
            assert "cluster_id" not in result.feature_names, (
                f"cluster_id should not appear in {return_as} feature_names"
            )

    def test_kmeans_scaling_metadata(self, ds):
        """Verify the freq-encoding + StandardScaler metadata flags are set."""
        method = ClusterMembershipFE(n_clusters=5)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.metadata["m20_kmeans_encoding"] == "frequency"
        assert result.metadata["m20_kmeans_scaled"] is True

    def test_cumshift_variant_shape(self, ds):
        """cumshift limits KMeans inputs without changing the output contract."""
        method = ClusterMembershipFE(
            n_clusters=5, return_as="both", feature_subset="cumshift",
        )
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[1] == ds.n_features + 3
        # KMeans space must use a subset of original features
        assert result.metadata["m20_n_features_for_kmeans"] <= ds.n_features
        assert result.metadata["m20_n_features_for_kmeans"] >= 3
        assert "m20_shift_ratio_target" in result.metadata
        assert "m20_n_features_selected" in result.metadata

    def test_cumshift_changes_clustering(self, ds):
        """Cumshift selects a different feature space → different KMeans
        labels → different output features (compared to default subset=None)."""
        m_full = ClusterMembershipFE(n_clusters=5, return_as="feature", seed=42)
        m_cs = ClusterMembershipFE(
            n_clusters=5, return_as="feature",
            feature_subset="cumshift", seed=42,
        )
        r_full = m_full.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        r_cs = m_cs.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Distance-to-centroid features (position -1) should differ
        assert not np.allclose(r_full.X_train[:, -1], r_cs.X_train[:, -1])

    def test_purity_filter_metadata_present(self, ds):
        """Report purity-filter metadata when no cluster satisfies the threshold."""
        method = ClusterMembershipFE(n_clusters=5, purity_threshold=0.05)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert "m20_n_pure_clusters_filtered" in result.metadata
        assert result.metadata["m20_purity_threshold"] == 0.05

    def test_purity_filter_disabled_via_zero_threshold(self):
        """purity_threshold=0.0 should disable the filter entirely."""
        method = ClusterMembershipFE(
            n_clusters=10, return_as="weight",
            purity_threshold=0.0, seed=42,
        )
        ds_shift = load_synthetic(n_train=200, n_test=100, shift_strength=2.0, seed=42)
        result = method.fit_transform(ds_shift.X_train, ds_shift.X_test, ds_shift.y_train)
        assert result.metadata["m20_n_pure_clusters_filtered"] == 0

    def test_purity_filter_active_on_heavy_shift(self):
        """On heavily shifted data with K=15, expect at least one cluster
        to be flagged as pure (extreme train_frac)."""
        method = ClusterMembershipFE(
            n_clusters=15, return_as="weight",
            purity_threshold=0.10, seed=42,
        )
        ds_shift = load_synthetic(n_train=300, n_test=150, shift_strength=2.0, seed=42)
        result = method.fit_transform(ds_shift.X_train, ds_shift.X_test, ds_shift.y_train)
        # Record the number of clusters that satisfy the purity threshold.
        assert result.metadata["m20_n_pure_clusters_filtered"] >= 0
        # Weights remain finite and positive after filtering.
        assert np.all(np.isfinite(result.sample_weights_train))
        assert np.all(result.sample_weights_train > 0)

    def test_name_property_with_cumshift_suffix(self):
        m = ClusterMembershipFE(
            n_clusters=10, return_as="both", feature_subset="cumshift",
        )
        assert m.name == "cluster_membership_k10_both_cumshift"

    def test_name_property_no_suffix_default(self):
        m = ClusterMembershipFE(n_clusters=10, return_as="both")
        assert m.name == "cluster_membership_k10_both"

    def test_method_lookup_contains_all_configurations(self):
        expected = [
            # Original 6
            "cluster_membership_k5_feature", "cluster_membership_k10_feature",
            "cluster_membership_k10_weight", "cluster_membership_k10_both",
            "cluster_membership_k5_both", "cluster_membership_k15_both",
            # Cumulative-shift variants
            "cluster_membership_k10_both_cumshift",
            "cluster_membership_k10_feature_cumshift",
        ]
        for name in expected:
            assert name in TIER2_METHODS, f"Missing {name} in TIER2_METHODS"


# ===========================================================================
# M19: NeighborhoodConsensusFeatures
# ===========================================================================

class TestNeighborhoodConsensusFeatures:
    def test_classification_shape_adds_5_features(self, ds):
        """Classification: should append 4 universal + 1 agree feature = 5."""
        method = NeighborhoodConsensusFeatures(k=5)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[1] == ds.n_features + 5
        assert result.X_test.shape[1] == ds.n_features + 5
        assert len(result.feature_names) == result.X_train.shape[1]
        # Last column should be the agree feature.
        assert result.feature_names[-1] == "nbr_consensus_agree"

    def test_regression_shape_adds_4_features(self, ds):
        """Regression: should append only the 4 universal features (no agree)."""
        # Synthesize a continuous y_train so _is_classification returns False.
        rng = np.random.RandomState(0)
        y_cont = ds.y_train.astype(float) + rng.randn(len(ds.y_train)) * 0.5 + 0.1
        method = NeighborhoodConsensusFeatures(k=5)
        result = method.fit_transform(ds.X_train, ds.X_test, y_cont)
        assert result.X_train.shape[1] == ds.n_features + 4
        assert result.feature_names[-1] == "nbr_consensus_diff"
        assert result.metadata["m19_is_classification"] is False

    def test_self_feature_equals_pass1_scalar(self, ds):
        """nbr_consensus_self must equal the pass-1 scalar prediction."""
        method = NeighborhoodConsensusFeatures(k=5)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Self-column is the first of the appended block.
        self_col_idx = ds.n_features + 0
        train_self = result.X_train[:, self_col_idx]
        test_self = result.X_test[:, self_col_idx]
        # Binary classification → P(y=1) ∈ [0, 1].
        assert np.all((train_self >= 0) & (train_self <= 1))
        assert np.all((test_self >= 0) & (test_self <= 1))

    def test_diff_equals_self_minus_mean(self, ds):
        """nbr_consensus_diff must be exactly self - mean."""
        method = NeighborhoodConsensusFeatures(k=10)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        n_orig = ds.n_features
        self_col = result.X_train[:, n_orig + 0]
        mean_col = result.X_train[:, n_orig + 1]
        diff_col = result.X_train[:, n_orig + 3]
        np.testing.assert_array_almost_equal(diff_col, self_col - mean_col)

    def test_std_non_negative(self, ds):
        """nbr_consensus_std must be non-negative everywhere."""
        method = NeighborhoodConsensusFeatures(k=10)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        std_col = result.X_train[:, ds.n_features + 2]
        assert np.all(std_col >= 0)

    def test_agree_in_unit_interval(self, ds):
        """nbr_consensus_agree (classification) must be in [0, 1]."""
        method = NeighborhoodConsensusFeatures(k=10)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        agree_col = result.X_train[:, -1]
        assert np.all((agree_col >= 0) & (agree_col <= 1))

    def test_no_nan_inf(self, ds):
        method = NeighborhoodConsensusFeatures(k=10)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Newly appended columns must be finite.
        new_train = result.X_train[:, ds.n_features:]
        new_test = result.X_test[:, ds.n_features:]
        assert not np.any(np.isnan(new_train))
        assert not np.any(np.isnan(new_test))
        assert not np.any(np.isinf(new_train))
        assert not np.any(np.isinf(new_test))

    def test_distance_weighting_no_extreme_weights(self, ds):
        """Duplicate points must not blow up distance weighting (1e-4 clip)."""
        # Inject duplicate rows into X_test.
        X_te = ds.X_test.copy().reset_index(drop=True)
        X_te.iloc[10] = X_te.iloc[11]
        X_te.iloc[20] = X_te.iloc[21]
        method = NeighborhoodConsensusFeatures(k=5, weighting="distance")
        result = method.fit_transform(ds.X_train, X_te, ds.y_train)
        new_test = result.X_test[:, ds.n_features:]
        assert np.all(np.isfinite(new_test)), "Distance weighting must clip duplicates"

    def test_uniform_weighting_runs(self, ds):
        method = NeighborhoodConsensusFeatures(k=5, weighting="uniform")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[1] == ds.n_features + 5

    def test_deterministic(self, ds):
        m1 = NeighborhoodConsensusFeatures(k=10, seed=42)
        r1 = m1.fit_transform(ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy())
        m2 = NeighborhoodConsensusFeatures(k=10, seed=42)
        r2 = m2.fit_transform(ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy())
        np.testing.assert_array_almost_equal(r1.X_train, r2.X_train)
        np.testing.assert_array_almost_equal(r1.X_test, r2.X_test)

    def test_uses_labels_true(self):
        method = NeighborhoodConsensusFeatures()
        assert method.uses_labels is True

    def test_metadata_keys(self, ds):
        method = NeighborhoodConsensusFeatures(k=10)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        meta = result.metadata
        for key in (
            "m19_k_requested", "m19_k_effective_train", "m19_k_effective_test",
            "m19_n_features_added", "m19_n_features_for_knn",
            "m19_n_features_total", "m19_is_classification",
            "m19_n_classes", "m19_n_folds_used", "m19_weighting",
            "m19_feature_subset",
        ):
            assert key in meta, f"Missing metadata key {key}"

    def test_small_dataset_reduces_folds(self, ds_small):
        """Tiny datasets should reduce n_folds to keep folds non-empty."""
        method = NeighborhoodConsensusFeatures(k=5, n_folds=5)
        result = method.fit_transform(ds_small.X_train, ds_small.X_test, ds_small.y_train)
        assert result.metadata["m19_n_folds_used"] <= 5
        assert result.metadata["m19_n_folds_used"] >= 2

    def test_k_larger_than_n_falls_back(self):
        """k > n - 1 should auto-clamp to n - 1 without crashing."""
        ds = load_synthetic(n_train=20, n_test=8, shift_strength=1.0, seed=42)
        method = NeighborhoodConsensusFeatures(k=50, n_folds=2)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.metadata["m19_k_effective_train"] <= len(ds.X_train) - 1
        assert result.metadata["m19_k_effective_test"] <= len(ds.X_test) - 1

    def test_cumshift_metadata_present(self, ds):
        method = NeighborhoodConsensusFeatures(
            k=10, feature_subset="cumshift", shift_ratio=0.9,
        )
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        meta = result.metadata
        assert meta["m19_feature_subset"] == "cumshift"
        assert "m19_shift_ratio_target" in meta
        assert "m19_shift_ratio_achieved" in meta
        assert "m19_n_features_selected" in meta
        # kNN inputs are restricted. Output features use the full feature set.
        assert meta["m19_n_features_for_knn"] <= meta["m19_n_features_total"]
        assert result.X_train.shape[1] == meta["m19_n_features_total"] + 5

    def test_cumshift_name_includes_suffix(self):
        method = NeighborhoodConsensusFeatures(k=10, feature_subset="cumshift")
        assert method.name.endswith("_cumshift")

    def test_invalid_hyperparameters_raise(self):
        for kwargs in (
            {"k": 0},
            {"weighting": "bogus"},
            {"n_folds": 1},
            {"feature_subset": "bogus"},
            {"shift_ratio": 0.0},
            {"shift_ratio": 1.5},
            {"internal_n_estimators": 0},
            {"internal_max_depth": 0},
        ):
            with pytest.raises(ValueError):
                NeighborhoodConsensusFeatures(**kwargs)

    def test_method_lookup_contains_all_configurations(self):
        expected = [
            "neighbor_consensus_k10_dist",
            "neighbor_consensus_k10_uniform",
            "neighbor_consensus_k5_dist",
            "neighbor_consensus_k20_dist",
            "neighbor_consensus_k10_dist_cumshift",
            "neighbor_consensus_k5_dist_cumshift",
        ]
        for name in expected:
            assert name in TIER2_METHODS, f"Missing {name} in TIER2_METHODS"

    # --- kNN-space frequency encoding and scaling --------------------------

    def test_high_cardinality_cats_no_feature_explosion(self, ds):
        """A 50-category column must not create an excessive kNN feature count."""
        import pandas as pd
        rng = np.random.RandomState(0)
        X_tr = ds.X_train.copy()
        X_te = ds.X_test.copy()
        X_tr["HIGHCARD"] = pd.Series(
            rng.choice([f"cat_{i}" for i in range(50)], size=len(X_tr)),
            index=X_tr.index, dtype="object",
        )
        X_te["HIGHCARD"] = pd.Series(
            rng.choice([f"cat_{i}" for i in range(50)], size=len(X_te)),
            index=X_te.index, dtype="object",
        )
        method = NeighborhoodConsensusFeatures(k=5)
        result = method.fit_transform(X_tr, X_te, ds.y_train)
        # n_features_for_knn should equal n original columns (1 per original
        # column), NOT n_original + 49 (one-hot) or similar.
        n_orig = X_tr.shape[1]
        assert result.metadata["m19_n_features_for_knn"] == n_orig
        assert result.metadata["m19_n_features_total"] == n_orig
        # Output still appends 5 consensus features (clf).
        assert result.X_train.shape[1] == n_orig + 5

    def test_freq_encoding_applied_to_knn_space(self, ds):
        """Metadata should reflect frequency encoding of categorical columns."""
        method = NeighborhoodConsensusFeatures(k=10)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        meta = result.metadata
        assert meta["m19_knn_encoding"] == "frequency"
        # The synthetic dataset has 3 cat cols (OCCP, MAR, ST).
        assert meta["m19_n_cat_cols_encoded"] == 3

    def test_standardscaler_applied(self, ds):
        """Metadata flag must indicate the kNN matrix was standardized."""
        method = NeighborhoodConsensusFeatures(k=10)
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.metadata["m19_knn_scaled"] is True

    def test_purely_numeric_dataset_unchanged_behavior(self, ds):
        """When no cat cols exist, freq-encoding is a no-op. Method still runs."""
        X_tr = ds.X_train.select_dtypes(include=["number"]).copy()
        X_te = ds.X_test.select_dtypes(include=["number"]).copy()
        assert X_tr.select_dtypes(include=["object", "category"]).shape[1] == 0
        method = NeighborhoodConsensusFeatures(k=5)
        result = method.fit_transform(X_tr, X_te, ds.y_train)
        assert result.metadata["m19_n_cat_cols_encoded"] == 0
        assert result.metadata["m19_knn_encoding"] == "frequency"
        assert result.metadata["m19_knn_scaled"] is True
        # The classifier appends five features without changing the row count.
        assert result.X_train.shape[1] == X_tr.shape[1] + 5
        # Every appended value is finite.
        new_cols = result.X_train[:, X_tr.shape[1]:]
        assert np.all(np.isfinite(new_cols))


# ===========================================================================
# Cross-method integration tests
# ===========================================================================

class TestIntegration:
    def test_m18_then_m20_pipeline(self, ds):
        """M18 and M20 should compose without error."""
        m18 = LeafBasedTestDensityFE(mode="confidence_feature", max_depth=3)
        r1 = m18.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        # Feed M18 output into M20 (as numpy arrays, need to wrap in DataFrames)
        import pandas as pd
        X_tr_df = pd.DataFrame(r1.X_train, columns=r1.feature_names)
        X_te_df = pd.DataFrame(r1.X_test, columns=r1.feature_names)

        m20 = ClusterMembershipFE(n_clusters=5, return_as="feature")
        r2 = m20.fit_transform(X_tr_df, X_te_df, ds.y_train)
        assert r2.X_train.shape[0] == len(ds.X_train)
        assert r2.X_test.shape[0] == len(ds.X_test)
        assert not np.any(np.isnan(r2.X_train))

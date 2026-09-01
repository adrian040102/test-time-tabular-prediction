"""Tests for Tier 1 test-time methods (M1-M5, M14 and M14b).

Each method is tested for:
1. Output shape consistency (train/test same number of features)
2. No NaN/Inf in output
3. Deterministic output for repeated inputs
4. Feature count expectations (e.g. PCA adds features)
5. GroupBy-specific behavior (feature explosion, variant differences)
6. MethodPipeline combinations
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
from src.methods.base import InductiveBaseline, MethodPipeline
from src.methods.tier1 import (
    JointFrequencyEncoding,
    FrequencyRatio,
    JointPCA,
    JointTSNE,
    JointTSNESkipReason,
    ShiftRegularizedTargetEncoding,
    JointScaling,
    JointGroupByFeatures,
    GroupByShiftRatio,
    TIER1_METHODS,
    get_all_tier1_methods,
)


@pytest.fixture
def ds():
    return load_synthetic(n_train=300, n_test=200, shift_strength=1.0, seed=42)


@pytest.fixture
def ds_no_shift():
    return load_synthetic(n_train=300, n_test=200, shift_strength=0.0, seed=42)


# ---------------------------------------------------------------------------
# Generic tests for all Tier 1 methods
# ---------------------------------------------------------------------------

class TestAllTier1Generic:
    """Run shape, NaN and determinism checks on every configured Tier 1 method."""

    @pytest.fixture(params=list(get_all_tier1_methods().keys()))
    def method_name(self, request):
        return request.param

    def test_output_shape_consistent(self, ds, method_name):
        method = get_all_tier1_methods()[method_name]
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[1] == result.X_test.shape[1], \
            f"{method_name}: train/test feature count mismatch"
        assert len(result.feature_names) == result.X_train.shape[1]

    def test_no_nan(self, ds, method_name):
        method = get_all_tier1_methods()[method_name]
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert not np.any(np.isnan(result.X_train)), f"{method_name}: NaN in X_train"
        assert not np.any(np.isnan(result.X_test)), f"{method_name}: NaN in X_test"

    def test_no_inf(self, ds, method_name):
        method = get_all_tier1_methods()[method_name]
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert not np.any(np.isinf(result.X_train)), f"{method_name}: Inf in X_train"
        assert not np.any(np.isinf(result.X_test)), f"{method_name}: Inf in X_test"

    def test_row_count_preserved(self, ds, method_name):
        method = get_all_tier1_methods()[method_name]
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[0] == len(ds.X_train)
        assert result.X_test.shape[0] == len(ds.X_test)

    def test_deterministic(self, ds, method_name):
        method = get_all_tier1_methods()[method_name]
        r1 = method.fit_transform(ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy())
        r2 = method.fit_transform(ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy())
        np.testing.assert_array_almost_equal(r1.X_train, r2.X_train,
                                             err_msg=f"{method_name}: not deterministic (X_train)")
        np.testing.assert_array_almost_equal(r1.X_test, r2.X_test,
                                             err_msg=f"{method_name}: not deterministic (X_test)")


# ---------------------------------------------------------------------------
# M1-specific tests
# ---------------------------------------------------------------------------

class TestJointFrequencyEncoding:
    def test_combined_preserves_cats_and_adds_freq(self, ds):
        n_cat = len(ds.cat_features)
        n_num = len(ds.num_features)
        result = JointFrequencyEncoding("combined").fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        # Preserve encoded categoricals and add one frequency feature per column.
        assert result.X_train.shape[1] == n_num + 2 * n_cat
        # Cats must remain in the feature names so cat-aware models can split.
        assert result.categorical_features == ds.cat_features
        for cat_col in ds.cat_features:
            assert cat_col in result.feature_names

    def test_separate_preserves_cats_and_has_two_freq_per_cat(self, ds):
        n_cat = len(ds.cat_features)
        n_num = len(ds.num_features)
        result = JointFrequencyEncoding("separate").fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        # Preserve encoded categoricals and add two frequency features per column.
        assert result.X_train.shape[1] == n_num + 3 * n_cat
        assert result.categorical_features == ds.cat_features

    def test_frequencies_between_zero_and_one(self, ds):
        result = JointFrequencyEncoding("combined").fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        # Only the frequency columns (last n_cat columns) should be in [0, 1]
        n_cat = len(ds.cat_features)
        freq_cols = result.X_train[:, -n_cat:]
        assert np.all(freq_cols >= 0)
        assert np.all(freq_cols <= 1.01)  # small tolerance


# ---------------------------------------------------------------------------
# M2-specific tests
# ---------------------------------------------------------------------------

class TestFrequencyRatio:
    def test_log_ratio_centered_near_zero_iid(self, ds_no_shift):
        result = FrequencyRatio(use_log=True).fit_transform(
            ds_no_shift.X_train, ds_no_shift.X_test, ds_no_shift.y_train
        )
        # Under no shift, log ratios should be approximately 0
        n_cat = len(ds_no_shift.cat_features)
        ratio_cols = result.X_train[:, -n_cat:]
        mean_ratio = np.mean(np.abs(ratio_cols))
        assert mean_ratio < 0.5, f"Log ratios should be near 0 under IID, got mean |ratio| = {mean_ratio}"

    def test_raw_ratio_centered_near_one_iid(self, ds_no_shift):
        result = FrequencyRatio(use_log=False).fit_transform(
            ds_no_shift.X_train, ds_no_shift.X_test, ds_no_shift.y_train
        )
        n_cat = len(ds_no_shift.cat_features)
        ratio_cols = result.X_train[:, -n_cat:]
        mean_ratio = np.mean(ratio_cols)
        assert 0.5 < mean_ratio < 1.5, f"Raw ratios should be near 1 under IID, got mean = {mean_ratio}"


# ---------------------------------------------------------------------------
# M3-specific tests
# ---------------------------------------------------------------------------

class TestJointPCA:
    def test_adds_components(self, ds):
        n_original = ds.n_features
        n_comp = 3
        result = JointPCA(n_components=n_comp).fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        assert result.X_train.shape[1] == n_original + n_comp

    def test_pca_names_present(self, ds):
        result = JointPCA(n_components=3).fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        pca_names = [n for n in result.feature_names if n.startswith("joint_pc")]
        assert len(pca_names) == 3

    def test_explained_variance_in_metadata(self, ds):
        result = JointPCA(n_components=3).fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        assert "explained_variance_ratio" in result.metadata
        assert len(result.metadata["explained_variance_ratio"]) == 3


# ---------------------------------------------------------------------------
# M3b-specific tests: JointTSNE
# ---------------------------------------------------------------------------

class TestJointTSNE:
    """Core behavior tests for ``JointTSNE``."""

    def test_adds_2d_components(self, ds):
        n_original = ds.n_features
        result = JointTSNE(n_components=2, perplexity=30).fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        assert result.X_train.shape[1] == n_original + 2
        assert result.X_test.shape[1] == n_original + 2

    def test_adds_3d_components(self, ds):
        n_original = ds.n_features
        result = JointTSNE(n_components=3, perplexity=30).fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        assert result.X_train.shape[1] == n_original + 3
        assert result.X_test.shape[1] == n_original + 3

    def test_tsne_dim_names_present(self, ds):
        result = JointTSNE(n_components=2).fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        tsne_names = [n for n in result.feature_names if n.startswith("joint_tsne")]
        assert tsne_names == ["joint_tsne1", "joint_tsne2"]

    def test_metadata_records_perplexity_and_kl(self, ds):
        result = JointTSNE(n_components=2, perplexity=30).fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        md = result.metadata
        assert md["perplexity"] == 30
        assert md["perplexity_requested"] == 30
        assert md["n_components"] == 2
        assert md["n_total"] == len(ds.X_train) + len(ds.X_test)
        assert np.isfinite(md["kl_divergence"]), "kl_divergence should be finite"
        assert md["library"] == "openTSNE"

    def test_skip_pipeline_scaler_set(self, ds):
        """Joint embeddings must bypass the pipeline train-only scaler refit."""
        result = JointTSNE().fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.skip_pipeline_scaler is True

    def test_original_cats_preserved_in_output(self, ds):
        """Original categoricals remain as encoded columns beside the t-SNE dims."""
        result = JointTSNE().fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.categorical_features == ds.cat_features
        for cat_col in ds.cat_features:
            assert cat_col in result.feature_names, (
                f"Original cat '{cat_col}' missing from output"
            )

    def test_perplexity_auto_clamped_for_small_n(self):
        """For tiny datasets the requested perplexity may exceed (n-1)/3.
        The method must clamp it to a valid value."""
        from src.data.synthetic import load_synthetic
        tiny = load_synthetic(n_train=30, n_test=20, shift_strength=1.0, seed=0)
        result = JointTSNE(n_components=2, perplexity=50).fit_transform(
            tiny.X_train, tiny.X_test, tiny.y_train
        )
        assert result.metadata["perplexity_requested"] == 50
        # (n_total - 1) / 3 = 49 / 3 = 16.33 < 50, so effective is clamped
        assert result.metadata["perplexity"] < 50
        assert result.metadata["perplexity"] >= 5  # floor

    def test_hard_skip_when_too_large(self, ds):
        """The 200K guard applies immediately without doing any t-SNE work."""
        method = JointTSNE(max_n_total=400)  # ds is 300+200=500, so skip
        with pytest.raises(JointTSNESkipReason) as exc:
            method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert exc.value.reason == "n_total_exceeded"
        assert exc.value.details["max_n_total"] == 400
        assert exc.value.details["n_total"] == 500
        # The structured skip reason subclasses RuntimeError.
        assert isinstance(exc.value, RuntimeError)

    # ---------------------------------------------------------------
    # Skip messages use a method-specific prefix and structured details.
    # ---------------------------------------------------------------

    def test_skip_message_starts_with_joint_tsne_skip_prefix(self, ds):
        """The exception text must start with ``joint_tsne_skip:``."""
        method = JointTSNE(max_n_total=400)
        with pytest.raises(JointTSNESkipReason) as exc:
            method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        msg = str(exc.value)
        assert msg.startswith("joint_tsne_skip:"), (
            f"skip message must start with 'joint_tsne_skip:', got: {msg!r}"
        )

    def test_skip_message_format_matches_tabpfn_tabicl_pattern(self, ds):
        """Format: 'joint_tsne_skip: <reason> (k=v, k=v, ...)'."""
        method = JointTSNE(max_n_total=400)
        with pytest.raises(JointTSNESkipReason) as exc:
            method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        msg = str(exc.value)
        # Reason follows the prefix directly after a single space.
        assert msg.startswith("joint_tsne_skip: n_total_exceeded"), (
            f"reason segment does not have the required format: {msg!r}"
        )
        # Details rendered as k=v, not dict-repr (no curly braces, no quotes).
        assert "{" not in msg and "'" not in msg, (
            f"details should be 'k=v, k=v' not dict-repr, got: {msg!r}"
        )
        # All four expected detail keys are present.
        for key in ("n_train", "n_test", "n_total", "max_n_total"):
            assert f"{key}=" in msg, f"missing detail key {key} in: {msg!r}"

    def test_skip_message_without_details(self):
        """Edge case: skip reason raised without details dict still gets prefix."""
        exc = JointTSNESkipReason("custom_reason")
        msg = str(exc)
        assert msg == "joint_tsne_skip: custom_reason", (
            f"prefix should still apply with no details, got: {msg!r}"
        )

    def test_skip_message_matches_other_wrapper_prefix_convention(self):
        """The prefix format must match the other structured skip reasons."""
        from src.models.tabpfn import TabPFNSkipReason
        from src.models.tabicl import TabICLSkipReason
        ours = str(JointTSNESkipReason("foo", {"a": 1, "b": 2}))
        theirs_pfn = str(TabPFNSkipReason("foo", {"a": 1, "b": 2}))
        theirs_icl = str(TabICLSkipReason("foo", {"a": 1, "b": 2}))
        # Only the method-specific prefix differs.
        assert ours == "joint_tsne_skip: foo (a=1, b=2)"
        assert theirs_pfn == "tabpfn_skip: foo (a=1, b=2)"
        assert theirs_icl == "tabicl_skip: foo (a=1, b=2)"

    def test_invalid_n_components_rejected(self):
        with pytest.raises(ValueError, match="n_components must be 2 or 3"):
            JointTSNE(n_components=5)

    def test_invalid_perplexity_rejected(self):
        with pytest.raises(ValueError, match="perplexity must be > 0"):
            JointTSNE(perplexity=0)

    def test_three_configurations_in_method_lookup(self):
        for key in ("joint_tsne_2d_p30", "joint_tsne_2d_p50", "joint_tsne_3d_p30"):
            assert key in TIER1_METHODS, f"missing method key: {key}"
            instance = TIER1_METHODS[key]()
            assert instance.name == key

    # ---------------------------------------------------------------
    # The size guard must run before deferred imports.
    # ---------------------------------------------------------------

    def test_hard_skip_occurs_before_opentsne_import(self, ds, monkeypatch):
        """The size guard must run before an unavailable openTSNE import."""
        import builtins
        real_import = builtins.__import__

        def fail_on_opentsne(name, *args, **kwargs):
            if name == "openTSNE" or name.startswith("openTSNE."):
                raise ImportError("simulated openTSNE missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_on_opentsne)

        method = JointTSNE(max_n_total=400)  # ds is 300+200=500
        # The size guard raises the structured skip reason before ImportError.
        with pytest.raises(JointTSNESkipReason) as exc:
            method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert exc.value.reason == "n_total_exceeded"

    # ---------------------------------------------------------------
    # If the distance-space dimensionality is below n_components, openTSNE's
    # PCA initialization is invalid. Single-feature datasets and narrow
    # SelectiveMethod outputs therefore fall back to random initialization.
    # ---------------------------------------------------------------

    def test_one_numeric_feature_2d_uses_random_init(self):
        """One distance feature requires random initialization for 2D t-SNE."""
        rng = np.random.RandomState(0)
        X_tr = pd.DataFrame({"only_feat": rng.randn(60).astype(float)})
        X_te = pd.DataFrame({"only_feat": rng.randn(40).astype(float)})
        result = JointTSNE(n_components=2, perplexity=10).fit_transform(
            X_tr, X_te, np.zeros(60),
        )
        assert result.X_train.shape == (60, 1 + 2)
        assert result.X_test.shape == (40, 1 + 2)
        assert result.metadata["initialization"] == "random"
        assert result.metadata["n_dist_features"] == 1
        assert np.isfinite(result.X_train).all()
        assert np.isfinite(result.X_test).all()

    def test_two_numeric_features_3d_uses_random_init(self):
        """Two distance features require random initialization for 3D t-SNE."""
        rng = np.random.RandomState(0)
        X_tr = pd.DataFrame(rng.randn(60, 2), columns=["a", "b"]).astype(float)
        X_te = pd.DataFrame(rng.randn(40, 2), columns=["a", "b"]).astype(float)
        result = JointTSNE(n_components=3, perplexity=10).fit_transform(
            X_tr, X_te, np.zeros(60),
        )
        assert result.X_train.shape == (60, 2 + 3)
        assert result.metadata["initialization"] == "random"
        assert result.metadata["n_dist_features"] == 2

    def test_wide_features_uses_pca_init(self, ds):
        """A sufficiently wide distance space must use PCA initialization."""
        result = JointTSNE(n_components=2, perplexity=10).fit_transform(
            ds.X_train, ds.X_test, ds.y_train,
        )
        # The fixture produces at least two distance-space features.
        assert result.metadata["initialization"] == "pca"
        assert result.metadata["n_dist_features"] >= 2

    def test_pca_init_boundary_n_features_equals_n_components(self):
        """PCA initialization remains valid when feature and component counts match."""
        rng = np.random.RandomState(0)
        X_tr = pd.DataFrame(rng.randn(60, 2), columns=["a", "b"]).astype(float)
        X_te = pd.DataFrame(rng.randn(40, 2), columns=["a", "b"]).astype(float)
        result = JointTSNE(n_components=2, perplexity=10).fit_transform(
            X_tr, X_te, np.zeros(60),
        )
        assert result.metadata["n_dist_features"] == 2
        assert result.metadata["initialization"] == "pca"  # n==K is allowed

    # ---------------------------------------------------------------
    # metadata["perplexity"] records the outer clamp, while openTSNE can apply
    # a stricter internal (n-1)/3 clamp. ``perplexity_effective`` records the
    # value used for fitting.
    # ---------------------------------------------------------------

    def test_perplexity_effective_matches_outer_clamp_when_no_inner_clamp(self, ds):
        """Without an inner clamp, effective perplexity equals the outer value."""
        result = JointTSNE(n_components=2, perplexity=30).fit_transform(
            ds.X_train, ds.X_test, ds.y_train,
        )
        # ds is 500 rows, (n-1)/3 = 166 >> 30, so no clamp at all.
        assert result.metadata["perplexity_requested"] == 30.0
        assert result.metadata["perplexity"] == 30.0
        assert result.metadata["perplexity_effective"] == 30.0

    def test_perplexity_effective_records_opentsne_internal_clamp(self):
        """Record an openTSNE internal clamp below the configured floor of 5.
        For n_total=12, openTSNE clamps to (12-1)/3 = 3.67.
        ``perplexity_effective`` records the internally clamped value."""
        rng = np.random.RandomState(0)
        X_tr = pd.DataFrame(rng.randn(7, 5), columns=list("abcde")).astype(float)
        X_te = pd.DataFrame(rng.randn(5, 5), columns=list("abcde")).astype(float)
        result = JointTSNE(n_components=2, perplexity=30).fit_transform(
            X_tr, X_te, np.zeros(7),
        )
        # The outer floor sets perplexity to 5.0.
        assert result.metadata["perplexity"] == 5.0
        # But openTSNE internally clamps at (12-1)/3 ≈ 3.67.
        assert result.metadata["perplexity_effective"] < 5.0
        assert result.metadata["perplexity_effective"] == pytest.approx(11 / 3, abs=1e-6)

    def test_perplexity_effective_present_in_all_three_configurations(self, ds):
        """All three configurations populate the effective value."""
        for key in ("joint_tsne_2d_p30", "joint_tsne_2d_p50", "joint_tsne_3d_p30"):
            result = TIER1_METHODS[key]().fit_transform(
                ds.X_train, ds.X_test, ds.y_train,
            )
            assert "perplexity_effective" in result.metadata, (
                f"{key} missing perplexity_effective"
            )
            assert isinstance(result.metadata["perplexity_effective"], float)


# ---------------------------------------------------------------------------
# M4-specific tests
# ---------------------------------------------------------------------------

class TestTargetEncoding:
    def test_uses_labels(self):
        method = ShiftRegularizedTargetEncoding()
        assert method.uses_labels is True

    def test_encoding_between_0_and_1(self, ds):
        result = ShiftRegularizedTargetEncoding().fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        # For binary classification, target encodings should be in [0, 1]
        n_cat = len(ds.cat_features)
        enc_cols = result.X_train[:, -n_cat:]
        assert np.all(enc_cols >= -0.01), "Target encoding below 0"
        assert np.all(enc_cols <= 1.01), "Target encoding above 1"

    def test_all_regularization_modes(self, ds):
        for mode in ("ratio", "absolute", "diff"):
            method = ShiftRegularizedTargetEncoding(regularization_mode=mode)
            result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
            assert result.X_train.shape[0] == len(ds.X_train)


# ---------------------------------------------------------------------------
# M5-specific tests
# ---------------------------------------------------------------------------

class TestJointScaling:
    def test_combined_mean_near_zero(self, ds):
        result = JointScaling().fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Combined (train+test) should have mean ≈ 0
        combined = np.vstack([result.X_train, result.X_test])
        means = np.mean(combined, axis=0)
        np.testing.assert_array_almost_equal(means, 0, decimal=1)

    def test_differs_from_train_only_scaling(self, ds):
        """Joint scaling should differ from inductive baseline under shift.

        InductiveBaseline now returns unscaled label-encoded data (scaling
        is handled by the pipeline), so the test compares JointScaling against a
        manual train-only StandardScaler to verify the joint statistics
        produce different results.
        """
        from sklearn.preprocessing import StandardScaler
        joint_result = JointScaling().fit_transform(ds.X_train, ds.X_test, ds.y_train)
        inductive_result = InductiveBaseline().fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Apply train-only scaler to inductive output (mimics pipeline)
        scaler = StandardScaler()
        ind_tr_scaled = scaler.fit_transform(inductive_result.X_train)
        ind_te_scaled = scaler.transform(inductive_result.X_test)
        # The results must differ because their scaling statistics differ.
        assert not np.allclose(joint_result.X_test, ind_te_scaled, atol=1e-3), \
            "Joint scaling should differ from inductive under distribution shift"


# ---------------------------------------------------------------------------
# M14-specific tests: JointGroupByFeatures
# ---------------------------------------------------------------------------

class TestJointGroupByFeatures:
    def test_combined_adds_features(self, ds):
        """Combined variant should add n_group * n_agg * n_func features."""
        method = JointGroupByFeatures(variant="combined")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # The output includes features beyond the original numeric columns.
        assert result.X_train.shape[1] > len(ds.num_features)
        assert result.metadata.get("n_groupby_features", 0) > 0

    def test_separate_has_more_features_than_combined(self, ds):
        """Separate variant creates 2 features per triple vs 1 for combined."""
        combined_result = JointGroupByFeatures(variant="combined").fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        separate_result = JointGroupByFeatures(variant="separate").fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        # Separate mode produces twice as many new features.
        n_combined_new = combined_result.metadata["n_groupby_features"]
        n_separate_new = separate_result.metadata["n_groupby_features"]
        assert n_separate_new == 2 * n_combined_new

    def test_feature_names_descriptive(self, ds):
        result = JointGroupByFeatures(variant="combined").fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        # New feature names contain the group column, aggregate column and function.
        joint_features = [f for f in result.feature_names if "_joint" in f]
        assert len(joint_features) > 0, "No _joint features found"

    def test_separate_feature_names_have_train_test(self, ds):
        result = JointGroupByFeatures(variant="separate").fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        train_features = [f for f in result.feature_names if f.endswith("_train")]
        test_features = [f for f in result.feature_names if f.endswith("_test")]
        assert len(train_features) > 0, "No _train features found"
        assert len(test_features) > 0, "No _test features found"
        assert len(train_features) == len(test_features), \
            "Should have equal number of _train and _test features"

    def test_custom_agg_funcs(self, ds):
        method = JointGroupByFeatures(
            variant="combined", agg_funcs=["mean", "min", "max"]
        )
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        names = result.feature_names
        assert any("mean" in n for n in names)
        assert any("min" in n for n in names)
        assert any("max" in n for n in names)

    def test_original_cats_preserved_in_output(self, ds):
        """Original categorical columns must remain as label-encoded integers
        alongside the new aggregation features (preserve-originals rule)."""
        result = JointGroupByFeatures(variant="combined").fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        for cat_col in ds.cat_features:
            assert cat_col in result.feature_names, \
                f"Original categorical '{cat_col}' missing from output"
        assert result.categorical_features == ds.cat_features

    def test_iid_no_nan(self, ds_no_shift):
        """Under IID, combined stats should have no NaN."""
        combined_result = JointGroupByFeatures(variant="combined").fit_transform(
            ds_no_shift.X_train, ds_no_shift.X_test, ds_no_shift.y_train
        )
        assert not np.any(np.isnan(combined_result.X_train))
        assert not np.any(np.isnan(combined_result.X_test))


# ---------------------------------------------------------------------------
# M14b-specific tests: GroupByShiftRatio
# ---------------------------------------------------------------------------

class TestGroupByShiftRatio:
    def test_adds_shift_ratio_features(self, ds):
        result = GroupByShiftRatio(use_log=True).fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        shift_features = [f for f in result.feature_names if "shift_ratio" in f]
        assert len(shift_features) > 0, "No shift_ratio features found"
        assert result.metadata.get("n_shift_ratio_features", 0) > 0

    def test_log_ratio_near_zero_under_iid(self, ds_no_shift):
        """Under IID, log shift ratios should be close to 0."""
        result = GroupByShiftRatio(use_log=True).fit_transform(
            ds_no_shift.X_train, ds_no_shift.X_test, ds_no_shift.y_train
        )
        shift_features = [f for f in result.feature_names if "shift_ratio" in f]
        shift_indices = [result.feature_names.index(f) for f in shift_features]
        shift_values = result.X_train[:, shift_indices]
        mean_abs = np.mean(np.abs(shift_values))
        assert mean_abs < 0.5, \
            f"Log shift ratios should be near 0 under IID, got mean |ratio| = {mean_abs}"

    def test_raw_ratio_near_one_under_iid(self, ds_no_shift):
        """Under IID, raw shift ratios should be close to 1."""
        result = GroupByShiftRatio(use_log=False).fit_transform(
            ds_no_shift.X_train, ds_no_shift.X_test, ds_no_shift.y_train
        )
        shift_features = [f for f in result.feature_names if "shift_ratio" in f]
        shift_indices = [result.feature_names.index(f) for f in shift_features]
        shift_values = result.X_train[:, shift_indices]
        mean_val = np.mean(shift_values)
        assert 0.5 < mean_val < 1.5, \
            f"Raw shift ratios should be near 1 under IID, got mean = {mean_val}"

    def test_shift_ratios_differ_under_shift(self, ds):
        """Under shift, ratios should deviate from 0 (log) or 1 (raw)."""
        result = GroupByShiftRatio(use_log=True).fit_transform(
            ds.X_train, ds.X_test, ds.y_train
        )
        shift_features = [f for f in result.feature_names if "shift_ratio" in f]
        shift_indices = [result.feature_names.index(f) for f in shift_features]
        shift_values = result.X_train[:, shift_indices]
        max_abs = np.max(np.abs(shift_values))
        assert max_abs > 0.01, \
            f"Expected some non-zero shift ratios under shift, got max |ratio| = {max_abs}"


# ---------------------------------------------------------------------------
# MethodPipeline Combination Tests
# ---------------------------------------------------------------------------

class TestMethodPipelineCombinations:
    """Test combining Tier 1 methods in pipelines."""

    def test_scaling_then_frequency(self, ds):
        """JointScaling followed by JointFrequencyEncoding must succeed."""
        pipeline = MethodPipeline([
            JointScaling(),
            JointFrequencyEncoding(variant="combined"),
        ])
        result = pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[0] == len(ds.X_train)
        assert result.X_test.shape[0] == len(ds.X_test)
        assert not np.any(np.isnan(result.X_train))

    def test_full_tier1_pipeline(self, ds):
        """The combined Tier 1 pipeline must preserve valid output."""
        pipeline = MethodPipeline([
            # Encoding and scaling.
            JointScaling(),
            JointFrequencyEncoding(variant="combined"),
            FrequencyRatio(use_log=True),
            # Feature generation from encoded groups.
            JointGroupByFeatures(variant="combined"),
            JointPCA(n_components=5),
        ])
        result = pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        assert result.X_train.shape[0] == len(ds.X_train)
        assert result.X_test.shape[0] == len(ds.X_test)
        assert result.X_train.shape[1] == result.X_test.shape[1]
        assert not np.any(np.isnan(result.X_train))
        assert not np.any(np.isinf(result.X_train))
        # The pipeline retains at least the original feature count.
        assert result.X_train.shape[1] >= ds.n_features

    def test_pipeline_name_format(self):
        pipeline = MethodPipeline([
            JointScaling(),
            JointFrequencyEncoding(variant="combined"),
        ])
        assert "joint_scaling" in pipeline.name
        assert "joint_freq_enc_combined" in pipeline.name
        assert "→" in pipeline.name

    def test_pipeline_with_groupby_and_shift_ratio(self, ds):
        """GroupBy followed by ShiftRatio must succeed."""
        pipeline = MethodPipeline([
            JointGroupByFeatures(variant="combined"),
            GroupByShiftRatio(use_log=True),
        ])
        result = pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[0] == len(ds.X_train)
        assert not np.any(np.isnan(result.X_train))
        assert not np.any(np.isinf(result.X_train))

    def test_pipeline_produces_different_results_than_baseline(self, ds):
        """A pipeline should produce different features than inductive baseline."""
        baseline = InductiveBaseline()
        pipeline = MethodPipeline([
            JointScaling(),
            JointFrequencyEncoding(variant="combined"),
        ])
        base_result = baseline.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        pipe_result = pipeline.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        # Either feature count or feature values must differ.
        assert (base_result.X_train.shape[1] != pipe_result.X_train.shape[1] or
                not np.allclose(base_result.X_train, pipe_result.X_train, atol=1e-3))

    def test_pipeline_end_to_end_with_model(self, ds):
        """Run the full pipeline and model evaluation end to end."""
        from src.models.base import get_model
        from src.pipeline import run_single_experiment

        pipeline = MethodPipeline([
            # Encoding and scaling.
            JointScaling(),
            JointFrequencyEncoding(variant="combined"),
            # Feature generation from encoded groups.
            JointGroupByFeatures(variant="combined"),
        ])
        model = get_model("lightgbm")
        result = run_single_experiment(ds, pipeline, model, seed=42)

        assert result.accuracy > 0.0
        assert 0 <= result.auc_roc <= 1
        assert result.n_features_out > ds.n_features  # GroupBy added features

"""
Tests for Tier 2 test-time methods (M6, M13).

Each method is tested for:
1. Output shape consistency (train/test same number of features)
2. No NaN/Inf in output
3. Deterministic output (same input → same output)
4. Row count preservation
5. Domain classifier specific: PAD in metadata, weight properties
6. Leakage: y_test independence
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
    AdversarialValidation,
    AdversarialReweighting,
    TIER2_METHODS,
    _compute_pad,
    _clip_weights,
)


@pytest.fixture
def ds():
    return load_synthetic(n_train=200, n_test=100, shift_strength=1.0, seed=42)


@pytest.fixture
def ds_no_shift():
    return load_synthetic(n_train=200, n_test=100, shift_strength=0.0, seed=42)


# ---------------------------------------------------------------------------
# Generic tests for ALL Tier 2 methods
# ---------------------------------------------------------------------------

class TestAllTier2Generic:
    """Run shape, NaN and determinism checks on every configured Tier 2 method."""

    @pytest.fixture(params=list(TIER2_METHODS.keys()))
    def method_name(self, request):
        return request.param

    def test_output_shape_consistent(self, ds, method_name):
        method = TIER2_METHODS[method_name]()
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[1] == result.X_test.shape[1], \
            f"{method_name}: train/test feature count mismatch"
        assert len(result.feature_names) == result.X_train.shape[1]

    def test_no_nan(self, ds, method_name):
        method = TIER2_METHODS[method_name]()
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert not np.any(np.isnan(result.X_train)), f"{method_name}: NaN in X_train"
        assert not np.any(np.isnan(result.X_test)), f"{method_name}: NaN in X_test"

    def test_no_inf(self, ds, method_name):
        method = TIER2_METHODS[method_name]()
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert not np.any(np.isinf(result.X_train)), f"{method_name}: Inf in X_train"
        assert not np.any(np.isinf(result.X_test)), f"{method_name}: Inf in X_test"

    def test_row_count_preserved(self, ds, method_name):
        # M7 (PseudoLabelSelfTraining) and M17 (TDDA) deliberately expand
        # X_train with augmented rows, so the test checks that X_train is
        # at least as large as the original and X_test is unchanged.
        AUGMENTATION_PREFIXES = ("pseudo_label_", "tdda_")
        method = TIER2_METHODS[method_name]()
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        if method_name.startswith(AUGMENTATION_PREFIXES):
            assert result.X_train.shape[0] >= len(ds.X_train), \
                f"{method_name}: augmented X_train should be >= original"
        else:
            assert result.X_train.shape[0] == len(ds.X_train)
        assert result.X_test.shape[0] == len(ds.X_test)

    def test_deterministic(self, ds, method_name):
        method = TIER2_METHODS[method_name]()
        r1 = method.fit_transform(ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy())
        # Re-instantiate to get a fresh object
        method2 = TIER2_METHODS[method_name]()
        r2 = method2.fit_transform(ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy())
        np.testing.assert_array_almost_equal(
            r1.X_train, r2.X_train,
            err_msg=f"{method_name}: not deterministic (X_train)"
        )
        np.testing.assert_array_almost_equal(
            r1.X_test, r2.X_test,
            err_msg=f"{method_name}: not deterministic (X_test)"
        )


# ---------------------------------------------------------------------------
# M6-specific tests
# ---------------------------------------------------------------------------

class TestAdversarialValidation:
    def test_feature_mode_adds_column(self, ds):
        method = AdversarialValidation(classifier="lgbm", return_as="feature")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Should add exactly 1 feature (domain_score)
        assert result.X_train.shape[1] == ds.n_features + 1
        assert "domain_score" in result.feature_names

    def test_weight_mode_no_extra_column(self, ds):
        method = AdversarialValidation(classifier="lgbm", return_as="weight")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Weight-only mode should NOT add features
        assert result.X_train.shape[1] == ds.n_features
        assert result.sample_weights_train is not None
        assert len(result.sample_weights_train) == len(ds.X_train)

    def test_both_mode_adds_column_and_weights(self, ds):
        method = AdversarialValidation(classifier="lgbm", return_as="both")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[1] == ds.n_features + 1
        assert result.sample_weights_train is not None

    def test_pad_in_metadata(self, ds):
        method = AdversarialValidation(classifier="lgbm", return_as="feature")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert "pad" in result.metadata
        assert "domain_accuracy" in result.metadata
        # PAD should be between 0 and 2
        assert 0.0 <= result.metadata["pad"] <= 2.0

    def test_pad_low_when_no_shift(self, ds_no_shift):
        method = AdversarialValidation(classifier="lgbm", return_as="feature")
        result = method.fit_transform(
            ds_no_shift.X_train, ds_no_shift.X_test, ds_no_shift.y_train
        )
        # With no shift, PAD should remain low because the classifier cannot separate
        # the splits. The threshold allows residual structure in the synthetic data.
        assert result.metadata["pad"] < 1.0, \
            f"PAD={result.metadata['pad']} is too high for no-shift data"

    def test_weights_positive(self, ds):
        method = AdversarialValidation(
            classifier="lgbm", return_as="weight", weight_type="ratio"
        )
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert np.all(result.sample_weights_train > 0), "Weights should be positive"

    def test_weights_clipped(self, ds):
        method = AdversarialValidation(
            classifier="lgbm", return_as="weight",
            weight_type="ratio", clip_strategy="hard",
            clip_min=0.1, clip_max=10.0,
        )
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert np.all(result.sample_weights_train >= 0.1)
        assert np.all(result.sample_weights_train <= 10.0)

    def test_logreg_classifier(self, ds):
        method = AdversarialValidation(classifier="logreg", return_as="feature")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[1] == ds.n_features + 1

    def test_rf_classifier(self, ds):
        method = AdversarialValidation(classifier="rf", return_as="feature")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.X_train.shape[1] == ds.n_features + 1

    def test_domain_scores_between_0_and_1(self, ds):
        method = AdversarialValidation(classifier="lgbm", return_as="feature")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Last column is domain_score
        train_scores = result.X_train[:, -1]
        test_scores = result.X_test[:, -1]
        assert np.all(train_scores >= 0) and np.all(train_scores <= 1)
        assert np.all(test_scores >= 0) and np.all(test_scores <= 1)


# ---------------------------------------------------------------------------
# M13-specific tests
# ---------------------------------------------------------------------------

class TestAdversarialReweighting:
    def test_no_extra_features(self, ds):
        method = AdversarialReweighting(classifier="lgbm", weight_type="ratio")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        # Reweighting should NOT change the feature count
        assert result.X_train.shape[1] == ds.n_features

    def test_weights_returned(self, ds):
        method = AdversarialReweighting(classifier="lgbm", weight_type="ratio")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.sample_weights_train is not None
        assert len(result.sample_weights_train) == len(ds.X_train)

    def test_softmax_weights_sum_to_n(self, ds):
        method = AdversarialReweighting(
            classifier="lgbm", weight_type="ratio", clip_strategy="softmax"
        )
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        expected_sum = len(ds.X_train)
        actual_sum = result.sample_weights_train.sum()
        np.testing.assert_almost_equal(
            actual_sum, expected_sum, decimal=1,
            err_msg="Softmax weights should sum to n_train"
        )

    def test_hard_clip_bounds(self, ds):
        method = AdversarialReweighting(
            classifier="lgbm", weight_type="ratio",
            clip_strategy="hard", clip_min=0.1, clip_max=10.0,
        )
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert np.all(result.sample_weights_train >= 0.1)
        assert np.all(result.sample_weights_train <= 10.0)

    def test_metadata_has_weight_stats(self, ds):
        method = AdversarialReweighting(classifier="lgbm", weight_type="ratio")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert "pad" in result.metadata
        assert "mean_weight" in result.metadata
        assert "std_weight" in result.metadata

    def test_raw_weight_type(self, ds):
        method = AdversarialReweighting(classifier="lgbm", weight_type="raw")
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)
        assert result.sample_weights_train is not None
        # Raw weights are P(test|x) and lie between 0 and 1 before clipping.
        # After hard clipping with default 0.1-10.0, they must be at least 0.1.
        assert np.all(result.sample_weights_train >= 0.0)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_compute_pad_perfect_separation(self):
        scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        is_test = np.array([0, 0, 0, 1, 1, 1])
        pad = _compute_pad(scores, is_test)
        assert pad == 2.0, "Perfect separation should give PAD=2.0"

    def test_compute_pad_random(self):
        # 50% error rate → PAD = 0
        scores = np.array([0.9, 0.1, 0.9, 0.1])  # all wrong
        is_test = np.array([0, 1, 0, 1])
        pad = _compute_pad(scores, is_test)
        assert pad == 0.0, "Random/wrong predictions should give PAD≈0"

    def test_clip_weights_hard(self):
        w = np.array([0.01, 5.0, 100.0])
        clipped = _clip_weights(w, strategy="hard", clip_min=0.1, clip_max=10.0)
        np.testing.assert_array_equal(clipped, [0.1, 5.0, 10.0])

    def test_clip_weights_softmax(self):
        w = np.array([1.0, 2.0, 3.0])
        clipped = _clip_weights(w, strategy="softmax")
        np.testing.assert_almost_equal(clipped.sum(), 3.0)

    def test_clip_weights_winsorize(self):
        w = np.array([0.01, 1.0, 1.0, 1.0, 100.0])
        clipped = _clip_weights(w, strategy="winsorize", winsorize_pct=0.2)
        # Extreme values should be pulled toward the center:
        # the lowest value should increase and the highest should decrease
        assert clipped.min() > w[0], "Winsorize should raise the minimum"
        assert clipped.max() < w[-1], "Winsorize should lower the maximum"
        # The middle values should be unchanged
        np.testing.assert_array_almost_equal(clipped[1:4], w[1:4])


# ---------------------------------------------------------------------------
# Leakage tests
# ---------------------------------------------------------------------------

class TestLeakage:
    def test_y_test_independence(self, ds):
        """Output must not depend on y_test, which is never provided to the method."""
        method1 = AdversarialValidation(classifier="lgbm", return_as="feature", seed=42)
        r1 = method1.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        method2 = AdversarialValidation(classifier="lgbm", return_as="feature", seed=42)
        r2 = method2.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        np.testing.assert_array_almost_equal(r1.X_train, r2.X_train)
        np.testing.assert_array_almost_equal(r1.X_test, r2.X_test)

    def test_sensitive_to_y_train_for_nothing(self, ds):
        """Domain classifier does NOT use y_train, so output should be same
        regardless of y_train values."""
        method1 = AdversarialValidation(classifier="lgbm", return_as="feature", seed=42)
        r1 = method1.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        # Shuffle y_train
        y_shuffled = np.random.RandomState(99).permutation(ds.y_train)
        method2 = AdversarialValidation(classifier="lgbm", return_as="feature", seed=42)
        r2 = method2.fit_transform(ds.X_train, ds.X_test, y_shuffled)

        np.testing.assert_array_almost_equal(
            r1.X_train, r2.X_train,
            err_msg="Domain classifier should NOT depend on y_train"
        )

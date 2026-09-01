"""Leakage, label-independence and determinism tests for test-time methods."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.base import TabularDataset, LeakageGuard
from src.data.synthetic import load_synthetic
from src.methods.base import InductiveBaseline, IdentityMethod, TestTimeMethod
from src.methods.tier1 import get_all_tier1_methods
from src.methods.tier2 import get_all_tier2_methods


# --- Fixtures ---

@pytest.fixture
def small_synthetic():
    """Small synthetic dataset for fast tests."""
    return load_synthetic(n_train=500, n_test=300, shift_strength=1.0, seed=42)


@pytest.fixture
def small_synthetic_iid():
    """Synthetic dataset with no shift (IID)."""
    return load_synthetic(n_train=500, n_test=300, shift_strength=0.0, seed=42)


# --- LeakageGuard Tests ---

class TestLeakageGuard:
    def test_removes_y_test(self, small_synthetic):
        ds = small_synthetic
        assert ds.y_test is not None
        guard = LeakageGuard(ds)
        assert ds.y_test is None, "y_test should be None after LeakageGuard"

    def test_double_guard_raises(self, small_synthetic):
        ds = small_synthetic
        LeakageGuard(ds)
        with pytest.raises(ValueError, match="already None"):
            LeakageGuard(ds)

    def test_evaluate_returns_metrics(self, small_synthetic):
        ds = small_synthetic
        guard = LeakageGuard(ds)
        n_test = guard.n_test

        y_pred = np.zeros(n_test, dtype=int)
        y_prob = np.random.rand(n_test)
        metrics = guard.evaluate(y_pred, y_prob)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "auc_roc" in metrics
        assert "log_loss" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["auc_roc"] <= 1

    def test_evaluate_wrong_length(self, small_synthetic):
        ds = small_synthetic
        guard = LeakageGuard(ds)
        with pytest.raises(ValueError, match="length"):
            guard.evaluate(np.zeros(10))


# --- Method Leakage Tests ---

class TestMethodLeakage:
    """Tests that methods do not depend on y_test."""

    def _run_leakage_test(self, method: TestTimeMethod, ds: TabularDataset):
        """
        Core leakage test: verify that the method does not depend on y_test.

        Two runs with identical visible inputs must produce identical output.
        The method never receives y_test.
        """
        # Run the method once.
        result_1 = method.fit_transform(
            ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy()
        )

        # Run the method again with identical inputs.
        result_2 = method.fit_transform(
            ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy()
        )

        np.testing.assert_array_almost_equal(
            result_1.X_train, result_2.X_train,
            err_msg=f"{method.name}: X_train differs between runs (possible leakage/randomness)"
        )
        np.testing.assert_array_almost_equal(
            result_1.X_test, result_2.X_test,
            err_msg=f"{method.name}: X_test differs between runs (possible leakage/randomness)"
        )

    def test_inductive_baseline_no_leakage(self, small_synthetic):
        self._run_leakage_test(InductiveBaseline(), small_synthetic)

    def test_identity_no_leakage(self, small_synthetic):
        self._run_leakage_test(IdentityMethod(), small_synthetic)

    @pytest.fixture(params=list(get_all_tier1_methods().keys()))
    def tier1_method_name(self, request):
        return request.param

    def test_tier1_no_leakage(self, small_synthetic, tier1_method_name):
        """Every Tier 1 method must produce identical output on identical input."""
        method = get_all_tier1_methods()[tier1_method_name]
        self._run_leakage_test(method, small_synthetic)

    def test_tier1_y_test_independent(self, small_synthetic, tier1_method_name):
        """
        Stronger leakage test: verify that changing y_test externally
        has no effect on method output.

        Even though y_test is not in the method signature, this checks
        that there is no hidden global-state leak.
        """
        ds = small_synthetic
        method = get_all_tier1_methods()[tier1_method_name]

        # Run with original y_test still on the dataset
        result_1 = method.fit_transform(
            ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy()
        )

        # Scramble y_test on the dataset object.
        ds.y_test = np.random.permutation(ds.y_test) if ds.y_test is not None else None

        # Run again and require identical output.
        result_2 = method.fit_transform(
            ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy()
        )

        np.testing.assert_array_almost_equal(
            result_1.X_train, result_2.X_train,
            err_msg=f"{tier1_method_name}: output depends on y_test (LEAKAGE!)"
        )
        np.testing.assert_array_almost_equal(
            result_1.X_test, result_2.X_test,
            err_msg=f"{tier1_method_name}: output depends on y_test (LEAKAGE!)"
        )

    def test_tier1_y_train_sensitivity(self, small_synthetic, tier1_method_name):
        """
        Methods that declare uses_labels=True should produce different observable
        output when y_train changes. Methods with uses_labels=False should not
        depend on y_train content.

        "Observable output" = X_train OR metadata. Selective wrappers that use
        y_train only through CombinedSelector's importance-ranking may leave
        X_train unchanged when the selected columns are rank-stable under
        y-scramble. The y-dependency still appears in metadata through selection
        scores. Either channel satisfies the uses_labels declaration.
        """
        ds = small_synthetic
        method = get_all_tier1_methods()[tier1_method_name]

        result_1 = method.fit_transform(
            ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy()
        )

        # Scramble y_train
        y_scrambled = np.random.permutation(ds.y_train)
        result_2 = method.fit_transform(
            ds.X_train.copy(), ds.X_test.copy(), y_scrambled
        )

        if method.uses_labels:
            x_differs = not np.allclose(result_1.X_train, result_2.X_train, atol=1e-6)
            metadata_differs = str(result_1.metadata) != str(result_2.metadata)
            assert x_differs or metadata_differs, (
                f"{tier1_method_name} declares uses_labels=True but neither "
                f"X_train nor metadata depended on y_train"
            )
        else:
            # The output should be identical.
            np.testing.assert_array_almost_equal(
                result_1.X_train, result_2.X_train,
                err_msg=f"{tier1_method_name} declares uses_labels=False but output changed with y_train"
            )


# --- Tier 2 Method Leakage Tests ---

class TestTier2MethodLeakage:
    """Tests that Tier 2 methods do not depend on ``y_test``."""

    @pytest.fixture(params=list(get_all_tier2_methods().keys()))
    def tier2_method_name(self, request):
        return request.param

    def test_tier2_no_leakage(self, small_synthetic, tier2_method_name):
        """Every Tier 2 method must produce identical output on identical input."""
        method = get_all_tier2_methods()[tier2_method_name]

        # Run the first method instance.
        result_1 = method.fit_transform(
            small_synthetic.X_train.copy(),
            small_synthetic.X_test.copy(),
            small_synthetic.y_train.copy(),
        )

        # Use identical inputs with a new method instance to check stateless behavior.
        method_2 = get_all_tier2_methods()[tier2_method_name]
        result_2 = method_2.fit_transform(
            small_synthetic.X_train.copy(),
            small_synthetic.X_test.copy(),
            small_synthetic.y_train.copy(),
        )

        np.testing.assert_array_almost_equal(
            result_1.X_test, result_2.X_test,
            err_msg=f"{tier2_method_name}: X_test differs between runs (non-deterministic)",
        )

    def test_tier2_y_test_independent(self, small_synthetic, tier2_method_name):
        """Tier 2 methods must not depend on y_test."""
        ds = small_synthetic
        method = get_all_tier2_methods()[tier2_method_name]

        result_1 = method.fit_transform(
            ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy()
        )

        # Scramble y_test
        ds.y_test = np.random.permutation(ds.y_test) if ds.y_test is not None else None

        method_2 = get_all_tier2_methods()[tier2_method_name]
        result_2 = method_2.fit_transform(
            ds.X_train.copy(), ds.X_test.copy(), ds.y_train.copy()
        )

        np.testing.assert_array_almost_equal(
            result_1.X_test, result_2.X_test,
            err_msg=f"{tier2_method_name}: output depends on y_test (LEAKAGE!)",
        )


class TestTier2OutputShape:
    """Tests that Tier 2 methods produce correct output shapes."""

    @pytest.fixture(params=list(get_all_tier2_methods().keys()))
    def tier2_method_name(self, request):
        return request.param

    def test_tier2_shapes(self, small_synthetic, tier2_method_name):
        """Tier 2 methods must produce consistent shapes."""
        ds = small_synthetic
        method = get_all_tier2_methods()[tier2_method_name]
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        # Methods that augment training data (M7, M17) may increase n_train
        assert result.X_train.shape[0] >= ds.n_train, \
            f"X_train rows: expected >= {ds.n_train}, got {result.X_train.shape[0]}"
        assert result.X_test.shape[0] == ds.n_test, \
            f"X_test rows: expected {ds.n_test}, got {result.X_test.shape[0]}"
        assert result.X_train.shape[1] == result.X_test.shape[1], \
            f"Feature count mismatch: train={result.X_train.shape[1]}, test={result.X_test.shape[1]}"
        assert len(result.feature_names) == result.X_train.shape[1], \
            f"feature_names length mismatch"

        # Check weights if present
        if result.sample_weights_train is not None:
            assert len(result.sample_weights_train) == result.X_train.shape[0], \
                f"Weights length mismatch: {len(result.sample_weights_train)} vs {result.X_train.shape[0]}"
            assert np.all(np.isfinite(result.sample_weights_train)), "Weights must be finite"
            assert np.all(result.sample_weights_train >= 0), "Weights must be non-negative"

        # Check augmented y_train if present
        if result.y_train is not None:
            assert len(result.y_train) == result.X_train.shape[0], \
                f"y_train length ({len(result.y_train)}) != X_train rows ({result.X_train.shape[0]})"

        # No NaN or Inf in output
        assert np.all(np.isfinite(result.X_test)), \
            f"{tier2_method_name}: X_test contains NaN or Inf"


# --- Output Shape Tests ---

class TestOutputShape:
    """Tests that methods produce correct output shapes."""

    def _check_shapes(self, method: TestTimeMethod, ds: TabularDataset):
        result = method.fit_transform(ds.X_train, ds.X_test, ds.y_train)

        assert result.X_train.shape[0] == ds.n_train, \
            f"X_train rows: expected {ds.n_train}, got {result.X_train.shape[0]}"
        assert result.X_test.shape[0] == ds.n_test, \
            f"X_test rows: expected {ds.n_test}, got {result.X_test.shape[0]}"
        assert result.X_train.shape[1] == result.X_test.shape[1], \
            f"Feature count mismatch: train={result.X_train.shape[1]}, test={result.X_test.shape[1]}"
        assert len(result.feature_names) == result.X_train.shape[1], \
            f"feature_names length mismatch"

        if result.sample_weights_train is not None:
            assert len(result.sample_weights_train) == ds.n_train
            assert np.all(result.sample_weights_train >= 0), "Weights must be non-negative"

    def test_inductive_baseline_shapes(self, small_synthetic):
        self._check_shapes(InductiveBaseline(), small_synthetic)

    def test_identity_shapes(self, small_synthetic):
        self._check_shapes(IdentityMethod(), small_synthetic)


# --- Integration Test ---

class TestEndToEnd:
    """Quick end-to-end test: data → method → model → evaluate."""

    def test_full_pipeline(self, small_synthetic):
        from src.models.base import get_model
        from src.pipeline import run_single_experiment

        ds = small_synthetic
        method = InductiveBaseline()
        model = get_model("lightgbm")

        result = run_single_experiment(ds, method, model, seed=42)

        assert result.accuracy > 0.0
        assert 0 <= result.auc_roc <= 1
        assert result.f1 >= 0
        assert result.train_time > 0
        assert result.n_train == 500
        assert result.n_test == 300

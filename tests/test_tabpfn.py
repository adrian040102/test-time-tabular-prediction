"""Unit tests for :class:`TabPFNWrapper`.

The tests cover basic CPU execution, size guards and pre-scaling contracts.
TabPFN is optional, so the module is skipped when the dependency is unavailable.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

tabpfn = pytest.importorskip("tabpfn")

from src.models.base import get_model  # noqa: E402
from src.models.tabpfn import (  # noqa: E402
    CPU_MAX_TRAIN_ROWS,
    MAX_CLASSES,
    MAX_FEATURES,
    MAX_TRAIN_ROWS,
    TabPFNSkipReason,
    TabPFNWrapper,
)

warnings.filterwarnings("ignore")


# --- Fixtures -----------------------------------------------------------


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n, d = 200, 6
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    y = (X.values @ rng.normal(size=d) > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(1)
    n, d, k = 240, 5, 3
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    W = rng.normal(size=(d, k))
    y = np.argmax(X.values @ W, axis=1)
    return X, y


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(2)
    n, d = 200, 5
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    coef = rng.normal(size=d)
    y = X.values @ coef + rng.normal(scale=0.1, size=n)
    return X, y


def _make_wrapper(task_type: str, **overrides):
    """Build a wrapper with minimal CPU settings for basic execution tests."""
    kwargs = dict(
        task_type=task_type,
        seed=42,
        device="cpu",
        n_estimators=2,
        ignore_pretraining_limits=True,
        cpu_max_rows=10_000,
    )
    kwargs.update(overrides)
    return TabPFNWrapper(**kwargs)


# Binary classification


class TestFitPredictBinary:
    def test_basic_fit_and_predict(self, binary_data):
        X, y = binary_data
        m = _make_wrapper("classification")
        m.fit(X, y)

        preds = m.predict(X)
        proba = m.predict_proba(X)

        assert preds.shape == (len(X),)
        # Binary output is one-dimensional P(class=1).
        assert proba.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})
        assert (proba >= 0).all() and (proba <= 1).all()
        assert m.model_ is not None
        assert m.classes_ is not None and len(m.classes_) == 2


# Multiclass classification


class TestFitPredictMulticlass:
    def test_basic_fit_and_predict(self, multiclass_data):
        X, y = multiclass_data
        m = _make_wrapper("classification")
        m.fit(X, y)

        preds = m.predict(X)
        proba = m.predict_proba(X)

        assert preds.shape == (len(X),)
        assert proba.shape == (len(X), 3)
        assert set(np.unique(preds)).issubset({0, 1, 2})
        # Probabilities sum to ~1 row-wise.
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)


# Regression


class TestFitPredictRegression:
    def test_basic_fit_and_predict(self, regression_data):
        X, y = regression_data
        m = _make_wrapper("regression")
        m.fit(X, y)

        preds = m.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()
        # Regression has no predict_proba.
        assert m.predict_proba(X) is None

        # Training-set RMSE must be below the target standard deviation.
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        assert rmse < y.std()


# --- 4. Size guards ----------------------------------------------------
#
# The guards run before TabPFN initialization. Inputs use minimal row or feature
# allocations where possible.


class TestSizeGuards:
    def test_train_rows_exceeded(self):
        n = MAX_TRAIN_ROWS + 1
        X = pd.DataFrame(np.zeros((n, 3)), columns=list("abc"))
        y = np.zeros(n, dtype=int)
        y[: n // 2] = 1
        m = _make_wrapper("classification")
        with pytest.raises(TabPFNSkipReason) as exc:
            m.fit(X, y)
        assert exc.value.reason == "train_rows_exceeded"
        assert exc.value.details["train_rows"] == n
        assert exc.value.details["limit"] == MAX_TRAIN_ROWS

    def test_features_exceeded(self):
        d = MAX_FEATURES + 1
        X = pd.DataFrame(np.zeros((20, d)), columns=[f"f{i}" for i in range(d)])
        y = np.array([0, 1] * 10)
        m = _make_wrapper("classification")
        with pytest.raises(TabPFNSkipReason) as exc:
            m.fit(X, y)
        assert exc.value.reason == "features_exceeded"
        assert exc.value.details["features"] == d
        assert exc.value.details["limit"] == MAX_FEATURES

    def test_classes_exceeded(self):
        n_classes = MAX_CLASSES + 1
        # n_samples must be ≥ n_classes. Make each class appear once.
        X = pd.DataFrame(np.eye(n_classes), columns=[f"f{i}" for i in range(n_classes)])
        y = np.arange(n_classes)
        m = _make_wrapper("classification")
        with pytest.raises(TabPFNSkipReason) as exc:
            m.fit(X, y)
        assert exc.value.reason == "classes_exceeded"
        assert exc.value.details["n_classes"] == n_classes
        assert exc.value.details["limit"] == MAX_CLASSES

    def test_cpu_too_large(self):
        # Use the default cpu_max_rows (1000).  Do not override.
        n = CPU_MAX_TRAIN_ROWS + 1
        X = pd.DataFrame(np.zeros((n, 3)), columns=list("abc"))
        y = np.zeros(n, dtype=int)
        y[: n // 2] = 1
        m = TabPFNWrapper(
            task_type="classification",
            seed=42,
            device="cpu",
            n_estimators=2,
            ignore_pretraining_limits=True,
        )
        with pytest.raises(TabPFNSkipReason) as exc:
            m.fit(X, y)
        assert exc.value.reason == "cpu_too_large"
        assert exc.value.details["train_rows"] == n
        assert exc.value.details["cpu_limit"] == CPU_MAX_TRAIN_ROWS

    def test_regression_no_class_guard(self, regression_data):
        """Regression must skip the class guard but retain the CPU row guard."""
        n = CPU_MAX_TRAIN_ROWS + 1
        X = pd.DataFrame(np.zeros((n, 3)), columns=list("abc"))
        y = np.linspace(0, 1, n)
        m = TabPFNWrapper(
            task_type="regression",
            seed=42,
            device="cpu",
            n_estimators=2,
            ignore_pretraining_limits=True,
        )
        with pytest.raises(TabPFNSkipReason) as exc:
            m.fit(X, y)
        assert exc.value.reason == "cpu_too_large"


# --- 5. Method-output compatibility ------------------------------------


class TestMethodOutputCompatibility:
    def test_was_prescaled_accepted(self, binary_data):
        X, y = binary_data
        m = _make_wrapper("classification")
        m.fit(X, y, _was_prescaled=True)
        assert m.model_ is not None

    def test_default_fit_succeeds(self, binary_data):
        """The wrapper must fit without the pre-scaling flag."""
        X, y = binary_data
        m = _make_wrapper("classification")
        m.fit(X, y)  # no exception
        assert m.model_ is not None


# --- 6. Model lookup wiring --------------------------------------------


class TestModelLookup:
    def test_get_model_returns_wrapper(self):
        m = get_model(
            "tabpfn_v2",
            task_type="classification",
            seed=42,
            device="cpu",
            n_estimators=2,
            ignore_pretraining_limits=True,
            cpu_max_rows=10_000,
        )
        assert isinstance(m, TabPFNWrapper)
        assert m.name == "tabpfn_v2"

    def test_prefers_raw_features_flag(self):
        m = get_model("tabpfn_v2", task_type="classification", seed=42, device="cpu")
        assert getattr(m, "prefers_raw_features", False) is True

    def test_accepts_method_generated_output_flag(self):
        m = get_model("tabpfn_v2", task_type="classification", seed=42, device="cpu")
        assert getattr(m, "refuses_prescaled_input", False) is False


# --- 7. Categorical-feature handling -----------------------------------


class TestCategoricalFeature:
    def test_cat_indices_resolve_from_names(self, binary_data):
        """``_cat_indices`` converts names to zero-based column indices."""
        X, _ = binary_data
        m = _make_wrapper("classification", categorical_features=["f0", "f3"])
        idx = m._cat_indices(X, None)
        assert idx == [0, 3]

    def test_cat_indices_explicit_overrides(self, binary_data):
        """The per-call ``explicit`` argument overrides the constructor list."""
        X, _ = binary_data
        m = _make_wrapper("classification", categorical_features=["f0"])
        idx = m._cat_indices(X, ["f2", "f4"])
        assert idx == [2, 4]

    def test_cat_indices_missing_columns_dropped(self, binary_data):
        """Names absent from ``X.columns`` are excluded."""
        X, _ = binary_data
        m = _make_wrapper("classification", categorical_features=["f0", "no_such_col"])
        idx = m._cat_indices(X, None)
        assert idx == [0]

    def test_cat_indices_returns_none_when_empty(self, binary_data):
        X, _ = binary_data
        m = _make_wrapper("classification")
        assert m._cat_indices(X, None) is None
        assert m._cat_indices(X, []) is None

    def test_fit_with_cat_kwarg_runs(self, binary_data):
        """End-to-end fit with declared categoricals produces predictions.

        ``f5`` has four unique integer codes and is declared categorical.
        """
        X, y = binary_data
        # Replace one column with low-cardinality integer codes.
        X = X.copy()
        rng = np.random.default_rng(99)
        X["f5"] = rng.integers(0, 4, size=len(X)).astype(float)
        m = _make_wrapper("classification")
        m.fit(X, y, categorical_feature=["f5"])
        proba = m.predict_proba(X)
        assert proba.shape == (len(X),)


# --- 8. Sample-weight handling -----------------------------------------


class TestSampleWeight:
    def test_sample_weight_ignored_with_warning(self, binary_data):
        X, y = binary_data
        m = _make_wrapper("classification")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.fit(X, y, sample_weight=np.ones(len(y)))
        # At least one RuntimeWarning mentioning sample_weight.
        msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        assert any("sample_weight" in s for s in msgs), msgs
        # Fit still proceeds.
        assert m.model_ is not None


# --- 9. Pipeline integration: method-generated output ------------------


class TestPipelineIntegration:
    def test_pipeline_accepts_joint_pca(self):
        """run_single_experiment accepts M3 JointPCA with TabPFNv2."""
        from src.data.base import TabularDataset
        from src.methods.tier1 import JointPCA
        from src.pipeline import run_single_experiment

        rng = np.random.default_rng(7)
        n_tr, n_te, d = 60, 40, 5
        X_tr = pd.DataFrame(
            rng.normal(size=(n_tr, d)), columns=[f"f{i}" for i in range(d)]
        )
        X_te = pd.DataFrame(
            rng.normal(size=(n_te, d)), columns=[f"f{i}" for i in range(d)]
        )
        y_tr = (X_tr.values @ rng.normal(size=d) > 0).astype(int)
        y_te = (X_te.values @ rng.normal(size=d) > 0).astype(int)

        ds = TabularDataset(
            name="prescaled_test_case",
            X_train=X_tr,
            X_test=X_te,
            y_train=y_tr,
            y_test=y_te,
            cat_features=[],
            num_features=list(X_tr.columns),
            task_type="classification",
        )

        method = JointPCA(n_components=2)
        model = _make_wrapper("classification")

        run_single_experiment(
            ds, method, model, seed=42, bootstrap_train=False,
        )

    def test_pipeline_accepts_scaling_safe_method(self, binary_data):
        """The pipeline must accept a method that does not request scaler bypass."""
        from src.data.base import TabularDataset
        from src.methods.base import InductiveBaseline
        from src.pipeline import run_single_experiment

        X, y = binary_data
        # Build train/test splits from the fixture.
        cut = len(X) // 2
        ds = TabularDataset(
            name="non_prescaled_test_case",
            X_train=X.iloc[:cut].reset_index(drop=True),
            X_test=X.iloc[cut:].reset_index(drop=True),
            y_train=y[:cut],
            y_test=y[cut:],
            cat_features=[],
            num_features=list(X.columns),
            task_type="classification",
        )

        result = run_single_experiment(
            ds, InductiveBaseline(), _make_wrapper("classification"),
            seed=42, bootstrap_train=False,
        )
        assert np.isfinite(result.auc_roc)

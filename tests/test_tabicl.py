"""Unit tests for :class:`TabICLWrapper`.

The tests cover basic CPU execution, the size guard and KV-cache reuse. TabICL is
optional, so the module is skipped when the dependency is unavailable.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

tabicl = pytest.importorskip("tabicl")

from src.models.base import get_model  # noqa: E402
from src.models.tabicl import (  # noqa: E402
    MAX_TRAIN_ROWS,
    TabICLSkipReason,
    TabICLWrapper,
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
def regression_data():
    rng = np.random.default_rng(2)
    n, d = 200, 5
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    coef = rng.normal(size=d)
    y = X.values @ coef + rng.normal(scale=0.1, size=n)
    return X, y


def _make_wrapper(task_type: str, **overrides):
    """Build a wrapper with minimal CPU settings."""
    kwargs = dict(
        task_type=task_type,
        seed=42,
        device="cpu",
        n_estimators=1,
        kv_cache=True,
        verbose=False,
    )
    kwargs.update(overrides)
    return TabICLWrapper(**kwargs)


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


# --- 3. KV-cache reuse ------------------------------------------------
#
# With kv_cache=True, fit-side projections are reused across prediction batches.


class TestKVCacheReuse:
    def test_two_batches_run(self, binary_data):
        """Two prediction batches must return independent finite probabilities."""
        X, y = binary_data
        rng = np.random.default_rng(101)
        X_te1 = pd.DataFrame(
            rng.normal(size=(40, X.shape[1])),
            columns=X.columns,
        )
        X_te2 = pd.DataFrame(
            rng.normal(size=(60, X.shape[1])),
            columns=X.columns,
        )

        m = _make_wrapper("classification", kv_cache=True)
        m.fit(X, y)

        p1 = m.predict_proba(X_te1)
        p2 = m.predict_proba(X_te2)

        assert p1.shape == (40,)
        assert p2.shape == (60,)
        assert np.isfinite(p1).all() and np.isfinite(p2).all()

    def test_second_call_not_slower(self, binary_data):
        """The second same-shape prediction must satisfy the cache timing limit."""
        X, y = binary_data
        rng = np.random.default_rng(7)
        X_te1 = pd.DataFrame(
            rng.normal(size=(40, X.shape[1])),
            columns=X.columns,
        )
        X_te2 = pd.DataFrame(
            rng.normal(size=(40, X.shape[1])),
            columns=X.columns,
        )

        m = _make_wrapper("classification", kv_cache=True)
        m.fit(X, y)

        # Perform one unmeasured prediction to initialize PyTorch.
        _ = m.predict_proba(X_te1)

        t0 = time.perf_counter()
        _ = m.predict_proba(X_te1)
        t_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        _ = m.predict_proba(X_te2)
        t_second = time.perf_counter() - t0

        # Both calls use the cache. The second should be within 2x of the first.
        # Floor of 0.05s avoids div-by-zero noise.
        assert t_second < max(2.0 * t_first, 0.05 + t_first), (
            f"KV cache appears disabled: first={t_first:.3f}s, "
            f"second={t_second:.3f}s"
        )


# --- 4. Many-class classification (>10 classes) -----------------------
#
# ``support_many_classes=True`` enables hierarchical classification. The test
# uses 15 classes and requires a probability matrix with 15 columns.
#
# TabICL does not support ``kv_cache=True`` when ``n_classes > 10``. The wrapper
# must disable the cache and issue a warning.


class TestManyClasses:
    @pytest.fixture
    def many_class_data(self):
        rng = np.random.default_rng(11)
        n, d, k = 240, 6, 15  # 16 samples per class
        X = pd.DataFrame(
            rng.normal(size=(n, d)),
            columns=[f"f{i}" for i in range(d)],
        )
        W = rng.normal(size=(d, k))
        y = np.argmax(X.values @ W, axis=1)
        # Ensure all k classes are represented. Otherwise classifier
        # would only see fewer than k.
        for cls in range(k):
            if cls not in y:
                y[cls] = cls
        return X, y, k

    def test_15_classes_runs(self, many_class_data):
        X, y, k = many_class_data
        m = _make_wrapper(
            "classification",
            support_many_classes=True,
            kv_cache=False,  # avoid the auto-downgrade warning
        )
        m.fit(X, y)

        proba = m.predict_proba(X)
        preds = m.predict(X)
        assert proba.shape == (X.shape[0], k)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-3)
        assert preds.shape == (X.shape[0],)
        assert set(np.unique(preds)).issubset(set(range(k)))

    def test_kv_cache_auto_disabled_for_many_classes(self, many_class_data):
        """The wrapper must disable the KV cache and warn above ten classes."""
        X, y, k = many_class_data
        m = _make_wrapper(
            "classification",
            support_many_classes=True,
            kv_cache=True,  # wrapper must downgrade unsupported cache settings
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.fit(X, y)
        msgs = [
            str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)
        ]
        assert any("kv_cache" in s and str(k) in s for s in msgs), msgs
        # Fit still completes and produces predictions.
        assert m.model_ is not None
        proba = m.predict_proba(X)
        assert proba.shape == (X.shape[0], k)


# --- 5. Size guard (500K rows) -----------------------------------------


class TestSizeGuard:
    def test_train_rows_exceeded(self):
        """The size guard must raise above ``MAX_TRAIN_ROWS``."""
        n = MAX_TRAIN_ROWS + 1
        # Test the guard directly before the end-to-end case below.
        m = _make_wrapper("classification")
        with pytest.raises(TabICLSkipReason) as exc:
            m._check_size_guard(n)
        assert exc.value.reason == "train_rows_exceeded"
        assert exc.value.details["train_rows"] == n
        assert exc.value.details["limit"] == MAX_TRAIN_ROWS

    def test_under_limit_passes_guard(self):
        """The size guard must accept values at or below the limit."""
        m = _make_wrapper("classification")
        m._check_size_guard(MAX_TRAIN_ROWS)  # The exact limit is valid.
        m._check_size_guard(1_000)
        m._check_size_guard(0)

    def test_fit_with_oversized_x_raises(self):
        """An oversized DataFrame must raise before model allocation."""
        n = MAX_TRAIN_ROWS + 1
        # Use a small frame whose reported length exceeds the limit.
        class _FakeLen(pd.DataFrame):
            @property
            def _constructor(self):
                return _FakeLen

            def __len__(self):
                return n

        df = _FakeLen({"a": [0.0], "b": [1.0]})
        y = np.array([0])

        m = _make_wrapper("classification")
        with pytest.raises(TabICLSkipReason) as exc:
            m.fit(df, y)
        assert exc.value.reason == "train_rows_exceeded"
        # No TabICL fit should have happened.
        assert m.model_ is None


# --- 6. Model lookup wiring --------------------------------------------


class TestModelLookup:
    def test_get_model_returns_wrapper(self):
        m = get_model(
            "tabicl_v2",
            task_type="classification",
            seed=42,
            device="cpu",
            n_estimators=1,
            verbose=False,
        )
        assert isinstance(m, TabICLWrapper)
        assert m.name == "tabicl_v2"

    def test_prefers_raw_features_flag(self):
        """TabICL must request raw heterogeneous inputs from the pipeline."""
        m = get_model("tabicl_v2", task_type="classification", seed=42, device="cpu")
        assert getattr(m, "prefers_raw_features", False) is True

    def test_refuses_prescaled_input_flag_is_false(self):
        """TabICL must accept pre-scaled inputs."""
        m = get_model("tabicl_v2", task_type="classification", seed=42, device="cpu")
        assert getattr(m, "refuses_prescaled_input", False) is False


# --- 7. Sample-weight handling -----------------------------------------


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


# --- 8. explicit categorical_feature contract -------------------------


class TestCategoricalFeatureKwarg:
    def test_categorical_feature_is_consumed(self, binary_data):
        """The wrapper must consume declared categorical feature names."""
        X, y = binary_data
        m = _make_wrapper("classification")
        # Pipeline forwards categorical_feature when prefers_raw_features
        # is True. The wrapper must consume it without error.
        m.fit(X, y, categorical_feature=["f0", "f3"])
        assert m._categorical_features == ["f0", "f3"]
        assert m.model_ is not None
        assert m.predict(X).shape == (len(X),)

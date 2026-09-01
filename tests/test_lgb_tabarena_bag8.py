"""Unit tests for :class:`LightGBMTabArenaBag8Wrapper`.

These tests cover the wrapper behavior used by ``run_single_experiment``:

* basic fit and prediction behavior on each task type.
* the bag-8 fit takes more than 1.5 times as long as the single-model wrapper.
* the same seed produces the same predictions on the same data.
* the ``categorical_feature`` argument casts columns to pandas ``category`` so
  AutoGluon recognizes them as categorical.

AutoGluon is optional. The module is skipped when AutoGluon is unavailable.
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

autogluon = pytest.importorskip("autogluon.tabular")

from src.models.base import get_model  # noqa: E402
from src.models.lgb_tabarena_bag8 import LightGBMTabArenaBag8Wrapper  # noqa: E402

warnings.filterwarnings("ignore")


# --- Fixtures ------------------------------------------------------------


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n, d = 200, 10
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
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
        model = LightGBMTabArenaBag8Wrapper(task_type="classification", seed=42)
        model.fit(X, y)

        preds = model.predict(X)
        proba = model.predict_proba(X)

        assert preds.shape == (len(X),)
        assert proba.shape == (len(X),)  # Binary output is one-dimensional P(class=1).
        assert set(np.unique(preds)).issubset({0, 1})
        assert (proba >= 0).all() and (proba <= 1).all()
        assert model.predictor_ is not None


# Multiclass classification


class TestFitPredictMulticlass:
    def test_basic_fit_and_predict(self, multiclass_data):
        X, y = multiclass_data
        model = LightGBMTabArenaBag8Wrapper(task_type="classification", seed=42)
        model.fit(X, y)

        preds = model.predict(X)
        proba = model.predict_proba(X)

        assert preds.shape == (len(X),)
        assert proba.shape == (len(X), 3)
        assert set(np.unique(preds)).issubset({0, 1, 2})
        # Probabilities sum to ~1 row-wise (AG may have tiny float drift).
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# Regression


class TestFitPredictRegression:
    def test_basic_fit_and_predict(self, regression_data):
        X, y = regression_data
        model = LightGBMTabArenaBag8Wrapper(task_type="regression", seed=42)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()
        assert model.predict_proba(X) is None

        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        assert rmse < y.std()


# --- 4. Bag-8 timing relative to the single model -----------------------


class TestBag8VsSingleModelTiming:
    """Verify that bag-8 fit duration reflects the eight-model configuration."""

    def test_bag8_slower_than_single(self):
        rng = np.random.default_rng(7)
        n, d = 600, 10
        X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
        y = (X.values @ rng.normal(size=d) > 0).astype(int)

        single = get_model("lightgbm_tabarena", task_type="classification", seed=42)
        t0 = time.perf_counter()
        single.fit(X, y)
        t_single = time.perf_counter() - t0

        bag8 = get_model("lightgbm_tabarena_bag8", task_type="classification", seed=42)
        t0 = time.perf_counter()
        bag8.fit(X, y)
        t_bag8 = time.perf_counter() - t0

        # Bag-8 fit time must exceed 1.5 times the single-model fit time.
        assert t_bag8 > 1.5 * t_single, (
            f"bag-8 fit time ({t_bag8:.2f}s) is not more than 1.5 times the "
            f"single-model fit time ({t_single:.2f}s)"
        )
        # Bag-8 fit time must also exceed one second on this input.
        assert t_bag8 > 1.0, (
            f"bag-8 fit time {t_bag8:.2f}s does not exceed one second"
        )


# --- 5. Reproducibility with the same seed -------------------------------
#
# Independent subprocesses isolate process-global state during reproducibility checks.


def _subprocess_fit_predict_script(
    task_type: str,
    seed: int,
    data_seed: int,
    n: int,
    d: int,
    out_path: str,
) -> str:
    """Return a self-contained Python script that fits the bag-8 wrapper
    on synthetic data and writes predictions to ``out_path`` as .npy."""
    return f"""
import warnings, sys
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, r"{PROJECT_ROOT}")
import numpy as np, pandas as pd
from src.models.lgb_tabarena_bag8 import LightGBMTabArenaBag8Wrapper

rng = np.random.default_rng({data_seed})
X = pd.DataFrame(rng.normal(size=({n}, {d})),
                 columns=[f"f{{i}}" for i in range({d})])
if "{task_type}" == "regression":
    coef = rng.normal(size={d})
    y = X.values @ coef + rng.normal(scale=0.1, size={n})
else:
    y = (X.values @ rng.normal(size={d}) > 0).astype(int)

m = LightGBMTabArenaBag8Wrapper(task_type="{task_type}", seed={seed})
m.fit(X, y)
out = m.predict(X) if "{task_type}" == "regression" else m.predict_proba(X)
np.save(r"{out_path}", out)
"""


def _run_in_subprocess(script: str, tmp_dir: Path, tag: str) -> None:
    """Run the given Python source in a fresh interpreter.

    The script is responsible for writing its own output to disk. This
    helper only starts the process and returns stderr on failure.
    """
    import subprocess

    src_path = tmp_dir / f"script_{tag}.py"
    src_path.write_text(script)
    result = subprocess.run(
        [sys.executable, "-u", str(src_path)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"subprocess fit failed ({tag}):\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


class TestReproducibility:
    """Run each fit in a fresh subprocess to isolate process-global state."""

    def test_same_seed_same_predictions_binary(self, tmp_path):
        out1 = tmp_path / "binary_1.npy"
        out2 = tmp_path / "binary_2.npy"
        script1 = _subprocess_fit_predict_script(
            "classification", seed=42, data_seed=0, n=200, d=10,
            out_path=str(out1),
        )
        script2 = _subprocess_fit_predict_script(
            "classification", seed=42, data_seed=0, n=200, d=10,
            out_path=str(out2),
        )
        # Two independent subprocess fits.
        _run_in_subprocess(script1, tmp_path, "bin_1")
        _run_in_subprocess(script2, tmp_path, "bin_2")
        p1 = np.load(out1)
        p2 = np.load(out2)
        np.testing.assert_allclose(p1, p2, atol=0.0, rtol=0.0)

    def test_same_seed_same_predictions_regression(self, tmp_path):
        out1 = tmp_path / "reg_1.npy"
        out2 = tmp_path / "reg_2.npy"
        script1 = _subprocess_fit_predict_script(
            "regression", seed=42, data_seed=2, n=250, d=6,
            out_path=str(out1),
        )
        script2 = _subprocess_fit_predict_script(
            "regression", seed=42, data_seed=2, n=250, d=6,
            out_path=str(out2),
        )
        _run_in_subprocess(script1, tmp_path, "reg_1")
        _run_in_subprocess(script2, tmp_path, "reg_2")
        p1 = np.load(out1)
        p2 = np.load(out2)
        np.testing.assert_allclose(p1, p2, atol=0.0, rtol=0.0)


# --- 6. Categorical-feature handling ------------------------------------


class TestCategoricalFeature:
    def test_cat_dtype_after_cast(self):
        """``_cast_cats`` returns a frame where declared cols are ``category``."""
        rng = np.random.default_rng(3)
        n = 200
        X = pd.DataFrame({
            "f_num0": rng.normal(size=n),
            "f_num1": rng.normal(size=n),
            "f_cat": rng.integers(0, 4, size=n),
        })
        m = LightGBMTabArenaBag8Wrapper(
            task_type="classification",
            seed=42,
            categorical_features=["f_cat"],
        )
        X_typed = m._cast_cats(X, ["f_cat"])
        assert isinstance(X_typed["f_cat"].dtype, pd.CategoricalDtype)
        # The input frame remains unchanged because the method copies it.
        assert not isinstance(X["f_cat"].dtype, pd.CategoricalDtype)
        # Numeric columns untouched.
        assert pd.api.types.is_float_dtype(X_typed["f_num0"].dtype)

    def test_fit_with_cats_runs(self):
        """End-to-end fit with declared categoricals produces predictions."""
        rng = np.random.default_rng(4)
        n = 250
        X = pd.DataFrame({
            "f_num": rng.normal(size=n),
            "f_cat": rng.integers(0, 4, size=n),
        })
        y = ((X["f_cat"].values * 0.5 + rng.normal(scale=0.3, size=n)) > 1.0).astype(int)
        m = LightGBMTabArenaBag8Wrapper(
            task_type="classification",
            seed=42,
            categorical_features=["f_cat"],
        )
        m.fit(X, y)
        proba = m.predict_proba(X)
        assert proba.shape == (n,)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_cat_arg_via_fit_kwarg(self, binary_data):
        """Per-call ``categorical_feature`` overrides the empty default."""
        X, y = binary_data
        m = LightGBMTabArenaBag8Wrapper(task_type="classification", seed=42)
        # Fit must accept a floating-point column declared as categorical.
        m.fit(X, y, categorical_feature=["f0"])
        assert m.predictor_ is not None


# --- 7. Model lookup wiring ---------------------------------------------


class TestModelLookup:
    def test_get_model_returns_bag8_wrapper(self):
        m = get_model("lightgbm_tabarena_bag8", task_type="classification", seed=42)
        assert isinstance(m, LightGBMTabArenaBag8Wrapper)
        assert m.name == "lightgbm_tabarena_bag8"

    def test_prefers_raw_features_flag(self):
        m = get_model("lightgbm_tabarena_bag8", task_type="classification", seed=42)
        assert getattr(m, "prefers_raw_features", False) is True

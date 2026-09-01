from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import load_dataset
from src.data.base import TabularDataset
from src.methods.baselines import NoScaling
from src.methods.base import (
    InductiveBaseline,
    MethodPipeline,
    RowProvenance,
    TestTimeMethod,
    TransformResult,
    _propagate_seed,
)
from src.methods.column_selection import ColumnSelection, ColumnSelector, AllColumnsSelector
from src.methods.selective import SelectiveMethod
from src.methods.tier1 import FrequencyRatio, _label_encode_cats
from src.methods.tier2 import OTDisplacementImproved, TestDirectedDataAugmentation as TDDAMethod
from src.models.base import ModelWrapper
from src.pipeline import run_single_experiment


class ConstantWeightMethod(TestTimeMethod):
    def __init__(self, weight: float = 2.0, *, drop_weight_rows: bool = False):
        self.weight = weight
        self.drop_weight_rows = drop_weight_rows

    @property
    def name(self) -> str:
        return "constant_weight"

    def fit_transform(self, X_train, X_test, y_train):
        X_tr = np.asarray(X_train, dtype=float)
        X_te = np.asarray(X_test, dtype=float)
        n_weights = len(X_tr) - 1 if self.drop_weight_rows else len(X_tr)
        return TransformResult(
            X_train=X_tr,
            X_test=X_te,
            feature_names=list(X_train.columns),
            sample_weights_train=np.full(n_weights, self.weight),
        )


class DuplicateRowsMethod(TestTimeMethod):
    def __init__(self, original_weight: float = 1.0, augment_weight: float = 3.0):
        self.original_weight = original_weight
        self.augment_weight = augment_weight

    @property
    def name(self) -> str:
        return "duplicate_rows"

    def fit_transform(self, X_train, X_test, y_train):
        X_tr = np.asarray(X_train, dtype=float)
        X_te = np.asarray(X_test, dtype=float)
        X_aug = np.vstack([X_tr, X_tr])
        y_aug = np.concatenate([np.asarray(y_train), np.asarray(y_train)])
        weights = np.concatenate([
            np.full(len(X_tr), self.original_weight),
            np.full(len(X_tr), self.augment_weight),
        ])
        return TransformResult(
            X_train=X_aug,
            X_test=X_te,
            feature_names=list(X_train.columns),
            sample_weights_train=weights,
            y_train=y_aug,
            row_provenance_train=RowProvenance.augmented(
                n_original_train=len(X_tr),
                augmented_source_split="train",
                augmented_source_position=np.arange(len(X_tr)),
                augmented_role="duplicate",
            ),
        )


class SeededNoOpMethod(TestTimeMethod):
    def __init__(self, seed: int = 42):
        self.seed = seed

    @property
    def name(self) -> str:
        return "seeded_noop"

    def fit_transform(self, X_train, X_test, y_train):
        return TransformResult(
            X_train=np.asarray(X_train, dtype=float),
            X_test=np.asarray(X_test, dtype=float),
            feature_names=list(X_train.columns),
        )


class OneCatSelector(ColumnSelector):
    @property
    def name(self) -> str:
        return "one_cat"

    def select(self, X_train, X_test, y_train):
        return ColumnSelection(cat_cols=["cat_a"], num_cols=[], scores={"cat_a": 1.0})


class CapturingRegressor(ModelWrapper):
    def __init__(self):
        self.y_seen = None

    @property
    def name(self) -> str:
        return "capturing_regressor"

    def fit(self, X_train, y_train, sample_weight=None):
        self.y_seen = np.asarray(y_train).copy()

    def predict(self, X_test):
        return np.zeros(len(X_test))


def numeric_frame(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({"x": np.arange(n, dtype=float), "z": np.linspace(0.0, 1.0, n)})


class TestMethodPipelineWeightComposition:
    def test_prior_weights_are_padded_across_augmentation(self):
        X_train = numeric_frame(4)
        X_test = numeric_frame(2)
        y = np.arange(4)
        pipe = MethodPipeline([
            ConstantWeightMethod(weight=2.0),
            DuplicateRowsMethod(original_weight=3.0, augment_weight=3.0),
        ])

        result = pipe.fit_transform(X_train, X_test, y)

        np.testing.assert_allclose(
            result.sample_weights_train,
            np.array([6.0] * 4 + [3.0] * 4),
        )

    def test_augmentation_then_reweighting_multiplies_one_to_one(self):
        X_train = numeric_frame(4)
        X_test = numeric_frame(2)
        y = np.arange(4)
        pipe = MethodPipeline([
            DuplicateRowsMethod(original_weight=1.0, augment_weight=2.0),
            ConstantWeightMethod(weight=3.0),
        ])

        result = pipe.fit_transform(X_train, X_test, y)

        np.testing.assert_allclose(
            result.sample_weights_train,
            np.array([3.0] * 4 + [6.0] * 4),
        )

    def test_weight_row_dropping_raises(self):
        X_train = numeric_frame(4)
        X_test = numeric_frame(2)
        y = np.arange(4)
        pipe = MethodPipeline([
            ConstantWeightMethod(weight=2.0),
            ConstantWeightMethod(weight=3.0, drop_weight_rows=True),
        ])

        with pytest.raises(ValueError, match="Row dropping is not supported"):
            pipe.fit_transform(X_train, X_test, y)


class TestSeedPropagation:
    def test_direct_method_seed_updates(self):
        method = SeededNoOpMethod()
        _propagate_seed(method, 99)
        assert method.seed == 99

    def test_method_pipeline_children_update(self):
        a = SeededNoOpMethod()
        b = SeededNoOpMethod()
        _propagate_seed(MethodPipeline([a, b]), 99)
        assert a.seed == 99
        assert b.seed == 99

    def test_selective_inner_method_updates(self):
        inner = SeededNoOpMethod()
        selective = SelectiveMethod(inner, AllColumnsSelector())
        _propagate_seed(selective, 99)
        assert inner.seed == 99


class TestAugmentedRegressionLogTarget:
    def test_tdda_augmented_targets_stay_in_logged_scale(self):
        X_train = pd.DataFrame({"x": np.linspace(0.0, 1.0, 8)})
        X_test = pd.DataFrame({"x": np.linspace(0.1, 0.9, 4)})
        y_train = np.exp(np.linspace(0.0, 3.0, 8))
        y_test = np.exp(np.linspace(0.2, 2.5, 4))
        dataset = TabularDataset(
            name="skewed_regression",
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cat_features=[],
            num_features=["x"],
            task_type="regression",
        )
        model = CapturingRegressor()
        method = TDDAMethod(k=1, lambda_=0.7, n_augments=1)

        result = run_single_experiment(
            dataset,
            method,
            model,
            seed=42,
            bootstrap_train=False,
            log_transform_target=True,
        )

        assert np.isfinite(result.rmse)
        assert model.y_seen is not None
        assert len(model.y_seen) == 2 * len(y_train)
        assert model.y_seen.max() <= np.log1p(y_train).max() + 1e-12


class TestSelectiveCategoricalFeatures:
    def test_inner_and_passthrough_cats_both_declared_on_partial_selection(self):
        """Selector picks cat_a (inner) → cat_b is passthrough. Both must be declared.

        SelectiveMethod combines the inner method's ``categorical_features``
        with passthrough categoricals. This allows raw-feature models to restore
        both groups as native categorical columns.
        """
        X_train = pd.DataFrame({
            "cat_a": ["a", "b", "a", "c"],
            "cat_b": ["x", "x", "y", "z"],
            "num": [1.0, 2.0, 3.0, 4.0],
        })
        X_test = pd.DataFrame({
            "cat_a": ["a", "c"],
            "cat_b": ["y", "z"],
            "num": [5.0, 6.0],
        })
        method = SelectiveMethod(FrequencyRatio(use_log=True), OneCatSelector())

        result = method.fit_transform(X_train, X_test, np.array([0, 1, 0, 1]))

        # Inner-method cat (cat_a) first, then passthrough cat (cat_b).
        # Stable dict.fromkeys order. Both must reach LightGBM as native cats.
        assert result.categorical_features == ["cat_a", "cat_b"]


class TestOTDisplacementFallback:
    def test_all_categorical_data_returns_no_displacement_features(self):
        X_train = pd.DataFrame({
            "cat_a": ["a", "b", "a", "c"],
            "cat_b": ["x", "x", "y", "z"],
        })
        X_test = pd.DataFrame({
            "cat_a": ["a", "c"],
            "cat_b": ["y", "z"],
        })
        method = OTDisplacementImproved(mode="selective", ks_threshold=0.0)

        result = method.fit_transform(X_train, X_test, np.array([0, 1, 0, 1]))
        enc_train, enc_test, _ = _label_encode_cats(X_train, X_test)

        assert result.metadata["selected_features"] == []
        assert not any(name.startswith("ot_disp_") for name in result.feature_names)
        np.testing.assert_allclose(result.X_train, enc_train.values.astype(float))
        np.testing.assert_allclose(result.X_test, enc_test.values.astype(float))

    def test_fallback_selects_only_numeric_columns(self):
        X_train = pd.DataFrame({
            "cat": ["a", "b", "a", "c", "d"],
            "num0": [0.0, 1.0, 2.0, 3.0, 4.0],
            "num1": [1.0, 1.5, 2.5, 3.5, 4.5],
            "num2": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        X_test = pd.DataFrame({
            "cat": ["a", "e", "f"],
            "num0": [0.5, 1.5, 2.5],
            "num1": [1.2, 2.2, 3.2],
            "num2": [4.5, 3.5, 2.5],
        })
        method = OTDisplacementImproved(mode="selective", ks_threshold=0.0)

        result = method.fit_transform(X_train, X_test, np.array([0, 1, 0, 1, 0]))

        assert set(result.metadata["selected_features"]) == {"num0", "num1", "num2"}
        assert "cat" not in result.metadata["selected_features"]


class TestMetadataReservedColumnGuard:
    """Ensure that metadata keys cannot overwrite core result columns.

    A method emitting a scalar metadata key named exactly like a reserved
    column (e.g. ``auc_roc``, ``seed``) would otherwise silently overwrite the
    real value when to_dict() merges metadata into the row dict. The guard
    namespaces such keys to ``meta_<key>``. This test retains that behavior
    even when current methods do not emit colliding keys.
    """

    def test_colliding_metadata_key_is_namespaced_not_overwriting(self):
        from src.pipeline import ExperimentResult

        exp = ExperimentResult(
            dataset="d", method="m", model="mod", seed=42,
            task_type="classification", accuracy=0.88, auc_roc=0.91,
            metadata={"auc_roc": -1.0, "seed": 999, "ess_ratio": 0.5},
        )
        d = exp.to_dict()

        # Reserved columns keep their real values, not the metadata values.
        assert d["auc_roc"] == 0.91
        assert d["seed"] == 42
        # Colliding metadata is preserved under a meta_ prefix (no info loss).
        assert d["meta_auc_roc"] == -1.0
        assert d["meta_seed"] == 999
        # Non-colliding metadata passes straight through unchanged.
        assert d["ess_ratio"] == 0.5

    def test_noncolliding_metadata_unaffected(self):
        """The common case (today's methods): ordinary metadata is untouched."""
        from src.pipeline import ExperimentResult

        exp = ExperimentResult(
            dataset="d", method="m", model="mod", seed=1,
            metadata={"ess": 12.0, "n_groupby_features": 7, "m16_distance_encoding": "frequency"},
        )
        d = exp.to_dict()
        assert d["ess"] == 12.0
        assert d["n_groupby_features"] == 7
        assert d["m16_distance_encoding"] == "frequency"
        # No spurious meta_ keys were created.
        assert "meta_ess" not in d


# --- Raw-feature categorical handling ----------------------------------------
#
# For models with prefers_raw_features=True (LightGBM bag8, TabPFN, TabICL),
# run_single_experiment passes the original categorical columns with object or
# category dtype rather than InductiveBaseline's label-encoded float values.
# Mock models exercise this pipeline path without requiring AutoGluon or torch.


class _CapturingModel(ModelWrapper):
    """Mock model that records the exact frame + fit kwargs it received."""

    def __init__(self, *, prefers_raw: bool):
        self.prefers_raw_features = prefers_raw
        self.X_seen: pd.DataFrame | None = None
        self.X_predict_seen: pd.DataFrame | None = None
        self.fit_kwargs: dict | None = None

    @property
    def name(self) -> str:
        return "capturing_raw" if self.prefers_raw_features else "capturing_nonraw"

    def fit(self, X_train, y_train, sample_weight=None, **fit_kwargs):
        self.X_seen = X_train.copy()
        self.fit_kwargs = dict(fit_kwargs)

    def predict(self, X_test):
        self.X_predict_seen = X_test.copy()
        return np.zeros(len(X_test))

    def predict_proba(self, X_test):
        self.X_predict_seen = X_test.copy()
        return np.full((len(X_test), 2), 0.5)


class _ProvenanceCapturingModel(_CapturingModel):
    @property
    def execution_metadata(self) -> dict[str, object]:
        return {"model_n_estimators": 2}


class _AugmentingCatMethod(TestTimeMethod):
    """Label-encodes cats (like InductiveBaseline) AND duplicates train rows.

    Used to verify the re-injection guard: because the train row count
    changes, the pipeline cannot positionally align the original cats, so it
    uses the method's encoded categorical features without raising an error.
    """

    @property
    def name(self) -> str:
        return "augmenting_cat"

    def fit_transform(self, X_train, X_test, y_train):
        from sklearn.preprocessing import LabelEncoder

        tr = X_train.copy()
        te = X_test.copy()
        cat_cols = tr.select_dtypes(include=["object", "category"]).columns.tolist()
        for c in cat_cols:
            le = LabelEncoder().fit(tr[c].astype(str))
            tr[c] = le.transform(tr[c].astype(str))
            tv = te[c].astype(str)
            known = tv.isin(le.classes_)
            te[c] = -1
            te.loc[known, c] = le.transform(tv[known])
        X_tr = tr.values.astype(float)
        X_tr_aug = np.vstack([X_tr, X_tr])  # duplicate -> row count changes
        return TransformResult(
            X_train=X_tr_aug,
            X_test=te.values.astype(float),
            feature_names=list(X_train.columns),
            y_train=np.concatenate([np.asarray(y_train), np.asarray(y_train)]),
            categorical_features=cat_cols if cat_cols else None,
            row_provenance_train=RowProvenance.augmented(
                n_original_train=len(X_tr),
                augmented_source_split="train",
                augmented_source_position=np.arange(len(X_tr)),
                augmented_role="duplicate",
            ),
        )


def _mixed_dataset(n_tr=40, n_te=20, seed=0):
    rng = np.random.RandomState(seed)

    def mk(n):
        return pd.DataFrame({
            "num1": rng.randn(n),
            "cat_country": rng.choice(["DE", "FR", "ES"], n),
            "num2": rng.randn(n) * 5,
            "cat_color": rng.choice(["red", "blue"], n),
        })

    Xtr, Xte = mk(n_tr), mk(n_te)
    return TabularDataset(
        name="mixed_toy",
        X_train=Xtr, y_train=rng.randint(0, 2, n_tr),
        X_test=Xte, y_test=rng.randint(0, 2, n_te),
        cat_features=["cat_country", "cat_color"],
        num_features=["num1", "num2"],
        task_type="classification",
    )


class TestRawFeatureCatParity:
    @pytest.mark.parametrize(
        "dataset_name",
        [
            "bank-marketing",
            "coil2000_insurance_policies",
            "HR_Analytics_Job_Change_of_Data_Scientists",
            "in_vehicle_coupon_recommendation",
            "Marketing_Campaign",
            "seismic-bumps",
        ],
    )
    def test_inductive_and_no_scaling_match_after_type_repair(
        self,
        dataset_name: str,
    ):
        """The canary pair must differ in neither values nor declarations."""
        try:
            baseline_dataset = load_dataset(
                dataset_name,
                fold_idx=0,
                max_samples=500,
                seed=0,
            )
            no_scaling_dataset = load_dataset(
                dataset_name,
                fold_idx=0,
                max_samples=500,
                seed=0,
            )
        except FileNotFoundError:
            pytest.skip(f"{dataset_name} cache is not present")

        baseline_model = _CapturingModel(prefers_raw=True)
        no_scaling_model = _CapturingModel(prefers_raw=True)
        run_single_experiment(
            baseline_dataset,
            InductiveBaseline(),
            baseline_model,
            seed=0,
            bootstrap_train=False,
        )
        run_single_experiment(
            no_scaling_dataset,
            NoScaling(),
            no_scaling_model,
            seed=0,
            bootstrap_train=False,
        )

        pd.testing.assert_frame_equal(
            baseline_model.X_seen,
            no_scaling_model.X_seen,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            baseline_model.X_predict_seen,
            no_scaling_model.X_predict_seen,
            check_exact=True,
        )
        assert baseline_model.fit_kwargs.get("categorical_feature", []) == (
            no_scaling_model.fit_kwargs.get("categorical_feature", [])
        )

    def test_model_execution_metadata_is_written_to_result(self):
        ds = _mixed_dataset()
        model = _ProvenanceCapturingModel(prefers_raw=True)
        exp = run_single_experiment(
            ds, InductiveBaseline(), model, seed=0, bootstrap_train=False
        )
        assert exp.to_dict()["model_n_estimators"] == 2

    def test_raw_model_receives_raw_object_categoricals(self):
        ds = _mixed_dataset()
        model = _CapturingModel(prefers_raw=True)
        run_single_experiment(ds, InductiveBaseline(), model, seed=0, bootstrap_train=False)
        X = model.X_seen
        # Cats arrive as raw objects (strings), not label-encoded floats.
        assert X["cat_country"].dtype == object
        assert X["cat_color"].dtype == object
        assert set(X["cat_country"]) <= {"DE", "FR", "ES"}
        # Numerics stay float.
        assert np.issubdtype(X["num1"].dtype, np.floating)
        assert np.issubdtype(X["num2"].dtype, np.floating)
        # The cat declaration is forwarded.
        assert set(model.fit_kwargs.get("categorical_feature", [])) == {
            "cat_country", "cat_color"
        }

    def test_reinjected_values_match_original(self):
        """Re-injection must be positionally faithful to dataset.X_train."""
        ds = _mixed_dataset()
        model = _CapturingModel(prefers_raw=True)
        run_single_experiment(ds, InductiveBaseline(), model, seed=0, bootstrap_train=False)
        np.testing.assert_array_equal(
            model.X_seen["cat_country"].to_numpy(),
            ds.X_train["cat_country"].to_numpy(),
        )
        np.testing.assert_array_equal(
            model.X_seen["cat_color"].to_numpy(),
            ds.X_train["cat_color"].to_numpy(),
        )

    def test_nonraw_model_keeps_label_encoded_float_path(self):
        ds = _mixed_dataset()
        model = _CapturingModel(prefers_raw=False)
        run_single_experiment(ds, InductiveBaseline(), model, seed=0, bootstrap_train=False)
        X = model.X_seen
        # All columns float (label-encoded + scaled). No cat kwarg forwarded.
        assert all(np.issubdtype(X[c].dtype, np.floating) for c in X.columns)
        assert "categorical_feature" not in (model.fit_kwargs or {})

    def test_all_numeric_dataset_is_noop_for_raw_model(self):
        rng = np.random.RandomState(1)
        mk = lambda n: pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
        ds = TabularDataset(
            name="num_only", X_train=mk(30), y_train=rng.randint(0, 2, 30),
            X_test=mk(10), y_test=rng.randint(0, 2, 10),
            cat_features=[], num_features=["a", "b"], task_type="classification",
        )
        model = _CapturingModel(prefers_raw=True)
        run_single_experiment(ds, InductiveBaseline(), model, seed=0, bootstrap_train=False)
        X = model.X_seen
        assert all(np.issubdtype(X[c].dtype, np.floating) for c in X.columns)
        # No cats declared -> no kwarg.
        assert "categorical_feature" not in (model.fit_kwargs or {})

    def test_augmenting_method_restores_cats_from_declared_provenance(self):
        """Augmented rows inherit raw cats from their declared source rows."""
        ds = _mixed_dataset()
        model = _CapturingModel(prefers_raw=True)
        run_single_experiment(ds, _AugmentingCatMethod(), model, seed=0, bootstrap_train=False)
        X = model.X_seen
        # Train rows doubled by augmentation.
        assert len(X) == 2 * ds.n_train
        assert X["cat_country"].dtype == object
        np.testing.assert_array_equal(
            X["cat_country"].to_numpy(),
            np.tile(ds.X_train["cat_country"].to_numpy(), 2),
        )
        assert set(model.fit_kwargs.get("categorical_feature", [])) == {
            "cat_country", "cat_color"
        }

    def test_bootstrap_toggle_changes_training_rows(self):
        """Check the worker's fold-driven bootstrap switch. bootstrap_train
        resamples (rows differ from original order). No-bootstrap is identity.

        A fresh dataset is built per run because LeakageGuard nulls y_test on
        first use.
        """
        original_cats = _mixed_dataset().X_train["cat_country"].to_numpy()

        ds_no = _mixed_dataset()
        m_no = _CapturingModel(prefers_raw=True)
        run_single_experiment(ds_no, InductiveBaseline(), m_no, seed=0, bootstrap_train=False)
        # Without bootstrap, the raw cats match the dataset 1:1 and in order.
        np.testing.assert_array_equal(m_no.X_seen["cat_country"].to_numpy(), original_cats)

        ds_bs = _mixed_dataset()
        m_bs = _CapturingModel(prefers_raw=True)
        run_single_experiment(ds_bs, InductiveBaseline(), m_bs, seed=0, bootstrap_train=True)
        # With bootstrap, the row multiset is a resample (same length, but the
        # ordered sequence almost surely differs on n=40).
        assert len(m_bs.X_seen) == ds_bs.n_train
        assert not np.array_equal(m_bs.X_seen["cat_country"].to_numpy(), original_cats)


# Bag8 wrapper end-to-end test when AutoGluon is available
#
# The mock-based tests above cover categorical re-injection. This test wires the
# LightGBMTabArenaBag8 wrapper through run_single_experiment on categorical data
# and checks that it receives raw categoricals plus the categorical_feature
# argument, fits successfully and returns a finite metric. It is skipped when
# AutoGluon is unavailable.


class TestBag8RealWrapperEndToEnd:
    def test_pipeline_hands_real_bag8_raw_cats_and_fits(self):
        pytest.importorskip("autogluon.tabular")
        from src.models.base import get_model

        ds = _mixed_dataset(n_tr=200, n_te=80)
        model = get_model("lightgbm_tabarena_bag8", task_type="classification", seed=0)
        assert getattr(model, "prefers_raw_features", False) is True

        # Capture the frame and keyword arguments received by the wrapper, then
        # dispatch to the underlying AutoGluon fit.
        captured: dict = {}
        orig_fit = model.fit

        def spy_fit(X_train, y_train, sample_weight=None, **fit_kwargs):
            captured["X"] = X_train.copy()
            captured["kwargs"] = dict(fit_kwargs)
            return orig_fit(X_train, y_train, sample_weight=sample_weight, **fit_kwargs)

        model.fit = spy_fit  # shadow the bound method on this instance only

        exp = run_single_experiment(
            ds, InductiveBaseline(), model, seed=0, bootstrap_train=False
        )

        X = captured["X"]
        # Cats reached the real wrapper as RAW objects (not label-encoded floats)
        # and positionally match the original frame.
        assert X["cat_country"].dtype == object
        assert X["cat_color"].dtype == object
        np.testing.assert_array_equal(
            X["cat_country"].to_numpy(), ds.X_train["cat_country"].to_numpy()
        )
        # Numerics stay float.
        assert np.issubdtype(X["num1"].dtype, np.floating)
        # The cat declaration is forwarded to the wrapper.
        assert set(captured["kwargs"].get("categorical_feature", [])) == {
            "cat_country", "cat_color"
        }
        # The real AutoGluon bag-8 fit accepted the raw frame and scored finite.
        assert np.isfinite(exp.auc_roc)
        assert 0.0 <= exp.auc_roc <= 1.0

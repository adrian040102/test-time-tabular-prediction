"""LightGBM bag-8 wrapper for AutoGluon's published TabArena GBM recipe.

This wrapper is registered as ``"lightgbm_tabarena_bag8"``. The separate
``"lightgbm_tabarena"`` wrapper implements a single-model configuration.

Recipe:

* Engine: ``autogluon.tabular.TabularPredictor.fit`` with
  ``hyperparameters={"GBM": {}}`` (AutoGluon's GBM-default HPs:
  ``num_boost_round=10000``, ``learning_rate=0.05``,
  library defaults for the rest).
* Bagging: ``num_bag_folds=8``, ``num_bag_sets=1``: 8 LightGBM
  boosters, each fit on 7/8 of the train data with the held-out 1/8
  as the early-stopping val set. Test predictions are the mean across
  all 8 boosters (probabilities for classification, raw values for
  regression).
* ``fit_weighted_ensemble=False``: only the GBM model itself. No
  AutoGluon ensemble layer (matches published GBM (default), which
  is a single model entry not the WeightedEnsemble row).
* ``prefers_raw_features = True``: pipeline skips the universal
  ``SimpleImputer`` + ``StandardScaler``. AutoGluon handles NaN
  natively and tree splits are scale-invariant.
* Categorical handling: declared columns are cast to pandas
  ``category`` dtype before being handed to AutoGluon, which then
  uses LightGBM's native categorical-splitting.

Implementation notes:

* AutoGluon writes model artifacts to disk. The wrapper allocates a per-instance
  :class:`tempfile.TemporaryDirectory` and lets GC clean it up. ``fit()``
  may be called multiple times. The temp dir is recreated each call.
* ``num_cpus=1`` by default, consistent with the single-worker-per-task
  execution setup. Override via ``**kwargs``.
* Sample weights are supported via AutoGluon's ``sample_weight``
  constructor argument based on column names. The wrapper adds a dedicated
  ``_sample_weight_`` column to the training frame when needed.
* Label encoding preserves the existing wrapper's ``predict`` /
  ``predict_proba`` output contract: binary → 1-D ``P(class=1)``,
  multiclass → ``(n, k)`` matrix sorted by encoded class index,
  regression → 1-D float array.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ModelWrapper


class LightGBMTabArenaBag8Wrapper(ModelWrapper):
    """LightGBM with the published TabArena GBM (default) recipe (8-fold bag-CV).

    Parameters
    ----------
    task_type
        ``"classification"`` or ``"regression"``.
    seed
        Forwarded through the GBM hyperparameter dictionary to each LightGBM
        booster. The bagging fold split uses AutoGluon's default seed.
    categorical_features
        Optional list of column names to declare as categorical. These
        columns are cast to pandas ``category`` dtype before AutoGluon
        receives the data. AutoGluon then uses LightGBM's native categorical
        splitting. Can also be supplied per-call via
        ``fit(..., categorical_feature=...)``.
    num_bag_folds
        Inner-CV bag size. ``8`` matches the published recipe and is
        not normally changed.
    num_bag_sets
        Number of independent bagging passes. ``1`` matches the
        published recipe.
    num_cpus
        CPU cap for AutoGluon. ``1`` matches the published recipe and the
        cluster's single-worker-per-task allocation. Override for local runs
        with a larger CPU allocation.
    time_limit
        Per-bag wallclock cap in seconds. ``None`` (the default)
        matches the published recipe, which uses ``num_boost_round=10000``
        with adaptive early stopping and no time cap.
    **kwargs
        Extra LightGBM ``hyperparameters["GBM"]`` overrides
        (e.g. ``num_leaves=63``). Empty dict by default keeps
        AutoGluon's GBM-default HPs intact.
    """

    prefers_raw_features: bool = True

    _TARGET_COL = "_target_"
    _WEIGHT_COL = "_sample_weight_"

    def __init__(
        self,
        task_type: str = "classification",
        seed: int = 42,
        categorical_features: list[str] | None = None,
        num_bag_folds: int = 8,
        num_bag_sets: int = 1,
        num_cpus: int = 1,
        time_limit: float | None = None,
        **kwargs: Any,
    ):
        self._task_type = task_type
        self._seed = int(seed)
        self._categorical_features = (
            list(categorical_features) if categorical_features else None
        )
        self._num_bag_folds = int(num_bag_folds)
        self._num_bag_sets = int(num_bag_sets)
        self._num_cpus = int(num_cpus)
        self._time_limit = time_limit
        self._gbm_hyperparameters: dict[str, Any] = dict(kwargs)

        self.predictor_ = None
        self.classes_ = None
        self._label_encoder = None
        self._n_classes = None
        self._feature_names = None
        self._tmpdir = None  # tempfile.TemporaryDirectory held until GC

    @property
    def name(self) -> str:
        return "lightgbm_tabarena_bag8"

    # --- Internals ---------------------------------------------------

    def _resolve_problem_type(self) -> str:
        """Return AutoGluon ``problem_type`` for this wrapper instance."""
        if self._task_type == "regression":
            return "regression"
        if self._n_classes is not None and self._n_classes > 2:
            return "multiclass"
        return "binary"

    def _resolve_eval_metric(self, problem_type: str) -> str:
        if problem_type == "binary":
            return "roc_auc"
        if problem_type == "multiclass":
            return "log_loss"
        return "rmse"

    def _resolve_cat_feature_list(
        self,
        X: pd.DataFrame,
        explicit: list[str] | None,
    ) -> list[str]:
        """Pick the set of columns to declare as categorical.

        Per-call ``explicit`` wins over instance-level
        ``categorical_features``. Columns not present in ``X.columns``
        are silently dropped: methods may have removed cat columns
        mid-pipeline.
        """
        cats = explicit if explicit is not None else self._categorical_features
        if not cats:
            return []
        return [c for c in cats if c in X.columns]

    def _cast_cats(
        self,
        X: pd.DataFrame,
        cat_cols: list[str],
    ) -> pd.DataFrame:
        """Return a copy of ``X`` with ``cat_cols`` cast to pandas
        ``category`` dtype so AutoGluon detects them as categorical.

        Columns that are already ``category`` are left untouched.
        """
        if not cat_cols:
            return X
        out = X.copy()
        for c in cat_cols:
            if not isinstance(out[c].dtype, pd.CategoricalDtype):
                out[c] = out[c].astype("category")
        return out

    def _build_train_frame(
        self,
        X: pd.DataFrame,
        y_enc: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> pd.DataFrame:
        df = X.copy()
        df[self._TARGET_COL] = y_enc
        if sample_weight is not None:
            df[self._WEIGHT_COL] = np.asarray(sample_weight, dtype=float)
        return df

    def _new_tmpdir(self):
        # Replace the temporary directory when the model is fitted again.
        # TemporaryDirectory handles cleanup through its finalizer.
        configured = os.environ.get("THESIS_TMPDIR") or tempfile.gettempdir()
        tmp_root = Path(configured).expanduser().resolve()
        allowed_value = os.environ.get("THESIS_ALLOWED_ROOT")
        if allowed_value:
            allowed_root = Path(allowed_value).expanduser().resolve()
            if not tmp_root.is_relative_to(allowed_root):
                raise RuntimeError(
                    f"Bag8 temporary root escaped authorized root: "
                    f"{tmp_root} is not below {allowed_root}"
                )
        tmp_root.mkdir(parents=True, exist_ok=True)
        if not os.access(tmp_root, os.W_OK):
            raise RuntimeError(f"Bag8 temporary root is not writable: {tmp_root}")
        self._tmpdir = tempfile.TemporaryDirectory(
            prefix="lgb_tabarena_bag8_", dir=tmp_root,
        )
        return self._tmpdir.name

    # --- Fit / predict -----------------------------------------------

    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        categorical_feature: list[str] | None = None,
        **fit_kwargs,
    ):
        from autogluon.tabular import TabularPredictor
        from sklearn.preprocessing import LabelEncoder

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(
                X_train,
                columns=[f"f{i}" for i in range(X_train.shape[1])],
            )
        self._feature_names = list(X_train.columns)

        y_train = np.asarray(y_train)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)

        if self._task_type == "classification":
            self._label_encoder = LabelEncoder()
            y_enc = self._label_encoder.fit_transform(y_train)
            self.classes_ = self._label_encoder.classes_
            self._n_classes = len(self.classes_)
        else:
            y_enc = y_train.astype(float)
            self._n_classes = None

        cat_cols = self._resolve_cat_feature_list(X_train, categorical_feature)
        # Persist the per-fit declaration so predict() casts the same columns.
        self._categorical_features = list(cat_cols)
        X_typed = self._cast_cats(X_train, cat_cols)
        train_df = self._build_train_frame(X_typed, y_enc, sample_weight)

        problem_type = self._resolve_problem_type()
        eval_metric = self._resolve_eval_metric(problem_type)

        predictor_kwargs: dict[str, Any] = dict(
            label=self._TARGET_COL,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=self._new_tmpdir(),
            verbosity=0,
        )
        if sample_weight is not None:
            predictor_kwargs["sample_weight"] = self._WEIGHT_COL
            predictor_kwargs["weight_evaluation"] = False

        self.predictor_ = TabularPredictor(**predictor_kwargs)

        # AutoGluon 1.4.0 does not accept ``random_seed`` in ``fit``. Pass the
        # wrapper seed through the GBM hyperparameter dictionary so each
        # LightGBM booster receives it. The bagging fold split retains
        # AutoGluon's default seed.
        gbm_hp: dict[str, Any] = {"seed": self._seed}
        gbm_hp.update(self._gbm_hyperparameters)

        ag_fit_kwargs: dict[str, Any] = dict(
            train_data=train_df,
            hyperparameters={"GBM": gbm_hp},
            num_bag_folds=self._num_bag_folds,
            num_bag_sets=self._num_bag_sets,
            fit_weighted_ensemble=False,
            time_limit=self._time_limit,
            num_cpus=self._num_cpus,
            verbosity=0,
        )
        # Caller-supplied AutoGluon overrides (e.g. ``presets``) take
        # precedence over the wrapper defaults.
        ag_fit_kwargs.update(fit_kwargs)

        self.predictor_.fit(**ag_fit_kwargs)

    def _predict_df(self, X_test) -> pd.DataFrame:
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=self._feature_names)
        elif self._feature_names is not None:
            cols = [c for c in self._feature_names if c in X_test.columns]
            X_test = X_test[cols]

        # Re-cast categorical columns to the dtypes used during fit. AutoGluon stores
        # categorical metadata, but the inbound test frame may have
        # come through a method that re-encoded cats as float. Re-cast
        # for symmetry with the training-side dtype handling.
        if self._categorical_features:
            cat_cols = [c for c in self._categorical_features if c in X_test.columns]
            X_test = self._cast_cats(X_test, cat_cols)
        return X_test

    def predict(self, X_test):
        X_test = self._predict_df(X_test)
        if self._task_type == "regression":
            return self.predictor_.predict(X_test).to_numpy()

        # Use the predictor's own predict() (returns label codes that
        # correspond to the LabelEncoder input), then map back to the
        # original labels.
        raw = self.predictor_.predict(X_test).to_numpy()
        return self._label_encoder.inverse_transform(raw.astype(int))

    def predict_proba(self, X_test):
        if self._task_type == "regression":
            return None
        X_test = self._predict_df(X_test)
        proba_df = self.predictor_.predict_proba(X_test, as_multiclass=True)
        # AutoGluon returns columns in self.predictor_.class_labels order.
        # The model was trained on LabelEncoder-encoded integers (0..k-1), so
        # this order is the encoder's class order.
        cls = list(self.predictor_.class_labels)
        # Re-order to encoder order (sorted ints) so column i = P(class i).
        target_order = sorted(cls)
        proba = proba_df[target_order].to_numpy()
        return self._format_proba(proba)

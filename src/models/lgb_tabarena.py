# Portions of this file adapt the LightGBM defaults and adaptive early-stopping
# formula from AutoGluon v1.4.0. They are reimplemented for this repository's
# model-wrapper contract. AutoGluon is licensed under Apache-2.0. See
# THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.

"""LightGBM wrapper using an AutoGluon-inspired GBM-default configuration.

The wrapper is registered separately from :class:`LightGBMWrapper`.

Recipe sources:
``autogluon/tabular/models/lgb/lgb_model.py:LGBModel._fit`` and
``.../lgb/hyperparameters/parameters.py``:

* Training engine: ``lgb.train()`` (not the sklearn ``LGBMClassifier`` shim).
* ``num_boost_round = 10_000``.
* ``learning_rate = 0.05`` + LightGBM library defaults for everything else
  (``num_leaves=31``, ``min_data_in_leaf=20``, ``feature_fraction=1.0``,
  ``bagging_fraction=1.0``, ``max_depth=-1``).
* Adaptive early-stopping patience: ``300`` if ``n_train <= 10_000``.
  Otherwise, use ``max(20, round(300 * 10_000 / n_train))``. This follows
  ``autogluon.core.models._utils.get_early_stopping_rounds``.
* Stopping metric per problem type: binary → ``auc``. Multiclass →
  ``multi_logloss``. Regression → ``rmse``.
* Optional ``categorical_feature=`` passing: the pipeline forwards this
  via ``model.fit(..., **fit_kwargs)`` when the upstream
  :class:`TransformResult` carries a populated ``categorical_features``
  list.
* ``prefers_raw_features = True`` is read by ``run_single_experiment`` to
  skip the universal ``SimpleImputer`` + ``StandardScaler`` (LightGBM
  handles NaN natively and tree splits are scale-invariant).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import ModelWrapper


def _adaptive_early_stopping_rounds(
    n_train: int,
    *,
    min_offset: int = 20,
    max_offset: int = 300,
    min_rows: int = 10_000,
) -> int:
    """Reimplementation of AutoGluon's adaptive patience formula.

    Source:
        ``autogluon/core/models/_utils.py:get_early_stopping_rounds``.
    """
    if n_train <= min_rows:
        return max_offset
    return max(min_offset, round(max_offset * min_rows / n_train))


class LightGBMTabArenaWrapper(ModelWrapper):
    """LightGBM wrapper with an AutoGluon-inspired GBM-default configuration.

    Parameters
    ----------
    task_type
        ``"classification"`` or ``"regression"``.
    seed
        Forwarded to ``random_state`` AND used as the val-split RNG.
    val_size
        Fraction of training rows held out for early-stopping. AutoGluon
        normally receives an externally built validation set. This wrapper creates
        the validation set internally so the pipeline contract (``model.fit(X, y)``)
        does not change.
    categorical_features
        Optional list of column names to declare as categorical. Tree
        splits then use LightGBM's native categorical-partitioning
        algorithm rather than treating ID-encoded floats as ordinals.
        Can also be supplied per-call via ``fit(..., categorical_feature=...)``.
    **kwargs
        Extra LightGBM ``params`` that override defaults (e.g.
        ``num_leaves=63``).
    """

    prefers_raw_features: bool = True

    def __init__(
        self,
        task_type: str = "classification",
        seed: int = 42,
        val_size: float = 0.2,
        categorical_features: list[str] | None = None,
        **kwargs,
    ):
        self._task_type = task_type
        self._seed = int(seed)
        self._val_size = float(val_size)
        self._categorical_features = list(categorical_features) if categorical_features else None
        self._extra_params = dict(kwargs)

        self.booster_ = None
        self.best_iteration_ = None
        self.classes_ = None
        self._label_encoder = None
        self._n_classes = None
        self._feature_names = None

    @property
    def name(self) -> str:
        return "lightgbm_tabarena"

    # --- Internals ---------------------------------------------------

    def _build_params(self, n_classes: int | None) -> dict:
        if self._task_type == "regression":
            objective = "regression"
            metric = "rmse"
        elif n_classes is not None and n_classes > 2:
            objective = "multiclass"
            metric = "multi_logloss"
        else:
            objective = "binary"
            metric = "auc"

        params = {
            "objective": objective,
            "metric": metric,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "max_depth": -1,
            "verbose": -1,
            "seed": self._seed,
            "deterministic": True,
            "force_col_wise": True,
        }
        if objective == "multiclass":
            params["num_class"] = n_classes

        params.update(self._extra_params)
        return params

    def _split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ):
        from sklearn.model_selection import train_test_split

        n = len(y)
        idx = np.arange(n)
        stratify = y if self._task_type == "classification" else None
        try:
            idx_tr, idx_val = train_test_split(
                idx,
                test_size=self._val_size,
                random_state=self._seed,
                stratify=stratify,
            )
        except ValueError:
            # Stratification can fail on extreme imbalance (minority class
            # too small to split). Fall back to a non-stratified split.
            idx_tr, idx_val = train_test_split(
                idx,
                test_size=self._val_size,
                random_state=self._seed,
                stratify=None,
            )

        X_tr = X.iloc[idx_tr] if isinstance(X, pd.DataFrame) else X[idx_tr]
        X_val = X.iloc[idx_val] if isinstance(X, pd.DataFrame) else X[idx_val]
        y_tr = y[idx_tr]
        y_val = y[idx_val]
        sw_tr = sample_weight[idx_tr] if sample_weight is not None else None
        sw_val = sample_weight[idx_val] if sample_weight is not None else None
        return X_tr, X_val, y_tr, y_val, sw_tr, sw_val

    def _resolve_cat_feature_arg(
        self,
        X: pd.DataFrame,
        explicit: list[str] | None,
    ) -> list[str] | str:
        """Pick the categorical_feature argument for lgb.Dataset.

        ``explicit`` (per-call) wins over instance-level ``categorical_features``
        (constructor). Anything not present in ``X.columns`` is silently
        dropped: methods may have removed cat columns mid-pipeline.
        Returns ``"auto"`` if nothing is declared.
        """
        cats = explicit if explicit is not None else self._categorical_features
        if not cats:
            return "auto"
        if isinstance(X, pd.DataFrame):
            cats = [c for c in cats if c in X.columns]
        return cats if cats else "auto"

    # --- Fit / predict -----------------------------------------------

    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        categorical_feature: list[str] | None = None,
        **fit_kwargs,
    ):
        import lightgbm as lgb
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

        X_tr, X_val, y_tr, y_val, sw_tr, sw_val = self._split(
            X_train, y_enc, sample_weight
        )

        cat_arg = self._resolve_cat_feature_arg(X_train, categorical_feature)

        train_set = lgb.Dataset(
            X_tr,
            label=y_tr,
            weight=sw_tr,
            categorical_feature=cat_arg,
            free_raw_data=False,
        )
        val_set = lgb.Dataset(
            X_val,
            label=y_val,
            weight=sw_val,
            categorical_feature=cat_arg,
            reference=train_set,
            free_raw_data=False,
        )

        n_train = len(y_tr)
        patience = _adaptive_early_stopping_rounds(n_train)

        params = self._build_params(self._n_classes)
        callbacks = [
            lgb.early_stopping(stopping_rounds=patience, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        self.booster_ = lgb.train(
            params,
            train_set,
            num_boost_round=10_000,
            valid_sets=[val_set],
            valid_names=["val"],
            callbacks=callbacks,
        )
        self.best_iteration_ = self.booster_.best_iteration

    def _raw_predict(self, X_test) -> np.ndarray:
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=self._feature_names)
        elif self._feature_names is not None:
            # Re-order columns to match training. Missing columns will
            # raise downstream. That is the desired behavior.
            cols = [c for c in self._feature_names if c in X_test.columns]
            X_test = X_test[cols]
        return self.booster_.predict(X_test, num_iteration=self.best_iteration_)

    def predict(self, X_test):
        raw = self._raw_predict(X_test)
        if self._task_type == "regression":
            return raw
        if self._n_classes == 2:
            labels = (raw >= 0.5).astype(int)
        else:
            labels = np.argmax(raw, axis=1)
        return self._label_encoder.inverse_transform(labels)

    def predict_proba(self, X_test):
        if self._task_type == "regression":
            return None
        raw = self._raw_predict(X_test)
        return self._format_proba(raw)

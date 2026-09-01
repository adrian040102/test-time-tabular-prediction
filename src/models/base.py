"""
Model wrappers providing a unified interface for classifiers and regressors.

Each wrapper supports fit, predict and optional sample_weight.
Classification wrappers additionally support predict_proba.
The task_type parameter controls which underlying sklearn/boosting model is used.
The seed is forwarded to supported estimator randomness controls.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ModelWrapper(ABC):
    """Abstract base class for all model wrappers."""

    # Explicit capability contract used before model.fit. Wrappers that do
    # not consume weights must override this with ``False``.
    supports_sample_weight: bool = True
    # Wrappers may override this when method-level pre-scaling is incompatible
    # with the model.
    refuses_prescaled_input: bool = False

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, sample_weight: np.ndarray | None = None):
        ...

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray | None:
        """Return class probabilities (classification only).

        For binary classification: returns 1-D array of P(class=1).
        For multiclass classification: returns 2-D array (n_samples, n_classes).
        For regression: returns None.
        """
        return None

    def _format_proba(self, proba: np.ndarray) -> np.ndarray:
        """Convert raw predict_proba output to the correct format.

        Binary (2 columns) → 1-D vector of P(class=1).
        Multiclass (>2 columns) → full 2-D matrix.
        """
        if proba.ndim == 2:
            if proba.shape[1] == 2:
                return proba[:, 1]
            return proba  # multiclass: return full matrix
        return proba  # already 1-D

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def task_type(self) -> str:
        return getattr(self, "_task_type", "classification")

    @property
    def execution_metadata(self) -> dict[str, object]:
        """Scalar model settings that must travel with every result row."""
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task_type={self.task_type!r})"


class LogisticRegressionWrapper(ModelWrapper):
    def __init__(self, task_type: str = "classification", seed: int = 42, **kwargs):
        self._task_type = task_type
        if task_type == "regression":
            from sklearn.linear_model import Ridge
            defaults = dict(random_state=seed)
            defaults.update(kwargs)
            self.model = Ridge(**defaults)
        else:
            from sklearn.linear_model import LogisticRegression
            defaults = dict(max_iter=1000, random_state=seed)
            defaults.update(kwargs)
            self.model = LogisticRegression(**defaults)

    @property
    def name(self) -> str:
        return "logreg" if self._task_type == "classification" else "ridge"

    def fit(self, X_train, y_train, sample_weight=None):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        if self._task_type == "regression":
            return None
        return self._format_proba(self.model.predict_proba(X_test))


class DecisionTreeWrapper(ModelWrapper):
    def __init__(self, task_type: str = "classification", seed: int = 42, **kwargs):
        self._task_type = task_type
        defaults = dict(random_state=seed)
        defaults.update(kwargs)
        if task_type == "regression":
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor(**defaults)
        else:
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier(**defaults)

    @property
    def name(self) -> str:
        return "decision_tree"

    def fit(self, X_train, y_train, sample_weight=None):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        if self._task_type == "regression":
            return None
        return self._format_proba(self.model.predict_proba(X_test))


class RandomForestWrapper(ModelWrapper):
    def __init__(self, task_type: str = "classification", seed: int = 42, **kwargs):
        self._task_type = task_type
        defaults = dict(n_estimators=200, random_state=seed, n_jobs=-1)
        defaults.update(kwargs)
        if task_type == "regression":
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**defaults)
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**defaults)

    @property
    def name(self) -> str:
        return "random_forest"

    def fit(self, X_train, y_train, sample_weight=None):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        if self._task_type == "regression":
            return None
        return self._format_proba(self.model.predict_proba(X_test))


class XGBoostWrapper(ModelWrapper):
    def __init__(self, task_type: str = "classification", seed: int = 42, **kwargs):
        import xgboost as xgb
        self._task_type = task_type
        defaults = dict(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=seed,
        )
        defaults.update(kwargs)
        if task_type == "regression":
            defaults.setdefault("eval_metric", "rmse")
            self.model = xgb.XGBRegressor(**defaults)
        else:
            defaults.setdefault("eval_metric", "logloss")
            self.model = xgb.XGBClassifier(**defaults)

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(self, X_train, y_train, sample_weight=None):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        if self._task_type == "regression":
            return None
        return self._format_proba(self.model.predict_proba(X_test))


class LightGBMWrapper(ModelWrapper):
    def __init__(self, task_type: str = "classification", seed: int = 42, **kwargs):
        import lightgbm as lgb
        self._task_type = task_type
        defaults = dict(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            subsample_freq=1,  # Required to enable row subsampling.
            verbosity=-1, random_state=seed,
        )
        defaults.update(kwargs)
        if task_type == "regression":
            self.model = lgb.LGBMRegressor(**defaults)
        else:
            self.model = lgb.LGBMClassifier(**defaults)

    @property
    def name(self) -> str:
        return "lightgbm"

    def fit(self, X_train, y_train, sample_weight=None):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        if self._task_type == "regression":
            return None
        return self._format_proba(self.model.predict_proba(X_test))


class CatBoostWrapper(ModelWrapper):
    def __init__(self, task_type: str = "classification", seed: int = 42, **kwargs):
        self._task_type = task_type
        defaults = dict(
            iterations=300, depth=6, learning_rate=0.1,
            verbose=0, random_seed=seed,
        )
        defaults.update(kwargs)
        if task_type == "regression":
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(**defaults)
        else:
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier(**defaults)

    @property
    def name(self) -> str:
        return "catboost"

    def fit(self, X_train, y_train, sample_weight=None):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict(self, X_test):
        return self.model.predict(X_test).flatten()

    def predict_proba(self, X_test):
        if self._task_type == "regression":
            return None
        return self._format_proba(self.model.predict_proba(X_test))


# --- Registry ---

def _make_lgb_tabarena(*args, **kwargs):
    """Deferred factory: avoids a circular import at base.py load time
    (lgb_tabarena imports ModelWrapper from this module)."""
    from src.models.lgb_tabarena import LightGBMTabArenaWrapper
    return LightGBMTabArenaWrapper(*args, **kwargs)


def _make_lgb_tabarena_bag8(*args, **kwargs):
    """Create the bag-8 variant only when requested.

    AutoGluon is imported when this factory is called because it is not
    installed in every environment that imports this module.
    """
    from src.models.lgb_tabarena_bag8 import LightGBMTabArenaBag8Wrapper
    return LightGBMTabArenaBag8Wrapper(*args, **kwargs)


def _make_tabpfn_v2(*args, **kwargs):
    """Create TabPFN-v2 only when requested.

    Importing TabPFN also imports PyTorch and may initiate a model download, so
    the dependency is loaded when this factory is called.
    """
    from src.models.tabpfn import TabPFNWrapper
    kwargs.setdefault("n_estimators", MODEL_EXECUTION_CONTRACTS["tabpfn_v2"]["model_n_estimators"])
    return TabPFNWrapper(*args, **kwargs)


def _make_tabicl_v2(*args, **kwargs):
    """Create the canonical TabICL-v2 wrapper with its configured estimator count."""
    from src.models.tabicl import TabICLWrapper
    kwargs.setdefault("n_estimators", MODEL_EXECUTION_CONTRACTS["tabicl_v2"]["model_n_estimators"])
    return TabICLWrapper(*args, **kwargs)


def _make_tabicl_v2_ne2(*args, **kwargs):
    """Create the two-estimator TabICL variant with the standard wrapper and guards."""
    from src.models.tabicl import TabICLWrapper
    kwargs.setdefault("n_estimators", 2)
    return TabICLWrapper(*args, **kwargs)


# Execution-critical settings are recorded with each result and reported
# independently by the wrapper.
MODEL_EXECUTION_CONTRACTS: dict[str, dict[str, object]] = {
    "tabicl_v2": {"model_n_estimators": 2},
    "tabicl_v2_ne2": {"model_n_estimators": 2},
    "tabpfn_v2": {"model_n_estimators": 8},
}


def get_model_execution_metadata(name: str) -> dict[str, object]:
    """Return a copy of the frozen execution contract for *name*."""
    return dict(MODEL_EXECUTION_CONTRACTS.get(name, {}))


MODEL_REGISTRY: dict[str, type[ModelWrapper]] = {
    "logreg": LogisticRegressionWrapper,
    "decision_tree": DecisionTreeWrapper,
    "random_forest": RandomForestWrapper,
    "xgboost": XGBoostWrapper,
    "lightgbm": LightGBMWrapper,
    "catboost": CatBoostWrapper,
    "lightgbm_tabarena": _make_lgb_tabarena,  # factory. Resolved at first call
    "lightgbm_tabarena_bag8": _make_lgb_tabarena_bag8,  # factory, imports AutoGluon
    "tabpfn_v2": _make_tabpfn_v2,  # factory, imports PyTorch and TabPFN
    "tabicl_v2": _make_tabicl_v2,  # factory, imports PyTorch and TabICL
    "tabicl_v2_ne2": _make_tabicl_v2_ne2,  # factory. n_estimators=2
}


def get_model(name: str, task_type: str = "classification", seed: int = 42, **kwargs) -> ModelWrapper:
    """Instantiate a model by registry name.

    Args:
        name: Model key from MODEL_REGISTRY.
        task_type: ``"classification"`` or ``"regression"``.
        seed: Random seed forwarded to the estimator's random_state.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](task_type=task_type, seed=seed, **kwargs)


def get_all_models(task_type: str = "classification", seed: int = 42, **kwargs) -> dict[str, ModelWrapper]:
    """Instantiate all registered models for a given task type."""
    return {name: cls(task_type=task_type, seed=seed, **kwargs) for name, cls in MODEL_REGISTRY.items()}

"""TabPFN-v2 wrapper for the test-time tabular benchmark.

Reference: Hollmann et al., "Accurate predictions on small data with a
tabular foundation model", Nature 637(8045):319-326, 2025.
Repository: https://github.com/PriorLabs/TabPFN

Role
----
Reference model evaluated on datasets that satisfy the size contract below.

Size guards (hard skip, raises :class:`TabPFNSkipReason`)
---------------------------------------------------------
- ``train_rows > 10_000``
- ``n_features > 500``
- classification with ``n_classes > 10``
- CPU mode with ``train_rows > cpu_max_rows`` (default 1_000)

Method-generated representations
--------------------------------
The pipeline bypasses its generic imputer and scaler for TabPFN. Deliberate
method outputs such as PCA scores, t-SNE coordinates and standardized columns
are nevertheless valid experimental inputs and are passed through as produced.

Categorical handling
--------------------
TabPFN's API takes ``categorical_features_indices`` (0-based int
indices into the column list passed to ``.fit()``).  The pipeline
forwards ``categorical_feature=list[str]`` (column names) when an
upstream method declared them. The wrapper converts names to indices with
``X_train.columns.get_loc``.  Names absent from the current column set
are silently dropped (a method may have removed cat columns).

Declared categoricals that contain numeric values may arrive as pandas
``object`` columns because the pipeline preserves their raw values. TabPFN
2.2.1 converts a mixed DataFrame to one object ndarray before its own dtype
repair. On datasets that also contain an all-NaN training column, that can
leave an object-typed ``[nan]`` category inside sklearn's ``OrdinalEncoder``
and make prediction fail in ``np.isnan``. The wrapper therefore normalizes
only numeric-valued declared categoricals to ``float64`` at the model
boundary, consistently for fit and predict, while still passing their
categorical indices to TabPFN. Category semantics and values are unchanged.

Sample weights
--------------
TabPFN does not accept ``sample_weight``.  When the pipeline passes one
(e.g. M6 adversarial validation upstream), the wrapper ignores it with a warning.
This follows the upstream behavior for transformer-style classifiers.
Methods that require sample weights should be excluded from TabPFN manifests
when configuring the sweep.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ModelWrapper


# --- Size guards -------------------------------------------------------
MAX_TRAIN_ROWS = 10_000
MAX_FEATURES = 500
MAX_CLASSES = 10
CPU_MAX_TRAIN_ROWS = 1_000


class TabPFNSkipReason(RuntimeError):
    """Structured skip signal from :class:`TabPFNWrapper`.

    Raised by :meth:`TabPFNWrapper.fit` when a run cannot proceed
    because of:

    - dataset size exceeding the documented pre-training limits.
    - CPU mode with more rows than ``cpu_max_rows`` allows.

    The exception subclasses :class:`RuntimeError` so the existing SLURM
    worker (``scripts/slurm/worker.py``) catches it and writes the
    message to the ``error`` column of the per-task CSV without crashing
    the array task.  Analysis scripts can filter on
    ``df["error"].str.startswith("tabpfn_skip:")``.

    Attributes
    ----------
    reason
        Short machine-readable key (e.g. ``"train_rows_exceeded"``).
    details
        Dict of relevant numbers for downstream filtering / debugging.
    """

    def __init__(self, reason: str, details: dict | None = None) -> None:
        self.reason = reason
        self.details = details or {}
        msg = f"tabpfn_skip: {reason}"
        if self.details:
            msg = f"{msg} ({', '.join(f'{k}={v}' for k, v in self.details.items())})"
        super().__init__(msg)


def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` → ``"cuda"`` if available else ``"cpu"``.

    Any other string is returned unchanged so callers can force a
    specific device (e.g. ``"cuda:1"``).
    """
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class TabPFNWrapper(ModelWrapper):
    """TabPFN-v2 (PriorLabs) wrapper.

    Parameters
    ----------
    task_type
        ``"classification"`` or ``"regression"``.
    seed
        Forwarded to TabPFN's ``random_state``.
    device
        ``"auto"`` (default), ``"cuda"`` or ``"cpu"``. ``"auto"`` picks
        ``cuda`` if torch reports it is available, otherwise ``cpu``.
    categorical_features
        Optional list of column names to declare as categorical. TabPFN
        receives them as 0-based column indices.  Can also be supplied
        per-call via ``fit(..., categorical_feature=[...])``.
    n_estimators
        Number of TabPFN ensemble members.  Default 8 (library default).
    ignore_pretraining_limits
        Forwarded to TabPFN. Default False: the wrapper-level size
        guards are stricter than TabPFN's own limits anyway, so this
        only matters for callers who bypass the guards.
    cpu_max_rows
        CPU-mode size threshold. Default
        :data:`CPU_MAX_TRAIN_ROWS` (1000). The production sweep runs on GPU,
        where this threshold does not apply.
    **kwargs
        Extra kwargs forwarded to ``TabPFNClassifier`` /
        ``TabPFNRegressor`` (e.g. ``inference_precision``,
        ``memory_saving_mode``).
    """

    prefers_raw_features: bool = True
    refuses_prescaled_input: bool = False
    supports_sample_weight: bool = False

    def __init__(
        self,
        task_type: str = "classification",
        seed: int = 42,
        device: str = "auto",
        categorical_features: list[str] | None = None,
        n_estimators: int = 8,
        ignore_pretraining_limits: bool = False,
        cpu_max_rows: int = CPU_MAX_TRAIN_ROWS,
        **kwargs: Any,
    ) -> None:
        self._task_type = task_type
        self._seed = int(seed)
        self._device_requested = device
        self._categorical_features = (
            list(categorical_features) if categorical_features else None
        )
        self._n_estimators = int(n_estimators)
        self._ignore_pretraining_limits = bool(ignore_pretraining_limits)
        self._cpu_max_rows = int(cpu_max_rows)
        self._extra_kwargs = dict(kwargs)

        self.model_ = None
        self.classes_ = None
        self._n_classes: int | None = None
        self._feature_names: list[str] | None = None
        self._resolved_device: str | None = None
        self._numeric_categorical_features: list[str] = []

    @property
    def name(self) -> str:
        return "tabpfn_v2"

    @property
    def execution_metadata(self) -> dict[str, object]:
        return {"model_n_estimators": self._n_estimators}

    # --- Guards -----------------------------------------------------

    def _check_size_guards(
        self,
        n_train: int,
        n_features: int,
        n_classes: int | None,
        device: str,
    ) -> None:
        if n_train > MAX_TRAIN_ROWS:
            raise TabPFNSkipReason(
                "train_rows_exceeded",
                {"train_rows": n_train, "limit": MAX_TRAIN_ROWS},
            )
        if n_features > MAX_FEATURES:
            raise TabPFNSkipReason(
                "features_exceeded",
                {"features": n_features, "limit": MAX_FEATURES},
            )
        if (
            self._task_type == "classification"
            and n_classes is not None
            and n_classes > MAX_CLASSES
        ):
            raise TabPFNSkipReason(
                "classes_exceeded",
                {"n_classes": n_classes, "limit": MAX_CLASSES},
            )
        if device == "cpu" and n_train > self._cpu_max_rows:
            raise TabPFNSkipReason(
                "cpu_too_large",
                {"train_rows": n_train, "cpu_limit": self._cpu_max_rows},
            )

    # --- Cat index resolution --------------------------------------

    def _cat_indices(
        self,
        X: pd.DataFrame,
        explicit: list[str] | None,
    ) -> list[int] | None:
        """Translate cat-feature names → 0-based column indices.

        Per-call ``explicit`` wins over the instance-level
        ``categorical_features``.  Names absent from ``X.columns`` are
        silently dropped: methods may have removed cat columns mid
        pipeline (e.g. M1 ``JointFrequencyEncoding`` replaces cats).
        Returns ``None`` when the resulting list would be empty so
        TabPFN falls back to its own categorical inference.
        """
        cats = explicit if explicit is not None else self._categorical_features
        if not cats:
            return None
        cols = list(X.columns)
        idx = [cols.index(c) for c in cats if c in cols]
        return idx or None

    @staticmethod
    def _numeric_view(series: pd.Series) -> pd.Series:
        """Infer Python-number object columns without coercing numeric strings."""
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype(object)
        return series.infer_objects(copy=False)

    def _normalize_numeric_categoricals(
        self,
        X: pd.DataFrame,
        *,
        cat_indices: list[int] | None = None,
        fitting: bool,
    ) -> pd.DataFrame:
        """Give numeric-valued categoricals a stable numeric model dtype.

        The categorical declaration remains authoritative: TabPFN still receives
        the same categorical indices. This translation is model-boundary-only and
        never mutates the pipeline frame supplied by the caller.
        """
        if fitting:
            indices = cat_indices or []
            numeric_cats: list[str] = []
            for index in indices:
                column = X.columns[index]
                inferred = self._numeric_view(X[column])
                if pd.api.types.is_numeric_dtype(inferred.dtype):
                    numeric_cats.append(column)
            self._numeric_categorical_features = numeric_cats

        if not self._numeric_categorical_features:
            return X

        normalized = X.copy()
        for column in self._numeric_categorical_features:
            if column not in normalized.columns:
                raise ValueError(
                    "TabPFN numeric categorical normalization is missing the "
                    f"fit-time column {column!r}"
                )
            inferred = self._numeric_view(normalized[column])
            if not pd.api.types.is_numeric_dtype(inferred.dtype):
                raise ValueError(
                    "TabPFN numeric categorical changed to a non-numeric dtype "
                    f"between fit and predict: {column!r} ({inferred.dtype})"
                )
            normalized[column] = pd.to_numeric(
                inferred, errors="raise"
            ).astype(np.float64)
        return normalized

    # --- Fit / predict ---------------------------------------------

    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        categorical_feature: list[str] | None = None,
        _was_prescaled: bool = False,
        **fit_kwargs: Any,
    ):
        """Fit the underlying TabPFN model.

        Parameters
        ----------
        X_train, y_train
            Training features (DataFrame or ndarray) and labels.
        sample_weight
            Ignored with a warning.  TabPFN does not accept weights.
        categorical_feature
            Optional list of column names to declare as categorical
            (overrides the constructor-level ``categorical_features``).
            Matches the kwarg the pipeline forwards from
            ``TransformResult.categorical_features``.
        _was_prescaled
            Backward-compatible pipeline metadata. TabPFN receives the
            method output as produced. The wrapper applies no eligibility
            rejection based on this flag.
        **fit_kwargs
            Currently unused.
        """
        if sample_weight is not None:
            warnings.warn(
                "TabPFNWrapper does not support sample_weight. Ignoring "
                f"weights array of length {len(sample_weight)}.",
                RuntimeWarning,
                stacklevel=2,
            )

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(
                X_train,
                columns=[f"f{i}" for i in range(X_train.shape[1])],
            )
        self._feature_names = list(X_train.columns)

        y_train = np.asarray(y_train)

        n_train, n_features = X_train.shape
        if self._task_type == "classification":
            self.classes_ = np.unique(y_train)
            self._n_classes = len(self.classes_)
        else:
            self._n_classes = None

        device = _resolve_device(self._device_requested)
        self._resolved_device = device
        self._check_size_guards(n_train, n_features, self._n_classes, device)

        cat_indices = self._cat_indices(X_train, categorical_feature)
        X_train = self._normalize_numeric_categoricals(
            X_train,
            cat_indices=cat_indices,
            fitting=True,
        )

        model_kwargs: dict[str, Any] = dict(
            n_estimators=self._n_estimators,
            categorical_features_indices=cat_indices,
            device=device,
            ignore_pretraining_limits=self._ignore_pretraining_limits,
            random_state=self._seed,
        )
        model_kwargs.update(self._extra_kwargs)

        if self._task_type == "regression":
            from tabpfn import TabPFNRegressor
            self.model_ = TabPFNRegressor(**model_kwargs)
        else:
            from tabpfn import TabPFNClassifier
            self.model_ = TabPFNClassifier(**model_kwargs)

        self.model_.fit(X_train, y_train)

    def _predict_df(self, X_test) -> pd.DataFrame:
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=self._feature_names)
        elif self._feature_names is not None:
            # Re-order to match training.  Missing columns are a contract
            # violation. The pipeline guarantees column consistency
            # between train and test, so a mismatch here means something
            # upstream replaced the test frame with the wrong shape.
            missing = [c for c in self._feature_names if c not in X_test.columns]
            if missing:
                raise ValueError(
                    "TabPFNWrapper.predict received a frame missing columns "
                    f"that were present at fit time: {missing[:5]}"
                    + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
                )
            X_test = X_test[self._feature_names]
        return self._normalize_numeric_categoricals(X_test, fitting=False)

    def predict(self, X_test):
        X_test = self._predict_df(X_test)
        return np.asarray(self.model_.predict(X_test))

    def predict_proba(self, X_test):
        if self._task_type == "regression":
            return None
        X_test = self._predict_df(X_test)
        proba = np.asarray(self.model_.predict_proba(X_test))
        return self._format_proba(proba)

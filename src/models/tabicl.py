"""TabICL-v2 wrapper for the test-time tabular benchmark.

Reference: Qu et al., "TabICL: A Tabular Foundation Model for In-Context
Learning on Large Data", arXiv:2502.05564, ICML 2025.
Repository: https://github.com/soda-inria/tabicl (BSD-3-Clause).

Role
----
Evaluation model alongside :class:`LightGBMTabArenaBag8Wrapper`. TabICL has no
hard column limit and supports CPU or disk offload on larger inputs.

Size limit (raises :class:`TabICLSkipReason`)
----------------------------------------------
The wrapper has one size limit:

- ``train_rows > 500_000``

Inputs below that threshold use TabICL's own
adaptive-precision and offload heuristics. ``offload_mode='auto'``
falls back to CPU / disk for large inputs that exceed VRAM.

KV cache
--------
``kv_cache=True`` is the wrapper default. TabICL's library default is False.
The cache is built during ``fit()`` and reused across ``predict`` and
``predict_proba`` calls on different test batches. This avoids recomputing
the representation for each prediction.

Categorical handling
--------------------
TabICL applies its own ordinal-encoding step internally on object /
string and categorical columns. The wrapper does not pre-encode or
forward a ``categorical_feature=`` declaration from upstream methods
(unlike :class:`LightGBMTabArenaWrapper`).  The wrapper accepts whatever
the pipeline forwards (raw or pre-scaled: TabICL does not have
TabPFN-v2's "raw inputs only" contract).

Sample weights
--------------
TabICL's sklearn-style ``.fit(X, y)`` API does not accept
``sample_weight``.  When the pipeline passes one (e.g. M6 adversarial
validation upstream), the wrapper ignores it with a warning. Reweighting methods
that require sample weights should be excluded from TabICL manifests when
configuring the sweep.

GPU / CPU
---------
``device="auto"`` (the wrapper default) selects torch.cuda if available and CPU otherwise.
TabICL's own ``device=None`` makes the same selection. The explicit "auto"
value is consistent with :class:`TabPFNWrapper`. TabICL can run on CPU through
its ``offload_mode='auto'`` behavior.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ModelWrapper


# --- Size threshold ----------------------------------------------------
MAX_TRAIN_ROWS = 500_000


class TabICLSkipReason(RuntimeError):
    """Structured skip signal from :class:`TabICLWrapper`.

    Raised by :meth:`TabICLWrapper.fit` when the input would push TabICL
    because ``train_rows > 500_000``.

    Uses the same interface as :class:`src.models.tabpfn.TabPFNSkipReason` so the SLURM
    worker (``scripts/slurm/worker.py``) catches it and writes the
    message to the ``error`` column of the per-task CSV without crashing
    the array task.  Analysis scripts can filter on
    ``df["error"].str.startswith("tabicl_skip:")``.

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
        msg = f"tabicl_skip: {reason}"
        if self.details:
            msg = f"{msg} ({', '.join(f'{k}={v}' for k, v in self.details.items())})"
        super().__init__(msg)


def _resolve_device(device: str | None) -> str | None:
    """Resolve ``"auto"`` → ``"cuda"`` if available else ``"cpu"``.

    ``None`` is passed through (TabICL's API also auto-selects on None).
    Any other string is returned unchanged so callers can force a
    specific device (e.g. ``"cuda:1"``).
    """
    if device is None or device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


class TabICLWrapper(ModelWrapper):
    """TabICL-v2 (Soda-Inria) wrapper.

    Parameters
    ----------
    task_type
        ``"classification"`` or ``"regression"``.
    seed
        Forwarded to TabICL's ``random_state``.
    device
        ``"auto"`` (default), ``"cuda"``, ``"cpu"``, ``"mps"`` or any
        valid ``torch.device`` string.  ``"auto"`` picks ``cuda`` if
        torch reports it is available, otherwise ``cpu``.
    n_estimators
        Number of TabICL ensemble members.  Default 8 (library default).
    kv_cache
        Cache training-data KV projections during ``fit()`` so
        subsequent ``predict`` / ``predict_proba`` calls reuse them.
        Default ``True`` in this wrapper for the per-fit speedup. Set
        to ``False`` (library default) or ``"repr"`` to override.
    support_many_classes
        Forward to TabICL.  Default ``True`` (library default: enables
        hierarchical classification for >10 classes).  Regression
        ignores this parameter.
    batch_size
        Inference batch size.  Default 8 (library default).  Lower
        values trade speed for memory.
    allow_auto_download
        Forward to TabICL.  Default ``True``: downloads the checkpoint
        from Hugging Face Hub on first use.
    verbose
        Forward to TabICL.  Default ``False``.
    **kwargs
        Extra kwargs forwarded to ``TabICLClassifier`` /
        ``TabICLRegressor`` (e.g. ``use_amp``, ``offload_mode``,
        ``softmax_temperature``).

    Raw-feature contract
    --------------------
    ``prefers_raw_features = True`` causes the pipeline to pass the original
    feature frame, including object and categorical columns. This enables
    TabICL's native ordinal encoding, normalization and missing-value handling.
    TabICL does not refuse pre-scaled input, so methods that emit
    ``skip_pipeline_scaler=True`` remain supported.
    """

    # See "Raw-feature contract" above.
    prefers_raw_features: bool = True
    supports_sample_weight: bool = False

    def __init__(
        self,
        task_type: str = "classification",
        seed: int = 42,
        device: str | None = "auto",
        n_estimators: int = 8,
        kv_cache: bool | str = True,
        support_many_classes: bool = True,
        batch_size: int | None = 8,
        allow_auto_download: bool = True,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self._task_type = task_type
        self._seed = int(seed)
        self._device_requested = device
        self._n_estimators = int(n_estimators)
        self._kv_cache = kv_cache
        self._support_many_classes = bool(support_many_classes)
        self._batch_size = batch_size
        self._allow_auto_download = bool(allow_auto_download)
        self._verbose = bool(verbose)
        self._extra_kwargs = dict(kwargs)

        self.model_ = None
        self.classes_ = None
        self._n_classes: int | None = None
        self._feature_names: list[str] | None = None
        self._categorical_features: list[str] = []
        self._resolved_device: str | None = None

    @property
    def name(self) -> str:
        return "tabicl_v2"

    @property
    def execution_metadata(self) -> dict[str, object]:
        return {"model_n_estimators": self._n_estimators}

    # --- Guard ------------------------------------------------------

    def _check_size_guard(self, n_train: int) -> None:
        if n_train > MAX_TRAIN_ROWS:
            raise TabICLSkipReason(
                "train_rows_exceeded",
                {"train_rows": n_train, "limit": MAX_TRAIN_ROWS},
            )

    # --- Fit / predict ---------------------------------------------

    def fit(
        self,
        X_train,
        y_train,
        sample_weight=None,
        **fit_kwargs: Any,
    ):
        """Fit the underlying TabICL model.

        Parameters
        ----------
        X_train, y_train
            Training features (DataFrame or ndarray) and labels.
        sample_weight
            Ignored with a warning.  TabICL's ``.fit(X, y)`` API does
            not accept weights.
        **fit_kwargs
            ``categorical_feature`` is consumed as an explicit list of
            categorical column names. Numeric-stored declared categories are
            cast to object dtype for TabICL's internal ordinal encoder.
        """
        if sample_weight is not None:
            warnings.warn(
                "TabICLWrapper does not support sample_weight. Ignoring "
                f"weights array of length {len(sample_weight)}.",
                RuntimeWarning,
                stacklevel=2,
            )

        categorical_feature = fit_kwargs.pop("categorical_feature", None)

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(
                X_train,
                columns=[f"f{i}" for i in range(X_train.shape[1])],
            )
        else:
            X_train = X_train.copy()
        self._feature_names = list(X_train.columns)
        self._categorical_features = [
            col for col in (categorical_feature or []) if col in X_train.columns
        ]
        for col in self._categorical_features:
            X_train[col] = X_train[col].astype("object")

        y_train = np.asarray(y_train)

        n_train = len(X_train)
        self._check_size_guard(n_train)

        if self._task_type == "classification":
            self.classes_ = np.unique(y_train)
            self._n_classes = len(self.classes_)
        else:
            self._n_classes = None

        device = _resolve_device(self._device_requested)
        self._resolved_device = device

        # TabICL raises ValueError at fit() if kv_cache is enabled with
        # >10 classes. The hierarchical many-class path is incompatible
        # with the KV cache. Downgrade to False with a warning so sweeps
        # do not crash on many-class datasets.
        effective_kv = self._kv_cache
        if (
            self._task_type == "classification"
            and self._support_many_classes
            and self._n_classes is not None
            and self._n_classes > 10
            and bool(effective_kv)
        ):
            warnings.warn(
                f"TabICLWrapper: disabling kv_cache for {self._n_classes}-class "
                "classification (TabICL requires kv_cache=False when "
                "support_many_classes=True and n_classes > 10).",
                RuntimeWarning,
                stacklevel=2,
            )
            effective_kv = False

        common_kwargs: dict[str, Any] = dict(
            n_estimators=self._n_estimators,
            kv_cache=effective_kv,
            batch_size=self._batch_size,
            allow_auto_download=self._allow_auto_download,
            device=device,
            random_state=self._seed,
            verbose=self._verbose,
        )

        if self._task_type == "regression":
            from tabicl import TabICLRegressor
            model_kwargs = dict(common_kwargs)
            model_kwargs.update(self._extra_kwargs)
            self.model_ = TabICLRegressor(**model_kwargs)
        else:
            from tabicl import TabICLClassifier
            model_kwargs = dict(common_kwargs)
            model_kwargs["support_many_classes"] = self._support_many_classes
            model_kwargs.update(self._extra_kwargs)
            self.model_ = TabICLClassifier(**model_kwargs)

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
                    "TabICLWrapper.predict received a frame missing columns "
                    f"that were present at fit time: {missing[:5]}"
                    + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
                )
            X_test = X_test[self._feature_names]
        if self._categorical_features:
            X_test = X_test.copy()
            for col in self._categorical_features:
                if col in X_test.columns:
                    X_test[col] = X_test[col].astype("object")
        return X_test

    def predict(self, X_test):
        X_test = self._predict_df(X_test)
        return np.asarray(self.model_.predict(X_test))

    def predict_proba(self, X_test):
        if self._task_type == "regression":
            return None
        X_test = self._predict_df(X_test)
        proba = np.asarray(self.model_.predict_proba(X_test))
        return self._format_proba(proba)

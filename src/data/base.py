"""
Base data abstractions for the thesis benchmark framework.

Provides:
- TabularDataset: Standardized container for train/test splits with metadata.
  Supports both classification and regression tasks via the ``task_type`` field.
- LeakageGuard: Separates y_test immediately after loading and only
  releases it for evaluation, ensuring the pipeline never has access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


@dataclass
class TabularDataset:
    """Standardized container for a tabular dataset with train/test split.

    Attributes:
        task_type: ``"classification"`` (default) or ``"regression"``.
    """

    name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray  # Only for evaluation. Immediately separated by LeakageGuard
    cat_features: list[str]
    num_features: list[str]
    task_type: Literal["classification", "regression"] = "classification"
    shift_description: str = ""  # e.g. "temporal: 2014→2018"
    metadata: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)

    @property
    def is_regression(self) -> bool:
        return self.task_type == "regression"

    def copy(self) -> "TabularDataset":
        """Return a shallow copy with an independent y_test for LeakageGuard.

        X_train and X_test are *not* deep-copied (they are read-only in the
        pipeline), but y_test gets its own copy. A LeakageGuard can set the copy
        to None without affecting other runs that use the same dataset.
        """
        return TabularDataset(
            name=self.name,
            X_train=self.X_train,
            X_test=self.X_test,
            y_train=self.y_train,
            y_test=self.y_test.copy() if self.y_test is not None else None,
            cat_features=self.cat_features,
            num_features=self.num_features,
            task_type=self.task_type,
            shift_description=self.shift_description,
            metadata=self.metadata,
        )

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.name}  ({self.task_type})",
            f"  Shift: {self.shift_description}",
            f"  Train: {self.n_train:,} samples",
            f"  Test:  {self.n_test:,} samples",
            f"  Features: {self.n_features} ({len(self.cat_features)} cat, {len(self.num_features)} num)",
            f"  Train label mean: {self.y_train.mean():.3f}",
        ]
        if self.y_test is not None:
            lines.append(f"  Test label mean:  {self.y_test.mean():.3f}")
        return "\n".join(lines)


class LeakageGuard:
    """
    Separates y_test immediately after loading and only releases it
    for evaluation after predictions have been made.

    The pipeline does not receive y_test. Only the guard can access y_test when
    computing evaluation metrics.

    Supports both classification and regression tasks (determined
    automatically from ``dataset.task_type``).

    Usage:
        dataset = load_some_dataset(...)
        guard = LeakageGuard(dataset)
        # dataset.y_test is now None: pipeline cannot access it
        # ... run pipeline, get y_pred, y_prob ...
        metrics = guard.evaluate(y_pred, y_prob)
    """

    def __init__(self, dataset: TabularDataset):
        if dataset.y_test is None:
            raise ValueError("y_test is already None: was LeakageGuard applied twice?")
        self._y_test = dataset.y_test.copy()
        self._n_test = len(dataset.X_test)
        self._task_type = dataset.task_type
        # Detect multiclass classification
        self._n_classes = len(np.unique(self._y_test)) if self._task_type == "classification" else 0
        self._is_multiclass = self._n_classes > 2
        # Remove y_test from the dataset object
        dataset.y_test = None

    def evaluate(
        self,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> dict[str, float]:
        """
        Evaluate predictions against the held-out y_test.

        For classification:
            y_pred: Hard predictions (0/1).
            y_prob: Predicted probabilities for the positive class (optional).

        For regression:
            y_pred: Continuous predictions.
            y_prob: Ignored.

        Returns:
            Dictionary with metric names and values.
        """
        if len(y_pred) != self._n_test:
            raise ValueError(
                f"y_pred length ({len(y_pred)}) != expected n_test ({self._n_test})"
            )

        if self._task_type == "regression":
            return self._evaluate_regression(y_pred)
        return self._evaluate_classification(y_pred, y_prob)

    def _evaluate_classification(
        self, y_pred: np.ndarray, y_prob: np.ndarray | None
    ) -> dict[str, float]:
        # Use appropriate averaging for multiclass
        avg = "macro" if self._is_multiclass else "binary"
        metrics: dict[str, float] = {
            "accuracy": accuracy_score(self._y_test, y_pred),
            "f1": f1_score(self._y_test, y_pred, average=avg, zero_division=0),
        }
        if y_prob is not None:
            prob_arr = np.asarray(y_prob)
            if self._is_multiclass:
                # y_prob should be (n_samples, n_classes) for multiclass
                if prob_arr.ndim == 2 and prob_arr.shape[1] == self._n_classes:
                    try:
                        metrics["auc_roc"] = roc_auc_score(
                            self._y_test, prob_arr, multi_class="ovr", average="macro"
                        )
                    except ValueError:
                        metrics["auc_roc"] = float("nan")
                    try:
                        metrics["log_loss"] = log_loss(
                            self._y_test, prob_arr, labels=np.arange(self._n_classes)
                        )
                    except ValueError:
                        metrics["log_loss"] = float("nan")
                else:
                    # Fallback: probabilities do not match expected shape
                    metrics["auc_roc"] = float("nan")
                    metrics["log_loss"] = float("nan")
            else:
                if len(prob_arr) != self._n_test:
                    raise ValueError(
                        f"y_prob length ({len(prob_arr)}) != expected n_test ({self._n_test})"
                    )
                try:
                    metrics["auc_roc"] = roc_auc_score(self._y_test, prob_arr)
                except ValueError:
                    metrics["auc_roc"] = float("nan")
                try:
                    metrics["log_loss"] = log_loss(self._y_test, prob_arr, labels=[0, 1])
                except ValueError:
                    metrics["log_loss"] = float("nan")
        return metrics

    def _evaluate_regression(self, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "rmse": float(np.sqrt(mean_squared_error(self._y_test, y_pred))),
            "mae": float(mean_absolute_error(self._y_test, y_pred)),
            "r2": float(r2_score(self._y_test, y_pred)),
        }

    @property
    def task_type(self) -> str:
        return self._task_type

    @property
    def n_test(self) -> int:
        return self._n_test

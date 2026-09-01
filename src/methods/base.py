"""
Base abstractions for test-time methods.

Every method receives X_train, X_test, y_train and returns transformed
features and optional sample weights. The y_test value is never part of the
signature: the LeakageGuard ensures it does not exist in the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MethodContext:
    """Dataset/model contract supplied by the experiment pipeline.

    Task type and categorical semantics are explicit instead of inferred from
    target values or pandas dtypes. ``raw_X_*`` are the exact method-boundary
    frames after any bootstrap selection.
    """

    task_type: Literal["classification", "regression"]
    categorical_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    raw_X_train: pd.DataFrame
    raw_X_test: pd.DataFrame
    model_name: str
    model_supports_sample_weight: bool

    def __post_init__(self) -> None:
        if self.task_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported task_type in MethodContext: {self.task_type!r}")
        missing = [
            col for col in self.categorical_features
            if col not in self.raw_X_train.columns or col not in self.raw_X_test.columns
        ]
        if missing:
            raise ValueError(
                "Declared categorical features are missing from a method input: "
                f"{missing}"
            )


@dataclass(frozen=True)
class RowProvenance:
    """Source row for every transformed training row."""

    source_split: np.ndarray
    source_position: np.ndarray
    role: np.ndarray

    def __post_init__(self) -> None:
        splits = np.asarray(self.source_split, dtype="U5").ravel().copy()
        positions = np.asarray(self.source_position, dtype=np.int64).ravel().copy()
        roles = np.asarray(self.role, dtype="U32").ravel().copy()
        if not (len(splits) == len(positions) == len(roles)):
            raise ValueError("RowProvenance arrays must have equal length")
        splits.setflags(write=False)
        positions.setflags(write=False)
        roles.setflags(write=False)
        object.__setattr__(self, "source_split", splits)
        object.__setattr__(self, "source_position", positions)
        object.__setattr__(self, "role", roles)

    @classmethod
    def identity_train(cls, n_rows: int) -> "RowProvenance":
        return cls(
            source_split=np.full(n_rows, "train", dtype="U5"),
            source_position=np.arange(n_rows, dtype=np.int64),
            role=np.full(n_rows, "original", dtype="U32"),
        )

    @classmethod
    def augmented(
        cls,
        *,
        n_original_train: int,
        augmented_source_split: str,
        augmented_source_position: np.ndarray,
        augmented_role: str,
    ) -> "RowProvenance":
        aug_pos = np.asarray(augmented_source_position, dtype=np.int64).ravel()
        return cls(
            source_split=np.concatenate([
                np.full(n_original_train, "train", dtype="U5"),
                np.full(len(aug_pos), augmented_source_split, dtype="U5"),
            ]),
            source_position=np.concatenate([
                np.arange(n_original_train, dtype=np.int64),
                aug_pos,
            ]),
            role=np.concatenate([
                np.full(n_original_train, "original", dtype="U32"),
                np.full(len(aug_pos), augmented_role, dtype="U32"),
            ]),
        )

    def validate(self, *, n_rows: int, n_train: int, n_test: int) -> None:
        if len(self.source_split) != n_rows:
            raise ValueError(
                "Row provenance length does not match transformed training rows: "
                f"{len(self.source_split)} != {n_rows}"
            )
        invalid_sources = sorted(set(self.source_split) - {"train", "test"})
        if invalid_sources:
            raise ValueError(f"Invalid row provenance source split(s): {invalid_sources}")
        if np.any(self.source_position < 0):
            raise ValueError("Row provenance positions must be non-negative")
        train_mask = self.source_split == "train"
        test_mask = self.source_split == "test"
        if train_mask.any() and np.any(self.source_position[train_mask] >= n_train):
            raise ValueError("Row provenance references a nonexistent training row")
        if test_mask.any() and np.any(self.source_position[test_mask] >= n_test):
            raise ValueError("Row provenance references a nonexistent test row")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "source_split": self.source_split,
            "source_position": self.source_position,
            "role": self.role,
        })


@dataclass
class TransformResult:
    """Result of a test-time method's fit_transform."""

    X_train: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    sample_weights_train: np.ndarray | None = None  # Optional: for reweighting methods
    y_train: np.ndarray | None = None  # Optional: overrides original y_train (for methods that augment the training set, e.g. M7 PseudoLabel, M17 TDDA)
    metadata: dict = field(default_factory=dict)  # For logging (e.g. PAD score, n_pseudo_labels)
    skip_pipeline_scaler: bool = False  # If True, pipeline skips its universal StandardScaler (used by M5 JointScaling which applies its own joint scaler)
    categorical_features: list[str] | None = None  # Names in feature_names that should be treated as categorical by cat-aware models (e.g. lightgbm_tabarena's lgb.train with categorical_feature=). None == auto-detect / not declared.
    row_provenance_train: RowProvenance | None = None  # Required when training-row cardinality changes.


class TestTimeMethod(ABC):
    """
    Base class for all test-time methods.

    Each method receives X_train, X_test, y_train as input and returns
    transformed features and optional weights. The y_test value is never passed
    because it does not exist in the method signature.
    """

    @abstractmethod
    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
    ) -> TransformResult:
        """
        Apply the test-time method.

        Args:
            X_train: Training features (DataFrame).
            X_test: Test features (DataFrame).
            y_train: Training labels.

        Returns:
            TransformResult with transformed features and optional weights.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this method configuration."""
        ...

    @property
    def uses_labels(self) -> bool:
        """Whether this method uses y_train (e.g. Target Encoding, Label Propagation)."""
        return False

    @property
    def may_emit_sample_weight(self) -> bool:
        """Whether any valid execution can emit ``sample_weights_train``.

        This is a pre-execution capability contract for manifest construction.
        Runtime validation still checks the concrete :class:`TransformResult`.
        """
        return False

    @property
    def produces_prescaled_output(self) -> bool:
        """Whether the method intentionally returns pre-scaled model features."""
        return False

    @property
    def supported_task_types(self) -> frozenset[str]:
        """Dataset task types for which the registered method is meaningful."""
        return frozenset({"classification", "regression"})

    @property
    def method_context(self) -> MethodContext | None:
        """Explicit pipeline context or ``None`` in direct unit calls."""
        return getattr(self, "_method_context", None)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


def _combine_sample_weights(
    all_weights: np.ndarray | None,
    new_weights: np.ndarray,
    method_name: str,
) -> np.ndarray:
    """Multiply pipeline weights, padding prior weights across augmentation.

    When a later method grows ``X_train`` (for example TDDA or pseudo-labeling),
    earlier sample weights only apply to the original rows.  Augmented rows get
    a neutral previous weight of 1.0, then the augmenting method's own weight.
    """
    new_w = np.asarray(new_weights, dtype=float).ravel()
    if all_weights is None:
        return new_w.copy()

    old_w = np.asarray(all_weights, dtype=float).ravel()
    if len(old_w) == len(new_w):
        return old_w * new_w
    if len(new_w) > len(old_w):
        padded = np.concatenate([old_w, np.ones(len(new_w) - len(old_w))])
        return padded * new_w
    raise ValueError(
        f"Method {method_name} returned fewer weights than accumulated. "
        "Row dropping is not supported in pipelines."
    )


def _propagate_seed(method: object, seed: int) -> None:
    """Recursively set ``seed`` on a method and common nested containers."""
    seen: set[int] = set()

    def walk(obj: object | None) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if hasattr(obj, "seed"):
            setattr(obj, "seed", seed)

        for child in getattr(obj, "methods", []) or []:
            walk(child)
        for attr in ("method", "selector", "base_selector"):
            walk(getattr(obj, attr, None))

    walk(method)


def _propagate_context(method: object, context: MethodContext) -> None:
    """Recursively bind an explicit context to wrappers and pipelines."""
    seen: set[int] = set()

    def walk(obj: object | None) -> None:
        if obj is None:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        setattr(obj, "_method_context", context)

        for child in getattr(obj, "methods", []) or []:
            walk(child)
        for attr in ("method", "selector", "base_selector"):
            walk(getattr(obj, attr, None))

    walk(method)


class IdentityMethod(TestTimeMethod):
    """
    Minimal preprocessing: label-encode categoricals, pass numerics through.

    The label encoder is fit on the combined train and test features to ensure
    consistent encoding. This is intentional for a "no-method" control because
    it isolates the effect of the downstream model without introducing
    unseen-category issues.  For a strictly inductive baseline (encoder fit
    on train only), use ``InductiveBaseline`` instead.
    """

    @property
    def name(self) -> str:
        return "identity"

    def fit_transform(self, X_train, X_test, y_train):
        from sklearn.preprocessing import LabelEncoder

        if isinstance(X_train, pd.DataFrame):
            feature_names = list(X_train.columns)
            X_train_enc = X_train.copy()
            X_test_enc = X_test.copy()

            # Label-encode categorical columns so they become numeric
            cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
            for col in cat_cols:
                le = LabelEncoder()
                combined = pd.concat([X_train_enc[col].astype(str), X_test_enc[col].astype(str)])
                le.fit(combined)
                X_train_enc[col] = le.transform(X_train_enc[col].astype(str))
                X_test_enc[col] = le.transform(X_test_enc[col].astype(str))

            X_tr = X_train_enc.values.astype(float)
            X_te = X_test_enc.values.astype(float)
        else:
            feature_names = [f"f{i}" for i in range(X_train.shape[1])]
            X_tr = X_train.astype(float)
            X_te = X_test.astype(float)

        return TransformResult(
            X_train=X_tr,
            X_test=X_te,
            feature_names=feature_names,
        )


class InductiveBaseline(TestTimeMethod):
    """
    Standard inductive baseline: LabelEncode categoricals (fit on train only).

    This is the baseline against which all test-time methods are compared.
    StandardScaler is not applied here because the pipeline
    (``run_single_experiment``) already applies a universal StandardScaler
    after every method.  Duplicating it here would cause double-scaling and
    make the baseline's preprocessing inconsistent with other methods.
    """

    @property
    def name(self) -> str:
        return "inductive_baseline"

    def fit_transform(self, X_train, X_test, y_train):
        from sklearn.preprocessing import LabelEncoder

        X_train_enc = X_train.copy()
        X_test_enc = X_test.copy()

        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        # Label encode categoricals. Fit on train only
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(X_train_enc[col].astype(str))

            X_train_enc[col] = le.transform(X_train_enc[col].astype(str))

            # Handle unseen categories in test → map to -1
            test_vals = X_test_enc[col].astype(str)
            known_mask = test_vals.isin(le.classes_)
            X_test_enc[col] = -1
            X_test_enc.loc[known_mask, col] = le.transform(test_vals[known_mask])

        X_tr_out = X_train_enc.values.astype(float)
        X_te_out = X_test_enc.values.astype(float)

        feature_names = list(X_train.columns)
        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            # Categorical columns remain as label-encoded floats. Declaring them here lets
            # cat-aware models (e.g. lightgbm_tabarena) use native categorical
            # splits. Models without this capability ignore the field.
            categorical_features=cat_cols if cat_cols else None,
        )


class MethodPipeline:
    """
    Combines multiple TestTimeMethod instances sequentially.

    Each method's output feeds into the next. Sample weights are
    combined multiplicatively across methods.
    """

    def __init__(self, methods: list[TestTimeMethod]):
        self.methods = methods

    @property
    def name(self) -> str:
        return " → ".join(m.name for m in self.methods)

    @property
    def may_emit_sample_weight(self) -> bool:
        return any(method.may_emit_sample_weight for method in self.methods)

    @property
    def produces_prescaled_output(self) -> bool:
        return any(method.produces_prescaled_output for method in self.methods)

    @property
    def supported_task_types(self) -> frozenset[str]:
        supported = frozenset({"classification", "regression"})
        for method in self.methods:
            supported = supported.intersection(method.supported_task_types)
        return supported

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
    ) -> TransformResult:
        current_train = X_train
        current_test = X_test
        current_y_train = y_train
        all_weights = None
        all_metadata = {}
        final_y_train = None
        any_skip_scaler = False
        final_cat_features: list[str] | None = None
        current_provenance = RowProvenance.identity_train(len(X_train))

        for i, method in enumerate(self.methods):
            # Ensure DataFrames for methods that expect them
            if not isinstance(current_train, pd.DataFrame):
                cols = [f"f{j}" for j in range(current_train.shape[1])]
                current_train = pd.DataFrame(current_train, columns=cols)
                current_test = pd.DataFrame(current_test, columns=cols)

            n_rows_before = len(current_train)
            result = method.fit_transform(current_train, current_test, current_y_train)

            if result.row_provenance_train is not None:
                current_provenance = result.row_provenance_train
            elif len(result.X_train) != n_rows_before:
                raise ValueError(
                    f"Method {method.name} changed training-row cardinality without "
                    "returning row_provenance_train"
                )

            # Wrap output back into DataFrames for the next method
            current_train = pd.DataFrame(result.X_train, columns=result.feature_names)
            current_test = pd.DataFrame(result.X_test, columns=result.feature_names)

            # If this method augmented y_train, propagate to subsequent methods
            if result.y_train is not None:
                current_y_train = result.y_train
                final_y_train = result.y_train

            # Combine weights multiplicatively
            if result.sample_weights_train is not None:
                all_weights = _combine_sample_weights(
                    all_weights, result.sample_weights_train, method.name
                )

            # Merge metadata
            if result.metadata:
                all_metadata[method.name] = result.metadata

            # Propagate skip_pipeline_scaler (OR: if any method requests it)
            if result.skip_pipeline_scaler:
                any_skip_scaler = True

            # Carry forward the final method's categorical-feature declaration. Only
            # the final step's output reaches the model, so only its
            # categorical_features matters downstream.
            final_cat_features = result.categorical_features

        return TransformResult(
            X_train=current_train.values,
            X_test=current_test.values,
            feature_names=list(current_train.columns),
            sample_weights_train=all_weights,
            y_train=final_y_train,
            metadata=all_metadata,
            skip_pipeline_scaler=any_skip_scaler,
            categorical_features=final_cat_features,
            row_provenance_train=current_provenance,
        )

    def __repr__(self) -> str:
        return f"MethodPipeline([{', '.join(repr(m) for m in self.methods)}])"


class CatPreservingMethodPipeline(MethodPipeline):
    """
    Variant of :class:`MethodPipeline` that re-injects the original
    categorical columns between steps.

    Motivation
    ----------
    The base ``MethodPipeline`` stores each step's output as a numpy float
    array in ``TransformResult.X_train`` and re-wraps it as a DataFrame for
    the next method. Conversion removes object and category dtypes. Methods that
    use ``select_dtypes(include=["object", "category"])`` therefore see no
    categorical columns from step 2 onward, even when the prior method retained
    the columns.
    Methods that explicitly drop the originals (M1 ``JointFrequencyEncoding``, M4
    target-encoding variants) make this unavoidable for greedy stacks like
    ``joint_freq → freq_ratio``.

    Behavior
    --------
    Before each method after the first runs, the pipeline checks which of the
    *original* categorical column names (captured once at the start of
    ``fit_transform``) are either absent or have lost their object/category
    dtype in the pipeline's current state.  Those columns are re-inserted
    with their original dtypes, overwriting any label-encoded shadow.  The
    next method receives the categorical columns again and can apply its
    transformation.

    The final output is returned exactly as the last method produced it:
    a NumPy float array, so downstream callers such as
    ``run_single_experiment`` continue to work unchanged.  Re-injection
    happens strictly between steps. The pipeline boundary is unaffected.

    Trade-offs
    ----------
    - Feature matrix may grow: encoded cat features (from step 1) and the
      original cats (label-encoded by whichever method consumed them in a
      later step) can both appear in the final output. Tree models can use
      redundant features. Linear models may receive collinear signals.
    - If a method mutates cat values in place (e.g. imputing to
      ``"MISSING"``), the re-injected version reflects the pre-method
      state.  No method in the current codebase does this.
    - Single-method pipelines are unchanged: re-injection happens only
      between steps.
    """

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
    ) -> TransformResult:
        # Record the original categorical columns for re-injection between
        # subsequent steps.
        original_cat_cols = X_train.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        orig_cats_train = (
            X_train[original_cat_cols].copy().reset_index(drop=True)
            if original_cat_cols else None
        )
        orig_cats_test = (
            X_test[original_cat_cols].copy().reset_index(drop=True)
            if original_cat_cols else None
        )

        current_train = X_train
        current_test = X_test
        current_y_train = y_train
        all_weights = None
        all_metadata: dict = {}
        final_y_train = None
        any_skip_scaler = False
        final_cat_features: list[str] | None = None
        current_provenance = RowProvenance.identity_train(len(X_train))

        for i, method in enumerate(self.methods):
            # Re-wrap numpy output from previous step back into DataFrames.
            if not isinstance(current_train, pd.DataFrame):
                cols = [f"f{j}" for j in range(current_train.shape[1])]
                current_train = pd.DataFrame(current_train, columns=cols)
                current_test = pd.DataFrame(current_test, columns=cols)

            # Reinsert original categorical columns after the first step. The
            # initial input already has the correct dtypes. If an earlier method
            # added rows, the original categorical rows no longer align with the
            # transformed data. In that case, use the transformed data unchanged
            # for the remaining steps.
            rows_changed = (
                original_cat_cols
                and (
                    len(current_train) != len(orig_cats_train)
                    or len(current_test) != len(orig_cats_test)
                )
            )
            if rows_changed:
                all_metadata.setdefault("_pipeline", {})[
                    "cat_reinjection_skipped_after_row_change"
                ] = True
            if i > 0 and original_cat_cols and not rows_changed:
                current_cat_cols = set(
                    current_train.select_dtypes(
                        include=["object", "category"]
                    ).columns
                )
                need_reinject = [
                    c for c in original_cat_cols if c not in current_cat_cols
                ]
                if need_reinject:
                    # If a label-encoded shadow of the cat still exists in
                    # the DataFrame under the same name, drop it first to avoid
                    # duplicate columns.
                    drop_first = [
                        c for c in need_reinject if c in current_train.columns
                    ]
                    if drop_first:
                        current_train = current_train.drop(columns=drop_first)
                        current_test = current_test.drop(columns=drop_first)
                    current_train = pd.concat(
                        [
                            current_train.reset_index(drop=True),
                            orig_cats_train[need_reinject].reset_index(drop=True),
                        ],
                        axis=1,
                    )
                    current_test = pd.concat(
                        [
                            current_test.reset_index(drop=True),
                            orig_cats_test[need_reinject].reset_index(drop=True),
                        ],
                        axis=1,
                    )

            n_rows_before = len(current_train)
            result = method.fit_transform(current_train, current_test, current_y_train)

            if result.row_provenance_train is not None:
                current_provenance = result.row_provenance_train
            elif len(result.X_train) != n_rows_before:
                raise ValueError(
                    f"Method {method.name} changed training-row cardinality without "
                    "returning row_provenance_train"
                )

            current_train = pd.DataFrame(result.X_train, columns=result.feature_names)
            current_test = pd.DataFrame(result.X_test, columns=result.feature_names)

            if result.y_train is not None:
                current_y_train = result.y_train
                final_y_train = result.y_train

            if result.sample_weights_train is not None:
                all_weights = _combine_sample_weights(
                    all_weights, result.sample_weights_train, method.name
                )

            if result.metadata:
                all_metadata[method.name] = result.metadata

            if result.skip_pipeline_scaler:
                any_skip_scaler = True

            final_cat_features = result.categorical_features

        return TransformResult(
            X_train=current_train.values,
            X_test=current_test.values,
            feature_names=list(current_train.columns),
            sample_weights_train=all_weights,
            y_train=final_y_train,
            metadata=all_metadata,
            skip_pipeline_scaler=any_skip_scaler,
            categorical_features=final_cat_features,
            row_provenance_train=current_provenance,
        )

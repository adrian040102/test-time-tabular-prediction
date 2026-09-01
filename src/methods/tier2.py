"""
Tier 2 Test-Time Methods: Model-Based Test-Time Methods.

These methods use auxiliary models or algorithmic procedures to extract
additional information from the combination of train and test data.

Methods:
    M6:  AdversarialValidation: OOF domain classifier as feature / weight / both
    M7:  PseudoLabelSelfTraining: iterative self-training with confidence threshold
    M8:  LabelPropagationFeature: OOF label propagation soft labels as feature
    M9:  OTDisplacementFeatures: optimal transport displacement vectors as features
    M13: AdversarialReweighting: importance weighting via domain classifier scores
    M16: NearestTestDistance: distance to k nearest test neighbors as feature / weight
    M17: TestDirectedDataAugmentation: mixup between train points and their test neighbors
    M18: LeafBasedTestDensityFE: train/test leaf occupancy analysis as shift diagnostic
    M19: NeighborhoodConsensusFeatures: two-pass neighborhood-consensus features from OOF predictions
    M20: ClusterMembershipFE: cluster-aware local reweighting and shift features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

from src.methods.base import RowProvenance, TestTimeMethod, TransformResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_encode_for_classifier(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Label-encode categoricals and return numeric arrays for the domain classifier.

    Delegates to :func:`preprocessing.encode_for_classifier`.
    """
    from src.methods.preprocessing import encode_for_classifier
    return encode_for_classifier(X_train, X_test)


def _build_domain_classifier(classifier: str, seed: int = 42):
    """Instantiate a lightweight classifier for domain discrimination."""
    if classifier == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            # The sklearn API applies row subsampling only when
            # ``subsample_freq`` is at least one, so it is omitted here.
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
            n_jobs=1,
        )
    elif classifier == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            random_state=seed,
        )
    elif classifier == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(
            f"Unknown domain classifier '{classifier}'. "
            "Options: 'lgbm', 'logreg', 'rf'"
        )


def _build_task_classifier(classifier: str, seed: int = 42, n_classes: int = 2):
    """
    Instantiate a classifier for *task-label* prediction (pseudo-labeling).

    Unlike ``_build_domain_classifier`` (tuned for binary domain discrimination),
    this uses slightly more capacity to handle multiclass and harder task labels.
    """
    if classifier == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            # Subsampling is omitted because it has no effect unless
            # subsample_freq is at least 1.
            colsample_bytree=0.8,
            num_class=n_classes if n_classes > 2 else 1,
            random_state=seed,
            verbose=-1,
            n_jobs=1,
        )
    elif classifier == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=500,
            solver="lbfgs",
            multi_class="auto",
            random_state=seed,
        )
    elif classifier == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=seed,
            n_jobs=1,
        )
    else:
        raise ValueError(
            f"Unknown task classifier '{classifier}'. "
            "Options: 'lgbm', 'logreg', 'rf'"
        )


def _compute_pad(oof_scores: np.ndarray, is_test: np.ndarray) -> float:
    """
    Compute Proxy-A-Distance (PAD) from OOF domain classifier scores.

    PAD = 2 * (1 - 2 * error_rate)
    where error_rate = fraction of misclassified samples at threshold 0.5.

    PAD ≈ 0 means no shift, PAD ≈ 2 means perfect separation.
    """
    predictions = (oof_scores >= 0.5).astype(int)
    error_rate = np.mean(predictions != is_test)
    pad = 2.0 * (1.0 - 2.0 * error_rate)
    return float(max(pad, 0.0))  # Clip at 0 (below-chance = no shift)


def _run_domain_oof(
    X_train: np.ndarray,
    X_test: np.ndarray,
    classifier: str = "lgbm",
    n_folds: int = 5,
    seed: int = 42,
    calibrate: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Shared out-of-fold domain-classification loop.

    Trains a domain classifier (train=0, test=1) using stratified k-fold
    OOF to produce per-sample shift scores.

    Args:
        X_train: Numeric training features (already encoded).
        X_test: Numeric test features (already encoded).
        classifier: Classifier type ("lgbm", "logreg", "rf").
        n_folds: Number of OOF folds.
        seed: Random seed.
        calibrate: Optional calibration method: "platt", "isotonic"
            or None. When set, OOF scores are
            post-hoc calibrated using CalibratedClassifierCV.

    Returns:
        train_scores: (n_train,) P(test|x) for training samples.
        test_scores:  (n_test,)  P(test|x) for test samples.
        pad: Proxy-A-Distance.
        domain_accuracy: OOF domain classification accuracy.
    """
    n_train = len(X_train)
    n_test = len(X_test)

    X_combined = np.vstack([X_train, X_test])
    is_test = np.array([0] * n_train + [1] * n_test)

    oof_scores = np.zeros(len(X_combined))
    kf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=seed
    )

    for fold_train_idx, fold_val_idx in kf.split(X_combined, is_test):
        clf = _build_domain_classifier(classifier, seed)
        X_fold_train = X_combined[fold_train_idx]
        X_fold_val = X_combined[fold_val_idx]
        # Only impute NaN for classifiers that cannot handle it natively.
        # LightGBM learns optimal split directions for missing values.
        # Filling with 0.0 destroys that signal.
        if classifier != "lgbm":
            X_fold_train = np.nan_to_num(X_fold_train, nan=0.0)
            X_fold_val = np.nan_to_num(X_fold_val, nan=0.0)

        if calibrate is not None:
            # Apply optional probability calibration.
            from sklearn.calibration import CalibratedClassifierCV
            cal_method = "sigmoid" if calibrate == "platt" else "isotonic"
            try:
                cal_clf = CalibratedClassifierCV(
                    clf, method=cal_method, cv=3,
                )
                cal_clf.fit(X_fold_train, is_test[fold_train_idx])
                proba = cal_clf.predict_proba(X_fold_val)
            except Exception:
                # Fallback: use uncalibrated if calibration fails
                clf.fit(X_fold_train, is_test[fold_train_idx])
                proba = clf.predict_proba(X_fold_val)
        else:
            clf.fit(X_fold_train, is_test[fold_train_idx])
            proba = clf.predict_proba(X_fold_val)

        oof_scores[fold_val_idx] = proba[:, 1]

    train_scores = oof_scores[:n_train]
    test_scores = oof_scores[n_train:]
    pad = _compute_pad(oof_scores, is_test)
    domain_accuracy = float(np.mean((oof_scores >= 0.5) == is_test))

    return train_scores, test_scores, pad, domain_accuracy


def _clip_weights(
    weights: np.ndarray,
    strategy: str = "hard",
    clip_min: float = 0.1,
    clip_max: float = 10.0,
    winsorize_pct: float = 0.05,
) -> np.ndarray:
    """
    Clip / normalize sample weights to avoid extreme values.

    Strategies:
        "hard": np.clip(weights, clip_min, clip_max)
        "normalize": linear normalization so weights sum to len(weights)
        "softmax": alias for "normalize" (Softmax-Normalisierung)
        "winsorize": clip extremes to the (pct, 1-pct) percentiles
    """
    if strategy == "hard":
        return np.clip(weights, clip_min, clip_max)
    elif strategy in ("normalize", "softmax"):
        # Linear normalization to preserve expected total weight
        w = weights.copy()
        total = w.sum()
        if total < 1e-10:
            # Use uniform weights when the total is numerically zero.
            return np.ones_like(w)
        w = w / total * len(w)
        return w
    elif strategy == "winsorize":
        lo = np.percentile(weights, winsorize_pct * 100)
        hi = np.percentile(weights, (1 - winsorize_pct) * 100)
        return np.clip(weights, lo, hi)
    else:
        raise ValueError(
            f"Unknown clip strategy '{strategy}'. "
            "Options: 'hard', 'normalize', 'softmax', 'winsorize'"
        )


# ===========================================================================
# M6: Domain Classifier with Out-of-Fold Prediction
# ===========================================================================

class AdversarialValidation(TestTimeMethod):
    """
    M6: Train a domain classifier (train=0, test=1) using OOF to produce
    shift scores. These scores can be returned as:

      - "feature": append domain score as an extra column
      - "weight":  use scores as sample_weights_train
      - "both":    do both

    The OOF procedure prevents a sample's training-fold model from predicting it during
    training, avoiding overly optimistic scores.

    Also computes PAD (Proxy-A-Distance) as a shift metric in metadata.
    """

    def __init__(
        self,
        classifier: str = "lgbm",
        n_folds: int = 5,
        return_as: str = "feature",
        weight_type: str = "ratio",
        clip_strategy: str = "hard",
        clip_min: float = 0.1,
        clip_max: float = 10.0,
        calibrate: str | None = None,
        tempering: str | float | None = None,
        pad_max: float = 1.0,
        seed: int = 42,
    ):
        assert return_as in ("feature", "weight", "both"), \
            f"return_as must be 'feature', 'weight' or 'both', got '{return_as}'"
        assert weight_type in ("ratio", "raw"), \
            f"weight_type must be 'ratio' or 'raw', got '{weight_type}'"
        assert calibrate in (None, "platt", "isotonic"), \
            f"calibrate must be None, 'platt' or 'isotonic', got '{calibrate}'"

        self.classifier = classifier
        self.n_folds = n_folds
        self.return_as = return_as
        self.weight_type = weight_type
        self.clip_strategy = clip_strategy
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.calibrate = calibrate
        self.tempering = tempering
        self.pad_max = pad_max
        self.seed = seed

    @property
    def name(self) -> str:
        parts = [f"adversarial_{self.classifier}"]
        if self.calibrate:
            parts.append(f"cal_{self.calibrate}")
        parts.append(f"as_{self.return_as}")
        if self.return_as in ("weight", "both"):
            parts.append(self.weight_type)
        return "_".join(parts)

    @property
    def may_emit_sample_weight(self) -> bool:
        return self.return_as in ("weight", "both")

    def fit_transform(self, X_train, X_test, y_train):
        # --- Step 1: Encode categoricals for the domain classifier ---
        X_tr_enc, X_te_enc = _label_encode_for_classifier(X_train, X_test)

        # --- Step 2–5: Shared out-of-fold domain classification ---
        train_scores, test_scores, pad, domain_accuracy = _run_domain_oof(
            X_tr_enc, X_te_enc,
            classifier=self.classifier,
            n_folds=self.n_folds,
            seed=self.seed,
            calibrate=self.calibrate,
        )

        # --- Step 6: Build output ---
        # Start with label-encoded features from the original data
        # (same as IdentityMethod / InductiveBaseline preprocessing)
        from src.methods.tier1 import _label_encode_cats
        X_tr_base, X_te_base, _ = _label_encode_cats(X_train, X_test)
        X_tr_out = X_tr_base.values.astype(float)
        X_te_out = X_te_base.values.astype(float)
        feature_names = list(X_train.columns)

        sample_weights = None

        # Add domain score as feature
        if self.return_as in ("feature", "both"):
            X_tr_out = np.column_stack([X_tr_out, train_scores])
            X_te_out = np.column_stack([X_te_out, test_scores])
            feature_names = feature_names + ["domain_score"]

        # Compute sample weights
        if self.return_as in ("weight", "both"):
            if self.weight_type == "ratio":
                # Importance weight: P(test|x) / P(train|x)
                # Higher weight for train samples that look like test
                raw_weights = train_scores / (1.0 - train_scores + 1e-8)
            else:
                # Raw P(test|x)
                raw_weights = train_scores.copy()

            sample_weights = _clip_weights(
                raw_weights,
                strategy=self.clip_strategy,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )

            # Weight tempering: blend domain weights toward uniform
            if self.tempering is not None:
                if self.tempering == "pad":
                    alpha = min(pad / self.pad_max, 1.0) if self.pad_max > 0 else 1.0
                else:
                    alpha = float(self.tempering)
                sample_weights = (1.0 - alpha) + alpha * sample_weights

        metadata = {
            "pad": pad,
            "domain_accuracy": domain_accuracy,
            "mean_train_score": float(train_scores.mean()),
            "mean_test_score": float(test_scores.mean()),
        }
        if sample_weights is not None:
            ess = float(sample_weights.sum()**2 / (sample_weights**2).sum())
            metadata["ess"] = ess
            metadata["ess_ratio"] = ess / len(sample_weights)
            if self.tempering is not None:
                metadata["tempering_alpha"] = alpha

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            sample_weights_train=sample_weights,
            metadata=metadata,
        )


# ===========================================================================
# M13: Sample Reweighting via Domain Classifier
# ===========================================================================

class AdversarialReweighting(TestTimeMethod):
    """
    M13: Use domain classifier shift scores purely as sample weights.

    This is a convenience wrapper that:
    1. Runs the domain classifier OOF internally
    2. Converts scores to importance weights
    3. Returns original features unchanged + weights

    The key difference from AdversarialValidation(return_as="weight") is that
    this class provides multiple clipping strategies as named variants and
    does not add the domain score as a feature. It only reweights.
    """

    def __init__(
        self,
        classifier: str = "lgbm",
        n_folds: int = 5,
        weight_type: str = "ratio",
        clip_strategy: str = "hard",
        clip_min: float = 0.1,
        clip_max: float = 10.0,
        winsorize_pct: float = 0.05,
        calibrate: str | None = None,
        tempering: str | float | None = None,
        pad_max: float = 1.0,
        seed: int = 42,
    ):
        self.classifier = classifier
        self.n_folds = n_folds
        self.weight_type = weight_type
        self.clip_strategy = clip_strategy
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.winsorize_pct = winsorize_pct
        self.calibrate = calibrate
        self.tempering = tempering
        self.pad_max = pad_max
        self.seed = seed

    @property
    def name(self) -> str:
        cal = f"_cal_{self.calibrate}" if self.calibrate else ""
        temp = "_tempered" if self.tempering is not None else ""
        return (
            f"adversarial_reweight_{self.classifier}{cal}_{self.weight_type}"
            f"_clip_{self.clip_strategy}{temp}"
        )

    @property
    def may_emit_sample_weight(self) -> bool:
        return True

    def fit_transform(self, X_train, X_test, y_train):
        # --- Shared out-of-fold domain classification ---
        X_tr_enc, X_te_enc = _label_encode_for_classifier(X_train, X_test)
        train_scores, _, pad, domain_accuracy = _run_domain_oof(
            X_tr_enc, X_te_enc,
            classifier=self.classifier,
            n_folds=self.n_folds,
            seed=self.seed,
            calibrate=self.calibrate,
        )

        # --- Compute importance weights ---
        if self.weight_type == "ratio":
            raw_weights = train_scores / (1.0 - train_scores + 1e-8)
        else:
            raw_weights = train_scores.copy()

        sample_weights = _clip_weights(
            raw_weights,
            strategy=self.clip_strategy,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
            winsorize_pct=self.winsorize_pct,
        )

        # Weight tempering: blend domain weights toward uniform
        tempering_alpha = None
        if self.tempering is not None:
            if self.tempering == "pad":
                tempering_alpha = min(pad / self.pad_max, 1.0) if self.pad_max > 0 else 1.0
            else:
                tempering_alpha = float(self.tempering)
            sample_weights = (1.0 - tempering_alpha) + tempering_alpha * sample_weights

        # --- Return original features (label-encoded) + weights ---
        from src.methods.tier1 import _label_encode_cats
        X_tr_base, X_te_base, _ = _label_encode_cats(X_train, X_test)

        ess = float(sample_weights.sum()**2 / (sample_weights**2).sum())
        metadata = {
            "pad": pad,
            "domain_accuracy": domain_accuracy,
            "mean_weight": float(sample_weights.mean()),
            "std_weight": float(sample_weights.std()),
            "min_weight": float(sample_weights.min()),
            "max_weight": float(sample_weights.max()),
            "ess": ess,
            "ess_ratio": ess / len(sample_weights),
        }
        if tempering_alpha is not None:
            metadata["tempering_alpha"] = tempering_alpha

        return TransformResult(
            X_train=X_tr_base.values.astype(float),
            X_test=X_te_base.values.astype(float),
            feature_names=list(X_train.columns),
            sample_weights_train=sample_weights,
            metadata=metadata,
        )


# ===========================================================================
# M7: Pseudo-Label Self-Training
# ===========================================================================

class PseudoLabelSkipReason(RuntimeError):
    """Raised when a pseudo-labeling method (M7) is applied to a regression target.

    Pseudo-labeling is classification-only: it augments the training set with
    high-confidence *class* predictions on the test features. A continuous
    target has no confident class, so the method raises ``PseudoLabelSkipReason``
    instead of fitting a classifier to continuous targets.

    Subclasses :class:`RuntimeError` so the SLURM worker writes
    ``pseudo_label_skip: <reason>`` to the per-task CSV ``error`` column without
    stopping the array task. ``JointTSNESkipReason``, ``TabPFNSkipReason`` and
    ``TabICLSkipReason`` use the same convention.
    """

    def __init__(self, reason: str, details: dict | None = None):
        self.reason = reason
        self.details = details or {}
        msg = f"pseudo_label_skip: {reason}"
        if self.details:
            msg = f"{msg} ({', '.join(f'{k}={v}' for k, v in self.details.items())})"
        super().__init__(msg)


def _pseudo_is_classification(y_train: np.ndarray) -> bool:
    """Heuristic target-type check shared by the M7 pseudo-labeling methods.

    Mirrors the M18/M19 ``_is_classification`` rule: bool/int/object targets
    and small-cardinality integer-valued floats are classification. Continuous
    floats are regression.
    """
    y_arr = np.asarray(y_train)
    unique_y = np.unique(y_arr)
    if y_arr.dtype.kind in "biuO":
        return True
    if len(unique_y) <= 50 and np.allclose(unique_y, np.round(unique_y)):
        return True
    return False


class PseudoLabelSelfTraining(TestTimeMethod):
    """
    M7: Iterative self-training with pseudo-labels.

    1. Train a lightweight model on (X_train, y_train).
    2. Predict on X_test. Keep samples above confidence threshold.
    3. Add high-confidence pseudo-labeled test samples to training set.
    4. Repeat for max_iterations.

    Returns the augmented training set (original + pseudo-labeled) and
    original test set. The pseudo-labeled samples can receive lower sample
    weight through ``pseudo_weight``.

    Args:
        confidence_threshold: Minimum predicted probability to accept a
            pseudo-label (default 0.9).
        max_iterations: Number of self-training rounds (default 3).
        classifier: Model used for pseudo-labeling ("lgbm", "logreg", "rf").
        pseudo_weight: Sample weight assigned to pseudo-labeled rows
            (default 0.5).  Real training rows always get weight 1.0.
        seed: Random seed.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        max_iterations: int = 3,
        classifier: str = "lgbm",
        pseudo_weight: float = 0.5,
        seed: int = 42,
    ):
        assert 0.5 < confidence_threshold < 1.0, \
            f"confidence_threshold must be in (0.5, 1.0), got {confidence_threshold}"
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.classifier = classifier
        self.pseudo_weight = pseudo_weight
        self.seed = seed

    @property
    def name(self) -> str:
        return (
            f"pseudo_label_{self.classifier}"
            f"_conf{self.confidence_threshold}"
            f"_iter{self.max_iterations}"
        )

    @property
    def uses_labels(self) -> bool:
        return True

    @property
    def may_emit_sample_weight(self) -> bool:
        return True

    @property
    def supported_task_types(self) -> frozenset[str]:
        return frozenset({"classification"})

    def fit_transform(self, X_train, X_test, y_train):
        # M7 pseudo-labeling is classification-only. Raise PseudoLabelSkipReason
        # instead of fitting a classifier to continuous regression targets.
        is_classification = (
            self.method_context.task_type == "classification"
            if self.method_context is not None
            else _pseudo_is_classification(np.asarray(y_train))
        )
        if not is_classification:
            raise PseudoLabelSkipReason(
                "regression_unsupported",
                {"n_unique": int(len(np.unique(y_train))),
                 "dtype": str(np.asarray(y_train).dtype)},
            )
        from src.methods.tier1 import _label_encode_cats

        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        y_tr = np.asarray(y_train).copy()

        n_original_train = len(X_tr_arr)

        # Track which test indices have already been pseudo-labeled
        available_mask = np.ones(len(X_te_arr), dtype=bool)
        pseudo_X_parts: list[np.ndarray] = []
        pseudo_y_parts: list[np.ndarray] = []
        pseudo_source_parts: list[np.ndarray] = []

        total_pseudo = 0
        n_iterations_executed = 0
        iteration_stats: list[dict] = []

        for iteration in range(self.max_iterations):
            n_iterations_executed += 1

            # Build current augmented training set
            if pseudo_X_parts:
                X_aug = np.vstack([X_tr_arr] + pseudo_X_parts)
                y_aug = np.concatenate([y_tr] + pseudo_y_parts)
            else:
                X_aug = X_tr_arr
                y_aug = y_tr

            # Train classifier. Use the dedicated task classifier which has
            # more capacity than the domain classifier (deeper trees, more
            # estimators) since task-label prediction can be harder.
            n_classes = len(np.unique(y_aug))
            clf = _build_task_classifier(
                self.classifier, self.seed + iteration, n_classes=n_classes
            )
            # Only impute NaN for classifiers that cannot handle it natively.
            # LightGBM learns optimal split directions for missing values.
            # Filling with 0.0 destroys that signal.
            X_aug_fit = X_aug if self.classifier == "lgbm" else np.nan_to_num(X_aug, nan=0.0)
            clf.fit(X_aug_fit, y_aug)

            # Predict on remaining test samples.
            if not available_mask.any():
                break

            X_remaining = X_te_arr[available_mask]
            if self.classifier != "lgbm":
                X_remaining = np.nan_to_num(X_remaining, nan=0.0)
            proba = clf.predict_proba(X_remaining)

            # For binary: proba shape (n, 2). For multiclass: (n, C)
            max_proba = proba.max(axis=1)
            # Map argmax indices back to the original label space.
            # The clf.classes_ attribute contains the actual labels, including
            # strings and non-contiguous integers.
            predicted_labels = clf.classes_[proba.argmax(axis=1)]

            # Select high-confidence samples
            confident = max_proba >= self.confidence_threshold
            n_new = confident.sum()

            if n_new == 0:
                break

            # Map back to original test indices
            remaining_indices = np.where(available_mask)[0]
            selected_indices = remaining_indices[confident]

            pseudo_X_parts.append(X_te_arr[selected_indices])
            pseudo_y_parts.append(predicted_labels[confident])
            pseudo_source_parts.append(selected_indices.copy())
            available_mask[selected_indices] = False
            total_pseudo += n_new

            iteration_stats.append({
                "iteration": iteration,
                "n_accepted": int(n_new),
                "mean_confidence": float(max_proba[confident].mean()),
            })

        # Build final augmented training arrays
        if pseudo_X_parts:
            X_tr_out = np.vstack([X_tr_arr] + pseudo_X_parts)
            y_tr_out = np.concatenate([y_tr] + pseudo_y_parts)
            # Weights: 1.0 for real, pseudo_weight for pseudo-labeled
            weights = np.concatenate([
                np.ones(n_original_train),
                np.full(total_pseudo, self.pseudo_weight),
            ])
        else:
            X_tr_out = X_tr_arr
            y_tr_out = y_tr
            weights = None

        pseudo_source_positions = (
            np.concatenate(pseudo_source_parts)
            if pseudo_source_parts else np.array([], dtype=np.int64)
        )
        row_provenance = RowProvenance.augmented(
            n_original_train=n_original_train,
            augmented_source_split="test",
            augmented_source_position=pseudo_source_positions,
            augmented_role="pseudo_label",
        )

        feature_names = list(X_tr_enc.columns)

        metadata = {
            "n_pseudo_labels": total_pseudo,
            "n_iterations_run": n_iterations_executed,
            "n_train_augmented": len(X_tr_out),
            "iteration_stats": iteration_stats,
        }

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_arr,
            feature_names=feature_names,
            sample_weights_train=weights,
            y_train=y_tr_out if total_pseudo > 0 else None,
            metadata=metadata,
            row_provenance_train=row_provenance,
        )


# ===========================================================================
# M16: Nearest-Test-Neighbor Distance
# ===========================================================================

class NearestTestDistance(TestTimeMethod):
    """
    M16: For each training sample, compute the distance to the k nearest
    test samples and compute the reverse distances for test samples. The method
    returns features, sample weights or both.

    Features produced (when return_as includes "feature"):
      - mean_dist_to_other:   mean distance to k nearest cross-distribution neighbors
      - min_dist_to_other:    distance to single nearest cross-distribution neighbor
      - mean_dist_self_density: mean distance to own k-NN (local density)
      - nearest_train_dist:   distance to nearest training point (LOO for
            train, direct for test).  Same semantics for both splits: gives
            the downstream model an explicit extrapolation signal.

    Weight formula (train only):
      weight_i = 1 / (1 + mean_dist_k_i)
      → nearby train points get higher weight.

    Args:
        k: Number of nearest neighbors (default 10).
        return_as: "feature", "weight" or "both".
        feature_weighting: How to weight/select features for distance computation.
            "none": unweighted Euclidean distance on all features
            "ks": soft-weight each feature by its KS statistic
            "filter": only use features with KS p-value < ks_threshold
            "cumshift": keep the fewest features capturing ``shift_ratio``
                         of total KS shift (adaptive dimensionality reduction)
        ks_threshold: p-value threshold for "filter" mode (default 0.05).
        shift_ratio: Fraction of total KS shift to capture in "cumshift"
            mode (default 0.9).  Floor of 3 features always kept.
        seed: Retained for a consistent method interface.
    """

    def __init__(
        self,
        k: int = 10,
        return_as: str = "both",
        feature_weighting: str = "none",
        ks_threshold: float = 0.05,
        shift_ratio: float = 0.9,
        seed: int = 42,
    ):
        assert return_as in ("feature", "weight", "both"), \
            f"return_as must be 'feature', 'weight' or 'both', got '{return_as}'"
        assert feature_weighting in ("none", "ks", "filter", "cumshift"), \
            f"feature_weighting must be 'none', 'ks', 'filter' or 'cumshift', got '{feature_weighting}'"
        self.k = k
        self.return_as = return_as
        self.feature_weighting = feature_weighting
        self.ks_threshold = ks_threshold
        self.shift_ratio = shift_ratio
        self.seed = seed

    @property
    def name(self) -> str:
        parts = [f"nearest_test_k{self.k}"]
        if self.feature_weighting != "none":
            parts.append(f"fw_{self.feature_weighting}")
        parts.append(f"as_{self.return_as}")
        return "_".join(parts)

    @property
    def may_emit_sample_weight(self) -> bool:
        return self.return_as in ("weight", "both")

    def _compute_ks_weights(self, X_train: np.ndarray, X_test: np.ndarray):
        """Compute per-feature KS statistics for distance weighting."""
        from scipy.stats import ks_2samp

        n_features = X_train.shape[1]
        ks_stats = np.zeros(n_features)
        ks_pvals = np.zeros(n_features)

        for j in range(n_features):
            tr_col = X_train[:, j]
            te_col = X_test[:, j]
            # Remove NaN for KS test
            tr_valid = tr_col[~np.isnan(tr_col)]
            te_valid = te_col[~np.isnan(te_col)]
            if len(tr_valid) > 1 and len(te_valid) > 1:
                stat, pval = ks_2samp(tr_valid, te_valid)
                ks_stats[j] = stat
                ks_pvals[j] = pval
            else:
                ks_stats[j] = 0.0
                ks_pvals[j] = 1.0

        return ks_stats, ks_pvals

    def fit_transform(self, X_train, X_test, y_train):
        from src.methods.tier1 import _label_encode_cats
        from src.methods.preprocessing import get_cat_cols

        # The output matrix remains label-encoded, and the original feature columns
        # remain unchanged. LightGBM and bag8 can use label-encoded categories.
        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        feature_cols = list(X_tr_enc.columns)

        # The distance space is frequency-encoded instead. Label-encoding cats
        # (Germany=1, France=2, Spain=3) makes Euclidean distance meaningless.
        # Joint train+test frequency represents how common each category is.
        # This is suitable for a distribution-shift distance. M19, M20 and
        # JointTSNE use the same representation. _frequency_encode_joint retains
        # column order, so selected_features corresponds to feature_cols.
        # On all-numeric data (cat_cols == []) this is a no-op: freq-encode and
        # label-encode produce the same float array, so the distance
        # features are unchanged.
        cat_cols = get_cat_cols(X_train)
        X_tr_dist_raw, X_te_dist_raw = _frequency_encode_joint(
            X_train, X_test, cat_cols
        )

        # Impute NaN with joint train+test per-feature median before distance
        # computation. (Freq-encoded cat cols carry no NaN by construction.)
        X_tr_clean, X_te_clean = _impute_median_joint(X_tr_dist_raw, X_te_dist_raw)

        # Standardize before distance computation
        scaler = StandardScaler()
        combined = np.vstack([X_tr_clean, X_te_clean])
        scaler.fit(combined)
        X_tr_scaled = scaler.transform(X_tr_clean)
        X_te_scaled = scaler.transform(X_te_clean)

        # Apply feature weighting / filtering for distance computation
        n_dist_features = X_tr_scaled.shape[1]
        dist_feature_info = {}

        if self.feature_weighting in ("ks", "filter", "cumshift"):
            ks_stats, ks_pvals = self._compute_ks_weights(X_tr_clean, X_te_clean)

            if self.feature_weighting == "filter":
                # Only keep features with significant shift
                mask = ks_pvals < self.ks_threshold
                if mask.sum() == 0:
                    mask = np.ones(X_tr_scaled.shape[1], dtype=bool)
                X_tr_dist = X_tr_scaled[:, mask]
                X_te_dist = X_te_scaled[:, mask]
                n_dist_features = int(mask.sum())

            elif self.feature_weighting == "cumshift":
                # Dynamic selection: fewest features capturing shift_ratio
                # of total distributional shift (measured by KS statistics)
                n_f = X_tr_scaled.shape[1]
                order = np.argsort(ks_stats)[::-1]
                cumsum = np.cumsum(ks_stats[order])
                total = cumsum[-1]

                if total < 1e-6:
                    # No measurable shift → use all features
                    selected_idx = np.arange(n_f)
                else:
                    n_keep = int(np.searchsorted(cumsum, total * self.shift_ratio)) + 1
                    n_keep = max(n_keep, 3)       # floor: avoid degenerate space
                    n_keep = min(n_keep, n_f)     # cap
                    selected_idx = order[:n_keep]

                X_tr_dist = X_tr_scaled[:, selected_idx]
                X_te_dist = X_te_scaled[:, selected_idx]
                n_dist_features = len(selected_idx)
                achieved = float(cumsum[n_dist_features - 1] / total) if total > 1e-6 else 0.0
                dist_feature_info = {
                    "shift_ratio_target": self.shift_ratio,
                    "shift_ratio_achieved": achieved,
                    "selected_features": [feature_cols[j] for j in selected_idx],
                }

            else:  # "ks". Soft weighting
                w = ks_stats.copy()
                if w.sum() > 0:
                    w = w / w.sum() * len(w)
                else:
                    w = np.ones(len(w))
                X_tr_dist = X_tr_scaled * np.sqrt(w)
                X_te_dist = X_te_scaled * np.sqrt(w)
        else:
            X_tr_dist = X_tr_scaled
            X_te_dist = X_te_scaled

        # Effective k (cannot exceed dataset sizes)
        k_train = min(self.k, len(X_te_dist) - 1, len(X_tr_dist) - 1)
        k_train = max(k_train, 1)

        # --- Train → Test distances ---
        nn_test = NearestNeighbors(n_neighbors=k_train, algorithm="auto", n_jobs=1)
        nn_test.fit(X_te_dist)
        dist_tr2te, _ = nn_test.kneighbors(X_tr_dist)  # (n_train, k)

        mean_dist_tr2te = dist_tr2te.mean(axis=1)  # (n_train,)
        min_dist_tr2te = dist_tr2te[:, 0]           # (n_train,)

        # --- Test → Train distances ---
        nn_train = NearestNeighbors(n_neighbors=k_train, algorithm="auto", n_jobs=1)
        nn_train.fit(X_tr_dist)
        dist_te2tr, _ = nn_train.kneighbors(X_te_dist)  # (n_test, k)

        mean_dist_te2tr = dist_te2tr.mean(axis=1)  # (n_test,)
        min_dist_te2tr = dist_te2tr[:, 0]           # (n_test,)

        # --- Build output ---
        feature_names = list(feature_cols)
        X_tr_out = X_tr_arr.copy()
        X_te_out = X_te_arr.copy()

        sample_weights = None

        if self.return_as in ("feature", "both"):
            # Compute self-distribution density (dist to own k-NN)
            k_self = min(self.k, len(X_tr_dist) - 1, len(X_te_dist) - 1)
            k_self = max(k_self, 1)

            # Train self-density
            nn_train_self = NearestNeighbors(
                n_neighbors=k_self + 1, algorithm="auto", n_jobs=1
            )
            nn_train_self.fit(X_tr_dist)
            dist_tr_self, _ = nn_train_self.kneighbors(X_tr_dist)
            # Skip the first neighbor (self, distance=0)
            mean_dist_tr_self = dist_tr_self[:, 1:].mean(axis=1)

            # Test self-density
            nn_test_self = NearestNeighbors(
                n_neighbors=min(k_self + 1, len(X_te_dist)),
                algorithm="auto", n_jobs=1,
            )
            nn_test_self.fit(X_te_dist)
            dist_te_self, _ = nn_test_self.kneighbors(X_te_dist)
            mean_dist_te_self = dist_te_self[:, 1:].mean(axis=1)

            # Nearest training support. Same semantics for both splits:
            #   Train: LOO nearest train neighbor (exclude self)
            #   Test:  nearest train neighbor (direct)
            # Gives the GBDT an explicit extrapolation signal.
            nearest_train_support_tr = dist_tr_self[:, 1]  # nearest non-self train point
            nearest_train_support_te = min_dist_te2tr       # nearest train point

            X_tr_out = np.column_stack([
                X_tr_out,
                mean_dist_tr2te,              # dist to other distribution
                min_dist_tr2te,               # min dist to other distribution
                mean_dist_tr_self,            # local density (own distribution)
                nearest_train_support_tr,     # extrapolation signal
            ])
            X_te_out = np.column_stack([
                X_te_out,
                mean_dist_te2tr,              # dist to other distribution
                min_dist_te2tr,               # min dist to other distribution
                mean_dist_te_self,            # local density (own distribution)
                nearest_train_support_te,     # extrapolation signal
            ])
            feature_names = feature_names + [
                "mean_dist_to_other", "min_dist_to_other",
                "mean_dist_self_density", "nearest_train_dist",
            ]

        if self.return_as in ("weight", "both"):
            # Higher weight for train samples close to test distribution
            sample_weights = 1.0 / (1.0 + mean_dist_tr2te)

        metadata = {
            "k": k_train,
            "feature_weighting": self.feature_weighting,
            "n_features_for_distance": n_dist_features,
            "n_features_total": X_tr_arr.shape[1],
            "mean_train_to_test_dist": float(mean_dist_tr2te.mean()),
            "mean_test_to_train_dist": float(mean_dist_te2tr.mean()),
            "m16_distance_encoding": "frequency",
            "n_cat_cols_encoded": len(cat_cols),
        }
        metadata.update(dist_feature_info)

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            sample_weights_train=sample_weights,
            metadata=metadata,
        )


# ===========================================================================
# M16v2: Nearest-Test-Neighbor with Relative Distances
# ===========================================================================


class NearestTestDistanceRelative(TestTimeMethod):
    """
    M16v2: Nearest-test-neighbor distance features using *relative* distances.

    The original ``NearestTestDistance`` (M16)
    produces absolute Euclidean distances (mean_dist_to_other,
    min_dist_to_other, mean_dist_self_density).  These depend on the number
    of features and their scale, making them non-comparable across datasets
    or even across different subsets of the same dataset.

    This version computes dimensionless relative distances:

    - ``relative_dist = dist_to_other / (dist_to_self + eps)``

      A value of ~1 means the sample is equidistant to both distributions.
      A value >> 1 means it is far from the other distribution relative to
      its own local density.  A value << 1 means it is an outlier in its
      own distribution but close to the other.

    Features produced (when return_as includes "feature"):
      - rel_dist_to_other:     dimensionless relative distance ratio
      - log_rel_dist_to_other: log1p transform that reduces the heavy right tail
      - mean_dist_self_density: absolute self-density for reference
      - nearest_train_dist:    distance to nearest training point (LOO for
            train, direct for test): extrapolation signal

    Both relative features are scale-free, interpretable and much more
    stable than raw Euclidean distances as input to downstream GBDT models.

    Args:
        k: Number of nearest neighbors (default 10).
        return_as: "feature", "weight" or "both".
        feature_weighting: How to weight/select features for distance computation.
            "none": unweighted Euclidean distance on all features
            "ks": soft-weight each feature by its KS statistic
            "filter": only use features with KS p-value < ks_threshold
            "cumshift": keep the fewest features capturing ``shift_ratio``
                         of total KS shift (adaptive dimensionality reduction)
        ks_threshold: p-value threshold for "filter" mode (default 0.05).
        shift_ratio: Fraction of total KS shift to capture in "cumshift"
            mode (default 0.9).  Floor of 3 features always kept.
        seed: Retained for a consistent method interface.
    """

    def __init__(
        self,
        k: int = 10,
        return_as: str = "both",
        feature_weighting: str = "none",
        ks_threshold: float = 0.05,
        shift_ratio: float = 0.9,
        seed: int = 42,
    ):
        assert return_as in ("feature", "weight", "both")
        assert feature_weighting in ("none", "ks", "filter", "cumshift")
        self.k = k
        self.return_as = return_as
        self.feature_weighting = feature_weighting
        self.ks_threshold = ks_threshold
        self.shift_ratio = shift_ratio
        self.seed = seed

    @property
    def name(self) -> str:
        parts = [f"nearest_test_rel_k{self.k}"]
        if self.feature_weighting != "none":
            parts.append(f"fw_{self.feature_weighting}")
        parts.append(f"as_{self.return_as}")
        return "_".join(parts)

    @property
    def may_emit_sample_weight(self) -> bool:
        return self.return_as in ("weight", "both")

    def _compute_ks_weights(self, X_train: np.ndarray, X_test: np.ndarray):
        """Compute per-feature KS statistics for distance weighting."""
        from scipy.stats import ks_2samp

        n_features = X_train.shape[1]
        ks_stats = np.zeros(n_features)
        ks_pvals = np.zeros(n_features)

        for j in range(n_features):
            tr_col = X_train[:, j]
            te_col = X_test[:, j]
            tr_valid = tr_col[~np.isnan(tr_col)]
            te_valid = te_col[~np.isnan(te_col)]
            if len(tr_valid) > 1 and len(te_valid) > 1:
                stat, pval = ks_2samp(tr_valid, te_valid)
                ks_stats[j] = stat
                ks_pvals[j] = pval
            else:
                ks_stats[j] = 0.0
                ks_pvals[j] = 1.0

        return ks_stats, ks_pvals

    def fit_transform(self, X_train, X_test, y_train):
        from src.methods.tier1 import _label_encode_cats
        from src.methods.preprocessing import get_cat_cols

        # The output matrix remains label-encoded, and the original feature columns
        # remain unchanged. LightGBM can use label-encoded categories.
        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        feature_cols = list(X_tr_enc.columns)

        # The distance space is frequency-encoded instead. Label-encoding cats
        # makes Euclidean distance meaningless (arbitrary codes). Joint
        # train+test frequency gives each category a meaningful number (rare vs.
        # common). Mirrors M19/M20/JointTSNE. _frequency_encode_joint preserves
        # column order, so cumshift's selected_features metadata stays aligned
        # with feature_cols. On all-numeric data (cat_cols == []) this is a
        # no-op: freq-encode and label-encode produce the same array.
        cat_cols = get_cat_cols(X_train)
        X_tr_dist_raw, X_te_dist_raw = _frequency_encode_joint(
            X_train, X_test, cat_cols
        )

        # Impute NaN with joint train+test per-feature median before distance
        # computation. (Freq-encoded cat cols carry no NaN by construction.)
        X_tr_clean, X_te_clean = _impute_median_joint(X_tr_dist_raw, X_te_dist_raw)

        # Standardize
        scaler = StandardScaler()
        combined = np.vstack([X_tr_clean, X_te_clean])
        scaler.fit(combined)
        X_tr_s = scaler.transform(X_tr_clean)
        X_te_s = scaler.transform(X_te_clean)

        # Feature weighting / filtering for distance computation
        n_dist_features = X_tr_s.shape[1]
        dist_feature_info = {}

        if self.feature_weighting in ("ks", "filter", "cumshift"):
            ks_stats, ks_pvals = self._compute_ks_weights(X_tr_clean, X_te_clean)

            if self.feature_weighting == "filter":
                mask = ks_pvals < self.ks_threshold
                if mask.sum() == 0:
                    mask = np.ones(X_tr_s.shape[1], dtype=bool)
                X_tr_d = X_tr_s[:, mask]
                X_te_d = X_te_s[:, mask]
                n_dist_features = int(mask.sum())

            elif self.feature_weighting == "cumshift":
                # Dynamic selection: fewest features capturing shift_ratio
                # of total distributional shift (measured by KS statistics)
                n_f = X_tr_s.shape[1]
                order = np.argsort(ks_stats)[::-1]
                cumsum = np.cumsum(ks_stats[order])
                total = cumsum[-1]

                if total < 1e-6:
                    selected_idx = np.arange(n_f)
                else:
                    n_keep = int(np.searchsorted(cumsum, total * self.shift_ratio)) + 1
                    n_keep = max(n_keep, 3)
                    n_keep = min(n_keep, n_f)
                    selected_idx = order[:n_keep]

                X_tr_d = X_tr_s[:, selected_idx]
                X_te_d = X_te_s[:, selected_idx]
                n_dist_features = len(selected_idx)
                achieved = float(cumsum[n_dist_features - 1] / total) if total > 1e-6 else 0.0
                dist_feature_info = {
                    "shift_ratio_target": self.shift_ratio,
                    "shift_ratio_achieved": achieved,
                    "selected_features": [feature_cols[j] for j in selected_idx],
                }

            else:  # "ks". Soft weighting
                w = ks_stats.copy()
                if w.sum() > 0:
                    w = w / w.sum() * len(w)
                else:
                    w = np.ones(len(w))
                X_tr_d = X_tr_s * np.sqrt(w)
                X_te_d = X_te_s * np.sqrt(w)
        else:
            X_tr_d = X_tr_s
            X_te_d = X_te_s

        eps = 1e-8
        k_eff = min(self.k, len(X_te_d) - 1, len(X_tr_d) - 1)
        k_eff = max(k_eff, 1)

        # --- Cross-distribution distances ---
        nn_test = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=1)
        nn_test.fit(X_te_d)
        dist_tr2te, _ = nn_test.kneighbors(X_tr_d)
        mean_dist_tr2te = dist_tr2te.mean(axis=1)

        nn_train = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=1)
        nn_train.fit(X_tr_d)
        dist_te2tr, _ = nn_train.kneighbors(X_te_d)
        mean_dist_te2tr = dist_te2tr.mean(axis=1)

        # --- Self-distribution distances ---
        k_self = min(self.k + 1, len(X_tr_d))
        nn_tr_self = NearestNeighbors(n_neighbors=k_self, algorithm="auto", n_jobs=1)
        nn_tr_self.fit(X_tr_d)
        dist_tr_self, _ = nn_tr_self.kneighbors(X_tr_d)
        mean_dist_tr_self = dist_tr_self[:, 1:].mean(axis=1)  # skip self

        k_self_te = min(self.k + 1, len(X_te_d))
        nn_te_self = NearestNeighbors(n_neighbors=k_self_te, algorithm="auto", n_jobs=1)
        nn_te_self.fit(X_te_d)
        dist_te_self, _ = nn_te_self.kneighbors(X_te_d)
        mean_dist_te_self = dist_te_self[:, 1:].mean(axis=1)  # skip self

        # --- Relative features ---
        rel_dist_train = mean_dist_tr2te / (mean_dist_tr_self + eps)
        rel_dist_test = mean_dist_te2tr / (mean_dist_te_self + eps)

        # Log-transform for better GBDT splits (relative dists can have
        # heavy right tail)
        log_rel_train = np.log1p(rel_dist_train)
        log_rel_test = np.log1p(rel_dist_test)

        feature_names = list(feature_cols)
        X_tr_out = X_tr_arr.copy()
        X_te_out = X_te_arr.copy()
        sample_weights = None

        if self.return_as in ("feature", "both"):
            # Nearest training support. Same semantics for both splits:
            #   Train: LOO nearest train neighbor (exclude self)
            #   Test:  nearest train neighbor (direct)
            nearest_train_support_tr = dist_tr_self[:, 1]
            nearest_train_support_te = dist_te2tr[:, 0]

            X_tr_out = np.column_stack([
                X_tr_out,
                rel_dist_train,
                log_rel_train,
                mean_dist_tr_self,
                nearest_train_support_tr,
            ])
            X_te_out = np.column_stack([
                X_te_out,
                rel_dist_test,
                log_rel_test,
                mean_dist_te_self,
                nearest_train_support_te,
            ])
            feature_names += [
                "rel_dist_to_other",
                "log_rel_dist_to_other",
                "mean_dist_self_density",
                "nearest_train_dist",
            ]

        if self.return_as in ("weight", "both"):
            # Weight: inverse of relative distance. Train samples that are
            # "close to test relative to their own density" get high weight
            sample_weights = 1.0 / (1.0 + rel_dist_train)

        metadata = {
            "k": k_eff,
            "feature_weighting": self.feature_weighting,
            "n_features_for_distance": n_dist_features,
            "n_features_total": X_tr_arr.shape[1],
            "mean_rel_dist_train": float(rel_dist_train.mean()),
            "mean_rel_dist_test": float(rel_dist_test.mean()),
            "m16_distance_encoding": "frequency",
            "n_cat_cols_encoded": len(cat_cols),
        }
        metadata.update(dist_feature_info)

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            sample_weights_train=sample_weights,
            metadata=metadata,
        )


# ===========================================================================
# M8: Label Propagation as Feature (OOF)
# ===========================================================================

class LabelPropagationFeature(TestTimeMethod):
    """
    M8: Use label propagation on the train+test k-NN graph to produce
    soft labels, then append them as a feature.

    The OOF trick avoids leakage: only train samples from *other* folds
    have visible labels. The current fold's train samples and all test
    samples receive propagated soft labels.  Test soft labels are averaged
    across all folds (ensembled).

    Args:
        k: Number of neighbors for the k-NN graph (default 15).
        n_folds: Number of OOF folds (default 5).
        alpha: Damping factor: fraction of label retained from iteration
            to iteration (default 0.5). Higher α assigns more weight to the
            original known labels than to propagated values.
        max_iter: Maximum propagation iterations (default 100).
        seed: Random seed.
    """

    def __init__(
        self,
        k: int = 15,
        n_folds: int = 5,
        alpha: float = 0.5,
        max_iter: int = 100,
        seed: int = 42,
    ):
        self.k = k
        self.n_folds = n_folds
        self.alpha = alpha
        self.max_iter = max_iter
        self.seed = seed

    @property
    def name(self) -> str:
        return f"label_prop_k{self.k}_a{self.alpha}"

    @property
    def uses_labels(self) -> bool:
        return True

    @staticmethod
    def _fallback_is_classification(y_train: np.ndarray) -> bool:
        """Compatibility fallback for direct calls without MethodContext."""
        return len(np.unique(y_train)) <= 20

    def _resolve_task_type(self, y_train: np.ndarray) -> tuple[bool, str]:
        if self.method_context is not None:
            return self.method_context.task_type == "classification", "context"
        return self._fallback_is_classification(y_train), "fallback_target_heuristic"

    def _propagate(
        self,
        W,
        labels: np.ndarray,
        known_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Run label propagation on a weight matrix.

        The clamping update for known nodes is:

            f[known] = α · y_fixed[known] + (1 - α) · (W·f)[known]

        Alpha convention: α controls *clamping strength*, which determines how
        strongly the known labels are retained during propagation. This is the
        reverse of the convention in Zhou et al. (2003), where α controls
        propagation strength. An α of 0.5 is equivalent under both conventions.
        Here, an α of 0.8 means strong clamping, while Zhou's α=0.8 means strong
        propagation.

        Args:
            W: (N, N) row-normalized weight matrix (sparse or dense).
            labels: (N,) initial label vector. Known labels are set.
                unknown labels are initialized to the global mean.
            known_mask: (N,) boolean: True for nodes with known labels.

        Returns:
            Tuple of (propagated_labels, converged, n_iterations):
            - propagated_labels: (N,) soft labels after propagation.
            - converged: bool: True if ||f_new - f||_∞ < 1e-4 before max_iter.
            - n_iterations: int: number of iterations performed.
        """
        f = labels.copy()
        y_fixed = labels.copy()
        converged = False
        n_iterations = 0

        for it in range(self.max_iter):
            # Propagation step: each node takes weighted average of neighbors
            f_new = W @ f
            # Ensure dense array for indexing (sparse @ dense → dense)
            if hasattr(f_new, 'toarray'):
                f_new = np.asarray(f_new).ravel()
            # Clamp known labels back (with damping)
            f_new[known_mask] = (
                self.alpha * y_fixed[known_mask]
                + (1 - self.alpha) * f_new[known_mask]
            )
            n_iterations = it + 1
            # Check convergence
            if np.abs(f_new - f).max() < 1e-4:
                converged = True
                break
            f = f_new

        return f, converged, n_iterations

    def fit_transform(self, X_train, X_test, y_train):
        from src.methods.tier1 import _label_encode_cats

        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        y_tr = np.asarray(y_train).astype(float)

        n_train = len(X_tr_arr)
        n_test = len(X_te_arr)

        # Standardize before building the graph
        X_combined_raw = np.vstack([
            np.nan_to_num(X_tr_arr, nan=0.0),
            np.nan_to_num(X_te_arr, nan=0.0),
        ])
        scaler = StandardScaler()
        X_combined = scaler.fit_transform(X_combined_raw)

        # Build k-NN graph once (shared across folds)
        k_eff = min(self.k, len(X_combined) - 1)
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=1)
        nn.fit(X_combined)
        distances, indices = nn.kneighbors(X_combined)

        # Build row-normalized weight matrix (sparse for O(N*k) memory)
        # Using RBF kernel weights: w_ij = exp(-d_ij^2 / sigma^2)
        from scipy import sparse as sp

        sigma = np.median(distances[:, -1]) + 1e-8  # Median of k-th neighbor distances
        N = len(X_combined)

        # Build sparse matrix from k-NN graph
        rows = np.repeat(np.arange(N), k_eff)
        cols = indices.ravel()
        vals = np.exp(-(distances.ravel() ** 2) / (sigma ** 2))
        W = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        # Make symmetric
        W = (W + W.T) / 2.0
        # Row-normalize
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        W = sp.diags(1.0 / row_sums) @ W

        # Global mean for initialization of unknown labels
        global_mean = y_tr.mean()

        # OOF procedure: split only training indices
        soft_labels_train = np.zeros(n_train)
        soft_labels_test = np.zeros(n_test)
        fold_convergence: list[dict] = []

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(
            kf.split(np.arange(n_train))
        ):
            # Initialize labels
            labels = np.full(N, global_mean)
            known_mask = np.zeros(N, dtype=bool)

            # Set known labels using training samples from the other folds.
            labels[fold_train_idx] = y_tr[fold_train_idx]
            known_mask[fold_train_idx] = True
            # fold_val_idx train samples and all test samples: unknown

            propagated, conv, n_it = self._propagate(W, labels, known_mask)
            fold_convergence.append({
                "fold": fold_idx,
                "converged": conv,
                "n_iterations": n_it,
            })

            # Collect OOF predictions for held-out train samples
            soft_labels_train[fold_val_idx] = propagated[fold_val_idx]
            # Ensemble test predictions
            soft_labels_test += propagated[n_train:] / self.n_folds

        # Append soft labels as features
        X_tr_out = np.column_stack([X_tr_arr, soft_labels_train])
        X_te_out = np.column_stack([X_te_arr, soft_labels_test])
        feature_names = list(X_tr_enc.columns) + ["label_prop_soft"]

        metadata = {
            "mean_soft_train": float(soft_labels_train.mean()),
            "std_soft_train": float(soft_labels_train.std()),
            "mean_soft_test": float(soft_labels_test.mean()),
            "correlation_soft_y": float(np.corrcoef(soft_labels_train, y_tr)[0, 1])
            if len(np.unique(y_tr)) > 1 else 0.0,
            "fold_convergence": fold_convergence,
            "m8_mean_iterations": float(
                np.mean([fc["n_iterations"] for fc in fold_convergence])
            ),
        }

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            metadata=metadata,
        )


# ===========================================================================
# M8v2: Feature-Weighted Label Propagation
# ===========================================================================

class LabelPropagationWeighted(LabelPropagationFeature):
    """
    M8v2: Label propagation with feature-weighted k-NN graph.

    The original M8 builds the k-NN graph on all
    features with equal weight.  If 90% of features are irrelevant to the
    label, nearest neighbors in full feature space may have very different
    labels, producing useless soft labels.

    Features in the distance calculation are weighted by
    ``shift_score × importance_score``:
    - shift_score: KS statistic for the magnitude of feature shift
    - importance_score: LightGBM feature importance for label prediction
    - product: assigns high weight only when a feature both shifted and matters
      for prediction. A feature that satisfies only one condition receives low
      weight.

    Also supports adaptive k based on dataset size.

    Inherits the ``_propagate`` method and OOF logic from
    ``LabelPropagationFeature``: only the graph construction differs.

    Args:
        k: Number of neighbors (default 15 or "auto" for adaptive).
        n_folds: OOF folds (default 5).
        alpha: Clamping strength (default 0.5).
        max_iter: Max propagation iterations (default 100).
        adaptive_k: If True, set k = clip(sqrt(n_total), 5, 50).
        importance_classifier: Model for computing feature importance
            ("lgbm" or None for equal importance).
        seed: Random seed.
    """

    def __init__(
        self,
        k: int = 15,
        n_folds: int = 5,
        alpha: float = 0.5,
        max_iter: int = 100,
        adaptive_k: bool = False,
        importance_classifier: str | None = "lgbm",
        seed: int = 42,
    ):
        super().__init__(k=k, n_folds=n_folds, alpha=alpha, max_iter=max_iter, seed=seed)
        self.adaptive_k = adaptive_k
        self.importance_classifier = importance_classifier

    @property
    def name(self) -> str:
        parts = [f"label_prop_weighted_k{self.k}_a{self.alpha}"]
        if self.adaptive_k:
            parts.append("adak")
        if self.importance_classifier:
            parts.append(f"imp_{self.importance_classifier}")
        return "_".join(parts)

    def _compute_feature_weights(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-feature weights as shift_score × importance_score.

        Returns:
            weights: (n_features,) array of non-negative weights.
        """
        from scipy.stats import ks_2samp

        n_features = X_train.shape[1]

        # --- Shift scores (KS statistic per feature) ---
        ks_stats = np.zeros(n_features)
        for j in range(n_features):
            tr_col = X_train[:, j]
            te_col = X_test[:, j]
            tr_valid = tr_col[~np.isnan(tr_col)]
            te_valid = te_col[~np.isnan(te_col)]
            if len(tr_valid) > 1 and len(te_valid) > 1:
                stat, _ = ks_2samp(tr_valid, te_valid)
                ks_stats[j] = stat

        # --- Importance scores ---
        if self.importance_classifier == "lgbm":
            try:
                from lightgbm import LGBMClassifier, LGBMRegressor

                X_clean = np.nan_to_num(X_train, nan=0.0)
                y_clean = y_train.copy()

                is_classification, _ = self._resolve_task_type(y_clean)
                if is_classification:
                    # Classification
                    model = LGBMClassifier(
                        n_estimators=50, max_depth=4,
                        random_state=self.seed, verbose=-1, n_jobs=1,
                    )
                else:
                    # Regression
                    model = LGBMRegressor(
                        n_estimators=50, max_depth=4,
                        random_state=self.seed, verbose=-1, n_jobs=1,
                    )
                model.fit(X_clean, y_clean)
                importances = model.feature_importances_.astype(float)
                # Normalize to [0, 1]
                if importances.max() > 0:
                    importances = importances / importances.max()
                else:
                    importances = np.ones(n_features)
            except Exception:
                # Fallback: equal importance
                importances = np.ones(n_features)
        else:
            importances = np.ones(n_features)

        # --- Combined weights: shift × importance ---
        weights = ks_stats * importances

        # Avoid all-zero weights (fallback to uniform)
        if weights.sum() < 1e-10:
            weights = np.ones(n_features)

        return weights

    def fit_transform(self, X_train, X_test, y_train):
        from src.methods.tier1 import _label_encode_cats
        from scipy import sparse as sp

        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        y_tr = np.asarray(y_train).astype(float)

        n_train = len(X_tr_arr)
        n_test = len(X_te_arr)

        # --- Compute feature weights ---
        X_tr_clean = np.nan_to_num(X_tr_arr, nan=0.0)
        X_te_clean = np.nan_to_num(X_te_arr, nan=0.0)
        feature_weights = self._compute_feature_weights(X_tr_clean, X_te_clean, y_tr)

        # --- Standardize, then apply feature weights ---
        X_combined_raw = np.vstack([X_tr_clean, X_te_clean])
        scaler = StandardScaler()
        X_combined = scaler.fit_transform(X_combined_raw)

        # Weight features: multiply by sqrt(w_j) so that
        # ||x - y||^2 = sum(w_j * (x_j - y_j)^2)
        sqrt_weights = np.sqrt(feature_weights)
        # Normalize so total weight = n_features (preserve distance scale)
        if sqrt_weights.sum() > 0:
            sqrt_weights = sqrt_weights / sqrt_weights.sum() * len(sqrt_weights)
        X_combined_weighted = X_combined * sqrt_weights

        # --- Adaptive k ---
        if self.adaptive_k:
            k_eff = int(np.clip(np.sqrt(len(X_combined)), 5, 50))
        else:
            k_eff = self.k
        k_eff = min(k_eff, len(X_combined) - 1)

        # --- Build k-NN graph on weighted features ---
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=1)
        nn.fit(X_combined_weighted)
        distances, indices = nn.kneighbors(X_combined_weighted)

        # Build sparse weight matrix (same as parent class)
        sigma = np.median(distances[:, -1]) + 1e-8
        N = len(X_combined)
        rows = np.repeat(np.arange(N), k_eff)
        cols = indices.ravel()
        vals = np.exp(-(distances.ravel() ** 2) / (sigma ** 2))
        W = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        W = (W + W.T) / 2.0
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        W = sp.diags(1.0 / row_sums) @ W

        # --- OOF label propagation (same logic as parent) ---
        global_mean = y_tr.mean()
        soft_labels_train = np.zeros(n_train)
        soft_labels_test = np.zeros(n_test)
        fold_convergence: list[dict] = []

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(
            kf.split(np.arange(n_train))
        ):
            labels = np.full(N, global_mean)
            known_mask = np.zeros(N, dtype=bool)
            labels[fold_train_idx] = y_tr[fold_train_idx]
            known_mask[fold_train_idx] = True

            propagated, conv, n_it = self._propagate(W, labels, known_mask)
            fold_convergence.append({
                "fold": fold_idx,
                "converged": conv,
                "n_iterations": n_it,
            })

            soft_labels_train[fold_val_idx] = propagated[fold_val_idx]
            soft_labels_test += propagated[n_train:] / self.n_folds

        # Build output
        X_tr_out = np.column_stack([X_tr_arr, soft_labels_train])
        X_te_out = np.column_stack([X_te_arr, soft_labels_test])
        feature_names = list(X_tr_enc.columns) + ["label_prop_weighted_soft"]

        # Count how many features have significant weight
        n_significant = int((feature_weights > 0.01).sum())
        is_classification, task_type_source = self._resolve_task_type(y_tr)

        metadata = {
            "mean_soft_train": float(soft_labels_train.mean()),
            "std_soft_train": float(soft_labels_train.std()),
            "mean_soft_test": float(soft_labels_test.mean()),
            "correlation_soft_y": float(np.corrcoef(soft_labels_train, y_tr)[0, 1])
            if len(np.unique(y_tr)) > 1 else 0.0,
            "k_effective": k_eff,
            "n_significant_features": n_significant,
            "n_features_total": len(feature_weights),
            "m8_is_classification": is_classification,
            "m8_task_type_source": task_type_source,
            "fold_convergence": fold_convergence,
        }

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            metadata=metadata,
        )


# ===========================================================================
# M8v3: Enhanced Label Propagation (Hard Selection + Multi-Feature)
# ===========================================================================

class LabelPropagationEnhanced(LabelPropagationWeighted):
    """
    M8v3: Enhanced label propagation with hard feature selection,
    multi-feature output and graph-derived proximity features.

    Mitigates degradation of k-nearest-neighbor graph quality in
    high-dimensional spaces, where distances can become less discriminative.
    It extends M8v2 with:

    1. **Hard feature selection.**  Computes per-feature weight as
       ``KS_shift_score × LightGBM_importance`` (same as M8v2), but applies
       a hard cutoff: only features with weight >= ``threshold × max_weight``
       are used for graph construction.  A floor of ``min_features`` prevents
       degenerate graphs. Original features are preserved in the output:
       only the kNN graph is built on the selected subset.

    2. **Multi-feature output.**  Instead of a single soft-label scalar:
       - Binary / regression: 1 soft label + 2 graph features = 3 new features
       - Multi-class (K classes): K soft probabilities + 2 graph features

    3. **Graph-derived proximity features** (essentially free from the kNN):
       - ``lp_frac_test_neighbors``: fraction of each sample's k neighbors
         that are test points.  Non-parametric train-test proximity measure
         (cf. M6's parametric domain classifier, M16's distance-based).
       - ``lp_mean_neighbor_dist``: mean distance to k neighbors.  Local
         density proxy: samples in sparse regions have larger values.

    4. **Per-class soft probabilities** for multi-class tasks.  M8/M8v2
       propagate the raw class index (0,1,2,...) as a single float, which
       is meaningless for nominal classes.  M8v3 propagates K one-hot
       indicator vectors and normalizes to proper probabilities.

    Inherits ``_compute_feature_weights`` from M8v2 and ``_propagate``
    from M8.

    Args:
        k: Number of neighbors (default 15).
        n_folds: OOF folds (default 5).
        alpha: Clamping strength (default 0.5).
        max_iter: Max propagation iterations (default 100).
        adaptive_k: If True, set k = clip(sqrt(n_total), 5, 50).
        feature_threshold: Hard cutoff for feature selection.  Keep features
            where ``weight >= threshold * max_weight``.
        min_features: Minimum features to keep after hard cutoff (default 3).
            If fewer meet the threshold, the top-``min_features`` by weight
            are kept instead.
        seed: Random seed.
    """

    def __init__(
        self,
        k: int = 15,
        n_folds: int = 5,
        alpha: float = 0.5,
        max_iter: int = 100,
        adaptive_k: bool = False,
        feature_threshold: float = 0.05,
        min_features: int = 3,
        seed: int = 42,
    ):
        super().__init__(
            k=k, n_folds=n_folds, alpha=alpha, max_iter=max_iter,
            adaptive_k=adaptive_k, importance_classifier="lgbm", seed=seed,
        )
        self.feature_threshold = feature_threshold
        self.min_features = min_features

    @property
    def name(self) -> str:
        parts = ["label_prop_v3"]
        parts.append(f"t{self.feature_threshold}")
        if self.adaptive_k:
            parts.append("adak")
        else:
            parts.append(f"k{self.k}")
        parts.append(f"a{self.alpha}")
        return "_".join(parts)

    def fit_transform(self, X_train, X_test, y_train):
        from src.methods.tier1 import _label_encode_cats
        from scipy import sparse as sp

        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        y_tr = np.asarray(y_train).astype(float)

        n_train = len(X_tr_arr)
        n_test = len(X_te_arr)
        n_features_orig = X_tr_arr.shape[1]

        # ==================================================================
        # 1. Compute feature weights (inherited from M8v2)
        # ==================================================================
        X_tr_clean = np.nan_to_num(X_tr_arr, nan=0.0)
        X_te_clean = np.nan_to_num(X_te_arr, nan=0.0)
        feature_weights = self._compute_feature_weights(
            X_tr_clean, X_te_clean, y_tr,
        )

        # ==================================================================
        # 2. Hard feature selection
        # ==================================================================
        max_w = feature_weights.max()
        if max_w > 0:
            keep_mask = feature_weights >= self.feature_threshold * max_w
        else:
            keep_mask = np.ones(n_features_orig, dtype=bool)

        # Retain at least min_features.
        if keep_mask.sum() < self.min_features:
            n_keep = min(self.min_features, n_features_orig)
            top_idx = np.argsort(feature_weights)[-n_keep:]
            keep_mask = np.zeros(n_features_orig, dtype=bool)
            keep_mask[top_idx] = True

        n_selected = int(keep_mask.sum())

        # ==================================================================
        # 3. Standardize selected features → build kNN graph
        # ==================================================================
        X_combined_raw = np.vstack([X_tr_clean, X_te_clean])
        scaler = StandardScaler()
        X_combined_std = scaler.fit_transform(X_combined_raw)
        # Graph built on selected features only. Output keeps all originals
        X_graph = X_combined_std[:, keep_mask]

        N = len(X_combined_raw)

        # Adaptive k
        if self.adaptive_k:
            k_eff = int(np.clip(np.sqrt(N), 5, 50))
        else:
            k_eff = self.k
        k_eff = min(k_eff, N - 1)

        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=1)
        nn.fit(X_graph)
        distances, indices = nn.kneighbors(X_graph)

        # ==================================================================
        # 4. Graph-derived features (extracted before weight matrix)
        # ==================================================================
        # frac_test_neighbors: non-parametric train-test proximity
        is_test_neighbor = indices >= n_train
        frac_test_all = is_test_neighbor.mean(axis=1).astype(float)
        frac_test_train = frac_test_all[:n_train]
        frac_test_test = frac_test_all[n_train:]

        # mean_neighbor_distance: local density proxy
        mean_dist_all = distances.mean(axis=1).astype(float)
        mean_dist_train = mean_dist_all[:n_train]
        mean_dist_test = mean_dist_all[n_train:]

        # ==================================================================
        # 5. Build sparse weight matrix (RBF kernel)
        # ==================================================================
        sigma = np.median(distances[:, -1]) + 1e-8
        rows = np.repeat(np.arange(N), k_eff)
        cols = indices.ravel()
        vals = np.exp(-(distances.ravel() ** 2) / (sigma ** 2))
        W = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
        W = (W + W.T) / 2.0
        row_sums_W = np.asarray(W.sum(axis=1)).ravel()
        row_sums_W[row_sums_W == 0] = 1.0
        W = sp.diags(1.0 / row_sums_W) @ W

        # ==================================================================
        # 6. Determine task type
        # ==================================================================
        classes = np.unique(y_tr)
        is_classification, task_type_source = self._resolve_task_type(y_tr)
        n_classes = len(classes) if is_classification else 1
        is_multiclass = is_classification and n_classes > 2

        # ==================================================================
        # 7. OOF label propagation
        # ==================================================================
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        fold_convergence: list[dict] = []

        if is_multiclass:
            # ----- Multi-class: propagate per-class one-hot indicators -----
            soft_train = np.zeros((n_train, n_classes))
            fold_results_test = np.zeros((self.n_folds, n_test, n_classes))

            for fold_idx, (fold_tr, fold_val) in enumerate(
                kf.split(np.arange(n_train))
            ):
                for c_idx, c in enumerate(classes):
                    # Initialize unknowns to class prior from this fold's
                    # training data (not the held-out fold)
                    class_prior = float((y_tr[fold_tr] == c).mean())
                    labels = np.full(N, class_prior)
                    known_mask = np.zeros(N, dtype=bool)
                    labels[fold_tr] = (y_tr[fold_tr] == c).astype(float)
                    known_mask[fold_tr] = True

                    propagated, conv, n_it = self._propagate(
                        W, labels, known_mask,
                    )

                    soft_train[fold_val, c_idx] = propagated[fold_val]
                    fold_results_test[fold_idx, :, c_idx] = propagated[n_train:]

                    # Log convergence once per fold (first class only)
                    if c_idx == 0:
                        fold_convergence.append({
                            "fold": fold_idx,
                            "converged": conv,
                            "n_iterations": n_it,
                        })

            # Average test soft labels across folds
            soft_test = fold_results_test.mean(axis=0)  # (n_test, n_classes)

            # Clip negative values (propagation can produce small negatives)
            soft_train = np.clip(soft_train, 0.0, None)
            soft_test = np.clip(soft_test, 0.0, None)

            # Normalize rows to proper probabilities (sum to 1)
            def _normalize_rows(arr):
                rs = arr.sum(axis=1, keepdims=True)
                rs[rs == 0] = 1.0
                return arr / rs

            soft_train = _normalize_rows(soft_train)
            soft_test = _normalize_rows(soft_test)

            soft_feature_names = [f"lp_soft_c{int(c)}" for c in classes]

        else:
            # ----- Binary classification or regression: single propagation -----
            global_mean = y_tr.mean()
            soft_labels_train = np.zeros(n_train)
            fold_results_test = np.zeros((self.n_folds, n_test))

            for fold_idx, (fold_tr, fold_val) in enumerate(
                kf.split(np.arange(n_train))
            ):
                labels = np.full(N, global_mean)
                known_mask = np.zeros(N, dtype=bool)
                labels[fold_tr] = y_tr[fold_tr]
                known_mask[fold_tr] = True

                propagated, conv, n_it = self._propagate(
                    W, labels, known_mask,
                )
                fold_convergence.append({
                    "fold": fold_idx,
                    "converged": conv,
                    "n_iterations": n_it,
                })

                soft_labels_train[fold_val] = propagated[fold_val]
                fold_results_test[fold_idx] = propagated[n_train:]

            # Average test soft labels across folds
            soft_labels_test = fold_results_test.mean(axis=0)

            # Reshape to 2D for uniform column_stack below
            soft_train = soft_labels_train.reshape(-1, 1)
            soft_test = soft_labels_test.reshape(-1, 1)
            soft_feature_names = ["lp_soft"]

        # ==================================================================
        # 8. Assemble output features
        # ==================================================================
        X_tr_out = np.column_stack([
            X_tr_arr,
            soft_train,
            frac_test_train.reshape(-1, 1),
            mean_dist_train.reshape(-1, 1),
        ])
        X_te_out = np.column_stack([
            X_te_arr,
            soft_test,
            frac_test_test.reshape(-1, 1),
            mean_dist_test.reshape(-1, 1),
        ])
        feature_names = (
            list(X_tr_enc.columns)
            + soft_feature_names
            + ["lp_frac_test_neighbors", "lp_mean_neighbor_dist"]
        )

        # ==================================================================
        # 9. Metadata
        # ==================================================================
        if not is_multiclass and len(np.unique(y_tr)) > 1:
            soft_flat = soft_train.ravel()
            if soft_flat.std() > 0:
                corr = float(np.corrcoef(soft_flat, y_tr)[0, 1])
            else:
                corr = 0.0
        else:
            corr = 0.0

        metadata = {
            "n_features_total": n_features_orig,
            "n_features_selected": n_selected,
            "features_selected_frac": round(n_selected / max(n_features_orig, 1), 3),
            "feature_threshold": self.feature_threshold,
            "k_effective": k_eff,
            "sigma": float(sigma),
            "n_classes": n_classes,
            "m8_is_classification": is_classification,
            "m8_task_type_source": task_type_source,
            "is_multiclass": is_multiclass,
            "n_output_features": len(soft_feature_names) + 2,
            "mean_soft_train": float(soft_train.mean()),
            "std_soft_train": float(soft_train.std()),
            "mean_soft_test": float(soft_test.mean()),
            "correlation_soft_y": corr,
            "mean_frac_test_train": float(frac_test_train.mean()),
            "mean_frac_test_test": float(frac_test_test.mean()),
            "mean_dist_train": float(mean_dist_train.mean()),
            "mean_dist_test": float(mean_dist_test.mean()),
            "fold_convergence": fold_convergence,
        }

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            metadata=metadata,
        )


# ===========================================================================
# M9: Optimal Transport Displacement Features
# ===========================================================================

class OTDisplacementFeatures(TestTimeMethod):
    """
    M9: Compute per-feature displacement vectors via univariate optimal
    transport (quantile matching) and append them as features.

    For each feature j and each sample i:
      - Find the quantile rank of x_ij within its source distribution.
      - Look up the value at that quantile in the other distribution.
      - displacement_ij = value_at_quantile - x_ij

    Also computes a scalar displacement magnitude (L2 norm across features).

    Each displacement records how far a feature value would move under the
    estimated transport from its source distribution to the other distribution.

    Args:
        append_magnitude: Whether to append the scalar displacement norm
            (default True).
        seed: Random seed (unused, for API consistency).
    """

    def __init__(self, append_magnitude: bool = True, seed: int = 42):
        self.append_magnitude = append_magnitude
        self.seed = seed

    @property
    def name(self) -> str:
        mag = "_mag" if self.append_magnitude else ""
        return f"ot_displacement{mag}"

    def fit_transform(self, X_train, X_test, y_train):
        from src.methods.tier1 import _label_encode_cats

        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)

        n_features = X_tr_arr.shape[1]
        disp_train = np.zeros_like(X_tr_arr)
        disp_test = np.zeros_like(X_te_arr)

        for j in range(n_features):
            tr_col = X_tr_arr[:, j]
            te_col = X_te_arr[:, j]

            # Handle NaN: use median fill for quantile computation
            tr_valid = tr_col.copy()
            te_valid = te_col.copy()
            tr_nan = np.isnan(tr_valid)
            te_nan = np.isnan(te_valid)
            if tr_nan.any():
                tr_valid[tr_nan] = np.nanmedian(tr_valid) if not np.all(tr_nan) else 0.0
            if te_nan.any():
                te_valid[te_nan] = np.nanmedian(te_valid) if not np.all(te_nan) else 0.0

            # --- Train displacement (train → test distribution) ---
            # Quantile rank of each train sample within train distribution
            from scipy.stats import rankdata
            tr_ranks = rankdata(tr_valid, method="average") / (len(tr_valid) + 1)
            # Value at those quantiles in the test distribution via interp
            te_sorted = np.sort(te_valid)
            te_quantiles = np.linspace(0, 1, len(te_sorted) + 2)[1:-1]  # exclude 0 and 1
            te_values_at_quantile = np.interp(tr_ranks, te_quantiles, te_sorted)
            disp_train[:, j] = te_values_at_quantile - tr_valid

            # --- Test displacement (test → train distribution, inverse) ---
            te_ranks = rankdata(te_valid, method="average") / (len(te_valid) + 1)
            tr_sorted = np.sort(tr_valid)
            tr_quantiles = np.linspace(0, 1, len(tr_sorted) + 2)[1:-1]
            tr_values_at_quantile = np.interp(te_ranks, tr_quantiles, tr_sorted)
            disp_test[:, j] = tr_values_at_quantile - te_valid

        # Displacement magnitude (L2 norm across features)
        mag_train = np.linalg.norm(disp_train, axis=1)
        mag_test = np.linalg.norm(disp_test, axis=1)

        # Build output: original features + displacement per feature + magnitude
        disp_names = [f"ot_disp_{c}" for c in X_tr_enc.columns]
        feature_names = list(X_tr_enc.columns) + disp_names

        X_tr_out = np.hstack([X_tr_arr, disp_train])
        X_te_out = np.hstack([X_te_arr, disp_test])

        if self.append_magnitude:
            X_tr_out = np.column_stack([X_tr_out, mag_train])
            X_te_out = np.column_stack([X_te_out, mag_test])
            feature_names = feature_names + ["ot_disp_magnitude"]

        metadata = {
            "mean_train_magnitude": float(mag_train.mean()),
            "mean_test_magnitude": float(mag_test.mean()),
            "max_train_magnitude": float(mag_train.max()),
        }

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            metadata=metadata,
        )


# ===========================================================================
# M9v2: Improved OT Displacement
# ===========================================================================

class OTDisplacementImproved(TestTimeMethod):
    """
    M9v2: OT displacement with five additional design choices:

    1. **Selective features**: Only computes displacements for features with
       significant distribution shift (KS test p < threshold).  Non-shifted
       features contribute near-zero displacements = pure noise.

    2. **Summary features**: Instead of (or in addition to) per-feature
       displacement vectors (which double the feature count), produces
       compact summary statistics:
       - mean_abs_displacement: average absolute shift across features
       - max_displacement: which feature shifted most for this sample
       - displacement_magnitude: L2 norm, also included in v1

    3. **Top-k selection**: Limits displacement features to the k most
       shifted features, which limits the number of new features.

    4. **Skip categorical features**: Quantile matching on label-encoded
       categories is semantically meaningless (displacement from cat 0 to
       cat 1.5 has no interpretation).  Categorical columns are detected
       by dtype before label encoding and excluded from displacement
       computation.  They still pass through as original features.

    5. **IQR normalization**: Raw displacements are in original feature
       units, so a feature with range [0, 1000] dominates summaries over
       a [0, 1] feature.  Each displacement column is divided by the
       combined (train+test) IQR of that feature, making displacements
       scale-invariant and summaries meaningful.

    **Optional absolute-value mode (``abs_disp``)**: By construction, the
    signed per-feature displacement column has sign-mirrored train/test
    means (``disp_train ≈ +Δ`` while ``disp_test ≈ −Δ``), which leaks
    distribution membership into the feature. Setting ``abs_disp=True``
    replaces every per-feature displacement with its absolute value,
    preserving the magnitude signal ("how far is this sample from the
    other distribution") without the sign-mirror artefact.

    **Optional PCA direction columns (``n_summary_pcs``):** Direction components
    can encode split membership because they capture the dominant train-to-test
    displacement direction. They are disabled by default and available only
    when explicitly requested.

    Args:
        mode: "summary_only" (compact), "selective" (only shifted features)
              or "selective_and_summary" (both).
        ks_threshold: p-value threshold for feature selection (default 0.05).
        top_k: Max number of features to compute displacements for.
            None = no limit beyond KS selection (default None).
        n_summary_pcs: Number of displacement-direction components (default 0).
        abs_disp: If True, emit absolute rather than signed per-feature
            displacement (default False).
        seed: Random seed.
    """

    def __init__(
        self,
        mode: str = "selective_and_summary",
        ks_threshold: float = 0.05,
        top_k: int | None = None,
        n_summary_pcs: int = 0,
        abs_disp: bool = False,
        seed: int = 42,
    ):
        assert mode in ("summary_only", "selective", "selective_and_summary"), \
            f"mode must be 'summary_only', 'selective' or 'selective_and_summary', got '{mode}'"
        self.mode = mode
        self.ks_threshold = ks_threshold
        self.top_k = top_k
        self.n_summary_pcs = n_summary_pcs
        self.abs_disp = abs_disp
        self.seed = seed

    @property
    def name(self) -> str:
        parts = [f"ot_disp_v2_{self.mode}"]
        if self.top_k is not None:
            parts.append(f"top{self.top_k}")
        if self.abs_disp:
            parts.append("abs")
        return "_".join(parts)

    def _select_shifted_features(
        self, X_train: np.ndarray, X_test: np.ndarray,
        numeric_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Select features with significant distribution shift via KS test.

        Args:
            X_train: Cleaned training array.
            X_test: Cleaned test array.
            numeric_mask: Boolean array (True = numeric).  Categorical
                features are excluded from displacement computation since
                quantile matching on label-encoded categories is meaningless.

        Returns:
            ks_stats: KS statistic per feature (for all features).
            selected_idx: Indices of selected features.
        """
        from scipy.stats import ks_2samp

        n_features = X_train.shape[1]
        ks_stats = np.zeros(n_features)
        ks_pvals = np.zeros(n_features)

        for j in range(n_features):
            # Skip categorical features. Quantile matching is meaningless
            if numeric_mask is not None and not numeric_mask[j]:
                ks_stats[j] = 0.0
                ks_pvals[j] = 1.0
                continue

            tr_col = X_train[:, j]
            te_col = X_test[:, j]
            tr_valid = tr_col[~np.isnan(tr_col)]
            te_valid = te_col[~np.isnan(te_col)]
            if len(tr_valid) > 1 and len(te_valid) > 1:
                stat, pval = ks_2samp(tr_valid, te_valid)
                ks_stats[j] = stat
                ks_pvals[j] = pval
            else:
                ks_stats[j] = 0.0
                ks_pvals[j] = 1.0

        # Select features with significant shift
        selected = np.where(ks_pvals < self.ks_threshold)[0]

        if len(selected) == 0:
            # Fallback: take top-5 numeric features by KS statistic. Keep
            # categoricals excluded even under fallback. OT on label-encoded
            # categories has no meaningful quantile geometry.
            if numeric_mask is not None:
                numeric_idx = np.where(numeric_mask)[0]
                if len(numeric_idx) == 0:
                    selected = np.array([], dtype=int)
                else:
                    n_take = min(5, len(numeric_idx))
                    order = np.argsort(ks_stats[numeric_idx])[-n_take:]
                    selected = numeric_idx[order]
            else:
                selected = np.argsort(ks_stats)[-min(5, n_features):]

        # Apply top_k limit: keep the most shifted features
        if self.top_k is not None and len(selected) > self.top_k:
            # Sort selected by KS stat descending, keep top_k
            order = np.argsort(ks_stats[selected])[::-1]
            selected = selected[order[: self.top_k]]

        return ks_stats, sorted(selected.tolist())

    def _compute_displacement(
        self, source: np.ndarray, target_sorted: np.ndarray, target_quantiles: np.ndarray
    ) -> np.ndarray:
        """Compute OT displacement for a single feature column."""
        from scipy.stats import rankdata

        ranks = rankdata(source, method="average") / (len(source) + 1)
        values_at_quantile = np.interp(ranks, target_quantiles, target_sorted)
        return values_at_quantile - source

    def fit_transform(self, X_train, X_test, y_train):
        from src.methods.tier1 import _label_encode_cats
        from src.methods.preprocessing import get_cat_cols
        from sklearn.decomposition import PCA as SkPCA

        # Identify categorical columns before encoding while dtypes are available.
        cat_cols_original = set(get_cat_cols(X_train))

        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        feature_cols = list(X_tr_enc.columns)

        # Boolean mask: True = numeric (meaningful for OT displacement)
        numeric_mask = np.array([col not in cat_cols_original for col in feature_cols])

        # Handle NaN
        X_tr_clean = X_tr_arr.copy()
        X_te_clean = X_te_arr.copy()
        for j in range(X_tr_arr.shape[1]):
            tr_nan = np.isnan(X_tr_clean[:, j])
            te_nan = np.isnan(X_te_clean[:, j])
            median_val = np.nanmedian(np.concatenate([X_tr_clean[:, j], X_te_clean[:, j]]))
            if not np.isfinite(median_val):
                median_val = 0.0
            if tr_nan.any():
                X_tr_clean[tr_nan, j] = median_val
            if te_nan.any():
                X_te_clean[te_nan, j] = median_val

        # --- Feature selection (categoricals excluded via numeric_mask) ---
        ks_stats, selected_idx = self._select_shifted_features(
            X_tr_clean, X_te_clean, numeric_mask=numeric_mask,
        )
        n_selected = len(selected_idx)

        if n_selected == 0:
            metadata = {
                "n_features_selected": 0,
                "n_features_total": X_tr_arr.shape[1],
                "n_cat_skipped": int((~numeric_mask).sum()),
                "selected_features": [],
                "mean_ks_stat_selected": 0.0,
                "mean_train_magnitude": 0.0,
                "mean_test_magnitude": 0.0,
                "note": "no numeric features available for OT displacement",
            }
            return TransformResult(
                X_train=X_tr_arr,
                X_test=X_te_arr,
                feature_names=feature_cols,
                metadata=metadata,
            )

        # --- Compute displacements for selected features ---
        disp_train = np.zeros((len(X_tr_clean), n_selected))
        disp_test = np.zeros((len(X_te_clean), n_selected))

        for i, j in enumerate(selected_idx):
            tr_col = X_tr_clean[:, j]
            te_col = X_te_clean[:, j]

            # Train → Test displacement
            te_sorted = np.sort(te_col)
            te_quantiles = np.linspace(0, 1, len(te_sorted) + 2)[1:-1]
            disp_train[:, i] = self._compute_displacement(tr_col, te_sorted, te_quantiles)

            # Test → Train displacement
            tr_sorted = np.sort(tr_col)
            tr_quantiles = np.linspace(0, 1, len(tr_sorted) + 2)[1:-1]
            disp_test[:, i] = self._compute_displacement(te_col, tr_sorted, tr_quantiles)

        # Optional absolute-value transformation removes the sign difference.
        # Signed displacement columns have train_mean ≈ +Δ / test_mean ≈ −Δ
        # by construction, leaking train-vs-test membership. |disp| preserves
        # the magnitude signal ("how far from the other distribution") while
        # producing symmetric train/test distributions.
        if self.abs_disp:
            disp_train = np.abs(disp_train)
            disp_test = np.abs(disp_test)

        # --- Normalize displacements by IQR (scale-invariant) ---
        for i, j in enumerate(selected_idx):
            combined_col = np.concatenate([X_tr_clean[:, j], X_te_clean[:, j]])
            iqr = float(np.percentile(combined_col, 75) - np.percentile(combined_col, 25))
            if iqr > 0:
                disp_train[:, i] /= iqr
                disp_test[:, i] /= iqr
            # iqr == 0 → constant feature, keep raw displacement (likely ~0)

        # --- Summary features ---
        summary_train_parts = []
        summary_test_parts = []
        summary_names = []

        # Always compute summaries
        # 1. Mean absolute displacement
        mean_abs_disp_train = np.abs(disp_train).mean(axis=1, keepdims=True)
        mean_abs_disp_test = np.abs(disp_test).mean(axis=1, keepdims=True)
        summary_train_parts.append(mean_abs_disp_train)
        summary_test_parts.append(mean_abs_disp_test)
        summary_names.append("ot_mean_abs_disp")

        # 2. Max absolute displacement
        max_abs_disp_train = np.abs(disp_train).max(axis=1, keepdims=True)
        max_abs_disp_test = np.abs(disp_test).max(axis=1, keepdims=True)
        summary_train_parts.append(max_abs_disp_train)
        summary_test_parts.append(max_abs_disp_test)
        summary_names.append("ot_max_abs_disp")

        # 3. L2 magnitude
        mag_train = np.linalg.norm(disp_train, axis=1, keepdims=True)
        mag_test = np.linalg.norm(disp_test, axis=1, keepdims=True)
        summary_train_parts.append(mag_train)
        summary_test_parts.append(mag_test)
        summary_names.append("ot_disp_magnitude")

        # Optional displacement-direction components. These may encode split
        # membership because they capture the dominant train-to-test direction,
        # so they are emitted only when explicitly requested.
        if n_selected >= 2 and self.n_summary_pcs > 0:
            n_pcs = min(self.n_summary_pcs, n_selected)
            disp_combined = np.vstack([disp_train, disp_test])
            pca = SkPCA(n_components=n_pcs)
            pca.fit(disp_combined)
            pca_train = pca.transform(disp_train)
            pca_test = pca.transform(disp_test)
            summary_train_parts.append(pca_train)
            summary_test_parts.append(pca_test)
            for k in range(n_pcs):
                summary_names.append(f"ot_disp_dir_pc{k+1}")

        # --- Build output ---
        feature_names = feature_cols.copy()

        if self.mode in ("selective", "selective_and_summary"):
            # Append per-feature displacements (selected only)
            disp_names = [f"ot_disp_{feature_cols[j]}" for j in selected_idx]
            X_tr_out = np.hstack([X_tr_arr, disp_train])
            X_te_out = np.hstack([X_te_arr, disp_test])
            feature_names = feature_names + disp_names
        else:
            X_tr_out = X_tr_arr
            X_te_out = X_te_arr

        if self.mode in ("summary_only", "selective_and_summary"):
            summary_train = np.hstack(summary_train_parts)
            summary_test = np.hstack(summary_test_parts)
            X_tr_out = np.hstack([X_tr_out, summary_train])
            X_te_out = np.hstack([X_te_out, summary_test])
            feature_names = feature_names + summary_names

        metadata = {
            "n_features_selected": n_selected,
            "n_features_total": X_tr_arr.shape[1],
            "n_cat_skipped": int((~numeric_mask).sum()),
            "selected_features": [feature_cols[j] for j in selected_idx],
            "mean_ks_stat_selected": float(ks_stats[selected_idx].mean()) if selected_idx else 0.0,
            "mean_train_magnitude": float(mag_train.mean()),
            "mean_test_magnitude": float(mag_test.mean()),
        }

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            metadata=metadata,
        )


# ===========================================================================
# M7v2: Calibrated Pseudo-Label Self-Training
# ===========================================================================

class PseudoLabelCalibrated(TestTimeMethod):
    """
    M7v2: Pseudo-labeling with probability calibration and confidence-based
    weights.

    **Probability calibration:** Platt scaling or isotonic regression is
    applied to the pseudo-labeling model's out-of-fold predictions before
    thresholding.

    **Confidence-proportional weights:**
    Instead of a fixed pseudo_weight for all pseudo-labeled samples,
    weight each sample proportionally to its calibrated confidence:
        weight = ((confidence - 0.5) / 0.5) ^ power
    A 99%-confident pseudo-label gets much higher weight than an
    81%-confident one.

    Args:
        confidence_threshold: Minimum calibrated probability (default 0.9).
        max_iterations: Self-training rounds (default 3).
        classifier: Model for pseudo-labeling ("lgbm", "logreg", "rf").
        calibration_method: "platt" (sigmoid) or "isotonic" (default "isotonic").
        weight_mode: How to assign pseudo-label weights:
            "fixed": constant weight (like original M7)
            "proportional": weight = ((conf - 0.5) / 0.5)
            "quadratic": weight = ((conf - 0.5) / 0.5)^2
        base_pseudo_weight: Base weight for fixed mode or scaling factor
            for proportional modes (default 0.5).
        n_cal_folds: Number of folds for calibration (default 3).
        seed: Random seed.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.9,
        max_iterations: int = 3,
        classifier: str = "lgbm",
        calibration_method: str = "isotonic",
        weight_mode: str = "proportional",
        base_pseudo_weight: float = 0.5,
        n_cal_folds: int = 3,
        seed: int = 42,
    ):
        assert 0.5 < confidence_threshold < 1.0
        assert calibration_method in ("platt", "isotonic")
        assert weight_mode in ("fixed", "proportional", "quadratic")
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.classifier = classifier
        self.calibration_method = calibration_method
        self.weight_mode = weight_mode
        self.base_pseudo_weight = base_pseudo_weight
        self.n_cal_folds = n_cal_folds
        self.seed = seed

    @property
    def name(self) -> str:
        return (
            f"pseudo_label_v2_{self.classifier}"
            f"_cal_{self.calibration_method}"
            f"_w_{self.weight_mode}"
            f"_conf{self.confidence_threshold}"
        )

    @property
    def uses_labels(self) -> bool:
        return True

    @property
    def may_emit_sample_weight(self) -> bool:
        return True

    @property
    def supported_task_types(self) -> frozenset[str]:
        return frozenset({"classification"})

    def _calibrate_model(self, clf, X_train, y_train):
        """
        Wrap a classifier with probability calibration using OOF.

        Returns a CalibratedClassifierCV that produces well-calibrated
        probabilities.
        """
        from sklearn.calibration import CalibratedClassifierCV

        method = "sigmoid" if self.calibration_method == "platt" else "isotonic"
        calibrated = CalibratedClassifierCV(
            clf,
            method=method,
            cv=self.n_cal_folds,
        )
        calibrated.fit(X_train, y_train)
        return calibrated

    def _compute_weight(self, confidence: np.ndarray) -> np.ndarray:
        """Compute per-sample weight based on confidence and weight_mode."""
        if self.weight_mode == "fixed":
            return np.full(len(confidence), self.base_pseudo_weight)
        elif self.weight_mode == "proportional":
            # Linear scaling: 0 at conf=0.5, 1.0 at conf=1.0
            raw = (confidence - 0.5) / 0.5
            return np.clip(raw, 0.0, 1.0) * self.base_pseudo_weight * 2
        else:  # quadratic
            raw = ((confidence - 0.5) / 0.5) ** 2
            return np.clip(raw, 0.0, 1.0) * self.base_pseudo_weight * 2

    def fit_transform(self, X_train, X_test, y_train):
        # M7 pseudo-labeling is classification-only. Raise the documented skip for
        # regression targets rather than building a degenerate classifier or raising
        # "Unknown label type: continuous".
        is_classification = (
            self.method_context.task_type == "classification"
            if self.method_context is not None
            else _pseudo_is_classification(np.asarray(y_train))
        )
        if not is_classification:
            raise PseudoLabelSkipReason(
                "regression_unsupported",
                {"n_unique": int(len(np.unique(y_train))),
                 "dtype": str(np.asarray(y_train).dtype)},
            )
        from src.methods.tier1 import _label_encode_cats

        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        y_tr = np.asarray(y_train).copy()

        n_original_train = len(X_tr_arr)

        available_mask = np.ones(len(X_te_arr), dtype=bool)
        pseudo_X_parts: list[np.ndarray] = []
        pseudo_y_parts: list[np.ndarray] = []
        pseudo_w_parts: list[np.ndarray] = []
        pseudo_source_parts: list[np.ndarray] = []

        total_pseudo = 0
        n_iterations_executed = 0
        calibration_used = False
        calibration_skipped_guard = False
        iteration_stats: list[dict] = []

        for iteration in range(self.max_iterations):
            n_iterations_executed += 1

            # Build current augmented training set
            if pseudo_X_parts:
                X_aug = np.vstack([X_tr_arr] + pseudo_X_parts)
                y_aug = np.concatenate([y_tr] + pseudo_y_parts)
            else:
                X_aug = X_tr_arr
                y_aug = y_tr

            # Train classifier
            n_classes = len(np.unique(y_aug))
            clf = _build_task_classifier(
                self.classifier, self.seed + iteration, n_classes=n_classes
            )

            # Only impute NaN for classifiers that cannot handle it natively.
            # LightGBM learns optimal split directions for missing values.
            # Filling with 0.0 destroys that signal.
            if self.classifier != "lgbm":
                X_aug_clean = np.nan_to_num(X_aug, nan=0.0)
            else:
                X_aug_clean = X_aug

            # Apply calibration only when enough rows are available to avoid
            # unstable small-sample calibration.
            cal_guard_threshold = max(50, self.n_cal_folds * 20)
            if len(X_aug_clean) >= cal_guard_threshold and n_classes >= 2:
                try:
                    clf = self._calibrate_model(clf, X_aug_clean, y_aug)
                    calibration_used = True
                except Exception:
                    # Fallback: use uncalibrated model
                    clf.fit(X_aug_clean, y_aug)
            else:
                calibration_skipped_guard = True
                clf.fit(X_aug_clean, y_aug)

            # Predict on remaining test samples
            if not available_mask.any():
                break

            X_remaining = X_te_arr[available_mask]
            if self.classifier != "lgbm":
                X_remaining = np.nan_to_num(X_remaining, nan=0.0)
            proba = clf.predict_proba(X_remaining)

            max_proba = proba.max(axis=1)
            predicted_labels = clf.classes_[proba.argmax(axis=1)]

            # Select high-confidence samples using calibrated probabilities.
            confident = max_proba >= self.confidence_threshold
            n_new = confident.sum()

            if n_new == 0:
                break

            # Map back to original test indices
            remaining_indices = np.where(available_mask)[0]
            selected_indices = remaining_indices[confident]

            pseudo_X_parts.append(X_te_arr[selected_indices])
            pseudo_y_parts.append(predicted_labels[confident])
            pseudo_source_parts.append(selected_indices.copy())

            # Confidence-proportional weights.
            selected_confidence = max_proba[confident]
            iter_weights = self._compute_weight(selected_confidence)
            pseudo_w_parts.append(iter_weights)

            available_mask[selected_indices] = False
            total_pseudo += n_new

            iteration_stats.append({
                "iteration": iteration,
                "n_accepted": int(n_new),
                "mean_confidence": float(selected_confidence.mean()),
                "mean_weight": float(iter_weights.mean()),
            })

        # Build final augmented training arrays
        if pseudo_X_parts:
            X_tr_out = np.vstack([X_tr_arr] + pseudo_X_parts)
            y_tr_out = np.concatenate([y_tr] + pseudo_y_parts)
            # Weights: 1.0 for real, variable for pseudo-labeled
            weights = np.concatenate([
                np.ones(n_original_train),
                *pseudo_w_parts,
            ])
        else:
            X_tr_out = X_tr_arr
            y_tr_out = y_tr
            weights = None

        pseudo_source_positions = (
            np.concatenate(pseudo_source_parts)
            if pseudo_source_parts else np.array([], dtype=np.int64)
        )
        row_provenance = RowProvenance.augmented(
            n_original_train=n_original_train,
            augmented_source_split="test",
            augmented_source_position=pseudo_source_positions,
            augmented_role="pseudo_label",
        )

        feature_names = list(X_tr_enc.columns)

        metadata = {
            "n_pseudo_labels": total_pseudo,
            "n_iterations_run": n_iterations_executed,
            "n_train_augmented": len(X_tr_out),
            "calibration_used": calibration_used,
            "m7v2_calibration_skipped": calibration_skipped_guard,
            "calibration_method": self.calibration_method,
            "weight_mode": self.weight_mode,
            "iteration_stats": iteration_stats,
        }
        if pseudo_w_parts:
            all_pseudo_weights = np.concatenate(pseudo_w_parts)
            metadata["mean_pseudo_weight"] = float(all_pseudo_weights.mean())
            metadata["std_pseudo_weight"] = float(all_pseudo_weights.std())

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_arr,
            feature_names=feature_names,
            sample_weights_train=weights,
            y_train=y_tr_out if total_pseudo > 0 else None,
            metadata=metadata,
            row_provenance_train=row_provenance,
        )


# ===========================================================================
# M17: Test-Directed Data Augmentation (TDDA)
# ===========================================================================

class TestDirectedDataAugmentation(TestTimeMethod):
    """
    M17: For each training sample, find its k nearest test neighbors
    and create augmented samples via mixup interpolation.

    x_aug = lambda_ * x_train + (1 - lambda_) * x_test_neighbor
    y_aug = y_train  (label comes from the training point)

    The augmented samples lie between training rows and nearby test rows. They
    retain the training-row labels while moving the feature values toward the
    test features.

    Args:
        k: Number of test neighbors per training sample (default 5).
        lambda_: Interpolation parameter.  Higher → stays closer to
            the training point (default 0.7).
        n_augments: Number of augmented samples per training point.
            Each uses a different test neighbor (default 1).
        augment_weight: Sample weight for augmented rows (default 0.5).
        feature_weighting: How to select features for kNN distance:
            "none": use all features (default)
            "cumshift": keep the fewest features capturing ``shift_ratio``
                         of total KS shift (adaptive dimensionality reduction).
                         Only affects kNN neighbor search. Mixup still
                         operates on all features.
        shift_ratio: Fraction of total KS shift to capture in "cumshift"
            mode (default 0.9).  Floor of 3 features always kept.
        seed: Random seed.
    """

    def __init__(
        self,
        k: int = 5,
        lambda_: float = 0.7,
        n_augments: int = 1,
        augment_weight: float = 0.5,
        feature_weighting: str = "none",
        shift_ratio: float = 0.9,
        seed: int = 42,
    ):
        assert 0.0 < lambda_ < 1.0, f"lambda_ must be in (0, 1), got {lambda_}"
        assert n_augments >= 1, f"n_augments must be >= 1, got {n_augments}"
        assert feature_weighting in ("none", "cumshift"), \
            f"feature_weighting must be 'none' or 'cumshift', got '{feature_weighting}'"
        self.k = k
        self.lambda_ = lambda_
        self.n_augments = n_augments
        self.augment_weight = augment_weight
        self.feature_weighting = feature_weighting
        self.shift_ratio = shift_ratio
        self.seed = seed

    @property
    def name(self) -> str:
        cs = "_cumshift" if self.feature_weighting == "cumshift" else ""
        return (
            f"tdda_k{self.k}_lam{self.lambda_}"
            f"_aug{self.n_augments}{cs}"
        )

    @property
    def uses_labels(self) -> bool:
        return True

    @property
    def may_emit_sample_weight(self) -> bool:
        return True

    def fit_transform(self, X_train, X_test, y_train):
        from src.methods.tier1 import _label_encode_cats

        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        y_tr = np.asarray(y_train)
        feature_cols = list(X_tr_enc.columns)

        n_train = len(X_tr_arr)

        # --- Median NaN imputation (per-feature, joint train+test) ---
        X_tr_clean = X_tr_arr.copy()
        X_te_clean = X_te_arr.copy()
        for j in range(X_tr_arr.shape[1]):
            combined_col = np.concatenate([X_tr_arr[:, j], X_te_arr[:, j]])
            median_val = np.nanmedian(combined_col)
            if not np.isfinite(median_val):
                median_val = 0.0
            tr_nan = np.isnan(X_tr_clean[:, j])
            te_nan = np.isnan(X_te_clean[:, j])
            if tr_nan.any():
                X_tr_clean[tr_nan, j] = median_val
            if te_nan.any():
                X_te_clean[te_nan, j] = median_val

        # --- Standardize for kNN ---
        scaler = StandardScaler()
        combined = np.vstack([X_tr_clean, X_te_clean])
        scaler.fit(combined)
        X_tr_scaled = scaler.transform(X_tr_clean)
        X_te_scaled = scaler.transform(X_te_clean)

        # --- cumshift feature selection for kNN distance ---
        n_knn_features = X_tr_scaled.shape[1]
        cumshift_info = {}

        if self.feature_weighting == "cumshift":
            from scipy.stats import ks_2samp

            n_f = X_tr_scaled.shape[1]
            ks_stats = np.zeros(n_f)
            for j in range(n_f):
                tr_col = X_tr_clean[:, j]
                te_col = X_te_clean[:, j]
                if len(tr_col) > 1 and len(te_col) > 1:
                    stat, _ = ks_2samp(tr_col, te_col)
                    ks_stats[j] = stat

            order = np.argsort(ks_stats)[::-1]
            cumsum = np.cumsum(ks_stats[order])
            total = cumsum[-1]

            if total < 1e-6:
                # No measurable shift → use all features
                selected_idx = np.arange(n_f)
            else:
                n_keep = int(np.searchsorted(cumsum, total * self.shift_ratio)) + 1
                n_keep = max(n_keep, 3)       # floor: avoid degenerate space
                n_keep = min(n_keep, n_f)     # cap
                selected_idx = order[:n_keep]

            X_tr_knn = X_tr_scaled[:, selected_idx]
            X_te_knn = X_te_scaled[:, selected_idx]
            n_knn_features = len(selected_idx)
            achieved = float(cumsum[n_knn_features - 1] / total) if total > 1e-6 else 0.0
            cumshift_info = {
                "shift_ratio_target": self.shift_ratio,
                "shift_ratio_achieved": achieved,
                "selected_features": [feature_cols[j] for j in selected_idx],
            }
        else:
            X_tr_knn = X_tr_scaled
            X_te_knn = X_te_scaled

        # --- Find k nearest test neighbors (on selected features) ---
        k_eff = min(self.k, len(X_te_knn))
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=1)
        nn.fit(X_te_knn)
        _, test_neighbor_idx = nn.kneighbors(X_tr_knn)  # (n_train, k_eff)

        # --- Generate augmented samples (mixup on imputed features) ---
        aug_X_parts = []
        aug_y_parts = []

        n_aug_per = min(self.n_augments, k_eff)

        for i in range(n_train):
            # Pick n_augments distinct test neighbors
            neighbors = test_neighbor_idx[i, :n_aug_per]

            for j_idx in range(n_aug_per):
                te_neighbor = X_te_clean[neighbors[j_idx]]
                # Mixup interpolation (on imputed values → NaN-free)
                x_aug = self.lambda_ * X_tr_clean[i] + (1.0 - self.lambda_) * te_neighbor
                aug_X_parts.append(x_aug)
                aug_y_parts.append(y_tr[i])

        aug_X = np.array(aug_X_parts)  # (n_train * n_aug_per, n_features)
        aug_y = np.array(aug_y_parts)

        # Combine original + augmented training data
        X_tr_out = np.vstack([X_tr_arr, aug_X])

        # Weights: 1.0 for original, augment_weight for augmented
        weights = np.concatenate([
            np.ones(n_train),
            np.full(len(aug_X), self.augment_weight),
        ])

        y_augmented = np.concatenate([y_tr, aug_y])
        row_provenance = RowProvenance.augmented(
            n_original_train=n_train,
            augmented_source_split="train",
            augmented_source_position=np.repeat(
                np.arange(n_train, dtype=np.int64), n_aug_per
            ),
            augmented_role="tdda_augment",
        )

        feature_names = feature_cols

        metadata = {
            "n_augmented": len(aug_X),
            "n_train_total": len(X_tr_out),
            "lambda": self.lambda_,
            "feature_weighting": self.feature_weighting,
            "n_features_for_knn": n_knn_features,
            "n_features_total": X_tr_arr.shape[1],
        }
        metadata.update(cumshift_info)

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_arr,
            feature_names=feature_names,
            sample_weights_train=weights,
            y_train=y_augmented,
            metadata=metadata,
            row_provenance_train=row_provenance,
        )


# ===========================================================================
# M17v2: Improved TDDA
# ===========================================================================

class TestDirectedDataAugmentationImproved(TestTimeMethod):
    """
    M17v2: TDDA with per-feature lambda and distance-dependent weights.

    The method adds two refinements:

    1. **Per-feature lambda:** The original uses a single lambda for all
       features.  But features that shifted a lot should be interpolated
       more aggressively toward the test distribution. The method computes the KS
       statistic per feature and set:

           lambda_j = base_lambda + (1 - base_lambda) * (1 - ks_stat_j)

       So unshifted features (KS≈0) get lambda≈1 (stay close to train)
       and heavily shifted features get lambda≈base_lambda (move toward
       test).  This means the mixup is *feature-wise*, not a single scalar.

    2. **Distance-dependent augment weight:** The original assigns a fixed
       weight to all augmented samples. Augmented points from nearby test
       neighbors receive more weight than points based on distant neighbors.
       The weight is:

           weight = base_weight / (1 + distance_to_neighbor)

       Nearby neighbors produce high-weight augmented points.

    Args:
        k: Number of test neighbors (default 5).
        base_lambda: Base interpolation (default 0.7).  This is the
            minimum lambda for the most shifted features.
        n_augments: Augmented samples per training point (default 1).
        base_augment_weight: Base weight before distance scaling (default 0.5).
        use_per_feature_lambda: Enable per-feature lambda (default True).
        use_distance_weights: Enable distance-dependent weighting (default True).
        feature_weighting: How to select features for kNN distance:
            "none": use all features (default)
            "cumshift": keep the fewest features capturing ``shift_ratio``
                         of total KS shift (adaptive dimensionality reduction).
                         Only affects kNN neighbor search. Mixup still
                         operates on all features.
        shift_ratio: Fraction of total KS shift to capture in "cumshift"
            mode (default 0.9).  Floor of 3 features always kept.
        seed: Random seed.
    """

    def __init__(
        self,
        k: int = 5,
        base_lambda: float = 0.7,
        n_augments: int = 1,
        base_augment_weight: float = 0.5,
        use_per_feature_lambda: bool = True,
        use_distance_weights: bool = True,
        feature_weighting: str = "none",
        shift_ratio: float = 0.9,
        seed: int = 42,
    ):
        assert 0.0 < base_lambda < 1.0
        assert n_augments >= 1
        assert feature_weighting in ("none", "cumshift"), \
            f"feature_weighting must be 'none' or 'cumshift', got '{feature_weighting}'"
        self.k = k
        self.base_lambda = base_lambda
        self.n_augments = n_augments
        self.base_augment_weight = base_augment_weight
        self.use_per_feature_lambda = use_per_feature_lambda
        self.use_distance_weights = use_distance_weights
        self.feature_weighting = feature_weighting
        self.shift_ratio = shift_ratio
        self.seed = seed

    @property
    def name(self) -> str:
        parts = [f"tdda_v2_k{self.k}_lam{self.base_lambda}"]
        if self.use_per_feature_lambda:
            parts.append("pfl")
        if self.use_distance_weights:
            parts.append("dw")
        if self.feature_weighting == "cumshift":
            parts.append("cumshift")
        return "_".join(parts)

    @property
    def uses_labels(self) -> bool:
        return True

    @property
    def may_emit_sample_weight(self) -> bool:
        return True

    def _compute_per_feature_lambda(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-feature interpolation lambda based on KS shift.

        Returns:
            lambdas: (n_features,): higher = closer to train.
        """
        from scipy.stats import ks_2samp

        n_features = X_train.shape[1]
        ks_stats = np.zeros(n_features)

        for j in range(n_features):
            tr_col = X_train[:, j]
            te_col = X_test[:, j]
            tr_valid = tr_col[~np.isnan(tr_col)]
            te_valid = te_col[~np.isnan(te_col)]
            if len(tr_valid) > 1 and len(te_valid) > 1:
                stat, _ = ks_2samp(tr_valid, te_valid)
                ks_stats[j] = stat

        # lambda_j = base_lambda + (1 - base_lambda) * (1 - ks_stat_j)
        # KS=0 → lambda=1.0 (fully train)
        # KS=1 → lambda=base_lambda (interpolate maximally toward test)
        lambdas = self.base_lambda + (1.0 - self.base_lambda) * (1.0 - ks_stats)

        return lambdas

    def fit_transform(self, X_train, X_test, y_train):
        from src.methods.tier1 import _label_encode_cats

        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        y_tr = np.asarray(y_train)
        feature_cols = list(X_tr_enc.columns)

        n_train = len(X_tr_arr)
        n_features = X_tr_arr.shape[1]

        # --- Median NaN imputation (per-feature, joint train+test) ---
        X_tr_clean = X_tr_arr.copy()
        X_te_clean = X_te_arr.copy()
        for j in range(n_features):
            combined_col = np.concatenate([X_tr_arr[:, j], X_te_arr[:, j]])
            median_val = np.nanmedian(combined_col)
            if not np.isfinite(median_val):
                median_val = 0.0
            tr_nan = np.isnan(X_tr_clean[:, j])
            te_nan = np.isnan(X_te_clean[:, j])
            if tr_nan.any():
                X_tr_clean[tr_nan, j] = median_val
            if te_nan.any():
                X_te_clean[te_nan, j] = median_val

        # --- Standardize for kNN ---
        scaler = StandardScaler()
        combined = np.vstack([X_tr_clean, X_te_clean])
        scaler.fit(combined)
        X_tr_scaled = scaler.transform(X_tr_clean)
        X_te_scaled = scaler.transform(X_te_clean)

        # --- Per-feature lambda ---
        if self.use_per_feature_lambda:
            lambdas = self._compute_per_feature_lambda(X_tr_clean, X_te_clean)
            # lambdas is (n_features,). Will broadcast during mixup
        else:
            lambdas = np.full(n_features, self.base_lambda)

        # --- cumshift feature selection for kNN distance ---
        n_knn_features = X_tr_scaled.shape[1]
        cumshift_info = {}

        if self.feature_weighting == "cumshift":
            from scipy.stats import ks_2samp

            n_f = X_tr_scaled.shape[1]
            ks_stats = np.zeros(n_f)
            for j in range(n_f):
                tr_col = X_tr_clean[:, j]
                te_col = X_te_clean[:, j]
                if len(tr_col) > 1 and len(te_col) > 1:
                    stat, _ = ks_2samp(tr_col, te_col)
                    ks_stats[j] = stat

            order = np.argsort(ks_stats)[::-1]
            cumsum = np.cumsum(ks_stats[order])
            total = cumsum[-1]

            if total < 1e-6:
                # No measurable shift → use all features
                selected_idx = np.arange(n_f)
            else:
                n_keep = int(np.searchsorted(cumsum, total * self.shift_ratio)) + 1
                n_keep = max(n_keep, 3)       # floor: avoid degenerate space
                n_keep = min(n_keep, n_f)     # cap
                selected_idx = order[:n_keep]

            X_tr_knn = X_tr_scaled[:, selected_idx]
            X_te_knn = X_te_scaled[:, selected_idx]
            n_knn_features = len(selected_idx)
            achieved = float(cumsum[n_knn_features - 1] / total) if total > 1e-6 else 0.0
            cumshift_info = {
                "shift_ratio_target": self.shift_ratio,
                "shift_ratio_achieved": achieved,
                "selected_features": [feature_cols[j] for j in selected_idx],
            }
        else:
            X_tr_knn = X_tr_scaled
            X_te_knn = X_te_scaled

        # --- Find k nearest test neighbors (on selected features, with distances) ---
        k_eff = min(self.k, len(X_te_knn))
        nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=1)
        nn.fit(X_te_knn)
        distances, test_neighbor_idx = nn.kneighbors(X_tr_knn)

        # --- Generate augmented samples (mixup on imputed features) ---
        aug_X_parts = []
        aug_y_parts = []
        aug_w_parts = []

        n_aug_per = min(self.n_augments, k_eff)

        for i in range(n_train):
            neighbors = test_neighbor_idx[i, :n_aug_per]
            neighbor_dists = distances[i, :n_aug_per]

            for j_idx in range(n_aug_per):
                te_neighbor = X_te_clean[neighbors[j_idx]]

                # Per-feature mixup (on imputed values → NaN-free)
                x_aug = lambdas * X_tr_clean[i] + (1.0 - lambdas) * te_neighbor
                aug_X_parts.append(x_aug)
                aug_y_parts.append(y_tr[i])

                # Distance-dependent weight
                if self.use_distance_weights:
                    dist = neighbor_dists[j_idx]
                    w = self.base_augment_weight / (1.0 + dist)
                else:
                    w = self.base_augment_weight
                aug_w_parts.append(w)

        aug_X = np.array(aug_X_parts)
        aug_y = np.array(aug_y_parts)
        aug_w = np.array(aug_w_parts)

        # Combine original + augmented
        X_tr_out = np.vstack([X_tr_arr, aug_X])
        weights = np.concatenate([np.ones(n_train), aug_w])
        y_augmented = np.concatenate([y_tr, aug_y])
        row_provenance = RowProvenance.augmented(
            n_original_train=n_train,
            augmented_source_split="train",
            augmented_source_position=np.repeat(
                np.arange(n_train, dtype=np.int64), n_aug_per
            ),
            augmented_role="tdda_augment",
        )

        feature_names = feature_cols

        metadata = {
            "n_augmented": len(aug_X),
            "n_train_total": len(X_tr_out),
            "base_lambda": self.base_lambda,
            "per_feature_lambda": self.use_per_feature_lambda,
            "distance_weights": self.use_distance_weights,
            "mean_augment_weight": float(aug_w.mean()),
            "std_augment_weight": float(aug_w.std()),
            "feature_weighting": self.feature_weighting,
            "n_features_for_knn": n_knn_features,
            "n_features_total": n_features,
        }
        if self.use_per_feature_lambda:
            metadata["mean_lambda"] = float(lambdas.mean())
            metadata["min_lambda"] = float(lambdas.min())
        metadata.update(cumshift_info)

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_arr,
            feature_names=feature_names,
            sample_weights_train=weights,
            y_train=y_augmented,
            metadata=metadata,
            row_provenance_train=row_provenance,
        )


# ---------------------------------------------------------------------------
# M18: GBDT Leaf Co-Occurrence Analysis
# ---------------------------------------------------------------------------

class LeafBasedTestDensityFE(TestTimeMethod):
    """
    M18: GBDT Leaf Co-Occurrence Analysis.

    Cross-fits auxiliary LightGBMs on (X_train, y_train), extracts leaf
    assignments for all samples inside each fold model and analyses per-leaf
    train/test occupancy as a distribution-shift diagnostic. Each training
    row receives features only from the fold model that excluded that row.
    Test features are averaged after feature construction, so leaf IDs from
    different GBDTs are never compared or concatenated.

    The auxiliary GBDT is trained directly on the (NaN-bearing) feature
    matrix. LightGBM learns optimal split directions for missing values
    natively, so no NaN imputation is performed before training.

    Modes
    -----
    confidence_feature
        Appends 3 features: leaf_minority_ratio, leaf_avg_train_frac,
        leaf_avg_size, which summarizes average leaf size across the
        model's decision trees.
    leaf_embedding_knn
        Uses leaf-assignment vectors as an embedding space. Builds k-NN
        with Hamming distance and appends 2 features:
        leaf_hamming_dist_to_other, leaf_cross_split_frac.

    Parameters
    ----------
    feature_subset : {"all", "cumshift"}
        Input columns used by the auxiliary GBDT:
        "all": train on every column (default. Tree partition is
                     task-driven, may include splits on non-shifted
                     predictive features).
        "cumshift": train only on the smallest set of columns whose
                     KS statistics cumulatively cover ``shift_ratio`` of
                     the total KS shift (floor of 3). Tree partition is
                     then shift-driven, focusing leaves on regions that
                     distinguish train from test. Same algorithm
                     used by M16/M17.
        Output features (the appended leaf-occupancy columns) are always
        appended to the full original feature matrix. Selection only
        affects what the auxiliary GBDT trains on.
    shift_ratio : float
        Fraction of total KS shift to capture in "cumshift" mode
        (default 0.9). Floor of 3 features always kept. Ignored when
        feature_subset == "all".
    n_folds : int
        Requested cross-fitting folds (default 5). Classification uses
        stratified folds when class counts permit. Otherwise deterministic
        shuffled K-fold is used as a documented fallback.
    """

    def __init__(
        self,
        mode: str = "confidence_feature",
        n_estimators: int = 100,
        max_depth: int = 6,
        k: int = 10,
        minority_threshold: float = 0.1,
        feature_subset: str = "all",
        shift_ratio: float = 0.9,
        n_folds: int = 5,
        seed: int = 42,
    ):
        if mode not in ("confidence_feature", "leaf_embedding_knn"):
            raise ValueError(f"Unknown mode '{mode}'")
        if feature_subset not in ("all", "cumshift"):
            raise ValueError(
                f"feature_subset must be 'all' or 'cumshift', got '{feature_subset}'"
            )
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        if not (0.0 < shift_ratio <= 1.0):
            raise ValueError(f"shift_ratio must be in (0, 1], got {shift_ratio}")
        if not (0.0 <= minority_threshold <= 1.0):
            raise ValueError(
                f"minority_threshold must be in [0, 1], got {minority_threshold}"
            )
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.k = k
        self.minority_threshold = minority_threshold
        self.feature_subset = feature_subset
        self.shift_ratio = shift_ratio
        self.n_folds = n_folds
        self.seed = seed

    @property
    def name(self) -> str:
        cs = "_cumshift" if self.feature_subset == "cumshift" else ""
        if self.mode == "confidence_feature":
            return f"leaf_density_conf_d{self.max_depth}_t{self.n_estimators}{cs}"
        # Embed mode also includes _t{n_estimators} to avoid name collisions
        # if a future variant changes the auxiliary tree count.
        return (
            f"leaf_density_embed_k{self.k}_d{self.max_depth}"
            f"_t{self.n_estimators}{cs}"
        )

    @property
    def uses_labels(self) -> bool:
        return True  # trains GBDT on y_train

    # ----- internal helpers --------------------------------------------------

    def _effective_n_estimators(self, n_train: int) -> int:
        """Reduce tree count for small datasets to avoid overfitting."""
        if n_train < 50:
            return min(self.n_estimators, 20)
        if n_train < 200:
            return min(self.n_estimators, 50)
        return self.n_estimators

    @staticmethod
    def _is_classification(y_train: np.ndarray) -> bool:
        """Infer the task type when no MethodContext is bound."""
        unique_y = np.unique(y_train)
        if y_train.dtype.kind in "biuO":
            return True
        if len(unique_y) <= 50 and np.allclose(unique_y, np.round(unique_y)):
            return True
        return False

    def _resolve_task_type(self, y_train: np.ndarray) -> tuple[bool, str]:
        if self.method_context is not None:
            return self.method_context.task_type == "classification", "context"
        return self._is_classification(y_train), "fallback_target_heuristic"

    def _crossfit_splits(
        self,
        y_train: np.ndarray,
        is_clf: bool,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
        """Build deterministic folds with a fallback for rare classes."""
        n_train = len(y_train)
        if n_train < 2:
            raise ValueError("M18 cross-fitting requires at least 2 training rows")

        requested = min(self.n_folds, n_train)
        splitter_name = "kfold"
        if is_clf:
            _, class_counts = np.unique(y_train, return_counts=True)
            if len(class_counts) >= 2 and int(class_counts.min()) >= 2:
                effective = min(requested, int(class_counts.min()))
                splitter = StratifiedKFold(
                    n_splits=effective,
                    shuffle=True,
                    random_state=self.seed,
                )
                splits = list(splitter.split(np.zeros((n_train, 1)), y_train))
                splitter_name = "stratified_kfold"
            else:
                effective = requested
                splitter = KFold(
                    n_splits=effective,
                    shuffle=True,
                    random_state=self.seed,
                )
                splits = list(splitter.split(np.zeros((n_train, 1))))
                splitter_name = "kfold_rare_class_fallback"
        else:
            effective = requested
            splitter = KFold(
                n_splits=effective,
                shuffle=True,
                random_state=self.seed,
            )
            splits = list(splitter.split(np.zeros((n_train, 1))))

        return splits, {
            "crossfit_n_folds_requested": int(self.n_folds),
            "crossfit_n_folds_effective": int(effective),
            "crossfit_splitter": splitter_name,
        }

    def _build_and_fit_gbdt(
        self, X: np.ndarray, y: np.ndarray, n_est: int, is_clf: bool,
    ):
        """Build, fit and return an LightGBM model."""
        params = dict(
            n_estimators=n_est,
            max_depth=self.max_depth,
            learning_rate=0.1,
            # Subsampling is omitted because it has no effect unless
            # subsample_freq is at least 1.
            colsample_bytree=0.8,
            random_state=self.seed,
            verbose=-1,
            n_jobs=1,
        )
        if is_clf:
            from lightgbm import LGBMClassifier
            gbdt = LGBMClassifier(**params)
        else:
            from lightgbm import LGBMRegressor
            gbdt = LGBMRegressor(**params)
        gbdt.fit(X, y)
        return gbdt

    def _get_leaf_ids(self, gbdt, X: np.ndarray) -> np.ndarray:
        """Extract leaf indices, shape (n_samples, n_trees)."""
        leaves = gbdt.predict(X, pred_leaf=True)
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        return leaves

    def _mode_features(
        self,
        leaf_train: np.ndarray,
        leaf_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
        if self.mode == "confidence_feature":
            return self._confidence_features(leaf_train, leaf_test)
        return self._embedding_knn_features(leaf_train, leaf_test)

    def _cross_fitted_leaf_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        n_est: int,
        is_clf: bool,
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
        """Create OOF train features and a fold-feature ensemble for test.

        Occupancy and neighbor statistics are computed separately inside each
        fold model's leaf space. All train/test feature rows participate in
        those unlabeled statistics, but a training row receives output only
        from a GBDT fitted without that row or its target.
        """
        splits, split_info = self._crossfit_splits(y_train, is_clf)
        n_added = 3 if self.mode == "confidence_feature" else 2
        train_features = np.empty((len(X_train), n_added), dtype=float)
        test_feature_sum = np.zeros((len(X_test), n_added), dtype=float)
        train_coverage = np.zeros(len(X_train), dtype=np.int64)
        n_trees_per_fold: list[int] = []
        mode_info_per_fold: list[dict] = []
        feature_names: list[str] | None = None

        for fit_idx, held_out_idx in splits:
            gbdt = self._build_and_fit_gbdt(
                X_train[fit_idx], y_train[fit_idx], n_est, is_clf,
            )
            leaf_train = self._get_leaf_ids(gbdt, X_train)
            leaf_test = self._get_leaf_ids(gbdt, X_test)
            fold_train, fold_test, fold_names, fold_info = self._mode_features(
                leaf_train, leaf_test,
            )
            if feature_names is None:
                feature_names = fold_names
            elif feature_names != fold_names:
                raise RuntimeError("M18 fold feature names are inconsistent")

            train_features[held_out_idx] = fold_train[held_out_idx]
            train_coverage[held_out_idx] += 1
            test_feature_sum += fold_test
            n_trees_per_fold.append(int(leaf_train.shape[1]))
            mode_info_per_fold.append(fold_info)

        if not np.all(train_coverage == 1):
            raise RuntimeError(
                "M18 cross-fitting must assign every training row exactly once"
            )
        if feature_names is None:
            raise RuntimeError("M18 cross-fitting produced no folds")

        test_features = test_feature_sum / len(splits)
        crossfit_info = {
            **split_info,
            "crossfit_train_coverage_min": int(train_coverage.min()),
            "crossfit_train_coverage_max": int(train_coverage.max()),
            "crossfit_test_aggregation": "mean_of_fold_derived_features",
            "crossfit_leaf_space_policy": "fold_local_no_cross_model_leaf_ids",
            "crossfit_occupancy_rows": "all_train_features_plus_all_test_features",
            "n_trees_per_fold": n_trees_per_fold,
        }
        if self.mode == "confidence_feature":
            constant_flags = [
                bool(info.get("confidence_features_constant", False))
                for info in mode_info_per_fold
            ]
            crossfit_info["confidence_features_constant"] = all(constant_flags)
            crossfit_info["confidence_features_constant_folds"] = int(
                sum(constant_flags)
            )
        else:
            # These values depend only on split sizes and requested k, so they
            # must agree across folds.
            first = mode_info_per_fold[0]
            if any(info != first for info in mode_info_per_fold[1:]):
                raise RuntimeError("M18 embedding metadata differs across folds")
            crossfit_info.update(first)

        return train_features, test_features, feature_names, crossfit_info

    # ----- mode implementations ----------------------------------------------

    def _confidence_features(
        self,
        leaf_train: np.ndarray,
        leaf_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
        """Compute 3 per-sample confidence features from leaf occupancy."""
        n_train = len(leaf_train)
        n_test = len(leaf_test)
        n_total = n_train + n_test
        n_trees = leaf_train.shape[1]

        leaf_all = np.vstack([leaf_train, leaf_test])
        is_train = np.zeros(n_total, dtype=bool)
        is_train[:n_train] = True

        avg_train_frac = np.zeros(n_total)
        avg_total_size = np.zeros(n_total)
        minority_count = np.zeros(n_total)

        for t in range(n_trees):
            col = leaf_all[:, t]
            unique_leaves, inverse = np.unique(col, return_inverse=True)
            n_leaves = len(unique_leaves)

            # count train and total samples per leaf
            train_per_leaf = np.zeros(n_leaves)
            total_per_leaf = np.zeros(n_leaves)
            np.add.at(train_per_leaf, inverse[:n_train], 1)
            np.add.at(total_per_leaf, inverse, 1)

            # per-sample statistics via inverse mapping
            tf = train_per_leaf[inverse] / total_per_leaf[inverse]
            total = total_per_leaf[inverse]

            avg_train_frac += tf
            avg_total_size += total

            # "other" fraction: for train samples it is test_frac (1-tf),
            # for test samples it is train_frac (tf)
            other_frac = np.where(is_train, 1.0 - tf, tf)
            minority_count += (other_frac < self.minority_threshold).astype(float)

        avg_train_frac /= n_trees
        avg_total_size /= n_trees
        minority_ratio = minority_count / n_trees

        # stack features: (n_total, 3)
        feats = np.column_stack([minority_ratio, avg_train_frac, avg_total_size])
        feat_names = ["leaf_minority_ratio", "leaf_avg_train_frac", "leaf_avg_size"]

        # Diagnostic: flag pathological cases where all three features are
        # constant across samples. Happens when every tree produces leaves
        # with identical train/test composition (e.g. a dataset that is
        # perfectly separable by one categorical with symmetric split counts).
        eps = 1e-12
        all_constant = bool(
            feats[:, 0].std() < eps
            and feats[:, 1].std() < eps
            and feats[:, 2].std() < eps
        )
        info = {"confidence_features_constant": all_constant}

        return feats[:n_train], feats[n_train:], feat_names, info

    def _embedding_knn_features(
        self,
        leaf_train: np.ndarray,
        leaf_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
        """Compute 2 per-sample features from k-NN in leaf-embedding space.

        Returns
        -------
        train_feats, test_feats : np.ndarray
            New per-sample features for each split.
        feat_names : list[str]
            Names for the 2 new features.
        info : dict
            Diagnostic info (e.g. effective k after small-data bounding).
        """
        n_train = len(leaf_train)
        n_test = len(leaf_test)
        # The cross-split kNN queries each split against the other split's
        # tree, so self can never appear in the neighbor list. Bound is
        # n_test / n_train, not n_test - 1 / n_train - 1.
        k = max(1, min(self.k, n_test, n_train))

        # --- distance to other split ---
        nn_test = NearestNeighbors(
            n_neighbors=k, metric="hamming", algorithm="brute", n_jobs=1,
        )
        nn_test.fit(leaf_test)
        dist_tr2te, _ = nn_test.kneighbors(leaf_train)  # (n_train, k)

        nn_train = NearestNeighbors(
            n_neighbors=k, metric="hamming", algorithm="brute", n_jobs=1,
        )
        nn_train.fit(leaf_train)
        dist_te2tr, _ = nn_train.kneighbors(leaf_test)  # (n_test, k)

        mean_dist_train = dist_tr2te.mean(axis=1)  # (n_train,)
        mean_dist_test = dist_te2tr.mean(axis=1)    # (n_test,)

        # --- cross-split neighbour fraction (vectorized) ---
        leaf_all = np.vstack([leaf_train, leaf_test])
        n_total = n_train + n_test
        is_train = np.zeros(n_total, dtype=bool)
        is_train[:n_train] = True

        # Request k_combined + 1 neighbors to drop the possible self entry.
        # The upper bound is therefore n_total - 1, not n_total - 2.
        k_combined = max(1, min(k, n_total - 1))
        nn_all = NearestNeighbors(
            n_neighbors=k_combined + 1,  # +1 to exclude self
            metric="hamming", algorithm="brute", n_jobs=1,
        )
        nn_all.fit(leaf_all)
        _, idx_all = nn_all.kneighbors(leaf_all)  # (n_total, k+1)

        # Identify the self-position in each row's neighbor list.
        # With NearestNeighbors, at most one neighbor index can equal
        # the row index, so each row has at most one True in self_mask.
        sample_idx = np.arange(n_total)[:, None]                # (n_total, 1)
        not_self_mask = idx_all != sample_idx                   # (n_total, k+1)

        # Stable-sort rows so the (≤1) self-entry moves to the end. After
        # sorting, the first k_combined columns hold non-self neighbors.
        sort_order = np.argsort(~not_self_mask, axis=1, kind="stable")
        sorted_idx = np.take_along_axis(idx_all, sort_order, axis=1)
        sorted_valid = np.take_along_axis(not_self_mask, sort_order, axis=1)
        chosen = sorted_idx[:, :k_combined]                     # (n_total, k_combined)
        chosen_valid = sorted_valid[:, :k_combined]             # (n_total, k_combined)

        # Cross == neighbor belongs to the other split than the source sample.
        neighbor_is_train = is_train[chosen]                    # (n_total, k_combined)
        own_is_train = is_train[:, None]                        # (n_total, 1)
        cross = (neighbor_is_train != own_is_train) & chosen_valid

        n_valid = chosen_valid.sum(axis=1)
        cross_count = cross.sum(axis=1).astype(float)
        cross_frac = np.zeros(n_total)
        nz = n_valid > 0
        cross_frac[nz] = cross_count[nz] / n_valid[nz]

        train_feats = np.column_stack([mean_dist_train, cross_frac[:n_train]])
        test_feats = np.column_stack([mean_dist_test, cross_frac[n_train:]])
        feat_names = ["leaf_hamming_dist_to_other", "leaf_cross_split_frac"]
        info = {"k_requested": int(self.k), "k_effective": int(k),
                "k_combined": int(k_combined)}

        return train_feats, test_feats, feat_names, info

    # ----- cumshift feature selection ---------------------------------------

    def _select_features_cumshift(
        self, X_tr_arr: np.ndarray, X_te_arr: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Select smallest feature set capturing ``shift_ratio`` of total KS shift.

        Same algorithm as M16/M17 cumshift mode. NaN values are skipped
        per column when computing the KS statistic.

        Returns
        -------
        selected_idx : np.ndarray
            Column indices to keep, sorted by descending KS shift.
        info : dict
            Diagnostic info for metadata (target ratio, achieved ratio,
            number of features kept).
        """
        from scipy.stats import ks_2samp

        n_f = X_tr_arr.shape[1]
        ks_stats = np.zeros(n_f)
        for j in range(n_f):
            tr = X_tr_arr[:, j]
            te = X_te_arr[:, j]
            tr_v = tr[~np.isnan(tr)]
            te_v = te[~np.isnan(te)]
            if len(tr_v) > 1 and len(te_v) > 1:
                stat, _ = ks_2samp(tr_v, te_v)
                ks_stats[j] = stat

        order = np.argsort(ks_stats)[::-1]
        cumsum = np.cumsum(ks_stats[order])
        total = cumsum[-1]

        if total < 1e-6:
            # No measurable shift → use all features
            selected_idx = np.arange(n_f)
            info = {
                "shift_ratio_target": self.shift_ratio,
                "shift_ratio_achieved": 0.0,
                "n_features_selected": n_f,
            }
            return selected_idx, info

        n_keep = int(np.searchsorted(cumsum, total * self.shift_ratio)) + 1
        n_keep = max(n_keep, 3)       # floor: keep at least 3 features
        n_keep = min(n_keep, n_f)     # cap at n_features
        selected_idx = order[:n_keep]
        achieved = float(cumsum[n_keep - 1] / total)
        info = {
            "shift_ratio_target": self.shift_ratio,
            "shift_ratio_achieved": achieved,
            "n_features_selected": int(n_keep),
        }
        return selected_idx, info

    # ----- main entry --------------------------------------------------------

    def fit_transform(self, X_train, X_test, y_train):
        from src.methods.tier1 import _label_encode_cats

        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        feature_cols = list(X_tr_enc.columns)

        # --- Optional cumshift feature selection for the auxiliary GBDT ---
        cumshift_info: dict = {}
        if self.feature_subset == "cumshift":
            selected_idx, cumshift_info = self._select_features_cumshift(
                X_tr_arr, X_te_arr,
            )
            cumshift_info["selected_features"] = [
                feature_cols[j] for j in selected_idx
            ]
            X_tr_for_gbdt = X_tr_arr[:, selected_idx]
            X_te_for_gbdt = X_te_arr[:, selected_idx]
        else:
            X_tr_for_gbdt = X_tr_arr
            X_te_for_gbdt = X_te_arr

        n_train = len(X_tr_arr)
        y_train = np.asarray(y_train)
        is_clf, task_type_source = self._resolve_task_type(y_train)
        n_est = self._effective_n_estimators(n_train)

        # NaN-aware cross-fitting: every auxiliary model uses only its fold's
        # fit rows and targets. Leaf IDs stay local to that model. Only the
        # fixed-dimensional derived features are combined across folds.
        new_tr, new_te, feat_names, crossfit_info = (
            self._cross_fitted_leaf_features(
                X_tr_for_gbdt,
                X_te_for_gbdt,
                y_train,
                n_est,
                is_clf,
            )
        )

        # Append leaf-derived features to the full original feature matrix.
        # (NaN preserved in the original columns. Downstream pipeline imputes).
        X_tr_out = np.column_stack([X_tr_arr, new_tr])
        X_te_out = np.column_stack([X_te_arr, new_te])
        feature_names = feature_cols + feat_names

        metadata = {
            "m18_mode": self.mode,
            "m18_n_trees": int(round(np.mean(crossfit_info["n_trees_per_fold"]))),
            "m18_n_features_added": len(feat_names),
            "m18_is_classification": is_clf,
            "m18_task_type_source": task_type_source,
            "m18_effective_n_estimators": n_est,
            "m18_feature_subset": self.feature_subset,
            "m18_n_features_for_gbdt": X_tr_for_gbdt.shape[1],
            "m18_n_features_total": X_tr_arr.shape[1],
        }
        if cumshift_info:
            metadata.update(
                {f"m18_{key}": v for key, v in cumshift_info.items()}
            )
        metadata.update(
            {f"m18_{key}": value for key, value in crossfit_info.items()}
        )

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# M20: Cluster-Then-Adapt
# ---------------------------------------------------------------------------

class ClusterMembershipFE(TestTimeMethod):
    """
    M20: Cluster-Then-Adapt.

    Clusters the combined train+test feature space with KMeans, then
    provides the downstream model with per-cluster shift diagnostics
    and/or local importance weights.

    The core insight: distribution shift is spatially heterogeneous.
    Some regions are heavily shifted, others barely. Global reweighting
    treats all regions equally. This method makes the adaptation local.

    Distance-space construction
    ---------------------------
    The KMeans distance space is built separately from the output matrix:

      1. Categoricals are jointly frequency-encoded.
      2. Optional cumshift selection retains the smallest feature set capturing
         ``shift_ratio`` of total KS shift, with a floor of three features.
      3. Missing values are filled with combined train/test medians.
      4. Joint standardization prevents high-magnitude features from dominating
         Euclidean distance.

    Output features are appended to the original label-encoded matrix to retain
    the same downstream interface:

    Modes (``return_as``)
    ---------------------
    feature
        Appends 3 features:
            - cluster_relative_size      (continuous density: total_c / N)
            - cluster_train_frac         (n_train_c / (n_train_c + n_test_c))
            - cluster_dist_to_centroid   (Euclidean dist in scaled KMeans space)
    weight
        Per-cluster importance weights for train samples (upweights train in
        test-heavy clusters). Also appends ``cluster_relative_size`` as a
        single feature so the model receives cluster information.
    both
        3 features + sample weights.

    Cluster purity filter
    ---------------------
    Per-cluster importance weights are unreliable when one split has little or
    no representation in a cluster. Fully pure clusters receive a neutral
    weight. Mixed-extreme clusters are downweighted. Set
    ``purity_threshold=0.0`` to disable the mixed-extreme filter.

    The raw KMeans ``cluster_id`` is not included in output features because
    label ordering is arbitrary. Relative size, train fraction and centroid
    distance provide label-invariant cluster descriptors. Cluster IDs remain
    in diagnostic metadata.

    Parameters
    ----------
    n_clusters : int, default 10
        Target number of KMeans clusters. Auto-reduced for tiny datasets via
        ``_effective_k`` (floor n_total // 10, min 2).
    return_as : {"feature", "weight", "both"}, default "both"
        Output mode (see above).
    weight_clip_strategy : str, default "normalize"
        Passed to ``_clip_weights``. Controls how raw n_te/n_tr ratios are
        capped/normalized before being returned as sample weights.
    feature_subset : {None, "cumshift"}, default None
        If "cumshift", restrict the KMeans distance computation to the
        smallest feature subset whose KS statistics cumulatively cover
        ``shift_ratio`` of total KS shift (floor 3). Output features are
        always appended to the full original feature matrix.
    shift_ratio : float, default 0.9
        Target cumshift ratio (only used when feature_subset == "cumshift").
    purity_threshold : float, default 0.05
        Per-cluster purity filter (see above). Must be in [0.0, 0.5).
    seed : int, default 42
        Random seed for KMeans initialization.
    """

    def __init__(
        self,
        n_clusters: int = 10,
        return_as: str = "both",
        weight_clip_strategy: str = "normalize",
        feature_subset: str | None = None,
        shift_ratio: float = 0.9,
        purity_threshold: float = 0.05,
        pure_train_weight: float = 0.1,
        seed: int = 42,
    ):
        if return_as not in ("feature", "weight", "both"):
            raise ValueError(f"Unknown return_as '{return_as}'")
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
        if feature_subset not in (None, "cumshift"):
            raise ValueError(
                f"feature_subset must be None or 'cumshift', got '{feature_subset}'"
            )
        if not (0.0 < shift_ratio <= 1.0):
            raise ValueError(f"shift_ratio must be in (0, 1], got {shift_ratio}")
        if not (0.0 <= purity_threshold < 0.5):
            raise ValueError(
                f"purity_threshold must be in [0.0, 0.5), got {purity_threshold}"
            )
        if not (0.0 < pure_train_weight <= 1.0):
            raise ValueError(
                f"pure_train_weight must be in (0.0, 1.0], got {pure_train_weight}"
            )
        self.n_clusters = n_clusters
        self.return_as = return_as
        self.weight_clip_strategy = weight_clip_strategy
        self.feature_subset = feature_subset
        self.shift_ratio = shift_ratio
        self.purity_threshold = purity_threshold
        self.pure_train_weight = pure_train_weight
        self.seed = seed

    @property
    def name(self) -> str:
        cs = "_cumshift" if self.feature_subset == "cumshift" else ""
        return f"cluster_membership_k{self.n_clusters}_{self.return_as}{cs}"

    @property
    def may_emit_sample_weight(self) -> bool:
        return self.return_as in ("weight", "both")

    # uses_labels remains False because clustering uses only X.

    def _effective_k(self, n_total: int) -> int:
        """Reduce cluster count for small datasets."""
        if n_total < self.n_clusters * 10:
            return max(2, n_total // 10)
        return self.n_clusters

    def _select_features_cumshift(
        self, X_tr_arr: np.ndarray, X_te_arr: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Select smallest feature set capturing ``shift_ratio`` of total KS shift.

        Same algorithm as M16/M17/M18/M19 cumshift mode (KS-only). NaN values
        are skipped per column when computing the KS statistic. Operates on
        the freq-encoded matrix so KS on former cat cols is mathematically
        sensible (numeric frequency comparison rather than label-code
        comparison).
        """
        from scipy.stats import ks_2samp

        n_f = X_tr_arr.shape[1]
        ks_stats = np.zeros(n_f)
        for j in range(n_f):
            tr = X_tr_arr[:, j]
            te = X_te_arr[:, j]
            tr_v = tr[~np.isnan(tr)]
            te_v = te[~np.isnan(te)]
            if len(tr_v) > 1 and len(te_v) > 1:
                stat, _ = ks_2samp(tr_v, te_v)
                ks_stats[j] = stat

        order = np.argsort(ks_stats)[::-1]
        cumsum = np.cumsum(ks_stats[order])
        total = cumsum[-1] if len(cumsum) else 0.0

        if total < 1e-6:
            selected_idx = np.arange(n_f)
            info = {
                "shift_ratio_target": self.shift_ratio,
                "shift_ratio_achieved": 0.0,
                "n_features_selected": n_f,
            }
            return selected_idx, info

        n_keep = int(np.searchsorted(cumsum, total * self.shift_ratio)) + 1
        n_keep = max(n_keep, 3)
        n_keep = min(n_keep, n_f)
        selected_idx = order[:n_keep]
        info = {
            "shift_ratio_target": self.shift_ratio,
            "shift_ratio_achieved": float(cumsum[n_keep - 1] / total),
            "n_features_selected": int(n_keep),
        }
        return selected_idx, info

    def fit_transform(self, X_train, X_test, y_train):
        from sklearn.cluster import KMeans
        from src.methods.tier1 import _label_encode_cats
        from src.methods.preprocessing import get_cat_cols

        # Step 1: use the original label-encoded matrix as the output base.
        # Consensus features are appended to it. The downstream pipeline
        # imputer and scaler handle any remaining NaN in this matrix.
        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        feature_cols = list(X_tr_enc.columns)

        n_train = len(X_tr_arr)
        n_test = len(X_te_arr)
        n_total = n_train + n_test

        # Step 2a: build the KMeans distance space separately.
        # Joint frequency encoding retains one column per categorical feature.
        # Numerics pass through.
        cat_cols = get_cat_cols(X_train)
        X_tr_km_raw, X_te_km_raw = _frequency_encode_joint(
            X_train, X_test, cat_cols,
        )

        # --- Step 2b: Optional cumshift feature selection on the freq-encoded
        # raw matrix (cumshift KS_2samp ignores NaN per column).
        cumshift_info: dict = {}
        if self.feature_subset == "cumshift":
            selected_idx, cumshift_info = self._select_features_cumshift(
                X_tr_km_raw, X_te_km_raw,
            )
            cumshift_info["selected_features"] = [
                feature_cols[j] for j in selected_idx
            ]
            X_tr_for_km = X_tr_km_raw[:, selected_idx]
            X_te_for_km = X_te_km_raw[:, selected_idx]
        else:
            X_tr_for_km = X_tr_km_raw
            X_te_for_km = X_te_km_raw

        # --- Step 2c: Median-impute (joint train+test). Freq-encoded cat cols
        # have no NaN by construction (NaN → "nan" category → frequency value),
        # so this only touches numeric columns.
        X_tr_for_km, X_te_for_km = _impute_median_joint(
            X_tr_for_km, X_te_for_km,
        )

        # Step 2d: fit StandardScaler jointly so frequency-encoded categorical
        # features and numerical features use a common scale.
        scaler = StandardScaler()
        scaler.fit(np.vstack([X_tr_for_km, X_te_for_km]))
        X_tr_for_km = scaler.transform(X_tr_for_km)
        X_te_for_km = scaler.transform(X_te_for_km)

        # --- Step 3: KMeans on combined scaled matrix.
        effective_k = self._effective_k(n_total)
        kmeans = KMeans(
            n_clusters=effective_k, random_state=self.seed, n_init=10,
        )
        combined_scaled = np.vstack([X_tr_for_km, X_te_for_km])
        kmeans.fit(combined_scaled)
        labels_train = kmeans.predict(X_tr_for_km)
        labels_test = kmeans.predict(X_te_for_km)

        # --- Step 4: Per-cluster statistics.
        train_frac_per_cluster = np.zeros(effective_k)
        weight_per_cluster = np.zeros(effective_k)
        relative_size_per_cluster = np.zeros(effective_k)
        is_fully_pure = np.zeros(effective_k, dtype=bool)
        cluster_sizes = {}

        for c in range(effective_k):
            n_tr_c = int((labels_train == c).sum())
            n_te_c = int((labels_test == c).sum())
            total_c = n_tr_c + n_te_c
            train_frac_per_cluster[c] = n_tr_c / (total_c + 1e-10)
            relative_size_per_cluster[c] = total_c / n_total
            # Weight = test/train ratio. Fallback to 1.0 for empty clusters.
            if n_tr_c == 0 or n_te_c == 0:
                weight_per_cluster[c] = 1.0  # neutral weight
                is_fully_pure[c] = True
            else:
                weight_per_cluster[c] = n_te_c / n_tr_c
            cluster_sizes[c] = {"n_train": n_tr_c, "n_test": n_te_c}

        # Step 5: clusters present in only one split retain the neutral weight
        # from Step 4. Clusters present in both splits receive
        # ``pure_train_weight`` when their training fraction is extreme because
        # the ratio is estimated from few rows in the smaller split. The two
        # cases are reported separately.
        n_pure_filtered = 0
        n_pure_train_clusters = int(
            (is_fully_pure & (train_frac_per_cluster > 0.5)).sum()
        )
        n_pure_test_clusters = int(
            (is_fully_pure & (train_frac_per_cluster < 0.5)).sum()
        )
        if self.purity_threshold > 0.0:
            lo = self.purity_threshold
            hi = 1.0 - self.purity_threshold
            extreme_mask = (
                (train_frac_per_cluster < lo) | (train_frac_per_cluster > hi)
            )
            mixed_extreme_mask = extreme_mask & ~is_fully_pure
            weight_per_cluster[mixed_extreme_mask] = self.pure_train_weight
            n_pure_filtered = int(mixed_extreme_mask.sum())

        # Step 6: build output features and append them to the original
        # label-encoded matrix. The downstream pipeline imputer and scaler handle
        # any NaN that remains.
        sample_weights = None

        if self.return_as in ("feature", "both"):
            rel_size_train = relative_size_per_cluster[labels_train]
            rel_size_test = relative_size_per_cluster[labels_test]
            cluster_tf_train = train_frac_per_cluster[labels_train]
            cluster_tf_test = train_frac_per_cluster[labels_test]
            dist_train = np.linalg.norm(
                X_tr_for_km - kmeans.cluster_centers_[labels_train], axis=1,
            )
            dist_test = np.linalg.norm(
                X_te_for_km - kmeans.cluster_centers_[labels_test], axis=1,
            )

            X_tr_out = np.column_stack([
                X_tr_arr, rel_size_train, cluster_tf_train, dist_train,
            ])
            X_te_out = np.column_stack([
                X_te_arr, rel_size_test, cluster_tf_test, dist_test,
            ])
            feature_names = feature_cols + [
                "cluster_relative_size",
                "cluster_train_frac",
                "cluster_dist_to_centroid",
            ]
        else:
            # Weight-only mode appends relative cluster size as a
            # label-invariant reference feature.
            rel_size_train = relative_size_per_cluster[labels_train]
            rel_size_test = relative_size_per_cluster[labels_test]
            X_tr_out = np.column_stack([X_tr_arr, rel_size_train])
            X_te_out = np.column_stack([X_te_arr, rel_size_test])
            feature_names = feature_cols + ["cluster_relative_size"]

        # --- Step 7: Sample weights.
        if self.return_as in ("weight", "both"):
            raw_weights = weight_per_cluster[labels_train]
            sample_weights = _clip_weights(
                raw_weights, strategy=self.weight_clip_strategy,
            )

        # --- Step 8: Metadata.
        metadata = {
            "m20_effective_k": effective_k,
            "m20_return_as": self.return_as,
            "m20_feature_subset": self.feature_subset,
            "m20_kmeans_encoding": "frequency",
            "m20_n_cat_cols_encoded": len(cat_cols),
            "m20_kmeans_scaled": True,
            "m20_n_features_for_kmeans": int(X_tr_for_km.shape[1]),
            "m20_n_features_total": int(X_tr_arr.shape[1]),
            "m20_cluster_sizes": cluster_sizes,
            "m20_cluster_train_fracs": {
                c: float(train_frac_per_cluster[c]) for c in range(effective_k)
            },
            "m20_cluster_relative_sizes": {
                c: float(relative_size_per_cluster[c]) for c in range(effective_k)
            },
            "m20_n_pure_clusters_filtered": n_pure_filtered,
            "m20_n_pure_train_clusters": n_pure_train_clusters,
            "m20_n_pure_test_clusters": n_pure_test_clusters,
            "m20_purity_threshold": float(self.purity_threshold),
            "m20_pure_train_weight": float(self.pure_train_weight),
        }
        if sample_weights is not None:
            metadata["m20_mean_weight"] = float(sample_weights.mean())
            metadata["m20_std_weight"] = float(sample_weights.std())
        if cumshift_info:
            metadata.update({f"m20_{k}": v for k, v in cumshift_info.items()})

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            sample_weights_train=sample_weights,
            metadata=metadata,
        )


# ===========================================================================
# M19: Neighborhood Consensus Features
# ===========================================================================

def _frequency_encode_joint(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Replace each categorical column with its joint train+test frequency.

    Used by M19's kNN distance space and M20's KMeans distance space to avoid
    the additional columns from one-hot encoding while giving categorical
    columns a meaningful numeric representation. NaN values are treated as
    their own category (via ``.astype(str)`` → "nan"), same convention as
    ``_label_encode_cats``. Numeric columns pass through unchanged.
    """
    X_tr = X_train_df.copy()
    X_te = X_test_df.copy()
    for col in cat_cols:
        combined = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)])
        freq_map = combined.value_counts(normalize=True).to_dict()
        X_tr[col] = X_tr[col].astype(str).map(freq_map).astype(float)
        X_te[col] = X_te[col].astype(str).map(freq_map).astype(float)
    return X_tr.values.astype(float), X_te.values.astype(float)


def _impute_median_joint(
    X_train: np.ndarray, X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Impute NaN values using statistics from the joint train and test matrix.

    M19 uses the imputed values for its k-NN distance space and M20 uses them
    for its KMeans distance space. Their generated features are appended to the
    original matrices, and the downstream pipeline handles remaining NaN values.

    If neither matrix contains NaN, the function returns the input arrays
    unchanged. Columns that contain only NaN values use 0.0.
    """
    if not (np.isnan(X_train).any() or np.isnan(X_test).any()):
        return X_train, X_test
    joint = np.vstack([X_train, X_test])
    medians = np.nanmedian(joint, axis=0)
    # Columns that are entirely NaN get a 0 fallback (rare).
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_train_imp = np.where(np.isnan(X_train), medians, X_train)
    X_test_imp = np.where(np.isnan(X_test), medians, X_test)
    return X_train_imp, X_test_imp


class NeighborhoodConsensusFeatures(TestTimeMethod):
    """
    M19: Neighborhood Consensus Features (two-pass test-time method).

    Pass 1: fit a lightweight internal model on (X_train, y_train). Compute
    out-of-fold (OOF) predictions on train and standard predictions on test.
    Reduce predictions to a per-sample scalar:
        - binary classification: P(y = positive class)
        - multiclass: max-class probability (model confidence)
        - regression: y_pred

    Build a k-NN graph within each split (train neighbors come from the train
    split, test neighbors from the test split: never cross splits at this
    stage). Then compute and append the following per-sample features:

        nbr_consensus_self    : the sample's own pass-1 scalar prediction
        nbr_consensus_mean    : weighted mean of neighbors' scalars
        nbr_consensus_std     : weighted std  of neighbors' scalars
        nbr_consensus_diff    : self - mean   (signed outlier signal)
        nbr_consensus_agree   : (classification only) unweighted fraction of
                                neighbors whose argmax class equals self's

    The downstream pipeline model in pass 2 then trains and predicts on
    (original features + consensus features). Because the features are
    constructed in a way that is symmetric across train and test (OOF on
    train, refit-and-predict on test), the model can learn to use them
    consistently.

    Parameters
    ----------
    k : int, default 10
        Neighborhood size for the kNN graph (within each split).
    weighting : {"distance", "uniform"}, default "distance"
        How to weight neighbors when computing the weighted mean/std.
    n_folds : int, default 5
        Number of folds for cross-validated predictions on train.
        Auto-reduced for tiny datasets.
    feature_subset : {None, "cumshift"}, default None
        If "cumshift", restrict the kNN distance computation to the smallest
        feature subset whose KS statistics cumulatively cover ``shift_ratio``
        of the total KS shift (floor 3). Same algorithm used by M16/M17/M18.
        Output features are always appended to the full original feature
        matrix.
    shift_ratio : float, default 0.9
        Target cumshift ratio (only used when feature_subset == "cumshift").
    internal_n_estimators : int, default 100
        Number of trees in the internal LightGBM pass-1 model.
    internal_max_depth : int, default 6
        Max depth of the internal LightGBM (kept moderate to prevent OOF
        overfit).
    seed : int, default 42
        Seed threaded through internal model + folds.
    """

    def __init__(
        self,
        k: int = 10,
        weighting: str = "distance",
        n_folds: int = 5,
        feature_subset: str | None = None,
        shift_ratio: float = 0.9,
        internal_n_estimators: int = 100,
        internal_max_depth: int = 6,
        seed: int = 42,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if weighting not in ("distance", "uniform"):
            raise ValueError(
                f"weighting must be 'distance' or 'uniform', got '{weighting}'"
            )
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        if feature_subset not in (None, "cumshift"):
            raise ValueError(
                f"feature_subset must be None or 'cumshift', got '{feature_subset}'"
            )
        if not (0.0 < shift_ratio <= 1.0):
            raise ValueError(f"shift_ratio must be in (0, 1], got {shift_ratio}")
        if internal_n_estimators < 1:
            raise ValueError(
                f"internal_n_estimators must be >= 1, got {internal_n_estimators}"
            )
        if internal_max_depth < 1:
            raise ValueError(
                f"internal_max_depth must be >= 1, got {internal_max_depth}"
            )
        self.k = k
        self.weighting = weighting
        self.n_folds = n_folds
        self.feature_subset = feature_subset
        self.shift_ratio = shift_ratio
        self.internal_n_estimators = internal_n_estimators
        self.internal_max_depth = internal_max_depth
        self.seed = seed

    @property
    def name(self) -> str:
        cs = "_cumshift" if self.feature_subset == "cumshift" else ""
        # Distance weighting abbreviated to "dist" to match other tier2 names.
        w = "dist" if self.weighting == "distance" else "uniform"
        return f"neighbor_consensus_k{self.k}_{w}{cs}"

    @property
    def uses_labels(self) -> bool:
        return True  # internal pass-1 model trains on y_train

    # ----- internal helpers -------------------------------------------------

    @staticmethod
    def _is_classification(y_train: np.ndarray) -> bool:
        """Infer the task type when no MethodContext is bound."""
        unique_y = np.unique(y_train)
        if y_train.dtype.kind in "biuO":
            return True
        if len(unique_y) <= 50 and np.allclose(unique_y, np.round(unique_y)):
            return True
        return False

    def _resolve_task_type(self, y_train: np.ndarray) -> tuple[bool, str]:
        """Use the dataset contract when available. Retain direct-call compatibility."""
        if self.method_context is not None:
            return self.method_context.task_type == "classification", "context"
        return self._is_classification(y_train), "fallback_target_heuristic"

    def _build_internal_model(self, is_clf: bool, n_classes: int):
        """Build a lightweight LightGBM for pass-1 predictions."""
        params = dict(
            n_estimators=self.internal_n_estimators,
            max_depth=self.internal_max_depth,
            learning_rate=0.1,
            colsample_bytree=0.8,
            min_data_in_leaf=5,
            random_state=self.seed,
            verbose=-1,
            n_jobs=1,
        )
        if is_clf:
            from lightgbm import LGBMClassifier
            if n_classes > 2:
                params["num_class"] = n_classes
            return LGBMClassifier(**params)
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)

    def _effective_n_folds(self, n_train: int, y_train: np.ndarray, is_clf: bool) -> int:
        """Reduce fold count for tiny datasets so each fold is non-empty."""
        n_folds = min(self.n_folds, max(2, n_train // 2))
        if is_clf:
            # Need at least n_folds samples per class for StratifiedKFold.
            _, counts = np.unique(y_train, return_counts=True)
            min_class = int(counts.min())
            n_folds = min(n_folds, max(2, min_class))
        return n_folds

    def _select_features_cumshift(
        self, X_tr_arr: np.ndarray, X_te_arr: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Select smallest feature set capturing ``shift_ratio`` of total KS shift.

        Same algorithm as M16/M17/M18 cumshift mode. NaN values are skipped
        per column when computing the KS statistic.
        """
        from scipy.stats import ks_2samp

        n_f = X_tr_arr.shape[1]
        ks_stats = np.zeros(n_f)
        for j in range(n_f):
            tr = X_tr_arr[:, j]
            te = X_te_arr[:, j]
            tr_v = tr[~np.isnan(tr)]
            te_v = te[~np.isnan(te)]
            if len(tr_v) > 1 and len(te_v) > 1:
                stat, _ = ks_2samp(tr_v, te_v)
                ks_stats[j] = stat

        order = np.argsort(ks_stats)[::-1]
        cumsum = np.cumsum(ks_stats[order])
        total = cumsum[-1] if len(cumsum) else 0.0

        if total < 1e-6:
            selected_idx = np.arange(n_f)
            info = {
                "shift_ratio_target": self.shift_ratio,
                "shift_ratio_achieved": 0.0,
                "n_features_selected": n_f,
            }
            return selected_idx, info

        n_keep = int(np.searchsorted(cumsum, total * self.shift_ratio)) + 1
        n_keep = max(n_keep, 3)
        n_keep = min(n_keep, n_f)
        selected_idx = order[:n_keep]
        info = {
            "shift_ratio_target": self.shift_ratio,
            "shift_ratio_achieved": float(cumsum[n_keep - 1] / total),
            "n_features_selected": int(n_keep),
        }
        return selected_idx, info

    def _pass1_predictions(
        self,
        X_train_arr: np.ndarray,
        X_test_arr: np.ndarray,
        y_train: np.ndarray,
        is_clf: bool,
        n_classes: int,
        n_folds: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Compute pass-1 scalar predictions for train (OOF) and test (refit).

        Returns
        -------
        scalar_train, scalar_test : np.ndarray, shape (n,)
            Per-sample scalar prediction (P(y=1) binary, max-prob multiclass,
            y_pred regression).
        argmax_train, argmax_test : np.ndarray | None, shape (n,)
            Per-sample argmax class label (classification only. None for
            regression).
        """
        from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold

        n_train = len(X_train_arr)

        if is_clf:
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=self.seed,
            )
            try:
                oof_proba = cross_val_predict(
                    self._build_internal_model(is_clf=True, n_classes=n_classes),
                    X_train_arr, y_train,
                    cv=cv, method="predict_proba", n_jobs=1,
                )
            except ValueError:
                # Stratification can fail for very rare classes. Fall back
                # to plain KFold.
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
                oof_proba = cross_val_predict(
                    self._build_internal_model(is_clf=True, n_classes=n_classes),
                    X_train_arr, y_train,
                    cv=cv, method="predict_proba", n_jobs=1,
                )

            # Refit on full train and predict on test.
            clf = self._build_internal_model(is_clf=True, n_classes=n_classes)
            clf.fit(X_train_arr, y_train)
            test_proba = clf.predict_proba(X_test_arr)

            # Reduce probabilities to a per-sample scalar.
            if n_classes == 2:
                # Pick column for the larger class label as "positive".
                pos_idx = 1 if oof_proba.shape[1] >= 2 else 0
                scalar_train = oof_proba[:, pos_idx]
                scalar_test = test_proba[:, pos_idx]
            else:
                scalar_train = oof_proba.max(axis=1)
                scalar_test = test_proba.max(axis=1)

            argmax_train = np.argmax(oof_proba, axis=1)
            argmax_test = np.argmax(test_proba, axis=1)
            return scalar_train, scalar_test, argmax_train, argmax_test

        # Regression
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        oof_pred = cross_val_predict(
            self._build_internal_model(is_clf=False, n_classes=1),
            X_train_arr, y_train, cv=cv, n_jobs=1,
        )
        reg = self._build_internal_model(is_clf=False, n_classes=1)
        reg.fit(X_train_arr, y_train)
        test_pred = reg.predict(X_test_arr)
        return oof_pred, test_pred, None, None

    @staticmethod
    def _impute_median_joint(
        X_train: np.ndarray, X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expose the shared ``_impute_median_joint`` helper through the class."""
        return _impute_median_joint(X_train, X_test)

    def _compute_consensus_features(
        self,
        X_knn: np.ndarray,
        scalar: np.ndarray,
        argmax: np.ndarray | None,
        is_clf: bool,
    ) -> tuple[np.ndarray, list[str], int]:
        """Compute consensus features for one split (train or test).

        Within-split kNN: each sample finds its k nearest *same-split*
        neighbors (excluding itself).
        """
        n = X_knn.shape[0]
        # Effective k is min(self.k, n - 1). If n < 2, return zero-valued
        # features so the downstream feature schema is preserved.
        k_eff = max(0, min(self.k, n - 1))
        n_features_out = 5 if is_clf else 4
        feat_names = [
            "nbr_consensus_self",
            "nbr_consensus_mean",
            "nbr_consensus_std",
            "nbr_consensus_diff",
        ]
        if is_clf:
            feat_names.append("nbr_consensus_agree")

        if k_eff < 1:
            feats = np.zeros((n, n_features_out))
            feats[:, 0] = scalar  # self is always defined
            return feats, feat_names, 0

        nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", n_jobs=1)
        nn.fit(X_knn)
        distances, indices = nn.kneighbors(X_knn)
        # Drop self (first column = 0 distance to self).
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        if self.weighting == "distance":
            clipped = np.maximum(distances, 1e-4)
            weights = 1.0 / clipped
            weights = weights / weights.sum(axis=1, keepdims=True)
        else:
            weights = np.full_like(distances, 1.0 / k_eff)

        nbr_scalars = scalar[indices]                             # (n, k_eff)
        weighted_mean = (weights * nbr_scalars).sum(axis=1)       # (n,)

        # Weighted variance: E[(x - mu)^2] under the row-normalized weights.
        diffs_sq = (nbr_scalars - weighted_mean[:, None]) ** 2
        weighted_var = (weights * diffs_sq).sum(axis=1)
        # Avoid floating-point negative variance (k_eff == 1 → 0 by design).
        weighted_var = np.maximum(weighted_var, 0.0)
        weighted_std = np.sqrt(weighted_var)

        diff = scalar - weighted_mean

        feats_list = [scalar, weighted_mean, weighted_std, diff]
        if is_clf and argmax is not None:
            nbr_argmax = argmax[indices]                          # (n, k_eff)
            agree = (nbr_argmax == argmax[:, None]).mean(axis=1)
            feats_list.append(agree)

        feats = np.column_stack(feats_list)
        return feats, feat_names, k_eff

    # ----- main entry --------------------------------------------------------

    def fit_transform(self, X_train, X_test, y_train):
        from src.methods.tier1 import _label_encode_cats
        from src.methods.preprocessing import get_cat_cols
        from sklearn.preprocessing import StandardScaler

        # --- Label-encode for the internal pass-1 model. LightGBM handles
        # label-encoded categoricals natively. This matrix also becomes the
        # base for the output (consensus features are appended to it).
        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        X_tr_arr = X_tr_enc.values.astype(float)
        X_te_arr = X_te_enc.values.astype(float)
        feature_cols = list(X_tr_enc.columns)

        n_train = len(X_tr_arr)
        is_clf, task_type_source = self._resolve_task_type(y_train)
        n_classes = int(len(np.unique(y_train))) if is_clf else 1

        # --- Pass 1: OOF on train + refit-and-predict on test.
        n_folds = self._effective_n_folds(n_train, y_train, is_clf)
        scalar_train, scalar_test, argmax_train, argmax_test = self._pass1_predictions(
            X_tr_arr, X_te_arr, y_train, is_clf, n_classes, n_folds,
        )

        # Build a separate kNN distance matrix.
        # Categorical columns use joint frequency encoding, with one output column each.
        # Numerical columns are preserved.
        # This matrix is used only for cumshift selection and kNN. It never
        # reaches the downstream model.
        cat_cols = get_cat_cols(X_train)
        X_tr_knn_raw, X_te_knn_raw = _frequency_encode_joint(
            X_train, X_test, cat_cols,
        )

        # --- Optional cumshift selection on the frequency-encoded matrix.
        cumshift_info: dict = {}
        if self.feature_subset == "cumshift":
            selected_idx, cumshift_info = self._select_features_cumshift(
                X_tr_knn_raw, X_te_knn_raw,
            )
            cumshift_info["selected_features"] = [
                feature_cols[j] for j in selected_idx
            ]
            X_tr_for_knn = X_tr_knn_raw[:, selected_idx]
            X_te_for_knn = X_te_knn_raw[:, selected_idx]
        else:
            X_tr_for_knn = X_tr_knn_raw
            X_te_for_knn = X_te_knn_raw

        # --- Median-impute (joint train+test). Freq-encoded cat cols have
        # no NaN by construction (NaN → "nan" category → frequency value),
        # so this only touches numeric columns.
        X_tr_for_knn, X_te_for_knn = self._impute_median_joint(
            X_tr_for_knn, X_te_for_knn,
        )

        # Standardize with a joint fit so frequency-encoded categorical features
        # and numerical features use a common scale.
        scaler = StandardScaler()
        scaler.fit(np.vstack([X_tr_for_knn, X_te_for_knn]))
        X_tr_for_knn = scaler.transform(X_tr_for_knn)
        X_te_for_knn = scaler.transform(X_te_for_knn)

        # --- Compute consensus features for each split.
        train_feats, feat_names, k_eff_train = self._compute_consensus_features(
            X_tr_for_knn, scalar_train, argmax_train, is_clf,
        )
        test_feats, _, k_eff_test = self._compute_consensus_features(
            X_te_for_knn, scalar_test, argmax_test, is_clf,
        )

        # Append features to the full original label-encoded feature
        # matrices. Downstream pipeline's imputer + scaler handle the rest.
        X_tr_out = np.column_stack([X_tr_arr, train_feats])
        X_te_out = np.column_stack([X_te_arr, test_feats])
        feature_names = feature_cols + feat_names

        metadata = {
            "m19_k_requested": int(self.k),
            "m19_k_effective_train": int(k_eff_train),
            "m19_k_effective_test": int(k_eff_test),
            "m19_n_features_added": len(feat_names),
            "m19_n_features_for_knn": int(X_tr_for_knn.shape[1]),
            "m19_n_features_total": int(X_tr_arr.shape[1]),
            "m19_is_classification": is_clf,
            "m19_task_type_source": task_type_source,
            "m19_n_classes": n_classes,
            "m19_n_folds_used": int(n_folds),
            "m19_weighting": self.weighting,
            "m19_feature_subset": self.feature_subset,
            "m19_knn_encoding": "frequency",
            "m19_n_cat_cols_encoded": len(cat_cols),
            "m19_knn_scaled": True,
        }
        if cumshift_info:
            metadata.update({f"m19_{k}": v for k, v in cumshift_info.items()})

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            metadata=metadata,
        )


# ===========================================================================
# Registries
# ===========================================================================

# M6 variants: domain classifier × return mode × weight type
TIER2_METHODS: dict[str, callable] = {
    # --- M6: Domain Classifier as Feature ---
    "adversarial_lgbm_feature": lambda: AdversarialValidation(
        classifier="lgbm", return_as="feature"
    ),
    "adversarial_logreg_feature": lambda: AdversarialValidation(
        classifier="logreg", return_as="feature"
    ),
    "adversarial_rf_feature": lambda: AdversarialValidation(
        classifier="rf", return_as="feature"
    ),
    # --- M6: Domain Classifier as Weight (ratio) ---
    "adversarial_lgbm_weight_ratio": lambda: AdversarialValidation(
        classifier="lgbm", return_as="weight", weight_type="ratio"
    ),
    "adversarial_lgbm_weight_raw": lambda: AdversarialValidation(
        classifier="lgbm", return_as="weight", weight_type="raw"
    ),
    # --- M6: Domain Classifier as Both ---
    "adversarial_lgbm_both_ratio": lambda: AdversarialValidation(
        classifier="lgbm", return_as="both", weight_type="ratio"
    ),
    # --- M6: Calibrated Domain Classifier ---
    "adversarial_lgbm_cal_iso_feature": lambda: AdversarialValidation(
        classifier="lgbm", return_as="feature", calibrate="isotonic"
    ),
    "adversarial_lgbm_cal_platt_feature": lambda: AdversarialValidation(
        classifier="lgbm", return_as="feature", calibrate="platt"
    ),
    "adversarial_lgbm_cal_iso_both_ratio": lambda: AdversarialValidation(
        classifier="lgbm", return_as="both", weight_type="ratio", calibrate="isotonic"
    ),
    # --- M13: Sample Reweighting (ratio, different clipping) ---
    "adversarial_reweight_lgbm_ratio_hard": lambda: AdversarialReweighting(
        classifier="lgbm", weight_type="ratio", clip_strategy="hard"
    ),
    "adversarial_reweight_lgbm_ratio_normalize": lambda: AdversarialReweighting(
        classifier="lgbm", weight_type="ratio", clip_strategy="normalize"
    ),
    "adversarial_reweight_lgbm_ratio_winsorize": lambda: AdversarialReweighting(
        classifier="lgbm", weight_type="ratio", clip_strategy="winsorize"
    ),
    # --- M13: Sample Reweighting (raw P(test|x)) ---
    "adversarial_reweight_lgbm_raw_hard": lambda: AdversarialReweighting(
        classifier="lgbm", weight_type="raw", clip_strategy="hard"
    ),
    # --- M13: Calibrated Reweighting ---
    "adversarial_reweight_lgbm_cal_iso_ratio_hard": lambda: AdversarialReweighting(
        classifier="lgbm", weight_type="ratio", clip_strategy="hard", calibrate="isotonic"
    ),
    "adversarial_reweight_lgbm_cal_iso_ratio_winsorize": lambda: AdversarialReweighting(
        classifier="lgbm", weight_type="ratio", clip_strategy="winsorize", calibrate="isotonic"
    ),
    # --- M13: PAD-Proportional Tempered Reweighting ---
    "adversarial_reweight_lgbm_ratio_hard_tempered": lambda: AdversarialReweighting(
        classifier="lgbm", weight_type="ratio", clip_strategy="hard", tempering="pad", pad_max=1.0
    ),
    "adversarial_reweight_lgbm_ratio_winsorize_tempered": lambda: AdversarialReweighting(
        classifier="lgbm", weight_type="ratio", clip_strategy="winsorize",
        tempering="pad", pad_max=1.0
    ),
    "adversarial_reweight_lgbm_cal_iso_ratio_hard_tempered": lambda: AdversarialReweighting(
        classifier="lgbm", weight_type="ratio", clip_strategy="hard",
        calibrate="isotonic", tempering="pad", pad_max=1.0
    ),
    # --- M7: Pseudo-Label Self-Training ---
    "pseudo_label_lgbm_conf90_iter3": lambda: PseudoLabelSelfTraining(
        classifier="lgbm", confidence_threshold=0.9, max_iterations=3
    ),
    "pseudo_label_lgbm_conf95_iter3": lambda: PseudoLabelSelfTraining(
        classifier="lgbm", confidence_threshold=0.95, max_iterations=3
    ),
    "pseudo_label_lgbm_conf80_iter5": lambda: PseudoLabelSelfTraining(
        classifier="lgbm", confidence_threshold=0.8, max_iterations=5
    ),
    "pseudo_label_lgbm_conf90_iter1": lambda: PseudoLabelSelfTraining(
        classifier="lgbm", confidence_threshold=0.9, max_iterations=1
    ),
    # --- M16: Nearest-Test-Neighbor Distance ---
    "nearest_test_k10_both": lambda: NearestTestDistance(
        k=10, return_as="both", feature_weighting="none"
    ),
    "nearest_test_k10_filter_both": lambda: NearestTestDistance(
        k=10, return_as="both", feature_weighting="filter"
    ),
    "nearest_test_k5_both": lambda: NearestTestDistance(
        k=5, return_as="both", feature_weighting="none"
    ),
    "nearest_test_k10_cumshift_both": lambda: NearestTestDistance(
        k=10, return_as="both", feature_weighting="cumshift"
    ),
    # --- M8: Label Propagation Feature ---
    "label_prop_k15_a05": lambda: LabelPropagationFeature(
        k=15, alpha=0.5
    ),
    "label_prop_k10_a05": lambda: LabelPropagationFeature(
        k=10, alpha=0.5
    ),
    "label_prop_k15_a08": lambda: LabelPropagationFeature(
        k=15, alpha=0.8
    ),
    # --- M9: OT Displacement Features ---
    "ot_displacement_mag": lambda: OTDisplacementFeatures(
        append_magnitude=True
    ),
    "ot_displacement_no_mag": lambda: OTDisplacementFeatures(
        append_magnitude=False
    ),
    # --- M17: Test-Directed Data Augmentation ---
    "tdda_k5_lam07_aug1": lambda: TestDirectedDataAugmentation(
        k=5, lambda_=0.7, n_augments=1
    ),
    "tdda_k5_lam06_aug1": lambda: TestDirectedDataAugmentation(
        k=5, lambda_=0.6, n_augments=1
    ),
    "tdda_k5_lam08_aug1": lambda: TestDirectedDataAugmentation(
        k=5, lambda_=0.8, n_augments=1
    ),
    "tdda_k5_lam07_aug2": lambda: TestDirectedDataAugmentation(
        k=5, lambda_=0.7, n_augments=2
    ),
    "tdda_k10_lam07_aug1": lambda: TestDirectedDataAugmentation(
        k=10, lambda_=0.7, n_augments=1
    ),
    "tdda_k5_lam07_aug1_cumshift": lambda: TestDirectedDataAugmentation(
        k=5, lambda_=0.7, n_augments=1, feature_weighting="cumshift"
    ),
    # --- M9v2: Improved OT Displacement ---
    "ot_disp_v2_summary_only": lambda: OTDisplacementImproved(
        mode="summary_only"
    ),
    "ot_disp_v2_selective": lambda: OTDisplacementImproved(
        mode="selective"
    ),
    "ot_disp_v2_selective_and_summary": lambda: OTDisplacementImproved(
        mode="selective_and_summary"
    ),
    "ot_disp_v2_selective_top10": lambda: OTDisplacementImproved(
        mode="selective", top_k=10
    ),
    "ot_disp_v2_selective_and_summary_top10": lambda: OTDisplacementImproved(
        mode="selective_and_summary", top_k=10
    ),
    # --- M9v2: Absolute per-feature displacement variants ---
    # Absolute displacement preserves magnitude while removing the
    # sign-mirrored split-membership signal of signed displacement.
    "ot_disp_v2_selective_abs": lambda: OTDisplacementImproved(
        mode="selective", abs_disp=True
    ),
    "ot_disp_v2_selective_and_summary_abs": lambda: OTDisplacementImproved(
        mode="selective_and_summary", abs_disp=True
    ),
    # --- M7v2: Calibrated Pseudo-Labeling ---
    "pseudo_label_v2_cal_iso_wprop_conf90": lambda: PseudoLabelCalibrated(
        calibration_method="isotonic", weight_mode="proportional",
        confidence_threshold=0.9, max_iterations=3,
    ),
    "pseudo_label_v2_cal_iso_wquad_conf90": lambda: PseudoLabelCalibrated(
        calibration_method="isotonic", weight_mode="quadratic",
        confidence_threshold=0.9, max_iterations=3,
    ),
    "pseudo_label_v2_cal_platt_wprop_conf90": lambda: PseudoLabelCalibrated(
        calibration_method="platt", weight_mode="proportional",
        confidence_threshold=0.9, max_iterations=3,
    ),
    "pseudo_label_v2_cal_iso_wprop_conf85": lambda: PseudoLabelCalibrated(
        calibration_method="isotonic", weight_mode="proportional",
        confidence_threshold=0.85, max_iterations=3,
    ),
    "pseudo_label_v2_cal_iso_wfixed_conf90": lambda: PseudoLabelCalibrated(
        calibration_method="isotonic", weight_mode="fixed",
        confidence_threshold=0.9, max_iterations=3,
    ),
    # --- M8v2: Feature-Weighted Label Propagation ---
    "label_prop_weighted_k15_a05": lambda: LabelPropagationWeighted(
        k=15, alpha=0.5, adaptive_k=False,
    ),
    "label_prop_weighted_k10_a05": lambda: LabelPropagationWeighted(
        k=10, alpha=0.5, adaptive_k=False,
    ),
    "label_prop_weighted_k15_a08": lambda: LabelPropagationWeighted(
        k=15, alpha=0.8, adaptive_k=False,
    ),
    "label_prop_weighted_adaptive_a05": lambda: LabelPropagationWeighted(
        k=15, alpha=0.5, adaptive_k=True,
    ),
    # --- M8v3: Enhanced Label Propagation (hard selection + multi-feature) ---
    "label_prop_v3_t005_k15": lambda: LabelPropagationEnhanced(
        k=15, alpha=0.5, feature_threshold=0.05,
    ),
    "label_prop_v3_t01_k15": lambda: LabelPropagationEnhanced(
        k=15, alpha=0.5, feature_threshold=0.1,
    ),
    "label_prop_v3_t005_k10": lambda: LabelPropagationEnhanced(
        k=10, alpha=0.5, feature_threshold=0.05,
    ),
    "label_prop_v3_adaptive": lambda: LabelPropagationEnhanced(
        k=15, alpha=0.5, feature_threshold=0.05, adaptive_k=True,
    ),
    # --- M17v2: Improved TDDA ---
    "tdda_v2_k5_lam07_pfl_dw": lambda: TestDirectedDataAugmentationImproved(
        k=5, base_lambda=0.7, n_augments=1,
        use_per_feature_lambda=True, use_distance_weights=True,
    ),
    "tdda_v2_k5_lam06_pfl_dw": lambda: TestDirectedDataAugmentationImproved(
        k=5, base_lambda=0.6, n_augments=1,
        use_per_feature_lambda=True, use_distance_weights=True,
    ),
    "tdda_v2_k5_lam07_pfl_nodw": lambda: TestDirectedDataAugmentationImproved(
        k=5, base_lambda=0.7, n_augments=1,
        use_per_feature_lambda=True, use_distance_weights=False,
    ),
    "tdda_v2_k5_lam07_nopfl_dw": lambda: TestDirectedDataAugmentationImproved(
        k=5, base_lambda=0.7, n_augments=1,
        use_per_feature_lambda=False, use_distance_weights=True,
    ),
    "tdda_v2_k10_lam07_pfl_dw": lambda: TestDirectedDataAugmentationImproved(
        k=10, base_lambda=0.7, n_augments=1,
        use_per_feature_lambda=True, use_distance_weights=True,
    ),
    "tdda_v2_k5_lam07_pfl_dw_cumshift": lambda: TestDirectedDataAugmentationImproved(
        k=5, base_lambda=0.7, n_augments=1,
        use_per_feature_lambda=True, use_distance_weights=True,
        feature_weighting="cumshift",
    ),
    # --- M16v2: Relative Distances ---
    "nearest_test_rel_k10_both": lambda: NearestTestDistanceRelative(
        k=10, return_as="both", feature_weighting="none",
    ),
    "nearest_test_rel_k5_both": lambda: NearestTestDistanceRelative(
        k=5, return_as="both", feature_weighting="none",
    ),
    "nearest_test_rel_k10_cumshift_both": lambda: NearestTestDistanceRelative(
        k=10, return_as="both", feature_weighting="cumshift",
    ),
    # --- M18: GBDT Leaf Co-Occurrence Analysis ---
    "leaf_density_conf": lambda: LeafBasedTestDensityFE(
        mode="confidence_feature", n_estimators=100, max_depth=6,
    ),
    "leaf_density_conf_shallow": lambda: LeafBasedTestDensityFE(
        mode="confidence_feature", n_estimators=100, max_depth=3,
    ),
    "leaf_density_conf_deep": lambda: LeafBasedTestDensityFE(
        mode="confidence_feature", n_estimators=150, max_depth=8,
    ),
    "leaf_density_embed_k10": lambda: LeafBasedTestDensityFE(
        mode="leaf_embedding_knn", k=10,
    ),
    "leaf_density_embed_k5": lambda: LeafBasedTestDensityFE(
        mode="leaf_embedding_knn", k=5,
    ),
    "leaf_density_embed_k20": lambda: LeafBasedTestDensityFE(
        mode="leaf_embedding_knn", k=20,
    ),
    "leaf_density_conf_cumshift": lambda: LeafBasedTestDensityFE(
        mode="confidence_feature", n_estimators=100, max_depth=6,
        feature_subset="cumshift",
    ),
    "leaf_density_embed_k10_cumshift": lambda: LeafBasedTestDensityFE(
        mode="leaf_embedding_knn", k=10, feature_subset="cumshift",
    ),
    # --- M20: Cluster-Then-Adapt ---
    "cluster_membership_k5_feature": lambda: ClusterMembershipFE(
        n_clusters=5, return_as="feature",
    ),
    "cluster_membership_k10_feature": lambda: ClusterMembershipFE(
        n_clusters=10, return_as="feature",
    ),
    "cluster_membership_k10_weight": lambda: ClusterMembershipFE(
        n_clusters=10, return_as="weight",
    ),
    "cluster_membership_k10_both": lambda: ClusterMembershipFE(
        n_clusters=10, return_as="both",
    ),
    "cluster_membership_k5_both": lambda: ClusterMembershipFE(
        n_clusters=5, return_as="both",
    ),
    "cluster_membership_k15_both": lambda: ClusterMembershipFE(
        n_clusters=15, return_as="both",
    ),
    "cluster_membership_k10_both_cumshift": lambda: ClusterMembershipFE(
        n_clusters=10, return_as="both", feature_subset="cumshift",
    ),
    "cluster_membership_k10_feature_cumshift": lambda: ClusterMembershipFE(
        n_clusters=10, return_as="feature", feature_subset="cumshift",
    ),
    # --- M19: Neighborhood Consensus Features ---
    "neighbor_consensus_k10_dist": lambda: NeighborhoodConsensusFeatures(
        k=10, weighting="distance",
    ),
    "neighbor_consensus_k10_uniform": lambda: NeighborhoodConsensusFeatures(
        k=10, weighting="uniform",
    ),
    "neighbor_consensus_k5_dist": lambda: NeighborhoodConsensusFeatures(
        k=5, weighting="distance",
    ),
    "neighbor_consensus_k20_dist": lambda: NeighborhoodConsensusFeatures(
        k=20, weighting="distance",
    ),
    "neighbor_consensus_k10_dist_cumshift": lambda: NeighborhoodConsensusFeatures(
        k=10, weighting="distance", feature_subset="cumshift",
    ),
    "neighbor_consensus_k5_dist_cumshift": lambda: NeighborhoodConsensusFeatures(
        k=5, weighting="distance", feature_subset="cumshift",
    ),
}


def get_tier2_method(name: str) -> TestTimeMethod:
    """Instantiate a Tier 2 method by registry name."""
    if name not in TIER2_METHODS:
        raise ValueError(
            f"Unknown Tier 2 method '{name}'. "
            f"Available: {sorted(TIER2_METHODS.keys())}"
        )
    return TIER2_METHODS[name]()


def get_all_tier2_methods() -> dict[str, TestTimeMethod]:
    """Instantiate all registered Tier 2 methods."""
    return {name: factory() for name, factory in TIER2_METHODS.items()}

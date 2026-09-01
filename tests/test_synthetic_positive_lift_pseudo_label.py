"""Synthetic positive-lift tests for M7 pseudo-label self-training.

The controlled orthogonal-subpopulation shift uses a downstream logistic
regression that cannot represent the subpopulation-by-direction interaction.
An auxiliary LightGBM can pseudo-label confident rows from the test-dominant
subpopulation. The augmented training set then adjusts the downstream model for
the test distribution.

Each of five variants must exceed ``InductiveBaseline`` by
``MIN_LIFT_VS_BASELINE`` across ``N_SEEDS``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.tier2 import PseudoLabelCalibrated, PseudoLabelSelfTraining
from tests._synthetic_data import make_pseudo_labelable
from tests._synthetic_evaluation import weighted_augment_auc


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
N_SAMPLES_TRAIN = 900   # Supports LightGBM splits with min_child_samples=20.
N_SAMPLES_TEST = 900
N_NOISE = 4
N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05

# Five pseudo-label variants evaluated on this mechanism.
EVALUATED_VARIANTS = [
    pytest.param(
        lambda: PseudoLabelSelfTraining(
            classifier="lgbm", confidence_threshold=0.9, max_iterations=3
        ),
        "pseudo_label_lgbm_conf90_iter3",
        id="pseudo_label_lgbm_conf90_iter3",
    ),
    pytest.param(
        lambda: PseudoLabelSelfTraining(
            classifier="lgbm", confidence_threshold=0.8, max_iterations=5
        ),
        "pseudo_label_lgbm_conf80_iter5",
        id="pseudo_label_lgbm_conf80_iter5",
    ),
    pytest.param(
        lambda: PseudoLabelCalibrated(
            calibration_method="isotonic", weight_mode="proportional",
            confidence_threshold=0.9, max_iterations=3,
        ),
        "pseudo_label_v2_cal_iso_wprop_conf90",
        id="pseudo_label_v2_cal_iso_wprop_conf90",
    ),
    pytest.param(
        lambda: PseudoLabelCalibrated(
            calibration_method="isotonic", weight_mode="quadratic",
            confidence_threshold=0.9, max_iterations=3,
        ),
        "pseudo_label_v2_cal_iso_wquad_conf90",
        id="pseudo_label_v2_cal_iso_wquad_conf90",
    ),
    pytest.param(
        lambda: PseudoLabelCalibrated(
            calibration_method="platt", weight_mode="proportional",
            confidence_threshold=0.9, max_iterations=3,
        ),
        "pseudo_label_v2_cal_platt_wprop_conf90",
        id="pseudo_label_v2_cal_platt_wprop_conf90",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen(seed: int):
    return make_pseudo_labelable(
        n_train=N_SAMPLES_TRAIN, n_test=N_SAMPLES_TEST, n_noise=N_NOISE, seed=seed
    )


def _run_variant_vs_baseline(method_factory):
    """Run ``method_factory()`` against ``InductiveBaseline`` for ``N_SEEDS``."""
    aucs_baseline: list[float] = []
    aucs_method: list[float] = []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = _gen(seed)
        aucs_baseline.append(
            weighted_augment_auc(InductiveBaseline(), X_tr, X_te, y_tr, y_te)
        )
        aucs_method.append(
            weighted_augment_auc(method_factory(), X_tr, X_te, y_tr, y_te)
        )
    return (
        float(np.mean(aucs_baseline)),
        float(np.mean(aucs_method)),
        aucs_baseline,
        aucs_method,
    )


# ---------------------------------------------------------------------------
# Positive-lift comparison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_factory,variant_label", EVALUATED_VARIANTS)
def test_pseudo_label_exceeds_baseline_on_covariate_shift(method_factory, variant_label):
    """Each pseudo-label variant must exceed the baseline by the required margin."""
    mean_baseline, mean_method, aucs_baseline, aucs_method = (
        _run_variant_vs_baseline(method_factory)
    )
    lift_vs_baseline = mean_method - mean_baseline
    per_seed_lifts = [m - b for m, b in zip(aucs_method, aucs_baseline)]

    msg = (
        f"\n  Variant: {variant_label}\n"
        f"  N seeds: {N_SEEDS}\n"
        f"  Mean AUC InductiveBaseline: {mean_baseline:.4f}\n"
        f"  Mean AUC {variant_label}:   {mean_method:.4f}\n"
        f"  Mean lift vs baseline:      {lift_vs_baseline:+.4f} "
        f"(threshold: >= +{MIN_LIFT_VS_BASELINE:.2f})\n"
        f"  Per-seed M7 AUCs:           {[round(a, 3) for a in aucs_method]}\n"
        f"  Per-seed baseline AUCs:     {[round(a, 3) for a in aucs_baseline]}\n"
        f"  Per-seed lifts:             {[round(l, 3) for l in per_seed_lifts]}\n"
    )

    assert lift_vs_baseline >= MIN_LIFT_VS_BASELINE, (
        f"{variant_label} AUC difference from InductiveBaseline is below "
        f"{MIN_LIFT_VS_BASELINE:.2f} on the covariate-shift test.{msg}"
    )


# ---------------------------------------------------------------------------
# Baseline and augmentation checks
# ---------------------------------------------------------------------------

def test_baseline_is_misspecified_on_subpop_shift():
    """Verify that the no-interaction logistic baseline remains below its ceiling."""
    aucs = [
        weighted_augment_auc(InductiveBaseline(), *_gen(seed))
        for seed in range(N_SEEDS)
    ]
    mean_auc = float(np.mean(aucs))
    assert mean_auc <= 0.75, (
        f"Baseline mean AUC {mean_auc:.4f} exceeds 0.75. "
        f"Per-seed values: {[round(a, 3) for a in aucs]}."
    )


def test_pseudo_labels_are_created_on_large_train():
    """The design must create pseudo-labels and augment the training rows."""
    X_tr, X_te, y_tr, y_te = _gen(0)
    result = PseudoLabelSelfTraining(
        classifier="lgbm", confidence_threshold=0.9, max_iterations=3
    ).fit_transform(X_tr, X_te, y_tr)
    n_pseudo = int((result.metadata or {}).get("n_pseudo_labels", 0))
    assert n_pseudo > 0, (
        f"The method created {n_pseudo} pseudo-labels. At least one is required."
    )
    # Augmentation must add rows and provide the corresponding labels.
    assert result.X_train.shape[0] > X_tr.shape[0]
    assert result.y_train is not None
    assert len(result.y_train) == result.X_train.shape[0]

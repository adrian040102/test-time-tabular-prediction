"""Synthetic positive-lift tests for M13 adversarial reweighting.

The controlled dataset contains two subpopulations with orthogonal decision
directions and a large train-to-test mixture shift. A logistic regression
without interactions fits a train-dominated compromise. Domain-classifier
importance weights emphasize the test-like subpopulation and adjust the model
for the test distribution.

Each of four variants must exceed ``InductiveBaseline`` by
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
from src.methods.tier2 import AdversarialReweighting
from tests._synthetic_data import make_covariate_shift_weighted
from tests._synthetic_evaluation import weighted_augment_auc


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
N_SAMPLES_TRAIN = 900
N_SAMPLES_TEST = 900
N_NOISE = 4
TRAIN_S1_FRAC = 0.15
TEST_S1_FRAC = 0.85
THETA1_DEG = 90.0
N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05

# Four reweighting variants evaluated on this mechanism.
EVALUATED_VARIANTS = [
    pytest.param(
        lambda: AdversarialReweighting(
            classifier="lgbm", weight_type="ratio", clip_strategy="hard",
            calibrate="isotonic", tempering="pad", pad_max=1.0,
        ),
        "adversarial_reweight_lgbm_cal_iso_ratio_hard_tempered",
        id="adversarial_reweight_lgbm_cal_iso_ratio_hard_tempered",
    ),
    pytest.param(
        lambda: AdversarialReweighting(
            classifier="lgbm", weight_type="ratio", clip_strategy="hard",
            calibrate="isotonic",
        ),
        "adversarial_reweight_lgbm_cal_iso_ratio_hard",
        id="adversarial_reweight_lgbm_cal_iso_ratio_hard",
    ),
    pytest.param(
        lambda: AdversarialReweighting(
            classifier="lgbm", weight_type="ratio", clip_strategy="hard",
        ),
        "adversarial_reweight_lgbm_ratio_hard",
        id="adversarial_reweight_lgbm_ratio_hard",
    ),
    pytest.param(
        lambda: AdversarialReweighting(
            classifier="lgbm", weight_type="ratio", clip_strategy="hard",
            tempering="pad", pad_max=1.0,
        ),
        "adversarial_reweight_lgbm_ratio_hard_tempered",
        id="adversarial_reweight_lgbm_ratio_hard_tempered",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gen(seed: int):
    return make_covariate_shift_weighted(
        n_train=N_SAMPLES_TRAIN,
        n_test=N_SAMPLES_TEST,
        n_noise=N_NOISE,
        train_s1_frac=TRAIN_S1_FRAC,
        test_s1_frac=TEST_S1_FRAC,
        theta1_deg=THETA1_DEG,
        seed=seed,
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
def test_reweighting_exceeds_baseline_on_covariate_shift(method_factory, variant_label):
    """Each reweighting variant must exceed the baseline by the required margin."""
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
        f"  Per-seed M13 AUCs:          {[round(a, 3) for a in aucs_method]}\n"
        f"  Per-seed baseline AUCs:     {[round(a, 3) for a in aucs_baseline]}\n"
        f"  Per-seed lifts:             {[round(l, 3) for l in per_seed_lifts]}\n"
    )

    assert lift_vs_baseline >= MIN_LIFT_VS_BASELINE, (
        f"{variant_label} AUC difference from InductiveBaseline is below "
        f"{MIN_LIFT_VS_BASELINE:.2f} on the covariate-shift test.{msg}"
    )


# ---------------------------------------------------------------------------
# Baseline ceiling
# ---------------------------------------------------------------------------

def test_baseline_is_misspecified_on_subpop_shift():
    """The no-interaction logistic baseline AUC must not exceed 0.75."""
    aucs: list[float] = []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = _gen(seed)
        aucs.append(weighted_augment_auc(InductiveBaseline(), X_tr, X_te, y_tr, y_te))

    mean_auc = float(np.mean(aucs))
    msg = (
        f"\n  Mean baseline AUC on subpop shift: {mean_auc:.4f} "
        f"(expected <= 0.75)\n"
        f"  Per-seed: {[round(a, 3) for a in aucs]}\n"
    )
    assert mean_auc <= 0.75, f"Baseline AUC exceeds 0.75.{msg}"

"""Synthetic threshold tests for M17 test-directed data augmentation.

The evaluated variants must produce finite AUC differences below 0.05 on the
directional-augmentation design. The baseline must also remain below its
configured ceiling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.tier2 import (
    TestDirectedDataAugmentation,
    TestDirectedDataAugmentationImproved,
)
from tests._synthetic_data import make_directional_augment
from tests._synthetic_evaluation import weighted_augment_auc


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
N_TRAIN = 150
N_TEST = 800
N_NOISE = 2
X0_SEP = 0.5
SPUR_STRENGTH = 2.0
N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05

# Variants evaluated by the below-threshold behavior test.
EVALUATED_VARIANTS = [
    pytest.param(
        lambda: TestDirectedDataAugmentation(k=5, lambda_=0.7, n_augments=1),
        "tdda_k5_lam07_aug1",
        id="tdda_k5_lam07_aug1",
    ),
    pytest.param(
        lambda: TestDirectedDataAugmentationImproved(
            k=5, base_lambda=0.7, n_augments=1,
            use_per_feature_lambda=True, use_distance_weights=True,
        ),
        "tdda_v2_k5_lam07_pfl_dw",
        id="tdda_v2_k5_lam07_pfl_dw",
    ),
    pytest.param(
        lambda: TestDirectedDataAugmentationImproved(
            k=5, base_lambda=0.7, n_augments=1,
            use_per_feature_lambda=True, use_distance_weights=True,
            feature_weighting="cumshift",
        ),
        "tdda_v2_k5_lam07_pfl_dw_cumshift",
        id="tdda_v2_k5_lam07_pfl_dw_cumshift",
    ),
]


def _gen(seed: int):
    return make_directional_augment(
        n_train=N_TRAIN, n_test=N_TEST, n_noise=N_NOISE,
        x0_sep=X0_SEP, spur_strength=SPUR_STRENGTH, seed=seed,
    )


def _run_variant_vs_baseline(method_factory):
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
# Below-threshold behavior test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_factory,variant_label", EVALUATED_VARIANTS)
def test_tdda_stays_below_threshold_on_designed_dataset(method_factory, variant_label):
    """Each M17 variant must remain below the configured AUC threshold."""
    mean_baseline, mean_method, _, _ = (
        _run_variant_vs_baseline(method_factory)
    )
    lift = mean_method - mean_baseline

    assert np.isfinite(lift), f"{variant_label}: lift is not finite"
    assert lift < MIN_LIFT_VS_BASELINE, (
        f"{variant_label} AUC difference {lift:+.4f} is not below "
        f"{MIN_LIFT_VS_BASELINE:.2f} on make_directional_augment."
    )


def test_baseline_remains_below_configured_ceiling():
    """The design baseline must remain below the configured ceiling."""
    aucs = [weighted_augment_auc(InductiveBaseline(), *_gen(seed)) for seed in range(N_SEEDS)]
    mean_auc = float(np.mean(aucs))
    assert mean_auc <= 0.70, (
        f"Baseline AUC {mean_auc:.4f} exceeds the configured ceiling of 0.70. "
        f"Per-seed values: {[round(a, 3) for a in aucs]}."
    )

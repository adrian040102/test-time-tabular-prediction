"""Synthetic positive-lift tests for M5 ``JointScaling``.

M5 fits a ``StandardScaler`` on the combined train and test features. Its
train-only version fits the scaler on training features only. The synthetic
design contains stable-scale signal features and nuisance features with greater
test variance. Joint scaling can use the test feature scale to reduce the
influence of those nuisance features under L2-regularized logistic regression.

The mean joint AUC must exceed both comparison AUCs by at least 0.05. A separate
control uses effectively unregularized logistic regression. Under that control,
per-feature affine scaling should not produce the same improvement because the
prediction ranking is invariant to the scaling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.tier1 import JointScaling
from src.methods.baselines import TrainOnlyScaling
from tests._synthetic_data import make_heteroscedastic_scale_shift
from tests._synthetic_evaluation import weighted_augment_auc


N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05
MIN_TESTTIME_VALUE = 0.05

BASELINE_CEILING = 0.72   # nuisance test-scale blowup suppresses the L2 baseline

GEN_KW = dict(n_train=400, n_test=2000, n_signal=4, n_noise=40,
              test_noise_scale=12.0, spurious=0.0, signal_strength=1.0)


def _unreg_logreg():
    """Return the rank-invariance control with ``C=1e6``.

    Without effective L2 regularization, per-feature standardization cannot
    change the prediction ranking.
    """
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, random_state=0, C=1e6)


def _run(downstream=None):
    """Return mean AUCs for the baseline, train-only version and joint version."""
    b, t, j = [], [], []
    for seed in range(N_SEEDS):
        data = make_heteroscedastic_scale_shift(seed=seed, **GEN_KW)
        b.append(weighted_augment_auc(InductiveBaseline(), *data, downstream=downstream))
        t.append(weighted_augment_auc(TrainOnlyScaling(), *data, downstream=downstream))
        j.append(weighted_augment_auc(JointScaling(), *data, downstream=downstream))
    return (float(np.mean(b)), float(np.mean(t)), float(np.mean(j)), np.array(j))


def test_joint_scaling_exceeds_both_comparisons():
    """Joint scaling must exceed both comparison AUCs by at least 0.05."""
    mean_b, mean_t, mean_j, j = _run()
    lift = mean_j - mean_b
    ttv = mean_j - mean_t

    msg = (
        f"\n  Variant: joint_scaling  (downstream: L2 LogisticRegression)\n"
        f"  Mean AUC InductiveBaseline:      {mean_b:.4f}\n"
        f"  Mean AUC train_only_scaling:     {mean_t:.4f}\n"
        f"  Mean AUC joint_scaling:          {mean_j:.4f}\n"
        f"  Lift vs baseline:   {lift:+.4f} (≥ +{MIN_LIFT_VS_BASELINE:.2f})\n"
        f"  Test-time value:    {ttv:+.4f} (≥ +{MIN_TESTTIME_VALUE:.2f})\n"
        f"  Per-seed joint AUCs: {[round(a, 3) for a in j]}\n"
    )
    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"joint_scaling AUC difference from the baseline is below "
        f"{MIN_LIFT_VS_BASELINE:.2f}.{msg}"
    )
    assert ttv >= MIN_TESTTIME_VALUE, (
        f"joint_scaling AUC difference from the train-only version is below "
        f"{MIN_TESTTIME_VALUE:.2f}.{msg}"
    )


def test_baseline_and_train_only_version_meet_expected_ceiling():
    """The test-inflated nuisance features reduce the L2 baseline and
    train-only version down. Both remain below the joint version and
    ``BASELINE_CEILING``. The baseline equals the train-only version because both
    use training-set standardization, so the recoverable difference comes from
    the test-feature scale."""
    mean_b, mean_t, mean_j, _ = _run()
    assert mean_b < BASELINE_CEILING, (
        f"Baseline AUC {mean_b:.4f} is not below {BASELINE_CEILING}. The "
        f"expected effect of the nuisance-feature scale no longer holds."
    )
    assert abs(mean_b - mean_t) < 0.02, (
        f"Inductive baseline {mean_b:.4f} and train_only_scaling {mean_t:.4f} "
        f"should coincide (both train-std normalization)."
    )
    assert mean_t < mean_j - MIN_TESTTIME_VALUE, (
        f"train_only_scaling {mean_t:.4f} is not at least {MIN_TESTTIME_VALUE:.2f} "
        f"below joint_scaling {mean_j:.4f}."
    )


def test_mechanism_is_regularization_conditional():
    """The AUC difference must stay below 0.05 without effective L2 regularization."""
    mean_b, mean_t, mean_j, _ = _run(downstream=_unreg_logreg)
    ttv_unreg = mean_j - mean_t
    assert ttv_unreg < MIN_TESTTIME_VALUE, (
        f"Under effectively unregularized logistic regression, the joint and "
        f"train-only AUC difference is {ttv_unreg:+.4f}, which is not below "
        f"{MIN_TESTTIME_VALUE:.2f}."
    )

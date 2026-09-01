"""Synthetic positive-lift test for M1 ``JointFrequencyEncoding``.

M1 has a train-only version (``freq_enc_train_only``). This test compares the
inductive baseline, train-only version and joint version. The test-time value is
``lift(joint) − lift(train_only)``. It measures whether the unlabeled test rows
help beyond the effect of frequency encoding itself.

Design based on label-related prevalence shift
--------------------------------------------
``make_prevalence_shift_cat`` builds one ``cat`` column whose category
prevalence differs between train and test. Train categories are sampled uniformly,
so the *train* frequency is uninformative. Test categories are sampled in
proportion to population prevalence. The label of a category depends on its
population prevalence.

* ``InductiveBaseline`` (label-encoded ``cat``) is near chance because a linear
  model cannot infer prevalence from one arbitrary ordinal code.
* ``freq_enc_train_only`` remains near chance because train frequency is approximately uniform.
* ``JointFrequencyEncoding`` (joint train+test frequency) predicts the label
  because the joint count approximates the test and population prevalence.

The test uses logistic regression. The mechanism is model-conditional because a tree
would memorise the encoded category and invalidate the intended control. See the
generator docstring.

Required mean differences over ``N_SEEDS`` replicates
-------------------------------------------------
  * joint AUC ≥ baseline AUC + ``MIN_LIFT_VS_BASELINE`` (= 0.05)
  * joint AUC ≥ train-only AUC + ``MIN_TESTTIME_VALUE`` (= 0.05)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.tier1 import JointFrequencyEncoding
from src.methods.baselines import TrainOnlyFrequencyEncoding
from tests._synthetic_data import make_prevalence_shift_cat
from tests._synthetic_evaluation import weighted_augment_auc


N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05
MIN_TESTTIME_VALUE = 0.05

# The baseline and train-only version must remain near chance.
BASELINE_CEILING = 0.65
TRAIN_ONLY_CEILING = 0.70

GEN_KW = dict(n_train=800, n_test=1600, n_categories=50, n_noise=3, alpha=1.0)


def _comparison_aucs(joint_factory):
    """Return per-seed AUC arrays for the baseline and both method versions."""
    b, t, j = [], [], []
    for seed in range(N_SEEDS):
        data = make_prevalence_shift_cat(seed=seed, **GEN_KW)
        b.append(weighted_augment_auc(InductiveBaseline(), *data))
        t.append(weighted_augment_auc(TrainOnlyFrequencyEncoding(), *data))
        j.append(weighted_augment_auc(joint_factory(), *data))
    return np.array(b), np.array(t), np.array(j)


@pytest.mark.parametrize(
    "joint_factory,variant_label",
    [
        (lambda: JointFrequencyEncoding(variant="combined"), "joint_freq_combined"),
        (lambda: JointFrequencyEncoding(variant="separate"), "joint_freq_separate"),
    ],
)
def test_joint_frequency_exceeds_both_comparisons(joint_factory, variant_label):
    """M1 must exceed the inductive baseline and train-only frequency version.

    The comparison with the train-only version isolates the value of counting
    unlabeled test rows from the value of frequency encoding.
    """
    b, t, j = _comparison_aucs(joint_factory)
    mean_b, mean_t, mean_j = float(b.mean()), float(t.mean()), float(j.mean())
    lift = mean_j - mean_b
    testtime_value = mean_j - mean_t

    msg = (
        f"\n  Variant: {variant_label}\n"
        f"  N seeds: {N_SEEDS}\n"
        f"  Mean AUC InductiveBaseline:    {mean_b:.4f}\n"
        f"  Mean AUC freq_enc_train_only:  {mean_t:.4f}\n"
        f"  Mean AUC {variant_label}:      {mean_j:.4f}\n"
        f"  Lift vs baseline:    {lift:+.4f} (threshold ≥ +{MIN_LIFT_VS_BASELINE:.2f})\n"
        f"  Test-time value:     {testtime_value:+.4f} (threshold ≥ +{MIN_TESTTIME_VALUE:.2f})\n"
        f"  Per-seed joint AUCs: {[round(a, 3) for a in j]}\n"
    )

    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"{variant_label} AUC minus inductive baseline AUC is below "
        f"{MIN_LIFT_VS_BASELINE:.2f}.{msg}"
    )
    assert testtime_value >= MIN_TESTTIME_VALUE, (
        f"{variant_label} AUC minus train-only AUC is below "
        f"{MIN_TESTTIME_VALUE:.2f}.{msg}"
    )


def test_controls_are_near_chance():
    """The baseline and train-only version must remain near chance.

    The near-chance controls isolate the information added by the joint version.
    The label-encoded category and train-only frequency contain
    little prevalence information under a linear model.
    """
    b, t, _ = _comparison_aucs(lambda: JointFrequencyEncoding(variant="combined"))
    mean_b, mean_t = float(b.mean()), float(t.mean())
    assert mean_b < BASELINE_CEILING, (
        f"Inductive baseline AUC {mean_b:.4f} is not below "
        f"{BASELINE_CEILING}."
    )
    assert mean_t < TRAIN_ONLY_CEILING, (
        f"Train-only frequency AUC {mean_t:.4f} is not below "
        f"{TRAIN_ONLY_CEILING}."
    )

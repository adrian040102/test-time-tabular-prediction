"""Synthetic positive-lift test for M2 ``FrequencyRatioImproved``.

M2 uses the same train-only frequency version as M1, but the ratio
``f_test/f_train`` is undefined without the test rows. The primary comparison is
with the inductive baseline. The train-only frequency version is a secondary
control that should remain near chance because train frequency is a prevalence
*level* rather than a *direction*.

Design: prevalence-shift direction
----------------------------------
``make_directional_prevalence_shift`` builds one ``sig`` categorical whose
categories each increase or decrease in prevalence between train and test. The
label is that direction, assigned independently of the train-prevalence level.
The data also contain three non-shifting categorical noise columns.

* ``InductiveBaseline`` remains near chance because the category code does not
  contain the direction.
* ``freq_enc_train_only`` remains near chance because it contains a level rather
  than a direction.
* ``FrequencyRatioImproved`` (ratio or difference of test and train frequency) represents the
  direction directly.

High cardinality (150 levels with about 10 train rows each) is necessary. It prevents the
baseline from memorising the deterministic category-to-direction map and
transferring it to test.

Model: logistic regression (a tree could memorise the encoded category).

Required mean differences over ``N_SEEDS`` replicates
----------------------------------------------------
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
from src.methods.tier1 import FrequencyRatioImproved
from src.methods.baselines import TrainOnlyFrequencyEncoding
from tests._synthetic_data import make_directional_prevalence_shift
from tests._synthetic_evaluation import weighted_augment_auc


N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05
MIN_TESTTIME_VALUE = 0.05

# Expected control ceilings.
BASELINE_CEILING = 0.62
TRAIN_ONLY_CEILING = 0.68

GEN_KW = dict(n_train=1500, n_test=1500, n_categories=150, n_noise_cols=3,
              n_noise_levels=15, ratio_up=3.0, ratio_down=1.0 / 3.0, alpha=1.0)

VARIANTS = [
    (lambda: FrequencyRatioImproved(mode="both", adaptive_alpha=True, noise_gate=True),
     "freq_ratio_v2_both_gated"),
    (lambda: FrequencyRatioImproved(mode="diff", adaptive_alpha=True, noise_gate=True),
     "freq_ratio_v2_diff_gated"),
    (lambda: FrequencyRatioImproved(mode="ratio", adaptive_alpha=True, noise_gate=False),
     "freq_ratio_v2_ratio_ungated"),
]


def _three_way(joint_factory):
    b, t, j = [], [], []
    for seed in range(N_SEEDS):
        data = make_directional_prevalence_shift(seed=seed, **GEN_KW)
        b.append(weighted_augment_auc(InductiveBaseline(), *data))
        t.append(weighted_augment_auc(TrainOnlyFrequencyEncoding(), *data))
        j.append(weighted_augment_auc(joint_factory(), *data))
    return np.array(b), np.array(t), np.array(j)


@pytest.mark.parametrize("joint_factory,variant_label", VARIANTS)
def test_freq_ratio_exceeds_both_comparisons(joint_factory, variant_label):
    """Each M2 configuration must exceed the baseline and train-only version.

    The second comparison shows that the ratio needs the test rows.
    """
    b, t, j = _three_way(joint_factory)
    mean_b, mean_t, mean_j = float(b.mean()), float(t.mean()), float(j.mean())
    lift = mean_j - mean_b
    testtime_value = mean_j - mean_t

    msg = (
        f"\n  Variant: {variant_label}\n"
        f"  Mean AUC InductiveBaseline:    {mean_b:.4f}\n"
        f"  Mean AUC freq_enc_train_only:  {mean_t:.4f}\n"
        f"  Mean AUC {variant_label}:      {mean_j:.4f}\n"
        f"  Lift vs baseline: {lift:+.4f} (≥ +{MIN_LIFT_VS_BASELINE:.2f})\n"
        f"  Test-time value:  {testtime_value:+.4f} (≥ +{MIN_TESTTIME_VALUE:.2f})\n"
        f"  Per-seed joint AUCs: {[round(a, 3) for a in j]}\n"
    )
    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"{variant_label} AUC minus inductive baseline AUC is below "
        f"{MIN_LIFT_VS_BASELINE:.2f}.{msg}"
    )
    assert testtime_value >= MIN_TESTTIME_VALUE, (
        f"{variant_label} AUC minus train-only frequency AUC is below "
        f"{MIN_TESTTIME_VALUE:.2f}.{msg}"
    )


def test_baseline_and_train_only_version_meet_expected_ceilings():
    """The baseline and train-only frequency version are near chance. A linear model
    cannot infer the shift *direction* from a category code or a train-frequency
    *level*."""
    b, t, _ = _three_way(VARIANTS[0][0])
    mean_b, mean_t = float(b.mean()), float(t.mean())
    assert mean_b < BASELINE_CEILING, (
        f"Inductive baseline AUC {mean_b:.4f} is not below {BASELINE_CEILING}."
    )
    assert mean_t < TRAIN_ONLY_CEILING, (
        f"Train-only frequency AUC {mean_t:.4f} is not below {TRAIN_ONLY_CEILING}."
    )


def test_shift_filter_keeps_the_shifted_column():
    """Verify the chi-square filtering outcomes.

    The shifted ``sig`` column must remain. At least one column must be identified
    as shifted. The non-shifting noise columns must be identified for exclusion.
    """
    data = make_directional_prevalence_shift(seed=0, **GEN_KW)
    method = FrequencyRatioImproved(mode="both", adaptive_alpha=True, noise_gate=True)
    res = method.fit_transform(data[0], data[1], data[2])
    md = res.metadata
    assert md["n_columns_shifted"] >= 1, (
        f"The shift filter removed the signal column. metadata={md}"
    )
    # At least two of the three non-shifting noise columns should be excluded.
    assert md["n_columns_gated"] >= 2, (
        f"Expected the non-shifting noise columns to be excluded. metadata={md}"
    )

"""Synthetic positive-lift test for M4 ``ShiftRegularizedTargetEncoding``.

M4 smooths a target encoding per category by ``m = base_m · adj``, where ``adj``
increases with the train-to-test frequency shift. Its train-only counterpart is a fixed-``m``
target encoding (``target_enc_standard_m{10,20}``).

Design with reversed concept drift correlated with the frequency shift
-------------------------------------------------------------------
``make_reversed_drift_target`` builds one high-cardinality categorical with
stable categories with means spread across the interior and no frequency shift.
A few drift categories are rare in train and dominant in test. Their train target
mean is extreme but their test mean is fully reversed. The mean-to-category-id map is
shuffled so the ordinal label code is uninformative (baseline ≈ chance).

* Plain target encoding (any fixed ``m``) encodes the drift cats by their
  confident but misleading train mean. It reverses their expected ordering on
  the drift-dominated test rows, producing AUC below chance there.
* M4 determines smoothing from test frequency. The drift categories are the
  frequency-shifted ones, so M4 shrinks them toward the global mean (neutralising
  the misleading signal) while preserving the unshifted stable signal. No
  single fixed ``m`` can do both. M4's ``ratio`` and ``absolute`` modes are
  compared with both prespecified fixed-``m`` train-only versions, m10 and m20.
  Using both controls avoids choosing one based on test AUC.

Required mean differences over ``N_SEEDS`` replicates for each primary configuration
----------------------------------------------------------------------
  * joint AUC ≥ baseline AUC + 0.05
  * joint AUC ≥ m10 train-only AUC + 0.05
  * joint AUC ≥ m20 train-only AUC + 0.05

Structural limitation of the ``diff`` mode
-------------------------------------------
``diff``'s ``adj = |f_test − f_train| + 1`` is bounded in ``[1, 2]``, so it can
only interpolate smoothing between m10 and m20. These values are nearly
identical for target encoding and cannot strongly down-weight the drift
categories. The comparison
requires ``target_enc_diff`` and ``target_enc_oof_diff`` to remain below the
+0.05 difference threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.tier1 import ShiftRegularizedTargetEncoding, ShiftRegularizedTargetEncodingOOF
from src.methods.baselines import StandardTargetEncoding
from tests._synthetic_data import make_reversed_drift_target
from tests._synthetic_evaluation import weighted_augment_auc


N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05
MIN_TESTTIME_VALUE = 0.05

BASELINE_CEILING = 0.60   # The shuffled label code keeps the baseline near chance.

GEN_KW = dict(n_train=3000, n_test=3000, n_stable=30, n_drift=3,
              stable_extreme=0.95, drift_extreme=0.98,
              train_drift_frac=0.03, test_drift_frac=0.35)

# Primary ratio and absolute configurations, with and without OOF encoding.
PRIMARY_VARIANTS = [
    (lambda: ShiftRegularizedTargetEncoding(regularization_mode="ratio"),
     "target_enc_ratio"),
    (lambda: ShiftRegularizedTargetEncodingOOF(regularization_mode="ratio"),
     "target_enc_oof_ratio"),
    (lambda: ShiftRegularizedTargetEncoding(regularization_mode="absolute"),
     "target_enc_absolute"),
    (lambda: ShiftRegularizedTargetEncodingOOF(regularization_mode="absolute"),
     "target_enc_oof_absolute"),
]

# Variants this design cannot isolate because of the bounded adjustment.
DIFF_VARIANTS = [
    (lambda: ShiftRegularizedTargetEncoding(regularization_mode="diff"),
     "target_enc_diff"),
    (lambda: ShiftRegularizedTargetEncodingOOF(regularization_mode="diff"),
     "target_enc_oof_diff"),
]


def _run(joint_factory):
    """Return mean AUCs for the baseline, train-only versions and joint version."""
    b, t10, t20, j = [], [], [], []
    for seed in range(N_SEEDS):
        data = make_reversed_drift_target(seed=seed, **GEN_KW)
        b.append(weighted_augment_auc(InductiveBaseline(), *data))
        t10.append(weighted_augment_auc(StandardTargetEncoding(m=10.0), *data))
        t20.append(weighted_augment_auc(StandardTargetEncoding(m=20.0), *data))
        j.append(weighted_augment_auc(joint_factory(), *data))
    return (float(np.mean(b)), float(np.mean(t10)),
            float(np.mean(t20)), float(np.mean(j)), np.array(j))


@pytest.mark.parametrize("joint_factory,variant_label", PRIMARY_VARIANTS)
def test_target_encoding_exceeds_all_comparisons(joint_factory, variant_label):
    """Each ratio or absolute configuration must exceed all three comparisons.

    Comparing with both prespecified fixed-``m`` train-only versions avoids
    selecting a smoothing constant based on test AUC.
    """
    mean_b, mean_t10, mean_t20, mean_j, j = _run(joint_factory)
    lift = mean_j - mean_b
    ttv10 = mean_j - mean_t10
    ttv20 = mean_j - mean_t20

    msg = (
        f"\n  Variant: {variant_label}  (downstream: LR)\n"
        f"  Mean AUC InductiveBaseline:        {mean_b:.4f}\n"
        f"  Mean AUC train-only m10:           {mean_t10:.4f}\n"
        f"  Mean AUC train-only m20:           {mean_t20:.4f}\n"
        f"  Mean AUC {variant_label}:          {mean_j:.4f}\n"
        f"  Lift vs baseline:   {lift:+.4f} (≥ +{MIN_LIFT_VS_BASELINE:.2f})\n"
        f"  Test-time value/m10:{ttv10:+.4f} (≥ +{MIN_TESTTIME_VALUE:.2f})\n"
        f"  Test-time value/m20:{ttv20:+.4f} (≥ +{MIN_TESTTIME_VALUE:.2f})\n"
        f"  Per-seed joint AUCs: {[round(a, 3) for a in j]}\n"
    )
    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"{variant_label} AUC minus baseline AUC is below {MIN_LIFT_VS_BASELINE:.2f}.{msg}"
    )
    assert ttv10 >= MIN_TESTTIME_VALUE, (
        f"{variant_label} AUC minus m10 train-only AUC is below {MIN_TESTTIME_VALUE:.2f}.{msg}"
    )
    assert ttv20 >= MIN_TESTTIME_VALUE, (
        f"{variant_label} AUC minus m20 train-only AUC is below {MIN_TESTTIME_VALUE:.2f}.{msg}"
    )


def test_baseline_and_train_only_versions_meet_expected_ceilings():
    """The baseline and train-only comparison must satisfy their expected limits."""
    mean_b, mean_t10, mean_t20, mean_j, _ = _run(PRIMARY_VARIANTS[0][0])
    assert mean_b < BASELINE_CEILING, (
        f"Inductive baseline AUC {mean_b:.4f} not near chance (< {BASELINE_CEILING})."
    )
    assert mean_t10 < mean_j - MIN_TESTTIME_VALUE, (
        f"Plain m10 train-only AUC {mean_t10:.4f} is not at least "
        f"{MIN_TESTTIME_VALUE:.2f} below the M4 ratio variant {mean_j:.4f}."
    )


@pytest.mark.parametrize("joint_factory,variant_label", DIFF_VARIANTS)
def test_diff_mode_is_structurally_limited(joint_factory, variant_label):
    """The ``diff`` mode's adjustment factor
    ``|f_test − f_train| + 1`` is bounded in ``[1, 2]``, so it only interpolates
    smoothing between m10 and m20 (near-identical for target encoding) and cannot
    isolate the mechanism on this design. The difference from the m10 train-only
    version must remain below +0.05."""
    mean_b, mean_t10, mean_t20, mean_j, _ = _run(joint_factory)
    ttv10 = mean_j - mean_t10
    assert ttv10 < MIN_TESTTIME_VALUE, (
        f"{variant_label} minus m10 train-only AUC is {ttv10:+.4f}, which is not "
        f"below {MIN_TESTTIME_VALUE:.2f}."
    )

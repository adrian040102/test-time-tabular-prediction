"""Controlled tests for M3 ``JointPCA``.

``make_pca_loading_shift_signal`` uses a weaker signal loading in the training
features and a stronger loading in the test features. Each joint configuration
is compared with the corresponding train-only version using the same component
count. A depth-four decision tree is used downstream. The constants below set
the required mean AUC differences across ``N_SEEDS`` replicates.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.tier1 import JointPCA
from src.methods.baselines import TrainOnlyPCA
from tests._synthetic_data import make_pca_loading_shift_signal
from tests._synthetic_evaluation import weighted_augment_auc


N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05
MIN_TESTTIME_VALUE = 0.05

# Expected ceilings: a shallow axis-aligned tree on the raw features is near
# chance because the boundary is diagonal. The train-only version does not include
# that direction among its returned components, so it stays near baseline.
BASELINE_CEILING = 0.65
TRAIN_ONLY_CEILING = 0.66

GEN_KW = dict(n_train=120, n_test=600, n_signal=8, n_noise=150,
              a_train=0.5, a_test=3.0, noise_sd=1.0)

# Each joint factory is paired with a train-only factory using the same component count.
VARIANTS = [
    (lambda: JointPCA(n_components=5), lambda: TrainOnlyPCA(n_components=5),
     "joint_pca_5"),
    (lambda: JointPCA(n_components=10), lambda: TrainOnlyPCA(n_components=10),
     "joint_pca_10"),
    (lambda: JointPCA(n_components=0.9), lambda: TrainOnlyPCA(n_components=0.9),
     "joint_pca_var90"),
    (lambda: JointPCA(n_components=0.95), lambda: TrainOnlyPCA(n_components=0.95),
     "joint_pca_var95"),
]


def _dtree():
    """Return the fixed ``DecisionTree(max_depth=4)`` downstream model."""
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(max_depth=4, random_state=0)


def _comparison_aucs(joint_factory, train_only_factory, downstream=_dtree):
    b, t, j = [], [], []
    for seed in range(N_SEEDS):
        data = make_pca_loading_shift_signal(seed=seed, **GEN_KW)
        b.append(weighted_augment_auc(InductiveBaseline(), *data, downstream=downstream))
        t.append(weighted_augment_auc(train_only_factory(), *data, downstream=downstream))
        j.append(weighted_augment_auc(joint_factory(), *data, downstream=downstream))
    return np.array(b), np.array(t), np.array(j)


@pytest.mark.parametrize("joint_factory,train_only_factory,variant_label", VARIANTS)
def test_joint_pca_exceeds_baseline_and_train_only_auc(
    joint_factory, train_only_factory, variant_label
):
    """Each M3 configuration must exceed the baseline and train-only PCA version.

    The difference shows that including test rows places the discriminative
    direction among the returned components.
    """
    b, t, j = _comparison_aucs(joint_factory, train_only_factory)
    mean_b, mean_t, mean_j = float(b.mean()), float(t.mean()), float(j.mean())
    lift = mean_j - mean_b
    testtime_value = mean_j - mean_t

    msg = (
        f"\n  Variant: {variant_label}  (downstream: DecisionTree(max_depth=4))\n"
        f"  Mean AUC InductiveBaseline:        {mean_b:.4f}\n"
        f"  Mean AUC for pca_train_only:       {mean_t:.4f}\n"
        f"  Mean AUC {variant_label}:          {mean_j:.4f}\n"
        f"  Lift vs baseline: {lift:+.4f} (≥ +{MIN_LIFT_VS_BASELINE:.2f})\n"
        f"  Test-time value:  {testtime_value:+.4f} (≥ +{MIN_TESTTIME_VALUE:.2f})\n"
        f"  Per-seed joint AUCs: {[round(a, 3) for a in j]}\n"
    )
    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"{variant_label} did not reach the required improvement over the "
        f"inductive baseline.{msg}"
    )
    assert testtime_value >= MIN_TESTTIME_VALUE, (
        f"{variant_label} did not reach the required test-time premium.{msg}"
    )


def test_baseline_and_train_only_version_meet_expected_ceilings():
    """A shallow tree on the raw features is near chance (the boundary is a
    diagonal it cannot axis-split). The corresponding train-only PCA version does
    not include that direction among the returned components, so it stays near
    baseline."""
    b, t, _ = _comparison_aucs(VARIANTS[0][0], VARIANTS[0][1])
    mean_b, mean_t = float(b.mean()), float(t.mean())
    assert mean_b < BASELINE_CEILING, (
        f"Inductive baseline AUC {mean_b:.4f} not near chance (< {BASELINE_CEILING})."
    )
    assert mean_t < TRAIN_ONLY_CEILING, (
        f"Train-only PCA AUC {mean_t:.4f} not near chance (< {TRAIN_ONLY_CEILING})."
    )


def test_effect_is_specific_to_tree_downstream():
    """Verify that the same data and method stay below the improvement threshold
    under logistic regression. PCA adds a linear projection to the complete raw
    feature set, which already spans that projection."""
    b, _, j = _comparison_aucs(
        VARIANTS[0][0], VARIANTS[0][1], downstream=None
    )  # None selects logistic regression.
    lr_lift = float(j.mean()) - float(b.mean())
    assert lr_lift < MIN_LIFT_VS_BASELINE, (
        f"Joint PCA improved logistic-regression AUC by {lr_lift:+.4f}. The "
        f"required upper bound is +{MIN_LIFT_VS_BASELINE:.2f}."
    )

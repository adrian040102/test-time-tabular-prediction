"""Synthetic positive-lift tests for M20 ``ClusterMembershipFE``.

M20 clusters the combined train and test feature space with KMeans. Feature-mode
variants append cluster-density summaries. Weight mode returns cluster-based
importance weights. The feature tests use a density-shift checkerboard, while the
weight test uses a covariate-shift design.

A train-only clustering control provides the feature-mode comparison without test
rows in the clustering or cluster counts. Each feature variant must exceed both
the inductive baseline and this control by at least 0.05. The weight variant must
exceed its inductive baseline by at least 0.05. Test labels are used only for the
final AUC calculation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.tier2 import ClusterMembershipFE
from tests._synthetic_data import make_cluster_density_shift, make_covariate_shift_weighted
from tests._synthetic_evaluation import weighted_augment_auc


N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05
MIN_TESTTIME_VALUE = 0.05

FEATURE_KW = dict(n_train=600, n_test=1500, n_noise=4, c=2.0, sigma=0.9)
WEIGHT_KW = dict(n_train=900, n_test=900, n_noise=4,
                 train_s1_frac=0.15, test_s1_frac=0.85, theta1_deg=90.0)

FEATURE_BASELINE_CEILING = 0.60   # The checkerboard linear baseline remains near chance.
CONTROL_CEILING = 0.60            # train-only cluster control is near chance on uniform train
WEIGHT_BASELINE_BAND = (0.55, 0.75)  # Expected range for the reweight baseline.

FEATURE_VARIANTS = [
    ("cluster_membership_k5_both",
     lambda: ClusterMembershipFE(n_clusters=5, return_as="both")),
    ("cluster_membership_k10_both",
     lambda: ClusterMembershipFE(n_clusters=10, return_as="both")),
    ("cluster_membership_k10_feature",
     lambda: ClusterMembershipFE(n_clusters=10, return_as="feature")),
    ("cluster_membership_k15_both",
     lambda: ClusterMembershipFE(n_clusters=15, return_as="both")),
    ("cluster_membership_k10_both_cumshift",
     lambda: ClusterMembershipFE(n_clusters=10, return_as="both", feature_subset="cumshift")),
]
WEIGHT_VARIANT = ("cluster_membership_k10_weight",
                  lambda: ClusterMembershipFE(n_clusters=10, return_as="weight"))

_CACHE: dict = {}


def _train_only_cluster_control(X_tr_df, X_te_df, y_tr, y_te, n_clusters=10):
    """Compute cluster features from training rows only.

    Test rows are assigned to training clusters but do not affect the fit or
    per-cluster counts. The downstream model is otherwise unchanged.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    Xtr = X_tr_df.values.astype(float)
    Xte = X_te_df.values.astype(float)
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    n_tr = len(Ztr)
    k = n_clusters if n_tr >= n_clusters * 10 else max(2, n_tr // 10)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Ztr)
    lab_tr = km.predict(Ztr)
    lab_te = km.predict(Zte)
    rel = np.array([(lab_tr == c).sum() / n_tr for c in range(k)])
    dist_tr = np.linalg.norm(Ztr - km.cluster_centers_[lab_tr], axis=1)
    dist_te = np.linalg.norm(Zte - km.cluster_centers_[lab_te], axis=1)
    Xtr_out = np.column_stack([Ztr, rel[lab_tr], dist_tr])
    Xte_out = np.column_stack([Zte, rel[lab_te], dist_te])
    clf = LogisticRegression(max_iter=1000, random_state=0).fit(Xtr_out, np.asarray(y_tr))
    return float(roc_auc_score(y_te, clf.predict_proba(Xte_out)[:, 1]))


def _mean(key, fn):
    if key not in _CACHE:
        _CACHE[key] = float(np.mean(fn()))
    return _CACHE[key]


def _feature_baseline():
    return _mean("feat_base", lambda: [
        weighted_augment_auc(InductiveBaseline(), *make_cluster_density_shift(seed=s, **FEATURE_KW))
        for s in range(N_SEEDS)])


def _feature_control():
    return _mean("feat_ctrl", lambda: [
        _train_only_cluster_control(*make_cluster_density_shift(seed=s, **FEATURE_KW), n_clusters=10)
        for s in range(N_SEEDS)])


def _weight_baseline():
    return _mean("wt_base", lambda: [
        weighted_augment_auc(InductiveBaseline(), *make_covariate_shift_weighted(seed=s, **WEIGHT_KW))
        for s in range(N_SEEDS)])


def _feature_joint(label, factory):
    return _mean(("feat", label), lambda: [
        weighted_augment_auc(factory(), *make_cluster_density_shift(seed=s, **FEATURE_KW))
        for s in range(N_SEEDS)])


@pytest.mark.parametrize("label,factory", FEATURE_VARIANTS)
def test_feature_variants_exceed_baseline_and_control(label, factory):
    """Each feature variant must exceed both comparison AUCs by at least 0.05."""
    base = _feature_baseline()
    ctrl = _feature_control()
    joint = _feature_joint(label, factory)
    lift = joint - base
    ttv = joint - ctrl
    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"{label} baseline AUC difference {lift:+.4f} is below "
        f"{MIN_LIFT_VS_BASELINE:.2f} "
        f"(baseline {base:.4f}, joint {joint:.4f})."
    )
    assert ttv >= MIN_TESTTIME_VALUE, (
        f"{label} control AUC difference {ttv:+.4f} is below "
        f"{MIN_TESTTIME_VALUE:.2f} "
        f"(train-only-cluster control {ctrl:.4f}, joint {joint:.4f})."
    )


def test_weight_variant_exceeds_baseline_on_reweight_design():
    """The weight variant must exceed the reweight baseline AUC by at least 0.05."""
    base = _weight_baseline()
    label, factory = WEIGHT_VARIANT
    joint = _mean(("wt", label), lambda: [
        weighted_augment_auc(factory(), *make_covariate_shift_weighted(seed=s, **WEIGHT_KW))
        for s in range(N_SEEDS)])
    lift = joint - base
    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"{label} AUC difference {lift:+.4f} is below {MIN_LIFT_VS_BASELINE:.2f} "
        f"(baseline {base:.4f}, joint {joint:.4f})."
    )


def test_baselines_meet_expected_ceilings():
    """The feature design puts the baseline at chance (XOR). The train-only cluster
    control also stays near chance because uniform training data produces nearly
    constant cluster features. The weight-design baseline remains below its
    specified ceiling."""
    fb = _feature_baseline()
    fc = _feature_control()
    wb = _weight_baseline()
    assert fb < FEATURE_BASELINE_CEILING, (
        f"Feature baseline {fb:.4f} not near chance (< {FEATURE_BASELINE_CEILING})."
    )
    assert fc < CONTROL_CEILING, (
        f"Train-only cluster control {fc:.4f} exceeds {CONTROL_CEILING}. "
        f"The expected uniform-training control behavior no longer holds."
    )
    assert WEIGHT_BASELINE_BAND[0] < wb < WEIGHT_BASELINE_BAND[1], (
        f"Reweight baseline {wb:.4f} outside the misspecified band {WEIGHT_BASELINE_BAND}."
    )


def test_control_stays_near_baseline_so_lift_is_test_time_value():
    """The train-only cluster control must stay within 0.05 of the baseline."""
    base = _feature_baseline()
    ctrl = _feature_control()
    assert abs(ctrl - base) < 0.05, (
        f"Train-only-cluster control {ctrl:.4f} differs from baseline {base:.4f} by "
        f"≥ 0.05. The control differs from the baseline more than expected."
    )

"""Synthetic positive-lift tests for M8 ``LabelPropagation``.

M8 builds a nearest-neighbor graph over the combined train and test rows. It
propagates known training labels and appends the resulting soft label as a
feature. The synthetic design has three modes with a nonlinear label pattern.
One class is rare in training data but common in test data, allowing unlabeled
test rows to connect its graph region.

A train-only control computes out-of-fold nearest-neighbor soft labels without
including test rows as graph nodes. Each joint variant must exceed both the
inductive baseline and this control by at least 0.05. Test labels are used only
for the final AUC calculation.
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
    LabelPropagationFeature, LabelPropagationWeighted, LabelPropagationEnhanced,
)
from tests._synthetic_data import make_three_mode_xor
from tests._synthetic_evaluation import weighted_augment_auc


N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05
MIN_TESTTIME_VALUE = 0.05

# Fixed M8 design using ``make_three_mode_xor``.
GEN_KW = dict(
    n_train=600, n_test=1500, n_informative=3, n_noise=4,
    train_weights=(0.475, 0.05, 0.475), test_weights=(0.05, 0.90, 0.05),
)
BASELINE_CEILING = 0.60          # The linear baseline remains near chance.
CONTROL_FLOOR_OVER_BASELINE = 0.20  # the control must be substantially stronger than the baseline

VARIANTS = [
    ("label_prop_weighted_adaptive_a05",
     lambda: LabelPropagationWeighted(k=15, alpha=0.5, adaptive_k=True)),
    ("label_prop_v3_adaptive",
     lambda: LabelPropagationEnhanced(k=15, alpha=0.5, feature_threshold=0.05, adaptive_k=True)),
    ("label_prop_weighted_k15_a05",
     lambda: LabelPropagationWeighted(k=15, alpha=0.5, adaptive_k=False)),
    ("label_prop_v3_t005_k15",
     lambda: LabelPropagationEnhanced(k=15, alpha=0.5, feature_threshold=0.05)),
    ("label_prop_k15_a05", lambda: LabelPropagationFeature(k=15, alpha=0.5)),
    ("label_prop_k10_a05", lambda: LabelPropagationFeature(k=10, alpha=0.5)),
]

_CACHE: dict = {}


def _train_only_knn_oof(X_tr_df, X_te_df, y_tr, y_te, k=15):
    """Compute a train-only nearest-neighbor soft-label control.

    Test rows are not graph nodes. The control uses the same out-of-fold procedure
    and neighbor count as the evaluated methods.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    Xtr = X_tr_df.values.astype(float)
    Xte = X_te_df.values.astype(float)
    sc = StandardScaler().fit(np.vstack([Xtr, Xte]))
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    y = np.asarray(y_tr).astype(float)
    n = len(Ztr)
    soft_tr = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for tri, vai in kf.split(np.arange(n)):
        nn = NearestNeighbors(n_neighbors=min(k, len(tri))).fit(Ztr[tri])
        _, idx = nn.kneighbors(Ztr[vai])
        soft_tr[vai] = y[tri][idx].mean(1)
    nn = NearestNeighbors(n_neighbors=min(k, n)).fit(Ztr)
    _, idx = nn.kneighbors(Zte)
    soft_te = y[idx].mean(1)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(np.column_stack([Ztr, soft_tr]), y)
    return float(roc_auc_score(y_te, clf.predict_proba(np.column_stack([Zte, soft_te]))[:, 1]))


def _mean_auc(key, factory):
    """Memoized mean test AUC of ``factory`` over N_SEEDS on the fixed design."""
    if key not in _CACHE:
        vals = [weighted_augment_auc(factory(), *make_three_mode_xor(seed=s, **GEN_KW))
                for s in range(N_SEEDS)]
        _CACHE[key] = float(np.mean(vals))
    return _CACHE[key]


def _baseline():
    return _mean_auc("baseline", InductiveBaseline)


def _control():
    if "control" not in _CACHE:
        vals = [_train_only_knn_oof(*make_three_mode_xor(seed=s, **GEN_KW), k=15)
                for s in range(N_SEEDS)]
        _CACHE["control"] = float(np.mean(vals))
    return _CACHE["control"]


@pytest.mark.parametrize("label,factory", VARIANTS)
def test_variant_exceeds_baseline(label, factory):
    """Each variant must exceed the inductive baseline AUC by at least 0.05."""
    base = _baseline()
    joint = _mean_auc(label, factory)
    lift = joint - base
    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"{label} AUC difference {lift:+.4f} is below {MIN_LIFT_VS_BASELINE:.2f} "
        f"(baseline {base:.4f}, joint {joint:.4f})."
    )


@pytest.mark.parametrize("label,factory", VARIANTS)
def test_variant_exceeds_train_only_control(label, factory):
    """Each variant must exceed the train-only control AUC by at least 0.05."""
    ctrl = _control()
    joint = _mean_auc(label, factory)
    ttv = joint - ctrl
    assert ttv >= MIN_TESTTIME_VALUE, (
        f"{label} AUC difference {ttv:+.4f} is below {MIN_TESTTIME_VALUE:.2f} "
        f"(control {ctrl:.4f}, joint {joint:.4f})."
    )


def test_baseline_meets_expected_ceiling():
    """XOR-by-mode is nonlinear, so raw LR remains near chance
    (below the ceiling)."""
    base = _baseline()
    assert base < BASELINE_CEILING, (
        f"Baseline {base:.4f} is not below {BASELINE_CEILING}. The baseline task "
        f"is easier than expected."
    )


def test_control_exceeds_baseline_floor():
    """The train-only control must exceed the baseline by the required amount."""
    base = _baseline()
    ctrl = _control()
    assert ctrl - base >= CONTROL_FLOOR_OVER_BASELINE, (
        f"Control AUC {ctrl:.4f} differs from baseline AUC {base:.4f} by less "
        f"than {CONTROL_FLOOR_OVER_BASELINE}."
    )

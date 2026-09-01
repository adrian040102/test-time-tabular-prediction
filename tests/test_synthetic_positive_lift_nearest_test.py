"""Synthetic positive-lift test for M16 NearestTestDistance.

The test evaluates the feature channel on a core/periphery covariate shift. A
shared core is close to both splits, while train and test peripheries occupy
orthogonal directions. The raw linear boundary does not transfer, but
cross-split distance features have the same direction of association in both
splits.

Each tested variant must exceed ``InductiveBaseline`` by
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
from src.methods.tier2 import NearestTestDistance, NearestTestDistanceRelative
from tests._synthetic_data import make_core_periphery_shift


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
N_SAMPLES_TRAIN = 800
N_SAMPLES_TEST = 800
N_INFORMATIVE = 2
N_NOISE = 4
CLUSTER_STD = 1.2
PERIPHERY_DISTANCE = 4.0
N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05

# Three configurations evaluated end to end.
EVALUATED_VARIANTS = [
    pytest.param(
        lambda: NearestTestDistance(k=5, return_as="both", feature_weighting="none"),
        "nearest_test_k5_both",
        id="nearest_test_k5_both",
    ),
    pytest.param(
        lambda: NearestTestDistance(k=10, return_as="both", feature_weighting="none"),
        "nearest_test_k10_both",
        id="nearest_test_k10_both",
    ),
    pytest.param(
        lambda: NearestTestDistanceRelative(
            k=10, return_as="both", feature_weighting="none"
        ),
        "nearest_test_rel_k10_both",
        id="nearest_test_rel_k10_both",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logistic_auc(method, X_tr_df, X_te_df, y_tr, y_te) -> float:
    """Fit ``method`` and train logistic regression, then return AUC.

    Only the feature channel is exercised. ``sample_weights_train`` is not passed
    to the model. Scaler handling follows the experiment pipeline contract.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    result = method.fit_transform(X_tr_df, X_te_df, y_tr)
    X_tr = result.X_train
    X_te = result.X_test
    if not result.skip_pipeline_scaler:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba))


def _run_variant_vs_baseline(method_factory) -> tuple[float, float, list[float], list[float]]:
    """Run ``method_factory()`` against ``InductiveBaseline`` for ``N_SEEDS``."""
    aucs_baseline: list[float] = []
    aucs_method: list[float] = []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = make_core_periphery_shift(
            n_train=N_SAMPLES_TRAIN,
            n_test=N_SAMPLES_TEST,
            n_informative=N_INFORMATIVE,
            n_noise=N_NOISE,
            cluster_std=CLUSTER_STD,
            periphery_distance=PERIPHERY_DISTANCE,
            seed=seed,
        )
        aucs_baseline.append(_logistic_auc(InductiveBaseline(), X_tr, X_te, y_tr, y_te))
        aucs_method.append(_logistic_auc(method_factory(), X_tr, X_te, y_tr, y_te))
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
def test_nearest_test_exceeds_baseline_on_core_periphery(method_factory, variant_label):
    """Each M16 variant must exceed the baseline by the configured margin."""
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
        f"  Per-seed M16 AUCs:          {[round(a, 3) for a in aucs_method]}\n"
        f"  Per-seed baseline AUCs:     {[round(a, 3) for a in aucs_baseline]}\n"
        f"  Per-seed lifts:             {[round(l, 3) for l in per_seed_lifts]}\n"
    )

    assert lift_vs_baseline >= MIN_LIFT_VS_BASELINE, (
        f"{variant_label} AUC difference from InductiveBaseline is below "
        f"{MIN_LIFT_VS_BASELINE:.2f} on the core/periphery test.{msg}"
    )


def test_baseline_does_not_transfer_across_splits():
    """Training AUC must be at least 0.90 and test AUC must not exceed 0.65."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    train_aucs: list[float] = []
    test_aucs: list[float] = []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = make_core_periphery_shift(
            n_train=N_SAMPLES_TRAIN,
            n_test=N_SAMPLES_TEST,
            n_informative=N_INFORMATIVE,
            n_noise=N_NOISE,
            cluster_std=CLUSTER_STD,
            periphery_distance=PERIPHERY_DISTANCE,
            seed=seed,
        )
        result = InductiveBaseline().fit_transform(X_tr, X_te, y_tr)
        scaler = StandardScaler()
        Xa = scaler.fit_transform(result.X_train)
        Xb = scaler.transform(result.X_test)
        clf = LogisticRegression(max_iter=1000, random_state=0)
        clf.fit(Xa, y_tr)
        train_aucs.append(float(roc_auc_score(y_tr, clf.predict_proba(Xa)[:, 1])))
        test_aucs.append(float(roc_auc_score(y_te, clf.predict_proba(Xb)[:, 1])))

    mean_train = float(np.mean(train_aucs))
    mean_test = float(np.mean(test_aucs))
    msg = (
        f"\n  Mean baseline TRAIN AUC: {mean_train:.4f} (expected >= 0.90)\n"
        f"  Mean baseline TEST  AUC: {mean_test:.4f} (expected <= 0.65)\n"
    )
    assert mean_train >= 0.90, f"Training AUC is below 0.90.{msg}"
    assert mean_test <= 0.65, f"Test AUC exceeds 0.65.{msg}"

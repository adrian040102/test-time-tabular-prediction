"""Synthetic positive-lift tests for M19 NeighborhoodConsensusFeatures.

A noisy circular classification problem keeps the raw linear baseline near chance,
while a covariate-shifted test split supplies within-test neighborhood structure.
The full method is compared with ``InductiveBaseline``.

A second comparison removes all consensus columns except the pass-one self
prediction. This separates neighborhood consensus from out-of-fold stacking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.tier2 import NeighborhoodConsensusFeatures
from tests._synthetic_data import make_noisy_consensus


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
N_SAMPLES_TRAIN = 900
N_SAMPLES_TEST = 900
N_INFORMATIVE = 2
N_NOISE = 4
LABEL_NOISE = 0.25
SHIFT = 0.5
N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05

# Four variants evaluated on the noisy-consensus mechanism.
EVALUATED_VARIANTS = [
    pytest.param(
        lambda: NeighborhoodConsensusFeatures(k=10, weighting="distance"),
        "neighbor_consensus_k10_dist",
        id="neighbor_consensus_k10_dist",
    ),
    pytest.param(
        lambda: NeighborhoodConsensusFeatures(k=10, weighting="uniform"),
        "neighbor_consensus_k10_uniform",
        id="neighbor_consensus_k10_uniform",
    ),
    pytest.param(
        lambda: NeighborhoodConsensusFeatures(k=5, weighting="distance"),
        "neighbor_consensus_k5_dist",
        id="neighbor_consensus_k5_dist",
    ),
    pytest.param(
        lambda: NeighborhoodConsensusFeatures(
            k=10, weighting="distance", feature_subset="cumshift"
        ),
        "neighbor_consensus_k10_dist_cumshift",
        id="neighbor_consensus_k10_dist_cumshift",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lr_auc_on_arrays(X_tr, X_te, y_tr, y_te, skip_scaler: bool) -> float:
    """Train logistic regression on the feature arrays and return test AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X_tr = np.asarray(X_tr)
    X_te = np.asarray(X_te)
    if not skip_scaler:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba))


def _logistic_auc(method, X_tr_df, X_te_df, y_tr, y_te) -> float:
    """Fit ``method`` and score logistic regression on its full feature set.

    M19 emits
    no sample weights, so only the feature channel is exercised.
    """
    result = method.fit_transform(X_tr_df, X_te_df, y_tr)
    return _lr_auc_on_arrays(
        result.X_train, result.X_test, y_tr, y_te, result.skip_pipeline_scaler
    )


def _full_and_self_auc(method, X_tr_df, X_te_df, y_tr, y_te) -> tuple[float, float]:
    """Fit ``method`` once and return full and self-only AUCs.

    The self-only features include all original columns and
    ``nbr_consensus_self``. Other ``nbr_consensus_*`` columns are excluded.
    """
    result = method.fit_transform(X_tr_df, X_te_df, y_tr)
    names = list(result.feature_names)
    X_tr = np.asarray(result.X_train)
    X_te = np.asarray(result.X_test)

    full = _lr_auc_on_arrays(X_tr, X_te, y_tr, y_te, result.skip_pipeline_scaler)

    keep = [
        i for i, nm in enumerate(names)
        if (not nm.startswith("nbr_consensus_")) or nm == "nbr_consensus_self"
    ]
    self_only = _lr_auc_on_arrays(
        X_tr[:, keep], X_te[:, keep], y_tr, y_te, result.skip_pipeline_scaler
    )
    return full, self_only


def _run_variant_vs_baseline(method_factory):
    """Run ``method_factory()`` against ``InductiveBaseline`` for ``N_SEEDS``.

    Returns (mean_baseline, mean_full, mean_self, baseline_list, full_list,
    self_list). The method is fit once per seed and both feature sets are scored
    from that fit.
    """
    aucs_baseline: list[float] = []
    aucs_full: list[float] = []
    aucs_self: list[float] = []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = make_noisy_consensus(
            n_train=N_SAMPLES_TRAIN,
            n_test=N_SAMPLES_TEST,
            n_informative=N_INFORMATIVE,
            n_noise=N_NOISE,
            label_noise=LABEL_NOISE,
            shift=SHIFT,
            seed=seed,
        )
        aucs_baseline.append(_logistic_auc(InductiveBaseline(), X_tr, X_te, y_tr, y_te))
        full, self_only = _full_and_self_auc(method_factory(), X_tr, X_te, y_tr, y_te)
        aucs_full.append(full)
        aucs_self.append(self_only)
    return (
        float(np.mean(aucs_baseline)),
        float(np.mean(aucs_full)),
        float(np.mean(aucs_self)),
        aucs_baseline,
        aucs_full,
        aucs_self,
    )


# ---------------------------------------------------------------------------
# Positive-lift comparison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_factory,variant_label", EVALUATED_VARIANTS)
def test_neighbor_consensus_exceeds_baseline_on_noisy_consensus(
    method_factory, variant_label
):
    """Each M19 variant must exceed the baseline by the configured margin."""
    mean_baseline, mean_full, mean_self, aucs_b, aucs_f, aucs_s = (
        _run_variant_vs_baseline(method_factory)
    )
    lift_vs_baseline = mean_full - mean_baseline
    per_seed_lifts = [f - b for f, b in zip(aucs_f, aucs_b)]
    consensus_over_stacking = mean_full - mean_self

    msg = (
        f"\n  Variant: {variant_label}\n"
        f"  N seeds: {N_SEEDS}\n"
        f"  Mean AUC InductiveBaseline: {mean_baseline:.4f}\n"
        f"  Mean AUC {variant_label} (full):   {mean_full:.4f}\n"
        f"  Mean lift vs baseline:      {lift_vs_baseline:+.4f} "
        f"(threshold: >= +{MIN_LIFT_VS_BASELINE:.2f})\n"
        f"  Mean AUC self-only (stacking):     {mean_self:.4f}\n"
        f"  Consensus over stacking:           {consensus_over_stacking:+.4f}\n"
        f"  Per-seed M19 AUCs:          {[round(a, 3) for a in aucs_f]}\n"
        f"  Per-seed baseline AUCs:     {[round(a, 3) for a in aucs_b]}\n"
        f"  Per-seed lifts:             {[round(l, 3) for l in per_seed_lifts]}\n"
    )

    assert lift_vs_baseline >= MIN_LIFT_VS_BASELINE, (
        f"{variant_label} AUC difference from InductiveBaseline is below "
        f"{MIN_LIFT_VS_BASELINE:.2f} on the noisy-consensus test.{msg}"
    )


# ---------------------------------------------------------------------------
# Consensus and stacking comparison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_factory,variant_label", EVALUATED_VARIANTS)
def test_consensus_and_stacking_aucs_are_finite(method_factory, variant_label):
    """Both consensus and self-only stacking AUCs must be finite."""
    _, mean_full, mean_self, _, _, _ = _run_variant_vs_baseline(method_factory)
    assert np.isfinite(mean_full), f"{variant_label}: full AUC is not finite"
    assert np.isfinite(mean_self), f"{variant_label}: self-only AUC is not finite"


# ---------------------------------------------------------------------------
# Baseline ceiling
# ---------------------------------------------------------------------------

def test_baseline_is_near_chance_on_circle():
    """The linear baseline AUC on the circular design must not exceed 0.60."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    aucs: list[float] = []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = make_noisy_consensus(
            n_train=N_SAMPLES_TRAIN,
            n_test=N_SAMPLES_TEST,
            n_informative=N_INFORMATIVE,
            n_noise=N_NOISE,
            label_noise=LABEL_NOISE,
            shift=SHIFT,
            seed=seed,
        )
        result = InductiveBaseline().fit_transform(X_tr, X_te, y_tr)
        scaler = StandardScaler()
        Xa = scaler.fit_transform(np.asarray(result.X_train))
        Xb = scaler.transform(np.asarray(result.X_test))
        clf = LogisticRegression(max_iter=1000, random_state=0)
        clf.fit(Xa, y_tr)
        aucs.append(float(roc_auc_score(y_te, clf.predict_proba(Xb)[:, 1])))

    mean_auc = float(np.mean(aucs))
    msg = (
        f"\n  Mean baseline AUC on circle: {mean_auc:.4f} "
        f"(expected <= 0.60)\n"
        f"  Per-seed: {[round(a, 3) for a in aucs]}\n"
    )
    assert mean_auc <= 0.60, f"Baseline AUC exceeds 0.60.{msg}"

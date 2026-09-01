"""Synthetic threshold tests for M9 OT displacement.

A three-mode mixture places the positive class in the two outer modes and
concentrates test mass in the middle mode. Absolute or magnitude displacement
features can identify mode membership even though the raw logistic-regression
baseline remains near chance.

Magnitude-based variants must exceed ``InductiveBaseline`` by the configured
threshold. The signed-v2 variant must remain below the threshold because train
and test displacements have mirrored signs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.tier2 import OTDisplacementFeatures, OTDisplacementImproved
from tests._synthetic_data import make_three_mode_xor


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
N_SAMPLES_TRAIN = 800
N_SAMPLES_TEST = 400
N_INFORMATIVE = 4
N_NOISE = 4
N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05

# Variants expected to exceed the positive-lift threshold.
PRIMARY_VARIANTS = [
    pytest.param(
        lambda: OTDisplacementImproved(mode="selective", abs_disp=True, seed=42),
        "ot_disp_v2_selective_abs",
        id="ot_disp_v2_selective_abs",
    ),
    pytest.param(
        lambda: OTDisplacementImproved(
            mode="selective_and_summary", abs_disp=True, seed=42
        ),
        "ot_disp_v2_selective_and_summary_abs",
        id="ot_disp_v2_selective_and_summary_abs",
    ),
    pytest.param(
        lambda: OTDisplacementFeatures(append_magnitude=True),
        "ot_displacement_mag",
        id="ot_displacement_mag",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logistic_auc(method, X_tr_df, X_te_df, y_tr, y_te) -> float:
    """Fit ``method`` and logistic regression on its features, then return AUC.

    M9 variants do not set ``skip_pipeline_scaler``, but the branch is retained
    for consistency with the experiment pipeline.
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
        X_tr, X_te, y_tr, y_te = make_three_mode_xor(
            n_train=N_SAMPLES_TRAIN,
            n_test=N_SAMPLES_TEST,
            n_informative=N_INFORMATIVE,
            n_noise=N_NOISE,
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

@pytest.mark.parametrize("method_factory,variant_label", PRIMARY_VARIANTS)
def test_ot_displacement_exceeds_baseline_on_three_mode_xor(
    method_factory, variant_label,
):
    """Require each magnitude-based variant to exceed the baseline threshold."""
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
        f"  Per-seed M9 AUCs:           {[round(a, 3) for a in aucs_method]}\n"
        f"  Per-seed baseline AUCs:     {[round(a, 3) for a in aucs_baseline]}\n"
        f"  Per-seed lifts:             {[round(l, 3) for l in per_seed_lifts]}\n"
    )

    assert lift_vs_baseline >= MIN_LIFT_VS_BASELINE, (
        f"{variant_label} AUC difference from InductiveBaseline is below "
        f"{MIN_LIFT_VS_BASELINE:.2f} on the three-mode XOR test.{msg}"
    )


def test_signed_v2_stays_below_threshold_due_to_sign_mirror():
    """The signed-v2 AUC difference must remain below the configured threshold."""
    mean_baseline, mean_method, aucs_baseline, aucs_method = _run_variant_vs_baseline(
        lambda: OTDisplacementImproved(mode="selective", abs_disp=False, seed=42)
    )
    lift_vs_baseline = mean_method - mean_baseline

    assert lift_vs_baseline < MIN_LIFT_VS_BASELINE, (
        f"ot_disp_v2_selective signed-v2 AUC difference {lift_vs_baseline:+.4f} "
        f"is not below {MIN_LIFT_VS_BASELINE:+.2f} on the three-mode XOR test."
    )

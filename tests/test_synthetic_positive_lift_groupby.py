"""Synthetic positive-lift test for M14 GroupByShiftRatio.

The log-ratio variant is evaluated on a controlled dataset where per-group
test-time mean shifts encode the label. Shuffled group codes and noisy row-level
values keep the logistic-regression baseline near chance, while the group-level
log ratio exposes the shared shift signal.

The mean method AUC must exceed InductiveBaseline by
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
from src.methods.tier1 import GroupByShiftRatio
from tests._synthetic_data import make_groupby_mean_shift


# ---------------------------------------------------------------------------
# Test parameters are defined in the module docstring.
# ---------------------------------------------------------------------------
N_SAMPLES_TRAIN = 6000
N_SAMPLES_TEST = 6000
N_CATS = 100
N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05

# Log-ratio variant evaluated on the designed mean-shift mechanism.
EVALUATED_VARIANTS = [
    pytest.param(
        lambda: GroupByShiftRatio(use_log=True),
        "groupby_shift_ratio_log",
        id="groupby_shift_ratio_log",
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

    X_tr = np.asarray(X_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
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

    GroupByShiftRatio emits no
    sample weights, so only the feature channel is exercised.
    """
    result = method.fit_transform(X_tr_df, X_te_df, y_tr)
    return _lr_auc_on_arrays(
        result.X_train, result.X_test, y_tr, y_te, result.skip_pipeline_scaler
    )


def _run_variant_vs_baseline(method_factory):
    """Run ``method_factory()`` against ``InductiveBaseline`` for ``N_SEEDS``."""
    aucs_baseline: list[float] = []
    aucs_method: list[float] = []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = make_groupby_mean_shift(
            n_train=N_SAMPLES_TRAIN,
            n_test=N_SAMPLES_TEST,
            n_cats=N_CATS,
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
def test_groupby_shift_ratio_exceeds_baseline_on_mean_shift(method_factory, variant_label):
    """The log-ratio variant must exceed the baseline by the required margin."""
    mean_baseline, mean_method, aucs_b, aucs_m = _run_variant_vs_baseline(method_factory)
    lift = mean_method - mean_baseline
    per_seed_lifts = [m - b for m, b in zip(aucs_m, aucs_b)]

    msg = (
        f"\n  Variant: {variant_label}\n"
        f"  N seeds: {N_SEEDS}\n"
        f"  Mean AUC InductiveBaseline: {mean_baseline:.4f}\n"
        f"  Mean AUC {variant_label}:   {mean_method:.4f}\n"
        f"  Mean lift vs baseline:      {lift:+.4f} "
        f"(threshold: >= +{MIN_LIFT_VS_BASELINE:.2f})\n"
        f"  Per-seed method AUCs:       {[round(a, 3) for a in aucs_m]}\n"
        f"  Per-seed baseline AUCs:     {[round(a, 3) for a in aucs_b]}\n"
        f"  Per-seed lifts:             {[round(l, 3) for l in per_seed_lifts]}\n"
    )

    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"{variant_label} AUC difference from InductiveBaseline is below "
        f"{MIN_LIFT_VS_BASELINE:.2f} on the groupby-mean-shift test.{msg}"
    )


# ---------------------------------------------------------------------------
# Baseline ceiling
# ---------------------------------------------------------------------------

def test_baseline_is_near_chance_on_mean_shift():
    """The mean linear-baseline AUC must not exceed 0.60."""
    aucs: list[float] = []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = make_groupby_mean_shift(
            n_train=N_SAMPLES_TRAIN,
            n_test=N_SAMPLES_TEST,
            n_cats=N_CATS,
            seed=seed,
        )
        aucs.append(_logistic_auc(InductiveBaseline(), X_tr, X_te, y_tr, y_te))

    mean_auc = float(np.mean(aucs))
    msg = (
        f"\n  Mean baseline AUC: {mean_auc:.4f} (expected <= 0.60)\n"
        f"  Per-seed: {[round(a, 3) for a in aucs]}\n"
    )
    assert mean_auc <= 0.60, f"Baseline AUC exceeds 0.60.{msg}"


# ---------------------------------------------------------------------------
# Shift-ratio feature contract
# ---------------------------------------------------------------------------

def test_shift_ratio_column_is_produced_and_separates_shift_labels():
    """The expected shift-ratio column must separate the two shift labels."""
    import pandas as pd

    X_tr, X_te, y_tr, y_te = make_groupby_mean_shift(seed=0)
    result = GroupByShiftRatio(use_log=True).fit_transform(X_tr, X_te, y_tr)

    shift_cols = [c for c in result.feature_names if c.endswith("_shift_ratio")]
    assert shift_cols == ["group_x_shift_ratio"], (
        f"expected exactly one shift-ratio column from the (group, x) pair, got "
        f"{shift_cols} (feature_names={list(result.feature_names)})"
    )

    sr = pd.DataFrame(result.X_test, columns=list(result.feature_names))[
        "group_x_shift_ratio"
    ].to_numpy(dtype=float)
    up_mean = sr[y_te == 1].mean()
    down_mean = sr[y_te == 0].mean()
    assert up_mean > down_mean, (
        f"shift ratio is not aligned with shift labels: up={up_mean:.3f} down={down_mean:.3f}"
    )

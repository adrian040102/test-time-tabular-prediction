"""Synthetic positive-lift tests for two groupby-selective wrappers.

A SelectiveMethod restricts its inner method to selected columns. For group-by
methods, this can limit the number of group-by features created from combinations
of grouping and aggregation columns. The substantive comparison is therefore
the wrapper versus its standalone base, in addition to the comparison with
InductiveBaseline.

The controlled high-dimensional dataset contains one signal group/numeric pair
and many noise pairs. Each wrapper uses parameters chosen for its feature
expansion and must exceed both configured comparison margins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.column_selection import CombinedSelector
from src.methods.tier1 import get_tier1_method
from tests._synthetic_data import make_high_dim_groupby_shift


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05
MIN_WRAPPER_OVER_STANDALONE = 0.01
BASELINE_CHANCE_MAX = 0.60

# Two groupby-selective wrappers, their standalone bases and per-method
# generator parameters.
VARIANTS = [
    pytest.param(
        {
            "label": "joint_groupby_combined__sel_combined_union",
            "base_key": "joint_groupby_combined",
            "wrap_key": "joint_groupby_combined__sel_combined_union",
            # Three functions (mean, count and std) are evaluated with 250 rows
            # and 50 noise pairs.
            "gen": dict(n_train=250, n_test=250, n_signal_pairs=1, n_noise_pairs=50,
                        n_levels_signal=8, n_levels_noise=60),
        },
        id="joint_groupby_combined__sel_combined_union",
    ),
    pytest.param(
        {
            "label": "groupby_shift_cohen_d__sel_combined_union",
            "base_key": "groupby_shift_cohen_d",
            "wrap_key": "groupby_shift_cohen_d__sel_combined_union",
            # With min_group_size=5, each noise group needs enough rows to
            # produce a non-zero Cohen's d feature.
            "gen": dict(n_train=200, n_test=200, n_signal_pairs=1, n_noise_pairs=50,
                        n_levels_signal=8, n_levels_noise=20),
        },
        id="groupby_shift_cohen_d__sel_combined_union",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lr_auc_on_arrays(X_tr, X_te, y_tr, y_te, skip_scaler: bool) -> float:
    """Train logistic regression on the given feature arrays, return test AUC."""
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
    return float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))


def _logistic_auc(method, X_tr_df, X_te_df, y_tr, y_te) -> float:
    """Fit ``method`` and score logistic regression on all output features.

    Feature channel only (these methods emit no sample weights / augmentation).
    """
    result = method.fit_transform(X_tr_df, X_te_df, y_tr)
    return _lr_auc_on_arrays(
        result.X_train, result.X_test, y_tr, y_te, result.skip_pipeline_scaler
    )


# Cache each configuration so the four tests reuse the same five-seed evaluation.
_RESULTS_CACHE: dict[str, dict] = {}


def _get_results(variant: dict) -> dict:
    """Evaluate the baseline, standalone method, wrapper and no-noise reference."""
    label = variant["label"]
    if label in _RESULTS_CACHE:
        return _RESULTS_CACHE[label]

    base_key, wrap_key, gen = variant["base_key"], variant["wrap_key"], variant["gen"]
    n_sig = gen["n_signal_pairs"]
    no_noise_gen = {**gen, "n_noise_pairs": 0}

    baseline, standalone, wrapper, no_noise_reference, complete_pairs = [], [], [], [], []
    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = make_high_dim_groupby_shift(seed=seed, **gen)
        baseline.append(_logistic_auc(InductiveBaseline(), X_tr, X_te, y_tr, y_te))
        standalone.append(_logistic_auc(get_tier1_method(base_key), X_tr, X_te, y_tr, y_te))
        wrapper.append(_logistic_auc(get_tier1_method(wrap_key), X_tr, X_te, y_tr, y_te))

        # Record how many complete signal pairs the union selects.
        chosen = set(CombinedSelector(combination="union").select(X_tr, X_te, y_tr).all_cols)
        complete_pairs.append(
            sum(1 for i in range(n_sig)
                if f"g_sig{i}" in chosen and f"x_sig{i}" in chosen)
        )

        # Evaluate the standalone method without noise pairs for comparison.
        Xc_tr, Xc_te, yc_tr, yc_te = make_high_dim_groupby_shift(
            seed=seed, **no_noise_gen
        )
        no_noise_reference.append(
            _logistic_auc(get_tier1_method(base_key), Xc_tr, Xc_te, yc_tr, yc_te)
        )

    res = {
        "baseline": baseline, "standalone": standalone, "wrapper": wrapper,
        "no_noise_reference": no_noise_reference, "complete_pairs": complete_pairs,
        "mean_baseline": float(np.mean(baseline)),
        "mean_standalone": float(np.mean(standalone)),
        "mean_wrapper": float(np.mean(wrapper)),
        "mean_no_noise_reference": float(np.mean(no_noise_reference)),
        "per_seed_wrapper_minus_standalone": [
            w - s for w, s in zip(wrapper, standalone)
        ],
    }
    _RESULTS_CACHE[label] = res
    return res


def _fmt(variant: dict, r: dict) -> str:
    lift_b = r["mean_wrapper"] - r["mean_baseline"]
    wrapper_minus_standalone = r["mean_wrapper"] - r["mean_standalone"]
    n_pos = sum(1 for d in r["per_seed_wrapper_minus_standalone"] if d > 0)
    return (
        f"\n  Variant: {variant['label']}\n"
        f"  N seeds: {N_SEEDS}  gen={variant['gen']}\n"
        f"  Mean AUC InductiveBaseline: {r['mean_baseline']:.4f}\n"
        f"  Mean AUC standalone base:   {r['mean_standalone']:.4f} "
        f"(no-noise reference {r['mean_no_noise_reference']:.4f})\n"
        f"  Mean AUC wrapper:           {r['mean_wrapper']:.4f}\n"
        f"  wrapper - baseline:         {lift_b:+.4f} (threshold >= +{MIN_LIFT_VS_BASELINE:.2f})\n"
        f"  wrapper - standalone:       {wrapper_minus_standalone:+.4f} "
        f"(threshold > +{MIN_WRAPPER_OVER_STANDALONE:.2f}, "
        f"{n_pos}/{N_SEEDS} seeds positive)\n"
        f"  Per-seed wrapper-standalone:"
        f"{[round(d, 3) for d in r['per_seed_wrapper_minus_standalone']]}\n"
        f"  Complete signal pairs selected (per seed): {r['complete_pairs']}\n"
    )


# ---------------------------------------------------------------------------
# Positive-lift comparison: wrapper > baseline and wrapper > standalone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_wrapper_exceeds_baseline_and_standalone(variant):
    """Require each wrapper to exceed both comparison values by the configured margins."""
    r = _get_results(variant)
    msg = _fmt(variant, r)

    lift_vs_baseline = r["mean_wrapper"] - r["mean_baseline"]
    wrapper_over_standalone = r["mean_wrapper"] - r["mean_standalone"]

    assert lift_vs_baseline >= MIN_LIFT_VS_BASELINE, (
        f"FAIL: {variant['label']} did not exceed InductiveBaseline by "
        f"{MIN_LIFT_VS_BASELINE:.2f}.{msg}"
    )
    assert wrapper_over_standalone > MIN_WRAPPER_OVER_STANDALONE, (
        f"FAIL: {variant['label']} did not exceed its standalone base by "
        f"{MIN_WRAPPER_OVER_STANDALONE:.2f}. The test requires column selection "
        f"to improve over using all columns.{msg}"
    )


# ---------------------------------------------------------------------------
# Control 1: the high-dimensional data keeps the baseline near chance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_baseline_is_near_chance(variant):
    """Verify that the raw-feature logistic-regression baseline is near chance.

    The shuffled codes and large within-group spread prevent a linear model from
    using the per-group shift in the raw features. A higher baseline would no
    longer isolate the information added by the wrapper.
    """
    r = _get_results(variant)
    assert r["mean_baseline"] <= BASELINE_CHANCE_MAX, (
        f"Baseline AUC exceeds the configured ceiling.{_fmt(variant, r)}"
    )


# ---------------------------------------------------------------------------
# Control 2: the union selects a complete signal pair
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_wrapper_selects_signal_pair(variant):
    """Verify that the selector retains a complete signal pair on most seeds.

    If the union dropped the signal group or its numeric partner, the wrapper's
    group-by method could not produce the shift feature. The LightGBM importance
    ranking varies across the small training samples, so the requirement applies
    to a majority of seeds.
    """
    r = _get_results(variant)
    n_with_pair = sum(1 for cp in r["complete_pairs"] if cp >= 1)
    assert n_with_pair >= (N_SEEDS + 1) // 2, (
        f"Union failed to select a complete signal pair on a majority of seeds "
        f"({n_with_pair}/{N_SEEDS}). Per seed: {r['complete_pairs']}.{_fmt(variant, r)}"
    )


# ---------------------------------------------------------------------------
# Control 3: noise-pair expansion reduces standalone performance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
def test_noise_pair_expansion_reduces_standalone_auc(variant):
    """Verify that noise-pair feature expansion reduces standalone AUC.

    The no-noise reference uses the same signal pair without the 50 noise pairs.
    The selector removes part of the unnecessary feature expansion.
    """
    r = _get_results(variant)
    assert r["mean_standalone"] < r["mean_no_noise_reference"] - 0.02, (
        f"The standalone version does not show the expected feature-expansion cost. "
        f"The no-noise reference must exceed the expanded feature set.{_fmt(variant, r)}"
    )

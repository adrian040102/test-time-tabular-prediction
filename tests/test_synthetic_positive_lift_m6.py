"""Synthetic positive-lift tests for M6 ``AdversarialValidation``.

M6 trains an out-of-fold domain classifier on train and test features. It returns
``P(test|x)`` as a feature or an importance weight. The feature-mode tests use a
radial label design where a nonlinear domain score supplies information that the
linear downstream model cannot derive from the raw features. The weight-mode test
uses a covariate-shift design where test-like training rows require greater weight.

Each primary configuration must exceed its inductive baseline by at least 0.05.
Cross-design controls verify that the two modes respond to different shift
structures. A linear domain classifier and a no-shift control must remain below
the same threshold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.tier2 import AdversarialValidation
from tests._synthetic_data import make_test_region_positive, make_covariate_shift_weighted
from tests._synthetic_evaluation import weighted_augment_auc


N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05

RADIAL_KW = dict(n_train=800, n_test=1500, n_noise=4,
                 sigma_train=2.0, sigma_test=0.9, radius=1.5)
REWEIGHT_KW = dict(n_train=900, n_test=900, n_noise=4,
                   train_s1_frac=0.15, test_s1_frac=0.85, theta1_deg=90.0)

RADIAL_BASELINE_CEILING = 0.60     # circle label -> linear baseline ≈ chance
REWEIGHT_BASELINE_CEILING = 0.72   # subpop×direction interaction -> misspecified LR ≈ 0.64

_CACHE: dict = {}


def _mean_auc(key, factory, gen, gen_kw):
    """Memoized mean test AUC of ``factory`` over N_SEEDS on a given generator."""
    ck = (key, gen.__name__, tuple(sorted(gen_kw.items())))
    if ck not in _CACHE:
        vals = [weighted_augment_auc(factory(), *gen(seed=s, **gen_kw)) for s in range(N_SEEDS)]
        _CACHE[ck] = float(np.mean(vals))
    return _CACHE[ck]


def _radial_baseline():
    return _mean_auc("baseline", InductiveBaseline, make_test_region_positive, RADIAL_KW)


def _reweight_baseline():
    return _mean_auc("baseline", InductiveBaseline, make_covariate_shift_weighted, REWEIGHT_KW)


FEATURE_VARIANTS = [
    ("adversarial_lgbm_feature",
     lambda: AdversarialValidation(classifier="lgbm", return_as="feature")),
    ("adversarial_lgbm_cal_iso_feature",
     lambda: AdversarialValidation(classifier="lgbm", return_as="feature", calibrate="isotonic")),
    ("adversarial_rf_feature",
     lambda: AdversarialValidation(classifier="rf", return_as="feature")),
    ("adversarial_lgbm_both_ratio",
     lambda: AdversarialValidation(classifier="lgbm", return_as="both", weight_type="ratio")),
]


@pytest.mark.parametrize("label,factory", FEATURE_VARIANTS)
def test_feature_variants_exceed_baseline_on_radial(label, factory):
    """Each nonlinear feature variant must exceed the baseline AUC by at least 0.05."""
    base = _radial_baseline()
    joint = _mean_auc(label, factory, make_test_region_positive, RADIAL_KW)
    lift = joint - base
    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"{label} AUC difference {lift:+.4f} is below {MIN_LIFT_VS_BASELINE:.2f} "
        f"(baseline {base:.4f}, joint {joint:.4f})."
    )


def test_weight_raw_exceeds_baseline_on_reweight():
    """The weight mode must exceed the reweight-design baseline by at least 0.05."""
    base = _reweight_baseline()
    joint = _mean_auc(
        "adversarial_lgbm_weight_raw",
        lambda: AdversarialValidation(classifier="lgbm", return_as="weight", weight_type="raw"),
        make_covariate_shift_weighted, REWEIGHT_KW)
    lift = joint - base
    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"weight_raw AUC difference {lift:+.4f} is below {MIN_LIFT_VS_BASELINE:.2f} "
        f"(baseline {base:.4f}, joint {joint:.4f})."
    )


def test_baselines_meet_expected_ceilings():
    """Both design baselines must remain below their expected ceilings."""
    rb = _radial_baseline()
    wb = _reweight_baseline()
    assert rb < RADIAL_BASELINE_CEILING, (
        f"Radial baseline {rb:.4f} not near chance (< {RADIAL_BASELINE_CEILING})."
    )
    assert wb < REWEIGHT_BASELINE_CEILING, (
        f"Reweight baseline {wb:.4f} not misspecified (< {REWEIGHT_BASELINE_CEILING})."
    )


def test_logreg_feature_is_structurally_limited():
    """A linear domain score must remain below the radial-design threshold."""
    base = _radial_baseline()
    joint = _mean_auc(
        "adversarial_logreg_feature",
        lambda: AdversarialValidation(classifier="logreg", return_as="feature"),
        make_test_region_positive, RADIAL_KW)
    lift = joint - base
    assert lift < MIN_LIFT_VS_BASELINE, (
        f"adversarial_logreg_feature AUC difference {lift:+.4f} is not below "
        f"{MIN_LIFT_VS_BASELINE:.2f} on the radial design."
    )


def test_modes_are_complementary():
    """Each mode must remain below the threshold on the other mode's design."""
    # Feature mode on the reweight design.
    feat_on_reweight = _mean_auc(
        "adversarial_lgbm_feature",
        lambda: AdversarialValidation(classifier="lgbm", return_as="feature"),
        make_covariate_shift_weighted, REWEIGHT_KW)
    assert feat_on_reweight - _reweight_baseline() < MIN_LIFT_VS_BASELINE, (
        "Feature mode unexpectedly helped on the reweight design."
    )
    # Weight mode on the radial design.
    weight_on_radial = _mean_auc(
        "adversarial_lgbm_weight_raw",
        lambda: AdversarialValidation(classifier="lgbm", return_as="weight", weight_type="raw"),
        make_test_region_positive, RADIAL_KW)
    assert weight_on_radial - _radial_baseline() < MIN_LIFT_VS_BASELINE, (
        "Weight mode unexpectedly helped on the radial design."
    )


def test_no_shift_control():
    """Without covariate shift, the feature AUC difference must stay below 0.05."""
    no_shift_kw = dict(RADIAL_KW, sigma_test=RADIAL_KW["sigma_train"])
    base = _mean_auc("baseline", InductiveBaseline, make_test_region_positive, no_shift_kw)
    joint = _mean_auc(
        "adversarial_lgbm_feature",
        lambda: AdversarialValidation(classifier="lgbm", return_as="feature"),
        make_test_region_positive, no_shift_kw)
    assert joint - base < MIN_LIFT_VS_BASELINE, (
        f"With no covariate shift, the domain-score AUC difference is "
        f"{joint - base:+.4f}, which is not below {MIN_LIFT_VS_BASELINE:.2f}."
    )

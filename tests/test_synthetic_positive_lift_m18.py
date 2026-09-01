"""Synthetic positive-lift tests for M18 ``LeafBasedTestDensityFE``.

M18 trains an auxiliary LightGBM on the training data and uses leaf assignments
to summarize train and test occupancy. Confidence-mode variants append leaf
density features and are evaluated on a density checkerboard. Embedding-mode
variants append cross-split leaf-distance features and are evaluated on a
core-periphery design.

Each variant must exceed the corresponding inductive baseline by at least 0.05.
A cross-design control verifies that the embedding mode stays below the threshold
on the density checkerboard. Test labels are used only for final AUC calculation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.tier2 import LeafBasedTestDensityFE
from tests._synthetic_data import make_density_checkerboard, make_core_periphery_shift
from tests._synthetic_evaluation import weighted_augment_auc


N_SEEDS = 5
MIN_LIFT_VS_BASELINE = 0.05

CONF_KW = dict(n_train=600, n_test=1500, n_noise=4, c=2.0, sigma=0.9)
EMBED_KW = dict(n_informative=2, n_noise=4, cluster_std=0.8)

CONF_BASELINE_CEILING = 0.60    # The checkerboard linear baseline remains near chance.
EMBED_BASELINE_CEILING = 0.60   # core/periphery asymmetry keeps the baseline near chance

CONF_VARIANTS = [
    ("leaf_density_conf",
     lambda: LeafBasedTestDensityFE(mode="confidence_feature", n_estimators=100, max_depth=6)),
    ("leaf_density_conf_shallow",
     lambda: LeafBasedTestDensityFE(mode="confidence_feature", n_estimators=100, max_depth=3)),
    ("leaf_density_conf_cumshift",
     lambda: LeafBasedTestDensityFE(mode="confidence_feature", n_estimators=100, max_depth=6, feature_subset="cumshift")),
]
EMBED_VARIANTS = [
    ("leaf_density_embed_k10",
     lambda: LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=10, n_estimators=100, max_depth=6)),
    ("leaf_density_embed_k5",
     lambda: LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=5, n_estimators=100, max_depth=6)),
    ("leaf_density_embed_k10_cumshift",
     lambda: LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=10, n_estimators=100, max_depth=6, feature_subset="cumshift")),
]

_CACHE: dict = {}


def _mean_auc(key, factory, gen, gen_kw):
    """Memoized mean test AUC of ``factory`` over N_SEEDS on a given generator."""
    ck = (key, gen.__name__, tuple(sorted(gen_kw.items())))
    if ck not in _CACHE:
        vals = [weighted_augment_auc(factory(), *gen(seed=s, **gen_kw)) for s in range(N_SEEDS)]
        _CACHE[ck] = float(np.mean(vals))
    return _CACHE[ck]


def _conf_baseline():
    return _mean_auc("baseline", InductiveBaseline, make_density_checkerboard, CONF_KW)


def _embed_baseline():
    return _mean_auc("baseline", InductiveBaseline, make_core_periphery_shift, EMBED_KW)


@pytest.mark.parametrize("label,factory", CONF_VARIANTS)
def test_conf_variants_exceed_baseline_on_checkerboard(label, factory):
    """Each confidence variant must exceed its baseline AUC by at least 0.05."""
    base = _conf_baseline()
    joint = _mean_auc(label, factory, make_density_checkerboard, CONF_KW)
    lift = joint - base
    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"{label} AUC difference {lift:+.4f} is below {MIN_LIFT_VS_BASELINE:.2f} "
        f"(baseline {base:.4f}, joint {joint:.4f})."
    )


@pytest.mark.parametrize("label,factory", EMBED_VARIANTS)
def test_embed_variants_exceed_baseline_on_core_periphery(label, factory):
    """Each embedding variant must exceed its baseline AUC by at least 0.05."""
    base = _embed_baseline()
    joint = _mean_auc(label, factory, make_core_periphery_shift, EMBED_KW)
    lift = joint - base
    assert lift >= MIN_LIFT_VS_BASELINE, (
        f"{label} AUC difference {lift:+.4f} is below {MIN_LIFT_VS_BASELINE:.2f} "
        f"(baseline {base:.4f}, joint {joint:.4f})."
    )


def test_baselines_meet_expected_ceilings():
    """Both designs keep the baseline below the expected ceiling. The checkerboard
    baseline is near chance because of XOR. The core/periphery baseline is near
    chance because the train boundary mislabels the orthogonal test periphery."""
    cb = _conf_baseline()
    eb = _embed_baseline()
    assert cb < CONF_BASELINE_CEILING, (
        f"Checkerboard baseline {cb:.4f} not near chance (< {CONF_BASELINE_CEILING})."
    )
    assert eb < EMBED_BASELINE_CEILING, (
        f"Core/periphery baseline {eb:.4f} not near chance (< {EMBED_BASELINE_CEILING})."
    )


def test_modes_are_complementary():
    """Embedding mode must remain below the threshold on the checkerboard design."""
    base = _conf_baseline()
    # Evaluate embedding mode on the density-checkerboard design.
    embed_on_checkerboard = _mean_auc(
        "embed_k10_on_checkerboard",
        lambda: LeafBasedTestDensityFE(mode="leaf_embedding_knn", k=10, n_estimators=100, max_depth=6),
        make_density_checkerboard, CONF_KW)
    assert embed_on_checkerboard - base < MIN_LIFT_VS_BASELINE, (
        f"Embedding-mode AUC difference {embed_on_checkerboard - base:+.4f} is "
        f"not below {MIN_LIFT_VS_BASELINE:.2f} on the density checkerboard."
    )

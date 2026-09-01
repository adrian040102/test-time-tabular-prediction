"""Synthetic positive-lift tests for JointTSNE.

Each evaluated configuration is compared with JointPCA and InductiveBaseline on
concentric circles with four Gaussian noise dimensions. Concentric circles
provide nonlinear structure that logistic regression cannot capture directly.
t-SNE can expose that structure.

Across ``N_SEEDS``, the mean JointTSNE AUC must exceed both comparison
methods by the configured margins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.base import InductiveBaseline
from src.methods.tier1 import JointPCA, JointTSNE


# ---------------------------------------------------------------------------
# Test parameters are defined in the module docstring.
# ---------------------------------------------------------------------------
N_SAMPLES = 1000        # An 80/20 split yields 800 training and 200 test rows.
N_NOISE_DIMS = 4
NOISE_DIM_SCALE = 0.3
CIRCLES_FACTOR = 0.3    # inner circle radius = 0.3 * outer
CIRCLES_NOISE = 0.05    # Gaussian noise added to circle coords
N_SEEDS = 5
MIN_LIFT_VS_PCA = 0.05      # Minimum lift required over the PCA comparison.
MIN_LIFT_VS_BASELINE = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _concentric_circles_with_noise(seed: int):
    """Generate concentric-circles dataset with low-dim Gaussian noise.

    Return ``X_train_df``, ``X_test_df``, ``y_train`` and ``y_test`` from an
    IID 80/20 split.
    All ``2 + N_NOISE_DIMS`` columns are numeric (no categoricals).
    """
    from sklearn.datasets import make_circles

    rng = np.random.RandomState(seed)
    X2d, y = make_circles(
        n_samples=N_SAMPLES,
        factor=CIRCLES_FACTOR,
        noise=CIRCLES_NOISE,
        random_state=seed,
    )
    noise = rng.standard_normal((N_SAMPLES, N_NOISE_DIMS)) * NOISE_DIM_SCALE
    X_full = np.hstack([X2d, noise])

    perm = rng.permutation(N_SAMPLES)
    n_train = int(N_SAMPLES * 0.8)
    tr_idx, te_idx = perm[:n_train], perm[n_train:]
    cols = [f"f{i}" for i in range(X_full.shape[1])]
    X_tr_df = pd.DataFrame(X_full[tr_idx], columns=cols)
    X_te_df = pd.DataFrame(X_full[te_idx], columns=cols)
    return X_tr_df, X_te_df, y[tr_idx].astype(int), y[te_idx].astype(int)


def _logistic_auc(method, X_tr_df, X_te_df, y_tr, y_te) -> float:
    """Run a method and train logistic regression, then return AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    result = method.fit_transform(X_tr_df, X_te_df, y_tr)
    X_tr = result.X_train
    X_te = result.X_test
    # Match the pipeline treatment of methods that preserve joint coordinates.
    if not result.skip_pipeline_scaler:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_te, proba))


# ---------------------------------------------------------------------------
# Positive-lift comparison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tsne_factory,variant_label",
    [
        (lambda: JointTSNE(n_components=2, perplexity=30), "joint_tsne_2d_p30"),
        (lambda: JointTSNE(n_components=2, perplexity=50), "joint_tsne_2d_p50"),
        (lambda: JointTSNE(n_components=3, perplexity=30), "joint_tsne_3d_p30"),
    ],
)
def test_joint_tsne_exceeds_pca_and_baseline_on_concentric_circles(
    tsne_factory, variant_label,
):
    """Require JointTSNE to exceed both comparison methods by the configured margins."""
    aucs_baseline: list[float] = []
    aucs_pca: list[float] = []
    aucs_tsne: list[float] = []

    for seed in range(N_SEEDS):
        X_tr, X_te, y_tr, y_te = _concentric_circles_with_noise(seed)

        aucs_baseline.append(
            _logistic_auc(InductiveBaseline(), X_tr, X_te, y_tr, y_te)
        )
        # Use the same component count for JointPCA and two-dimensional t-SNE.
        aucs_pca.append(
            _logistic_auc(JointPCA(n_components=2), X_tr, X_te, y_tr, y_te)
        )
        aucs_tsne.append(
            _logistic_auc(tsne_factory(), X_tr, X_te, y_tr, y_te)
        )

    mean_baseline = float(np.mean(aucs_baseline))
    mean_pca = float(np.mean(aucs_pca))
    mean_tsne = float(np.mean(aucs_tsne))

    lift_vs_baseline = mean_tsne - mean_baseline
    lift_vs_pca = mean_tsne - mean_pca

    msg = (
        f"\n  Variant: {variant_label}\n"
        f"  N seeds: {N_SEEDS}\n"
        f"  Mean AUC InductiveBaseline: {mean_baseline:.4f}\n"
        f"  Mean AUC JointPCA(2):       {mean_pca:.4f}\n"
        f"  Mean AUC {variant_label}:   {mean_tsne:.4f}\n"
        f"  Lift vs baseline: {lift_vs_baseline:+.4f} "
        f"(threshold: >= +{MIN_LIFT_VS_BASELINE:.2f})\n"
        f"  Lift vs PCA:      {lift_vs_pca:+.4f} "
        f"(threshold: >= +{MIN_LIFT_VS_PCA:.2f})\n"
        f"  Per-seed t-SNE AUCs: {[round(a, 3) for a in aucs_tsne]}\n"
    )

    assert lift_vs_baseline >= MIN_LIFT_VS_BASELINE, (
        f"JointTSNE AUC difference from InductiveBaseline is below "
        f"{MIN_LIFT_VS_BASELINE:.2f}.{msg}"
    )
    assert lift_vs_pca >= MIN_LIFT_VS_PCA, (
        f"JointTSNE AUC difference from JointPCA(2) is below "
        f"{MIN_LIFT_VS_PCA:.2f}.{msg}"
    )

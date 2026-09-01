"""
Synthetic income prediction dataset with controllable covariate shift.

Simulates the ACSIncome scenario: predict whether income > 50k.
The shift models temporal changes (inflation, demographic shifts):
  - Numeric features shift their means
  - Categorical distributions change
  - The decision boundary stays roughly the same

Supports multiple shift modes:
  - "full" (default): All features shift proportionally to shift_strength.
  - "partial": Only a subset of features shift (tests column selection).
  - "label_shift": P(Y) changes while P(X|Y) stays the same.

Args:
    shift_strength: 0.0 = no shift (IID), 1.0 = moderate, 5.0 = extreme
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.data.base import TabularDataset

# Features that shift in "partial" mode (3 of 8: 2 num + 1 cat).
# AGEP shifts via mean, SCHL shifts via distribution, OCCP shifts via distribution.
PARTIAL_SHIFT_FEATURES = {"AGEP", "SCHL", "OCCP"}

# Controlled-series settings. Auxiliary synthetic datasets retain their own
# size settings. ``load_dataset`` applies these values only to the five
# canonical ``synthetic_shift_*`` names below.
CONTROLLED_SHIFT_STRENGTHS: tuple[float, ...] = (0.0, 0.5, 1.0, 3.0, 5.0)
CONTROLLED_SYNTHETIC_N_TRAIN = 10_000
CONTROLLED_SYNTHETIC_N_TEST = 10_000
CONTROLLED_SYNTHETIC_SEED = 42


def controlled_shift_strength(name: str) -> float | None:
    """Return the canonical shift strength for a controlled-series name, else ``None``."""
    prefix = "synthetic_shift_"
    if not name.startswith(prefix):
        return None
    try:
        shift = float(name.removeprefix(prefix))
    except ValueError:
        return None
    return shift if shift in CONTROLLED_SHIFT_STRENGTHS else None


def _shifted_education_dist(shift: float) -> np.ndarray:
    """Simulate education inflation: more people with higher education over time."""
    base = np.array([0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.07,
                     0.08, 0.09, 0.10, 0.10, 0.09, 0.08, 0.07, 0.07])
    adjustment = np.linspace(-0.01, 0.01, 16) * shift
    p = np.clip(base + adjustment, 0.005, None)
    return p / p.sum()


def _shifted_occupation_dist(shift: float) -> np.ndarray:
    """Simulate occupational shift: more tech/mgmt, less labor over time."""
    base = np.array([0.18, 0.15, 0.15, 0.20, 0.17, 0.15])
    adjustment = np.array([0.02, -0.01, 0.04, -0.02, -0.03, 0.00]) * shift
    p = np.clip(base + adjustment, 0.02, None)
    return p / p.sum()


def _shifted_marital_dist(shift: float) -> np.ndarray:
    """Simulate demographic shift: fewer married, more single over time."""
    base = np.array([0.45, 0.30, 0.15, 0.10])
    adjustment = np.array([-0.03, 0.03, 0.01, -0.01]) * shift
    p = np.clip(base + adjustment, 0.02, None)
    return p / p.sum()


def _shifted_state_dist(shift: float) -> np.ndarray:
    """Simulate population migration: more TX/FL, less NY/IL."""
    base = np.array([0.25, 0.20, 0.20, 0.18, 0.17])
    adjustment = np.array([-0.01, 0.03, -0.02, 0.02, -0.02]) * shift
    p = np.clip(base + adjustment, 0.02, None)
    return p / p.sum()


def _generate_features(
    n: int,
    shift: float,
    rng: np.random.RandomState,
    shift_features: set[str] | None = None,
) -> pd.DataFrame:
    """Generate feature matrix with optional distributional shift.

    Args:
        n: Number of samples.
        shift: Shift strength (0.0 = no shift).
        rng: Random state.
        shift_features: If provided, only these features get shifted.
            Others are generated with shift=0.0. None means all shift.
    """
    def _s(feat_name: str) -> float:
        """Effective shift for this feature."""
        if shift_features is None:
            return shift
        return shift if feat_name in shift_features else 0.0

    age = rng.normal(40 + _s("AGEP") * 2, 12, n).clip(16, 90)
    education = rng.choice(
        np.arange(1, 17), size=n, p=_shifted_education_dist(_s("SCHL")),
    ).astype(float)
    hours_per_week = rng.normal(40 + _s("WKHP") * 1.5, 10, n).clip(1, 80)
    occupation = rng.choice(
        ["mgmt", "sales", "tech", "service", "labor", "office"],
        size=n, p=_shifted_occupation_dist(_s("OCCP")),
    )
    marital = rng.choice(
        ["married", "single", "divorced", "widowed"],
        size=n, p=_shifted_marital_dist(_s("MAR")),
    )
    state = rng.choice(
        ["CA", "TX", "NY", "FL", "IL"],
        size=n, p=_shifted_state_dist(_s("ST")),
    )
    capital_gain = rng.exponential(500 + _s("CAPITALGAIN") * 200, n).clip(0, 100_000)
    capital_loss = rng.exponential(50 + _s("CAPITALLOSS") * 20, n).clip(0, 5_000)

    return pd.DataFrame({
        "AGEP": age,
        "SCHL": education,
        "WKHP": hours_per_week,
        "OCCP": occupation,
        "MAR": marital,
        "ST": state,
        "CAPITALGAIN": capital_gain,
        "CAPITALLOSS": capital_loss,
    })


def _generate_labels(df: pd.DataFrame, rng: np.random.RandomState) -> np.ndarray:
    """Generate income labels with a fixed threshold while feature values shift."""
    score = (
        0.03 * df["AGEP"]
        + 0.15 * df["SCHL"]
        + 0.02 * df["WKHP"]
        + 0.0002 * df["CAPITALGAIN"]
        - 0.0003 * df["CAPITALLOSS"]
        + df["OCCP"].map({"mgmt": 1.5, "tech": 1.2, "office": 0.3,
                          "sales": 0.5, "service": -0.5, "labor": -0.3})
        + df["MAR"].map({"married": 0.5, "single": -0.2,
                         "divorced": -0.1, "widowed": -0.3})
    )
    prob = 1 / (1 + np.exp(-(score - 4.5)))
    return (rng.random(len(df)) < prob).astype(int)


def _generate_label_shift_data(
    n_train: int,
    n_test: int,
    target_test_prevalence: float,
    rng: np.random.RandomState,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Generate data with label shift: P(Y) changes but P(X|Y) stays the same.

    Strategy: generate train IID, then collect positive and negative test
    examples separately in small batches until each class has enough examples.
    """
    # Generate training data (natural prevalence, ~0.43)
    X_train = _generate_features(n_train, shift=0.0, rng=rng)
    y_train = _generate_labels(X_train, rng)

    n_pos_needed = int(n_test * target_test_prevalence)
    n_neg_needed = n_test - n_pos_needed

    # Collect positive and negative examples in batches
    pos_frames: list[pd.DataFrame] = []
    neg_frames: list[pd.DataFrame] = []
    n_pos_have, n_neg_have = 0, 0
    batch_size = 20_000

    while n_pos_have < n_pos_needed or n_neg_have < n_neg_needed:
        X_batch = _generate_features(batch_size, shift=0.0, rng=rng)
        y_batch = _generate_labels(X_batch, rng)

        if n_pos_have < n_pos_needed:
            mask_pos = y_batch == 1
            pos_frames.append(X_batch[mask_pos])
            n_pos_have += mask_pos.sum()

        if n_neg_have < n_neg_needed:
            mask_neg = y_batch == 0
            neg_frames.append(X_batch[mask_neg])
            n_neg_have += mask_neg.sum()

    X_pos = pd.concat(pos_frames, ignore_index=True).iloc[:n_pos_needed]
    X_neg = pd.concat(neg_frames, ignore_index=True).iloc[:n_neg_needed]

    X_test = pd.concat([X_pos, X_neg], ignore_index=True)
    y_test = np.array([1] * n_pos_needed + [0] * n_neg_needed)

    # Shuffle test set
    shuffle_idx = rng.permutation(n_test)
    X_test = X_test.iloc[shuffle_idx].reset_index(drop=True)
    y_test = y_test[shuffle_idx]

    return X_train, y_train, X_test, y_test


def load_synthetic(
    n_train: int = 50_000,
    n_test: int = 50_000,
    shift_strength: float = 1.0,
    shift_mode: Literal["full", "partial", "label_shift"] = "full",
    label_shift_prevalence: float = 0.65,
    seed: int = 42,
    max_samples: int | None = None,
    name_override: str | None = None,
) -> TabularDataset:
    """
    Generate synthetic income prediction data with controllable shift.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        shift_strength: 0.0 = IID, 1.0 = moderate, 5.0 = extreme.
            Ignored when shift_mode="label_shift".
        shift_mode: Type of shift to apply.
            "full": All features shift (default, original behavior).
            "partial": Only AGEP, SCHL, OCCP shift (3/8 features).
            "label_shift": P(Y) changes, P(X|Y) preserved.
        label_shift_prevalence: Target P(Y=1) in test set for label_shift
            mode. Train prevalence is ~0.43 (natural). Default 0.65.
        seed: Random seed for reproducibility.
        max_samples: Cap on train/test size.

    Returns:
        TabularDataset with all metadata populated.
    """
    if max_samples is not None:
        n_train = min(n_train, max_samples)
        n_test = min(n_test, max_samples)

    rng = np.random.RandomState(seed)

    cat_features = ["OCCP", "MAR", "ST"]
    num_features = ["AGEP", "SCHL", "WKHP", "CAPITALGAIN", "CAPITALLOSS"]

    if shift_mode == "label_shift":
        X_train, y_train, X_test, y_test = _generate_label_shift_data(
            n_train, n_test, label_shift_prevalence, rng,
        )
        name = "synthetic_label_shift"
        shift_desc = (
            f"synthetic label shift: P(Y=1) train~0.43 -> test~{label_shift_prevalence}"
        )
        meta = {
            "shift_mode": "label_shift",
            "label_shift_prevalence": label_shift_prevalence,
            "seed": seed,
        }
    else:
        shift_features = PARTIAL_SHIFT_FEATURES if shift_mode == "partial" else None
        X_train = _generate_features(n_train, shift=0.0, rng=rng)
        y_train = _generate_labels(X_train, rng)
        X_test = _generate_features(
            n_test, shift=shift_strength, rng=rng,
            shift_features=shift_features,
        )
        y_test = _generate_labels(X_test, rng)

        if shift_mode == "partial":
            name = f"synthetic_partial_shift"
            shift_desc = (
                f"synthetic partial covariate shift (strength={shift_strength}, "
                f"shifted features: {sorted(PARTIAL_SHIFT_FEATURES)})"
            )
        else:
            name = f"synthetic_shift_{shift_strength}"
            shift_desc = f"synthetic covariate shift (strength={shift_strength})"

        meta = {
            "shift_strength": shift_strength,
            "shift_mode": shift_mode,
            "seed": seed,
        }
        if shift_features is not None:
            meta["shifted_features"] = sorted(shift_features)

    return TabularDataset(
        name=name_override or name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cat_features=cat_features,
        num_features=num_features,
        shift_description=shift_desc,
        metadata=meta,
    )

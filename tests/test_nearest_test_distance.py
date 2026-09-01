"""Tests for M16 (NearestTestDistance) / M16v2 (NearestTestDistanceRelative).

These methods compute kNN distances between the train and test distributions
and append the resulting distances as features or sample weights. The distance
space frequency-encodes categorical columns using joint train-and-test
frequencies, consistent with M19, M20 and JointTSNE. Arbitrary label codes do not
provide meaningful Euclidean distances. Frequency encoding represents category
rarity, which is relevant to distribution shift.

The output matrix used by the downstream model remains label-encoded. Only the
matrix that feeds the distance pipeline is frequency-encoded.

On all-numeric data, ``_frequency_encode_joint`` and ``_label_encode_cats``
produce the same float array, so this preprocessing only changes datasets with
categorical columns.
"""

import numpy as np
import pandas as pd
import pytest

from src.methods.tier1 import _label_encode_cats
from src.methods.preprocessing import get_cat_cols
from src.methods.tier2 import (
    NearestTestDistance,
    NearestTestDistanceRelative,
    TIER2_METHODS,
    _frequency_encode_joint,
)

BOTH_CLASSES = [NearestTestDistance, NearestTestDistanceRelative]

# Evaluated configurations covered here. All use feature_weighting="none".
EVALUATED_VARIANTS = [
    "nearest_test_k5_both",
    "nearest_test_k10_both",
    "nearest_test_rel_k10_both",
]


# ---------------------------------------------------------------------------
# Fixtures / data builders
# ---------------------------------------------------------------------------

def _mixed_frame(seed: int = 0):
    """A mixed cat+num frame with rare/common categories and a real shift."""
    rng = np.random.RandomState(seed)
    n_train, n_test = 80, 50
    cat1_tr = rng.choice(["x", "y", "z"], size=n_train, p=[0.7, 0.2, 0.1])
    cat1_te = rng.choice(["x", "y", "z"], size=n_test, p=[0.3, 0.3, 0.4])  # shift
    cat2_tr = rng.choice(["p", "q"], size=n_train, p=[0.9, 0.1])
    cat2_te = rng.choice(["p", "q"], size=n_test, p=[0.5, 0.5])
    Xtr = pd.DataFrame({
        "c1": cat1_tr,
        "c2": cat2_tr,
        "n1": rng.randn(n_train),
        "n2": rng.randn(n_train) + 1.0,
        "n3": rng.randn(n_train),
    })
    Xte = pd.DataFrame({
        "c1": cat1_te,
        "c2": cat2_te,
        "n1": rng.randn(n_test) + 0.5,
        "n2": rng.randn(n_test),
        "n3": rng.randn(n_test),
    })
    y = (rng.rand(n_train) > 0.5).astype(int)
    return Xtr, Xte, y


def _all_numeric_frame(seed: int = 1):
    rng = np.random.RandomState(seed)
    Xtr = pd.DataFrame(rng.randn(60, 6), columns=[f"n{i}" for i in range(6)])
    Xte = pd.DataFrame(rng.randn(40, 6), columns=[f"n{i}" for i in range(6)])
    y = (rng.rand(60) > 0.5).astype(int)
    return Xtr, Xte, y


def _rare_far_frame():
    """Train = 100% category 'A'. Test = 50% 'A' + 50% 'B'. The single numeric
    column is constant, so the categorical drives the distance entirely.
    'B' is absent from train, so 'B' test rows must be farther from train."""
    n_train, n_test = 60, 40
    cat_tr = ["A"] * n_train
    cat_te = ["A"] * (n_test // 2) + ["B"] * (n_test // 2)
    Xtr = pd.DataFrame({"grp": cat_tr, "num": [1.0] * n_train})
    Xte = pd.DataFrame({"grp": cat_te, "num": [1.0] * n_test})
    y = np.array([0, 1] * (n_train // 2))
    return Xtr, Xte, y


# ---------------------------------------------------------------------------
# Helper-level: the frequency encoding itself
# ---------------------------------------------------------------------------

def test_all_numeric_freq_encode_bit_identical_to_label_encode():
    """Without categorical columns, frequency and label encoding are identical."""
    Xtr, Xte, _ = _all_numeric_frame()
    assert get_cat_cols(Xtr) == []

    f_tr, f_te = _frequency_encode_joint(Xtr, Xte, get_cat_cols(Xtr))
    le_tr, le_te, _ = _label_encode_cats(Xtr, Xte)

    assert np.array_equal(f_tr, le_tr.values.astype(float))
    assert np.array_equal(f_te, le_te.values.astype(float))


def test_frequency_encoding_reflects_rarity_not_label_codes():
    """The Germany/France/Spain argument: label-encoding produces arbitrary
    ordinal codes (alphabetical: F=0, G=1, S=2 → F nearer G than S), but the
    distance space must reflect *rarity*. Build a 3-category column whose joint
    frequencies induce a different distance ordering than label codes would."""
    cats = (["G"] * 60) + (["S"] * 30) + (["F"] * 10)  # n = 100
    X = pd.DataFrame({"country": cats})
    X2 = pd.DataFrame({"country": cats})  # same dist → joint freq well-defined
    assert get_cat_cols(X) == ["country"]

    f_tr, _ = _frequency_encode_joint(X, X2, ["country"])
    # one representative row per category
    val = {c: float(f_tr[cats.index(c), 0]) for c in ("G", "S", "F")}

    # joint counts G=120, S=60, F=20 over 200 → 0.6 / 0.3 / 0.1
    assert val["F"] == pytest.approx(0.1)
    assert val["S"] == pytest.approx(0.3)
    assert val["G"] == pytest.approx(0.6)
    assert val["F"] < val["S"] < val["G"]              # ordered by rarity

    # In frequency space F is closer to S than to G. Label codes (F=0, G=1, S=2)
    # give the opposite (|F-S|=2 > |F-G|=1), proving the distance space
    # uses frequencies rather than arbitrary label codes.
    assert abs(val["F"] - val["S"]) < abs(val["F"] - val["G"])


def test_frequency_encode_preserves_numeric_columns():
    """Numerics must pass through unchanged. Only cats are replaced."""
    rng = np.random.RandomState(3)
    num = rng.randn(20)
    cats = ["a"] * 10 + ["b"] * 10
    X = pd.DataFrame({"cat": cats, "num": num})
    f_tr, _ = _frequency_encode_joint(X, X, ["cat"])
    # column order preserved: col 0 = cat (freq), col 1 = num (untouched)
    assert np.allclose(f_tr[:, 1], num)


# ---------------------------------------------------------------------------
# Method-level: all-numeric (no-op) behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", BOTH_CLASSES)
def test_all_numeric_runs_finite_and_deterministic(cls):
    Xtr, Xte, y = _all_numeric_frame()
    n_feat = Xtr.shape[1]

    r1 = cls(k=5, return_as="both", feature_weighting="none").fit_transform(Xtr, Xte, y)
    r2 = cls(k=5, return_as="both", feature_weighting="none").fit_transform(Xtr, Xte, y)

    assert r1.X_train.shape == (len(Xtr), n_feat + 4)
    assert r1.X_test.shape == (len(Xte), n_feat + 4)
    assert np.isfinite(r1.X_train).all() and np.isfinite(r1.X_test).all()
    # determinism
    assert np.array_equal(r1.X_train, r2.X_train)
    assert np.array_equal(r1.X_test, r2.X_test)
    # metadata records the encoding + zero cat cols
    assert r1.metadata["m16_distance_encoding"] == "frequency"
    assert r1.metadata["n_cat_cols_encoded"] == 0


# ---------------------------------------------------------------------------
# Method-level: cat-bearing behaviour
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", BOTH_CLASSES)
def test_cat_bearing_runs_clean(cls):
    Xtr, Xte, y = _mixed_frame()
    n_feat = Xtr.shape[1]
    n_cat = len(get_cat_cols(Xtr))
    assert n_cat == 2

    r = cls(k=5, return_as="both", feature_weighting="none").fit_transform(Xtr, Xte, y)

    assert r.X_train.shape == (len(Xtr), n_feat + 4)
    assert r.X_test.shape == (len(Xte), n_feat + 4)
    assert np.isfinite(r.X_train).all() and np.isfinite(r.X_test).all()
    assert r.metadata["m16_distance_encoding"] == "frequency"
    assert r.metadata["n_cat_cols_encoded"] == n_cat


@pytest.mark.parametrize("cls", BOTH_CLASSES)
def test_cat_bearing_deterministic(cls):
    Xtr, Xte, y = _mixed_frame()
    r1 = cls(k=10, return_as="both", feature_weighting="none").fit_transform(Xtr, Xte, y)
    r2 = cls(k=10, return_as="both", feature_weighting="none").fit_transform(Xtr, Xte, y)
    assert np.array_equal(r1.X_train, r2.X_train)
    assert np.array_equal(r1.X_test, r2.X_test)


@pytest.mark.parametrize("cls", BOTH_CLASSES)
def test_rare_category_sits_far_from_common(cls):
    """A test category absent from train (rare) must produce a larger
    nearest-train distance than a category present in train (common)."""
    Xtr, Xte, y = _rare_far_frame()
    r = cls(k=5, return_as="feature", feature_weighting="none").fit_transform(Xtr, Xte, y)

    idx = r.feature_names.index("nearest_train_dist")
    nd = r.X_test[:, idx]
    a_dist = nd[:20].mean()   # first 20 test rows are common 'A'
    b_dist = nd[20:].mean()   # last 20 test rows are rare 'B' (absent in train)
    assert b_dist > a_dist + 1e-6


@pytest.mark.parametrize("cls", BOTH_CLASSES)
@pytest.mark.parametrize("fw", ["none", "ks", "filter", "cumshift"])
def test_all_feature_weightings_run_on_cats(cls, fw):
    """ks / filter / cumshift share fit_transform with the plain path. Verify
    each runs clean on cat-bearing data and reports frequency encoding."""
    Xtr, Xte, y = _mixed_frame()
    r = cls(k=5, return_as="both", feature_weighting=fw).fit_transform(Xtr, Xte, y)
    assert np.isfinite(r.X_train).all() and np.isfinite(r.X_test).all()
    assert r.metadata["m16_distance_encoding"] == "frequency"
    if fw == "cumshift":
        # selected_features metadata must stay aligned with the original
        # column names (freq-encode preserves column order).
        selected = set(r.metadata.get("selected_features", []))
        assert selected <= set(Xtr.columns)
        assert len(selected) >= 1


@pytest.mark.parametrize("cls", BOTH_CLASSES)
def test_handles_nan_in_numeric(cls):
    """NaN in numeric columns must be median-imputed in the distance space. The
    appended distance features stay finite (original cols keep their NaN for the
    downstream imputer)."""
    Xtr, Xte, y = _mixed_frame()
    Xtr = Xtr.copy()
    Xte = Xte.copy()
    Xtr.loc[Xtr.index[:5], "n1"] = np.nan
    Xte.loc[Xte.index[:3], "n2"] = np.nan

    r = cls(k=5, return_as="both", feature_weighting="none").fit_transform(Xtr, Xte, y)
    # last 4 cols are the appended distance features
    assert np.isfinite(r.X_train[:, -4:]).all()
    assert np.isfinite(r.X_test[:, -4:]).all()


@pytest.mark.parametrize("cls", BOTH_CLASSES)
def test_row_counts_preserved(cls):
    Xtr, Xte, y = _mixed_frame()
    r = cls(k=5, return_as="both", feature_weighting="none").fit_transform(Xtr, Xte, y)
    assert r.X_train.shape[0] == len(Xtr)
    assert r.X_test.shape[0] == len(Xte)
    if r.sample_weights_train is not None:
        assert len(r.sample_weights_train) == len(Xtr)


# ---------------------------------------------------------------------------
# The three evaluated configurations run end to end.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", EVALUATED_VARIANTS)
def test_evaluated_variant_end_to_end(key):
    assert key in TIER2_METHODS
    Xtr, Xte, y = _mixed_frame()
    m = TIER2_METHODS[key]()
    r = m.fit_transform(Xtr, Xte, y)
    assert np.isfinite(r.X_train).all() and np.isfinite(r.X_test).all()
    assert r.metadata["m16_distance_encoding"] == "frequency"
    assert r.metadata["n_cat_cols_encoded"] == len(get_cat_cols(Xtr))
    assert r.X_train.shape[0] == len(Xtr)
    assert r.X_test.shape[0] == len(Xte)

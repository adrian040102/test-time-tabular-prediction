"""Synthetic-data generators used by the test suite.

Each function returns training features, test features, training labels and test
labels under a controlled data-generating process. The function docstrings define
the relevant distribution shift and control comparison. Test labels are used only
for evaluation and are never passed to a method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_three_mode_xor(
    n_train: int = 800,
    n_test: int = 400,
    n_informative: int = 4,
    n_noise: int = 4,
    train_weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    test_weights: tuple[float, float, float] = (0.1, 0.8, 0.1),
    mode_locations: tuple[float, float, float] = (-2.0, 0.0, 2.0),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate a three-mode mixture for the M9 tests.

    Outer modes have label 1 and the middle mode has label 0. The test split assigns
    more probability to the middle mode.
    """
    rng = np.random.RandomState(seed)
    train_weights_arr = np.asarray(train_weights, dtype=float)
    test_weights_arr = np.asarray(test_weights, dtype=float)
    mode_locations_arr = np.asarray(mode_locations, dtype=float)

    if not np.isclose(train_weights_arr.sum(), 1.0):
        raise ValueError(f"train_weights must sum to 1, got {train_weights_arr.sum()}")
    if not np.isclose(test_weights_arr.sum(), 1.0):
        raise ValueError(f"test_weights must sum to 1, got {test_weights_arr.sum()}")
    if len(mode_locations_arr) != 3:
        raise ValueError(f"mode_locations must have length 3, got {len(mode_locations_arr)}")

    def _sample(n: int, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mode_id = rng.choice(3, size=n, p=w)
        # XOR-by-mode: outer modes (0 and 2) are y=1, middle mode (1) is y=0.
        y = (mode_id != 1).astype(int)
        X_inf = rng.standard_normal((n, n_informative)) + mode_locations_arr[mode_id, None]
        X_noi = rng.standard_normal((n, n_noise))
        X = np.hstack([X_inf, X_noi])
        return X, y

    X_tr, y_tr = _sample(n_train, train_weights_arr)
    X_te, y_te = _sample(n_test, test_weights_arr)
    cols = [f"inf{i}" for i in range(n_informative)] + [f"noi{i}" for i in range(n_noise)]
    return (
        pd.DataFrame(X_tr, columns=cols),
        pd.DataFrame(X_te, columns=cols),
        y_tr,
        y_te,
    )


def make_core_periphery_shift(
    n_train: int = 800,
    n_test: int = 800,
    n_informative: int = 2,
    n_noise: int = 4,
    core_frac: float = 0.5,
    periphery_distance: float = 4.0,
    cluster_std: float = 1.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate a shared class-one core and split-specific class-zero peripheries.

    The two peripheries lie in orthogonal feature directions. This geometry is used
    by the M16 tests.
    """
    if n_informative < 2:
        raise ValueError(f"n_informative must be >= 2, got {n_informative}")

    rng = np.random.RandomState(seed)
    D, d = periphery_distance, n_informative
    mu_core = np.zeros(d)
    mu_train_periph = np.zeros(d)
    mu_train_periph[0] = D                      # train periphery: direction e0
    mu_test_periph = np.zeros(d)
    mu_test_periph[1] = D                       # test periphery: direction e1

    def _split(n: int, mu_periph: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_core = int(round(core_frac * n))
        n_per = n - n_core
        X_core = rng.normal(0, cluster_std, (n_core, d)) + mu_core
        X_per = rng.normal(0, cluster_std, (n_per, d)) + mu_periph
        X_inf = np.vstack([X_core, X_per])
        y = np.concatenate([np.ones(n_core, dtype=int), np.zeros(n_per, dtype=int)])
        X_noi = rng.normal(0, 1.0, (n, n_noise))
        X = np.hstack([X_inf, X_noi])
        order = rng.permutation(n)
        return X[order], y[order]

    X_tr, y_tr = _split(n_train, mu_train_periph)
    X_te, y_te = _split(n_test, mu_test_periph)
    cols = [f"inf{i}" for i in range(d)] + [f"noi{i}" for i in range(n_noise)]
    return (
        pd.DataFrame(X_tr, columns=cols),
        pd.DataFrame(X_te, columns=cols),
        y_tr,
        y_te,
    )


def make_noisy_consensus(
    n_train: int = 900,
    n_test: int = 900,
    n_informative: int = 2,
    n_noise: int = 4,
    label_noise: float = 0.25,
    shift: float = 0.5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate a circular classification problem for the M19 tests.

    Training labels include configurable noise. The test features have a
    configurable location shift.
    """
    if n_informative < 1:
        raise ValueError(f"n_informative must be >= 1, got {n_informative}")
    if not (0.0 <= label_noise <= 1.0):
        raise ValueError(f"label_noise must be in [0, 1], got {label_noise}")

    rng = np.random.RandomState(seed)

    def _split(n: int, center_shift: float) -> tuple[np.ndarray, np.ndarray]:
        X_inf = rng.normal(center_shift, 1.0, (n, n_informative))
        r2 = (X_inf ** 2).sum(axis=1)
        y = (r2 > np.median(r2)).astype(int)            # circle: not linearly separable
        X_noi = rng.normal(0.0, 1.0, (n, n_noise))
        return np.hstack([X_inf, X_noi]), y

    X_tr, y_tr_true = _split(n_train, 0.0)
    X_te, y_te = _split(n_test, shift)                  # mild covariate shift on inf dims
    flip = rng.rand(n_train) < label_noise              # inject label noise in training only
    y_tr = y_tr_true.copy()
    y_tr[flip] = 1 - y_tr[flip]
    cols = [f"inf{i}" for i in range(n_informative)] + [f"noi{i}" for i in range(n_noise)]
    return (
        pd.DataFrame(X_tr, columns=cols),
        pd.DataFrame(X_te, columns=cols),
        y_tr,
        y_te,
    )


def make_covariate_shift_weighted(
    n_train: int = 900,
    n_test: int = 900,
    n_noise: int = 4,
    train_s1_frac: float = 0.15,
    test_s1_frac: float = 0.85,
    theta1_deg: float = 90.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate two subpopulations with different linear decision directions.

    The subpopulation prevalence changes between training and test data. The M13
    tests use this change to evaluate sample reweighting.
    """
    if not (0.0 <= train_s1_frac <= 1.0 and 0.0 <= test_s1_frac <= 1.0):
        raise ValueError("train_s1_frac and test_s1_frac must be in [0, 1]")

    rng = np.random.RandomState(seed)
    theta1 = np.deg2rad(theta1_deg)
    dir0 = np.array([1.0, 0.0])
    dir1 = np.array([np.cos(theta1), np.sin(theta1)])

    def _split(n: int, s1_frac: float) -> tuple[np.ndarray, np.ndarray]:
        s = (rng.rand(n) < s1_frac).astype(int)
        x = rng.normal(0.0, 1.0, (n, 2))                 # decision plane (no marginal shift)
        proj = np.where(s == 0, x @ dir0, x @ dir1)
        y = (proj > 0).astype(int)
        noi = rng.normal(0.0, 1.0, (n, n_noise))
        X = np.column_stack([s.astype(float), x, noi])
        return X, y

    X_tr, y_tr = _split(n_train, train_s1_frac)
    X_te, y_te = _split(n_test, test_s1_frac)
    cols = ["subpop", "dir0", "dir1"] + [f"noi{i}" for i in range(n_noise)]
    return (
        pd.DataFrame(X_tr, columns=cols),
        pd.DataFrame(X_te, columns=cols),
        y_tr,
        y_te,
    )


def make_pseudo_labelable(
    n_train: int = 900,
    n_test: int = 900,
    n_noise: int = 4,
    train_s1_frac: float = 0.15,
    test_s1_frac: float = 0.85,
    theta1_deg: float = 90.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate the covariate-shift scenario used by the M7 tests.

    This calls make_covariate_shift_weighted with defaults that provide enough
    training rows for the internal LightGBM model.
    """
    return make_covariate_shift_weighted(
        n_train=n_train,
        n_test=n_test,
        n_noise=n_noise,
        train_s1_frac=train_s1_frac,
        test_s1_frac=test_s1_frac,
        theta1_deg=theta1_deg,
        seed=seed,
    )


def make_directional_augment(
    n_train: int = 150,
    n_test: int = 800,
    n_noise: int = 2,
    x0_sep: float = 0.5,
    spur_strength: float = 2.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate a weak stable feature and a strong training-only spurious feature.

    The M17 tests use this distribution to evaluate test-directed augmentation when
    the spurious association does not persist in the test data.
    """
    rng = np.random.RandomState(seed)

    def _draw(n: int, train: bool) -> tuple[np.ndarray, np.ndarray]:
        y = rng.randint(0, 2, n)
        sgn = np.where(y == 1, 1.0, -1.0)
        x0 = rng.normal(0.0, 1.0, n) + sgn * x0_sep            # weak informative
        if train:
            x1 = sgn * spur_strength + rng.normal(0.0, 0.3, n)  # spurious: tracks y in train
        else:
            x1 = rng.normal(0.0, spur_strength + 0.3, n)        # independent in test
        noi = rng.normal(0.0, 1.0, (n, n_noise))
        X = np.column_stack([x0, x1, noi])
        return X, y

    X_tr, y_tr = _draw(n_train, True)
    X_te, y_te = _draw(n_test, False)
    cols = ["x0_inf", "x1_spur"] + [f"noi{i}" for i in range(n_noise)]
    return (
        pd.DataFrame(X_tr, columns=cols),
        pd.DataFrame(X_te, columns=cols),
        y_tr,
        y_te,
    )


def make_groupby_mean_shift(
    n_train: int = 6000,
    n_test: int = 6000,
    n_cats: int = 100,
    base_center: float = 50.0,
    sigma: float = 60.0,
    ratio_up: float = 1.6,
    ratio_down: float = 0.625,
    label_noise: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate group-specific test mean shifts for the group-by tests.

    Training feature means are common across groups. Test means shift up or down by
    group, and the shift direction determines the label.
    """
    if n_cats % 2 != 0:
        raise ValueError(f"n_cats must be even (equal up/down split), got {n_cats}")

    rng = np.random.default_rng(seed)
    half = n_cats // 2
    codes = np.arange(n_cats)
    rng.shuffle(codes)  # decouple the shift label from the ordinal code value
    up = set(codes[:half].tolist())
    shift_label = np.array([1 if c in up else 0 for c in range(n_cats)], dtype=int)
    # The training center is independent of the group, so training x contains no
    # label signal. Only the per-group multiplicative test shift encodes the label.
    # The method estimates it from the per-group mean because a single x value is
    # too noisy when sigma is large.
    ratio = np.where(shift_label == 1, ratio_up, ratio_down)

    def _split(n: int, is_test: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        group = np.repeat(np.arange(n_cats), n // n_cats)
        if len(group) < n:  # top up the remainder if n not divisible by n_cats
            group = np.concatenate([group, rng.integers(0, n_cats, n - len(group))])
        rng.shuffle(group)
        center = base_center * (ratio[group] if is_test else 1.0)
        x = rng.normal(center, sigma, n)
        y = shift_label[group].copy()
        return group, x, y

    g_tr, x_tr, y_tr = _split(n_train, is_test=False)
    g_te, x_te, y_te = _split(n_test, is_test=True)
    flip = rng.random(n_train) < label_noise
    y_tr[flip] = 1 - y_tr[flip]

    X_tr = pd.DataFrame({"group": g_tr.astype(int), "x": x_tr})
    X_te = pd.DataFrame({"group": g_te.astype(int), "x": x_te})
    return X_tr, X_te, y_tr.astype(int), y_te.astype(int)


def make_high_dim_groupby_shift(
    n_train: int = 250,
    n_test: int = 250,
    n_signal_pairs: int = 1,
    n_noise_pairs: int = 50,
    n_levels_signal: int = 8,
    n_levels_noise: int = 60,
    base_center: float = 50.0,
    sigma: float = 60.0,
    ratio_up: float = 1.6,
    ratio_down: float = 0.625,
    label_noise: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate signal and noise group-feature pairs for selective group-by tests.

    Only the signal pairs have group-specific test mean shifts related to the
    label. The noise pairs have stable distributions.
    """
    if n_levels_signal % 2 != 0:
        raise ValueError(f"n_levels_signal must be even, got {n_levels_signal}")
    rng = np.random.default_rng(seed)

    y_tr = rng.integers(0, 2, n_train)
    y_te = rng.integers(0, 2, n_test)

    cols_tr: dict[str, np.ndarray] = {}
    cols_te: dict[str, np.ndarray] = {}

    half = n_levels_signal // 2
    for i in range(n_signal_pairs):
        codes = np.arange(n_levels_signal)
        rng.shuffle(codes)  # decouple the shift label from the ordinal code value
        up_codes = codes[:half]
        down_codes = codes[half:]
        shift_label = np.zeros(n_levels_signal, dtype=int)
        shift_label[up_codes] = 1

        def _assign(y: np.ndarray) -> np.ndarray:
            g = np.empty(len(y), dtype=int)
            up_mask = y == 1
            g[up_mask] = rng.choice(up_codes, size=int(up_mask.sum()))
            g[~up_mask] = rng.choice(down_codes, size=int((~up_mask).sum()))
            return g

        g_tr = _assign(y_tr)
        g_te = _assign(y_te)
        ratio = np.where(shift_label == 1, ratio_up, ratio_down)
        x_tr = rng.normal(base_center, sigma, n_train)
        x_te = rng.normal(base_center * ratio[g_te], sigma, n_test)

        cols_tr[f"g_sig{i}"] = g_tr
        cols_tr[f"x_sig{i}"] = x_tr
        cols_te[f"g_sig{i}"] = g_te
        cols_te[f"x_sig{i}"] = x_te

    for j in range(n_noise_pairs):
        cols_tr[f"g_noi{j}"] = rng.integers(0, n_levels_noise, n_train)
        cols_te[f"g_noi{j}"] = rng.integers(0, n_levels_noise, n_test)
        cols_tr[f"x_noi{j}"] = rng.normal(base_center, sigma, n_train)
        cols_te[f"x_noi{j}"] = rng.normal(base_center, sigma, n_test)  # NO shift

    flip = rng.random(n_train) < label_noise
    y_tr = y_tr.copy()
    y_tr[flip] = 1 - y_tr[flip]

    return (
        pd.DataFrame(cols_tr),
        pd.DataFrame(cols_te),
        y_tr.astype(int),
        y_te.astype(int),
    )


def make_prevalence_shift_cat(
    n_train: int = 800,
    n_test: int = 1600,
    n_categories: int = 50,
    n_noise: int = 3,
    alpha: float = 1.0,
    label_noise: float = 0.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate a categorical prevalence shift for the M1 tests.

    Training levels are sampled uniformly. Test prevalence varies by level and
    determines the label.
    """
    rng = np.random.RandomState(seed)

    # Population prevalence per category and the prevalence-keyed label.
    p_pop = rng.dirichlet(np.ones(n_categories) * alpha)
    order = np.argsort(p_pop)
    label_of_cat = np.zeros(n_categories, dtype=int)
    label_of_cat[order[n_categories // 2:]] = 1  # upper-prevalence half → y=1

    # Train: uniform category sampling makes train frequency uninformative.
    # Test: sampling by population prevalence makes test and joint frequency informative.
    train_cat = rng.randint(0, n_categories, size=n_train)
    test_cat = rng.choice(n_categories, size=n_test, p=p_pop)

    y_tr = label_of_cat[train_cat].copy()
    y_te = label_of_cat[test_cat].copy()
    if label_noise > 0:
        flip = rng.rand(n_train) < label_noise
        y_tr[flip] = 1 - y_tr[flip]

    cols_tr: dict[str, np.ndarray] = {
        "cat": np.array([f"cat_{c}" for c in train_cat], dtype=object)
    }
    cols_te: dict[str, np.ndarray] = {
        "cat": np.array([f"cat_{c}" for c in test_cat], dtype=object)
    }
    for j in range(n_noise):
        cols_tr[f"n{j}"] = rng.standard_normal(n_train)
        cols_te[f"n{j}"] = rng.standard_normal(n_test)

    return (
        pd.DataFrame(cols_tr),
        pd.DataFrame(cols_te),
        y_tr.astype(int),
        y_te.astype(int),
    )


def make_directional_prevalence_shift(
    n_train: int = 1500,
    n_test: int = 1500,
    n_categories: int = 150,
    n_noise_cols: int = 3,
    n_noise_levels: int = 15,
    ratio_up: float = 3.0,
    ratio_down: float = 1.0 / 3.0,
    alpha: float = 1.0,
    label_noise: float = 0.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate categorical levels whose prevalence increases or decreases in test data.

    The direction of the prevalence change determines the label. This distribution
    is used by the M2 tests.
    """
    rng = np.random.RandomState(seed)

    direction = np.zeros(n_categories, dtype=int)
    direction[rng.permutation(n_categories)[: n_categories // 2]] = 1  # half UP

    q = rng.dirichlet(np.ones(n_categories) * alpha)  # train prevalence (level)
    factor = np.where(direction == 1, ratio_up, ratio_down)
    r = q * factor
    r = r / r.sum()                                    # test prevalence

    train_cat = rng.choice(n_categories, size=n_train, p=q)
    test_cat = rng.choice(n_categories, size=n_test, p=r)
    y_tr = direction[train_cat].copy()
    y_te = direction[test_cat].copy()
    if label_noise > 0:
        flip = rng.rand(n_train) < label_noise
        y_tr[flip] = 1 - y_tr[flip]

    cols_tr: dict[str, np.ndarray] = {
        "sig": np.array([f"s{c}" for c in train_cat], dtype=object)
    }
    cols_te: dict[str, np.ndarray] = {
        "sig": np.array([f"s{c}" for c in test_cat], dtype=object)
    }
    # Noise categorical columns: identical distribution on both splits (no shift)
    for j in range(n_noise_cols):
        pj = rng.dirichlet(np.ones(n_noise_levels))
        tr = rng.choice(n_noise_levels, size=n_train, p=pj)
        te = rng.choice(n_noise_levels, size=n_test, p=pj)
        cols_tr[f"noi{j}"] = np.array([f"n{c}" for c in tr], dtype=object)
        cols_te[f"noi{j}"] = np.array([f"n{c}" for c in te], dtype=object)

    return (
        pd.DataFrame(cols_tr),
        pd.DataFrame(cols_te),
        y_tr.astype(int),
        y_te.astype(int),
    )


def make_pca_loading_shift_signal(
    n_train: int = 120,
    n_test: int = 600,
    n_signal: int = 8,
    n_noise: int = 150,
    a_train: float = 0.5,
    a_test: float = 3.0,
    noise_sd: float = 1.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate a diagonal latent signal with different training and test loadings.

    The signal loading is weaker in training data and stronger in test data. The M3
    tests compare joint PCA with the corresponding train-only version.
    """
    rng = np.random.RandomState(seed)
    z_tr = rng.randn(n_train)
    z_te = rng.randn(n_test)
    y_tr = (z_tr > 0).astype(int)
    y_te = (z_te > 0).astype(int)

    # Equal-magnitude random-sign loading: signal spread evenly across the
    # signal features so NO single feature is strongly discriminative (a pure
    # diagonal that axis-aligned tree splits cannot capture).
    u = rng.choice([-1.0, 1.0], size=n_signal) / np.sqrt(n_signal)

    Xs_tr = a_train * np.outer(z_tr, u) + noise_sd * rng.randn(n_train, n_signal)
    Xs_te = a_test * np.outer(z_te, u) + noise_sd * rng.randn(n_test, n_signal)
    Xn_tr = noise_sd * rng.randn(n_train, n_noise)
    Xn_te = noise_sd * rng.randn(n_test, n_noise)

    cols = [f"sig{i}" for i in range(n_signal)] + [f"noise{i}" for i in range(n_noise)]
    X_tr = pd.DataFrame(np.hstack([Xs_tr, Xn_tr]), columns=cols)
    X_te = pd.DataFrame(np.hstack([Xs_te, Xn_te]), columns=cols)
    return X_tr, X_te, y_tr.astype(int), y_te.astype(int)


def make_reversed_drift_target(
    n_train: int = 3000,
    n_test: int = 3000,
    n_stable: int = 30,
    n_drift: int = 3,
    stable_extreme: float = 0.95,
    drift_extreme: float = 0.98,
    train_drift_frac: float = 0.03,
    test_drift_frac: float = 0.35,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate stable categories and shifted categories with reversed test target means.

    The frequency shift identifies the categories with concept drift. This
    distribution is used by the M4 tests.
    """
    rng = np.random.RandomState(seed)
    n_total = n_stable + n_drift

    stable_tr = np.linspace(1 - stable_extreme, stable_extreme, n_stable)
    stable_te = stable_tr.copy()                          # no concept drift
    drift_tr = np.where(np.arange(n_drift) % 2 == 0, drift_extreme, 1 - drift_extreme)
    drift_te = 1.0 - drift_tr                              # full reversal on test
    train_mean = np.concatenate([stable_tr, drift_tr])
    test_mean = np.concatenate([stable_te, drift_te])

    # Shuffle the mean -> category-id map so the ordinal label code is
    # uninformative. Otherwise, the inductive baseline uses the signal from the
    # code. The names[c] value is the string identifier for logical category c.
    names = rng.permutation(n_total)

    def _split(n, drift_frac):
        n_dr = int(n * drift_frac)
        n_st = n - n_dr
        st = rng.randint(0, n_stable, size=n_st)
        dr = n_stable + rng.randint(0, n_drift, size=n_dr)
        cat = np.concatenate([st, dr])
        rng.shuffle(cat)
        return cat

    train_cat = _split(n_train, train_drift_frac)
    test_cat = _split(n_test, test_drift_frac)
    y_tr = (rng.rand(n_train) < train_mean[train_cat]).astype(int)
    y_te = (rng.rand(n_test) < test_mean[test_cat]).astype(int)
    X_tr = pd.DataFrame({"cat": np.array([f"k{names[c]}" for c in train_cat], dtype=object)})
    X_te = pd.DataFrame({"cat": np.array([f"k{names[c]}" for c in test_cat], dtype=object)})
    return X_tr, X_te, y_tr.astype(int), y_te.astype(int)


def make_heteroscedastic_scale_shift(
    n_train: int = 400,
    n_test: int = 2000,
    n_signal: int = 4,
    n_noise: int = 40,
    test_noise_scale: float = 12.0,
    spurious: float = 0.0,
    signal_strength: float = 1.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate stable signal features and noise features with larger test variance.

    The M5 tests use this heteroscedastic shift to compare joint scaling with the
    corresponding train-only version.
    """
    rng = np.random.RandomState(seed)

    # Signal has a linear relationship and stable scale on both splits.
    w = rng.randn(n_signal)
    Xs_tr = rng.randn(n_train, n_signal)
    Xs_te = rng.randn(n_test, n_signal)
    y_tr = (signal_strength * (Xs_tr @ w) + rng.randn(n_train) > 0).astype(int)
    y_te = (signal_strength * (Xs_te @ w) + rng.randn(n_test) > 0).astype(int)

    # Nuisance features use train ~N(0,1), with an optional small train-only
    # spurious component. Test variance is multiplied by `test_noise_scale`.
    y_tr_signed = (2.0 * y_tr - 1.0)
    Xn_tr = rng.randn(n_train, n_noise) + spurious * y_tr_signed[:, None]
    Xn_te = test_noise_scale * rng.randn(n_test, n_noise)

    cols = [f"sig{i}" for i in range(n_signal)] + [f"noise{i}" for i in range(n_noise)]
    X_tr = pd.DataFrame(np.hstack([Xs_tr, Xn_tr]), columns=cols)
    X_te = pd.DataFrame(np.hstack([Xs_te, Xn_te]), columns=cols)
    return X_tr, X_te, y_tr.astype(int), y_te.astype(int)


def make_test_region_positive(
    n_train: int = 800,
    n_test: int = 1500,
    n_noise: int = 4,
    sigma_train: float = 2.0,
    sigma_test: float = 0.9,
    radius: float = 1.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate radial labels with test features concentrated in the positive region.

    The M6 feature-mode tests use the domain-classifier output to represent this
    difference in feature density.
    """
    rng = np.random.RandomState(seed)
    Xi_tr = sigma_train * rng.randn(n_train, 2)
    Xi_te = sigma_test * rng.randn(n_test, 2)
    y_tr = ((Xi_tr ** 2).sum(1) < radius ** 2).astype(int)
    y_te = ((Xi_te ** 2).sum(1) < radius ** 2).astype(int)
    Xn_tr = rng.randn(n_train, n_noise)
    Xn_te = rng.randn(n_test, n_noise)
    cols = ["x0", "x1"] + [f"noise{i}" for i in range(n_noise)]
    X_tr = pd.DataFrame(np.hstack([Xi_tr, Xn_tr]), columns=cols)
    X_te = pd.DataFrame(np.hstack([Xi_te, Xn_te]), columns=cols)
    return X_tr, X_te, y_tr.astype(int), y_te.astype(int)


def make_density_checkerboard(
    n_train: int = 600,
    n_test: int = 1500,
    n_noise: int = 4,
    c: float = 2.0,
    sigma: float = 0.9,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate XOR quadrants with class-dependent training and test densities.

    The M18 tests use leaf-occupancy features to represent the density difference.
    """
    rng = np.random.RandomState(seed)
    centers = np.array([(+1, +1), (-1, +1), (-1, -1), (+1, -1)], dtype=float)
    qlabel = np.array([1, 0, 1, 0])  # same-sign quadrants are y=1
    train_w = np.array([0.03, 0.47, 0.03, 0.47])  # y=1 quads test-sparse in train
    test_w = np.array([0.47, 0.03, 0.47, 0.03])   # y=1 quads test-dense

    def _sample(n: int, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = rng.choice(4, size=n, p=w)
        Xi = centers[q] * c + sigma * rng.randn(n, 2)
        Xn = rng.randn(n, n_noise)
        return np.hstack([Xi, Xn]), qlabel[q]

    X_tr, y_tr = _sample(n_train, train_w)
    X_te, y_te = _sample(n_test, test_w)
    cols = ["x0", "x1"] + [f"noise{i}" for i in range(n_noise)]
    return (
        pd.DataFrame(X_tr, columns=cols),
        pd.DataFrame(X_te, columns=cols),
        y_tr.astype(int),
        y_te.astype(int),
    )


def make_cluster_density_shift(
    n_train: int = 600,
    n_test: int = 1500,
    n_noise: int = 4,
    c: float = 2.0,
    sigma: float = 0.9,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate XOR clusters with class-dependent test density.

    Training data is uniform across clusters. Test data places more probability on
    the class-one clusters. This distribution is used by the M20 tests.
    """
    rng = np.random.RandomState(seed)
    centers = np.array([(+1, +1), (-1, +1), (-1, -1), (+1, -1)], dtype=float)
    qlabel = np.array([1, 0, 1, 0])                  # same-sign blobs are y=1 (XOR)
    train_w = np.array([0.25, 0.25, 0.25, 0.25])     # uniform training weights
    test_w = np.array([0.47, 0.03, 0.47, 0.03])      # y=1 blobs are test-dense and y=0 blobs are sparse

    def _sample(n: int, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = rng.choice(4, size=n, p=w)
        Xi = centers[q] * c + sigma * rng.randn(n, 2)
        Xn = rng.randn(n, n_noise)
        return np.hstack([Xi, Xn]), qlabel[q]

    X_tr, y_tr = _sample(n_train, train_w)
    X_te, y_te = _sample(n_test, test_w)
    cols = ["x0", "x1"] + [f"noise{i}" for i in range(n_noise)]
    return (
        pd.DataFrame(X_tr, columns=cols),
        pd.DataFrame(X_te, columns=cols),
        y_tr.astype(int),
        y_te.astype(int),
    )

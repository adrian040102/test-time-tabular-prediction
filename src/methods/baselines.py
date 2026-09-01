"""Train-only versions of feature transformations.

Each method applies the same transformation as its joint version but fits the
transformation using training rows only. Comparing the joint version with the
train-only version measures the test-time premium. Comparing the train-only
version with ``InductiveBaseline`` measures the effect of the transformation.

For M5, the comparison uses ``TrainOnlyScaling`` and ``JointScaling``.
``NoScaling`` remains available for reproducing saved experiment records. It is
excluded from evaluated method lists because it duplicates ``InductiveBaseline``
at the prediction-model boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

from src.methods.base import TestTimeMethod, TransformResult


# ---------------------------------------------------------------------------
# Helpers (shared with tier1)
# ---------------------------------------------------------------------------

def _label_encode_cats_train_only(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit categorical encodings on training data only."""
    X_tr = X_train.copy()
    X_te = X_test.copy()
    cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(X_tr[col].astype(str))
        X_tr[col] = le.transform(X_tr[col].astype(str))
        # Handle unseen categories in test
        test_vals = X_te[col].astype(str)
        known_mask = test_vals.isin(le.classes_)
        X_te[col] = -1
        X_te.loc[known_mask, col] = le.transform(test_vals[known_mask])
    return X_tr, X_te, cat_cols


def _get_cat_cols(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=["object", "category"]).columns.tolist()


def _get_num_cols(X: pd.DataFrame) -> list[str]:
    return X.select_dtypes(include=["number"]).columns.tolist()


# ===========================================================================
# B-M1: Frequency Encoding (Train-Only)
# ===========================================================================

class TrainOnlyFrequencyEncoding(TestTimeMethod):
    """
    Baseline B for M1: Frequency encoding computed on training data only.

    Each category is replaced by its relative frequency in the training set.
    This isolates the encoding benefit (frequency vs. label encoding) from
    the test-time benefit (joint frequency computation).
    """

    @property
    def name(self) -> str:
        return "freq_enc_train_only"

    def fit_transform(self, X_train, X_test, y_train):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        cat_cols = _get_cat_cols(X_train)

        for col in cat_cols:
            train_vals = X_tr[col].astype(str)
            test_vals = X_te[col].astype(str)

            # Frequencies computed on train only
            freq_map = train_vals.value_counts(normalize=True).to_dict()

            col_name = f"{col}_freq_train"
            X_tr[col_name] = train_vals.map(freq_map).fillna(0).astype(float)
            X_te[col_name] = test_vals.map(freq_map).fillna(0).astype(float)

        # Preserve original categorical columns as train-only label-encoded
        # integers alongside the new encoded features, as in joint M1/M4.
        # Preserve the same columns as the joint version so dropped columns do not
        # affect the joint-versus-train-only comparison.
        X_tr, X_te, _ = _label_encode_cats_train_only(X_tr, X_te)

        feature_names = list(X_tr.columns)
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=feature_names,
            categorical_features=cat_cols if cat_cols else None,
        )


# ===========================================================================
# B-M3: PCA (Train-Only)
# ===========================================================================

class TrainOnlyPCA(TestTimeMethod):
    """
    Baseline B for M3: PCA fitted on training data only.

    Same as JointPCA, but the PCA model is fitted only on training data.
    Test data is projected using the train-fitted PCA.  PCA runs on
    **numeric columns only**. Categoricals are label-encoded for
    pass-through but excluded from PCA.

    Args:
        n_components: Number of PCA components to add (int, default 5)
            or a float in (0, 1) for adaptive selection by explained
            variance fraction (e.g. 0.9 = 90% variance).
        seed: random_state for PCA. Only affects the randomized-SVD path
            (wide datasets, int n_components). It has no effect for full or
            covariance-eigh SVD and for float n_components. Threaded by
            ``_propagate_seed`` so it tracks the run seed. Default 42.
    """

    def __init__(self, n_components: int | float = 5, seed: int = 42):
        self.n_components = n_components
        self.seed = int(seed)

    @property
    def name(self) -> str:
        if isinstance(self.n_components, float):
            pct = int(self.n_components * 100)
            return f"pca_train_only_var{pct}"
        return f"pca_train_only_{self.n_components}"

    def fit_transform(self, X_train, X_test, y_train):
        X_tr, X_te, cat_cols = _label_encode_cats_train_only(X_train, X_test)
        cat_set = set(cat_cols)

        all_cols = list(X_tr.columns)
        num_cols = [c for c in all_cols if c not in cat_set]

        # Full feature arrays (label-encoded cats + original numerics, unscaled).
        # These pass through unchanged. The pipeline imputes missing values downstream.
        X_tr_vals = X_tr[all_cols].values.astype(float)
        X_te_vals = X_te[all_cols].values.astype(float)

        # Edge case: no numeric columns → skip PCA, return label-encoded features
        if len(num_cols) == 0:
            return TransformResult(
                X_train=X_tr_vals,
                X_test=X_te_vals,
                feature_names=all_cols,
                categorical_features=cat_cols if cat_cols else None,
                metadata={
                    "explained_variance_ratio": [],
                    "n_pca_components": 0,
                    "note": "PCA skipped because there are no numerical columns",
                },
            )

        # Extract numeric columns only for PCA
        X_tr_num = X_tr[num_cols].values.astype(float)
        X_te_num = X_te[num_cols].values.astype(float)

        # Impute missing numerical values with the training median.
        if np.isnan(X_tr_num).any() or np.isnan(X_te_num).any():
            col_medians = np.nanmedian(X_tr_num, axis=0)
            for j in range(X_tr_num.shape[1]):
                nan_val = col_medians[j] if np.isfinite(col_medians[j]) else 0.0
                mask_tr = np.isnan(X_tr_num[:, j])
                mask_te = np.isnan(X_te_num[:, j])
                if mask_tr.any():
                    X_tr_num[mask_tr, j] = nan_val
                if mask_te.any():
                    X_te_num[mask_te, j] = nan_val

        # Fit the scaler on training data only.
        scaler = StandardScaler()
        scaler.fit(X_tr_num)
        X_tr_scaled = scaler.transform(X_tr_num)
        X_te_scaled = scaler.transform(X_te_num)

        # Fit PCA on training data only.
        n_comp = min(self.n_components, len(num_cols))
        # _propagate_seed passes the run seed to random_state so
        # the randomized-SVD path on wide datasets is reproducible and
        # independent of global np.random state. No-op for full / covariance-
        # eigh SVD and for float (variance-fraction) n_components. JointPCA uses
        # the same seed logic, which keeps the comparison consistent.
        pca = PCA(n_components=n_comp, random_state=self.seed)
        pca.fit(X_tr_scaled)

        pca_train = pca.transform(X_tr_scaled)
        pca_test = pca.transform(X_te_scaled)
        n_actual = pca_train.shape[1]  # actual count (differs from n_comp when adaptive)
        pca_names = [f"pca_train_pc{i+1}" for i in range(n_actual)]

        # Append to original (unscaled) features
        X_tr_out = np.hstack([X_tr_vals, pca_train])
        X_te_out = np.hstack([X_te_vals, pca_test])
        feature_names = all_cols + pca_names

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            categorical_features=cat_cols if cat_cols else None,
            metadata={
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "n_pca_components": n_actual,
                "n_numeric_cols": len(num_cols),
                "n_categorical_cols": len(cat_cols),
            },
        )


# ===========================================================================
# B-M4: Standard Target Encoding (no shift regularization)
# ===========================================================================

class StandardTargetEncoding(TestTimeMethod):
    """
    Baseline B for M4: Standard smoothed target encoding (train-only).

    Uses the classic formula:
        encoding(cat) = (n_cat * mean_y_cat + m * global_mean) / (n_cat + m)

    where m is a fixed smoothing parameter (no test-frequency adjustment).
    This isolates the effect of test-frequency regularization in M4.
    """

    def __init__(self, m: float = 10.0):
        self.m = m

    @property
    def name(self) -> str:
        return f"target_enc_standard_m{int(self.m)}"

    @property
    def uses_labels(self) -> bool:
        return True

    def fit_transform(self, X_train, X_test, y_train):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        cat_cols = _get_cat_cols(X_train)
        y = np.asarray(y_train).astype(float)
        global_mean = y.mean()

        for col in cat_cols:
            train_vals = X_tr[col].astype(str)
            test_vals = X_te[col].astype(str)

            # Per-category stats from training data
            cat_stats = pd.DataFrame({"cat": train_vals.values, "y": y})
            grouped = cat_stats.groupby("cat")["y"]
            cat_mean = grouped.mean().to_dict()
            cat_count = grouped.count().to_dict()

            all_cats = set(cat_mean) | set(test_vals.unique().astype(str))
            encoding_map = {}
            for cat in all_cats:
                n_cat = cat_count.get(cat, 0)
                mean_y_cat = cat_mean.get(cat, global_mean)
                encoding = (n_cat * mean_y_cat + self.m * global_mean) / (n_cat + self.m)
                encoding_map[cat] = encoding

            col_name = f"{col}_target_enc"
            X_tr[col_name] = train_vals.map(encoding_map).fillna(global_mean).astype(float)
            X_te[col_name] = test_vals.map(encoding_map).fillna(global_mean).astype(float)

        # Preserve original categorical columns as train-only label-encoded
        # integers alongside the new encoded features, as in joint M1/M4.
        # Preserve the same columns as the joint version so dropped columns do not
        # affect the joint-versus-train-only comparison.
        X_tr, X_te, _ = _label_encode_cats_train_only(X_tr, X_te)

        feature_names = list(X_tr.columns)
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=feature_names,
            categorical_features=cat_cols if cat_cols else None,
        )


# ===========================================================================
# B-M5: Scaling Ablation (Three-Tier)
# ===========================================================================

class NoScaling(TestTimeMethod):
    """
    M5 control excluded from scientific comparisons. It encodes categorical
    features and leaves numerical features unchanged. The pipeline can still
    apply its standard post-method scaling.
    """

    @property
    def name(self) -> str:
        return "no_scaling"

    def fit_transform(self, X_train, X_test, y_train):
        X_tr, X_te, cat_cols = _label_encode_cats_train_only(X_train, X_test)

        feature_names = list(X_tr.columns)
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=feature_names,
        )


class TrainOnlyScaling(TestTimeMethod):
    """
    M5 ablation tier 2: StandardScaler fit on training data only.

    This is the standard inductive approach to scaling.
    Compared with JointScaling (tier1.py) to measure the
    benefit of including test data in the scaler fit.
    """

    @property
    def name(self) -> str:
        return "train_only_scaling"

    def fit_transform(self, X_train, X_test, y_train):
        X_tr, X_te, cat_cols = _label_encode_cats_train_only(X_train, X_test)

        X_tr_vals = X_tr.values.astype(float)
        X_te_vals = X_te.values.astype(float)

        # Fit the scaler on training data only.
        scaler = StandardScaler()
        scaler.fit(X_tr_vals)

        X_tr_out = scaler.transform(X_tr_vals)
        X_te_out = scaler.transform(X_te_vals)

        feature_names = list(X_tr.columns)
        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
        )


# ===========================================================================
# B-M14: Group-By Features (Train-Only)
# ===========================================================================

class TrainOnlyGroupByFeatures(TestTimeMethod):
    """
    Baseline B for M14: Group-by aggregations computed on training data only.

    Same aggregations as JointGroupByFeatures but the statistics are computed
    exclusively from training data.  Test data is mapped using the
    train-computed group statistics.
    """

    def __init__(
        self,
        group_cols: list[str] | str = "auto",
        agg_cols: list[str] | str = "auto",
        agg_funcs: list[str] | None = None,
        max_unique: int = 100,
    ):
        self.group_cols = group_cols
        self.agg_cols = agg_cols
        self.agg_funcs = agg_funcs or ["mean", "count", "std"]
        self.max_unique = max_unique

    @property
    def name(self) -> str:
        return "groupby_train_only"

    def _resolve_columns(self, X_train: pd.DataFrame):
        """Resolve 'auto' column selections using training data only.

        Uses the same logic as JointGroupByFeatures._resolve_columns: use categorical
        columns as group columns, with low-cardinality numerics as fallback
        only when no categoricals exist.
        """
        if self.group_cols == "auto":
            cat_cols = _get_cat_cols(X_train)
            group_cols = [c for c in cat_cols
                          if X_train[c].nunique() <= self.max_unique]

            # Fallback: if no categoricals found, detect low-cardinality
            # numeric columns (label-encoded/scaled categoricals)
            if not group_cols:
                for c in _get_num_cols(X_train):
                    if X_train[c].nunique() <= self.max_unique:
                        group_cols.append(c)
        else:
            group_cols = list(self.group_cols)

        if self.agg_cols == "auto":
            num_cols = _get_num_cols(X_train)
            agg_cols = [c for c in num_cols
                        if c not in group_cols and X_train[c].std() > 0]
        else:
            agg_cols = list(self.agg_cols)

        return group_cols, agg_cols

    def fit_transform(self, X_train, X_test, y_train):
        X_tr, X_te, cat_cols_encoded = _label_encode_cats_train_only(X_train, X_test)

        group_cols, agg_cols = self._resolve_columns(X_train)

        if not group_cols or not agg_cols:
            orig_cat_cols = _get_cat_cols(X_train)
            remaining_cats = [c for c in orig_cat_cols if c in X_tr.columns]
            feature_names = list(X_tr.columns)
            return TransformResult(
                X_train=X_tr.values.astype(float),
                X_test=X_te.values.astype(float),
                feature_names=feature_names,
                categorical_features=remaining_cats if remaining_cats else None,
            )

        # Collect new columns in dicts to avoid DataFrame fragmentation
        new_cols_tr: dict[str, np.ndarray] = {}
        new_cols_te: dict[str, np.ndarray] = {}
        n_new = 0
        for gcol in group_cols:
            if gcol not in X_tr.columns:
                continue
            for acol in agg_cols:
                if acol not in X_tr.columns or acol == gcol:
                    continue
                for func in self.agg_funcs:
                    # Compute the aggregation on training data only.
                    agg_map = (
                        X_tr.groupby(X_tr[gcol].astype(str))[acol]
                        .agg(func).to_dict()
                    )
                    global_fallback = X_tr[acol].astype(float).agg(func)

                    col_name = f"{gcol}_{acol}_{func}_train"
                    new_cols_tr[col_name] = (
                        X_tr[gcol].astype(str).map(agg_map)
                        .fillna(global_fallback).astype(float).values
                    )
                    new_cols_te[col_name] = (
                        X_te[gcol].astype(str).map(agg_map)
                        .fillna(global_fallback).astype(float).values
                    )
                    n_new += 1

        # Batch-append all new columns at once (avoids fragmentation)
        if new_cols_tr:
            X_tr = pd.concat([X_tr, pd.DataFrame(new_cols_tr, index=X_tr.index)], axis=1)
            X_te = pd.concat([X_te, pd.DataFrame(new_cols_te, index=X_te.index)], axis=1)

        # Preserve all original categorical and group columns as label-encoded
        # integers. JointGroupByFeatures also retains these columns.
        X_tr = X_tr.replace([np.inf, -np.inf], 0).fillna(0)
        X_te = X_te.replace([np.inf, -np.inf], 0).fillna(0)

        orig_cat_cols = _get_cat_cols(X_train)
        remaining_cats = [c for c in orig_cat_cols if c in X_tr.columns]
        feature_names = list(X_tr.columns)
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=feature_names,
            metadata={"n_groupby_features": n_new},
            categorical_features=remaining_cats if remaining_cats else None,
        )


# ===========================================================================
# Registry
# ===========================================================================

BASELINE_METHODS: dict[str, callable] = {
    # B-M1: Frequency encoding (train-only)
    "freq_enc_train_only": lambda: TrainOnlyFrequencyEncoding(),
    # B-M3: PCA (train-only)
    "pca_train_only_5": lambda: TrainOnlyPCA(n_components=5),
    "pca_train_only_10": lambda: TrainOnlyPCA(n_components=10),
    "pca_train_only_var90": lambda: TrainOnlyPCA(n_components=0.9),
    "pca_train_only_var95": lambda: TrainOnlyPCA(n_components=0.95),
    # B-M4: Standard target encoding (no shift regularization)
    "target_enc_standard_m10": lambda: StandardTargetEncoding(m=10.0),
    "target_enc_standard_m5": lambda: StandardTargetEncoding(m=5.0),
    "target_enc_standard_m20": lambda: StandardTargetEncoding(m=20.0),
    # Excluded from scientific analyses.
    "no_scaling": lambda: NoScaling(),
    "train_only_scaling": lambda: TrainOnlyScaling(),
    # B-M14: Group-by (train-only)
    "groupby_train_only": lambda: TrainOnlyGroupByFeatures(),
}


def get_baseline_method(name: str) -> TestTimeMethod:
    """Instantiate a baseline method by registry name."""
    if name not in BASELINE_METHODS:
        raise ValueError(
            f"Unknown baseline method '{name}'. "
            f"Available: {sorted(BASELINE_METHODS.keys())}"
        )
    return BASELINE_METHODS[name]()

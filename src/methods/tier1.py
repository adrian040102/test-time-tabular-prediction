"""
Tier 1 Test-Time Methods: Simple Joint Feature Engineering.

These methods share a common principle: instead of computing features
only on training data, they include test data features (never labels).

Methods:
    M1:   JointFrequencyEncoding: frequency counts over train+test
    M2:   FrequencyRatio: freq_test / freq_train per category
    M3:   JointPCA: PCA fitted on combined feature space
    M3b:  JointTSNE: t-SNE fitted on combined feature space
    M4:   ShiftRegularizedTargetEncoding: target encoding with test-frequency smoothing
    M5:   JointScaling: StandardScaler fitted on train+test
    M14:  JointGroupByFeatures: group-by aggregations over train+test
    M14b: GroupByShiftRatio: mean_test / mean_train per group (shift signal)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

from src.methods.base import TestTimeMethod, TransformResult


# ---------------------------------------------------------------------------
# Helpers delegated to the shared preprocessing module
# ---------------------------------------------------------------------------

from src.methods.preprocessing import (
    label_encode_joint as _label_encode_cats_impl,
    get_cat_cols as _get_cat_cols,
    get_num_cols as _get_num_cols,
)


def _label_encode_cats(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Label-encode all categorical columns (fit on train+test combined).

    Wrapper around :func:`preprocessing.label_encode_joint` used by the tier 1
    and tier 2 modules.
    """
    return _label_encode_cats_impl(X_train, X_test)


# ===========================================================================
# M1: Joint Frequency Encoding
# ===========================================================================

class JointFrequencyEncoding(TestTimeMethod):
    """
    M1: Replace each category with its frequency computed over train+test.

    Two variants (controlled by `variant`):
      - "combined": single feature = freq(cat) over train+test
      - "separate": two features per cat col = freq_train(cat), freq_test(cat)
    """

    def __init__(self, variant: str = "combined"):
        assert variant in ("combined", "separate"), \
            f"variant must be 'combined' or 'separate', got '{variant}'"
        self.variant = variant

    @property
    def name(self) -> str:
        return f"joint_freq_enc_{self.variant}"

    def fit_transform(self, X_train, X_test, y_train):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        cat_cols = _get_cat_cols(X_train)
        new_feature_names = []

        for col in cat_cols:
            train_vals = X_tr[col].astype(str)
            test_vals = X_te[col].astype(str)

            if self.variant == "combined":
                # Frequency over train+test combined
                combined = pd.concat([train_vals, test_vals])
                freq_map = combined.value_counts(normalize=True).to_dict()
                col_name = f"{col}_freq_joint"
                X_tr[col_name] = train_vals.map(freq_map).fillna(0).astype(float)
                X_te[col_name] = test_vals.map(freq_map).fillna(0).astype(float)
                new_feature_names.append(col_name)
            else:
                # Separate: freq_train and freq_test
                freq_train = train_vals.value_counts(normalize=True).to_dict()
                freq_test = test_vals.value_counts(normalize=True).to_dict()
                col_tr = f"{col}_freq_train"
                col_te_name = f"{col}_freq_test"
                X_tr[col_tr] = train_vals.map(freq_train).fillna(0).astype(float)
                X_te[col_tr] = test_vals.map(freq_train).fillna(0).astype(float)
                X_tr[col_te_name] = train_vals.map(freq_test).fillna(0).astype(float)
                X_te[col_te_name] = test_vals.map(freq_test).fillna(0).astype(float)
                new_feature_names.extend([col_tr, col_te_name])

        # Preserve original cats as label-encoded integers alongside the new
        # freq features so cat-aware models can still use native cat splits.
        for col in cat_cols:
            le = LabelEncoder()
            combined = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)])
            le.fit(combined)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_te[col] = le.transform(X_te[col].astype(str))

        feature_names = list(X_tr.columns)
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=feature_names,
            categorical_features=cat_cols if cat_cols else None,
        )


# ===========================================================================
# M2: Train/Test Frequency Ratio
# ===========================================================================

class FrequencyRatio(TestTimeMethod):
    """
    M2: Compute freq_test(cat) / freq_train(cat) per category.

    With Laplace smoothing to avoid division by zero.
    The ratio is optionally log-transformed for symmetry.

    Args:
        use_log: Whether to log-transform the ratio (default True).
        alpha: Laplace smoothing constant (default 1.0).
    """

    def __init__(self, use_log: bool = True, alpha: float = 1.0):
        self.use_log = use_log
        self.alpha = alpha

    @property
    def name(self) -> str:
        log_str = "log" if self.use_log else "raw"
        return f"freq_ratio_{log_str}"

    def fit_transform(self, X_train, X_test, y_train):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        cat_cols = _get_cat_cols(X_train)
        n_train = len(X_tr)
        n_test = len(X_te)

        for col in cat_cols:
            train_vals = X_tr[col].astype(str)
            test_vals = X_te[col].astype(str)

            # Compute smoothed proportions per category
            train_counts = train_vals.value_counts().to_dict()
            test_counts = test_vals.value_counts().to_dict()
            all_cats = set(train_counts) | set(test_counts)

            ratio_map = {}
            for cat in all_cats:
                f_train = (train_counts.get(cat, 0) + self.alpha) / (n_train + self.alpha * len(all_cats))
                f_test = (test_counts.get(cat, 0) + self.alpha) / (n_test + self.alpha * len(all_cats))
                ratio = f_test / f_train
                if self.use_log:
                    ratio = np.log(ratio)
                else:
                    # Clip raw ratios to limit extreme values for linear models.
                    ratio = np.clip(ratio, 0.01, 100.0)
                ratio_map[cat] = ratio

            col_name = f"{col}_freq_ratio"
            X_tr[col_name] = train_vals.map(ratio_map).astype(float)
            X_te[col_name] = test_vals.map(ratio_map).astype(float)

        # Label-encode original categoricals (retain category identity
        # alongside the ratio features so the model gets both prediction
        # signal and shift signal)
        for col in cat_cols:
            le = LabelEncoder()
            combined = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)])
            le.fit(combined)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_te[col] = le.transform(X_te[col].astype(str))

        feature_names = list(X_tr.columns)
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=feature_names,
            categorical_features=cat_cols if cat_cols else None,
        )


# ===========================================================================
# M2v2: Improved Frequency Ratio
# ===========================================================================

class FrequencyRatioImproved(TestTimeMethod):
    """
    M2v2 frequency-ratio features with three optional components:

    1. Adaptive smoothing sets ``alpha = 1.0 / n_categories`` for each column.

    2. Subtraction provides ``freq_test - freq_train`` as an alternative to
       the ratio and represents absolute changes in prevalence.

    3. Shift filtering uses a chi-squared test for each categorical column.
       The generated feature is set to zero when ``p > gate_threshold``.

    Args:
        mode: "ratio" (log ratio, like original), "diff" (subtraction)
              or "both" (two features per column).
        adaptive_alpha: Use adaptive smoothing (default True).
        noise_gate: Apply chi-squared shift filtering (default True).
        gate_threshold: p-value threshold: columns with p > this are
            zeroed out (default 0.05).
    """

    def __init__(
        self,
        mode: str = "ratio",
        adaptive_alpha: bool = True,
        noise_gate: bool = True,
        gate_threshold: float = 0.05,
    ):
        assert mode in ("ratio", "diff", "both")
        self.mode = mode
        self.adaptive_alpha = adaptive_alpha
        self.noise_gate = noise_gate
        self.gate_threshold = gate_threshold

    @property
    def name(self) -> str:
        parts = [f"freq_ratio_v2_{self.mode}"]
        if self.noise_gate:
            parts.append("gated")
        else:
            parts.append("ungated")
        return "_".join(parts)

    def _chi2_pvalue(self, train_vals: pd.Series, test_vals: pd.Series) -> float:
        """Compute chi-squared p-value for distribution shift in a categorical column."""
        from scipy.stats import chi2_contingency

        all_cats = sorted(set(train_vals.unique()) | set(test_vals.unique()))
        if len(all_cats) <= 1:
            return 1.0  # No shift possible with 1 category

        train_counts = train_vals.value_counts()
        test_counts = test_vals.value_counts()
        observed = np.array([
            [train_counts.get(c, 0) for c in all_cats],
            [test_counts.get(c, 0) for c in all_cats],
        ])

        # Remove zero-sum columns to avoid chi2 issues
        col_mask = observed.sum(axis=0) > 0
        observed = observed[:, col_mask]

        if observed.shape[1] <= 1:
            return 1.0

        try:
            _, pval, _, _ = chi2_contingency(observed)
            return float(pval)
        except Exception:
            return 1.0

    def fit_transform(self, X_train, X_test, y_train):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        cat_cols = _get_cat_cols(X_train)
        n_train = len(X_tr)
        n_test = len(X_te)
        metadata = {}

        for col in cat_cols:
            train_vals = X_tr[col].astype(str)
            test_vals = X_te[col].astype(str)

            train_counts = train_vals.value_counts().to_dict()
            test_counts = test_vals.value_counts().to_dict()
            all_cats = set(train_counts) | set(test_counts)
            n_cats = len(all_cats)

            # Adapt the smoothing strength to the number of observed categories.
            alpha = (1.0 / n_cats) if self.adaptive_alpha else 1.0

            # Optionally suppress features when the category distributions do not shift.
            is_shifted = True
            if self.noise_gate:
                pval = self._chi2_pvalue(train_vals, test_vals)
                is_shifted = pval < self.gate_threshold
                metadata[f"chi2_pval_{col}"] = pval

            ratio_map = {}
            diff_map = {}
            for cat in all_cats:
                f_train = (train_counts.get(cat, 0) + alpha) / (n_train + alpha * n_cats)
                f_test = (test_counts.get(cat, 0) + alpha) / (n_test + alpha * n_cats)

                if is_shifted:
                    ratio_map[cat] = np.log(f_test / f_train)
                    diff_map[cat] = f_test - f_train
                else:
                    # Zero out. No signal for unshifted columns
                    ratio_map[cat] = 0.0
                    diff_map[cat] = 0.0

            if self.mode in ("ratio", "both"):
                col_name = f"{col}_freq_ratio_v2"
                X_tr[col_name] = train_vals.map(ratio_map).astype(float)
                X_te[col_name] = test_vals.map(ratio_map).astype(float)

            if self.mode in ("diff", "both"):
                col_name = f"{col}_freq_diff"
                X_tr[col_name] = train_vals.map(diff_map).astype(float)
                X_te[col_name] = test_vals.map(diff_map).astype(float)

        # Label-encode original categoricals (retain category identity
        # alongside the ratio/diff features so the model gets both prediction
        # signal and shift signal)
        for col in cat_cols:
            le = LabelEncoder()
            combined = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)])
            le.fit(combined)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_te[col] = le.transform(X_te[col].astype(str))

        feature_names = list(X_tr.columns)
        n_gated = sum(1 for k, v in metadata.items() if k.startswith("chi2_pval_") and v >= self.gate_threshold)
        metadata["n_columns_gated"] = n_gated
        metadata["n_columns_shifted"] = len(cat_cols) - n_gated

        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=feature_names,
            metadata=metadata,
            categorical_features=cat_cols if cat_cols else None,
        )


# ===========================================================================
# M3: Joint PCA
# ===========================================================================

class JointPCA(TestTimeMethod):
    """
    M3: Fit PCA on the combined (train+test) numerical feature space.

    PCA runs on **numeric columns only**: categoricals are label-encoded
    for pass-through but excluded from PCA (label-encoded integers have no
    meaningful Euclidean distance for PCA to exploit).

    The resulting components are appended as additional features
    (original features are kept). Both splits live in the same
    coordinate system because they come from the same PCA fit.

    Args:
        n_components: Number of PCA components to add (int, default 5)
            or a float in (0, 1) for adaptive selection by explained
            variance fraction (e.g. 0.9 = 90% variance).
        seed: random_state for PCA. Only affects the randomized-SVD path
            (wide datasets where sklearn 'auto' picks it for int
            n_components). A no-op for full / covariance-eigh SVD and for
            float (variance-fraction) n_components. Threaded by
            ``_propagate_seed`` so it tracks the run seed. Default 42.
    """

    def __init__(self, n_components: int | float = 5, seed: int = 42):
        self.n_components = n_components
        self.seed = int(seed)

    @property
    def name(self) -> str:
        if isinstance(self.n_components, float):
            pct = int(self.n_components * 100)
            return f"joint_pca_var{pct}"
        return f"joint_pca_{self.n_components}"

    @property
    def produces_prescaled_output(self) -> bool:
        return True

    def fit_transform(self, X_train, X_test, y_train):
        X_tr, X_te, cat_cols = _label_encode_cats(X_train, X_test)
        cat_set = set(cat_cols)

        all_cols = list(X_tr.columns)
        num_cols = [c for c in all_cols if c not in cat_set]

        # Full feature arrays (label-encoded cats + original numerics, unscaled).
        # These pass through as-is. Pipeline handles NaN imputation downstream.
        X_tr_vals = X_tr[all_cols].values.astype(float)
        X_te_vals = X_te[all_cols].values.astype(float)

        # Edge case: no numeric columns → skip PCA, return label-encoded features
        if len(num_cols) == 0:
            return TransformResult(
                X_train=X_tr_vals,
                X_test=X_te_vals,
                feature_names=all_cols,
                # Declare the label-encoded passthrough cats so raw-feature
                # models (bag8/TabPFN/TabICL) receive native categorical splits.
                # The main return below and TrainOnlyPCA use the same declaration,
                # so both versions handle all-categorical datasets consistently.
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

        # Impute missing numeric values with combined medians before PCA.
        combined_num = np.vstack([X_tr_num, X_te_num])
        if np.isnan(combined_num).any():
            col_medians = np.nanmedian(combined_num, axis=0)
            for j in range(X_tr_num.shape[1]):
                nan_val = col_medians[j] if np.isfinite(col_medians[j]) else 0.0
                mask_tr = np.isnan(X_tr_num[:, j])
                mask_te = np.isnan(X_te_num[:, j])
                if mask_tr.any():
                    X_tr_num[mask_tr, j] = nan_val
                if mask_te.any():
                    X_te_num[mask_te, j] = nan_val

        # Scale numeric columns before PCA (combined fit)
        scaler = StandardScaler()
        combined_num = np.vstack([X_tr_num, X_te_num])
        scaler.fit(combined_num)
        X_tr_scaled = scaler.transform(X_tr_num)
        X_te_scaled = scaler.transform(X_te_num)

        # Fit PCA on combined scaled numeric data
        # For int n_components: clamp to number of numeric features
        # For float n_components (e.g. 0.9): min() is a no-op (float < n_features)
        n_comp = min(self.n_components, len(num_cols))
        # Use the run seed for randomized SVD without depending on global
        # NumPy state.
        pca = PCA(n_components=n_comp, random_state=self.seed)
        combined_scaled = np.vstack([X_tr_scaled, X_te_scaled])
        pca.fit(combined_scaled)

        # Transform both splits
        pca_train = pca.transform(X_tr_scaled)
        pca_test = pca.transform(X_te_scaled)
        n_actual = pca_train.shape[1]  # actual count (differs from n_comp when adaptive)
        pca_names = [f"joint_pc{i+1}" for i in range(n_actual)]

        # Pass the original unscaled numerical features through unchanged, as in
        # TrainOnlyPCA. Joint scaling (X_tr_scaled/X_te_scaled) is used only to
        # fit and transform the PCA above. It is not written into
        # the passthrough block. This keeps the numeric columns byte-identical
        # between the joint and train-only versions. Their comparison isolates the
        # PCA fit without scaling asymmetry in the passthrough block.
        # skip_pipeline_scaler stays True below so the pipeline's train-only
        # StandardScaler cannot refit on the components and alter the joint coordinate
        # system used by PCA.
        # Label-encoded categorical columns pass through as integers. Tree models can
        # use them. Scale-sensitive models should one-hot encode them upstream.
        X_tr_out = np.hstack([X_tr_vals, pca_train])
        X_te_out = np.hstack([X_te_vals, pca_test])
        feature_names = all_cols + pca_names

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            skip_pipeline_scaler=True,
            # Declare the label-encoded passthrough cats so raw-feature models
            # (LightGBM bag8) receive native categorical splits, as in
            # JointTSNE (M3b). No-op for non-raw models / all-numeric datasets.
            categorical_features=cat_cols if cat_cols else None,
            metadata={
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "n_pca_components": n_actual,
                "n_numeric_cols": len(num_cols),
                "n_categorical_cols": len(cat_cols),
            },
        )


# ===========================================================================
# M3b: Joint t-SNE
# ===========================================================================

class JointTSNESkipReason(RuntimeError):
    """Raised when :class:`JointTSNE` cannot run on the supplied split sizes.

    The current guard triggers when ``n_train + n_test > max_n_total``. The
    structured prefix allows callers to record an unsupported run as a skip.
    """

    def __init__(self, reason: str, details: dict | None = None):
        self.reason = reason
        self.details = details or {}
        # Prefix matches TabPFNSkipReason / TabICLSkipReason so analysis
        # scripts can do `df["error"].str.startswith("joint_tsne_skip:")`
        # across all skip-aware models/methods uniformly.
        msg = f"joint_tsne_skip: {reason}"
        if self.details:
            msg = f"{msg} ({', '.join(f'{k}={v}' for k, v in self.details.items())})"
        super().__init__(msg)


class JointTSNE(TestTimeMethod):
    """M3b: Fit t-SNE on the combined train and test feature space.

    Categorical columns are jointly frequency-encoded, numerical columns are
    jointly median-imputed and all distance-space columns are jointly scaled
    before fitting openTSNE. The output preserves label-encoded original
    features and appends ``joint_tsne1`` through
    ``joint_tsne{n_components}``. ``skip_pipeline_scaler=True`` preserves the
    geometry of the joint embedding.

    Args:
        n_components: Number of embedding dimensions (2 or 3).
        perplexity: Requested neighborhood size, clamped for small datasets.
        max_n_total: Skip threshold for ``n_train + n_test``.
        n_jobs: Number of openTSNE worker threads.
        seed: Random state passed to openTSNE. Exact embeddings may still vary
            across processor types or numerical environments.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        max_n_total: int = 200_000,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        if n_components not in (2, 3):
            raise ValueError(
                f"n_components must be 2 or 3, got {n_components}"
            )
        if perplexity <= 0:
            raise ValueError(f"perplexity must be > 0, got {perplexity}")
        if max_n_total < 100:
            raise ValueError(
                f"max_n_total must be >= 100, got {max_n_total}"
            )
        self.n_components = n_components
        self.perplexity = float(perplexity)
        self.max_n_total = int(max_n_total)
        self.n_jobs = int(n_jobs)
        self.seed = int(seed)

    @property
    def name(self) -> str:
        return f"joint_tsne_{self.n_components}d_p{int(self.perplexity)}"

    @property
    def produces_prescaled_output(self) -> bool:
        return True

    def fit_transform(self, X_train, X_test, y_train):
        # Apply the size guard before deferred imports so skipped runs avoid
        # importing openTSNE and related helpers.
        n_train = len(X_train)
        n_test = len(X_test)
        n_total = n_train + n_test
        if n_total > self.max_n_total:
            raise JointTSNESkipReason(
                "n_total_exceeded",
                {
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_total": n_total,
                    "max_n_total": self.max_n_total,
                },
            )

        # Deferred imports keep module-import time short and make reuse of the
        # tier 2 helpers explicit at the call site.
        from openTSNE import TSNE as _OpenTSNE
        from src.methods.tier2 import (
            _frequency_encode_joint,
            _impute_median_joint,
        )

        # --- Step 1: identify cats once (used for both output base and
        # distance space).
        cat_cols = _get_cat_cols(X_train)

        # Step 2: build the t-SNE distance space used by the M19 and M20 pattern.
        # Freq-encode cats joint train+test → joint median impute → joint
        # StandardScaler.  Cats enter the t-SNE fit on a numerically
        # meaningful continuous scale comparable to numerics.
        X_tr_km_raw, X_te_km_raw = _frequency_encode_joint(
            X_train, X_test, cat_cols,
        )
        # Frequency-encoded categorical columns contain no NaN because NaN is
        # encoded as a string. Imputation affects only numerical columns.
        X_tr_km, X_te_km = _impute_median_joint(X_tr_km_raw, X_te_km_raw)
        ds_scaler = StandardScaler()
        ds_scaler.fit(np.vstack([X_tr_km, X_te_km]))
        X_tr_dist = ds_scaler.transform(X_tr_km)
        X_te_dist = ds_scaler.transform(X_te_km)
        combined_dist = np.vstack([X_tr_dist, X_te_dist])

        # Use PCA initialization when the distance space has enough dimensions.
        # Otherwise use random initialization because openTSNE's PCA initializer
        # requires at least ``n_components`` input features.
        n_dist_features = int(combined_dist.shape[1])
        if n_dist_features >= self.n_components:
            init_used = "pca"
        else:
            init_used = "random"

        # --- Step 3: joint t-SNE fit on the combined matrix.  Clamp
        # perplexity to satisfy the openTSNE convention
        # ``perplexity < (n_samples - 1) / 3``.
        eff_perp = min(self.perplexity, max(5.0, (n_total - 1) / 3.0))
        # openTSNE's FFT/interpolation method supports at most two output
        # dimensions, so use Barnes-Hut for higher-dimensional embeddings.
        neg_grad = "bh" if self.n_components > 2 else "auto"
        tsne = _OpenTSNE(
            n_components=self.n_components,
            perplexity=eff_perp,
            random_state=self.seed,
            n_jobs=self.n_jobs,
            initialization=init_used,
            negative_gradient_method=neg_grad,
        )
        embedding_obj = tsne.fit(combined_dist)
        # ``kl_divergence`` lives on the TSNEEmbedding instance (which
        # subclasses np.ndarray). Extract before casting to a plain ndarray.
        kl_div = float(getattr(embedding_obj, "kl_divergence", float("nan")))

        # Record the effective perplexity after openTSNE's internal clamp. Fall
        # back to the outer clamped value if the library API changes.
        try:
            effective_list = list(embedding_obj.affinities.effective_perplexities_)
            actual_perp = float(effective_list[0]) if effective_list else float(eff_perp)
        except (AttributeError, IndexError, TypeError):
            actual_perp = float(eff_perp)

        embedding = np.asarray(embedding_obj)
        tsne_train = embedding[:n_train]
        tsne_test = embedding[n_train:]
        tsne_names = [f"joint_tsne{i+1}" for i in range(self.n_components)]

        # Step 4: build the output matrix from encoded categorical features and joint
        # numerics in the original column order.  Per the M3 docstring
        # rationale: emit joint-scaled numerics so ``skip_pipeline_scaler``
        # preserves the joint coordinate system end-to-end.
        X_tr_enc, X_te_enc, _ = _label_encode_cats(X_train, X_test)
        all_cols = list(X_tr_enc.columns)
        X_tr_vals = X_tr_enc.values.astype(float)
        X_te_vals = X_te_enc.values.astype(float)
        num_cols = [c for c in all_cols if c not in set(cat_cols)]

        X_tr_out_base = X_tr_vals.copy()
        X_te_out_base = X_te_vals.copy()
        if num_cols:
            num_idx = [all_cols.index(c) for c in num_cols]
            # StandardScaler is per-column. The joint-scaled numerics in the
            # distance space match what a numerics-only joint
            # StandardScaler would have produced. Reuse these values instead
            # of fitting another scaler.
            X_tr_out_base[:, num_idx] = X_tr_dist[:, num_idx]
            X_te_out_base[:, num_idx] = X_te_dist[:, num_idx]

        X_tr_out = np.hstack([X_tr_out_base, tsne_train])
        X_te_out = np.hstack([X_te_out_base, tsne_test])
        feature_names = all_cols + tsne_names

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            skip_pipeline_scaler=True,
            categorical_features=cat_cols if cat_cols else None,
            metadata={
                # Three perplexity fields: requested, outer-clamped and the
                # effective value after openTSNE's internal clamp.
                "perplexity": float(eff_perp),
                "perplexity_requested": float(self.perplexity),
                "perplexity_effective": float(actual_perp),
                "n_components": int(self.n_components),
                "n_total": int(n_total),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "kl_divergence": kl_div,
                "n_numeric_cols": len(num_cols),
                "n_categorical_cols": len(cat_cols),
                "n_dist_features": n_dist_features,
                "initialization": init_used,  # "pca" or low-dimensional fallback "random"
                "negative_gradient_method": neg_grad,  # "bh" for 3d, "auto" for 2d
                "n_jobs": int(self.n_jobs),
                "library": "openTSNE",
            },
        )


# ===========================================================================
# M4: Test-Regularized Target Encoding
# ===========================================================================

class ShiftRegularizedTargetEncoding(TestTimeMethod):
    """
    M4: Target encoding with smoothing regulated by test frequency.

    Formula:
        encoding(cat) = (n_cat * mean_y_cat + m * global_mean) / (n_cat + m)

    where m depends on the adjustment factor derived from test frequencies.

    The adjustment_factor depends on `regularization_mode`:
      - "ratio":    freq_test(cat) / freq_train(cat)
      - "absolute": freq_test(cat) / mean_freq_test
      - "diff":     |freq_test(cat) - freq_train(cat)| + 1

    Normal direction:   m = base_m * adj  (more shift → more smoothing)
    Inverted direction: m = base_m / adj  (more shift → less smoothing,
        giving the training signal more weight for test-important categories)

    Args:
        base_m: Base smoothing parameter (default 10).
        regularization_mode: How to compute m from test frequencies.
        adj_cap: Cap on adjustment factor (default None = uncapped).
            Prevents extreme smoothing in either direction.
        invert: If True, invert the adjustment direction:
            m = base_m / adj instead of m = base_m * adj.
    """

    def __init__(
        self,
        base_m: float = 10.0,
        regularization_mode: str = "ratio",
        adj_cap: float | None = None,
        invert: bool = False,
    ):
        assert regularization_mode in ("ratio", "absolute", "diff")
        self.base_m = base_m
        self.regularization_mode = regularization_mode
        self.adj_cap = adj_cap
        self.invert = invert

    @property
    def name(self) -> str:
        mode = self.regularization_mode
        if self.invert:
            mode = f"inv_{mode}"
        base = f"target_enc_{mode}"
        if self.adj_cap is not None:
            base += f"_cap{int(self.adj_cap)}"
        return base

    @property
    def uses_labels(self) -> bool:
        return True

    def fit_transform(self, X_train, X_test, y_train):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        cat_cols = _get_cat_cols(X_train)
        y = np.asarray(y_train).astype(float)
        global_mean = y.mean()

        adj_stats = {}  # per-column diagnostics

        for col in cat_cols:
            train_vals = X_tr[col].astype(str)
            test_vals = X_te[col].astype(str)

            # Compute per-category statistics from training data
            cat_stats = pd.DataFrame({"cat": train_vals.values, "y": y})
            grouped = cat_stats.groupby("cat")["y"]
            cat_mean = grouped.mean().to_dict()
            cat_count = grouped.count().to_dict()

            # Compute test frequencies
            test_freq = test_vals.value_counts(normalize=True).to_dict()
            train_freq = train_vals.value_counts(normalize=True).to_dict()
            mean_test_freq = np.mean(list(test_freq.values())) if test_freq else 1.0

            all_cats = set(train_freq) | set(test_freq)
            encoding_map = {}
            col_adjs = []  # track adj values for diagnostics

            for cat in all_cats:
                n_cat = cat_count.get(cat, 0)
                mean_y_cat = cat_mean.get(cat, global_mean)
                f_train = train_freq.get(cat, 0)
                f_test = test_freq.get(cat, 0)

                # Compute adjustment factor
                if self.regularization_mode == "ratio":
                    adj = (f_test + 1e-6) / (f_train + 1e-6)
                elif self.regularization_mode == "absolute":
                    adj = (f_test + 1e-6) / (mean_test_freq + 1e-6)
                else:  # "diff"
                    adj = abs(f_test - f_train) + 1.0

                # Cap adjustment factor
                if self.adj_cap is not None:
                    adj = min(adj, self.adj_cap)

                col_adjs.append(adj)

                # Compute m: normal or inverted direction
                if self.invert:
                    m = self.base_m / max(adj, 1e-6)
                else:
                    m = self.base_m * adj

                encoding = (n_cat * mean_y_cat + m * global_mean) / (n_cat + m)
                encoding_map[cat] = encoding

            # Per-column diagnostics
            n_capped = 0
            if self.adj_cap is not None and col_adjs:
                n_capped = sum(1 for a in col_adjs if a >= self.adj_cap)
            adj_stats[col] = {
                "mean_adj": float(np.mean(col_adjs)) if col_adjs else 0.0,
                "max_adj": float(np.max(col_adjs)) if col_adjs else 0.0,
                "min_adj": float(np.min(col_adjs)) if col_adjs else 0.0,
                "n_categories": len(col_adjs),
                "n_capped": n_capped,
            }

            col_name = f"{col}_target_enc"
            X_tr[col_name] = train_vals.map(encoding_map).fillna(global_mean).astype(float)
            X_te[col_name] = test_vals.map(encoding_map).fillna(global_mean).astype(float)

        # Preserve original cats as label-encoded integers alongside the
        # target-encoded columns so cat-aware models can use native splits.
        for col in cat_cols:
            le = LabelEncoder()
            combined = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)])
            le.fit(combined)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_te[col] = le.transform(X_te[col].astype(str))

        feature_names = list(X_tr.columns)
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=feature_names,
            categorical_features=cat_cols if cat_cols else None,
            metadata={
                "adj_stats": adj_stats,
                "n_cat_cols": len(cat_cols),
                "adj_cap": self.adj_cap,
                "invert": self.invert,
            },
        )


# ===========================================================================
# M4v2: OOF Target Encoding with Shift Regularization
# ===========================================================================

class ShiftRegularizedTargetEncodingOOF(TestTimeMethod):
    """
    M4v2: Target encoding with OOF to prevent target leakage on training data.

    The original M4 computes mean_y_cat using all
    training samples including sample i itself.  This means the encoding of
    sample i contains information about y_i: a form of target leakage that
    inflates apparent accuracy on training data and can cause overfitting.

    For each fold, out-of-fold encoding performs these steps:
    - Compute category means from samples in the other folds only.
    - Apply the encoding to the held-out fold's training samples.
    - Test data always receives the encoding computed from all training data
      (no leakage since y_test is never used).

    The shift regularization from M4 (test-frequency adjustment of m) is
    preserved: the OOF procedure is orthogonal to it.

    Args:
        base_m: Base smoothing parameter (default 10).
        regularization_mode: How to compute m from test frequencies
            ("ratio", "absolute", "diff").
        n_folds: Number of OOF folds (default 5).
        seed: Random seed for fold splitting.
        adj_cap: Cap on adjustment factor (default None = uncapped).
        invert: If True, invert the adjustment direction:
            m = base_m / adj instead of m = base_m * adj.
    """

    def __init__(
        self,
        base_m: float = 10.0,
        regularization_mode: str = "ratio",
        n_folds: int = 5,
        seed: int = 42,
        adj_cap: float | None = None,
        invert: bool = False,
    ):
        assert regularization_mode in ("ratio", "absolute", "diff")
        self.base_m = base_m
        self.regularization_mode = regularization_mode
        self.n_folds = n_folds
        self.seed = seed
        self.adj_cap = adj_cap
        self.invert = invert

    @property
    def name(self) -> str:
        mode = self.regularization_mode
        if self.invert:
            mode = f"inv_{mode}"
        base = f"target_enc_oof_{mode}"
        if self.n_folds != 5:
            base += f"_{self.n_folds}fold"
        if self.adj_cap is not None:
            base += f"_cap{int(self.adj_cap)}"
        return base

    @property
    def uses_labels(self) -> bool:
        return True

    def _compute_encoding(
        self,
        train_vals: pd.Series,
        y: np.ndarray,
        test_freq: dict,
        train_freq: dict,
        global_mean: float,
    ) -> dict:
        """
        Compute per-category target encoding with shift regularization.

        This is the same formula as M4 but factored out so it can be called
        on different subsets (full train or fold-specific train).

        Returns:
            Tuple of (encoding_map, list_of_adj_values).
        """
        cat_stats = pd.DataFrame({"cat": train_vals.values, "y": y})
        grouped = cat_stats.groupby("cat")["y"]
        cat_mean = grouped.mean().to_dict()
        cat_count = grouped.count().to_dict()

        mean_test_freq = np.mean(list(test_freq.values())) if test_freq else 1.0
        all_cats = set(train_freq) | set(test_freq) | set(cat_mean)
        encoding_map = {}
        adj_values = []

        for cat in all_cats:
            n_cat = cat_count.get(cat, 0)
            mean_y_cat = cat_mean.get(cat, global_mean)
            f_train = train_freq.get(cat, 0)
            f_test = test_freq.get(cat, 0)

            if self.regularization_mode == "ratio":
                adj = (f_test + 1e-6) / (f_train + 1e-6)
            elif self.regularization_mode == "absolute":
                adj = (f_test + 1e-6) / (mean_test_freq + 1e-6)
            else:  # "diff"
                adj = abs(f_test - f_train) + 1.0

            # Cap adjustment factor
            if self.adj_cap is not None:
                adj = min(adj, self.adj_cap)

            adj_values.append(adj)

            # Compute m: normal or inverted direction
            if self.invert:
                m = self.base_m / max(adj, 1e-6)
            else:
                m = self.base_m * adj

            encoding = (n_cat * mean_y_cat + m * global_mean) / (n_cat + m)
            encoding_map[cat] = encoding

        return encoding_map, adj_values

    def fit_transform(self, X_train, X_test, y_train):
        from sklearn.model_selection import KFold

        X_tr = X_train.copy()
        X_te = X_test.copy()
        cat_cols = _get_cat_cols(X_train)
        y = np.asarray(y_train).astype(float)
        global_mean = y.mean()

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        adj_stats = {}  # per-column diagnostics

        for col in cat_cols:
            train_vals = X_tr[col].astype(str)
            test_vals = X_te[col].astype(str)

            # Test frequencies (available for shift regularization)
            test_freq = test_vals.value_counts(normalize=True).to_dict()
            # Full-train frequencies (used for regularization adjustment)
            full_train_freq = train_vals.value_counts(normalize=True).to_dict()

            # --- OOF encoding for training data ---
            oof_encodings = pd.Series(index=X_tr.index, dtype=float)

            for fold_train_idx, fold_val_idx in kf.split(X_tr):
                # Compute the encoding from the other folds only.
                fold_train_vals = train_vals.iloc[fold_train_idx]
                fold_y = y[fold_train_idx]

                # Use fold-specific train frequencies for the encoding,
                # but full test frequencies for the shift regularization
                fold_train_freq = fold_train_vals.value_counts(normalize=True).to_dict()

                encoding_map, _ = self._compute_encoding(
                    fold_train_vals, fold_y,
                    test_freq, fold_train_freq, global_mean,
                )

                # Apply to held-out fold
                fold_val_vals = train_vals.iloc[fold_val_idx]
                oof_encodings.iloc[fold_val_idx] = (
                    fold_val_vals.map(encoding_map).fillna(global_mean).astype(float)
                )

            # --- Full-data encoding for test data ---
            full_encoding_map, col_adjs = self._compute_encoding(
                train_vals, y,
                test_freq, full_train_freq, global_mean,
            )

            # Per-column diagnostics (from full-data encoding)
            n_capped = 0
            if self.adj_cap is not None and col_adjs:
                n_capped = sum(1 for a in col_adjs if a >= self.adj_cap)
            adj_stats[col] = {
                "mean_adj": float(np.mean(col_adjs)) if col_adjs else 0.0,
                "max_adj": float(np.max(col_adjs)) if col_adjs else 0.0,
                "min_adj": float(np.min(col_adjs)) if col_adjs else 0.0,
                "n_categories": len(col_adjs),
                "n_capped": n_capped,
            }

            col_name = f"{col}_target_enc_oof"
            X_tr[col_name] = oof_encodings.values
            X_te[col_name] = test_vals.map(full_encoding_map).fillna(global_mean).astype(float)

        # Preserve original cats as label-encoded integers alongside the OOF
        # target-encoded columns so cat-aware models can use native splits.
        for col in cat_cols:
            le = LabelEncoder()
            combined = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)])
            le.fit(combined)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_te[col] = le.transform(X_te[col].astype(str))

        feature_names = list(X_tr.columns)
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=feature_names,
            categorical_features=cat_cols if cat_cols else None,
            metadata={
                "n_folds": self.n_folds,
                "base_m": self.base_m,
                "adj_stats": adj_stats,
                "n_cat_cols": len(cat_cols),
                "adj_cap": self.adj_cap,
                "invert": self.invert,
            },
        )


# ===========================================================================
# M5: Joint Scaling
# ===========================================================================

class JointScaling(TestTimeMethod):
    """
    M5: StandardScaler fitted on combined train+test data.

    Instead of fitting the scaler on training data only, the method fits it on
    the union of both splits. This produces a more stable
    normalization when feature distributions have shifted.

    Categorical columns are label-encoded before scaling.
    """

    @property
    def name(self) -> str:
        return "joint_scaling"

    @property
    def produces_prescaled_output(self) -> bool:
        return True

    def fit_transform(self, X_train, X_test, y_train):
        X_tr, X_te, cat_cols = _label_encode_cats(X_train, X_test)

        X_tr_vals = X_tr.values.astype(float)
        X_te_vals = X_te.values.astype(float)

        # Fit scaler on combined data
        combined = np.vstack([X_tr_vals, X_te_vals])
        scaler = StandardScaler()
        scaler.fit(combined)

        X_tr_out = scaler.transform(X_tr_vals)
        X_te_out = scaler.transform(X_te_vals)

        feature_names = list(X_tr.columns)
        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            skip_pipeline_scaler=True,  # M5 already applied joint StandardScaler. Pipeline re-scaler would overwrite joint statistics
        )


# ===========================================================================
# M14: Joint Group-By Features
# ===========================================================================

class JointGroupByFeatures(TestTimeMethod):
    """
    M14: Group-by aggregations computed over train+test combined.

    For each (group_col, agg_col, agg_func) triple, compute statistics
    like mean, count, std over the combined train+test data, giving the
    model information about how feature distributions look in the test set.

    Two variants (controlled by `variant`):
      - "combined": single feature = agg(num_col | group) over train+test
      - "separate": two features per triple = agg_train(num_col | group),
                    agg_test(num_col | group): the difference is itself
                    a shift signal the model can learn from.

    Args:
        group_cols: Categorical columns to group by ("auto" = all with <= 100 unique).
        agg_cols: Numerical columns to aggregate ("auto" = all with std > 0).
        agg_funcs: Aggregation functions (default: ["mean", "count", "std"]).
        variant: "combined" or "separate".
        max_unique: Maximum unique values for auto-selected group columns.
    """

    def __init__(
        self,
        group_cols: list[str] | str = "auto",
        agg_cols: list[str] | str = "auto",
        agg_funcs: list[str] | None = None,
        variant: str = "combined",
        max_unique: int = 100,
    ):
        assert variant in ("combined", "separate"), \
            f"variant must be 'combined' or 'separate', got '{variant}'"
        self.group_cols = group_cols
        self.agg_cols = agg_cols
        self.agg_funcs = agg_funcs or ["mean", "count", "std"]
        self.variant = variant
        self.max_unique = max_unique

    @property
    def name(self) -> str:
        if self.agg_funcs != ["mean", "count", "std"]:
            suffix = "_".join(self.agg_funcs)
            return f"joint_groupby_{suffix}_only"
        return f"joint_groupby_{self.variant}"

    def _resolve_columns(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Resolve 'auto' column selections.

        Handles two scenarios:
        1. Raw data: categorical columns (object/category dtype) are used as groups.
        2. Post-Stufe-1 data: categoricals have been label-encoded and possibly
           scaled into floats. In this case, low-cardinality numerical
           columns as group candidates. A numeric column with <= max_unique
           distinct values is treated as a discrete group column: this correctly
           identifies label-encoded categories even after StandardScaler.
        """
        if self.group_cols == "auto":
            # First try: actual categorical columns
            cat_cols = _get_cat_cols(X_train)
            group_cols = [c for c in cat_cols if X_train[c].nunique() <= self.max_unique]

            # Fallback: if no categoricals found, detect low-cardinality numeric
            # columns (label-encoded and possibly scaled categoricals)
            if not group_cols:
                for c in _get_num_cols(X_train):
                    if X_train[c].nunique() <= self.max_unique:
                        group_cols.append(c)
        else:
            group_cols = list(self.group_cols)

        if self.agg_cols == "auto":
            num_cols = _get_num_cols(X_train)
            # Exclude group columns from aggregation targets
            agg_cols = [c for c in num_cols
                        if c not in group_cols and X_train[c].std() > 0]
        else:
            agg_cols = list(self.agg_cols)

        return group_cols, agg_cols

    def fit_transform(self, X_train, X_test, y_train):
        X_tr = X_train.copy()
        X_te = X_test.copy()
        n_train = len(X_tr)
        n_test = len(X_te)

        group_cols, agg_cols = self._resolve_columns(X_train, X_test)

        # If no suitable columns found, fall back to identity-like behavior
        if not group_cols or not agg_cols:
            X_tr_enc, X_te_enc, fallback_cats = _label_encode_cats(X_tr, X_te)
            feature_names = list(X_tr_enc.columns)
            return TransformResult(
                X_train=X_tr_enc.values.astype(float),
                X_test=X_te_enc.values.astype(float),
                feature_names=feature_names,
                metadata={"n_groupby_features": 0},
                categorical_features=fallback_cats if fallback_cats else None,
            )

        # Tag rows with origin for combined aggregation
        combined = pd.concat([X_tr, X_te], ignore_index=True)
        origin = np.array(["train"] * n_train + ["test"] * n_test)
        n_new_features = 0

        # Collect new columns in dicts to avoid DataFrame fragmentation
        # (repeated frame.insert is O(n^2) and triggers PerformanceWarning)
        new_cols_tr: dict[str, np.ndarray] = {}
        new_cols_te: dict[str, np.ndarray] = {}

        for gcol in group_cols:
            group_key = combined[gcol].astype(str)

            for acol in agg_cols:
                agg_values = combined[acol].astype(float)

                for func in self.agg_funcs:
                    if self.variant == "combined":
                        # Aggregate over train+test combined
                        agg_map = (
                            pd.DataFrame({"g": group_key, "v": agg_values})
                            .groupby("g")["v"]
                            .agg(func)
                            .to_dict()
                        )
                        col_name = f"{gcol}_{acol}_{func}_joint"
                        new_cols_tr[col_name] = X_tr[gcol].astype(str).map(agg_map).astype(float).values
                        new_cols_te[col_name] = X_te[gcol].astype(str).map(agg_map).astype(float).values
                        n_new_features += 1
                    else:
                        # Separate aggregations for train and test
                        df_agg = pd.DataFrame({
                            "g": group_key, "v": agg_values, "origin": origin
                        })
                        train_agg = (
                            df_agg[df_agg["origin"] == "train"]
                            .groupby("g")["v"].agg(func).to_dict()
                        )
                        test_agg = (
                            df_agg[df_agg["origin"] == "test"]
                            .groupby("g")["v"].agg(func).to_dict()
                        )

                        col_tr_name = f"{gcol}_{acol}_{func}_train"
                        col_te_name = f"{gcol}_{acol}_{func}_test"
                        global_fallback = agg_values.agg(func)

                        new_cols_tr[col_tr_name] = (
                            X_tr[gcol].astype(str).map(train_agg)
                            .fillna(global_fallback).astype(float).values
                        )
                        new_cols_te[col_tr_name] = (
                            X_te[gcol].astype(str).map(train_agg)
                            .fillna(global_fallback).astype(float).values
                        )
                        new_cols_tr[col_te_name] = (
                            X_tr[gcol].astype(str).map(test_agg)
                            .fillna(global_fallback).astype(float).values
                        )
                        new_cols_te[col_te_name] = (
                            X_te[gcol].astype(str).map(test_agg)
                            .fillna(global_fallback).astype(float).values
                        )
                        n_new_features += 2

        # Batch-append all new columns at once (avoids fragmentation)
        if new_cols_tr:
            X_tr = pd.concat([X_tr, pd.DataFrame(new_cols_tr, index=X_tr.index)], axis=1)
            X_te = pd.concat([X_te, pd.DataFrame(new_cols_te, index=X_te.index)], axis=1)

        # Preserve all original categorical columns as label-encoded
        # integers. Group cols stay alongside the per-group aggregation
        # features so cat-aware models can split on them natively.
        orig_cat_cols = _get_cat_cols(X_train)
        for c in orig_cat_cols:
            if c in X_tr.columns:
                le = LabelEncoder()
                combined_vals = pd.concat([X_tr[c].astype(str), X_te[c].astype(str)])
                le.fit(combined_vals)
                X_tr[c] = le.transform(X_tr[c].astype(str))
                X_te[c] = le.transform(X_te[c].astype(str))

        # Fill any remaining NaN from aggregation (e.g. std of single-element group)
        X_tr = X_tr.fillna(0)
        X_te = X_te.fillna(0)

        feature_names = list(X_tr.columns)
        remaining_cats = [c for c in orig_cat_cols if c in X_tr.columns]
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=feature_names,
            metadata={"n_groupby_features": n_new_features},
            categorical_features=remaining_cats if remaining_cats else None,
        )


# ===========================================================================
# M14b: Group-By Shift Ratio
# ===========================================================================

class GroupByShiftRatio(TestTimeMethod):
    """
    M14b: For each (group_col, num_col) pair, compute the ratio
    mean_test(num_col | group) / mean_train(num_col | group).

    This directly captures per-group shift magnitude: a ratio of 1.0
    means no shift for that group, while large deviations indicate
    the group's feature distribution changed between train and test.

    Optionally log-transformed for symmetry (log(1) = 0 means no shift).

    Args:
        group_cols: Categorical columns to group by ("auto" = all with <= 100 unique).
        agg_cols: Numerical columns to compute ratios for ("auto" = all with std > 0).
        use_log: Log-transform the ratio (default True).
        alpha: Smoothing constant to avoid division by zero.
        max_unique: Maximum unique values for auto-selected group columns.
    """

    def __init__(
        self,
        group_cols: list[str] | str = "auto",
        agg_cols: list[str] | str = "auto",
        use_log: bool = True,
        alpha: float = 1e-6,
        max_unique: int = 100,
    ):
        self.group_cols = group_cols
        self.agg_cols = agg_cols
        self.use_log = use_log
        self.alpha = alpha
        self.max_unique = max_unique

    @property
    def name(self) -> str:
        log_str = "log" if self.use_log else "raw"
        return f"groupby_shift_ratio_{log_str}"

    def _resolve_columns(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        """
        Resolve column selections. Same logic as JointGroupByFeatures:
        detects low-cardinality numeric columns when no categoricals are present.
        """
        if self.group_cols == "auto":
            cat_cols = _get_cat_cols(X_train)
            group_cols = [c for c in cat_cols if X_train[c].nunique() <= self.max_unique]

            # Fallback: detect low-cardinality numeric columns
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
        X_tr = X_train.copy()
        X_te = X_test.copy()

        group_cols, agg_cols = self._resolve_columns(X_train, X_test)

        if not group_cols or not agg_cols:
            X_tr_enc, X_te_enc, fallback_cats = _label_encode_cats(X_tr, X_te)
            feature_names = list(X_tr_enc.columns)
            return TransformResult(
                X_train=X_tr_enc.values.astype(float),
                X_test=X_te_enc.values.astype(float),
                feature_names=feature_names,
                metadata={"n_shift_ratio_features": 0},
                categorical_features=fallback_cats if fallback_cats else None,
            )

        # Collect new columns in dicts to avoid DataFrame fragmentation
        new_cols_tr: dict[str, np.ndarray] = {}
        new_cols_te: dict[str, np.ndarray] = {}
        n_new_features = 0

        for gcol in group_cols:
            for acol in agg_cols:
                # Compute group means separately
                train_mean = (
                    X_tr.groupby(X_tr[gcol].astype(str))[acol]
                    .mean().to_dict()
                )
                test_mean = (
                    X_te.groupby(X_te[gcol].astype(str))[acol]
                    .mean().to_dict()
                )

                # Global fallback
                global_train_mean = X_tr[acol].astype(float).mean()
                global_test_mean = X_te[acol].astype(float).mean()

                all_groups = set(train_mean) | set(test_mean)
                ratio_map = {}
                for g in all_groups:
                    m_train = train_mean.get(g, global_train_mean) + self.alpha
                    m_test = test_mean.get(g, global_test_mean) + self.alpha
                    ratio = m_test / m_train
                    if self.use_log:
                        # Avoid log of negative (possible with shifted means)
                        ratio = np.log(max(ratio, 1e-10))
                    ratio_map[g] = ratio

                col_name = f"{gcol}_{acol}_shift_ratio"
                new_cols_tr[col_name] = X_tr[gcol].astype(str).map(ratio_map).fillna(0).astype(float).values
                new_cols_te[col_name] = X_te[gcol].astype(str).map(ratio_map).fillna(0).astype(float).values
                n_new_features += 1

        # Batch-append all new columns at once (avoids fragmentation)
        if new_cols_tr:
            X_tr = pd.concat([X_tr, pd.DataFrame(new_cols_tr, index=X_tr.index)], axis=1)
            X_te = pd.concat([X_te, pd.DataFrame(new_cols_te, index=X_te.index)], axis=1)

        # Preserve all original categorical columns as label-encoded
        # integers alongside the shift-ratio features.
        orig_cat_cols = _get_cat_cols(X_train)
        for c in orig_cat_cols:
            if c in X_tr.columns:
                le = LabelEncoder()
                combined_vals = pd.concat([X_tr[c].astype(str), X_te[c].astype(str)])
                le.fit(combined_vals)
                X_tr[c] = le.transform(X_tr[c].astype(str))
                X_te[c] = le.transform(X_te[c].astype(str))

        # Handle any NaN/Inf
        X_tr = X_tr.replace([np.inf, -np.inf], 0).fillna(0)
        X_te = X_te.replace([np.inf, -np.inf], 0).fillna(0)

        feature_names = list(X_tr.columns)
        remaining_cats = [c for c in orig_cat_cols if c in X_tr.columns]
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=feature_names,
            metadata={"n_shift_ratio_features": n_new_features},
            categorical_features=remaining_cats if remaining_cats else None,
        )


# ===========================================================================
# M14bv2: GroupBy Shift with Cohen's d
# ===========================================================================


class GroupByShiftCohenD(TestTimeMethod):
    """
    M14bv2: Group-by shift features using Cohen's d instead of mean ratio.

    The original ``GroupByShiftRatio`` computes
    ``mean_test / mean_train`` per (group_col, agg_col) pair.  This ratio:

    - Is undefined / numerically unstable when the train group mean is near
      zero (the denominator collapses).
    - Is scale-dependent: a ratio of 2.0 could mean (0.001 → 0.002) or
      (100 → 200): very different practical magnitudes.
    - Is non-symmetric: shrinkage and growth are not on the same scale
      (0.5 vs 2.0 for a doubling).

    This version replaces the ratio with **Cohen's d**, a standardized effect
    size defined as ``(mean_test - mean_train) / pooled_std``.  Cohen's d:

    - Is well-defined when means are near zero (only the *difference*
      matters, not the ratio).
    - Is scale-free: d = 0.5 always means "half a standard deviation of
      shift" regardless of the original units.
    - Is symmetric: positive and negative shifts are on the same scale.

    The feature is set to 0 when the pooled standard deviation is zero.
    """

    def __init__(
        self,
        group_cols: list[str] | str = "auto",
        agg_cols: list[str] | str = "auto",
        max_unique: int = 100,
        min_group_size: int = 5,
    ):
        self.group_cols = group_cols
        self.agg_cols = agg_cols
        self.max_unique = max_unique
        self.min_group_size = min_group_size

    @property
    def name(self) -> str:
        base = "groupby_shift_cohen_d"
        if self.min_group_size != 5:
            return f"{base}_min{self.min_group_size}"
        return base

    def _resolve_columns(self, X_train: pd.DataFrame, X_test: pd.DataFrame):
        if self.group_cols == "auto":
            cat_cols = _get_cat_cols(X_train)
            group_cols = [c for c in cat_cols
                          if X_train[c].nunique() <= self.max_unique]
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

    @staticmethod
    def _cohen_d(
        group_train: np.ndarray,
        group_test: np.ndarray,
        min_n: int = 5,
    ) -> float:
        """Compute Cohen's d between two samples.

        Returns 0 when either sample is too small or the pooled std is zero.
        """
        n1, n2 = len(group_train), len(group_test)
        if n1 < min_n or n2 < min_n:
            return 0.0
        m1, m2 = group_train.mean(), group_test.mean()
        s1, s2 = group_train.var(ddof=1), group_test.var(ddof=1)
        pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)
        if pooled_std < 1e-12:
            return 0.0
        return (m2 - m1) / pooled_std

    def fit_transform(self, X_train, X_test, y_train):
        X_tr = X_train.copy()
        X_te = X_test.copy()

        group_cols, agg_cols = self._resolve_columns(X_train, X_test)

        if not group_cols or not agg_cols:
            X_tr_enc, X_te_enc, fallback_cats = _label_encode_cats(X_tr, X_te)
            return TransformResult(
                X_train=X_tr_enc.values.astype(float),
                X_test=X_te_enc.values.astype(float),
                feature_names=list(X_tr_enc.columns),
                metadata={"n_cohen_d_features": 0},
                categorical_features=fallback_cats if fallback_cats else None,
            )

        # Collect new columns in dicts to avoid DataFrame fragmentation
        new_cols_tr: dict[str, np.ndarray] = {}
        new_cols_te: dict[str, np.ndarray] = {}
        n_new = 0
        for gcol in group_cols:
            g_tr = X_tr[gcol].astype(str)
            g_te = X_te[gcol].astype(str)
            for acol in agg_cols:
                a_tr = X_tr[acol].astype(float)
                a_te = X_te[acol].astype(float)

                all_groups = set(g_tr.unique()) | set(g_te.unique())
                d_map = {}
                for g in all_groups:
                    vals_tr = a_tr[g_tr == g].dropna().values
                    vals_te = a_te[g_te == g].dropna().values
                    d_map[g] = self._cohen_d(
                        vals_tr, vals_te, min_n=self.min_group_size,
                    )

                col_name = f"{gcol}_{acol}_shift_d"
                new_cols_tr[col_name] = g_tr.map(d_map).fillna(0).astype(float).values
                new_cols_te[col_name] = g_te.map(d_map).fillna(0).astype(float).values
                n_new += 1

        # Batch-append all new columns at once (avoids fragmentation)
        if new_cols_tr:
            X_tr = pd.concat([X_tr, pd.DataFrame(new_cols_tr, index=X_tr.index)], axis=1)
            X_te = pd.concat([X_te, pd.DataFrame(new_cols_te, index=X_te.index)], axis=1)

        # Preserve all original categorical columns as label-encoded
        # integers alongside the Cohen's d features.
        orig_cat_cols = _get_cat_cols(X_train)
        for c in orig_cat_cols:
            if c in X_tr.columns:
                le = LabelEncoder()
                combined_vals = pd.concat([X_tr[c].astype(str), X_te[c].astype(str)])
                le.fit(combined_vals)
                X_tr[c] = le.transform(X_tr[c].astype(str))
                X_te[c] = le.transform(X_te[c].astype(str))

        X_tr = X_tr.replace([np.inf, -np.inf], 0).fillna(0)
        X_te = X_te.replace([np.inf, -np.inf], 0).fillna(0)

        remaining_cats = [c for c in orig_cat_cols if c in X_tr.columns]
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=list(X_tr.columns),
            metadata={"n_cohen_d_features": n_new},
            categorical_features=remaining_cats if remaining_cats else None,
        )


# ===========================================================================
# M14v2: Targeted Group-By Features
# ===========================================================================

class TargetedGroupByFeatures(TestTimeMethod):
    """
    M14v2: Group-by features computed only for the most shifted columns.

    The original M14 creates group-by features for
    all (group_col × agg_col × agg_func) combinations, which can produce
    hundreds of features for wide datasets.  This dilutes signal in tree
    splits.

    This method selects group and aggregation columns separately:

    1. Ranks *categorical* columns by chi-squared shift score.
    2. Only uses the top-k most shifted categoricals as group columns.
    3. Optionally filters *numerical* aggregation targets to those with
       significant KS shift, so only features with detected shift are aggregated.
    4. Caps the number of new features at max_features.

    Selection applies directly to group columns instead of filtering
    all features uniformly.

    Args:
        variant: "combined" or "separate" (same as M14).
        top_k_groups: Max number of categorical group columns to use
            (default 3).  Selected by highest chi-squared statistic.
        filter_agg_cols: If True, also filter numerical aggregation
            targets to those with significant KS shift (default False).
        ks_threshold: p-value threshold for filtering agg columns (0.1).
        agg_funcs: Aggregation functions (default: ["mean", "count"]).
            Reduced from M14's default of ["mean", "count", "std"] to
            further limit feature count.
        max_features: Hard cap on total new features (default 30).
        max_unique: Max unique values per group column (default 100).
    """

    def __init__(
        self,
        variant: str = "combined",
        top_k_groups: int = 3,
        filter_agg_cols: bool = False,
        ks_threshold: float = 0.1,
        agg_funcs: list[str] | None = None,
        max_features: int = 30,
        max_unique: int = 100,
    ):
        assert variant in ("combined", "separate")
        self.variant = variant
        self.top_k_groups = top_k_groups
        self.filter_agg_cols = filter_agg_cols
        self.ks_threshold = ks_threshold
        self.agg_funcs = agg_funcs or ["mean", "count"]
        self.max_features = max_features
        self.max_unique = max_unique

    @property
    def name(self) -> str:
        parts = ["targeted_groupby"]
        if self.variant != "combined":
            parts.append(self.variant)
        parts.append(f"top{self.top_k_groups}")
        if self.filter_agg_cols:
            parts.append("shifted_agg")
        return "_".join(parts)

    def _rank_cat_cols_by_shift(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> list[tuple[str, float]]:
        """Rank categorical columns by chi-squared shift, descending."""
        from scipy.stats import chi2_contingency

        cat_cols = _get_cat_cols(X_train)
        # Also consider low-cardinality numeric columns as potential group cols
        num_cols = _get_num_cols(X_train)
        low_card_num = [
            c for c in num_cols
            if X_train[c].nunique() <= self.max_unique
        ]
        candidate_cols = [
            c for c in (cat_cols + low_card_num)
            if X_train[c].nunique() <= self.max_unique
        ]

        scored = []
        for col in candidate_cols:
            tr_vals = X_train[col].astype(str)
            te_vals = X_test[col].astype(str)
            all_cats = sorted(set(tr_vals.unique()) | set(te_vals.unique()))
            if len(all_cats) <= 1:
                scored.append((col, 0.0))
                continue

            tr_counts = tr_vals.value_counts()
            te_counts = te_vals.value_counts()
            observed = np.array([
                [tr_counts.get(c, 0) for c in all_cats],
                [te_counts.get(c, 0) for c in all_cats],
            ])
            col_mask = observed.sum(axis=0) > 0
            observed = observed[:, col_mask]
            if observed.shape[1] <= 1:
                scored.append((col, 0.0))
                continue

            try:
                stat, _, _, _ = chi2_contingency(observed)
                scored.append((col, float(stat)))
            except Exception:
                scored.append((col, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _filter_num_cols_by_shift(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, group_cols: list[str]
    ) -> list[str]:
        """Filter numerical columns to those with significant KS shift."""
        from scipy.stats import ks_2samp

        num_cols = _get_num_cols(X_train)
        num_cols = [c for c in num_cols if c not in group_cols]

        if not self.filter_agg_cols:
            # No filtering: return all non-zero-variance numeric columns
            return [c for c in num_cols if X_train[c].std() > 1e-10]

        selected = []
        for col in num_cols:
            if X_train[col].std() < 1e-10:
                continue
            tr_vals = X_train[col].dropna().values
            te_vals = X_test[col].dropna().values
            if len(tr_vals) < 2 or len(te_vals) < 2:
                continue
            _, pval = ks_2samp(tr_vals, te_vals)
            if pval < self.ks_threshold:
                selected.append(col)

        # Fallback: if no columns pass, take all
        if not selected:
            selected = [c for c in num_cols if X_train[c].std() > 1e-10]

        return selected

    def fit_transform(self, X_train, X_test, y_train):
        X_tr = X_train.copy()
        X_te = X_test.copy()

        # --- Select top-k shifted group columns ---
        ranked = self._rank_cat_cols_by_shift(X_train, X_test)
        group_cols = [col for col, _ in ranked[: self.top_k_groups]]
        group_scores = {col: score for col, score in ranked[: self.top_k_groups]}

        if not group_cols:
            # No categorical columns → pass through with label encoding
            X_tr, X_te, fallback_cats = _label_encode_cats(X_tr, X_te)
            feature_names = list(X_tr.columns)
            return TransformResult(
                X_train=X_tr.values.astype(float),
                X_test=X_te.values.astype(float),
                feature_names=feature_names,
                metadata={"n_group_cols": 0, "n_new_features": 0},
                categorical_features=fallback_cats if fallback_cats else None,
            )

        # --- Select aggregation columns ---
        agg_cols = self._filter_num_cols_by_shift(X_train, X_test, group_cols)

        if not agg_cols:
            X_tr, X_te, fallback_cats = _label_encode_cats(X_tr, X_te)
            feature_names = list(X_tr.columns)
            return TransformResult(
                X_train=X_tr.values.astype(float),
                X_test=X_te.values.astype(float),
                feature_names=feature_names,
                metadata={"n_group_cols": len(group_cols), "n_agg_cols": 0, "n_new_features": 0},
                categorical_features=fallback_cats if fallback_cats else None,
            )

        # --- Compute group-by features ---
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

                if n_new >= self.max_features:
                    break

                for func in self.agg_funcs:
                    if n_new >= self.max_features:
                        break

                    if self.variant == "combined":
                        combined = pd.concat([X_tr, X_te])
                        agg_map = (
                            combined.groupby(combined[gcol].astype(str))[acol]
                            .agg(func).to_dict()
                        )
                        global_val = combined[acol].astype(float).agg(func)
                        col_name = f"{gcol}_{acol}_{func}_joint"
                        new_cols_tr[col_name] = X_tr[gcol].astype(str).map(agg_map).fillna(global_val).astype(float).values
                        new_cols_te[col_name] = X_te[gcol].astype(str).map(agg_map).fillna(global_val).astype(float).values
                        n_new += 1
                    else:  # "separate"
                        train_agg = X_tr.groupby(X_tr[gcol].astype(str))[acol].agg(func).to_dict()
                        test_agg = X_te.groupby(X_te[gcol].astype(str))[acol].agg(func).to_dict()
                        global_tr = X_tr[acol].astype(float).agg(func)
                        global_te = X_te[acol].astype(float).agg(func)

                        col_tr = f"{gcol}_{acol}_{func}_train"
                        col_te = f"{gcol}_{acol}_{func}_test"
                        new_cols_tr[col_tr] = X_tr[gcol].astype(str).map(train_agg).fillna(global_tr).astype(float).values
                        new_cols_te[col_tr] = X_te[gcol].astype(str).map(train_agg).fillna(global_tr).astype(float).values
                        new_cols_tr[col_te] = X_tr[gcol].astype(str).map(test_agg).fillna(global_te).astype(float).values
                        new_cols_te[col_te] = X_te[gcol].astype(str).map(test_agg).fillna(global_te).astype(float).values
                        n_new += 2

        # Batch-append all new columns at once (avoids fragmentation)
        if new_cols_tr:
            X_tr = pd.concat([X_tr, pd.DataFrame(new_cols_tr, index=X_tr.index)], axis=1)
            X_te = pd.concat([X_te, pd.DataFrame(new_cols_te, index=X_te.index)], axis=1)

        # Preserve all original categorical columns as label-encoded
        # integers alongside the targeted aggregation features.
        orig_cat_cols = _get_cat_cols(X_train)
        for c in orig_cat_cols:
            if c in X_tr.columns:
                le = LabelEncoder()
                combined_vals = pd.concat([X_tr[c].astype(str), X_te[c].astype(str)])
                le.fit(combined_vals)
                X_tr[c] = le.transform(X_tr[c].astype(str))
                X_te[c] = le.transform(X_te[c].astype(str))

        X_tr = X_tr.replace([np.inf, -np.inf], 0).fillna(0)
        X_te = X_te.replace([np.inf, -np.inf], 0).fillna(0)

        feature_names = list(X_tr.columns)
        remaining_cats = [c for c in orig_cat_cols if c in X_tr.columns]
        return TransformResult(
            X_train=X_tr.values.astype(float),
            X_test=X_te.values.astype(float),
            feature_names=feature_names,
            metadata={
                "n_group_cols": len(group_cols),
                "n_agg_cols": len(agg_cols),
                "n_new_features": n_new,
                "group_col_scores": group_scores,
            },
            categorical_features=remaining_cats if remaining_cats else None,
        )


# ===========================================================================
# Registry
# ===========================================================================

TIER1_METHODS = {
    "joint_freq_combined": lambda: JointFrequencyEncoding(variant="combined"),
    "joint_freq_separate": lambda: JointFrequencyEncoding(variant="separate"),
    "freq_ratio_log": lambda: FrequencyRatio(use_log=True),
    "joint_pca_5": lambda: JointPCA(n_components=5),
    "joint_pca_10": lambda: JointPCA(n_components=10),
    "joint_pca_var90": lambda: JointPCA(n_components=0.9),
    "joint_pca_var95": lambda: JointPCA(n_components=0.95),
    "target_enc_ratio": lambda: ShiftRegularizedTargetEncoding(regularization_mode="ratio"),
    "target_enc_absolute": lambda: ShiftRegularizedTargetEncoding(regularization_mode="absolute"),
    "target_enc_diff": lambda: ShiftRegularizedTargetEncoding(regularization_mode="diff"),
    "target_enc_ratio_cap5": lambda: ShiftRegularizedTargetEncoding(
        regularization_mode="ratio", adj_cap=5.0,
    ),
    "target_enc_inv_ratio": lambda: ShiftRegularizedTargetEncoding(
        regularization_mode="ratio", invert=True,
    ),
    "joint_scaling": lambda: JointScaling(),
    "joint_groupby_combined": lambda: JointGroupByFeatures(variant="combined"),
    "joint_groupby_separate": lambda: JointGroupByFeatures(variant="separate"),
    "joint_groupby_mean_only": lambda: JointGroupByFeatures(variant="combined", agg_funcs=["mean"]),
    "groupby_shift_ratio_log": lambda: GroupByShiftRatio(use_log=True),
    "groupby_shift_ratio_raw": lambda: GroupByShiftRatio(use_log=False),
    # --- M2v2: Improved Frequency Ratio ---
    "freq_ratio_v2_ratio_gated": lambda: FrequencyRatioImproved(
        mode="ratio", adaptive_alpha=True, noise_gate=True
    ),
    "freq_ratio_v2_diff_gated": lambda: FrequencyRatioImproved(
        mode="diff", adaptive_alpha=True, noise_gate=True
    ),
    "freq_ratio_v2_both_gated": lambda: FrequencyRatioImproved(
        mode="both", adaptive_alpha=True, noise_gate=True
    ),
    "freq_ratio_v2_ratio_ungated": lambda: FrequencyRatioImproved(
        mode="ratio", adaptive_alpha=True, noise_gate=False
    ),
    # --- M4v2: OOF Target Encoding ---
    "target_enc_oof_ratio": lambda: ShiftRegularizedTargetEncodingOOF(
        regularization_mode="ratio", n_folds=5,
    ),
    "target_enc_oof_diff": lambda: ShiftRegularizedTargetEncodingOOF(
        regularization_mode="diff", n_folds=5,
    ),
    "target_enc_oof_absolute": lambda: ShiftRegularizedTargetEncodingOOF(
        regularization_mode="absolute", n_folds=5,
    ),
    "target_enc_oof_ratio_cap5": lambda: ShiftRegularizedTargetEncodingOOF(
        regularization_mode="ratio", n_folds=5, adj_cap=5.0,
    ),
    "target_enc_oof_inv_ratio": lambda: ShiftRegularizedTargetEncodingOOF(
        regularization_mode="ratio", n_folds=5, invert=True,
    ),
    # --- M14v2: Targeted Group-By ---
    "targeted_groupby_top3": lambda: TargetedGroupByFeatures(
        top_k_groups=3, max_features=30,
    ),
    "targeted_groupby_top5": lambda: TargetedGroupByFeatures(
        top_k_groups=5, max_features=30,
    ),
    "targeted_groupby_top3_shifted_agg": lambda: TargetedGroupByFeatures(
        top_k_groups=3, max_features=30, filter_agg_cols=True,
    ),
    "targeted_groupby_top5_shifted_agg": lambda: TargetedGroupByFeatures(
        top_k_groups=5, max_features=30, filter_agg_cols=True,
    ),
    # --- M14bv2: GroupBy Shift Cohen's d ---
    "groupby_shift_cohen_d": lambda: GroupByShiftCohenD(),
    "groupby_shift_cohen_d_min10": lambda: GroupByShiftCohenD(min_group_size=10),
    # --- M3b: Joint t-SNE ---
    "joint_tsne_2d_p30": lambda: JointTSNE(n_components=2, perplexity=30),
    "joint_tsne_2d_p50": lambda: JointTSNE(n_components=2, perplexity=50),
    "joint_tsne_3d_p30": lambda: JointTSNE(n_components=3, perplexity=30),
}


# ---------------------------------------------------------------------------
# Selective variants: apply methods only to columns selected by a strategy
# ---------------------------------------------------------------------------
def _build_selective_methods() -> dict[str, callable]:
    """
    Build selective versions of methods defined in this module.

    CombinedSelector is the default selection strategy. Selected columns
    must be shifted according to KS or Cramér's V and predictively important
    according to LightGBM feature importance.

    Two variants per method:
      - intersection: columns must rank highly on shift and importance
      - union: columns can rank highly on either dimension

    Naming convention: {method}__sel_{selector}
    E.g. "joint_freq_combined__sel_combined_intersection"
    """
    from src.methods.selective import SelectiveMethod
    from src.methods.column_selection import CombinedSelector

    # Which methods benefit most from column selection
    base_methods = {
        "joint_freq_combined": lambda: JointFrequencyEncoding(variant="combined"),
        "joint_freq_separate": lambda: JointFrequencyEncoding(variant="separate"),
        "freq_ratio_log": lambda: FrequencyRatio(use_log=True),
        "target_enc_ratio": lambda: ShiftRegularizedTargetEncoding(regularization_mode="ratio"),
        "joint_groupby_combined": lambda: JointGroupByFeatures(variant="combined"),
        "groupby_shift_cohen_d": lambda: GroupByShiftCohenD(),
    }

    # CombinedSelector with two modes: strict (intersection) and permissive (union)
    selectors = {
        "combined_intersection": lambda: CombinedSelector(combination="intersection"),
        "combined_union": lambda: CombinedSelector(combination="union"),
    }

    registry = {}
    for method_name, method_factory in base_methods.items():
        for sel_name, sel_factory in selectors.items():
            key = f"{method_name}__sel_{sel_name}"
            # Use default argument binding to avoid late-binding closure issue
            registry[key] = (
                lambda mf=method_factory, sf=sel_factory: SelectiveMethod(mf(), sf())
            )

    return registry


SELECTIVE_METHODS = _build_selective_methods()


def get_tier1_method(name: str) -> TestTimeMethod:
    """Instantiate a Tier 1 method by registry name (includes selective variants)."""
    all_methods = {**TIER1_METHODS, **SELECTIVE_METHODS}
    if name not in all_methods:
        raise ValueError(
            f"Unknown Tier 1 method '{name}'. "
            f"Available: {sorted(all_methods.keys())}"
        )
    return all_methods[name]()


def get_all_tier1_methods() -> dict[str, TestTimeMethod]:
    """Instantiate all Tier 1 methods (base + selective variants)."""
    all_methods = {**TIER1_METHODS, **SELECTIVE_METHODS}
    return {name: factory() for name, factory in all_methods.items()}

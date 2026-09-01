"""
Column Selection Strategies for Targeted Feature Engineering.

These selectors identify columns before applying transformations. This can
reduce irrelevant transformed features.

Strategies:
    ShiftColumnSelector: per-column shift tests (KS for numeric, χ² for categorical)
    DomainClassifierSelector: train/test domain classifier feature importances
    CombinedSelector: intersection of shift and importance
    TopKSelector: simple top-k by any scoring function

Usage:
    selector = ShiftColumnSelector(top_k=10)
    selected = selector.select(X_train, X_test, y_train)
    # selected.cat_cols, selected.num_cols: columns to engineer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class ColumnSelection:
    """Result of a column selection strategy."""

    cat_cols: list[str]  # Selected categorical columns
    num_cols: list[str]  # Selected numerical columns
    scores: dict[str, float] = field(default_factory=dict)  # col → score (for logging)

    @property
    def all_cols(self) -> list[str]:
        return self.cat_cols + self.num_cols

    @property
    def n_selected(self) -> int:
        return len(self.cat_cols) + len(self.num_cols)


class ColumnSelector(ABC):
    """Base class for column selection strategies."""

    @property
    def method_context(self):
        """Pipeline-bound MethodContext, if this selector is wrapped."""
        return getattr(self, "_method_context", None)

    @abstractmethod
    def select(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
    ) -> ColumnSelection:
        """
        Select columns worth engineering.

        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training labels.

        Returns:
            ColumnSelection with the chosen columns and their scores.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class AllColumnsSelector(ColumnSelector):
    """Dummy selector that returns all columns (no filtering)."""

    @property
    def name(self) -> str:
        return "all_columns"

    def select(self, X_train, X_test, y_train):
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        return ColumnSelection(cat_cols=cat_cols, num_cols=num_cols)


# ===========================================================================
# Strategy 1: Per-Column Shift Detection (KS + Chi-Squared)
# ===========================================================================

class ShiftColumnSelector(ColumnSelector):
    """
    Select columns with the largest distribution shift between train and test.

    Uses KS-test for numerical columns and chi-squared test for categorical
    columns. Columns are ranked by shift magnitude (1 - p_value) and the
    top-k or those exceeding a significance threshold are returned.

    The selector focuses on columns with detected distribution shift.

    Args:
        top_k: Select the top-k most shifted columns. If None, use p_value_threshold.
        top_fraction: Select the top fraction (e.g. 0.3 = top 30%). Overridden by top_k.
        p_value_threshold: Only select columns with p < threshold (default 0.05).
            Used as fallback when neither top_k nor top_fraction is set.
        min_cols: Minimum number of columns to always select (default 1).
    """

    def __init__(
        self,
        top_k: int | None = None,
        top_fraction: float | None = None,
        p_value_threshold: float = 0.05,
        min_cols: int = 1,
    ):
        self.top_k = top_k
        self.top_fraction = top_fraction
        self.p_value_threshold = p_value_threshold
        self.min_cols = min_cols

    @property
    def name(self) -> str:
        if self.top_k is not None:
            return f"shift_selector_top{self.top_k}"
        if self.top_fraction is not None:
            return f"shift_selector_frac{self.top_fraction}"
        return f"shift_selector_p{self.p_value_threshold}"

    def _score_numeric(self, train_col: pd.Series, test_col: pd.Series) -> float:
        """KS-test statistic for a numeric column. Higher = more shift."""
        train_vals = train_col.dropna().values.astype(float)
        test_vals = test_col.dropna().values.astype(float)
        if len(train_vals) < 2 or len(test_vals) < 2:
            return 0.0
        stat, p_value = stats.ks_2samp(train_vals, test_vals)
        return stat  # KS statistic itself is a good shift measure

    def _score_categorical(self, train_col: pd.Series, test_col: pd.Series) -> float:
        """Chi-squared based shift score for a categorical column."""
        train_vals = train_col.astype(str)
        test_vals = test_col.astype(str)

        # Build frequency tables aligned to the same categories
        all_cats = sorted(set(train_vals.unique()) | set(test_vals.unique()))
        if len(all_cats) <= 1:
            return 0.0

        train_counts = train_vals.value_counts()
        test_counts = test_vals.value_counts()

        # Align to same categories
        observed_train = np.array([train_counts.get(c, 0) for c in all_cats], dtype=float)
        observed_test = np.array([test_counts.get(c, 0) for c in all_cats], dtype=float)

        # Normalize to proportions, then use symmetric chi-squared
        n_train = observed_train.sum()
        n_test = observed_test.sum()
        if n_train == 0 or n_test == 0:
            return 0.0

        # Expected under null: pooled proportions
        pooled = (observed_train + observed_test) / (n_train + n_test)
        expected_train = pooled * n_train
        expected_test = pooled * n_test

        # Avoid division by zero
        mask = expected_train > 0
        if mask.sum() < 2:
            return 0.0

        chi2_train = np.sum((observed_train[mask] - expected_train[mask]) ** 2 / expected_train[mask])
        chi2_test = np.sum((observed_test[mask] - expected_test[mask]) ** 2 / expected_test[mask])
        chi2_total = chi2_train + chi2_test

        # Normalize by degrees of freedom for comparability across different cardinalities
        df = mask.sum() - 1
        return chi2_total / max(df, 1)

    def select(self, X_train, X_test, y_train):
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

        # Score all columns
        scores = {}
        for col in num_cols:
            scores[col] = self._score_numeric(X_train[col], X_test[col])
        for col in cat_cols:
            scores[col] = self._score_categorical(X_train[col], X_test[col])

        # Determine how many to select
        all_cols_sorted = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        n_total = len(all_cols_sorted)

        if self.top_k is not None:
            n_select = min(self.top_k, n_total)
        elif self.top_fraction is not None:
            n_select = max(int(n_total * self.top_fraction), self.min_cols)
        else:
            # In threshold mode, apply the configured significance test for each
            # feature type.
            selected_set = set()
            for col in num_cols:
                train_vals = X_train[col].dropna().values.astype(float)
                test_vals = X_test[col].dropna().values.astype(float)
                if len(train_vals) >= 2 and len(test_vals) >= 2:
                    _, p = stats.ks_2samp(train_vals, test_vals)
                    if p < self.p_value_threshold:
                        selected_set.add(col)
            for col in cat_cols:
                train_vals = X_train[col].astype(str)
                test_vals = X_test[col].astype(str)
                all_cats = sorted(set(train_vals.unique()) | set(test_vals.unique()))
                if len(all_cats) > 1:
                    train_counts = train_vals.value_counts()
                    test_counts = test_vals.value_counts()
                    obs_train = np.array([train_counts.get(c, 0) for c in all_cats])
                    obs_test = np.array([test_counts.get(c, 0) for c in all_cats])
                    # Chi-squared test on contingency table
                    contingency = np.vstack([obs_train, obs_test])
                    # Remove zero-sum columns
                    col_sums = contingency.sum(axis=0)
                    contingency = contingency[:, col_sums > 0]
                    if contingency.shape[1] > 1:
                        chi2, p, _, _ = stats.chi2_contingency(contingency)
                        if p < self.p_value_threshold:
                            selected_set.add(col)

            # Ensure minimum
            if len(selected_set) < self.min_cols and all_cols_sorted:
                for c in all_cols_sorted:
                    selected_set.add(c)
                    if len(selected_set) >= self.min_cols:
                        break

            selected_cat = [c for c in cat_cols if c in selected_set]
            selected_num = [c for c in num_cols if c in selected_set]
            return ColumnSelection(
                cat_cols=selected_cat,
                num_cols=selected_num,
                scores=scores,
            )

        # Top-k / top-fraction mode
        n_select = max(n_select, self.min_cols)
        selected_set = set(all_cols_sorted[:n_select])
        selected_cat = [c for c in cat_cols if c in selected_set]
        selected_num = [c for c in num_cols if c in selected_set]

        return ColumnSelection(
            cat_cols=selected_cat,
            num_cols=selected_num,
            scores=scores,
        )


# ===========================================================================
# Strategy 2: Domain Classifier Feature Importance
# ===========================================================================

class DomainClassifierSelector(ColumnSelector):
    """
    Select columns that best distinguish train from test data.

    Trains a lightweight classifier (train=0, test=1) and uses its feature
    importances to identify which columns drive the domain shift. This
    captures multivariate shift patterns that per-column tests miss.

    Args:
        top_k: Number of top columns to select.
        top_fraction: Fraction of columns to select (default 0.3).
        classifier: Which classifier to use ("lgbm" or "rf").
        min_cols: Minimum columns to select.
        max_samples: Subsample each split to this many rows for speed (default 30000).
    """

    def __init__(
        self,
        top_k: int | None = None,
        top_fraction: float = 0.3,
        classifier: str = "lgbm",
        min_cols: int = 1,
        max_samples: int = 30_000,
    ):
        self.top_k = top_k
        self.top_fraction = top_fraction
        self.classifier = classifier
        self.min_cols = min_cols
        self.max_samples = max_samples

    @property
    def name(self) -> str:
        k_str = f"top{self.top_k}" if self.top_k else f"frac{self.top_fraction}"
        return f"domain_clf_selector_{k_str}"

    def select(self, X_train, X_test, y_train):
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

        # Subsample for speed on large datasets (folktables can be 500K+ rows)
        rng = np.random.RandomState(42)
        X_tr_sub = X_train
        X_te_sub = X_test
        if len(X_train) > self.max_samples:
            idx = rng.choice(len(X_train), self.max_samples, replace=False)
            X_tr_sub = X_train.iloc[idx]
        if len(X_test) > self.max_samples:
            idx = rng.choice(len(X_test), self.max_samples, replace=False)
            X_te_sub = X_test.iloc[idx]

        # Prepare data: label-encode categoricals for the domain classifier
        X_tr_enc = X_tr_sub.copy()
        X_te_enc = X_te_sub.copy()
        for col in cat_cols:
            le = LabelEncoder()
            combined = pd.concat([X_tr_enc[col].astype(str), X_te_enc[col].astype(str)])
            le.fit(combined)
            X_tr_enc[col] = le.transform(X_tr_enc[col].astype(str))
            X_te_enc[col] = le.transform(X_te_enc[col].astype(str))

        X_combined = pd.concat([X_tr_enc, X_te_enc], ignore_index=True)
        domain_label = np.array([0] * len(X_tr_enc) + [1] * len(X_te_enc))

        # Handle NaN/Inf
        X_combined = X_combined.fillna(0)
        X_combined = X_combined.replace([np.inf, -np.inf], 0)

        # Train domain classifier
        if self.classifier == "lgbm":
            try:
                import lightgbm as lgb
                clf = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    verbose=-1,
                    n_jobs=1,
                    random_state=42,
                )
            except ImportError:
                # Fallback to RF
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=100, max_depth=4, n_jobs=1, random_state=42)
        else:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=100, max_depth=4, n_jobs=1, random_state=42)

        clf.fit(X_combined.values.astype(float), domain_label)

        # Extract feature importances
        importances = clf.feature_importances_
        all_columns = list(X_combined.columns)
        scores = {col: float(imp) for col, imp in zip(all_columns, importances)}

        # Rank and select
        cols_sorted = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        n_total = len(cols_sorted)

        if self.top_k is not None:
            n_select = min(self.top_k, n_total)
        else:
            n_select = max(int(n_total * self.top_fraction), self.min_cols)

        n_select = max(n_select, self.min_cols)
        selected_set = set(cols_sorted[:n_select])

        selected_cat = [c for c in cat_cols if c in selected_set]
        selected_num = [c for c in num_cols if c in selected_set]

        return ColumnSelection(
            cat_cols=selected_cat,
            num_cols=selected_num,
            scores=scores,
        )


# ===========================================================================
# Strategy 3: Combined Selector (Shift + Importance Intersection)
# ===========================================================================

class CombinedSelector(ColumnSelector):
    """
    Select columns that are shifted and predictively important.

    Combines two signals:
      1. Shift magnitude (from ShiftColumnSelector)
      2. Feature importance for the prediction task (from a baseline model)

    Selected columns must score above the threshold for shift and predictive
    importance.

    Args:
        shift_top_fraction: Fraction of columns considered "shifted" (default 0.5).
        importance_top_fraction: Fraction of columns considered "important" (default 0.5).
        combination: How to combine: "intersection" (both) or "union" (either).
        min_cols: Minimum columns to select.
    """

    def __init__(
        self,
        shift_top_fraction: float = 0.5,
        importance_top_fraction: float = 0.5,
        combination: str = "intersection",
        min_cols: int = 1,
        max_samples: int = 30_000,
    ):
        assert combination in ("intersection", "union")
        self.shift_top_fraction = shift_top_fraction
        self.importance_top_fraction = importance_top_fraction
        self.combination = combination
        self.min_cols = min_cols
        self.max_samples = max_samples

    @property
    def name(self) -> str:
        return f"combined_selector_{self.combination}"

    @staticmethod
    def _fallback_is_classification(y_train: np.ndarray) -> bool:
        unique_y = np.unique(y_train)
        if len(unique_y) > 50:
            return False
        try:
            return bool(np.allclose(unique_y, np.round(unique_y)))
        except (ValueError, TypeError):
            return False

    def _resolve_task_type(self, y_train: np.ndarray) -> tuple[bool, str]:
        if self.method_context is not None:
            return self.method_context.task_type == "classification", "context"
        return self._fallback_is_classification(y_train), "fallback_target_heuristic"

    def select(self, X_train, X_test, y_train):
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        all_cols = cat_cols + num_cols

        if not all_cols:
            return ColumnSelection(cat_cols=[], num_cols=[], scores={})

        # Subsample for speed on large datasets
        rng = np.random.RandomState(42)
        X_tr_sub = X_train
        X_te_sub = X_test
        y_tr_sub = y_train
        if len(X_train) > self.max_samples:
            idx = rng.choice(len(X_train), self.max_samples, replace=False)
            X_tr_sub = X_train.iloc[idx]
            # np.asarray() forces positional indexing. Protects against pandas
            # Series with duplicate labels (e.g., after bootstrap resampling in
            # run_single_experiment, which reindexes X_train but not y_train).
            y_tr_sub = np.asarray(y_train)[idx]
        if len(X_test) > self.max_samples:
            idx = rng.choice(len(X_test), self.max_samples, replace=False)
            X_te_sub = X_test.iloc[idx]

        # --- Signal 1: Shift scores ---
        # Use V2 (KS + Cramér's V) for comparable [0,1] scores across
        # numeric and categorical columns.
        shift_selector = ShiftColumnSelectorV2(top_fraction=1.0)  # Get scores for all
        shift_result = shift_selector.select(X_tr_sub, X_te_sub, y_tr_sub)
        shift_scores = shift_result.scores

        # --- Signal 2: Predictive importance ---
        # Train a quick model and extract feature importances
        X_tr_enc = X_tr_sub.copy()
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(X_tr_enc[col].astype(str))
            X_tr_enc[col] = le.transform(X_tr_enc[col].astype(str))

        X_tr_vals = X_tr_enc.values.astype(float)
        X_tr_vals = np.nan_to_num(X_tr_vals, nan=0.0, posinf=1e10, neginf=-1e10)

        is_classification, _ = self._resolve_task_type(np.asarray(y_tr_sub))

        try:
            import lightgbm as lgb
            if is_classification:
                clf = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    verbose=-1, n_jobs=1, random_state=42,
                )
            else:
                clf = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.1,
                    verbose=-1, n_jobs=1, random_state=42,
                )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if is_classification:
                clf = RandomForestClassifier(n_estimators=100, max_depth=4, n_jobs=1, random_state=42)
            else:
                clf = RandomForestRegressor(n_estimators=100, max_depth=4, n_jobs=1, random_state=42)

        clf.fit(X_tr_vals, y_tr_sub)
        importances = clf.feature_importances_
        importance_scores = {col: float(imp) for col, imp in zip(X_tr_enc.columns, importances)}

        # --- Combine: rank on both dimensions ---
        n_total = len(all_cols)
        n_shift = max(int(n_total * self.shift_top_fraction), 1)
        n_importance = max(int(n_total * self.importance_top_fraction), 1)

        shift_ranked = sorted(all_cols, key=lambda c: shift_scores.get(c, 0), reverse=True)
        importance_ranked = sorted(all_cols, key=lambda c: importance_scores.get(c, 0), reverse=True)

        shifted_set = set(shift_ranked[:n_shift])
        important_set = set(importance_ranked[:n_importance])

        if self.combination == "intersection":
            selected_set = shifted_set & important_set
        else:
            selected_set = shifted_set | important_set

        # Ensure minimum selection (fall back to top shifted if intersection is empty)
        if len(selected_set) < self.min_cols:
            for c in shift_ranked:
                selected_set.add(c)
                if len(selected_set) >= self.min_cols:
                    break

        # Combined score for logging (product of normalized ranks)
        combined_scores = {}
        for col in all_cols:
            s_rank = shift_ranked.index(col) / max(n_total - 1, 1)
            i_rank = importance_ranked.index(col) / max(n_total - 1, 1)
            combined_scores[col] = (1 - s_rank) * (1 - i_rank)  # Higher = better

        selected_cat = [c for c in cat_cols if c in selected_set]
        selected_num = [c for c in num_cols if c in selected_set]

        return ColumnSelection(
            cat_cols=selected_cat,
            num_cols=selected_num,
            scores=combined_scores,
        )


# ===========================================================================
# Strategy 1v2: Comparable Shift Scores via Cramér's V
# ===========================================================================


class ShiftColumnSelectorV2(ColumnSelector):
    """
    Column selector with comparable shift scores across numeric and
    categorical features.

    Numeric columns use the KS statistic and categorical columns use Cramér's
    V. Both scores lie in [0, 1], allowing the feature types to be ranked
    together:

    - **Numeric:** KS statistic (already [0, 1]): unchanged.
    - **Categorical:** Cramér's V = ``sqrt(chi2 / (n * (min(r, c) - 1)))``,
      which is also bounded [0, 1].  A value of 0 means identical
      distributions. 1 means completely disjoint.

    """

    def __init__(
        self,
        top_k: int | None = None,
        top_fraction: float | None = None,
        p_value_threshold: float = 0.05,
        min_cols: int = 1,
    ):
        self.top_k = top_k
        self.top_fraction = top_fraction
        self.p_value_threshold = p_value_threshold
        self.min_cols = min_cols

    @property
    def name(self) -> str:
        if self.top_k is not None:
            return f"shift_v2_selector_top{self.top_k}"
        if self.top_fraction is not None:
            return f"shift_v2_selector_frac{self.top_fraction}"
        return f"shift_v2_selector_p{self.p_value_threshold}"

    @staticmethod
    def _score_numeric(train_col: pd.Series, test_col: pd.Series) -> float:
        """KS statistic: already in [0, 1]."""
        tr = train_col.dropna().values.astype(float)
        te = test_col.dropna().values.astype(float)
        if len(tr) < 2 or len(te) < 2:
            return 0.0
        stat, _ = stats.ks_2samp(tr, te)
        return stat

    @staticmethod
    def _score_categorical(train_col: pd.Series, test_col: pd.Series) -> float:
        """Cramér's V: bounded in [0, 1], comparable with KS."""
        tr = train_col.astype(str)
        te = test_col.astype(str)

        all_cats = sorted(set(tr.unique()) | set(te.unique()))
        if len(all_cats) <= 1:
            return 0.0

        tr_counts = tr.value_counts()
        te_counts = te.value_counts()
        obs_tr = np.array([tr_counts.get(c, 0) for c in all_cats], dtype=float)
        obs_te = np.array([te_counts.get(c, 0) for c in all_cats], dtype=float)

        # Build 2×k contingency table
        contingency = np.vstack([obs_tr, obs_te])
        # Remove zero-sum columns
        col_sums = contingency.sum(axis=0)
        contingency = contingency[:, col_sums > 0]
        if contingency.shape[1] <= 1:
            return 0.0

        try:
            chi2, _, _, _ = stats.chi2_contingency(contingency)
        except ValueError:
            return 0.0

        n = contingency.sum()
        r, k = contingency.shape  # r=2 (train/test), k=n_categories
        denom = n * (min(r, k) - 1)
        if denom <= 0:
            return 0.0
        return np.sqrt(chi2 / denom)

    def select(self, X_train, X_test, y_train):
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

        scores: dict[str, float] = {}
        for col in num_cols:
            scores[col] = self._score_numeric(X_train[col], X_test[col])
        for col in cat_cols:
            scores[col] = self._score_categorical(X_train[col], X_test[col])

        all_sorted = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        n_total = len(all_sorted)

        if self.top_k is not None:
            n_select = min(self.top_k, n_total)
        elif self.top_fraction is not None:
            n_select = max(int(n_total * self.top_fraction), self.min_cols)
        else:
            # Threshold mode: select columns whose score exceeds a
            # percentile-based cutoff (since both KS and V are in [0,1],
            # a simple score threshold can be derived from the p-value
            # significance testing).
            selected_set: set[str] = set()

            for col in num_cols:
                tr = X_train[col].dropna().values.astype(float)
                te = X_test[col].dropna().values.astype(float)
                if len(tr) >= 2 and len(te) >= 2:
                    _, p = stats.ks_2samp(tr, te)
                    if p < self.p_value_threshold:
                        selected_set.add(col)

            for col in cat_cols:
                tr = X_train[col].astype(str)
                te = X_test[col].astype(str)
                all_cats = sorted(set(tr.unique()) | set(te.unique()))
                if len(all_cats) > 1:
                    tr_c = tr.value_counts()
                    te_c = te.value_counts()
                    obs_t = np.array([tr_c.get(c, 0) for c in all_cats])
                    obs_e = np.array([te_c.get(c, 0) for c in all_cats])
                    cont = np.vstack([obs_t, obs_e])
                    cs = cont.sum(axis=0)
                    cont = cont[:, cs > 0]
                    if cont.shape[1] > 1:
                        try:
                            _, p, _, _ = stats.chi2_contingency(cont)
                            if p < self.p_value_threshold:
                                selected_set.add(col)
                        except ValueError:
                            pass

            if len(selected_set) < self.min_cols and all_sorted:
                for c in all_sorted:
                    selected_set.add(c)
                    if len(selected_set) >= self.min_cols:
                        break

            return ColumnSelection(
                cat_cols=[c for c in cat_cols if c in selected_set],
                num_cols=[c for c in num_cols if c in selected_set],
                scores=scores,
            )

        # top-k / top-fraction mode
        n_select = max(n_select, self.min_cols)
        selected_set = set(all_sorted[:n_select])

        return ColumnSelection(
            cat_cols=[c for c in cat_cols if c in selected_set],
            num_cols=[c for c in num_cols if c in selected_set],
            scores=scores,
        )


# ===========================================================================
# Strategy 4: OOF Column Selector Wrapper
# ===========================================================================


class OOFColumnSelector(ColumnSelector):
    """
    The base selector is run across K training subsets and the selected columns
    are aggregated by majority, intersection or union. For each fold:

    1. Run the base selector on (out-of-fold train, full test).
    2. Record which columns were selected.

    Then aggregate across folds:

    - ``"majority"``: a column is kept if selected in ≥ 50% of folds.
    - ``"intersection"``: a column is kept only if selected in all folds.
    - ``"union"``: a column is kept if selected in ANY fold.

    The final selection is the aggregated set.
    """

    def __init__(
        self,
        base_selector: ColumnSelector,
        n_folds: int = 5,
        aggregation: str = "majority",
        seed: int = 42,
    ):
        assert aggregation in ("majority", "intersection", "union")
        self.base_selector = base_selector
        self.n_folds = n_folds
        self.aggregation = aggregation
        self.seed = seed

    @property
    def name(self) -> str:
        return f"oof_{self.base_selector.name}_{self.aggregation}"

    def select(self, X_train, X_test, y_train):
        from sklearn.model_selection import KFold

        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        all_cols = cat_cols + num_cols

        if not all_cols:
            return ColumnSelection(cat_cols=[], num_cols=[], scores={})

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        col_counts: dict[str, int] = {c: 0 for c in all_cols}
        fold_scores: dict[str, list[float]] = {c: [] for c in all_cols}

        for _, oof_idx in kf.split(X_train):
            X_oof = X_train.iloc[oof_idx]
            y_oof = y_train[oof_idx] if isinstance(y_train, np.ndarray) else y_train.iloc[oof_idx]

            result = self.base_selector.select(X_oof, X_test, y_oof)
            for c in result.all_cols:
                if c in col_counts:
                    col_counts[c] += 1
            for c, s in result.scores.items():
                if c in fold_scores:
                    fold_scores[c].append(s)

        # Aggregate
        if self.aggregation == "majority":
            threshold = self.n_folds / 2.0
            selected = {c for c, cnt in col_counts.items() if cnt >= threshold}
        elif self.aggregation == "intersection":
            selected = {c for c, cnt in col_counts.items() if cnt == self.n_folds}
        else:  # union
            selected = {c for c, cnt in col_counts.items() if cnt > 0}

        # Ensure at least 1 column selected. Fall back to most-voted
        if not selected:
            best = max(col_counts, key=col_counts.get)
            selected = {best}

        # Mean scores across folds for logging
        avg_scores = {
            c: float(np.mean(ss)) if ss else 0.0
            for c, ss in fold_scores.items()
        }

        return ColumnSelection(
            cat_cols=[c for c in cat_cols if c in selected],
            num_cols=[c for c in num_cols if c in selected],
            scores=avg_scores,
        )


# ===========================================================================
# Selector Registry
# ===========================================================================

COLUMN_SELECTORS = {
    "all": lambda: AllColumnsSelector(),
    "shift_top5": lambda: ShiftColumnSelector(top_k=5),
    "shift_top10": lambda: ShiftColumnSelector(top_k=10),
    "shift_top30pct": lambda: ShiftColumnSelector(top_fraction=0.3),
    "shift_significant": lambda: ShiftColumnSelector(p_value_threshold=0.05),
    "domain_clf_top30pct": lambda: DomainClassifierSelector(top_fraction=0.3),
    "domain_clf_top50pct": lambda: DomainClassifierSelector(top_fraction=0.5),
    "combined_intersection": lambda: CombinedSelector(combination="intersection"),
    "combined_union": lambda: CombinedSelector(
        combination="union", shift_top_fraction=0.3, importance_top_fraction=0.3,
    ),
    # --- V2: Comparable shift scores via Cramér's V ---
    "shift_v2_top5": lambda: ShiftColumnSelectorV2(top_k=5),
    "shift_v2_top10": lambda: ShiftColumnSelectorV2(top_k=10),
    "shift_v2_top30pct": lambda: ShiftColumnSelectorV2(top_fraction=0.3),
    "shift_v2_significant": lambda: ShiftColumnSelectorV2(p_value_threshold=0.05),
    # --- OOF column selectors ---
    "oof_shift_top5_majority": lambda: OOFColumnSelector(
        base_selector=ShiftColumnSelector(top_k=5), aggregation="majority",
    ),
    "oof_shift_top10_majority": lambda: OOFColumnSelector(
        base_selector=ShiftColumnSelector(top_k=10), aggregation="majority",
    ),
    "oof_shift_v2_top5_majority": lambda: OOFColumnSelector(
        base_selector=ShiftColumnSelectorV2(top_k=5), aggregation="majority",
    ),
    "oof_shift_v2_top10_majority": lambda: OOFColumnSelector(
        base_selector=ShiftColumnSelectorV2(top_k=10), aggregation="majority",
    ),
    "oof_shift_top10_intersection": lambda: OOFColumnSelector(
        base_selector=ShiftColumnSelector(top_k=10), aggregation="intersection",
    ),
    "oof_combined_intersection_majority": lambda: OOFColumnSelector(
        base_selector=CombinedSelector(combination="intersection"), aggregation="majority",
    ),
}


def get_column_selector(name: str) -> ColumnSelector:
    """Instantiate a column selector by registry name."""
    if name not in COLUMN_SELECTORS:
        raise ValueError(
            f"Unknown column selector '{name}'. Available: {list(COLUMN_SELECTORS.keys())}"
        )
    return COLUMN_SELECTORS[name]()

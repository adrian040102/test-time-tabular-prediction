"""Covariate-shift diagnostics for ``TabularDataset`` objects.

Numerical features are evaluated with Kolmogorov-Smirnov statistics,
Wasserstein distances and moment differences. Categorical features are
evaluated with chi-squared tests, Jensen-Shannon divergence and total variation.
The aggregate report also includes a numerical-feature domain classifier.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# Per-feature result containers

@dataclass
class NumericalFeatureShift:
    feature: str
    ks_statistic: float
    ks_pvalue: float
    wasserstein: float
    mean_train: float
    mean_test: float
    std_train: float
    std_test: float


@dataclass
class CategoricalFeatureShift:
    feature: str
    chi2_statistic: float
    chi2_pvalue: float
    js_divergence: float
    total_variation: float


@dataclass
class ShiftReport:
    """Full shift diagnostic report for a dataset."""

    dataset_name: str
    shift_description: str
    n_train: int
    n_test: int

    numerical_results: list[NumericalFeatureShift]
    categorical_results: list[CategoricalFeatureShift]

    # Aggregate metrics
    frac_significant_num: float  # fraction of num features with p < alpha
    frac_significant_cat: float  # fraction of cat features with p < alpha
    mean_ks_statistic: float
    mean_js_divergence: float
    domain_classifier_accuracy: float | None  # None if not computed
    alpha: float = 0.05

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"Shift diagnostics: {self.dataset_name}",
            f"  Shift: {self.shift_description}",
            f"  Train: {self.n_train:,}  |  Test: {self.n_test:,}",
            "",
            f"  Numerical features ({len(self.numerical_results)}):",
            f"    Mean KS statistic:      {self.mean_ks_statistic:.4f}",
            f"    Significant (p<{self.alpha}):   {self.frac_significant_num:.1%}",
            "",
            f"  Categorical features ({len(self.categorical_results)}):",
            f"    Mean JS divergence:     {self.mean_js_divergence:.4f}",
            f"    Significant (p<{self.alpha}):   {self.frac_significant_cat:.1%}",
        ]
        if self.domain_classifier_accuracy is not None:
            lines.append("")
            lines.append(f"  Domain classifier acc:    {self.domain_classifier_accuracy:.3f}")
            lines.append(f"    (0.5 = no shift, 1.0 = perfectly separable)")
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Return per-feature results as a single DataFrame."""
        rows = []
        for r in self.numerical_results:
            rows.append({
                "feature": r.feature,
                "type": "numerical",
                "ks_stat": r.ks_statistic,
                "ks_pvalue": r.ks_pvalue,
                "wasserstein": r.wasserstein,
                "mean_train": r.mean_train,
                "mean_test": r.mean_test,
                "std_train": r.std_train,
                "std_test": r.std_test,
                "significant": r.ks_pvalue < self.alpha,
            })
        for r in self.categorical_results:
            rows.append({
                "feature": r.feature,
                "type": "categorical",
                "chi2_stat": r.chi2_statistic,
                "chi2_pvalue": r.chi2_pvalue,
                "js_divergence": r.js_divergence,
                "total_variation": r.total_variation,
                "significant": r.chi2_pvalue < self.alpha,
            })
        return pd.DataFrame(rows)

    @property
    def overall_shift_score(self) -> float:
        """
        Single 0–1 score summarising overall shift magnitude.

        Computed as the mean of three normalised components:
          1. Mean KS statistic (numerical features)
          2. Mean JS divergence (categorical features), scaled to [0, 1]
          3. Domain classifier accuracy, rescaled from [0.5, 1] → [0, 1]

        Components are skipped if not available (e.g. no cat features).
        """
        components = []
        if self.numerical_results:
            components.append(min(self.mean_ks_statistic, 1.0))
        if self.categorical_results:
            # JS divergence is in [0, ln(2)] for binary distributions.
            # Normalise to [0, 1].
            components.append(min(self.mean_js_divergence / 0.693, 1.0))
        if self.domain_classifier_accuracy is not None:
            components.append(
                max(0.0, min(1.0, (self.domain_classifier_accuracy - 0.5) * 2))
            )
        return float(np.mean(components)) if components else 0.0


# Main diagnostics class

class ShiftDiagnostics:
    """
    Compute covariate shift statistics for a TabularDataset.

    Parameters
    ----------
    dataset : TabularDataset
        Must have both X_train, X_test populated.
    alpha : float
        Significance level for statistical tests (Bonferroni-corrected
        internally).
    max_domain_clf_samples : int
        Cap per split for the domain classifier (speed).
    """

    def __init__(
        self,
        dataset,  # TabularDataset. No type hint to avoid circular import
        alpha: float = 0.05,
        max_domain_clf_samples: int = 10_000,
    ):
        self.ds = dataset
        self.alpha = alpha
        self.max_domain_clf_samples = max_domain_clf_samples

    # Public API

    def full_report(self, run_domain_classifier: bool = True) -> ShiftReport:
        """Run all diagnostics and return a ShiftReport."""
        num_results = self._analyse_numerical()
        cat_results = self._analyse_categorical()

        n_total = len(num_results) + len(cat_results)
        bonferroni_alpha = self.alpha / max(n_total, 1)

        frac_sig_num = (
            np.mean([r.ks_pvalue < bonferroni_alpha for r in num_results])
            if num_results else 0.0
        )
        frac_sig_cat = (
            np.mean([r.chi2_pvalue < bonferroni_alpha for r in cat_results])
            if cat_results else 0.0
        )
        mean_ks = (
            np.mean([r.ks_statistic for r in num_results])
            if num_results else 0.0
        )
        mean_js = (
            np.mean([r.js_divergence for r in cat_results])
            if cat_results else 0.0
        )

        domain_acc = None
        if run_domain_classifier:
            domain_acc = self._domain_classifier_accuracy()

        return ShiftReport(
            dataset_name=self.ds.name,
            shift_description=self.ds.shift_description,
            n_train=self.ds.n_train,
            n_test=self.ds.n_test,
            numerical_results=num_results,
            categorical_results=cat_results,
            frac_significant_num=float(frac_sig_num),
            frac_significant_cat=float(frac_sig_cat),
            mean_ks_statistic=float(mean_ks),
            mean_js_divergence=float(mean_js),
            domain_classifier_accuracy=domain_acc,
            alpha=self.alpha,
        )

    # Numerical features

    def _analyse_numerical(self) -> list[NumericalFeatureShift]:
        results = []
        for col in self.ds.num_features:
            if col not in self.ds.X_train.columns:
                continue
            train_vals = self.ds.X_train[col].dropna().values.astype(float)
            test_vals = self.ds.X_test[col].dropna().values.astype(float)

            if len(train_vals) == 0 or len(test_vals) == 0:
                continue

            ks_stat, ks_p = sp_stats.ks_2samp(train_vals, test_vals)
            wass = float(sp_stats.wasserstein_distance(train_vals, test_vals))

            results.append(NumericalFeatureShift(
                feature=col,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_p),
                wasserstein=wass,
                mean_train=float(np.mean(train_vals)),
                mean_test=float(np.mean(test_vals)),
                std_train=float(np.std(train_vals)),
                std_test=float(np.std(test_vals)),
            ))
        return results

    # Categorical features

    def _analyse_categorical(self) -> list[CategoricalFeatureShift]:
        results = []
        for col in self.ds.cat_features:
            if col not in self.ds.X_train.columns:
                continue
            train_series = self.ds.X_train[col].astype(str)
            test_series = self.ds.X_test[col].astype(str)

            all_cats = sorted(set(train_series.unique()) | set(test_series.unique()))
            if len(all_cats) < 2:
                continue

            train_counts = train_series.value_counts().reindex(all_cats, fill_value=0)
            test_counts = test_series.value_counts().reindex(all_cats, fill_value=0)

            # Chi-squared (add small constant to avoid zero cells)
            observed = np.array([train_counts.values, test_counts.values], dtype=float)
            observed += 1e-10
            chi2_stat, chi2_p, _, _ = sp_stats.chi2_contingency(observed)

            # Normalised distributions
            p = train_counts.values.astype(float)
            q = test_counts.values.astype(float)
            p = p / p.sum()
            q = q / q.sum()

            # Jensen–Shannon divergence
            m = 0.5 * (p + q)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                js = float(0.5 * sp_stats.entropy(p, m) + 0.5 * sp_stats.entropy(q, m))

            # Total variation
            tv = float(0.5 * np.sum(np.abs(p - q)))

            results.append(CategoricalFeatureShift(
                feature=col,
                chi2_statistic=float(chi2_stat),
                chi2_pvalue=float(chi2_p),
                js_divergence=js,
                total_variation=tv,
            ))
        return results

    # Domain classifier

    def _domain_classifier_accuracy(self) -> float | None:
        """Fit a logistic-regression domain classifier on numerical features.

        Interpret cross-validated accuracy relative to the domain-label
        class-balance baseline. Higher accuracy indicates stronger linear
        separability.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return None

        num_cols = [c for c in self.ds.num_features if c in self.ds.X_train.columns]
        if not num_cols:
            return None

        X_tr = self.ds.X_train[num_cols].copy()
        X_te = self.ds.X_test[num_cols].copy()

        # Subsample for speed
        n = self.max_domain_clf_samples
        if len(X_tr) > n:
            idx = np.random.RandomState(42).choice(len(X_tr), n, replace=False)
            X_tr = X_tr.iloc[idx]
        if len(X_te) > n:
            idx = np.random.RandomState(43).choice(len(X_te), n, replace=False)
            X_te = X_te.iloc[idx]

        X = pd.concat([X_tr, X_te], ignore_index=True).fillna(0).values
        y = np.array([0] * len(X_tr) + [1] * len(X_te))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=200, solver="lbfgs", random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="accuracy")

        return float(scores.mean())


# Analyze multiple datasets.

def analyse_all_datasets(
    datasets: dict[str, object],
    run_domain_classifier: bool = True,
) -> pd.DataFrame:
    """
    Run shift diagnostics on a dictionary of {name: TabularDataset} and
    return a summary DataFrame with one row per dataset.
    """
    rows = []
    for name, ds in datasets.items():
        try:
            diag = ShiftDiagnostics(ds)
            report = diag.full_report(run_domain_classifier=run_domain_classifier)
            rows.append({
                "dataset": name,
                "shift_description": report.shift_description,
                "n_train": report.n_train,
                "n_test": report.n_test,
                "n_num_features": len(report.numerical_results),
                "n_cat_features": len(report.categorical_results),
                "mean_ks_stat": report.mean_ks_statistic,
                "frac_significant_num": report.frac_significant_num,
                "mean_js_div": report.mean_js_divergence,
                "frac_significant_cat": report.frac_significant_cat,
                "domain_clf_acc": report.domain_classifier_accuracy,
                "overall_shift_score": report.overall_shift_score,
            })
        except Exception as e:
            print(f"  ⚠ {name}: {e}")
            rows.append({"dataset": name, "error": str(e)})

    return pd.DataFrame(rows)

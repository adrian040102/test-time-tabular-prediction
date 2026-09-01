"""
SelectiveMethod: wrapper that applies a method only to selected columns.

This wrapper applies a method to selected columns:
1. Runs a ColumnSelector to identify which columns to engineer
2. Passes only selected columns to the inner method
3. Keeps unselected columns as-is (label-encoded for categoricals)
4. Concatenates both halves in the output

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.methods.base import TestTimeMethod, TransformResult
from src.methods.column_selection import ColumnSelector, AllColumnsSelector, CombinedSelector


class SelectiveMethod(TestTimeMethod):
    """
    Wraps any TestTimeMethod and restricts it to selected columns only.

    Unselected columns are passed through with minimal preprocessing
    (label encoding for categoricals). The final output concatenates
    the method's transformed features with the passthrough features.

    Args:
        method: The inner method to apply selectively.
        selector: A ColumnSelector that decides which columns to engineer.
    """

    def __init__(self, method: TestTimeMethod, selector: ColumnSelector):
        self.method = method
        self.selector = selector

    @property
    def name(self) -> str:
        return f"{self.method.name}[{self.selector.name}]"

    @property
    def uses_labels(self) -> bool:
        return self.method.uses_labels or self._selector_uses_labels()

    @property
    def may_emit_sample_weight(self) -> bool:
        return self.method.may_emit_sample_weight

    @property
    def produces_prescaled_output(self) -> bool:
        return self.method.produces_prescaled_output

    @property
    def supported_task_types(self) -> frozenset[str]:
        return self.method.supported_task_types

    def _selector_uses_labels(self) -> bool:
        """Check if the column selector itself depends on y_train."""
        # CombinedSelector trains an LGBM on (X_train, y_train) for importance
        # scores, so any SelectiveMethod wrapping it depends on y_train.
        return isinstance(self.selector, CombinedSelector)

    def fit_transform(self, X_train, X_test, y_train):
        # Step 1: Run column selector
        selection = self.selector.select(X_train, X_test, y_train)

        all_cat = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        all_num = X_train.select_dtypes(include=["number"]).columns.tolist()
        all_cols = list(X_train.columns)

        selected_cols = selection.all_cols
        passthrough_cols = [c for c in all_cols if c not in selected_cols]

        # Edge case: if no columns selected, fall back to applying method to all
        if not selected_cols:
            result = self.method.fit_transform(X_train, X_test, y_train)
            result.metadata["column_selection"] = {
                "selector": self.selector.name,
                "n_selected": 0,
                "n_total": len(all_cols),
                "fallback": "all_columns",
            }
            return result

        # If all columns are selected, run the method directly.
        if not passthrough_cols:
            result = self.method.fit_transform(X_train, X_test, y_train)
            result.metadata["column_selection"] = {
                "selector": self.selector.name,
                "n_selected": len(selected_cols),
                "n_total": len(all_cols),
                "selected_cols": selected_cols,
                "scores": selection.scores,
            }
            return result

        # Step 2: Split into selected and passthrough
        X_train_selected = X_train[selected_cols].copy()
        X_test_selected = X_test[selected_cols].copy()

        X_train_pass = X_train[passthrough_cols].copy()
        X_test_pass = X_test[passthrough_cols].copy()

        # Step 3: Run inner method on selected columns only
        result = self.method.fit_transform(X_train_selected, X_test_selected, y_train)

        # Step 4: Label-encode passthrough categoricals
        pass_cat_cols = X_train_pass.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        for col in pass_cat_cols:
            le = LabelEncoder()
            combined = pd.concat([
                X_train_pass[col].astype(str),
                X_test_pass[col].astype(str),
            ])
            le.fit(combined)
            X_train_pass[col] = le.transform(X_train_pass[col].astype(str))
            X_test_pass[col] = le.transform(X_test_pass[col].astype(str))

        pass_train = X_train_pass.values.astype(float)
        pass_test = X_test_pass.values.astype(float)
        pass_names = list(X_train_pass.columns)

        # Step 5: Concatenate method output + passthrough
        # Handle augmenting methods (M7, M17) that add rows to X_train.
        # The extra rows represent pseudo-labeled test samples. Pad passthrough
        # columns with NaN for those rows (the pipeline's imputer handles NaN).
        n_method_train = result.X_train.shape[0]
        n_orig_train = pass_train.shape[0]
        if n_method_train > n_orig_train:
            n_extra = n_method_train - n_orig_train
            pad = np.full((n_extra, pass_train.shape[1]), np.nan)
            pass_train = np.vstack([pass_train, pad])

        X_tr_out = np.hstack([result.X_train, pass_train])
        X_te_out = np.hstack([result.X_test, pass_test])
        feature_names = result.feature_names + pass_names

        # Merge metadata
        metadata = dict(result.metadata) if result.metadata else {}
        metadata["column_selection"] = {
            "selector": self.selector.name,
            "n_selected": len(selected_cols),
            "n_passthrough": len(passthrough_cols),
            "n_total": len(all_cols),
            "selected_cat": selection.cat_cols,
            "selected_num": selection.num_cols,
            "scores": {k: round(v, 4) for k, v in selection.scores.items()},
        }

        # Declare every categorical column that remains in the output,
        # including label-encoded passthrough columns. Preserve stable ordering.
        inner_cats = result.categorical_features or []
        merged_cats = list(dict.fromkeys(list(inner_cats) + list(pass_cat_cols)))
        declared_cats = [c for c in merged_cats if c in feature_names] or None

        return TransformResult(
            X_train=X_tr_out,
            X_test=X_te_out,
            feature_names=feature_names,
            sample_weights_train=result.sample_weights_train,
            y_train=result.y_train,
            metadata=metadata,
            skip_pipeline_scaler=result.skip_pipeline_scaler,
            categorical_features=declared_cats,
        )

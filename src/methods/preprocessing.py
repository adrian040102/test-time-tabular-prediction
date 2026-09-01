"""
Shared preprocessing utilities for all methods.

This module centralizes label encoding and column-detection helpers shared by
the method implementations. Per-file helpers remain as compatibility
re-exports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def get_cat_cols(X: pd.DataFrame) -> list[str]:
    """Return categorical column names (object or category dtype)."""
    return X.select_dtypes(include=["object", "category"]).columns.tolist()


def get_num_cols(X: pd.DataFrame) -> list[str]:
    """Return numeric column names."""
    return X.select_dtypes(include=["number"]).columns.tolist()


# ---------------------------------------------------------------------------
# Label encoding: joint (transductive)
# ---------------------------------------------------------------------------

def label_encode_joint(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Label-encode all categorical columns, fit on train+test combined.

    This is the transductive encoding used by most test-time methods:
    the encoder receives both splits to produce a consistent category-to-integer mapping.

    Returns:
        X_train_enc: Encoded training DataFrame (copy).
        X_test_enc:  Encoded test DataFrame (copy).
        cat_cols:    List of columns that were encoded.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    cat_cols = get_cat_cols(X_train)

    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)])
        le.fit(combined)
        X_tr[col] = le.transform(X_tr[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))

    return X_tr, X_te, cat_cols


# ---------------------------------------------------------------------------
# Label encoding: inductive (train-only)
# ---------------------------------------------------------------------------

def label_encode_train_only(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    unseen_value: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Label-encode all categorical columns, fit on training data only.

    Categories in the test set that were not seen during training are
    mapped to ``unseen_value`` (default -1).

    Returns:
        X_train_enc: Encoded training DataFrame (copy).
        X_test_enc:  Encoded test DataFrame (copy).
        cat_cols:    List of columns that were encoded.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    cat_cols = get_cat_cols(X_train)

    for col in cat_cols:
        le = LabelEncoder()
        le.fit(X_tr[col].astype(str))

        X_tr[col] = le.transform(X_tr[col].astype(str))

        # Handle unseen categories in test
        test_vals = X_te[col].astype(str)
        known_mask = test_vals.isin(le.classes_)
        X_te[col] = unseen_value
        X_te.loc[known_mask, col] = le.transform(test_vals[known_mask])

    return X_tr, X_te, cat_cols


# ---------------------------------------------------------------------------
# Encode to numeric arrays (for classifiers that need plain np.ndarray)
# ---------------------------------------------------------------------------

def encode_for_classifier(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Label-encode categoricals and return float numpy arrays.

    This wrapper provides numerical input to domain classifiers and k-NN.
    It applies joint encoding to the training and test rows.
    """
    X_tr, X_te, _ = label_encode_joint(X_train, X_test)
    return X_tr.values.astype(float), X_te.values.astype(float)

"""Evaluate methods that return sample weights or additional training rows.

``weighted_augment_auc`` applies the same target, sample-weight and scaling
rules as the experiment pipeline. It also supports the inductive baseline and
an optional downstream estimator factory.
"""
from __future__ import annotations

import numpy as np


def _make_logreg():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=1000, random_state=0)


def weighted_augment_auc(method, X_tr_df, X_te_df, y_tr, y_te, downstream=None) -> float:
    """Fit a method and return binary test AUC from the transformed data.

    Parameters
    ----------
    method
        A ``TestTimeMethod`` (or ``InductiveBaseline``).
    X_tr_df, X_te_df
        Training and test feature DataFrames.
    y_tr, y_te
        Training and test labels. Test labels are used only to calculate AUC.
    downstream
        Optional zero-argument factory returning an unfitted estimator with a
        ``predict_proba`` method. Defaults to ``LogisticRegression(max_iter=1000,
        random_state=0)``. The tests in this module use the logistic-regression
        default.

    Returns
    -------
    float
        ``roc_auc_score`` on the test split.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    result = method.fit_transform(X_tr_df, X_te_df, y_tr)
    X_tr = np.asarray(result.X_train)
    X_te = np.asarray(result.X_test)

    # Use the transformed target when it has one value per transformed row.
    if result.y_train is not None and len(result.y_train) == X_tr.shape[0]:
        y_for_model = np.asarray(result.y_train)
    else:
        y_for_model = np.asarray(y_tr)

    # Pass sample weights to the estimator after checking their length.
    sample_weights = (
        None
        if result.sample_weights_train is None
        else np.asarray(result.sample_weights_train)
    )
    if sample_weights is not None and len(sample_weights) != X_tr.shape[0]:
        raise ValueError(
            f"sample_weights_train length ({len(sample_weights)}) does not match "
            f"transformed train rows ({X_tr.shape[0]})"
        )

    # Standardize unless the method already produced joint coordinates.
    if not result.skip_pipeline_scaler:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    clf = (downstream or _make_logreg)()
    clf.fit(X_tr, y_for_model, sample_weight=sample_weights)
    return float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))

"""Mixed-type train-versus-test separability diagnostic.

The diagnostic treats the training/test origin as a binary label, applies the
project's established mixed-type encoding and reports a five-fold
cross-validated ROC AUC from a histogram gradient-boosted tree.

The folds are explicitly shuffled and stratified.  This matters for temporal
datasets whose stored rows are ordered by time: unshuffled folds would evaluate
different time blocks in different folds and can make strongly separable
domains appear indistinguishable when reversed fold AUCs cancel high fold AUCs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder


def compute_domain_auc(
    dataset,
    *,
    cap_per_split: int = 10_000,
    random_state: int = 42,
) -> tuple[float, int]:
    """Return mixed-type Domain AUC and the number of rows used.

    Parameters
    ----------
    dataset:
        A ``TabularDataset``-compatible object with ``X_train``, ``X_test``
        and ``cat_features`` attributes.
    cap_per_split:
        Maximum number of rows drawn independently from either split.
    random_state:
        Seed shared by subsampling, cross-validation and the classifier.
    """
    columns = list(dataset.X_train.columns)
    x_train = dataset.X_train[columns].copy()
    x_test = dataset.X_test[columns].copy()

    rng = np.random.RandomState(random_state)
    if len(x_train) > cap_per_split:
        x_train = x_train.iloc[
            rng.choice(len(x_train), cap_per_split, replace=False)
        ]
    if len(x_test) > cap_per_split:
        x_test = x_test.iloc[
            rng.choice(len(x_test), cap_per_split, replace=False)
        ]

    categorical = [c for c in dataset.cat_features if c in columns]
    numerical = [c for c in columns if c not in categorical]
    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []

    if categorical:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        encoder.fit(
            pd.concat(
                [
                    x_train[categorical].astype(str),
                    x_test[categorical].astype(str),
                ],
                ignore_index=True,
            )
        )
        train_blocks.append(encoder.transform(x_train[categorical].astype(str)))
        test_blocks.append(encoder.transform(x_test[categorical].astype(str)))

    if numerical:
        train_blocks.append(
            x_train[numerical].apply(pd.to_numeric, errors="coerce").to_numpy()
        )
        test_blocks.append(
            x_test[numerical].apply(pd.to_numeric, errors="coerce").to_numpy()
        )

    if not train_blocks:
        raise ValueError("Domain AUC requires at least one input feature")

    x = np.vstack([np.hstack(train_blocks), np.hstack(test_blocks)])
    y = np.array([0] * len(x_train) + [1] * len(x_test))
    classifier = HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=120,
        random_state=random_state,
    )
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )
    auc = cross_val_score(
        classifier,
        x,
        y,
        cv=cv,
        scoring="roc_auc",
    ).mean()
    return float(auc), int(len(x))

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.evaluation import domain_auc as domain_auc_module


def test_domain_auc_uses_seeded_shuffled_stratified_folds(monkeypatch):
    dataset = SimpleNamespace(
        X_train=pd.DataFrame(
            {
                "x": np.arange(20, dtype=float),
                "category": ["a", "b"] * 10,
            }
        ),
        X_test=pd.DataFrame(
            {
                "x": np.arange(20, 40, dtype=float),
                "category": ["b", "c"] * 10,
            }
        ),
        cat_features=["category"],
    )
    captured = {}

    def fake_cross_val_score(estimator, x, y, *, cv, scoring):
        captured["cv"] = cv
        captured["scoring"] = scoring
        captured["rows"] = len(x)
        captured["columns"] = x.shape[1]
        return np.full(5, 0.75)

    monkeypatch.setattr(domain_auc_module, "cross_val_score", fake_cross_val_score)

    auc, rows = domain_auc_module.compute_domain_auc(dataset)

    assert auc == 0.75
    assert rows == 40
    assert captured["rows"] == 40
    assert captured["columns"] == 2
    assert captured["scoring"] == "roc_auc"
    assert isinstance(captured["cv"], StratifiedKFold)
    assert captured["cv"].n_splits == 5
    assert captured["cv"].shuffle is True
    assert captured["cv"].random_state == 42

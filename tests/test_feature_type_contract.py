"""Regression checks for the stored feature-type contract."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.openml_datasets import (
    _apply_openml_feature_value_repairs,
    _resolve_openml_feature_types,
)
from src.data.parquet_loader import load_parquet_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABARENA_ROOT = PROJECT_ROOT / "data" / "tabarena" / "datasets"

REPAIRED_CATEGORICALS = {
    "bank-marketing": {"poutcome"},
    "coil2000_insurance_policies": {
        "avgAge",
        "contributionPrivateThirdPartyInsurance",
    },
    "HR_Analytics_Job_Change_of_Data_Scientists": {"experience"},
    "in_vehicle_coupon_recommendation": {"coupon"},
    "seismic-bumps": {"ghazard"},
}


@pytest.mark.parametrize(
    ("dataset_name", "expected_categorical"),
    REPAIRED_CATEGORICALS.items(),
)
def test_repaired_text_columns_are_declared_categorical(
    dataset_name: str,
    expected_categorical: set[str],
):
    dataset_dir = TABARENA_ROOT / dataset_name
    if not dataset_dir.is_dir():
        pytest.skip(f"{dataset_name} cache is not present")

    meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
    assert expected_categorical <= set(meta["cat_features"])
    assert expected_categorical.isdisjoint(meta["num_features"])

    for filename in ("data.parquet", "train.parquet", "test.parquet"):
        frame = pd.read_parquet(dataset_dir / filename)
        assert expected_categorical <= set(frame.columns)
        for column in meta["num_features"]:
            assert pd.api.types.is_numeric_dtype(frame[column]), (
                dataset_name,
                filename,
                column,
                frame[column].dtype,
            )


def test_marketing_campaign_date_is_an_ordered_day_offset():
    dataset_dir = TABARENA_ROOT / "Marketing_Campaign"
    if not dataset_dir.is_dir():
        pytest.skip("Marketing_Campaign cache is not present")

    meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
    assert "Dt_Customer" in meta["num_features"]
    assert "Dt_Customer" not in meta["cat_features"]
    for filename in ("data.parquet", "train.parquet", "test.parquet"):
        frame = pd.read_parquet(dataset_dir / filename)
        values = frame["Dt_Customer"]
        assert pd.api.types.is_integer_dtype(values)
        assert values.isna().sum() == 0
        assert int(values.min()) >= 0
        assert int(values.max()) <= 699


@pytest.mark.parametrize(
    "dataset_name",
    [*REPAIRED_CATEGORICALS, "Marketing_Campaign"],
)
def test_repaired_dataset_loads_in_default_and_fold_modes(dataset_name: str):
    if not (TABARENA_ROOT / dataset_name).is_dir():
        pytest.skip(f"{dataset_name} cache is not present")
    default_dataset = load_parquet_dataset(dataset_name)
    fold = load_parquet_dataset(dataset_name, fold_idx=0)
    assert default_dataset.cat_features == fold.cat_features
    assert default_dataset.num_features == fold.num_features
    for dataset in (default_dataset, fold):
        assert set(dataset.cat_features).isdisjoint(dataset.num_features)
        assert set(dataset.cat_features) | set(dataset.num_features) == set(
            dataset.X_train.columns
        )
        for column in dataset.num_features:
            assert pd.api.types.is_numeric_dtype(dataset.X_train[column])
            assert pd.api.types.is_numeric_dtype(dataset.X_test[column])


def test_openml_repair_is_reproducible_from_raw_values():
    frame = pd.DataFrame(
        {
            "Dt_Customer": ["2012-07-30", "2014-06-29"],
            "value": [1.0, 2.0],
        }
    )
    repaired = _apply_openml_feature_value_repairs(frame, 46940)
    assert repaired["Dt_Customer"].tolist() == [0, 699]
    assert str(repaired["Dt_Customer"].dtype) == "int32"

    categorical_frame = pd.DataFrame(
        {"poutcome": ["unknown", "success"], "value": [1.0, 2.0]}
    )
    cat, num = _resolve_openml_feature_types(
        categorical_frame,
        [False, False],
        list(categorical_frame.columns),
        46910,
    )
    assert cat == ["poutcome"]
    assert num == ["value"]


def test_unknown_text_as_numerical_fails_closed():
    frame = pd.DataFrame({"text_feature": ["a", "b"], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="non-numeric columns declared numerical"):
        _resolve_openml_feature_types(
            frame,
            [False, False],
            list(frame.columns),
            999999,
        )


def test_parquet_loader_rejects_text_declared_numerical(tmp_path: Path):
    dataset_dir = tmp_path / "invalid_contract"
    dataset_dir.mkdir()
    train = pd.DataFrame(
        {
            "text_feature": ["a", "b", "c", "d"],
            "value": [1.0, 2.0, 3.0, 4.0],
            "__target__": [0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "text_feature": ["a", "d"],
            "value": [5.0, 6.0],
            "__target__": [0, 1],
        }
    )
    train.to_parquet(dataset_dir / "train.parquet", index=False)
    test.to_parquet(dataset_dir / "test.parquet", index=False)
    (dataset_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "invalid_contract",
                "task_type": "classification",
                "cat_features": [],
                "num_features": ["text_feature", "value"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="declares non-numeric stored columns"):
        load_parquet_dataset("invalid_contract", data_root=tmp_path)

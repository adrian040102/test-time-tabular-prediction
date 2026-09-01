"""Canonical contract for the five controlled covariate-shift datasets."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset
from src.data.parquet_loader import load_parquet_dataset
from src.data.synthetic import (
    CONTROLLED_SHIFT_STRENGTHS,
    CONTROLLED_SYNTHETIC_N_TEST,
    CONTROLLED_SYNTHETIC_N_TRAIN,
    CONTROLLED_SYNTHETIC_SEED,
)


@pytest.mark.parametrize("shift", CONTROLLED_SHIFT_STRENGTHS)
def test_named_controlled_dataset_recreates_expected_10k_split(shift: float):
    dataset = load_dataset(f"synthetic_shift_{shift}")

    assert dataset.n_train == CONTROLLED_SYNTHETIC_N_TRAIN == 10_000
    assert dataset.n_test == CONTROLLED_SYNTHETIC_N_TEST == 10_000
    assert dataset.task_type == "classification"
    assert dataset.metadata == {
        "shift_strength": shift,
        "shift_mode": "full",
        "seed": CONTROLLED_SYNTHETIC_SEED,
    }
    assert dataset.cat_features == ["OCCP", "MAR", "ST"]
    assert dataset.num_features == [
        "AGEP",
        "SCHL",
        "WKHP",
        "CAPITALGAIN",
        "CAPITALLOSS",
    ]


def test_controlled_series_has_identical_training_data_and_schema():
    datasets = [load_dataset(f"synthetic_shift_{shift}") for shift in CONTROLLED_SHIFT_STRENGTHS]
    reference = datasets[0]

    for dataset in datasets[1:]:
        pd.testing.assert_frame_equal(dataset.X_train, reference.X_train, check_exact=True)
        assert (dataset.y_train == reference.y_train).all()
        assert list(dataset.X_test.columns) == list(reference.X_test.columns)
        assert list(dataset.X_test.dtypes.astype(str)) == list(
            reference.X_test.dtypes.astype(str)
        )


@pytest.mark.parametrize("shift", CONTROLLED_SHIFT_STRENGTHS)
def test_experiment_seed_uses_stored_controlled_split(shift: float):
    """Seed 0 is a model identity, not permission to regenerate the split."""

    name = f"synthetic_shift_{shift}"
    expected = load_parquet_dataset(name, seed=0)
    observed = load_dataset(name, seed=0)

    pd.testing.assert_frame_equal(observed.X_train, expected.X_train, check_exact=True)
    pd.testing.assert_frame_equal(observed.X_test, expected.X_test, check_exact=True)
    assert (observed.y_train == expected.y_train).all()
    assert (observed.y_test == expected.y_test).all()
    assert observed.metadata["seed"] == CONTROLLED_SYNTHETIC_SEED


@pytest.mark.parametrize("shift", CONTROLLED_SHIFT_STRENGTHS)
def test_canary_subsample_uses_stored_controlled_split(shift: float):
    name = f"synthetic_shift_{shift}"
    expected = load_parquet_dataset(name, seed=0, max_samples=500)
    observed = load_dataset(name, seed=0, max_samples=500)

    pd.testing.assert_frame_equal(observed.X_train, expected.X_train, check_exact=True)
    pd.testing.assert_frame_equal(observed.X_test, expected.X_test, check_exact=True)
    assert (observed.y_train == expected.y_train).all()
    assert (observed.y_test == expected.y_test).all()


def test_auxiliary_synthetic_size_contracts_are_unchanged():
    small = load_dataset("synthetic_small_test")
    large = load_dataset("synthetic_large_test")

    assert (small.n_train, small.n_test) == (50_000, 5_000)
    assert (large.n_train, large.n_test) == (5_000, 50_000)

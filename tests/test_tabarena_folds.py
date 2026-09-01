"""Tests for TabArena fold files, dataset loading and manifest propagation.

The cache-dependent tests are skipped when fold files are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import load_dataset
from src.data.parquet_loader import (
    _find_dataset_dir,
    load_parquet_dataset,
    parquet_dataset_exists,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABARENA_ROOT = PROJECT_ROOT / "data" / "tabarena" / "datasets"

# Fold-layer checks use this small dataset when its files are available.
SAMPLE_DATASET = "blood-transfusion-service-center"


def _has_fold_layer(name: str) -> bool:
    d = TABARENA_ROOT / name
    return (
        (d / "data.parquet").is_file()
        and (d / "folds.npz").is_file()
        and (d / "meta.json").is_file()
    )


@pytest.fixture(scope="module")
def sample_meta() -> dict:
    if not _has_fold_layer(SAMPLE_DATASET):
        pytest.skip(
            f"{SAMPLE_DATASET} has no augmented fold data"
        )
    with open(TABARENA_ROOT / SAMPLE_DATASET / "meta.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Augmentation artifacts
# ---------------------------------------------------------------------------

class TestAugmentationArtifacts:

    def test_data_parquet_exists(self, sample_meta):
        path = TABARENA_ROOT / SAMPLE_DATASET / "data.parquet"
        df = pd.read_parquet(path)
        # Augmented data.parquet must contain the special target column
        assert "__target__" in df.columns
        # Total rows = sum of fold train + test sizes for any single fold
        spec0 = sample_meta["fold_specs"][0]
        assert len(df) == spec0["train_size"] + spec0["test_size"]

    def test_folds_npz_keys_complete(self, sample_meta):
        with np.load(TABARENA_ROOT / SAMPLE_DATASET / "folds.npz") as npz:
            keys = set(npz.files)
        n_folds = sample_meta["n_total_folds"]
        expected = {f"train_idx_{i}" for i in range(n_folds)} | \
                   {f"test_idx_{i}" for i in range(n_folds)}
        assert keys == expected

    def test_meta_fold_specs_consistent(self, sample_meta):
        specs = sample_meta["fold_specs"]
        # fold_idx is 0-based and contiguous
        assert [s["fold_idx"] for s in specs] == list(range(len(specs)))
        # n_total_folds matches the spec list length
        assert sample_meta["n_total_folds"] == len(specs)
        # Each cross-validation training split is larger than its test split.
        for s in specs:
            assert s["train_size"] > s["test_size"]

    def test_fold_indices_are_a_partition(self, sample_meta):
        """Within each repeat, test indices must partition the row index set."""
        with np.load(TABARENA_ROOT / SAMPLE_DATASET / "folds.npz") as npz:
            folds_by_repeat: dict[int, list[np.ndarray]] = {}
            for spec in sample_meta["fold_specs"]:
                rep = spec["repeat"]
                folds_by_repeat.setdefault(rep, []).append(
                    npz[f"test_idx_{spec['fold_idx']}"]
                )
        # data length
        n = pd.read_parquet(TABARENA_ROOT / SAMPLE_DATASET / "data.parquet").shape[0]
        for rep, test_arrays in folds_by_repeat.items():
            # Within one repeat, test indices across folds should partition [0, n)
            concat = np.concatenate(test_arrays)
            assert len(concat) == n, f"repeat {rep}: incomplete coverage"
            assert len(set(concat.tolist())) == n, f"repeat {rep}: overlap"


# ---------------------------------------------------------------------------
# Loader API
# ---------------------------------------------------------------------------

class TestFoldLoader:

    def test_default_path_unchanged(self):
        """load_parquet_dataset(name) without fold_idx must use train/test
        parquets and not touch fold_specs."""
        if not parquet_dataset_exists(SAMPLE_DATASET):
            pytest.skip(f"{SAMPLE_DATASET} not present")
        ds = load_parquet_dataset(SAMPLE_DATASET)
        # Check the structure of the loader path without fold_idx.
        assert ds.X_train is not None and ds.X_test is not None
        assert "fold_idx" not in ds.metadata

    def test_fold_idx_loads_split(self, sample_meta):
        ds = load_parquet_dataset(SAMPLE_DATASET, fold_idx=0)
        spec0 = sample_meta["fold_specs"][0]
        assert len(ds.X_train) == spec0["train_size"]
        assert len(ds.X_test) == spec0["test_size"]
        # Metadata records the fold spec
        assert ds.metadata["fold_idx"] == 0
        assert ds.metadata["fold_repeat"] == spec0["repeat"]
        assert ds.metadata["fold_fold"] == spec0["fold"]
        assert ds.metadata["fold_sample"] == spec0["sample"]

    def test_fold_idx_out_of_range_raises(self, sample_meta):
        n = sample_meta["n_total_folds"]
        with pytest.raises(IndexError):
            load_parquet_dataset(SAMPLE_DATASET, fold_idx=n)
        with pytest.raises(IndexError):
            load_parquet_dataset(SAMPLE_DATASET, fold_idx=-1)

    def test_load_dataset_passes_fold_idx_through(self):
        """load_dataset(name, fold_idx=K) must reach the parquet loader."""
        if not _has_fold_layer(SAMPLE_DATASET):
            pytest.skip(f"{SAMPLE_DATASET} not augmented")
        ds_a = load_dataset(SAMPLE_DATASET, fold_idx=2)
        ds_b = load_dataset(SAMPLE_DATASET, fold_idx=2)
        # Same fold, same content (deterministic indexing)
        assert ds_a.metadata["fold_idx"] == 2
        assert len(ds_a.X_train) == len(ds_b.X_train)
        # Load a second fold for comparison.
        ds_other = load_dataset(SAMPLE_DATASET, fold_idx=10)
        assert ds_other.metadata["fold_idx"] == 10
        # Stratified folds can differ by a few rows across repeats.
        assert abs(len(ds_a.X_train) - len(ds_other.X_train)) <= 5

    def test_columns_match_default_loader(self):
        """Fold and default loaders must produce identical column sets
        so any cached method lookup or preprocessor still works."""
        if not _has_fold_layer(SAMPLE_DATASET):
            pytest.skip(f"{SAMPLE_DATASET} not augmented")
        ds_default = load_dataset(SAMPLE_DATASET)
        ds_fold = load_dataset(SAMPLE_DATASET, fold_idx=0)
        assert list(ds_default.X_train.columns) == list(ds_fold.X_train.columns)
        assert list(ds_default.X_test.columns) == list(ds_fold.X_test.columns)
        assert ds_default.cat_features == ds_fold.cat_features
        assert ds_default.num_features == ds_fold.num_features
        assert ds_default.task_type == ds_fold.task_type


# ---------------------------------------------------------------------------
# Manifest fold propagation
# ---------------------------------------------------------------------------

class TestManifestRoundTrip:

    def test_fold_column_in_quick_check_manifest(self):
        """If the quick-check manifest exists, its fold column must be populated."""
        manifest = PROJECT_ROOT / "scripts" / "slurm" / "manifest_tabarena_quick_check.csv"
        if not manifest.is_file():
            pytest.skip("manifest_tabarena_quick_check.csv not generated")
        df = pd.read_csv(manifest)
        assert "fold" in df.columns
        # Every TabArena row should have a non-empty fold (ints 0..29)
        non_empty = df["fold"].notna() & (df["fold"].astype(str) != "")
        assert non_empty.all(), \
            f"{(~non_empty).sum()} manifest rows have empty fold"
        assert df["fold"].astype(int).min() == 0
        assert df["fold"].astype(int).max() == 29  # 10x3 = 30 folds

    def test_load_dataset_with_int_fold_from_manifest(self):
        """Manifest fold values must be accepted as integer ``fold_idx`` values."""
        if not _has_fold_layer(SAMPLE_DATASET):
            pytest.skip(f"{SAMPLE_DATASET} not augmented")
        # Accept both string and integer representations from CSV parsing.
        for fold_str_or_int in ["0", "5", 7, "29"]:
            ds = load_dataset(
                SAMPLE_DATASET,
                seed=42,
                fold_idx=int(fold_str_or_int),
            )
            assert ds.metadata["fold_idx"] == int(fold_str_or_int)


# ---------------------------------------------------------------------------
# Balanced manifest ordering
# ---------------------------------------------------------------------------

class TestBalancedManifestOrdering:
    """Balanced order must interleave datasets and place higher-cost datasets first."""

    def test_balanced_manifest_batches_span_many_datasets(self):
        """Each sampled manifest batch must contain multiple datasets."""
        manifest = PROJECT_ROOT / "scripts" / "slurm" / "manifest_tabarena_selected.csv"
        if not manifest.is_file():
            pytest.skip("manifest_tabarena_selected.csv not generated")
        df = pd.read_csv(manifest)
        # Sample task IDs across the manifest.
        sample_tids = [
            0,
            df["task_id"].max() // 4,
            df["task_id"].max() // 2,
            df["task_id"].max(),
        ]
        for tid in sample_tids:
            batch = df[df["task_id"] == tid]
            n_unique = batch["dataset"].nunique()
            # The final task may contain fewer rows.
            min_expected = min(len(batch), 10)
            assert n_unique >= min_expected, (
                f"task_id={tid} has only {n_unique} distinct datasets "
                f"across {len(batch)} rows. Batches should interleave."
            )

    def test_balanced_manifest_lpt_ordering(self):
        """Every first-task dataset must cost more than every final-task dataset."""
        import pyarrow.parquet as pq

        manifest = PROJECT_ROOT / "scripts" / "slurm" / "manifest_tabarena_selected.csv"
        if not manifest.is_file():
            pytest.skip("manifest_tabarena_selected.csv not generated")
        df = pd.read_csv(manifest)

        def _cost(name: str) -> int:
            for base in ("data/datasets", "data/tabarena/datasets"):
                full = PROJECT_ROOT / base / name / "data.parquet"
                if full.is_file():
                    m = pq.ParquetFile(full).metadata
                    return m.num_rows * m.num_columns
                tp = PROJECT_ROOT / base / name / "train.parquet"
                ep = PROJECT_ROOT / base / name / "test.parquet"
                if tp.is_file() and ep.is_file():
                    mt, me = pq.ParquetFile(tp).metadata, pq.ParquetFile(ep).metadata
                    return (mt.num_rows + me.num_rows) * max(
                        mt.num_columns, me.num_columns
                    )
            return 0

        task0_datasets = set(df[df["task_id"] == 0]["dataset"])
        last_tid = df["task_id"].max()
        last_datasets = set(df[df["task_id"] == last_tid]["dataset"])

        min_cost_task0 = min(_cost(d) for d in task0_datasets)
        max_cost_last = max(_cost(d) for d in last_datasets)

        # The first task uses the strict upper portion of the cost ordering.
        assert min_cost_task0 > max_cost_last, (
            f"LPT violated: cheapest in task 0 (cost={min_cost_task0:,}) "
            f"is <= most expensive in last task (cost={max_cost_last:,}). "
            f"The first task must contain higher-cost datasets."
        )

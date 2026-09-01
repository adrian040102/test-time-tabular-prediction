"""
Tests for the TabReD infrastructure: preprocessing scripts, parquet loader
``time_col`` propagation and the gap-widening sampler.

Each preprocessing-script test builds a small synthetic raw-input fixture in
the format expected upstream, invokes the script via
``subprocess`` (so argparse + main path is exercised), then verifies the
output schema (train.parquet / test.parquet / meta.json).

The Sberbank and Home Credit scripts have schema-only import tests because their
raw inputs require several files or a nested directory. End-to-end validation
requires the separately obtained source data described in ``DATASETS.md``.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_DIR = PROJECT_ROOT / "scripts" / "preprocessing"
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.base import TabularDataset
from src.data.tabred_sampling import (
    default_gap_sizes,
    enumerate_gap_windows,
    sample_tabred,
    slice_gap_dataset,
)


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #

def _run_preprocessing(
    script_name: str,
    raw_dir: Path,
    output_dir: Path,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Invoke a preprocessing script as a subprocess and return the result."""
    script_path = PREPROCESSING_DIR / script_name
    assert script_path.is_file(), f"missing script {script_path}"
    args = [
        sys.executable,
        str(script_path),
        "--raw-dir",
        str(raw_dir),
        "--output-dir",
        str(output_dir),
        "--overwrite",
    ]
    if extra_args:
        args.extend(extra_args)
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _assert_valid_output(output_dir: Path) -> dict:
    """Common schema assertions on a freshly-built dataset directory."""
    assert (output_dir / "train.parquet").is_file(), "train.parquet not written"
    assert (output_dir / "test.parquet").is_file(), "test.parquet not written"
    assert (output_dir / "meta.json").is_file(), "meta.json not written"

    with open(output_dir / "meta.json") as f:
        meta = json.load(f)

    assert meta["task_type"] in ("classification", "regression")
    assert meta["time_col"] == "__time__"
    assert meta["n_train"] > 0
    assert meta["n_test"] > 0
    assert meta["n_features"] == len(meta["cat_features"]) + len(meta["num_features"])

    train = pd.read_parquet(output_dir / "train.parquet")
    test = pd.read_parquet(output_dir / "test.parquet")
    assert "__target__" in train.columns
    assert "__time__" in train.columns
    assert "__target__" in test.columns
    assert "__time__" in test.columns
    # train ends strictly before test begins
    assert train["__time__"].max() < test["__time__"].min(), (
        "Train/test temporal windows overlap"
    )
    return meta


# --------------------------------------------------------------------------- #
# _common.py helpers                                                          #
# --------------------------------------------------------------------------- #

class TestCommonHelpers:
    """Standalone tests for the shared helpers in _common.py."""

    def test_ordinal_encode_collapses_rare_cats(self):
        sys.path.insert(0, str(PREPROCESSING_DIR))
        try:
            common = importlib.import_module("_common")
        finally:
            sys.path.pop(0)
        # 100 rows: A x 60, B x 38, C x 1, D x 1. min_frequency=1/50 → both
        # C and D are below the 2-count threshold and get folded into a
        # shared infrequent bucket, as documented by scikit-learn.
        df = pd.DataFrame({"x": ["A"] * 60 + ["B"] * 38 + ["C", "D"]})
        out = common.ordinal_encode(df, ["x"], min_frequency=1 / 50)
        assert out["x"].dtype == np.int64
        # 3 distinct codes: A, B and one shared infrequent bucket for {C, D}.
        assert out["x"].nunique() == 3
        # C and D must share a code (the infrequent bucket).
        c_code = out["x"].iloc[-2]
        d_code = out["x"].iloc[-1]
        assert c_code == d_code, (
            f"C and D should fold into the same infrequent bucket, got "
            f"{c_code} and {d_code}"
        )

    def test_ordinal_encode_handles_missing(self):
        sys.path.insert(0, str(PREPROCESSING_DIR))
        try:
            common = importlib.import_module("_common")
        finally:
            sys.path.pop(0)
        df = pd.DataFrame({"x": ["A", "B", None, "A", "B"]})
        out = common.ordinal_encode(df, ["x"], min_frequency=1 / 10)
        # All codes must be non-negative ints (NaN lifted to max+1).
        assert (out["x"] >= 0).all()
        assert out["x"].dtype == np.int64

    def test_default_temporal_split_drops_val(self):
        sys.path.insert(0, str(PREPROCESSING_DIR))
        try:
            common = importlib.import_module("_common")
        finally:
            sys.path.pop(0)
        df = pd.DataFrame({
            "__time__": pd.to_datetime([
                "2024-01-01", "2024-01-15",
                "2024-02-05", "2024-02-20",
                "2024-03-10", "2024-04-01",
            ]),
            "__target__": [0, 1, 0, 1, 0, 1],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        spec = common.SplitSpec(val_start="2024-02-01", test_start="2024-03-01")
        train, test = common.default_temporal_split(df, spec)
        assert len(train) == 2  # 01-01, 01-15
        assert len(test) == 2  # 03-10, 04-01
        # Val rows excluded entirely.
        assert all(train["__time__"] < pd.Timestamp("2024-02-01"))
        assert all(test["__time__"] >= pd.Timestamp("2024-03-01"))


# --------------------------------------------------------------------------- #
# Preprocessing scripts (with synthetic raw-input fixtures)                   #
# --------------------------------------------------------------------------- #

class TestWeatherScript:
    """Run tabred_weather.py with a fixture-generated input."""

    def _build_raw(self, raw_dir: Path) -> None:
        # Microsecond-since-epoch fact_time spanning Jan→Aug 2023 so the
        # default split (2023-06 / 2023-07) produces a non-empty train+test.
        n = 500
        rng = np.random.default_rng(0)
        # fact_time is stored as Unix seconds in the source and preprocessing output.
        times_ns = pd.date_range("2023-01-01", "2023-08-15", periods=n).view("int64")
        times_s = times_ns // 10**9  # ns → s
        df = pd.DataFrame({
            "fact_time": times_s,
            "fact_temperature": rng.normal(15, 5, n).astype("float32"),
            "fact_latitude": rng.uniform(40, 60, n).astype("float32"),
            "fact_longitude": rng.uniform(20, 50, n).astype("float32"),
            "fact_station_id": rng.integers(0, 100, n).astype("int64"),
            "apply_time_rl": rng.integers(0, 86400, n).astype("int64"),
            "num_0": rng.normal(0, 1, n).astype("float32"),
            "num_1": rng.uniform(0, 100, n).astype("float32"),
            "bin_available_0": rng.integers(0, 2, n).astype("int64"),
            "bin_available_1": rng.integers(0, 2, n).astype("int64"),
        })
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_dir / "weather.parquet", index=False)

    def test_runs_and_emits_valid_schema(self, tmp_path):
        raw = tmp_path / "raw"
        out = tmp_path / "tabred_weather"
        self._build_raw(raw)
        # subsample-frac=1.0 retains all 500 synthetic rows.
        r = _run_preprocessing(
            "tabred_weather.py", raw, out, extra_args=["--subsample-frac", "1.0"]
        )
        assert r.returncode == 0, f"stderr: {r.stderr}\nstdout: {r.stdout}"
        meta = _assert_valid_output(out)
        assert meta["task_type"] == "regression"
        # Bin-available cols should be in features.
        assert any("available" in c for c in meta["num_features"])
        assert meta["cat_features"] == []  # weather has no cat features


class TestCookingTimeScript:
    def _build_raw(self, raw_dir: Path) -> None:
        n = 600
        rng = np.random.default_rng(1)
        # Time spanning Nov 2023 → mid Jan 2024 so default split (12-21 / 12-28) hits.
        times = pd.date_range("2023-11-01", "2024-01-10", periods=n)
        df = pd.DataFrame({
            "timestamp": times,
            "cooking_time_minutes": rng.uniform(2, 90, n).astype("float32"),
            "cat_0": rng.integers(0, 50, n).astype("int64"),  # excluded
            "cat_1": rng.integers(0, 5, n).astype("int64"),
            "cat_2": rng.integers(0, 5, n).astype("int64"),  # excluded
            "cat_3": rng.integers(0, 5, n).astype("int64"),  # excluded
            "cat_4": rng.integers(0, 6, n).astype("int64"),
            "num_0": rng.normal(0, 1, n).astype("float32"),
            "num_1": rng.uniform(0, 10, n).astype("float32"),
            "bin_0": rng.integers(0, 2, n).astype("int64"),
        })
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_dir / "cooking_time.parquet", index=False)

    def test_runs_and_emits_valid_schema(self, tmp_path):
        raw = tmp_path / "raw"
        out = tmp_path / "tabred_cooking_time"
        self._build_raw(raw)
        r = _run_preprocessing(
            "tabred_cooking_time.py", raw, out, extra_args=["--subsample-frac", "1.0"]
        )
        assert r.returncode == 0, f"stderr: {r.stderr}\nstdout: {r.stdout}"
        meta = _assert_valid_output(out)
        assert meta["task_type"] == "regression"
        # cat_1 and cat_4 should be cat features. Cat_0/cat_2/cat_3 excluded.
        assert "cat_1" in meta["cat_features"]
        assert "cat_4" in meta["cat_features"]
        assert "cat_0" not in meta["cat_features"]
        assert "cat_2" not in meta["cat_features"]
        assert "cat_3" not in meta["cat_features"]


class TestDeliveryEtaScript:
    def _build_raw(self, raw_dir: Path) -> None:
        n = 800
        rng = np.random.default_rng(2)
        times = pd.date_range("2023-11-01", "2024-01-05", periods=n)
        df = pd.DataFrame({
            "timestamp": times,
            "delivery_eta_minutes": rng.uniform(2, 60, n).astype("float32"),
            "cat_0": rng.integers(0, 5, n).astype("int64"),
            "cat_1": rng.integers(0, 5, n).astype("int64"),  # excluded
            "cat_2": rng.integers(0, 5, n).astype("int64"),  # excluded
            "cat_3": rng.integers(0, 5, n).astype("int64"),
            "num_0": rng.normal(0, 1, n).astype("float32"),
            "bin_0": rng.integers(0, 2, n).astype("int64"),
        })
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_dir / "delivery_eta.parquet", index=False)

    def test_runs_and_emits_valid_schema(self, tmp_path):
        raw = tmp_path / "raw"
        out = tmp_path / "tabred_delivery_eta"
        self._build_raw(raw)
        r = _run_preprocessing(
            "tabred_delivery_eta.py", raw, out, extra_args=["--subsample-frac", "1.0"]
        )
        assert r.returncode == 0, f"stderr: {r.stderr}\nstdout: {r.stdout}"
        meta = _assert_valid_output(out)
        assert meta["task_type"] == "regression"
        assert "cat_1" not in meta["cat_features"]
        assert "cat_2" not in meta["cat_features"]


class TestMapsRoutingScript:
    def _build_raw(self, raw_dir: Path) -> None:
        n = 700
        rng = np.random.default_rng(3)
        times = pd.date_range("2023-10-15", "2023-12-15", periods=n)
        df = pd.DataFrame({
            "timestamp": times,
            "target_log_spkm": rng.normal(0, 1, n).astype("float32"),
            "track_length": rng.uniform(100, 5000, n).astype("float32"),
            "cat_0": rng.integers(0, 5, n).astype("int64"),
            "cat_1": rng.integers(0, 5, n).astype("int64"),
            "num_0": rng.normal(0, 1, n).astype("float32"),
            "num_1": rng.uniform(0, 10, n).astype("float32"),
        })
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(raw_dir / "maps_routing.parquet", index=False)

    def test_runs_and_emits_valid_schema(self, tmp_path):
        raw = tmp_path / "raw"
        out = tmp_path / "tabred_maps_routing"
        self._build_raw(raw)
        r = _run_preprocessing(
            "tabred_maps_routing.py", raw, out, extra_args=["--subsample-frac", "1.0"]
        )
        assert r.returncode == 0, f"stderr: {r.stderr}\nstdout: {r.stdout}"
        meta = _assert_valid_output(out)
        assert meta["task_type"] == "regression"


class TestHomesiteScript:
    def _build_raw(self, raw_dir: Path) -> None:
        n = 1000
        rng = np.random.default_rng(4)
        dates = pd.date_range("2013-01-01", "2015-05-20", periods=n)
        df = pd.DataFrame({
            "QuoteNumber": np.arange(n),
            "Original_Quote_Date": dates.strftime("%Y-%m-%d"),
            "QuoteConversion_Flag": rng.integers(0, 2, n).astype("int64"),
            "Field6": rng.integers(0, 5, n).astype("int64"),
            "Field7": rng.normal(0, 1, n).astype("float32"),
            "Field8A": rng.integers(-1, 5, n).astype("int64"),  # -1s replaced by NaN
            "Field9B": rng.choice([-1, 0, 1, 2], n).astype("int64"),
            "BinaryFlag": rng.choice(["N", "Y"], n),
            "CatString": rng.choice(["alpha", "beta", "gamma"], n),
        })
        raw_dir.mkdir(parents=True, exist_ok=True)
        # Write train.csv inside a zip named train.csv.zip (matches upstream).
        csv_path = raw_dir / "train.csv"
        df.to_csv(csv_path, index=False)
        with zipfile.ZipFile(raw_dir / "train.csv.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname="train.csv")
        csv_path.unlink()  # the script uses the copy in the zip archive

    def test_runs_and_emits_valid_schema(self, tmp_path):
        raw = tmp_path / "raw"
        out = tmp_path / "tabred_homesite"
        self._build_raw(raw)
        r = _run_preprocessing("tabred_homesite.py", raw, out)
        assert r.returncode == 0, f"stderr: {r.stderr}\nstdout: {r.stdout}"
        meta = _assert_valid_output(out)
        assert meta["task_type"] == "classification"
        assert meta["extra"]["n_classes"] == 2
        # CatString with object dtype should appear in cat_features.
        assert "CatString" in meta["cat_features"]
        # Binary N/Y flag and -1-corrected ordinal cols are numerics.
        assert "BinaryFlag" in meta["num_features"]


class TestEcomOffersScript:
    def _build_raw(self, raw_dir: Path) -> None:
        rng = np.random.default_rng(5)
        # offers.csv.gz
        n_offers = 30
        offers = pd.DataFrame({
            "offer": np.arange(n_offers),
            "category": rng.integers(0, 5, n_offers),
            "quantity": rng.integers(1, 5, n_offers),
            "company": rng.integers(0, 10, n_offers),
            "offervalue": rng.uniform(0.5, 5.0, n_offers).astype("float32"),
            "brand": rng.integers(0, 8, n_offers),
        })
        # trainHistory.csv.gz contains one offer per shopper from March to April 2013.
        n_shoppers = 200
        history = pd.DataFrame({
            "id": np.arange(n_shoppers),
            "chain": rng.integers(0, 5, n_shoppers),
            "offer": rng.integers(0, n_offers, n_shoppers),
            "market": rng.integers(0, 3, n_shoppers),
            "repeattrips": rng.integers(0, 3, n_shoppers),
            "repeater": rng.choice(["t", "f"], n_shoppers),
            "offerdate": pd.date_range("2013-03-01", "2013-05-01", periods=n_shoppers).strftime("%Y-%m-%d"),
        })
        # transactions.csv.gz contains five transactions per shopper.
        n_tx = n_shoppers * 5
        tx = pd.DataFrame({
            "id": np.repeat(np.arange(n_shoppers), 5),
            "chain": rng.integers(0, 5, n_tx),
            "dept": rng.integers(0, 20, n_tx),
            "category": rng.integers(0, 5, n_tx),
            "company": rng.integers(0, 10, n_tx),
            "brand": rng.integers(0, 8, n_tx),
            "date": pd.date_range("2013-01-01", "2013-04-01", periods=n_tx).strftime("%Y-%m-%d"),
            "productsize": rng.uniform(1, 100, n_tx).astype("float32"),
            "productmeasure": rng.choice(["OZ", "LB", "EA"], n_tx),
            "purchasequantity": rng.integers(1, 5, n_tx),
            "purchaseamount": rng.uniform(1.0, 30.0, n_tx).astype("float32"),
        })
        raw_dir.mkdir(parents=True, exist_ok=True)
        offers.to_csv(raw_dir / "offers.csv.gz", index=False, compression="gzip")
        history.to_csv(raw_dir / "trainHistory.csv.gz", index=False, compression="gzip")
        tx.to_csv(raw_dir / "transactions.csv.gz", index=False, compression="gzip")

    def test_runs_and_emits_valid_schema(self, tmp_path):
        raw = tmp_path / "raw"
        out = tmp_path / "tabred_ecom_offers"
        self._build_raw(raw)
        r = _run_preprocessing("tabred_ecom_offers.py", raw, out)
        assert r.returncode == 0, f"stderr: {r.stderr}\nstdout: {r.stdout}"
        meta = _assert_valid_output(out)
        assert meta["task_type"] == "classification"
        # Should have aggregated has_bought_* features.
        assert any("has_bought" in c for c in meta["num_features"])


class TestComplexScriptsImport:
    """Import-level tests for the Sberbank and Home Credit scripts.

    Full end-to-end fixtures require large nested input layouts. This test
    verifies that the modules parse and expose the upstream SplitSpec values.
    """

    @pytest.mark.parametrize(
        "module_name, expected_val_start, expected_test_start",
        [
            ("tabred_sberbank", "2014-06-30", "2014-12-01"),
            ("tabred_homecredit", "2020-01-01", "2020-05-01"),
        ],
    )
    def test_split_spec_matches_upstream(
        self, module_name, expected_val_start, expected_test_start
    ):
        sys.path.insert(0, str(PREPROCESSING_DIR))
        try:
            mod = importlib.import_module(module_name)
        finally:
            sys.path.pop(0)
        spec = mod.SPLIT
        assert spec.val_start == expected_val_start
        assert spec.test_start == expected_test_start


# --------------------------------------------------------------------------- #
# Parquet loader: time_col propagation                                        #
# --------------------------------------------------------------------------- #

@pytest.fixture
def tiny_tabred_dataset_dir(tmp_path: Path) -> Path:
    """Build a tiny tabred_*-style dataset folder for loader/sampler tests.

    Schema: 8 train rows + 6 test rows, one numeric + one cat feature,
    binary classification target, monotonic __time__ within each split.
    """
    out = tmp_path / "tabred_synthetic"
    out.mkdir()
    train = pd.DataFrame({
        "num_feat": np.arange(8, dtype=np.float32),
        "cat_feat": np.array([0, 1, 0, 1, 0, 1, 2, 2], dtype=np.int64),
        "__target__": np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        "__time__": pd.to_datetime([f"2024-01-{d:02d}" for d in range(1, 9)]),
    })
    test = pd.DataFrame({
        "num_feat": np.arange(8, 14, dtype=np.float32),
        "cat_feat": np.array([0, 1, 2, 0, 1, 2], dtype=np.int64),
        "__target__": np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
        # Test starts 2024-02-01 (gap of ~3 weeks vs end of train).
        "__time__": pd.to_datetime([f"2024-02-{d:02d}" for d in range(1, 7)]),
    })
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    meta = {
        "name": "tabred_synthetic",
        "task_type": "classification",
        "cat_features": ["cat_feat"],
        "num_features": ["num_feat"],
        "time_col": "__time__",
        "shift_description": "tabred synthetic for testing",
        "n_train": 8,
        "n_test": 6,
        "n_features": 2,
        "extra": {"source": "tabred", "test_fixture": True},
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out


class TestParquetLoaderTimeCol:
    def test_load_drops_time_col_from_features(self, tiny_tabred_dataset_dir):
        from src.data.parquet_loader import load_parquet_dataset
        ds = load_parquet_dataset(
            "tabred_synthetic", data_root=tiny_tabred_dataset_dir.parent
        )
        # Feature matrices contain only declared features and exclude __time__.
        assert "__time__" not in ds.X_train.columns
        assert "__time__" not in ds.X_test.columns
        assert set(ds.X_train.columns) == {"num_feat", "cat_feat"}

    def test_time_arrays_in_metadata(self, tiny_tabred_dataset_dir):
        from src.data.parquet_loader import load_parquet_dataset
        ds = load_parquet_dataset(
            "tabred_synthetic", data_root=tiny_tabred_dataset_dir.parent
        )
        assert ds.metadata["time_col"] == "__time__"
        assert len(ds.metadata["time_train"]) == ds.n_train
        assert len(ds.metadata["time_test"]) == ds.n_test
        # Monotonic train, monotonic test, train < test.
        assert np.all(np.diff(ds.metadata["time_train"].astype("datetime64[ns]")) > np.timedelta64(0))
        assert ds.metadata["time_train"].max() < ds.metadata["time_test"].min()

    def test_load_dataset_explicit_dispatch(self, tiny_tabred_dataset_dir, monkeypatch):
        """Verify the tabred_* fallback raises a clear error when no parquet exists."""
        from src.data import load_dataset
        # tabred_ghost is not present anywhere. Should raise FileNotFoundError
        # mentioning the preprocessing script.
        with pytest.raises(FileNotFoundError, match="tabred_ghost.py"):
            load_dataset("tabred_ghost", data_root=tiny_tabred_dataset_dir.parent / "void")

    def test_loader_unaffected_for_non_tabred_datasets(self, tmp_path):
        """time_col logic must be a no-op when meta.json does not declare it."""
        out = tmp_path / "plain_dataset"
        out.mkdir()
        train = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "__target__": [0, 1, 0],
        })
        test = pd.DataFrame({
            "x": [4.0, 5.0],
            "__target__": [1, 0],
        })
        train.to_parquet(out / "train.parquet", index=False)
        test.to_parquet(out / "test.parquet", index=False)
        meta = {
            "name": "plain_dataset",
            "task_type": "classification",
            "cat_features": [],
            "num_features": ["x"],
            "n_train": 3, "n_test": 2, "n_features": 1,
            "shift_description": "",
            "extra": {},
        }
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        from src.data.parquet_loader import load_parquet_dataset
        ds = load_parquet_dataset("plain_dataset", data_root=tmp_path)
        assert "time_col" not in ds.metadata
        assert "time_train" not in ds.metadata
        assert list(ds.X_train.columns) == ["x"]


# --------------------------------------------------------------------------- #
# Gap-widening sampler                                                        #
# --------------------------------------------------------------------------- #

@pytest.fixture
def loaded_tiny_dataset(tiny_tabred_dataset_dir) -> TabularDataset:
    from src.data.parquet_loader import load_parquet_dataset
    return load_parquet_dataset(
        "tabred_synthetic", data_root=tiny_tabred_dataset_dir.parent
    )


class TestSampler:
    def test_basic_shrink(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        sampled = sample_tabred(ds, x_train_first=3, y_test_last=2)
        assert sampled.n_train == 3
        assert sampled.n_test == 2
        # Training rows are the three earliest training times.
        sampled_train_times = pd.to_datetime(sampled.metadata["time_train"])
        original_train_times = pd.to_datetime(ds.metadata["time_train"])
        np.testing.assert_array_equal(
            sampled_train_times.sort_values().values,
            np.sort(original_train_times.values)[:3],
        )
        # Test rows are the two latest test times.
        sampled_test_times = pd.to_datetime(sampled.metadata["time_test"])
        original_test_times = pd.to_datetime(ds.metadata["time_test"])
        np.testing.assert_array_equal(
            sampled_test_times.sort_values().values,
            np.sort(original_test_times.values)[-2:],
        )

    def test_widens_temporal_gap(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        pre_gap = pd.Timestamp(ds.metadata["time_test"].min()) - pd.Timestamp(
            ds.metadata["time_train"].max()
        )
        sampled = sample_tabred(ds, x_train_first=2, y_test_last=2)
        post_gap = pd.Timestamp(sampled.metadata["time_test"].min()) - pd.Timestamp(
            sampled.metadata["time_train"].max()
        )
        assert post_gap > pre_gap, f"sampler should widen gap. pre={pre_gap}, post={post_gap}"

    def test_y_train_aligned_after_slice(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        sampled = sample_tabred(ds, x_train_first=4, y_test_last=4)
        assert len(sampled.y_train) == sampled.n_train
        assert len(sampled.y_test) == sampled.n_test
        # Row-major bijection: the earliest 4 train rows by time should match
        # the y_train of the same row indices in the source.
        order = np.argsort(ds.metadata["time_train"], kind="stable")[:4]
        np.testing.assert_array_equal(
            np.sort(sampled.y_train), np.sort(np.asarray(ds.y_train)[order])
        )

    def test_clipping_when_request_exceeds_size(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        sampled = sample_tabred(ds, x_train_first=999, y_test_last=999)
        # Requests above the available size return the complete splits.
        assert sampled.n_train == ds.n_train
        assert sampled.n_test == ds.n_test

    def test_rejects_non_positive_sizes(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        with pytest.raises(ValueError):
            sample_tabred(ds, x_train_first=0, y_test_last=3)
        with pytest.raises(ValueError):
            sample_tabred(ds, x_train_first=3, y_test_last=0)
        with pytest.raises(ValueError):
            sample_tabred(ds, x_train_first=-1, y_test_last=3)

    def test_rejects_dataset_without_time_col(self):
        ds = TabularDataset(
            name="no_time",
            X_train=pd.DataFrame({"x": [1.0, 2.0]}),
            X_test=pd.DataFrame({"x": [3.0, 4.0]}),
            y_train=np.array([0, 1]),
            y_test=np.array([0, 1]),
            cat_features=[],
            num_features=["x"],
            task_type="classification",
        )
        with pytest.raises(ValueError, match="time_col"):
            sample_tabred(ds, x_train_first=1, y_test_last=1)

    def test_metadata_records_sample_spec(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        sampled = sample_tabred(ds, x_train_first=5, y_test_last=3)
        spec = sampled.metadata["sample_spec"]
        assert spec["x_train_first"] == 5
        assert spec["y_test_last"] == 3
        assert spec["n_train_pre_sample"] == ds.n_train
        assert spec["n_test_pre_sample"] == ds.n_test

    def test_output_name_is_traceable(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        sampled = sample_tabred(ds, x_train_first=4, y_test_last=2)
        assert sampled.name == f"{ds.name}__sample_4_2"

    def test_does_not_mutate_input(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        pre_n_train = ds.n_train
        pre_n_test = ds.n_test
        pre_train_time = np.array(ds.metadata["time_train"], copy=True)
        _ = sample_tabred(ds, x_train_first=2, y_test_last=2)
        assert ds.n_train == pre_n_train
        assert ds.n_test == pre_n_test
        np.testing.assert_array_equal(np.array(ds.metadata["time_train"]), pre_train_time)


# Alternative training periods for a fixed test period

class TestGapSizes:
    def test_caps_and_floors(self):
        # Big dataset -> capped at (8000, 5000).
        assert default_gap_sizes(300_000, 40_000) == (8000, 5000)
        # Sberbank-like -> n_train = n//3, n_test = n//2.
        assert default_gap_sizes(18_847, 4_647) == (6282, 2323)
        # Small -> floors (2500 train, 1000 test).
        assert default_gap_sizes(3_000, 1_000) == (2500, 1000)

    def test_extreme_windows_dont_overlap_in_third_branch(self):
        # The //3 rule guarantees 2*n_train <= n_total so the earliest and
        # latest windows are disjoint, which is the purpose of using equal sizes.
        for total in (18_847, 24_000, 30_000, 240_000):
            n_train, _ = default_gap_sizes(total, 5_000)
            if n_train == total // 3:
                assert 2 * n_train <= total


class TestEnumerateGapWindows:
    def test_fixes_latest_test_window(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        plan = enumerate_gap_windows(ds, n_train=4, n_test=2, n_positions=3)
        all_test = np.asarray(ds.metadata["time_test"])
        sel = all_test[plan.test_idx]
        np.testing.assert_array_equal(
            np.sort(pd.to_datetime(sel).values),
            np.sort(pd.to_datetime(all_test).values)[-2:],
        )

    def test_windows_have_equal_sizes(self, loaded_tiny_dataset):
        plan = enumerate_gap_windows(loaded_tiny_dataset, 4, 2, n_positions=3)
        assert len(plan.windows) >= 2
        assert all(len(w.train_idx) == 4 for w in plan.windows)
        assert len(plan.test_idx) == 2
        assert plan.n_train_total == 8 and plan.n_test_total == 6

    def test_all_train_before_test(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        plan = enumerate_gap_windows(ds, 4, 2, n_positions=3)
        t_test = np.asarray(ds.metadata["time_test"])[plan.test_idx]
        t_train_all = np.asarray(ds.metadata["time_train"])
        for w in plan.windows:
            assert pd.Timestamp(t_train_all[w.train_idx].max()) < pd.Timestamp(t_test.min())

    def test_oldest_and_newest_windows(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        plan = enumerate_gap_windows(ds, 4, 2, n_positions=3)
        order = np.argsort(ds.metadata["time_train"], kind="stable")
        oldest = min(plan.windows, key=lambda w: w.start_position)
        newest = max(plan.windows, key=lambda w: w.start_position)
        np.testing.assert_array_equal(oldest.train_idx, order[:4])
        np.testing.assert_array_equal(newest.train_idx, order[-4:])

    def test_deterministic(self, loaded_tiny_dataset):
        p1 = enumerate_gap_windows(loaded_tiny_dataset, 4, 2, n_positions=3)
        p2 = enumerate_gap_windows(loaded_tiny_dataset, 4, 2, n_positions=3)
        np.testing.assert_array_equal(p1.test_idx, p2.test_idx)
        for a, b in zip(p1.windows, p2.windows):
            np.testing.assert_array_equal(a.train_idx, b.train_idx)

    def test_rejects_invalid_args(self, loaded_tiny_dataset):
        with pytest.raises(ValueError):
            enumerate_gap_windows(loaded_tiny_dataset, 0, 2)
        with pytest.raises(ValueError):
            enumerate_gap_windows(loaded_tiny_dataset, 4, 0)
        with pytest.raises(ValueError):
            enumerate_gap_windows(loaded_tiny_dataset, 4, 2, n_positions=0)

    def test_requires_time_metadata(self):
        ds = TabularDataset(
            name="no_time",
            X_train=pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}),
            X_test=pd.DataFrame({"x": [5.0, 6.0]}),
            y_train=np.array([0, 1, 0, 1]),
            y_test=np.array([0, 1]),
            cat_features=[],
            num_features=["x"],
            task_type="classification",
        )
        with pytest.raises(ValueError, match="time_col"):
            enumerate_gap_windows(ds, 2, 1)


class TestSliceGapDataset:
    def test_shapes_and_y_alignment(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        plan = enumerate_gap_windows(ds, 4, 2, n_positions=3)
        w = plan.windows[0]
        sliced = slice_gap_dataset(ds, w.train_idx, plan.test_idx, "gap_test")
        assert sliced.n_train == 4 and sliced.n_test == 2
        assert sliced.name == f"{ds.name}__gap_test"
        assert sliced.metadata["gap_version"] == "gap_test"
        np.testing.assert_array_equal(
            np.asarray(sliced.y_train), np.asarray(ds.y_train)[w.train_idx]
        )
        np.testing.assert_array_equal(
            np.asarray(sliced.metadata["time_train"]),
            np.asarray(ds.metadata["time_train"])[w.train_idx],
        )

    def test_no_temporal_overlap(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        plan = enumerate_gap_windows(ds, 4, 2, n_positions=3)
        for w in plan.windows:
            sliced = slice_gap_dataset(ds, w.train_idx, plan.test_idx, "v")
            assert pd.Timestamp(np.max(sliced.metadata["time_train"])) < pd.Timestamp(
                np.min(sliced.metadata["time_test"])
            )

    def test_handles_none_y_test(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        ds2 = TabularDataset(
            name=ds.name,
            X_train=ds.X_train,
            X_test=ds.X_test,
            y_train=ds.y_train,
            y_test=None,
            cat_features=ds.cat_features,
            num_features=ds.num_features,
            task_type=ds.task_type,
            metadata=ds.metadata,
        )
        plan = enumerate_gap_windows(ds, 4, 2, n_positions=3)
        sliced = slice_gap_dataset(ds2, plan.windows[0].train_idx, plan.test_idx, "v")
        assert sliced.y_test is None
        assert sliced.n_train == 4

    def test_does_not_mutate_input(self, loaded_tiny_dataset):
        ds = loaded_tiny_dataset
        pre_time = np.array(ds.metadata["time_train"], copy=True)
        plan = enumerate_gap_windows(ds, 4, 2, n_positions=3)
        _ = slice_gap_dataset(ds, plan.windows[0].train_idx, plan.test_idx, "v")
        np.testing.assert_array_equal(np.array(ds.metadata["time_train"]), pre_time)
        assert ds.n_train == 8 and ds.n_test == 6

"""Data loading utilities."""

import importlib
from pathlib import Path

from src.data.base import TabularDataset, LeakageGuard

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "datasets"

# These five datasets are fixed, stored experiment inputs. Their experiment
# identity uses seed 0, while the immutable split itself was generated with
# seed 42. Prefer the packaged parquet files so a model seed can never
# accidentally regenerate a different dataset split.
_CONTROLLED_SYNTHETIC_DATASETS = {
    "synthetic_shift_0.0",
    "synthetic_shift_0.5",
    "synthetic_shift_1.0",
    "synthetic_shift_3.0",
    "synthetic_shift_5.0",
}

# Registry of dataset loaders for CLI access
DATASET_REGISTRY = {
    # --- Synthetic ---
    "synthetic": "src.data.synthetic:load_synthetic",
    # --- Folktables (US Census, all 5 tasks) ---
    "folktables_income": "src.data.folktables:load_folktables_income",
    "folktables_coverage": "src.data.folktables:load_folktables_coverage",
    "folktables_employment": "src.data.folktables:load_folktables_employment",
    "folktables_travel_time": "src.data.folktables:load_folktables_travel_time",
    "folktables_mobility": "src.data.folktables:load_folktables_mobility",
    # --- UCI standalone datasets ---
    "bike_sharing": "src.data.bike_sharing:load_bike_sharing",
    "gas_sensor_drift": "src.data.gas_sensor_drift:load_gas_sensor_drift",
    "diabetes_hospital": "src.data.diabetes_hospital:load_diabetes_hospital",
    # --- CDC BRFSS (geographic shift by US census region) ---
    "brfss_diabetes": "src.data.brfss:load_brfss_diabetes",
    # --- Lending Club (temporal / financial shift) ---
    "lending_club": "src.data.lending_club:load_lending_club",
    # --- Yandex Shifts ---
    "shifts_weather": "src.data.shifts_weather:load_shifts_weather",
    # --- Tschalzev et al. (NeurIPS 2024 D&B) via OpenML ---
    "tschalzev_mbgm": "src.data.openml_datasets:load_tschalzev_dataset",
    "tschalzev_aeac": "src.data.openml_datasets:load_tschalzev_dataset",
    "tschalzev_ogpcc": "src.data.openml_datasets:load_tschalzev_dataset",
    "tschalzev_sctp": "src.data.openml_datasets:load_tschalzev_dataset",
    # --- TabReD (Rubachev et al., ICLR 2025. ArXiv:2406.19380) ---
    # Eight datasets with natural temporal shift. Each
    # ships as train.parquet + test.parquet + meta.json (with a top-level
    # ``time_col`` field so the gap-widening sampler in
    # ``src.data.tabred_sampling`` can re-slice at sweep time). Build each
    # dataset from raw Kaggle inputs with
    # ``scripts/preprocessing/tabred_<name>.py``. These datasets require
    # separately prepared source files.
    "tabred_sberbank": "src.data.parquet_loader:load_parquet_dataset",
    "tabred_homesite": "src.data.parquet_loader:load_parquet_dataset",
    "tabred_ecom_offers": "src.data.parquet_loader:load_parquet_dataset",
    "tabred_homecredit": "src.data.parquet_loader:load_parquet_dataset",
    "tabred_cooking_time": "src.data.parquet_loader:load_parquet_dataset",
    "tabred_delivery_eta": "src.data.parquet_loader:load_parquet_dataset",
    "tabred_maps_routing": "src.data.parquet_loader:load_parquet_dataset",
    "tabred_weather": "src.data.parquet_loader:load_parquet_dataset",
}


def load_dataset(name: str, **kwargs) -> TabularDataset:
    """Load a dataset by registry name with keyword arguments.

    Tries parquet first (fast, offline, no cache dirs). Falls back to the
    original API-based loaders if parquet is not available. Parameterized
    synthetic datasets use the generator, except for the five fixed controlled
    datasets, which prefer their stored experiment splits.
    """
    # Parquet fast-path. Most synthetic designs remain parameterized and are
    # generated fresh. Only the five fixed controlled inputs use stored files.
    if not name.startswith("synthetic") or name in _CONTROLLED_SYNTHETIC_DATASETS:
        from src.data.parquet_loader import parquet_dataset_exists, load_parquet_dataset
        if parquet_dataset_exists(name):
            pq_kwargs = {
                k: v for k, v in kwargs.items()
                if k in ("seed", "max_samples", "data_root", "fold_idx")
            }
            return load_parquet_dataset(name, **pq_kwargs)

    # Config-based loader for YAML files in configs/datasets/.
    config_path = _CONFIGS_DIR / f"{name}.yaml"
    if config_path.is_file():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        module_path, func_name = cfg["loader"].rsplit(":", 1)
        mod = importlib.import_module(module_path)
        loader_fn = getattr(mod, func_name)
        params = dict(cfg.get("params", {}))
        params.update(kwargs)  # CLI kwargs override config
        return loader_fn(**params)

    # Original loaders for fallback or first-time use.
    if name.startswith("synthetic"):
        from src.data.synthetic import load_synthetic
        if name == "synthetic_label_shift":
            kwargs.setdefault("shift_mode", "label_shift")
        elif name == "synthetic_partial_shift":
            kwargs.setdefault("shift_mode", "partial")
            kwargs.setdefault("shift_strength", 1.0)
        elif name == "synthetic_small_test":
            kwargs.setdefault("n_train", 50_000)
            kwargs.setdefault("n_test", 5_000)
            kwargs.setdefault("shift_strength", 1.0)
            kwargs.setdefault("name_override", "synthetic_small_test")
        elif name == "synthetic_large_test":
            kwargs.setdefault("n_train", 5_000)
            kwargs.setdefault("n_test", 50_000)
            kwargs.setdefault("shift_strength", 1.0)
            kwargs.setdefault("name_override", "synthetic_large_test")
        elif "_shift_" in name:
            from src.data.synthetic import (
                CONTROLLED_SYNTHETIC_N_TEST,
                CONTROLLED_SYNTHETIC_N_TRAIN,
                CONTROLLED_SYNTHETIC_SEED,
                controlled_shift_strength,
            )
            shift = float(name.split("_shift_")[1])
            kwargs.setdefault("shift_strength", shift)
            if controlled_shift_strength(name) is not None:
                kwargs.setdefault("n_train", CONTROLLED_SYNTHETIC_N_TRAIN)
                kwargs.setdefault("n_test", CONTROLLED_SYNTHETIC_N_TEST)
                kwargs.setdefault("seed", CONTROLLED_SYNTHETIC_SEED)
        return load_synthetic(**kwargs)

    elif name.startswith("folktables_income"):
        from src.data.folktables import load_folktables_income
        # Parse years from name like "folktables_income_2014_2018"
        parts = name.split("_")
        if len(parts) >= 4:
            kwargs.setdefault("train_year", parts[2])
            kwargs.setdefault("test_year", parts[3])
        return load_folktables_income(**kwargs)

    elif name.startswith("folktables_coverage"):
        from src.data.folktables import load_folktables_coverage
        parts = name.split("_")
        if len(parts) >= 4:
            kwargs.setdefault("train_year", parts[2])
            kwargs.setdefault("test_year", parts[3])
        return load_folktables_coverage(**kwargs)

    elif name.startswith("folktables_employment"):
        from src.data.folktables import load_folktables_employment
        parts = name.split("_")
        if len(parts) >= 4:
            kwargs.setdefault("train_year", parts[2])
            kwargs.setdefault("test_year", parts[3])
        return load_folktables_employment(**kwargs)

    elif name.startswith("folktables_travel_time"):
        from src.data.folktables import load_folktables_travel_time
        parts = name.split("_")
        if len(parts) >= 5:
            kwargs.setdefault("train_year", parts[3])
            kwargs.setdefault("test_year", parts[4])
        return load_folktables_travel_time(**kwargs)

    elif name.startswith("folktables_mobility"):
        from src.data.folktables import load_folktables_mobility
        parts = name.split("_")
        if len(parts) >= 4:
            kwargs.setdefault("train_year", parts[2])
            kwargs.setdefault("test_year", parts[3])
        return load_folktables_mobility(**kwargs)

    elif name.startswith("diabetes_hospital"):
        from src.data.diabetes_hospital import load_diabetes_hospital
        return load_diabetes_hospital(**kwargs)

    elif name.startswith("brfss_"):
        from src.data.brfss import load_brfss
        task = name.replace("brfss_", "")
        return load_brfss(task=task, **kwargs)

    elif name.startswith("lending_club"):
        from src.data.lending_club import load_lending_club
        return load_lending_club(**kwargs)

    elif name.startswith("bike_sharing"):
        from src.data.bike_sharing import load_bike_sharing
        return load_bike_sharing(**kwargs)

    elif name.startswith("gas_sensor_drift"):
        from src.data.gas_sensor_drift import load_gas_sensor_drift
        return load_gas_sensor_drift(**kwargs)

    elif name.startswith("shifts_weather"):
        from src.data.shifts_weather import load_shifts_weather
        return load_shifts_weather(**kwargs)

    elif name.startswith("tschalzev_"):
        from src.data.openml_datasets import load_tschalzev_dataset
        short_name = name.replace("tschalzev_", "")
        return load_tschalzev_dataset(short_name=short_name, **kwargs)

    elif name.startswith("openml_"):
        from src.data.openml_datasets import load_openml_dataset
        openml_id = int(name.replace("openml_", ""))
        return load_openml_dataset(openml_id=openml_id, **kwargs)

    elif name.startswith("tabred_"):
        # TabReD datasets only resolve through the parquet fast-path above.
        # Reaching this branch means that per-dataset preprocessing has not run.
        short = name.replace("tabred_", "")
        raise FileNotFoundError(
            f"TabReD dataset '{name}' has no built artifacts under "
            f"data/datasets/{name}/. "
            f"Run `python scripts/preprocessing/tabred_{short}.py` first "
            f"(see thesis/docs/tabred_kaggle_setup.md for raw-data setup)."
        )

    else:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

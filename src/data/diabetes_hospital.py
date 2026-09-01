"""
UCI Diabetes 130-US Hospitals loader with an admission-source shift.

The dataset covers 1999–2008. The configured split trains on emergency-related
admissions and tests on other admission sources.

Task: binary classification (readmitted within 30 days vs. not).

Source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.base import TabularDataset


_DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)

# Admission-source and discharge-disposition IDs are coded categorical variables.

# Features to use (after cleaning)
_CAT_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
]

_NUM_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# Medication columns (binary: changed / not changed)
_MED_FEATURES = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "glipizide", "glyburide", "pioglitazone",
    "rosiglitazone", "acarbose", "insulin",
]


def _group_icd9(code: str) -> str:
    """Map ICD-9 codes into broad disease groups."""
    if code == "?" or pd.isna(code):
        return "unknown"
    try:
        c = float(code.replace("E", "").replace("V", ""))
    except (ValueError, AttributeError):
        return "other"

    if code.startswith("V"):
        return "supplementary"
    if code.startswith("E"):
        return "external_causes"
    if 390 <= c <= 459 or c == 785:
        return "circulatory"
    if 460 <= c <= 519 or c == 786:
        return "respiratory"
    if 520 <= c <= 579 or c == 787:
        return "digestive"
    if 250 <= c < 251:
        return "diabetes"
    if 800 <= c <= 999:
        return "injury"
    if 710 <= c <= 739:
        return "musculoskeletal"
    if 580 <= c <= 629 or c == 788:
        return "genitourinary"
    if 140 <= c <= 239:
        return "neoplasms"
    return "other"


def _download_and_cache(cache_dir: str | None = None) -> pd.DataFrame:
    """Download and extract the Diabetes 130-Hospitals dataset."""
    import urllib.request

    if cache_dir is None:
        base = os.environ.get(
            "THESIS_DATA_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "thesis_data"),
        )
        cache_dir = os.path.join(base, "diabetes_hospital")
    os.makedirs(cache_dir, exist_ok=True)

    csv_path = os.path.join(cache_dir, "diabetic_data.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, na_values="?")

    zip_path = os.path.join(cache_dir, "diabetes_hospital.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading Diabetes 130-Hospitals dataset to {cache_dir} ...")
        urllib.request.urlretrieve(_DOWNLOAD_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    # Find the main data CSV (may be in a subdirectory)
    candidates = list(Path(cache_dir).rglob("diabetic_data.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"diabetic_data.csv not found after extraction. "
            f"Contents: {list(Path(cache_dir).rglob('*'))}"
        )
    return pd.read_csv(str(candidates[0]), na_values="?")


def _get_region(row_hospital_id: int) -> str:
    """Region assignment is performed directly in ``load_diabetes_hospital``."""
    # Split logic is implemented in ``load_diabetes_hospital``.
    pass


def load_diabetes_hospital(
    train_regions: list[str] | None = None,
    test_regions: list[str] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    cache_dir: str | None = None,
) -> TabularDataset:
    """
    Load the dataset with an admission-source shift.

    Training uses emergency-related admissions (IDs 1, 2, 3 and 7). Testing
    uses other admission sources.

    Args:
        train_regions: Admission source groups for training.
                       Default: ["emergency"].
        test_regions:  Admission source groups for testing.
                       Default: ["non_emergency"].
        max_samples: Optional cap on samples per split.
        seed: Random seed for subsampling.
        cache_dir: Optional cache directory.

    Returns:
        TabularDataset (task_type="classification", binary readmission).
    """
    if train_regions is None:
        train_regions = ["emergency"]
    if test_regions is None:
        test_regions = ["non_emergency"]

    df = _download_and_cache(cache_dir)

    # Clean and prepare the data.
    # Remove duplicate patient encounters (keep first)
    df = df.drop_duplicates(subset=["patient_nbr"], keep="first")

    # Create binary readmission target
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)

    # Group ICD-9 diagnosis codes
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_group"] = df[col].apply(_group_icd9)

    # Emergency-related admission-source IDs.
    emergency_ids = {1, 2, 3, 7}
    df["admission_group"] = df["admission_source_id"].apply(
        lambda x: "emergency" if x in emergency_ids else "non_emergency"
    )

    # Cast coded IDs to string (they are categorical, not ordinal)
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[col] = df[col].astype(str)

    # Medication columns: simplify to binary (changed vs. not)
    for med in _MED_FEATURES:
        if med in df.columns:
            df[med] = (df[med] != "No").astype(str)

    all_cat = _CAT_FEATURES + _MED_FEATURES
    features = all_cat + _NUM_FEATURES

    # Fill missing categoricals
    for col in all_cat:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)

    # Split by admission group.
    train_mask = df["admission_group"].isin(train_regions)
    test_mask = df["admission_group"].isin(test_regions)

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            f"Empty split. train={train_regions} ({train_mask.sum()} rows), "
            f"test={test_regions} ({test_mask.sum()} rows)."
        )

    available_features = [f for f in features if f in df.columns]
    cat_available = [f for f in all_cat if f in df.columns]
    num_available = [f for f in _NUM_FEATURES if f in df.columns]

    X_train = df.loc[train_mask, available_features].reset_index(drop=True)
    y_train = df.loc[train_mask, "readmitted_30d"].values.astype(int)
    X_test = df.loc[test_mask, available_features].reset_index(drop=True)
    y_test = df.loc[test_mask, "readmitted_30d"].values.astype(int)

    if max_samples:
        X_train, y_train = _subsample(X_train, y_train, max_samples, seed)
        X_test, y_test = _subsample(X_test, y_test, max_samples, seed + 1)

    shift_desc = f"institutional: {'+'.join(train_regions)}→{'+'.join(test_regions)}"

    return TabularDataset(
        name="diabetes_hospital",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cat_features=cat_available,
        num_features=num_available,
        task_type="classification",
        shift_description=shift_desc,
        metadata={
            "source": "uci",
            "task": "diabetes_30day_readmission",
            "n_classes": 2,
            "train_regions": train_regions,
            "test_regions": test_regions,
        },
    )


def _subsample(
    X: pd.DataFrame, y: np.ndarray, n: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(X) <= n:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X.iloc[idx].reset_index(drop=True), y[idx]

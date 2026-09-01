"""Save selected method inputs and outputs for inspection.

A capture stores the pre-transform, post-method and post-pipeline state for one
experiment. A YAML file selects the ``(dataset, method, model, seed)`` runs.

Capture YAML shape:

    captures:
      - {dataset: credit-g, method: inductive_baseline, model: tabpfn_v2, seed: 0, fold: 0}
      - ...

``fold`` is optional. If it is omitted, the worker saves every fold under a
fold-specific filename.

``scripts/slurm/worker.py`` and ``scripts/run_experiment.py`` accept
`--capture-config <yaml>` and `--capture-dir <dir>`. For each run, if the
tuple is in the capture set, ``run_single_experiment`` saves an artifact to
`<dir>/{dataset}__{method}__{model}__s{seed}.pkl`.

Saved captures can be loaded with :func:`load_artifact`.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.methods.base import RowProvenance


__all__ = [
    "CaptureKey",
    "load_audit",
    "build_artifact",
    "save_artifact",
    "load_artifact",
    "artifact_filename",
]

# Number of head rows to snapshot from each split. This is sufficient for
# inspecting transformations without reproducing the complete dataset.
HEAD_N = 50


@dataclass(frozen=True)
class CaptureKey:
    """Identify one experiment run for audit-set lookup."""

    dataset: str
    method: str
    model: str
    seed: int
    fold: int | None = None

    def as_tuple(self) -> tuple[str, str, str, int, int | None]:
        return (
            self.dataset,
            self.method,
            self.model,
            int(self.seed),
            None if self.fold is None else int(self.fold),
        )


def load_audit(path: str | Path) -> set[tuple[str, str, str, int, int | None]]:
    """Parse capture YAML into (dataset, method, model, seed, fold) tuples.

    A ValueError for unknown keys identifies typos instead of silently skipping
    a requested capture.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    entries = cfg.get("captures", [])
    required = {"dataset", "method", "model", "seed"}
    allowed = required | {"fold"}
    audit: set[tuple[str, str, str, int, int | None]] = set()
    for i, e in enumerate(entries):
        missing = required - set(e)
        extra = set(e) - allowed
        if missing:
            raise ValueError(f"Capture entry {i} missing keys: {missing} (got {e})")
        if extra:
            raise ValueError(f"Capture entry {i} has unknown keys: {extra} (allowed: {allowed})")
        fold = None if e.get("fold") is None else int(e["fold"])
        audit.add((
            str(e["dataset"]), str(e["method"]), str(e["model"]),
            int(e["seed"]), fold,
        ))
    return audit


def artifact_filename(key: CaptureKey) -> str:
    """Construct the on-disk filename for a given capture key."""
    fold_tag = "" if key.fold is None else f"__f{key.fold}"
    return f"{key.dataset}__{key.method}__{key.model}__s{key.seed}{fold_tag}.pkl"


def _head_df(arr_or_df, columns, n=HEAD_N):
    """Return first n rows as a DataFrame, using supplied columns when arr is a numpy array."""
    if arr_or_df is None:
        return None
    if hasattr(arr_or_df, "head"):
        return arr_or_df.head(n).copy()
    arr = np.asarray(arr_or_df)
    cols = list(columns) if columns is not None else [f"col_{i}" for i in range(arr.shape[1])]
    if len(cols) != arr.shape[1]:
        cols = [f"col_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr[:n], columns=cols).copy()


def _head_array(arr, n=HEAD_N):
    if arr is None:
        return None
    return np.asarray(arr)[:n].copy()


def _series_sha256(series: pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(series, index=True, categorize=True)
    schema = json.dumps(
        {"name": str(series.name), "dtype": str(series.dtype)},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(schema + hashed.to_numpy().tobytes()).hexdigest()


def _frame_contract(frame: pd.DataFrame) -> dict[str, Any]:
    """Stable integrity/schema summary for a captured model-input frame."""
    hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    schema = json.dumps(
        {
            "columns": [str(c) for c in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "shape": list(frame.shape),
        },
        sort_keys=True,
    ).encode("utf-8")
    numeric = frame.select_dtypes(include=["number"])
    inf_by_column = {
        str(col): int(np.isinf(numeric[col].to_numpy(dtype=float)).sum())
        for col in numeric.columns
    }
    return {
        "shape": tuple(frame.shape),
        "columns": [str(c) for c in frame.columns],
        "dtypes": {str(c): str(frame[c].dtype) for c in frame.columns},
        "missing_by_column": {
            str(c): int(frame[c].isna().sum()) for c in frame.columns
        },
        "inf_by_numeric_column": inf_by_column,
        "sha256": hashlib.sha256(
            schema + hashed.to_numpy().tobytes()
        ).hexdigest(),
        "column_sha256": {
            str(c): _series_sha256(frame[c]) for c in frame.columns
        },
    }


def _array_sha256(arr) -> str | None:
    if arr is None:
        return None
    series = pd.Series(np.asarray(arr).ravel())
    return _series_sha256(series)


@lru_cache(maxsize=1)
def _runtime_fingerprint() -> dict[str, Any]:
    """Hash executable source and lockfiles used by a capture run."""
    thesis_root = Path(__file__).resolve().parent.parent
    paths = sorted((thesis_root / "src").rglob("*.py"))
    for name in ("pyproject.toml", "requirements.lock"):
        path = thesis_root / name
        if path.exists():
            paths.append(path)
    digest = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for path in sorted(set(paths)):
        relative = path.relative_to(thesis_root).as_posix()
        content = path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        file_hashes[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return {
        "source_tree_sha256": digest.hexdigest(),
        "source_file_sha256": file_hashes,
        "python": sys.version,
        "platform": platform.platform(),
    }


def build_artifact(
    *,
    key: CaptureKey,
    task_type: str,
    # Pre-method state
    X_train_pre: pd.DataFrame,
    X_test_pre: pd.DataFrame,
    y_train_pre: np.ndarray,
    cat_features: list[str],
    num_features: list[str],
    # Post-method state (the TransformResult fields)
    X_train_post,
    X_test_post,
    feature_names_post: list[str],
    y_train_post: np.ndarray | None,
    sample_weights_post: np.ndarray | None,
    method_metadata: dict,
    skip_pipeline_scaler: bool,
    # Exact model-boundary state (immediately before model.fit)
    X_train_model_input: pd.DataFrame,
    X_test_model_input: pd.DataFrame,
    y_train_model_input: np.ndarray,
    sample_weights_model_input: np.ndarray | None,
    categorical_features_model_input: list[str],
    row_provenance_train: RowProvenance,
    model_supports_sample_weight: bool,
) -> dict[str, Any]:
    """Assemble the captured-artifact dict.

    The exact pre-fit frames and aligned arrays are stored in full. Pre/post
    previews remain for convenient browsing and stable hashes make silent
    value/schema drift detectable without loading every cell interactively.
    """
    feature_names_pre = list(X_train_pre.columns)
    return {
        "schema_version": 2,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "key": {
            "dataset": key.dataset, "method": key.method,
            "model": key.model, "seed": int(key.seed),
            "fold": None if key.fold is None else int(key.fold),
        },
        "task_type": task_type,
        "runtime_fingerprint": _runtime_fingerprint(),
        "feature_names": {
            "pre": feature_names_pre,
            "post": list(feature_names_post),
            "cat_features": list(cat_features),
            "num_features": list(num_features),
        },
        "shapes": {
            "X_train_pre":   tuple(X_train_pre.shape),
            "X_test_pre":    tuple(X_test_pre.shape),
            "X_train_post":  tuple(np.asarray(X_train_post).shape),
            "X_test_post":   tuple(np.asarray(X_test_post).shape),
            "X_train_model_input": tuple(X_train_model_input.shape),
            "X_test_model_input":  tuple(X_test_model_input.shape),
            "feature_delta_method":  int(np.asarray(X_train_post).shape[1]) - int(X_train_pre.shape[1]),
            "y_train_augmented": y_train_post is not None and len(y_train_post) != len(y_train_pre),
            "sample_weights_present": sample_weights_post is not None,
            "skip_pipeline_scaler": bool(skip_pipeline_scaler),
        },
        "pre_method": {
            "X_train_head": _head_df(X_train_pre, None),
            "X_test_head":  _head_df(X_test_pre, None),
            "y_train_head": _head_array(y_train_pre),
        },
        "post_method": {
            "X_train_head":         _head_df(X_train_post, feature_names_post),
            "X_test_head":          _head_df(X_test_post, feature_names_post),
            "y_train_head":         _head_array(y_train_post),
            "sample_weights_head":  _head_array(sample_weights_post),
            "metadata":             dict(method_metadata or {}),
        },
        "model_input": {
            "X_train": X_train_model_input.copy(deep=True),
            "X_test": X_test_model_input.copy(deep=True),
            "y_train": np.asarray(y_train_model_input).copy(),
            "sample_weights": (
                None if sample_weights_model_input is None
                else np.asarray(sample_weights_model_input).copy()
            ),
            "row_provenance_train": row_provenance_train.to_frame(),
            "categorical_features": list(categorical_features_model_input),
            "categorical_indices": [
                int(X_train_model_input.columns.get_loc(col))
                for col in categorical_features_model_input
                if col in X_train_model_input.columns
            ],
            "model_supports_sample_weight": bool(model_supports_sample_weight),
            "sample_weight_consumed": bool(
                sample_weights_model_input is not None
                and model_supports_sample_weight
            ),
            "contract": {
                "X_train": _frame_contract(X_train_model_input),
                "X_test": _frame_contract(X_test_model_input),
                "y_train_sha256": _array_sha256(y_train_model_input),
                "sample_weights_sha256": _array_sha256(
                    sample_weights_model_input
                ),
                "row_provenance_sha256": _frame_contract(
                    row_provenance_train.to_frame()
                )["sha256"],
            },
        },
        # The capture browser reads this preview under the post_pipeline key.
        "post_pipeline": {
            "X_train_head": X_train_model_input.head(HEAD_N).copy(),
            "X_test_head": X_test_model_input.head(HEAD_N).copy(),
        },
    }


def save_artifact(path: str | Path, artifact: dict) -> None:
    """Create the parent directory when needed and store the pickled artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_artifact(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

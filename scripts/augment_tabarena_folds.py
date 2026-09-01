#!/usr/bin/env python3
"""
Augment existing TabArena dataset folders with the official CV-fold structure.

For each dataset under ``data/tabarena/datasets/<name>/``, the script retains
the existing train/test parquets, writes the full OpenML-order dataset and fold
indices and extends ``meta.json`` with fold metadata:

    data.parquet
    folds.npz
    meta.json

The fold layer enables ``load_dataset(name, fold_idx=K)`` to reproduce TabArena's
official protocol:
  * 10 repeats × 3 folds (= 30 splits) for datasets with n < 2500
  * 3 repeats × 3 folds  (= 9 splits)  for datasets with n >= 2500

Usage:
    # Augment every dataset under data/tabarena/datasets/
    python scripts/augment_tabarena_folds.py

    # Only specific datasets (by folder name)
    python scripts/augment_tabarena_folds.py --names blood-transfusion-service-center anneal

    # Re-export even if folds.npz already exists
    python scripts/augment_tabarena_folds.py --overwrite

The script is resumable: re-running skips datasets that already have a complete
``folds.npz`` with the fold count specified in ``meta.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.data.openml_datasets import (
    _apply_openml_feature_value_repairs,
    _ensure_openml,
    _openml_feature_repair_notes,
    _resolve_openml_feature_types,
)

TARGET_COL = "__target__"
DEFAULT_ROOT = PROJECT_ROOT / "data" / "tabarena" / "datasets"


# ---------------------------------------------------------------------------
# Per-dataset augmentation
# ---------------------------------------------------------------------------

def _load_existing_meta(ds_dir: Path) -> dict:
    with open(ds_dir / "meta.json") as f:
        return json.load(f)


def _drop_id_columns(X: pd.DataFrame, cat_indicator, feature_names):
    """Mirror the ID-drop heuristic used in load_openml_dataset() so that
    data.parquet ends up with the same column set as the existing single split."""
    id_cols = []
    for col in X.columns:
        if X[col].dtype == object or hasattr(X[col], "cat"):
            if X[col].nunique() >= len(X) * 0.9:
                id_cols.append(col)
    if id_cols:
        X = X.drop(columns=id_cols)
        if cat_indicator is not None:
            mask = [c not in id_cols for c in feature_names]
            cat_indicator = [ci for ci, keep in zip(cat_indicator, mask) if keep]
            feature_names = [fn for fn, keep in zip(feature_names, mask) if keep]
    return X, cat_indicator, feature_names


def _augment_one(
    folder_name: str,
    ds_dir: Path,
    overwrite: bool,
) -> dict:
    """Add data.parquet + folds.npz + extended meta.json to one dataset folder."""
    meta_path = ds_dir / "meta.json"
    data_path = ds_dir / "data.parquet"
    folds_path = ds_dir / "folds.npz"

    if not meta_path.exists():
        return {"name": folder_name, "status": "missing_meta"}

    meta = _load_existing_meta(ds_dir)
    extra = meta.get("extra", {})
    task_id = extra.get("openml_task_id")
    dataset_id = extra.get("openml_dataset_id")
    if task_id is None or dataset_id is None:
        return {"name": folder_name, "status": "missing_openml_ids"}

    # Skip if already done & meta records the fold spec
    already_done = (
        data_path.exists()
        and folds_path.exists()
        and "fold_specs" in meta
        and not overwrite
    )
    if already_done:
        return {
            "name": folder_name, "status": "skipped",
            "n_total_folds": meta.get("n_total_folds"),
        }

    openml = _ensure_openml()

    # Fetch the task and its cross-validation grid.
    task = openml.tasks.get_task(
        task_id,
        download_data=False,
        download_qualities=False,
        download_features_meta_data=False,
    )
    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    n_total = n_repeats * n_folds * n_samples

    # Fetch the complete data in the original OpenML row order.
    dataset = task.get_dataset()
    X, y, cat_indicator, feature_names = dataset.get_data(
        target=dataset.default_target_attribute,
    )
    if X is None or y is None:
        raise ValueError(
            f"OpenML dataset {dataset_id} returned empty data. "
            f"Check https://www.openml.org/d/{dataset_id}"
        )
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)

    # ID-column drop (same heuristic as the existing single split so the
    # data.parquet column set matches train.parquet/test.parquet)
    X, cat_indicator, feature_names = _drop_id_columns(X, cat_indicator, feature_names)
    X = _apply_openml_feature_value_repairs(X, dataset_id)
    cat_features, num_features = _resolve_openml_feature_types(
        X,
        cat_indicator,
        feature_names,
        dataset_id,
    )

    # Target → numeric (matches load_openml_dataset behavior)
    y_arr = np.asarray(y)
    task_type = meta.get("task_type")
    if task_type == "classification" and y_arr.dtype.kind in ("O", "U", "S"):
        le = LabelEncoder()
        y_arr = le.fit_transform(y_arr)
    if task_type == "classification":
        y_arr = y_arr.astype(int)

    # Persist data.parquet (full, original OpenML row order)
    full = X.copy()
    full[TARGET_COL] = y_arr
    full.to_parquet(data_path, index=False)

    # Iterate over the OpenML cross-validation grid and collect indices.
    fold_specs = []
    npz_payload: dict[str, np.ndarray] = {}
    fold_idx = 0
    for rep in range(n_repeats):
        for fold in range(n_folds):
            for samp in range(n_samples):
                tr, te = task.get_train_test_split_indices(
                    repeat=rep, fold=fold, sample=samp,
                )
                tr = np.asarray(tr, dtype=np.int64)
                te = np.asarray(te, dtype=np.int64)
                npz_payload[f"train_idx_{fold_idx}"] = tr
                npz_payload[f"test_idx_{fold_idx}"] = te
                fold_specs.append({
                    "fold_idx": fold_idx,
                    "repeat": int(rep),
                    "fold": int(fold),
                    "sample": int(samp),
                    "train_size": int(tr.size),
                    "test_size": int(te.size),
                })
                fold_idx += 1

    # Persist folds.npz
    np.savez_compressed(folds_path, **npz_payload)

    # Extend meta.json and retain all existing keys.
    meta["n_total_folds"] = n_total
    meta["fold_specs"] = fold_specs
    meta["cat_features"] = cat_features
    meta["num_features"] = num_features
    meta["n_features"] = len(cat_features) + len(num_features)
    meta["extra"] = {
        **extra,
        "n_repeats": int(n_repeats),
        "n_folds": int(n_folds),
        "n_samples": int(n_samples),
        "data_parquet": "data.parquet",
        "folds_npz": "folds.npz",
        "feature_type_repairs": _openml_feature_repair_notes(dataset_id),
        "fold_protocol": (
            f"OpenML official CV: {n_repeats} repeats x {n_folds} folds "
            f"x {n_samples} samples = {n_total} splits "
            f"(stratified for classification)"
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "name": folder_name,
        "status": "ok",
        "n_total_folds": n_total,
        "n_rows": int(len(full)),
        "n_features": int(full.shape[1] - 1),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Augment TabArena dataset folders with official CV folds."
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help=f"Datasets root (default: {DEFAULT_ROOT})")
    p.add_argument("--names", nargs="*", default=None,
                   help="Only augment these folder names (default: all under --root).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-export folds.npz even if it already exists.")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit the number of datasets to process.")
    p.add_argument("--continue-on-error", action="store_true", default=True,
                   help="Keep going if a dataset fails (default: True).")
    args = p.parse_args()

    root: Path = args.root
    if not root.is_dir():
        print(f"Datasets root does not exist: {root}", flush=True)
        return 1

    # Discover dataset folders (each must already have meta.json)
    folders = sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "meta.json").is_file()
    )
    if args.names:
        wanted = set(args.names)
        folders = [p for p in folders if p.name in wanted]
        unknown = wanted - {p.name for p in folders}
        if unknown:
            print(f"WARN: requested names not found under {root}: {sorted(unknown)}",
                  flush=True)
    if args.limit:
        folders = folders[: args.limit]

    print(f"Augmenting {len(folders)} dataset(s) under {root}\n", flush=True)

    results = []
    t_start = time.time()
    for i, ds_dir in enumerate(folders, 1):
        name = ds_dir.name
        print(f"  [{i}/{len(folders)}] {name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = _augment_one(name, ds_dir, overwrite=args.overwrite)
            elapsed = time.time() - t0
            if r["status"] == "ok":
                print(f"OK ({elapsed:.1f}s, n_total_folds={r['n_total_folds']}, "
                      f"n_rows={r['n_rows']})", flush=True)
            elif r["status"] == "skipped":
                print(f"SKIP (already augmented, n_total_folds={r['n_total_folds']})",
                      flush=True)
            else:
                print(f"WARN ({r['status']})", flush=True)
            results.append(r)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {e}", flush=True)
            traceback.print_exc()
            results.append({"name": name, "status": "failed", "error": str(e)})
            if not args.continue_on_error:
                raise
        gc.collect()

    # Summary
    n_ok = sum(r["status"] == "ok" for r in results)
    n_skip = sum(r["status"] == "skipped" for r in results)
    n_fail = sum(r["status"] == "failed" for r in results)
    n_warn = len(results) - n_ok - n_skip - n_fail

    # Update _index.json with augmentation summary (additive)
    idx_path = root / "_index.json"
    if idx_path.is_file():
        with open(idx_path) as f:
            idx = json.load(f)
        idx["folds_augmented_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        idx["folds_augmentation"] = {
            "ok": n_ok, "skipped": n_skip, "failed": n_fail, "warned": n_warn,
            "total_attempted": len(results),
        }
        with open(idx_path, "w") as f:
            json.dump(idx, f, indent=2)

    total = time.time() - t_start
    print(
        f"\nDone in {total:.1f}s: "
        f"ok={n_ok}  skipped={n_skip}  warned={n_warn}  failed={n_fail}",
        flush=True,
    )
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

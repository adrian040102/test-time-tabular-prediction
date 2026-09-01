#!/usr/bin/env python3
"""
Download the TabArena datasets from OpenML Study 457 to local parquet files.

Output structure (one folder per dataset, mirroring the existing 26-dataset layout
under ``data/datasets/``):

    data/tabarena/datasets/<dataset_name>/
        train.parquet      # X_train + __target__
        test.parquet       # X_test  + __target__
        meta.json          # openml IDs, task_type, cat/num features, sizes

A summary index is also written:

    data/tabarena/datasets/_index.json

Usage:
    # Download everything (slow first time, OpenML caches in ~/.cache/openml):
    python scripts/download_tabarena.py

    # Download only a few specific datasets (by OpenML dataset ID):
    python scripts/download_tabarena.py --dataset-ids 46905 46922

    # Re-download even if folders already exist:
    python scripts/download_tabarena.py --overwrite

    # Use a different output root:
    python scripts/download_tabarena.py --output data/tabarena/datasets

Notes:
  - TabArena's official protocol is repeated CV (10x3 for small, 3x3 for large),
    not a single train/test split. This script saves a single stratified IID split with
    seed=42 / test_size=0.2 for compatibility with the existing pipeline. The
    full OpenML IDs are preserved in meta.json so the CV protocol can be
    reconstructed later without re-downloading.
  - Datasets are downloaded via the openml package and cached in
    ~/.cache/openml (or $SCRATCH/.cache/openml on HPC).
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

from src.data.openml_datasets import (
    TABARENA_STUDY_ID,
    _ensure_openml,
    load_openml_dataset,
)


TARGET_COL = "__target__"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "tabarena" / "datasets"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_tabarena_task_table() -> list[dict]:
    """
    Query OpenML Study 457 and return one record per task with
    ``{task_id, dataset_id, dataset_name}``.
    """
    openml = _ensure_openml()
    suite = openml.study.get_suite(TABARENA_STUDY_ID)
    task_ids = list(suite.tasks)

    rows = []
    for tid in task_ids:
        task = openml.tasks.get_task(
            tid,
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False,
        )
        rows.append({
            "task_id": int(tid),
            "dataset_id": int(task.dataset_id),
            "dataset_name": task.get_dataset().name,
        })
    return rows


def safe_dataset_dirname(name: str) -> str:
    """Sanitize dataset name for filesystem use."""
    keep = "-_.()"
    return "".join(c if (c.isalnum() or c in keep) else "_" for c in name).strip("_")


# ---------------------------------------------------------------------------
# Per-dataset download
# ---------------------------------------------------------------------------

def _download_one(
    dataset_id: int,
    dataset_name: str,
    task_id: int | None,
    output_root: Path,
    seed: int = 42,
    test_size: float = 0.2,
    overwrite: bool = False,
) -> dict:
    """Download one TabArena dataset, write parquet and metadata and return its status."""
    folder_name = safe_dataset_dirname(dataset_name)
    ds_dir = output_root / folder_name
    meta_path = ds_dir / "meta.json"

    if meta_path.exists() and not overwrite:
        return {"dataset_id": dataset_id, "name": dataset_name,
                "status": "skipped", "folder": folder_name}

    dataset = load_openml_dataset(
        openml_id=dataset_id,
        name=dataset_name,
        task_type=None,         # auto-detect
        test_size=test_size,
        max_samples=None,
        seed=seed,
    )

    train_df = dataset.X_train.copy()
    train_df[TARGET_COL] = dataset.y_train

    test_df = dataset.X_test.copy()
    test_df[TARGET_COL] = dataset.y_test

    ds_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(ds_dir / "train.parquet", index=False)
    test_df.to_parquet(ds_dir / "test.parquet", index=False)

    meta = {
        "name": dataset.name,
        "task_type": dataset.task_type,
        "cat_features": dataset.cat_features,
        "num_features": dataset.num_features,
        "shift_description": "TabArena (IID, single random split)",
        "n_train": dataset.n_train,
        "n_test": dataset.n_test,
        "n_features": dataset.n_features,
        "seed_used": seed,
        "max_samples_used": None,
        "extra": {
            "source": "tabarena",
            "openml_study_id": TABARENA_STUDY_ID,
            "openml_dataset_id": dataset_id,
            "openml_task_id": task_id,
            "openml_dataset_name": dataset_name,
            "test_size": test_size,
            "n_total": dataset.n_train + dataset.n_test,
            "feature_type_repairs": dataset.metadata.get(
                "feature_type_repairs", []
            ),
            "split_protocol": (
                "Single IID split with seed 42. Classification targets are "
                "stratified when each class has at least two samples. The "
                "TabArena official protocol uses repeated CV (10x3 for "
                "n<2500, 3x3 otherwise)."
            ),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "dataset_id": dataset_id,
        "name": dataset_name,
        "status": "ok",
        "folder": folder_name,
        "n_train": dataset.n_train,
        "n_test": dataset.n_test,
        "n_features": dataset.n_features,
        "task_type": dataset.task_type,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Download all TabArena datasets (OpenML Study 457) to parquet."
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output root (default: {DEFAULT_OUTPUT})")
    p.add_argument("--dataset-ids", type=int, nargs="*", default=None,
                   help="Only download these OpenML dataset IDs (default: all 51).")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit the number of datasets to process.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--overwrite", action="store_true",
                   help="Re-download even if meta.json already exists.")
    p.add_argument("--continue-on-error", action="store_true", default=True,
                   help="Keep going if a dataset fails (default: True).")
    args = p.parse_args()

    output_root: Path = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Querying OpenML Study {TABARENA_STUDY_ID} ...", flush=True)
    try:
        all_tasks = fetch_tabarena_task_table()
    except Exception as e:
        print(f"Failed to fetch study task list: {e}", flush=True)
        traceback.print_exc()
        return 1

    print(f"  {len(all_tasks)} tasks in study", flush=True)

    # Filter
    if args.dataset_ids:
        wanted = set(args.dataset_ids)
        tasks = [t for t in all_tasks if t["dataset_id"] in wanted]
        print(f"  filtering to {len(tasks)} dataset(s) by --dataset-ids", flush=True)
    else:
        tasks = all_tasks

    if args.limit:
        tasks = tasks[: args.limit]
        print(f"  limited to first {len(tasks)} task(s)", flush=True)

    print(f"\nDownloading {len(tasks)} datasets to {output_root}\n", flush=True)

    results = []
    t_start = time.time()
    for i, t in enumerate(tasks, 1):
        ds_id = t["dataset_id"]
        ds_name = t["dataset_name"]
        task_id = t["task_id"]

        print(f"  [{i}/{len(tasks)}] {ds_name} (dataset_id={ds_id}) ...",
              end=" ", flush=True)
        t0 = time.time()
        try:
            r = _download_one(
                dataset_id=ds_id,
                dataset_name=ds_name,
                task_id=task_id,
                output_root=output_root,
                seed=args.seed,
                test_size=args.test_size,
                overwrite=args.overwrite,
            )
            elapsed = time.time() - t0
            if r["status"] == "ok":
                print(f"OK ({elapsed:.1f}s, n_train={r['n_train']}, "
                      f"n_test={r['n_test']}, p={r['n_features']})", flush=True)
            else:
                print(f"SKIP (already exported)", flush=True)
            results.append(r)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {e}", flush=True)
            results.append({
                "dataset_id": ds_id,
                "name": ds_name,
                "status": "failed",
                "error": str(e),
            })
            if not args.continue_on_error:
                raise
        gc.collect()

    # Write index
    index_path = output_root / "_index.json"
    with open(index_path, "w") as f:
        json.dump({
            "study_id": TABARENA_STUDY_ID,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_in_study": len(all_tasks),
            "attempted": len(tasks),
            "ok": sum(r["status"] == "ok" for r in results),
            "skipped": sum(r["status"] == "skipped" for r in results),
            "failed": sum(r["status"] == "failed" for r in results),
            "datasets": results,
        }, f, indent=2)

    total = time.time() - t_start
    n_ok = sum(r["status"] == "ok" for r in results)
    n_skip = sum(r["status"] == "skipped" for r in results)
    n_fail = sum(r["status"] == "failed" for r in results)
    print(f"\nDone in {total:.1f}s: ok={n_ok}  skipped={n_skip}  failed={n_fail}",
          flush=True)
    print(f"Index written to {index_path}", flush=True)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Merge all per-task result CSVs into a single sweep_results.csv.

Usage:
    python scripts/slurm/collect_results.py
    python scripts/slurm/collect_results.py --runs-dir results/runs --output results/sweep_results.csv

Also summarizes expected and discovered task files, missing task IDs and
recorded error rows.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Collect SLURM results.")
    p.add_argument("--runs-dir", default="results/runs", help="Directory with per-task CSVs")
    p.add_argument("--manifest", default="scripts/slurm/manifest.csv", help="Manifest to check completeness")
    p.add_argument("--output", default="results/sweep_results.csv", help="Merged output CSV")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    task_files = sorted(runs_dir.glob("task_*.csv"))

    if not task_files:
        print(f"No result files found in {runs_dir}/")
        return

    # Load and merge
    dfs = []
    n_errors = 0
    for f in task_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            n_errors += df["error"].notna().sum() if "error" in df.columns else 0
        except Exception as e:
            print(f"  WARNING: Could not read {f}: {e}")

    merged = pd.concat(dfs, ignore_index=True)

    # Compare discovered task files with the manifest.
    manifest = pd.read_csv(args.manifest)
    n_expected = len(manifest)
    n_tasks_expected = manifest["task_id"].nunique()
    n_tasks_found = len(task_files)
    n_runs_found = len(merged)

    # Find missing tasks
    expected_ids = set(manifest["task_id"].unique())
    found_ids = set()
    for f in task_files:
        try:
            tid = int(f.stem.split("_")[1])
            found_ids.add(tid)
        except (IndexError, ValueError):
            pass
    missing_ids = sorted(expected_ids - found_ids)

    # Write merged output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS COLLECTED")
    print(f"  Task files found: {n_tasks_found} / {n_tasks_expected}")
    print(f"  Runs collected:   {n_runs_found} / {n_expected}")
    print(f"  Errors:           {n_errors}")
    if missing_ids:
        print(f"  Missing tasks:    {len(missing_ids)}")
        if len(missing_ids) <= 20:
            print(f"    IDs: {missing_ids}")
            # Format the task IDs for a scheduler-specific array command.
            id_str = ",".join(str(i) for i in missing_ids)
            print(f"\n  Missing array-task IDs:")
            print(f"    {id_str} (use with your scheduler's array command)")
        else:
            print(f"    First 20: {missing_ids[:20]}")
            # Save a range-independent list for scheduler-specific resubmission.
            print(f"\n  Saving all missing task IDs:")
            missing_file = runs_dir / "missing_tasks.txt"
            with open(missing_file, "w") as f:
                f.write(",".join(str(i) for i in missing_ids))
            print(f"    See: {missing_file}")
    else:
        print(f"  Status:           All complete ✓")
    print(f"\n  Output: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run a benchmark sweep over datasets, methods, models and seeds.

The dataset is reloaded for each experiment because LeakageGuard removes direct
access to test labels. Results are appended after each run to support resumption.

Usage:
    python scripts/run_sweep.py \
        --config configs/experiments/tabarena_final_lgb_bag8.yaml

    python scripts/run_sweep.py \
        --config configs/experiments/tabarena_final_lgb_bag8.yaml \
        --datasets diabetes --models lightgbm_tabarena_bag8 --seeds 0
"""

from __future__ import annotations

import argparse
import sys
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import pandas as pd

from src.data import load_dataset
from src.models.base import get_model
from src.pipeline import run_single_experiment
from src.utils import load_config, check_cache_env

# Warn when cache directories use the home directory, which may have limited
# quota on shared systems.
check_cache_env(strict=False)


# ---------------------------------------------------------------------------
# Method registry (same as run_experiment.py but gathered in one place)
# ---------------------------------------------------------------------------
def _build_method_registry():
    from src.methods.base import InductiveBaseline, IdentityMethod
    from src.methods.tier1 import TIER1_METHODS, SELECTIVE_METHODS
    from src.methods.tier2 import TIER2_METHODS
    from src.methods.baselines import BASELINE_METHODS

    base = {
        "inductive_baseline": lambda: InductiveBaseline(),
        "identity": lambda: IdentityMethod(),
    }
    # TIER1_METHODS values may be instances or lambdas
    all_methods = {**base}
    for k, v in TIER1_METHODS.items():
        if callable(v) and not hasattr(v, "fit_transform"):
            all_methods[k] = v          # already a factory / lambda
        else:
            all_methods[k] = lambda _v=v: _v  # wrap instance in lambda

    # Selective variants (method + column selector combinations)
    all_methods.update(SELECTIVE_METHODS)

    # Tier 2 methods (domain classifier, reweighting, etc.)
    all_methods.update(TIER2_METHODS)

    # Train-only intermediate baselines.
    all_methods.update(BASELINE_METHODS)

    return all_methods


METHOD_REGISTRY = _build_method_registry()


def get_method(name: str):
    if name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{name}'. Available: {sorted(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[name]()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run a full benchmark sweep.")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file (datasets, models, methods, seeds, params)")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Override: dataset names")
    p.add_argument("--models", nargs="*", default=None,
                   help="Override: model names")
    p.add_argument("--methods", nargs="*", default=None,
                   help="Override: method names")
    p.add_argument("--seeds", nargs="*", type=int, default=None,
                   help="Override: random seeds")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Override: max samples per split")
    p.add_argument("--results-dir", type=str, default="results",
                   help="Directory for output CSVs")
    p.add_argument("--tag", type=str, default=None,
                   help="Optional tag appended to output filename")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # --- Build configuration (config file + CLI overrides) -----------------
    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    datasets = args.datasets or cfg.get("datasets", ["synthetic"])
    models   = args.models   or cfg.get("models",   ["lightgbm"])
    methods  = args.methods  or cfg.get("methods",  ["inductive_baseline"])
    seeds    = args.seeds    or cfg.get("seeds",    [42])
    params   = cfg.get("params", {})
    max_samples = args.max_samples or params.get("max_samples")

    # --- Output path -------------------------------------------------------
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    csv_path = results_dir / f"sweep_results{tag}.csv"

    # Resume support: load existing results to skip completed runs.
    # Use method_key (the CLI or configuration name) rather than the
    # method's display name, which can differ (e.g. "joint_freq_combined"
    # vs "joint_freq_enc_combined").
    done_keys: set[tuple] = set()
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(csv_path)
        except pd.errors.ParserError:
            # CSV got corrupted (e.g. unquoted commas in metadata fields).
            # Fall back to Python engine which is more forgiving.
            try:
                existing = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
                print(f"WARNING: CSV had parse errors, skipped invalid lines. "
                      f"Loaded {len(existing)} valid rows.")
            except Exception:
                print(f"WARNING: Could not parse {csv_path}, starting fresh.")
                existing = pd.DataFrame()
        key_col = "method_key" if "method_key" in existing.columns else "method"
        for _, row in existing.iterrows():
            if pd.notna(row.get("error", float("nan"))) if "error" in row.index else False:
                continue  # do not skip failed runs. Retry them
            done_keys.add((
                str(row.get("dataset", "")),
                str(row[key_col]),
                str(row.get("model", "")),
                int(row.get("seed", 0)),
            ))
        print(f"Resuming: {len(done_keys)} runs already completed in {csv_path}")

    # --- Compute run matrix ------------------------------------------------
    combos = [
        (d, me, mo, s)
        for d in datasets
        for me in methods
        for mo in models
        for s in seeds
    ]
    combos = [c for c in combos if c not in done_keys]
    total = len(combos)

    print(f"\n{'=' * 72}")
    print(f"BENCHMARK SWEEP: {total} runs")
    print(f"  Datasets: {datasets}")
    print(f"  Methods:  {methods}")
    print(f"  Models:   {models}")
    print(f"  Seeds:    {seeds}")
    if max_samples:
        print(f"  Max samples/split: {max_samples}")
    print(f"  Output:   {csv_path}")
    print(f"{'=' * 72}\n")

    # --- Run loop ----------------------------------------------------------
    t_sweep_start = time.time()
    n_ok, n_fail = 0, 0

    for i, (ds_name, method_name, model_name, seed) in enumerate(combos, 1):
        tag_str = f"[{i}/{total}]"
        print(f"{tag_str} {ds_name} | {method_name} | {model_name} | seed={seed}", end=" … ")

        row: dict = {
            "dataset": ds_name,
            "method_key": method_name,
            "model": model_name,
            "seed": seed,
        }

        try:
            # Reload dataset every time (LeakageGuard sets y_test=None).
            # Pass seed so synthetic datasets generate different data per seed.
            # Real-world datasets ignore the seed (fixed train/test split).
            ds_kwargs = {"seed": seed}
            if max_samples:
                ds_kwargs["max_samples"] = max_samples
            dataset = load_dataset(ds_name, **ds_kwargs)

            method = get_method(method_name)
            model = get_model(model_name, task_type=dataset.task_type, seed=seed)

            # Apply a log transform to nonnegative regression targets with skewness above 3.
            log_target = False
            if dataset.task_type == "regression" and (dataset.y_train >= 0).all():
                from scipy.stats import skew as _skew
                if _skew(dataset.y_train) > 3.0:
                    log_target = True

            result = run_single_experiment(
                dataset, method, model, seed=seed,
                log_transform_target=log_target,
            )
            row.update(result.to_dict())
            print(result.summary_line())
            n_ok += 1

        except Exception as e:
            import traceback
            row["error"] = str(e)
            print(f"FAILED: {e}")
            traceback.print_exc()  # Print the full traceback for debugging.
            n_fail += 1

        # Append after every run so completed rows are retained.
        df_row = pd.DataFrame([row])
        df_row.to_csv(
            csv_path,
            mode="a",
            header=not csv_path.exists() or (i == 1 and not done_keys),
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
        )

    elapsed = time.time() - t_sweep_start

    print(f"\n{'=' * 72}")
    print(f"SWEEP COMPLETE: {n_ok} succeeded, {n_fail} failed, {elapsed:.0f}s total")
    print(f"Results: {csv_path}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

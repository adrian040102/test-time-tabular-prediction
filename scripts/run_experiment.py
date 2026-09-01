#!/usr/bin/env python3
"""
CLI entry point for running a single experiment.

Usage:
    python scripts/run_experiment.py --dataset synthetic --model lightgbm
    python scripts/run_experiment.py --dataset synthetic_shift_2.0 --model xgboost --seed 123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset
from src.methods.base import InductiveBaseline
from src.models.base import get_model, MODEL_REGISTRY
from src.pipeline import run_single_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run a thesis experiment.")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        help="Dataset name (e.g. 'synthetic', 'synthetic_shift_1.0', "
                             "'folktables_income_2014_2018')")
    parser.add_argument("--model", type=str, default="lightgbm",
                        help=f"Model name. Options: {list(MODEL_REGISTRY.keys())}")
    parser.add_argument("--method", type=str, default="inductive_baseline",
                        help="Method name (default: inductive_baseline)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per split (for prototyping)")
    parser.add_argument("--shift-strength", type=float, default=1.0,
                        help="Shift strength for synthetic data")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--capture-config", type=str, default=None,
                        help="Optional capture-audit YAML. If this run's "
                             "(dataset, method, model, seed) matches an entry, "
                             "save a pickled snapshot to --capture-dir.")
    parser.add_argument("--capture-dir", type=str, default="results/captures",
                        help="Directory for capture artifacts.")
    return parser.parse_args()


def get_method(name: str):
    """Get a method instance by name."""
    from src.methods.base import IdentityMethod
    from src.methods.tier1 import TIER1_METHODS, SELECTIVE_METHODS
    from src.methods.tier2 import TIER2_METHODS
    from src.methods.baselines import BASELINE_METHODS

    base_methods = {
        "inductive_baseline": lambda: InductiveBaseline(),
        "identity": lambda: IdentityMethod(),
    }
    # Merge base methods with Tier 1 + Tier 2 registries + selective variants + baselines
    all_methods = {**base_methods, **TIER1_METHODS, **SELECTIVE_METHODS, **TIER2_METHODS, **BASELINE_METHODS}
    if name not in all_methods:
        raise ValueError(f"Unknown method '{name}'. Available: {sorted(all_methods.keys())}")
    return all_methods[name]()


def main():
    args = parse_args()

    print("=" * 70)
    print("THESIS EXPERIMENT")
    print("=" * 70)

    # Load dataset
    dataset_kwargs = {}
    if args.max_samples:
        dataset_kwargs["max_samples"] = args.max_samples
    if "synthetic" in args.dataset:
        dataset_kwargs["shift_strength"] = args.shift_strength

    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, **dataset_kwargs)
    print(dataset.summary())

    # Get method and model (pass task_type so the right estimator is used)
    method = get_method(args.method)
    model = get_model(args.model, task_type=dataset.task_type, seed=args.seed)

    print(f"Method: {method.name}")
    print(f"Model:  {model.name}")
    print(f"Seed:   {args.seed}")
    print()

    # Resolve capture path if the (dataset, method, model, seed) is on the audit list
    capture_path = None
    if args.capture_config:
        from src.capture import CaptureKey, artifact_filename, load_audit
        audit = load_audit(args.capture_config)
        key = CaptureKey(args.dataset, args.method, model.name, args.seed)
        if key.as_tuple() in audit:
            capture_dir = Path(args.capture_dir)
            capture_dir.mkdir(parents=True, exist_ok=True)
            capture_path = capture_dir / artifact_filename(key)
            print(f"Capture enabled: {capture_path}")

    # Run experiment
    result = run_single_experiment(dataset, method, model, seed=args.seed,
                                   capture_path=capture_path,
                                   capture_method_key=args.method)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(result.summary_line())
    print(f"\nFeatures: {result.n_features_in} -> {result.n_features_out}")
    print(f"Samples:  {result.n_train:,} train, {result.n_test:,} test")

    # Save single result
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame([result.to_dict()])
    csv_path = results_dir / f"{args.dataset}_{args.method}_{args.model}_s{args.seed}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()

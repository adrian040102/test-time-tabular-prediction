#!/usr/bin/env python3
"""
Generate a job manifest for SLURM array jobs.

Reads a sweep YAML configuration and writes a CSV in which each row represents
one dataset, method, model, seed and fold combination. Rows are grouped into
batches so that each SLURM array task handles the configured number of runs.

Optional dataset, method, model and seed filters select values already present
in the YAML configuration.

Usage:
    python scripts/slurm/generate_manifest.py \
        --config configs/experiments/tabarena_final_lgb_bag8.yaml \
        --batch-size 5 \
        --output scripts/slurm/manifest.csv

Output CSV columns:
    task_id, run_idx, dataset, method, model, seed, max_samples, fold
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config


def _resolve_folds_for_dataset(name: str) -> list[int] | None:
    """Return [0..N-1] if the dataset has fold metadata, else None.

    Looks for ``fold_specs`` in the dataset's ``meta.json`` (only TabArena-style
    datasets augmented via ``scripts/augment_tabarena_folds.py`` will have it).
    Searches the same locations as the parquet loader: primary data root and
    ``data/tabarena/datasets/`` fallback.
    """
    candidates = [
        PROJECT_ROOT / "data" / "datasets" / name / "meta.json",
        PROJECT_ROOT / "data" / "tabarena" / "datasets" / name / "meta.json",
    ]
    for path in candidates:
        if path.is_file():
            with open(path) as f:
                meta = json.load(f)
            specs = meta.get("fold_specs")
            if specs:
                return [int(s["fold_idx"]) for s in specs]
            return None
    return None


def _resolve_n_train(name: str) -> int | None:
    """Return ``n_train`` from ``meta.json`` if present.

    The size-dependent fold policy uses three outer folds below 2500 training rows
    and one outer fold otherwise.
    """
    candidates = [
        PROJECT_ROOT / "data" / "datasets" / name / "meta.json",
        PROJECT_ROOT / "data" / "tabarena" / "datasets" / name / "meta.json",
    ]
    for path in candidates:
        if path.is_file():
            with open(path) as f:
                meta = json.load(f)
            n = meta.get("n_train")
            return int(n) if n is not None else None
    return None


def _resolve_dataset_weight(name: str) -> float:
    """Return a rough per-run cost proxy for ``name`` (rows × columns).

    Used by ``--order balanced`` to place the longest-running jobs first. The
    calculation reads Parquet metadata for row and column counts, not the data.
    Falls back to 1.0 if the dataset folder cannot be located.
    """
    # Lazy import to keep generate_manifest.py startup fast for --help etc.
    import pyarrow.parquet as pq  # type: ignore

    candidates = [
        PROJECT_ROOT / "data" / "datasets" / name,
        PROJECT_ROOT / "data" / "tabarena" / "datasets" / name,
    ]
    for ds_dir in candidates:
        # Prefer full data.parquet (TabArena-style). Fall back to train+test.
        full = ds_dir / "data.parquet"
        if full.is_file():
            pf = pq.ParquetFile(full)
            return float(pf.metadata.num_rows) * float(pf.metadata.num_columns)
        train_p, test_p = ds_dir / "train.parquet", ds_dir / "test.parquet"
        if train_p.is_file() and test_p.is_file():
            pt, pe = pq.ParquetFile(train_p), pq.ParquetFile(test_p)
            rows = pt.metadata.num_rows + pe.metadata.num_rows
            cols = max(pt.metadata.num_columns, pe.metadata.num_columns)
            return float(rows) * float(cols)
    return 1.0


def _apply_filter(yaml_values, user_selection, field_name):
    """Filter yaml_values to the user's selection. Error on unknown values.

    Preserves YAML order (deterministic task_id assignment across invocations
    that select the same subset in different CLI orderings)."""
    if user_selection is None:
        return yaml_values
    yaml_set = set(yaml_values)
    unknown = [v for v in user_selection if v not in yaml_set]
    if unknown:
        raise SystemExit(
            f"Unknown {field_name} in --{field_name}: {unknown}\n"
            f"Available in YAML: {yaml_values}"
        )
    user_set = set(user_selection)
    return [v for v in yaml_values if v in user_set]


def main():
    p = argparse.ArgumentParser(description="Generate SLURM job manifest.")
    p.add_argument("--config", required=True, help="Sweep YAML config file")
    p.add_argument("--batch-size", type=int, default=5,
                   help="Runs per SLURM array task (default: 5)")
    p.add_argument("--output", default="scripts/slurm/manifest.csv",
                   help="Output manifest CSV path")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Subset of YAML datasets (space-separated). Omit to use all.")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of YAML methods (space-separated). Omit to use all.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Subset of YAML models (space-separated). Omit to use all.")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Subset of YAML seeds (space-separated ints). Omit to use all.")
    p.add_argument("--folds", default="auto",
                   choices=["auto", "all", "none", "first"],
                   help=(
                       "Per-dataset fold expansion (default: auto). "
                       "'auto'/'all' expand each dataset to its OpenML folds if "
                       "fold_specs is present in meta.json, else single split. "
                       "'first' uses only fold_idx=0. 'none' uses the default single split."
                   ))
    p.add_argument("--max-folds", type=int, default=None,
                   help="If set, limit the number of folds per dataset for a quick check.")
    p.add_argument(
        "--order",
        default="balanced",
        choices=["dataset_outermost", "balanced"],
                   help=(
                       "Task ordering strategy (default: balanced). "
                       "'dataset_outermost' uses a dataset-outermost Cartesian "
                       "product, so each batch contains almost one dataset. "
                       "'balanced' = interleave datasets across batches, "
                       "ordered by decreasing processing time."
                   ))
    p.add_argument("--task-id-start", type=int, default=0,
                   help=(
                       "Offset added to every task_id (default 0 = no change). "
                       "Use for an ADD-ON array that must land in a results dir "
                       "already occupied by a running sweep: set this above the "
                       "live manifest's max task_id so the new task_NNNN.csv "
                       "files cannot overwrite existing ones. Submit the add-on "
                       "with --array=START-END and TASK_OFFSET=0."
                   ))
    args = p.parse_args()

    cfg = load_config(args.config)
    datasets = _apply_filter(cfg["datasets"], args.datasets, "datasets")
    methods = _apply_filter(cfg["methods"], args.methods, "methods")
    models = _apply_filter(cfg["models"], args.models, "models")
    seeds = _apply_filter(cfg["seeds"], args.seeds, "seeds")
    params = cfg.get("params", {}) or {}
    max_samples = params.get("max_samples", "")

    # Optional YAML-level fold policy and per-dataset overrides. YAML files
    # without these keys retain CLI-controlled fold expansion. A configured
    # fold policy takes precedence over --folds and per-dataset overrides take
    # precedence over the policy.
    #
    # Supported fold_policy values:
    #   "first": one fold per fold-bearing dataset
    #   "small_datasets_three_folds": three folds below 2,500 training rows
    # Datasets without fold metadata emit a single fold="" entry.
    fold_policy = params.get("fold_policy")
    fold_overrides_raw = params.get("fold_overrides", {}) or {}
    fold_overrides: dict[str, int] = {
        str(k): int(v) for k, v in fold_overrides_raw.items()
    }
    if fold_policy is not None and fold_policy not in ("first", "small_datasets_three_folds"):
        raise SystemExit(
            f"Unknown params.fold_policy: {fold_policy!r}. "
            f"Supported: 'first', 'small_datasets_three_folds'."
        )
    unknown_overrides = sorted(set(fold_overrides) - set(datasets))
    if unknown_overrides:
        raise SystemExit(
            f"params.fold_overrides references unknown datasets "
            f"(not in this YAML's datasets list): {unknown_overrides}"
        )

    # Optional unified evaluation policy:
    #   "folds3_else1": fold-bearing datasets use up to three official folds.
    #   Fixed-split datasets use one evaluation. The first configured seed is
    #   used in both cases and models train on the complete training partition
    #   without bootstrap sampling. Per-dataset fold overrides still apply.
    # This policy supersedes fold_policy. The two cannot be combined.
    eval_policy = params.get("eval_policy")
    if eval_policy is not None and eval_policy != "folds3_else1":
        raise SystemExit(
            f"Unknown params.eval_policy: {eval_policy!r}. "
            f"Supported: 'folds3_else1'."
        )
    if eval_policy == "folds3_else1" and fold_policy is not None:
        raise SystemExit(
            "params.eval_policy: folds3_else1 cannot be combined with "
            "params.fold_policy. folds3_else1 defines fold behavior itself."
        )

    # Resolve per-dataset fold list. Datasets without fold_specs in meta.json
    # always run with fold="" (the default single split), regardless of --folds.
    fold_mode = args.folds
    per_dataset_folds: dict[str, list] = {}
    fold_summary: dict[str, int] = {"with_folds": 0, "without_folds": 0}
    for ds in datasets:
        folds = _resolve_folds_for_dataset(ds) if fold_mode != "none" else None
        if folds is None:
            per_dataset_folds[ds] = [""]
            fold_summary["without_folds"] += 1
            continue
        # Determine the fold count for this dataset.
        if ds in fold_overrides:
            n_folds = fold_overrides[ds]
        elif fold_policy == "first":
            n_folds = 1
        elif fold_policy == "small_datasets_three_folds":
            n_train = _resolve_n_train(ds)
            n_folds = 3 if (n_train is not None and n_train < 2500) else 1
        elif eval_policy == "folds3_else1":
            # 3 official folds for every fold-bearing dataset.
            n_folds = 3
        else:
            # Default CLI behavior.
            if fold_mode == "first":
                n_folds = 1
            elif args.max_folds is not None:
                n_folds = args.max_folds
            else:
                n_folds = len(folds)
        per_dataset_folds[ds] = folds[:n_folds]
        fold_summary["with_folds"] += 1

    # Full Cartesian product (dataset × fold × method × model × seed)
    # Ordering strategy controls how tuples are laid out in the manifest,
    # which directly determines how SLURM array batches distribute work across
    # workers. See --order help above for the motivation.
    # folds3_else1 defines per-dataset (seed, fold) evaluation pairs. Fixed-split
    # datasets use one evaluation at the first configured seed. Fold-bearing
    # datasets vary the fold at that seed.
    per_dataset_evals: dict[str, list[tuple]] | None = None
    if eval_policy == "folds3_else1":
        parity_seed = seeds[0]
        per_dataset_evals = {}
        for ds in datasets:
            folds = per_dataset_folds[ds]
            if folds == [""]:        # fixed-split dataset
                per_dataset_evals[ds] = [(parity_seed, "")]
            else:                     # fold-bearing dataset
                per_dataset_evals[ds] = [(parity_seed, f) for f in folds]

    combos: list[tuple] = []
    if eval_policy == "folds3_else1":
        if args.order == "dataset_outermost":
            for ds in datasets:
                for (s, fold) in per_dataset_evals[ds]:
                    for me, mo in itertools.product(methods, models):
                        combos.append((ds, me, mo, s, fold))
        else:  # balanced: interleave datasets, longest-cost-first
            ds_by_cost = sorted(
                datasets, key=lambda d: _resolve_dataset_weight(d), reverse=True,
            )
            max_evals = max(len(per_dataset_evals[d]) for d in datasets)
            for eval_pos in range(max_evals):
                for me, mo in itertools.product(methods, models):
                    for ds in ds_by_cost:
                        evals = per_dataset_evals[ds]
                        if eval_pos < len(evals):
                            s, fold = evals[eval_pos]
                            combos.append((ds, me, mo, s, fold))
    elif args.order == "dataset_outermost":
        # Dataset is the outer loop. Each batch then contains approximately one
        # dataset, which can leave workers idle near the end when dataset sizes
        # differ substantially.
        for ds in datasets:
            for fold in per_dataset_folds[ds]:
                for me, mo, s in itertools.product(methods, models, seeds):
                    combos.append((ds, me, mo, s, fold))
    else:  # balanced
        # Each batch spans many datasets. Larger datasets are emitted first so
        # shorter jobs remain available while the longest jobs finish.
        #   1) sort datasets by per-run cost (rows × cols) descending
        #   2) outer loop over fold-position, method, model, seed
        #   3) inner loop over the cost-sorted dataset list
        ds_by_cost = sorted(
            datasets, key=lambda d: _resolve_dataset_weight(d), reverse=True,
        )
        max_folds = max(len(per_dataset_folds[d]) for d in datasets)
        for fold_pos in range(max_folds):
            for me, mo, s in itertools.product(methods, models, seeds):
                for ds in ds_by_cost:
                    folds = per_dataset_folds[ds]
                    if fold_pos < len(folds):
                        combos.append((ds, me, mo, s, folds[fold_pos]))
    total_runs = len(combos)

    # Assign task_ids
    batch_size = args.batch_size
    n_tasks = (total_runs + batch_size - 1) // batch_size

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id", "run_idx", "dataset", "method", "model", "seed",
            "max_samples", "fold",
        ])
        for i, (ds, me, mo, s, fold) in enumerate(combos):
            task_id = i // batch_size + args.task_id_start
            run_idx = i % batch_size
            writer.writerow([task_id, run_idx, ds, me, mo, s, max_samples, fold])

    any_filter = any(x is not None for x in (args.datasets, args.methods, args.models, args.seeds))
    print(f"Manifest written: {out_path}")
    if any_filter:
        print(f"  Filters applied:")
        print(f"    datasets: {len(datasets)} (of {len(cfg['datasets'])})")
        print(f"    methods:  {len(methods)} (of {len(cfg['methods'])})")
        print(f"    models:   {len(models)} (of {len(cfg['models'])})  -> {models}")
        print(f"    seeds:    {len(seeds)} (of {len(cfg['seeds'])})  -> {seeds}")
    if fold_summary["with_folds"]:
        ex_ds = next((d for d in datasets if per_dataset_folds[d] != [""]), None)
        ex_folds = len(per_dataset_folds[ex_ds]) if ex_ds else 0
        policy_str = f", policy={fold_policy}" if fold_policy else ""
        ovr_str = (
            f", overrides={fold_overrides}" if fold_overrides else ""
        )
        print(f"  Folds:       mode={fold_mode}{policy_str}{ovr_str}, "
              f"{fold_summary['with_folds']} datasets with folds "
              f"(e.g. {ex_ds}: {ex_folds} folds), "
              f"{fold_summary['without_folds']} default single-split")
    print(f"  Total runs:  {total_runs}")
    print(f"  Batch size:  {batch_size}")
    lo = args.task_id_start
    hi = args.task_id_start + n_tasks - 1
    print(f"  Array tasks: {n_tasks}  (use --array={lo}-{hi})")
    print(f"  Order:       {args.order}")
    # Diagnostic: how many distinct datasets appear in the first/last batch?
    # Balanced should approach min(batch_size, n_datasets). Dataset-outermost should be 1.
    if combos:
        first_batch = combos[:batch_size]
        last_batch = combos[-batch_size:]
        n_ds_first = len({c[0] for c in first_batch})
        n_ds_last = len({c[0] for c in last_batch})
        print(f"  Batch mix:   first={n_ds_first} datasets / "
              f"last={n_ds_last} datasets "
              f"(ideal for load balance: ~min(batch_size, n_datasets)"
              f"={min(batch_size, len(datasets))})")
    print(f"  Max parallel: 50  (use %50 throttle)")
    print(f"  SLURM flag:  --array={lo}-{hi}%50")


if __name__ == "__main__":
    main()

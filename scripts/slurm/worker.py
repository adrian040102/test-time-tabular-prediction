#!/usr/bin/env python3
"""
SLURM array-task worker.

Reads $SLURM_ARRAY_TASK_ID, looks up its batch of runs from the manifest,
executes them and writes results to results/runs/task_{id}.csv.

Each task is fully self-contained: it loads data, runs the experiment
and writes its own output file.  No inter-task coordination is needed.

Usage (directly or through a scheduler wrapper):
    SLURM_ARRAY_TASK_ID=0 python scripts/slurm/worker.py \
        --manifest scripts/slurm/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data import load_dataset
from src.models.base import get_model, get_model_execution_metadata
from src.pipeline import run_single_experiment
from src.result_status import is_deterministic_skip_error
from src.utils import check_cache_env

# Warn when cache directories use the home directory, which may have limited
# quota on shared systems.
check_cache_env(strict=False)

# ---------------------------------------------------------------------------
# Bag-8 subprocess isolation
# ---------------------------------------------------------------------------
# AutoGluon mutates process-global state across successive TabularPredictor
# fits (logging handlers, temp-dir registry, internal caches).  When the
# worker processes a batch of ≥2 bag-8 rows in the same process, later fits
# can produce different results than the first.  The reproducibility contract
# test (tests/test_lgb_tabarena_bag8.py:subprocess_reproducibility) catches
# this by using fresh subprocess.run() isolation. This worker uses the same
# batch runtime via spawn-mode multiprocessing.Process per bag-8 row.

_BAG8_MODEL_NAME = "lightgbm_tabarena_bag8"


def _bag8_worker_inner(
    result_queue,
    project_root_str: str,
    ds_name: str,
    ds_kwargs: dict,
    method_name: str,
    model_name: str,
    seed: int,
    capture_path_str: str | None,
) -> None:
    """Module-level target for spawn-mode subprocess.

    Re-imports everything in a fresh process so AutoGluon global state cannot
    leak between consecutive bag-8 fits in the same batch.  Must be a
    module-level function so the ``spawn`` start method can pickle it.
    """
    import sys as _sys
    _sys.path.insert(0, project_root_str)

    from src.data import load_dataset as _load
    from src.models.base import get_model as _get_model
    from src.pipeline import run_single_experiment as _run
    from src.methods.base import InductiveBaseline, IdentityMethod
    from src.methods.tier1 import TIER1_METHODS, SELECTIVE_METHODS
    from src.methods.tier2 import TIER2_METHODS
    from src.methods.baselines import BASELINE_METHODS
    from src.methods.composed import COMPOSED_METHODS
    from src.methods.random_pipelines import RANDOM_METHODS

    _registry = {
        "inductive_baseline": lambda: InductiveBaseline(),
        "identity": lambda: IdentityMethod(),
    }
    for k, v in TIER1_METHODS.items():
        if callable(v) and not hasattr(v, "fit_transform"):
            _registry[k] = v
        else:
            _registry[k] = lambda _v=v: _v
    _registry.update(SELECTIVE_METHODS)
    _registry.update(TIER2_METHODS)
    _registry.update(BASELINE_METHODS)
    _registry.update(COMPOSED_METHODS)
    _registry.update(RANDOM_METHODS)

    try:
        dataset = _load(ds_name, **ds_kwargs)
        method = _registry[method_name]()
        model = _get_model(model_name, task_type=dataset.task_type, seed=seed)

        # Detect the log transformation with the same rule as the main worker loop.
        log_target = False
        if dataset.task_type == "regression" and (dataset.y_train >= 0).all():
            from scipy.stats import skew as _skew
            if _skew(dataset.y_train) > 3.0:
                log_target = True

        # Train on the complete training partition. Fold and seed repetition are
        # defined by the manifest. Bootstrap sampling is disabled.
        bootstrap = False
        result = _run(dataset, method, model, seed=seed,
                      log_transform_target=log_target, bootstrap_train=bootstrap,
                      capture_path=capture_path_str,
                      capture_fold=ds_kwargs.get("fold_idx"),
                      capture_method_key=method_name)
        result_queue.put(("ok", result.to_dict()))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def _run_bag8_in_isolated_process(
    ds_name: str,
    ds_kwargs: dict,
    method_name: str,
    model_name: str,
    seed: int,
    capture_path: Path | str | None = None,
    timeout_seconds: int = 3600,
) -> dict:
    """Run one bag-8 experiment in a fresh spawn-mode subprocess.

    Returns the same dict as ``ExperimentResult.to_dict()``.
    Raises ``RuntimeError`` on subprocess failure or timeout.

    Consume the queue before joining the process because
    recommended by the multiprocessing documentation. This avoids deadlock if a
    payload exceeds the pipe buffer. The timeout also detects stalled children.
    """
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    proc = ctx.Process(
        target=_bag8_worker_inner,
        args=(result_queue, str(PROJECT_ROOT), ds_name, ds_kwargs, method_name,
              model_name, seed, str(capture_path) if capture_path is not None else None),
    )
    proc.start()

    try:
        status, payload = _await_bag8_result(
            proc,
            result_queue,
            timeout_seconds=timeout_seconds,
            identity=f"{ds_name} | {method_name} | seed={seed}",
        )
    except RuntimeError:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=5)
        raise

    # Subprocess wrote to queue. Drain by joining (should return immediately).
    proc.join(timeout=10)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)

    if status == "ok":
        return payload
    raise RuntimeError(f"bag-8 subprocess error: {payload}")


def _await_bag8_result(
    proc,
    result_queue,
    *,
    timeout_seconds: float,
    identity: str,
):
    """Wait for a Bag8 payload while detecting child death promptly."""
    import queue as _queue

    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"bag-8 subprocess timeout after {timeout_seconds}s "
                f"({identity}, exit={proc.exitcode})"
            )
        try:
            status, payload = result_queue.get(timeout=min(1.0, remaining))
            break
        except _queue.Empty:
            if proc.is_alive():
                continue
            # Give the multiprocessing queue feeder one final opportunity to
            # flush a payload after the child exits.
            try:
                status, payload = result_queue.get(timeout=0.5)
                break
            except _queue.Empty:
                proc.join(timeout=1)
                raise RuntimeError(
                    "bag-8 subprocess exited before returning a result "
                    f"({identity}, exit={proc.exitcode})"
                )
    return status, payload


def _canonical_optional_int(value: object) -> int | str:
    """Normalize CSV integers while preserving an empty optional value."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return ""
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"Expected an integer-like value, got {value!r}")
    return int(number)


def _identity_multiset(frame: pd.DataFrame, *, manifest: bool) -> Counter:
    """Return exact experiment identities for a manifest or task output."""
    method_column = "method" if manifest else "method_key"
    required = {"dataset", method_column, "model", "seed"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing identity columns: {sorted(missing)}")

    identities: Counter = Counter()
    for _, row in frame.iterrows():
        identities[(
            str(row["dataset"]),
            str(row[method_column]),
            str(row["model"]),
            _canonical_optional_int(row["seed"]),
            _canonical_optional_int(row.get("fold", "")),
        )] += 1
    return identities


def _completion_status(
    existing: pd.DataFrame,
    expected_runs: pd.DataFrame,
) -> tuple[bool, str]:
    """Validate exact task identity and terminal error state before skipping."""
    if len(existing) != len(expected_runs):
        return False, f"row-count mismatch {len(existing)} != {len(expected_runs)}"
    try:
        observed = _identity_multiset(existing, manifest=False)
        expected = _identity_multiset(expected_runs, manifest=True)
    except (TypeError, ValueError) as exc:
        return False, f"invalid identity schema: {exc}"
    if observed != expected:
        missing = sum((expected - observed).values())
        unexpected = sum((observed - expected).values())
        return False, f"identity mismatch (missing={missing}, unexpected={unexpected})"

    if "error" not in existing.columns:
        return True, "all exact identities completed"
    errors = existing["error"]
    is_skip = errors.map(is_deterministic_skip_error)
    n_errors = int((errors.notna() & ~is_skip).sum())
    if n_errors:
        return False, f"{n_errors} non-skip error rows"
    return True, "all exact identities completed or deterministically skipped"


def _read_completion_status(
    out_path: Path,
    expected_runs: pd.DataFrame,
) -> tuple[bool, str]:
    """Treat malformed or unreadable prior output as incomplete, never final."""
    try:
        existing = pd.read_csv(out_path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return False, f"unreadable prior output: {type(exc).__name__}: {exc}"
    return _completion_status(existing, expected_runs)


def _atomic_write_csv(frame: pd.DataFrame, out_path: Path) -> None:
    """Durably replace a task CSV through a same-directory temporary file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out_path.name}.", suffix=".tmp", dir=out_path.parent,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(
                handle,
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
                lineterminator="\n",
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# Method registry (same as run_sweep.py)
# ---------------------------------------------------------------------------
def _build_method_registry():
    from src.methods.base import InductiveBaseline, IdentityMethod
    from src.methods.tier1 import TIER1_METHODS, SELECTIVE_METHODS
    from src.methods.tier2 import TIER2_METHODS
    from src.methods.baselines import BASELINE_METHODS
    from src.methods.composed import COMPOSED_METHODS

    base = {
        "inductive_baseline": lambda: InductiveBaseline(),
        "identity": lambda: IdentityMethod(),
    }
    all_methods = {**base}
    for k, v in TIER1_METHODS.items():
        if callable(v) and not hasattr(v, "fit_transform"):
            all_methods[k] = v
        else:
            all_methods[k] = lambda _v=v: _v

    # Selective variants (method + column selector combinations)
    all_methods.update(SELECTIVE_METHODS)

    # Tier 2 methods (domain classifier, reweighting, etc.)
    all_methods.update(TIER2_METHODS)

    # Train-only intermediate baselines (groupby_train_only, pca_train_only_*, target_enc_standard_m10)
    all_methods.update(BASELINE_METHODS)

    # Composed joint and train-only pipelines for the exploratory composition
    # experiment. See src/methods/composed.py.
    all_methods.update(COMPOSED_METHODS)

    # Optional seeded random five-method chains.
    # Empty dict when configs/experiments/random_pipelines_draws.json is absent.
    from src.methods.random_pipelines import RANDOM_METHODS
    all_methods.update(RANDOM_METHODS)

    return all_methods


METHOD_REGISTRY = _build_method_registry()


def get_method(name: str):
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Available: {sorted(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name]()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Path to manifest.csv")
    p.add_argument("--results-dir", default="results/runs", help="Per-task output dir")
    p.add_argument("--capture-config", default=None,
                   help="Optional path to a capture-audit YAML. Runs whose "
                        "(dataset, method, model, seed) appear under `captures:` "
                        "save a pickled pre/post snapshot to --capture-dir.")
    p.add_argument("--capture-dir", default="results/captures",
                   help="Directory for capture artifacts (only used if --capture-config is set).")
    args = p.parse_args()

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) + int(os.environ.get("TASK_OFFSET", 0))
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "local")

    # Resolve relative paths against project root (handles any SLURM cwd)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir

    # Load capture audit (optional. Empty set means "no captures")
    capture_audit: set[tuple[str, str, str, int, int | None]] = set()
    capture_dir: Path | None = None
    if args.capture_config:
        from src.capture import load_audit
        audit_path = Path(args.capture_config)
        if not audit_path.is_absolute():
            audit_path = PROJECT_ROOT / audit_path
        capture_audit = load_audit(audit_path)
        capture_dir = Path(args.capture_dir)
        if not capture_dir.is_absolute():
            capture_dir = PROJECT_ROOT / capture_dir
        capture_dir.mkdir(parents=True, exist_ok=True)
        print(f"Capture audit loaded: {len(capture_audit)} entries from {audit_path}", flush=True)
        print(f"Capture output dir:   {capture_dir}", flush=True)

    # Read the manifest and select the assigned task.
    manifest = pd.read_csv(manifest_path)
    my_runs = manifest[manifest["task_id"] == task_id]

    if my_runs.empty:
        print(f"Task {task_id}: no runs assigned, exiting.", flush=True)
        return

    # Output file
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"task_{task_id:04d}.csv"

    # Skip a task only when every expected row completed successfully.
    # Rerunning the same array command retries tasks with failed rows.
    if out_path.exists():
        complete, detail = _read_completion_status(out_path, my_runs)
        if complete:
            print(
                f"Task {task_id}: already completed with exact identities "
                f"({len(my_runs)} runs), skipping.",
                flush=True,
            )
            return
        print(f"Task {task_id}: prior output incomplete ({detail}). Re-running.", flush=True)

    print(f"Task {task_id} (job {job_id}): {len(my_runs)} runs", flush=True)

    rows = []
    for _, run in my_runs.iterrows():
        ds_name = run["dataset"]
        method_name = run["method"]
        model_name = run["model"]
        seed = int(run["seed"])
        max_samples = int(run["max_samples"]) if pd.notna(run["max_samples"]) and run["max_samples"] != "" else None
        # Optional fold column for TabArena official-CV sweeps. It is absent for default
        # single-split sweeps. Empty string / NaN / missing column → no fold.
        fold_idx: int | None = None
        if "fold" in run.index:
            v = run["fold"]
            if pd.notna(v) and str(v) != "":
                fold_idx = int(v)

        fold_tag = f" | fold={fold_idx}" if fold_idx is not None else ""
        print(f"  {ds_name} | {method_name} | {model_name} | seed={seed}{fold_tag}",
              end=" … ", flush=True)

        row = {
            "dataset": ds_name,
            "method_key": method_name,
            "model": model_name,
            "seed": seed,
            "fold": fold_idx if fold_idx is not None else "",
        }
        execution_contract = get_model_execution_metadata(model_name)
        row.update(execution_contract)

        try:
            ds_kwargs = {"seed": seed}
            if max_samples:
                ds_kwargs["max_samples"] = max_samples
            if fold_idx is not None:
                ds_kwargs["fold_idx"] = fold_idx

            capture_path = None
            capture_identity = (ds_name, method_name, model_name, seed, fold_idx)
            capture_wildcard = (ds_name, method_name, model_name, seed, None)
            if capture_audit and (
                capture_identity in capture_audit or capture_wildcard in capture_audit
            ):
                from src.capture import CaptureKey, artifact_filename
                capture_path = capture_dir / artifact_filename(
                    CaptureKey(ds_name, method_name, model_name, seed, fold_idx)
                )

            if model_name == _BAG8_MODEL_NAME:
                # Each bag-8 run gets a fresh subprocess to prevent AutoGluon
                # process-global state drift across consecutive fits in a batch.
                payload = _run_bag8_in_isolated_process(
                    ds_name, ds_kwargs, method_name, model_name, seed,
                    capture_path=capture_path,
                    # The per-run wall-time limit is configurable for datasets requiring more time.
                    timeout_seconds=int(os.environ.get("BAG8_TIMEOUT", "10000")),
                )
                for key, expected in execution_contract.items():
                    observed = payload.get(key, expected)
                    if observed != expected:
                        raise RuntimeError(
                            f"model execution contract mismatch for {key}: "
                            f"expected {expected!r}, got {observed!r}"
                        )
                row.update(payload)
                print("ok (bag-8 subprocess)", flush=True)
            else:
                dataset = load_dataset(ds_name, **ds_kwargs)
                method = get_method(method_name)
                model = get_model(model_name, task_type=dataset.task_type, seed=seed)

                # Auto-detect log transform for highly skewed regression targets
                log_target = False
                if dataset.task_type == "regression" and (dataset.y_train >= 0).all():
                    from scipy.stats import skew as _skew
                    if _skew(dataset.y_train) > 3.0:
                        log_target = True

                # Train on the complete training partition. Fold and seed repetition are
                # defined by the manifest. Bootstrap sampling is disabled.
                bootstrap = False
                result = run_single_experiment(
                    dataset, method, model, seed=seed,
                    bootstrap_train=bootstrap,
                    log_transform_target=log_target,
                    capture_path=capture_path,
                    capture_fold=fold_idx,
                    capture_method_key=method_name,
                )
                payload = result.to_dict()
                for key, expected in execution_contract.items():
                    if payload.get(key) != expected:
                        raise RuntimeError(
                            f"model execution contract mismatch for {key}: "
                            f"expected {expected!r}, got {payload.get(key)!r}"
                        )
                row.update(payload)
                print(result.summary_line(), flush=True)

        except Exception as e:
            row["error"] = str(e)
            print(f"FAILED: {e}", flush=True)

        rows.append(row)

        # Checkpoint after every cell. Same-directory atomic replacement reduces the
        # risk of malformed output after interruption.
        _atomic_write_csv(pd.DataFrame(rows), out_path)

    print(f"Task {task_id}: done, wrote {len(rows)} results to {out_path}", flush=True)


if __name__ == "__main__":
    main()

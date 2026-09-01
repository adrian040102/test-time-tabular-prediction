"""
Create gap-widened TabReD dataset versions.

The script reads the measured windows produced by
``scripts/characterize_gap_widening.py`` and writes selected train/test slices
under ``data/datasets/``.

Each dataset receives a newest-window version, an oldest-window version and,
when distinct, a maximum-shift version. The test window and sample sizes remain
fixed. LightGBM is configured for every version. TabICL and TabPFN are configured
for the two endpoint versions. Labels are not used to select windows and the
original raw downloads are not required once the natural-split parquets exist.

Run from the repository root:
    python scripts/build_gap_widened_tabred.py
    python scripts/build_gap_widened_tabred.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset  # noqa: E402
from src.data.tabred_sampling import (  # noqa: E402
    default_gap_sizes,
    enumerate_gap_windows,
    slice_gap_dataset,
)

TARGET_COL = "__target__"
TIME_COL = "__time__"

KEEP = [
    "tabred_weather",
    "tabred_delivery_eta",
    "tabred_sberbank",
    "tabred_homesite",
]

BAG8 = "lightgbm_tabarena_bag8"
GPU_MODELS = ["tabicl_v2", "tabpfn_v2"]

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "data" / "datasets"
GAP_DIR = ROOT / "results" / "gap_widening"
WINDOWS_CSV = GAP_DIR / "gap_widening_windows.csv"


def _plan_versions(sub: pd.DataFrame) -> dict[str, str]:
    """Map version suffix -> window tag for one dataset's window curve.

    Returns an ordered dict-like with keys among {gap_near, gap_far,
    gap_maxdiff}. Drift collapses gap_far == gap_maxdiff to a single
    gap_far folder.
    """
    newest_tag = sub.loc[sub["start_position"].idxmax(), "tag"]
    oldest_tag = sub.loc[sub["start_position"].idxmin(), "tag"]
    maxdiff_tag = sub.loc[sub["shift_score"].idxmax(), "tag"]

    versions: dict[str, str] = {"gap_near": newest_tag}
    if maxdiff_tag == oldest_tag:
        # drift: oldest is the most different -> one high-end folder
        versions["gap_far"] = oldest_tag
    else:
        versions["gap_far"] = oldest_tag
        versions["gap_maxdiff"] = maxdiff_tag

    # Remove duplicate windows so each window tag produces one folder.
    seen: dict[str, str] = {}
    deduped: dict[str, str] = {}
    for suffix, tag in versions.items():
        if tag in seen:
            print(
                f"    Duplicate: {suffix} window == {seen[tag]} window (tag {tag}). "
                f"Skipping duplicate folder."
            )
            continue
        seen[tag] = suffix
        deduped[suffix] = tag
    return deduped


def _high_end_suffix(versions: dict[str, str]) -> str:
    """The high-shift end folder: gap_maxdiff if present else gap_far."""
    return "gap_maxdiff" if "gap_maxdiff" in versions else "gap_far"


def _write_version(
    source_ds,
    source_name: str,
    suffix: str,
    tag: str,
    win_row: pd.Series,
    plan,
    tag_to_window: dict,
    include_gpu_models: bool,
    dry_run: bool,
) -> dict:
    """Create one dataset version and return its index entry."""
    w = tag_to_window[tag]
    sliced = slice_gap_dataset(source_ds, w.train_idx, plan.test_idx, suffix)
    folder = OUT_ROOT / sliced.name  # e.g. tabred_weather__gap_near

    models = [BAG8] + (GPU_MODELS if include_gpu_models else [])

    clf = win_row.get("domain_clf_acc")
    clf_val = None if pd.isna(clf) else float(clf)

    entry = {
        "name": sliced.name,
        "source_dataset": source_name,
        "gap_version": suffix,
        "gpu_models_included": bool(include_gpu_models),
        "models": models,
        "n_train": int(len(sliced.X_train)),
        "n_test": int(len(sliced.X_test)),
        "shift_score": float(win_row["shift_score"]),
        "domain_clf_acc": clf_val,
        "mean_ks": float(win_row["mean_ks"]),
        "train_time_start": win_row["train_time_start"],
        "train_time_end": win_row["train_time_end"],
        "test_time_start": win_row["test_time_start"],
        "test_time_end": win_row["test_time_end"],
    }

    print(
        f"    {sliced.name:<34} train {entry['train_time_start'][:10]}"
        f"..{entry['train_time_end'][:10]}  n_train={entry['n_train']:,}"
        f"  shift={entry['shift_score']:.3f}  models={'+'.join(m.split('_')[0] for m in models)}"
    )

    if dry_run:
        return entry

    # Reconstruct on-disk frames: features + target + time (loader selects by name).
    train_df = sliced.X_train.copy()
    train_df[TARGET_COL] = np.asarray(sliced.y_train)
    train_df[TIME_COL] = np.asarray(sliced.metadata["time_train"])
    test_df = sliced.X_test.copy()
    test_df[TARGET_COL] = np.asarray(sliced.y_test)
    test_df[TIME_COL] = np.asarray(sliced.metadata["time_test"])

    # Require the training period to end before the test period starts.
    if train_df[TIME_COL].max() >= test_df[TIME_COL].min():
        raise ValueError(
            f"{sliced.name}: train/test time overlap "
            f"(max train {train_df[TIME_COL].max()} >= min test {test_df[TIME_COL].min()})"
        )

    folder.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(folder / "train.parquet", index=False)
    test_df.to_parquet(folder / "test.parquet", index=False)

    meta = {
        "name": sliced.name,
        "task_type": source_ds.task_type,
        "cat_features": list(source_ds.cat_features),
        "num_features": list(source_ds.num_features),
        "time_col": TIME_COL,
        "shift_description": (
            f"{source_name} version '{suffix}'. The training period from "
            f"{entry['train_time_start'][:10]} to {entry['train_time_end'][:10]} "
            f"was compared with the fixed latest test period. The measured "
            f"shift was {entry['shift_score']:.3f}."
        ),
        "n_train": entry["n_train"],
        "n_test": entry["n_test"],
        "n_features": len(source_ds.cat_features) + len(source_ds.num_features),
        "extra": {
            "source": "tabred",
            "tabred_split": "gap_widened",
            "source_dataset": source_name,
            "gap_version": suffix,
            "gpu_models_included": bool(include_gpu_models),
            "measured_shift_score": entry["shift_score"],
            "measured_domain_clf_acc": clf_val,
            "measured_mean_ks": entry["mean_ks"],
            "window_tag": tag,
            "window_start_position": int(w.start_position),
            "window_start_frac": float(w.start_frac),
            "n_train_total": int(plan.n_train_total),
            "n_test_total": int(plan.n_test_total),
            "train_time_start": entry["train_time_start"],
            "train_time_end": entry["train_time_end"],
            "test_time_start": entry["test_time_start"],
            "test_time_end": entry["test_time_end"],
        },
    }
    with open(folder / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=KEEP)
    ap.add_argument("--positions", type=int, default=10)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing parquet/meta.",
    )
    args = ap.parse_args()

    if not WINDOWS_CSV.is_file():
        sys.exit(
            f"Missing {WINDOWS_CSV}. Run scripts/characterize_gap_widening.py first."
        )
    windows = pd.read_csv(WINDOWS_CSV)

    index_entries: list[dict] = []
    for name in args.datasets:
        sub = windows[windows["dataset"] == name].copy()
        if sub.empty:
            print(f"\n=== {name} ===\n  SKIP (no rows in {WINDOWS_CSV.name})")
            continue

        ds = load_dataset(name)
        n_train, n_test = default_gap_sizes(len(ds.X_train), len(ds.X_test))
        plan = enumerate_gap_windows(ds, n_train, n_test, n_positions=args.positions)
        tag_to_window = {w.tag: w for w in plan.windows}

        versions = _plan_versions(sub)
        high_end = _high_end_suffix(versions)
        gpu_suffixes = {"gap_near", high_end}

        kind = (
            "shared far and maximum-difference period"
            if "gap_maxdiff" not in versions
            else "separate maximum-difference period"
        )
        print(f"\n=== {name} ===  [{kind}]  {len(versions)} version(s)")

        score_by_tag = sub.set_index("tag")
        for suffix, tag in versions.items():
            win_row = score_by_tag.loc[tag]
            entry = _write_version(
                source_ds=ds,
                source_name=name,
                suffix=suffix,
                tag=tag,
                win_row=win_row,
                plan=plan,
                tag_to_window=tag_to_window,
                include_gpu_models=(suffix in gpu_suffixes),
                dry_run=args.dry_run,
            )
            index_entries.append(entry)

    # Write the dataset index used during manifest generation.
    if index_entries and not args.dry_run:
        index_path = GAP_DIR / "gap_datasets_index.json"
        with open(index_path, "w") as f:
            json.dump(
                {
                    "n_positions": args.positions,
                    "bag8_model": BAG8,
                    "gpu_models": GPU_MODELS,
                    "datasets": index_entries,
                },
                f,
                indent=2,
            )
        n_bag8 = len(index_entries)
        n_gpu = sum(1 for e in index_entries if e["gpu_models_included"])
        print(
            f"\nWrote {len(index_entries)} version folders under {OUT_ROOT}"
            f"\n  bag8 datasets : {n_bag8}"
            f"\n  GPU datasets  : {n_gpu} (gap_near + high-shift end per dataset)"
            f"\nWrote index: {index_path}"
        )
    elif args.dry_run:
        n_gpu = sum(1 for e in index_entries if e["gpu_models_included"])
        print(
            f"\n[dry-run] would write {len(index_entries)} folders "
            f"({n_gpu} with GPU-model coverage). No files written."
        )
    else:
        print("\nNothing to write.")


if __name__ == "__main__":
    main()

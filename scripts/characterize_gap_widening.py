"""
Characterize gap-widening for the TabReD datasets.

For each dataset, the script fixes the test window at the latest rows, slides a
fixed-size training window over the earlier time-sorted data, measures feature
distribution differences and selects the newest, oldest and maximum-shift
training windows. Labels are not used in window selection.

Outputs under ``results/gap_widening/``:
    gap_widening_windows.csv
    gap_widening_chosen.csv

Run from the repository root:
    python scripts/characterize_gap_widening.py
    python scripts/characterize_gap_widening.py --datasets tabred_sberbank
    python scripts/characterize_gap_widening.py --positions 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Support direct execution from the repository checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset  # noqa: E402
from src.data.tabred_sampling import (  # noqa: E402
    default_gap_sizes,
    enumerate_gap_windows,
    slice_gap_dataset,
)
from src.evaluation.shift_diagnostics import ShiftDiagnostics  # noqa: E402


DEFAULT_DATASETS = [
    "tabred_sberbank",
    "tabred_homesite",
    "tabred_weather",
    "tabred_cooking_time",
    "tabred_delivery_eta",
    "tabred_ecom_offers",
]

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "gap_widening"


def _short(ts: str | None) -> str:
    return (ts or "?")[:10]


def characterize_one(name: str, n_positions: int) -> tuple[list[dict], list[dict]]:
    """Return (window_rows, chosen_rows) for one dataset."""
    ds = load_dataset(name)
    n_tr_total = len(ds.X_train)
    n_te_total = len(ds.X_test)
    n_train, n_test = default_gap_sizes(n_tr_total, n_te_total)

    plan = enumerate_gap_windows(ds, n_train, n_test, n_positions=n_positions)

    print(f"\n=== {name} ===")
    print(
        f"  full sizes : train={n_tr_total:,}  test={n_te_total:,}"
        f"   -> window sizes: n_train={n_train:,}  n_test={n_test:,}"
    )
    print(
        f"  fixed test : {_short(plan.test_time_start)} .. "
        f"{_short(plan.test_time_end)}  ({plan.n_test:,} rows)"
    )
    print(f"  scanning {len(plan.windows)} train-window positions:")

    window_rows: list[dict] = []
    for w in plan.windows:
        temp = slice_gap_dataset(ds, w.train_idx, plan.test_idx, w.tag)
        rep = ShiftDiagnostics(temp).full_report(run_domain_classifier=True)
        clf = rep.domain_classifier_accuracy
        row = {
            "dataset": name,
            "tag": w.tag,
            "start_position": w.start_position,
            "start_frac": round(w.start_frac, 3),
            "n_train": n_train,
            "n_test": n_test,
            "train_time_start": w.train_time_start,
            "train_time_end": w.train_time_end,
            "test_time_start": plan.test_time_start,
            "test_time_end": plan.test_time_end,
            "shift_score": round(float(rep.overall_shift_score), 4),
            "domain_clf_acc": (round(float(clf), 4) if clf is not None else None),
            "mean_ks": round(float(rep.mean_ks_statistic), 4),
        }
        window_rows.append(row)
        clf_str = f"{clf:.3f}" if clf is not None else "  -  "
        print(
            f"    frac {w.start_frac:0.2f}  train {_short(w.train_time_start)}"
            f"..{_short(w.train_time_end)}  shift={row['shift_score']:.3f}"
            f"  clf_acc={clf_str}  ks={row['mean_ks']:.3f}"
        )

    df = pd.DataFrame(window_rows)
    newest = df.loc[df["start_position"].idxmax()]
    oldest = df.loc[df["start_position"].idxmin()]
    maxdiff = df.loc[df["shift_score"].idxmax()]

    chosen_rows: list[dict] = []
    for label, r in [
        ("newest_train", newest),
        ("oldest_train", oldest),
        ("most_different_train", maxdiff),
    ]:
        chosen_rows.append(
            {
                "dataset": name,
                "version": label,
                "tag": r["tag"],
                "n_train": int(r["n_train"]),
                "n_test": int(r["n_test"]),
                "shift_score": float(r["shift_score"]),
                "domain_clf_acc": (
                    float(r["domain_clf_acc"])
                    if r["domain_clf_acc"] is not None
                    and not pd.isna(r["domain_clf_acc"])
                    else None
                ),
                "mean_ks": float(r["mean_ks"]),
                "train_time_start": r["train_time_start"],
                "train_time_end": r["train_time_end"],
            }
        )

    # Plain-language verdict for this dataset.
    coincide = maxdiff["tag"] == oldest["tag"]
    spread = float(df["shift_score"].max() - df["shift_score"].min())
    print(
        f"  picks: newest shift={newest['shift_score']:.3f} | "
        f"oldest shift={oldest['shift_score']:.3f} | "
        f"most-different shift={maxdiff['shift_score']:.3f} "
        f"(train {_short(maxdiff['train_time_start'])}..{_short(maxdiff['train_time_end'])})"
    )
    if coincide:
        print(
            "  -> most-different == oldest  : DRIFT-like "
            "(calendar gap is a good proxy here)."
        )
    else:
        print(
            "  -> most-different != oldest: the largest measured shift "
            "does not occur at the oldest window."
        )
    print(f"  -> shift spread across windows = {spread:.3f}"
          + ("   [limited shift range]" if spread < 0.05 else ""))

    return window_rows, chosen_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Which TabReD datasets to characterize.",
    )
    ap.add_argument(
        "--positions",
        type=int,
        default=10,
        help="Number of train-window positions to scan per dataset.",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_windows: list[dict] = []
    all_chosen: list[dict] = []
    for name in args.datasets:
        try:
            win_rows, chosen_rows = characterize_one(name, args.positions)
        except FileNotFoundError as e:
            print(f"\n=== {name} ===\n  SKIP (not on disk): {e}")
            continue
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"\n=== {name} ===\n  ERROR: {type(e).__name__}: {e}")
            continue
        all_windows.extend(win_rows)
        all_chosen.extend(chosen_rows)

    if all_windows:
        win_path = OUT_DIR / "gap_widening_windows.csv"
        chosen_path = OUT_DIR / "gap_widening_chosen.csv"
        pd.DataFrame(all_windows).to_csv(win_path, index=False)
        pd.DataFrame(all_chosen).to_csv(chosen_path, index=False)
        print(f"\nWrote {win_path}")
        print(f"Wrote {chosen_path}")
    else:
        print("\nNo datasets were characterized. No files were written.")


if __name__ == "__main__":
    main()

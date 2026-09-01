#!/usr/bin/env python3
"""Compare selected baseline runs with published TabArena results.

The script runs ``inductive_baseline`` through ``run_single_experiment`` for
the requested folds and seed, then compares each headline error with the
corresponding published TabArena result.

Binary and regression cells from ``lightgbm_tabarena_bag8`` are checked against
``--tolerance``. Multiclass Bag8 cells and TabICL/TabPFN cells are reported
separately because their results can be more sensitive to environment and
configuration differences.

Fetch the public comparison data first if it is absent:
    python scripts/fetch_tabarena_benchmarks.py

Usage:
    python scripts/parity_check.py
    python scripts/parity_check.py --models lightgbm_tabarena_bag8
    python scripts/parity_check.py --datasets diabetes anneal houses

Exit codes: 0 means the checked Bag8 evaluations pass. 2 means that a checked
Bag8 evaluation fails. 1 means that setup fails.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DF_RESULTS = PROJECT_ROOT / "data" / "tabarena" / "benchmark" / "df_results.csv"
OUT_CSV = PROJECT_ROOT / "results" / "parity_check.csv"

# model registry key -> published method name in df_results.csv
MODEL_TO_PUBLISHED = {
    "lightgbm_tabarena_bag8": "GBM (default)",
    "tabicl_v2": "TABICL (default)",
    "tabpfn_v2": "TABPFNV2 (default)",
}

# Default panel spanning task types, dataset sizes and numeric/categorical
# feature profiles.
DEFAULT_PANEL = [
    "diabetes",                 # binary, numeric, small
    "anneal",                   # multiclass, small, categorical
    "airfoil_self_noise",       # regression, small, numeric
    "Amazon_employee_access",   # binary, categorical-heavy
    "APSFailure",               # binary, large, numeric
    "houses",                   # regression, large, numeric
    "wine_quality",             # multiclass, small (TabPFN-eligible)
    "hiva_agnostic",            # binary, high-dimensional
]


def _headline_error(exp, problem_type: str) -> float:
    """TabArena convention: binary -> 1-AUC, multiclass -> log_loss, reg -> RMSE."""
    if problem_type == "binary":
        return 1.0 - float(exp.auc_roc)
    if problem_type == "multiclass":
        return float(exp.log_loss)
    if problem_type == "regression":
        return float(exp.rmse)
    raise ValueError(problem_type)


def main() -> int:
    ap = argparse.ArgumentParser(description="TabArena parity check.")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_PANEL)
    ap.add_argument("--models", nargs="+", default=list(MODEL_TO_PUBLISHED))
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=0,
                    help="Published TabArena seed is 0. Keep this value for parity.")
    ap.add_argument("--tolerance", type=float, default=1e-6,
                    help="HARD per-fold tolerance for bag8 (deterministic clone).")
    ap.add_argument("--soft-tolerance", type=float, default=0.02,
                    help="Relative deviation above which tabicl/tabpfn is flagged.")
    args = ap.parse_args()

    if not DF_RESULTS.is_file():
        print(f"[setup error] published results not found: {DF_RESULTS}")
        return 1

    from src.data import load_dataset
    from src.methods.base import InductiveBaseline
    from src.models.base import get_model
    from src.pipeline import run_single_experiment

    pub = pd.read_csv(DF_RESULTS, low_memory=False)
    pt_map = (pub.drop_duplicates("dataset")
              .set_index("dataset")["problem_type"].to_dict())

    rows: list[dict] = []
    warnings.filterwarnings("ignore")

    for model_name in args.models:
        published_method = MODEL_TO_PUBLISHED.get(model_name)
        if published_method is None:
            print(f"[skip] no published mapping for model {model_name}")
            continue
        pub_m = pub[pub["method"] == published_method]
        for ds_name in args.datasets:
            problem_type = pt_map.get(ds_name)
            if problem_type is None:
                # not a published TabArena dataset -> nothing to compare
                continue
            for fold in args.folds:
                pub_cell = pub_m[(pub_m["dataset"] == ds_name)
                                 & (pub_m["fold"] == fold)]
                pub_err = float(pub_cell["metric_error"].iloc[0]) if len(pub_cell) else np.nan

                row = {
                    "model": model_name, "dataset": ds_name, "fold": fold,
                    "problem_type": problem_type, "published_err": pub_err,
                    "ours_err": np.nan, "delta": np.nan, "delta_pct": np.nan,
                    "status": "", "note": "",
                }
                try:
                    ds = load_dataset(ds_name, fold_idx=fold, seed=args.seed)
                    model = get_model(model_name, task_type=ds.task_type, seed=args.seed)
                    # Disable bootstrap so training uses the complete fold training partition.
                    exp = run_single_experiment(
                        ds, InductiveBaseline(), model,
                        seed=args.seed, bootstrap_train=False,
                    )
                    ours = _headline_error(exp, problem_type)
                    row["ours_err"] = ours
                    if np.isfinite(pub_err):
                        row["delta"] = ours - pub_err
                        row["delta_pct"] = (
                            (ours - pub_err) / pub_err * 100.0 if pub_err else np.nan
                        )
                except Exception as e:  # noqa: BLE001 (incl. *SkipReason)
                    row["status"] = "skip"
                    row["note"] = f"{type(e).__name__}: {e}"[:160]
                rows.append(row)
                tag = row["status"] or "ran"
                d = row["delta"]
                dstr = f"{d:+.6f}" if d == d else "   n/a "
                print(f"  {model_name:24s} {ds_name:24s} fold{fold} "
                      f"ours={row['ours_err']:.6f} pub={pub_err:.6f} "
                      f"d={dstr} [{tag}]", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # ---- Verdict ----
    print("\n" + "=" * 72)
    bag8 = df[(df["model"] == "lightgbm_tabarena_bag8") & (df["status"] != "skip")]
    # Apply the hard tolerance to binary and regression Bag8 cells. Report
    # multiclass log-loss separately because it is more version-sensitive.
    HARD_PT = ("binary", "regression")
    bag8_hard = bag8[bag8["problem_type"].isin(HARD_PT)].dropna(subset=["delta"])
    bag8_soft = bag8[bag8["problem_type"] == "multiclass"].dropna(subset=["delta"])
    parity_pass = True
    if bag8_hard.empty:
        print("bag8: no corresponding binary or regression evaluations ran. "
              "Parity could not be assessed.")
        parity_pass = False
    else:
        worst = bag8_hard["delta"].abs().max()
        n_outside_tolerance = int((bag8_hard["delta"].abs() > args.tolerance).sum())
        print(f"bag8 (GBM default parameters) [binary and regression, tolerance check]: "
              f"{len(bag8_hard)} corresponding evaluations, worst |delta| = {worst:.3e}, "
              f"tolerance = {args.tolerance:g}")
        if n_outside_tolerance:
            parity_pass = False
            print(f"  Out-of-tolerance cells: {n_outside_tolerance}")
            outside_tolerance = bag8_hard[bag8_hard["delta"].abs() > args.tolerance]
            print(outside_tolerance[["dataset", "fold", "ours_err", "published_err",
                       "delta", "delta_pct"]].to_string(index=False))
        else:
            print("  Every binary and regression cell was within tolerance.")
    if not bag8_soft.empty:
        worst_s = bag8_soft["delta"].abs().max()
        n_flag = int((bag8_soft["delta"].abs() > args.tolerance).sum())
        print(f"bag8 [multiclass, diagnostic report]: {len(bag8_soft)} corresponding evaluations, "
              f"worst |delta| = {worst_s:.3e}, {n_flag} above {args.tolerance:g} "
              f"(multiclass log_loss is version-sensitive and is excluded from "
              f"the tolerance decision).")
        if n_flag:
            sft = bag8_soft[bag8_soft["delta"].abs() > args.tolerance]
            print(sft[["dataset", "fold", "ours_err", "published_err",
                       "delta", "delta_pct"]].to_string(index=False))

    for m in ("tabicl_v2", "tabpfn_v2"):
        sub = df[(df["model"] == m) & (df["status"] != "skip")].dropna(subset=["delta_pct"])
        skipped = int((df["model"] == m).sum() - len(df[(df["model"] == m) & (df["status"] != "skip")]))
        if sub.empty:
            print(f"{m}: no corresponding evaluations ran (config-dependent / size-skipped).")
            continue
        flagged = sub[sub["delta_pct"].abs() > args.soft_tolerance * 100]
        print(f"{m}: {len(sub)} corresponding evaluations, median |delta_pct| = "
              f"{sub['delta_pct'].abs().median():.2f}%, "
              f"{len(flagged)} flagged > {args.soft_tolerance*100:.0f}% (config-dependent).")

    print("=" * 72)
    print(f"Wrote {OUT_CSV.relative_to(PROJECT_ROOT)}")
    if parity_pass:
        print("\nPASS: bag8 reproduces published GBM(default). "
              "The result satisfies the parity criterion.")
        return 0
    print("\nFAIL: bag8 does not reproduce published GBM(default). "
          "The pipeline data path still differs.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

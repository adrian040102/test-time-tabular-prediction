#!/usr/bin/env python3
"""
Fetch public TabArena benchmark data from Hugging Face.

The downloaded files provide per-fold model results and an aggregated
leaderboard for comparison with locally evaluated methods.

Default output:
    data/tabarena/benchmark/
        df_results.parquet
        df_results.csv
        tabarena_leaderboard.csv
        fetch_info.json

Usage:
    python scripts/fetch_tabarena_benchmarks.py
    python scripts/fetch_tabarena_benchmarks.py --output some/other/dir
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Public benchmark files mirrored from the Hugging Face dataset repository.
HF_REPO = "TabArena/benchmark_results"
HF_FILES = [
    "df_results.parquet",         # per-fold results
    "df_results.csv",             # per-fold results in CSV form
    "tabarena_leaderboard.csv",   # aggregated leaderboard
]

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "tabarena" / "benchmark"


def hf_resolve_url(repo: str, filename: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"


def _download(url: str, dest: Path) -> int:
    """Download ``url`` to ``dest`` and return the number of bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    n_bytes = 0
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)  # 1 MB
            if not chunk:
                break
            f.write(chunk)
            n_bytes += len(chunk)
    tmp.replace(dest)
    return n_bytes


def main():
    p = argparse.ArgumentParser(description="Fetch TabArena leaderboard CSV.")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help=f"Output dir (default: {DEFAULT_OUTPUT})")
    p.add_argument("--revision", default="main",
                   help="HF revision/branch (default: main).")
    p.add_argument("--files", nargs="*", default=None,
                   help=f"Override file list (default: {HF_FILES})")
    args = p.parse_args()

    files = args.files or HF_FILES
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(files)} file(s) from {HF_REPO}@{args.revision}",
          flush=True)
    print(f"  -> {out_dir}\n", flush=True)

    info = {
        "repo": HF_REPO,
        "revision": args.revision,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": [],
    }

    for fname in files:
        url = hf_resolve_url(HF_REPO, fname, revision=args.revision)
        dest = out_dir / fname
        print(f"  {fname} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            n_bytes = _download(url, dest)
            elapsed = time.time() - t0
            print(f"OK ({n_bytes / 1e6:.2f} MB, {elapsed:.1f}s)", flush=True)
            info["files"].append({
                "name": fname,
                "url": url,
                "bytes": n_bytes,
                "status": "ok",
            })
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            info["files"].append({
                "name": fname,
                "url": url,
                "status": "failed",
                "error": str(e),
            })

    info_path = out_dir / "fetch_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nFetch info written to {info_path}", flush=True)

    # Validate the required columns in the per-fold results.
    pq = out_dir / "df_results.parquet"
    if pq.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(pq)
            print(f"\nPer-fold results ({pq.name}): "
                  f"{len(df):,} rows x {df.shape[1]} columns", flush=True)
            print(f"  columns: {list(df.columns)}", flush=True)
            if "dataset" in df.columns:
                n_datasets = df["dataset"].nunique()
                print(f"  unique datasets: {n_datasets}", flush=True)
            if "method" in df.columns:
                n_methods = df["method"].nunique()
                print(f"  unique methods:  {n_methods}", flush=True)
        except Exception as e:
            print(f"\n(could not preview df_results.parquet: {e})", flush=True)

    lb = out_dir / "tabarena_leaderboard.csv"
    if lb.exists():
        try:
            import pandas as pd
            df = pd.read_csv(lb, nrows=5)
            print(f"\nLeaderboard ({lb.name}): {df.shape[1]} columns",
                  flush=True)
            print(f"  columns: {list(df.columns)}", flush=True)
        except Exception as e:
            print(f"\n(could not preview leaderboard: {e})", flush=True)

    n_failed = sum(f["status"] == "failed" for f in info["files"])
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

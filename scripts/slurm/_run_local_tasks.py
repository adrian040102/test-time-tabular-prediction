"""Local task-loop driver: run a manifest's tasks 0..N-1 sequentially in one
activated env. Each task is an isolated subprocess (so the bag-8 spawn inherits
the activated PATH via sys.executable). Per-task CSVs -> resumable + monitorable.

Usage:
  python scripts/slurm/_run_local_tasks.py <manifest> <results_dir> <n_tasks>
"""
import os, subprocess, sys

manifest, results_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
for t in range(n):
    env = dict(os.environ, SLURM_ARRAY_TASK_ID=str(t),
               BAG8_TIMEOUT=os.environ.get("BAG8_TIMEOUT", "3600"))
    print(f"\n===== TASK {t}/{n-1} =====", flush=True)
    subprocess.run(
        [sys.executable, "scripts/slurm/worker.py",
         "--manifest", manifest, "--results-dir", results_dir],
        env=env, check=False,
    )
print("\nAll tasks complete.", flush=True)

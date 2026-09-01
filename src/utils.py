"""
Utility functions for the experiment framework.
"""

from __future__ import annotations

from pathlib import Path

import yaml
import numpy as np


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Seed the available random-number generators."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if not."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_cache_env(strict: bool = False) -> list[str]:
    """
    Check that cache-heavy environment variables do not point to $HOME.

    Call this early in any entry-point script (run_sweep, worker, etc.)
    to catch misconfigurations before they fill the home directory quota.

    Args:
        strict: If True, raise RuntimeError on problems. If False,
                print warnings and return the list of problem descriptions.

    Returns:
        List of warning strings (empty if everything is OK).
    """
    import os
    import warnings

    home = os.path.expanduser("~")
    checks = {
        "OPENML_CACHE_DIRECTORY": "OpenML datasets (can be >50 GB)",
        "THESIS_DATA_CACHE": "Thesis datasets (bike_sharing, shifts_weather, etc.)",
        "FOLKTABLES_CACHE_DIR": "US Census / folktables data",
        "XDG_CACHE_HOME": "General XDG cache (pip, misc)",
    }

    problems = []
    for var, desc in checks.items():
        val = os.environ.get(var, "")
        if not val or val.startswith(home):
            loc = val if val else f"(not set -> defaults to {home})"
            problems.append(f"  {var} = {loc}  [{desc}]")

    if problems:
        msg = (
            "Cache directories point to your home directory!\n"
            "This will likely exceed your disk quota.\n"
            "Set the cache environment variables to writable locations with "
            "sufficient space before running. See the repository README.\n\n"
            + "\n".join(problems)
        )
        if strict:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg, stacklevel=2)

    return problems

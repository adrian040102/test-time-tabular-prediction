"""Seeded five-method pipelines for an auxiliary composition analysis.

For each dataset, two seeded draws select variants from five distinct method
families and order them by method type in :class:`CatPreservingMethodPipeline`.
When present, draw specifications are loaded from
``configs/experiments/random_pipelines_draws.json`` and registered as
``random_<dataset>_d<draw>``. These auxiliary keys remain outside the counted
tier and baseline registries.

Row-augmenting members may prevent later categorical re-injection when row
counts diverge. The pipeline records this in metadata. Sample weights are
combined according to the pipeline weight contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.methods.base import CatPreservingMethodPipeline

ROOT = Path(__file__).resolve().parent.parent.parent  # src/methods/ -> repository root
DRAWS_JSON = ROOT / "configs" / "experiments" / "random_pipelines_draws.json"


def _component_registry() -> dict:
    """Return all single-method factories used by the sweep worker.

    The result excludes composed, lattice and random keys because a random
    chain contains only single methods.
    """
    from src.methods.tier1 import TIER1_METHODS, SELECTIVE_METHODS
    from src.methods.tier2 import TIER2_METHODS
    from src.methods.baselines import BASELINE_METHODS

    reg: dict = {}
    for k, v in TIER1_METHODS.items():
        if callable(v) and not hasattr(v, "fit_transform"):
            reg[k] = v
        else:
            reg[k] = lambda _v=v: _v
    reg.update(SELECTIVE_METHODS)
    reg.update(TIER2_METHODS)
    reg.update(BASELINE_METHODS)
    return reg


def make_random_chain(method_keys: tuple[str, ...] | list[str]) -> CatPreservingMethodPipeline:
    """Instantiate the recorded chain (fresh component instances per call)."""
    reg = _component_registry()
    missing = [k for k in method_keys if k not in reg]
    if missing:
        raise KeyError(
            f"random chain references unknown method keys {missing}. "
            "Check the optional random-pipeline draw specification"
        )
    return CatPreservingMethodPipeline([reg[k]() for k in method_keys])


def _load_random_methods() -> dict[str, callable]:
    if not DRAWS_JSON.exists():
        print(
            f"[random_pipelines] optional auxiliary draws are absent ({DRAWS_JSON}). "
            "Random-pipeline methods are disabled. "
            "They are outside the reported protocol.",
            file=sys.stderr,
        )
        return {}
    spec = json.loads(DRAWS_JSON.read_text(encoding="utf-8"))
    out: dict[str, callable] = {}
    for draw in spec["draws"]:
        out[draw["key"]] = (
            lambda _keys=tuple(draw["methods"]): make_random_chain(_keys)
        )
    return out


RANDOM_METHODS: dict[str, callable] = _load_random_methods()

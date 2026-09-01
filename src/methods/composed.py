"""Composed test-time pipelines and corresponding train-only pipelines.

The pipelines combine frequency encoding, group-by aggregation and an optional
low-dimensional PCA representation. They are registered separately from the
canonical single-method configurations.

Implementation constraints:

  * Chain with :class:`CatPreservingMethodPipeline`, not plain
    :class:`MethodPipeline`. The plain pipeline re-wraps each step's output as
    a float array and loses categorical dtype between steps, so the groupby
    step would see zero cats after the freq step consumes them.  CatPreserving
    re-injects the original cats at every step boundary (the same wrapper
    greedy uses for ``len(methods) >= 2`` stacks). The sweep worker calls
    ``run_single_experiment`` directly and does not wrap the pipeline, so the factory
    itself must return the cat-preserving wrapper.

  * Components are created from the same registry factories used for
    single-method runs (``TIER1_METHODS`` / ``BASELINE_METHODS``).

  * Two compositions have joint and train-only versions:
      - C   = freq -> groupby -> pca. ``JointPCA`` sets
              ``skip_pipeline_scaler=True`` so a generic scaler does not alter
              its coordinates.
      - Cfg = freq -> groupby         (reduced version without PCA).

Each pair differs in whether the test features inform the transformations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.methods.base import (
    CatPreservingMethodPipeline,
    MethodPipeline,
    TestTimeMethod,
    TransformResult,
)
from src.methods.baselines import (
    BASELINE_METHODS,
    TrainOnlyGroupByFeatures,
)
from src.methods.tier1 import TIER1_METHODS, JointGroupByFeatures


# --- component builders (fresh instance per call, via the canonical factories)
def _joint_freq():
    return TIER1_METHODS["joint_freq_combined"]()


def _joint_groupby():
    return TIER1_METHODS["joint_groupby_combined"]()


def _joint_pca10():
    return TIER1_METHODS["joint_pca_10"]()


def _train_only_freq():
    return BASELINE_METHODS["freq_enc_train_only"]()


def _train_only_groupby():
    return BASELINE_METHODS["groupby_train_only"]()


def _train_only_pca10():
    return BASELINE_METHODS["pca_train_only_10"]()


# --- composed pipeline factories --------------------------------------------
def make_composed_fgp_joint() -> CatPreservingMethodPipeline:
    """Combine joint frequency encoding, group-by aggregation and PCA."""
    return CatPreservingMethodPipeline([_joint_freq(), _joint_groupby(), _joint_pca10()])


def make_composed_fgp_train_only() -> CatPreservingMethodPipeline:
    """Use train-only frequency encoding, group-by aggregation and PCA."""
    return CatPreservingMethodPipeline(
        [_train_only_freq(), _train_only_groupby(), _train_only_pca10()]
    )


def make_composed_fg_joint() -> CatPreservingMethodPipeline:
    """Combine joint frequency encoding and group-by aggregation without PCA."""
    return CatPreservingMethodPipeline([_joint_freq(), _joint_groupby()])


def make_composed_fg_train_only() -> CatPreservingMethodPipeline:
    """Use train-only frequency encoding and group-by aggregation."""
    return CatPreservingMethodPipeline([_train_only_freq(), _train_only_groupby()])


# ==============================================================================
# Composition lattice
# ==============================================================================
# Predefined lattice over steps that have joint and train-only versions
#   f = frequency encoding    (joint_freq_combined   / freq_enc_train_only)
#   t = target encoding       (target_enc_ratio      / target_enc_standard_m10)
#   g = capped group-by aggregation, implemented by CappedGroupByStep below
#   p = PCA-10                (joint_pca_10          / pca_train_only_10)
# in the canonical order f -> t -> g -> p. Every subset of at least two steps
# is registered, together with the capped single-group reference pair and the
# plain-pipeline diagnostic comparison.


def _joint_target():
    return TIER1_METHODS["target_enc_ratio"]()


def _train_only_target():
    return BASELINE_METHODS["target_enc_standard_m10"]()


# --- capped groupby step ------------------------------------------------------
# Suffixes for columns added by earlier lattice encoder steps. These define the
# composition, so the cap keeps them first. Values in these columns differ
# between the joint and train-only versions. The cap rule may use only their
# names, not their values, to select the same features for both versions.
_ENCODER_COL_SUFFIXES = ("_freq_joint", "_freq_train", "_target_enc")


@dataclass
class GroupBySpec:
    """Explicit, deterministic groupby spec resolved from the arriving X_train."""

    group_cols: list[str]
    agg_cols: list[str]
    agg_funcs: list[str]
    audit: dict = field(default_factory=dict)


def resolve_capped_groupby_spec(
    X_train: pd.DataFrame,
    max_new_features: int = 300,
    max_unique: int = 100,
) -> GroupBySpec:
    """Resolve an explicit (group_cols, agg_cols, agg_funcs) spec under a cap.

    Mirrors the canonical auto-resolution of JointGroupByFeatures /
    TrainOnlyGroupByFeatures (cats with <= max_unique uniques as groups,
    low-cardinality-numeric fallback, numerics with std>0 as agg targets), then
    applies the predefined deterministic reduction when the projected new
    feature count exceeds ``max_new_features``:

      1. under budget -> pass through unchanged, as in uncapped automatic mode
      2. drop 'std' and 'count' agg funcs (keep 'mean')
      3. rank agg cols: encoder-derived columns first (name order), then
         original numerical columns by descending training variance (stable tie-break = frame
         order), then take the top K within budget
      4. last resort (>max_new group cols): drop highest-train-cardinality
         group cols first

    The rule uses only X_train. For encoder columns whose values depend on the
    version, it uses only names and structure. The joint and train-only versions
    therefore select the same features and differ only in the aggregation values.
    """
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    group_cols = [c for c in cat_cols if X_train[c].nunique() <= max_unique]
    if not group_cols:
        # Fallback identical to the canonical classes: low-cardinality numerics
        # (label-encoded categoricals from an earlier stage).
        group_cols = [c for c in num_cols if X_train[c].nunique() <= max_unique]

    group_set = set(group_cols)
    agg_cols = [
        c for c in num_cols
        if c not in group_set and X_train[c].std() > 0
    ]
    funcs = ["mean", "count", "std"]

    audit: dict = {
        "cap": int(max_new_features),
        "n_group_candidates": len(group_cols),
        "n_agg_candidates": len(agg_cols),
        "projected_uncapped": len(group_cols) * len(agg_cols) * len(funcs),
        "reduced": False,
    }

    if not group_cols or not agg_cols:
        # Canonical classes do nothing when either list is empty.
        return GroupBySpec(group_cols, agg_cols, funcs, audit)

    if len(group_cols) * len(agg_cols) * len(funcs) <= max_new_features:
        return GroupBySpec(group_cols, agg_cols, funcs, audit)

    audit["reduced"] = True

    # Tier 2 reduction: mean only.
    funcs = ["mean"]

    # Tier 3: rank aggregate columns. Encoder-derived columns come first in name
    # order because their values differ between versions but their names do not.
    # The remaining numerical columns follow in descending training variance.
    # The values are the same in both versions. A stable sort retains frame
    # order when variances are equal.
    enc_cols = [c for c in agg_cols if c.endswith(_ENCODER_COL_SUFFIXES)]
    other_cols = [c for c in agg_cols if not c.endswith(_ENCODER_COL_SUFFIXES)]
    variances = {
        c: float(v) if np.isfinite(v) else 0.0
        for c, v in X_train[other_cols].var().items()
    } if other_cols else {}
    other_ranked = sorted(other_cols, key=lambda c: -variances[c])
    ranked = enc_cols + other_ranked

    # Tier 4: group cols, lowest train-cardinality kept first. Dropping only
    # applies when one aggregate column over all groups exceeds the budget.
    g_ranked = sorted(group_cols, key=lambda c: X_train[c].nunique())
    groups_sel: list[str] = []
    aggs_sel: list[str] = []
    for n_g in range(len(g_ranked), 0, -1):
        k = max_new_features // (n_g * len(funcs))
        if k >= 1:
            groups_sel = g_ranked[:n_g]
            aggs_sel = ranked[: min(k, len(ranked))]
            break
    if not groups_sel:  # max_new_features < 1 is unusual but defined
        groups_sel, aggs_sel = g_ranked[:1], ranked[:1]

    audit["n_groups_selected"] = len(groups_sel)
    audit["n_aggs_selected"] = len(aggs_sel)
    audit["projected_capped"] = len(groups_sel) * len(aggs_sel) * len(funcs)
    audit["funcs"] = list(funcs)
    return GroupBySpec(groups_sel, aggs_sel, funcs, audit)


class CappedGroupByStep(TestTimeMethod):
    """Groupby step with a deterministic new-feature cap (lattice g-step).

    Resolve the specification from X_train so the joint and train-only versions
    select the same features. The canonical class then handles the transformation.
    The specification is recomputed inside every fit_transform call and is not
    stored on the instance.
    """

    def __init__(self, joint: bool, max_new_features: int = 300):
        self.joint = bool(joint)
        self.max_new_features = int(max_new_features)

    @property
    def name(self) -> str:
        base = "joint_groupby" if self.joint else "groupby_train_only"
        return f"{base}_cap{self.max_new_features}"

    def fit_transform(self, X_train, X_test, y_train) -> TransformResult:
        spec = resolve_capped_groupby_spec(X_train, self.max_new_features)
        if self.joint:
            inner = JointGroupByFeatures(
                group_cols=spec.group_cols,
                agg_cols=spec.agg_cols,
                agg_funcs=spec.agg_funcs,
                variant="combined",
            )
        else:
            inner = TrainOnlyGroupByFeatures(
                group_cols=spec.group_cols,
                agg_cols=spec.agg_cols,
                agg_funcs=spec.agg_funcs,
            )
        result = inner.fit_transform(X_train, X_test, y_train)
        result.metadata = {**(result.metadata or {}), "groupby_cap": spec.audit}
        return result


# --- lattice factories --------------------------------------------------------
# The order f -> t -> g -> p is fixed: encode, aggregate and then compress.
# The "g" version is the capped single-group-by reference.
LATTICE_COMPOSITIONS: list[str] = [
    "g",
    "ft", "fg", "fp", "tg", "tp", "gp",
    "ftg", "ftp", "fgp", "tgp", "ftgp",
]

_LATTICE_STEP_JOINT = {
    "f": _joint_freq,
    "t": _joint_target,
    "g": lambda: CappedGroupByStep(joint=True),
    "p": _joint_pca10,
}
_LATTICE_STEP_TRAIN_ONLY = {
    "f": _train_only_freq,
    "t": _train_only_target,
    "g": lambda: CappedGroupByStep(joint=False),
    "p": _train_only_pca10,
}


def make_lattice_pipeline(composition: str, joint: bool) -> CatPreservingMethodPipeline:
    """Build one lattice version, such as ``make_lattice_pipeline("fgp", joint=True)``."""
    steps = _LATTICE_STEP_JOINT if joint else _LATTICE_STEP_TRAIN_ONLY
    unknown = [c for c in composition if c not in steps]
    if unknown:
        raise ValueError(f"Unknown lattice step(s) {unknown} in {composition!r}")
    return CatPreservingMethodPipeline([steps[c]() for c in composition])


def make_lattice_plain_fgp(joint: bool) -> MethodPipeline:
    """Run fgp through the plain pipeline without categorical re-injection.

    After the frequency step label-encodes the categorical columns, conversion
    to a float array removes their categorical dtype. The group-by step then uses
    low-cardinality numerical group detection. The difference from
    ``lattice_fgp_*`` measures the effect of categorical re-injection.
    """
    steps = _LATTICE_STEP_JOINT if joint else _LATTICE_STEP_TRAIN_ONLY
    return MethodPipeline([steps[c]() for c in "fgp"])


# --- registry ----------------------------------------------------------------
# Auxiliary methods are merged into the worker registry but kept outside the
# canonical tier and baseline registries.
COMPOSED_METHODS: dict[str, callable] = {
    # Auxiliary single-composition keys use the uncapped group-by step. The
    # lattice keys below use the capped step.
    "composed_fgp_joint": make_composed_fgp_joint,
    "composed_fgp_train_only": make_composed_fgp_train_only,
    "composed_fg_joint": make_composed_fg_joint,
    "composed_fg_train_only": make_composed_fg_train_only,
}

for _comp in LATTICE_COMPOSITIONS:
    COMPOSED_METHODS[f"lattice_{_comp}_joint"] = (
        lambda _c=_comp: make_lattice_pipeline(_c, joint=True)
    )
    COMPOSED_METHODS[f"lattice_{_comp}_train_only"] = (
        lambda _c=_comp: make_lattice_pipeline(_c, joint=False)
    )

COMPOSED_METHODS["lattice_fgp_plain_joint"] = lambda: make_lattice_plain_fgp(joint=True)
COMPOSED_METHODS["lattice_fgp_plain_train_only"] = lambda: make_lattice_plain_fgp(joint=False)

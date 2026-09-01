"""Temporal subsampling utilities for TabReD datasets.

The parquet loader stores the training and test timestamps in dataset metadata.
``sample_tabred`` selects the earliest training rows and latest test rows after
sorting each split by time. It returns a new ``TabularDataset`` without changing
the parquet files or the input object. The selected row counts are recorded in
``metadata["sample_spec"]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.base import TabularDataset


def _coerce_time_array(arr: Any) -> np.ndarray:
    """Coerce a time array into pandas-comparable values for sorting.

    Accepts pandas.DatetimeIndex, np.datetime64 array, Python datetimes,
    pandas.Timestamp values or numeric epoch-style values. The returned flat
    one-dimensional array supports stable argsort.
    """
    if arr is None:
        raise ValueError("Time array is None.")
    a = np.asarray(arr)
    if a.dtype.kind in ("M", "f", "i", "u"):
        # Datetime64, float, int, uint. All sortable.
        return a.ravel()
    # Object dtype: try to coerce via pandas.
    return pd.to_datetime(pd.Series(a.ravel())).to_numpy()


def sample_tabred(
    dataset: TabularDataset,
    x_train_first: int,
    y_test_last: int,
) -> TabularDataset:
    """Return a shrunk dataset with widened temporal gap.

    Parameters
    ----------
    dataset:
        A TabReD-style :class:`TabularDataset` produced by ``load_dataset``.
        Must carry ``time_col``, ``time_train``, ``time_test`` keys in
        ``metadata`` (the parquet loader stashes them automatically when
        ``meta.json`` declares a ``time_col``).
    x_train_first:
        Number of earliest-in-time train rows to keep. Must be positive.
        Values >= ``len(X_train)`` are clipped (returns full train).
    y_test_last:
        Number of latest-in-time test rows to keep. Must be positive.
        Values >= ``len(X_test)`` are clipped (returns full test).

    Returns
    -------
    A new :class:`TabularDataset` with ``X_train``, ``X_test``,
    ``y_train``, ``y_test`` re-sliced and a refreshed ``metadata`` that
    carries the new ``time_train``/``time_test`` arrays plus a
    ``sample_spec`` field for traceability.

    Raises
    ------
    ValueError
        If ``dataset.metadata`` lacks the time columns or if either
        sample size is non-positive.
    """
    if x_train_first <= 0:
        raise ValueError(f"x_train_first must be > 0, got {x_train_first}")
    if y_test_last <= 0:
        raise ValueError(f"y_test_last must be > 0, got {y_test_last}")

    meta = dataset.metadata or {}
    time_col = meta.get("time_col")
    if not time_col:
        raise ValueError(
            f"Dataset {dataset.name!r} has no 'time_col' in metadata: "
            f"sample_tabred only works on TabReD-style datasets."
        )
    if "time_train" not in meta or "time_test" not in meta:
        raise ValueError(
            f"Dataset {dataset.name!r} carries 'time_col' but no time arrays. "
            f"Re-load via load_dataset() to populate metadata['time_train']."
        )

    t_train = _coerce_time_array(meta["time_train"])
    t_test = _coerce_time_array(meta["time_test"])

    n_train = len(dataset.X_train)
    n_test = len(dataset.X_test)
    if len(t_train) != n_train:
        raise ValueError(
            f"metadata['time_train'] has {len(t_train)} entries but "
            f"X_train has {n_train} rows."
        )
    if len(t_test) != n_test:
        raise ValueError(
            f"metadata['time_test'] has {len(t_test)} entries but "
            f"X_test has {n_test} rows."
        )

    # Use stable argsort so ties retain their original order.
    train_order = np.argsort(t_train, kind="stable")
    test_order = np.argsort(t_test, kind="stable")

    keep_train_n = min(x_train_first, n_train)
    keep_test_n = min(y_test_last, n_test)

    train_idx = train_order[:keep_train_n]
    test_idx = test_order[n_test - keep_test_n :]

    X_train = dataset.X_train.iloc[train_idx].reset_index(drop=True)
    X_test = dataset.X_test.iloc[test_idx].reset_index(drop=True)
    y_train = np.asarray(dataset.y_train)[train_idx]
    # y_test may be None if a LeakageGuard already nulled it. Only slice if present.
    if dataset.y_test is not None:
        y_test = np.asarray(dataset.y_test)[test_idx]
    else:
        y_test = None

    new_t_train = t_train[train_idx]
    new_t_test = t_test[test_idx]

    new_metadata = dict(meta)
    new_metadata["time_train"] = new_t_train
    new_metadata["time_test"] = new_t_test
    new_metadata["sample_spec"] = {
        "x_train_first": int(keep_train_n),
        "y_test_last": int(keep_test_n),
        "x_train_first_requested": int(x_train_first),
        "y_test_last_requested": int(y_test_last),
        "n_train_pre_sample": int(n_train),
        "n_test_pre_sample": int(n_test),
    }
    # Record a string summary when the dtype supports timestamp conversion.
    try:
        gap_after = pd.Timestamp(new_t_test.min()) - pd.Timestamp(new_t_train.max())
        new_metadata["sample_gap_after"] = str(gap_after)
    except (TypeError, ValueError):
        new_metadata["sample_gap_after"] = None

    return TabularDataset(
        name=f"{dataset.name}__sample_{keep_train_n}_{keep_test_n}",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cat_features=list(dataset.cat_features),
        num_features=list(dataset.num_features),
        task_type=dataset.task_type,
        shift_description=(
            f"{dataset.shift_description} | sampled first-{keep_train_n} train / "
            f"last-{keep_test_n} test"
        ),
        metadata=new_metadata,
    )


# Alternative training periods for a fixed test period
#
# These functions move a fixed-size training period through the time-sorted
# training history while holding the latest test period constant. The caller can
# compare distributional differences across periods. Calendar distance and
# distributional difference can diverge for seasonal datasets.


def default_gap_sizes(
    n_train_total: int,
    n_test_total: int,
    *,
    train_cap: int = 8000,
    train_floor: int = 2500,
    test_cap: int = 5000,
    test_floor: int = 1000,
) -> tuple[int, int]:
    """Pick per-dataset (n_train, n_test) window sizes for gap-widening.

    The defaults satisfy these constraints:
      * ``n_train = clip(n_train_total // 3, train_floor, train_cap)``: the
        ``// 3`` guarantees the earliest and latest windows do not overlap
        (2 * (n//3) < n), so the calendar extremes are genuinely distinct.
        ``train_floor=2500`` preserves the configured one-outer-fold policy.
        ``train_cap=8000`` keeps every window TabPFN-eligible (<= 10K).
      * ``n_test = clip(n_test_total // 2, test_floor, test_cap)``.

    Both results are clipped to the available row counts.
    """
    n_train = min(train_cap, n_train_total // 3)
    n_train = max(train_floor, n_train)
    n_train = min(n_train, n_train_total)
    n_test = min(test_cap, n_test_total // 2)
    n_test = max(test_floor, n_test)
    n_test = min(n_test, n_test_total)
    return int(n_train), int(n_test)


def _fmt_time(value: Any) -> str | None:
    """Return an ISO string for one time value or ``None`` when conversion fails."""
    try:
        return pd.Timestamp(value).isoformat()
    except (TypeError, ValueError, OverflowError):
        try:
            return str(value)
        except Exception:  # pragma: no cover - final fallback
            return None


@dataclass
class GapWindow:
    """One candidate training window before a result suffix is assigned."""

    tag: str  # e.g. "pos_0000". Unique per window position
    train_idx: np.ndarray  # positional indices into dataset.X_train
    start_position: int  # offset into the time-sorted train order
    start_frac: float  # start_position / max_start, in [0, 1]
    train_time_start: str | None
    train_time_end: str | None


@dataclass
class GapWindowPlan:
    """The fixed test window + all candidate train windows for one dataset."""

    dataset_name: str
    n_train: int
    n_test: int
    n_train_total: int
    n_test_total: int
    test_idx: np.ndarray  # positional indices into dataset.X_test (latest n_test)
    test_time_start: str | None
    test_time_end: str | None
    windows: list[GapWindow]


def _validate_time_metadata(dataset: TabularDataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_train, t_test) coerced arrays. Raise if metadata is missing."""
    meta = dataset.metadata or {}
    if not meta.get("time_col"):
        raise ValueError(
            f"Dataset {dataset.name!r} has no 'time_col' in metadata: "
            f"gap-widening only works on TabReD-style datasets."
        )
    if "time_train" not in meta or "time_test" not in meta:
        raise ValueError(
            f"Dataset {dataset.name!r} carries 'time_col' but no time arrays. "
            f"Re-load via load_dataset() to populate metadata['time_train']."
        )
    t_train = _coerce_time_array(meta["time_train"])
    t_test = _coerce_time_array(meta["time_test"])
    if len(t_train) != len(dataset.X_train):
        raise ValueError(
            f"metadata['time_train'] has {len(t_train)} entries but X_train "
            f"has {len(dataset.X_train)} rows."
        )
    if len(t_test) != len(dataset.X_test):
        raise ValueError(
            f"metadata['time_test'] has {len(t_test)} entries but X_test "
            f"has {len(dataset.X_test)} rows."
        )
    return t_train, t_test


def enumerate_gap_windows(
    dataset: TabularDataset,
    n_train: int,
    n_test: int,
    n_positions: int = 10,
) -> GapWindowPlan:
    """Build a fixed test period and a sequence of candidate training periods.

    The test period contains the latest ``n_test`` test rows and is fixed across
    all candidates. The train candidates are contiguous blocks of ``n_train``
    time-sorted train rows, started at ``n_positions`` evenly spaced offsets
    from 0 (earliest) to ``n_train_total - n_train`` (latest). No ``y`` is used.

    Returns a :class:`GapWindowPlan`. The caller measures each window's shift
    vs the fixed test window and picks newest / oldest / most-different.
    """
    if n_train <= 0 or n_test <= 0:
        raise ValueError(f"n_train and n_test must be > 0 (got {n_train}, {n_test})")
    if n_positions <= 0:
        raise ValueError(f"n_positions must be > 0 (got {n_positions})")

    t_train, t_test = _validate_time_metadata(dataset)
    n_train_total = len(t_train)
    n_test_total = len(t_test)

    n_train = min(n_train, n_train_total)
    n_test = min(n_test, n_test_total)

    train_order = np.argsort(t_train, kind="stable")
    test_order = np.argsort(t_test, kind="stable")

    # Fixed test = latest n_test rows.
    test_idx = test_order[n_test_total - n_test :]
    test_times = t_test[test_idx]

    max_start = n_train_total - n_train
    if max_start <= 0:
        starts = [0]
    else:
        starts = sorted(
            set(int(s) for s in np.linspace(0, max_start, n_positions).astype(int))
        )

    windows: list[GapWindow] = []
    for s in starts:
        w_order = train_order[s : s + n_train]
        w_times = t_train[w_order]
        windows.append(
            GapWindow(
                tag=f"pos_{s:06d}",
                train_idx=w_order,
                start_position=int(s),
                start_frac=float(s / max_start) if max_start > 0 else 0.0,
                train_time_start=_fmt_time(np.min(w_times)),
                train_time_end=_fmt_time(np.max(w_times)),
            )
        )

    return GapWindowPlan(
        dataset_name=dataset.name,
        n_train=int(n_train),
        n_test=int(n_test),
        n_train_total=int(n_train_total),
        n_test_total=int(n_test_total),
        test_idx=test_idx,
        test_time_start=_fmt_time(np.min(test_times)),
        test_time_end=_fmt_time(np.max(test_times)),
        windows=windows,
    )


def slice_gap_dataset(
    dataset: TabularDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    tag: str,
) -> TabularDataset:
    """Return a new dataset sliced to the given train/test positional indices.

    Pure function (never mutates the input). Slices ``X``/``y`` and the
    ``time_train``/``time_test`` side-arrays in lockstep and records the
    version ``tag`` under ``metadata['gap_version']``. ``y_test`` is sliced
    only if present (a LeakageGuard may already have nulled it).
    """
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    X_train = dataset.X_train.iloc[train_idx].reset_index(drop=True)
    X_test = dataset.X_test.iloc[test_idx].reset_index(drop=True)
    y_train = np.asarray(dataset.y_train)[train_idx]
    y_test = (
        np.asarray(dataset.y_test)[test_idx]
        if dataset.y_test is not None
        else None
    )

    new_meta = dict(dataset.metadata or {})
    if new_meta.get("time_train") is not None:
        new_meta["time_train"] = _coerce_time_array(new_meta["time_train"])[train_idx]
    if new_meta.get("time_test") is not None:
        new_meta["time_test"] = _coerce_time_array(new_meta["time_test"])[test_idx]
    new_meta["gap_version"] = tag

    return TabularDataset(
        name=f"{dataset.name}__{tag}",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cat_features=list(dataset.cat_features),
        num_features=list(dataset.num_features),
        task_type=dataset.task_type,
        shift_description=(
            f"{dataset.shift_description} | gap version {tag} "
            f"(train {len(train_idx)} / test {len(test_idx)})"
        ),
        metadata=new_meta,
    )

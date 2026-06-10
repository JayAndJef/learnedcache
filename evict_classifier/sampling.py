"""Vectorized one-pass sampling of eviction-time candidate rows.

Every candidate row corresponds to a (prior access a_i of page p, eviction event
E in the interval (t_i, next_access_of_p)) pair, with
``label = 1 iff (next_access - E) < horizon``. Instead of streaming all such
pairs through a Python loop, we compute the per-access interval boundaries with a
sort + a few ``searchsorted`` calls, then draw a bounded, class-balanced sample
entirely in numpy. This pays O(n_access) once in vectorized code (seconds), not a
per-record Python loop per epoch (minutes).

Right-censoring: eviction events later than ``last_access_ts - horizon`` cannot
have their reuse window observed, so they are dropped before sampling/splitting.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

TS_COL = "ts"
PAGE_KEY_COLS = ("dm", "dn", "in", "of")
DERIVED_FEATURE_COL = "time_since_last_access_at_eviction"


@dataclass
class WorkloadSample:
    """Materialised, in-memory training + holdout data for one workload."""

    x_train: np.ndarray  # (N, n_feat) raw float32 features
    y_train: np.ndarray  # (N,) float32 binary labels
    disc_sample: np.ndarray  # (M, n_feat) natural-ratio sample for discretizer fit
    x_eval: np.ndarray  # (E, n_feat) temporal-holdout features (may be empty)
    y_eval: np.ndarray  # (E,) holdout labels
    n_pos_seen: int  # total positive candidates available (train events)
    n_neg_seen: int
    class_weight: dict[int, float] | None


def _validate_fields(arr: np.ndarray, required: tuple[str, ...], label: str) -> None:
    names = arr.dtype.names or ()
    missing = [f for f in required if f not in names]
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")


def _interval_bounds(
    ev: np.ndarray,
    ts_s: np.ndarray,
    next_ts: np.ndarray,
    horizon: float,
    residency_cap: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-access eviction-index bounds against sorted events *ev*.

    Returns ``(lo, pos_lo, hi)`` such that, for access i, events in ``[lo, hi)``
    fall in ``(t_i, next_i)``; of those, ``[pos_lo, hi)`` are within ``horizon``
    of the next access (label 1) and ``[lo, pos_lo)`` are not (label 0).

    ``residency_cap`` truncates each window to ``(t_i, t_i + cap)``: the eviction
    log carries no page identity, so without a cap a page counts as a candidate
    long after it would realistically have been evicted from a full cache. The
    cap restricts training to events where the page is plausibly still resident
    (i.e. its in-list time-since-access is below the cache turnover time).
    """
    win_end = next_ts if residency_cap is None else np.minimum(next_ts, ts_s + residency_cap)
    lo = np.searchsorted(ev, ts_s, side="right")
    hi = np.searchsorted(ev, win_end, side="left")  # next_ts==inf -> len(ev)
    hi = np.maximum(hi, lo)
    pos_start = np.maximum(ts_s, next_ts - horizon)
    pos_lo = np.searchsorted(ev, pos_start, side="right")
    pos_lo = np.clip(pos_lo, lo, hi)
    return lo, pos_lo, hi


def _draw(counts: np.ndarray, starts: np.ndarray, n: int, rng: np.random.RandomState):
    """Draw *n* (record, event_index) samples ~ uniform over all events.

    Record i is chosen with probability proportional to ``counts[i]`` (its number
    of eligible events), then a uniform event offset in ``[0, counts[i])`` is
    picked, giving global event index ``starts[i] + offset``.
    """
    counts = counts.astype(np.int64)
    total = int(counts.sum())
    if n <= 0 or total == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    cdf = np.cumsum(counts)
    rec = np.searchsorted(cdf, rng.random(n) * total, side="right")
    rec = np.clip(rec, 0, len(counts) - 1)
    c = counts[rec]
    off = np.minimum((rng.random(n) * c).astype(np.int64), np.maximum(c - 1, 0))
    return rec, starts[rec] + off


def _rows(
    rec: np.ndarray, e_idx: np.ndarray, ev: np.ndarray, ts_s: np.ndarray, feats_s: np.ndarray
) -> np.ndarray:
    """Assemble feature rows: prior-access features + derived time-since-access."""
    out = np.empty((len(rec), feats_s.shape[1] + 1), dtype=np.float32)
    out[:, :-1] = feats_s[rec]
    out[:, -1] = (ev[e_idx] - ts_s[rec]).astype(np.float32)
    return out


@dataclass
class _TrialSample:
    x_train: np.ndarray
    y_train: np.ndarray
    disc_sample: np.ndarray
    x_eval: np.ndarray
    y_eval: np.ndarray
    n_pos: int
    n_neg: int


def sample_trial(
    access: np.ndarray,
    eviction: np.ndarray,
    discretize_cols: list[str],
    *,
    horizon: float,
    n_train: int,
    n_eval: int,
    disc_sample_size: int,
    balanced: bool,
    holdout_frac: float,
    rng: np.random.RandomState,
    residency_cap: float | None = None,
) -> _TrialSample:
    """Vectorized candidate sampling for one access/eviction trial.

    Feature rows come from the page's most recent prior access record, with
    ``TSA = event - prior_access_ts`` appended — exactly the state the purged
    kernel policies hold at eviction time (feature maps are only written at
    ``folio_accessed``; insertions and evictions mutate no state).
    """
    _validate_fields(access, (TS_COL, *PAGE_KEY_COLS, *discretize_cols), "access")
    _validate_fields(eviction, (TS_COL,), "eviction")

    ts = access[TS_COL].astype(np.float64)
    dm, dn, ino, of = (access[c] for c in PAGE_KEY_COLS)

    # Sort by page key, then ts (lexsort's last key is primary).
    order = np.lexsort((ts, of, ino, dn, dm))
    ts_s = ts[order]
    feats_s = np.column_stack(
        [access[c].astype(np.float32)[order] for c in discretize_cols]
    )

    def materialize(rec: np.ndarray, e_idx: np.ndarray, ev: np.ndarray) -> np.ndarray:
        return _rows(rec, e_idx, ev, ts_s, feats_s)

    # next-access timestamp within each page (inf at the last access of a page).
    same_page = (
        (dm[order][1:] == dm[order][:-1])
        & (dn[order][1:] == dn[order][:-1])
        & (ino[order][1:] == ino[order][:-1])
        & (of[order][1:] == of[order][:-1])
    )
    next_ts = np.full(len(ts_s), np.inf)
    next_ts[:-1][same_page] = ts_s[1:][same_page]

    # Right-censoring: keep only events whose reuse window is observable.
    ev_all = np.sort(eviction[TS_COL].astype(np.float64))
    ev_valid = ev_all[ev_all <= ts.max() - horizon]
    if holdout_frac > 0.0 and len(ev_valid) >= 2:
        split = np.quantile(ev_valid, 1.0 - holdout_frac)
        ev_train, ev_eval = ev_valid[ev_valid < split], ev_valid[ev_valid >= split]
    else:
        ev_train, ev_eval = ev_valid, ev_valid[:0]

    # ── Training sample ──
    lo, pos_lo, hi = _interval_bounds(ev_train, ts_s, next_ts, horizon, residency_cap)
    n_pos_arr, n_neg_arr = hi - pos_lo, pos_lo - lo
    n_pos, n_neg = int(n_pos_arr.sum()), int(n_neg_arr.sum())

    if balanced:
        n_p = n_train // 2
        pr, pe = _draw(n_pos_arr, pos_lo, n_p, rng)
        nr, ne = _draw(n_neg_arr, lo, n_train - n_p, rng)
        x_train = np.concatenate(
            [materialize(pr, pe, ev_train), materialize(nr, ne, ev_train)]
        )
        y_train = np.concatenate(
            [np.ones(len(pr), np.float32), np.zeros(len(nr), np.float32)]
        )
    else:
        rec, e_idx = _draw(n_pos_arr + n_neg_arr, lo, n_train, rng)
        x_train = materialize(rec, e_idx, ev_train)
        y_train = (e_idx >= pos_lo[rec]).astype(np.float32)

    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    # Natural-ratio sample for fitting the discretizer.
    drec, de = _draw(n_pos_arr + n_neg_arr, lo, disc_sample_size, rng)
    disc_sample = materialize(drec, de, ev_train)

    # ── Holdout sample (natural ratio) ──
    if len(ev_eval) > 0:
        elo, epos_lo, ehi = _interval_bounds(
            ev_eval, ts_s, next_ts, horizon, residency_cap
        )
        erec, ee = _draw(ehi - elo, elo, n_eval, rng)
        x_eval = materialize(erec, ee, ev_eval)
        y_eval = (ee >= epos_lo[erec]).astype(np.float32)
    else:
        n_feat = len(discretize_cols) + 1
        x_eval = np.empty((0, n_feat), np.float32)
        y_eval = np.empty(0, np.float32)

    return _TrialSample(x_train, y_train, disc_sample, x_eval, y_eval, n_pos, n_neg)


def collect_workload_sample(
    pairs: Iterator[tuple[int, np.ndarray, np.ndarray]],
    discretize_cols: list[str],
    *,
    horizon: float,
    target_rows: int,
    balanced: bool = True,
    disc_sample_size: int = 200_000,
    eval_rows: int = 300_000,
    holdout_frac: float = 0.2,
    residency_cap: float | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> WorkloadSample:
    """Sample a bounded training/holdout set across a workload's iters."""
    rng = np.random.RandomState(random_state)
    parts: list[_TrialSample] = []
    n_pos_seen = n_neg_seen = 0

    for trial_id, access, eviction in pairs:
        ts = sample_trial(
            access, eviction, discretize_cols,
            horizon=horizon, n_train=target_rows, n_eval=eval_rows,
            disc_sample_size=disc_sample_size, balanced=balanced,
            holdout_frac=holdout_frac, rng=rng, residency_cap=residency_cap,
        )
        parts.append(ts)
        n_pos_seen += ts.n_pos
        n_neg_seen += ts.n_neg
        if verbose:
            print(
                f"  trial {trial_id}: {len(access):,} accesses -> "
                f"{len(ts.x_train):,} train / {len(ts.x_eval):,} holdout rows "
                f"({ts.n_pos:,} pos / {ts.n_neg:,} neg available)"
            )

    def _cat(attr: str) -> np.ndarray:
        return np.concatenate([getattr(p, attr) for p in parts], axis=0)

    x_train, y_train = _cat("x_train"), _cat("y_train")
    if len(x_train) > target_rows:  # multiple iters overshoot the budget
        keep = rng.choice(len(x_train), size=target_rows, replace=False)
        x_train, y_train = x_train[keep], y_train[keep]

    total = n_pos_seen + n_neg_seen
    class_weight = None
    if total and n_pos_seen and n_neg_seen:
        class_weight = {0: total / (2.0 * n_neg_seen), 1: total / (2.0 * n_pos_seen)}

    if verbose and total:
        print(
            f"  collected {len(x_train):,} train rows; "
            f"true positive rate {n_pos_seen / total:.4f}"
        )

    return WorkloadSample(
        x_train=x_train,
        y_train=y_train,
        disc_sample=_cat("disc_sample"),
        x_eval=_cat("x_eval"),
        y_eval=_cat("y_eval"),
        n_pos_seen=n_pos_seen,
        n_neg_seen=n_neg_seen,
        class_weight=class_weight,
    )

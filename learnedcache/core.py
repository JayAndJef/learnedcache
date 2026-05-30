"""
Core logic for learned cache training and export.
"""

from __future__ import annotations

import bisect
import json
import pickle
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from learnedcache.binary_loading import (
    build_pairs_from_binary,
    discover_workloads_and_iters,
)
from learnedcache.helpers import save_evaluation_outputs
from learnedcache.loading import read_access_eviction_trial_pairs, transform_logs_to_csvs
from learnedcache.models import build_model
from learnedcache.preprocess import (
    fit_discretizer_from_sample,
    one_hot_encode_features,
    train_and_transform_discretizer,
    transform_discretizer_batch,
)

PAGE_KEY_COLS = ["dm", "dn", "in", "of"]
TS_COL = "ts"

DERIVED_FEATURE_COL = "time_since_last_access_at_eviction"
TARGET_COL = "time_until_next_reuse_from_eviction"
NO_REUSE_LABEL_OFFSET = 1.0

def run_transform_logs(log_pattern: str, verbose: bool = True) -> None:
    """Transform raw log files to CSV format."""
    if verbose:
        print(f"Transforming logs matching: {log_pattern}")
    transform_logs_to_csvs(log_pattern)
    if verbose:
        print("Done.")

def _validate_required_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _require_numeric(df: pd.DataFrame, columns: list[str], df_name: str) -> pd.DataFrame:
    """Coerce required columns to numeric in-place; fail on invalid values. Returns same DataFrame.

    Preserves existing numeric dtypes (uint32, uint64, etc.) without upcast.
    Non-numeric columns are downcast to the smallest unsigned integer type
    when all values are non-negative.
    """
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="unsigned")
        if df[col].isna().any():
            raise ValueError(f"{df_name} has non-numeric values in required column '{col}'.")
    return df


def _validate_array_fields(
    arr: np.ndarray, required_fields: list[str], label: str
) -> None:
    """Validate that a structured array has all required fields."""
    names = arr.dtype.names or ()
    missing = [f for f in required_fields if f not in names]
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")


def _access_to_sorted_dict(
    access: pd.DataFrame | np.ndarray,
    trial_id: int,
    required_cols: list[str],
    discretize_cols: list[str],
    label: str,
) -> dict[str, np.ndarray]:
    """Normalize access data into a dict of 1D arrays, sorted by page key then timestamp.

    Works for both CSV DataFrames and binary structured arrays.

    Sorts by (PAGE_KEY_COLS, TS_COL) using ``np.lexsort`` so that all
    accesses for the same page key are **contiguous** in memory.  This is
    required for the diff-based group boundary detection in
    ``_build_eviction_supervised_df`` -- a simple ``argsort`` by ts alone
    would interleave different pages that happen to share nearby timestamps.

    Returns a dict with keys: ``TS_COL``, each PAGE_KEY_COL, and each
    discretize_col.
    """
    if isinstance(access, pd.DataFrame):
        _validate_required_columns(access, required_cols, label)
        access = _require_numeric(access, [TS_COL, *discretize_cols], label)

        # lexsort: last array is the primary sort key.
        # We want page-key-group then timestamp within each group.
        sort_idx = np.lexsort(
            [access[TS_COL].to_numpy(dtype=np.float64)]
            + [access[col].to_numpy() for col in reversed(PAGE_KEY_COLS)]
        )

        result: dict[str, np.ndarray] = {
            TS_COL: access[TS_COL].to_numpy(dtype=np.float64)[sort_idx],
            **{col: access[col].to_numpy()[sort_idx] for col in PAGE_KEY_COLS},
            **{
                col: access[col].to_numpy(dtype=np.float64)[sort_idx]
                for col in discretize_cols
            },
        }
    else:
        # Binary structured array
        _validate_array_fields(access, required_cols, label)

        sort_idx = np.lexsort(
            [access[TS_COL].astype(np.float64)]
            + [access[col] for col in reversed(PAGE_KEY_COLS)]
        )

        result = {
            TS_COL: access[TS_COL].astype(np.float64)[sort_idx],
            **{col: access[col][sort_idx] for col in PAGE_KEY_COLS},
            **{
                col: access[col].astype(np.float64)[sort_idx]
                for col in discretize_cols
            },
        }

    return result


def _build_trial_supervised_df(
    trial_id: int,
    access: pd.DataFrame | np.ndarray,
    eviction: pd.DataFrame | np.ndarray,
    discretize_cols: list[str],
) -> pd.DataFrame:
    """Build supervised DataFrame for a single access+eviction trial.

    Returns a DataFrame with columns ``[trial_id, eviction_ts,
    DERIVED_FEATURE_COL, TARGET_COL] + PAGE_KEY_COLS + discretize_cols``.
    The ``TARGET_COL`` column still contains ``NaN`` for pages that are
    never reused within the trial — the caller must fill these with a
    global ``no_reuse_label`` after all trials have been processed.

    This is the per-trial building block used by both the legacy
    full-materialisation path (``_build_eviction_supervised_df``) and
    the new streaming path.
    """
    EVICTION_TS_COL = "eviction_ts"
    access_required = sorted(set(PAGE_KEY_COLS + [TS_COL] + discretize_cols))
    eviction_required = [TS_COL]

    # --- Normalize eviction ---
    if isinstance(eviction, pd.DataFrame):
        _validate_required_columns(eviction, eviction_required, f"eviction trial {trial_id}")
        eviction = _require_numeric(eviction, [TS_COL], f"eviction trial {trial_id}")
        eviction_ts_arr = eviction[TS_COL].to_numpy(dtype=np.float64)
    else:
        _validate_array_fields(eviction, eviction_required, f"eviction trial {trial_id}")
        eviction_ts_arr = eviction[TS_COL].astype(np.float64)
    eviction_sort = np.argsort(eviction_ts_arr)
    eviction_ts_arr = eviction_ts_arr[eviction_sort]
    n_evictions = len(eviction_ts_arr)

    # --- Normalize access into sorted dict ---
    sorted_access = _access_to_sorted_dict(
        access, trial_id, access_required, discretize_cols,
        f"access trial {trial_id}",
    )

    # --- Per-page group boundaries via numpy diff ---
    n_access = len(sorted_access[TS_COL])
    group_boundary = np.zeros(n_access, dtype=bool)
    for col in PAGE_KEY_COLS:
        col_vals = sorted_access[col]
        group_boundary[1:] |= (col_vals[1:] != col_vals[:-1])
    group_starts = np.concatenate([[0], np.where(group_boundary)[0]])
    group_ends = np.concatenate([group_starts[1:], [n_access]])

    BATCH_SIZE = 50  # pages per batch; small batches keep DataFrames small
    trial_results: list[pd.DataFrame] = []
    batch_arrays: list[dict[str, np.ndarray]] = []

    for start, end in zip(group_starts, group_ends):
        page_access_ts = sorted_access[TS_COL][start:end]
        n_page_accesses = len(page_access_ts)

        # side='right' -> pos is first index with ts > eviction_ts.
        #   prior access idx = pos-1  (ts <= eviction_ts, exact match OK)
        #   next access idx  = pos    (ts >  eviction_ts, exact match excluded)
        pos = np.searchsorted(page_access_ts, eviction_ts_arr, side="right")

        has_prior = pos > 0
        prior_idx = np.clip(pos - 1, 0, n_page_accesses - 1)
        has_future = pos < n_page_accesses
        future_idx = np.clip(pos, 0, n_page_accesses - 1)

        page_data: dict[str, np.ndarray] = {
            EVICTION_TS_COL: eviction_ts_arr,
            DERIVED_FEATURE_COL: eviction_ts_arr
            - np.where(has_prior, page_access_ts[prior_idx], np.nan),
            TARGET_COL: np.where(
                has_future, page_access_ts[future_idx], np.nan
            )
            - eviction_ts_arr,
            "trial_id": np.full(n_evictions, trial_id),
        }

        for col in PAGE_KEY_COLS:
            page_data[col] = np.full(n_evictions, sorted_access[col][start])

        for col in discretize_cols:
            col_vals = sorted_access[col][start:end]
            page_data[col] = np.where(has_prior, col_vals[prior_idx], np.nan)

        batch_arrays.append(page_data)

        if len(batch_arrays) >= BATCH_SIZE:
            trial_results.append(
                pd.DataFrame(
                    {
                        col: np.concatenate([b[col] for b in batch_arrays])
                        for col in batch_arrays[0]
                    }
                ).dropna(subset=[DERIVED_FEATURE_COL])
            )
            batch_arrays.clear()

    if batch_arrays:
        trial_results.append(
            pd.DataFrame(
                {
                    col: np.concatenate([b[col] for b in batch_arrays])
                    for col in batch_arrays[0]
                }
            ).dropna(subset=[DERIVED_FEATURE_COL])
        )
        batch_arrays.clear()

    if trial_results:
        return pd.concat(trial_results, ignore_index=True)
    # No supervised rows for this trial (e.g. zero evictions or no prior accesses).
    return pd.DataFrame(columns=[
        "trial_id", EVICTION_TS_COL, DERIVED_FEATURE_COL, TARGET_COL,
    ] + PAGE_KEY_COLS + discretize_cols)


def _build_eviction_supervised_df(
    access_eviction_pairs: Iterable[tuple[int, pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]],
    discretize_cols: list[str],
) -> pd.DataFrame:
    """Build the FULL supervised DataFrame by concatenating all trials.

    Legacy path — kept for backward compatibility with existing callers
    and tests.  Prefer the streaming path (``_run_train_ranker_streaming``)
    for new code to avoid materialising the entire dataset in memory.
    """
    if not discretize_cols:
        raise ValueError("discretize_cols cannot be empty.")

    EVICTION_TS_COL = "eviction_ts"
    all_dfs: list[pd.DataFrame] = []

    for trial_id, access, eviction in access_eviction_pairs:
        trial_df = _build_trial_supervised_df(
            trial_id, access, eviction, discretize_cols,
        )
        if len(trial_df) > 0:
            all_dfs.append(trial_df)

    if not all_dfs:
        raise ValueError("No supervised rows were generated from access+eviction streams.")

    supervised_df = pd.concat(all_dfs, ignore_index=True)

    max_finite = supervised_df[TARGET_COL].max()
    no_reuse_label = (
        max_finite if pd.notnull(max_finite) else 0.0
    ) + NO_REUSE_LABEL_OFFSET
    supervised_df[TARGET_COL] = supervised_df[TARGET_COL].fillna(no_reuse_label)

    return supervised_df[
        ["trial_id", EVICTION_TS_COL, DERIVED_FEATURE_COL, TARGET_COL]
        + PAGE_KEY_COLS
        + discretize_cols
    ]


class StreamingSupervisedGenerator:
    """Yields ``(x_diff, y_diff)`` pair batches for one trial's eviction events.

    Processes eviction events in configurable chunks and access pages in
    small groups so that peak memory is bounded by a single page-group ×
    eviction-chunk working set (typically a few MB).

    Parameters
    ----------
    access_arr:
        Structured numpy array (memmap or in-memory) of access records.
    eviction_arr:
        Structured numpy array (or DataFrame) of eviction records.
    discretizer:
        A fitted :class:`~sklearn.preprocessing.KBinsDiscretizer`.
    n_bins_list:
        Bin counts per discretized feature.
    discretize_cols:
        Feature column / field names.
    chunk_events:
        Number of eviction events to process in each forward-window advance.
    lookahead_us:
        Forward-lookahead window in microseconds.
    pairs_per_event:
        Number of pairs to sample per eviction event.
    max_pairs_total:
        Optional cap on total pairs.
    batch_size:
        Mini-batch size for yielded ``(x_diff, y_diff)`` tuples.
    page_batch_size:
        Number of pages to process together within one eviction chunk.
    random_state:
        Seed for the per-chunk pair-sampling RNG.
    no_reuse_label:
        Surrogate target value for pages whose next access lies beyond
        the lookahead window.
    """

    def __init__(
        self,
        access_arr: np.ndarray,
        eviction_arr: np.ndarray,
        discretizer: Any,
        n_bins_list: list[int],
        discretize_cols: list[str],
        no_reuse_label: float,
        *,
        chunk_events: int = 50,
        lookahead_us: int | None = None,  # auto-scaled if None
        pairs_per_event: int = 512,
        max_pairs_total: int | None = None,
        batch_size: int = 256,
        page_batch_size: int = 50,
        random_state: int = 42,
    ) -> None:
        if chunk_events <= 0:
            raise ValueError("chunk_events must be > 0.")

        self._discretizer = discretizer
        self._n_bins_list = n_bins_list
        self._discretize_cols = list(discretize_cols)
        self._no_reuse_label = no_reuse_label
        self._chunk_events = chunk_events
        self._lookahead_us = lookahead_us  # may be overridden below
        self._pairs_per_event = pairs_per_event
        self._max_pairs_total = max_pairs_total
        self._batch_size = batch_size
        self._page_batch_size = page_batch_size
        self._random_state = random_state

        # --- Access: store the ORIGINAL array + ts-sort index ---
        # We never copy or materialise the access data — all access goes
        # through the sort index into the original (often memmap) array.
        if isinstance(access_arr, pd.DataFrame):
            _validate_required_columns(
                access_arr, [TS_COL] + self._discretize_cols + PAGE_KEY_COLS,
                "access",
            )
            access_arr = _require_numeric(
                access_arr, [TS_COL] + self._discretize_cols, "access",
            )
            self._access_arr = access_arr
            self._access_is_df = True
        else:
            _validate_array_fields(
                access_arr,
                [TS_COL] + self._discretize_cols + PAGE_KEY_COLS,
                "access",
            )
            self._access_arr = access_arr
            self._access_is_df = False

        # ts-sort index — the only per-record allocation (~ N * 8 bytes).
        raw_ts = (
            self._access_arr[TS_COL].to_numpy(dtype=np.float64)
            if self._access_is_df
            else self._access_arr[TS_COL].astype(np.float64)
        )
        self._ts_order = np.argsort(raw_ts)
        del raw_ts
        self._n_access = len(self._ts_order)

        # --- Evictions: sort by ts (eviction arrays are tiny, 1–2 MB) ---
        if isinstance(eviction_arr, pd.DataFrame):
            _validate_required_columns(eviction_arr, [TS_COL], "eviction")
            eviction_arr = _require_numeric(eviction_arr, [TS_COL], "eviction")
            self._eviction_ts = np.sort(
                eviction_arr[TS_COL].to_numpy(dtype=np.float64)
            )
        else:
            _validate_array_fields(eviction_arr, [TS_COL], "eviction")
            self._eviction_ts = np.sort(eviction_arr[TS_COL].astype(np.float64))
        self._n_evictions = len(self._eviction_ts)

        # --- Page state: page_key -> (ts_list, feat_rows) ---
        # Lists grow during _advance_access_window; converted to numpy
        # inside the per-page-group processing loop.
        # Auto-scale lookahead to ~30% of the eviction time span (unit-agnostic).
        # The eviction array is fully loaded and tiny (1–2 MB), so computing
        # its full min/max range is essentially free.
        if lookahead_us is None or lookahead_us <= 0:
            ev_ts_for_range = self._eviction_ts
            if len(ev_ts_for_range) >= 2:
                ev_range = float(ev_ts_for_range[-1] - ev_ts_for_range[0])
            else:
                ev_range = 1e6
            lookahead_us = max(int(ev_range * 0.30), 1_000_000)
        self._lookahead_us = lookahead_us

        self._page_state: dict[tuple, tuple[list, list]] = defaultdict(lambda: ([], []))
        self._access_ptr = 0

    # ------------------------------------------------------------------
    # Public generator entry-point
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return self._generate()

    def _generate(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(x_batch, y_batch)`` tuples consumed by ``model.fit()``."""
        ev_ptr = 0
        rng = np.random.RandomState(self._random_state)
        n_events = self._n_evictions

        while ev_ptr < n_events:
            chunk_end = min(ev_ptr + self._chunk_events, n_events)
            ev_chunk = self._eviction_ts[ev_ptr:chunk_end]

            # --- 1. Advance access window ---
            max_lookahead = ev_chunk[-1] + self._lookahead_us
            self._advance_access_window(max_lookahead)

            # --- 2. Process pages in groups ---
            page_keys = list(self._page_state.keys())
            pbs = self._page_batch_size

            for pg_start in range(0, len(page_keys), pbs):
                pg_end = min(pg_start + pbs, len(page_keys))
                group_keys = page_keys[pg_start:pg_end]

                # Build candidate rows for this page-group × eviction-chunk
                candidates, targets, event_ids = self._build_page_group_rows(
                    group_keys, ev_chunk,
                )
                if candidates is None:
                    continue

                # Discretize
                discretized = transform_discretizer_batch(candidates, self._discretizer)
                del candidates

                # One-hot encode
                onehot = one_hot_encode_features(discretized, self._n_bins_list)
                del discretized

                # Sample pairwise diffs within events
                try:
                    x_diff, y_diff, _stats = _sample_pairwise_diffs_by_event(
                        x_full=onehot,
                        y_full=targets,
                        event_ids=event_ids,
                        pairs_per_event=self._pairs_per_event,
                        random_state=rng.randint(0, 2**31 - 1),
                        max_pairs_total=self._max_pairs_total,
                    )
                except ValueError:
                    # All pairs were ties (e.g., lookahead too short) — skip
                    del onehot, targets, event_ids
                    continue
                del onehot, targets, event_ids

                # Yield mini-batches
                perm = rng.permutation(len(x_diff))
                bs = self._batch_size
                for i in range(0, len(x_diff), bs):
                    idx = perm[i : i + bs]
                    yield x_diff[idx], y_diff[idx]

                del x_diff, y_diff

            # --- 3. Prune stale page state ---
            self._prune_page_state(ev_chunk[0])

            ev_ptr = chunk_end

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_access_window(self, max_ts: float) -> None:
        """Read access records up to *max_ts* into ``_page_state``.

        Uses the ts-sort index to iterate in chronological order without
        ever materialising a sorted copy of the access array.
        """
        ptr = self._access_ptr
        ts_order = self._ts_order
        n = self._n_access
        arr = self._access_arr
        disc_cols = self._discretize_cols

        if self._access_is_df:
            while ptr < n:
                idx = ts_order[ptr]
                ts_val = float(arr[TS_COL].iloc[idx])
                if ts_val > max_ts:
                    break
                key = (
                    int(arr["dm"].iloc[idx]),
                    int(arr["dn"].iloc[idx]),
                    int(arr["in"].iloc[idx]),
                    int(arr["of"].iloc[idx]),
                )
                feats = [float(arr[c].iloc[idx]) for c in disc_cols]
                ts_list, feat_list = self._page_state[key]
                ts_list.append(ts_val)
                feat_list.append(feats)
                ptr += 1
        else:
            while ptr < n:
                idx = ts_order[ptr]
                ts_val = float(arr["ts"][idx])
                if ts_val > max_ts:
                    break
                key = (
                    int(arr["dm"][idx]),
                    int(arr["dn"][idx]),
                    int(arr["in"][idx]),
                    int(arr["of"][idx]),
                )
                feats = [float(arr[c][idx]) for c in disc_cols]
                ts_list, feat_list = self._page_state[key]
                ts_list.append(ts_val)
                feat_list.append(feats)
                ptr += 1

        self._access_ptr = ptr

    def _build_page_group_rows(
        self,
        group_keys: list[tuple],
        ev_chunk: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Build candidate rows for *group_keys* against *ev_chunk*.

        Returns ``(candidates, targets, event_ids)`` — each is ``None``
        if no rows survived filtering.
        """
        E = len(ev_chunk)
        n_disc = len(self._discretize_cols)
        n_features = 1 + n_disc  # derived + discretize_cols
        ev_ts = np.asarray(ev_chunk, dtype=np.float64)

        feat_chunks: list[np.ndarray] = []   # each (K, n_features) float32
        targ_chunks: list[np.ndarray] = []   # each (K,) float32
        evt_chunks: list[np.ndarray] = []    # each (K,) int32

        for key in group_keys:
            ts_list, feat_list = self._page_state[key]
            if not ts_list:
                continue

            page_ts = np.asarray(ts_list, dtype=np.float64)
            page_feats = np.asarray(feat_list, dtype=np.float32)
            A = len(page_ts)

            pos = np.searchsorted(page_ts, ev_ts, side="right")
            has_prior = pos > 0
            if not has_prior.any():
                continue

            prior_idx = np.clip(pos - 1, 0, A - 1)
            has_future = pos < A

            # target: time_until_next_reuse
            future_ts = np.where(has_future, page_ts[np.clip(pos, 0, A - 1)], np.nan)
            target = np.where(
                has_future, future_ts - ev_ts, self._no_reuse_label,
            ).astype(np.float32)

            # derived feature: time_since_last_access_at_eviction
            derived = (ev_ts - page_ts[prior_idx]).astype(np.float32)

            keep = has_prior
            k = int(keep.sum())
            rows = np.zeros((k, n_features), dtype=np.float32)
            rows[:, 0] = derived[keep]
            rows[:, 1:] = page_feats[prior_idx[keep]]

            feat_chunks.append(rows)
            targ_chunks.append(target[keep])
            evt_chunks.append(np.where(keep)[0].astype(np.int32))

        if not feat_chunks:
            return None, None, None

        candidates = np.concatenate(feat_chunks, axis=0)
        targets = np.concatenate(targ_chunks, axis=0)
        event_ids = np.concatenate(evt_chunks, axis=0)
        return candidates, targets, event_ids

    def _prune_page_state(self, cutoff_ts: float) -> None:
        """Discard access records with ts < *cutoff_ts*.

        For each page we always keep the single access record immediately
        preceding *cutoff_ts* (if one exists) because it is the most
        recent prior access for the next eviction chunk.
        """
        for key in list(self._page_state.keys()):
            ts_list, feat_list = self._page_state[key]
            if not ts_list:
                del self._page_state[key]
                continue

            idx = bisect.bisect_left(ts_list, cutoff_ts)
            keep_from = max(0, idx - 1)
            if keep_from >= len(ts_list):
                del self._page_state[key]
                continue

            if keep_from > 0:
                self._page_state[key] = (ts_list[keep_from:], feat_list[keep_from:])


def _sample_pairwise_diffs_by_event(
    x_full: np.ndarray,
    y_full: np.ndarray,
    event_ids: np.ndarray,
    pairs_per_event: int,
    random_state: int,
    max_pairs_total: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if pairs_per_event <= 0:
        raise ValueError("pairs_per_event must be > 0.")

    # Defensive dtype guards: enforce compact types regardless of caller.
    x_full = np.asarray(x_full, dtype=np.float32)
    y_full = np.asarray(y_full, dtype=np.float32)

    rng = np.random.RandomState(random_state)

    # --- Presorted fast-path: skip O(N log N) argsort when already sorted ---
    if np.all(event_ids[:-1] <= event_ids[1:]):
        x_sorted = x_full
        y_sorted = y_full
        sorted_event_ids = event_ids
    else:
        sort_idx = np.argsort(event_ids)
        x_sorted = x_full[sort_idx]
        y_sorted = y_full[sort_idx]
        sorted_event_ids = event_ids[sort_idx]

    # --- Fast group-boundary detection via diff ---
    change = np.concatenate([[True], sorted_event_ids[1:] != sorted_event_ids[:-1]])
    first_indices = np.where(change)[0]
    n_unique = len(first_indices)
    counts = np.diff(first_indices, append=len(sorted_event_ids))

    eligible = counts >= 2
    first_indices = first_indices[eligible]
    counts = counts[eligible]

    if len(counts) == 0:
        raise ValueError("No events with >= 2 samples found.")

    total_events = len(counts)
    total_requested = total_events * pairs_per_event

    # Per-group metadata, expanded to one entry per requested pair.
    offsets = np.repeat(first_indices, pairs_per_event).astype(np.intp)
    ns = np.repeat(counts, pairs_per_event).astype(np.intp)

    # idx_a: uniform random within each group's local [0, count) range.
    # idx_b: guaranteed != idx_a in one pass via offset ∈ [1, ns).
    idx_a_local = (rng.rand(total_requested) * ns).astype(np.intp)
    offset = (1 + rng.rand(total_requested) * (ns - 1)).astype(np.intp)
    idx_b_local = (idx_a_local + offset) % ns

    sorted_pos_a = idx_a_local + offsets
    sorted_pos_b = idx_b_local + offsets
    del idx_a_local, idx_b_local, offsets, ns, offset

    # Drop ties (y_a == y_b) which provide no learning signal.
    y_a, y_b = y_sorted[sorted_pos_a], y_sorted[sorted_pos_b]
    keep = y_a != y_b
    ties_dropped = len(keep) - int(keep.sum())
    sorted_pos_a, sorted_pos_b = sorted_pos_a[keep], sorted_pos_b[keep]
    y_a, y_b = y_a[keep], y_b[keep]

    if len(sorted_pos_a) == 0:
        raise ValueError("No pairwise samples generated after tie-drop.")

    # Optional post-hoc cap on total pair count.
    if max_pairs_total is not None and 0 < max_pairs_total < len(sorted_pos_a):
        chosen = rng.choice(len(sorted_pos_a), size=max_pairs_total, replace=False).astype(np.intp)
        sorted_pos_a, sorted_pos_b = sorted_pos_a[chosen], sorted_pos_b[chosen]
        y_a, y_b = y_a[chosen], y_b[chosen]

    # One-shot subtraction (x_sorted is float32 after defensive cast above,
    # so x_sorted[...] - x_sorted[...] stays in float32 -- no float64 temps).
    x_out = x_sorted[sorted_pos_a] - x_sorted[sorted_pos_b]
    y_out = (y_a < y_b).astype(np.float32)

    stats = {
        "events_total": n_unique,
        "events_with_pairs": total_events,
        "sampled_before_tie_drop": total_requested,
        "ties_dropped": ties_dropped,
        "pairs_after_tie_drop": len(x_out),
    }
    return x_out, y_out, stats


def _make_pair_generator(
    trial_iter: Iterable[tuple[int, np.ndarray, np.ndarray]],
    discretizer: Any,
    n_bins_list: list[int],
    discretize_cols: list[str],
    no_reuse_label: float,
    *,
    chunk_events: int = 50,
    lookahead_us: int | None = None,  # auto-scaled if None
    pairs_per_event: int = 512,
    max_pairs_total: int | None = None,
    batch_size: int = 256,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(x_batch, y_batch)`` tuples for training or validation.

    Wraps infinitely so that ``model.fit()`` with ``steps_per_epoch``
    never exhausts the generator mid-epoch.  Each pass through
    *trial_iter* creates fresh ``StreamingSupervisedGenerator``
    instances (which re-read from memory-mapped data).  Explicit
    ``del`` and ``gc.collect`` calls ensure old generators are freed
    before new ones are created.
    """
    import gc

    rng = np.random.RandomState(random_state)
    while True:
        for trial_id, access, eviction in trial_iter:
            stream = StreamingSupervisedGenerator(
                access_arr=access,
                eviction_arr=eviction,
                discretizer=discretizer,
                n_bins_list=n_bins_list,
                discretize_cols=discretize_cols,
                no_reuse_label=no_reuse_label,
                chunk_events=chunk_events,
                lookahead_us=lookahead_us,
                pairs_per_event=pairs_per_event,
                max_pairs_total=max_pairs_total,
                batch_size=batch_size,
                random_state=rng.randint(0, 2**31 - 1),
            )
            for x_batch, y_batch in stream:
                yield x_batch, y_batch
            del stream
            gc.collect()  # eagerly reclaim sort-index + page-state memory


def _run_train_ranker_streaming(
    pairs_fn: Any,  # Callable[[], Iterator]
    n_trials: int,
    source_description: str,
    output_dir: Path,
    discretize_cols: list[str],
    n_bins: int,
    max_epochs: int,
    batch_size: int,
    pairs_per_event: int,
    max_pairs_total: int | None,
    random_state: int,
    verbose: bool,
) -> dict[str, Any]:
    """Streaming three-phase pipeline: collect stats → fit discretizer → train.

    Phase 1: One pass through all trials to collect a discretizer
    subsample, track the global target maximum, and count eviction
    events (for ``steps_per_epoch`` estimation).

    Phase 2: Fit ``KBinsDiscretizer`` on the subsample.

    Phase 3: Build a Keras model and train via Python generators that
    stream eviction events in small chunks with bounded forward
    lookahead.  The full training dataset is never simultaneously in RAM.
    """
    model_feature_cols = [*discretize_cols, DERIVED_FEATURE_COL]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_trial_id: int | None = (n_trials - 1) if n_trials >= 2 else None
    has_holdout = test_trial_id is not None

    # ── Phase 1: collect discretizer subsample + global target max ──
    MAX_SAMPLE = 200_000
    discretizer_sample_rows: list[np.ndarray] = []
    n_collected = 0
    global_target_max = -np.inf
    n_train_events = 0
    n_test_events = 0
    n_train_rows = 0
    n_test_rows = 0

    if verbose:
        print("Phase 1: collecting discretizer subsample and counting events...")

    # We sample discretize_cols directly from the raw access arrays
    # (no expensive eviction join) and approximate the derived-feature
    # distribution from inter-access gaps.  This avoids building the
    # full per-trial supervised DataFrame which is the dominant memory
    # consumer.
    rng = np.random.RandomState(random_state)

    for trial_id, access, eviction in pairs_fn():
        is_train = (trial_id != test_trial_id)

        # Count eviction events
        if isinstance(eviction, pd.DataFrame):
            n_events = int(eviction[TS_COL].nunique())
        else:
            n_events = len(np.unique(eviction[TS_COL]))
        if is_train:
            n_train_events += n_events
        else:
            n_test_events += n_events

        if not is_train or n_collected >= MAX_SAMPLE:
            continue

        # Sample raw access features for discretizer fitting.
        if isinstance(access, pd.DataFrame):
            n_access = len(access)
            n_take = min(MAX_SAMPLE - n_collected, n_access)
            indices = rng.choice(n_access, size=n_take, replace=False)
            sample_feats = access[discretize_cols].iloc[indices].to_numpy(dtype=np.float64)
            # Approximate derived feature from inter-access ts gaps
            ts_all = access[TS_COL].to_numpy(dtype=np.float64)
            if len(ts_all) >= 2:
                ts_sorted = np.sort(ts_all)
                ts_diffs = np.diff(ts_sorted)
                n_derived = min(n_take, len(ts_diffs))
                derived_sample = rng.choice(ts_diffs, size=n_derived, replace=True).astype(np.float64)
            else:
                derived_sample = np.zeros(n_take, dtype=np.float64)
        else:
            n_access = len(access)
            n_take = min(MAX_SAMPLE - n_collected, n_access)
            indices = rng.choice(n_access, size=n_take, replace=False)
            sample_feats = np.column_stack(
                [access[col].astype(np.float64)[indices] for col in discretize_cols]
            )
            ts_all = access[TS_COL].astype(np.float64)
            if len(ts_all) >= 2:
                ts_sorted = np.sort(ts_all)
                ts_diffs = np.diff(ts_sorted)
                n_derived = min(n_take, len(ts_diffs))
                derived_sample = rng.choice(ts_diffs, size=n_derived, replace=True).astype(np.float64)
            else:
                derived_sample = np.zeros(n_take, dtype=np.float64)

        # Combine: [discretize_cols | DERIVED_FEATURE_COL]
        combined = np.column_stack([sample_feats, derived_sample.reshape(-1, 1)])
        discretizer_sample_rows.append(combined)
        n_collected += n_take

        # Track global target max from access ts range (upper bound on reuse).
        # The maximum possible reuse time ≤ max_ts − min_ts for the trial.
        ts_min = float(ts_all.min()) if len(ts_all) > 0 else 0.0
        ts_max = float(ts_all.max()) if len(ts_all) > 0 else 0.0
        global_target_max = max(global_target_max, ts_max - ts_min)
        del sample_feats, derived_sample, combined, ts_all
        if n_access >= 2:
            del ts_diffs

    if n_collected == 0:
        raise ValueError("No training feature rows for discretizer fitting.")

    no_reuse_label = (
        global_target_max if np.isfinite(global_target_max) else 0.0
    ) + NO_REUSE_LABEL_OFFSET

    if verbose:
        print(f"  Collected {n_collected:,} rows for discretizer sample")
        print(f"  Global target max: {global_target_max:.3f}  →  no_reuse_label: {no_reuse_label:.3f}")
        print(f"  Train events: {n_train_events:,}  |  Test events: {n_test_events:,}")
        print(f"  Train rows:   {n_train_rows:,}  |  Test rows:   {n_test_rows:,}")

    # ── Phase 2: fit discretizer ──
    if verbose:
        print("Phase 2: fitting discretizer...")

    disc_sample = np.vstack(discretizer_sample_rows)
    del discretizer_sample_rows
    discretizer = fit_discretizer_from_sample(
        disc_sample, n_bins=n_bins, strategy="quantile", random_state=random_state,
    )
    del disc_sample

    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(len(model_feature_cols))]
    n_encoded_features = sum(n_bins_list)

    if verbose:
        print(f"  Bins per feature: {n_bins_list}")
        print(f"  Total one-hot features: {n_encoded_features}")

    # ── Phase 3: train ──
    if verbose:
        print("Phase 3: training with streaming generator...")

    # Estimate steps_per_epoch and validation_steps from event counts.
    # These are upper bounds — tie-dropping may reduce actual pairs, but
    # the infinite generator wrapper handles the slight overestimation.
    train_max = max_pairs_total or (n_train_events * pairs_per_event)
    train_pairs_est = min(n_train_events * pairs_per_event, train_max)
    steps_per_epoch = max(1, int(np.ceil(train_pairs_est / batch_size)))

    val_steps = None
    if has_holdout:
        val_max = max_pairs_total or (n_test_events * pairs_per_event)
        val_pairs_est = min(n_test_events * pairs_per_event, val_max)
        val_steps = max(1, int(np.ceil(val_pairs_est / batch_size)))

    if verbose:
        print(f"  steps_per_epoch: {steps_per_epoch}, validation_steps: {val_steps}")
        print(f"  chunk_events={50}, lookahead_us=auto-scaled")

    # Build training / validation generators.
    def _train_trials():
        for tup in pairs_fn():
            if tup[0] != test_trial_id:
                yield tup

    train_gen = _make_pair_generator(
        _train_trials(),
        discretizer=discretizer,
        n_bins_list=n_bins_list,
        discretize_cols=discretize_cols,
        no_reuse_label=no_reuse_label,
        chunk_events=50,
        lookahead_us=None,  # auto-scaled per workload
        pairs_per_event=pairs_per_event,
        max_pairs_total=max_pairs_total,
        batch_size=batch_size,
        random_state=random_state,
    )

    val_gen = None
    if has_holdout:
        def _val_trials():
            for tup in pairs_fn():
                if tup[0] == test_trial_id:
                    yield tup

        val_gen = _make_pair_generator(
            _val_trials(),
            discretizer=discretizer,
            n_bins_list=n_bins_list,
            discretize_cols=discretize_cols,
            no_reuse_label=no_reuse_label,
            chunk_events=50,
            lookahead_us=None,  # auto-scaled per workload
            pairs_per_event=pairs_per_event,
            max_pairs_total=max_pairs_total,
            batch_size=batch_size,
            random_state=random_state + 1,
        )

    # Build model.
    model = build_model(n_encoded_features)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    if verbose:
        model.summary()

    # Train.
    if has_holdout:
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1 if verbose else 0,
        )
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=[early_stop],
            verbose=1 if verbose else 0,
        )
    else:
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs,
            verbose=1 if verbose else 0,
        )

    # ── Post-training evaluation (holdout case) ──
    pairwise_accuracy = 0.0
    y_pred_prob = None
    x_diff_test = np.empty((0, n_encoded_features), dtype=np.float32)
    y_test_pairs = np.empty(0, dtype=np.float32)
    test_pair_stats: dict[str, int] = {
        "events_total": 0, "events_with_pairs": 0,
        "sampled_before_tie_drop": 0, "ties_dropped": 0,
        "pairs_after_tie_drop": 0,
    }
    n_test_pairs_val = 0

    if has_holdout and test_trial_id is not None:
        if verbose:
            print("Evaluating on holdout trial...")
        x_test_list: list[np.ndarray] = []
        y_test_list: list[np.ndarray] = []
        test_events_total = 0
        rng = np.random.RandomState(random_state + 1)

        for trial_id, access, eviction in pairs_fn():
            if trial_id != test_trial_id:
                continue

            # Stream through the holdout trial once for evaluation.
            stream = StreamingSupervisedGenerator(
                access_arr=access,
                eviction_arr=eviction,
                discretizer=discretizer,
                n_bins_list=n_bins_list,
                discretize_cols=discretize_cols,
                no_reuse_label=no_reuse_label,
                chunk_events=50,
                lookahead_us=None,  # auto-scaled per workload
                pairs_per_event=pairs_per_event,
                max_pairs_total=max_pairs_total,
                batch_size=batch_size,
                random_state=rng.randint(0, 2**31 - 1),
            )
            # We need the full x_diff and y_diff for evaluation, not mini-batches.
            # So accumulate across chunks.
            for x_diff_chunk, y_diff_chunk in stream:
                x_test_list.append(x_diff_chunk)
                y_test_list.append(y_diff_chunk)
            break  # only one holdout trial

        if x_test_list:
            x_diff_test = np.concatenate(x_test_list, axis=0)
            y_test_pairs = np.concatenate(y_test_list, axis=0)
            n_test_pairs_val = len(y_test_pairs)
            del x_test_list, y_test_list

            y_pred_prob = model.predict(x_diff_test, verbose=0).ravel()
            pairwise_accuracy = float(accuracy_score(
                y_test_pairs.astype(np.int32),
                (y_pred_prob >= 0.5).astype(np.int32),
            ))
            if verbose:
                print(f"Pairwise Test Accuracy: {pairwise_accuracy:.4f}")
                print(f"Trained for {len(history.history.get('loss', []))} epochs")

    # ── Save artifacts ──
    model_path = output_dir / "model.keras"
    model.save(model_path)
    if verbose:
        print(f"Saved model → {model_path}")

    discretizer_path = output_dir / "discretizer.pkl"
    with discretizer_path.open("wb") as f:
        pickle.dump(discretizer, f)
    if verbose:
        print(f"Saved discretizer → {discretizer_path}")

    weights = model.get_layer("ranking_weight").get_weights()[0].ravel()
    train_pair_stats_dict: dict[str, int] = {
        "events_total": n_train_events,
        "events_with_pairs": n_train_events,
        "sampled_before_tie_drop": n_train_events * pairs_per_event,
        "ties_dropped": 0,
        "pairs_after_tie_drop": steps_per_epoch * batch_size,
    }

    save_evaluation_outputs(
        history=history,
        weights=weights,
        x_eval_full=x_diff_test,
        column_names=model_feature_cols,
        n_bins_list=n_bins_list,
        output_dir=output_dir,
        y_true=y_test_pairs,
        y_pred_prob=y_pred_prob,
        pairwise_accuracy=pairwise_accuracy,
        access_pattern=source_description or "",
        n_rows=n_train_rows + n_test_rows,
        n_train_rows=n_train_rows,
        n_test_rows=n_test_rows,
        epochs_trained=len(history.history.get("loss", [])),
        n_train_pairs=steps_per_epoch * batch_size * len(history.history.get("loss", [])),
        n_test_pairs=n_test_pairs_val,
        train_pair_stats=train_pair_stats_dict,
        test_pair_stats=test_pair_stats,
    )

    if verbose:
        print("Done.")

    return {
        "model": model,
        "discretizer": discretizer,
        "n_bins_list": n_bins_list,
        "discretize_cols": model_feature_cols,
        "pairwise_accuracy": pairwise_accuracy,
        "history": history,
        "n_train_pairs": steps_per_epoch * batch_size * len(history.history.get("loss", [])),
        "n_test_pairs": n_test_pairs_val,
        "train_pair_stats": train_pair_stats_dict,
        "test_pair_stats": test_pair_stats,
    }


def run_train_ranker(
    access_pattern: str | None = None,
    eviction_pattern: str | None = None,
    pairs: Iterable[tuple[int, pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]] | None = None,
    pairs_fn: Any = None,  # Callable[[], Iterator] — streaming path
    n_trials: int | None = None,  # needed for streaming path
    source_description: str | None = None,
    output_dir: Path = Path("."),
    discretize_cols: list[str] = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"],
    n_bins: int = 10,
    max_epochs: int = 50,
    batch_size: int = 256,
    pairs_per_event: int = 512,
    max_pairs_total: int | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a linear Bradley-Terry pairwise-diff ranker on eviction-time events.

    Supports three input paths:

    * **Streaming** (``pairs_fn`` or ``access_pattern`` + ``eviction_pattern``):
      Uses per-chunk eviction-event batching with bounded forward lookahead.
      The full training dataset is never simultaneously in RAM.

    * **Legacy** (``pairs``): loads all data into a single DataFrame before
      training.  Kept for backward compatibility.
    """
    # ── Resolve input path ──
    use_streaming: bool
    if pairs_fn is not None:
        # Explicit streaming path (binary data from run_train_from_binary, etc.)
        use_streaming = True
        if n_trials is None:
            n_trials = sum(1 for _ in pairs_fn())
        if source_description is None:
            source_description = "binary data (streaming)"
    elif access_pattern is not None and eviction_pattern is not None:
        # CSV pattern path → construct pairs_fn internally
        use_streaming = True
        n_trials = count_access_eviction_trial_pairs(access_pattern, eviction_pattern)
        pairs_fn = lambda: read_access_eviction_trial_pairs(  # type: ignore[assignment]
            access_pattern, eviction_pattern
        )
        source_description = f"access={access_pattern}, eviction={eviction_pattern}"
    elif pairs is not None:
        use_streaming = False
        if source_description is None:
            source_description = "binary data (legacy)"
    else:
        raise ValueError(
            "Must provide one of: pairs_fn, access_pattern+eviction_pattern, or pairs"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_streaming:
        return _run_train_ranker_streaming(
            pairs_fn=pairs_fn,
            n_trials=n_trials,
            source_description=source_description or "",
            output_dir=output_dir,
            discretize_cols=discretize_cols,
            n_bins=n_bins,
            max_epochs=max_epochs,
            batch_size=batch_size,
            pairs_per_event=pairs_per_event,
            max_pairs_total=max_pairs_total,
            random_state=random_state,
            verbose=verbose,
        )

    # ── Legacy materialisation path (unchanged) ──
    if verbose:
        print("Building eviction-time supervised dataset...")
    df = _build_eviction_supervised_df(pairs, discretize_cols)

    model_feature_cols = [*discretize_cols, DERIVED_FEATURE_COL]

    if verbose:
        print(f"Built {len(df)} supervised rows")
        print(
            f"Target stats ({TARGET_COL}): min={df[TARGET_COL].min():.3f}, "
            f"median={df[TARGET_COL].median():.3f}, max={df[TARGET_COL].max():.3f}"
        )
        print(f"Model features: {model_feature_cols}")

    trial_ids = np.sort(df["trial_id"].unique())
    has_holdout = len(trial_ids) >= 2

    if has_holdout:
        test_trial_id = trial_ids[-1]
        train_trials = trial_ids[:-1]
        train_mask = df["trial_id"].isin(train_trials).to_numpy()
        test_mask = (df["trial_id"] == test_trial_id).to_numpy()
    else:
        train_mask = np.ones(len(df), dtype=bool)
        test_mask = np.zeros(len(df), dtype=bool)
        if verbose:
            print("Single trial: no holdout set. Training on all data.")

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    if len(train_idx) == 0 or (has_holdout and len(test_idx) == 0):
        raise ValueError("Invalid trial split produced empty train or test set.")

    if verbose:
        print(f"Split by trial_id: train={train_trials.tolist()} | test={[int(test_trial_id)]}")
        print(f"  Train rows: {len(train_idx)}  |  Test rows: {len(test_idx)}")

    train_features_df = df.iloc[train_idx][model_feature_cols].reset_index(drop=True)
    test_features_df = df.iloc[test_idx][model_feature_cols].reset_index(drop=True)

    if verbose:
        print("Discretizing features (fit on train only)...")
    train_discretized, discretizer = train_and_transform_discretizer(
        train_features_df,
        n_bins=n_bins,
        strategy="quantile",
    )
    del train_features_df  # free input DataFrame (keeps ~864 MB alive)
    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(len(model_feature_cols))]
    if verbose:
        print(f"Bins per discretized feature: {n_bins_list}")

    _transformed_test = discretizer.transform(test_features_df)
    test_discretized = _transformed_test.data.astype(np.int8).reshape(test_features_df.shape)
    del _transformed_test, test_features_df

    x_train_full = one_hot_encode_features(train_discretized, n_bins_list)
    x_test_full = one_hot_encode_features(test_discretized, n_bins_list)
    y_train_raw = df.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float32)
    y_test_raw = df.iloc[test_idx][TARGET_COL].to_numpy(dtype=np.float32)

    # NumPy-based event encoding avoids pd.factorize(list(zip(...))) which
    # converts millions of rows to Python tuples — a major CPU/memory sink.
    # np.unique on a structured array returns inverse indices in C.
    _train_tid = df.iloc[train_idx]["trial_id"].to_numpy(dtype=np.int32)
    _train_ets = df.iloc[train_idx]["eviction_ts"].to_numpy(dtype=np.float64)
    _train_evt = np.empty(len(_train_tid), dtype=[("tid", np.int32), ("ets", np.float64)])
    _train_evt["tid"] = _train_tid
    _train_evt["ets"] = _train_ets
    _, train_events = np.unique(_train_evt, return_inverse=True)
    del _train_tid, _train_ets, _train_evt

    x_diff_train, y_train_pairs, train_pair_stats = _sample_pairwise_diffs_by_event(
        x_full=x_train_full,
        y_full=y_train_raw,
        event_ids=train_events,
        pairs_per_event=pairs_per_event,
        random_state=random_state,
        max_pairs_total=max_pairs_total,
    )

    if has_holdout:
        _test_tid = df.iloc[test_idx]["trial_id"].to_numpy(dtype=np.int32)
        _test_ets = df.iloc[test_idx]["eviction_ts"].to_numpy(dtype=np.float64)
        _test_evt = np.empty(len(_test_tid), dtype=[("tid", np.int32), ("ets", np.float64)])
        _test_evt["tid"] = _test_tid
        _test_evt["ets"] = _test_ets
        _, test_events = np.unique(_test_evt, return_inverse=True)
        del _test_tid, _test_ets, _test_evt
        x_diff_test, y_test_pairs, test_pair_stats = _sample_pairwise_diffs_by_event(
            x_full=x_test_full,
            y_full=y_test_raw,
            event_ids=test_events,
            pairs_per_event=pairs_per_event,
            random_state=random_state + 1,
            max_pairs_total=max_pairs_total,
        )
    else:
        x_diff_test = np.empty((0, x_train_full.shape[1]), dtype=np.float32)
        y_test_pairs = np.empty(0, dtype=np.float32)
        test_pair_stats = {"events_total": 0, "events_with_pairs": 0,
                           "sampled_before_tie_drop": 0, "ties_dropped": 0,
                           "pairs_after_tie_drop": 0}

    n_encoded_features = x_train_full.shape[1]
    if verbose:
        print(f"Feature matrix shape: train={x_train_full.shape}, test={x_test_full.shape}")
        print(f"  One-hot features: {n_encoded_features} (from bins {n_bins_list})")
        print(
            "Pairwise sampling train stats: "
            f"{train_pair_stats}, label_balance={float(np.mean(y_train_pairs)):.3f}"
        )
        if has_holdout:
            print(
                "Pairwise sampling test stats: "
                f"{test_pair_stats}, label_balance={float(np.mean(y_test_pairs)):.3f}"
            )

    if verbose:
        print("Building pairwise-diff model...")
    model = build_model(n_encoded_features)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    if verbose:
        model.summary()

    if verbose:
        print(f"Training (max_epochs={max_epochs}, batch_size={batch_size})...")

    if has_holdout:
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1 if verbose else 0,
        )
        history = model.fit(
            x_diff_train,
            y_train_pairs,
            epochs=max_epochs,
            batch_size=batch_size,
            validation_data=(x_diff_test, y_test_pairs),
            callbacks=[early_stop],
            verbose=1 if verbose else 0,
        )
    else:
        history = model.fit(
            x_diff_train,
            y_train_pairs,
            epochs=max_epochs,
            batch_size=batch_size,
            verbose=1 if verbose else 0,
        )

    if has_holdout:
        if verbose:
            print("Evaluating...")
        y_pred_prob = model.predict(x_diff_test, verbose=0).ravel()
        pairwise_accuracy = float(accuracy_score(
            y_test_pairs.astype(np.int32),
            (y_pred_prob >= 0.5).astype(np.int32),
        ))
        if verbose:
            print(f"Pairwise Test Accuracy: {pairwise_accuracy:.4f}")
            print(f"Trained for {len(history.history.get('loss', []))} epochs")
    else:
        pairwise_accuracy = 0.0
        y_pred_prob = None
        if verbose:
            print(f"Trained for {len(history.history.get('loss', []))} epochs")
            print("No holdout set: skipping test evaluation and visualizations.")

    model_path = output_dir / "model.keras"
    model.save(model_path)
    if verbose:
        print(f"Saved model → {model_path}")

    discretizer_path = output_dir / "discretizer.pkl"
    with discretizer_path.open("wb") as f:
        pickle.dump(discretizer, f)
    if verbose:
        print(f"Saved discretizer → {discretizer_path}")

    weights = model.get_layer("ranking_weight").get_weights()[0].ravel()
    save_evaluation_outputs(
        history=history,
        weights=weights,
        x_eval_full=x_diff_test,
        column_names=model_feature_cols,
        n_bins_list=n_bins_list,
        output_dir=output_dir,
        y_true=y_test_pairs,
        y_pred_prob=y_pred_prob,
        pairwise_accuracy=pairwise_accuracy,
        access_pattern=source_description or "",
        n_rows=len(df),
        n_train_rows=len(train_idx),
        n_test_rows=len(test_idx),
        epochs_trained=len(history.history.get("loss", [])),
        n_train_pairs=len(y_train_pairs),
        n_test_pairs=len(y_test_pairs),
        train_pair_stats=train_pair_stats,
        test_pair_stats=test_pair_stats,
    )

    if verbose:
        print("Done.")

    return {
        "model": model,
        "discretizer": discretizer,
        "n_bins_list": n_bins_list,
        "discretize_cols": model_feature_cols,
        "pairwise_accuracy": pairwise_accuracy,
        "history": history,
        "n_train_pairs": len(y_train_pairs),
        "n_test_pairs": len(y_test_pairs),
        "train_pair_stats": train_pair_stats,
        "test_pair_stats": test_pair_stats,
    }

def run_train_from_binary(
    data_dir: str | Path,
    output_dir: Path,
    workloads: list[str] | None = None,
    discretize_cols: list[str] | None = None,
    n_bins: int = 10,
    max_epochs: int = 50,
    batch_size: int = 256,
    pairs_per_event: int = 512,
    max_pairs_total: int | None = None,
    pair_random_state: int = 42,
    weight_scale: int = 10000,
    verbose: bool = False,
) -> dict[str, dict[str, Any]]:
    """Train one model per workload from binary cache trace logs.

    Discovers workloads under data_dir, builds trial pairs from iter dirs,
    trains a pairwise ranker per workload, and exports BPF weights.

    Returns:
        Dict mapping workload_name -> train_result dict.
    """
    if discretize_cols is None:
        discretize_cols = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"]

    export_feature_names = [*discretize_cols, DERIVED_FEATURE_COL]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workload_map = discover_workloads_and_iters(data_dir, workloads)

    results: dict[str, dict[str, Any]] = {}
    for wl_name, iter_dirs in workload_map.items():
        wl_output_dir = output_dir / wl_name
        print(f"\n{'=' * 80}")
        print(f"Workload: {wl_name} ({len(iter_dirs)} iter(s))")
        print(f"Output: {wl_output_dir}")
        print(f"{'=' * 80}")

        source = f"binary data from {data_dir}/{wl_name}"

        train_result = run_train_ranker(
            pairs_fn=lambda: build_pairs_from_binary(iter_dirs),
            n_trials=len(iter_dirs),
            source_description=source,
            output_dir=wl_output_dir,
            discretize_cols=discretize_cols,
            n_bins=n_bins,
            max_epochs=max_epochs,
            batch_size=batch_size,
            pairs_per_event=pairs_per_event,
            max_pairs_total=max_pairs_total,
            random_state=pair_random_state,
            verbose=verbose,
        )

        export_file = wl_output_dir / "model_weights.json"
        run_export_model(
            model_dir=wl_output_dir,
            output_file=export_file,
            weight_scale=weight_scale,
            feature_names=export_feature_names,
            verbose=verbose,
            model=train_result["model"],
            discretizer=train_result["discretizer"],
        )

        # Free heavy in-memory objects — model, history, and discretizer are
        # saved to disk and no longer needed. Only lightweight metadata
        # remains in the result dict (accuracy, pair counts, stats).
        del train_result["model"]
        del train_result["history"]
        del train_result["discretizer"]

        results[wl_name] = train_result

        if verbose:
            n_iters = len(iter_dirs)
            acc = train_result.get("pairwise_accuracy", 0.0)
            print(f"\nWorkload '{wl_name}' complete ({n_iters} iter(s), acc={acc:.4f})\n")

    return results


def run_export_model(
    model_dir: Path,
    output_file: Path,
    weight_scale: int = 10000,
    feature_names: list[str] = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie", DERIVED_FEATURE_COL],
    verbose: bool = True,
    model: Any = None,
    discretizer: Any = None,
) -> dict[str, Any]:
    """Export trained model to BPF-compatible JSON format.

    Args:
        model: Optional in-memory Keras model. If None, loads from model_dir/model.keras.
        discretizer: Optional in-memory KBinsDiscretizer. If None, loads from model_dir/discretizer.pkl.
    """
    if model is None or discretizer is None:
        if verbose:
            print(f"Loading model from {model_dir}...")
        discretizer_path = model_dir / "discretizer.pkl"
        if not discretizer_path.exists():
            raise FileNotFoundError(f"{discretizer_path} not found")
        with discretizer_path.open("rb") as f:
            discretizer = pickle.load(f)
        model_path = model_dir / "model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found")
        model = keras.models.load_model(model_path)
    else:
        if verbose:
            print("Using provided in-memory model and discretizer.")

    weights = model.get_layer("ranking_weight").get_weights()[0].ravel()

    if len(discretizer.bin_edges_) != len(feature_names):
        raise ValueError(
            "feature_names length does not match trained discretizer feature count: "
            f"{len(feature_names)} vs {len(discretizer.bin_edges_)}"
        )

    n_features = len(feature_names)
    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(n_features)]

    if verbose:
        print(f"Features: {feature_names}")
        print(f"Bins per feature: {n_bins_list}")
        print(f"Total one-hot features: {sum(n_bins_list)}")
        print(f"Weight vector shape: {weights.shape}")

    model_data: dict[str, Any] = {
        "feature_names": feature_names,
        "n_features": n_features,
        "weight_scale": weight_scale,
        "features": [],
    }

    weight_idx = 0
    for feat_idx, feat_name in enumerate(feature_names):
        n_bins_feat = n_bins_list[feat_idx]
        all_edges = discretizer.bin_edges_[feat_idx]
        interior_edges = all_edges[1:-1].tolist()

        feat_weights_float = weights[weight_idx : weight_idx + n_bins_feat]
        feat_weights_int = (feat_weights_float * weight_scale).astype(np.int64).tolist()

        feature_data = {
            "index": feat_idx,
            "name": feat_name,
            "n_bins": n_bins_feat,
            "bin_edges": [int(x) for x in interior_edges],
            "weights_float": feat_weights_float.tolist(),
            "weights_int": feat_weights_int,
        }
        model_data["features"].append(feature_data)
        weight_idx += n_bins_feat

        if verbose:
            print(
                f"  {feat_name}: {n_bins_feat} bins, "
                f"weights [{feat_weights_float.min():.4f}, {feat_weights_float.max():.4f}]"
            )

    with Path(output_file).open("w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)

    if verbose:
        print(f"\nExported model to {output_file}")
        print(f"Weight scale factor: {weight_scale}")
        print("Ready to load into BPF maps!")

    return model_data

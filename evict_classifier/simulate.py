"""Trace-driven page-cache simulator.

Replays the access stream from the binary logs and measures hit rate at a fixed
capacity for four policies:

* ``fifo``    -- classic FIFO replacement.
* ``lru``     -- classic LRU replacement.
* ``belady``  -- Belady's MIN (offline-optimal upper bound, bypass allowed).
* ``protect`` -- the trained binary classifier under the *kernel's actual*
  selection semantics (``__bpf_cache_ext_list_sample``): folios are taken from
  the front of the list in ``batch x sample`` chunks, the min-score folio of
  each ``sample``-sized group is evicted, survivors rotate to the tail.

Simulation model: the access log is the request stream; a miss inserts the
page; insertions beyond capacity trigger eviction. Differences from the live
system are deliberate and documented in KNOWN_ISSUES.md / IMPROVEMENTS.md:
readahead-only insertions are absent (every simulated page is inserted by an
access), and protect-policy features use the page's last *access-log row*
(training semantics) rather than re-simulating the in-kernel EMA state machine.

Hit rates are reported for the full stream and for the final 20% of accesses
("tail"), since per-workload models saw the earlier portion during training.
"""

from __future__ import annotations

import glob
import heapq
import json
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .loading import read_binary_access_log
from .sampling import PAGE_KEY_COLS, TS_COL

RAW_FEATURE_COLS = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"]

# Must mirror cache_ext_fifo_ml_protect.bpf.c.
PROTECT_BASE = 1 << 40
TSA_CAP = 1 << 36
DEFAULT_BATCH = 32  # request_nr_folios_to_evict (SWAP_CLUSTER_MAX)
DEFAULT_SAMPLE = 5  # sampling_options.sample_size

TAIL_FRAC = 0.2


@dataclass
class Stream:
    """A time-ordered access stream with page ids and raw model features."""

    ts: np.ndarray  # int64 ns, ascending
    page: np.ndarray  # int64 page ids in [0, n_pages)
    feats: np.ndarray  # (n, 8) float64 raw feature columns (RAW_FEATURE_COLS)
    n_pages: int

    def __len__(self) -> int:
        return len(self.page)


def load_stream(iter_dir: str | Path) -> Stream:
    """Load one iter directory's access log as a time-ordered stream."""
    files = sorted(glob.glob(str(Path(iter_dir) / "mglru_lc_access_*.bin")))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected 1 access log in {iter_dir}, got {len(files)}")
    access = read_binary_access_log(files[0])

    ts = access[TS_COL].astype(np.int64)
    order = np.argsort(ts, kind="stable")

    keys = np.empty(len(access), dtype=[(c, access.dtype[c]) for c in PAGE_KEY_COLS])
    for c in PAGE_KEY_COLS:
        keys[c] = access[c]
    _, inverse = np.unique(keys, return_inverse=True)

    feats = np.column_stack(
        [access[c].astype(np.float64)[order] for c in RAW_FEATURE_COLS]
    )
    page = inverse[order].astype(np.int64)
    return Stream(ts=ts[order], page=page, feats=feats, n_pages=int(page.max()) + 1)


def compute_next_use(page: np.ndarray) -> np.ndarray:
    """Stream index of each access's next same-page access (len(page) = never)."""
    n = len(page)
    idx = np.lexsort((np.arange(n), page))
    nxt = np.full(n, n, dtype=np.int64)
    same = page[idx][1:] == page[idx][:-1]
    nxt[idx[:-1][same]] = idx[1:][same]
    return nxt


def _result(name: str, n: int, hits: int, hits_tail: int, capacity: int) -> dict:
    tail_n = n - int(n * (1 - TAIL_FRAC))
    return {
        "policy": name,
        "capacity_pages": capacity,
        "accesses": n,
        "hits": hits,
        "hit_rate": hits / n if n else 0.0,
        "tail_accesses": tail_n,
        "tail_hits": hits_tail,
        "tail_hit_rate": hits_tail / tail_n if tail_n else 0.0,
    }


def run_fifo(stream: Stream, capacity: int) -> dict:
    n = len(stream)
    tail_start = int(n * (1 - TAIL_FRAC))
    resident = bytearray(stream.n_pages)
    dq: deque[int] = deque()
    hits = hits_tail = 0

    for i, p in enumerate(stream.page.tolist()):
        if resident[p]:
            hits += 1
            if i >= tail_start:
                hits_tail += 1
            continue
        if len(dq) >= capacity:
            resident[dq.popleft()] = 0
        resident[p] = 1
        dq.append(p)
    return _result("fifo", n, hits, hits_tail, capacity)


def run_lru(stream: Stream, capacity: int) -> dict:
    n = len(stream)
    tail_start = int(n * (1 - TAIL_FRAC))
    od: OrderedDict[int, None] = OrderedDict()
    hits = hits_tail = 0

    for i, p in enumerate(stream.page.tolist()):
        if p in od:
            od.move_to_end(p)
            hits += 1
            if i >= tail_start:
                hits_tail += 1
            continue
        if len(od) >= capacity:
            od.popitem(last=False)
        od[p] = None
    return _result("lru", n, hits, hits_tail, capacity)


def run_belady(stream: Stream, capacity: int) -> dict:
    """Belady's MIN with bypass: evict the resident page (possibly the one just
    inserted) whose next use is farthest in the future."""
    n = len(stream)
    tail_start = int(n * (1 - TAIL_FRAC))
    next_use = compute_next_use(stream.page)

    # Heap entries are single packed ints for speed/memory:
    # key = (n - next_use) << PAGE_BITS | page  -> min-heap pops farthest next use.
    page_bits = max(int(stream.n_pages - 1).bit_length(), 1)
    resident = bytearray(stream.n_pages)
    cur_next = np.zeros(stream.n_pages, dtype=np.int64)
    heap: list[int] = []
    size = hits = hits_tail = 0
    page_mask = (1 << page_bits) - 1

    pages = stream.page.tolist()
    nxt = next_use.tolist()
    for i in range(n):
        p = pages[i]
        nu = nxt[i]
        if resident[p]:
            hits += 1
            if i >= tail_start:
                hits_tail += 1
            cur_next[p] = nu
            heapq.heappush(heap, ((n - nu) << page_bits) | p)
            continue

        resident[p] = 1
        size += 1
        cur_next[p] = nu
        heapq.heappush(heap, ((n - nu) << page_bits) | p)
        while size > capacity:
            key = heapq.heappop(heap)
            q = key & page_mask
            q_nu = n - (key >> page_bits)
            if resident[q] and cur_next[q] == q_nu:  # not a stale entry
                resident[q] = 0
                size -= 1
    return _result("belady", n, hits, hits_tail, capacity)


def load_model(model_file: str | Path) -> dict[str, Any]:
    """Load exported classifier JSON into arrays for integer scoring."""
    data = json.loads(Path(model_file).read_text())
    return {
        "edges": [np.asarray(f["bin_edges"], dtype=np.float64) for f in data["features"]],
        "weights": [np.asarray(f["weights_int"], dtype=np.int64) for f in data["features"]],
        "bias": int(data["bias_int"]),
        "threshold": int(data["threshold_int"]),
    }


def _protect_scores(
    model: dict[str, Any], feats_rows: np.ndarray, tsa: np.ndarray
) -> np.ndarray:
    """Kernel-faithful integer scores: protect band + recency tiebreak."""
    k = len(feats_rows)
    logit = np.full(k, model["bias"], dtype=np.int64)
    for f in range(len(model["edges"])):
        vals = tsa.astype(np.float64) if f == len(RAW_FEATURE_COLS) else feats_rows[:, f]
        bins = np.searchsorted(model["edges"][f], vals, side="right")
        logit += model["weights"][f][bins]
    protected = logit > model["threshold"]
    return np.where(protected, PROTECT_BASE, 0) - np.minimum(tsa, TSA_CAP)


def run_protect(
    stream: Stream,
    capacity: int,
    model: dict[str, Any],
    *,
    batch: int = DEFAULT_BATCH,
    sample: int = DEFAULT_SAMPLE,
) -> dict:
    n = len(stream)
    tail_start = int(n * (1 - TAIL_FRAC))
    resident = bytearray(stream.n_pages)
    last_row = [0] * stream.n_pages  # page -> latest access row (valid when resident)
    dq: deque[int] = deque()
    size = hits = hits_tail = 0

    pages = stream.page.tolist()
    ts = stream.ts
    feats = stream.feats

    for i in range(n):
        p = pages[i]
        last_row[p] = i
        if resident[p]:
            hits += 1
            if i >= tail_start:
                hits_tail += 1
            continue

        resident[p] = 1
        dq.append(p)
        size += 1
        while size > capacity:
            now = int(ts[i])
            k = min(batch * sample, size)
            cand = [dq.popleft() for _ in range(k)]
            rows = np.fromiter((last_row[q] for q in cand), dtype=np.int64, count=k)
            tsa = now - ts[rows]
            scores = _protect_scores(model, feats[rows], tsa)

            # Min of each consecutive `sample`-sized group is evicted.
            n_full = k // sample
            evict: list[int] = []
            if n_full:
                grouped = scores[: n_full * sample].reshape(n_full, sample)
                evict.extend(
                    (np.argmin(grouped, axis=1) + np.arange(n_full) * sample).tolist()
                )
            if k % sample:
                evict.append(n_full * sample + int(np.argmin(scores[n_full * sample :])))

            evict_set = set(evict)
            for j, q in enumerate(cand):
                if j in evict_set:
                    resident[q] = 0
                    size -= 1
                else:
                    dq.append(q)
    return _result("protect", n, hits, hits_tail, capacity)


def simulate_workload(
    iter_dir: str | Path,
    model_file: str | Path,
    capacities: list[int],
    *,
    policies: list[str] = ["fifo", "lru", "belady", "protect"],
    verbose: bool = True,
) -> list[dict]:
    """Run all requested policies x capacities for one workload iter."""
    stream = load_stream(iter_dir)
    model = load_model(model_file) if "protect" in policies else None
    unique_pages = stream.n_pages
    compulsory_ceiling = 1.0 - unique_pages / len(stream)
    if verbose:
        print(
            f"  stream: {len(stream):,} accesses, {unique_pages:,} pages "
            f"(compulsory-miss hit-rate ceiling {compulsory_ceiling:.3f})"
        )

    runners = {
        "fifo": run_fifo,
        "lru": run_lru,
        "belady": run_belady,
        "protect": lambda s, c: run_protect(s, c, model),
    }
    results = []
    for cap in capacities:
        for name in policies:
            res = runners[name](stream, cap)
            res["compulsory_ceiling"] = compulsory_ceiling
            results.append(res)
            if verbose:
                print(
                    f"  cap={cap:>9,} {name:>8}: hit_rate={res['hit_rate']:.4f} "
                    f"tail={res['tail_hit_rate']:.4f}"
                )
    return results

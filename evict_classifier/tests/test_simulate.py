"""Tests for the trace-driven simulator (textbook reference strings)."""

import numpy as np

from evict_classifier.simulate import (
    RAW_FEATURE_COLS,
    Stream,
    compute_next_use,
    run_belady,
    run_fifo,
    run_lru,
    run_protect,
)

# Silberschatz's classic reference string: with 3 frames, FIFO faults 15,
# LRU faults 12, OPT (Belady) faults 9.
TEXTBOOK = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]


def _stream(pages: list[int]) -> Stream:
    page = np.asarray(pages, dtype=np.int64)
    n = len(page)
    return Stream(
        ts=np.arange(n, dtype=np.int64),
        page=page,
        feats=np.zeros((n, len(RAW_FEATURE_COLS)), dtype=np.float64),
        n_pages=int(page.max()) + 1,
    )


def _trivial_model(threshold: int) -> dict:
    # one bin per feature, zero weights: logit == bias == 0 for every page.
    return {
        "edges": [np.empty(0, dtype=np.float64)] * 9,
        "weights": [np.zeros(1, dtype=np.int64)] * 9,
        "bias": 0,
        "threshold": threshold,
    }


def test_compute_next_use():
    nxt = compute_next_use(np.array([1, 2, 1, 1, 2]))
    assert nxt.tolist() == [2, 4, 3, 5, 5]  # 5 == len == never again


def test_fifo_textbook():
    res = run_fifo(_stream(TEXTBOOK), capacity=3)
    assert res["accesses"] - res["hits"] == 15


def test_lru_textbook():
    res = run_lru(_stream(TEXTBOOK), capacity=3)
    assert res["accesses"] - res["hits"] == 12


def test_belady_textbook():
    # Textbook OPT (no bypass) faults 9 on this string; our MIN allows bypass
    # (evicting the just-inserted page), whose true optimum is 8 -- verified by
    # exhaustive search over all victim choices including bypass.
    res = run_belady(_stream(TEXTBOOK), capacity=3)
    assert res["accesses"] - res["hits"] == 8


def test_belady_beats_or_ties_everyone():
    rng = np.random.RandomState(0)
    pages = rng.zipf(1.3, size=3000) % 100
    s = _stream(pages.tolist())
    misses = {
        name: r["accesses"] - r["hits"]
        for name, r in {
            "fifo": run_fifo(s, 20),
            "lru": run_lru(s, 20),
            "belady": run_belady(s, 20),
        }.items()
    }
    assert misses["belady"] <= misses["fifo"]
    assert misses["belady"] <= misses["lru"]


def test_protect_never_protect_with_batch1_equals_fifo():
    # threshold so high nothing is protected, batch=1 sample=1 -> exact FIFO.
    s = _stream(TEXTBOOK)
    res = run_protect(s, capacity=3, model=_trivial_model(10**9), batch=1, sample=1)
    assert res["accesses"] - res["hits"] == 15


def test_protect_all_protected_still_makes_progress():
    # threshold below bias=0 -> everything protected; eviction must still
    # evict one per group (stalest), keeping size bounded at capacity.
    s = _stream(TEXTBOOK * 5)
    res = run_protect(s, capacity=3, model=_trivial_model(-1), batch=1, sample=2)
    assert res["hits"] > 0
    assert res["accesses"] == len(TEXTBOOK) * 5

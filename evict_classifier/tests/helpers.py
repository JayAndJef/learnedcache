"""Synthetic structured-array builders for tests."""

from __future__ import annotations

import numpy as np

from evict_classifier.loading import _ACCESS_DTYPE, _EVICTION_DTYPE

DISCRETIZE_COLS = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"]


def make_access(records: list[dict]) -> np.ndarray:
    """Build an access array from a list of field dicts (missing fields -> 0)."""
    arr = np.zeros(len(records), dtype=_ACCESS_DTYPE)
    for i, rec in enumerate(records):
        for key, val in rec.items():
            arr[key][i] = val
    return arr


def make_eviction(ts_list: list[int]) -> np.ndarray:
    arr = np.zeros(len(ts_list), dtype=_EVICTION_DTYPE)
    arr["ts"] = ts_list
    return arr


def make_synthetic_trial(
    n_hot: int = 30, n_cold: int = 30, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Build a trial with a clean reuse signal at a horizon of ~5 ts units.

    Hot pages are accessed twice 2 units apart (reused soon); cold pages once.
    One eviction event sits between each hot page's two accesses, so hot pages
    label positive and cold pages negative.
    """
    rng = np.random.RandomState(seed)
    records: list[dict] = []
    evictions: list[int] = []

    def feat_fields() -> dict:
        return {c: int(rng.randint(1, 1000)) for c in DISCRETIZE_COLS}

    base = 1000
    for i in range(n_hot):
        t = base + i * 10
        ino = 100 + (i % 5)  # a few hot inodes
        records.append({"ts": t, "dm": 1, "dn": 0, "in": ino, "of": i, **feat_fields()})
        records.append({"ts": t + 2, "dm": 1, "dn": 0, "in": ino, "of": i, **feat_fields()})
        evictions.append(t + 1)  # between the two accesses

    for j in range(n_cold):
        t = base + (n_hot + j) * 10
        ino = 900 + (j % 5)  # cold inodes
        records.append({"ts": t, "dm": 1, "dn": 0, "in": ino, "of": 10_000 + j, **feat_fields()})
        evictions.append(t + 1)  # after the only access -> no future

    records.sort(key=lambda r: r["ts"])
    return make_access(records), make_eviction(sorted(evictions))

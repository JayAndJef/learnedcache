"""Tests for the turnover estimator (insertion/eviction logs only)."""

import numpy as np
import pytest

from evict_classifier.loading import _INSERTION_DTYPE, estimate_turnover

from .helpers import make_access, make_eviction

_MS = 1_000_000  # ns


def _write_logs(tmp_path, ins_ts, ev_ts, access_ts=None):
    iter_dir = tmp_path / "iter_1"
    iter_dir.mkdir()
    ins = np.zeros(len(ins_ts), dtype=_INSERTION_DTYPE)
    ins["ts"] = ins_ts
    ins.tofile(iter_dir / "mglru_lc_insertion_1.bin")
    make_eviction(ev_ts).tofile(iter_dir / "mglru_lc_eviction_1.bin")
    access_ts = access_ts if access_ts is not None else [0, max(ins_ts) + 1]
    make_access([{"ts": t} for t in access_ts]).tofile(
        iter_dir / "mglru_lc_access_1.bin"
    )
    return iter_dir


def _two_bursts():
    """100 insertions 1 ms apart, a 10 s idle gap, then 100 more."""
    burst1 = [i * _MS for i in range(100)]
    burst2 = [burst1[-1] + 10_000 * _MS + i * _MS for i in range(100)]
    return burst1 + burst2


def test_estimate_turnover_active_periods_and_capacity(tmp_path):
    ins_ts = _two_bursts()
    # First eviction lands after the 150th insertion -> capacity estimate 150.
    first_ev = ins_ts[149] + 1
    last_access = ins_ts[-1] + 5 * _MS
    iter_dir = _write_logs(
        tmp_path, ins_ts, [first_ev, ins_ts[160]], access_ts=[0, last_access]
    )

    est = estimate_turnover(iter_dir)
    assert est.capacity_pages == 150
    assert est.capacity_estimated
    assert est.n_insertions == 200
    assert est.label_window_ns == pytest.approx(last_access - first_ev)
    # The 10 s idle gap is excluded: 198 active gaps of 1 ms each.
    active_ns = 198 * _MS
    assert est.active_seconds == pytest.approx(active_ns / 1e9)
    rate_per_ns = 199 / active_ns
    assert est.insertion_rate_per_s == pytest.approx(rate_per_ns * 1e9)
    assert est.turnover_ns == pytest.approx(150 / rate_per_ns)


def test_estimate_turnover_capacity_override(tmp_path):
    ins_ts = _two_bursts()
    iter_dir = _write_logs(tmp_path, ins_ts, [ins_ts[149] + 1])

    est = estimate_turnover(iter_dir, capacity_pages=600)
    assert est.capacity_pages == 600
    assert not est.capacity_estimated
    assert est.turnover_ns == pytest.approx(600 / (199 / (198 * _MS)))


def test_estimate_turnover_empty_eviction_log_raises(tmp_path):
    iter_dir = _write_logs(tmp_path, _two_bursts(), [])
    with pytest.raises(ValueError, match="never filled"):
        estimate_turnover(iter_dir)
    # ... but a supplied capacity makes the eviction log unnecessary.
    est = estimate_turnover(iter_dir, capacity_pages=100)
    assert est.capacity_pages == 100
    assert est.label_window_ns is None


def test_estimate_turnover_missing_insertion_log_raises(tmp_path):
    iter_dir = tmp_path / "iter_1"
    iter_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        estimate_turnover(iter_dir)

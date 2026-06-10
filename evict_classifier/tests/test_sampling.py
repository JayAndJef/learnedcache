"""Tests for the vectorized candidate sampler."""

import numpy as np

from evict_classifier.sampling import (
    _draw,
    _interval_bounds,
    collect_workload_sample,
    sample_trial,
)

from .helpers import DISCRETIZE_COLS, make_access, make_eviction, make_synthetic_trial


def test_interval_bounds_partition_events():
    # one page accessed at 10 and 30; events between are split by the horizon.
    ts_s = np.array([10.0, 30.0])
    next_ts = np.array([30.0, np.inf])
    ev = np.array([5.0, 12.0, 25.0, 28.0, 40.0])  # sorted events
    lo, pos_lo, hi = _interval_bounds(ev, ts_s, next_ts, horizon=5.0)
    # access0 interval (10,30): events 12,25,28 -> [lo,hi)=[1,4); within 5 of 30
    # are 25?,28 -> E>25 => indices for >25 ('right' of 25)=3 -> pos covers {28}.
    assert lo[0] == 1 and hi[0] == 4
    assert pos_lo[0] == 3  # events >25: just 28 -> [3,4) positive, [1,3) negative
    # access1 has no next access -> all its events are negative, none positive.
    assert hi[1] == lo[1] or pos_lo[1] == hi[1]


def test_interval_bounds_residency_cap_truncates_window():
    # one access at 10, never reused; events at 15, 30, 60.
    ts_s = np.array([10.0])
    next_ts = np.array([np.inf])
    ev = np.array([15.0, 30.0, 60.0])
    # uncapped: all three events are candidates.
    lo, pos_lo, hi = _interval_bounds(ev, ts_s, next_ts, horizon=5.0)
    assert (hi - lo)[0] == 3
    # cap 25: window is (10, 35) -> events 15 and 30 only; 60 is excluded
    # because the page would no longer be resident.
    lo, pos_lo, hi = _interval_bounds(ev, ts_s, next_ts, horizon=5.0, residency_cap=25.0)
    assert lo[0] == 0 and hi[0] == 2
    assert pos_lo[0] == hi[0]  # no reuse -> no positive band


def test_sample_trial_residency_cap_drops_stale_candidates():
    # page accessed at 10 only; eviction events at 15 (fresh) and 90 (stale).
    # A later unrelated access lifts ts.max so neither event is censored.
    access = make_access(
        [
            {"ts": 10, "dm": 1, "dn": 0, "in": 7, "of": 3},
            {"ts": 200, "dm": 2, "dn": 0, "in": 9, "of": 0},
        ]
    )
    kw = dict(
        horizon=5.0, n_train=20, n_eval=0, disc_sample_size=10,
        balanced=False, holdout_frac=0.0,
    )
    uncapped = sample_trial(
        access, make_eviction([15, 90]), DISCRETIZE_COLS,
        rng=np.random.RandomState(0), **kw,
    )
    capped = sample_trial(
        access, make_eviction([15, 90]), DISCRETIZE_COLS,
        rng=np.random.RandomState(0), residency_cap=30.0, **kw,
    )
    # page-7 contributes 2 candidate events uncapped, 1 with the cap; the
    # unrelated page at 200 has no events after it. (Plus page-9 contributes 0.)
    assert uncapped.n_neg == 2
    assert capped.n_neg == 1
    # the surviving candidate is the fresh one: derived TSA = 15 - 10 = 5.
    assert np.allclose(capped.x_train[:, -1], 5.0)


def test_draw_indices_in_range():
    rng = np.random.RandomState(0)
    counts = np.array([0, 3, 0, 2], dtype=np.int64)
    starts = np.array([100, 200, 300, 400], dtype=np.int64)
    rec, e_idx = _draw(counts, starts, n=500, rng=rng)
    assert len(rec) == 500
    # only records with positive counts can be chosen
    assert set(np.unique(rec)).issubset({1, 3})
    # event index stays within the record's [start, start+count) band
    for r, e in zip(rec, e_idx):
        assert starts[r] <= e < starts[r] + counts[r]


def test_draw_empty_when_no_events():
    rng = np.random.RandomState(0)
    rec, e_idx = _draw(np.zeros(4, np.int64), np.arange(4), n=10, rng=rng)
    assert len(rec) == 0 and len(e_idx) == 0


def test_sample_trial_labels_and_features():
    # page p: access at 10 (features pd=11) then 20; eviction at 15.
    # interval (10,20), event 15, horizon 8 -> next(20)-15=5<8 -> positive.
    # A later unrelated access at 100 lifts ts.max so event 15 isn't censored.
    access = make_access(
        [
            {"ts": 10, "dm": 1, "dn": 0, "in": 7, "of": 3, "pd": 11},
            {"ts": 20, "dm": 1, "dn": 0, "in": 7, "of": 3, "pd": 22},
            {"ts": 100, "dm": 2, "dn": 0, "in": 9, "of": 0, "pd": 1},
        ]
    )
    res = sample_trial(
        access, make_eviction([15]), DISCRETIZE_COLS,
        horizon=8.0, n_train=10, n_eval=0, disc_sample_size=10,
        balanced=False, holdout_frac=0.0, rng=np.random.RandomState(0),
    )
    assert res.n_pos == 1 and res.n_neg == 0
    assert res.x_train.shape[1] == len(DISCRETIZE_COLS) + 1
    assert (res.y_train == 1.0).all()
    # derived = 15 - 10 = 5; prior features from the ts=10 record (pd=11).
    assert np.allclose(res.x_train[:, -1], 5.0)
    assert np.allclose(res.x_train[:, DISCRETIZE_COLS.index("pd")], 11.0)


def test_sample_trial_censors_tail_events():
    # last access at 20; horizon 10 -> events after 20-10=10 are censored.
    access = make_access(
        [
            {"ts": 5, "dm": 1, "dn": 0, "in": 7, "of": 3},
            {"ts": 20, "dm": 1, "dn": 0, "in": 7, "of": 3},
        ]
    )
    # event at 8 is observable (<=10); event at 18 is censored (>10).
    res = sample_trial(
        access, make_eviction([8, 18]), DISCRETIZE_COLS,
        horizon=10.0, n_train=20, n_eval=0, disc_sample_size=10,
        balanced=False, holdout_frac=0.0, rng=np.random.RandomState(0),
    )
    # only the event at 8 survives; it falls in (5,20), next=20, 20-8=12>10 -> neg.
    assert res.n_pos == 0
    assert res.n_neg == 1


def test_interval_bounds_match_bruteforce():
    """Randomized check of the vectorized bands against an O(n^2) reference."""
    rng = np.random.RandomState(123)
    for trial in range(20):
        horizon = float(rng.randint(1, 30))
        cap = float(rng.randint(5, 50)) if trial % 2 else None
        ts_s = np.sort(rng.randint(0, 200, size=15).astype(np.float64))
        # next_ts must be > ts (simulate per-page next access), some inf
        gaps = rng.randint(1, 60, size=15).astype(np.float64)
        next_ts = ts_s + gaps
        next_ts[rng.rand(15) < 0.3] = np.inf
        ev = np.sort(rng.randint(0, 250, size=25).astype(np.float64))

        lo, pos_lo, hi = _interval_bounds(ev, ts_s, next_ts, horizon, cap)

        for i in range(15):
            for j, e in enumerate(ev):
                in_window = ts_s[i] < e < next_ts[i]
                if cap is not None:
                    in_window = in_window and e < ts_s[i] + cap
                is_pos = in_window and (next_ts[i] - e) < horizon
                got_in = lo[i] <= j < hi[i]
                got_pos = pos_lo[i] <= j < hi[i]
                assert got_in == in_window, (trial, i, j)
                assert got_pos == is_pos, (trial, i, j)


def _make_insertion(records: list[dict]) -> np.ndarray:
    from evict_classifier.loading import _INSERTION_DTYPE

    arr = np.zeros(len(records), dtype=_INSERTION_DTYPE)
    for i, rec in enumerate(records):
        for key, val in rec.items():
            arr[key][i] = val
    return arr


def test_insertion_anchor_rewrites_tsa_and_page_features():
    # page (1,0,7,3): access at 10 (pd=11, fq=5, p2=22), next access at 70.
    # The page is re-inserted at u=50 (so it was evicted in (10, 50)); an
    # eviction event at E=60 must see the kernel's fresh-entry state:
    # TSA = 60-50 = 10, pd/p2 = UNKNOWN sentinel, fq = 0.
    access = make_access(
        [
            {"ts": 10, "dm": 1, "dn": 0, "in": 7, "of": 3, "pd": 11, "fq": 5, "p2": 22},
            {"ts": 70, "dm": 1, "dn": 0, "in": 7, "of": 3, "pd": 1, "fq": 6, "p2": 1},
            {"ts": 1000, "dm": 2, "dn": 0, "in": 9, "of": 0},  # lifts ts.max (censoring)
        ]
    )
    insertion = _make_insertion([{"ts": 50, "dm": 1, "dn": 0, "in": 7, "ix": 3}])
    kw = dict(
        horizon=15.0, n_train=8, n_eval=0, disc_sample_size=4,
        balanced=False, holdout_frac=0.0,
    )

    plain = sample_trial(
        access, make_eviction([60]), DISCRETIZE_COLS,
        rng=np.random.RandomState(0), **kw,
    )
    # without the insertion log: stale anchor (TSA = 60-10) and stale features.
    assert np.allclose(plain.x_train[:, -1], 50.0)
    assert np.allclose(plain.x_train[:, DISCRETIZE_COLS.index("pd")], 11.0)

    fixed = sample_trial(
        access, make_eviction([60]), DISCRETIZE_COLS,
        rng=np.random.RandomState(0), insertion=insertion, **kw,
    )
    assert fixed.n_anchor_corrected > 0
    assert np.allclose(fixed.x_train[:, -1], 10.0)  # anchored at insertion
    sent = np.float32(2.0**64)
    assert np.allclose(fixed.x_train[:, DISCRETIZE_COLS.index("pd")], sent)
    assert np.allclose(fixed.x_train[:, DISCRETIZE_COLS.index("p2")], sent)
    assert np.allclose(fixed.x_train[:, DISCRETIZE_COLS.index("fq")], 0.0)
    # label unchanged: next access 70, E=60, 10 < horizon 15 -> positive.
    assert (fixed.y_train == 1.0).all()


def test_insertion_anchor_ignores_insertions_before_prior_access():
    # insertion at u=5 precedes the access at 10 -> not in (t_i, E]; no patch.
    access = make_access(
        [
            {"ts": 10, "dm": 1, "dn": 0, "in": 7, "of": 3, "pd": 11},
            {"ts": 1000, "dm": 2, "dn": 0, "in": 9, "of": 0},
        ]
    )
    insertion = _make_insertion([{"ts": 5, "dm": 1, "dn": 0, "in": 7, "ix": 3}])
    res = sample_trial(
        access, make_eviction([60]), DISCRETIZE_COLS,
        horizon=15.0, n_train=8, n_eval=0, disc_sample_size=4,
        balanced=False, holdout_frac=0.0,
        rng=np.random.RandomState(0), insertion=insertion,
    )
    assert res.n_anchor_corrected == 0
    assert np.allclose(res.x_train[:, -1], 50.0)
    assert np.allclose(res.x_train[:, DISCRETIZE_COLS.index("pd")], 11.0)


def test_insertion_anchor_uses_latest_insertion():
    access = make_access(
        [
            {"ts": 10, "dm": 1, "dn": 0, "in": 7, "of": 3},
            {"ts": 1000, "dm": 2, "dn": 0, "in": 9, "of": 0},
        ]
    )
    insertion = _make_insertion(
        [
            {"ts": 20, "dm": 1, "dn": 0, "in": 7, "ix": 3},
            {"ts": 55, "dm": 1, "dn": 0, "in": 7, "ix": 3},
        ]
    )
    res = sample_trial(
        access, make_eviction([60]), DISCRETIZE_COLS,
        horizon=15.0, n_train=8, n_eval=0, disc_sample_size=4,
        balanced=False, holdout_frac=0.0,
        rng=np.random.RandomState(0), insertion=insertion,
    )
    assert np.allclose(res.x_train[:, -1], 5.0)  # 60 - 55, latest insertion wins


def test_collect_balanced_both_classes():
    access, eviction = make_synthetic_trial(n_hot=40, n_cold=40, seed=1)
    sample = collect_workload_sample(
        iter([(0, access, eviction)]),
        DISCRETIZE_COLS,
        horizon=5.0,
        target_rows=4000,
        balanced=True,
        disc_sample_size=500,
        eval_rows=500,
        holdout_frac=0.2,
        random_state=0,
        verbose=False,
    )
    assert sample.x_train.shape[1] == len(DISCRETIZE_COLS) + 1
    assert sample.n_pos_seen > 0 and sample.n_neg_seen > 0
    n_pos = int(sample.y_train.sum())
    assert 0 < n_pos < len(sample.y_train)  # both classes present
    assert len(sample.disc_sample) > 0

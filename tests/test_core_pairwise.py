import numpy as np
import pytest

from learnedcache.core import _sample_pairwise_diffs_by_event

def test_sample_pairwise_diffs_by_event_stays_within_event_boundaries() -> None:
    # Event-local feature design: first coordinate is constant within each event.
    # Same-event diffs must therefore have first coordinate = 0.
    x_full = np.array(
        [
            [0.0, 0.0],   # event 0
            [0.0, 1.0],   # event 0
            [0.0, 2.0],   # event 0
            [10.0, 0.0],  # event 1
            [10.0, 1.0],  # event 1
            [20.0, 0.0],  # event 2 (single row, no pairs)
        ],
        dtype=np.float32,
    )
    y_full = np.array([1.0, 2.0, 3.0, 5.0, 2.0, 9.0], dtype=np.float32)
    event_ids = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)

    x_diff, y_pairs, stats = _sample_pairwise_diffs_by_event(
        x_full=x_full,
        y_full=y_full,
        event_ids=event_ids,
        pairs_per_event=64,
        random_state=42,
    )

    assert len(x_diff) == len(y_pairs)
    assert len(x_diff) > 0
    assert np.allclose(x_diff[:, 0], 0.0)
    assert stats["events_total"] == 3
    assert stats["events_with_pairs"] >= 2

def test_sample_pairwise_diffs_by_event_drops_ties_and_errors_if_only_ties() -> None:
    x_full = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_full = np.array([7.0, 7.0], dtype=np.float32)  # tie only
    event_ids = np.array([0, 0], dtype=np.int64)

    with pytest.raises(ValueError, match="No pairwise samples generated after tie-drop"):
        _sample_pairwise_diffs_by_event(
            x_full=x_full,
            y_full=y_full,
            event_ids=event_ids,
            pairs_per_event=32,
            random_state=7,
        )

def test_sample_pairwise_diffs_by_event_no_reuse_as_worse_rank_and_global_cap() -> None:
    # y=100 models no-reuse surrogate (worse than finite 2.0).
    x_a = np.array([1.0, 0.0], dtype=np.float32)  # finite reuse
    x_b = np.array([0.0, 1.0], dtype=np.float32)  # no-reuse surrogate

    x_full = np.vstack([x_a, x_b]).astype(np.float32)
    y_full = np.array([2.0, 100.0], dtype=np.float32)
    event_ids = np.array([0, 0], dtype=np.int64)

    x_diff, y_pairs, _stats = _sample_pairwise_diffs_by_event(
        x_full=x_full,
        y_full=y_full,
        event_ids=event_ids,
        pairs_per_event=256,
        random_state=123,
        max_pairs_total=40,
    )

    assert len(x_diff) == len(y_pairs)
    assert len(x_diff) <= 40  # global cap respected

    diff_ab = x_a - x_b
    diff_ba = x_b - x_a

    mask_ab = np.all(np.isclose(x_diff, diff_ab), axis=1)
    mask_ba = np.all(np.isclose(x_diff, diff_ba), axis=1)

    # If A=finite and B=no-reuse, A should be labeled sooner (1).
    if mask_ab.any():
        assert np.all(y_pairs[mask_ab] == 1.0)

    # Reverse ordering should label 0.
    if mask_ba.any():
        assert np.all(y_pairs[mask_ba] == 0.0)
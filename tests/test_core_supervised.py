import pandas as pd
import pytest

from learnedcache.core import (
    DERIVED_FEATURE_COL,
    TARGET_COL,
    _build_eviction_supervised_df,
)

def _row(dm: int, dn: int, in_: int, of: int, ts: float, pd_val: float) -> dict[str, float]:
    return {
        "dm": dm,
        "dn": dn,
        "in": in_,
        "of": of,
        "ts": ts,
        "pd": pd_val,
    }

def _pick(
    df: pd.DataFrame,
    trial_id: int,
    dm: int,
    dn: int,
    in_: int,
    of: int,
    eviction_ts: float,
) -> pd.Series:
    return df[
        (df["trial_id"] == trial_id)
        & (df["dm"] == dm)
        & (df["dn"] == dn)
        & (df["in"] == in_)
        & (df["of"] == of)
        & (df["eviction_ts"] == eviction_ts)
    ].iloc[0]

def test_build_eviction_supervised_df_multi_trial_no_cross_boundary_leakage() -> None:
    # Trial 0: shared page A + trial-0-unique page B
    access_df_0 = pd.DataFrame(
        [
            _row(1, 1, 1, 1, 1, 10),    # A
            _row(2, 2, 2, 2, 2, 20),    # B
            _row(1, 1, 1, 1, 5, 11),    # A update
            _row(1, 1, 1, 1, 9, 12),    # A reuse
            _row(2, 2, 2, 2, 12, 21),   # B reuse
        ]
    )
    eviction_df_0 = pd.DataFrame([{"ts": 6}, {"ts": 10}])

    # Trial 1: shared page A + trial-1-unique page C
    access_df_1 = pd.DataFrame(
        [
            _row(1, 1, 1, 1, 100, 110),   # A (shared key, different trial)
            _row(3, 3, 3, 3, 101, 130),   # C
            _row(1, 1, 1, 1, 106, 111),   # A reuse
            _row(3, 3, 3, 3, 109, 131),   # C reuse
        ]
    )
    eviction_df_1 = pd.DataFrame([{"ts": 103}, {"ts": 108}])

    out = _build_eviction_supervised_df(
        access_eviction_pairs=[
            (0, access_df_0, eviction_df_0),
            (1, access_df_1, eviction_df_1),
        ],
        discretize_cols=["pd"],
    )

    assert DERIVED_FEATURE_COL in out.columns
    assert TARGET_COL in out.columns

    # 2 tracked pages at each eviction in each trial => 8 rows
    assert len(out) == 8

    # Finite targets across both trials: {3, 6, 2, 3, 6, 1}; max=6 => no_reuse=7
    no_reuse_expected = 7.0

    # ---- Trial 0 checks ----
    t0_a6 = _pick(out, 0, 1, 1, 1, 1, 6.0)
    t0_b6 = _pick(out, 0, 2, 2, 2, 2, 6.0)
    t0_a10 = _pick(out, 0, 1, 1, 1, 1, 10.0)
    t0_b10 = _pick(out, 0, 2, 2, 2, 2, 10.0)

    assert t0_a6[DERIVED_FEATURE_COL] == pytest.approx(1.0)     # 6 - 5
    assert t0_a6[TARGET_COL] == pytest.approx(3.0)              # 9 - 6
    assert t0_a6["pd"] == pytest.approx(11.0)

    assert t0_b6[DERIVED_FEATURE_COL] == pytest.approx(4.0)     # 6 - 2
    assert t0_b6[TARGET_COL] == pytest.approx(6.0)              # 12 - 6
    assert t0_b6["pd"] == pytest.approx(20.0)

    # Critical anti-leak assertion:
    # If cross-trial leakage were allowed, this could incorrectly reuse trial-1 A@100.
    assert t0_a10[DERIVED_FEATURE_COL] == pytest.approx(1.0)    # 10 - 9
    assert t0_a10[TARGET_COL] == pytest.approx(no_reuse_expected)
    assert t0_a10["pd"] == pytest.approx(12.0)

    assert t0_b10[DERIVED_FEATURE_COL] == pytest.approx(8.0)    # 10 - 2
    assert t0_b10[TARGET_COL] == pytest.approx(2.0)             # 12 - 10
    assert t0_b10["pd"] == pytest.approx(20.0)

    # ---- Trial 1 checks ----
    t1_a103 = _pick(out, 1, 1, 1, 1, 1, 103.0)
    t1_c103 = _pick(out, 1, 3, 3, 3, 3, 103.0)
    t1_a108 = _pick(out, 1, 1, 1, 1, 1, 108.0)
    t1_c108 = _pick(out, 1, 3, 3, 3, 3, 108.0)

    # Critical anti-leak assertion:
    # Must use trial-1 A@100, not trial-0 A@9.
    assert t1_a103[DERIVED_FEATURE_COL] == pytest.approx(3.0)   # 103 - 100
    assert t1_a103[TARGET_COL] == pytest.approx(3.0)            # 106 - 103
    assert t1_a103["pd"] == pytest.approx(110.0)

    assert t1_c103[DERIVED_FEATURE_COL] == pytest.approx(2.0)   # 103 - 101
    assert t1_c103[TARGET_COL] == pytest.approx(6.0)            # 109 - 103
    assert t1_c103["pd"] == pytest.approx(130.0)

    assert t1_a108[DERIVED_FEATURE_COL] == pytest.approx(2.0)   # 108 - 106
    assert t1_a108[TARGET_COL] == pytest.approx(no_reuse_expected)
    assert t1_a108["pd"] == pytest.approx(111.0)

    assert t1_c108[DERIVED_FEATURE_COL] == pytest.approx(7.0)   # 108 - 101
    assert t1_c108[TARGET_COL] == pytest.approx(1.0)            # 109 - 108
    assert t1_c108["pd"] == pytest.approx(130.0)

    no_reuse_rows = out[out[TARGET_COL] == no_reuse_expected]
    assert len(no_reuse_rows) == 2

def test_build_eviction_supervised_df_fails_on_missing_required_column() -> None:
    access_df = pd.DataFrame(
        [
            {"dm": 1, "dn": 1, "in": 1, "of": 1, "ts": 1},  # missing "pd"
        ]
    )
    eviction_df = pd.DataFrame([{"ts": 2}])

    with pytest.raises(ValueError, match="missing required columns"):
        _build_eviction_supervised_df(
            access_eviction_pairs=[(0, access_df, eviction_df)],
            discretize_cols=["pd"],
        )
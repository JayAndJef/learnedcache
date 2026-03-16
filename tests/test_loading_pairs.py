import pandas as pd
import pytest

from learnedcache.loading import read_access_eviction_trial_pairs

def test_read_access_eviction_trial_pairs_pairs_by_sorted_token(tmp_path) -> None:
    access_a = tmp_path / "a_access.csv"
    access_b = tmp_path / "b_access.csv"
    eviction_a = tmp_path / "a_eviction.csv"
    eviction_b = tmp_path / "b_eviction.csv"

    pd.DataFrame([{"ts": 1, "dm": 1, "dn": 1, "in": 1, "of": 1, "pd": 10}]).to_csv(access_b, index=False)
    pd.DataFrame([{"ts": 2, "dm": 2, "dn": 2, "in": 2, "of": 2, "pd": 20}]).to_csv(access_a, index=False)
    pd.DataFrame([{"ts": 100}]).to_csv(eviction_b, index=False)
    pd.DataFrame([{"ts": 200}]).to_csv(eviction_a, index=False)

    pairs = read_access_eviction_trial_pairs(
        access_pattern=str(tmp_path / "*_access.csv"),
        eviction_pattern=str(tmp_path / "*_eviction.csv"),
    )

    assert len(pairs) == 2

    trial0_id, trial0_access, trial0_eviction = pairs[0]
    trial1_id, trial1_access, trial1_eviction = pairs[1]

    assert trial0_id == 0
    assert trial1_id == 1

    # Token sort order should be: a, then b
    assert int(trial0_access.iloc[0]["dm"]) == 2
    assert int(trial0_eviction.iloc[0]["ts"]) == 200
    assert int(trial1_access.iloc[0]["dm"]) == 1
    assert int(trial1_eviction.iloc[0]["ts"]) == 100

def test_read_access_eviction_trial_pairs_raises_when_no_common_tokens(tmp_path) -> None:
    pd.DataFrame([{"ts": 1, "dm": 1, "dn": 1, "in": 1, "of": 1, "pd": 10}]).to_csv(
        tmp_path / "only_access_access.csv", index=False
    )
    pd.DataFrame([{"ts": 2}]).to_csv(tmp_path / "different_eviction.csv", index=False)

    with pytest.raises(ValueError, match="No matching access/eviction file pairs found by token"):
        read_access_eviction_trial_pairs(
            access_pattern=str(tmp_path / "*_access.csv"),
            eviction_pattern=str(tmp_path / "*_eviction.csv"),
        )

def test_read_access_eviction_trial_pairs_raises_on_partial_overlap_tokens(tmp_path) -> None:
    pd.DataFrame([{"ts": 1, "dm": 1, "dn": 1, "in": 1, "of": 1, "pd": 10}]).to_csv(
        tmp_path / "shared_access.csv", index=False
    )
    pd.DataFrame([{"ts": 2, "dm": 2, "dn": 2, "in": 2, "of": 2, "pd": 20}]).to_csv(
        tmp_path / "access_only_access.csv", index=False
    )
    pd.DataFrame([{"ts": 3}]).to_csv(tmp_path / "shared_eviction.csv", index=False)

    with pytest.raises(ValueError, match="token sets must match exactly"):
        read_access_eviction_trial_pairs(
            access_pattern=str(tmp_path / "*_access.csv"),
            eviction_pattern=str(tmp_path / "*_eviction.csv"),
        )

"""End-to-end smoke test: synthetic .bin files -> train -> exported JSON."""

import json

import numpy as np

from evict_classifier.loading import _INSERTION_DTYPE
from evict_classifier.train import train_workload

from .helpers import make_synthetic_trial


def test_train_workload_end_to_end(tmp_path):
    access, eviction = make_synthetic_trial(n_hot=40, n_cold=40, seed=2)

    iter_dir = tmp_path / "ycsb_test" / "iter_1"
    iter_dir.mkdir(parents=True)
    access.tofile(iter_dir / "mglru_lc_access_1.bin")
    eviction.tofile(iter_dir / "mglru_lc_eviction_1.bin")

    out_dir = tmp_path / "out"
    result = train_workload(
        "ycsb_test",
        [iter_dir],
        out_dir,
        horizon=5.0,  # raw ts units in this synthetic trial
        target_rows=2000,
        n_bins=4,
        max_epochs=2,
        batch_size=256,
        disc_sample_size=500,
        eval_rows=500,
        random_state=0,
        verbose=False,
    )

    assert result["n_train_rows"] > 0
    weights = out_dir / "ycsb_test" / "model_weights.json"
    assert weights.exists()
    data = json.loads(weights.read_text())
    assert data["model_type"] == "binary_reuse_classifier"
    assert "bias_int" in data and "threshold_int" in data
    assert len(data["features"]) == 9
    assert (out_dir / "ycsb_test" / "model.keras").exists()
    assert (out_dir / "ycsb_test" / "discretizer.pkl").exists()
    assert (out_dir / "ycsb_test" / "feature_importance.png").stat().st_size > 0
    metrics = json.loads((out_dir / "ycsb_test" / "metrics.json").read_text())
    assert metrics["horizon_source"] == "manual"
    assert metrics["horizon_ns"] == 5.0
    assert metrics["residency_cap_ns"] == 5.0  # None follows the horizon


def test_train_workload_auto_horizon(tmp_path):
    access, eviction = make_synthetic_trial(n_hot=40, n_cold=40, seed=2)

    iter_dir = tmp_path / "ycsb_test" / "iter_1"
    iter_dir.mkdir(parents=True)
    access.tofile(iter_dir / "mglru_lc_access_1.bin")
    eviction.tofile(iter_dir / "mglru_lc_eviction_1.bin")

    # Insertions 1 ts-unit apart (rate 1/unit); the trial's first eviction is at
    # ts 1001, preceded by 5 insertions -> capacity 5 -> turnover = 5 units,
    # matching the manual smoke test's horizon.
    ins = np.zeros(605, dtype=_INSERTION_DTYPE)
    ins["ts"] = np.arange(996, 996 + 605)
    ins.tofile(iter_dir / "mglru_lc_insertion_1.bin")

    out_dir = tmp_path / "out"
    result = train_workload(
        "ycsb_test",
        [iter_dir],
        out_dir,
        horizon=None,
        target_rows=2000,
        n_bins=4,
        max_epochs=2,
        batch_size=256,
        disc_sample_size=500,
        eval_rows=500,
        random_state=0,
        verbose=False,
    )

    assert result["n_train_rows"] > 0
    metrics = json.loads((out_dir / "ycsb_test" / "metrics.json").read_text())
    assert metrics["horizon_source"] == "auto"
    assert metrics["horizon_ns"] == 5.0
    assert metrics["residency_cap_ns"] == 5.0
    assert metrics["capacity_pages"] == 5
    assert metrics["capacity_estimated"] is True

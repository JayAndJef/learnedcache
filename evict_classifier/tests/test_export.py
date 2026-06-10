"""Tests for the BPF export contract (bias/threshold + per-feature arrays)."""

import json

import numpy as np

from evict_classifier.export import DEFAULT_FEATURE_NAMES, export_classifier
from evict_classifier.models import build_binary_classifier
from evict_classifier.preprocess import fit_discretizer_from_sample


def _fit(n_features=9, n_bins=4, seed=0):
    rng = np.random.RandomState(seed)
    disc = fit_discretizer_from_sample(rng.rand(500, n_features) * 1000, n_bins=n_bins)
    n_bins_list = [len(disc.bin_edges_[i]) - 1 for i in range(n_features)]
    model = build_binary_classifier(sum(n_bins_list))
    return model, disc, n_bins_list


def test_export_has_bias_threshold_and_feature_arrays(tmp_path):
    model, disc, n_bins_list = _fit()
    out = tmp_path / "model_weights.json"
    data = export_classifier(
        out, model, disc, threshold=0.5, weight_scale=1000, verbose=False
    )

    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == data

    assert data["model_type"] == "binary_reuse_classifier"
    assert data["feature_names"] == DEFAULT_FEATURE_NAMES
    assert data["weight_scale"] == 1000
    assert data["threshold"] == 0.5
    assert data["threshold_int"] == 500
    assert data["bias_int"] == round(data["bias"] * 1000)

    assert len(data["features"]) == len(DEFAULT_FEATURE_NAMES)
    for i, feat in enumerate(data["features"]):
        assert feat["index"] == i
        assert feat["n_bins"] == n_bins_list[i]
        assert len(feat["weights_int"]) == feat["n_bins"]
        assert len(feat["weights_float"]) == feat["n_bins"]
        assert len(feat["bin_edges"]) == feat["n_bins"] - 1


def test_export_rejects_feature_count_mismatch(tmp_path):
    model, disc, _ = _fit(n_features=9)
    try:
        export_classifier(
            tmp_path / "x.json", model, disc,
            feature_names=["only", "three", "names"], verbose=False,
        )
    except ValueError as e:
        assert "feature_names" in str(e)
    else:
        raise AssertionError("expected ValueError on feature-count mismatch")

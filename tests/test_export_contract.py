import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from learnedcache.core import run_export_model

class _FakeLayer:
    def __init__(self, weights: np.ndarray) -> None:
        self._weights = weights

    def get_weights(self):
        return [self._weights]

class _FakeModel:
    def __init__(self, weights: np.ndarray) -> None:
        self._layer = _FakeLayer(weights)

    def get_layer(self, name: str):
        assert name == "ranking_weight"
        return self._layer

def test_run_export_model_fails_on_feature_count_mismatch(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()

    discretizer = SimpleNamespace(
        bin_edges_=[
            np.array([0.0, 1.0, 2.0], dtype=float),
            np.array([0.0, 1.0, 2.0], dtype=float),
        ]
    )
    with (model_dir / "discretizer.pkl").open("wb") as f:
        pickle.dump(discretizer, f)

    (model_dir / "model.keras").write_text("stub", encoding="utf-8")

    fake_weights = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float32)
    monkeypatch.setattr(
        "learnedcache.core.keras.models.load_model",
        lambda _path: _FakeModel(fake_weights),
    )

    with pytest.raises(ValueError, match="feature_names length does not match trained discretizer"):
        run_export_model(
            model_dir=model_dir,
            output_file=tmp_path / "out.json",
            feature_names=["only_one_feature"],
            verbose=False,
        )

def test_run_export_model_writes_expected_metadata_shape(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "model_dir"
    model_dir.mkdir()

    discretizer = SimpleNamespace(
        bin_edges_=[
            np.array([0.0, 1.0, 2.0], dtype=float),  # 2 bins
            np.array([0.0, 10.0, 20.0, 30.0], dtype=float),  # 3 bins
        ]
    )
    with (model_dir / "discretizer.pkl").open("wb") as f:
        pickle.dump(discretizer, f)

    (model_dir / "model.keras").write_text("stub", encoding="utf-8")

    # total one-hot dims = 2 + 3 = 5
    fake_weights = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    monkeypatch.setattr(
        "learnedcache.core.keras.models.load_model",
        lambda _path: _FakeModel(fake_weights),
    )

    output_file = tmp_path / "model_weights.json"
    out = run_export_model(
        model_dir=model_dir,
        output_file=output_file,
        feature_names=["f1", "f2"],
        verbose=False,
    )

    assert out["n_features"] == 2
    assert len(out["features"]) == 2
    assert out["features"][0]["n_bins"] == 2
    assert out["features"][1]["n_bins"] == 3

    written = json.loads(output_file.read_text(encoding="utf-8"))
    assert written["feature_names"] == ["f1", "f2"]
    assert len(written["features"]) == 2
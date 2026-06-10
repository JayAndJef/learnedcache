"""Export a trained binary classifier to BPF-compatible JSON.

Same per-feature ``bin_edges`` + quantized ``weights_int`` contract as the
ranker's exporter, plus two top-level scalars the ranker lacks: a quantized
``bias`` and a ``threshold``. The ``cache_ext_fifo_ml_protect`` policy decides
``sum(weights_int[bin]) + bias_int > threshold_int`` -> predicted reused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import WEIGHT_LAYER_NAME
from .sampling import DERIVED_FEATURE_COL

# Eviction-time feature order -- the BPF contract: must match the enum in
# cache_ext_fifo_ml_protect.bpf.c.
DEFAULT_FEATURE_NAMES = [
    "pd", "sz", "fq", "sd", "p2", "id", "i2", "ie", DERIVED_FEATURE_COL,
]


def export_classifier(
    output_file: str | Path,
    model: Any,
    discretizer: Any,
    *,
    feature_names: list[str] = DEFAULT_FEATURE_NAMES,
    weight_scale: int = 10000,
    threshold: float = 0.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Serialize *model* + *discretizer* to the BPF JSON schema.

    Args:
        threshold: decision threshold in *logit* units. The exported
            ``threshold_int = round(threshold * weight_scale)``; the in-kernel
            score (``sum(weight_int[bin]) + bias_int``) is compared against it.
            Default 0.0 corresponds to ``P(reuse) > 0.5``.
    """
    layer = model.get_layer(WEIGHT_LAYER_NAME)
    layer_weights = layer.get_weights()
    weights = layer_weights[0].ravel()
    bias = float(layer_weights[1][0]) if len(layer_weights) > 1 else 0.0

    if len(discretizer.bin_edges_) != len(feature_names):
        raise ValueError(
            "feature_names length does not match discretizer feature count: "
            f"{len(feature_names)} vs {len(discretizer.bin_edges_)}"
        )

    n_features = len(feature_names)
    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(n_features)]

    if verbose:
        print(f"Features: {feature_names}")
        print(f"Bins per feature: {n_bins_list}")
        print(f"Total one-hot features: {sum(n_bins_list)}  |  bias: {bias:.4f}")

    model_data: dict[str, Any] = {
        "model_type": "binary_reuse_classifier",
        "feature_names": feature_names,
        "n_features": n_features,
        "weight_scale": weight_scale,
        "bias": bias,
        "bias_int": int(round(bias * weight_scale)),
        "threshold": threshold,
        "threshold_int": int(round(threshold * weight_scale)),
        "features": [],
    }

    # Bin edges are loaded as u64 in the kernel. Float rounding of the
    # UNKNOWN_*-sentinel feature values (2^64-1) can push an edge to exactly
    # 2^64, which would overflow the loader's json_object_get_uint64 -- clamp.
    _U64_MAX = 2**64 - 1

    weight_idx = 0
    for feat_idx, feat_name in enumerate(feature_names):
        n_bins_feat = n_bins_list[feat_idx]
        interior_edges = [
            min(max(edge, 0.0), _U64_MAX)
            for edge in discretizer.bin_edges_[feat_idx][1:-1].tolist()
        ]

        feat_weights_float = weights[weight_idx : weight_idx + n_bins_feat]
        feat_weights_int = (feat_weights_float * weight_scale).astype(np.int64).tolist()

        model_data["features"].append(
            {
                "index": feat_idx,
                "name": feat_name,
                "n_bins": n_bins_feat,
                "bin_edges": [int(x) for x in interior_edges],
                "weights_float": feat_weights_float.tolist(),
                "weights_int": feat_weights_int,
            }
        )
        weight_idx += n_bins_feat

        if verbose:
            print(
                f"  {feat_name}: {n_bins_feat} bins, "
                f"weights [{feat_weights_float.min():.4f}, {feat_weights_float.max():.4f}]"
            )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)

    if verbose:
        print(f"\nExported model to {output_file} (weight_scale={weight_scale})")

    return model_data

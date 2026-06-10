"""Weight visualization for the binary reuse classifier.

Mirrors the ranker's ``feature_importance.png`` (one bar per one-hot bin,
red = pushes toward "not reused" / evict, green = pushes toward "reused" /
protect), with the bias annotated since it shifts the decision boundary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import WEIGHT_LAYER_NAME


def save_weight_plot(
    model,
    feature_names: list[str],
    n_bins_list: list[int],
    output_path: str | Path,
) -> Path:
    """Save a per-bin weight bar chart for the trained linear classifier."""
    layer_weights = model.get_layer(WEIGHT_LAYER_NAME).get_weights()
    weights = layer_weights[0].ravel()
    bias = float(layer_weights[1][0]) if len(layer_weights) > 1 else 0.0

    bin_labels = [
        f"{name}_{b}"
        for name, n_bins in zip(feature_names, n_bins_list)
        for b in range(n_bins)
    ]
    if len(bin_labels) != len(weights):
        raise ValueError(
            f"{len(bin_labels)} bin labels but {len(weights)} weights."
        )

    fig, ax = plt.subplots(figsize=(25, 10))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in weights]
    ax.bar(range(len(weights)), weights, color=colors)
    ax.set_xticks(range(len(weights)))
    ax.set_xticklabels(bin_labels, rotation=90, fontsize=10)
    ax.set_ylabel("Weight")
    ax.set_title(
        f"Binary Reuse Classifier Weights (bias = {bias:+.4f}; "
        "green = toward reused/protect, red = toward not-reused/evict)"
    )
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Feature-group separators for readability.
    edge = 0
    for n_bins in n_bins_list[:-1]:
        edge += n_bins
        ax.axvline(x=edge - 0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

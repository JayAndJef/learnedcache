#!/usr/bin/env python3
"""
Export trained eviction-time pointwise model to JSON for BPF loading.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import keras
import numpy as np
import typer

DERIVED_FEATURE_COL = "time_since_last_access_at_eviction"
DEFAULT_FEATURE_NAMES = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie", DERIVED_FEATURE_COL]

app = typer.Typer()

@app.command()
def export_model(
    model_dir: Path = typer.Option(..., help="Directory containing trained model artifacts"),
    output_file: Path = typer.Option("model_weights.json", help="Output JSON file"),
    weight_scale: int = typer.Option(
        10000,
        help="Scale factor for quantizing float weights to integers",
    ),
    feature_names: list[str] = typer.Option(
        DEFAULT_FEATURE_NAMES,
        help="Feature names in BPF enum order",
    ),
) -> None:
    """
    Export trained pointwise model to BPF-compatible JSON format.

    Expected files in model_dir:
    - discretizer.pkl: fitted KBinsDiscretizer
    - model.keras: trained Keras model
    """
    typer.echo(f"Loading model from {model_dir}...")

    discretizer_path = model_dir / "discretizer.pkl"
    if not discretizer_path.exists():
        typer.echo(f"Error: {discretizer_path} not found", err=True)
        raise typer.Exit(1)

    with discretizer_path.open("rb") as f:
        discretizer = pickle.load(f)

    model_path = model_dir / "model.keras"
    if not model_path.exists():
        typer.echo(f"Error: {model_path} not found", err=True)
        raise typer.Exit(1)

    model = keras.models.load_model(model_path)
    weights = model.get_layer("ranking_weight").get_weights()[0].ravel()

    n_features = len(feature_names)
    if len(discretizer.bin_edges_) != n_features:
        typer.echo(
            "Error: feature_names length does not match trained discretizer feature count "
            f"({n_features} vs {len(discretizer.bin_edges_)})",
            err=True,
        )
        raise typer.Exit(1)

    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(n_features)]

    typer.echo(f"Features: {feature_names}")
    typer.echo(f"Bins per feature: {n_bins_list}")
    typer.echo(f"Total one-hot features: {sum(n_bins_list)}")
    typer.echo(f"Weight vector shape: {weights.shape}")

    model_data: dict[str, Any] = {
        "feature_names": feature_names,
        "n_features": n_features,
        "weight_scale": weight_scale,
        "features": [],
    }

    weight_idx = 0
    for feat_idx, feat_name in enumerate(feature_names):
        n_bins = n_bins_list[feat_idx]
        all_edges = discretizer.bin_edges_[feat_idx]
        interior_edges = all_edges[1:-1].tolist()

        feat_weights_float = weights[weight_idx : weight_idx + n_bins]
        feat_weights_int = (feat_weights_float * weight_scale).astype(np.int64).tolist()

        feature_data = {
            "index": feat_idx,
            "name": feat_name,
            "n_bins": n_bins,
            "bin_edges": [int(x) for x in interior_edges],
            "weights_float": feat_weights_float.tolist(),
            "weights_int": feat_weights_int,
        }
        model_data["features"].append(feature_data)
        weight_idx += n_bins

        typer.echo(
            f"  {feat_name}: {n_bins} bins, "
            f"weights range [{feat_weights_float.min():.4f}, {feat_weights_float.max():.4f}]"
        )

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)

    typer.echo(f"\nExported model to {output_file}")
    typer.echo(f"Weight scale factor: {weight_scale}")
    typer.echo("Ready to load into BPF maps!")

if __name__ == "__main__":
    app()
#!/usr/bin/env python3
"""
Export trained pairwise ranker model to JSON format for BPF loading.

This script loads a trained model from pairwise_ranker.py and exports:
- Number of bins per feature
- Bin edges for discretization
- Quantized integer weights

Output: model_weights.json
"""

import json
import pickle
import numpy as np
from pathlib import Path
import typer
import keras

app = typer.Typer()


@app.command()
def export_model(
    model_dir: Path = typer.Option(..., help="Directory containing trained model artifacts"),
    output_file: Path = typer.Option("model_weights.json", help="Output JSON file"),
    weight_scale: int = typer.Option(10000, help="Scale factor for quantizing float weights to integers"),
):
    """
    Export trained pairwise ranker model to BPF-compatible JSON format.

    Expected files in model_dir:
    - discretizer.pkl: Fitted KBinsDiscretizer
    - model.keras: Trained Keras model
    """

    typer.echo(f"Loading model from {model_dir}...")

    # Load discretizer
    discretizer_path = model_dir / "discretizer.pkl"
    if not discretizer_path.exists():
        typer.echo(f"Error: {discretizer_path} not found", err=True)
        raise typer.Exit(1)

    with open(discretizer_path, "rb") as f:
        discretizer = pickle.load(f)

    # Load Keras model
    model_path = model_dir / "model.keras"
    if not model_path.exists():
        typer.echo(f"Error: {model_path} not found", err=True)
        raise typer.Exit(1)

    model = keras.models.load_model(model_path)

    # Extract weights
    w = model.get_layer("ranking_weight").get_weights()[0].ravel()

    # Feature names (must match BPF enum order)
    feature_names = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"]

    # Extract discretizer info
    n_features = len(feature_names)
    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(n_features)]

    typer.echo(f"Features: {feature_names}")
    typer.echo(f"Bins per feature: {n_bins_list}")
    typer.echo(f"Total one-hot features: {sum(n_bins_list)}")
    typer.echo(f"Weight vector shape: {w.shape}")

    # Build output structure
    model_data = {
        "feature_names": feature_names,
        "n_features": n_features,
        "weight_scale": weight_scale,
        "features": []
    }

    # Process each feature
    weight_idx = 0
    for feat_idx, feat_name in enumerate(feature_names):
        n_bins = n_bins_list[feat_idx]

        # Extract only interior bin edges (sklearn uses [-inf, interior_edges, +inf])
        # bin_edges_[i] has (n_bins + 1) edges, we only need the interior (n_bins - 1) edges
        all_edges = discretizer.bin_edges_[feat_idx]
        print(f"edges for {feat_name}: {all_edges}")
        interior_edges = all_edges[1:-1].tolist()  # Skip first and last edge

        # Extract weights for this feature's bins
        feat_weights_float = w[weight_idx:weight_idx + n_bins]
        feat_weights_int = (feat_weights_float * weight_scale).astype(np.int64).tolist()

        feature_data = {
            "index": feat_idx,
            "name": feat_name,
            "n_bins": n_bins,
            "bin_edges": interior_edges,  # Only interior edges
            "weights_float": feat_weights_float.tolist(),
            "weights_int": feat_weights_int,
        }

        model_data["features"].append(feature_data)
        weight_idx += n_bins

        typer.echo(f"  {feat_name}: {n_bins} bins, weights range [{feat_weights_float.min():.4f}, {feat_weights_float.max():.4f}]")

    # Write to JSON
    with open(output_file, "w") as f:
        json.dump(model_data, f, indent=2)

    typer.echo(f"\nExported model to {output_file}")
    typer.echo(f"Weight scale factor: {weight_scale}")
    typer.echo("Ready to load into BPF maps!")


if __name__ == "__main__":
    app()

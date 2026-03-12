#!/usr/bin/env python3
"""
Learned Cache CLI - Unified command-line interface for cache eviction model training and export.

Commands:
    transform-logs:     Convert raw logs to CSV format
    train-ranker:       Train a pairwise ranker model on cache access traces
    export-model:       Export trained model to BPF-compatible JSON format
    train-and-export:   Train model and export to BPF (2-step: train → export)
    full-pipeline:      Complete workflow (3-step: transform → train → export)
"""

from pathlib import Path
from typing import Annotated

import typer

from learnedcache.core import run_transform_logs, run_train_ranker, run_export_model

app = typer.Typer(help="Learned Cache - Train and export cache eviction models")



@app.command()
def transform_logs(
    log_pattern: Annotated[str, typer.Option(help="Glob pattern for input log files")],
    verbose: Annotated[bool, typer.Option()] = False,
) -> None:
    """Transform raw log files to CSV format."""
    run_transform_logs(log_pattern, verbose=verbose)


@app.command()
def train_ranker(
    file_pattern: Annotated[str, typer.Option(help="Glob pattern for input CSV files")],
    output_dir: Annotated[Path, typer.Option(help="Directory to save model and artifacts")],
    discretize_cols: Annotated[list[str], typer.Option(help="Columns to discretize")] = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"],
    n_bins: Annotated[int, typer.Option(help="Number of bins for discretization")] = 10,
    max_epochs: Annotated[int, typer.Option(help="Maximum training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 256,
    sampling_multiplier: Annotated[float, typer.Option(help="Pair sampling multiplier")] = 1.0,
    random_state: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
    verbose: Annotated[bool, typer.Option()] = False,
) -> None:
    """Train a linear pairwise ranker model on cache access traces."""
    run_train_ranker(
        file_pattern=file_pattern,
        output_dir=output_dir,
        discretize_cols=discretize_cols,
        n_bins=n_bins,
        max_epochs=max_epochs,
        batch_size=batch_size,
        sampling_multiplier=sampling_multiplier,
        random_state=random_state,
        verbose=verbose,
    )


@app.command()
def export_model(
    model_dir: Annotated[Path, typer.Option(help="Directory containing trained model artifacts")],
    output_file: Annotated[Path, typer.Option(help="Output JSON file")] = Path("model_weights.json"),
    weight_scale: Annotated[int, typer.Option(help="Scale factor for quantizing weights")] = 10000,
    feature_names: Annotated[list[str], typer.Option(help="Feature names in BPF enum order")] = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"],
    verbose: Annotated[bool, typer.Option()] = False,
) -> None:
    """Export trained model to BPF-compatible JSON format."""
    try:
        run_export_model(
            model_dir=model_dir,
            output_file=output_file,
            weight_scale=weight_scale,
            feature_names=feature_names,
            verbose=verbose,
        )
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def train_and_export(
    file_pattern: Annotated[str, typer.Option(help="Glob pattern for input CSV files")],
    output_dir: Annotated[Path, typer.Option(help="Directory to save model and artifacts")],
    export_filename: Annotated[str, typer.Option(help="Filename for BPF export JSON")] = "model_weights.json",
    discretize_cols: Annotated[list[str], typer.Option(help="Columns to discretize")] = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"],
    n_bins: Annotated[int, typer.Option(help="Number of bins for discretization")] = 10,
    max_epochs: Annotated[int, typer.Option(help="Maximum training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 256,
    sampling_multiplier: Annotated[float, typer.Option(help="Pair sampling multiplier")] = 1.0,
    random_state: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
    weight_scale: Annotated[int, typer.Option(help="Scale factor for quantizing weights")] = 10000,
    verbose: Annotated[bool, typer.Option()] = False,
) -> None:
    """Train model and export to BPF format (2-step pipeline: train → export)."""

    output_dir = Path(output_dir)
    export_file = output_dir / export_filename

    typer.echo("=" * 80)
    typer.echo("STEP 1: TRAINING MODEL")
    typer.echo("=" * 80)
    run_train_ranker(
        file_pattern=file_pattern,
        output_dir=output_dir,
        discretize_cols=discretize_cols,
        n_bins=n_bins,
        max_epochs=max_epochs,
        batch_size=batch_size,
        sampling_multiplier=sampling_multiplier,
        random_state=random_state,
        verbose=verbose,
    )

    typer.echo("\n" + "=" * 80)
    typer.echo("STEP 2: EXPORTING MODEL TO BPF FORMAT")
    typer.echo("=" * 80)
    try:
        run_export_model(
            model_dir=output_dir,
            output_file=export_file,
            weight_scale=weight_scale,
            feature_names=discretize_cols,
            verbose=verbose,
        )
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("\n" + "=" * 80)
    typer.echo("✅ COMPLETE: Model trained and exported!")
    typer.echo("=" * 80)
    typer.echo(f"Model artifacts: {output_dir}")
    typer.echo(f"BPF export: {export_file}")


@app.command()
def full_pipeline(
    log_pattern: Annotated[str, typer.Option(help="Glob pattern for input log files")],
    output_dir: Annotated[Path, typer.Option(help="Directory to save model and artifacts")],
    export_filename: Annotated[str, typer.Option(help="Filename for BPF export JSON")] = "model_weights.json",
    discretize_cols: Annotated[list[str], typer.Option(help="Columns to discretize")] = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"],
    n_bins: Annotated[int, typer.Option(help="Number of bins for discretization")] = 10,
    max_epochs: Annotated[int, typer.Option(help="Maximum training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 256,
    sampling_multiplier: Annotated[float, typer.Option(help="Pair sampling multiplier")] = 1.0,
    random_state: Annotated[int, typer.Option(help="Random seed for reproducibility")] = 42,
    weight_scale: Annotated[int, typer.Option(help="Scale factor for quantizing weights")] = 10000,
    verbose: Annotated[bool, typer.Option()] = False,
) -> None:
    """Complete pipeline: transform logs → train model → export to BPF format (3-step)."""

    output_dir = Path(output_dir)
    export_file = output_dir / export_filename

    typer.echo("=" * 80)
    typer.echo("STEP 1: TRANSFORMING LOGS TO CSV")
    typer.echo("=" * 80)
    run_transform_logs(log_pattern, verbose=verbose)

    # Derive CSV pattern from log pattern
    csv_pattern = log_pattern.replace(".log", "_access.csv")

    typer.echo("\n" + "=" * 80)
    typer.echo("STEP 2: TRAINING MODEL")
    typer.echo("=" * 80)
    run_train_ranker(
        file_pattern=csv_pattern,
        output_dir=output_dir,
        discretize_cols=discretize_cols,
        n_bins=n_bins,
        max_epochs=max_epochs,
        batch_size=batch_size,
        sampling_multiplier=sampling_multiplier,
        random_state=random_state,
        verbose=verbose,
    )

    typer.echo("\n" + "=" * 80)
    typer.echo("STEP 3: EXPORTING MODEL TO BPF FORMAT")
    typer.echo("=" * 80)
    try:
        run_export_model(
            model_dir=output_dir,
            output_file=export_file,
            weight_scale=weight_scale,
            feature_names=discretize_cols,
            verbose=verbose,
        )
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("\n" + "=" * 80)
    typer.echo("✅ COMPLETE: Full pipeline finished!")
    typer.echo("=" * 80)
    typer.echo(f"Model artifacts: {output_dir}")
    typer.echo(f"BPF export: {export_file}")

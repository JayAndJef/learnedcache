#!/usr/bin/env python3
"""
Learned Cache CLI - unified interface for log transform, model training, and export.
"""

from pathlib import Path
from typing import Annotated

import typer

from learnedcache.core import (
    run_export_model,
    run_train_from_binary,
    run_train_ranker,
    run_transform_logs,
)

DEFAULT_DISCRETIZE_COLS = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"]
DERIVED_FEATURE_COL = "time_since_last_access_at_eviction"
DEFAULT_EXPORT_FEATURE_NAMES = [*DEFAULT_DISCRETIZE_COLS, DERIVED_FEATURE_COL]

app = typer.Typer(help="Learned Cache - train and export eviction-time pairwise rankers")

@app.command()
def transform_logs(
    log_pattern: Annotated[str, typer.Option(help="Glob pattern for input log files")],
    verbose: Annotated[bool, typer.Option()] = False,
) -> None:
    """Transform raw log files to CSV format."""
    run_transform_logs(log_pattern, verbose=verbose)

@app.command()
def train_ranker(
    access_pattern: Annotated[str, typer.Option(help="Glob pattern for access CSV files")],
    eviction_pattern: Annotated[str, typer.Option(help="Glob pattern for eviction CSV files")],
    output_dir: Annotated[Path, typer.Option(help="Directory to save model and artifacts")],
    discretize_cols: Annotated[
        list[str], typer.Option(help="Columns to discretize")
    ] = DEFAULT_DISCRETIZE_COLS,
    n_bins: Annotated[int, typer.Option(help="Number of bins for discretization")] = 10,
    max_epochs: Annotated[int, typer.Option(help="Maximum training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 256,
    pairs_per_event: Annotated[
        int, typer.Option(help="Sampled pair count per (trial_id, eviction_ts) event")
    ] = 512,
    max_pairs_total: Annotated[
        int | None, typer.Option(help="Optional cap on total sampled pairs")
    ] = None,
    max_page_groups_per_chunk: Annotated[
        int | None, typer.Option(help="Cap page groups per eviction chunk (None = all)")
    ] = None,
    pair_random_state: Annotated[
        int, typer.Option(help="Random seed for pair sampling")
    ] = 42,
) -> None:
    """Train a linear eviction-time pairwise-diff ranker on access+eviction traces."""
    run_train_ranker(
        access_pattern=access_pattern,
        eviction_pattern=eviction_pattern,
        output_dir=output_dir,
        discretize_cols=discretize_cols,
        n_bins=n_bins,
        max_epochs=max_epochs,
        batch_size=batch_size,
        pairs_per_event=pairs_per_event,
        max_pairs_total=max_pairs_total,
        max_page_groups_per_chunk=max_page_groups_per_chunk,
        random_state=pair_random_state,
        verbose=True,
    )

@app.command()
def export_model(
    model_dir: Annotated[Path, typer.Option(help="Directory containing model artifacts")],
    output_file: Annotated[Path, typer.Option(help="Output JSON file")] = Path("model_weights.json"),
    weight_scale: Annotated[int, typer.Option(help="Scale factor for quantizing weights")] = 10000,
    feature_names: Annotated[
        list[str], typer.Option(help="Feature names in BPF enum order")
    ] = DEFAULT_EXPORT_FEATURE_NAMES,
    verbose: Annotated[bool, typer.Option(help="Enable verbose output")] = False,
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
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

@app.command()
def train_and_export(
    access_pattern: Annotated[str, typer.Option(help="Glob pattern for access CSV files")],
    eviction_pattern: Annotated[str, typer.Option(help="Glob pattern for eviction CSV files")],
    output_dir: Annotated[Path, typer.Option(help="Directory to save model and artifacts")],
    export_filename: Annotated[
        str, typer.Option(help="Filename for BPF export JSON")
    ] = "model_weights.json",
    discretize_cols: Annotated[
        list[str], typer.Option(help="Columns to discretize")
    ] = DEFAULT_DISCRETIZE_COLS,
    n_bins: Annotated[int, typer.Option(help="Number of bins for discretization")] = 10,
    max_epochs: Annotated[int, typer.Option(help="Maximum training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 256,
    pairs_per_event: Annotated[
        int, typer.Option(help="Sampled pair count per (trial_id, eviction_ts) event")
    ] = 512,
    max_pairs_total: Annotated[
        int | None, typer.Option(help="Optional cap on total sampled pairs")
    ] = None,
    max_page_groups_per_chunk: Annotated[
        int | None, typer.Option(help="Cap page groups per eviction chunk (None = all)")
    ] = None,
    pair_random_state: Annotated[
        int, typer.Option(help="Random seed for pair sampling")
    ] = 42,
    weight_scale: Annotated[int, typer.Option(help="Scale factor for quantizing weights")] = 10000,
    verbose: Annotated[bool, typer.Option()] = False,
) -> None:
    """Train pairwise model and export to BPF format (train -> export)."""

    output_dir = Path(output_dir)
    export_file = output_dir / export_filename

    typer.echo("=" * 80)
    typer.echo("STEP 1: TRAINING PAIRWISE MODEL")
    typer.echo("=" * 80)
    train_result = run_train_ranker(
        access_pattern=access_pattern,
        eviction_pattern=eviction_pattern,
        output_dir=output_dir,
        discretize_cols=discretize_cols,
        n_bins=n_bins,
        max_epochs=max_epochs,
        batch_size=batch_size,
        pairs_per_event=pairs_per_event,
        max_pairs_total=max_pairs_total,
        max_page_groups_per_chunk=max_page_groups_per_chunk,
        random_state=pair_random_state,
        verbose=True,
    )

    export_feature_names = [*discretize_cols, DERIVED_FEATURE_COL]

    typer.echo("\n" + "=" * 80)
    typer.echo("STEP 2: EXPORTING MODEL TO BPF FORMAT")
    typer.echo("=" * 80)
    run_export_model(
        model_dir=output_dir,
        output_file=export_file,
        weight_scale=weight_scale,
        feature_names=export_feature_names,
        verbose=True,
        model=train_result["model"],
        discretizer=train_result["discretizer"],
    )

    typer.echo("\n" + "=" * 80)
    typer.echo("COMPLETE: Model trained and exported")
    typer.echo("=" * 80)
    typer.echo(f"Model artifacts: {output_dir}")
    typer.echo(f"BPF export: {export_file}")

@app.command()
def full_pipeline(
    log_pattern: Annotated[str, typer.Option(help="Glob pattern for input log files")],
    output_dir: Annotated[Path, typer.Option(help="Directory to save model and artifacts")],
    export_filename: Annotated[
        str, typer.Option(help="Filename for BPF export JSON")
    ] = "model_weights.json",
    discretize_cols: Annotated[
        list[str], typer.Option(help="Columns to discretize")
    ] = DEFAULT_DISCRETIZE_COLS,
    n_bins: Annotated[int, typer.Option(help="Number of bins for discretization")] = 10,
    max_epochs: Annotated[int, typer.Option(help="Maximum training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 256,
    pairs_per_event: Annotated[
        int, typer.Option(help="Sampled pair count per (trial_id, eviction_ts) event")
    ] = 512,
    max_pairs_total: Annotated[
        int | None, typer.Option(help="Optional cap on total sampled pairs")
    ] = None,
    max_page_groups_per_chunk: Annotated[
        int | None, typer.Option(help="Cap page groups per eviction chunk (None = all)")
    ] = None,
    pair_random_state: Annotated[
        int, typer.Option(help="Random seed for pair sampling")
    ] = 42,
    weight_scale: Annotated[int, typer.Option(help="Scale factor for quantizing weights")] = 10000,
    verbose: Annotated[bool, typer.Option()] = False,
) -> None:
    """Complete workflow: transform logs -> train pairwise model -> export model."""

    output_dir = Path(output_dir)
    export_file = output_dir / export_filename

    typer.echo("=" * 80)
    typer.echo("STEP 1: TRANSFORMING LOGS TO CSV")
    typer.echo("=" * 80)
    run_transform_logs(log_pattern, verbose=verbose)

    access_pattern = log_pattern.replace(".log", "_access.csv")
    eviction_pattern = log_pattern.replace(".log", "_eviction.csv")

    typer.echo("\n" + "=" * 80)
    typer.echo("STEP 2: TRAINING PAIRWISE MODEL")
    typer.echo("=" * 80)
    train_result = run_train_ranker(
        access_pattern=access_pattern,
        eviction_pattern=eviction_pattern,
        output_dir=output_dir,
        discretize_cols=discretize_cols,
        n_bins=n_bins,
        max_epochs=max_epochs,
        batch_size=batch_size,
        pairs_per_event=pairs_per_event,
        max_pairs_total=max_pairs_total,
        max_page_groups_per_chunk=max_page_groups_per_chunk,
        random_state=pair_random_state,
        verbose=True,
    )

    export_feature_names = [*discretize_cols, DERIVED_FEATURE_COL]

    typer.echo("\n" + "=" * 80)
    typer.echo("STEP 3: EXPORTING MODEL TO BPF FORMAT")
    typer.echo("=" * 80)
    run_export_model(
        model_dir=output_dir,
        output_file=export_file,
        weight_scale=weight_scale,
        feature_names=export_feature_names,
        verbose=True,
        model=train_result["model"],
        discretizer=train_result["discretizer"],
    )

    typer.echo("\n" + "=" * 80)
    typer.echo("COMPLETE: Full pipeline finished")
    typer.echo("=" * 80)
    typer.echo(f"Model artifacts: {output_dir}")
    typer.echo(f"BPF export: {export_file}")


@app.command()
def train_from_binary(
    data_dir: Annotated[Path, typer.Option(help="Base directory containing workload subdirectories")],
    output_dir: Annotated[Path, typer.Option(help="Base output directory")] = Path("./output"),
    workloads: Annotated[
        list[str] | None, typer.Option(help="Workloads to process (default: all)")
    ] = None,
    discretize_cols: Annotated[
        list[str], typer.Option(help="Columns to discretize")
    ] = DEFAULT_DISCRETIZE_COLS,
    n_bins: Annotated[int, typer.Option(help="Number of bins for discretization")] = 10,
    max_epochs: Annotated[int, typer.Option(help="Maximum training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 256,
    pairs_per_event: Annotated[
        int, typer.Option(help="Sampled pair count per (trial_id, eviction_ts) event")
    ] = 512,
    max_pairs_total: Annotated[
        int | None, typer.Option(help="Optional cap on total sampled pairs")
    ] = None,
    max_page_groups_per_chunk: Annotated[
        int | None, typer.Option(help="Cap page groups per eviction chunk (None = all)")
    ] = None,
    pair_random_state: Annotated[
        int, typer.Option(help="Random seed for pair sampling")
    ] = 42,
    weight_scale: Annotated[int, typer.Option(help="Scale factor for quantizing weights")] = 10000,
    verbose: Annotated[bool, typer.Option()] = False,
) -> None:
    """Train pairwise ranker from binary cache trace logs across workloads."""
    results = run_train_from_binary(
        data_dir=data_dir,
        output_dir=output_dir,
        workloads=workloads,
        discretize_cols=discretize_cols,
        n_bins=n_bins,
        max_epochs=max_epochs,
        batch_size=batch_size,
        pairs_per_event=pairs_per_event,
        max_pairs_total=max_pairs_total,
        max_page_groups_per_chunk=max_page_groups_per_chunk,
        pair_random_state=pair_random_state,
        weight_scale=weight_scale,
        verbose=verbose,
    )
    typer.echo("\n" + "=" * 80)
    typer.echo("All workloads complete.")
    typer.echo("=" * 80)
    for name, result in results.items():
        typer.echo(f"  {name}: accuracy={result.get('pairwise_accuracy', 'N/A')}")
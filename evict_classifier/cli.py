"""CLI for the eviction-time binary reuse classifier.

Standalone Typer app -- run via ``python -m evict_classifier``. Independent of
``learnedcache``'s CLI.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .train import train

app = typer.Typer(
    add_completion=False,
    help="Train an eviction-time binary reuse classifier for cache_ext_fifo_ml_protect.",
)

# Binary-log timestamps are nanoseconds (bpf_ktime_get_ns); the horizon is given
# in seconds for convenience and converted here.
_NS_PER_SECOND = 1_000_000_000.0


@app.callback()
def _main() -> None:
    """Eviction-time binary reuse classifier for cache_ext_fifo_ml_protect."""


@app.command("train")
def train_cmd(
    data_dir: Path = typer.Option(
        ..., "--data-dir", help="Dir of <workload>/iter_*/ binary logs.", exists=True
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Output dir (one subdir per workload)."
    ),
    workload: list[str] = typer.Option(
        None, "--workload", help="Restrict to these workloads (repeatable)."
    ),
    horizon_seconds: float = typer.Option(
        None, "--horizon",
        help="Reuse horizon H in seconds (label = reused within H). Omit to derive "
        "H from the measured cache turnover (capacity / insertion rate).",
    ),
    capacity_pages: int = typer.Option(
        None, "--capacity",
        help="Cache capacity in pages for the turnover estimate (auto-horizon mode "
        "only); omit to estimate it as the insertion count before the first eviction.",
    ),
    target_rows: int = typer.Option(
        2_000_000, "--target-rows", help="Reservoir budget (total training rows)."
    ),
    balanced: bool = typer.Option(
        True, "--balance/--no-balance",
        help="Class-balanced reservoir vs natural ratio + class weights.",
    ),
    n_bins: int = typer.Option(10, "--n-bins", help="Quantile bins per feature."),
    max_epochs: int = typer.Option(50, "--max-epochs"),
    batch_size: int = typer.Option(4096, "--batch-size"),
    threshold: float = typer.Option(
        0.0, "--threshold", help="Decision threshold in logit units (0 => P>0.5)."
    ),
    weight_scale: int = typer.Option(10000, "--weight-scale"),
    residency_cap_seconds: float = typer.Option(
        None, "--residency-cap",
        help="Exclude candidates idle longer than this (seconds) as implausibly "
        "still-resident; roughly the cache turnover time. Omit to follow the "
        "horizon (manual or measured); 0 disables.",
    ),
    holdout_frac: float = typer.Option(
        0.2, "--holdout-frac", help="Temporal-holdout fraction per workload."
    ),
    random_state: int = typer.Option(42, "--seed"),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Train + export a classifier per workload under DATA_DIR."""
    results = train(
        data_dir=data_dir,
        output_dir=output_dir,
        workloads=workload or None,
        horizon=(
            horizon_seconds * _NS_PER_SECOND if horizon_seconds is not None else None
        ),
        capacity_pages=capacity_pages,
        target_rows=target_rows,
        balanced=balanced,
        n_bins=n_bins,
        max_epochs=max_epochs,
        batch_size=batch_size,
        threshold=threshold,
        weight_scale=weight_scale,
        residency_cap=(
            residency_cap_seconds * _NS_PER_SECOND
            if residency_cap_seconds is not None
            else None
        ),
        holdout_frac=holdout_frac,
        random_state=random_state,
        verbose=not quiet,
    )
    print("\n=== summary ===")
    for name, res in results.items():
        m = res.get("metrics") or {}
        auc = m.get("auc")
        auc_str = f"{auc:.4f}" if isinstance(auc, float) else "n/a"
        print(
            f"{name}: {res['n_train_rows']:,} rows, "
            f"pos_rate {res['true_positive_rate']:.3f}, holdout AUC {auc_str} "
            f"-> {res['output_dir']}/model_weights.json"
        )


if __name__ == "__main__":
    app()

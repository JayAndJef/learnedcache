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
        10.0, "--horizon", help="Reuse horizon H in seconds (label = reused within H)."
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
        30.0, "--residency-cap",
        help="Exclude candidates idle longer than this (seconds) as implausibly "
        "still-resident; roughly the cache turnover time. 0 disables.",
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
        horizon=horizon_seconds * _NS_PER_SECOND,
        target_rows=target_rows,
        balanced=balanced,
        n_bins=n_bins,
        max_epochs=max_epochs,
        batch_size=batch_size,
        threshold=threshold,
        weight_scale=weight_scale,
        residency_cap=(
            residency_cap_seconds * _NS_PER_SECOND if residency_cap_seconds > 0 else None
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


@app.command("simulate")
def simulate_cmd(
    data_dir: Path = typer.Option(
        ..., "--data-dir", help="Dir of <workload>/iter_*/ binary logs.", exists=True
    ),
    model_root: Path = typer.Option(
        ..., "--model-root",
        help="Dir containing <workload>/model_weights.json (train output dir).",
    ),
    output_dir: Path = typer.Option(..., "--output-dir"),
    workload: list[str] = typer.Option(
        None, "--workload", help="Restrict to these workloads (repeatable)."
    ),
    capacities: str = typer.Option(
        "262144,524288,1048576",
        "--capacities", help="Comma-separated cache sizes in PAGES (4 KiB each).",
    ),
    policies: str = typer.Option(
        "fifo,lru,belady,protect", "--policies", help="Comma-separated policy list."
    ),
) -> None:
    """Trace-driven hit-rate simulation: baselines vs Belady vs the protect policy."""
    import json as _json

    from .loading import discover_workloads_and_iters
    from .simulate import simulate_workload

    caps = [int(c) for c in capacities.split(",") if c]
    pols = [p.strip() for p in policies.split(",") if p.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}
    for name, iter_dirs in discover_workloads_and_iters(data_dir, workload or None).items():
        print(f"=== {name} ===")
        model_file = model_root / name / "model_weights.json"
        if "protect" in pols and not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        all_results[name] = simulate_workload(
            iter_dirs[0], model_file, caps, policies=pols
        )

    out_file = output_dir / "simulation_results.json"
    out_file.write_text(_json.dumps(all_results, indent=2))
    print(f"\nResults -> {out_file}")

    _plot_simulation(all_results, output_dir)


def _plot_simulation(all_results: dict[str, list[dict]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, results in all_results.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for metric, ax in zip(("hit_rate", "tail_hit_rate"), axes):
            by_policy: dict[str, list[tuple[int, float]]] = {}
            for r in results:
                by_policy.setdefault(r["policy"], []).append(
                    (r["capacity_pages"], r[metric])
                )
            for policy, pts in by_policy.items():
                pts.sort()
                gb = [c * 4096 / 2**30 for c, _ in pts]
                ax.plot(gb, [v for _, v in pts], marker="o", label=policy)
            ax.set_xlabel("Cache capacity (GiB)")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_title(f"{name} — {metric.replace('_', ' ')}")
            ax.grid(True, alpha=0.3)
            ax.legend()
        plt.tight_layout()
        path = output_dir / f"hit_rate_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Plot -> {path}")


if __name__ == "__main__":
    app()

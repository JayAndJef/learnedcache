"""Training orchestration for the eviction-time binary reuse classifier.

Per workload: one streaming pass fills bounded reservoirs
(``sampling.collect_workload_sample``), the discretizer is fit on the subsample,
the reservoir is one-hot encoded once, and the linear classifier is trained
**in-memory** for many fast epochs. Evaluation uses the temporal-holdout tail at
the natural class ratio; artifacts (``model.keras``, ``discretizer.pkl``,
``model_weights.json``, ``eval_report.txt``) are written per workload.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import keras
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .export import DEFAULT_FEATURE_NAMES, export_classifier
from .loading import (
    build_pairs_from_binary,
    discover_workloads_and_iters,
    estimate_turnover,
)
from .models import build_binary_classifier
from .plots import save_weight_plot
from .preprocess import (
    fit_discretizer_from_sample,
    one_hot_encode_features,
    transform_discretizer_batch,
)
from .sampling import WorkloadSample, collect_workload_sample

# Eviction-time raw features; the derived time-since-access feature is appended
# by the sampler, giving the 9-feature vector the BPF policy also computes.
DEFAULT_DISCRETIZE_COLS = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"]


def _evaluate(
    model: keras.Model, x_eval: np.ndarray, y_eval: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Holdout metrics at the deployed decision threshold (logit units)."""
    prob = model.predict(x_eval, verbose=0).ravel()
    prob_threshold = float(1.0 / (1.0 + np.exp(-threshold)))
    pred = (prob > prob_threshold).astype(np.int32)
    y_true = y_eval.astype(np.int32)

    auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "n_eval": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "auc": auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "prob_threshold": prob_threshold,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _write_eval_report(path: Path, name: str, metrics: dict[str, Any]) -> None:
    if "confusion" not in metrics:
        path.write_text(f"{name}: no holdout set (single eviction event or empty).\n")
        return
    c = metrics["confusion"]
    path.write_text(
        f"Eviction-time binary reuse classifier -- {name}\n"
        f"{'=' * 48}\n"
        f"horizon ({metrics.get('horizon_source', 'manual')}) : "
        f"{metrics['horizon_ns'] / 1e9:.1f}s\n"
        f"holdout rows         : {metrics['n_eval']:,}\n"
        f"positive rate        : {metrics['positive_rate']:.4f}\n"
        f"AUC                  : {metrics['auc']:.4f}\n"
        f"precision / recall   : {metrics['precision']:.4f} / {metrics['recall']:.4f}\n"
        f"F1                   : {metrics['f1']:.4f}\n"
        f"decision prob thresh : {metrics['prob_threshold']:.4f}\n"
        f"confusion (tn fp fn tp): {c['tn']} {c['fp']} {c['fn']} {c['tp']}\n"
    )


def train_workload(
    name: str,
    iter_dirs: list[Path],
    output_dir: str | Path,
    *,
    discretize_cols: list[str] = DEFAULT_DISCRETIZE_COLS,
    horizon: float | None,
    capacity_pages: int | None = None,
    target_rows: int = 2_000_000,
    balanced: bool = True,
    n_bins: int = 10,
    max_epochs: int = 50,
    batch_size: int = 4096,
    threshold: float = 0.0,
    weight_scale: int = 10000,
    disc_sample_size: int = 200_000,
    eval_rows: int = 300_000,
    holdout_frac: float = 0.2,
    residency_cap: float | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train, evaluate, and export one workload's classifier.

    ``horizon`` and ``residency_cap`` are in raw timestamp units (nanoseconds
    for the binary logs); the CLI exposes them in seconds and converts. Labels
    are positive when the next reuse occurs within ``horizon`` of the eviction
    moment; candidates idle longer than ``residency_cap`` are excluded as
    implausibly still-resident (see ``sampling._interval_bounds``).

    When ``horizon`` is None it is derived from the measured cache turnover
    (fill/rotation time = capacity / insertion rate -- see
    ``loading.estimate_turnover``); ``capacity_pages`` optionally overrides the
    insertions-until-first-eviction capacity estimate. A None ``residency_cap``
    follows the horizon (they are the same physical quantity); pass 0 to
    disable the cap entirely.
    """
    out = Path(output_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    horizon_provenance: dict[str, Any] = {"horizon_source": "manual"}
    if horizon is None:
        estimates = [estimate_turnover(d, capacity_pages) for d in iter_dirs]
        turnover = float(np.mean([e.turnover_ns for e in estimates]))
        # Right-censoring drops events later than last_access - H, and evictions
        # only start after the cold-start fill (~one turnover in). If H is not
        # well inside [first_eviction, last_access], no labelable events remain
        # (a short trace at a big cgroup barely turns over once) -- clamp to
        # half that window.
        windows = [e.label_window_ns for e in estimates if e.label_window_ns]
        horizon = min(turnover, min(windows) / 2) if windows else turnover
        e = estimates[0]
        horizon_provenance = {
            "horizon_source": "auto",
            "turnover_ns": turnover,
            "horizon_clamped": horizon < turnover,
            "capacity_pages": e.capacity_pages,
            "capacity_estimated": e.capacity_estimated,
            "insertion_rate_per_s": e.insertion_rate_per_s,
            "active_seconds": e.active_seconds,
            "n_insertions": e.n_insertions,
        }
        if verbose:
            cap_src = "estimated" if e.capacity_estimated else "given"
            clamp_note = (
                f" (clamped to {horizon / 1e9:.1f}s -- half the observable "
                f"first-eviction-to-last-access window)" if horizon < turnover else ""
            )
            print(
                f"{name}: rate {e.insertion_rate_per_s:,.0f}/s over "
                f"{e.active_seconds:.1f}s active, capacity ~{e.capacity_pages:,} "
                f"pages ({cap_src}) -> turnover ~{turnover / 1e9:.1f}s{clamp_note}"
            )
    if residency_cap is None:
        residency_cap = horizon
    elif residency_cap <= 0:
        residency_cap = None  # explicitly disabled

    if verbose:
        print(f"\n=== {name}: collecting training sample ===")

    sample: WorkloadSample = collect_workload_sample(
        build_pairs_from_binary(iter_dirs),
        discretize_cols,
        horizon=horizon,
        target_rows=target_rows,
        balanced=balanced,
        disc_sample_size=disc_sample_size,
        eval_rows=eval_rows,
        holdout_frac=holdout_frac,
        residency_cap=residency_cap,
        random_state=random_state,
        verbose=verbose,
    )
    if len(sample.x_train) == 0:
        raise ValueError(f"{name}: no training rows collected (check data / horizon).")

    # ── Discretize + one-hot the reservoir, once ──
    discretizer = fit_discretizer_from_sample(
        sample.disc_sample, n_bins=n_bins, random_state=random_state
    )
    n_feat = sample.x_train.shape[1]
    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(n_feat)]
    n_encoded = sum(n_bins_list)

    x_train = one_hot_encode_features(
        transform_discretizer_batch(sample.x_train, discretizer), n_bins_list
    )
    validation_data = None
    if len(sample.x_eval) > 0:
        x_eval = one_hot_encode_features(
            transform_discretizer_batch(sample.x_eval, discretizer), n_bins_list
        )
        validation_data = (x_eval, sample.y_eval)

    if verbose:
        print(f"=== {name}: training ({n_encoded} one-hot features) ===")

    # ── In-memory training ──
    model = build_binary_classifier(n_encoded)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    callbacks = []
    if validation_data is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_auc", mode="max", patience=5, restore_best_weights=True
            )
        )
    model.fit(
        x_train,
        sample.y_train,
        validation_data=validation_data,
        epochs=max_epochs,
        batch_size=batch_size,
        class_weight=(None if balanced else sample.class_weight),
        callbacks=callbacks,
        verbose=(2 if verbose else 0),
    )

    metrics: dict[str, Any] = {
        "horizon_ns": float(horizon),
        "residency_cap_ns": float(residency_cap) if residency_cap is not None else None,
        **horizon_provenance,
    }
    if validation_data is not None:
        metrics |= _evaluate(model, validation_data[0], validation_data[1], threshold)
        if verbose:
            print(
                f"=== {name}: holdout AUC {metrics['auc']:.4f} "
                f"precision {metrics['precision']:.3f} recall {metrics['recall']:.3f} ==="
            )

    # ── Persist artifacts ──
    model.save(out / "model.keras")
    with (out / "discretizer.pkl").open("wb") as f:
        pickle.dump(discretizer, f)
    export_classifier(
        out / "model_weights.json",
        model,
        discretizer,
        weight_scale=weight_scale,
        threshold=threshold,
        verbose=verbose,
    )
    save_weight_plot(model, DEFAULT_FEATURE_NAMES, n_bins_list, out / "feature_importance.png")
    _write_eval_report(out / "eval_report.txt", name, metrics)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return {
        "workload": name,
        "output_dir": str(out),
        "n_train_rows": int(len(sample.x_train)),
        "n_encoded_features": n_encoded,
        "true_positive_rate": (
            sample.n_pos_seen / (sample.n_pos_seen + sample.n_neg_seen)
            if (sample.n_pos_seen + sample.n_neg_seen)
            else 0.0
        ),
        "metrics": metrics,
    }


def train(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    workloads: list[str] | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> dict[str, dict[str, Any]]:
    """Train one classifier per discovered workload under *data_dir*."""
    wl_map = discover_workloads_and_iters(data_dir, workloads)
    results: dict[str, dict[str, Any]] = {}
    for name, iter_dirs in wl_map.items():
        results[name] = train_workload(
            name, iter_dirs, output_dir, verbose=verbose, **kwargs
        )
    return results

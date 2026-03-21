from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc as sklearn_auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)

plt.style.use("seaborn-v0_8-paper")
sns.set_palette("pastel")
plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["font.family"] = "Courier New"
plt.rcParams["font.size"] = 30
plt.rcParams["axes.labelsize"] = 30
plt.rcParams["axes.titlesize"] = 40
plt.rcParams["xtick.labelsize"] = 30
plt.rcParams["ytick.labelsize"] = 30
plt.rcParams["legend.fontsize"] = 30
plt.rcParams["figure.dpi"] = 300

def save_training_visualizations(
    history: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    primary_metric: float,
    output_dir: Path,
) -> None:
    """Save pairwise training curves, ROC curve, and confusion matrix as separate images."""
    train_metric = history.history.get("accuracy", [])
    val_metric = history.history.get("val_accuracy", [])
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    train_auc_hist = history.history.get("auc", [])
    val_auc_hist = history.history.get("val_auc", [])

    y_pred_prob = np.asarray(y_pred).ravel()
    y_pred_cls = (y_pred_prob >= 0.5).astype(int)
    y_true_cls = np.asarray(y_true).astype(int)
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])

    fpr, tpr, _ = roc_curve(y_true_cls, y_pred_prob)
    roc_auc = sklearn_auc(fpr, tpr)

    rc = {
        "font.size": 26,
        "axes.labelsize": 26,
        "axes.titlesize": 32,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
    }

    # ── Pairwise Ranking Accuracy ─────────────────────────────────────────
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(10, 10))
        if len(train_metric) > 0:
            ax.plot(train_metric, label="Train Acc")
        if len(val_metric) > 0:
            ax.plot(val_metric, label="Val Acc")
        ax.set_title("Accuracy over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        if len(train_metric) > 0 or len(val_metric) > 0:
            ax.legend()
        ax.grid(True)
        plt.tight_layout()
        p = output_dir / "accuracy.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"Accuracy curve → {p}")

    # ── Live AUC per Epoch ────────────────────────────────────────────────
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(10, 10))
        if len(train_auc_hist) > 0:
            ax.plot(train_auc_hist, label="Train AUC")
        if len(val_auc_hist) > 0:
            ax.plot(val_auc_hist, label="Val AUC")
        ax.set_title("AUC over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")
        if len(train_auc_hist) > 0 or len(val_auc_hist) > 0:
            ax.legend()
        ax.grid(True)
        plt.tight_layout()
        p = output_dir / "live_auc.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"Live AUC curve → {p}")

    # ── Pairwise BCE Loss ─────────────────────────────────────────────────
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(10, 10))
        if len(train_loss) > 0:
            ax.plot(train_loss, label="Train Loss")
        if len(val_loss) > 0:
            ax.plot(val_loss, label="Val Loss")
        ax.set_title("Loss over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        if len(train_loss) > 0 or len(val_loss) > 0:
            ax.legend()
        ax.grid(True)
        plt.tight_layout()
        p = output_dir / "loss.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"Loss curve → {p}")

    # ── ROC Curve ─────────────────────────────────────────────────────────
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
        ax.set_title(f"ROC Curve  —  AUC: {roc_auc:.4f}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower right")
        ax.grid(True)
        plt.tight_layout()
        p = output_dir / "roc_curve.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"ROC curve → {p}")

    # ── Confusion Matrix ──────────────────────────────────────────────────
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["B sooner", "A sooner"],
            yticklabels=["B sooner", "A sooner"],
        )
        ax.set_title(f"Confusion Matrix\nAcc: {primary_metric:.4f}  AUC: {roc_auc:.4f}")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        p = output_dir / "confusion_matrix.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"Confusion matrix → {p}")

_ABBREV_SKIP = {"since", "at", "from", "of", "in", "the", "a", "an", "access", "by", "to", "for"}

def _col_abbrev(name: str) -> str:
    """Shorten a column name to initials, skipping common connector words.

    Single-word / already-short names (no underscores) are returned as-is.
    Example: 'time_since_last_access_at_eviction' -> 'tle'
    """
    parts = name.split("_")
    if len(parts) == 1:
        return name
    return "".join(p[0] for p in parts if p not in _ABBREV_SKIP)

def save_evaluation_report(
    model: Any,
    x_eval_full: np.ndarray,
    column_names: list[str],
    n_bins_list: list[int],
    output_dir: Path,
    access_pattern: str = "",
    eviction_pattern: str = "",
    n_rows: int = 0,
    n_train_rows: int = 0,
    n_test_rows: int = 0,
    epochs_trained: int = 0,
    objective: str = "pairwise_diff",
    pairwise_accuracy: float = 0.0,
    n_train_pairs: int = 0,
    n_test_pairs: int = 0,
    train_pair_stats: dict[str, int] | None = None,
    test_pair_stats: dict[str, int] | None = None,
    y_true: np.ndarray | None = None,
) -> None:
    """Save pairwise evaluation report with feature importance and sample scores.

    Produces a human-readable eval_report.txt that matches the format used in
    outputs/createdelete-swing/eval_report.txt.
    """
    # Learned weight vector (linear, before sigmoid)
    weights = model.get_layer("ranking_weight").get_weights()[0].ravel()

    # Build feature bin names for plotting
    feature_names: list[str] = []
    for col, n_bins in zip(column_names, n_bins_list):
        abbrev = _col_abbrev(col)
        for bin_idx in range(n_bins):
            feature_names.append(f"{abbrev}_bin{bin_idx}")

    # Feature importance plot
    fig_w, ax_w = plt.subplots(figsize=(25, 10))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in weights]
    ax_w.bar(range(len(weights)), weights, color=colors)
    ax_w.set_xticks(range(len(weights)))
    ax_w.set_xticklabels(feature_names, rotation=90, fontsize=14)
    ax_w.set_ylabel("Weight")
    ax_w.set_title("Learned Pairwise Ranking Weights")
    ax_w.axhline(y=0, color="black", linewidth=0.5)
    ax_w.grid(True, alpha=0.3)

    plt.tight_layout()
    importance_path = output_dir / "feature_importance.png"
    fig_w.savefig(importance_path)
    plt.close(fig_w)
    print(f"Feature importance → {importance_path}")

    sample_items = x_eval_full[:10]
    if len(sample_items) > 0:
        raw_sample_scores = np.dot(sample_items, weights)
        probs_sample = 1.0 / (1.0 + np.exp(-raw_sample_scores))
    else:
        raw_sample_scores = np.array([])
        probs_sample = np.array([])

    roc_auc_score: float | None = None
    if y_true is not None and len(x_eval_full) > 0:
        raw_full_scores = np.dot(x_eval_full, weights)
        probs_full = 1.0 / (1.0 + np.exp(-raw_full_scores))
        preds_full = (probs_full >= 0.5).astype(int)
        try:
            fpr_r, tpr_r, _ = roc_curve(y_true.astype(int), probs_full)
            roc_auc_score = float(sklearn_auc(fpr_r, tpr_r))
        except Exception:
            roc_auc_score = None
        try:
            cls_report_str = classification_report(
                y_true,
                preds_full,
                target_names=["B reused sooner", "A reused sooner"],
                digits=2,
            )
        except Exception:
            cls_report_str = "Classification report could not be computed.\n"
    else:
        cls_report_str = "No true labels supplied for classification report.\n"

    order = np.argsort(raw_sample_scores) if len(raw_sample_scores) > 0 else np.array([], dtype=int)

    eval_path = output_dir / "eval_report.txt"
    with eval_path.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Pairwise Ranker Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"File pattern: {access_pattern}\n")
        f.write(f"Total rows loaded: {n_rows}\n")
        f.write(f"Training pairs: {n_train_pairs}\n")
        f.write(f"Test pairs: {n_test_pairs}\n")
        f.write(f"Epochs trained: {epochs_trained}\n\n")

        f.write(f"Pairwise Test Accuracy: {pairwise_accuracy:.4f}\n")
        if roc_auc_score is not None:
            f.write(f"Pairwise Test AUC:      {roc_auc_score:.4f}\n")
        f.write("\n")

        f.write("Classification Report:\n")
        f.write("                 precision    recall  f1-score   support\n\n")
        if "No true labels" in cls_report_str:
            f.write(cls_report_str + "\n")
        else:
            f.write(cls_report_str + "\n")

        f.write(f"Weight vector shape: {weights.shape}\n")
        f.write(f"Weight vector: {weights.tolist()}\n\n")

        f.write("Sample item scores (higher = reused sooner = keep):\n")
        for idx, score in enumerate(raw_sample_scores):
            f.write(f"  Item {idx}: score = {float(score):.4f}\n")
        f.write("\n")

        evict_items = ", ".join(f"np.int64({int(x)})" for x in order)
        f.write("Eviction order (first to evict -> last):\n")
        f.write(f"  [{evict_items}]\n\n")

    print(f"Evaluation report → {eval_path}")

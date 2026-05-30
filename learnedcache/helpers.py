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

def save_evaluation_outputs(
    history: Any,
    weights: np.ndarray,
    x_eval_full: np.ndarray,
    column_names: list[str],
    n_bins_list: list[int],
    output_dir: Path,
    y_true: np.ndarray | None = None,
    y_pred_prob: np.ndarray | None = None,
    pairwise_accuracy: float = 0.0,
    access_pattern: str = "",
    n_rows: int = 0,
    n_train_rows: int = 0,
    n_test_rows: int = 0,
    epochs_trained: int = 0,
    n_train_pairs: int = 0,
    n_test_pairs: int = 0,
    train_pair_stats: dict[str, int] | None = None,
    test_pair_stats: dict[str, int] | None = None,
) -> None:
    """Save all evaluation outputs: training curves (6 PNGs) + eval report (txt).

    Consolidates the former save_training_visualizations and save_evaluation_report
    into a single pass. Scores are computed once from weights and reused across
    all outputs (ROC curve, confusion matrix, sample scores, eviction order).
    """
    # ── Feature bin names (needed for feature importance plot) ──────────────
    feature_names: list[str] = []
    for col, n_bins in zip(column_names, n_bins_list):
        abbrev = _col_abbrev(col)
        for bin_idx in range(n_bins):
            feature_names.append(f"{abbrev}_bin{bin_idx}")

    # ── Single score computation ────────────────────────────────────────────
    x_eval_full = np.asarray(x_eval_full, dtype=np.float32)
    has_data = len(x_eval_full) > 0

    raw_scores: np.ndarray | None = None
    raw_sample_scores = np.array([])
    roc_auc_score: float | None = None
    cls_report_str = "No true labels supplied for classification report.\n"
    fpr: np.ndarray | None = None
    tpr: np.ndarray | None = None
    cm = None

    if has_data:
        raw_scores = np.dot(x_eval_full, weights)
        del x_eval_full  # free one-hot matrix (largest object)

        # Sample scores: take a *copy* of first 10 entries so the slice
        # does not keep the parent score vector alive as a view.
        raw_sample_scores = raw_scores[:10].copy()

        # Use pre-computed probabilities from caller when available
        # (avoids recomputing sigmoid(model.predict() == sigmoid(dot(x, w))).
        # Falls back to computing from raw scores for backward compat.
        if y_pred_prob is not None:
            probs = np.asarray(y_pred_prob, dtype=np.float32).ravel()
        else:
            probs = np.float32(1.0) / (np.float32(1.0) + np.exp(-raw_scores))

        # Sign-based predictions: sigmoid(x) >= 0.5 iff x >= 0
        preds = (raw_scores >= 0).astype(np.int8)

        # ROC curve + classification report (free intermediates after each step)
        if y_true is not None and len(probs) > 0:
            y_true_cls = np.asarray(y_true, dtype=np.int8)
            try:
                fpr, tpr, _ = roc_curve(y_true_cls, probs)
                roc_auc_score = float(sklearn_auc(fpr, tpr))
            except Exception:
                pass
            del probs

            try:
                cls_report_str = classification_report(
                    y_true_cls, preds,
                    target_names=["B reused sooner", "A reused sooner"],
                    digits=2,
                )
            except Exception:
                cls_report_str = "Classification report could not be computed.\n"

            try:
                cm = confusion_matrix(y_true_cls, preds, labels=[0, 1])
            except Exception:
                pass
            del preds

        del raw_scores

    # ═════════════════════════════════════════════════════════════════════════
    #  Save 6 PNGs
    # ═════════════════════════════════════════════════════════════════════════
    train_metric = history.history.get("accuracy", [])
    val_metric = history.history.get("val_accuracy", [])
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    train_auc_hist = history.history.get("auc", [])
    val_auc_hist = history.history.get("val_auc", [])

    rc = {
        "font.size": 26,
        "axes.labelsize": 26,
        "axes.titlesize": 32,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
    }

    # ── Accuracy over Epochs ─────────────────────────────────────────────
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

    # ── AUC over Epochs ──────────────────────────────────────────────────
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

    # ── Loss over Epochs ─────────────────────────────────────────────────
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

    # ── ROC Curve ────────────────────────────────────────────────────────
    if fpr is not None and tpr is not None and roc_auc_score is not None:
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score:.4f})", linewidth=2)
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Random")
            ax.set_title(f"ROC Curve  —  AUC: {roc_auc_score:.4f}")
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

    # ── Confusion Matrix ─────────────────────────────────────────────────
    if cm is not None:
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
            title = f"Confusion Matrix\nAcc: {pairwise_accuracy:.4f}"
            if roc_auc_score is not None:
                title += f"  AUC: {roc_auc_score:.4f}"
            ax.set_title(title)
            ax.set_ylabel("True")
            ax.set_xlabel("Predicted")
            plt.tight_layout()
            p = output_dir / "confusion_matrix.png"
            fig.savefig(p)
            plt.close(fig)
            print(f"Confusion matrix → {p}")

    # ── Learned Pairwise Ranking Weights ─────────────────────────────────
    with plt.rc_context(rc):
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
        p = output_dir / "feature_importance.png"
        fig_w.savefig(p)
        plt.close(fig_w)
        print(f"Feature importance → {p}")

    # ═════════════════════════════════════════════════════════════════════════
    #  Save eval_report.txt
    # ═════════════════════════════════════════════════════════════════════════
    report_path = output_dir / "eval_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Pairwise Ranker Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"File pattern: {access_pattern}\n")
        f.write(f"Total rows loaded: {n_rows}\n")
        f.write(f"Train rows: {n_train_rows}\n")
        f.write(f"Test rows: {n_test_rows}\n")
        f.write(f"Training pairs: {n_train_pairs}\n")
        f.write(f"Test pairs: {n_test_pairs}\n")
        f.write(f"Epochs trained: {epochs_trained}\n\n")

        f.write(f"Pairwise Test Accuracy: {pairwise_accuracy:.4f}\n")
        if roc_auc_score is not None:
            f.write(f"Pairwise Test AUC:      {roc_auc_score:.4f}\n")
        f.write("\n")

        f.write("Classification Report:\n")
        f.write("                 precision    recall  f1-score   support\n\n")
        f.write(cls_report_str + "\n")

        f.write(f"Weight vector shape: {weights.shape}\n")
        f.write(f"Weight vector: {weights.tolist()}\n\n")

        f.write("Sample item scores (higher = reused sooner = keep):\n")
        for idx, score in enumerate(raw_sample_scores):
            f.write(f"  Item {idx}: score = {float(score):.4f}\n")
        f.write("\n")

        if len(raw_sample_scores) > 0:
            order = np.argsort(raw_sample_scores)
            evict_items = ", ".join(f"np.int64({int(x)})" for x in order)
            f.write("Eviction order (first to evict -> last):\n")
            f.write(f"  [{evict_items}]\n\n")

    print(f"Evaluation report → {report_path}")

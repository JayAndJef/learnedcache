from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

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
    """Save pairwise training curves and confusion matrix."""
    train_metric = history.history.get("accuracy", [])
    val_metric = history.history.get("val_accuracy", [])
    train_loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    y_pred_cls = (np.asarray(y_pred).ravel() >= 0.5).astype(int)
    y_true_cls = np.asarray(y_true).astype(int)
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])

    with plt.rc_context(
        {
            "font.size": 20,
            "axes.labelsize": 20,
            "axes.titlesize": 24,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }
    ):
        fig = plt.figure(figsize=(16, 5))

        ax1 = plt.subplot(1, 3, 1)
        if len(train_metric) > 0:
            ax1.plot(train_metric, label="Train Acc")
        if len(val_metric) > 0:
            ax1.plot(val_metric, label="Val Acc")
        ax1.set_title("Pairwise Ranking Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        if len(train_metric) > 0 or len(val_metric) > 0:
            ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(1, 3, 2)
        if len(train_loss) > 0:
            ax2.plot(train_loss, label="Train Loss")
        if len(val_loss) > 0:
            ax2.plot(val_loss, label="Val Loss")
        ax2.set_title("Pairwise BCE Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        if len(train_loss) > 0 or len(val_loss) > 0:
            ax2.legend()
        ax2.grid(True)

        ax3 = plt.subplot(1, 3, 3)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax3,
            xticklabels=["B sooner", "A sooner"],
            yticklabels=["B sooner", "A sooner"],
        )
        ax3.set_title(f"Confusion Matrix\nAccuracy: {primary_metric:.4f}")
        ax3.set_ylabel("True")
        ax3.set_xlabel("Predicted")

        plt.tight_layout()
        fig_path = output_dir / "training_curves.png"
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Training visualizations → {fig_path}")

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
        for bin_idx in range(n_bins):
            feature_names.append(f"{col}_bin{bin_idx}")

    # Feature importance plot (unchanged)
    fig_w, ax_w = plt.subplots(figsize=(25, 10))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in weights]
    ax_w.bar(range(len(weights)), weights, color=colors)
    ax_w.set_xticks(range(len(weights)))
    ax_w.set_xticklabels(feature_names, rotation=90, fontsize=8)
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

    if y_true is not None and len(x_eval_full) > 0:
        raw_full_scores = np.dot(x_eval_full, weights)
        probs_full = 1.0 / (1.0 + np.exp(-raw_full_scores))
        preds_full = (probs_full >= 0.5).astype(int)
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

        f.write(f"Pairwise Test Accuracy: {pairwise_accuracy:.4f}\n\n")

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

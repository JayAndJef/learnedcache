from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
) -> None:
    """Save pairwise evaluation report with feature importance and sample scores."""
    weights = model.get_layer("ranking_weight").get_weights()[0].ravel()

    feature_names: list[str] = []
    for col, n_bins in zip(column_names, n_bins_list):
        for bin_idx in range(n_bins):
            feature_names.append(f"{col}_bin{bin_idx}")

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
    scores = model.predict(sample_items, verbose=0).ravel() if len(sample_items) > 0 else np.array([])
    order = np.argsort(scores) if len(scores) > 0 else np.array([], dtype=int)

    eval_path = output_dir / "eval_report.txt"
    with eval_path.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Eviction-Time Pairwise-Diff Ranker Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Objective: {objective}\n")
        f.write(f"Access pattern: {access_pattern}\n")
        f.write(f"Eviction pattern: {eviction_pattern}\n")
        f.write(f"Total supervised rows: {n_rows}\n")
        f.write(f"Training rows: {n_train_rows}\n")
        f.write(f"Test rows: {n_test_rows}\n")
        f.write(f"Training pairs: {n_train_pairs}\n")
        f.write(f"Test pairs: {n_test_pairs}\n")
        f.write(f"Epochs trained: {epochs_trained}\n")
        f.write(f"Pairwise accuracy: {pairwise_accuracy:.4f}\n\n")

        f.write(f"Train pair stats: {train_pair_stats or {}}\n")
        f.write(f"Test pair stats: {test_pair_stats or {}}\n\n")

        f.write(f"Weight vector shape: {weights.shape}\n")
        f.write(f"Weight vector: {weights}\n\n")

        f.write("Sample pairwise probabilities (A reused sooner):\n")
        for idx, score in enumerate(scores):
            f.write(f"  Item {idx}: p = {score:.4f}\n")

        f.write("\nSample ascending probability order (low -> high):\n")
        f.write(f"  {list(order)}\n")

    print(f"Evaluation report → {eval_path}")
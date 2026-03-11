import typer
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

def save_training_visualizations(history, Y_test_pairs, Y_pred, accuracy, output_dir):
    """Save training curves and confusion matrix."""
    fig = plt.figure(figsize=(16, 5))

    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(history.history["accuracy"], label="Train Acc")
    ax1.plot(history.history["val_accuracy"], label="Val Acc")
    ax1.set_title("Pairwise Ranking Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Pairwise Ranking Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Binary Crossentropy")
    ax2.legend()
    ax2.grid(True)

    ax3 = plt.subplot(1, 3, 3)
    cm = confusion_matrix(Y_test_pairs.astype(int), Y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        ax=ax3,
        xticklabels=["B sooner", "A sooner"],
        yticklabels=["B sooner", "A sooner"],
    )
    ax3.set_title(f"Confusion Matrix\nAccuracy: {accuracy:.4f}")
    ax3.set_ylabel("True")
    ax3.set_xlabel("Predicted")

    plt.tight_layout()
    fig_path = output_dir / "training_curves.png"
    fig.savefig(fig_path)
    plt.close(fig)
    typer.echo(f"Saved training visualizations → {fig_path}")


def save_evaluation_report(model, X_test_full, column_names, n_bins_list, output_dir):
    """Save evaluation report with feature importance and sample scores."""
    w = model.get_layer("ranking_weight").get_weights()[0].ravel()

    feature_names = []
    for col, n_bins in zip(column_names, n_bins_list):
        for b in range(n_bins):
            feature_names.append(f"{col}_bin{b}")

    fig_w, ax_w = plt.subplots(figsize=(25, 10))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in w]
    ax_w.bar(range(len(w)), w, color=colors)
    ax_w.set_xticks(range(len(w)))
    ax_w.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax_w.set_ylabel("Weight")
    ax_w.set_title("Learned Ranking Weights (green=keep, red=evict)")
    ax_w.axhline(y=0, color="black", linewidth=0.5)
    ax_w.grid(True, alpha=0.3)

    plt.tight_layout()
    importance_path = output_dir / "feature_importance.png"
    fig_w.savefig(importance_path)
    plt.close(fig_w)
    typer.echo(f"Saved feature importance → {importance_path}")

    sample_items = X_test_full[:10]
    scores = sample_items @ w
    order = np.argsort(scores)

    eval_path = output_dir / "eval_report.txt"
    with open(eval_path, "w") as f:
        f.write("=== Pairwise Ranker Evaluation Report ===\n\n")
        f.write(f"Weight vector shape: {w.shape}\n")
        f.write(f"Weight range: [{w.min():.4f}, {w.max():.4f}]\n\n")

        f.write("Feature weights:\n")
        for fname, weight in zip(feature_names, w):
            f.write(f"  {fname:20s}: {weight:8.4f}\n")

        f.write("\nSample item scores (higher = reused sooner = keep):\n")
        for i, s in enumerate(scores):
            f.write(f"  Item {i}: score = {s:.4f}\n")

        f.write(f"\nEviction order (first to evict -> last):\n")
        f.write(f"  {list(order)}\n")

    typer.echo(f"Saved evaluation report → {eval_path}")

import typer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def save_training_visualizations(history, Y_test_pairs, Y_pred, accuracy, output_dir):
    """Save training curves and confusion matrix."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Crossentropy")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["accuracy"], label="Train Acc")
    ax2.plot(history.history["val_accuracy"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    cm = confusion_matrix(Y_test_pairs.astype(int), Y_pred)
    im = ax3.imshow(cm, cmap="Blues", interpolation="nearest")
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["B sooner", "A sooner"])
    ax3.set_yticklabels(["B sooner", "A sooner"])

    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    ax3.set_title(f"Confusion Matrix\nAccuracy: {accuracy:.4f}")
    ax3.set_ylabel("True")
    ax3.set_xlabel("Predicted")

    plt.tight_layout()
    fig_path = output_dir / "training_curves.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    typer.echo(f"Saved training visualizations → {fig_path}")


def save_evaluation_report(model, X_test_full, column_names, n_bins_list, output_dir):
    """Save evaluation report with feature importance and sample scores."""
    w = model.get_layer("ranking_weight").get_weights()[0].ravel()

    feature_names = []
    for col, n_bins in zip(column_names, n_bins_list):
        for b in range(n_bins):
            feature_names.append(f"{col}_bin{b}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(w)), w)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Weight")
    ax.set_title("Feature Importance (Weight per One-Hot Discretized Bin)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    importance_path = output_dir / "feature_importance.png"
    fig.savefig(importance_path, dpi=150)
    plt.close(fig)
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

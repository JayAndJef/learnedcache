#!/usr/bin/env python3
"""
Learned Cache CLI - Unified command-line interface for cache eviction model training and export.

Commands:
    train-ranker: Train a pairwise ranker model on cache access traces
    export-model: Export trained model to BPF-compatible JSON format
    transform-logs: Convert raw logs to CSV format
"""

import json
import pickle
from pathlib import Path
from typing import Annotated

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from keras import layers
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

from learnedcache.loading import read_csvs_to_dataframe, transform_logs_to_csvs

app = typer.Typer(help="Learned Cache - Train and export cache eviction models")


def train_and_transform_discretizer(
    X_train: pd.DataFrame,
    n_bins: int = 5,
    encode: str = "ordinal",
    strategy: str = "quantile",
    subsample: int | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, KBinsDiscretizer]:
    """Train a KBinsDiscretizer and transform features."""
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode=encode,
        strategy=strategy,
        subsample=subsample,
        random_state=random_state,
        quantile_method="averaged_inverted_cdf",
    )
    X_transformed = discretizer.fit_transform(X_train)
    X_transformed_df = pd.DataFrame(
        X_transformed, columns=X_train.columns, index=X_train.index
    )
    return X_transformed_df, discretizer


def one_hot_encode_features(discretized_df: pd.DataFrame, n_bins_list: list[int]) -> np.ndarray:
    """Manually one-hot encode discretized features into a numpy array."""
    encoded_parts = []
    for col_idx, (col_name, n_bins) in enumerate(zip(discretized_df.columns, n_bins_list)):
        col_vals = discretized_df[col_name].astype(int).values
        one_hot = np.zeros((len(col_vals), n_bins), dtype=np.float32)
        one_hot[np.arange(len(col_vals)), col_vals] = 1.0
        encoded_parts.append(one_hot)
    return np.concatenate(encoded_parts, axis=1)


def generate_pair_diffs(
    X_np: np.ndarray, Y_np: np.ndarray, n_pairs: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate pairwise training data as feature differences (vectorized).

    Returns
    -------
    X_diff : np.ndarray, shape (n_valid_pairs, n_features)
    labels : np.ndarray, shape (n_valid_pairs,)
        1.0 if A reused sooner, 0.0 otherwise.
    """
    rng = np.random.RandomState(seed)
    n = len(X_np)

    idx_a = rng.randint(0, n, size=n_pairs)
    idx_b = rng.randint(0, n, size=n_pairs)

    same = idx_a == idx_b
    while same.any():
        idx_b[same] = rng.randint(0, n, size=same.sum())
        same = idx_a == idx_b

    y_a = Y_np[idx_a]
    y_b = Y_np[idx_b]

    mask = y_a != y_b
    idx_a = idx_a[mask]
    idx_b = idx_b[mask]
    y_a = y_a[mask]
    y_b = y_b[mask]

    X_diff = X_np[idx_a] - X_np[idx_b]
    labels = (y_a < y_b).astype(np.float32)

    return X_diff, labels


@app.command()
def transform_logs(
    log_pattern: Annotated[str, typer.Option(help="Glob pattern for input log files")],
    output_suffix: Annotated[str, typer.Option(help="Suffix for output CSV files")] = "access.csv",
) -> None:
    """Transform raw log files to CSV format."""
    typer.echo(f"Transforming logs matching: {log_pattern}")
    transform_logs_to_csvs(log_pattern, output_suffix=output_suffix)
    typer.echo("Done.")


@app.command()
def train_ranker(
    file_pattern: Annotated[str, typer.Option(help="Glob pattern for input CSV files")],
    output_dir: Annotated[Path, typer.Option(help="Directory to save model and artifacts")],
    discretize_cols: Annotated[list[str], typer.Option(help="Columns to discretize")] = ["t", "z", "f"],
    raw_cols: Annotated[list[str], typer.Option(help="Raw columns to include")] = ["s", "trial_id"],
    n_bins: Annotated[int, typer.Option(help="Number of bins for discretization")] = 10,
    max_epochs: Annotated[int, typer.Option(help="Maximum training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 256,
    sampling_multiplier: Annotated[float, typer.Option(help="Pair sampling multiplier")] = 1.0,
) -> None:
    """Train a linear pairwise ranker model on cache access traces."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_csvs_to_dataframe(file_pattern)
    typer.echo(f"Loaded {len(df)} rows")

    Y = df.groupby(["trial_id", "dm", "dn", "in", "of"])["pd"].shift(-1)
    labeled = Y.notna().sum()
    unlabeled = Y.isna().sum()
    typer.echo(f"  Labeled: {labeled} | Unlabeled: {unlabeled} | Coverage: {labeled / len(Y) * 100:.2f}%")

    Y.fillna(1e15, inplace=True)
    Y_np = Y.values

    typer.echo(f"Discretizing features: {discretize_cols}")
    featureset_df, discretizer = train_and_transform_discretizer(
        df[discretize_cols], n_bins=n_bins, strategy="quantile"
    )
    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(len(discretize_cols))]
    typer.echo(f"  Bins per feature: {n_bins_list}")

    X_onehot = one_hot_encode_features(featureset_df, n_bins_list)
    X_raw = df[raw_cols].values.astype(np.float32)
    X_full = np.concatenate([X_onehot, X_raw], axis=1)
    n_encoded_features = X_full.shape[1]
    typer.echo(f"  Feature matrix shape: {X_full.shape}")

    typer.echo("Splitting data...")
    X_train_full, X_test_full, Y_train_raw, Y_test_raw = train_test_split(
        X_full, Y_np, test_size=0.2, random_state=42
    )

    typer.echo("Generating pairwise training data...")
    N_TRAIN_PAIRS = int(len(Y_train_raw) * sampling_multiplier)
    N_TEST_PAIRS = int(len(Y_test_raw) * sampling_multiplier)

    X_diff_train, Y_train_pairs = generate_pair_diffs(X_train_full, Y_train_raw, N_TRAIN_PAIRS, seed=42)
    X_diff_test, Y_test_pairs = generate_pair_diffs(X_test_full, Y_test_raw, N_TEST_PAIRS, seed=123)

    typer.echo(f"  Train pairs: {len(X_diff_train)} | Test pairs: {len(X_diff_test)}")

    input_diff = layers.Input(shape=(n_encoded_features,), name="feature_diff")
    output = layers.Dense(1, activation="sigmoid", use_bias=False, name="ranking_weight")(input_diff)
    model = keras.Model(inputs=input_diff, outputs=output, name="LinearPairwiseRanker")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    typer.echo(f"Training (max_epochs={max_epochs}, batch_size={batch_size})...")
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    history = model.fit(
        X_diff_train, Y_train_pairs,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(X_diff_test, Y_test_pairs),
        callbacks=[early_stop],
        verbose=1,
    )

    Y_pred_prob = model.predict(X_diff_test, verbose=0).ravel()
    Y_pred = (Y_pred_prob >= 0.5).astype(int)
    accuracy = accuracy_score(Y_test_pairs.astype(int), Y_pred)

    report = classification_report(
        Y_test_pairs.astype(int), Y_pred,
        target_names=["B reused sooner", "A reused sooner"]
    )
    typer.echo(f"\nTest Accuracy: {accuracy:.4f}")
    typer.echo(f"Trained for {len(history.history['loss'])} epochs")
    typer.echo(f"\n{report}")

    _save_training_visualizations(history, Y_test_pairs, Y_pred, accuracy, output_dir)

    model_path = output_dir / "model.keras"
    model.save(model_path)
    typer.echo(f"Saved model → {model_path}")

    discretizer_path = output_dir / "discretizer.pkl"
    with open(discretizer_path, "wb") as f:
        pickle.dump(discretizer, f)
    typer.echo(f"Saved discretizer → {discretizer_path}")

    _save_evaluation_report(model, X_test_full, discretize_cols, n_bins_list, output_dir)

    typer.echo("Done.")


def _save_training_visualizations(history, Y_test_pairs, Y_pred, accuracy, output_dir):
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


def _save_evaluation_report(model, X_test_full, discretize_cols, n_bins_list, output_dir):
    """Save evaluation report with feature importance and sample scores."""
    w = model.get_layer("ranking_weight").get_weights()[0].ravel()

    feature_names = []
    for col, n_bins in zip(discretize_cols, n_bins_list):
        for b in range(n_bins):
            feature_names.append(f"{col}_bin{b}")
    feature_names.extend(["s", "trial_id"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(w)), w)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Weight")
    ax.set_title("Feature Importance (Weight per One-Hot Bin + Raw Features)")
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


@app.command()
def export_model(
    model_dir: Annotated[Path, typer.Option(help="Directory containing trained model artifacts")],
    output_file: Annotated[Path, typer.Option(help="Output JSON file")] = Path("model_weights.json"),
    weight_scale: Annotated[int, typer.Option(help="Scale factor for quantizing weights")] = 10000,
    feature_names: Annotated[list[str], typer.Option(help="Feature names in BPF enum order")] = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"],
) -> None:
    """Export trained model to BPF-compatible JSON format."""

    typer.echo(f"Loading model from {model_dir}...")

    discretizer_path = model_dir / "discretizer.pkl"
    if not discretizer_path.exists():
        typer.echo(f"Error: {discretizer_path} not found", err=True)
        raise typer.Exit(1)

    with open(discretizer_path, "rb") as f:
        discretizer = pickle.load(f)

    model_path = model_dir / "model.keras"
    if not model_path.exists():
        typer.echo(f"Error: {model_path} not found", err=True)
        raise typer.Exit(1)

    model = keras.models.load_model(model_path)

    w = model.get_layer("ranking_weight").get_weights()[0].ravel()

    n_features = len(feature_names)
    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(n_features)]

    typer.echo(f"Features: {feature_names}")
    typer.echo(f"Bins per feature: {n_bins_list}")
    typer.echo(f"Total one-hot features: {sum(n_bins_list)}")
    typer.echo(f"Weight vector shape: {w.shape}")

    model_data = {
        "feature_names": feature_names,
        "n_features": n_features,
        "weight_scale": weight_scale,
        "features": []
    }

    weight_idx = 0
    for feat_idx, feat_name in enumerate(feature_names):
        n_bins = n_bins_list[feat_idx]

        all_edges = discretizer.bin_edges_[feat_idx]
        interior_edges = all_edges[1:-1].tolist()

        feat_weights_float = w[weight_idx:weight_idx + n_bins]
        feat_weights_int = (feat_weights_float * weight_scale).astype(np.int64).tolist()

        feature_data = {
            "index": feat_idx,
            "name": feat_name,
            "n_bins": n_bins,
            "bin_edges": [int(x) for x in interior_edges],
            "weights_float": feat_weights_float.tolist(),
            "weights_int": feat_weights_int,
        }

        model_data["features"].append(feature_data)
        weight_idx += n_bins

        typer.echo(f"  {feat_name}: {n_bins} bins, weights [{feat_weights_float.min():.4f}, {feat_weights_float.max():.4f}]")

    with open(output_file, "w") as f:
        json.dump(model_data, f, indent=2)

    typer.echo(f"\nExported model to {output_file}")
    typer.echo(f"Weight scale factor: {weight_scale}")
    typer.echo("Ready to load into BPF maps!")
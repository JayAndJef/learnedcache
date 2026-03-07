"""
Pairwise Ranker CLI for Learned Cache Eviction

Trains a linear pairwise ranker (Bradley-Terry model) on cache access traces
and outputs evaluation metrics and visualizations.

Usage:
    python pairwise_ranker.py --file-pattern 'data/fileserver/*4_amended_access.csv' --output-dir outputs/gen5
"""

import glob
from pathlib import Path
from typing import Annotated

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

app = typer.Typer(help="Train a linear pairwise ranker for cache eviction.")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def read_csvs_to_dataframe(file_pattern: str) -> pd.DataFrame:
    """
    Reads multiple CSV files and concatenates them into a single dataframe.
    Adds a trial_id column based on the sequence in which files are loaded.
    """
    filepaths = sorted(glob.glob(file_pattern))
    if not filepaths:
        typer.echo(f"No files matched pattern: {file_pattern}", err=True)
        raise typer.Exit(code=1)
    dataframes = []
    for trial_id, filepath in enumerate(filepaths):
        df = pd.read_csv(filepath)
        df["trial_id"] = trial_id
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def train_and_transform_discretizer(
    X_train: pd.DataFrame,
    n_bins: int = 5,
    encode: str = "ordinal",
    strategy: str = "quantile",
    subsample: int | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, KBinsDiscretizer]:
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
        X_transformed,
        columns=X_train.columns,
        index=X_train.index,
    )
    return X_transformed_df, discretizer


def one_hot_encode_features(
    discretized_df: pd.DataFrame, n_bins_list: list[int]
) -> np.ndarray:
    """
    Manually one-hot encode discretized features into a numpy array.

    Returns
    -------
    np.ndarray of shape (n_samples, sum(n_bins_list))
    """
    encoded_parts = []
    for col_name, n_bins in zip(discretized_df.columns, n_bins_list):
        col_vals = discretized_df[col_name].astype(int).values
        one_hot = np.zeros((len(col_vals), n_bins), dtype=np.float32)
        one_hot[np.arange(len(col_vals)), col_vals] = 1.0
        encoded_parts.append(one_hot)
    return np.concatenate(encoded_parts, axis=1)


def generate_pair_diffs(
    X_np: np.ndarray,
    Y_np: np.ndarray,
    n_pairs: int,
    seed: int = 42,
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

    # Filter out ties
    mask = y_a != y_b
    idx_a = idx_a[mask]
    idx_b = idx_b[mask]
    y_a = y_a[mask]
    y_b = y_b[mask]

    X_diff = X_np[idx_a] - X_np[idx_b]
    labels = (y_a < y_b).astype(np.float32)

    return X_diff, labels


# ---------------------------------------------------------------------------
# Main CLI command
# ---------------------------------------------------------------------------

@app.command()
def main(
    file_pattern: Annotated[
        str, typer.Option(help="Glob pattern for input CSV files to load.")
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Directory to save visualizations and eval data.")
    ],
    max_epochs: Annotated[
        int, typer.Option(help="Maximum number of training epochs.")
    ] = 10,
    batch_size: Annotated[
        int, typer.Option(help="Batch size for training.")
    ] = 256,
    sampling_multiplier: Annotated[
        float, typer.Option(help="Multiplier for number of pairs to generate (relative to dataset size).")
    ] = 1.0,
) -> None:
    """Train a linear pairwise ranker and save evaluation artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Data Loading
    # ------------------------------------------------------------------
    typer.echo("Loading data...")
    df = read_csvs_to_dataframe(file_pattern)
    typer.echo(f"Loaded {len(df)} rows from pattern: {file_pattern}")

    # ------------------------------------------------------------------
    # 2. Build Labels (Next-Reuse Time)
    # ------------------------------------------------------------------
    typer.echo("Computing labels...")
    Y = df.groupby(["trial_id", "dm", "dn", "in", "of"])["pd"].shift(-1)
    labeled = Y.notna().sum()
    unlabeled = Y.isna().sum()
    typer.echo(f"  Labeled rows: {labeled}  |  Unlabeled: {unlabeled}  |  Coverage: {labeled / len(Y) * 100:.2f}%")

    Y.fillna(1e15, inplace=True)
    Y_np = Y.values

    # ------------------------------------------------------------------
    # 3. Random Train/Test Split (80/20)
    # ------------------------------------------------------------------
    typer.echo("Splitting data randomly (80/20)...")
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42
    )
    typer.echo(f"  Train rows: {len(train_idx)}  |  Test rows: {len(test_idx)}")

    # ------------------------------------------------------------------
    # 4. Feature Discretization (fit on train only)
    # ------------------------------------------------------------------
    typer.echo("Discretizing features (fit on train only)...")
    discretize_cols = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"]

    train_features_df = df.iloc[train_idx][discretize_cols].reset_index(drop=True)
    test_features_df = df.iloc[test_idx][discretize_cols].reset_index(drop=True)

    train_discretized, discretizer = train_and_transform_discretizer(
        train_features_df, n_bins=10, strategy="quantile"
    )
    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(len(discretize_cols))]
    typer.echo(f"Bins per discretized feature: {n_bins_list}")

    # Transform test data using the train-fitted discretizer
    test_discretized_np = discretizer.transform(test_features_df)
    test_discretized = pd.DataFrame(
        test_discretized_np,
        columns=discretize_cols,
    )

    # ------------------------------------------------------------------
    # 5. One-Hot Encode + Build Feature Matrices
    # ------------------------------------------------------------------
    X_train_full = one_hot_encode_features(train_discretized, n_bins_list)
    X_test_full = one_hot_encode_features(test_discretized, n_bins_list)
    n_encoded_features = X_train_full.shape[1]

    typer.echo(f"Feature matrix shape: train={X_train_full.shape}, test={X_test_full.shape}")
    typer.echo(f"  One-hot features: {n_encoded_features} (from bins {n_bins_list})")

    Y_train_raw = Y_np[train_idx]
    Y_test_raw = Y_np[test_idx]

    # ------------------------------------------------------------------
    # 6. Generate Pairwise Training Data
    # ------------------------------------------------------------------
    typer.echo("Generating pairwise training data...")

    N_TRAIN_PAIRS = int(len(Y_train_raw) * sampling_multiplier)
    N_TEST_PAIRS = int(len(Y_test_raw) * sampling_multiplier)

    X_diff_train, Y_train_pairs = generate_pair_diffs(
        X_train_full, Y_train_raw, N_TRAIN_PAIRS, seed=42
    )
    X_diff_test, Y_test_pairs = generate_pair_diffs(
        X_test_full, Y_test_raw, N_TEST_PAIRS, seed=123
    )

    typer.echo(f"  Training pairs: {len(Y_train_pairs)} (multiplier: {sampling_multiplier})")
    typer.echo(f"  Test pairs:     {len(Y_test_pairs)} (multiplier: {sampling_multiplier})")
    typer.echo(f"  Train label balance (frac A sooner): {Y_train_pairs.mean():.3f}")
    typer.echo(f"  Test  label balance (frac A sooner): {Y_test_pairs.mean():.3f}")

    # ------------------------------------------------------------------
    # 7. Build Model
    # ------------------------------------------------------------------
    import keras
    from keras import layers
    from keras.callbacks import EarlyStopping

    typer.echo("Building model...")
    input_diff = layers.Input(shape=(n_encoded_features,), name="feature_diff")
    output = layers.Dense(1, activation="sigmoid", use_bias=False, name="ranking_weight")(input_diff)

    model = keras.Model(inputs=input_diff, outputs=output, name="LinearPairwiseRanker")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    # ------------------------------------------------------------------
    # 8. Training
    # ------------------------------------------------------------------
    typer.echo(f"Training (max_epochs={max_epochs}, batch_size={batch_size})...")
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_diff_train,
        Y_train_pairs,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(X_diff_test, Y_test_pairs),
        callbacks=[early_stop],
        verbose=1,
    )

    # ------------------------------------------------------------------
    # 9. Evaluation
    # ------------------------------------------------------------------
    typer.echo("Evaluating...")
    Y_pred_prob = model.predict(X_diff_test, verbose=0).ravel()
    Y_pred = (Y_pred_prob >= 0.5).astype(int)
    accuracy = accuracy_score(Y_test_pairs.astype(int), Y_pred)

    report = classification_report(
        Y_test_pairs.astype(int),
        Y_pred,
        target_names=["B reused sooner", "A reused sooner"],
    )

    typer.echo(f"Pairwise Test Accuracy: {accuracy:.4f}")
    typer.echo(f"Trained for {len(history.history['loss'])} epochs")
    typer.echo("\nClassification Report:")
    typer.echo(report)

    # --- Plot: training curves + confusion matrix ---
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
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    ax3 = plt.subplot(1, 3, 3)
    cm = confusion_matrix(Y_test_pairs.astype(int), Y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax3,
        xticklabels=["B sooner", "A sooner"],
        yticklabels=["B sooner", "A sooner"],
    )
    ax3.set_title(f"Confusion Matrix\nAccuracy: {accuracy:.4f}")
    ax3.set_ylabel("True")
    ax3.set_xlabel("Predicted")

    plt.tight_layout()
    training_fig_path = output_dir / "training_curves.png"
    fig.savefig(training_fig_path, dpi=150)
    plt.close(fig)
    typer.echo(f"Saved training curves & confusion matrix → {training_fig_path}")

    # ------------------------------------------------------------------
    # 10. Save Model and Discretizer for BPF Loading
    # ------------------------------------------------------------------
    import pickle

    model_path = output_dir / "model.keras"
    model.save(model_path)
    typer.echo(f"Saved model → {model_path}")

    discretizer_path = output_dir / "discretizer.pkl"
    with open(discretizer_path, "wb") as f:
        pickle.dump(discretizer, f)
    typer.echo(f"Saved discretizer → {discretizer_path}")

    # ------------------------------------------------------------------
    # 11. Extract Weight Vector
    # ------------------------------------------------------------------
    w = model.get_layer("ranking_weight").get_weights()[0].ravel()

    sample_items = X_test_full[:10]
    scores = sample_items @ w
    order = np.argsort(scores)

    # --- Plot: feature weights ---
    feature_names = []
    for col, n_bins in zip(discretize_cols, n_bins_list):
        for b in range(n_bins):
            feature_names.append(f"{col}_bin{b}")

    fig_w, ax_w = plt.subplots(figsize=(14, 5))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in w]
    ax_w.bar(range(len(w)), w, color=colors)
    ax_w.set_xticks(range(len(w)))
    ax_w.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax_w.set_ylabel("Weight")
    ax_w.set_title("Learned Ranking Weights (green=keep, red=evict)")
    ax_w.axhline(y=0, color="black", linewidth=0.5)
    ax_w.grid(True, alpha=0.3)
    plt.tight_layout()
    weights_fig_path = output_dir / "feature_weights.png"
    fig_w.savefig(weights_fig_path, dpi=150)
    plt.close(fig_w)
    typer.echo(f"Saved feature weights plot → {weights_fig_path}")

    # --- Write eval report ---
    eval_path = output_dir / "eval_report.txt"
    with open(eval_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Pairwise Ranker Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"File pattern: {file_pattern}\n")
        f.write(f"Total rows loaded: {len(df)}\n")
        f.write(f"Training pairs: {len(Y_train_pairs)}\n")
        f.write(f"Test pairs: {len(Y_test_pairs)}\n")
        f.write(f"Epochs trained: {len(history.history['loss'])}\n\n")

        f.write(f"Pairwise Test Accuracy: {accuracy:.4f}\n\n")

        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n")

        f.write(f"Weight vector shape: {w.shape}\n")
        f.write(f"Weight vector: {w}\n\n")

        f.write("Sample item scores (higher = reused sooner = keep):\n")
        for i, s in enumerate(scores):
            f.write(f"  Item {i}: score = {s:.4f}\n")

        f.write(f"\nEviction order (first to evict -> last):\n")
        f.write(f"  {list(order)}\n")

    typer.echo(f"Saved eval report → {eval_path}")
    typer.echo("Done.")


if __name__ == "__main__":
    app()
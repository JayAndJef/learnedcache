"""
Core logic functions for learned cache training and export (decoupled from CLI).
"""

import json
import pickle
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from learnedcache.loading import read_csvs_to_dataframe, transform_logs_to_csvs
from learnedcache.preprocess import one_hot_encode_features, generate_pair_diffs, train_and_transform_discretizer
from learnedcache.models import build_model
from learnedcache.helpers import save_evaluation_report, save_training_visualizations


def run_transform_logs(log_pattern: str, verbose: bool = True) -> None:
    """Transform raw log files to CSV format."""
    if verbose:
        print(f"Transforming logs matching: {log_pattern}")
    transform_logs_to_csvs(log_pattern)
    if verbose:
        print("Done.")


def run_train_ranker(
    file_pattern: str,
    output_dir: Path,
    discretize_cols: list[str] = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"],
    n_bins: int = 10,
    max_epochs: int = 50,
    batch_size: int = 256,
    sampling_multiplier: float = 1.0,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """Train a linear pairwise ranker model on cache access traces.

    Args:
        file_pattern: Glob pattern for input CSV files
        output_dir: Directory to save model and artifacts
        discretize_cols: Columns to discretize
        n_bins: Number of bins for discretization
        max_epochs: Maximum training epochs
        batch_size: Training batch size
        sampling_multiplier: Pair sampling multiplier
        random_state: Random seed for reproducibility
        verbose: Whether to print progress messages

    Returns:
        dict with keys: model, discretizer, n_bins_list, discretize_cols, accuracy, history
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------
    if verbose:
        print("Loading data...")
    df = read_csvs_to_dataframe(file_pattern)
    if verbose:
        print(f"Loaded {len(df)} rows from pattern: {file_pattern}")

    # ------------------------------------------------------------------
    # 2. Compute Labels (Next Reuse Time)
    # ------------------------------------------------------------------
    if verbose:
        print("Computing labels...")
    Y = df.groupby(["trial_id", "dm", "dn", "in", "of"])["pd"].shift(-1)
    labeled = Y.notna().sum()
    unlabeled = Y.isna().sum()
    if verbose:
        print(f"  Labeled rows: {labeled}  |  Unlabeled: {unlabeled}  |  Coverage: {labeled / len(Y) * 100:.2f}%")

    Y.fillna(1e15, inplace=True)
    Y_np = Y.values

    # ------------------------------------------------------------------
    # 3. Split Data (BEFORE discretization to avoid data leakage)
    # ------------------------------------------------------------------
    if verbose:
        print("Splitting data randomly (80/20)...")
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=random_state
    )
    if verbose:
        print(f"  Train rows: {len(train_idx)}  |  Test rows: {len(test_idx)}")

    # ------------------------------------------------------------------
    # 4. Discretize Features (fit on train only)
    # ------------------------------------------------------------------
    if verbose:
        print("Discretizing features (fit on train only)...")
    train_features_df = df.iloc[train_idx][discretize_cols].reset_index(drop=True)
    test_features_df = df.iloc[test_idx][discretize_cols].reset_index(drop=True)

    train_discretized, discretizer = train_and_transform_discretizer(
        train_features_df, n_bins=n_bins, strategy="quantile"
    )
    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(len(discretize_cols))]
    if verbose:
        print(f"Bins per discretized feature: {n_bins_list}")

    test_discretized_np = discretizer.transform(test_features_df)
    test_discretized = pd.DataFrame(
        test_discretized_np,
        columns=discretize_cols,
    )

    # ------------------------------------------------------------------
    # 5. One-Hot Encode
    # ------------------------------------------------------------------
    X_train_full = one_hot_encode_features(train_discretized, n_bins_list)
    X_test_full = one_hot_encode_features(test_discretized, n_bins_list)
    n_encoded_features = X_train_full.shape[1]

    if verbose:
        print(f"Feature matrix shape: train={X_train_full.shape}, test={X_test_full.shape}")
        print(f"  One-hot features: {n_encoded_features} (from bins {n_bins_list})")

    Y_train_raw = Y_np[train_idx]
    Y_test_raw = Y_np[test_idx]

    # ------------------------------------------------------------------
    # 6. Generate Pairwise Training Data
    # ------------------------------------------------------------------
    if verbose:
        print("Generating pairwise training data...")
    N_TRAIN_PAIRS = int(len(Y_train_raw) * sampling_multiplier)
    N_TEST_PAIRS = int(len(Y_test_raw) * sampling_multiplier)

    X_diff_train, Y_train_pairs = generate_pair_diffs(
        X_train_full, Y_train_raw, N_TRAIN_PAIRS, seed=random_state
    )
    X_diff_test, Y_test_pairs = generate_pair_diffs(
        X_test_full, Y_test_raw, N_TEST_PAIRS, seed=random_state
    )

    if verbose:
        print(f"  Training pairs: {len(Y_train_pairs)} (multiplier: {sampling_multiplier})")
        print(f"  Test pairs:     {len(Y_test_pairs)} (multiplier: {sampling_multiplier})")
        print(f"  Train label balance (frac A sooner): {Y_train_pairs.mean():.3f}")
        print(f"  Test  label balance (frac A sooner): {Y_test_pairs.mean():.3f}")

    # ------------------------------------------------------------------
    # 7. Build Model
    # ------------------------------------------------------------------
    if verbose:
        print("Building model...")
    model = build_model(n_encoded_features)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    if verbose:
        model.summary()

    # ------------------------------------------------------------------
    # 8. Training
    # ------------------------------------------------------------------
    if verbose:
        print(f"Training (max_epochs={max_epochs}, batch_size={batch_size})...")
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1 if verbose else 0,
    )

    history = model.fit(
        X_diff_train,
        Y_train_pairs,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(X_diff_test, Y_test_pairs),
        callbacks=[early_stop],
        verbose=1 if verbose else 0,
    )

    # ------------------------------------------------------------------
    # 9. Evaluation
    # ------------------------------------------------------------------
    if verbose:
        print("Evaluating...")
    Y_pred_prob = model.predict(X_diff_test, verbose=0).ravel()
    Y_pred = (Y_pred_prob >= 0.5).astype(int)
    accuracy = accuracy_score(Y_test_pairs.astype(int), Y_pred)

    report = classification_report(
        Y_test_pairs.astype(int),
        Y_pred,
        target_names=["B reused sooner", "A reused sooner"],
    )

    if verbose:
        print(f"Pairwise Test Accuracy: {accuracy:.4f}")
        print(f"Trained for {len(history.history['loss'])} epochs")
        print("\nClassification Report:")
        print(report)

    save_training_visualizations(history, Y_test_pairs, Y_pred, accuracy, output_dir)

    # ------------------------------------------------------------------
    # 10. Save Model and Discretizer
    # ------------------------------------------------------------------
    model_path = output_dir / "model.keras"
    model.save(model_path)
    if verbose:
        print(f"Saved model → {model_path}")

    discretizer_path = output_dir / "discretizer.pkl"
    with open(discretizer_path, "wb") as f:
        pickle.dump(discretizer, f)
    if verbose:
        print(f"Saved discretizer → {discretizer_path}")

    # ------------------------------------------------------------------
    # 11. Save Evaluation Report
    # ------------------------------------------------------------------
    save_evaluation_report(
        model, X_test_full, discretize_cols, n_bins_list, output_dir,
        file_pattern=file_pattern,
        n_rows=len(df),
        n_train_pairs=len(Y_train_pairs),
        n_test_pairs=len(Y_test_pairs),
        epochs_trained=len(history.history['loss']),
        accuracy=accuracy,
        classification_report_str=report,
    )

    if verbose:
        print("Done.")

    return {
        "model": model,
        "discretizer": discretizer,
        "n_bins_list": n_bins_list,
        "discretize_cols": discretize_cols,
        "accuracy": accuracy,
        "history": history,
    }


def run_export_model(
    model_dir: Path,
    output_file: Path,
    weight_scale: int = 10000,
    feature_names: list[str] = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"],
    verbose: bool = True,
) -> dict:
    """Export trained model to BPF-compatible JSON format.

    Args:
        model_dir: Directory containing trained model artifacts
        output_file: Output JSON file
        weight_scale: Scale factor for quantizing weights
        feature_names: Feature names in BPF enum order
        verbose: Whether to print progress messages

    Returns:
        dict: The exported model data
    """
    if verbose:
        print(f"Loading model from {model_dir}...")

    discretizer_path = model_dir / "discretizer.pkl"
    if not discretizer_path.exists():
        raise FileNotFoundError(f"{discretizer_path} not found")

    with open(discretizer_path, "rb") as f:
        discretizer = pickle.load(f)

    model_path = model_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found")

    model = keras.models.load_model(model_path)

    w = model.get_layer("ranking_weight").get_weights()[0].ravel()

    n_features = len(feature_names)
    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(n_features)]

    if verbose:
        print(f"Features: {feature_names}")
        print(f"Bins per feature: {n_bins_list}")
        print(f"Total one-hot features: {sum(n_bins_list)}")
        print(f"Weight vector shape: {w.shape}")

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

        if verbose:
            print(f"  {feat_name}: {n_bins} bins, weights [{feat_weights_float.min():.4f}, {feat_weights_float.max():.4f}]")

    with open(output_file, "w") as f:
        json.dump(model_data, f, indent=2)

    if verbose:
        print(f"\nExported model to {output_file}")
        print(f"Weight scale factor: {weight_scale}")
        print("Ready to load into BPF maps!")

    return model_data

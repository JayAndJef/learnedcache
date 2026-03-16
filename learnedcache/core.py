"""
Core logic for learned cache training and export.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import keras
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from learnedcache.helpers import save_evaluation_report, save_training_visualizations
from learnedcache.loading import read_access_eviction_trial_pairs, transform_logs_to_csvs
from learnedcache.models import build_model
from learnedcache.preprocess import one_hot_encode_features, train_and_transform_discretizer

PAGE_KEY_COLS = ["dm", "dn", "in", "of"]
TS_COL = "ts"

DERIVED_FEATURE_COL = "time_since_last_access_at_eviction"
TARGET_COL = "time_until_next_reuse_from_eviction"
NO_REUSE_LABEL_OFFSET = 1.0

def run_transform_logs(log_pattern: str, verbose: bool = True) -> None:
    """Transform raw log files to CSV format."""
    if verbose:
        print(f"Transforming logs matching: {log_pattern}")
    transform_logs_to_csvs(log_pattern)
    if verbose:
        print("Done.")

def _validate_required_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")

def _require_numeric(df: pd.DataFrame, columns: list[str], df_name: str) -> pd.DataFrame:
    """Return a copy with required columns coerced to numeric; fail on invalid values."""
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any():
            raise ValueError(f"{df_name} has non-numeric values in required column '{col}'.")
    return out


def _build_eviction_supervised_df(
    access_eviction_pairs: list[tuple[int, pd.DataFrame, pd.DataFrame]],
    discretize_cols: list[str],
) -> pd.DataFrame:
    if not discretize_cols:
        raise ValueError("discretize_cols cannot be empty.")

    all_dfs: list[pd.DataFrame] = []
    access_required = sorted(set(PAGE_KEY_COLS + [TS_COL] + discretize_cols))
    eviction_required = [TS_COL]

    for trial_id, access_df, eviction_df in access_eviction_pairs:
        _validate_required_columns(
            access_df, access_required, f"access trial {trial_id}"
        )
        _validate_required_columns(
            eviction_df, eviction_required, f"eviction trial {trial_id}"
        )

        access_df = _require_numeric(
            access_df, [TS_COL, *discretize_cols], f"access trial {trial_id}"
        )
        eviction_df = _require_numeric(
            eviction_df, [TS_COL], f"eviction trial {trial_id}"
        )

        access_sorted = access_df.sort_values(TS_COL).reset_index(drop=True)
        eviction_sorted = eviction_df.sort_values(TS_COL).reset_index(drop=True)

        unique_pages = access_sorted[PAGE_KEY_COLS].drop_duplicates()
        eviction_expanded = eviction_sorted.merge(
            unique_pages, how="cross"
        ).sort_values(TS_COL)

        features_df = pd.merge_asof(
            eviction_expanded,
            access_sorted.rename(columns={TS_COL: "last_access_ts"}),
            left_on=TS_COL,
            right_on="last_access_ts",
            by=PAGE_KEY_COLS,
            direction="backward",
        )

        labels_df = pd.merge_asof(
            eviction_expanded,
            access_sorted[PAGE_KEY_COLS + [TS_COL]].rename(
                columns={TS_COL: "next_access_ts"}
            ),
            left_on=TS_COL,
            right_on="next_access_ts",
            by=PAGE_KEY_COLS,
            direction="forward",
            allow_exact_matches=False,
        )

        trial_result = features_df.copy()
        trial_result["next_access_ts"] = labels_df["next_access_ts"]
        trial_result["trial_id"] = trial_id
        trial_result["eviction_ts"] = trial_result[TS_COL]

        trial_result[DERIVED_FEATURE_COL] = (
            trial_result["eviction_ts"] - trial_result["last_access_ts"]
        )
        trial_result[TARGET_COL] = (
            trial_result["next_access_ts"] - trial_result["eviction_ts"]
        )

        trial_result = trial_result.dropna(subset=[DERIVED_FEATURE_COL])
        all_dfs.append(trial_result)

    if not all_dfs:
        raise ValueError(
            "No supervised rows were generated from access+eviction streams."
        )

    supervised_df = pd.concat(all_dfs, ignore_index=True)

    max_finite = supervised_df[TARGET_COL].max()
    no_reuse_label = (
        max_finite if pd.notnull(max_finite) else 0.0
    ) + NO_REUSE_LABEL_OFFSET
    supervised_df[TARGET_COL] = supervised_df[TARGET_COL].fillna(no_reuse_label)

    return supervised_df[
        ["trial_id", "eviction_ts", DERIVED_FEATURE_COL, TARGET_COL]
        + PAGE_KEY_COLS
        + discretize_cols
    ]


def _sample_pairwise_diffs_by_event(
    x_full: np.ndarray,
    y_full: np.ndarray,
    event_ids: np.ndarray,
    pairs_per_event: int,
    random_state: int,
    max_pairs_total: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Sample pairwise diffs within each event only.

    Event is typically (trial_id, eviction_ts). Ties are dropped (y_a == y_b).
    """
    if pairs_per_event <= 0:
        raise ValueError("pairs_per_event must be > 0.")

    rng = np.random.RandomState(random_state)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    events_total = 0
    events_with_pairs = 0
    ties_dropped = 0
    sampled_before_tie_drop = 0

    unique_events = np.unique(event_ids)
    for event_id in unique_events:
        events_total += 1
        local_idx = np.where(event_ids == event_id)[0]
        n = len(local_idx)
        if n < 2:
            continue

        idx_a_local = rng.randint(0, n, size=pairs_per_event)
        idx_b_local = rng.randint(0, n, size=pairs_per_event)

        same = idx_a_local == idx_b_local
        while same.any():
            idx_b_local[same] = rng.randint(0, n, size=int(same.sum()))
            same = idx_a_local == idx_b_local

        idx_a = local_idx[idx_a_local]
        idx_b = local_idx[idx_b_local]

        y_a = y_full[idx_a]
        y_b = y_full[idx_b]

        sampled_before_tie_drop += len(idx_a)

        mask = y_a != y_b
        ties_dropped += int((~mask).sum())
        if not mask.any():
            continue

        idx_a = idx_a[mask]
        idx_b = idx_b[mask]
        y_a = y_a[mask]
        y_b = y_b[mask]

        x_diff = x_full[idx_a] - x_full[idx_b]
        labels = (y_a < y_b).astype(np.float32)

        x_parts.append(x_diff.astype(np.float32, copy=False))
        y_parts.append(labels)
        events_with_pairs += 1

    if not x_parts:
        raise ValueError("No pairwise samples generated after tie-drop; cannot train pairwise ranker.")

    x_out = np.concatenate(x_parts, axis=0)
    y_out = np.concatenate(y_parts, axis=0)

    if max_pairs_total is not None and max_pairs_total > 0 and len(y_out) > max_pairs_total:
        keep = rng.choice(len(y_out), size=max_pairs_total, replace=False)
        x_out = x_out[keep]
        y_out = y_out[keep]

    stats = {
        "events_total": events_total,
        "events_with_pairs": events_with_pairs,
        "sampled_before_tie_drop": sampled_before_tie_drop,
        "ties_dropped": ties_dropped,
        "pairs_after_tie_drop": int(len(y_out)),
    }
    return x_out, y_out, stats

def run_train_ranker(
    access_pattern: str,
    eviction_pattern: str,
    output_dir: Path,
    discretize_cols: list[str] = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie"],
    n_bins: int = 10,
    max_epochs: int = 50,
    batch_size: int = 256,
    pairs_per_event: int = 64,
    max_pairs_total: int | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train a linear Bradley-Terry pairwise-diff ranker on eviction-time events."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("Loading access+eviction trial pairs...")
    pairs = read_access_eviction_trial_pairs(access_pattern, eviction_pattern)

    if verbose:
        print("Building eviction-time supervised dataset...")
    df = _build_eviction_supervised_df(pairs, discretize_cols)

    model_feature_cols = [*discretize_cols, DERIVED_FEATURE_COL]

    if verbose:
        print(f"Built {len(df)} supervised rows")
        print(
            f"Target stats ({TARGET_COL}): min={df[TARGET_COL].min():.3f}, "
            f"median={df[TARGET_COL].median():.3f}, max={df[TARGET_COL].max():.3f}"
        )
        print(f"Model features: {model_feature_cols}")

    trial_ids = np.sort(df["trial_id"].unique())
    if len(trial_ids) < 2:
        raise ValueError("Need at least 2 trials for holdout split (train=all but last, test=last).")

    test_trial_id = trial_ids[-1]
    train_trials = trial_ids[:-1]

    train_mask = df["trial_id"].isin(train_trials).to_numpy()
    test_mask = (df["trial_id"] == test_trial_id).to_numpy()

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Invalid trial split produced empty train or test set.")

    if verbose:
        print(f"Split by trial_id: train={train_trials.tolist()} | test={[int(test_trial_id)]}")
        print(f"  Train rows: {len(train_idx)}  |  Test rows: {len(test_idx)}")

    train_features_df = df.iloc[train_idx][model_feature_cols].reset_index(drop=True)
    test_features_df = df.iloc[test_idx][model_feature_cols].reset_index(drop=True)

    if verbose:
        print("Discretizing features (fit on train only)...")
    train_discretized, discretizer = train_and_transform_discretizer(
        train_features_df,
        n_bins=n_bins,
        strategy="quantile",
    )
    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(len(model_feature_cols))]
    if verbose:
        print(f"Bins per discretized feature: {n_bins_list}")

    test_discretized_np = discretizer.transform(test_features_df)
    test_discretized = pd.DataFrame(test_discretized_np, columns=model_feature_cols)

    x_train_full = one_hot_encode_features(train_discretized, n_bins_list)
    x_test_full = one_hot_encode_features(test_discretized, n_bins_list)
    y_train_raw = df.iloc[train_idx][TARGET_COL].to_numpy(dtype=np.float32)
    y_test_raw = df.iloc[test_idx][TARGET_COL].to_numpy(dtype=np.float32)

    train_events = pd.factorize(
        list(
            zip(
                df.iloc[train_idx]["trial_id"].to_numpy(),
                df.iloc[train_idx]["eviction_ts"].to_numpy(),
            )
        )
    )[0]
    test_events = pd.factorize(
        list(
            zip(
                df.iloc[test_idx]["trial_id"].to_numpy(),
                df.iloc[test_idx]["eviction_ts"].to_numpy(),
            )
        )
    )[0]

    x_diff_train, y_train_pairs, train_pair_stats = _sample_pairwise_diffs_by_event(
        x_full=x_train_full,
        y_full=y_train_raw,
        event_ids=train_events,
        pairs_per_event=pairs_per_event,
        random_state=random_state,
        max_pairs_total=max_pairs_total,
    )
    x_diff_test, y_test_pairs, test_pair_stats = _sample_pairwise_diffs_by_event(
        x_full=x_test_full,
        y_full=y_test_raw,
        event_ids=test_events,
        pairs_per_event=pairs_per_event,
        random_state=random_state + 1,
        max_pairs_total=max_pairs_total,
    )

    n_encoded_features = x_train_full.shape[1]
    if verbose:
        print(f"Feature matrix shape: train={x_train_full.shape}, test={x_test_full.shape}")
        print(f"  One-hot features: {n_encoded_features} (from bins {n_bins_list})")
        print(
            "Pairwise sampling train stats: "
            f"{train_pair_stats}, label_balance={float(np.mean(y_train_pairs)):.3f}"
        )
        print(
            "Pairwise sampling test stats: "
            f"{test_pair_stats}, label_balance={float(np.mean(y_test_pairs)):.3f}"
        )

    if verbose:
        print("Building pairwise-diff model...")
    model = build_model(n_encoded_features)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    if verbose:
        model.summary()

    if verbose:
        print(f"Training (max_epochs={max_epochs}, batch_size={batch_size})...")
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1 if verbose else 0,
    )

    history = model.fit(
        x_diff_train,
        y_train_pairs,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_data=(x_diff_test, y_test_pairs),
        callbacks=[early_stop],
        verbose=1 if verbose else 0,
    )

    if verbose:
        print("Evaluating...")
    y_pred_prob = model.predict(x_diff_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(np.float32)
    pairwise_accuracy = float(accuracy_score(y_test_pairs.astype(np.int32), y_pred.astype(np.int32)))

    if verbose:
        print(f"Pairwise Test Accuracy: {pairwise_accuracy:.4f}")
        print(f"Trained for {len(history.history.get('loss', []))} epochs")

    save_training_visualizations(
        history=history,
        y_true=y_test_pairs,
        y_pred=y_pred_prob,
        primary_metric=pairwise_accuracy,
        output_dir=output_dir,
    )

    model_path = output_dir / "model.keras"
    model.save(model_path)
    if verbose:
        print(f"Saved model → {model_path}")

    discretizer_path = output_dir / "discretizer.pkl"
    with discretizer_path.open("wb") as f:
        pickle.dump(discretizer, f)
    if verbose:
        print(f"Saved discretizer → {discretizer_path}")

    save_evaluation_report(
        model=model,
        x_eval_full=x_diff_test,
        column_names=model_feature_cols,
        n_bins_list=n_bins_list,
        output_dir=output_dir,
        access_pattern=access_pattern,
        eviction_pattern=eviction_pattern,
        n_rows=len(df),
        n_train_rows=len(train_idx),
        n_test_rows=len(test_idx),
        epochs_trained=len(history.history.get("loss", [])),
        objective="pairwise_diff",
        pairwise_accuracy=pairwise_accuracy,
        n_train_pairs=len(y_train_pairs),
        n_test_pairs=len(y_test_pairs),
        train_pair_stats=train_pair_stats,
        test_pair_stats=test_pair_stats,
    )

    if verbose:
        print("Done.")

    return {
        "model": model,
        "discretizer": discretizer,
        "n_bins_list": n_bins_list,
        "discretize_cols": model_feature_cols,
        "pairwise_accuracy": pairwise_accuracy,
        "history": history,
        "n_train_pairs": len(y_train_pairs),
        "n_test_pairs": len(y_test_pairs),
        "train_pair_stats": train_pair_stats,
        "test_pair_stats": test_pair_stats,
    }

def run_export_model(
    model_dir: Path,
    output_file: Path,
    weight_scale: int = 10000,
    feature_names: list[str] = ["pd", "sz", "fq", "sd", "p2", "id", "i2", "ie", DERIVED_FEATURE_COL],
    verbose: bool = True,
) -> dict[str, Any]:
    """Export trained model to BPF-compatible JSON format."""
    if verbose:
        print(f"Loading model from {model_dir}...")

    discretizer_path = model_dir / "discretizer.pkl"
    if not discretizer_path.exists():
        raise FileNotFoundError(f"{discretizer_path} not found")

    with discretizer_path.open("rb") as f:
        discretizer = pickle.load(f)

    model_path = model_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found")

    model = keras.models.load_model(model_path)
    weights = model.get_layer("ranking_weight").get_weights()[0].ravel()

    if len(discretizer.bin_edges_) != len(feature_names):
        raise ValueError(
            "feature_names length does not match trained discretizer feature count: "
            f"{len(feature_names)} vs {len(discretizer.bin_edges_)}"
        )

    n_features = len(feature_names)
    n_bins_list = [len(discretizer.bin_edges_[i]) - 1 for i in range(n_features)]

    if verbose:
        print(f"Features: {feature_names}")
        print(f"Bins per feature: {n_bins_list}")
        print(f"Total one-hot features: {sum(n_bins_list)}")
        print(f"Weight vector shape: {weights.shape}")

    model_data: dict[str, Any] = {
        "feature_names": feature_names,
        "n_features": n_features,
        "weight_scale": weight_scale,
        "features": [],
    }

    weight_idx = 0
    for feat_idx, feat_name in enumerate(feature_names):
        n_bins_feat = n_bins_list[feat_idx]
        all_edges = discretizer.bin_edges_[feat_idx]
        interior_edges = all_edges[1:-1].tolist()

        feat_weights_float = weights[weight_idx : weight_idx + n_bins_feat]
        feat_weights_int = (feat_weights_float * weight_scale).astype(np.int64).tolist()

        feature_data = {
            "index": feat_idx,
            "name": feat_name,
            "n_bins": n_bins_feat,
            "bin_edges": [int(x) for x in interior_edges],
            "weights_float": feat_weights_float.tolist(),
            "weights_int": feat_weights_int,
        }
        model_data["features"].append(feature_data)
        weight_idx += n_bins_feat

        if verbose:
            print(
                f"  {feat_name}: {n_bins_feat} bins, "
                f"weights [{feat_weights_float.min():.4f}, {feat_weights_float.max():.4f}]"
            )

    with Path(output_file).open("w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)

    if verbose:
        print(f"\nExported model to {output_file}")
        print(f"Weight scale factor: {weight_scale}")
        print("Ready to load into BPF maps!")

    return model_data

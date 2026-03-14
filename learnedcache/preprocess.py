import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

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

    Efficiently samples pairs by grouping samples by their label values,
    then sampling across different groups to avoid ties.

    Returns
    -------
    X_diff : np.ndarray, shape (n_pairs, n_features)
    labels : np.ndarray, shape (n_pairs,)
        1.0 if A reused sooner, 0.0 otherwise.
    """
    rng = np.random.RandomState(seed)
    n = len(X_np)

    unique_y, inverse_indices = np.unique(Y_np, return_inverse=True)
    n_unique = len(unique_y)

    if n_unique == 1:
        raise ValueError(
            f"Cannot generate pairs: all {n} samples have the same label value. "
            "No non-tie pairs are possible."
        )

    groups = [np.where(inverse_indices == i)[0] for i in range(n_unique)]
    group_sizes = np.array([len(g) for g in groups])

    idx_a = np.empty(n_pairs, dtype=np.int64)
    idx_b = np.empty(n_pairs, dtype=np.int64)

    for i in range(n_pairs):
        group_a, group_b = rng.choice(n_unique, size=2, replace=False)

        idx_a[i] = rng.choice(groups[group_a])
        idx_b[i] = rng.choice(groups[group_b])

    y_a = Y_np[idx_a]
    y_b = Y_np[idx_b]

    X_diff = X_np[idx_a] - X_np[idx_b]
    labels = (y_a < y_b).astype(np.float32)

    return X_diff, labels
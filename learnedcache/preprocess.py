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
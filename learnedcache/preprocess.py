import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


def train_and_transform_discretizer(
    x_train: pd.DataFrame,
    n_bins: int = 5,
    encode: str = "ordinal",
    strategy: str = "quantile",
    subsample: int | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, KBinsDiscretizer]:
    """Fit a KBinsDiscretizer on training features and return transformed features."""
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode=encode,
        strategy=strategy,
        subsample=subsample,
        random_state=random_state,
    )
    x_transformed = discretizer.fit_transform(x_train)
    x_transformed_df = pd.DataFrame(
        x_transformed,
        columns=x_train.columns,
        index=x_train.index,
    )
    return x_transformed_df, discretizer


def one_hot_encode_features(
    discretized_df: pd.DataFrame,
    n_bins_list: list[int],
) -> np.ndarray:
    """One-hot encode discretized integer-bin feature columns."""
    encoded_parts: list[np.ndarray] = []

    for col_name, n_bins in zip(discretized_df.columns, n_bins_list):
        col_vals = discretized_df[col_name].to_numpy(dtype=np.int64, copy=False)

        if np.any(col_vals < 0) or np.any(col_vals >= n_bins):
            raise ValueError(
                f"Invalid bin index in column '{col_name}'. "
                f"Expected values in [0, {n_bins - 1}]."
            )

        one_hot = np.zeros((len(col_vals), n_bins), dtype=np.float32)
        one_hot[np.arange(len(col_vals)), col_vals] = 1.0
        encoded_parts.append(one_hot)

    if not encoded_parts:
        raise ValueError("No columns were available for one-hot encoding.")

    return np.concatenate(encoded_parts, axis=1)
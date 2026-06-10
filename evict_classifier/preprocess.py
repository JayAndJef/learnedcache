"""Feature discretization and one-hot encoding.

Copied verbatim from ``learnedcache.preprocess`` (kept independent so the ranker
package is untouched). A ``KBinsDiscretizer`` (quantile, ordinal) is fit on a
subsample, then applied in batches; one-hot encoding uses a pre-allocated int8
buffer with in-place column writes.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


def fit_discretizer_from_sample(
    sample: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
    random_state: int | None = 42,
) -> KBinsDiscretizer:
    """Fit a ``KBinsDiscretizer`` on a pre-subsampled feature matrix.

    ``subsample=None`` is passed because the caller has already subsampled.
    """
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy=strategy,
        subsample=None,  # caller already subsampled
        random_state=random_state,
    )
    discretizer.fit(sample)
    return discretizer


def transform_discretizer_batch(
    features: np.ndarray,
    discretizer: KBinsDiscretizer,
) -> np.ndarray:
    """Transform *features* through a fitted discretizer, returning int8.

    Extracts ``CSR.data`` directly and reshapes, avoiding the float64 dense
    intermediate that ``.toarray()`` would create.

    Returns an int8 array of shape ``(n_rows, n_features)`` with ordinal bin
    indices in ``[0, n_bins_per_feature)``.
    """
    csr = discretizer.transform(features)
    # csr.data may be a memoryview in newer scipy -- wrap with np.asarray.
    result = np.asarray(csr.data, dtype=np.int8).reshape(features.shape)
    del csr
    return result


def one_hot_encode_features(
    discretized: np.ndarray,
    n_bins_list: list[int],
) -> np.ndarray:
    """One-hot encode discretized integer-bin feature columns from a 2-D ndarray.

    Uses a single pre-allocated int8 buffer (one-hot values are {0,1}) with
    in-place advanced-index writes.
    """
    if discretized.ndim != 2:
        raise ValueError(f"Expected 2-D array, got {discretized.ndim}-D.")

    n_rows, n_cols = discretized.shape
    if n_rows == 0 or not n_bins_list:
        raise ValueError("No columns were available for one-hot encoding.")
    if n_cols != len(n_bins_list):
        raise ValueError(
            f"Array has {n_cols} columns but n_bins_list has {len(n_bins_list)} entries."
        )

    total_n_bins = sum(n_bins_list)
    out = np.zeros((n_rows, total_n_bins), dtype=np.int8)
    row_idx = np.arange(n_rows)

    col_offset = 0
    for col_idx, n_bins in enumerate(n_bins_list):
        col_vals = np.asarray(discretized[:, col_idx], dtype=np.int8)
        if np.any(col_vals < 0) or np.any(col_vals >= n_bins):
            raise ValueError(
                f"Invalid bin index in column {col_idx}. "
                f"Expected values in [0, {n_bins - 1}]."
            )
        out[row_idx, col_offset + col_vals.astype(np.intp)] = 1
        col_offset += n_bins

    return out

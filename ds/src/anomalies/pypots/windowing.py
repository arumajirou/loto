from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindowedSeries:
    """Windowed representation of a single univariate series.

    X shape: [n_windows, seq_len, n_features]
    map_ds_idx shape: [n_windows, seq_len] mapping to original ds index (0..T-1)
    """

    X: np.ndarray
    map_ds_idx: np.ndarray


def make_sliding_windows(
    y: np.ndarray,
    seq_len: int,
    stride: int,
    n_features: int = 1,
) -> WindowedSeries:
    """Convert a 1D array y (length T) into sliding windows.

    Notes
    -----
    PyPOTS expects X with shape [n_samples, n_steps, n_features].
    See API docs for predict()/detect().
    """
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}")
    if seq_len <= 1:
        raise ValueError("seq_len must be > 1")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    T = int(y.shape[0])
    if T < seq_len:
        return WindowedSeries(X=np.empty((0, seq_len, n_features), dtype=np.float32), map_ds_idx=np.empty((0, seq_len), dtype=np.int64))

    starts = list(range(0, T - seq_len + 1, stride))
    n_windows = len(starts)
    X = np.empty((n_windows, seq_len, n_features), dtype=np.float32)
    map_idx = np.empty((n_windows, seq_len), dtype=np.int64)
    for i, s in enumerate(starts):
        window = y[s : s + seq_len]
        X[i, :, 0] = window.astype(np.float32, copy=False)
        if n_features > 1:
            # if user later expands to multivariate, they should feed that directly.
            for f in range(1, n_features):
                X[i, :, f] = window.astype(np.float32, copy=False)
        map_idx[i, :] = np.arange(s, s + seq_len, dtype=np.int64)
    return WindowedSeries(X=X, map_ds_idx=map_idx)


def split_train_test_windows(X: np.ndarray, train_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Split windows into train and test (test is always all windows).

    Using the head portion for training preserves temporal causality.
    """
    if not (0.0 < train_frac <= 1.0):
        raise ValueError("train_frac must be in (0, 1]")
    n = int(X.shape[0])
    if n == 0:
        return X, X
    n_train = max(1, int(np.floor(n * train_frac)))
    X_train = X[:n_train]
    X_test = X
    return X_train, X_test

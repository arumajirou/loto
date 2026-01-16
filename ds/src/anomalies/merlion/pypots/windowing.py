# /mnt/e/env/ts/tslib/ds/src/anomalies/pypots/windowing.py
import numpy as np
import pandas as pd

class TimeSeriesWindower:
    def __init__(self, seq_len: int, stride: int = 1):
        self.seq_len = seq_len
        self.stride = stride
        self.last_ds_indices = None

    def transform(self, values: np.ndarray):
        n_steps = len(values)
        if n_steps < self.seq_len:
            return None, None
        
        indices = np.arange(0, n_steps - self.seq_len + 1, self.stride)
        windows = [values[i : i + self.seq_len] for i in indices]
        self.last_ds_indices = indices + self.seq_len - 1
        return np.array(windows, dtype=np.float32), self.last_ds_indices

    def inverse_transform_scores(self, scores: np.ndarray, original_len: int):
        full_scores = np.full(original_len, np.nan)
        if self.last_ds_indices is not None:
            full_scores[self.last_ds_indices] = scores
        return pd.Series(full_scores).ffill().bfill().values
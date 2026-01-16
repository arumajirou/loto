from __future__ import annotations

import json
import math
import random
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    created_ts: datetime


def _set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch may not be importable in some environments
        pass


def resolve_model_class(model_name: str):
    """Resolve PyPOTS anomaly_detection model class by name.

    Mapped from PyPOTS 1.0 docs modules.
    """
    import importlib

    name = str(model_name).strip()
    mapping = {
        "SAITS": "saits",
        "TimeMixerPP": "timemixerpp",
        "TEFN": "tefn",
        "ImputeFormer": "imputeformer",
        "PatchTST": "patchtst",
        "iTransformer": "itransformer",
        "Crossformer": "crossformer",
        "Pyraformer": "pyraformer",
        "FEDformer": "fedformer",
        "Informer": "informer",
        "Transformer": "transformer",
        "ETSformer": "etsformer",
        "TimeMixer": "timemixer",
        "NonstationaryTransformer": "nonstationary_transformer",
        "FiLM": "film",
        "TimesNet": "timesnet",
        "DLinear": "dlinear",
        "SegRNN": "segrnn",
        "SCINet": "scinet",
        "Autoformer": "autoformer",
        "Reformer": "reformer",
    }
    mod = mapping.get(name)
    if not mod:
        # try best-effort
        mod = name.lower()
    module = importlib.import_module(f"pypots.anomaly_detection.{mod}")
    if not hasattr(module, name):
        raise ImportError(f"Model class '{name}' not found in pypots.anomaly_detection.{mod}")
    return getattr(module, name)


def build_model_kwargs(
    *,
    model_cls,
    n_steps: int,
    n_features: int,
    anomaly_rate: float,
    device: Optional[str],
    common: Dict[str, Any],
    model_specific: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct kwargs respecting the model signature."""
    import inspect

    sig = inspect.signature(model_cls.__init__)
    accepted = set(sig.parameters.keys())
    accepted.discard("self")

    kwargs: Dict[str, Any] = {}
    # required basics
    for k, v in {
        "n_steps": int(n_steps),
        "n_features": int(n_features),
        "anomaly_rate": float(anomaly_rate),
    }.items():
        if k in accepted:
            kwargs[k] = v

    # common training args
    for k, v in common.items():
        if k in accepted:
            kwargs[k] = v

    if device is not None and "device" in accepted:
        kwargs["device"] = device

    # model-specific args
    for k, v in model_specific.items():
        if k in accepted:
            kwargs[k] = v

    # --- heuristics for common required params (so config can stay minimal) ---
    if model_cls.__name__ == "DLinear":
        if "moving_avg_window_size" in accepted and "moving_avg_window_size" not in kwargs:
            # common default in literature; bounded by seq length
            kwargs["moving_avg_window_size"] = int(max(3, min(25, max(3, n_steps // 2))))
        if "individual" in accepted and "individual" not in kwargs:
            kwargs["individual"] = False

    # verify required params without defaults are provided
    missing_required = []
    for pname, p in sig.parameters.items():
        if pname == "self":
            continue
        if p.default is inspect._empty and pname not in kwargs:
            missing_required.append(pname)
    if missing_required:
        raise ValueError(
            f"Missing required params for {model_cls.__name__}: {missing_required}. "
            "Provide them in [pypots.model.<ModelName>] section."
        )
    return kwargs


def fit_and_score_windows(
    detector,
    *,
    train_set: Dict[str, Any],
    test_set: Dict[str, Any],
    file_type: str = "hdf5",
    amp_enabled: bool = False,
    extra_forward_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float]:
    """Fit detector and compute continuous anomaly score per window-step.

    Returns
    -------
    test_energy_flat : np.ndarray, shape [n_test_windows * n_steps]
    threshold : float

    Notes
    -----
    PyPOTS BaseNNDetector.predict computes a per-timestep reconstruction MSE score,
    flattens it, then sets a percentile threshold using anomaly_rate.
    We replicate that but keep the continuous score.
    (See BaseNNDetector.predict in PyPOTS source.)
    """
    import torch
    from torch.cuda.amp import autocast
    from torch.utils.data import DataLoader
    from pypots.data.dataset.base import BaseDataset

    extra_forward_kwargs = extra_forward_kwargs or {}

    detector.fit(train_set)

    detector.model.eval()
    train_dataset = BaseDataset(
        detector.train_set,
        return_X_ori=False,
        return_X_pred=False,
        return_y=False,
        file_type=file_type,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=detector.batch_size,
        shuffle=False,
        num_workers=getattr(detector, "num_workers", 0),
    )
    test_dataset = BaseDataset(
        test_set,
        return_X_ori=False,
        return_X_pred=False,
        return_y=False,
        file_type=file_type,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=detector.batch_size,
        shuffle=False,
        num_workers=getattr(detector, "num_workers", 0),
    )

    mse = torch.nn.MSELoss(reduction="none")

    def collect_energy(loader) -> np.ndarray:
        collector = []
        for data in loader:
            inputs = detector._assemble_input_for_testing(data)
            with autocast(enabled=bool(amp_enabled)):
                results = detector.model(inputs, **extra_forward_kwargs)
            outputs = results["reconstruction"]
            score = torch.mean(mse(inputs["X"], outputs), dim=-1)  # [B, n_steps]
            collector.append(score.detach().cpu().numpy())
        if not collector:
            return np.array([], dtype=np.float32)
        arr = np.concatenate(collector, axis=0)
        return arr.reshape(-1)

    train_energy = collect_energy(train_loader)
    test_energy = collect_energy(test_loader)
    if train_energy.size == 0 and test_energy.size == 0:
        return np.array([], dtype=np.float32), float("nan")
    combined = np.concatenate([train_energy, test_energy], axis=0) if train_energy.size else test_energy

    # detector.anomaly_rate is required to be in (0,1)
    anomaly_rate = float(getattr(detector, "anomaly_rate"))
    threshold = float(np.percentile(combined, 100 - anomaly_rate * 100))
    return test_energy.astype(np.float32), threshold


def process_one_series(
    *,
    series_df: pd.DataFrame,
    run_meta: RunMeta,
    model_name: str,
    pipeline: str,
    seq_len: int,
    stride: int,
    train_frac: float,
    anomaly_rate: float,
    device: Optional[str],
    seed: int,
    hist_prefix: str,
    lib_prefix: str,
    sep: str,
    include_pipeline: bool,
    save_mode: str,
    pypots_common_kwargs: Dict[str, Any],
    pypots_model_kwargs: Dict[str, Any],
    amp_enabled: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Run PyPOTS anomaly detection for one (loto, unique_id, ts_type) series."""
    try:
        from .naming import feature_name  # type: ignore
        from .windowing import make_sliding_windows, split_train_test_windows  # type: ignore
    except Exception:  # running as a plain script (no package context)
        from naming import feature_name  # type: ignore
        from windowing import make_sliding_windows, split_train_test_windows  # type: ignore

    _set_reproducible_seed(seed)

    # defensive copy + sorting
    df = series_df[["loto", "unique_id", "ts_type", "ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    if df.empty:
        raise ValueError("empty series after ds parsing")

    y = df["y"].astype(float).to_numpy()
    T = int(y.shape[0])

    windowed = make_sliding_windows(y=y, seq_len=seq_len, stride=stride, n_features=1)
    X = windowed.X
    map_idx = windowed.map_ds_idx
    if X.shape[0] == 0:
        raise ValueError(f"series too short for seq_len={seq_len} (T={T})")

    X_train, X_test = split_train_test_windows(X, train_frac=train_frac)
    train_set = {"X": X_train}
    test_set = {"X": X_test}

    model_cls = resolve_model_class(model_name)
    kwargs = build_model_kwargs(
        model_cls=model_cls,
        n_steps=seq_len,
        n_features=1,
        anomaly_rate=anomaly_rate,
        device=device,
        common=pypots_common_kwargs,
        model_specific=pypots_model_kwargs,
    )

    detector = model_cls(**kwargs)

    test_energy_flat, threshold = fit_and_score_windows(
        detector,
        train_set=train_set,
        test_set=test_set,
        amp_enabled=amp_enabled,
    )
    if test_energy_flat.size == 0 or not np.isfinite(threshold):
        raise RuntimeError("failed to compute anomaly score/threshold")

    # mapping to original ds index
    map_flat = map_idx.reshape(-1)
    if map_flat.shape[0] != test_energy_flat.shape[0]:
        raise RuntimeError(
            f"mapping size mismatch: map={map_flat.shape[0]} vs score={test_energy_flat.shape[0]}"
        )

    # aggregate to per-timestamp score (mean across overlapping windows)
    sum_score = np.bincount(map_flat, weights=test_energy_flat, minlength=T).astype(np.float64)
    cnt = np.bincount(map_flat, minlength=T).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        score_ts = (sum_score / cnt).astype(np.float32)
    score_ts[cnt == 0] = np.nan
    is_anom_ts = (score_ts > threshold) & np.isfinite(score_ts)

    # output columns
    col_is = feature_name(
        "is_anomaly",
        hist_prefix=hist_prefix,
        lib_prefix=lib_prefix,
        pipeline=pipeline,
        sep=sep,
        include_pipeline=include_pipeline,
    )
    col_score = feature_name(
        "anomaly_score",
        hist_prefix=hist_prefix,
        lib_prefix=lib_prefix,
        pipeline=pipeline,
        sep=sep,
        include_pipeline=include_pipeline,
    )
    col_thr = feature_name(
        "threshold",
        hist_prefix=hist_prefix,
        lib_prefix=lib_prefix,
        pipeline=pipeline,
        sep=sep,
        include_pipeline=include_pipeline,
    )

    out = df.copy()
    out["run_id"] = run_meta.run_id
    out["created_ts"] = run_meta.created_ts
    out["library"] = "pypots"
    out["model"] = model_name
    out["pipeline"] = pipeline
    out[col_is] = is_anom_ts.astype(bool)
    out[col_score] = score_ts.astype(float)
    out[col_thr] = float(threshold)

    if save_mode == "full":
        sfid = f"{df.loc[0,'loto']}__{df.loc[0,'unique_id']}__{df.loc[0,'ts_type']}"
        out["sfid"] = sfid
        out["series_len"] = T
        out["ds_idx"] = np.arange(T, dtype=np.int64)
        out["model_params_json"] = json.dumps(kwargs, ensure_ascii=False, default=str)
        out["fit_ok"] = True

    return out, None


def safe_process_one_series(**kwargs) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Wrapper returning (result_df, failure_df)."""
    try:
        res, _ = process_one_series(**kwargs)
        return res, None
    except Exception as e:
        series_df: pd.DataFrame = kwargs.get("series_df")
        run_meta: RunMeta = kwargs.get("run_meta")
        model_name = kwargs.get("model_name")
        pipeline = kwargs.get("pipeline")
        df0 = series_df.iloc[0] if series_df is not None and not series_df.empty else {}
        sfid = f"{df0.get('loto','?')}__{df0.get('unique_id','?')}__{df0.get('ts_type','?')}"
        fail = {
            "run_id": getattr(run_meta, "run_id", None),
            "created_ts": getattr(run_meta, "created_ts", datetime.now(timezone.utc)),
            "sfid": sfid,
            "model": model_name,
            "pipeline": pipeline,
            "reason": type(e).__name__,
            "error_msg": str(e),
            "traceback": traceback.format_exc(limit=20),
            "series_len": int(len(series_df)) if series_df is not None else None,
        }
        return None, pd.DataFrame([fail])

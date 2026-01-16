# /mnt/e/env/ts/tslib/ds/src/anomalies/merlion/merlion_models.py
from __future__ import annotations

import importlib
import inspect
from typing import Any, Tuple


MODEL_IMPORTS: dict[str, tuple[str, str, str]] = {
    "DefaultDetector": ("merlion.models.defaults", "DefaultDetectorConfig", "DefaultDetector"),
    "SpectralResidual": ("merlion.models.anomaly.spectral_residual", "SpectralResidualConfig", "SpectralResidual"),
    "ZMS": ("merlion.models.anomaly.zms", "ZMSConfig", "ZMS"),
    "WindStats": ("merlion.models.anomaly.windstats", "WindStatsConfig", "WindStats"),
    "IsolationForest": ("merlion.models.anomaly.isolation_forest", "IsolationForestConfig", "IsolationForest"),
    # 必要なら追加（JDKが要るモデルもあるので注意）
}


def _filter_kwargs_for_ctor(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Configの __init__ シグネチャに存在するキーだけ残す（未知キーで落ちるのを防ぐ）。
    ただし **kwargs を持つConfigはフィルタしない。
    """
    try:
        sig = inspect.signature(cls.__init__)
    except Exception:
        return kwargs

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs

    allowed = {k for k in params.keys() if k not in ("self",)}
    return {k: v for k, v in kwargs.items() if k in allowed}


def resolve_model(name: str):
    if name not in MODEL_IMPORTS:
        raise ValueError(f"Unsupported Merlion model name: {name}. Add it to MODEL_IMPORTS.")
    mod, cfg_cls_name, model_cls_name = MODEL_IMPORTS[name]
    m = importlib.import_module(mod)
    cfg_cls = getattr(m, cfg_cls_name)
    model_cls = getattr(m, model_cls_name)
    return cfg_cls, model_cls


def build_model(name: str, params: dict[str, Any]):
    cfg_cls, model_cls = resolve_model(name)
    safe_params = _filter_kwargs_for_ctor(cfg_cls, params)
    cfg = cfg_cls(**safe_params)
    model = model_cls(cfg)
    return model, safe_params

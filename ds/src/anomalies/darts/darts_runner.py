# /mnt/e/env/ts/tslib/ds/src/anomalies/darts/darts_runner.py
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# ---- Darts imports（失敗したら環境問題） ----
try:
    from darts import TimeSeries
    from darts.ad import (
        ForecastingAnomalyModel,
        QuantileDetector,
        ThresholdDetector,
        NormScorer,
        KMeansScorer,
    )
    from darts.models import LinearRegressionModel
except Exception as e:
    raise SystemExit(
        f"\n[ImportError] darts import失敗: {e}\n"
        "例: pip install -U u8darts\n"
        "（深層学習モデルまで使うなら: pip install -U 'u8darts[torch]' など）\n"
    )


MODEL_FACTORY = {
    "LinearRegressionModel": LinearRegressionModel,
}

SCORER_FACTORY = {
    "NormScorer": NormScorer,
    "KMeansScorer": KMeansScorer,
}

DETECTOR_FACTORY = {
    "QuantileDetector": QuantileDetector,
    "ThresholdDetector": ThresholdDetector,
}


def _ts_to_df(ts: TimeSeries, value_name: str) -> pd.DataFrame:
    """
    Darts TimeSeries -> DataFrame (index=ds_idx, col=value_name)
    Dartsは to_dataframe() が推奨。:contentReference[oaicite:3]{index=3}
    """
    df = ts.to_dataframe()
    # 1列想定（univariate）
    col = df.columns[0]
    out = df.rename(columns={col: value_name})
    out = out.reset_index().rename(columns={out.index.name or "index": "ds_idx"})
    return out


def _make_series_from_values(y: np.ndarray) -> TimeSeries:
    """
    RangeIndex(0..n-1) のTimeSeriesを作る（周波数推定の事故を回避しやすい）。
    from_values() はRangeIndexを生成。:contentReference[oaicite:4]{index=4}
    """
    y2 = np.asarray(y, dtype=np.float32)
    return TimeSeries.from_values(y2)


def _resolve_start(start_cfg: Any, lags: int, n: int) -> Any:
    """
    start は Darts 仕様で int/float/timestamp 等を許す。
    ここでは RangeIndex 前提なので:
      - "auto": lags を採用（最低限の履歴確保）
      - int: そのまま
      - float: そのまま（割合start）
    """
    if start_cfg is None:
        return lags
    if isinstance(start_cfg, str) and start_cfg.lower() == "auto":
        return lags
    return start_cfg


def run_one_pipeline_on_one_series(
    *,
    y: np.ndarray,
    base_df: pd.DataFrame,   # ds_idx, y, ds_real, loto, unique_id, ts_type
    pipeline_name: str,
    model_name: str,
    model_params: Dict[str, Any],
    scorer_name: str,
    scorer_params: Dict[str, Any],
    detector_name: str,
    detector_params: Dict[str, Any],
    detector_fit_on: str,    # all|train
    train_frac: float,
    forecast_horizon: int,
    start_cfg: Any,
    verbose: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    予測モデル＋スコアラーで anomaly score を作り、Detectorで二値化。
    Darts公式の構成（AnomalyModel/Scorer/Detector）。:contentReference[oaicite:5]{index=5}
    """
    # ---- build model ----
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"unknown model: {model_name}")
    if scorer_name not in SCORER_FACTORY:
        raise ValueError(f"unknown scorer: {scorer_name}")
    if detector_name not in DETECTOR_FACTORY:
        raise ValueError(f"unknown detector: {detector_name}")

    # LinearRegressionModelは lags が実質必須
    lags = int(model_params.get("lags", 14))
    # 予測モデル
    forecasting_model = MODEL_FACTORY[model_name](**model_params)

    # スコアラー（例: NormScorer は予測と実測のノルム差分を返す）:contentReference[oaicite:6]{index=6}
    scorer = SCORER_FACTORY[scorer_name](**scorer_params)

    # AnomalyModel（ForecastingAnomalyModel は score(return_model_prediction=True) で予測も返せる）:contentReference[oaicite:7]{index=7}
    anomaly_model = ForecastingAnomalyModel(model=forecasting_model, scorer=[scorer])

    ts = _make_series_from_values(y)
    start = _resolve_start(start_cfg, lags=lags, n=len(y))

    # fit（allow_model_training=True で中の予測モデルも学習する）:contentReference[oaicite:8]{index=8}
    anomaly_model.fit(
        ts,
        allow_model_training=True,
        forecast_horizon=int(forecast_horizon),
        start=start,
        verbose=verbose,
    )

    score_out = anomaly_model.score(
        ts,
        forecast_horizon=int(forecast_horizon),
        start=start,
        return_model_prediction=True,
        verbose=verbose,
    )

    # blog/公式例では (anomaly_scores, model_forecasting) のタプル:contentReference[oaicite:9]{index=9}
    if isinstance(score_out, tuple) and len(score_out) == 2:
        scores_ts, pred_ts = score_out
    else:
        scores_ts, pred_ts = score_out, None

    # scorerをリストで渡してるので scores_ts は list[TimeSeries] の可能性がある
    if isinstance(scores_ts, list):
        # まずは1本目だけ（必要なら将来multi対応）
        scores_ts = scores_ts[0]

    score_df = _ts_to_df(scores_ts, "anomaly_score")

    pred_df = None
    if pred_ts is not None:
        # 予測系列（y_hat）も保存（可視化/検証に便利）
        if isinstance(pred_ts, list):
            pred_ts = pred_ts[0]
        pred_df = _ts_to_df(pred_ts, "y_hat")

    # ---- Detectorで二値化 ----
    detector = DETECTOR_FACTORY[detector_name](**detector_params)

    if detector_fit_on == "train":
        # score_dfの先頭 train_frac でfit（素朴だが再現性は高い）
        nfit = max(1, int(len(score_df) * float(train_frac)))
        fit_part = score_df.iloc[:nfit]
        fit_ts = TimeSeries.from_times_and_values(
            times=pd.Index(fit_part["ds_idx"].astype(int)),
            values=fit_part["anomaly_score"].to_numpy(dtype=np.float32),
        )
        detector.fit(fit_ts)
    else:
        # 全期間でfit
        detector.fit(scores_ts)

    binary_ts = detector.detect(scores_ts)
    bin_df = _ts_to_df(binary_ts, "is_anomaly")

    # Detectorは 0/1 のことが多いので bool化
    bin_df["is_anomaly"] = bin_df["is_anomaly"].astype(float).fillna(0.0) > 0.0

    # ---- merge（全dsを残す：scoreが無い初期点はNULLのまま）----
    out = base_df.merge(score_df, on="ds_idx", how="left")
    if pred_df is not None:
        out = out.merge(pred_df, on="ds_idx", how="left")
    else:
        out["y_hat"] = np.nan
    out = out.merge(bin_df, on="ds_idx", how="left")
    out["is_anomaly"] = out["is_anomaly"].astype("boolean").fillna(False).astype(bool)

    # 付帯メタ
    out["library"] = "darts"
    out["pipeline"] = pipeline_name
    out["model"] = model_name
    out["scorer"] = scorer_name
    out["detector"] = detector_name
    out["forecast_horizon"] = int(forecast_horizon)
    out["start"] = start
    out["train_frac"] = float(train_frac)

    meta = {
        "pipeline": pipeline_name,
        "model": {"name": model_name, "params": model_params},
        "scorer": {"name": scorer_name, "params": scorer_params},
        "detector": {"name": detector_name, "params": detector_params, "fit_on": detector_fit_on, "train_frac": float(train_frac)},
        "forecast_horizon": int(forecast_horizon),
        "start": start,
    }
    return out, meta

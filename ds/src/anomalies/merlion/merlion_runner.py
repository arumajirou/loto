# /mnt/e/env/ts/tslib/ds/src/anomalies/merlion/merlion_runner.py
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from merlion.utils import TimeSeries  # Merlion公式の基本データ型 :contentReference[oaicite:5]{index=5}

from merlion_models import build_model


def sanitize(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


@dataclass(frozen=True)
class SeriesPayload:
    loto: str
    unique_id: str
    ts_type: str
    ds: np.ndarray  # datetime64[ns]
    y: np.ndarray   # float


def _ts_from_arrays(ds: np.ndarray, y: np.ndarray) -> TimeSeries:
    idx = pd.DatetimeIndex(ds)
    s = pd.Series(y, index=idx).sort_index()
    # 時刻重複があれば最後を採用（Merlionは重複に厳しい設定のことがある）
    s = s[~s.index.duplicated(keep="last")]
    return TimeSeries.from_pd(s, check_times=True, drop_nan=True)


def _to_1d_pd(ts: TimeSeries) -> pd.Series:
    x = ts.to_pd()
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x


def run_one_series_wide(
    payload: SeriesPayload,
    model_names: List[str],
    model_params_map: Dict[str, Dict[str, Any]],
    feature_prefix_template: str,
    library: str,
    train_frac: float,
    min_len: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    戻り：
      - wide_df: 1系列×全時点 の wide 形式（モデル別特徴量列）
      - meta: failure用情報
    """
    n = len(payload.y)
    if n < min_len:
        raise ValueError(f"too_short<{min_len}")

    ts_full = _ts_from_arrays(payload.ds, payload.y)
    # train/test split index（時系列の順序に依存）
    # TimeSeriesは時系列順に整列済みなので、pdに戻して分割するのが確実
    full_pd = _to_1d_pd(ts_full)
    times = full_pd.index
    y_sorted = full_pd.values.astype(float)
    n2 = len(y_sorted)

    split = int(max(1, np.floor(n2 * float(train_frac))))
    split = min(split, n2 - 1) if n2 >= 2 else 1

    ts_train = TimeSeries.from_pd(pd.Series(y_sorted[:split], index=times[:split]), check_times=True, drop_nan=True)
    ts_test  = TimeSeries.from_pd(pd.Series(y_sorted[split:],  index=times[split:]),  check_times=True, drop_nan=True)

    base = pd.DataFrame({
        "loto": payload.loto,
        "unique_id": payload.unique_id,
        "ts_type": payload.ts_type,
        "ds": times,
    })

    failures: list[dict[str, Any]] = []
    model_meta: list[dict[str, Any]] = []

    feat_prefix = feature_prefix_template.format(library=library)

    for mn in model_names:
        safe_mn = sanitize(mn)
        params = model_params_map.get(mn, {})
        try:
            model, used_params = build_model(mn, params)
            # train() は train上の生スコアを返す :contentReference[oaicite:6]{index=6}
            train_scores_ts = model.train(train_data=ts_train)
            train_raw = _to_1d_pd(train_scores_ts)

            # post_rule は「校正＋閾値」などの後処理（callable） :contentReference[oaicite:7]{index=7}
            train_pp = _to_1d_pd(model.post_rule(train_scores_ts))

            test_raw_ts = model.get_anomaly_score(time_series=ts_test, time_series_prev=ts_train)
            test_pp_ts  = model.get_anomaly_label(time_series=ts_test, time_series_prev=ts_train)

            test_raw = _to_1d_pd(test_raw_ts)
            test_pp  = _to_1d_pd(test_pp_ts)

            score_raw = pd.Series(index=times, dtype=float)
            score_pp  = pd.Series(index=times, dtype=float)
            score_raw.loc[train_raw.index] = train_raw.values
            score_pp.loc[train_pp.index] = train_pp.values
            score_raw.loc[test_raw.index] = test_raw.values
            score_pp.loc[test_pp.index] = test_pp.values

            # Merlionのlabelは「非異常=0、異常=非0」になる想定 :contentReference[oaicite:8]{index=8}
            is_anom = (score_pp.fillna(0.0) != 0.0).astype(bool)

            base[f"{feat_prefix}{safe_mn}__score_raw"] = score_raw.values
            base[f"{feat_prefix}{safe_mn}__score_pp"]  = score_pp.values
            base[f"{feat_prefix}{safe_mn}__is_anomaly"] = is_anom.values

            model_meta.append({"model": mn, "params": used_params})

        except Exception as e:
            failures.append({"model": mn, "error": str(e)})
            # 欠損列として入れる（JOINしやすくする）
            base[f"{feat_prefix}{safe_mn}__score_raw"] = np.nan
            base[f"{feat_prefix}{safe_mn}__score_pp"]  = np.nan
            base[f"{feat_prefix}{safe_mn}__is_anomaly"] = False
            model_meta.append({"model": mn, "params": params, "failed": True, "error": str(e)})

    meta = {
        "series_len": int(n2),
        "model_params_json": json.dumps(model_meta, ensure_ascii=False),
        "failures": failures,
    }
    return base, meta

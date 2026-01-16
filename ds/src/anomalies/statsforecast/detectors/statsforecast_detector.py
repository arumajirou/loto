import numpy as np
import pandas as pd

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoCES, AutoTheta, MSTL

MODEL_FACTORY = {
    "AutoARIMA": AutoARIMA,
    "AutoETS": AutoETS,
    "AutoCES": AutoCES,
    "AutoTheta": AutoTheta,
    "MSTL": MSTL,
}

def run_statsforecast_on_series(
    *,
    sfid: str,
    ds_idx: np.ndarray,
    ds_real: np.ndarray,
    y: np.ndarray,
    loto: str,
    unique_id: str,
    ts_type: str,
    model_specs: list[dict],
    level: int,
    horizon: int,
    run_id: str,
    created_ts,
    save_only_anomalies: bool,
) -> tuple[list[dict], list[dict]]:
    """
    1系列に対して、複数モデルを順に試す（モデル失敗を隔離）。
    戻り値: (rows, failures)
    """
    rows: list[dict] = []
    fails: list[dict] = []

    base_map = pd.DataFrame({"ds": ds_idx.astype(int), "ds_real": ds_real})
    ser_df = pd.DataFrame({"unique_id": sfid, "ds": ds_idx.astype(int), "y": y.astype(float)})

    for spec in model_specs:
        mn = spec["model"]
        params = spec.get("params", {}) or {}

        if mn not in MODEL_FACTORY:
            fails.append({
                "run_id": run_id, "created_ts": created_ts,
                "library": "statsforecast", "model": mn,
                "sf_unique_id": sfid, "reason": "unknown_model",
                "error_msg": f"MODEL_FACTORYに存在しません: {mn}",
                "series_len": int(len(y)),
            })
            continue

        # MSTLはseason_length必須寄りなので保険
        if mn == "MSTL" and "season_length" not in params:
            params["season_length"] = [7]

        try:
            model = MODEL_FACTORY[mn](**params)
            sf = StatsForecast(models=[model], freq=1, n_jobs=1)

            _ = sf.forecast(df=ser_df, h=int(horizon), level=[int(level)], fitted=True)
            ins = sf.forecast_fitted_values()

            pred_col = mn
            lo_col = f"{mn}-lo-{int(level)}"
            hi_col = f"{mn}-hi-{int(level)}"

            if pred_col not in ins.columns or lo_col not in ins.columns or hi_col not in ins.columns:
                raise RuntimeError(f"missing interval cols: {pred_col},{lo_col},{hi_col}")

            tmp = ins[["ds", "y", pred_col, lo_col, hi_col]].copy()
            tmp = tmp.merge(base_map, on="ds", how="left")

            tmp["is_anomaly"] = ~tmp["y"].between(tmp[lo_col], tmp[hi_col])

            width = (tmp[hi_col] - tmp[lo_col]).replace(0, np.nan)
            below = (tmp[lo_col] - tmp["y"]) / width
            above = (tmp["y"] - tmp[hi_col]) / width
            tmp["anomaly_score"] = np.where(tmp["y"] < tmp[lo_col], below,
                                     np.where(tmp["y"] > tmp[hi_col], above, 0.0))
            tmp["anomaly_side"] = np.where(tmp["y"] < tmp[lo_col], "low",
                                    np.where(tmp["y"] > tmp[hi_col], "high", "none"))

            tmp = tmp.rename(columns={pred_col: "y_hat", lo_col: "y_lo", hi_col: "y_hi"})
            if save_only_anomalies:
                tmp = tmp[tmp["is_anomaly"]]

            for r in tmp.itertuples(index=False):
                rows.append({
                    "loto": loto,
                    "unique_id": unique_id,
                    "ts_type": ts_type,
                    "ds": pd.Timestamp(r.ds_real),
                    "y": float(r.y) if r.y is not None else None,
                    "y_hat": float(r.y_hat) if r.y_hat is not None else None,
                    "y_lo": float(r.y_lo) if r.y_lo is not None else None,
                    "y_hi": float(r.y_hi) if r.y_hi is not None else None,
                    "is_anomaly": bool(r.is_anomaly),
                    "anomaly_side": str(r.anomaly_side),
                    "anomaly_score": float(r.anomaly_score) if r.anomaly_score is not None else None,
                    "library": "statsforecast",
                    "model": mn,
                    "level": float(level),
                    "fit_ok": True,
                    "error_msg": None,
                    "series_len": int(len(y)),
                    "run_id": run_id,
                    "created_ts": created_ts,
                })

        except Exception as e:
            fails.append({
                "run_id": run_id, "created_ts": created_ts,
                "library": "statsforecast", "model": mn,
                "sf_unique_id": sfid, "reason": "fit_failed",
                "error_msg": str(e),
                "series_len": int(len(y)),
            })

    return rows, fails

import numpy as np
import pandas as pd

def _try_get_xp(use_gpu: bool):
    if not use_gpu:
        return np, "numpy"
    try:
        import cupy as cp  # optional
        # CUDAが無い/壊れてる時は例外になり得るので握る
        _ = cp.zeros((1,))
        return cp, "cupy"
    except Exception:
        return np, "numpy"

def hampel_detect(
    *,
    sfid: str,
    ds_real: np.ndarray,
    y: np.ndarray,
    loto: str,
    unique_id: str,
    ts_type: str,
    window: int,
    k: float,
    use_gpu: bool,
    run_id: str,
    created_ts,
    save_only_anomalies: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Hampel: 移動中央値±k*MAD（MAD=中央値絶対偏差）
    StatsForecastが不安定な系列でも必ず動く“保険”。
    GPU(CuPy)は簡易対応（使えなければ自動でNumPyにフォールバック）。
    """
    rows: list[dict] = []
    fails: list[dict] = []

    xp, backend = _try_get_xp(use_gpu)

    try:
        yv = xp.asarray(y, dtype=xp.float64)
        n = int(yv.shape[0])
        w = max(1, int(window))
        half = w // 2

        med = xp.empty(n, dtype=xp.float64)
        lo = xp.empty(n, dtype=xp.float64)
        hi = xp.empty(n, dtype=xp.float64)

        # rolling medianは汎用にループ（GPUでも動くが高速とは限らない）
        for i in range(n):
            a = max(0, i - half)
            b = min(n, i + half + 1)
            x = yv[a:b]
            m = xp.nanmedian(x)
            mad = xp.nanmedian(xp.abs(x - m))
            sigma = 1.4826 * mad
            med[i] = m
            lo[i] = m - k * sigma
            hi[i] = m + k * sigma

        is_anom = xp.logical_not(xp.logical_and(yv >= lo, yv <= hi))
        width = (hi - lo)
        width = xp.where(width == 0, xp.nan, width)
        score = xp.where(yv < lo, (lo - yv) / width, xp.where(yv > hi, (yv - hi) / width, 0.0))

        # cupy -> numpyへ戻す
        if backend == "cupy":
            import cupy as cp
            med = cp.asnumpy(med); lo = cp.asnumpy(lo); hi = cp.asnumpy(hi)
            is_anom = cp.asnumpy(is_anom); score = cp.asnumpy(score)
        else:
            med = np.asarray(med); lo = np.asarray(lo); hi = np.asarray(hi)
            is_anom = np.asarray(is_anom); score = np.asarray(score)

        for i in range(len(y)):
            if save_only_anomalies and (not bool(is_anom[i])):
                continue
            side = "low" if y[i] < lo[i] else ("high" if y[i] > hi[i] else "none")
            rows.append({
                "loto": loto,
                "unique_id": unique_id,
                "ts_type": ts_type,
                "ds": pd.Timestamp(ds_real[i]),
                "y": float(y[i]),
                "y_hat": float(med[i]) if np.isfinite(med[i]) else None,
                "y_lo": float(lo[i]) if np.isfinite(lo[i]) else None,
                "y_hi": float(hi[i]) if np.isfinite(hi[i]) else None,
                "is_anomaly": bool(is_anom[i]),
                "anomaly_side": side,
                "anomaly_score": float(score[i]) if np.isfinite(score[i]) else None,
                "library": "hampel",
                "model": "Hampel",
                "level": None,
                "fit_ok": True,
                "error_msg": None,
                "series_len": int(len(y)),
                "run_id": run_id,
                "created_ts": created_ts,
            })

        return rows, fails

    except Exception as e:
        fails.append({
            "run_id": run_id, "created_ts": created_ts,
            "library": "hampel", "model": "Hampel",
            "sf_unique_id": sfid, "reason": "hampel_failed",
            "error_msg": str(e),
            "series_len": int(len(y)),
        })
        return [], fails

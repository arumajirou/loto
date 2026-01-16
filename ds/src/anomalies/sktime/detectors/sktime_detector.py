import importlib
import numpy as np
import pandas as pd

def _import_detector_class(name_or_path: str):
    name = name_or_path.strip()

    SHORT = {
        "STRAY": "sktime.detection.stray.STRAY",
        "SubLOF": "sktime.detection.lof.SubLOF",
        "PyODDetector": "sktime.detection.adapters.PyODDetector",
    }

    path = SHORT.get(name, name)
    if "." not in path:
        raise ImportError(f"Detectorの指定が不正です: {name_or_path}")

    mod_path, cls_name = path.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)

def _coerce_labels(labels, n: int) -> np.ndarray:
    if isinstance(labels, pd.DataFrame):
        if "labels" in labels.columns:
            arr = labels["labels"].to_numpy()
        else:
            arr = labels.iloc[:, 0].to_numpy()
    elif isinstance(labels, pd.Series):
        arr = labels.to_numpy()
    else:
        arr = np.asarray(labels)

    arr = np.ravel(arr)
    if arr.size != n:
        raise ValueError(f"labels length mismatch: got={arr.size}, expected={n}")

    if arr.dtype == bool:
        return arr

    arr = arr.astype(float)
    arr = np.where(np.isnan(arr), 0.0, arr)
    return arr != 0.0

def _try_scores(detector, X):
    for meth in ("fit_transform_scores", "transform_scores", "predict_scores"):
        if hasattr(detector, meth):
            try:
                out = getattr(detector, meth)(X)
                if isinstance(out, pd.DataFrame):
                    for c in ("scores", "score", "anomaly_score", "label_score"):
                        if c in out.columns:
                            return out[c].to_numpy().astype(float)
                    return out.iloc[:, 0].to_numpy().astype(float)
                if isinstance(out, pd.Series):
                    return out.to_numpy().astype(float)
                return np.asarray(out, dtype=float)
            except Exception:
                continue
    return None

def _apply_sublof_safety(y: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    """
    SubLOF向け安全策：
    - auto_tune_neighbors: ユニーク数に合わせて n_neighbors を下げる（無理な近傍を防ぐ）
    - jitter_eps: 重複が多い場合に微小ノイズを加えてタイを崩す
    """
    p = dict(params) if params else {}

    auto_tune = bool(p.pop("auto_tune_neighbors", False))
    jitter_eps = p.pop("jitter_eps", None)

    y2 = y.astype(float, copy=True)
    uniq = int(pd.Series(y2).nunique(dropna=True))
    n = int(len(y2))

    if auto_tune:
        # LOF系は「近傍数 < ユニーク数」が現実的
        nn = int(p.get("n_neighbors", 20))
        nn2 = max(5, min(nn, max(5, uniq - 1)))
        p["n_neighbors"] = nn2

    # 重複がかなり多い場合だけジッタ（極小）
    if jitter_eps is not None:
        try:
            eps = float(jitter_eps)
            if eps > 0 and uniq < max(10, int(0.2 * n)):  # ユニーク率が低いとき
                rng = np.random.default_rng(0)
                y2 = y2 + rng.normal(0.0, eps, size=n)
        except Exception:
            pass

    # window_size が n_neighbors より小さいと不安定になりやすいので保険
    if "window_size" in p and "n_neighbors" in p:
        try:
            ws = int(p["window_size"])
            nn = int(p["n_neighbors"])
            if ws <= nn:
                p["window_size"] = max(ws, nn * 4)
        except Exception:
            pass

    return y2, p

def run_sktime_on_series(
    *,
    sfid: str,
    ds_real: np.ndarray,
    y: np.ndarray,
    loto: str,
    unique_id: str,
    ts_type: str,
    model_specs: list[dict],
    run_id: str,
    created_ts,
    save_only_anomalies: bool,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    fails: list[dict] = []

    n = int(len(y))
    X_base = pd.DataFrame({"y": y.astype(float)})

    for spec in model_specs:
        mn = spec["model"]
        params = spec.get("params", {}) or {}

        try:
            Det = _import_detector_class(mn)
        except Exception as e:
            fails.append({
                "run_id": run_id, "created_ts": created_ts,
                "library": "sktime", "model": mn,
                "sf_unique_id": sfid, "reason": "import_failed",
                "error_msg": str(e), "series_len": n,
            })
            continue

        try:
            X = X_base
            p = params

            # SubLOFだけ安全策（重複警告の実害を下げる）
            if mn.endswith("SubLOF") or mn == "SubLOF":
                y2, p2 = _apply_sublof_safety(y, params)
                X = pd.DataFrame({"y": y2})
                p = p2

            det = Det(**p)

            if hasattr(det, "fit_transform"):
                labels = det.fit_transform(X)
            else:
                det.fit(X)
                labels = det.transform(X)

            lab = _coerce_labels(labels, n)

            scores = _try_scores(det, X)
            if scores is None or np.asarray(scores).size != n:
                scores = lab.astype(float)

            for i in range(n):
                if save_only_anomalies and (not bool(lab[i])):
                    continue
                rows.append({
                    "loto": loto,
                    "unique_id": unique_id,
                    "ts_type": ts_type,
                    "ds": pd.Timestamp(ds_real[i]),
                    "y": float(y[i]),
                    "y_hat": None,
                    "y_lo": None,
                    "y_hi": None,
                    "is_anomaly": bool(lab[i]),
                    "anomaly_side": "none",
                    "anomaly_score": float(scores[i]) if np.isfinite(scores[i]) else None,
                    "library": "sktime",
                    "model": mn if "." not in mn else mn.split(".")[-1],
                    "level": None,
                    "fit_ok": True,
                    "error_msg": None,
                    "series_len": n,
                    "run_id": run_id,
                    "created_ts": created_ts,
                })

        except Exception as e:
            fails.append({
                "run_id": run_id, "created_ts": created_ts,
                "library": "sktime", "model": mn,
                "sf_unique_id": sfid, "reason": "fit_failed",
                "error_msg": str(e), "series_len": n,
            })

    return rows, fails

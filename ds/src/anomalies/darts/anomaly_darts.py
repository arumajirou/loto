# /mnt/e/env/ts/tslib/ds/src/anomalies/darts/anomaly_darts.py
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from config_utils import (
    default_ini_path,
    load_ini,
    cfg_get,
    parse_value,
)
from db_io import (
    make_pg_url,
    make_engine,
    load_loto_base,
    ensure_schema,
    save_results,
    save_failures,
    create_indexes,
)
from darts_runner import run_one_pipeline_on_one_series


def _series_key(row: pd.Series) -> str:
    return f"{row['loto']}__{row['unique_id']}__{row['ts_type']}"


def _build_base_and_groups(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    df = df.dropna(subset=["y"]).copy()

    df["sfid"] = df.apply(_series_key, axis=1)
    df = df.sort_values(["sfid", "ds"])
    df["ds_idx"] = df.groupby("sfid").cumcount().astype(int)

    lens = df.groupby("sfid")["y"].size().sort_values(ascending=False)
    series_ids = lens.index.tolist()

    base = df[["sfid", "ds_idx", "ds", "y", "loto", "unique_id", "ts_type"]].rename(columns={"ds": "ds_real"})
    return base, series_ids


def _load_pipeline_defs(cfg, enabled_names: List[str]) -> List[Dict[str, Any]]:
    out = []
    for name in enabled_names:
        sec = f"pipeline.{name}"
        if not cfg.has_section(sec):
            raise ValueError(f"missing section: [{sec}]")

        model = cfg_get(cfg, sec, "model", "LinearRegressionModel")
        model_params = cfg_get(cfg, sec, "model_params", {"lags": 14})
        scorer = cfg_get(cfg, sec, "scorer", "NormScorer")
        scorer_params = cfg_get(cfg, sec, "scorer_params", {"ord": 1, "component_wise": False})
        detector = cfg_get(cfg, sec, "detector", "QuantileDetector")
        detector_params = cfg_get(cfg, sec, "detector_params", {"high_quantile": 0.99})
        detector_fit_on = cfg_get(cfg, sec, "detector_fit_on", "all")

        if not isinstance(model_params, dict):
            raise ValueError(f"{sec}.model_params must be dict-like")
        if not isinstance(scorer_params, dict):
            raise ValueError(f"{sec}.scorer_params must be dict-like")
        if not isinstance(detector_params, dict):
            raise ValueError(f"{sec}.detector_params must be dict-like")

        out.append({
            "pipeline_name": name,
            "model_name": model,
            "model_params": model_params,
            "scorer_name": scorer,
            "scorer_params": scorer_params,
            "detector_name": detector,
            "detector_params": detector_params,
            "detector_fit_on": detector_fit_on,
        })
    return out


def _worker_run_series(payload: Dict[str, Any]) -> Tuple[List[pd.DataFrame], List[Dict[str, Any]]]:
    y = payload["y"]
    base_df = payload["base_df"]
    pipelines = payload["pipelines"]
    run_spec = payload["run_spec"]

    out_frames: List[pd.DataFrame] = []
    fails: List[Dict[str, Any]] = []

    for p in pipelines:
        try:
            out_df, _meta = run_one_pipeline_on_one_series(
                y=y,
                base_df=base_df,
                pipeline_name=p["pipeline_name"],
                model_name=p["model_name"],
                model_params=p["model_params"],
                scorer_name=p["scorer_name"],
                scorer_params=p["scorer_params"],
                detector_name=p["detector_name"],
                detector_params=p["detector_params"],
                detector_fit_on=p["detector_fit_on"],
                train_frac=run_spec["train_frac"],
                forecast_horizon=run_spec["forecast_horizon"],
                start_cfg=run_spec["start"],
                verbose=run_spec["verbose"],
            )
            out_frames.append(out_df)
        except Exception as e:
            fails.append({
                "sfid": base_df["sfid"].iloc[0],
                "pipeline": p["pipeline_name"],
                "reason": "run_failed",
                "error_msg": str(e),
                "series_len": int(len(y)),
            })

    return out_frames, fails


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", default=None)

    # source DB (dataset)
    ap.add_argument("--src-db-user", default=os.getenv("SRC_DB_USER"))
    ap.add_argument("--src-db-pass", default=os.getenv("SRC_DB_PASS"))
    ap.add_argument("--src-db-host", default=os.getenv("SRC_DB_HOST"))
    ap.add_argument("--src-db-port", default=os.getenv("SRC_DB_PORT"))
    ap.add_argument("--src-db-name", default=os.getenv("SRC_DB_NAME"))

    # target DB (anomaly)
    ap.add_argument("--out-db-user", default=os.getenv("OUT_DB_USER"))
    ap.add_argument("--out-db-pass", default=os.getenv("OUT_DB_PASS"))
    ap.add_argument("--out-db-host", default=os.getenv("OUT_DB_HOST"))
    ap.add_argument("--out-db-port", default=os.getenv("OUT_DB_PORT"))
    ap.add_argument("--out-db-name", default=os.getenv("OUT_DB_NAME"))

    ap.add_argument("--src-schema", default=None)
    ap.add_argument("--src-table", default=None)
    ap.add_argument("--where-sql", default=None)

    ap.add_argument("--out-schema", default=None)
    ap.add_argument("--out-table", default=None)
    ap.add_argument("--failures-table", default=None)
    ap.add_argument("--if-exists", default=None, choices=["replace", "append"])
    ap.add_argument("--chunksize", type=int, default=None)

    ap.add_argument("--series-limit", type=int, default=None)
    ap.add_argument("--min-len", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--parallel-backend", default=None, choices=["process", "thread"])
    ap.add_argument("--forecast-horizon", type=int, default=None)
    ap.add_argument("--start", default=None)     # "auto" / int / float
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--verbose", default=None)   # true/false
    ap.add_argument("--pipelines", default=None)

    # 追加：保存列の制御
    ap.add_argument("--save-mode", default=None, choices=["minimal", "full"])
    ap.add_argument("--feature-prefix-template", default=None)

    args = ap.parse_args()

    ini_path = default_ini_path(args.config)
    cfg = load_ini(ini_path)

    # ---- source DB ----
    src_user = args.src_db_user or cfg_get(cfg, "source_db", "user", "postgres")
    src_pass = args.src_db_pass or cfg_get(cfg, "source_db", "pass", "z")
    src_host = args.src_db_host or cfg_get(cfg, "source_db", "host", "127.0.0.1")
    src_port = args.src_db_port or cfg_get(cfg, "source_db", "port", "5432")
    src_name = args.src_db_name or cfg_get(cfg, "source_db", "name", "dataset")

    # ---- target DB ----
    out_user = args.out_db_user or cfg_get(cfg, "target_db", "user", "postgres")
    out_pass = args.out_db_pass or cfg_get(cfg, "target_db", "pass", "z")
    out_host = args.out_db_host or cfg_get(cfg, "target_db", "host", "127.0.0.1")
    out_port = args.out_db_port or cfg_get(cfg, "target_db", "port", "5432")
    out_name = args.out_db_name or cfg_get(cfg, "target_db", "name", "anomaly")

    # ---- table ----
    src_schema = args.src_schema or cfg_get(cfg, "source", "schema", "public")
    src_table  = args.src_table  or cfg_get(cfg, "source", "table", "loto_base")
    where_sql  = args.where_sql if args.where_sql is not None else cfg_get(cfg, "source", "where_sql", "")

    out_schema = args.out_schema or cfg_get(cfg, "output", "schema", "public")
    out_table  = args.out_table  or cfg_get(cfg, "output", "table", "darts")
    failures_table = args.failures_table or cfg_get(cfg, "output", "failures_table", "darts_failures")
    if_exists  = args.if_exists  or cfg_get(cfg, "output", "if_exists", "replace")
    chunksize  = args.chunksize  or cfg_get(cfg, "output", "chunksize", 50000)

    # ---- save column controls ----
    save_mode = args.save_mode or cfg_get(cfg, "output", "save_mode", "minimal")
    save_mode = str(save_mode).strip().lower()
    if save_mode not in ("minimal", "full"):
        save_mode = "minimal"

    library_name = "darts"

    # テンプレ（例: "hist_{library}__"）
    feature_prefix_template = (
        args.feature_prefix_template
        or cfg_get(cfg, "output", "feature_prefix_template", "hist_{library}__")
    )
    feature_prefix_template = str(feature_prefix_template)
    feature_prefix = feature_prefix_template.format(library=library_name)

    # ---- run ----
    series_limit = args.series_limit if args.series_limit is not None else cfg_get(cfg, "run", "series_limit", 0)
    min_len = args.min_len if args.min_len is not None else cfg_get(cfg, "run", "min_len", 50)
    n_jobs = args.n_jobs if args.n_jobs is not None else cfg_get(cfg, "run", "n_jobs", 4)
    parallel_backend = args.parallel_backend or cfg_get(cfg, "run", "parallel_backend", "process")
    forecast_horizon = args.forecast_horizon if args.forecast_horizon is not None else cfg_get(cfg, "run", "forecast_horizon", 1)
    start = args.start if args.start is not None else cfg_get(cfg, "run", "start", "auto")
    start = parse_value(start)
    train_frac = args.train_frac if args.train_frac is not None else cfg_get(cfg, "run", "train_frac", 0.8)
    verbose = args.verbose if args.verbose is not None else cfg_get(cfg, "run", "verbose", False)
    if isinstance(verbose, str):
        verbose = parse_value(verbose)

    enabled = args.pipelines or cfg_get(cfg, "pipelines", "enabled", "lr_norm_q99")
    enabled_names = [x.strip() for x in str(enabled).split(",") if x.strip()]
    pipelines = _load_pipeline_defs(cfg, enabled_names)

    # ---- DB engines ----
    src_engine = make_engine(make_pg_url(src_user, src_pass, src_host, src_port, src_name))
    out_engine = make_engine(make_pg_url(out_user, out_pass, out_host, out_port, out_name))

    print(f"SOURCE DB={src_name}  TABLE={src_schema}.{src_table}")
    print(f"TARGET DB={out_name}  TABLE={out_schema}.{out_table}")

    # ---- load ----
    raw = load_loto_base(src_engine, schema=src_schema, table=src_table, where_sql=str(where_sql))
    base, series_ids = _build_base_and_groups(raw)

    if int(series_limit) > 0:
        series_ids = series_ids[: int(series_limit)]

    lens = base.groupby("sfid")["y"].size()
    series_ids = [sid for sid in series_ids if int(lens.get(sid, 0)) >= int(min_len)]

    if not series_ids:
        raise SystemExit("対象系列が空です（series_limit/min_len/where_sql を見直してください）")

    run_id = str(uuid.uuid4())
    created_ts = datetime.now(timezone.utc)

    run_spec = {
        "forecast_horizon": int(forecast_horizon),
        "start": start,
        "train_frac": float(train_frac),
        "verbose": bool(verbose),
    }

    failures: List[Dict[str, Any]] = []
    out_frames: List[pd.DataFrame] = []

    Executor = ProcessPoolExecutor if parallel_backend == "process" else ThreadPoolExecutor
    max_workers = int(n_jobs) if int(n_jobs) > 0 else None

    payloads = []
    for sid in series_ids:
        sdf = base[base["sfid"] == sid].sort_values("ds_idx")
        y = sdf["y"].to_numpy(dtype=np.float32)
        payloads.append({
            "y": y,
            "base_df": sdf[["sfid", "ds_idx", "ds_real", "y", "loto", "unique_id", "ts_type"]].copy(),
            "pipelines": pipelines,
            "run_spec": run_spec,
        })

    with Executor(max_workers=max_workers) as ex:
        futs = [ex.submit(_worker_run_series, p) for p in payloads]
        for fut in as_completed(futs):
            series_outs, series_fails = fut.result()
            failures.extend(series_fails)
            for df in series_outs:
                out_frames.append(df)

    if not out_frames:
        raise SystemExit("出力が空です（全パイプライン失敗の可能性）。failuresを確認してください。")

    out_df = pd.concat(out_frames, ignore_index=True)

    # ---- run meta ----
    pipeline_meta_json = json.dumps(pipelines, ensure_ascii=False)
    out_df["run_id"] = run_id
    out_df["created_ts"] = created_ts
    out_df["series_len"] = out_df.groupby("sfid")["y"].transform("size").astype(int)
    out_df["model_params_json"] = pipeline_meta_json

    # 保存列名を統一
    out_df["ds"] = out_df["ds_real"]
    out_df = out_df.drop(columns=["ds_real"])

    # ---- 特徴量列に hist 接頭辞 ----
    feat_map = {
        "y_hat": f"{feature_prefix}y_hat",
        "anomaly_score": f"{feature_prefix}anomaly_score",
        "is_anomaly": f"{feature_prefix}is_anomaly",
    }
    out_df = out_df.rename(columns=feat_map)

    # ---- 保存する列を切り替え ----
    base_cols = ["loto", "unique_id", "ts_type", "ds", "y"]
    feat_cols = [feat_map["y_hat"], feat_map["anomaly_score"], feat_map["is_anomaly"]]
    common_meta = ["library", "pipeline", "run_id", "created_ts"]

    if save_mode == "minimal":
        keep = base_cols + feat_cols + common_meta
    else:
        # full: デバッグ/再現性用の列も残す
        keep = base_cols + feat_cols + [
            "model", "scorer", "detector",
            "forecast_horizon", "start", "train_frac",
            "series_len", "model_params_json",
            "sfid", "ds_idx",
        ] + common_meta

    out_df = out_df[keep]

    fail_df = pd.DataFrame(failures)
    if len(fail_df) > 0:
        fail_df["run_id"] = run_id
        fail_df["created_ts"] = created_ts

    # ---- save ----
    ensure_schema(out_engine, out_schema)
    save_results(out_df, out_engine, out_schema, out_table, if_exists=if_exists, chunksize=int(chunksize))
    create_indexes(
        out_engine,
        out_schema,
        out_table,
        anomaly_flag_col=feat_map["is_anomaly"],  # 例: "hist_darts__is_anomaly"
    )
    if len(fail_df) > 0:
        save_failures(fail_df, out_engine, out_schema, failures_table)

    print(f"OK run_id={run_id} series={len(series_ids)} pipelines={[p['pipeline_name'] for p in pipelines]} save_mode={save_mode}")
    print(f"Failures -> {out_schema}.{failures_table} rows={len(fail_df)}")


if __name__ == "__main__":
    main()

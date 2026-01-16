from __future__ import annotations

import argparse
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Support running both as a package module and as a plain script from this folder.
try:
    from .config_utils import AppConfig, load_ini, merge_overrides, require  # type: ignore
    from .db_utils import (
        TableRef,
        make_engine_from_config,
        read_one_series,
        read_series_keys,
        ensure_table,
        ensure_columns,
        ensure_indexes,
        write_dataframe,
    )  # type: ignore
    from .naming import make_pipeline_name, feature_name  # type: ignore
    from .pypots_runner import RunMeta, safe_process_one_series  # type: ignore
except Exception:
    from config_utils import AppConfig, load_ini, merge_overrides, require  # type: ignore
    from db_utils import (
        TableRef,
        make_engine_from_config,
        read_one_series,
        read_series_keys,
        ensure_table,
        ensure_columns,
        ensure_indexes,
        write_dataframe,
    )  # type: ignore
    from naming import make_pipeline_name, feature_name  # type: ignore
    from pypots_runner import RunMeta, safe_process_one_series  # type: ignore


def _worker_process_series(
    row: Dict[str, Any],
    assigned_device: Optional[str],
    *,
    db_source_section: Dict[str, Any],
    source_schema: str,
    source_table: str,
    where_sql: Optional[str],
    run_meta: RunMeta,
    models: List[str],
    seq_len: int,
    stride: int,
    train_frac: float,
    anomaly_rate: float,
    seed: int,
    hist_prefix: str,
    lib_prefix: str,
    sep: str,
    include_pipeline: bool,
    save_mode: str,
    pypots_common_kwargs: Dict[str, Any],
    pypots_model_kwargs_map: Dict[str, Dict[str, Any]],
    amp_enabled: bool,
    fail_fast: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Top-level worker for multiprocessing (must be picklable)."""
    src_engine = make_engine_from_config(db_source_section)

    loto, unique_id, ts_type = row["loto"], row["unique_id"], row["ts_type"]
    series_df = read_one_series(
        src_engine,
        source_schema=source_schema,
        source_table=source_table,
        where_sql=where_sql,
        loto=loto,
        unique_id=unique_id,
        ts_type=ts_type,
    )

    base_df = None
    failures = []
    used_models = []
    used_pipelines = []

    for m in models:
        pipeline = make_pipeline_name(m, seq_len, stride)
        model_kwargs = pypots_model_kwargs_map.get(m, {})

        res_df, fail_df = safe_process_one_series(
            series_df=series_df,
            run_meta=run_meta,
            model_name=m,
            pipeline=pipeline,
            seq_len=seq_len,
            stride=stride,
            train_frac=train_frac,
            anomaly_rate=anomaly_rate,
            device=assigned_device,
            seed=seed,
            hist_prefix=hist_prefix,
            lib_prefix=lib_prefix,
            sep=sep,
            include_pipeline=include_pipeline,
            save_mode=save_mode,
            pypots_common_kwargs=pypots_common_kwargs,
            pypots_model_kwargs=model_kwargs,
            amp_enabled=amp_enabled,
        )
        if res_df is not None:
            base_df = _merge_model_outputs(
                base_df,
                res_df,
                hist_prefix=hist_prefix,
                lib_prefix=lib_prefix,
                sep=sep,
            )
            used_models.append(m)
            used_pipelines.append(pipeline)
        if fail_df is not None:
            failures.append(fail_df)
            if fail_fast:
                break

    if base_df is not None:
        base_df["run_id"] = run_meta.run_id
        base_df["created_ts"] = run_meta.created_ts
        base_df["library"] = "pypots"
        base_df["model"] = "|".join(used_models) if used_models else None
        base_df["pipeline"] = "|".join(used_pipelines) if used_pipelines else None
        base_df["models_json"] = str(used_models)
        base_df["pipelines_json"] = str(used_pipelines)

    fail_df_all = pd.concat(failures, ignore_index=True) if failures else None
    return base_df, fail_df_all


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyPOTS anomaly detection runner (DB -> PyPOTS -> DB)")
    p.add_argument("--config", required=True, help="Path to config.ini")
    p.add_argument("--series-limit", type=int, default=None)
    p.add_argument("--models", type=str, default=None, help="Comma-separated model names (e.g. DLinear,TimesNet)")
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 | cuda:0,cuda:1")
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--train-frac", type=float, default=None)
    p.add_argument("--save-mode", type=str, choices=["minimal", "full"], default=None)
    p.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    return p.parse_args(argv)


def _devices_from_config(device_cfg: Any) -> Tuple[List[Optional[str]], bool]:
    """Return (devices_list, using_cuda). devices_list is per-process assigned device string."""
    if device_cfg is None or str(device_cfg).strip() == "":
        return [None], False
    s = str(device_cfg).strip()
    if s.lower() in ("cpu", "none"):
        return ["cpu"], False
    if "," in s:
        devs = [x.strip() for x in s.split(",") if x.strip()]
        using_cuda = any(d.startswith("cuda") for d in devs)
        return devs, using_cuda
    if s == "cuda":
        # expand to all visible GPUs if torch is available
        try:
            import torch

            n = torch.cuda.device_count()
            if n > 0:
                return [f"cuda:{i}" for i in range(n)], True
        except Exception:
            pass
        return ["cuda"], True
    return [s], s.startswith("cuda")


def _model_list(cfg: AppConfig, cli_models: Optional[str]) -> List[str]:
    if cli_models:
        return [m.strip() for m in cli_models.split(",") if m.strip()]
    models = cfg.get("pypots", "models", "DLinear")
    if isinstance(models, list):
        return [str(m).strip() for m in models if str(m).strip()]
    return [m.strip() for m in str(models).split(",") if m.strip()]


def _collect_feature_columns(df: pd.DataFrame, hist_prefix: str, lib_prefix: str, sep: str) -> List[str]:
    prefix = f"{hist_prefix}{sep}{lib_prefix}{sep}"
    return [c for c in df.columns if c.startswith(prefix)]


def _merge_model_outputs(
    base: Optional[pd.DataFrame],
    new_df: pd.DataFrame,
    *,
    hist_prefix: str,
    lib_prefix: str,
    sep: str,
) -> pd.DataFrame:
    if base is None:
        return new_df
    feats = _collect_feature_columns(new_df, hist_prefix, lib_prefix, sep)
    keep = ["loto", "unique_id", "ts_type", "ds"] + feats
    add = new_df[keep]
    merged = base.merge(add, on=["loto", "unique_id", "ts_type", "ds"], how="left")
    return merged


def _ensure_output_tables(
    engine,
    *,
    out_schema: str,
    out_table: str,
    fail_schema: str,
    fail_table: str,
    base_feature_cols: List[str],
    save_mode: str,
) -> Tuple[TableRef, TableRef]:
    out_ref = TableRef(db_name="anomaly", schema=out_schema, table=out_table)
    fail_ref = TableRef(db_name="anomaly", schema=fail_schema, table=fail_table)

    # minimal column set (PostgreSQL-friendly)
    base_cols = [
        ("loto", "TEXT"),
        ("unique_id", "TEXT"),
        ("ts_type", "TEXT"),
        ("ds", "TIMESTAMPTZ"),
        ("y", "DOUBLE PRECISION"),
        ("run_id", "TEXT"),
        ("created_ts", "TIMESTAMPTZ"),
        ("library", "TEXT"),
        ("model", "TEXT"),
        ("pipeline", "TEXT"),
        ("models_json", "TEXT"),
        ("pipelines_json", "TEXT"),
    ]
    ensure_table(engine, out_ref, base_cols)

    # extra columns in full mode
    if str(save_mode) == "full":
        ensure_columns(
            engine,
            out_ref,
            [
                ("sfid", "TEXT"),
                ("series_len", "INTEGER"),
                ("ds_idx", "INTEGER"),
                ("model_params_json", "TEXT"),
                ("fit_ok", "BOOLEAN"),
            ],
        )
    # feature columns are dynamic; add if missing
    feat_cols = [(c, "DOUBLE PRECISION") for c in base_feature_cols if c.endswith("anomaly_score") or c.endswith("threshold")]
    feat_cols += [(c, "BOOLEAN") for c in base_feature_cols if c.endswith("is_anomaly")]
    ensure_columns(engine, out_ref, feat_cols)

    # indexes
    ensure_indexes(
        engine,
        out_ref,
        [
            f"CREATE INDEX IF NOT EXISTS idx_pypots_key ON {out_ref.full_name} (loto, unique_id, ts_type, ds)",
            f"CREATE INDEX IF NOT EXISTS idx_pypots_run ON {out_ref.full_name} (run_id)",
        ],
    )

    fail_cols = [
        ("run_id", "TEXT"),
        ("created_ts", "TIMESTAMPTZ"),
        ("sfid", "TEXT"),
        ("model", "TEXT"),
        ("pipeline", "TEXT"),
        ("reason", "TEXT"),
        ("error_msg", "TEXT"),
        ("traceback", "TEXT"),
        ("series_len", "INTEGER"),
    ]
    ensure_table(engine, fail_ref, fail_cols)
    ensure_indexes(
        engine,
        fail_ref,
        [
            f"CREATE INDEX IF NOT EXISTS idx_pypots_fail_run ON {fail_ref.full_name} (run_id)",
            f"CREATE INDEX IF NOT EXISTS idx_pypots_fail_sfid ON {fail_ref.full_name} (sfid)",
        ],
    )
    return out_ref, fail_ref


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    cfg = load_ini(args.config)
    cfg = merge_overrides(
        cfg,
        {
            "run.series_limit": args.series_limit,
            "pypots.models": args.models,
            "pypots.device": args.device,
            "window.seq_len": args.seq_len,
            "window.stride": args.stride,
            "window.train_frac": args.train_frac,
            "output.save_mode": args.save_mode,
            "run.fail_fast": True if args.fail_fast else None,
        },
    )

    # required
    source_schema = require(cfg, "source", "schema")
    source_table = require(cfg, "source", "table")
    out_schema = require(cfg, "output", "schema")
    out_table = require(cfg, "output", "table")

    # optional
    where_sql = cfg.get("source", "where_sql", None)
    series_limit = cfg.get("run", "series_limit", None)
    min_len = int(cfg.get("run", "min_len", 64))
    seed = int(cfg.get("run", "seed", 42))
    fail_fast = bool(cfg.get("run", "fail_fast", False))

    seq_len = int(cfg.get("window", "seq_len", 64))
    stride = int(cfg.get("window", "stride", 1))
    train_frac = float(cfg.get("window", "train_frac", 0.8))

    anomaly_rate = float(cfg.get("pypots", "anomaly_rate", 0.01))
    amp_enabled = bool(cfg.get("pypots", "amp_enabled", False))

    hist_prefix = str(cfg.get("naming", "hist_prefix", "hist"))
    lib_prefix = str(cfg.get("naming", "lib_prefix", "pypots"))
    sep = str(cfg.get("naming", "sep", "__"))
    include_pipeline = bool(cfg.get("naming", "include_pipeline", True))

    models = _model_list(cfg, args.models)
    if not models:
        print("No models specified", file=sys.stderr)
        return 2

    devices, using_cuda = _devices_from_config(cfg.get("pypots", "device", None))
    if using_cuda:
        # CUDA + multiprocessing: spawn is safer than fork
        try:
            import multiprocessing as mp

            mp.set_start_method("spawn", force=True)
        except Exception:
            pass

    # DB engines
    src_engine = make_engine_from_config(cfg.section("db_source"))
    tgt_engine = make_engine_from_config(cfg.section("db_target"))

    # select series
    keys_df = read_series_keys(
        src_engine,
        source_schema=source_schema,
        source_table=source_table,
        where_sql=where_sql,
        series_limit=series_limit,
        min_len=min_len,
    )
    if keys_df.empty:
        print("No series matched (check where_sql/min_len/series_limit)")
        return 0

    # feature columns for all pipelines (pre-create)
    pipeline_names = [make_pipeline_name(m, seq_len, stride) for m in models]
    feature_cols = []
    for pnm in pipeline_names:
        for field in ("is_anomaly", "anomaly_score", "threshold"):
            feature_cols.append(
                feature_name(
                    field,
                    hist_prefix=hist_prefix,
                    lib_prefix=lib_prefix,
                    pipeline=pnm,
                    sep=sep,
                    include_pipeline=include_pipeline,
                )
            )

    out_ref, fail_ref = _ensure_output_tables(
        tgt_engine,
        out_schema=out_schema,
        out_table=out_table,
        fail_schema=cfg.get("output", "fail_schema", out_schema),
        fail_table=cfg.get("output", "fail_table", f"{out_table}_failures"),
        base_feature_cols=feature_cols,
        save_mode=str(cfg.get("output", "save_mode", "minimal")),
    )

    run_id = str(cfg.get("run", "run_id", uuid.uuid4().hex))
    created_ts = datetime.now(timezone.utc)
    run_meta = RunMeta(run_id=run_id, created_ts=created_ts)

    # pypots common kwargs
    pypots_common_kwargs: Dict[str, Any] = {
        "batch_size": int(cfg.get("pypots", "batch_size", 32)),
        "epochs": int(cfg.get("pypots", "epochs", 50)),
        "patience": cfg.get("pypots", "patience", None),
        "num_workers": int(cfg.get("pypots", "num_workers", 0)),
        "verbose": bool(cfg.get("pypots", "verbose", True)),
        "model_saving_strategy": cfg.get("pypots", "model_saving_strategy", None),
        "saving_path": cfg.get("pypots", "saving_path", None),
    }
    # remove None values (signature-sensitive)
    pypots_common_kwargs = {k: v for k, v in pypots_common_kwargs.items() if v is not None}

    save_mode = str(cfg.get("output", "save_mode", "minimal"))

    max_workers = int(cfg.get("run", "max_workers", len(devices) if using_cuda else (os.cpu_count() or 2)))
    if using_cuda:
        max_workers = min(max_workers, len(devices))
    max_workers = max(1, max_workers)

    futures = []
    failures_all = []

    # collect model-specific kwargs (picklable)
    pypots_model_kwargs_map: Dict[str, Dict[str, Any]] = {
        m: cfg.section(f"pypots.model.{m}") for m in models
    }

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i, row in enumerate(keys_df.to_dict("records")):
            dev = devices[i % len(devices)] if devices else None
            futures.append(
                ex.submit(
                    _worker_process_series,
                    row,
                    dev,
                    db_source_section=cfg.section("db_source"),
                    source_schema=source_schema,
                    source_table=source_table,
                    where_sql=where_sql,
                    run_meta=run_meta,
                    models=models,
                    seq_len=seq_len,
                    stride=stride,
                    train_frac=train_frac,
                    anomaly_rate=anomaly_rate,
                    seed=seed,
                    hist_prefix=hist_prefix,
                    lib_prefix=lib_prefix,
                    sep=sep,
                    include_pipeline=include_pipeline,
                    save_mode=save_mode,
                    pypots_common_kwargs=pypots_common_kwargs,
                    pypots_model_kwargs_map=pypots_model_kwargs_map,
                    amp_enabled=amp_enabled,
                    fail_fast=fail_fast,
                )
            )

        for fut in as_completed(futures):
            res_df, fail_df = fut.result()
            if res_df is not None and not res_df.empty:
                write_dataframe(
                    tgt_engine,
                    out_ref,
                    res_df,
                    if_exists="append",
                    chunksize=int(cfg.get("output", "chunksize", 5000)),
                )
            if fail_df is not None and not fail_df.empty:
                failures_all.append(fail_df)
                write_dataframe(
                    tgt_engine,
                    fail_ref,
                    fail_df,
                    if_exists="append",
                    chunksize=1000,
                )
                if fail_fast:
                    break

    # optional: print summary
    n_fail = sum(len(x) for x in failures_all) if failures_all else 0
    print(f"run_id={run_id} done. series={len(keys_df)} failures={n_fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

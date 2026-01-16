# /mnt/e/env/ts/tslib/ds/src/anomalies/merlion/anomaly_merlion.py
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

# ★ 相対importをやめて同階層importにする
from config_utils import (
    load_ini, cfg_get, cfg_section_dict,
    DBConn, OutSpec, RunSpec, NamingSpec
)
from db_utils import make_engine, ensure_schema, read_loto_base, write_df, create_indexes
from merlion_runner import SeriesPayload, run_one_series_wide


def _build_specs(cfg_path: str, args: argparse.Namespace):
    cfg = load_ini(cfg_path)

    # source
    src = DBConn(
        user=str(args.src_user or cfg_get(cfg, "db_source", "user", "postgres")),
        password=str(args.src_pass or cfg_get(cfg, "db_source", "pass", "z")),
        host=str(args.src_host or cfg_get(cfg, "db_source", "host", "127.0.0.1")),
        port=int(args.src_port or cfg_get(cfg, "db_source", "port", 5432)),
        dbname=str(args.src_name or cfg_get(cfg, "db_source", "name", "dataset")),
        schema=str(args.src_schema or cfg_get(cfg, "db_source", "schema", "public")),
        table=str(args.src_table or cfg_get(cfg, "db_source", "table", "loto_base")),
        where_sql=str(args.where_sql if args.where_sql is not None else cfg_get(cfg, "db_source", "where_sql", "")),
    )

    # target
    out = OutSpec(
        schema=str(args.out_schema or cfg_get(cfg, "db_target", "schema", "public")),
        table=str(args.out_table or cfg_get(cfg, "db_target", "table", "merlion")),
        failures_table=str(args.failures_table or cfg_get(cfg, "db_target", "failures_table", "merlion_failures")),
        if_exists=str(args.if_exists or cfg_get(cfg, "db_target", "if_exists", "replace")),
        chunksize=int(args.chunksize or cfg_get(cfg, "db_target", "chunksize", 50000)),
    )
    tgt_user = str(args.out_user or cfg_get(cfg, "db_target", "user", "postgres"))
    tgt_pass = str(args.out_pass or cfg_get(cfg, "db_target", "pass", "z"))
    tgt_host = str(args.out_host or cfg_get(cfg, "db_target", "host", "127.0.0.1"))
    tgt_port = int(args.out_port or cfg_get(cfg, "db_target", "port", 5432))
    tgt_name = str(args.out_name or cfg_get(cfg, "db_target", "name", "anomaly"))

    # run
    models_csv = str(args.models or cfg_get(cfg, "run", "models", "DefaultDetector"))
    models = [m.strip() for m in models_csv.split(",") if m.strip()]
    run = RunSpec(
        models=models,
        series_limit=int(args.series_limit if args.series_limit is not None else cfg_get(cfg, "run", "series_limit", 0)),
        min_len=int(args.min_len if args.min_len is not None else cfg_get(cfg, "run", "min_len", 50)),
        train_frac=float(args.train_frac if args.train_frac is not None else cfg_get(cfg, "run", "train_frac", 0.8)),
        n_jobs=int(args.n_jobs if args.n_jobs is not None else cfg_get(cfg, "run", "n_jobs", 4)),
        mp_start_method=str(args.mp_start_method or cfg_get(cfg, "run", "mp_start_method", "spawn")),
        flush_rows=int(args.flush_rows if args.flush_rows is not None else cfg_get(cfg, "run", "flush_rows", 200000)),
        fail_fast=bool(args.fail_fast if args.fail_fast is not None else cfg_get(cfg, "run", "fail_fast", False)),
        save_failures=bool(cfg_get(cfg, "run", "save_failures", True)),
    )

    # naming
    naming = NamingSpec(
        library=str(cfg_get(cfg, "naming", "library", "merlion")),
        feature_prefix_template=str(cfg_get(cfg, "naming", "feature_prefix_template", "hist_{library}__")),
        meta_prefix_template=str(cfg_get(cfg, "naming", "meta_prefix_template", "{library}__")),
        save_mode=str(args.save_mode or cfg_get(cfg, "naming", "save_mode", "wide")),
        include_y=bool(cfg_get(cfg, "naming", "include_y", False)),
        include_debug_cols=bool(cfg_get(cfg, "naming", "include_debug_cols", False)),
    )

    # per-model params
    model_params_map: Dict[str, Dict[str, Any]] = {}
    for mn in models:
        sec = f"model.{mn}"
        model_params_map[mn] = cfg_section_dict(cfg, sec)

    return src, (tgt_user, tgt_pass, tgt_host, tgt_port, tgt_name), out, run, naming, model_params_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)

    # source overrides
    ap.add_argument("--src-user", default=None)
    ap.add_argument("--src-pass", default=None)
    ap.add_argument("--src-host", default=None)
    ap.add_argument("--src-port", default=None)
    ap.add_argument("--src-name", default=None)
    ap.add_argument("--src-schema", default=None)
    ap.add_argument("--src-table", default=None)
    ap.add_argument("--where-sql", default=None)

    # target overrides
    ap.add_argument("--out-user", default=None)
    ap.add_argument("--out-pass", default=None)
    ap.add_argument("--out-host", default=None)
    ap.add_argument("--out-port", default=None)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--out-schema", default=None)
    ap.add_argument("--out-table", default=None)
    ap.add_argument("--failures-table", default=None)
    ap.add_argument("--if-exists", default=None, choices=["replace", "append"])
    ap.add_argument("--chunksize", type=int, default=None)

    # run overrides
    ap.add_argument("--models", default=None)
    ap.add_argument("--series-limit", type=int, default=None)
    ap.add_argument("--min-len", type=int, default=None)
    ap.add_argument("--train-frac", type=float, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--mp-start-method", default=None)
    ap.add_argument("--flush-rows", type=int, default=None)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--save-mode", default=None, choices=["wide", "long"])

    args = ap.parse_args()

    src, tgt_conn, out, run, naming, model_params_map = _build_specs(args.config, args)

    # engines
    src_engine = make_engine(src.user, src.password, src.host, src.port, src.dbname)
    tgt_user, tgt_pass, tgt_host, tgt_port, tgt_name = tgt_conn
    out_engine = make_engine(tgt_user, tgt_pass, tgt_host, tgt_port, tgt_name)

    print(f"SOURCE DB={src.dbname}  TABLE={src.schema}.{src.table}")
    print(f"TARGET DB={tgt_name}  TABLE={out.schema}.{out.table}")

    ensure_schema(out_engine, out.schema)

    # load
    df = read_loto_base(src_engine, src.schema, src.table, src.where_sql)

    # series id
    df["sfid"] = df["loto"].astype(str) + "__" + df["unique_id"].astype(str) + "__" + df["ts_type"].astype(str)
    df = df.sort_values(["sfid", "ds"])

    # series list
    sizes = df.groupby("sfid")["y"].size().sort_values(ascending=False)
    sfids = sizes.index.tolist()
    if run.series_limit > 0:
        sfids = sfids[: run.series_limit]

    # meta
    run_id = str(__import__("uuid").uuid4())
    created_ts = datetime.now(timezone.utc)

    # buffer write
    buffer: list[pd.DataFrame] = []
    buffered_rows = 0
    failures: list[dict] = []

    write_mode = out.if_exists
    wrote_any = False
    n_jobs = max(1, int(run.n_jobs))

    def flush():
        nonlocal buffer, buffered_rows, write_mode, wrote_any
        if not buffer:
            return
        big = pd.concat(buffer, ignore_index=True)

        meta_prefix = naming.meta_prefix_template.format(library=naming.library)
        big[f"{meta_prefix}run_id"] = run_id
        big[f"{meta_prefix}created_ts"] = created_ts

        write_df(out_engine, big, out.schema, out.table, if_exists=write_mode, chunksize=out.chunksize)
        wrote_any = True
        if write_mode == "replace":
            write_mode = "append"
        buffer = []
        buffered_rows = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = []
        for sfid in sfids:
            g = df[df["sfid"] == sfid]
            payload = SeriesPayload(
                loto=str(g["loto"].iloc[0]),
                unique_id=str(g["unique_id"].iloc[0]),
                ts_type=str(g["ts_type"].iloc[0]),
                ds=g["ds"].to_numpy(),
                y=g["y"].to_numpy(dtype=float),
            )
            futs.append(ex.submit(
                run_one_series_wide,
                payload,
                run.models,
                model_params_map,
                naming.feature_prefix_template,
                naming.library,
                run.train_frac,
                run.min_len,
            ))

        for fut in as_completed(futs):
            try:
                series_df, meta = fut.result()
                buffer.append(series_df)
                buffered_rows += len(series_df)

                if meta.get("failures"):
                    failures.append({
                        "run_id": run_id,
                        "created_ts": created_ts,
                        "sfid": None,
                        "reason": "partial_failures",
                        "error_msg": json.dumps(meta["failures"], ensure_ascii=False),
                    })

                if buffered_rows >= run.flush_rows:
                    flush()

            except Exception as e:
                failures.append({
                    "run_id": run_id,
                    "created_ts": created_ts,
                    "sfid": None,
                    "reason": "series_failed",
                    "error_msg": str(e),
                })
                if run.fail_fast:
                    raise

    flush()

    if wrote_any:
        meta_prefix = naming.meta_prefix_template.format(library=naming.library)
        create_indexes(
            out_engine,
            out.schema,
            out.table,
            naming.save_mode,
            run_id_col=f"{meta_prefix}run_id",   # ← ここが重要（merlion__run_id）
        )

    if run.save_failures and failures:
        fail_df = pd.DataFrame(failures)
        write_df(out_engine, fail_df, out.schema, out.failures_table, if_exists="append", chunksize=50000)

    print(f"OK run_id={run_id} series={len(sfids)} models={run.models} save_mode={naming.save_mode}")
    print(f"Failures -> {out.schema}.{out.failures_table} rows={len(failures)}")


if __name__ == "__main__":
    main()

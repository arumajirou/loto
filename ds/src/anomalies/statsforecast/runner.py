import os
import uuid
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from config_utils import AppConfig
from db_utils import Database

def _set_thread_env(threads: int):
    t = str(max(1, int(threads)))
    os.environ["OMP_NUM_THREADS"] = t
    os.environ["OPENBLAS_NUM_THREADS"] = t
    os.environ["MKL_NUM_THREADS"] = t
    os.environ["VECLIB_MAXIMUM_THREADS"] = t
    os.environ["NUMEXPR_NUM_THREADS"] = t
    os.environ["NUMBA_NUM_THREADS"] = t

def _series_jobs(df_y: pd.DataFrame, min_len: int, series_limit: int):
    df_y["sfid"] = (
        df_y["loto"].astype(str) + "__" +
        df_y["unique_id"].astype(str) + "__" +
        df_y["ts_type"].astype(str)
    )
    df_y = df_y.sort_values(["sfid", "ds"])
    df_y["ds_idx"] = df_y.groupby("sfid").cumcount() + 1

    lens = df_y.groupby("sfid")["y"].size().sort_values(ascending=False)
    keep = lens[lens >= int(min_len)]
    sfids = keep.index.tolist()
    if int(series_limit) > 0:
        sfids = sfids[: int(series_limit)]

    jobs = []
    for sfid in sfids:
        s = df_y[df_y["sfid"] == sfid]
        jobs.append({
            "sfid": sfid,
            "loto": str(s["loto"].iloc[0]),
            "unique_id": str(s["unique_id"].iloc[0]),
            "ts_type": str(s["ts_type"].iloc[0]),
            "ds_idx": s["ds_idx"].to_numpy(dtype=np.int32),
            "ds_real": s["ds"].to_numpy(dtype="datetime64[ns]"),
            "y": s["y"].to_numpy(dtype=np.float64),
        })
    return jobs

def _worker(job: dict, cfg_dict: dict):
    _set_thread_env(cfg_dict["threads_per_worker"])

    from detectors.statsforecast_detector import run_statsforecast_on_series
    from detectors.hampel_detector import hampel_detect

    rows, fails = [], []

    r, f = run_statsforecast_on_series(
        sfid=job["sfid"],
        ds_idx=job["ds_idx"],
        ds_real=job["ds_real"],
        y=job["y"],
        loto=job["loto"],
        unique_id=job["unique_id"],
        ts_type=job["ts_type"],
        model_specs=cfg_dict["model_specs"],
        level=cfg_dict["level"],
        horizon=cfg_dict["horizon"],
        run_id=cfg_dict["run_id"],
        created_ts=cfg_dict["created_ts"],
        save_only_anomalies=cfg_dict["save_only_anomalies"],
    )
    rows.extend(r); fails.extend(f)

    if cfg_dict["enable_hampel"]:
        r2, f2 = hampel_detect(
            sfid=job["sfid"],
            ds_real=job["ds_real"],
            y=job["y"],
            loto=job["loto"],
            unique_id=job["unique_id"],
            ts_type=job["ts_type"],
            window=cfg_dict["hampel_window"],
            k=cfg_dict["hampel_k"],
            use_gpu=cfg_dict["use_gpu"],
            run_id=cfg_dict["run_id"],
            created_ts=cfg_dict["created_ts"],
            save_only_anomalies=cfg_dict["save_only_anomalies"],
        )
        rows.extend(r2); fails.extend(f2)

    return rows, fails

class PipelineRunner:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        # source DB URL（dataset）
        src_db_url = (
            f"postgresql+psycopg2://{cfg.src_db_user}:{cfg.src_db_pass}"
            f"@{cfg.src_db_host}:{cfg.src_db_port}/{cfg.src_db_name}"
        )
        # target DB URL（anomaly）
        tgt_db_url = (
            f"postgresql+psycopg2://{cfg.tgt_db_user}:{cfg.tgt_db_pass}"
            f"@{cfg.tgt_db_host}:{cfg.tgt_db_port}/{cfg.tgt_db_name}"
        )

        self.db_src = Database(src_db_url)
        self.db_tgt = Database(tgt_db_url)

    def run(self):
        cfg = self.cfg
        run_id = str(uuid.uuid4())
        created_ts = datetime.now(timezone.utc)

        # 1) load from SOURCE DB (dataset)
        df = self.db_src.load_source(cfg.src_schema, cfg.src_table, cfg.where_sql)
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df = df.dropna(subset=["ds"])
        df_y = df.dropna(subset=["y"]).copy()

        # 2) jobs
        jobs = _series_jobs(df_y, cfg.min_len, cfg.series_limit)
        if not jobs:
            raise RuntimeError("有効な系列がありません（min_len/where_sql を見直してください）。")

        # 3) prepare TARGET DB (anomaly)
        self.db_tgt.ensure_schema(cfg.out_schema)

        worker_cfg = {
            "model_specs": cfg.model_specs,
            "level": int(cfg.level),
            "horizon": int(cfg.horizon),
            "enable_hampel": bool(cfg.enable_hampel),
            "hampel_window": int(cfg.hampel_window),
            "hampel_k": float(cfg.hampel_k),
            "use_gpu": bool(cfg.use_gpu),
            "save_only_anomalies": bool(cfg.save_only_anomalies),
            "threads_per_worker": int(cfg.threads_per_worker),
            "run_id": run_id,
            "created_ts": created_ts,
        }

        first_write_mode = cfg.if_exists
        pending_rows, pending_fails = [], []

        def flush():
            nonlocal first_write_mode, pending_rows, pending_fails
            if pending_rows:
                out_df = pd.DataFrame(pending_rows)
                self.db_tgt.save_results_chunk(
                    out_df, cfg.out_schema, cfg.out_table,
                    if_exists=first_write_mode, chunksize=int(cfg.chunksize)
                )
                first_write_mode = "append"
                pending_rows = []
            if cfg.save_failures and pending_fails:
                fail_df = pd.DataFrame(pending_fails)
                self.db_tgt.save_failures(fail_df, cfg.out_schema, cfg.failures_table, chunksize=50000)
                pending_fails = []

        with ProcessPoolExecutor(max_workers=int(cfg.max_workers)) as ex:
            futs = [ex.submit(_worker, job, worker_cfg) for job in jobs]
            done = 0
            for fut in as_completed(futs):
                rows, fails = fut.result()
                pending_rows.extend(rows)
                pending_fails.extend(fails)
                done += 1
                if done % int(cfg.flush_every) == 0:
                    flush()

        flush()
        self.db_tgt.create_indexes(cfg.out_schema, cfg.out_table, cfg.failures_table)

        print(f"SOURCE DB={cfg.src_db_name}  TABLE={cfg.src_schema}.{cfg.src_table}")
        print(f"TARGET DB={cfg.tgt_db_name}  TABLE={cfg.out_schema}.{cfg.out_table}")
        print(f"OK run_id={run_id} series={len(jobs)} models={cfg.enabled_models}")
        if cfg.save_failures:
            print(f"Failures -> {cfg.out_schema}.{cfg.failures_table}")

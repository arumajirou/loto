# /mnt/e/env/ts/tslib/ds/src/anomalies/sktime/runner.py
import os
import uuid
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import numpy as np
import pandas as pd

from config_utils import AppConfig
from db_utils import Database

LIBRARY_NAME = "sktime"

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
            "ds_real": s["ds"].to_numpy(dtype="datetime64[ns]"),
            "y": s["y"].to_numpy(dtype=np.float64),
        })
    return jobs

def _worker(job: dict, cfg_dict: dict):
    _set_thread_env(cfg_dict["threads_per_worker"])

    warnings.filterwarnings(
        "ignore",
        message=r"Duplicate values are leading to incorrect results\..*",
        category=UserWarning,
    )

    from detectors.sktime_detector import run_sktime_on_series

    return run_sktime_on_series(
        sfid=job["sfid"],
        ds_real=job["ds_real"],
        y=job["y"],
        loto=job["loto"],
        unique_id=job["unique_id"],
        ts_type=job["ts_type"],
        model_specs=cfg_dict["model_specs"],
        run_id=cfg_dict["run_id"],
        created_ts=cfg_dict["created_ts"],
        save_only_anomalies=cfg_dict["save_only_anomalies"],
    )

def _apply_prefix(df: pd.DataFrame, *, prefix: str, feature_cols: list[str], save_mode: str, prefix_run_metadata: bool):
    """
    特徴量列にだけ prefix を付ける（JOIN時の衝突回避）
    """
    rename_map = {}
    for c in feature_cols:
        if c in df.columns:
            rename_map[c] = f"{prefix}{c}"

    # minimal時：run_id / created_ts も衝突しやすいので必要ならprefix
    if save_mode == "minimal" and prefix_run_metadata:
        for c in ("run_id", "created_ts"):
            if c in df.columns:
                rename_map[c] = f"{prefix}{c}"

    df2 = df.rename(columns=rename_map)

    # minimal: JOINに必要なもの + y + feature(+run meta)
    if save_mode == "minimal":
        keep = ["loto", "unique_id", "ts_type", "ds", "y"]
        keep += [rename_map.get(c, c) for c in feature_cols if (rename_map.get(c, c) in df2.columns)]
        if prefix_run_metadata:
            for c in (f"{prefix}run_id", f"{prefix}created_ts"):
                if c in df2.columns:
                    keep.append(c)
        else:
            for c in ("run_id", "created_ts"):
                if c in df2.columns:
                    keep.append(c)
        keep = [c for c in keep if c in df2.columns]
        df2 = df2[keep]

    anomaly_col = rename_map.get("is_anomaly", "is_anomaly")
    model_col = "model"  # sktimeは model 列を残す（full時）。minimal時は列が無ければ db_utils 側でスキップされる

    return df2, anomaly_col, model_col

class PipelineRunner:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        src_db_url = (
            f"postgresql+psycopg2://{cfg.src_db_user}:{cfg.src_db_pass}"
            f"@{cfg.src_db_host}:{cfg.src_db_port}/{cfg.src_db_name}"
        )
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

        # 1) load from dataset
        df = self.db_src.load_source(cfg.src_schema, cfg.src_table, cfg.where_sql)
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df = df.dropna(subset=["ds"])
        df_y = df.dropna(subset=["y"]).copy()

        jobs = _series_jobs(df_y, cfg.min_len, cfg.series_limit)
        if not jobs:
            raise RuntimeError("有効な系列がありません（min_len/where_sql を見直してください）。")

        # 2) prepare anomaly
        self.db_tgt.ensure_schema(cfg.out_schema)

        worker_cfg = {
            "model_specs": cfg.model_specs,
            "threads_per_worker": int(cfg.threads_per_worker),
            "save_only_anomalies": bool(cfg.save_only_anomalies),
            "run_id": run_id,
            "created_ts": created_ts,
        }

        first_write_mode = cfg.if_exists
        pending_rows, pending_fails = [], []
        last_anomaly_col = "is_anomaly"
        last_model_col = "model"

        def flush():
            nonlocal first_write_mode, pending_rows, pending_fails, last_anomaly_col, last_model_col
            if pending_rows:
                out_df = pd.DataFrame(pending_rows)

                prefix = str(cfg.feature_prefix_template).format(library=LIBRARY_NAME)
                out_df2, anomaly_col, model_col = _apply_prefix(
                    out_df,
                    prefix=prefix,
                    feature_cols=list(cfg.feature_cols),
                    save_mode=str(cfg.save_mode),
                    prefix_run_metadata=bool(cfg.prefix_run_metadata),
                )
                last_anomaly_col = anomaly_col
                last_model_col = model_col

                self.db_tgt.save_results_chunk(
                    out_df2, cfg.out_schema, cfg.out_table,
                    if_exists=first_write_mode, chunksize=int(cfg.chunksize)
                )
                first_write_mode = "append"
                pending_rows = []

            if cfg.save_failures and pending_fails:
                fail_df = pd.DataFrame(pending_fails)
                self.db_tgt.save_failures(fail_df, cfg.out_schema, cfg.failures_table, chunksize=50000)
                pending_fails = []

        # 3) parallel
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

                if cfg.fail_fast and len(fails) > 0:
                    flush()
                    raise RuntimeError(f"fail_fast: detector failed. example={fails[0]}")

        flush()
        # 列が無いなら db_utils 側が勝手にスキップする
        self.db_tgt.create_indexes(cfg.out_schema, cfg.out_table, cfg.failures_table, anomaly_col=last_anomaly_col, model_col=last_model_col)

        print(f"SOURCE DB={cfg.src_db_name}  TABLE={cfg.src_schema}.{cfg.src_table}")
        print(f"TARGET DB={cfg.tgt_db_name}  TABLE={cfg.out_schema}.{cfg.out_table}")
        print(f"OK run_id={run_id} series={len(jobs)} models={cfg.enabled_models} save_mode={cfg.save_mode}")
        if cfg.save_failures:
            print(f"Failures -> {cfg.out_schema}.{cfg.failures_table}")

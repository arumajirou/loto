# /mnt/e/env/ts/tslib/ds/src/anomalies/sktime/config_utils.py
import os
import ast
import argparse
import configparser
from dataclasses import dataclass
from typing import List, Optional

def _strip_inline_comment(s: str) -> str:
    if s is None:
        return s
    t = s.strip()
    for sep in (";", "#"):
        if sep in t:
            t = t.split(sep, 1)[0].strip()
    return t

def parse_value(v: str):
    if v is None:
        return None
    s = _strip_inline_comment(v)
    if s == "":
        return ""
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        pass
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")) or (s.startswith("(") and s.endswith(")")):
        try:
            return ast.literal_eval(s)
        except Exception:
            return s
    return s

def load_ini(path: str):
    cfg = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=(";", "#"),
    )
    if os.path.exists(path):
        cfg.read(path, encoding="utf-8")
    return cfg

def cfg_get(cfg, section, key, default=None):
    if cfg.has_option(section, key):
        return parse_value(cfg.get(section, key))
    return default

def default_ini_path(cli_path: str | None, file_dir: str):
    if cli_path:
        return cli_path
    return os.path.join(file_dir, "config.ini")

def parse_bool_override(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None

def parse_csv_list(v, default: List[str]) -> List[str]:
    if v is None:
        return list(default)
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    s = _strip_inline_comment(str(v))
    if not s:
        return list(default)
    return [x.strip() for x in s.split(",") if x.strip()]

@dataclass(frozen=True)
class AppConfig:
    # source DB
    src_db_user: str
    src_db_pass: str
    src_db_host: str
    src_db_port: str
    src_db_name: str

    # target DB
    tgt_db_user: str
    tgt_db_pass: str
    tgt_db_host: str
    tgt_db_port: str
    tgt_db_name: str

    # source table
    src_schema: str
    src_table: str
    where_sql: str

    # output table
    out_schema: str
    out_table: str
    failures_table: str
    if_exists: str
    chunksize: int
    flush_every: int

    # output schema design
    save_mode: str  # "full" | "minimal"
    feature_prefix_template: str
    feature_cols: list[str]
    prefix_run_metadata: bool

    # run
    min_len: int
    series_limit: int
    max_workers: int
    threads_per_worker: int
    fail_fast: bool
    save_failures: bool
    save_only_anomalies: bool

    # models
    enabled_models: list[str]
    model_specs: list[dict]  # [{"model":..., "params": {...}}, ...]

def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)

    ap.add_argument("--where-sql", default=None)
    ap.add_argument("--models", default=None, help="例: STRAY,SubLOF もしくは module.Class の完全修飾名")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--threads-per-worker", type=int, default=None)
    ap.add_argument("--series-limit", type=int, default=None)
    ap.add_argument("--save-only-anomalies", default=None)  # true/false
    ap.add_argument("--fail-fast", default=None)            # true/false

    # 追加：カラム接頭辞まわり
    ap.add_argument("--save-mode", default=None, choices=["full", "minimal"])
    ap.add_argument("--feature-prefix-template", default=None)  # 例: hist_{library}__
    ap.add_argument("--feature-cols", default=None)             # CSV
    ap.add_argument("--prefix-run-metadata", default=None)      # true/false

    return ap

def build_model_specs(cfg, enabled_models: list[str], override_models: str | None):
    if override_models:
        enabled_models = [x.strip() for x in override_models.split(",") if x.strip()]

    specs = []
    for mn in enabled_models:
        sec = f"model.{mn}"
        params = {}
        if cfg.has_section(sec):
            for k, v in cfg.items(sec):
                params[k] = parse_value(v)
        specs.append({"model": mn, "params": params})
    return enabled_models, specs

def from_ini_and_args(ini_path: str, args, env=os.environ) -> AppConfig:
    cfg = load_ini(ini_path)

    # source DB (dataset)
    src_db_user = env.get("SRC_DB_USER") or cfg_get(cfg, "db_source", "user", "postgres")
    src_db_pass = env.get("SRC_DB_PASS") or cfg_get(cfg, "db_source", "pass", "z")
    src_db_host = env.get("SRC_DB_HOST") or cfg_get(cfg, "db_source", "host", "127.0.0.1")
    src_db_port = env.get("SRC_DB_PORT") or cfg_get(cfg, "db_source", "port", "5432")
    src_db_name = env.get("SRC_DB_NAME") or cfg_get(cfg, "db_source", "name", "dataset")

    # target DB (anomaly)
    tgt_db_user = env.get("TGT_DB_USER") or cfg_get(cfg, "db_target", "user", "postgres")
    tgt_db_pass = env.get("TGT_DB_PASS") or cfg_get(cfg, "db_target", "pass", "z")
    tgt_db_host = env.get("TGT_DB_HOST") or cfg_get(cfg, "db_target", "host", "127.0.0.1")
    tgt_db_port = env.get("TGT_DB_PORT") or cfg_get(cfg, "db_target", "port", "5432")
    tgt_db_name = env.get("TGT_DB_NAME") or cfg_get(cfg, "db_target", "name", "anomaly")

    # source table
    src_schema = cfg_get(cfg, "source", "schema", "public")
    src_table  = cfg_get(cfg, "source", "table", "loto_base")
    where_sql  = args.where_sql if args.where_sql is not None else cfg_get(cfg, "source", "where_sql", "")

    # output
    out_schema = cfg_get(cfg, "output", "schema", "public")
    out_table  = cfg_get(cfg, "output", "table", "sktime")
    failures_table = cfg_get(cfg, "output", "failures_table", "sktime_failures")
    if_exists  = cfg_get(cfg, "output", "if_exists", "replace")
    chunksize  = int(cfg_get(cfg, "output", "chunksize", 50000))
    flush_every = int(cfg_get(cfg, "output", "flush_every", 1))

    # 追加：prefix設定
    save_mode = str(cfg_get(cfg, "output", "save_mode", "full")).strip().lower()
    if save_mode not in ("full", "minimal"):
        save_mode = "full"

    feature_prefix_template = str(cfg_get(cfg, "output", "feature_prefix_template", "hist_{library}__"))
    default_feature_cols = ["y_hat", "y_lo", "y_hi", "is_anomaly", "anomaly_side", "anomaly_score"]
    feature_cols = parse_csv_list(cfg_get(cfg, "output", "feature_cols", ",".join(default_feature_cols)), default_feature_cols)

    prefix_run_metadata = bool(cfg_get(cfg, "output", "prefix_run_metadata", False))

    # run
    min_len = int(cfg_get(cfg, "run", "min_len", 20))
    series_limit = int(cfg_get(cfg, "run", "series_limit", 0))
    max_workers = int(cfg_get(cfg, "run", "max_workers", 4))
    threads_per_worker = int(cfg_get(cfg, "run", "threads_per_worker", 1))
    fail_fast = bool(cfg_get(cfg, "run", "fail_fast", False))
    save_failures = bool(cfg_get(cfg, "run", "save_failures", True))
    save_only_anomalies = bool(cfg_get(cfg, "run", "save_only_anomalies", False))

    # CLI overrides
    if args.max_workers is not None:
        max_workers = int(args.max_workers)
    if args.threads_per_worker is not None:
        threads_per_worker = int(args.threads_per_worker)
    if args.series_limit is not None:
        series_limit = int(args.series_limit)

    sb = parse_bool_override(args.save_only_anomalies)
    if sb is not None:
        save_only_anomalies = sb
    ff = parse_bool_override(args.fail_fast)
    if ff is not None:
        fail_fast = ff

    if args.save_mode is not None:
        save_mode = str(args.save_mode).strip().lower()
    if args.feature_prefix_template is not None:
        feature_prefix_template = str(args.feature_prefix_template)
    if args.feature_cols is not None:
        feature_cols = parse_csv_list(args.feature_cols, default_feature_cols)
    prm = parse_bool_override(args.prefix_run_metadata)
    if prm is not None:
        prefix_run_metadata = prm

    enabled_models_default = [x.strip() for x in str(cfg_get(cfg, "models", "enabled", "STRAY")).split(",") if x.strip()]
    enabled_models, model_specs = build_model_specs(cfg, enabled_models_default, args.models)

    return AppConfig(
        src_db_user=src_db_user, src_db_pass=src_db_pass, src_db_host=src_db_host, src_db_port=str(src_db_port), src_db_name=src_db_name,
        tgt_db_user=tgt_db_user, tgt_db_pass=tgt_db_pass, tgt_db_host=tgt_db_host, tgt_db_port=str(tgt_db_port), tgt_db_name=tgt_db_name,
        src_schema=src_schema, src_table=src_table, where_sql=str(where_sql or ""),
        out_schema=out_schema, out_table=out_table, failures_table=failures_table,
        if_exists=if_exists, chunksize=chunksize, flush_every=flush_every,
        save_mode=save_mode, feature_prefix_template=feature_prefix_template, feature_cols=feature_cols, prefix_run_metadata=prefix_run_metadata,
        min_len=min_len, series_limit=series_limit,
        max_workers=max_workers, threads_per_worker=threads_per_worker,
        fail_fast=fail_fast, save_failures=save_failures, save_only_anomalies=save_only_anomalies,
        enabled_models=enabled_models, model_specs=model_specs,
    )

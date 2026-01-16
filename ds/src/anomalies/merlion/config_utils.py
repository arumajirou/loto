# /mnt/e/env/ts/tslib/ds/src/anomalies/merlion/config_utils.py
from __future__ import annotations

import ast
import configparser
from dataclasses import dataclass
from typing import Any, Optional


def parse_value(v: Optional[str]) -> Any:
    if v is None:
        return None
    s = v.strip()
    if s == "":
        return ""
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    # int/float
    try:
        if any(c in s for c in (".", "e", "E")):
            return float(s)
        return int(s)
    except ValueError:
        pass
    # list/dict/tuple
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")) or (s.startswith("(") and s.endswith(")")):
        try:
            return ast.literal_eval(s)
        except Exception:
            return s
    return s


def load_ini(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=(";", "#"),  # ← ここ重要（iniの行末コメントで落ちない）
    )
    with open(path, "r", encoding="utf-8") as f:
        cfg.read_file(f)
    return cfg


def cfg_get(cfg: configparser.ConfigParser, section: str, key: str, default: Any = None) -> Any:
    if cfg.has_option(section, key):
        return parse_value(cfg.get(section, key))
    return default


def cfg_section_dict(cfg: configparser.ConfigParser, section: str) -> dict[str, Any]:
    if not cfg.has_section(section):
        return {}
    out: dict[str, Any] = {}
    for k, v in cfg.items(section):
        out[k] = parse_value(v)
    return out


@dataclass(frozen=True)
class DBConn:
    user: str
    password: str
    host: str
    port: int
    dbname: str
    schema: str
    table: str
    where_sql: str = ""


@dataclass(frozen=True)
class OutSpec:
    schema: str
    table: str
    failures_table: str
    if_exists: str
    chunksize: int


@dataclass(frozen=True)
class RunSpec:
    models: list[str]
    series_limit: int
    min_len: int
    train_frac: float
    n_jobs: int
    mp_start_method: str
    flush_rows: int
    fail_fast: bool
    save_failures: bool


@dataclass(frozen=True)
class NamingSpec:
    library: str
    feature_prefix_template: str
    meta_prefix_template: str
    save_mode: str  # wide or long
    include_y: bool
    include_debug_cols: bool

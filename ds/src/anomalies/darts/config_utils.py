# /mnt/e/env/ts/tslib/ds/src/anomalies/darts/config_utils.py
from __future__ import annotations

import ast
import configparser
import os
import json
from dataclasses import dataclass
from typing import Any, Optional


def _strip_inline_comment(s: str) -> str:
    """
    INIでありがちな:
      series_limit = 0 ; 0=無制限
    のような「値の後ろコメント」を安全に除去。
    """
    if s is None:
        return s
    out = []
    i = 0
    in_squote = False
    in_dquote = False
    while i < len(s):
        ch = s[i]
        if ch == "'" and not in_dquote:
            in_squote = not in_squote
            out.append(ch)
            i += 1
            continue
        if ch == '"' and not in_squote:
            in_dquote = not in_dquote
            out.append(ch)
            i += 1
            continue

        if not in_squote and not in_dquote:
            if ch in (";", "#"):
                prev = out[-1] if out else ""
                if prev in (" ", "\t"):
                    break
        out.append(ch)
        i += 1
    return "".join(out).rstrip()


def parse_value(v: Any) -> Any:
    """
    Pythonリテラル（True/False/None, {'a':1}）と
    JSONリテラル（true/false/null, {"a":1}）の両方を吸収する。
    """
    if v is None:
        return None
    if not isinstance(v, str):
        return v

    s = _strip_inline_comment(v).strip()
    if s == "":
        return ""

    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None

    # 数値
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        pass

    # list/dict/tuple っぽい
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")) or (s.startswith("(") and s.endswith(")")):
        # 1) まず Python literal_eval
        try:
            return ast.literal_eval(s)
        except Exception:
            pass

        # 2) JSON として読む（true/false/null を許す）
        try:
            return json.loads(s)
        except Exception:
            pass

        # 3) どうしてもダメなら、JSONっぽいbool/nullをPythonに寄せて再挑戦
        try:
            s2 = (
                s.replace(": true", ": True")
                 .replace(": false", ": False")
                 .replace(": null", ": None")
            )
            return ast.literal_eval(s2)
        except Exception:
            return s

    return s


def load_ini(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if os.path.exists(path):
        cfg.read(path, encoding="utf-8")
    return cfg


def cfg_get(cfg: configparser.ConfigParser, section: str, key: str, default: Any = None) -> Any:
    if cfg.has_option(section, key):
        return parse_value(cfg.get(section, key))
    return default


def default_ini_path(cli_path: Optional[str]) -> str:
    if cli_path:
        return cli_path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "config.ini")


@dataclass
class DbConn:
    user: str
    password: str
    host: str
    port: str
    name: str


@dataclass
class SourceSpec:
    schema: str
    table: str
    where_sql: str


@dataclass
class OutputSpec:
    schema: str
    table: str
    failures_table: str
    if_exists: str
    chunksize: int


@dataclass
class RunSpec:
    series_limit: int
    min_len: int
    n_jobs: int
    parallel_backend: str
    forecast_horizon: int
    start: Any
    train_frac: float
    verbose: bool


@dataclass
class PipelineSpec:
    enabled: list[str]


@dataclass
class PipelineDef:
    name: str
    model: str
    model_params: dict
    scorer: str
    scorer_params: dict
    detector: str
    detector_params: dict
    detector_fit_on: str

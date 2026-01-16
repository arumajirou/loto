from __future__ import annotations

import configparser
import json
import os
from dataclasses import dataclass
from typing import Any, Dict


def _strip_inline_comment(v: str) -> str:
    """Strip inline comments (INI) supporting ';' or '#'"""
    if v is None:
        return ""
    s = str(v)
    for sym in (";", "#"):
        if sym in s:
            s = s.split(sym, 1)[0]
    return s.strip()


def _parse_scalar(v: str) -> Any:
    """Best-effort parse for ini string values.

    Order:
      - JSON objects/arrays
      - bool
      - int
      - float
      - string
    """
    s = _strip_inline_comment(v)
    if s == "":
        return ""

    # JSON (object/array)
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            pass

    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None

    # int
    try:
        if low.startswith("0") and len(low) > 1 and low[1].isdigit():
            # keep leading-zero strings as strings (avoid octal confusion)
            raise ValueError
        return int(s)
    except Exception:
        pass

    # float
    try:
        return float(s)
    except Exception:
        return s


def _as_list(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v)
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


@dataclass
class AppConfig:
    raw: Dict[str, Dict[str, Any]]

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)

    def section(self, section: str) -> Dict[str, Any]:
        return dict(self.raw.get(section, {}))


def load_ini(path: str) -> AppConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")

    raw: Dict[str, Dict[str, Any]] = {}
    for sec in parser.sections():
        raw[sec] = {}
        for k, v in parser.items(sec):
            raw[sec][k] = _parse_scalar(v)
    return AppConfig(raw=raw)


def merge_overrides(cfg: AppConfig, overrides: Dict[str, Any]) -> AppConfig:
    """Merge CLI overrides into config.

    overrides is a dict with keys like "section.key".
    """
    raw = {sec: dict(vals) for sec, vals in cfg.raw.items()}
    for k, v in overrides.items():
        if v is None:
            continue
        if "." not in k:
            continue
        sec, key = k.split(".", 1)
        raw.setdefault(sec, {})[key] = v
    return AppConfig(raw=raw)


def require(cfg: AppConfig, section: str, key: str) -> Any:
    v = cfg.get(section, key, None)
    if v is None or v == "":
        raise ValueError(f"Missing required config: [{section}] {key}")
    return v


def parse_models(cfg: AppConfig) -> list[str]:
    models = _as_list(cfg.get("pypots", "models", "DLinear"))
    return models or ["DLinear"]


def parse_devices(cfg: AppConfig) -> list[str]:
    device = cfg.get("pypots", "device", "cpu")
    if isinstance(device, list):
        return [str(x) for x in device]
    s = str(device).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            return [str(x) for x in arr]
        except Exception:
            pass
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]

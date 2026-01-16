# /mnt/e/env/ts/tslib/ds/src/db/utils/format_utils.py
from __future__ import annotations
from typing import Any, Iterable, List

def _one_line(s: str) -> str:
    return " ".join(s.split())

def format_value(v: Any, max_chars: int = 140) -> str:
    if v is None:
        return "NULL"
    s = str(v)
    s = _one_line(s)
    if len(s) > max_chars:
        s = s[: max_chars - 1] + "…"
    return s

def format_bullets(values: Iterable[Any], max_items: int = 50, max_chars: int = 140) -> str:
    vals = list(values)[:max_items]
    if not vals:
        return "- (なし)"
    lines: List[str] = []
    for v in vals:
        lines.append(f"- {format_value(v, max_chars=max_chars)}")
    return "\n".join(lines)

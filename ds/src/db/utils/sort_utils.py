# /mnt/e/env/ts/tslib/ds/src/db/utils/sort_utils.py
from __future__ import annotations
from typing import Iterable, List, Optional, TypeVar, Any

T = TypeVar("T")

def sorted_alpha(items: Iterable[T]) -> List[T]:
    """
    アルファベット順（大小無視）で安定ソートして返す。
    None が混じっても落ちないように末尾扱いにする。
    """
    def key_func(x: Any) -> tuple[int, str]:
        if x is None:
            return (1, "")
        return (0, str(x).casefold())

    return sorted(list(items), key=key_func)

# /mnt/e/env/ts/tslib/ds/src/db/services/pg_unique_values.py
from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine


def quote_ident(ident: str) -> str:
    """
    SQL識別子（schema/table/column）を安全にダブルクォートでエスケープする。
    例: a"b -> "a""b"
    """
    return '"' + ident.replace('"', '""') + '"'


def fetch_top_unique_values(
    engine: Engine,
    schema: str,
    table: str,
    column: str,
    limit: int = 50,
) -> List[Any]:
    """
    指定カラムのユニーク値 上位(limit)件を頻度順で返す。
    返り値は value のリスト（型はDBの実型のまま）。
    """
    q_schema = quote_ident(schema)
    q_table = quote_ident(table)
    q_col = quote_ident(column)

    # value の型によっては ORDER BY value が失敗することがあるので、まずは count 降順のみで安定させる
    sql = f"""
        SELECT {q_col} AS value, COUNT(*) AS n
        FROM {q_schema}.{q_table}
        GROUP BY {q_col}
        ORDER BY n DESC
        LIMIT :limit
    """

    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"limit": int(limit)}).fetchall()

    return [r[0] for r in rows]

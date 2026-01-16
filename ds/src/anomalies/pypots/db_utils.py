from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


def quote_ident(name: str) -> str:
    """Double-quote an identifier (PostgreSQL-safe)."""
    s = str(name).replace('"', '""')
    return f'"{s}"'


@dataclass(frozen=True)
class TableRef:
    db_name: str
    schema: str
    table: str

    @property
    def full_name(self) -> str:
        return f"{quote_ident(self.schema)}.{quote_ident(self.table)}"


def make_engine_from_config(section: Dict[str, Any]) -> Engine:
    """Build a SQLAlchemy engine from a config section.

    Supported patterns:
      - url=postgresql+psycopg2://user:pass@host:5432/db
      - name=dataset -> env var DB_URL_DATASET is used if present
      - host/port/user/password/database + optional driver
    """
    url = section.get("url")
    if url:
        return create_engine(url, pool_pre_ping=True)

    name = section.get("name")
    if name:
        env_key = f"DB_URL_{str(name).upper()}"
        env_url = os.environ.get(env_key)
        if env_url:
            return create_engine(env_url, pool_pre_ping=True)

    driver = section.get("driver", "postgresql+psycopg2")
    host = section.get("host")
    port = section.get("port", 5432)
    user = section.get("user")
    password = section.get("password")
    database = section.get("database")
    if not all([host, user, database]):
        raise ValueError(
            "DB config must provide either url=..., name=... (with env DB_URL_NAME), "
            "or host/user/database."
        )
    # password may be empty
    if password is None:
        password = ""
    return create_engine(
        f"{driver}://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
    )


def ensure_schema(engine: Engine, schema: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)}"))


def ensure_table(engine: Engine, tref: TableRef, columns_sql: List[Tuple[str, str]]) -> None:
    """Create table if not exists with given columns.

    columns_sql: list of (colname, sql_type_expr)
    """
    ensure_schema(engine, tref.schema)
    cols = ",\n  ".join([f"{quote_ident(c)} {t}" for c, t in columns_sql])
    ddl = f"CREATE TABLE IF NOT EXISTS {tref.full_name} (\n  {cols}\n)"
    with engine.begin() as conn:
        conn.execute(text(ddl))


def ensure_columns(engine: Engine, tref: TableRef, columns_sql: List[Tuple[str, str]]) -> None:
    """Add missing columns via ALTER TABLE ... ADD COLUMN IF NOT EXISTS."""
    insp = inspect(engine)
    existing = {c["name"] for c in insp.get_columns(tref.table, schema=tref.schema)}
    missing = [(c, t) for (c, t) in columns_sql if c not in existing]
    if not missing:
        return
    with engine.begin() as conn:
        for c, t in missing:
            conn.execute(text(f"ALTER TABLE {tref.full_name} ADD COLUMN IF NOT EXISTS {quote_ident(c)} {t}"))


def ensure_indexes(engine: Engine, tref: TableRef, index_sql_statements: Iterable[str]) -> None:
    with engine.begin() as conn:
        for stmt in index_sql_statements:
            conn.execute(text(stmt))


def read_series_keys(
    engine: Engine,
    source_schema: str,
    source_table: str,
    where_sql: str | None,
    series_limit: int | None,
    min_len: int,
) -> pd.DataFrame:
    where = f"WHERE {where_sql}" if where_sql else ""
    limit = f"LIMIT {int(series_limit)}" if series_limit else ""
    sql = f"""
    SELECT loto, unique_id, ts_type, COUNT(*) AS series_len
    FROM {quote_ident(source_schema)}.{quote_ident(source_table)}
    {where}
    GROUP BY 1,2,3
    HAVING COUNT(*) >= :min_len
    ORDER BY series_len DESC
    {limit}
    """
    return pd.read_sql(text(sql), engine, params={"min_len": int(min_len)})


def read_one_series(
    engine: Engine,
    source_schema: str,
    source_table: str,
    where_sql: str | None,
    loto: Any,
    unique_id: Any,
    ts_type: Any,
) -> pd.DataFrame:
    where_parts = ["loto = :loto", "unique_id = :unique_id", "ts_type = :ts_type"]
    params = {"loto": loto, "unique_id": unique_id, "ts_type": ts_type}
    if where_sql:
        where_parts.append(f"({where_sql})")
    where = "WHERE " + " AND ".join(where_parts)
    sql = f"""
    SELECT loto, unique_id, ts_type, ds, y
    FROM {quote_ident(source_schema)}.{quote_ident(source_table)}
    {where}
    ORDER BY ds ASC
    """
    df = pd.read_sql(text(sql), engine, params=params)
    return df


def write_dataframe(
    engine: Engine,
    tref: TableRef,
    df: pd.DataFrame,
    if_exists: str = "append",
    chunksize: int = 5000,
) -> None:
    if df is None or df.empty:
        return
    df.to_sql(
        name=tref.table,
        schema=tref.schema,
        con=engine,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
    )

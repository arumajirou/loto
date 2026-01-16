# /mnt/e/env/ts/tslib/ds/src/anomalies/darts/db_io.py
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def make_pg_url(user: str, password: str, host: str, port: str, db: str) -> str:
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def make_engine(pg_url: str) -> Engine:
    return create_engine(pg_url, pool_pre_ping=True)


def load_loto_base(engine: Engine, schema: str, table: str, where_sql: str = "") -> pd.DataFrame:
    where_clause = f"WHERE {where_sql}" if where_sql else ""
    sql = text(f"""
        SELECT loto, unique_id, ts_type, ds, y
        FROM {schema}."{table}"
        {where_clause}
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    return df


def ensure_schema(engine: Engine, schema: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))


def save_results(
    df: pd.DataFrame,
    engine: Engine,
    schema: str,
    table: str,
    if_exists: str,
    chunksize: int,
) -> None:
    df.to_sql(
        table,
        engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
    )


def save_failures(df: pd.DataFrame, engine: Engine, schema: str, table: str) -> None:
    if df is None or len(df) == 0:
        return
    df.to_sql(
        table,
        engine,
        schema=schema,
        if_exists="append",
        index=False,
        chunksize=50000,
        method="multi",
    )


def create_indexes(
    engine: Engine,
    schema: str,
    table: str,
    *,
    anomaly_flag_col: str = "is_anomaly",   # ★ここを可変に
    pipeline_col: str = "pipeline",
    ds_col: str = "ds",
    run_id_col: str = "run_id",
) -> None:
    """
    列名変更に耐えるよう、異常フラグ列を引数化。
    """
    with engine.begin() as conn:
        # キー系
        conn.execute(text(f'''
            CREATE INDEX IF NOT EXISTS idx_{table}_key
            ON "{schema}"."{table}" (loto, unique_id, ts_type, {ds_col}, {pipeline_col});
        '''))

        # 異常フラグ（hist_is_anomaly 等に対応）
        conn.execute(text(f'''
            CREATE INDEX IF NOT EXISTS idx_{table}_anomaly
            ON "{schema}"."{table}" ({anomaly_flag_col}, {pipeline_col});
        '''))

        # run_id
        conn.execute(text(f'''
            CREATE INDEX IF NOT EXISTS idx_{table}_run
            ON "{schema}"."{table}" ({run_id_col});
        '''))

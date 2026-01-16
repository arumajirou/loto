# /mnt/e/env/ts/tslib/ds/src/anomalies/merlion/db_utils.py
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.types import Float, Boolean, DateTime, Integer, Text as SqlText


def make_engine(user: str, password: str, host: str, port: int, dbname: str) -> Engine:
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)


def read_loto_base(engine: Engine, schema: str, table: str, where_sql: str = "") -> pd.DataFrame:
    where_clause = f"WHERE {where_sql}" if where_sql else ""
    q = text(f"""
        SELECT loto, unique_id, ts_type, ds, y
        FROM {schema}."{table}"
        {where_clause}
        ORDER BY loto, unique_id, ts_type, ds
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds"])
    df = df.dropna(subset=["y"])
    return df


def ensure_schema(engine: Engine, schema: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))


def table_has_column(engine: Engine, schema: str, table: str, col: str) -> bool:
    q = text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table AND column_name = :col
        LIMIT 1
    """)
    with engine.connect() as conn:
        r = conn.execute(q, {"schema": schema, "table": table, "col": col}).fetchone()
    return r is not None


def write_df(
    engine: Engine,
    df: pd.DataFrame,
    schema: str,
    table: str,
    if_exists: str,
    chunksize: int,
) -> None:
    dtype = {}
    for c in df.columns:
        # bool
        if c.endswith("__is_anomaly") or c.endswith("_is_anomaly") or c == "is_anomaly":
            dtype[c] = Boolean()
        # float
        elif c.endswith("__score_raw") or c.endswith("__score_pp") or c.endswith("_score") or c == "score":
            dtype[c] = Float()
        # datetime
        elif c in ("ds",) or c.endswith("created_ts"):
            dtype[c] = DateTime()
        # int
        elif c in ("series_len",):
            dtype[c] = Integer()
        # text
        elif c in ("loto", "unique_id", "ts_type", "model", "pipeline", "sfid") or c.endswith("run_id") or c.endswith("params_json"):
            dtype[c] = SqlText()

    df.to_sql(
        table,
        engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
        method="multi",
        dtype=dtype if dtype else None,
    )


def create_indexes(
    engine: Engine,
    schema: str,
    table: str,
    mode: str,
    *,
    run_id_col: str = "run_id",
) -> None:
    """
    - wide: (loto, unique_id, ts_type, ds)
    - long: (loto, unique_id, ts_type, ds, model) があればそれで
    - run_id: 指定列が存在する場合のみ作成（例: merlion__run_id）
    """
    key_cols = ["loto", "unique_id", "ts_type", "ds"]

    # longの場合 model列があるなら入れる
    if mode == "long" and table_has_column(engine, schema, table, "model"):
        key_cols.append("model")

    key_cols_sql = ", ".join([f'"{c}"' for c in key_cols])

    with engine.begin() as conn:
        conn.execute(text(f'''
            CREATE INDEX IF NOT EXISTS idx_{table}_key
            ON "{schema}"."{table}" ({key_cols_sql});
        '''))

        # run_id列は設定次第で名前が変わるので存在確認してから作る
        if table_has_column(engine, schema, table, run_id_col):
            conn.execute(text(f'''
                CREATE INDEX IF NOT EXISTS idx_{table}_run
                ON "{schema}"."{table}" ("{run_id_col}");
            '''))
        else:
            # フォールバック：昔の run_id があればそっち
            if table_has_column(engine, schema, table, "run_id"):
                conn.execute(text(f'''
                    CREATE INDEX IF NOT EXISTS idx_{table}_run
                    ON "{schema}"."{table}" ("run_id");
                '''))
            # なければスキップ（ここで落とさない）

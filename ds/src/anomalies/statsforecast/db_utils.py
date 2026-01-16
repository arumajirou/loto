import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import Float, Boolean, DateTime, Integer, Text as SqlText

RESULT_DTYPES = {
    "loto": SqlText(),
    "unique_id": SqlText(),
    "ts_type": SqlText(),
    "ds": DateTime(),
    "y": Float(),
    "y_hat": Float(),
    "y_lo": Float(),
    "y_hi": Float(),
    "is_anomaly": Boolean(),
    "anomaly_side": SqlText(),
    "anomaly_score": Float(),
    "library": SqlText(),
    "model": SqlText(),
    "level": Float(),
    "fit_ok": Boolean(),
    "error_msg": SqlText(),
    "series_len": Integer(),
    "run_id": SqlText(),
    "created_ts": DateTime(),
}

FAIL_DTYPES = {
    "run_id": SqlText(),
    "created_ts": DateTime(),
    "library": SqlText(),
    "model": SqlText(),
    "sf_unique_id": SqlText(),
    "reason": SqlText(),
    "error_msg": SqlText(),
    "series_len": Integer(),
}

class Database:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url, pool_pre_ping=True)

    def ensure_schema(self, schema: str):
        with self.engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))

    def load_source(self, schema: str, table: str, where_sql: str):
        where_clause = f"WHERE {where_sql}" if where_sql else ""
        sql = text(f"""
            SELECT loto, unique_id, ts_type, ds, y
            FROM {schema}."{table}"
            {where_clause}
        """)
        with self.engine.connect() as conn:
            return pd.read_sql(sql, conn)

    def save_results_chunk(self, df: pd.DataFrame, schema: str, table: str, if_exists: str, chunksize: int):
        df.to_sql(
            table,
            self.engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            method="multi",
            dtype=RESULT_DTYPES,
        )

    def save_failures(self, df: pd.DataFrame, schema: str, table: str, chunksize: int):
        df.to_sql(
            table,
            self.engine,
            schema=schema,
            if_exists="append",
            index=False,
            chunksize=chunksize,
            method="multi",
            dtype=FAIL_DTYPES,
        )

    def create_indexes(self, schema: str, table: str, failures_table: str):
        with self.engine.begin() as conn:
            conn.execute(text(f'''
                CREATE INDEX IF NOT EXISTS idx_{table}_key
                ON "{schema}"."{table}" (loto, unique_id, ts_type, ds, model);
            '''))
            conn.execute(text(f'''
                CREATE INDEX IF NOT EXISTS idx_{table}_anomaly
                ON "{schema}"."{table}" (is_anomaly, model);
            '''))
            conn.execute(text(f'''
                CREATE INDEX IF NOT EXISTS idx_{table}_run
                ON "{schema}"."{table}" (run_id);
            '''))
            conn.execute(text(f'''
                CREATE INDEX IF NOT EXISTS idx_{failures_table}_run
                ON "{schema}"."{failures_table}" (run_id, model);
            '''))

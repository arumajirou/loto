# /mnt/e/env/ts/tslib/ds/src/anomalies/sktime/db_utils.py
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import Float, Boolean, DateTime, Integer, Text as SqlText

def infer_result_dtypes(columns: list[str]) -> dict:
    """
    列名から SQLAlchemy 型を推定（prefix付きでも動く）
    """
    dt = {}
    for c in columns:
        if c == "ds" or c.endswith("__ds"):
            dt[c] = DateTime()
        elif c == "y" or c.endswith("__y"):
            dt[c] = Float()
        elif c.endswith("y_hat") or c.endswith("y_lo") or c.endswith("y_hi"):
            dt[c] = Float()
        elif c.endswith("is_anomaly"):
            dt[c] = Boolean()
        elif c.endswith("anomaly_score"):
            dt[c] = Float()
        elif c.endswith("anomaly_side"):
            dt[c] = SqlText()
        elif c.endswith("series_len"):
            dt[c] = Integer()
        elif c.endswith("level"):
            dt[c] = Float()
        elif c.endswith("fit_ok"):
            dt[c] = Boolean()
        elif c.endswith("created_ts"):
            dt[c] = DateTime()
        else:
            # loto, unique_id, ts_type, model, library, run_id, error_msg 等
            dt[c] = SqlText()
    return dt

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
        dtypes = infer_result_dtypes(list(df.columns))
        df.to_sql(
            table,
            self.engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            method="multi",
            dtype=dtypes,
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

    def _table_columns(self, schema: str, table: str) -> set[str]:
        sql = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = :schema AND table_name = :table
        """)
        with self.engine.connect() as conn:
            rows = conn.execute(sql, {"schema": schema, "table": table}).fetchall()
        return {r[0] for r in rows}

    def create_indexes(self, schema: str, table: str, failures_table: str, *, anomaly_col: str = "is_anomaly", model_col: str = "model"):
        cols = self._table_columns(schema, table)

        with self.engine.begin() as conn:
            # key index（存在する列だけで作る）
            key_cols = ["loto", "unique_id", "ts_type", "ds"]
            if model_col in cols:
                key_cols.append(model_col)

            if all(c in cols for c in key_cols):
                conn.execute(text(f'''
                    CREATE INDEX IF NOT EXISTS idx_{table}_key
                    ON "{schema}"."{table}" ({", ".join(key_cols)});
                '''))

            # anomaly index
            if anomaly_col in cols:
                use_cols = [anomaly_col]
                if model_col in cols:
                    use_cols.append(model_col)
                conn.execute(text(f'''
                    CREATE INDEX IF NOT EXISTS idx_{table}_anomaly
                    ON "{schema}"."{table}" ({", ".join(use_cols)});
                '''))

            # run_id index（prefixされてても endswith で拾えるよう、runner側で anomaly_col を渡す設計）
            run_candidates = [c for c in cols if c.endswith("run_id")]
            for rc in run_candidates[:1]:
                conn.execute(text(f'''
                    CREATE INDEX IF NOT EXISTS idx_{table}_run
                    ON "{schema}"."{table}" ({rc});
                '''))

            # failures
            fcols = self._table_columns(schema, failures_table)
            if "run_id" in fcols and "model" in fcols:
                conn.execute(text(f'''
                    CREATE INDEX IF NOT EXISTS idx_{failures_table}_run
                    ON "{schema}"."{failures_table}" (run_id, model);
                '''))

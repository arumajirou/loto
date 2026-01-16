# /mnt/e/env/ts/tslib/ds/src/anomalies/pypots/db_utils.py
import pandas as pd
from sqlalchemy import create_engine
import os

class DBManager:
    def __init__(self):
        # 環境変数や設定から接続情報を取得（適宜書き換えてください）
        self.engine = create_engine("postgresql://user:pass@localhost:5432/postgres")

    def read_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql(query, self.engine)

    def write_table(self, df: pd.DataFrame, table_name: str, schema: str, if_exists='append'):
        df.to_sql(table_name, self.engine, schema=schema, if_exists=if_exists, index=False)
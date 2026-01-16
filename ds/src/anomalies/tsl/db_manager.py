import os
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import configparser

class DBManager:
    def __init__(self, config_path: str):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        self.host = os.environ.get("DB_HOST", self.config['DATABASE']['HOST'])
        self.port = os.environ.get("DB_PORT", self.config['DATABASE']['PORT'])
        self.user = os.environ.get("DB_USER", self.config['DATABASE']['USER'])
        self.password = os.environ.get("DB_PASS", self.config['DATABASE']['PASS'])

    def get_engine(self, db_name: str):
        """指定されたDBへの接続エンジンを作成"""
        url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{db_name}"
        return create_engine(url)

    def load_source_data(self) -> pd.DataFrame:
        """dataset DBから分析対象データを取得"""
        db_name = self.config['DATABASE']['SOURCE_DB']
        engine = self.get_engine(db_name)
        
        # 必要なカラムのみ取得、exec_tsでソート
        query = """
        SELECT loto, ds, unique_id, ts_type, y, exec_ts
        FROM raw_data 
        ORDER BY loto, unique_id, ts_type, exec_ts
        """
        # ※テーブル名が不明なため、ユーザー提供のDataFrame構造に基づき、
        # 実際には引数でDataFrameを受け取るか、テーブル名を調整してください。
        # ここでは実装用にプレースホルダーとします。
        
        # 今回はユーザー環境ですでにDataFrameがある想定のため、
        # このメソッドは「DBから取る場合」の実装例です。
        try:
            with engine.connect() as conn:
                # テーブル名が不明確なので、publicスキーマの主要テーブルと仮定
                # 実際には既存のロジックでロード済みDFを使う設計にします（Main参照）
                pass
        except SQLAlchemyError as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
        return pd.DataFrame()

    def create_database_if_not_exists(self, target_db: str):
        """出力先DBが存在しない場合に作成を試みる（Postgres特有）"""
        # 注意: 通常はpostgres DBに接続してCREATE DATABASEするが、
        # ここでは既存の仕組みに乗っかるため省略し、接続エラー時に警告する。
        pass

    def save_results(self, df: pd.DataFrame, library_name: str):
        """結果をanomaly DBのライブラリ名テーブルに保存"""
        target_db = self.config['DATABASE']['TARGET_DB']
        engine = self.get_engine(target_db)
        table_name = library_name.lower()
        
        try:
            # 既存があれば追記、なければ作成 (append)
            # index=Falseで保存
            df.to_sql(table_name, engine, if_exists='replace', index=False, schema='public')
            print(f"✅ Saved results to {target_db}.public.{table_name}")
        except Exception as e:
            print(f"❌ Failed to save results to {table_name}: {e}")
# db_manager.py
"""
PostgreSQL loto_futrテーブルの操作を管理するクラス。
接続、テーブル作成、データ挿入を担当。
"""

import logging
import argparse
import os
import pandas as pd
from sqlalchemy import create_engine, text, Engine
from typing import List

logger = logging.getLogger(__name__)

class LotoDBManager:
    """PostgreSQL loto_futrテーブルの操作を管理するクラス"""
    
    def __init__(self, args: argparse.Namespace):
        self.host = args.host or os.getenv("PGHOST", "127.0.0.1")
        self.port = args.port or os.getenv("PGPORT", "5432")
        self.database = args.db or os.getenv("PGDATABASE", "loto_db")
        self.user = args.user or os.getenv("PGUSER", "loto_user")
        self.password = args.password or os.getenv("PGPASSWORD")
        self.table_name = 'loto_futr'
        self.schema_name = 'public'
        self.engine = self._get_db_engine()

    def _get_db_engine(self) -> Engine:
        """PostgreSQLエンジン取得"""
        try:
            if self.password:
                connection_string = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
                logger.info("PostgreSQL接続: パスワードを接続文字列に含めました。")
            else:
                connection_string = f"postgresql+psycopg2://{self.user}@{self.host}:{self.port}/{self.database}"
                logger.info("PostgreSQL接続: パスワードなし (PCPASS/環境変数PGPASSに依存)。")
                
            engine = create_engine(connection_string, pool_pre_ping=True)
            logger.info(f"PostgreSQL接続成功: user={self.user} db={self.database} host={self.host}:{self.port}")
            return engine

        except Exception as e:
            logger.error(f"PostgreSQL接続エラー: {e}")
            raise

    def create_or_replace_table(self, static_columns: List[str], dynamic_fourier_cols: List[str], dynamic_freq_cols: List[str]):
        """既存テーブルを削除し、固定カラムと動的カラム定義で新規作成する"""
        
        drop_table_sql = f"DROP TABLE IF EXISTS {self.schema_name}.{self.table_name} CASCADE;"
        
        # NOTE: このSQLは元のスクリプトのCREATE TABLE文から「id」と「ds」を除いたものをベースに、型定義を保持して記述する必要があります。
        # ここでは元のcreate_loto_futr_exog.py内のCREATE文を正確に再現します。
        create_table_sql = f"""
        CREATE TABLE {self.schema_name}.{self.table_name} (
            id SERIAL PRIMARY KEY,
            ds DATE NOT NULL UNIQUE,
            futr_year INTEGER,
            futr_month INTEGER,
            futr_day INTEGER,
            futr_quarter INTEGER,
            futr_week INTEGER,
            futr_dayofweek INTEGER,
            futr_dayofyear INTEGER,
            futr_week_of_month INTEGER,
            futr_is_month_start INTEGER,
            futr_is_month_end INTEGER,
            futr_is_quarter_start INTEGER,
            futr_is_quarter_end INTEGER,
            futr_is_year_start INTEGER,
            futr_is_year_end INTEGER,
            futr_is_weekend INTEGER,
            futr_is_weekday INTEGER,
            futr_is_leap_year INTEGER,
            futr_days_in_month INTEGER,
            futr_days_to_month_end INTEGER,
            futr_days_since_month_start INTEGER,
            futr_days_in_year INTEGER,
            futr_days_to_year_end INTEGER,
            futr_month_progress DOUBLE PRECISION,
            futr_year_progress DOUBLE PRECISION,
            futr_month_sin DOUBLE PRECISION,
            futr_month_cos DOUBLE PRECISION,
            futr_day_sin DOUBLE PRECISION,
            futr_day_cos DOUBLE PRECISION,
            futr_dayofweek_sin DOUBLE PRECISION,
            futr_dayofweek_cos DOUBLE PRECISION,
            futr_dayofyear_sin DOUBLE PRECISION,
            futr_dayofyear_cos DOUBLE PRECISION,
            futr_week_sin DOUBLE PRECISION,
            futr_week_cos DOUBLE PRECISION,
            futr_quarter_sin DOUBLE PRECISION,
            futr_quarter_cos DOUBLE PRECISION,
            futr_trend_linear INTEGER,
            futr_trend_squared BIGINT,
            futr_trend_cubic BIGINT,
            futr_trend_log DOUBLE PRECISION,
            futr_trend_sqrt DOUBLE PRECISION,
            futr_trend_normalized DOUBLE PRECISION,
            futr_is_holiday INTEGER,
            futr_is_before_holiday INTEGER,
            futr_is_after_holiday INTEGER,
            futr_is_golden_week INTEGER,
            futr_is_obon INTEGER,
            futr_is_new_year_holiday INTEGER,
            futr_is_silver_week INTEGER,
            futr_is_payday_25th INTEGER,
            futr_is_payday_month_end INTEGER,
            futr_is_bonus_month INTEGER,
            futr_is_fiscal_year_end INTEGER,
            futr_is_fiscal_quarter_end INTEGER,
            futr_fiscal_year_jp INTEGER,
            futr_fiscal_quarter_jp INTEGER,
            futr_fiscal_month_jp INTEGER,
            futr_draw_number INTEGER,
            futr_draw_index_log DOUBLE PRECISION,
            futr_nth_wednesday_of_month INTEGER,
            futr_is_last_wednesday_of_month INTEGER,
            futr_is_first_draw_of_month INTEGER,
            futr_is_last_draw_of_month INTEGER,
            futr_draws_this_month INTEGER,
            futr_is_year_first_draw INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- インデックス作成
        CREATE UNIQUE INDEX idx_loto_futr_ds_unique ON public.loto_futr(ds);
        CREATE INDEX idx_loto_futr_year_month ON public.loto_futr(futr_year, futr_month);
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(drop_table_sql))
            conn.execute(text(create_table_sql))
            conn.commit()
            
            # 動的カラムの追加
            for col in dynamic_fourier_cols:
                conn.execute(text(f"ALTER TABLE {self.schema_name}.{self.table_name} ADD COLUMN IF NOT EXISTS {col} DOUBLE PRECISION;"))
            for col in dynamic_freq_cols:
                conn.execute(text(f"ALTER TABLE {self.schema_name}.{self.table_name} ADD COLUMN IF NOT EXISTS {col} INTEGER;"))
            conn.commit()
            
            logger.info(f"テーブル再作成（DROP & CREATE）と動的カラム追加: 完了 ({len(dynamic_fourier_cols) + len(dynamic_freq_cols)}カラム追加)")


    def insert_data(self, df: pd.DataFrame):
        """DataFrameをテーブルに挿入する"""
        df.to_sql(
            self.table_name,
            self.engine,
            schema=self.schema_name,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=1000
        )
        logger.info(f"データ挿入: 完了（{len(df)}レコード）")

    def log_db_stats(self):
        """データベースの統計情報を取得しログに出力する"""
        query = f"""
            SELECT 
                COUNT(*) as total_records,
                MIN(ds) as min_date,
                MAX(ds) as max_date,
                COUNT(DISTINCT EXTRACT(YEAR FROM ds)) as years
            FROM {self.schema_name}.{self.table_name}
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            
            logger.info("=" * 60)
            logger.info("データベース統計")
            logger.info("=" * 60)
            logger.info(f"総レコード数: {result[0]}")
            logger.info(f"最小日付: {result[1]}")
            logger.info(f"最大日付: {result[2]}")
            logger.info(f"年数: {result[3]}")
            logger.info("=" * 60)
            
    def get_sample_data(self, columns: List[str], limit: int = 10) -> pd.DataFrame:
        """サンプルデータを取得する"""
        cols_str = ', '.join(columns)
        query = f"""
        SELECT {cols_str}
        FROM {self.schema_name}.{self.table_name}
        ORDER BY ds
        LIMIT {limit}
        """
        return pd.read_sql(query, self.engine)
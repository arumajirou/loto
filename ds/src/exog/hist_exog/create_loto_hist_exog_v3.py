# /mnt/e/env/ts/tslib/ds/src/exog/hist_exog/create_loto_hist_exog_v3.py

import argparse
import sys
import time
import os
import numpy as np
from typing import Dict, List, Any
from sqlalchemy import create_engine  # NameError対策：これを追加

# 分割したファイルからのインポート
from hist_exog_utils import logger, LotoUtils
from db_resource_manager import DBResourceManager, ResourceMonitor
from exog_feature_creator import ExogFeatureCreator 

# 外部ライブラリのインポート（プログレスバー用）
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x 

# ==========================================
# 定数
# ==========================================
DEFAULT_INPUT_TABLE = "loto_base"
DEFAULT_OUTPUT_TABLE = "loto_hist_exog"
DEFAULT_RESOURCE_TABLE = "loto_resource_log"

# ==========================================
# MainProcessor クラス (全体の流れを制御)
# ==========================================
class MainProcessor:
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.t0 = time.time()
        self.engine = None
        self.db_manager = None
        self.monitor = None

    def run(self) -> int:
        logger.info(f"Start Historical Exogenous Feature Creation (Input: {self.args.schema}.{self.args.input_table})")

        # 1. DB接続 (特徴量用)
        try:
            self.engine = LotoUtils.get_db_engine(self.args)
            self.db_manager = DBResourceManager(self.engine, self.args.schema)
        except Exception as e:
            logger.error(f"Failed to initialize DB Manager: {e}")
            return 1

        # 2. リソースログ用データベースエンジンの生成
        log_engine = None
        if getattr(self.args, 'resource_db', None):
            log_url = f"postgresql+psycopg2://{self.args.user}:{self.args.password}@{self.args.host}:{self.args.port}/{self.args.resource_db}"
            log_engine = create_engine(log_url)
            logger.info(f"Resource logs will be saved to separate DB: {self.args.resource_db}")

        # 3. モニターの初期化と開始
        self.monitor = ResourceMonitor(
            resource_table=self.args.resource_table,
            db_manager=self.db_manager,
            resource_engine=log_engine
        )
        self.monitor.start()
        
        df_final = None

        try:
            # --- 重要：ここからが実際の処理ロジック ---
            # Step 1: データ取得
            logger.info(f"Step 1/3: Fetching data and creating SQL-based features...")
            df_base = self.db_manager.fetch_and_preprocess_data(
                self.args.input_table,
                self.args.sql_feature_table
            )
            if df_base.empty:
                logger.error(f"Data not found in {self.args.schema}.{self.args.input_table}.")
                return 1
            
            # Step 2: Pythonでの特徴量生成
            logger.info(f"Step 2/3: Generating complex features in Python (Parallel processing)...")
            creator = ExogFeatureCreator(self.args.n_jobs)
            key_cols = ['loto', 'unique_id', 'ds', 'ts_type']
            df_final = creator.create_features_parallel(
                df_base, 
                tqdm_func=tqdm, 
                key_cols=key_cols
            )

            # Step 3: DBへの保存
            logger.info(f"Step 3/3: Saving final features to {self.args.output_table}...")
            if 'y' in df_final.columns:
                df_final['y'] = np.floor(df_final['y']).fillna(0).astype('Int64')
            self.db_manager.save_exog_features(df_final, self.args.output_table)
            # --------------------------------------

        except Exception as e:
            elapsed = time.time() - self.t0
            rows = len(df_final) if df_final is not None else 0
            self.monitor.log_error(f"Feature Creation Error: {e}", elapsed=elapsed)
            logger.error(f"処理中にエラーが発生しました: {e}", exc_info=True)
            return 1
            
        finally:
            # リソース監視の停止と完了ログ保存
            if self.monitor:
                self.monitor.stop()
                elapsed = time.time() - self.t0
                rows = len(df_final) if df_final is not None else 0
                self.monitor.log_completion(elapsed, rows)
                logger.info(f"Process ended. Elapsed: {elapsed:.1f}s")
            
        return 0

# ==========================================
# メインエントリポイント
# ==========================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create Historical Exogenous Features.")
    
    # 接続設定
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="5432")
    ap.add_argument("--db", default="dataset")
    ap.add_argument("--user", default="postgres")
    ap.add_argument("--password", default="z")
    ap.add_argument("--schema", default="public")
    
    # テーブル設定
    ap.add_argument("--input-table", default=DEFAULT_INPUT_TABLE)
    ap.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE)
    
    # リソースログ設定
    ap.add_argument("--resource-db", default=None, help="Database for resource logs")
    ap.add_argument("--resource-table", default="loto", help="Table for resource logs")
    
    ap.add_argument("--sql-feature-table", default="loto_temp_sql_feat")
    ap.add_argument("--n-jobs", type=int, default=-1)

    args = ap.parse_args()

    processor = MainProcessor(args)
    sys.exit(processor.run())
# create_loto_futr_exog.py
"""
外生変数生成＆PostgreSQL格納スクリプト (リファクタリング版)
===========================================

exog_feature_generator.py から特徴量を生成し、
db_manager.py を使用して public.loto_futr テーブルに格納

Author: AI Research Team
Date: 2025-12-16 (Refactored)
Version: 2.0.0
"""

import sys
import argparse
import logging
import pandas as pd
import warnings
import os

# --- 新しく作成したモジュールをインポート ---
# NOTE: 実行時に sys.path に追加されている必要がある
try:
    from exog_feature_generator import ExogFeatureGenerator
    from db_manager import LotoDBManager
except ImportError as e:
    # 実行環境に応じてパスを調整する
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    from exog_feature_generator import ExogFeatureGenerator
    from db_manager import LotoDBManager
# ---------------------------------------------

warnings.filterwarnings('ignore')

# ==========================================
# ロガー設定
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # ログファイルのパスは環境に合わせて修正が必要な場合があります
        logging.FileHandler('/mnt/e/env/ts/tslib/ds/logs/loto_futr_exog.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# メイン関数
# ==========================================
def main(args: argparse.Namespace):
    """メイン実行"""
    logger.info("=" * 60)
    logger.info("外生変数生成＆PostgreSQL格納 開始 (v2.0.0)")
    logger.info("=" * 60)
    
    try:
        # 1. 特徴量生成オブジェクトの初期化と実行
        generator = ExogFeatureGenerator(max_fourier_k=5, max_freq=100)
        df = generator.generate()
        
        logger.info(f"最終データフレーム: {df.shape}")
        logger.info(f"カラム数: {len(df.columns)}")
        
        # 2. DB操作オブジェクトの初期化
        db_manager = LotoDBManager(args)
        
        # 3. テーブルスキーマの定義と作成/置換
        # ds を除く固定カラム名リストを抽出
        static_cols_with_futr = [col for col in df.columns if not col.startswith(('futr_fourier_', 'futr_freq')) and col != 'ds']
        fourier_cols = [col for col in df.columns if col.startswith('futr_fourier_')]
        freq_cols = [col for col in df.columns if col.startswith('futr_freq')]
        
        db_manager.create_or_replace_table(static_cols_with_futr, fourier_cols, freq_cols)
        
        # 4. データ挿入
        db_manager.insert_data(df)
        
        # 5. 確認クエリとログ
        db_manager.log_db_stats()
        
        # 6. サンプルデータ表示
        sample_cols = [
            'ds', 'futr_year', 'futr_month', 'futr_day',
            'futr_month_sin', 'futr_month_cos',
            'futr_trend_linear',
            'futr_is_holiday',
            'futr_nth_wednesday_of_month',
            'futr_freq2', 'futr_freq3', 'futr_freq10', 'futr_freq52', 'futr_freq100'
        ]
        sample_df = db_manager.get_sample_data(sample_cols, limit=10)
        
        logger.info("\nサンプルデータ（最初の10行）:")
        logger.info("\n" + sample_df.to_string())
        
        logger.info("\n" + "=" * 60)
        logger.info("外生変数生成＆PostgreSQL格納 完了！")
        logger.info("=" * 60)
        
        return True
    
    except Exception as e:
        logger.error(f"エラー発生: {e}", exc_info=True)
        return False
        
if __name__ == '__main__':
    # 引数パーサーの定義は元のスクリプトから引き継ぐ
    ap = argparse.ArgumentParser(description="Generate future exogenous variables and store them in PostgreSQL (loto_futr table).")
    
    # 接続設定引数
    ap.add_argument("--host", default=None, help="PostgreSQL host (default: 127.0.0.1 or PGHOST)")
    ap.add_argument("--port", default=None, help="PostgreSQL port (default: 5432 or PGPORT)")
    ap.add_argument("--db", default=None, help="PostgreSQL database name (default: loto_db or PGDATABASE)")
    ap.add_argument("--user", default=None, help="PostgreSQL user (default: loto_user or PGUSER)")
    ap.add_argument("--password", default=None, help="PostgreSQL password (default: None or PGPASSWORD)")
    
    args = ap.parse_args()
    success = main(args)
    sys.exit(0 if success else 1)
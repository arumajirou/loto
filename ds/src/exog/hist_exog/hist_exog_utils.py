# ds/src/hist_exog/hist_exog_utils.py

import os
import sys
import logging
import argparse
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from typing import Dict, List, Any

# ==========================================
# ロガー設定
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 特徴量パラメータ（共通定数）
# ==========================================
ROLLING_WINDOWS = [7, 14, 28, 56, 91, 182] 
LAGS = [1, 2, 3, 7, 14, 28, 56, 91, 364]
EWM_ALPHAS = [0.1, 0.3, 0.5]
COMPLEX_WINDOWS = [28, 56, 91]

# ==========================================
# ユーティリティクラス
# ==========================================
class LotoUtils:
    """ロギング、DB接続、psql識別子などのユーティリティ集"""

    @staticmethod
    def get_db_engine(args: argparse.Namespace) -> Engine:
        """PostgreSQLエンジン取得（引数または環境変数）"""
        host = args.host or os.getenv("PGHOST", "127.0.0.1")
        port = args.port or os.getenv("PGPORT", "5432")
        db = args.db or os.getenv("PGDATABASE", "loto_db")
        user = args.user or os.getenv("PGUSER", "loto_user")
        password = args.password or os.getenv("PGPASSWORD", "z")

        url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        try:
            engine = create_engine(url)
            logger.info(f"PostgreSQL接続成功: db={db} host={host}:{port}")
            return engine
        except Exception as e:
            logger.error(f"PostgreSQL接続エラー: {e}")
            raise

    @staticmethod
    def qident(name: str) -> str:
        """PostgreSQL識別子を引用符で囲む"""
        return '"' + name.replace('"', '""') + '"'
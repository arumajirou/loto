# ds/src/base/loto_common.py

import argparse
import os
import sys
import logging
import subprocess
from typing import Dict, Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

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

# --------------------------
# 設定・定数
# --------------------------
URLS: Dict[str, str] = {
    "mini": "https://loto-life.net/csv/mini",
    "loto6": "https://loto-life.net/csv/loto6",
    "loto7": "https://loto-life.net/csv/loto7",
    "bingo5": "https://loto-life.net/csv/bingo5",
    "numbers3": "https://loto-life.net/csv/numbers3",
    "numbers4": "https://loto-life.net/csv/numbers4",
}
ENCODINGS_TO_TRY = ["utf-8-sig", "cp932", "shift_jis", "utf-8"]

# --------------------------
# DB接続・ユーティリティ関数
# --------------------------
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

def get_env_conn(args: argparse.Namespace) -> Dict[str, str]:
    """引数または環境変数から接続情報を構築（psqlコマンド用）"""
    host = args.host or os.getenv("PGHOST") or os.getenv("PG_HOST") or "127.0.0.1"
    port = args.port or os.getenv("PGPORT") or os.getenv("PG_PORT") or "5432"
    db = args.db or os.getenv("PGDATABASE") or os.getenv("PG_DB") or "loto_db"
    user = args.user or os.getenv("PGUSER") or os.getenv("PG_USER") or "loto_user"
    password = args.password or os.getenv("PGPASSWORD") or os.getenv("PG_PASSWORD")
    schema = args.schema or os.getenv("PGSCHEMA") or os.getenv("PG_SCHEMA") or "public"

    env = {"host": host, "port": port, "db": db, "user": user, "schema": schema}
    if password:
        env["password"] = password
    return env

def qident(name: str) -> str:
    """PostgreSQL識別子を引用符で囲む"""
    return '"' + name.replace('"', '""') + '"'

def run_psql(sql: str, env: Dict[str, str], psql_bin: str = "psql", capture: bool = False) -> subprocess.CompletedProcess:
    """psqlコマンドを実行する"""
    e = os.environ.copy()
    e["PGHOST"] = env["host"]
    e["PGPORT"] = str(env["port"])
    e["PGDATABASE"] = env["db"]
    e["PGUSER"] = env["user"]
    e["PG_SCHEMA"] = env["schema"]
    if env.get("password"):
        e["PGPASSWORD"] = env["password"]
    else:
        e.pop("PGPASSWORD", None)

    e["PAGER"] = "cat"
    e["PSQLRC"] = "/dev/null"

    args = [psql_bin, "-v", "ON_ERROR_STOP=1"]
    args += ["-v", f"schema={env['schema']}"]

    if capture:
        args += ["-A", "-t", "-F", ","]
    return subprocess.run(
        args,
        input=sql,
        text=True,
        env=e,
        capture_output=capture,
        check=True, # エラー時に例外を発生させる
    )
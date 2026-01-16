import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import argparse

def create_database(target_db, host, port, user, password):
    print(f"--- Creating Database: {target_db} ---")
    print(f"Host: {host}:{port}, User: {user}")
    
    try:
        # 1. デフォルトの postgres データベースに接続
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, dbname="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # 2. 存在確認
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{target_db}'")
        exists = cur.fetchone()

        if not exists:
            # 3. DB作成
            # 識別子としてダブルクォートで囲むことで、ハイフン等が含まれてもエラーにならないようにする
            cur.execute(f'CREATE DATABASE "{target_db}"')
            print(f"✅ Database '{target_db}' created successfully.")
        else:
            print(f"ℹ️ Database '{target_db}' already exists.")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # 引数解析の設定
    parser = argparse.ArgumentParser(description="Create a new PostgreSQL database.")

    # 必須に近い引数 (デフォルト値あり)
    parser.add_argument(
        "--db_name", 
        type=str, 
        default="dataset", 
        help="Name of the database to create (default: dataset)"
    )

    # 接続情報の引数 (環境変数をデフォルト値に使用)
    parser.add_argument("--host", type=str, default=os.environ.get("DB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=str, default=os.environ.get("DB_PORT", "5432"))
    parser.add_argument("--user", type=str, default=os.environ.get("DB_USER", "postgres"))
    parser.add_argument("--password", type=str, default=os.environ.get("DB_PASS", "z"))

    args = parser.parse_args()

    # 関数実行
    create_database(
        target_db=args.db_name,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password
    )
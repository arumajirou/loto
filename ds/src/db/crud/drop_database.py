import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import argparse
import sys

def drop_database(target_db, host, port, user, password, force=False):
    print(f"--- 🗑️ Dropping Database: {target_db} ---")
    
    # 重要なDBを誤って消さないための簡易ガード
    protected_dbs = ['postgres', 'information_schema']
    if target_db in protected_dbs:
        print(f"❌ Error: '{target_db}' is a protected system database.")
        return

    # 最終確認 (forceオプションがない場合)
    if not force:
        confirm = input(f"⚠️ Are you sure you want to PERMANENTLY DELETE database '{target_db}'? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            return

    try:
        # 1. 管理用DB 'postgres' に接続
        # (削除対象のDB自体には接続できないため)
        conn = psycopg2.connect(
            host=host, port=port, user=user, password=password, dbname="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # 2. 既存の接続を強制切断 (これがないと "database is being accessed" エラーになることが多い)
        print(f"Terminating connections to '{target_db}'...")
        kill_connections_sql = f"""
        SELECT pg_terminate_backend(pg_stat_activity.pid)
        FROM pg_stat_activity
        WHERE pg_stat_activity.datname = '{target_db}'
          AND pid <> pg_backend_pid();
        """
        cur.execute(kill_connections_sql)

        # 3. DB削除実行
        print(f"Dropping database '{target_db}'...")
        cur.execute(f'DROP DATABASE IF EXISTS "{target_db}"')
        
        print(f"✅ Database '{target_db}' dropped successfully.")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drop a PostgreSQL database.")

    # 削除対象DB名 (必須)
    parser.add_argument("db_name", type=str, help="Name of the database to drop")

    # オプション
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--host", type=str, default=os.environ.get("DB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=str, default=os.environ.get("DB_PORT", "5432"))
    parser.add_argument("--user", type=str, default=os.environ.get("DB_USER", "postgres"))
    parser.add_argument("--password", type=str, default=os.environ.get("DB_PASS", "z"))

    args = parser.parse_args()

    drop_database(
        target_db=args.db_name,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        force=args.force
    )
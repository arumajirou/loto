import argparse
import time
import subprocess
import sys
import os
import traceback
from datetime import datetime
from sqlalchemy import text
from config import Config
from db_utils import get_engine, create_foundation_model_table

def get_next_task(engine):
    with engine.connect() as conn:
        query = text("""
            UPDATE foundation_model
            SET status = 'RUNNING', start_time = :now
            WHERE id = (
                SELECT id FROM foundation_model
                WHERE status = 'PENDING'
                ORDER BY id ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id, model_name, batch_size, num_workers, limit_rows, context_length
        """)
        result = conn.execute(query, {"now": datetime.now()}).fetchone()
        conn.commit()
        return result

def update_task_status(engine, task_id, status, error_log=None):
    with engine.connect() as conn:
        query = text("""
            UPDATE foundation_model
            SET status = :status, end_time = :now, error_log = :error_log
            WHERE id = :id
        """)
        conn.execute(query, {
            "status": status,
            "now": datetime.now(),
            "error_log": error_log,
            "id": task_id
        })
        conn.commit()

def run_task(task):
    # ジョブ管理DBへの接続
    engine = get_engine(Config.METADATA_DB)
    task_id, model, batch, workers, limit, context_len = task
    
    print(f"\n[Job Runner] Starting Task ID: {task_id} | Model: {model}")
    
    cmd = [
        sys.executable, "main.py",
        "--model", model,
        "--batch_size", str(batch),
        "--num_workers", str(workers),
        "--limit", str(limit)
    ]
    if context_len is not None:
        cmd.extend(["--seq_len", str(context_len)])

    try:
        env = os.environ.copy()
        # メインプロセスに必要な環境変数を渡す
        env["DB_HOST"] = Config.DB_HOST
        env["DB_PORT"] = Config.DB_PORT
        env["DB_USER"] = Config.DB_USER
        env["DB_PASS"] = Config.DB_PASS
        # DB名はConfig内で固定定義されているので渡さなくても良いが、
        # 念のためホスト情報は渡しておく
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            env=env,
            check=True 
        )
        
        print(f"[Job Runner] Task {task_id} COMPLETED.")
        print(result.stdout[-500:])
        update_task_status(engine, task_id, "COMPLETED", error_log=result.stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"[Job Runner] Task {task_id} FAILED.")
        error_msg = f"Exit Code: {e.returncode}\n\nSTDOUT:\n{e.stdout[-2000:]}\n\nSTDERR:\n{e.stderr[-2000:]}"
        print(e.stderr)
        update_task_status(engine, task_id, "ERROR", error_log=error_msg)
    except Exception as e:
        print(f"[Job Runner] System Error: {e}")
        update_task_status(engine, task_id, "ERROR", error_log=str(traceback.format_exc()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_host", type=str)
    # ... 他の引数が必要なら追加
    args = parser.parse_args()
    Config.update_from_args(args)

    print(f"[Job Runner] Metadata DB: {Config.METADATA_DB} | Feature DB: {Config.FEATURE_DB}")

    # 初期化
    create_foundation_model_table()
    
    # ジョブ監視ループ (Metadata DBに接続)
    engine = get_engine(Config.METADATA_DB)
    print("[Job Runner] Waiting for tasks...")
    
    while True:
        try:
            task = get_next_task(engine)
            if task:
                run_task(task)
            else:
                print("[Job Runner] No pending tasks found. Exiting.")
                break
        except Exception as e:
            print(f"[Job Runner] Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
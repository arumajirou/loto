import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError
from typing import List, Tuple
from config import Config

def get_engine(db_name: str) -> Engine:
    return create_engine(Config.get_db_url(db_name))

def create_database_if_not_exists(target_db_name: str):
    """DBが存在しない場合に作成"""
    sys_engine = create_engine(Config.get_db_url("postgres"), isolation_level="AUTOCOMMIT")
    with sys_engine.connect() as conn:
        exists = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{target_db_name}'")).scalar()
        if not exists:
            print(f"[DB] Creating database: {target_db_name}")
            try:
                conn.execute(text(f"CREATE DATABASE {target_db_name}"))
            except ProgrammingError as e:
                print(f"[DB] Error creating database: {e}")

def sanitize_table_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_").lower()

# --- Feature DB (保存用) への操作 ---

def create_model_table(model_name: str, embedding_dim: int) -> Tuple[str, List[str]]:
    """特徴量テーブルを作成 (FEATURE_DB: model_exog)"""
    table_name = sanitize_table_name(model_name)
    engine = get_engine(Config.FEATURE_DB) # ★変更点
    
    feature_cols = [f"hist_{table_name}_{i}" for i in range(embedding_dim)]
    
    base_cols_def = """
        loto TEXT,
        unique_id TEXT,
        ts_type TEXT,
        ds TIMESTAMP,
    """
    feature_cols_def = ",\n        ".join([f"{col} FLOAT" for col in feature_cols])
    pk_def = "PRIMARY KEY (loto, unique_id, ts_type, ds)"
    
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {base_cols_def}
        {feature_cols_def},
        {pk_def}
    );
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    return table_name, feature_cols

def create_resource_table(model_name: str):
    """リソースログテーブルを作成 (RESOURCE_DB)"""
    table_name = sanitize_table_name(model_name)
    engine = get_engine(Config.RESOURCE_DB)
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        cpu_percent FLOAT,
        ram_used_gb FLOAT,
        gpu_util_percent FLOAT,
        vram_used_gb FLOAT,
        disk_read_mb FLOAT,
        disk_write_mb FLOAT
    );
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    return table_name

# --- Metadata DB (設定用) への操作 ---

def create_foundation_model_table():
    """ジョブ管理テーブルを作成 (METADATA_DB: foundation_model)"""
    engine = get_engine(Config.METADATA_DB) # ★変更点
    
    ddl = """
    CREATE TABLE IF NOT EXISTS foundation_model (
        id SERIAL PRIMARY KEY,
        model_name TEXT NOT NULL,
        batch_size INT DEFAULT 16,
        num_workers INT DEFAULT 4,
        limit_rows INT DEFAULT 0,
        context_length INT,
        status TEXT DEFAULT 'PENDING',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        execution_command TEXT,
        error_log TEXT
    );
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    print(f"[DB] Table 'foundation_model' is ready in {Config.METADATA_DB}.")

def get_hf_models_from_snapshots() -> pd.DataFrame:
    """hf_snapshotsからモデル情報を取得 (METADATA_DB: foundation_model)"""
    engine = get_engine(Config.METADATA_DB) # ★変更点
    try:
        query = """
        SELECT DISTINCT ON (repo_id)
            repo_id,
            context_length,
            model_type,
            status
        FROM hf_snapshots
        ORDER BY repo_id, created_at DESC
        """
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"[Warning] Could not read hf_snapshots from {Config.METADATA_DB}: {e}")
        return pd.DataFrame()
import pandas as pd
from sqlalchemy import text
from db_utils import get_engine, create_foundation_model_table, get_hf_models_from_snapshots
from config import Config

def register_job(model_name, batch_size=16, limit=0, context_length=None):
    """
    【手動登録用】指定したモデルを1つ登録する
    """
    engine = get_engine(Config.METADATA_DB)
    
    # コンテキスト長の簡易自動設定 (指定がない場合)
    if context_length is None:
        lower_name = model_name.lower()
        if "patchtst" in lower_name:
            context_length = 512
        elif "chronos" in lower_name and "small" in lower_name:
            context_length = 512
        elif "timesfm" in lower_name:
            context_length = 2048
    
    query = text("""
        INSERT INTO foundation_model (model_name, batch_size, limit_rows, context_length, status)
        VALUES (:model, :batch, :limit, :ctx, 'PENDING')
    """)
    
    with engine.connect() as conn:
        conn.execute(query, {
            "model": model_name, 
            "batch": batch_size, 
            "limit": limit,
            "ctx": context_length
        })
        conn.commit()
    
    print(f"[Register] Manually registered: {model_name} (Ctx: {context_length})")

def register_from_snapshots(batch_size_default=16, limit_default=0):
    """
    【自動登録用】hf_snapshotsテーブルから一括登録する
    """
    engine = get_engine(Config.METADATA_DB)
    
    print(f"[Register] Fetching models from {Config.METADATA_DB}.hf_snapshots...")
    df_models = get_hf_models_from_snapshots()
    
    if df_models.empty:
        print("[Error] No models found or table missing.")
        return

    existing_jobs = set()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT model_name FROM foundation_model WHERE status IN ('PENDING', 'RUNNING')")).fetchall()
            existing_jobs = {r[0] for r in rows}
    except Exception:
        pass

    registered_count = 0
    with engine.connect() as conn:
        for _, row in df_models.iterrows():
            model_name = row['repo_id']
            ctx_len = row['context_length']
            if pd.isna(ctx_len):
                ctx_len = None
            else:
                ctx_len = int(ctx_len)

            if model_name in existing_jobs:
                continue

            # 安全なモデルのみ許可するフィルタ (一時的)
            safe_keywords = ["granite", "chronos", "gpt2", "timesfm"]
            if not any(k in model_name.lower() for k in safe_keywords):
                continue

            current_batch = batch_size_default
            if "large" in model_name.lower() or "500m" in model_name.lower():
                current_batch = 4

            query = text("""
                INSERT INTO foundation_model (model_name, batch_size, limit_rows, context_length, status)
                VALUES (:model, :batch, :limit, :ctx, 'PENDING')
            """)
            
            conn.execute(query, {
                "model": model_name, 
                "batch": current_batch, 
                "limit": limit_default,
                "ctx": ctx_len
            })
            registered_count += 1
            print(f"[Register] Added: {model_name} (Ctx: {ctx_len}, Batch: {current_batch})")
        
        conn.commit()
    
    print(f"\n[Done] Registered {registered_count} safe jobs.")

def main():
    create_foundation_model_table()
    # デフォルトではスナップショットからの自動登録を行う
    # register_from_snapshots() 
    pass 

if __name__ == "__main__":
    main()
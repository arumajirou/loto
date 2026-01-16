import os
import sys
import io
import csv
import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from tqdm import tqdm
from chronos import ChronosPipeline

# ==========================================
# 設定
# ==========================================
MODEL_ID = "amazon/chronos-t5-small" 
BATCH_SIZE = 32
# Chronos T5の制限（通常512）に合わせて分割サイズを設定
# ギリギリだと特殊トークン等の影響があるため、少し余裕を持たせるか512とする
# ここでは安全のため512単位で処理します
CHUNK_SIZE = 512 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DB接続設定
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASS = os.getenv("PGPASSWORD", "z")
PG_HOST = os.getenv("PGHOST", "127.0.0.1")
PG_PORT = os.getenv("PGPORT", "5432")
PG_DB   = os.getenv("PGDATABASE", "loto_db")
DB_URL  = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

class LotoChronosFeatureGenerator:
    def __init__(self):
        self.engine = create_engine(DB_URL)
        print(f"--- Initializing Chronos Pipeline ({MODEL_ID}) on {DEVICE} ---")
        
        self.pipeline = ChronosPipeline.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,
            torch_dtype=torch.float32,
        )
        print("Model loaded successfully.")

    def load_data(self, table_name="loto_base"):
        print(f"--- Loading data from {table_name} ---")
        query = f"""
            SELECT loto, unique_id, ts_type, ds, y 
            FROM public."{table_name}"
            ORDER BY loto, unique_id, ts_type, ds
        """
        try:
            df = pd.read_sql(query, self.engine)
            df['ds'] = pd.to_datetime(df['ds'])
            print(f"Data loaded: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def preprocess_series(self, y_series):
        y_values = torch.tensor(y_series, dtype=torch.float32)
        return y_values

    def generate_features(self, df):
        print("--- Generating Chronos Embeddings (Chunked) ---")
        
        grouped = df.groupby(['loto', 'unique_id', 'ts_type'])
        final_data = []
        batch_contexts = []
        batch_meta = []
        
        for name, group in tqdm(grouped, desc="Processing"):
            y_values = group['y'].values
            ds_values = group['ds'].values
            total_len = len(y_values)
            
            if total_len == 0: continue
            
            # --- 修正: 長い系列を CHUNK_SIZE ごとに分割して登録 ---
            # これにより全期間の特徴量を取得する
            for start_idx in range(0, total_len, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, total_len)
                
                # 分割したデータ
                chunk_y = y_values[start_idx:end_idx]
                chunk_len = len(chunk_y)
                
                context_tensor = self.preprocess_series(chunk_y)
                
                batch_contexts.append(context_tensor)
                batch_meta.append({
                    'key': name,
                    'ds_chunk': ds_values[start_idx:end_idx], # 対応する日付部分
                    'length': chunk_len
                })
                
                # バッチサイズに達したら推論
                if len(batch_contexts) >= BATCH_SIZE:
                    self._process_batch(batch_contexts, batch_meta, final_data)
                    batch_contexts = []
                    batch_meta = []
        
        # 残りのバッチ
        if batch_contexts:
            self._process_batch(batch_contexts, batch_meta, final_data)
            
        result_df = pd.DataFrame(final_data)
        print(f"Generated features for {len(result_df)} points.")
        return result_df

    def _process_batch(self, contexts, metas, result_list):
        with torch.no_grad():
            embeddings, _ = self.pipeline.embed(contexts)
            
        embeddings = embeddings.float().cpu().numpy()
        
        for i, meta in enumerate(metas):
            loto, unique_id, ts_type = meta['key']
            ds_chunk = meta['ds_chunk']
            original_len = meta['length']
            
            # ChronosはEOSトークンなどを付ける場合があるため長さを合わせる
            seq_emb = embeddings[i]
            valid_emb = seq_emb[:original_len]
            
            for t in range(len(valid_emb)):
                if t < len(ds_chunk):
                    features = valid_emb[t]
                    
                    result_list.append({
                        'loto': loto,
                        'unique_id': unique_id,
                        'ts_type': ts_type,
                        'ds': ds_chunk[t], # 正しい日付
                        'hist_chronos_embed': features
                    })

    def save_to_db(self, df, table_name="loto_chronos_features"):
        if df is None or df.empty:
            print("No data to save.")
            return

        print(f"--- Saving to {table_name} using FAST COPY (Flattened) ---")
        print("Flattening embeddings into columns...")
        
        embed_df = pd.DataFrame(
            df['hist_chronos_embed'].tolist(), 
            index=df.index
        ).add_prefix('hist_chronos_embed_')
        
        df_flat = pd.concat([df[['loto', 'unique_id', 'ts_type', 'ds']], embed_df], axis=1)
        embed_cols = embed_df.columns.tolist()
        print(f"Expanded to {len(embed_cols)} feature columns (Prefix: hist_chronos_embed_).")

        conn = self.engine.raw_connection()
        cur = conn.cursor()
        
        try:
            print("Recreating table...")
            embed_col_defs = ", ".join([f'"{col}" FLOAT' for col in embed_cols])
            
            create_table_sql = f'''
                CREATE TABLE "{table_name}" (
                    loto TEXT,
                    unique_id TEXT,
                    ts_type TEXT,
                    ds TIMESTAMP,
                    {embed_col_defs}
                )
            '''
            
            cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            cur.execute(create_table_sql)
            conn.commit()

            print("Writing data to memory buffer...")
            output = io.StringIO()
            df_flat.to_csv(
                output, 
                sep='\t', 
                header=False, 
                index=False,
                quoting=csv.QUOTE_NONE,
                escapechar='\\'
            )
            output.seek(0)

            print("Executing COPY command...")
            cur.copy_expert(
                f'COPY "{table_name}" FROM STDIN WITH (FORMAT CSV, DELIMITER E\'\\t\')', 
                output
            )
            conn.commit()
            
            print("Creating indexes...")
            cur.execute(f'CREATE INDEX idx_{table_name}_keys ON "{table_name}" (loto, unique_id, ts_type, ds)')
            conn.commit()
            
            print(f"Successfully saved {len(df)} rows to {table_name}.")
            
        except Exception as e:
            print(f"Error saving data: {e}")
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def run(self):
        df = self.load_data()
        if df is not None:
            features_df = self.generate_features(df)
            self.save_to_db(features_df)
            print("All processes completed.")

if __name__ == "__main__":
    generator = LotoChronosFeatureGenerator()
    generator.run()
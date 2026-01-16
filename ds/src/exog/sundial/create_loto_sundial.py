import os
import sys
import io
import csv
import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM

# ==========================================
# 設定
# ==========================================
MODEL_ID = "thuml/sundial-base-128m"
BATCH_SIZE = 8
CHUNK_SIZE = 512
PCA_COMPONENTS = 64

# デバイスの自動選択
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# DB接続設定
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASS = os.getenv("PGPASSWORD", "z")
PG_HOST = os.getenv("PGHOST", "127.0.0.1")
PG_PORT = os.getenv("PGPORT", "5432")
PG_DB   = os.getenv("PGDATABASE", "loto_db")
DB_URL  = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

class LotoSundialFeatureGenerator:
    def __init__(self):
        self.engine = create_engine(DB_URL)
        print(f"--- Initializing Sundial ({MODEL_ID}) on {DEVICE} ---")
        
        # 修正: 常に float32 でロードする（CUDAでも）
        # Sundialの内部実装との互換性を保つため
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            torch_dtype=torch.float32, 
        )
        self.model.eval()
        self.model.to(DEVICE)
        
        self.d_model = getattr(self.model.config, "n_embd", 768)
        self.patch_size = getattr(self.model.config, "patch_length", 16)
        
        print(f"Model loaded successfully.")
        print(f"Embedding Dimension: {self.d_model}")
        print(f"Patch Size: {self.patch_size}")
        print(f"Target PCA Dimension: {PCA_COMPONENTS}")

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
        y = np.array(y_series, dtype=np.float32)
        nans = np.isnan(y)
        if np.any(nans):
            x = np.arange(len(y))
            if np.all(nans):
                y[:] = 0.0
            else:
                y[nans] = np.interp(x[nans], x[~nans], y[~nans])
        
        mean = np.mean(y)
        std = np.std(y) + 1e-5
        y = (y - mean) / std

        return y

    def generate_features(self, df):
        print("--- Generating Sundial Embeddings (Chunked + PCA) ---")
        
        grouped = df.groupby(['loto', 'unique_id', 'ts_type'])
        
        raw_embeddings_map = [] 
        batch_inputs = []
        batch_metas = []
        
        # 1. Inference
        for name, group in tqdm(grouped, desc="Inference"):
            y_values = group['y'].values
            ds_values = group['ds'].values
            total_len = len(y_values)
            
            if total_len == 0: continue
            
            for start_idx in range(0, total_len, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, total_len)
                chunk_len = end_idx - start_idx
                
                chunk_y = y_values[start_idx:end_idx]
                
                processed_y = self.preprocess_series(chunk_y)
                
                final_pad_len = CHUNK_SIZE - len(processed_y)
                if final_pad_len > 0:
                    processed_y = np.pad(processed_y, (0, final_pad_len), 'constant')
                
                batch_inputs.append(processed_y)
                batch_metas.append({
                    'key': name,
                    'ds_chunk': ds_values[start_idx:end_idx],
                    'valid_len': chunk_len
                })
                
                if len(batch_inputs) >= BATCH_SIZE:
                    self._process_batch_inference(batch_inputs, batch_metas, raw_embeddings_map)
                    batch_inputs = []
                    batch_metas = []
        
        if batch_inputs:
             self._process_batch_inference(batch_inputs, batch_metas, raw_embeddings_map)
        
        # 2. PCA Training & Transform
        print("Step 2: PCA Training & Transformation...")
        if not raw_embeddings_map:
            return pd.DataFrame()

        all_embeddings = np.array([item['embedding'] for item in raw_embeddings_map])
        
        pca = PCA(n_components=PCA_COMPONENTS)
        pca_embeddings = pca.fit_transform(all_embeddings)
        print(f"Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        # 3. Formatting
        print("Step 3: Formatting...")
        final_data = []
        
        for i, item in enumerate(raw_embeddings_map):
            meta = item['meta']
            reduced_emb = pca_embeddings[i]
            
            ds_chunk = meta['ds_chunk']
            valid_len = meta['valid_len']
            loto, unique_id, ts_type = meta['key']
            
            for t in range(valid_len):
                if t < len(ds_chunk):
                    final_data.append({
                        'loto': loto,
                        'unique_id': unique_id,
                        'ts_type': ts_type,
                        'ds': ds_chunk[t],
                        'hist_sundial_embed': reduced_emb.tolist()
                    })
                    
        result_df = pd.DataFrame(final_data)
        print(f"Generated features for {len(result_df)} points.")
        return result_df

    def _process_batch_inference(self, inputs_list, metas, output_list):
        # 修正: float32でTensorを作成
        batch_tensor = torch.tensor(np.array(inputs_list), dtype=torch.float32)
        batch_tensor = batch_tensor.to(DEVICE)

        with torch.no_grad():
            outputs = self.model(
                batch_tensor, 
                output_hidden_states=True,
                return_dict=True
            )
            
        last_hidden_state = outputs.hidden_states[-1] # (B, P, D)
        embeddings_batch = last_hidden_state.float().cpu().numpy()
        
        # Pooling
        embeddings_pooled = embeddings_batch.mean(axis=1) # (B, D)
        
        for i, emb in enumerate(embeddings_pooled):
            output_list.append({
                'meta': metas[i],
                'embedding': emb
            })

    def save_to_db(self, df, table_name="loto_sundial_features"):
        if df is None or df.empty:
            print("No data to save.")
            return

        print(f"--- Saving to {table_name} using FAST COPY (Flattened Columns) ---")
        print("Flattening embeddings into columns...")
        
        embed_df = pd.DataFrame(
            df['hist_sundial_embed'].tolist(), 
            index=df.index
        ).add_prefix('hist_sundial_embed_')
        
        df_flat = pd.concat([df[['loto', 'unique_id', 'ts_type', 'ds']], embed_df], axis=1)
        embed_cols = embed_df.columns.tolist()
        print(f"Expanded to {len(embed_cols)} feature columns.")

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

            print("Writing to CSV buffer...")
            output = io.StringIO()
            df_flat.to_csv(output, sep='\t', header=False, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
            output.seek(0)

            print("Executing COPY...")
            cur.copy_expert(f'COPY "{table_name}" FROM STDIN WITH (FORMAT CSV, DELIMITER E\'\\t\')', output)
            conn.commit()
            
            print("Creating index...")
            cur.execute(f'CREATE INDEX idx_{table_name}_keys ON "{table_name}" (loto, unique_id, ts_type, ds)')
            conn.commit()
            
            print(f"Successfully saved {len(df)} rows.")
            
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
    generator = LotoSundialFeatureGenerator()
    generator.run()
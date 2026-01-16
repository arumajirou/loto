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

# Totoのインポート
try:
    from toto.model.toto import Toto
    from toto.data.util.dataset import MaskedTimeseries
except ImportError:
    print("Error: 'toto-ts' not found. Please run 'pip install toto-ts'")
    sys.exit(1)

# ==========================================
# 設定
# ==========================================
MODEL_ID = "Datadog/Toto-Open-Base-1.0"
BATCH_SIZE = 8
CHUNK_SIZE = 512
PCA_COMPONENTS = 64

# デバイス自動選択
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# DB接続設定
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASS = os.getenv("PGPASSWORD", "z")
PG_HOST = os.getenv("PGHOST", "127.0.0.1")
PG_PORT = os.getenv("PGPORT", "5432")
PG_DB   = os.getenv("PGDATABASE", "loto_db")
DB_URL  = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

class LotoTotoFeatureGenerator:
    def __init__(self):
        self.engine = create_engine(DB_URL)
        print(f"--- Initializing Toto ({MODEL_ID}) on {DEVICE} ---")
        
        self.model_wrapper = Toto.from_pretrained(MODEL_ID)
        self.model_wrapper.eval()
        self.model_wrapper.to(DEVICE)
        
        self.model = self.model_wrapper.model
        
        if hasattr(self.model, 'embed_dim'):
            self.d_model = self.model.embed_dim
        else:
            self.d_model = 768
        
        print(f"Model loaded successfully.")
        print(f"Embedding Dimension: {self.d_model}")
        print(f"Target PCA Dimension: {PCA_COMPONENTS}")
        
        self._hook_output = None
        self._register_hook()

    def _register_hook(self):
        def hook_fn(module, input, output):
            self._hook_output = output
            
        target_module = None
        if hasattr(self.model, 'transformer'):
            target_module = self.model.transformer
        elif hasattr(self.model, 'encoder'):
            target_module = self.model.encoder
            
        if target_module is None:
            for name, module in self.model.named_children():
                if 'transformer' in name.lower() or 'encoder' in name.lower():
                    target_module = module
                    break
        
        if target_module:
            target_module.register_forward_hook(hook_fn)
            print(f"Hook registered on: {target_module.__class__.__name__}")
        else:
            print("Warning: Could not find transformer module. Registering hook on backbone root.")
            self.model.register_forward_hook(hook_fn)

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

    def preprocess_series(self, y_series, ds_series):
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
        
        timestamps = ds_series.astype(np.int64) // 10**9
        if len(timestamps) > 1:
            diffs = np.diff(timestamps)
            interval = np.median(diffs)
            if interval == 0: interval = 86400 
        else:
            interval = 86400 
            
        return y, timestamps, interval

    def generate_features(self, df):
        print("--- Generating Toto Embeddings (Chunked + PCA) ---")
        
        grouped = df.groupby(['loto', 'unique_id', 'ts_type'])
        
        raw_embeddings_map = [] 
        batch_inputs = []
        batch_metas = []
        
        for name, group in tqdm(grouped, desc="Inference"):
            y_values = group['y'].values
            ds_values = group['ds'].values
            total_len = len(y_values)
            
            if total_len == 0: continue
            
            full_y, full_ts, interval = self.preprocess_series(y_values, ds_values)
            
            for start_idx in range(0, total_len, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, total_len)
                chunk_len = end_idx - start_idx
                
                chunk_y = full_y[start_idx:end_idx]
                chunk_ts = full_ts[start_idx:end_idx]
                
                final_pad_len = CHUNK_SIZE - chunk_len
                padding_mask = np.ones(CHUNK_SIZE, dtype=bool)
                
                if final_pad_len > 0:
                    chunk_y = np.pad(chunk_y, (0, final_pad_len), 'constant')
                    chunk_ts = np.pad(chunk_ts, (0, final_pad_len), 'constant')
                    padding_mask[chunk_len:] = False
                
                batch_inputs.append({
                    'values': chunk_y,
                    'timestamps': chunk_ts,
                    'interval': interval,
                    'mask': padding_mask
                })
                
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
        
        print("Step 2: PCA Training & Transformation...")
        if not raw_embeddings_map:
            return pd.DataFrame()

        all_embeddings = np.array([item['embedding'] for item in raw_embeddings_map])
        
        pca = PCA(n_components=PCA_COMPONENTS)
        pca_embeddings = pca.fit_transform(all_embeddings)
        print(f"Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
        
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
                        'hist_toto_embed': reduced_emb.tolist()
                    })
                    
        result_df = pd.DataFrame(final_data)
        print(f"Generated features for {len(result_df)} points.")
        return result_df

    def _process_batch_inference(self, inputs_list, metas, output_list):
        B = len(inputs_list)
        T = CHUNK_SIZE
        
        values = torch.zeros((B, 1, T), dtype=torch.float32, device=DEVICE)
        padding_mask = torch.zeros((B, 1, T), dtype=torch.bool, device=DEVICE)
        id_mask = torch.zeros((B, 1, T), dtype=torch.float32, device=DEVICE)
        
        for i, item in enumerate(inputs_list):
            values[i, 0, :] = torch.from_numpy(item['values'])
            padding_mask[i, 0, :] = torch.from_numpy(item['mask'])
        
        with torch.no_grad():
            self._hook_output = None 
            
            # --- 修正: 正しい引数名で渡す ---
            _ = self.model(
                inputs=values, 
                input_padding_mask=padding_mask,
                id_mask=id_mask
            )
            
        hidden_states = self._hook_output
        
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
            
        # (Batch, Variables, Time, Hidden) -> (B, 1, T, H)
        if len(hidden_states.shape) == 4:
            hidden_states = hidden_states.squeeze(1) 
            
        embeddings_batch = hidden_states.float().cpu().numpy()
        
        # Pooling
        embeddings_pooled = embeddings_batch.mean(axis=1) 
        
        for i, emb in enumerate(embeddings_pooled):
            output_list.append({
                'meta': metas[i],
                'embedding': emb
            })

    def save_to_db(self, df, table_name="loto_toto_features"):
        if df is None or df.empty:
            print("No data to save.")
            return

        print(f"--- Saving to {table_name} using FAST COPY (Flattened Columns) ---")
        print("Flattening embeddings into columns...")
        
        embed_df = pd.DataFrame(
            df['hist_toto_embed'].tolist(), 
            index=df.index
        ).add_prefix('hist_toto_embed_')
        
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
    generator = LotoTotoFeatureGenerator()
    generator.run()
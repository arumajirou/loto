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

# momentfmのインポート
try:
    from momentfm import MOMENTPipeline
except ImportError:
    print("Error: 'momentfm' not found. Please run 'pip install momentfm'")
    sys.exit(1)

# ==========================================
# 設定
# ==========================================
MODEL_ID = "AutonLab/MOMENT-1-large"
BATCH_SIZE = 8
CHUNK_SIZE = 512
PCA_COMPONENTS = 64  # 次元圧縮後のサイズ (1024 -> 64)

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

class LotoMomentFeatureGenerator:
    def __init__(self):
        self.engine = create_engine(DB_URL)
        print(f"--- Initializing MOMENT Pipeline ({MODEL_ID}) on {DEVICE} ---")
        
        self.model = MOMENTPipeline.from_pretrained(
            MODEL_ID,
            model_kwargs={'task_name': 'embedding'},
        )
        self.model.init()
        self.model.to(DEVICE)
        
        # コンフィグからの情報取得（フォールバック付き）
        if hasattr(self.model.config, 'd_model'):
            self.d_model = self.model.config.d_model
        elif hasattr(self.model.config, 'hidden_size'):
            self.d_model = self.model.config.hidden_size
        else:
            self.d_model = 1024
            
        print(f"Model loaded successfully.")
        print(f"Original Embedding Dimension: {self.d_model}")
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
        
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return y_tensor

    def generate_features(self, df):
        print("--- Generating MOMENT Embeddings (Chunked + PCA) ---")
        
        grouped = df.groupby(['loto', 'unique_id', 'ts_type'])
        
        # PCA学習用の一時保存リスト
        # メモリ節約のため、全データの埋め込みを保持せず、
        # 一部の代表データ(あるいは全データ)を使ってPCAを学習後、変換する
        # ここでは全データを一度メモリに載せるアプローチをとるが、
        # メモリ不足ならバッチごとにランダムサンプリングしてPCA学習する方式に切り替える
        
        raw_embeddings_map = [] # (index, embedding) のタプル
        batch_inputs = []
        batch_indices = [] # 後で再構成するためのインデックス情報
        
        processed_count = 0
        
        # 1. 全データの埋め込みベクトル生成（PCA前）
        print("Step 1: Inference...")
        for name, group in tqdm(grouped, desc="Inference"):
            y_values = group['y'].values
            ds_values = group['ds'].values
            total_len = len(y_values)
            
            if total_len == 0: continue
            
            for start_idx in range(0, total_len, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, total_len)
                chunk_len = end_idx - start_idx
                
                chunk_y = y_values[start_idx:end_idx]
                if chunk_len < CHUNK_SIZE:
                    chunk_y = np.pad(chunk_y, (0, CHUNK_SIZE - chunk_len), 'constant')
                
                input_tensor = self.preprocess_series(chunk_y)
                
                batch_inputs.append(input_tensor)
                
                # このバッチが対応するメタデータを保存
                batch_indices.append({
                    'loto': name[0],
                    'unique_id': name[1],
                    'ts_type': name[2],
                    'ds_chunk': ds_values[start_idx:end_idx],
                    'valid_len': chunk_len
                })
                
                if len(batch_inputs) >= BATCH_SIZE:
                    self._process_batch_inference(batch_inputs, batch_indices, raw_embeddings_map)
                    batch_inputs = []
                    batch_indices = []

        if batch_inputs:
             self._process_batch_inference(batch_inputs, batch_indices, raw_embeddings_map)
        
        # raw_embeddings_map は [{'meta': ..., 'embedding': np.array}, ...] のリストになっているはず
        
        # 2. PCA学習と変換
        print("Step 2: PCA Training & Transformation...")
        if not raw_embeddings_map:
            return pd.DataFrame()

        # 埋め込み行列の作成 (N_chunks, 1024)
        all_embeddings = np.array([item['embedding'] for item in raw_embeddings_map])
        
        pca = PCA(n_components=PCA_COMPONENTS)
        pca_embeddings = pca.fit_transform(all_embeddings)
        
        print(f"Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        # 3. 結果の整形
        print("Step 3: Formatting...")
        final_data = []
        
        for i, item in enumerate(raw_embeddings_map):
            meta = item['meta']
            reduced_emb = pca_embeddings[i] # (64,)
            
            ds_chunk = meta['ds_chunk']
            valid_len = meta['valid_len']
            
            # 各時点に同じ圧縮特徴量を付与
            for t in range(valid_len):
                if t < len(ds_chunk):
                    final_data.append({
                        'loto': meta['loto'],
                        'unique_id': meta['unique_id'],
                        'ts_type': meta['ts_type'],
                        'ds': ds_chunk[t],
                        'hist_moment_embed': reduced_emb.tolist() # リスト化
                    })
                    
        result_df = pd.DataFrame(final_data)
        print(f"Generated features for {len(result_df)} points.")
        return result_df

    def _process_batch_inference(self, inputs_list, metas, output_list):
        batch_tensor = torch.stack(inputs_list).to(DEVICE)
        batch_size = batch_tensor.shape[0]
        seq_len = batch_tensor.shape[2]
        
        input_mask = torch.ones((batch_size, seq_len), device=DEVICE)
        for i, meta in enumerate(metas):
            valid_len = meta['valid_len']
            if valid_len < seq_len:
                input_mask[i, valid_len:] = 0

        with torch.no_grad():
            output = self.model(x_enc=batch_tensor, input_mask=input_mask)
        
        if hasattr(output, 'embeddings'):
            embeddings = output.embeddings.float().cpu().numpy()
        else:
            embeddings = getattr(output, 'last_hidden_state', None)
            if embeddings is None:
                embeddings = output[0]
            embeddings = embeddings.float().cpu().numpy()

        if len(embeddings.shape) == 3:
            if embeddings.shape[1] == CHUNK_SIZE: 
                 embeddings = embeddings.mean(axis=1)
            else:
                 embeddings = embeddings.squeeze(1)
        
        # 結果をリストに追加
        for i, emb in enumerate(embeddings):
            output_list.append({
                'meta': metas[i],
                'embedding': emb # (1024,)
            })

    def save_to_db(self, df, table_name="loto_moment_features"):
        if df is None or df.empty:
            print("No data to save.")
            return

        print(f"--- Saving to {table_name} using FAST COPY (Flattened Columns) ---")
        print("Flattening embeddings into columns...")
        
        # リストを展開してカラムにする (hist_moment_embed_0 ... hist_moment_embed_63)
        embed_df = pd.DataFrame(
            df['hist_moment_embed'].tolist(), 
            index=df.index
        ).add_prefix('hist_moment_embed_')
        
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
    generator = LotoMomentFeatureGenerator()
    generator.run()
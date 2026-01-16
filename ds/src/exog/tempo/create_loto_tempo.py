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
from omegaconf import OmegaConf

# ==========================================
# 1. ローカルのTEMPOモジュールを読み込む設定
# ==========================================
# git cloneしたTEMPOディレクトリへの絶対パス
TEMPO_REPO_PATH = "/mnt/e/env/ts/tslib/model/TEMPO"

if TEMPO_REPO_PATH not in sys.path:
    sys.path.append(TEMPO_REPO_PATH)

try:
    from tempo.models.TEMPO import TEMPO
    print("Successfully imported TEMPO from local path.")
except ImportError as e:
    print(f"Error importing TEMPO: {e}")
    print(f"Please check if '{TEMPO_REPO_PATH}' exists and contains 'tempo' directory.")
    sys.exit(1)

# ==========================================
# 設定
# ==========================================
REPO_ID = "Melady/TEMPO"
FILENAME = "TEMPO-80M_v1.pth"
# キャッシュディレクトリ（NoneならデフォルトのHFキャッシュを使用）
CACHE_DIR = None 

BATCH_SIZE = 8
# TEMPOの学習設定に合わせて336などを設定（後でモデルから自動取得を試みる）
DEFAULT_CHUNK_SIZE = 336 
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

class LotoTempoFeatureGenerator:
    def __init__(self):
        self.engine = create_engine(DB_URL)
        print(f"--- Initializing TEMPO ({REPO_ID}) on {DEVICE} ---")
        
        try:
            # pretrainedモデルのロード
            self.model = TEMPO.load_pretrained_model(
                device=torch.device(DEVICE),
                repo_id=REPO_ID,
                filename=FILENAME,
                cache_dir=CACHE_DIR
            )
        except Exception as e:
            print(f"Failed to load via load_pretrained_model: {e}")
            sys.exit(1)
            
        self.model.eval()
        self.model.to(DEVICE)
        
        # コンテキスト長の取得
        if hasattr(self.model, 'seq_len'):
            self.chunk_size = self.model.seq_len
        else:
            self.chunk_size = DEFAULT_CHUNK_SIZE
            
        # 次元数の確認 (GPT2ベースなので通常768)
        self.d_model = 768 
        
        print(f"Model loaded successfully.")
        print(f"Sequence Length (CHUNK_SIZE): {self.chunk_size}")
        print(f"Target PCA Dimension: {PCA_COMPONENTS}")
        
        # Hook用の変数を準備
        self._hook_output = None
        self._register_hook()

    def _register_hook(self):
        """GPT2の出力を取得するためのフック"""
        def hook_fn(module, input, output):
            # outputは通常 (last_hidden_state, past_key_values, ...)
            if hasattr(output, 'last_hidden_state'):
                self._hook_output = output.last_hidden_state
            elif isinstance(output, tuple):
                self._hook_output = output[0]
            else:
                self._hook_output = output
        
        # TEMPOは self.gpt2_trend をメインのバックボーンとして使用
        if hasattr(self.model, 'gpt2_trend'):
            self.model.gpt2_trend.register_forward_hook(hook_fn)
            print("Hook registered on: self.model.gpt2_trend")
        else:
            print("Error: Could not find gpt2_trend module in TEMPO model.")

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
        # NaN処理
        y = np.array(y_series, dtype=np.float32)
        nans = np.isnan(y)
        if np.any(nans):
            x = np.arange(len(y))
            if np.all(nans):
                y[:] = 0.0
            else:
                y[nans] = np.interp(x[nans], x[~nans], y[~nans])
        
        # 標準化 (TEMPOは内部でRevInを行うが、入力も整えておく)
        mean = np.mean(y)
        std = np.std(y) + 1e-5
        y = (y - mean) / std
        return y

    def generate_features(self, df):
        print("--- Generating TEMPO Embeddings (Chunked + PCA) ---")
        
        grouped = df.groupby(['loto', 'unique_id', 'ts_type'])
        
        raw_embeddings_map = [] 
        batch_inputs = []
        batch_metas = []
        
        for name, group in tqdm(grouped, desc="Inference"):
            y_values = group['y'].values
            ds_values = group['ds'].values
            total_len = len(y_values)
            
            if total_len == 0: continue
            
            # CHUNK_SIZE ごとに処理
            for start_idx in range(0, total_len, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_len)
                chunk_len = end_idx - start_idx
                
                chunk_y = y_values[start_idx:end_idx]
                
                processed_y = self.preprocess_series(chunk_y)
                
                # TEMPOは固定長入力を期待する場合が多いため、不足分をパディング
                final_pad_len = self.chunk_size - len(processed_y)
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
            
            # 各時点に埋め込みベクトルを割り当て
            for t in range(valid_len):
                if t < len(ds_chunk):
                    final_data.append({
                        'loto': loto,
                        'unique_id': unique_id,
                        'ts_type': ts_type,
                        'ds': ds_chunk[t],
                        'hist_tempo_embed': reduced_emb.tolist()
                    })
                    
        result_df = pd.DataFrame(final_data)
        print(f"Generated features for {len(result_df)} points.")
        return result_df

    def _process_batch_inference(self, inputs_list, metas, output_list):
        B = len(inputs_list)
        T = self.chunk_size
        
        # TEMPO入力形状: [Batch, Length, Metrics(Channels)]
        # Univariateなので Metrics=1
        batch_tensor = torch.tensor(np.array(inputs_list), dtype=torch.float32).unsqueeze(2).to(DEVICE) # (B, T, 1)
        
        with torch.no_grad():
            self._hook_output = None 
            # forward実行 (test=Trueで予測モード)
            # 戻り値は無視し、Hookで取得したgpt2_trendの出力を使用する
            _ = self.model(batch_tensor, test=True)
            
        # Hook出力: (Batch, Seq_Len_Tokenized, Hidden)
        # TEMPOはデータをパッチ化してトークンにするため、Seq_Lenは T とは異なる
        hidden_states = self._hook_output
        embeddings_batch = hidden_states.float().cpu().numpy()
        
        # 平均プーリングして、このチャンク全体の特徴量とする
        embeddings_pooled = embeddings_batch.mean(axis=1) # (B, Hidden)
        
        for i, emb in enumerate(embeddings_pooled):
            output_list.append({
                'meta': metas[i],
                'embedding': emb
            })

    def save_to_db(self, df, table_name="loto_tempo_features"):
        if df is None or df.empty:
            print("No data to save.")
            return

        print(f"--- Saving to {table_name} using FAST COPY (Flattened Columns) ---")
        print("Flattening embeddings into columns...")
        
        embed_df = pd.DataFrame(
            df['hist_tempo_embed'].tolist(), 
            index=df.index
        ).add_prefix('hist_tempo_embed_')
        
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
    generator = LotoTempoFeatureGenerator()
    generator.run()
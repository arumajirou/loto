import os
import sys
import io
import csv
import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from tqdm import tqdm
from unittest.mock import MagicMock

# ==========================================
# 1. Import Error 回避のための Mock 処理
# ==========================================
sys.modules["lightning"] = MagicMock()
sys.modules["lightning.pytorch"] = MagicMock()
sys.modules["lightning.fabric"] = MagicMock()

try:
    from uni2ts.model.moirai2.module import Moirai2Module
    from uni2ts.common.torch_util import packed_causal_attention_mask
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# ==========================================
# 設定
# ==========================================
MODEL_NAME = "Salesforce/moirai-2.0-R-small"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DB接続設定
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASS = os.getenv("PGPASSWORD", "z")
PG_HOST = os.getenv("PGHOST", "127.0.0.1")
PG_PORT = os.getenv("PGPORT", "5432")
PG_DB   = os.getenv("PGDATABASE", "loto_db")
DB_URL  = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"

class LotoMoirai2FeatureGenerator:
    def __init__(self):
        self.engine = create_engine(DB_URL)
        print(f"--- Initializing Moirai 2.0 ({MODEL_NAME}) on {DEVICE} ---")
        
        # モデルロード
        self.model = Moirai2Module.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.model.to(DEVICE)
        
        # モデルからパッチサイズと次元数を動的に取得
        self.patch_size = self.model.patch_size
        self.d_model = self.model.d_model
        
        print(f"Model loaded successfully.")
        print(f"Patch Size: {self.patch_size}")
        print(f"Embedding Dimension: {self.d_model}")

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

    def preprocess_batch(self, y_batch_list):
        max_len = max([len(y) for y in y_batch_list])
        pad_len = (self.patch_size - (max_len % self.patch_size)) % self.patch_size
        target_seq_len = (max_len + pad_len) // self.patch_size
        
        batch_target = torch.zeros((len(y_batch_list), target_seq_len, self.patch_size), dtype=torch.float32)
        batch_observed_mask = torch.zeros((len(y_batch_list), target_seq_len, self.patch_size), dtype=torch.bool)
        batch_sample_id = torch.zeros((len(y_batch_list), target_seq_len), dtype=torch.int64)
        batch_time_id = torch.zeros((len(y_batch_list), target_seq_len), dtype=torch.int64)
        batch_variate_id = torch.zeros((len(y_batch_list), target_seq_len), dtype=torch.int64)
        batch_prediction_mask = torch.zeros((len(y_batch_list), target_seq_len), dtype=torch.bool)
        
        valid_lengths = []

        for i, y in enumerate(y_batch_list):
            y_np = np.array(y, dtype=np.float32)
            original_len = len(y_np)
            valid_lengths.append(original_len)
            
            nan_mask = np.isnan(y_np)
            y_np[nan_mask] = 0.0
            
            pad_needed = target_seq_len * self.patch_size - original_len
            y_padded = np.pad(y_np, (0, pad_needed), 'constant', constant_values=0)
            mask_padded = np.pad(~nan_mask, (0, pad_needed), 'constant', constant_values=False)
            
            batch_target[i] = torch.from_numpy(y_padded).view(-1, self.patch_size)
            batch_observed_mask[i] = torch.from_numpy(mask_padded).view(-1, self.patch_size)
            batch_time_id[i] = torch.arange(target_seq_len)

        return {
            'target': batch_target.to(DEVICE),
            'observed_mask': batch_observed_mask.to(DEVICE),
            'sample_id': batch_sample_id.to(DEVICE),
            'time_id': batch_time_id.to(DEVICE),
            'variate_id': batch_variate_id.to(DEVICE),
            'prediction_mask': batch_prediction_mask.to(DEVICE),
        }, valid_lengths

    def get_embeddings(self, batch_inputs):
        with torch.no_grad():
            loc, scale = self.model.scaler(
                batch_inputs['target'],
                batch_inputs['observed_mask'] * ~batch_inputs['prediction_mask'].unsqueeze(-1),
                batch_inputs['sample_id'],
                batch_inputs['variate_id'],
            )
            scaled_target = (batch_inputs['target'] - loc) / scale
            
            input_tokens = torch.cat(
                [scaled_target, batch_inputs['observed_mask'].to(torch.float32)], dim=-1
            )
            reprs = self.model.in_proj(input_tokens)
            
            attn_mask = packed_causal_attention_mask(
                batch_inputs['sample_id'], 
                batch_inputs['time_id']
            ).to(DEVICE)
            
            embeddings = self.model.encoder(
                reprs,
                attn_mask,
                time_id=batch_inputs['time_id'],
                var_id=batch_inputs['variate_id'],
            )
            return embeddings.cpu().numpy()

    def generate_features(self, df):
        print("--- Generating Moirai 2.0 Embeddings ---")
        grouped = df.groupby(['loto', 'unique_id', 'ts_type'])
        
        final_data = []
        batch_y = []
        batch_meta = []
        
        for name, group in tqdm(grouped, desc="Processing"):
            y_values = group['y'].values
            ds_values = group['ds'].values
            if len(y_values) == 0: continue
            
            batch_y.append(y_values)
            batch_meta.append({'key': name, 'ds': ds_values})
            
            if len(batch_y) >= BATCH_SIZE:
                self._process_batch(batch_y, batch_meta, final_data)
                batch_y = []
                batch_meta = []
        
        if batch_y:
            self._process_batch(batch_y, batch_meta, final_data)
            
        result_df = pd.DataFrame(final_data)
        print(f"Generated features for {len(result_df)} points.")
        return result_df

    def _process_batch(self, batch_y, batch_meta, final_data):
        inputs, valid_lengths = self.preprocess_batch(batch_y)
        embeddings = self.get_embeddings(inputs) 
        
        for i, meta in enumerate(batch_meta):
            loto, unique_id, ts_type = meta['key']
            ds_values = meta['ds']
            original_len = valid_lengths[i]
            seq_emb = embeddings[i]
            
            for p_idx, patch_emb in enumerate(seq_emb):
                features = patch_emb
                start_idx = p_idx * self.patch_size
                end_idx = start_idx + self.patch_size
                
                for t_idx in range(start_idx, end_idx):
                    if t_idx < original_len:
                        final_data.append({
                            'loto': loto,
                            'unique_id': unique_id,
                            'ts_type': ts_type,
                            'ds': ds_values[t_idx],
                            'moirai_embed': features
                        })

    # 修正: テーブル名をデフォルトで 'loto_moirai_features' に変更
    def save_to_db(self, df, table_name="loto_moirai_features"):
        if df is None or df.empty:
            print("No data to save.")
            return

        print(f"--- Saving to {table_name} using FAST COPY (Flattened Columns) ---")
        
        # 1. Embeddingを展開してDataFrameを作成
        print("Flattening embeddings into columns...")
        
        # 修正: 接頭辞を 'hist_moirai_embed_' に変更
        embed_df = pd.DataFrame(
            df['moirai_embed'].tolist(), 
            index=df.index
        ).add_prefix('hist_moirai_embed_')
        
        # 元のキーカラムと結合
        df_flat = pd.concat([df[['loto', 'unique_id', 'ts_type', 'ds']], embed_df], axis=1)
        
        # Embeddingカラムのリスト
        embed_cols = embed_df.columns.tolist()
        num_embed_cols = len(embed_cols)
        print(f"Expanded to {num_embed_cols} feature columns (Prefix: hist_moirai_embed_).")

        # 2. 生のコネクション取得
        conn = self.engine.raw_connection()
        cur = conn.cursor()
        
        try:
            # 3. テーブル再作成
            print("Recreating table...")
            
            # 埋め込みカラムの定義を作成
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

            # 4. メモリ上のCSVバッファに書き出し
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

            # 5. COPYコマンド実行
            print("Executing COPY command...")
            cur.copy_expert(
                f'COPY "{table_name}" FROM STDIN WITH (FORMAT CSV, DELIMITER E\'\\t\')', 
                output
            )
            conn.commit()
            
            # 6. インデックス作成
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
            print("Done.")

if __name__ == "__main__":
    generator = LotoMoirai2FeatureGenerator()
    generator.run()
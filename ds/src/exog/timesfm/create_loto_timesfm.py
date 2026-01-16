import os
import sys
import io
import csv
import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from sklearn.decomposition import PCA
from tqdm import tqdm

# ==========================================
# パスとインポートの設定 (TimesFM 2.5用)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# ds/src/timesfm -> model/timesfm/src へのパス解決
lib_path = os.path.abspath(os.path.join(current_dir, "../../../model/timesfm/src"))

if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# 競合回避のため、既存のtimesfmモジュールを削除
modules_to_remove = [m for m in sys.modules if m.startswith("timesfm")]
for m in modules_to_remove:
    del sys.modules[m]

try:
    import timesfm
    # クラスのインポート試行
    try:
        from timesfm import TimesFM_2p5_200M_torch
    except ImportError:
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
except ImportError as e:
    raise ImportError(f"Could not import TimesFM 2.5 from {lib_path}. Error: {e}")

# ==========================================
# 特徴量生成クラス
# ==========================================
class LotoTimesFMFeatureGenerator:
    def __init__(self, model_name="google/timesfm-2.5-200m-pytorch", device=None):
        # DB接続設定
        self.pg_user = os.getenv("PGUSER", "postgres")
        self.pg_pass = os.getenv("PGPASSWORD", "z")
        self.pg_host = os.getenv("PGHOST", "127.0.0.1")
        self.pg_port = os.getenv("PGPORT", "5432")
        self.pg_db = os.getenv("PGDATABASE", "loto_db")
        self.db_url = f"postgresql://{self.pg_user}:{self.pg_pass}@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        self.engine = create_engine(self.db_url)
        
        # モデル設定
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.patch_len = 32  # TimesFM 2.5 のパッチサイズ
        self.model = None
        
        print(f"Initialized with device: {self.device}")

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

    def setup_model(self):
        print(f"--- Loading TimesFM Model: {self.model_name} ---")
        # モデルロード（位置引数を使用）
        self.tfm = TimesFM_2p5_200M_torch.from_pretrained(self.model_name)
        self.model = self.tfm.model
        self.model.eval()
        self.model.to(self.device)
        print("Model loaded successfully.")

    def preprocess_series(self, y_series):
        # 線形補間
        y = np.array(y_series, dtype=np.float32)
        nans = np.isnan(y)
        if np.any(nans):
            x = np.arange(len(y))
            if np.all(nans):
                y[:] = 0.0
            else:
                y[nans] = np.interp(x[nans], x[~nans], y[~nans])
        
        # パッチサイズの倍数になるようにゼロパディング
        L = len(y)
        P = self.patch_len
        pad_len = (P - (L % P)) % P
        if pad_len > 0:
            y_padded = np.pad(y, (pad_len, 0), 'constant', constant_values=0)
            mask_padded = np.pad(np.ones(L, dtype=bool), (pad_len, 0), 'constant', constant_values=False)
        else:
            y_padded = y
            mask_padded = np.ones(L, dtype=bool)
            
        return y_padded, mask_padded, pad_len

    def generate_embeddings(self, df, pca_components=64):
        if self.model is None:
            self.setup_model()

        print(f"--- Generating Embeddings (PCA={pca_components}) ---")
        
        grouped = df.groupby(['loto', 'unique_id', 'ts_type'])
        
        all_embeddings = [] 
        batch_inputs = []
        batch_masks = []
        batch_meta = []
        
        BATCH_SIZE = 16
        
        # 1. データ準備
        print("Preprocessing...")
        for name, group in tqdm(grouped, desc="Preparing data"):
            y_values = group['y'].values
            ds_values = group['ds'].values
            
            if len(y_values) == 0: continue

            y_padded, mask_padded, pad_len = self.preprocess_series(y_values)
            
            batch_inputs.append(y_padded)
            batch_masks.append(mask_padded)
            batch_meta.append({
                'key': name,
                'ds': ds_values,
                'pad_len': pad_len,
                'num_patches': len(y_padded) // self.patch_len
            })

        # 2. 推論実行
        print("Running inference...")
        raw_embeddings_map = {}
        
        for i in tqdm(range(0, len(batch_inputs), BATCH_SIZE), desc="Inference"):
            batch_slice_inputs = batch_inputs[i:i+BATCH_SIZE]
            batch_slice_masks = batch_masks[i:i+BATCH_SIZE]
            batch_slice_meta = batch_meta[i:i+BATCH_SIZE]
            
            batch_max_len = max(len(x) for x in batch_slice_inputs)
            
            tensor_inputs = torch.zeros((len(batch_slice_inputs), batch_max_len), dtype=torch.float32, device=self.device)
            tensor_masks = torch.zeros((len(batch_slice_inputs), batch_max_len), dtype=torch.bool, device=self.device)
            
            for j, (inp, msk) in enumerate(zip(batch_slice_inputs, batch_slice_masks)):
                l = len(inp)
                tensor_inputs[j, -l:] = torch.from_numpy(inp).to(self.device)
                tensor_masks[j, -l:] = torch.from_numpy(msk).to(self.device)
            
            # TimesFM入力形状へ変形
            tensor_inputs_patched = tensor_inputs.view(len(batch_slice_inputs), -1, self.patch_len)
            tensor_masks_patched = tensor_masks.view(len(batch_slice_inputs), -1, self.patch_len)

            with torch.no_grad():
                outputs, _ = self.model(tensor_inputs_patched, tensor_masks_patched)
                output_embeddings = outputs[1]
                
            output_embeddings = output_embeddings.cpu().numpy()
            
            for j, meta in enumerate(batch_slice_meta):
                valid_patches = meta['num_patches']
                emb = output_embeddings[j, -valid_patches:, :]
                
                raw_embeddings_map[meta['key']] = emb
                
                # PCA学習用にサンプリング（全データだとメモリ厳しい場合）
                if len(all_embeddings) < 20000:
                    for k in range(emb.shape[0]):
                        all_embeddings.append(emb[k])

        # 3. PCA学習
        print("Training PCA...")
        pca = PCA(n_components=pca_components)
        if len(all_embeddings) > 0:
            pca.fit(np.array(all_embeddings))
        else:
            print("No embeddings generated.")
            return None
            
        # 4. 結果の整形（全レコードへの拡張）
        print("Formatting and applying PCA to ALL records...")
        final_data = []
        
        for name, emb in tqdm(raw_embeddings_map.items(), desc="Applying PCA"):
            # 次元圧縮 (N, 1280) -> (N, 64)
            emb_pca = pca.transform(emb)
            
            loto, unique_id, ts_type = name
            meta = [m for m in batch_meta if m['key'] == name][0]
            ds_values = meta['ds']
            pad_len = meta['pad_len']
            
            for i in range(len(emb_pca)):
                # このパッチ (i) がカバーする範囲
                start_idx_padded = i * self.patch_len
                end_idx_padded = (i + 1) * self.patch_len
                
                # 特徴量をリストとして保持（後でDataFrame化）
                features = emb_pca[i]
                
                # パッチ内の各時点について
                for padded_idx in range(start_idx_padded, end_idx_padded):
                    original_idx = padded_idx - pad_len
                    
                    if 0 <= original_idx < len(ds_values):
                        target_ds = ds_values[original_idx]
                        
                        # 辞書にデータを追加（featuresはnumpy arrayのまま）
                        final_data.append({
                            'loto': loto,
                            'unique_id': unique_id,
                            'ts_type': ts_type,
                            'ds': target_ds,
                            'hist_timesfm_embed': features
                        })

        result_df = pd.DataFrame(final_data)
        print(f"Generated features for {len(result_df)} points.")
        return result_df

    def save_to_db(self, df, table_name="loto_timesfm_features"):
        if df is None or df.empty:
            print("No data to save.")
            return

        print(f"--- Saving to {table_name} using FAST COPY (Flattened Columns) ---")
        
        # 1. Embeddingを展開してDataFrameを作成
        print("Flattening embeddings into columns...")
        
        # hist_timesfm_embedカラム(numpy array)をリスト化して、DataFrameとして展開
        # 列名は hist_timesfm_embed_0, hist_timesfm_embed_1, ...
        embed_df = pd.DataFrame(
            df['hist_timesfm_embed'].tolist(), 
            index=df.index
        ).add_prefix('hist_timesfm_embed_')
        
        # 元のキーカラムと結合
        df_flat = pd.concat([df[['loto', 'unique_id', 'ts_type', 'ds']], embed_df], axis=1)
        
        # Embeddingカラムのリスト
        embed_cols = embed_df.columns.tolist()
        num_embed_cols = len(embed_cols)
        print(f"Expanded to {num_embed_cols} feature columns (Prefix: hist_timesfm_embed_).")

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
            # pandas to_csv は高速
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
            features_df = self.generate_embeddings(df)
            self.save_to_db(features_df)
            print("All processes completed.")

if __name__ == "__main__":
    generator = LotoTimesFMFeatureGenerator()
    generator.run()
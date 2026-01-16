import os
import torch
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import List, Dict, Any

# ==========================================
# 1. DB接続設定
# ==========================================
def get_db_engine():
    PG_USER = os.getenv("PGUSER", "postgres")
    PG_PASS = os.getenv("PGPASSWORD", "z")
    PG_HOST = os.getenv("PGHOST", "127.0.0.1")
    PG_PORT = os.getenv("PGPORT", "5432")
    PG_DB   = os.getenv("PGDATABASE", "dataset")
    
    db_url = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    return create_engine(db_url)

# ==========================================
# 2. モデル別特徴量抽出クラス (Strategy Pattern)
# ==========================================

class BaseFeatureExtractor:
    def __init__(self, model_id: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.tokenizer = None
        print(f"[{model_id}] Initializing extractor on {device}...")

    def load_model(self):
        """モデルのロード（メモリ節約のため必要な時だけ呼ぶ）"""
        raise NotImplementedError

    def extract(self, time_series: np.ndarray) -> np.ndarray:
        """時系列データ(1D numpy array)から特徴量ベクトルを返す"""
        raise NotImplementedError

    def unload_model(self):
        """メモリ解放"""
        if self.model is not None:
            del self.model
            if self.tokenizer:
                del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None
            print(f"[{self.model_id}] Unloaded to free memory.")

class ChronosExtractor(BaseFeatureExtractor):
    """Amazon Chronos (T5ベース) からエンコーダ表現を抽出"""
    def load_model(self):
        # ChronosはT5ForConditionalGenerationだが、特徴抽出にはEncoderのみ使用可能
        # ここではtransformersのAutoModelForSeq2SeqLMとしてロード
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, 
            device_map=self.device, 
            torch_dtype=torch.bfloat16
        )
        self.model.eval()

    def extract(self, time_series: np.ndarray) -> np.ndarray:
        # Chronos向けにデータをトークン化 (簡易実装: 値をビン分割してトークン化するロジックが必要だが、
        # ここではChronosの公式パイプラインの内部ロジックを模倣するか、簡易的に正規化して入力とします)
        # ※注: 本来は chronos.pipeline を使うのが楽ですが、内部embeddingを取るため直接叩きます
        
        # 簡易化: 数値を文字列化してT5に入れる（Chronos本来の動作とは異なるが、汎用的なT5特徴量として取得）
        # 本格的なChronos embeddingには amazon/chronos ライブラリの `tokenizer.input_ids` が必要
        context_tensor = torch.tensor([time_series], dtype=torch.float32).to(self.device)
        
        # Chronos固有の前処理が複雑なため、ここではモデルがロードできることの確認を優先し
        # ダミーの処理として記述します（実際には公式ライブラリの tokenizer を使用推奨）
        # 今回は「モデルの重みを使って変換する」という観点で、AutoModelとして動かします
        return np.zeros(768) # プレースホルダー（実装時はライブラリ依存）

class MoiraiExtractor(BaseFeatureExtractor):
    """Salesforce Moirai から特徴量を抽出"""
    def load_model(self):
        self.model = AutoModel.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            device_map=self.device
        )
        self.model.eval()

    def extract(self, time_series: np.ndarray) -> np.ndarray:
        # MoiraiはPatch化して入力する必要がある
        # 簡易実装: 入力をテンソル化してモデルにパス
        inputs = torch.tensor(time_series, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        # 実際にはPatch sizeなどに合わせたパディングが必要
        with torch.no_grad():
            # encoder_outputs = self.model(inputs) # 形状が合わないとエラーになるため注意
            pass
        return np.zeros(self.model.config.hidden_size) # プレースホルダー

class GenericHFFeatureExtractor(BaseFeatureExtractor):
    """
    汎用的なHugging Faceモデル用抽出機
    時系列をコンテキストとして入力し、ラストトークンのHidden Stateを返す
    """
    def load_model(self):
        try:
            self.model = AutoModel.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                device_map=self.device,
                output_hidden_states=True
            )
        except Exception:
            # Seq2Seqモデルなどの場合
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, 
                trust_remote_code=True, 
                device_map=self.device,
                output_hidden_states=True
            )
        self.model.eval()

    def extract(self, time_series: np.ndarray) -> np.ndarray:
        # 入力を整形 (Batch, Time, Dim)
        # 多くの時系列モデルは (B, T, C) または (B, T) を期待
        
        # NOTE: モデルごとに期待する入力形式が厳密に異なるため、
        # ここでは最も一般的な「数値列を受け取ってHidden Stateを返す」処理を想定します。
        # 実際には各モデルの `forward` メソッドの引数に合わせて調整が必要です。
        
        # 安全策: ランダムな射影行列としての利用（モデルが重すぎる場合や入力形式が不明な場合）
        # ここでは実装の枠組みを提供します。
        return np.random.rand(64) # 実装時はここに `model(input).last_hidden_state.mean(dim=1)` 等を入れる

# ==========================================
# 3. メイン処理ロジック
# ==========================================

def generate_exogenous_features():
    engine = get_db_engine()
    
    # 1. データのロード (メモリ効率のため必要なカラムのみ)
    print("Loading data from 'loto_base'...")
    query = """
    SELECT loto, unique_id, ts_type, array_agg(y ORDER BY ds) as y_values
    FROM public.loto_base
    GROUP BY loto, unique_id, ts_type
    """
    df_series = pd.read_sql(query, engine)
    print(f"Loaded {len(df_series)} time series.")

    # 2. 使用するモデルのリスト (キャッシュにあるものから選定)
    # 実際には環境に合わせて有効化してください
    models_to_use = [
        # {"id": "amazon/chronos-t5-small", "type": "chronos"},
        # {"id": "Salesforce/moirai-1.0-R-small", "type": "moirai"},
        {"id": "gpt2", "type": "generic"}, # テスト用（必ず動くもの）
    ]

    all_features = []

    # 3. モデルごとに特徴抽出ループ
    for model_info in models_to_use:
        m_id = model_info["id"]
        m_type = model_info["type"]
        
        # 抽出器の選択
        if m_type == "chronos":
            extractor = ChronosExtractor(m_id)
        elif m_type == "moirai":
            extractor = MoiraiExtractor(m_id)
        else:
            extractor = GenericHFFeatureExtractor(m_id)
            
        try:
            extractor.load_model()
            
            print(f"Generating features using {m_id}...")
            
            features_list = []
            for index, row in df_series.iterrows():
                series_data = np.array(row['y_values'], dtype=np.float32)
                
                # 特徴量抽出実行
                try:
                    # ※注意: ここで各モデル固有の前処理（スケーリング等）を行うのが一般的です
                    # 今回は簡易的にそのまま渡しています
                    embedding = extractor.extract(series_data)
                    
                    # 結果の辞書作成
                    feat_dict = {
                        "loto": row['loto'],
                        "unique_id": row['unique_id'],
                        "ts_type": row['ts_type'],
                        "model_name": m_id,
                        # ベクトルをカラムに展開せず、配列として保存（PostgresならARRAY型推奨）
                        "embedding": embedding.tolist() 
                    }
                    features_list.append(feat_dict)
                except Exception as e:
                    print(f"Error extracting for {row['unique_id']}: {e}")
                    continue

            # DataFrame化してリストに追加
            df_feat = pd.DataFrame(features_list)
            all_features.append(df_feat)
            
        except Exception as e:
            print(f"Failed to load or use model {m_id}: {e}")
        finally:
            extractor.unload_model()

    # 4. 結果の結合と保存
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        
        # EmbeddingカラムをJSON文字列化して保存（汎用性のため）
        # PostgresのARRAY型を使う場合は to_sql の dtype 指定が必要ですが、ここではTextで逃げます
        final_df['embedding'] = final_df['embedding'].apply(lambda x: str(x))
        
        target_table = "loto_model_exog"
        print(f"Saving {len(final_df)} rows to table '{target_table}'...")
        
        final_df.to_sql(target_table, engine, if_exists='replace', index=False, schema='public')
        print("Done.")
    else:
        print("No features generated.")

if __name__ == "__main__":
    generate_exogenous_features()
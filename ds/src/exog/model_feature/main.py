import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import os

# パッケージパス解決
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 自作モジュール
from config import Config
from db_utils import get_engine, create_database_if_not_exists, create_model_table
from monitor import ResourceMonitor
from models import FeatureExtractorModel

# --- Dataset Definition ---
class LotoDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        df: loto_baseテーブルから取得したデータフレーム。
            columns: [loto, unique_id, ts_type, ds, y]
        """
        self.groups = []
        print("[Data] Grouping data by unique series...")
        # (loto, unique_id, ts_type) でグループ化
        grouped = df.groupby(['loto', 'unique_id', 'ts_type'])
        
        for name, group in tqdm(grouped, desc="Creating sequences"):
            vals = group['y'].values.astype(np.float32)
            dates = group['ds'].tolist()
            
            self.groups.append({
                'key': name, # (loto, unique_id, ts_type)
                'values': vals,
                'dates': dates,
                'length': len(vals)
            })

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        return self.groups[idx]

def collate_fn(batch):
    """
    可変長の時系列をバッチ化するためのパディング処理
    """
    values_list = [torch.tensor(item['values']) for item in batch]
    lengths = torch.tensor([item['length'] for item in batch])
    
    # バッチ内のパディング (モデルへの入力前に揃える)
    padded_values = torch.nn.utils.rnn.pad_sequence(values_list, batch_first=True, padding_value=0.0)
    
    # Mask作成
    mask = torch.zeros_like(padded_values)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0
        
    metadata = []
    for item in batch:
        metadata.append({
            'key': item['key'],
            'dates': item['dates'],
            'length': item['length']
        })
        
    return padded_values, mask, metadata

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser(description="Generate Time-Series Embeddings")
    
    # 実行パラメータ
    parser.add_argument("--model", type=str, required=True, help="HuggingFace Model ID")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--limit", type=int, default=0, help="Limit input rows for testing (0=All)")
    
    # DBから渡される強制シーケンス長 (hf_snapshotsのcontext_length)
    parser.add_argument("--seq_len", type=int, default=None, help="Enforced sequence length from DB")
    
    # DB接続パラメータ (Config上書き用)
    parser.add_argument("--db_host", type=str, help="Database Host")
    parser.add_argument("--db_port", type=str, help="Database Port")
    parser.add_argument("--db_user", type=str, help="Database User")
    parser.add_argument("--db_pass", type=str, help="Database Password")
    
    args = parser.parse_args()

    # 設定の確定
    Config.update_from_args(args)
    
    print(f"[System] Feature DB: {Config.FEATURE_DB} | Resource DB: {Config.RESOURCE_DB}")

    # 1. DB準備 (保存先DBが存在しなければ作成)
    create_database_if_not_exists(Config.FEATURE_DB)
    create_database_if_not_exists(Config.RESOURCE_DB)
    
    # 2. リソース監視開始
    monitor = ResourceMonitor(args.model)
    monitor.start()

    try:
        # 3. データロード (Source DB: dataset)
        print(f"[Data] Loading from {Config.SOURCE_DB}...")
        src_engine = get_engine(Config.SOURCE_DB)
        limit_clause = f"LIMIT {args.limit}" if args.limit > 0 else ""
        
        query = f"""
        SELECT loto, unique_id, ts_type, ds, y
        FROM {Config.SOURCE_TABLE}
        ORDER BY loto, unique_id, ts_type, ds ASC
        {limit_clause}
        """
        
        df = pd.read_sql(query, src_engine)
        if df.empty:
            print("No data found.")
            return

        dataset = LotoDataset(df)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=args.num_workers
        )

        # 4. モデルロード (DB指定のseq_lenを適用)
        model_wrapper = FeatureExtractorModel(args.model, force_seq_len=args.seq_len)
        hidden_dim = model_wrapper.get_embedding_dim()
        
        # 5. 保存先テーブル作成 (Feature DB: model_exog)
        # db_utils内で Config.FEATURE_DB を使用するように修正済み
        table_name, feature_cols = create_model_table(args.model, hidden_dim)
        
        # 保存用エンジンを取得
        target_engine = get_engine(Config.FEATURE_DB)

        # 6. 推論ループ
        print(f"[Inference] Starting inference loop for {len(dataset)} series...")
        
        data_buffer = []
        BUFFER_SIZE = 2000 
        
        for batch_values, batch_mask, metadata in tqdm(dataloader, desc="Processing Batches"):
            try:
                # 推論 (B, Max_Len, Hidden)
                # models.py 内で seq_len に合わせたパディング/トリミングが行われる
                embeddings = model_wrapper.forward(batch_values, batch_mask)
                embeddings = embeddings.cpu().numpy() # GPU -> CPU
                
                # Unpadding & レコード生成
                for i, meta in enumerate(metadata):
                    loto, uid, ttype = meta['key']
                    dates = meta['dates']
                    valid_len = meta['length']
                    
                    # 有効な長さにスライス (embeddingsは右詰めされている可能性があるため注意が必要だが、
                    # models.pyで復元処理(右端取得)済みのため、ここでは0からvalid_lenでOKとは限らない)
                    # models.py の修正により、embeddings は「入力と同じ長さ」か「seq_len」のどちらかで返ってくる。
                    # ここでは「日付の数」と合わせる必要がある。
                    
                    # モデル出力の系列長
                    out_seq_len = embeddings.shape[1]
                    
                    # 出力が日付数より多い場合 (Paddingされている) -> 右端(最新)から日付数分を取る
                    if out_seq_len >= valid_len:
                        valid_emb = embeddings[i, -valid_len:, :]
                    else:
                        # 出力が日付数より少ない場合 (モデルの制限でTrimされた) -> 日付もTrimする
                        # エラーにせず、計算できた分だけ保存する方針
                        valid_emb = embeddings[i, :, :]
                        dates = dates[-out_seq_len:] # 日付も最新分に合わせる
                        
                    # 整合性チェック
                    if len(dates) != valid_emb.shape[0]:
                        print(f"[Error] Length mismatch after processing for {uid}. Dates: {len(dates)}, Emb: {valid_emb.shape[0]}")
                        continue
                        
                    # 行データ作成
                    for t in range(len(dates)):
                        row = {
                            'loto': loto,
                            'unique_id': uid,
                            'ts_type': ttype,
                            'ds': dates[t]
                        }
                        # 特徴量展開
                        feats = {f"hist_{table_name}_{d}": float(valid_emb[t, d]) for d in range(hidden_dim)}
                        row.update(feats)
                        data_buffer.append(row)
                
                # バッファ保存
                if len(data_buffer) >= BUFFER_SIZE:
                    save_df = pd.DataFrame(data_buffer)
                    save_df.to_sql(table_name, target_engine, if_exists='append', index=False, method='multi', chunksize=500)
                    data_buffer = [] 

            except Exception as e:
                print(f"[Error] Batch processing failed: {e}")
                # import traceback
                # traceback.print_exc()
                continue

        # 残りのバッファを保存
        if data_buffer:
            print("[DB] Saving remaining records...")
            save_df = pd.DataFrame(data_buffer)
            save_df.to_sql(table_name, target_engine, if_exists='append', index=False, method='multi', chunksize=500)

        print("[Done] Processing complete.")

    except KeyboardInterrupt:
        print("[System] Interrupted by user.")
    except Exception as e:
        print(f"[System] Critical Error: {e}")
        # import traceback
        # traceback.print_exc()
        raise e 
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()
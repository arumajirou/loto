import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

# 設定とモジュールパスの解決
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from db_manager import DBManager
from models.detectors import get_detectors

warnings.filterwarnings('ignore')

class AnomalyBatchProcessor:
    def __init__(self, config_path: str, input_df: pd.DataFrame = None):
        self.db = DBManager(config_path)
        self.input_df = input_df
        # 設定ファイルから読み込む汚染率
        self.contamination = float(self.db.config['SETTINGS'].get('CONTAMINATION', 0.05))

    def run_library(self, library_name: str):
        """指定されたライブラリ（sklearn, pyod等）のモデルを実行して保存"""
        print(f"🚀 Starting anomaly detection using library: {library_name}")
        
        models = get_detectors(library_name, self.contamination)
        if not models:
            print(f"⚠️ No models found or library not available for: {library_name}")
            return

        if self.input_df is None:
            # DBからロードする場合（今回はNotebookから渡される想定なのでスキップ）
            print("Error: Input DataFrame is missing.")
            return

        df = self.input_df.copy()
        
        # 結果格納用辞書の初期化
        # キー: (loto, unique_id, ts_type) -> {exec_ts: [], result: []} は重いので
        # 元のDFにカラムを追加していく方式をとる
        
        # グルーピングキー
        group_keys = ['loto', 'unique_id', 'ts_type']
        
        # モデルごとにカラムを作成（初期値NaNまたは0）
        for model in models:
            col_name = f"hist_{model.name}"
            df[col_name] = 0
        
        # グループ化して処理
        # tqdmで進捗表示
        grouped = df.groupby(group_keys)
        
        print(f"Processing {len(grouped)} groups...")
        
        results_list = []
        
        for name, group in tqdm(grouped):
            # nameは (loto, unique_id, ts_type) のタプル
            # 時系列順であることを保証
            group = group.sort_values('exec_ts')
            y_values = group['y'].values
            
            # 欠損値処理 (線形補間 -> 前後埋め)
            if np.isnan(y_values).any():
                s = pd.Series(y_values)
                s = s.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
                y_values = s.values

            # 全てNaNあるいはデータ点数が少なすぎる場合はスキップ（すべて正常とする）
            if len(y_values) < 10 or np.isnan(y_values).any():
                continue

            # 各モデル実行
            for model in models:
                try:
                    preds = model.detect(y_values)
                    # 元のインデックスに対応させるため、結果をマッピング
                    col_name = f"hist_{model.name}"
                    # groupのindexを使ってdf本体更新は遅いので、結果リストに保存する方式推奨
                    # ここでは group に列を追加して後で結合する方式をとる
                    group[col_name] = preds
                except Exception as e:
                    # 個別の計算エラーはログに出して続行
                    # print(f"Error in {name} with {model.name}: {e}")
                    pass
            
            results_list.append(group)

        # 結果の結合
        if results_list:
            final_df = pd.concat(results_list)
            # 必要なカラムのみ抽出
            base_cols = ['loto', 'ds', 'unique_id', 'ts_type', 'exec_ts', 'y']
            result_cols = [f"hist_{m.name}" for m in models]
            output_df = final_df[base_cols + result_cols]
            
            # DB保存
            self.db.save_results(output_df, library_name)
        else:
            print("No results generated.")

if __name__ == "__main__":
    # スタンドアロンテスト用
    pass
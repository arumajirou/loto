import pandas as pd
import numpy as np
import logging
from joblib import Parallel, delayed
import warnings

# sktime transformers
from sktime.transformations.panel.catch22 import Catch22
from sktime.transformations.series.summarize import SummaryTransformer
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sktime.transformations.panel.rocket import MiniRocket

from . import config

logger = logging.getLogger(__name__)

# 特定の警告を無視（TSFreshなどが頻繁に出すため）
warnings.filterwarnings("ignore")

def _process_single_series(group_key, group_df, transformers_dict):
    """
    並列処理用の単一タスク関数
    1つの系列(group_df)を受け取り、辞書にある全てのTransformerを適用して結合する
    """
    loto, unique_id, ts_type = group_key
    
    # 時系列順にソートしてSeriesを作成
    y_series = group_df.sort_values('ds').set_index('ds')['y']
    
    # 結果格納用リスト
    generated_dfs = []
    
    try:
        # 各Transformerを順次適用
        for name, (transformer, prefix) in transformers_dict.items():
            try:
                # fit_transformを実行
                # 注意: sktimeの一部のtransformerはfitが必要
                # データ量が少ないため、都度fit_transformしてもコストは許容範囲と想定
                feats = transformer.fit_transform(y_series)
                
                # DataFrameでない場合(Seriesの場合)の補正
                if isinstance(feats, pd.Series):
                    feats = feats.to_frame().T
                
                # カラムリネーム
                feats.columns = [f"{prefix}{c}" for c in feats.columns]
                
                # indexのリセット（結合のため）
                feats.reset_index(drop=True, inplace=True)
                generated_dfs.append(feats)
                
            except Exception as e:
                # 1つの特徴量生成が失敗しても他は通す
                # logger.debug(f"Failed {name} for {group_key}: {e}")
                pass

        if not generated_dfs:
            return None

        # 横方向に結合
        final_feats = pd.concat(generated_dfs, axis=1)
        
        # キー列を追加
        final_feats['loto'] = loto
        final_feats['unique_id'] = unique_id
        final_feats['ts_type'] = ts_type
        
        return final_feats

    except Exception as e:
        return None

class SktimeFeatureGenerator:
    def __init__(self):
        self.transformers = {}
        
        # 1. Catch22
        if config.USE_CATCH22:
            self.transformers['catch22'] = (
                Catch22(replace_nans=True, n_jobs=1),
                config.PREFIX_CATCH22
            )
            
        # 2. Summary Statistics
        if config.USE_SUMMARY:
            # 基本統計量
            self.transformers['summary'] = (
                SummaryTransformer(),
                config.PREFIX_SUMMARY
            )
            
        # 3. TSFresh
        if config.USE_TSFRESH:
            self.transformers['tsfresh'] = (
                TSFreshFeatureExtractor(
                    default_fc_parameters=config.TSFRESH_SETTING,
                    show_warnings=False,
                    n_jobs=1  # joblibの外側で並列化するためここは1
                ),
                config.PREFIX_TSFRESH
            )
            
        # 4. MiniRocket
        if config.USE_MINIROCKET:
            self.transformers['minirocket'] = (
                MiniRocket(
                    num_kernels=config.MINIROCKET_KERNELS,
                    random_state=42,
                    n_jobs=1
                ),
                config.PREFIX_MINIROCKET
            )

    def generate(self, df):
        """
        特徴量生成のメインプロセス (並列化対応)
        """
        if not self.transformers:
            logger.warning("No transformers selected in config.py")
            return pd.DataFrame()

        logger.info(f"Active Transformers: {list(self.transformers.keys())}")
        logger.info("Preprocessing: Preparing data...")
        
        df_clean = df.copy()
        df_clean['ds'] = pd.to_datetime(df_clean['ds'])
        
        # 欠損値補完（共通前処理）
        df_clean['y'] = df_clean.groupby(['loto', 'unique_id', 'ts_type'])['y'].transform(
            lambda x: x.ffill().bfill().fillna(0)
        )

        # グループ化してリスト化
        groups = [
            (name, group) 
            for name, group in df_clean.groupby(['loto', 'unique_id', 'ts_type'])
        ]
        
        n_series = len(groups)
        logger.info(f"Starting parallel feature generation on {n_series} series (n_jobs=-1)...")
        
        # --- Joblibによる並列実行 ---
        # transformers辞書を各ワーカーに渡す
        parallel_results = Parallel(n_jobs=-1, backend='loky')(
            delayed(_process_single_series)(
                name, 
                group, 
                self.transformers
            ) 
            for name, group in groups
        )
        
        # 結果のフィルタリングと結合
        results = [res for res in parallel_results if res is not None]
        success_count = len(results)
        
        logger.info(f"Processing completed. Success: {success_count}/{n_series}")

        if not results:
            logger.warning("No features were generated.")
            return pd.DataFrame()

        final_df = pd.concat(results, axis=0).reset_index(drop=True)
        
        # カラム順序の調整 (キーを先頭に)
        key_cols = ['loto', 'unique_id', 'ts_type']
        feature_cols = [c for c in final_df.columns if c not in key_cols]
        
        # 列名重複の回避（念のため）
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        
        # 再取得（重複排除後）
        feature_cols = [c for c in final_df.columns if c not in key_cols]
        final_df = final_df[key_cols + feature_cols]
        
        return final_df
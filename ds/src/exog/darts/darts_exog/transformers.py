import pandas as pd
import numpy as np
import logging
from tqdm import tqdm

# Darts imports
from darts import TimeSeries
from darts.dataprocessing.transformers import Mapper, MissingValuesFiller
from darts.utils.timeseries_generation import datetime_attribute_timeseries

from . import config

logger = logging.getLogger(__name__)

class DartsFeatureGenerator:
    def __init__(self):
        self.imputer = MissingValuesFiller()

    def _df_to_darts(self, df):
        """
        DataFrameを (key) -> TimeSeries の辞書に変換する
        """
        logger.info("Converting DataFrame to Darts TimeSeries objects...")
        series_dict = {}
        
        # グループ化: (loto, unique_id, ts_type)
        grouped = df.groupby(['loto', 'unique_id', 'ts_type'])
        
        for key, group in tqdm(grouped, desc="Converting"):
            # 時間インデックスの設定
            df_grp = group.sort_values('ds')
            
            # 頻度推定が難しい場合があるため、単純な時系列として扱う
            # fill_missing_dates=True だと頻度が必須になるため、
            # ここでは欠損日付は埋めず、存在するデータのみでTimeSeriesを作る
            # (注意: 回号ベースの場合、日付の穴は無視できる)
            try:
                ts = TimeSeries.from_dataframe(
                    df_grp, 
                    time_col='ds', 
                    value_cols='y',
                    freq=None # 推論させる
                )
                series_dict[key] = ts
            except Exception as e:
                # 頻度推論に失敗した場合（日付が飛んでいる等）、整列済みと仮定して index を貼り直す手もあるが
                # 今回はログを出してスキップまたは強行
                logger.warning(f"Failed to create TimeSeries for {key}: {e}")
                
        return series_dict

    def generate(self, df):
        """
        全特徴量生成のメインプロセス
        """
        # 1. 前処理: データクリーニング
        # グループごとの欠損補完（Pandasレベルで事前にやっておく方がDarts変換時に安全）
        df_clean = df.copy()
        df_clean['y'] = df_clean.groupby(['loto', 'unique_id', 'ts_type'])['y'].transform(
            lambda x: x.ffill().bfill().fillna(0)
        )
        
        # 2. Darts TimeSeries に変換
        ts_dict = self._df_to_darts(df_clean)
        
        result_dfs = []
        
        logger.info("Generating features for each series...")
        # 各系列ごとに処理（並列化も可能だが、Darts内部処理に任せるかループで処理）
        for key, ts in tqdm(ts_dict.items(), desc="Processing"):
            loto, unique_id, ts_type = key
            
            # 特徴量格納用リスト (TimeSeries)
            features = []
            
            # --- A. カレンダー特徴量 (Datetime Attributes) ---
            # ts の時間インデックスを使って生成
            for attr in config.TIME_ATTRS:
                try:
                    # TimeSeriesを作成
                    cov = datetime_attribute_timeseries(ts, attribute=attr, one_hot=False)
                    # カラム名変更
                    cov = cov.with_columns_renamed(cov.columns, [f"{config.PREFIX}time_{attr}"])
                    features.append(cov)
                except Exception:
                    pass # 週番号など一部取得できない属性がある場合の対策

            # --- B. ラグ特徴量 (Shift) ---
            for lag in config.LAGS:
                # shift(正) = 未来のデータを現在に持ってくる(リーク)
                # shift(負) = 過去のデータを現在に持ってくる(ラグ) -> Dartsのshiftは正で遅延させる
                shifted = ts.shift(lag)
                shifted = shifted.with_columns_renamed(shifted.columns, [f"{config.PREFIX}lag_{lag}"])
                features.append(shifted)

            # --- C. 移動窓統計 (Rolling Window) ---
            # DartsにはWindowTransformerがあるが、ここではPandasのrollingをラップしたMapper等が使いやすい
            # TimeSeries.pd_dataframe() 経由で計算して戻すのが最速かつ確実
            
            ts_pd = ts.pd_dataframe()
            
            # Rolling
            for window in config.WINDOW_SIZES:
                rolled = ts_pd.rolling(window=window)
                for stat in config.WINDOW_STATS:
                    try:
                        # skewなどは数値計算エラーが出る可能性があるのでケア
                        if stat == 'skew':
                            feat_series = rolled.skew()
                        elif stat == 'var':
                            feat_series = rolled.var()
                        else:
                            # mean, std, min, max, sum
                            feat_series = getattr(rolled, stat)()
                            
                        # Series -> TimeSeries
                        # 名前
                        col_name = f"{config.PREFIX}roll_{window}_{stat}"
                        feat_ts = TimeSeries.from_times_and_values(ts.time_index, feat_series.values, columns=[col_name])
                        features.append(feat_ts)
                    except Exception:
                        pass

            # --- D. 階差 (Differences) ---
            for period in config.DIFF_PERIODS:
                diffed = ts.diff(n=period)
                diffed = diffed.with_columns_renamed(diffed.columns, [f"{config.PREFIX}diff_{period}"])
                features.append(diffed)

            # --- 統合と整形 ---
            if features:
                # 全特徴量を横結合 (concatenate along 'component' axis)
                # Dartsでは axis=1 が component
                try:
                    full_ts = ts.concatenate(features, axis=1)
                    
                    # 再び DataFrame に戻す
                    df_res = full_ts.pd_dataframe().reset_index()
                    
                    # キー列を追加
                    df_res['loto'] = loto
                    df_res['unique_id'] = unique_id
                    df_res['ts_type'] = ts_type
                    
                    # ds列の名前を合わせる (index名が 'time' になっている場合がある)
                    if 'ds' not in df_res.columns:
                        # datetime indexの名前を探す、あるいは一番目のカラムが時間と仮定
                        time_col = df_res.columns[0]
                        df_res.rename(columns={time_col: 'ds'}, inplace=True)

                    result_dfs.append(df_res)
                    
                except Exception as e:
                    logger.warning(f"Failed to concat features for {key}: {e}")

        if not result_dfs:
            return pd.DataFrame()

        # 全系列を縦結合
        logger.info("Concatenating all results...")
        final_df = pd.concat(result_dfs, axis=0, ignore_index=True)
        
        # カラム順序整理
        key_cols = ['loto', 'unique_id', 'ts_type', 'ds']
        feat_cols = [c for c in final_df.columns if c not in key_cols]
        final_df = final_df[key_cols + feat_cols]
        
        return final_df
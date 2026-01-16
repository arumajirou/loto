import pandas as pd
import numpy as np
from scipy.stats import linregress
from . import config

def _calculate_slope(y):
    """トレンドの傾きを計算（欠損がある場合はNone）"""
    y_clean = y.dropna()
    if len(y_clean) < 2:
        return np.nan
    x = np.arange(len(y_clean))
    slope, _, _, _, _ = linregress(x, y_clean)
    return slope

def _calculate_autocorr(y, lag):
    """自己相関を計算"""
    if len(y) <= lag:
        return np.nan
    return y.autocorr(lag=lag)

def calculate_series_stats(group_df):
    """
    1つの系列(loto, unique_id, ts_type)に対する統計量を計算する
    """
    # グループキー情報の取得
    first_row = group_df.iloc[0]
    result = {
        'loto': first_row['loto'],
        'unique_id': first_row['unique_id'],
        'ts_type': first_row['ts_type'],
        'ds_min': group_df['ds'].min(),
        'ds_max': group_df['ds'].max(),
        'span_days': (group_df['ds'].max() - group_df['ds'].min()).days
    }
    
    y = group_df['y']
    
    # 3.1 基本統計
    result['count'] = len(y)
    result['missing_rate'] = y.isna().mean()
    result['mean'] = y.mean()
    result['std'] = y.std()
    result['min'] = y.min()
    result['max'] = y.max()
    
    # 分位点
    quantiles = y.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    result['q05'] = quantiles[0.05]
    result['q25'] = quantiles[0.25]
    result['q50'] = quantiles[0.50]
    result['q75'] = quantiles[0.75]
    result['q95'] = quantiles[0.95]
    
    # 3.2 直近統計
    # 最後の有効値を取得
    y_valid = y.dropna()
    if not y_valid.empty:
        result['last'] = y_valid.iloc[-1]
        
        # 差分
        if len(y_valid) >= 2:
            result['diff1_last'] = y_valid.iloc[-1] - y_valid.iloc[-2]
        else:
            result['diff1_last'] = np.nan
            
        # 窓関数統計 (config.WINDOWS = [7, 14, 30])
        for w in config.WINDOWS:
            if len(y_valid) >= w:
                recent = y_valid.iloc[-w:]
                result[f'last{w}_mean'] = recent.mean()
                result[f'last{w}_std'] = recent.std()
            else:
                result[f'last{w}_mean'] = np.nan
                result[f'last{w}_std'] = np.nan
    else:
        # 全てNULLの場合
        result['last'] = np.nan
        result['diff1_last'] = np.nan
        for w in config.WINDOWS:
            result[f'last{w}_mean'] = np.nan
            result[f'last{w}_std'] = np.nan

    # 3.3 形状（トレンド・自己相関）
    result['trend_slope'] = _calculate_slope(y)
    result['autocorr_lag1'] = _calculate_autocorr(y, 1)
    result['autocorr_lag7'] = _calculate_autocorr(y, 7)
    
    return pd.Series(result)

def generate_hist_features(df):
    """
    DataFrame全体に対して特徴量生成を実行
    """
    # applyの実行
    # 注意: データ量が非常に多い場合、並列化(joblib/pandarallel)を検討するが、
    # 今回は45万行程度なのでpandas applyで数分以内に収まると想定。
    features = df.groupby(['loto', 'unique_id', 'ts_type']).apply(calculate_series_stats)
    
    # indexがMultiIndexになっているのでリセットするが、
    # calculate_series_stats内でキーを辞書に入れているので、drop=Trueでも良いが安全策で。
    features = features.reset_index(drop=True)
    
    # カラム名に接頭辞をつける（キー列以外）
    key_cols = ['loto', 'unique_id', 'ts_type', 'ds_min', 'ds_max', 'span_days'] # ds系はhistつけないか、要件に合わせて調整
    
    # 要件: "特徴量カラムにはhistという接頭辞"
    # ds_minなども分析上の特徴量になりうるので、ts_typeなどの主キー以外は全てhist_をつける方針にする
    rename_map = {}
    for col in features.columns:
        if col not in ['loto', 'unique_id', 'ts_type']:
            rename_map[col] = f"{config.PREFIX}{col}"
            
    features = features.rename(columns=rename_map)
    
    return features
# ds/src/hist_exog/exog_feature_creator.py

import pandas as pd
import numpy as np
import os
from typing import Dict, Any, List, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Scipyの統計関数
from scipy.signal import periodogram, find_peaks
from scipy.stats import entropy, iqr, skew, kurtosis

# 外部ライブラリのインポート
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x 

try:
    import antropy as ant
    ANTROPY_AVAILABLE = True
except ImportError:
    ant = None
    ANTROPY_AVAILABLE = False

# 内部モジュールからの定数をインポート
from hist_exog_utils import logger, ROLLING_WINDOWS, EWM_ALPHAS, COMPLEX_WINDOWS

# ==========================================
# 複雑性・周波数特徴量の計算ロジック (ヘルパー関数)
# ==========================================

def _calc_spectral_entropy(x, fs=1.0):
    try:
        _, psd = periodogram(x, fs, scaling='density')
        psd_norm = psd / np.sum(psd)
        return entropy(psd_norm + 1e-10)
    except:
        return np.nan

def _calc_permutation_entropy(x):
    if not ANTROPY_AVAILABLE: return np.nan
    try:
        return ant.perm_entropy(x, order=3, delay=1, normalize=True)
    except:
        return np.nan

def _calc_higuchi_fd(x):
    if not ANTROPY_AVAILABLE: return np.nan
    try:
        return ant.higuchi_fd(x, kmax=5)
    except:
        return np.nan

def _calc_cid_ce(x):
    if len(x) < 2: return 0.0
    diff = np.diff(x)
    return np.sqrt(np.sum(diff ** 2))

def _calc_num_peaks(x):
    try:
        peaks, _ = find_peaks(x, distance=1)
        return len(peaks)
    except:
        return np.nan

def _calc_autocorr(x, lag=1):
    x_series = pd.Series(x)
    if len(x_series.dropna()) <= lag: return np.nan
    return x_series.autocorr(lag=lag)


def _generate_features_for_series(group: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """
    単一の時系列グループ (loto, unique_id, ts_type) に対する特徴量生成 (サブプロセス実行)
    """
    
    y_past = group['y'].shift(1)
    sql_cols = [c for c in group.columns if c.startswith('sql_')]
    
    feats = pd.DataFrame(index=group.index)
    
    # --- B. Pythonで計算する統計ローリング ---
    for w in ROLLING_WINDOWS:
        roller = y_past.rolling(window=w, min_periods=max(1, w//2))
        
        feats[f'py_roll_std_{w}'] = roller.std()
        feats[f'py_roll_median_{w}'] = roller.median()
        feats[f'py_roll_range_{w}'] = roller.max() - roller.min()
        feats[f'py_roll_skew_{w}'] = roller.skew()
        feats[f'py_roll_kurt_{w}'] = roller.kurt()
        feats[f'py_roll_q10_{w}'] = roller.quantile(0.1)
        feats[f'py_roll_q90_{w}'] = roller.quantile(0.9)
        # raw=FalseでSeriesを渡し、dropna()を実行 (修正済)
        feats[f'py_roll_iqr_{w}'] = roller.apply(lambda x: iqr(x.dropna()), raw=False)

    # --- C. 変化率・差分 ---
    for k in [1, 7]:
        feats[f'py_diff_{k}'] = y_past.diff(k)
        feats[f'py_pct_change_{k}'] = y_past.pct_change(k).replace([np.inf, -np.inf], 0)

    # --- D. 指数平滑化 ---
    for alpha in EWM_ALPHAS:
        ewm = y_past.ewm(alpha=alpha, adjust=False)
        feats[f'py_ewm_mean_{alpha}'] = ewm.mean()
        feats[f'py_ewm_std_{alpha}'] = ewm.std()

    # --- E. 拡大窓統計 ---
    feats['py_expand_mean'] = y_past.expanding().mean()
    feats['py_expand_std'] = y_past.expanding().std()

    # --- F. 異常検知スコア的指標 ---
    for w in ROLLING_WINDOWS:
        med = feats[f'py_roll_median_{w}']
        iqr_val = feats[f'py_roll_q90_{w}'] - feats[f'py_roll_q10_{w}'] 
        feats[f'py_rob_z_{w}'] = (y_past - med) / (iqr_val + 1e-6)

    # --- G. 複雑性・周波数・スペクトル特徴 ---
    if ANTROPY_AVAILABLE:
        for w in COMPLEX_WINDOWS:
            min_periods = max(5, w//2)
            roller = y_past.rolling(w, min_periods=min_periods)
            
            # raw=Falseを使用し、x.dropna().values で NumPy 配列に変換 (修正済)
            feats[f'py_ac1_{w}'] = roller.apply(lambda x: _calc_autocorr(x.dropna(), 1), raw=False) 
            feats[f'py_spec_ent_{w}'] = roller.apply(lambda x: _calc_spectral_entropy(x.dropna().values), raw=False)
            feats[f'py_perm_ent_{w}'] = roller.apply(lambda x: _calc_permutation_entropy(x.dropna().values), raw=False)
            feats[f'py_higuchi_{w}'] = roller.apply(lambda x: _calc_higuchi_fd(x.dropna().values), raw=False)
            feats[f'py_cid_{w}'] = roller.apply(lambda x: _calc_cid_ce(x.dropna().values), raw=False)
            feats[f'py_peaks_{w}'] = roller.apply(lambda x: _calc_num_peaks(x.dropna().values), raw=False)

    # --- H. ドメイン・カレンダー特徴 ---
    feats['py_is_odd'] = y_past % 2
    feats['py_last_digit'] = y_past % 10
    
    # 元のキー情報と計算結果を結合
    result_df = pd.concat([group[key_cols + ['y']], group[sql_cols], feats], axis=1)
    
    return result_df

# ==========================================
# 特徴量生成クラス (並列処理を含む)
# ==========================================
class ExogFeatureCreator:
    
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = os.cpu_count() if n_jobs == -1 else n_jobs
        if not ANTROPY_AVAILABLE:
             logger.warning("Antropy not available. Complexity features will be skipped or set to NaN.")

    def create_features_parallel(self, df_base: pd.DataFrame, tqdm_func: Callable, key_cols: List[str]) -> pd.DataFrame:
        """
        全グループに対して並列で特徴量生成を実行する
        """
        
        groups = [group for _, group in df_base.groupby(['loto', 'unique_id', 'ts_type'], sort=False)]
        total_groups = len(groups)
        
        func = partial(_generate_features_for_series, key_cols=key_cols)

        results_list: List[pd.DataFrame] = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            
            future_to_group = {executor.submit(func, group): (group['loto'].iloc[0], group['unique_id'].iloc[0], group['ts_type'].iloc[0]) for group in groups}
            
            with tqdm_func(as_completed(future_to_group), total=total_groups, desc="Generating Exog Features") as pbar:
                for future in pbar:
                    group_info = future_to_group[future]
                    try:
                        df_feats = future.result()
                        results_list.append(df_feats)
                    except Exception as exc:
                        logger.error(f"Error processing group {group_info}: {exc}")
                        pass
        
        # エラーで結果リストが空の場合を考慮
        if not results_list:
            raise RuntimeError("No objects to concatenate: Feature generation failed for all groups.")
            
        df_final = pd.concat(results_list, axis=0)
        
        # 最終的な欠損値処理 (キーとy以外を0埋め)
        key_cols_all = key_cols + ['y']
        feature_cols = [c for c in df_final.columns if c not in key_cols_all]
        df_final[feature_cols] = df_final[feature_cols].fillna(0).astype('float64')

        logger.info(f"Generated {len(feature_cols)} features. Final Shape: {df_final.shape}")
            
        return df_final
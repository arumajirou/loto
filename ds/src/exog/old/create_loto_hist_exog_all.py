import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
from tqdm import tqdm

# ==========================================
# 1. 設定・定数
# ==========================================
DEFAULT_INPUT_TABLE = "loto_y_ts"
DEFAULT_OUTPUT_TABLE = "loto_hist_exog_all"
DEFAULT_BASE_TABLE = "loto_base" 

ROLLING_WINDOWS = [7, 14, 28, 56, 91, 182] 
LAGS = [1, 2, 3, 7, 14, 28, 56, 91, 364]
EWM_ALPHAS = [0.1, 0.3, 0.5]
COMPLEX_WINDOWS = [28, 56, 91]

# 並列処理設定
N_JOBS = os.cpu_count() or 4 

# 外部ライブラリのインポート（変更なし）
try:
    from scipy.signal import periodogram, find_peaks
    from scipy.stats import entropy, skew, kurtosis 
    import antropy as ant
except ImportError:
    print("WARNING: 'scipy' or 'antropy' library not fully available. Complexity features will be 0.")
    ant = None

# ==========================================
# 2. DB接続・データ取得 (SQLで特徴量を計算)
# ==========================================
def get_db_engine():
    host = os.environ.get("PGHOST", "127.0.0.1")
    port = os.environ.get("PGPORT", "5432")
    dbname = os.environ.get("PGDATABASE", "loto_db")
    user = os.environ.get("PGUSER", "loto_user")
    password = os.environ.get("PGPASSWORD", "z")
    
    url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    try:
        engine = create_engine(url)
        return engine
    except Exception as e:
        print(f"ERROR: PostgreSQL接続エラー: {e}")
        raise

def build_sql_features(schema: str, table_name: str, rolling_windows: List[int], lags: List[int]) -> str:
    """PostgreSQLのウィンドウ関数を使用して特徴量計算を行うSQLクエリを構築する"""
    
    window_def = "PARTITION BY loto, unique_id, ts_type ORDER BY ds"
    
    sql_parts = [
        "t.loto", "t.unique_id", "t.ds", "t.ts_type", 
        "t.y_transformed AS y", # y_transformed を新しいターゲット変数 y にリネーム
        "t.y_transformed AS y_past_for_python" # Pythonの特徴量計算用（t-1を内部でシフトする）
    ]
    
    # 1. ラグ特徴量 (LAG)
    for lag in lags:
        sql_parts.append(
            f"CAST(LAG(t.y_transformed, {lag}) OVER ({window_def}) AS double precision) AS lag_{lag}" # CASTで型を統一
        )

    # 2. ローリング集計 (AVG, SUM, STDDEV)
    for w in rolling_windows:
        roll_window_def = f"ROWS BETWEEN {w} PRECEDING AND 1 PRECEDING"
        
        sql_parts.append(
            f"AVG(t.y_transformed) OVER ({window_def} {roll_window_def}) AS roll_mean_{w}"
        )
        sql_parts.append(
            f"STDDEV(t.y_transformed) OVER ({window_def} {roll_window_def}) AS roll_std_{w}"
        )
        sql_parts.append(
            f"SUM(t.y_transformed) OVER ({window_def} {roll_window_def}) AS roll_sum_{w}"
        )
        
        # 3. 差分 (LAGを利用)
        for k in [1, 7]:
            sql_parts.append(
                f"t.y_transformed - LAG(t.y_transformed, {k}) OVER ({window_def}) AS diff_{k}"
            )

    sql_select = ",\n".join(sql_parts)

    return f"""
SELECT 
{sql_select}
FROM {schema}.{table_name} t
WHERE t.y_transformed IS NOT NULL
ORDER BY loto, unique_id, ts_type, ds
"""


def fetch_y_transformed_data_with_sql_features(engine, schema, table_name) -> pd.DataFrame:
    """loto_y_tsからSQLで計算された特徴量を加えてデータを取得"""
    print(f"Fetching data from {schema}.{table_name} with SQL Window Functions (4-key output)...")
    
    query = build_sql_features(schema, table_name, ROLLING_WINDOWS, LAGS)

    try:
        df = pd.read_sql(query, engine)
        df['ds'] = pd.to_datetime(df['ds']).dt.date
        
        # SQLから取得した特徴量列を float64 に統一 (NaNを含む可能性があるため)
        feature_cols_from_sql = [c for c in df.columns if c not in ['loto', 'unique_id', 'ds', 'ts_type', 'y', 'y_past_for_python']]
        df[feature_cols_from_sql] = df[feature_cols_from_sql].astype(np.float64)
        
        return df
    except Exception as e:
        print(f"ERROR: SQL特徴量生成クエリの実行に失敗: {e}")
        return pd.DataFrame()


def fetch_base_data_for_merge(engine, schema, table_name) -> pd.DataFrame:
    """loto_baseからキーとy_rawを取得（最終結合用 - 今回は使用しないが関数は維持）"""
    return pd.DataFrame()


# ==========================================
# 3. 特徴量計算ロジック（Pythonで複雑な特徴量のみ計算）
# ==========================================

# calc_* 関数は変更なし
def calc_spectral_entropy(x, fs=1.0):
    try:
        _, psd = periodogram(x, fs)
        psd_norm = psd / np.sum(psd)
        return entropy(psd_norm)
    except:
        return 0.0

def calc_permutation_entropy(x):
    if ant is None: return 0.0
    try:
        return ant.perm_entropy(x, order=3, normalize=True)
    except:
        return 0.0

def calc_higuchi_fd(x):
    if ant is None: return 0.0
    try:
        return ant.higuchi_fd(x, kmax=5)
    except:
        return 0.0

def calc_cid_ce(x):
    if len(x) < 2: return 0.0
    diff = np.diff(x)
    return np.sqrt(np.sum(diff ** 2))

def calc_num_peaks(x):
    try:
        peaks, _ = find_peaks(x)
        return len(peaks)
    except:
        return 0

def calc_autocorr(x, lag=1):
    if len(x) <= lag: return 0.0
    return pd.Series(x).autocorr(lag=lag) 


def generate_python_features(group: pd.DataFrame, y_col: str = 'y_past_for_python') -> pd.DataFrame:
    """
    1つの時系列(group)に対するPythonでの特徴量生成（EWM, 複雑性, ドメイン特徴のみ）
    """
    
    y_past = group[y_col] 
    feats = pd.DataFrame(index=group.index) 
    
    # --- D. 指数平滑化 (6種) ---
    for alpha in EWM_ALPHAS:
        ewm = y_past.ewm(alpha=alpha, adjust=False)
        feats[f'ewm_mean_{alpha}'] = ewm.mean()
        feats[f'ewm_std_{alpha}'] = ewm.std()

    # --- E. 拡大窓統計 (2種) ---
    feats['expand_mean'] = y_past.expanding().mean()
    feats['expand_std'] = y_past.expanding().std()

    # --- F. 複雑性・周波数・スペクトル特徴 (18種) ---
    if ant is not None:
        for w in COMPLEX_WINDOWS:
            roller_complex = y_past.rolling(window=w, min_periods=w)
            
            feats[f'ac1_{w}'] = roller_complex.apply(lambda x: calc_autocorr(x[~np.isnan(x)], 1), raw=True).astype(np.float64)
            feats[f'spec_ent_{w}'] = roller_complex.apply(lambda x: calc_spectral_entropy(x[~np.isnan(x)]), raw=True).astype(np.float64)
            feats[f'perm_ent_{w}'] = roller_complex.apply(lambda x: calc_permutation_entropy(x[~np.isnan(x)]), raw=True).astype(np.float64)
            feats[f'higuchi_{w}'] = roller_complex.apply(lambda x: calc_higuchi_fd(x[~np.isnan(x)]), raw=True).astype(np.float64)
            feats[f'cid_{w}'] = roller_complex.apply(lambda x: calc_cid_ce(x[~np.isnan(x)]), raw=True).astype(np.float64)
            feats[f'peaks_{w}'] = roller_complex.apply(lambda x: calc_num_peaks(x[~np.isnan(x)]), raw=True).astype(np.float64)
    else:
        for w in COMPLEX_WINDOWS:
            feats[f'ac1_{w}'] = 0.0
            feats[f'spec_ent_{w}'] = 0.0
            feats[f'perm_ent_{w}'] = 0.0
            feats[f'higuchi_{w}'] = 0.0
            feats[f'cid_{w}'] = 0.0
            feats[f'peaks_{w}'] = 0.0

    # --- G. ドメイン・カレンダー特徴 (2種) ---
    try:
        y_int = y_past.astype('Int64').fillna(0)
        feats['is_odd'] = y_int % 2
        feats['last_digit'] = y_int % 10
    except:
        feats['is_odd'] = 0
        feats['last_digit'] = 0
    
    # 全てのPython生成特徴量をfloat64に統一
    py_feat_cols = [c for c in feats.columns]
    feats[py_feat_cols] = feats[py_feat_cols].astype(np.float64)

    return feats


def process_group(key, group):
    """並列処理用のラッパー関数"""
    
    # 1. Pythonで計算する複雑な特徴量を生成
    df_py_feats = generate_python_features(group, y_col='y_past_for_python')
    
    # 2. SQLで計算された特徴量とキーを抽出
    # y_transformed, y_past_for_python は含めない
    key_and_sql_feats_cols = [c for c in group.columns if c not in ['y_transformed', 'y_past_for_python']]
    df_sql_and_keys = group[key_and_sql_feats_cols].reset_index(drop=True)

    # 3. 全特徴量を結合
    # df_py_feats は元のインデックスを持つが、df_sql_and_keys は reset_index(drop=True) されているため、
    # 結合する前に df_py_feats もリセットする必要がある
    df_py_feats = df_py_feats.reset_index(drop=True)
    df_all_feats = pd.concat([df_sql_and_keys, df_py_feats], axis=1)
    
    # 4. 最終結果を返す（列の順番と型は main で処理）
    return df_all_feats

# ==========================================
# 4. メイン処理・DB保存
# ==========================================
def save_to_db(df, engine, schema, table_name):
    """DataFrameをDBに保存し、インデックスを作成"""
    print(f"Saving {len(df)} rows to {schema}.{table_name}...")
    
    with engine.begin() as conn:
        # このテーブルはロングフォーマット（4キー）
        df.to_sql(table_name, conn, schema=schema, if_exists='replace', index=False,
                  dtype={
                      'loto': sqlalchemy.types.Text,
                      'unique_id': sqlalchemy.types.Text,
                      'ds': sqlalchemy.types.Date,
                      'ts_type': sqlalchemy.types.Text, # ts_type がキーに含まれる
                      'y': sqlalchemy.types.Float,      # 新しいターゲット変数 y
                  })
        
        # ★ 修正箇所: 最終テーブルへのユニークインデックスを (loto, unique_id, ds, ts_type) の4キー
        conn.execute(text(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_unique ON {schema}.{table_name} (loto, unique_id, ds, ts_type);"))
        # 検索用インデックス
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_loto_ds ON {schema}.{table_name} (loto, ds);"))
    print("DB Save Completed.")

def main():
    parser = argparse.ArgumentParser(description="Create Historical Exogenous Features from all y_transformed series.")
    parser.add_argument("--input-table", default=DEFAULT_INPUT_TABLE, help=f"Input table name (default: {DEFAULT_INPUT_TABLE})")
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE, help=f"Output table name (default: {DEFAULT_OUTPUT_TABLE})")
    parser.add_argument("--base-table", default=DEFAULT_BASE_TABLE, help=f"Base table for original y (default: {DEFAULT_BASE_TABLE})")
    parser.add_argument("--schema", default=os.environ.get("PG_SCHEMA", "public"), help="Database schema (default: public)")
    
    args = parser.parse_args()

    start_time = time.time()
    engine = get_db_engine()
    
    # 1. SQLで特徴量を計算しながら loto_y_ts からデータ取得
    df_y_ts = fetch_y_transformed_data_with_sql_features(engine, args.schema, args.input_table)
    if df_y_ts.empty:
        print(f"Data not found or SQL feature generation failed for {args.schema}.{args.input_table}.")
        return 1

    # 2. loto_baseからキーとy_rawを取得 (今回は使用しない)
    
    # loto, unique_id, ds, ts_type ごとにグループ化
    groups = list(df_y_ts.groupby(['loto', 'unique_id', 'ts_type'], sort=False))
    total_groups = len(groups)
    print(f"Processing {total_groups} series using {N_JOBS} cores for complex feature computation.")

    # 3. Pythonで複雑な特徴量のみを並列実行
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_group)(key, group.sort_values('ds').reset_index(drop=True))
        for key, group in tqdm(groups, total=total_groups, desc="Generating Features (Parallel)")
    )
    
    # 4. 全結合
    # df_final は (loto, unique_id, ds, ts_type) と y、全特徴量を含む
    df_final = pd.concat(results, axis=0).reset_index(drop=True) 
    
    # 5. 欠損値処理 (キーと y 以外を0埋め)
    key_y_cols = ['loto', 'unique_id', 'ds', 'ts_type', 'y']
    
    # データフレームの列が多いため、欠損値処理の対象列を明示的に取得
    feature_cols = [c for c in df_final.columns if c not in key_y_cols]
    
    # ★ 欠損値処理を実行
    df_final.loc[:, feature_cols] = df_final.loc[:, feature_cols].fillna(0)
    
    print(f"Generated Features: {len(feature_cols)} columns")
    print(f"Total Shape: {df_final.shape}")

    # 6. DB保存
    save_to_db(df_final, engine, args.schema, args.output_table)
    
    elapsed = time.time() - start_time
    print(f"Completed successfully in {elapsed:.1f} seconds.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from scipy.signal import periodogram, find_peaks
from scipy.stats import entropy, iqr, skew, kurtosis

# 外部ライブラリのインポート（環境にない場合はスキップ）
try:
    import antropy as ant
except ImportError:
    ant = None
    print("WARNING: 'antropy' library not found. Complexity features will be 0.")

# ==========================================
# 1. 設定・定数（デフォルト値）
# ==========================================
DEFAULT_INPUT_TABLE = "loto_base"
DEFAULT_OUTPUT_TABLE = "loto_hist_exog"

# 特徴量生成パラメータ（100種以上生成するための設定）
# 窓幅：短期(1週)、中期(2週, 1ヶ月)、長期(2ヶ月, 3ヶ月, 半年)
ROLLING_WINDOWS = [7, 14, 28, 56, 91, 182] 
# ラグ：直近、1週間前、...、1年前
LAGS = [1, 2, 3, 7, 14, 28, 56, 91, 364]
# 指数平滑化の減衰係数
EWM_ALPHAS = [0.1, 0.3, 0.5]
# 計算コストの高い特徴量を計算する窓（長すぎると遅くなるため限定）
COMPLEX_WINDOWS = [28, 56, 91]

# ==========================================
# 2. DB接続・データ取得
# ==========================================
def get_db_engine():
    # 環境変数から接続情報を取得
    host = os.environ.get("PGHOST", "127.0.0.1")
    port = os.environ.get("PGPORT", "5432")
    dbname = os.environ.get("PGDATABASE", "loto_db")
    user = os.environ.get("PGUSER", "loto_user")
    password = os.environ.get("PGPASSWORD", "z")
    
    url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url)

def fetch_base_data(engine, schema, table_name):
    """
    指定された入力テーブルからデータを取得し、重複排除を行う
    """
    print(f"Fetching data from {schema}.{table_name}...")
    
    # updated_ts があれば最新を優先するために取得
    query = f"""
    SELECT loto, unique_id, ds, y, updated_ts
    FROM {schema}.{table_name}
    """
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        # updated_tsがない場合のフォールバック
        print(f"Warning: {e}. Trying without updated_ts...")
        query_simple = f"SELECT loto, unique_id, ds, y FROM {schema}.{table_name}"
        df = pd.read_sql(query_simple, engine)
        df['updated_ts'] = pd.Timestamp.now()

    if df.empty:
        return df

    df['ds'] = pd.to_datetime(df['ds'])
    df['updated_ts'] = pd.to_datetime(df['updated_ts'])
    
    # --- 重複排除 (Deduplication) ---
    # loto, unique_id, ds が同じなら、updated_ts が新しい方を残す
    before_len = len(df)
    df = df.sort_values('updated_ts').drop_duplicates(subset=['loto', 'unique_id', 'ds'], keep='last')
    after_len = len(df)
    
    if before_len != after_len:
        print(f"Deduplication: Dropped {before_len - after_len} duplicate rows.")
    
    # 処理用にソート
    df = df.sort_values(['loto', 'unique_id', 'ds']).reset_index(drop=True)
    return df[['loto', 'unique_id', 'ds', 'y']] # 必要な列のみ返す

# ==========================================
# 3. 特徴量計算ロジック
# ==========================================

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

def generate_features_for_series(group):
    """
    1つの時系列(group)に対する特徴量生成
    ※ 前提: groupは既に ds順でソート済み
    """
    # 【最重要】リーク防止: yを1つずらした系列(t-1以前)のみを使用する
    y_past = group['y'].shift(1)
    
    feats = pd.DataFrame(index=group.index)
    
    # --- A. ラグ特徴 (9種) ---
    for lag in LAGS:
        feats[f'lag_{lag}'] = y_past.shift(lag - 1)
    
    # --- B. 基礎統計ローリング (6窓 * 11種 = 66種) ---
    for w in ROLLING_WINDOWS:
        roller = y_past.rolling(window=w, min_periods=max(1, w//2))
        
        feats[f'roll_mean_{w}'] = roller.mean()
        feats[f'roll_std_{w}'] = roller.std()
        feats[f'roll_max_{w}'] = roller.max()
        feats[f'roll_min_{w}'] = roller.min()
        feats[f'roll_median_{w}'] = roller.median()
        feats[f'roll_range_{w}'] = feats[f'roll_max_{w}'] - feats[f'roll_min_{w}']
        feats[f'roll_sum_{w}'] = roller.sum()
        feats[f'roll_skew_{w}'] = roller.skew()
        feats[f'roll_kurt_{w}'] = roller.kurt()
        feats[f'roll_q10_{w}'] = roller.quantile(0.1)
        feats[f'roll_q90_{w}'] = roller.quantile(0.9)

    # --- C. 変化率・差分 (4種) ---
    for k in [1, 7]:
        feats[f'diff_{k}'] = y_past.diff(k)
        feats[f'pct_change_{k}'] = y_past.pct_change(k).replace([np.inf, -np.inf], 0)

    # --- D. 指数平滑化 (3 alphas * 2 stats = 6種) ---
    for alpha in EWM_ALPHAS:
        ewm = y_past.ewm(alpha=alpha, adjust=False)
        feats[f'ewm_mean_{alpha}'] = ewm.mean()
        feats[f'ewm_std_{alpha}'] = ewm.std()

    # --- E. 拡大窓統計 (2種) ---
    feats['expand_mean'] = y_past.expanding().mean()
    feats['expand_std'] = y_past.expanding().std()

    # --- F. 異常検知スコア的指標 (Robust Z-Score) (6種) ---
    for w in ROLLING_WINDOWS:
        med = feats[f'roll_median_{w}']
        iqr_val = feats[f'roll_q90_{w}'] - feats[f'roll_q10_{w}']
        feats[f'rob_z_{w}'] = (y_past - med) / (iqr_val + 1e-6)

    # --- G. 複雑性・周波数・スペクトル特徴 (3窓 * 6種 = 18種) ---
    if ant is not None:
        for w in COMPLEX_WINDOWS:
            feats[f'ac1_{w}'] = y_past.rolling(w).apply(lambda x: calc_autocorr(x, 1), raw=True)
            feats[f'spec_ent_{w}'] = y_past.rolling(w).apply(calc_spectral_entropy, raw=True)
            feats[f'perm_ent_{w}'] = y_past.rolling(w).apply(calc_permutation_entropy, raw=True)
            feats[f'higuchi_{w}'] = y_past.rolling(w).apply(calc_higuchi_fd, raw=True)
            feats[f'cid_{w}'] = y_past.rolling(w).apply(calc_cid_ce, raw=True)
            feats[f'peaks_{w}'] = y_past.rolling(w).apply(calc_num_peaks, raw=True)

    # --- H. ドメイン・カレンダー特徴 (2種) ---
    feats['is_odd'] = y_past % 2
    feats['last_digit'] = y_past % 10

    return feats

# ==========================================
# 4. メイン処理・DB保存
# ==========================================
def save_to_db(df, engine, schema, table_name):
    print(f"Saving {len(df)} rows to {schema}.{table_name}...")
    
    with engine.begin() as conn:
        df.to_sql(table_name, conn, schema=schema, if_exists='replace', index=False,
                  dtype={
                      'loto': sqlalchemy.types.Text,
                      'unique_id': sqlalchemy.types.Text,
                      'ds': sqlalchemy.types.Date,
                      'y': sqlalchemy.types.BigInteger
                  })
        
        # 一意性インデックス
        conn.execute(text(f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_unique 
            ON {schema}.{table_name} (loto, unique_id, ds);
        """))
        # 検索用インデックス
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_loto_ds 
            ON {schema}.{table_name} (loto, ds);
        """))

def main():
    # 引数解析
    parser = argparse.ArgumentParser(description="Create Historical Exogenous Features")
    parser.add_argument("--input-table", default=DEFAULT_INPUT_TABLE, help=f"Input table name (default: {DEFAULT_INPUT_TABLE})")
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE, help=f"Output table name (default: {DEFAULT_OUTPUT_TABLE})")
    parser.add_argument("--schema", default=os.environ.get("PG_SCHEMA", "public"), help="Database schema (default: public)")
    
    args = parser.parse_args()

    start_time = time.time()
    engine = get_db_engine()
    
    # 1. データ取得
    df_base = fetch_base_data(engine, args.schema, args.input_table)
    if df_base.empty:
        print(f"Data not found in {args.schema}.{args.input_table}.")
        return

    results = []
    # loto, unique_id ごとにグループ化
    groups = df_base.groupby(['loto', 'unique_id'], sort=False)
    total_groups = len(groups)
    print(f"Processing {total_groups} series...")
    
    count = 0
    for (loto, uid), group in groups:
        # 日付順ソート
        group = group.sort_values('ds').reset_index(drop=True)
        
        # 2. 特徴量生成
        df_feats = generate_features_for_series(group)
        
        # 結合
        df_merged = pd.concat([group[['loto', 'unique_id', 'ds', 'y']], df_feats], axis=1)
        results.append(df_merged)
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total_groups} series...")

    # 全結合
    df_final = pd.concat(results, axis=0)
    
    # 3. 欠損値処理 (キー以外を0埋め)
    key_cols = ['loto', 'unique_id', 'ds', 'y']
    feature_cols = [c for c in df_final.columns if c not in key_cols]
    df_final[feature_cols] = df_final[feature_cols].fillna(0)
    
    print(f"Generated Features: {len(feature_cols)} columns")
    print(f"Total Shape: {df_final.shape}")

    # 4. DB保存
    save_to_db(df_final, engine, args.schema, args.output_table)
    
    elapsed = time.time() - start_time
    print(f"Completed successfully in {elapsed:.1f} seconds.")

if __name__ == "__main__":
    main()
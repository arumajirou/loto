import os
import sys
import pandas as pd
import numpy as np
import csv
from io import StringIO
from sqlalchemy import create_engine, text
import warnings
from functools import partial

# 数値計算上の警告（0割りなど）を無視
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# ライブラリのインポート
# ==========================================
try:
    from tsflex.features import FeatureCollection, MultipleFeatureDescriptors, FuncWrapper
    from tsfresh.feature_extraction import feature_calculators as fc
    import scipy.stats as ss
except ImportError as e:
    print(f"エラー: 必要なライブラリが見つかりません。詳細: {e}")
    sys.exit(1)

# ==========================================
# DB接続設定
# ==========================================
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASS = os.getenv("PGPASSWORD", "z")
PG_HOST = os.getenv("PGHOST", "127.0.0.1")
PG_PORT = os.getenv("PGPORT", "5432")
PG_DB   = os.getenv("PGDATABASE", "loto_db")

DB_URL = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
engine = create_engine(DB_URL)

# ==========================================
# 関数定義
# ==========================================

def load_data():
    """loto_baseからデータを読み込む"""
    print("Loading data from loto_base...")
    query = """
    SELECT loto, unique_id, ts_type, ds, y 
    FROM public.loto_base 
    WHERE y IS NOT NULL
    ORDER BY loto, unique_id, ts_type, ds
    """
    df = pd.read_sql(query, engine)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

def slope(x):
    """線形回帰の傾きを計算する軽量関数"""
    if len(x) < 2: return 0.0
    return np.polyfit(np.arange(len(x)), x, 1)[0]

def define_features_maximized():
    """
    特徴量を最大化するための設定定義
    """
    # ---------------------------------------------------------
    # 1. 基本統計量 (引数なし)
    # ---------------------------------------------------------
    basic_funcs = [
        np.min, np.max, np.mean, np.median, np.std, np.sum, np.var,
        ss.skew, ss.kurtosis,
        fc.abs_energy,
        fc.mean_abs_change,
        fc.mean_change,
        fc.mean_second_derivative_central,
        fc.variance_larger_than_standard_deviation,
        fc.count_above_mean,
        fc.count_below_mean,
        fc.first_location_of_maximum,
        fc.first_location_of_minimum,
        fc.last_location_of_maximum,
        fc.last_location_of_minimum,
        fc.longest_strike_above_mean,
        fc.longest_strike_below_mean,
        # fc.num_peaks は削除 (引数が必要なため下記へ移動)
        slope, # トレンドの傾き (カスタム関数)
    ]

    # ---------------------------------------------------------
    # 2. 引数付き関数 (分位点、ピーク数など)
    # ---------------------------------------------------------
    
    # Number of Peaks (ピーク数): 幅 n=1, 3, 5 で計算
    # 修正: fc.number_peaks を使用し、引数 n を指定
    for n in [1, 3, 5]:
        f = partial(fc.number_peaks, n=n)
        f.__name__ = f"number_peaks_n{n}"
        basic_funcs.append(f)

    # Quantiles (分位点)
    quantiles = [0.1, 0.25, 0.75, 0.9]
    for q in quantiles:
        f = partial(np.quantile, q=q)
        f.__name__ = f"quantile_{str(q).replace('.', '')}"
        basic_funcs.append(f)

    # Ratio beyond r sigma (異常値の割合)
    for r in [1, 2]:
        f = partial(fc.ratio_beyond_r_sigma, r=r)
        f.__name__ = f"ratio_beyond_{r}_sigma"
        basic_funcs.append(f)

    # CID CE (Complexity)
    f_cid = partial(fc.cid_ce, normalize=True)
    f_cid.__name__ = "cid_ce_norm"
    basic_funcs.append(f_cid)

    # ---------------------------------------------------------
    # 3. ウィンドウ設定の拡充
    # ---------------------------------------------------------
    # 短期(3)から長期(50)まで幅広く設定
    windows = [3, 5, 10, 20, 50] 
    
    # ---------------------------------------------------------
    # 4. 設定の構築
    # ---------------------------------------------------------
    fc_config = FeatureCollection(
        MultipleFeatureDescriptors(
            functions=basic_funcs,
            series_names=["y"],   
            windows=windows,      
            strides=1,            
        )
    )
    return fc_config

def process_group(group_df, fc_config):
    """並列処理用: 1グループの特徴量計算"""
    # 時系列インデックスの設定
    group_df = group_df.set_index('ds').sort_index()
    
    # 特徴量計算 (エラーハンドリング付き)
    try:
        df_feats = fc_config.calculate(group_df, return_df=True, show_progress=False)
    except Exception:
        # 計算不能なデータ（行数が極端に少ないなど）の場合は空を返す
        return pd.DataFrame()
    
    # インデックス(ds)をカラムに戻す
    df_feats = df_feats.reset_index()
    
    # メタデータ付与
    df_feats['loto'] = group_df['loto'].iloc[0]
    df_feats['unique_id'] = group_df['unique_id'].iloc[0]
    df_feats['ts_type'] = group_df['ts_type'].iloc[0]
    
    return df_feats

def main_process(df_base):
    print("Configuring MAXIMIZED features...")
    fc_config = define_features_maximized()
    
    groups = [g for _, g in df_base.groupby(['loto', 'unique_id', 'ts_type'])]
    total_groups = len(groups)
    
    if not groups:
        print("警告: データが空です。")
        return pd.DataFrame()

    from joblib import Parallel, delayed
    from tqdm import tqdm
    
    n_jobs = os.cpu_count()
    print(f"Using {n_jobs} cores for processing {total_groups} groups.")
    print("Starting parallel execution (This may take longer due to increased features)...")

    # ジェネレータとして結果を受け取る
    parallel_runner = Parallel(n_jobs=n_jobs, return_as="generator")

    process_generator = parallel_runner(
        delayed(process_group)(group, fc_config) 
        for group in groups
    )
    
    results = []
    
    for res in tqdm(process_generator, total=total_groups, desc="Overall Progress", unit="group"):
        if not res.empty:
            results.append(res)
    
    if not results:
        print("No features generated.")
        return pd.DataFrame()

    print("Concatenating results...")
    df_final = pd.concat(results, axis=0)
    return df_final

def psql_insert_copy(table, conn, keys, data_iter):
    """高速書き込み用ヘルパー (COPYコマンド)"""
    s_buf = StringIO()
    writer = csv.writer(s_buf)
    writer.writerows(data_iter)
    s_buf.seek(0)

    raw_conn = conn.connection
    with raw_conn.cursor() as cur:
        if table.schema:
            table_name = f"{table.schema}.{table.name}"
        else:
            table_name = table.name
            
        columns = ', '.join([f'"{k}"' for k in keys])
        sql = f"COPY {table_name} ({columns}) FROM STDIN WITH CSV"
        try:
            cur.copy_expert(sql=sql, file=s_buf)
        except Exception as e:
            print(f"COPY error: {e}")
            raise

def rename_and_save(df_features):
    """保存処理"""
    
    if df_features.empty:
        print("保存するデータがありません。")
        return

    # カラム名のリネーム
    new_cols = {}
    for col in df_features.columns:
        if col in ['ds', 'loto', 'unique_id', 'ts_type']:
            continue
        # 記号置換
        clean_col = col.replace("__", "_").replace("=", "").replace("w", "w_")
        new_cols[col] = f"hist_{clean_col}"
        
    df_features = df_features.rename(columns=new_cols)
    df_features = df_features.fillna(0)
    
    # メモリ節約のため型変換
    float_cols = df_features.select_dtypes(include=['float64']).columns
    df_features[float_cols] = df_features[float_cols].astype('float32')
    
    print(f"Saving {len(df_features)} rows and {len(df_features.columns)} columns to 'loto_tsfresh'...")
    
    try:
        df_features.to_sql(
            'loto_tsfresh', 
            engine, 
            if_exists='replace', 
            index=False, 
            method=psql_insert_copy,
            chunksize=10000 
        )
        print("Data saved successfully via COPY command.")
    except Exception as e:
        print(f"保存中にエラーが発生しました: {e}")
        return

    # インデックス作成
    with engine.connect() as conn:
        print("Creating index...")
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tsfresh_keys ON loto_tsfresh (loto, unique_id, ts_type, ds)"))
    
    print("Done.")

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    df_base = load_data()
    df_feats = main_process(df_base)
    rename_and_save(df_feats)
import os
import sys
import pandas as pd
import numpy as np
import csv
from io import StringIO
from sqlalchemy import create_engine, text
import warnings

# 数値計算上の警告（0割りなど）を無視してログをクリーンにする
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================
# ライブラリのインポート
# ==========================================
try:
    from tsflex.features import FeatureCollection, MultipleFeatureDescriptors
    # tsfreshの特徴量計算関数を再利用
    from tsfresh.feature_extraction import feature_calculators as fc
    import scipy.stats as ss
    from functools import partial
except ImportError as e:
    print(f"エラー: 必要なライブラリが見つかりません。詳細: {e}")
    print("pip install tsflex scipy tsfresh を実行してください。")
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

def define_features():
    """
    tsflexの特徴量設定定義
    """
    # 1. 基本的な統計関数
    basic_funcs = [
        np.min, np.max, np.mean, np.median, np.std, 
        ss.skew, ss.kurtosis,          
        fc.abs_energy,                 
        fc.sample_entropy,             
        fc.mean_abs_change,            
        fc.mean_change                 
    ]

    # 2. 引数が必要な関数 (cid_ce) の設定
    # normalize=True を固定した関数を作成
    cid_ce_normalized = partial(fc.cid_ce, normalize=True)
    cid_ce_normalized.__name__ = "cid_ce_norm" 

    basic_funcs.append(cid_ce_normalized)

    # 3. ウィンドウ設定 (過去5回, 10回, 25回)
    windows = [5, 10, 25] 
    
    # 4. 設定の構築
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
    """
    1つのグループ (loto, unique_id, ts_type) に対して特徴量を計算
    """
    # 時系列インデックスの設定
    group_df = group_df.set_index('ds').sort_index()
    
    # tsflexで特徴量計算
    df_feats = fc_config.calculate(group_df, return_df=True, show_progress=False)
    
    # インデックス(ds)をカラムに戻す
    df_feats = df_feats.reset_index()
    
    # メタデータ(loto, unique_id等)を付与
    df_feats['loto'] = group_df['loto'].iloc[0]
    df_feats['unique_id'] = group_df['unique_id'].iloc[0]
    df_feats['ts_type'] = group_df['ts_type'].iloc[0]
    
    return df_feats

def main_process(df_base):
    print("Configuring features...")
    fc_config = define_features()
    
    # データをグループに分割
    groups = [g for _, g in df_base.groupby(['loto', 'unique_id', 'ts_type'])]
    total_groups = len(groups)
    
    if not groups:
        print("警告: データが空です。")
        return pd.DataFrame()

    # Joblibによる並列実行
    from joblib import Parallel, delayed
    from tqdm import tqdm
    
    n_jobs = os.cpu_count()
    print(f"Using {n_jobs} cores for processing {total_groups} groups.")
    print("Starting parallel execution...")

    # return_as="generator" で完了したものから順次受け取る
    parallel_runner = Parallel(n_jobs=n_jobs, return_as="generator")

    process_generator = parallel_runner(
        delayed(process_group)(group, fc_config) 
        for group in groups
    )
    
    results = []
    
    # プログレスバーで全体の進捗を表示
    for res in tqdm(process_generator, total=total_groups, desc="Overall Progress", unit="group"):
        results.append(res)
    
    if not results:
        print("No features generated.")
        return pd.DataFrame()

    print("Concatenating results...")
    df_final = pd.concat(results, axis=0)
    return df_final

def psql_insert_copy(table, conn, keys, data_iter):
    """
    Pandas to_sql の method 引数に渡す高速書き込み関数
    PostgreSQLの COPY FROM STDIN を使用
    """
    # データをCSV形式のメモリバッファに書き込む
    s_buf = StringIO()
    writer = csv.writer(s_buf)
    writer.writerows(data_iter)
    s_buf.seek(0)

    # raw_connectionを取得してカーソルを作成
    raw_conn = conn.connection
    with raw_conn.cursor() as cur:
        # スキーマ名が含まれているか確認
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
    """カラム名を整形してDBに保存 (高速COPY版)"""
    
    if df_features.empty:
        print("保存するデータがありません。")
        return

    # カラム名のリネーム: "y__min__w=5" -> "hist_y_min_w5"
    new_cols = {}
    for col in df_features.columns:
        if col in ['ds', 'loto', 'unique_id', 'ts_type']:
            continue
        
        clean_col = col.replace("__", "_").replace("=", "").replace("w", "w_")
        new_cols[col] = f"hist_{clean_col}"
        
    df_features = df_features.rename(columns=new_cols)
    
    # NaNを0埋め
    df_features = df_features.fillna(0)
    
    print(f"Saving {len(df_features)} rows to 'loto_tsfresh' using COPY method...")
    
    try:
        # DBへ保存 (methodに高速化関数を指定)
        df_features.to_sql(
            'loto_tsfresh', 
            engine, 
            if_exists='replace', 
            index=False, 
            method=psql_insert_copy,  # 高速書き込みを使用
            chunksize=10000 
        )
        print("Data saved via COPY command.")
    except Exception as e:
        print(f"保存中にエラーが発生しました: {e}")
        return

    # インデックス作成 (検索高速化)
    with engine.connect() as conn:
        print("Creating index...")
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tsfresh_keys ON loto_tsfresh (loto, unique_id, ts_type, ds)"))
    
    print("Done.")

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":
    # 1. データロード
    df_base = load_data()
    
    # 2. 特徴量生成 (tsflexで高速実行)
    df_feats = main_process(df_base)
    
    # 3. 保存 (高速COPY)
    rename_and_save(df_feats)
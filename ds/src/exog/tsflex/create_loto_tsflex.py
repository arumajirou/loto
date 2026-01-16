import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# tsflex 関連のインポート
try:
    from tsflex.features import FeatureCollection, MultipleFeatureDescriptors
    from tsflex.features.integrations import tsfresh_settings_wrapper
    from tsflex.processing import SeriesProcessor
    import scipy.stats as ss
    # tsfreshの計算関数の一部を借用（インストール済み前提）
    from tsfresh.feature_extraction import feature_calculators as fc
except ImportError:
    print("エラー: tsflex がインストールされていません。'pip install tsflex' を実行してください。")
    sys.exit(1)

# DB接続設定
PG_USER = os.getenv("PGUSER", "postgres")
PG_PASS = os.getenv("PGPASSWORD", "z")
PG_HOST = os.getenv("PGHOST", "127.0.0.1")
PG_PORT = os.getenv("PGPORT", "5432")
PG_DB   = os.getenv("PGDATABASE", "loto_db")

DB_URL = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
engine = create_engine(DB_URL)

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
    # 基本的な統計量関数リスト
    basic_funcs = [
        np.min, np.max, np.mean, np.median, np.std, 
        ss.skew, ss.kurtosis,
        fc.abs_energy,
        fc.cid_ce,
        fc.sample_entropy
    ]

    # ウィンドウ設定 (過去5回, 10回, 20回)
    windows = [5, 10, 20]
    
    # 特徴量コレクションの作成
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
    # tsflexはindexが時間であることを期待します
    group_df = group_df.set_index('ds').sort_index()
    
    # 計算実行
    df_feats = fc_config.calculate(group_df, return_df=True, show_progress=False)
    
    # インデックス(ds)をカラムに戻す
    df_feats = df_feats.reset_index()
    
    # メタデータを付与
    df_feats['loto'] = group_df['loto'].iloc[0]
    df_feats['unique_id'] = group_df['unique_id'].iloc[0]
    df_feats['ts_type'] = group_df['ts_type'].iloc[0]
    
    return df_feats

def main_process(df_base):
    print("Configuring features...")
    fc_config = define_features()
    
    print("Processing groups (Parallel execution)...")
    
    # グループ化
    groups = [g for _, g in df_base.groupby(['loto', 'unique_id', 'ts_type'])]
    
    # Joblibによる並列化
    from joblib import Parallel, delayed
    from tqdm import tqdm
    
    results = Parallel(n_jobs=os.cpu_count())(
        delayed(process_group)(group, fc_config) 
        for group in tqdm(groups, desc="Extracting features")
    )
    
    if not results:
        print("No features generated.")
        return pd.DataFrame()

    df_final = pd.concat(results, axis=0)
    return df_final

def rename_and_save(df_features):
    """カラム名を整形して保存"""
    
    # tsflexのカラム名 "y__min__w=5" -> "hist_y_min_w5" に変換
    new_cols = {}
    for col in df_features.columns:
        if col in ['ds', 'loto', 'unique_id', 'ts_type']:
            continue
        clean_col = col.replace("__", "_").replace("=", "").replace("w", "w_")
        new_cols[col] = f"hist_{clean_col}"
        
    df_features = df_features.rename(columns=new_cols)
    df_features = df_features.fillna(0)
    
    print(f"Saving {len(df_features)} rows to 'loto_tsflex'...")
    
    # 保存
    df_features.to_sql(
        'loto_tsflex', 
        engine, 
        if_exists='replace', 
        index=False, 
        method='multi', 
        chunksize=5000
    )
    
    # インデックス作成
    with engine.connect() as conn:
        print("Creating index...")
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tsflex_keys ON loto_tsflex (loto, unique_id, ts_type, ds)"))
    
    print("Done.")

if __name__ == "__main__":
    df = load_data()
    df_feats = main_process(df)
    rename_and_save(df_feats)
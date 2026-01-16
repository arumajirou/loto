import argparse
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import sqlalchemy
import logging

# ==========================================
# ロガー設定
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# DB接続・ユーティリティ関数
# ==========================================
def get_db_engine(args: argparse.Namespace) -> Engine:
    """PostgreSQLエンジン取得（引数または環境変数）"""
    host = args.host or os.getenv("PGHOST", "127.0.0.1")
    port = args.port or os.getenv("PGPORT", "5432")
    db = args.db or os.getenv("PGDATABASE", "loto_db")
    user = args.user or os.getenv("PGUSER", "loto_user")
    password = args.password or os.getenv("PGPASSWORD", "z")

    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    try:
        engine = create_engine(url)
        logger.info(f"PostgreSQL接続成功: db={db} host={host}:{port}")
        return engine
    except Exception as e:
        logger.error(f"PostgreSQL接続エラー: {e}")
        raise

def fetch_base_data(engine: Engine, schema: str, table_name: str) -> pd.DataFrame:
    """loto_baseからデータを取得"""
    query = f"""
    SELECT loto, unique_id, ds, y
    FROM {schema}.{table_name}
    ORDER BY loto, unique_id, ds
    """
    try:
        df = pd.read_sql(query, engine)
        df['ds'] = pd.to_datetime(df['ds']).dt.date
        df['y'] = pd.to_numeric(df['y'], errors='coerce').astype('float64')
        return df.dropna(subset=['y'])
    except Exception as e:
        logger.error(f"テーブル {schema}.{table_name} の読み込みエラー: {e}")
        return pd.DataFrame()


def transform_and_melt(df: pd.DataFrame, transforms: Dict[str, str]) -> pd.DataFrame:
    """
    グループごとに変換を適用し、結果を縦持ち (Long Format) に変換
    """
    results_list: List[pd.DataFrame] = []
    groups = df.groupby(['loto', 'unique_id'], sort=False)

    for (loto, uid), group in groups:
        group = group.sort_values('ds').copy()
        
        # 変換済みデータを格納する辞書 (キーはts_type名)
        transformed_data: Dict[str, pd.Series] = {}
        y_raw = group['y']

        # --- 変換ロジック ---
        for ts_type, logic in transforms.items():
            # ... (変換ロジックは変更なし) ...
            if logic == "raw":
                y_new = y_raw
            elif logic == "cumsum":
                y_new = y_raw.cumsum()
            elif logic.startswith("rolling"):
                try:
                    parts = logic.split('.')
                    window_str = parts[0].replace('rolling(', '').replace(')', '')
                    window = int(window_str)
                    agg_func = parts[1].lower().replace('()', '')
                    roller = y_raw.rolling(window=window, min_periods=1)
                    
                    if agg_func == 'sum':
                        y_new = roller.sum()
                    elif agg_func == 'mean':
                        y_new = roller.mean()
                    else:
                        logger.warning(f"不明な集計関数: {agg_func}")
                        continue
                except Exception as e:
                    logger.error(f"Rolling計算エラー ({ts_type}): {e}")
                    continue
            elif logic.startswith("diff"):
                try:
                    lag = int(logic.replace('diff(', '').replace(')', ''))
                    y_new = y_raw.diff(lag)
                except Exception:
                    continue
            elif logic == "is_odd":
                y_new = y_raw % 2
            else:
                continue

            transformed_data[ts_type] = y_new
        # --- 変換ロジックここまで ---
            
        # --- FIX START ---
        # 1. キー列と生データを格納する辞書を作成
        data_to_combine = {
            'loto': loto, 
            'unique_id': uid, 
            'ds': group['ds'], 
            'y_raw': y_raw
        }
        
        # 2. 変換済みデータを新しい列として追加
        data_to_combine.update(transformed_data)

        # 3. 全てのデータを含むDataFrameを作成
        df_to_melt = pd.DataFrame(data_to_combine)

        # 4. 縦持ち (Melt) に変換 (df_to_meltを使用)
        df_melt = pd.melt(
            df_to_melt,
            id_vars=['loto', 'unique_id', 'ds', 'y_raw'],
            value_vars=list(transformed_data.keys()),
            var_name='ts_type',
            value_name='y_transformed'
        )
        # --- FIX END ---
        
        results_list.append(df_melt)

    return pd.concat(results_list, ignore_index=True)
    
def main():
    ap = argparse.ArgumentParser(description="Generate Transformed Time Series (y) for Exogenous Feature Creation.")
    
    # 接続設定
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--user", default=None)
    ap.add_argument("--password", default=None)
    ap.add_argument("--schema", default="public")
    
    # テーブル設定
    ap.add_argument("--input-table", default="loto_base", help="Input table containing original y values (default: loto_base)")
    ap.add_argument("--output-table", default="loto_y_ts", help="Output table for transformed series (default: loto_y_ts)")
    ap.add_argument("--mode", choices=["replace", "append"], default="replace", help="DB write mode")

    args = ap.parse_args()
    
    t0 = time.time()
    
    # 1. 接続
    engine = get_db_engine(args)
    
    # 2. 変換ロジック定義
    # 新しい加工ロジックを追加する場合は、ここに追加する
    TRANSFORMS: Dict[str, str] = {
        "raw": "raw",                 # オリジナルy
        "cumsum": "cumsum",           # 累積和
        "roll3_sum": "rolling(3).sum()",    # 3期間合計
        "roll7_mean": "rolling(7).mean()",  # 7期間平均
        "diff1": "diff(1)",           # 1階差分
        "is_odd": "is_odd"            # 奇数フラグ
    }
    logger.info(f"定義された変換: {list(TRANSFORMS.keys())}")
    
    # 3. データ取得
    df_base = fetch_base_data(engine, args.schema, args.input_table)
    if df_base.empty:
        logger.error("入力データが空です。処理を中止します。")
        return 1

    # 4. 変換と縦持ち化
    df_transformed = transform_and_melt(df_base, TRANSFORMS)
    
    # 5. DB書き込み
    logger.info(f"最終データフレーム: {df_transformed.shape}. {len(TRANSFORMS)}種類の時系列を生成。")
    
    output_table = args.output_table
    output_schema = args.schema
    
    try:
        # 主キー: (loto, unique_id, ds, ts_type)
        key_cols = ['loto', 'unique_id', 'ds', 'ts_type']
        
        if args.mode == "replace":
            # REPLACEモード: DROPしてCREATE
            df_transformed.to_sql(output_table, engine, schema=output_schema, if_exists='replace', index=False,
                                  dtype={'ds': sqlalchemy.types.Date, 'y_raw': sqlalchemy.types.Float, 'y_transformed': sqlalchemy.types.Float})
            
            with engine.begin() as conn:
                # UNIQUE INDEXの作成
                conn.execute(text(f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_{output_table}_key 
                    ON {output_schema}.{output_table} (loto, unique_id, ds, ts_type);
                """))
            logger.info("テーブルを新規作成し、データを投入しました (REPLACE mode)。")
            
        elif args.mode == "append":
            # APPENDモードはキー重複時の処理が複雑になるため、このスクリプトではREPLACE推奨
            logger.warning("APPEND modeはキー重複エラーを無視できないため、REPLACE modeを推奨します。")
            df_transformed.to_sql(output_table, engine, schema=output_schema, if_exists='append', index=False,
                                  dtype={'ds': sqlalchemy.types.Date, 'y_raw': sqlalchemy.types.Float, 'y_transformed': sqlalchemy.types.Float})
            logger.info("データを追記しました (APPEND mode)。キー重複エラーが発生した場合、トランザクションは中断します。")


    except Exception as e:
        logger.error(f"DB書き込みエラー: {e}")
        return 1
        
    elapsed = time.time() - t0
    logger.info(f"処理完了。所要時間: {elapsed:.2f}秒。")
    return 0

if __name__ == "__main__":
    sys.exit(main())
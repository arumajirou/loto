import pandas as pd
from sqlalchemy import create_engine, text
from . import config
import logging

logger = logging.getLogger(__name__)

def get_source_engine():
    return create_engine(config.DB_URI_SOURCE)

def get_target_engine():
    return create_engine(config.DB_URI_TARGET)

def load_data():
    """
    loto_base からデータを読み込み、dsをdatetime型に変換して返す
    """
    engine = get_source_engine()
    query = f"""
        SELECT loto, ds, unique_id, ts_type, y 
        FROM public.{config.TABLE_SOURCE}
        ORDER BY loto, unique_id, ts_type, ds
    """
    logger.info(f"Loading data from {config.TABLE_SOURCE}...")
    try:
        df = pd.read_sql(query, engine)
        
        # ds の型変換（安全性重視）
        # errors='coerce' で変換できないものは NaT にして後で除外も可能だが、
        # ここではデータ品質監視のため、あえてエラーになれば気づけるようにする
        if df['ds'].dtype == 'object':
            df['ds'] = pd.to_datetime(df['ds'])
            
        logger.info(f"Loaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

def save_data(df, table_name=config.TABLE_TARGET):
    """
    作成した特徴量を model_exog に保存する
    """
    engine = get_target_engine()
    logger.info(f"Saving {len(df)} features to public.{table_name}...")
    
    try:
        # 既存テーブルを置換（運用に合わせて append に変更も可）
        df.to_sql(
            table_name, 
            engine, 
            if_exists='replace', 
            index=False, 
            method='multi',
            chunksize=5000
        )
        
        # 主キー制約の追加（to_sqlでは作成されないため）
        with engine.connect() as conn:
            conn.execute(text(f"""
                ALTER TABLE public.{table_name} 
                ADD PRIMARY KEY (loto, unique_id, ts_type);
            """))
            conn.commit()
            
        logger.info("Save completed successfully.")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise
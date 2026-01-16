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
    loto_base からデータをロードし、最低限の型変換を行う
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
        
        # ds を datetime 型へ (Dartsで必須)
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
    logger.info(f"Saving {len(df)} rows to public.{table_name}...")
    
    try:
        # 主キー重複時は全置換する戦略 (replace)
        # ※データ量が多い場合は to_sql の method='multi' や chunksize を調整
        df.to_sql(
            table_name, 
            engine, 
            if_exists='replace', 
            index=False, 
            chunksize=5000,
            method='multi'
        )
        
        # 主キー制約の追加 (Composite Primary Key)
        with engine.connect() as conn:
            conn.execute(text(f"""
                ALTER TABLE public.{table_name} 
                ADD PRIMARY KEY (loto, unique_id, ts_type, ds);
            """))
            conn.commit()
            
        logger.info("Save completed successfully.")
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise
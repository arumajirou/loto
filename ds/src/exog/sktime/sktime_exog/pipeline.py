import logging
from . import db, transformers, config

# ロギング設定
logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def run():
    """
    sktime ETL実行メイン関数
    """
    try:
        # 1. Load Data
        df_raw = db.load_data()
        
        # 2. Initialize Generator
        generator = transformers.SktimeFeatureGenerator()
        
        # 3. Generate Features (Transform)
        df_features = generator.generate(df_raw)
        
        # 4. Save Data
        if not df_features.empty:
            db.save_data(df_features)
        else:
            logger.warning("Skipping save because feature DataFrame is empty.")
            
    except Exception as e:
        logger.error(f"Pipeline failed abnormally: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    run()
import logging
from . import db, transformers, config

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def run():
    """
    Darts 特徴量生成パイプラインの実行
    """
    try:
        # 1. Load
        df_raw = db.load_data()
        
        # 2. Transform
        generator = transformers.DartsFeatureGenerator()
        df_features = generator.generate(df_raw)
        
        # 3. Save
        if not df_features.empty:
            db.save_data(df_features)
        else:
            logger.warning("No features generated. Skipping save.")
            
    except Exception as e:
        logger.error(f"Darts Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    run()
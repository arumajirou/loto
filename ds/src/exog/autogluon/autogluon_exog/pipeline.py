import logging
from autogluon.common.features.feature_metadata import FeatureMetadata
from . import db, features_hist, config

# ロギング設定
logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def validate_features_with_autogluon(df):
    """
    AutoGluonのFeatureMetadataを使用して型を推論し、
    意図した型（数値型）になっているか検証・ログ出力する。
    """
    logger.info("Validating features with AutoGluon FeatureMetadata...")
    
    # AutoGluonによる型推論
    metadata = FeatureMetadata.from_df(df)
    
    logger.info("--- AutoGluon Detected Types ---")
    logger.info(metadata)
    
    # 属性から型マップを取得（修正済み）
    type_map = metadata.type_map_raw
    warnings = []
    
    for col, dtype in type_map.items():
        if col.startswith(config.PREFIX):
            # 日付系(hist_ds_)は数値チェックから除外してOKとする
            if "ds_" in col:
                continue

            # それ以外は int か float であるべき
            if dtype not in ['float', 'int']:
                warnings.append(f"Column '{col}' detected as '{dtype}', expected numeric.")
    
    if warnings:
        logger.warning("Type Mismatch Warnings:")
        for w in warnings:
            logger.warning(w)
    else:
        logger.info("All hist_ features are correctly detected as numeric types.")
        
    return metadata

def run():
    """
    ETL実行メイン関数
    """
    try:
        # 1. Load
        df_raw = db.load_data()
        
        # 2. Transform (Generate Features)
        logger.info("Generating historical features...")
        df_features = features_hist.generate_hist_features(df_raw)
        
        # 3. Validate (AutoGluon)
        _ = validate_features_with_autogluon(df_features)
        
        # 4. Save
        db.save_data(df_features)
        
    except Exception as e:
        logger.error(f"Pipeline failed abnormally: {e}")
        # 詳細なスタックトレースを表示
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    run()
import os

# データベース設定
DB_URI_SOURCE = os.getenv(
    "DB_URI_SOURCE", 
    "postgresql+psycopg2://postgres:z@localhost:5432/dataset"
)
DB_URI_TARGET = os.getenv(
    "DB_URI_TARGET", 
    "postgresql+psycopg2://postgres:z@localhost:5432/model_exog"
)

# テーブル名
TABLE_SOURCE = "loto_base"
TABLE_TARGET = "sktime"

# --- 特徴量生成設定 ---

# 1. Catch22 (22 features)
USE_CATCH22 = True
PREFIX_CATCH22 = "hist_c22_"

# 2. Summary Statistics (mean, std, min, max, quantiles...)
USE_SUMMARY = True
PREFIX_SUMMARY = "hist_summ_"

# 3. TSFresh (Comprehensive features)
# default_fc_parameters options: 'minimal' (top few), 'efficient' (hundreds), 'comprehensive' (thousands)
# DBの列数制限(Postgresは1600列)に注意。'efficient'以上は列数が多すぎる可能性があります。
USE_TSFRESH = True
TSFRESH_SETTING = "minimal"  # 'minimal' creates ~8-10 features per series.
PREFIX_TSFRESH = "hist_tsf_"

# 4. MiniRocket (Shape features)
# num_kernels: 生成される特徴量数（84の倍数が推奨される）
USE_MINIROCKET = True
MINIROCKET_KERNELS = 84  # 4996 is default, but too large for DB. 84 is fast and light.
PREFIX_MINIROCKET = "hist_mrock_"

# ロギング設定
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
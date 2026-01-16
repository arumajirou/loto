import os

# データベース設定
# ※ 環境に合わせて user/pass を修正してください
# 入力: 時系列データ
DB_URI_SOURCE = os.getenv(
    "DB_URI_SOURCE", 
    "postgresql+psycopg2://postgres:z@localhost:5432/dataset"
)
# 出力: 特徴量ストア
DB_URI_TARGET = os.getenv(
    "DB_URI_TARGET", 
    "postgresql+psycopg2://postgres:z@localhost:5432/model_exog"
)

# テーブル名
TABLE_SOURCE = "loto_base"
TABLE_TARGET = "autogluon"

# 特徴量生成設定
# 窓関数で計算する期間（点数ベース）
WINDOWS = [7, 14, 30]

# カラム接頭辞
PREFIX = "hist_"

# ログ設定
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
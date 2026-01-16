import os

# データベース接続情報
# 【重要】パスワード等は環境変数または適切な値に変更してください
DB_URI_SOURCE = os.getenv(
    "DB_URI_SOURCE", 
    "postgresql+psycopg2://postgres:z@localhost:5432/dataset"
)
DB_URI_TARGET = os.getenv(
    "DB_URI_TARGET", 
    "postgresql+psycopg2://postgres:z@localhost:5432/model_exog"
)

# テーブル定義
TABLE_SOURCE = "loto_base"
TABLE_TARGET = "darts"

# --- 特徴量生成パラメータ ---

# 1. カレンダー特徴量 (Datetime Attributes)
# Pandas DatetimeIndex の属性名
TIME_ATTRS = [
    "year", "month", "day", "dayofweek", "dayofyear", 
    "weekofyear", "quarter", "days_in_month",
    "is_month_start", "is_month_end", "is_quarter_start", "is_quarter_end", "is_year_start", "is_year_end"
]

# 2. ラグ特徴量 (Lags/Shift)
# 何回前の値をとるか
LAGS = [1, 7, 14, 28, 30]

# 3. 移動窓統計 (Rolling Window)
# 窓サイズ
WINDOW_SIZES = [7, 14, 28]
# 統計関数 (pandas.Series method name)
WINDOW_STATS = ["mean", "std", "min", "max", "sum", "var", "skew"]

# 4. 階差 (Differences)
# 周期 (1=前回, 7=前週)
DIFF_PERIODS = [1, 7]

# カラム接頭辞
PREFIX = "hist_dyn_"

# ロギング設定
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
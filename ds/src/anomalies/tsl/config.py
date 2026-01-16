# /mnt/e/env/ts/tslib/ds/src/anomalies/tsl/config.py
import os

# ユーザー環境情報 (Explorerの情報に基づき更新)
DB_USER = "postgres"
DB_PASS = "z"
DB_HOST = "127.0.0.1"
DB_PORT = "5432"

# ソースDB: loto_base テーブルがある場所
SOURCE_DB_NAME = "dataset"

# ターゲットDB: 結果を保存する場所
TARGET_DB_NAME = "anomaly"

# 接続文字列
SOURCE_DB_URI = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{SOURCE_DB_NAME}"
TARGET_DB_URI = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{TARGET_DB_NAME}"

# TSLライブラリのルートパス
TSL_ROOT_PATH = "/mnt/e/env/ts/tslib/model/Time-Series-Library"
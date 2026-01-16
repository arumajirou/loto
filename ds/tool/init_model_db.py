import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
from importlib.metadata import version, PackageNotFoundError

# ---------------------------------------------------------
# 1. 設定とデータ定義
# ---------------------------------------------------------

# DB接続情報 (ユーザー環境変数またはデフォルト値)
DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASS = os.environ.get("DB_PASS", "z")
TARGET_DB_NAME = "model" # 作成するDB名
DEFAULT_DB_NAME = "postgres" # DB作成操作用の接続先

# 特定された時系列予測ライブラリのリスト (名前, カテゴリ, 説明)
# pip listの名前(PyPI名)と一致させます
TARGET_LIBRARIES = [
    # 1. 基盤モデル・生成AI系
    ("timesfm", "Foundation Model", "Google製時系列基盤モデル"),
    ("chronos-forecasting", "Foundation Model", "Amazon製言語モデル技術応用時系列予測"),
    ("lag-llama", "Foundation Model", "LLaMAベースの確率的時系列予測基盤モデル"),
    ("granite-tsfm", "Foundation Model", "IBM製時系列基盤モデル(Tiny Time Mixers等)"),
    ("uni2ts", "Foundation Model", "統一データ形式を用いる基盤モデル(Moirai等)"),
    ("gluonts", "Foundation Model", "Amazon製確率的時系列モデリング(Deep Learning多数)"),
    
    # 2. ディープラーニング特化系
    ("neuralforecast", "Deep Learning", "Nixtla製DLモデルライブラリ(N-BEATS, NHITS等)"),
    ("flood_forecast", "Deep Learning", "PyTorchベース深層学習予測フレームワーク(Flow Forecast)"),
    ("ai4ts", "Deep Learning", "時系列向けAIツールキット"),
    ("neuralprophet", "Deep Learning", "Prophetのニューラルネットワーク拡張版"),
    
    # 3. 統合フレームワーク・AutoML系
    ("darts", "Unified Framework", "統計からDLまで統一APIで扱えるフレームワーク"),
    ("sktime", "Unified Framework", "Scikit-learn互換の時系列分析・予測統合ツールボックス"),
    ("autots", "AutoML", "モデル選択とハイパーパラメータ調整を自動化するAutoML"),
    ("salesforce-merlion", "Unified Framework", "Salesforce製時系列AIライブラリ(異常検知と予測)"),
    ("BasicTS", "Benchmark", "時系列予測のベンチマーク用フレームワーク"),
    
    # 4. 統計・機械学習・計量経済学系
    ("statsmodels", "Statistical", "統計モデリングの基本ライブラリ(ARIMA等)"),
    ("statsforecast", "Statistical", "Nixtla製高速統計モデルライブラリ(AutoARIMA等)"),
    ("mlforecast", "Machine Learning", "Nixtla製MLベース(LGBM等)予測ライブラリ"),
    ("skforecast", "Machine Learning", "Sklearn回帰モデルを時系列予測に転用するラッパー"),
    ("prophet", "Statistical", "Meta製加法モデルに基づく予測ライブラリ"),
    ("pmdarima", "Statistical", "Auto-ARIMA機能を提供するライブラリ"),
    ("hierarchicalforecast", "Statistical", "階層的時系列調整(Reconciliation)ライブラリ"),
    ("arch", "Econometrics", "金融時系列のボラティリティ予測(ARCH/GARCH)"),
    ("nfoursid", "System Identification", "状態空間モデルのシステム同定"),
    
    # 5. その他
    ("pypots", "Imputation", "欠損値のある時系列データの予測・補完特化"),
]

# ---------------------------------------------------------
# 2. データベース作成関数
# ---------------------------------------------------------
def create_database(host, port, user, password, default_db, new_db_name):
    """postgres等のデフォルトDBに接続し、新しいDBを作成する"""
    # default databaseへの接続URL
    url = f"postgresql://{user}:{password}@{host}:{port}/{default_db}"
    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    
    try:
        with engine.connect() as conn:
            # DBの存在確認
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{new_db_name}'"))
            if not result.fetchone():
                print(f"INFO: Creating database '{new_db_name}'...")
                conn.execute(text(f"CREATE DATABASE {new_db_name}"))
                print(f"SUCCESS: Database '{new_db_name}' created.")
            else:
                print(f"INFO: Database '{new_db_name}' already exists.")
    except Exception as e:
        print(f"ERROR: Failed to create database. {e}")
        # すでに存在する場合などでエラーになることがあるが、接続を試みるために続行
    finally:
        engine.dispose()

# ---------------------------------------------------------
# 3. データ処理とテーブル作成・挿入
# ---------------------------------------------------------
def setup_lib_table():
    # 1. DB作成の試行
    create_database(DB_HOST, DB_PORT, DB_USER, DB_PASS, DEFAULT_DB_NAME, TARGET_DB_NAME)

    # 2. 対象DBへの接続エンジン作成
    target_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{TARGET_DB_NAME}"
    engine = create_engine(target_url)

    # 3. データフレームの作成 (現在のインストールバージョンを取得)
    data = []
    for name, category, desc in TARGET_LIBRARIES:
        try:
            # パッケージ名の大文字小文字やハイフン/アンダースコアの揺らぎを吸収してバージョン取得を試みる
            # 基本的にリストの名前を使用
            ver = version(name)
        except PackageNotFoundError:
            ver = "Not Installed"
        
        data.append({
            "library_name": name,
            "category": category,
            "version": ver,
            "description": desc
        })
    
    df = pd.DataFrame(data)

    # 4. テーブル作成とデータ挿入 (replace: 既存なら作り直し)
    # テーブル名: lib
    try:
        print(f"INFO: Inserting {len(df)} records into table 'lib'...")
        df.to_sql('lib', engine, if_exists='replace', index=False)
        
        # プライマリキーなどを設定したい場合はここでALTER TABLEを実行しますが、
        # 今回は簡易的な保存のため、このままとします。
        
        print("SUCCESS: Data insertion complete.")
        
        # 5. 結果の確認表示
        print("\n--- Saved Data in 'model.lib' ---")
        with engine.connect() as conn:
            result_df = pd.read_sql("SELECT * FROM lib", conn)
            print(result_df.to_string(index=False))
            
    except Exception as e:
        print(f"ERROR: Failed to interact with database '{TARGET_DB_NAME}'. {e}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    setup_lib_table()
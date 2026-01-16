import os

class Config:
    """
    設定管理クラス
    """
    # 接続情報
    DB_HOST = os.environ.get("DB_HOST", "127.0.0.1")
    DB_PORT = os.environ.get("DB_PORT", "5432")
    DB_USER = os.environ.get("DB_USER", "postgres")
    # パスワード優先順位: DB_PASS > PGPASSWORD > デフォルト
    DB_PASS = os.environ.get("DB_PASS", os.environ.get("PGPASSWORD", "password"))

    # --- データベース名の定義 ---
    
    # 1. 管理・設定用DB (hf_snapshots, foundation_modelテーブル)
    METADATA_DB = "foundation_model"
    
    # 2. 結果保存用DB (特徴量テーブル)
    FEATURE_DB = "model_exog"
    
    # 3. リソースログ用DB
    RESOURCE_DB = "resource"

    # 4. 入力データソースDB
    SOURCE_DB = os.environ.get("SOURCE_DB", "dataset")
    SOURCE_TABLE = "loto_base"

    MONITOR_INTERVAL = 1.0

    @classmethod
    def get_db_url(cls, db_name: str) -> str:
        """指定されたDB名への接続URLを生成"""
        return f"postgresql+psycopg2://{cls.DB_USER}:{cls.DB_PASS}@{cls.DB_HOST}:{cls.DB_PORT}/{db_name}"

    @classmethod
    def update_from_args(cls, args):
        """引数で設定を上書き（ホスト情報など）"""
        if hasattr(args, 'db_host') and args.db_host:
            cls.DB_HOST = args.db_host
        if hasattr(args, 'db_port') and args.db_port:
            cls.DB_PORT = args.db_port
        if hasattr(args, 'db_user') and args.db_user:
            cls.DB_USER = args.db_user
        if hasattr(args, 'db_pass') and args.db_pass:
            cls.DB_PASS = args.db_pass
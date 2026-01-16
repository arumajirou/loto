# ds/src/hist_exog/db_resource_manager.py

import pandas as pd
import numpy as np
import os
import threading
import psutil
import time
from datetime import datetime
from sqlalchemy.engine import Engine
from sqlalchemy import text, types
from typing import List, Dict, Any, Optional

# 内部モジュールからのインポート
from hist_exog_utils import logger, LotoUtils, ROLLING_WINDOWS, LAGS

# 外部ライブラリのチェック
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


# ==========================================
# 定数
# ==========================================
DEFAULT_RESOURCE_TABLE = "loto_resource_log"
DEFAULT_SQL_FEATURE_TABLE = "loto_temp_sql_feat"


# ==========================================
# ResourceMonitor クラス
# ==========================================
# ds/src/hist_exog/db_resource_manager.py (修正版)

class ResourceMonitor:
    """リソース監視とロギング (外部データベースエンジン対応版)"""
    
    def __init__(self, resource_table: str, db_manager: Any, interval: float = 5.0, resource_engine: Engine = None):
        """
        引数:
            resource_table: ログを保存するテーブル名
            db_manager: DBResourceManagerのインスタンス
            interval: 監視間隔（秒）
            resource_engine: (任意) ログ保存専用のSQLAlchemy Engine。
                             指定がない場合は db_manager.engine を使用。
        """
        self.resource_table = resource_table
        self.db_manager = db_manager
        self.schema = getattr(db_manager, 'schema', 'public')
        
        # ログ出力先エンジンの切り分け
        # resource_engineが渡された場合はそちらを、そうでなければメインDBを使用
        self.engine = resource_engine if resource_engine is not None else db_manager.engine
        
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = time.time()
        
        if PSUTIL_AVAILABLE:
            try:
                self._initial_io = psutil.disk_io_counters()
                self._initial_net = psutil.net_io_counters()
            except Exception as e:
                logger.warning(f"IOカウンタの初期化に失敗しました: {e}")
                self._initial_io = None
                self._initial_net = None
        else:
            self._initial_io = None
            self._initial_net = None

        logger.info(f"ResourceMonitor initialized. Target table: {self.schema}.{self.resource_table}")
        if resource_engine:
            logger.info(f"Resource logs will be sent to a dedicated resource database.")

    def _get_current_metrics(self) -> Dict[str, Any]:
        """現在のCPU、メモリ、IOのメトリクスを取得"""
        now = datetime.now()
        metrics = {
            'timestamp': now,
            'elapsed_seconds': time.time() - self._start_time,
            'status': 'RUNNING',
            'cpu_percent': np.nan,
            'memory_percent': np.nan,
            'disk_read_bytes': 0,
            'disk_write_bytes': 0,
            'net_sent_bytes': 0,
            'net_recv_bytes': 0,
            'message': 'Metrics not available (psutil missing)',
            'total_rows': 0 
        }
        
        if PSUTIL_AVAILABLE:
            current_io = psutil.disk_io_counters()
            current_net = psutil.net_io_counters()
            
            metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
            metrics['memory_percent'] = psutil.virtual_memory().percent
            metrics['disk_read_bytes'] = current_io.read_bytes - self._initial_io.read_bytes
            metrics['disk_write_bytes'] = current_io.write_bytes - self._initial_io.write_bytes
            metrics['net_sent_bytes'] = current_net.bytes_sent - self._initial_net.bytes_sent
            metrics['net_recv_bytes'] = current_net.bytes_recv - self._initial_net.bytes_recv
            metrics['message'] = f"CPU:{metrics['cpu_percent']:.1f}%, MEM:{metrics['memory_percent']:.1f}%"
        
        return metrics

    def _monitor_loop(self):
        """バックグラウンドでリソースを定期的に監視し、DBにログを記録"""
        while not self._stop_event.wait(self.interval):
            try:
                metrics = self._get_current_metrics()
                metrics['table_name'] = self.resource_table
                self.db_manager.log_resource(metrics, engine=self.engine)
            except Exception as e:
                logger.error(f"リソース監視スレッドでエラーが発生しました: {e}", exc_info=True)
                break
                
    def start(self):
        """監視を開始"""
        self._start_time = time.time()
        self._stop_event.clear()
        
        if PSUTIL_AVAILABLE:
            self._thread = threading.Thread(target=self._monitor_loop)
            self._thread.daemon = True 
            self._thread.start()
            logger.info(f"リソース監視を開始しました (間隔: {self.interval}秒)")
        else:
            logger.warning("psutilが利用できません。リソース監視はスキップされます。")

    def stop(self):
        """監視を停止"""
        if self._thread:
            self._stop_event.set()
            self._thread.join()
            logger.info("リソース監視を停止しました")
        
    def log_completion(self, elapsed_time: float, total_rows: int):
        """処理完了時のログを記録"""
        metrics = self._get_current_metrics()
        metrics['table_name'] = self.resource_table
        metrics['elapsed_seconds'] = elapsed_time
        metrics['status'] = 'COMPLETED'
        metrics['message'] = '処理が正常に完了しました'
        metrics['total_rows'] = total_rows
        self.db_manager.log_resource(metrics, engine=self.engine)

    def log_error(self, message: str, elapsed: float):
        """エラー発生時のログを記録"""
        metrics = self._get_current_metrics()
        metrics['table_name'] = self.resource_table
        metrics['elapsed_seconds'] = elapsed
        metrics['status'] = 'ERROR'
        metrics['message'] = message
        self.db_manager.log_resource(metrics, engine=self.engine)


# ==========================================
# DBResourceManager クラス
# ==========================================
class DBResourceManager:
    """DB接続、データ取得、SQLでの特徴量前処理、DBへの保存、ログ記録を担当"""
    
    def __init__(self, engine: Engine, schema: str):
        self.engine = engine
        self.schema = schema
        
    def _get_sql_feature_query(self, input_table: str, temp_sql_table: str) -> str:
        """PostgreSQLのウィンドウ関数で高速に計算できる特徴量 (ラグ, 単純なローリング) を定義"""
        
        key_cols = ['loto', 'unique_id', 'ds', 'ts_type']
        
        partition_sql = ", ".join([LotoUtils.qident(c) for c in ['loto', 'unique_id', 'ts_type']])
        order_sql = LotoUtils.qident('ds')
        
        sql_features = []
        
        for lag in LAGS:
            sql_features.append(
                f"LAG(y, {lag}) OVER (PARTITION BY {partition_sql} ORDER BY {order_sql}) AS sql_lag_{lag}"
            )

        for w in ROLLING_WINDOWS:
            window_spec = f"ROWS BETWEEN {w-1} PRECEDING AND 1 PRECEDING"
            
            sql_features.append(
                f"AVG(y) OVER (PARTITION BY {partition_sql} ORDER BY {order_sql} {window_spec}) AS sql_roll_mean_{w}"
            )
            sql_features.append(
                f"SUM(y) OVER (PARTITION BY {partition_sql} ORDER BY {order_sql} {window_spec}) AS sql_roll_sum_{w}"
            )
            sql_features.append(
                f"MIN(y) OVER (PARTITION BY {partition_sql} ORDER BY {order_sql} {window_spec}) AS sql_roll_min_{w}"
            )
            sql_features.append(
                f"MAX(y) OVER (PARTITION BY {partition_sql} ORDER BY {order_sql} {window_spec}) AS sql_roll_max_{w}"
            )
            
        
        sql_query = f"""
        CREATE TEMPORARY TABLE {LotoUtils.qident(temp_sql_table)} AS
        SELECT 
            loto, unique_id, ds, ts_type, y, updated_ts,
            {', '.join(sql_features)}
        FROM 
            {LotoUtils.qident(self.schema)}.{LotoUtils.qident(input_table)}
        ORDER BY 
            loto, unique_id, ts_type, ds;
        """
        
        return sql_query

    def fetch_and_preprocess_data(self, input_table: str, temp_sql_table: str) -> pd.DataFrame:
        """
        DBからデータを取得し、SQLで計算できる特徴量を作成し、一時テーブルに格納する
        """
        logger.info(f"Creating SQL features in temporary table {temp_sql_table}...")
        
        with self.engine.begin() as conn:
            # 1. SQL特徴量生成と一時テーブル作成
            sql_query = self._get_sql_feature_query(input_table, temp_sql_table)
            conn.execute(text(sql_query))

            # 2. 一時テーブルから全データを取得
            query_fetch = f"""
            SELECT *
            FROM {LotoUtils.qident(temp_sql_table)}
            ORDER BY loto, unique_id, ts_type, ds;
            """
            df = pd.read_sql(query_fetch, conn)
            
            # 3. 一時テーブルの削除
            conn.execute(text(f"DROP TABLE {LotoUtils.qident(temp_sql_table)};"))
            
        logger.info(f"Fetched {len(df)} rows. SQL features created and data is ready for Python processing.")
        return df

    def save_exog_features(self, df: pd.DataFrame, table_name: str):
        """特徴量データフレームをDBに保存（REPLACEモードでインデックス設定）"""
        
        key_cols = ['loto', 'unique_id', 'ds', 'ts_type']
        all_cols = key_cols + [c for c in df.columns if c not in key_cols]
        
        dtype_map = {
            'loto': types.Text, 'unique_id': types.Text, 'ds': types.Date, 'ts_type': types.Text,
            'y': types.BigInteger
        }
        feature_cols = [c for c in df.columns if c not in key_cols + ['y']]
        for c in feature_cols:
            if c == 'updated_ts':
                dtype_map[c] = types.DateTime
            else:
                dtype_map[c] = types.Float
        
        logger.info(f"Saving {len(df)} rows to {self.schema}.{table_name} (Using 'replace' mode)...")
        
        with self.engine.begin() as conn:
            # 1. to_sqlで全書き換え (replace)
            df[all_cols].to_sql(table_name, conn, schema=self.schema, if_exists='replace', index=False, dtype=dtype_map)
            
            # 2. インデックス設定
            keys_sql = ", ".join([LotoUtils.qident(c) for c in key_cols])
            conn.execute(text(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS uidx_{table_name}_key 
                ON {self.schema}.{table_name} ({keys_sql});
            """))
            conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_loto_ds 
                ON {self.schema}.{table_name} ({LotoUtils.qident('loto')}, {LotoUtils.qident('ds')});
            """))
            
            logger.info(f"[OK] {table_name} にデータを保存しました。")

    def log_resource(self, monitor_data: Dict[str, Any], engine: Optional[Engine] = None):
        """リソース監視データを指定されたテーブルに記録する (カラム不一致時のフォールバックを強化)"""
        table_name = monitor_data.pop('table_name', DEFAULT_RESOURCE_TABLE)
        df_log = pd.DataFrame([monitor_data])
        target_engine = engine if engine is not None else self.engine

        

        # リソースログテーブルのdtype定義 (total_rows を含む)
        dtype_map = {
            'timestamp': types.DateTime,
            'cpu_percent': types.Float, 'memory_percent': types.Float,
            'disk_read_bytes': types.BigInteger, 'disk_write_bytes': types.BigInteger,
            'net_sent_bytes': types.BigInteger, 'net_recv_bytes': types.BigInteger,
            'elapsed_seconds': types.Float, 'status': types.Text, 'message': types.Text,
            'total_rows': types.BigInteger, 
        }
        
        try:
            # 1. APPEND を試みる
            df_log.to_sql(table_name, target_engine, schema=self.schema, if_exists='append', index=False)
        except Exception as e:
            # 2. APPEND 失敗時 (カラム不一致)
            logger.error(f"リソースログのAPPENDエラー ({e.__class__.__name__}). テーブルを再作成します.")
            try:
                with target_engine.begin() as conn:
                    # 既存テーブルをドロップし、新しいスキーマで作成
                    conn.execute(text(f"DROP TABLE IF EXISTS {LotoUtils.qident(self.schema)}.{LotoUtils.qident(table_name)};"))
                
                # 新しいスキーマで作成し、データを挿入
                df_log.to_sql(table_name, target_engine, schema=self.schema, if_exists='append', index=False, dtype=dtype_map)
                logger.info(f"リソースログテーブル {self.schema}.{table_name} を再作成しました。")
            except Exception as e2:
                logger.error(f"リソースログテーブルの再作成エラー: {e2}")
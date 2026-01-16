# /mnt/e/env/ts/tslib/ds/src/anomalies/tsl/run_anomaly_suite.py
import sys
import os
import uuid
import shutil
import logging
import traceback
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
import inspect

# 設定ファイル読み込み
import config

# TSLライブラリへのパスを通す
sys.path.append(config.TSL_ROOT_PATH)

# ログ設定
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("anomaly_suite.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. モンキーパッチによる互換性修正 (Import前に行う準備)
# ---------------------------------------------------------
try:
    # まずDataset_Customをimport
    from data_provider.data_loader import Dataset_Custom
    
    # オリジナルの __init__ を退避
    original_init = Dataset_Custom.__init__
    
    # 検査して有効な引数名を取得
    sig = inspect.signature(original_init)
    valid_keys = set(sig.parameters.keys())
    valid_keys.discard('self')
    
    # data_factory.pyは args をキーワード引数として渡すため、ここで除外してはならない
    # valid_keys.discard('args') 
    
    def patched_init(self, *args, **kwargs):
        """
        余計なキーワード引数 (win_sizeなど) を削除してから
        オリジナルの __init__ を呼び出すラッパー
        """
        # 有効なキーだけ残す
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        
        # 削除されたキーがあればログに出す（デバッグ用）
        removed = set(kwargs) - set(filtered_kwargs)
        if removed:
            pass 
            
        original_init(self, *args, **filtered_kwargs)
        
    # クラスメソッドを差し替え
    Dataset_Custom.__init__ = patched_init
    logger.info("🔧 Monkey Patch Applied: Dataset_Custom now tolerates extra arguments.")

except ImportError:
    logger.warning("⚠️ Could not import Dataset_Custom for patching. Proceeding anyway.")

# TSL本体のインポート
try:
    from exp.exp_anomaly_detection import Exp_Anomaly_Detection
except ImportError:
    logger.error(f"Error: Could not import TSL from {config.TSL_ROOT_PATH}")
    sys.exit(1)


# ---------------------------------------------------------
# 2. 設定クラス (全てのモデルに対応するパラメータを網羅)
# ---------------------------------------------------------
class TSLArgs:
    """TSLのargparse.Namespaceを模倣する設定クラス"""
    def __init__(self, **kwargs):
        # --- 基本設定 ---
        self.task_name = 'anomaly_detection'
        self.is_training = 1
        self.model_id = 'test'
        self.model = 'Autoformer'
        self.data = 'custom'
        self.root_path = './dataset/'
        self.data_path = 'data.csv'
        self.features = 'M'
        self.target = 'OT'
        self.freq = 'h'
        self.checkpoints = './checkpoints/'

        # --- モデル構造 ---
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 0
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 128
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 256
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.05
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.output_attention = False
        self.do_predict = False
        
        # --- 学習制御 ---
        self.num_workers = 0
        self.itr = 1
        self.train_epochs = 3
        self.batch_size = 32
        self.patience = 3
        self.learning_rate = 0.001
        self.des = 'test'
        self.loss = 'mse'
        self.lradj = 'type1'
        self.use_amp = False

        # --- GPU ---
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0'
        self.gpu_type = 'cuda'

        # --- モデル固有パラメータ ---
        self.num_kernels = 6       # TimesNet用
        self.top_k = 5             # TimesNet
        self.p_hidden_dims = [128, 128] # Nonstationary_Transformer用
        self.p_hidden_layers = 2   # Nonstationary_Transformer用
        
        # --- その他 ---
        self.anomaly_ratio = 0.25
        self.bucket_size = 4
        self.n_hashes = 4
        self.is_revexp = 0         # Autoformer等で要求されることがある
        
        # データ拡張比率
        self.augmentation_ratio = 0
        
        # 季節性パターン
        self.seasonal_patterns = 'Monthly'

        # ユーザー指定値で上書き
        self.__dict__.update(kwargs)


# ---------------------------------------------------------
# 3. データ処理・実行ロジック
# ---------------------------------------------------------
def get_db_engine(uri):
    return create_engine(uri)

def fetch_loto_data(engine):
    """loto_baseテーブルから全データを取得"""
    query = "SELECT loto, ds, unique_id, ts_type, y FROM public.loto_base"
    logger.info("Fetching data from DB...")
    df = pd.read_sql(query, engine)
    return df

def prepare_dataset(df_subset, temp_dir, task_id):
    pivot_df = df_subset.pivot(index='ds', columns='unique_id', values='y')
    pivot_df.reset_index(inplace=True)
    
    # Dataset_Customは 'date' カラムを期待するためリネーム
    pivot_df.rename(columns={'ds': 'date'}, inplace=True)
    
    pivot_df.sort_values('date', inplace=True)
    pivot_df = pivot_df.fillna(0) # 欠損埋め

    filename = f"dataset_{task_id}.csv"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, filename)
    pivot_df.to_csv(file_path, index=False)
    
    # カラム名リスト (date以外)
    feature_cols = [c for c in pivot_df.columns if c != 'date']
    return filename, feature_cols

def save_results_to_db(engine, results_df, table_name="tsl"):
    results_df.to_sql(table_name, engine, if_exists='append', index=False, method='multi', chunksize=1000)
    logger.info(f"Saved {len(results_df)} rows to table '{table_name}'")

def run_all_models():
    run_id = str(uuid.uuid4())
    logger.info(f"Starting Anomaly Detection Suite. Run ID: {run_id}")

    source_engine = get_db_engine(config.SOURCE_DB_URI)
    target_engine = get_db_engine(config.TARGET_DB_URI)

    try:
        df_all = fetch_loto_data(source_engine)
    except Exception as e:
        logger.critical(f"Failed to fetch data from Source DB: {e}")
        return

    # 実行対象モデル
    models_to_run = [
        'TimesNet', 'Autoformer', 'Transformer', 'DLinear', 
        'Nonstationary_Transformer', 'FiLM', 'iTransformer'
    ]

    loto_types = df_all['loto'].unique()
    ts_types = df_all['ts_type'].unique()
    temp_data_dir = os.path.join(os.getcwd(), 'temp_data')
    
    try:
        for loto in loto_types:
            for ts_type in ts_types:
                df_subset = df_all[(df_all['loto'] == loto) & (df_all['ts_type'] == ts_type)]
                if df_subset.empty: continue
                
                task_id = f"{loto}_{ts_type}"
                try:
                    csv_filename, feature_cols = prepare_dataset(df_subset, temp_data_dir, task_id)
                except Exception as e:
                    logger.warning(f"Skipping {task_id}: Data prep failed ({e})")
                    continue
                
                if not feature_cols:
                    logger.warning(f"Skipping {task_id}: No feature columns found.")
                    continue

                logger.info(f"Processing: {task_id} (Features: {len(feature_cols)})")
                c_out = len(feature_cols)
                
                target_col = feature_cols[-1]

                for model_name in models_to_run:
                    logger.info(f"  Running Model: {model_name}")
                    
                    # 修正: すべてのモデルでGPUを有効化
                    use_gpu = True

                    try:
                        args = TSLArgs(
                            model_id=f'{task_id}_{model_name}',
                            model=model_name,
                            root_path=temp_data_dir,
                            data_path=csv_filename,
                            enc_in=c_out,
                            dec_in=c_out,
                            c_out=c_out,
                            target=target_col,
                            use_gpu=use_gpu, 
                            gpu=0 if use_gpu else -1
                        )

                        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                            args.model_id, args.model, args.data, args.features,
                            args.seq_len, args.label_len, args.pred_len,
                            args.d_model, args.n_heads, args.e_layers, args.d_layers,
                            args.d_ff, args.factor, args.embed, args.distil, args.des
                        )
                        args.setting = setting

                        exp = Exp_Anomaly_Detection(args)
                        
                        exp.train(setting)
                        exp.test(setting)

                        res_path = os.path.join('./results', setting)
                        score_path = os.path.join(res_path, 'score.npy')
                        
                        if os.path.exists(score_path):
                            scores = np.load(score_path)
                            target_dates = pd.read_csv(os.path.join(temp_data_dir, csv_filename))['date'].values
                            
                            valid_len = len(scores)
                            valid_dates = target_dates[-valid_len:]
                            
                            result_rows = []
                            for i, date_val in enumerate(valid_dates):
                                result_rows.append({
                                    'run_id': run_id,
                                    'model_name': model_name,
                                    'loto': loto,
                                    'ts_type': ts_type,
                                    'ds': date_val,
                                    'unique_id': 'ALL',
                                    'anomaly_score': float(scores[i]),
                                    'created_at': datetime.now()
                                })
                            
                            if result_rows:
                                save_results_to_db(target_engine, pd.DataFrame(result_rows))
                        else:
                            logger.warning(f"    No output file found at {score_path}")

                    except Exception as e:
                        logger.error(f"    Error running {model_name} on {task_id}: {e}")
                        continue

    except Exception as main_e:
        logger.critical(f"Critical execution error: {main_e}")
    
    finally:
        if os.path.exists(temp_data_dir):
            shutil.rmtree(temp_data_dir)
        logger.info("Cleanup completed.")

if __name__ == "__main__":
    run_all_models()
# /mnt/e/env/ts/tslib/ds/src/anomalies/pypots/pypots_runner.py
import numpy as np
import pandas as pd
import torch
from pypots.anomaly import TimesNet
from windowing import TimeSeriesWindower

class PyPOTSModelRunner:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cpu')
        self.model_name = config.get('model', 'TimesNet')
        self.seq_len = int(config.get('seq_len', 48))
        self.epochs = int(config.get('epochs', 5))
        self.batch_size = int(config.get('batch_size', 32))
        self.anomaly_rate = float(config.get('anomaly_rate', 0.05))

    def run_inference(self, df_series: pd.DataFrame):
        y_values = df_series['y'].values.reshape(-1, 1)
        windower = TimeSeriesWindower(seq_len=self.seq_len)
        X_3d, _ = windower.transform(y_values)
        
        if X_3d is None: return None, None
        
        # モデル初期化
        model = TimesNet(
            n_steps=self.seq_len, n_features=1, 
            epochs=self.epochs, batch_size=self.batch_size, device=self.device
        )
        
        # 学習と推論
        model.fit({"X": X_3d})
        results = model.predict({"X": X_3d})
        
        # スコア抽出
        scores_3d = results['anomaly_score'] 
        full_scores = windower.inverse_transform_scores(scores_3d, len(df_series))
        
        threshold = np.nanpercentile(full_scores, 100 * (1 - self.anomaly_rate))
        is_anomaly = (full_scores > threshold).astype(int)
        
        return full_scores, is_anomaly
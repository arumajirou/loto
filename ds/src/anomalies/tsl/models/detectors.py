import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# PyODはオプション扱い（importエラー回避）
try:
    from pyod.models.copod import COPOD
    from pyod.models.iforest import IForest as PyODIForest
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False

class BaseDetector(ABC):
    def __init__(self, contamination=0.05):
        self.contamination = contamination

    @abstractmethod
    def detect(self, values: np.ndarray) -> np.ndarray:
        """
        異常検知を実行する
        Return: 0 (Normal), 1 (Anomaly) のnumpy array
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

class SklearnIsolationForestDetector(BaseDetector):
    def detect(self, values: np.ndarray) -> np.ndarray:
        # 2次元配列に変換
        X = values.reshape(-1, 1)
        # NaN埋め（念のため）
        if np.isnan(X).any():
            return np.zeros(len(X)) # エラー回避のためのダミー
            
        clf = IsolationForest(contamination=self.contamination, random_state=42, n_jobs=-1)
        preds = clf.fit_predict(X)
        # sklearnは 1:正常, -1:異常。これを 0:正常, 1:異常 に変換
        return np.where(preds == -1, 1, 0)

    @property
    def name(self) -> str:
        return "iforest"

class SklearnLOFDetector(BaseDetector):
    def detect(self, values: np.ndarray) -> np.ndarray:
        X = values.reshape(-1, 1)
        # LOFはpredictがない（fit_predictのみ）
        clf = LocalOutlierFactor(contamination=self.contamination, n_jobs=-1)
        preds = clf.fit_predict(X)
        return np.where(preds == -1, 1, 0)

    @property
    def name(self) -> str:
        return "lof"

class PyodCOPODDetector(BaseDetector):
    def detect(self, values: np.ndarray) -> np.ndarray:
        if not PYOD_AVAILABLE:
            return np.zeros(len(values))
        X = values.reshape(-1, 1)
        clf = COPOD(contamination=self.contamination)
        clf.fit(X)
        # pyodは 0:正常, 1:異常 (そのままでOK)
        return clf.labels_

    @property
    def name(self) -> str:
        return "copod"

def get_detectors(library: str, contamination: float) -> list[BaseDetector]:
    detectors = []
    if library == 'sklearn':
        detectors.append(SklearnIsolationForestDetector(contamination))
        detectors.append(SklearnLOFDetector(contamination))
    elif library == 'pyod' and PYOD_AVAILABLE:
        detectors.append(PyodCOPODDetector(contamination))
        # 必要に応じて追加
    return detectors
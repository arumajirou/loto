import threading
import time
import psutil
import pandas as pd
from datetime import datetime
try:
    import pynvml
    HAS_GPU_MONITOR = True
except ImportError:
    HAS_GPU_MONITOR = False

from config import Config
from db_utils import get_engine, create_resource_table

class ResourceMonitor(threading.Thread):
    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.table_name = create_resource_table(model_name)
        self.engine = get_engine(Config.RESOURCE_DB)
        self.stop_event = threading.Event()
        self.interval = Config.MONITOR_INTERVAL

        if HAS_GPU_MONITOR:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) # GPU 0を想定
            except Exception as e:
                print(f"[Monitor] GPU monitoring failed to init: {e}")
                self.gpu_handle = None
        else:
            self.gpu_handle = None

    def run(self):
        print(f"[Monitor] Started monitoring for {self.model_name}")
        while not self.stop_event.is_set():
            try:
                stats = self._get_stats()
                df = pd.DataFrame([stats])
                df.to_sql(self.table_name, self.engine, if_exists='append', index=False)
            except Exception as e:
                print(f"[Monitor] Error logging stats: {e}")
            
            time.sleep(self.interval)

    def _get_stats(self):
        # CPU & RAM
        cpu_pct = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        ram_gb = ram.used / (1024 ** 3)
        
        # Disk IO (Delta計算は省略し、カウンタまたは簡易スナップショットとする)
        # ※厳密な速度計算には前回値との差分が必要だが、ここでは簡易的に現在のIOカウンタ取得負荷のみ考慮
        disk = psutil.disk_io_counters()
        read_mb = disk.read_bytes / (1024 ** 2)
        write_mb = disk.write_bytes / (1024 ** 2)

        # GPU
        gpu_util = 0.0
        vram_gb = 0.0
        if self.gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_util = float(util.gpu)
                vram_gb = mem.used / (1024 ** 3)
            except Exception:
                pass

        return {
            "timestamp": datetime.now(),
            "cpu_percent": cpu_pct,
            "ram_used_gb": round(ram_gb, 2),
            "gpu_util_percent": gpu_util,
            "vram_used_gb": round(vram_gb, 2),
            "disk_read_mb": round(read_mb, 2), # 累積値
            "disk_write_mb": round(write_mb, 2) # 累積値
        }

    def stop(self):
        self.stop_event.set()
        if self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        self.join()
        print("[Monitor] Stopped.")
import time
import os
import csv
import numpy as np

class TelemetryListener:
    def __init__(self, filepath="/tmp/enb_metrics.csv", feature_dim=128):
        self.filepath = filepath
        self.feature_dim = feature_dim
        self.is_running = False

    def parse_last_csv_line(self):
        if not os.path.exists(self.filepath):
            return None, np.zeros(self.feature_dim, dtype='float32')
            
        try:
            with open(self.filepath, 'r') as f:
                lines = f.readlines()
                if len(lines) < 2:
                    return None, np.zeros(self.feature_dim, dtype='float32')
                
                header = [h.strip() for h in lines[0].split(',')]
                last_line = [v.strip() for v in lines[-1].split(',')]
                data = dict(zip(header, last_line))
                
                # Extraction des KPIs srsRAN
                rsrp = float(data.get("rsrp", -100.0))
                bler = float(data.get("bler", 0.0))
                throughput = float(data.get("dl_brate", 0.0)) / 1e6  # Conversion en Mbps
                
                raw_features = [rsrp / -140.0, bler, throughput / 100.0]
                vector = np.zeros(self.feature_dim, dtype='float32')
                vector[:len(raw_features)] = raw_features
                
                return {"rsrp": rsrp, "bler": bler, "throughput": throughput}, vector
        except Exception:
            return None, np.zeros(self.feature_dim, dtype='float32')

    def listen(self, callback):
        self.is_running = True
        print(f"[TelemetryListener] Surveillance active sur {self.filepath}...")
        last_mtime = 0
        while self.is_running:
            if os.path.exists(self.filepath):
                mtime = os.path.getmtime(self.filepath)
                if mtime > last_mtime:
                    last_mtime = mtime
                    metrics, vector = self.parse_last_csv_line()
                    if metrics:
                        callback(metrics, vector)
            time.sleep(1)

    def stop(self):
        self.is_running = False
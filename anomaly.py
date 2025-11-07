import numpy as np

def detect_anomalies(y, trend_low, k=3.0):
    residual = y - trend_low
    resid_std = np.std(residual)
    anomaly_mask = np.zeros_like(residual, dtype=bool)
    if resid_std > 0:
        anomaly_mask = np.abs(residual) > (k * resid_std)
    return residual, anomaly_mask

import numpy as np
from scipy.signal import periodogram

def compute_periodogram(y_norm, fs=1.0):
    f, Pxx = periodogram(y_norm, fs=fs, scaling='density', window='hann', detrend='linear')
    with np.errstate(divide='ignore'):
        period_days = np.where(f > 0, 1.0 / f, np.inf)
    return f, Pxx, period_days

def top_periods(f, Pxx, period_days, top_n=8):
    valid_idx = np.where(f > 0)[0]
    if valid_idx.size > 0:
        idx = valid_idx[np.argsort(Pxx[valid_idx])][-top_n:][::-1]
        return [(f[i], period_days[i], Pxx[i]) for i in idx]
    else:
        return []

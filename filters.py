import numpy as np
from scipy.signal import butter, filtfilt, wiener
import pandas as pd
import warnings


def butter_filter(data, cutoff_period_days, btype='low', fs=1.0, order=4):
    data = np.ravel(data)
    if len(data) < 3:
        warnings.warn("Data too short for filtering; returning original data")
        return data.copy()
    cutoff = 1.0 / float(cutoff_period_days)
    nyq = 0.5 * fs
    Wn = cutoff / nyq
    if not 0 < Wn < 1:
        warnings.warn(f"Wn={Wn:.4f} out of (0,1), returning original data")
        return data.copy()
    b, a = butter(order, Wn, btype=btype)
    padlen = min(3 * max(len(a), len(b)), len(data) - 1)
    try:
        return filtfilt(b, a, data, padlen=padlen)
    except ValueError:
        warnings.warn("filtfilt failed; fallback to lfilter")
        from scipy.signal import lfilter
        return lfilter(b, a, data)


def moving_average(data, window=7):
    return pd.Series(data).rolling(window=window, min_periods=1, center=True).mean().values


def apply_wiener(data):
    try:
        return wiener(data)
    except Exception:
        return moving_average(data)


def kalman_1d(observations, process_var=1e-3, measurement_var=1.0):
    n = len(observations)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhatminus = np.zeros(n)
    Pminus = np.zeros(n)
    K = np.zeros(n)
    if n == 0:
        return xhat
    xhat[0] = observations[0]
    P[0] = 1.0
    for k in range(1, n):
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + process_var
        K[k] = Pminus[k] / (Pminus[k] + measurement_var)
        xhat[k] = xhatminus[k] + K[k] * (observations[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return xhat

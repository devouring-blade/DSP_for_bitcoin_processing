import numpy as np

def fit_ar_lstsq(series, p):
    series = np.ravel(series)
    N = len(series)
    if N <= p:
        raise ValueError("Series too short for AR model")
    X = np.zeros((N - p, p))
    for i in range(p):
        X[:, i] = series[p - i - 1: N - i - 1]
    y_ar = series[p:]
    coeffs, *_ = np.linalg.lstsq(X, y_ar, rcond=None)
    return coeffs.flatten()

def predict_ar(series, coeffs, steps):
    p = len(coeffs)
    history = list(np.ravel(series)[-p:])
    preds = []
    for _ in range(steps):
        x = np.array(history[-p:][::-1])
        pred = np.dot(x, coeffs)
        preds.append(pred)
        history.append(pred)
    return np.array(preds)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

from data_loader import load_btc_data
from filters import butter_filter, moving_average, apply_wiener, kalman_1d
from spectrum import compute_periodogram, top_periods
from ar_model import fit_ar_lstsq, predict_ar
from anomaly import detect_anomalies

# -------- Config ---------
TICKER = "BTC-USD"
START = "2019-01-01"
RESAMPLE = "1D"
LOWPASS_CUTOFF_DAYS = 30
HIGHPASS_CUTOFF_DAYS = 7
MA_WINDOW = 7
AR_LAGS = 10
FORECAST_STEPS = 7
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# -------------------------

# Load data
ts, y = load_btc_data(TICKER, START, None, RESAMPLE)

# Detrend & normalize
y_ma = moving_average(y, MA_WINDOW)
y_detrended = y - y_ma
y_norm = (y_detrended - np.mean(y_detrended)) / (np.std(y_detrended)+1e-12)

# Spectrum
f, Pxx, period_days = compute_periodogram(y_norm)
top_period_list = top_periods(f, Pxx, period_days)

# Filters
trend_low = butter_filter(y, LOWPASS_CUTOFF_DAYS, 'low')
osc_short = butter_filter(y, HIGHPASS_CUTOFF_DAYS, 'high')
wien = apply_wiener(y)
kalman_trend = kalman_1d(y, process_var=1e-2, measurement_var=np.var(y)*0.1+1e-6)

# Anomalies
residual, anomaly_mask = detect_anomalies(y, trend_low)

# AR Forecast
if len(y) > AR_LAGS:
    try:
        ar_coeffs = fit_ar_lstsq(y, AR_LAGS)
        forecast = predict_ar(y, ar_coeffs, FORECAST_STEPS)
    except:
        forecast = np.full(FORECAST_STEPS, y[-1])
else:
    forecast = np.full(FORECAST_STEPS, y[-1])
forecast_dates = pd.date_range(start=ts.index[-1]+timedelta(days=1), periods=FORECAST_STEPS, freq='D')

# Plot
plt.figure(figsize=(14, 8))
plt.plot(ts.index, y, label='Price')
plt.plot(ts.index, trend_low, label=f'Low-pass {LOWPASS_CUTOFF_DAYS}d')
plt.plot(ts.index, y_ma, label=f'MA {MA_WINDOW}d')
plt.plot(ts.index, kalman_trend, label='Kalman', linestyle='--')
plt.scatter(ts.index[anomaly_mask], y[anomaly_mask], color='red', label='Anomaly')
plt.plot(forecast_dates, forecast, label='AR forecast', linestyle='-.', marker='o')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(10,6))
mask = f>0
if np.any(mask):
    plt.semilogx(period_days[mask], Pxx[mask])
    plt.gca().invert_xaxis()
    plt.xlabel("Period (days)")
    plt.ylabel("Power")
    plt.title("Periodogram")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

# Save CSV
df_out = pd.DataFrame({
    "date": ts.index,
    "price": y,
    "trend_low": trend_low,
    "ma": y_ma,
    "kalman": kalman_trend,
    "residual": residual,
    "anomaly": anomaly_mask
})
df_out.to_csv(f"{OUTPUT_DIR}/btc_dsp_results.csv", index=False)

print("Top spectral periods:")
for ff, pdays, pwr in top_period_list:
    print(f" - period ≈ {pdays:.2f} days, freq={ff:.5f}, power={pwr:.5e}")

import yfinance as yf
import pandas as pd


def load_btc_data(ticker="BTC-USD", start="2019-01-01", end=None, resample="1D"):
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.shape[0] == 0:
        raise SystemExit("No data downloaded. Check ticker or internet.")

    ts = data['Close'].dropna()

    if resample != "1D":
        ts = ts.resample(resample).last().ffill()

    ts.index = pd.to_datetime(ts.index)
    inferred = ts.index.inferred_freq
    if inferred is None:
        try:
            ts = ts.asfreq('D').ffill()
        except Exception:
            ts = ts.copy()
    else:
        ts = ts.asfreq(inferred).ffill()

    y = ts.values.astype(float).ravel()
    return ts, y

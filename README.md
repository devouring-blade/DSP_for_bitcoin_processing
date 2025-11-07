## Requirements
**Language:** Python  
**Version:** 3.12.5  

---

## Libraries
**NumPy** → numerical computation and vector/matrix operations  
**Pandas** → handling time series and tabular data  
**Matplotlib** → data visualization and plotting  
**yfinance** → downloading Bitcoin (or stock) price data  
**SciPy** → signal processing (filtering, periodogram, Wiener, Butterworth)  

---

## Project Structure
```plaintext
DSP_for_bitcoin_processing/
│
├── main.py          — main file: loads data, calls modules, and plots results
├── data_loader.py   — module for fetching data from yfinance
├── filters.py       — module containing Butterworth, Moving Average (MA), Wiener, and Kalman filters
├── spectrum.py      — module for FFT / periodogram signal analysis
├── ar_model.py      — module for AutoRegressive (AR) forecasting
├── anomaly.py       — module for anomaly detection
├── utils.py         — utility functions (e.g., ensure_1d, plot helpers)
└── outputs/         — directory for saving results (plots + CSV files)


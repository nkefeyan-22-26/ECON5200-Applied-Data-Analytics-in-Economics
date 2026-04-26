# Lab 21: Time Series Forecasting — ARIMA, GARCH & Bootstrap

**Course:** ECON 5200: Causal Machine Learning & Applied Analytics
**Lab Type:** Diagnosis-First | Forecasting & Volatility Modeling

---

## Objective

Diagnose and correct a deliberately flawed ARIMA pipeline on U.S. CPI data, extend the analysis to conditional volatility modeling of S&P 500 returns using GARCH(1,1), and build a reusable forecast evaluation module with distribution-free bootstrap confidence intervals.

---

## Data Sources

- **U.S. CPI (CPIAUCNS)** — Monthly, not seasonally adjusted, from FRED (2000–present)
- **S&P 500 (^GSPC)** — Daily adjusted close prices via Yahoo Finance (2000–2024)

---

## Methodology

### Part 1 — Diagnostic Analysis

Identified three deliberate modeling errors in a provided ARIMA pipeline for CPI:

- **Stationarity error:** `ARIMA(2,0,1)` fit to raw CPI levels despite the ADF test confirming a unit root. Fitting ARMA to a non-stationary series produces spurious estimates.
- **Seasonality omission:** `ARIMA(2,1,1)` with `d=1` addressed the trend but ignored monthly seasonal structure. ACF of residuals showed significant spikes at lags 12, 24, and 36, confirming seasonal autocorrelation leaking into residuals.
- **Missing diagnostic:** The pipeline jumped directly to forecasting without running the Ljung-Box test. Skipping this check means forecast confidence intervals are unreliable when residuals are still autocorrelated.

### Part 2 — Corrected SARIMA Pipeline

Fixed all three errors:

1. Applied seasonal differencing (`diff(12).diff()`) before the ADF test to correctly assess stationarity — `regression='c'` used on the differenced series after removing the trend.
2. Used `pmdarima.auto_arima` with `seasonal=True`, `m=12`, `d=1`, `D=1` to select optimal SARIMA order via AIC.
3. Re-fit the selected model with `statsmodels SARIMAX` and ran Ljung-Box at lags 12 and 24 before producing any forecast.

**Verification checkpoints passed:**
- ADF p-value on doubly-differenced CPI < 0.05
- Ljung-Box p-values at lags 12 and 24 > 0.05
- Seasonal ACF spikes at lags 12, 24 absent from SARIMA residuals

### Part 3 — GARCH(1,1) on S&P 500

Modeled conditional volatility of daily S&P 500 log returns using GARCH(1,1):

$$\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$$

- Fit using the `arch` library with `mean='Constant'`, `vol='Garch'`, `p=1`, `q=1`, `dist='normal'`
- α₁ + β₁ confirmed < 1 (variance stationarity)
- Annotated conditional volatility plot with major crisis periods: Sep 11, Lehman Brothers collapse, COVID-19 crash, 2022 bear market

### Part 4 — `forecast_evaluation.py` Module

Built a reusable portfolio module with two functions:

**`compute_mase(actual, forecast, insample, m)`**
Computes Mean Absolute Scaled Error relative to a naive seasonal benchmark. MASE < 1 indicates the model outperforms the naive forecast.

**`backtest_expanding_window(series, model_fn, min_train, horizon, step)`**
Runs an expanding-window backtest starting from `min_train` observations. At each origin, fits the model, forecasts `horizon` steps, records errors and MASE, then expands the training window by `step`. Returns a tidy DataFrame with columns `origin`, `horizon`, `actual`, `forecast`, `error`, `abs_error`, `mase`.

### Challenge — Block Bootstrap Forecast Intervals

Implemented distribution-free forecast confidence intervals via moving block bootstrap:

1. Extract SARIMA residuals from the fitted model
2. For each of 500 bootstrap iterations, resample overlapping blocks of residuals (block size = 6) to preserve autocorrelation and heteroskedasticity structure
3. Construct bootstrap forecast paths as point forecast + resampled residuals
4. Compute percentile-based 95% intervals across paths

This produces wider, more honest intervals than the standard parametric ARIMA approach when residuals exhibit volatility clustering or heavy tails.

---

## Key Findings

- One regular difference is insufficient to stationarize CPI — both a trend difference (`d=1`) and a seasonal difference (`D=1`) are required, reflecting dual unit roots in the raw series.
- Plain ARIMA visibly fails on monthly macroeconomic data: seasonal autocorrelation at lags 12 and 24 persists in residuals until SARIMA terms are added.
- S&P 500 GARCH(1,1) estimates yield α₁ + β₁ ≈ 0.9826, implying a volatility shock half-life of roughly 39.5 days. This confirms that financial stress (2008, COVID) elevates conditional variance for weeks before mean-reversion.
- Block bootstrap intervals are materially wider than parametric ARIMA CIs for CPI, reflecting the non-Gaussian, heteroskedastic nature of macro residuals.

---

## Repository Structure

```
lab-21-forecasting/
├── notebooks/
│   └── lab_ch21_diagnostic.ipynb   # Full lab notebook with outputs
├── src/
│   └── forecast_evaluation.py      # Reusable MASE + backtest module
└── README.md
```

---

## Environment

```
Python 3.12
statsmodels | pmdarima | arch | fredapi | yfinance
numpy | pandas | matplotlib
```

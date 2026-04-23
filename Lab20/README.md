# Time Series Diagnostics & Advanced Decomposition

**ECON 5200: Causal Machine Learning & Applied Analytics — Topic 20 of 26**

---

## Objective

A diagnosis-first investigation of classical decomposition pitfalls, unit root misspecification, and multi-seasonal analysis applied to FRED macroeconomic data, culminating in a production-grade Python module and interactive Streamlit dashboard.

---

## Methodology

- **STL decomposition diagnosis:** Identified the application of additive STL to multiplicative retail sales data (RSXFSN). Seasonal amplitude grew proportionally with the trend level — a signature of multiplicative structure. Applied a log-transform prior to decomposition, converting the multiplicative model to additive and stabilizing the seasonal component. Verified via seasonal amplitude ratio (target: 0.7–1.3).

- **ADF test misspecification diagnosis:** Corrected a flawed unit root test on Real GDP (GDPC1) that used `regression='n'`, omitting both the constant and deterministic trend from the ADF auxiliary regression. This biased the test statistic toward spurious rejection of the unit root null. Fixed to `regression='ct'` and supplemented with KPSS to construct a 2×2 decision table yielding a robust NON-STATIONARY verdict.

- **MSTL multi-seasonal decomposition:** Applied Multiple STL decomposition to simulated hourly electricity demand with overlapping daily (period=24) and weekly (period=168) seasonal cycles. Confirmed successful separation of both components, with residual standard deviation matching the true noise level of 15 MW.

- **Moving block bootstrap:** Implemented a moving block bootstrap (block_size=8 quarters) over 200 replicates to generate pointwise 90% confidence bands around the STL trend for log-GDP. Block resampling preserves the autocorrelation structure of economic residuals, which standard i.i.d. bootstrap destroys.

- **PELT structural break detection:** Applied the PELT algorithm with an RBF cost function to quarterly GDP growth. Ran per-segment ADF and KPSS tests on each identified regime to assess whether stationarity conclusions change across structural regimes.

- **Production module:** Packaged all analytical functions into a reusable `src/decompose.py` module with type hints, docstrings, and a self-test suite. Functions: `run_stl()`, `test_stationarity()`, `detect_breaks()`, `run_mstl()`, `block_bootstrap_trend()`.

- **Interactive dashboard:** Built a Streamlit application with live FRED integration, supporting STL, MSTL, and Classical decomposition methods, stationarity test output, PELT break overlays, and block bootstrap confidence band generation.

---

## Key Findings

- **GDP is I(1):** Real GDP in levels fails to reject the unit root null under the correctly specified ADF test (`regression='ct'`, p > 0.05). First differences are stationary, confirming the standard macroeconomic result that GDP must be differenced once before modeling.

- **Structural breaks cluster around known macro shocks:** PELT detected breaks near 1973 (oil shock), 1982 (Volcker disinflation), 2008 (financial crisis), and 2020 (COVID shock) — consistent with periods of abrupt change in both the mean and variance of GDP growth.

- **Bootstrap uncertainty is recession-dependent:** Confidence bands around the STL trend are materially wider during the 2008–2009 financial crisis and the 2020 contraction than during the 2010–2019 expansion, reflecting the elevated variance and persistence of residuals during economic downturns.

- **MSTL successfully separates overlapping seasonal cycles:** Residual standard deviation after MSTL decomposition matched the true noise level (σ = 15 MW), confirming that both the daily and weekly seasonal components were cleanly extracted with no leakage between them.

---

## How to Reproduce

```bash
git clone https://github.com/YOUR_USERNAME/econ-lab-20-time-series.git
cd econ-lab-20-time-series
pip install -r requirements.txt
jupyter lab notebooks/lab-ch20-diagnostic.ipynb
streamlit run app.py
```

A free FRED API key is required. Register at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) and paste it into the notebook setup cell or the Streamlit sidebar.

---

## Repository Structure

```
econ-lab-20-time-series/
├── README.md
├── requirements.txt
├── app.py
├── verification-log.md
├── notebooks/
│   └── lab-ch20-diagnostic.ipynb
├── src/
│   └── decompose.py
└── figures/
    ├── stl_decomposition.png
    ├── bootstrap_ci.png
    └── structural_breaks.png
```

---

## Dependencies

See `requirements.txt`. Key packages: `fredapi`, `statsmodels`, `ruptures`, `streamlit`, `matplotlib`, `pandas`, `numpy`.

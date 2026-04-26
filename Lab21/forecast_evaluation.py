"""
forecast_evaluation.py — Forecast Evaluation & Backtesting Module

Reusable functions for computing MASE and running expanding-window
backtests on time series forecasting models.

Author: [Your Name]
Course: ECON 5200, Lab 21
"""

import numpy as np
import pandas as pd
from typing import Callable


def compute_mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    insample: np.ndarray,
    m: int = 1
) -> float:
    """Compute Mean Absolute Scaled Error.

    MASE < 1: model beats the naive seasonal benchmark.
    MASE > 1: naive benchmark is better.

    Args:
        actual:   True out-of-sample values
        forecast: Model predictions (same length as actual)
        insample: In-sample (training) data for naive baseline
        m:        Seasonal period (1=random walk, 12=monthly seasonal)

    Returns:
        MASE score (float)
    """
    actual   = np.array(actual)
    forecast = np.array(forecast)
    insample = np.array(insample)

    mae_forecast = np.mean(np.abs(actual - forecast))

    # Naive seasonal baseline: predict insample[t] = insample[t - m]
    naive_errors = insample[m:] - insample[:-m]
    mae_naive    = np.mean(np.abs(naive_errors))

    if mae_naive == 0:
        raise ValueError("Naive MAE is zero — in-sample series is constant.")

    return mae_forecast / mae_naive


def backtest_expanding_window(
    series: pd.Series,
    model_fn: Callable,
    min_train: int = 120,
    horizon: int = 12,
    step: int = 12
) -> pd.DataFrame:
    """Expanding-window time series backtest.

    Starting from min_train observations, fit the model and forecast
    'horizon' steps ahead. Then expand the training window by 'step'
    observations and repeat until the series is exhausted.

    Args:
        series:    Full time series with DatetimeIndex
        model_fn:  Callable(train_series) -> np.ndarray of length horizon
        min_train: Minimum number of training observations
        horizon:   Number of steps to forecast per iteration
        step:      Observations added to the training window per iteration

    Returns:
        DataFrame with columns:
        ['origin', 'horizon', 'actual', 'forecast', 'error', 'abs_error', 'mase']
    """
    records = []
    n = len(series)

    for origin in range(min_train, n - horizon + 1, step):
        train  = series.iloc[:origin]
        actual = series.iloc[origin:origin + horizon].values

        forecast = np.array(model_fn(train))

        errors     = actual - forecast
        abs_errors = np.abs(errors)

        # MASE relative to seasonal naive on the training window (m=12 for monthly)
        mase_val = compute_mase(actual, forecast, train.values, m=12)

        for h in range(horizon):
            records.append({
                'origin':    series.index[origin],
                'horizon':   h + 1,
                'actual':    actual[h],
                'forecast':  forecast[h],
                'error':     errors[h],
                'abs_error': abs_errors[h],
                'mase':      mase_val,
            })

    return pd.DataFrame(records)


# --- Quick self-test ---
if __name__ == '__main__':
    print('forecast_evaluation.py loaded successfully.\n')

    rng      = np.random.default_rng(42)
    insample = rng.normal(0, 1, 120)
    actual   = rng.normal(0, 1, 12)
    forecast = rng.normal(0, 1, 12)

    mase = compute_mase(actual, forecast, insample, m=12)
    print(f'Test MASE: {mase:.4f}')
    print(f'  (expected ~1.0 for random forecast vs random naive)\n')

    # Expanding window test with a naive model
    idx    = pd.date_range('2010-01-01', periods=180, freq='MS')
    series = pd.Series(rng.normal(0, 1, 180), index=idx)

    def naive_model(train):
        return np.repeat(train.iloc[-1], 12)

    results = backtest_expanding_window(series, naive_model, min_train=120, horizon=12, step=12)
    print('Expanding window backtest results (first 5 rows):')
    print(results.head())
    print(f'\nMean MASE across all windows: {results["mase"].mean():.4f}')
    print(f'Mean MAE across all steps:    {results["abs_error"].mean():.4f}')

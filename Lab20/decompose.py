"""
decompose.py — Reusable time-series decomposition module
ECON 5200: Lab 20
"""

from __future__ import annotations
import warnings
from typing import List
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL, MSTL
from statsmodels.tsa.stattools import adfuller, kpss

try:
    import ruptures as rpt
    _HAS_RUPTURES = True
except ImportError:
    _HAS_RUPTURES = False

warnings.filterwarnings("ignore")


def run_stl(
    series: pd.Series,
    period: int,
    log_transform: bool = False,
    robust: bool = True,
) -> object:
    """Apply STL decomposition with optional log-transform.

    Use log_transform=True when the series has multiplicative seasonality
    (i.e. seasonal amplitude grows proportionally with the trend level).
    The log converts Y = T x S x R into log(Y) = log(T) + log(S) + log(R),
    making the structure additive so STL produces a stable seasonal component.

    Args:
        series: Time series with DatetimeIndex and freq set.
        period: Seasonal period (12=monthly, 4=quarterly).
        log_transform: If True, applies np.log before STL.
        robust: If True, downweights outliers via bisquare weights.

    Returns:
        STL result object with .trend, .seasonal, .resid attributes.

    Raises:
        ValueError: If series has non-positive values and log_transform=True.
        ValueError: If series.index.freq is None.
    """
    if series.index.freq is None:
        raise ValueError(
            "series.index.freq is None. Set frequency before calling run_stl(), "
            "e.g. series.index.freq = 'MS'."
        )

    work = series.dropna().copy()

    if log_transform:
        if (work <= 0).any():
            raise ValueError(
                "log_transform=True requires all values to be strictly positive."
            )
        work = np.log(work)

    return STL(work, period=period, robust=robust).fit()


def test_stationarity(
    series: pd.Series,
    alpha: float = 0.05,
) -> dict:
    """Run ADF + KPSS and return the 2x2 decision table verdict.

    ADF null hypothesis:  unit root present (series is non-stationary).
    KPSS null hypothesis: series IS stationary.
    The two tests are complementary — using both reduces Type I and II errors.

    2x2 decision table:
        ADF rejects + KPSS does not reject  =>  stationary
        ADF does not reject + KPSS rejects  =>  non-stationary
        Both reject                          =>  contradictory
        Neither rejects                      =>  inconclusive

    Args:
        series: Univariate time series (levels or differences).
        alpha:  Significance level for both tests. Default 0.05.

    Returns:
        dict with keys: adf_stat, adf_p, kpss_stat, kpss_p, verdict.
        verdict is one of: 'stationary', 'non-stationary',
        'contradictory', 'inconclusive'.
    """
    clean = series.dropna()

    if len(clean) < 20:
        raise ValueError(
            f"Series has only {len(clean)} observations. "
            "At least 20 are required for reliable ADF/KPSS testing."
        )

    adf_stat, adf_p, _, _, _, _ = adfuller(clean, regression='ct', autolag='AIC')
    kpss_stat, kpss_p, _, _     = kpss(clean, regression='ct', nlags='auto')

    adf_rej  = adf_p  < alpha
    kpss_rej = kpss_p < alpha

    if   adf_rej and not kpss_rej:  verdict = 'stationary'
    elif not adf_rej and kpss_rej:  verdict = 'non-stationary'
    elif adf_rej and kpss_rej:      verdict = 'contradictory'
    else:                           verdict = 'inconclusive'

    return {
        'adf_stat':  round(adf_stat,  4),
        'adf_p':     round(adf_p,     4),
        'kpss_stat': round(kpss_stat, 4),
        'kpss_p':    round(kpss_p,    4),
        'verdict':   verdict,
    }


def detect_breaks(
    series: pd.Series,
    pen: float = 10.0,
) -> List[pd.Timestamp]:
    """Detect structural breaks using the PELT algorithm.

    PELT minimises a penalised cost function (RBF) to find the globally
    optimal set of changepoints. The penalty controls the bias-variance
    tradeoff: higher pen = fewer breaks, lower pen = more breaks.

    Args:
        series: Time series with DatetimeIndex.
        pen:    PELT penalty parameter. Default 10.0.

    Returns:
        List of pd.Timestamp marking the first observation of each new regime.
        Returns empty list if no breaks are found.

    Raises:
        ImportError: If the ruptures package is not installed.
    """
    if not _HAS_RUPTURES:
        raise ImportError(
            "The ruptures package is required. Install with: pip install ruptures"
        )

    clean = series.dropna()
    algo  = rpt.Pelt(model='rbf').fit(clean.values)
    raw   = algo.predict(pen=pen)

    # ruptures returns the index AFTER the last obs of each segment.
    # The final value is always len(signal) — exclude it.
    return [clean.index[bp] for bp in raw if bp < len(clean)]


# -----------------------------------------------------------
# Self-test block — run with: python src/decompose.py
# -----------------------------------------------------------

if __name__ == '__main__':
    print('=' * 50)
    print('decompose.py self-tests')
    print('=' * 50)

    np.random.seed(42)
    idx = pd.date_range('2000-01-01', periods=120, freq='MS')
    ts  = pd.Series(
        np.linspace(100, 200, 120)
        + 20 * np.sin(2 * np.pi * np.arange(120) / 12)
        + np.random.normal(0, 5, 120),
        index=idx
    )
    ts.index.freq = 'MS'

    # Test 1: run_stl
    res = run_stl(ts, period=12)
    assert hasattr(res, 'trend'), 'run_stl must return object with .trend'
    print('run_stl:                     PASS')

    # Test 2: test_stationarity on levels → non-stationary
    r = test_stationarity(ts)
    assert r['verdict'] == 'non-stationary', \
        f"Expected 'non-stationary', got '{r['verdict']}'"
    print('test_stationarity (levels):  PASS')

    # Test 3: test_stationarity on first diff → stationary
    r2 = test_stationarity(ts.diff().dropna())
    assert r2['verdict'] == 'stationary', \
        f"Expected 'stationary', got '{r2['verdict']}'"
    print('test_stationarity (diff):    PASS')

    # Test 4: detect_breaks
    if _HAS_RUPTURES:
        ts_break = ts.copy()
        ts_break.iloc[60:] += 50
        breaks = detect_breaks(ts_break, pen=5)
        assert len(breaks) >= 1, 'Expected at least one break'
        print(f'detect_breaks:               PASS ({len(breaks)} break(s) found)')
    else:
        print('detect_breaks:               SKIPPED (ruptures not installed)')

    print('=' * 50)
    print('All tests passed.')
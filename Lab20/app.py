import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from fredapi import Fred
from decompose import run_stl, run_mstl, test_stationarity, detect_breaks, block_bootstrap_trend

st.set_page_config(page_title='Time Series Decomposition', layout='wide')
st.title('Lab 20 — Interactive Time Series Decomposition')

# ── Sidebar inputs ────────────────────────────────────────────
st.sidebar.header('Settings')
api_key   = st.sidebar.text_input('FRED API Key', type='password')
series_id = st.sidebar.text_input('FRED Series ID', value='RSXFSN')
method    = st.sidebar.selectbox('Decomposition method', ['STL', 'MSTL', 'Classical'])
log_flag  = st.sidebar.checkbox('Log-transform (multiplicative data)', value=True)
period1   = st.sidebar.slider('Primary period', 2, 52, 12)
penalty   = st.sidebar.slider('PELT penalty (breaks)', 1.0, 50.0, 10.0)
run_boot  = st.sidebar.checkbox('Run block bootstrap (slow)')
block_sz  = st.sidebar.slider('Bootstrap block size', 2, 20, 8)
n_boot    = st.sidebar.slider('Bootstrap replicates', 50, 300, 200)

if not api_key:
    st.info('Enter your FRED API key in the sidebar to load data.')
    st.stop()

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data(key, sid):
    fred = Fred(api_key=key)
    s = fred.get_series(sid).dropna()
    s.index = pd.DatetimeIndex(s.index)
    s.index.freq = pd.infer_freq(s.index)
    return s

try:
    raw = load_data(api_key, series_id)
except Exception as e:
    st.error(f'Failed to load {series_id}: {e}')
    st.stop()

st.success(f'Loaded {series_id}: {len(raw)} observations | freq={raw.index.freq}')

# ── Raw series plot ───────────────────────────────────────────
st.subheader('Raw Series')
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(raw.index, raw.values, linewidth=0.8, color='#2c3e50')
ax.set_title(series_id)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Decomposition ─────────────────────────────────────────────
st.subheader(f'{method} Decomposition')
work = np.log(raw) if log_flag and (raw > 0).all() else raw.copy()

try:
    if method == 'STL':
        res = run_stl(work, period=period1, log_transform=False)
        components = {
            'Observed': work,
            'Trend':    res.trend,
            'Seasonal': res.seasonal,
            'Residual': res.resid,
        }
    elif method == 'MSTL':
        res = run_mstl(work, periods=[period1])
        components = {
            'Observed': work,
            'Trend':    res.trend,
            'Seasonal': res.seasonal.iloc[:, 0],
            'Residual': res.resid,
        }
    else:
        from statsmodels.tsa.seasonal import seasonal_decompose
        res = seasonal_decompose(
            work, model='additive',
            period=period1, extrapolate_trend='freq'
        )
        components = {
            'Observed': work,
            'Trend':    pd.Series(res.trend,    index=work.index),
            'Seasonal': pd.Series(res.seasonal, index=work.index),
            'Residual': pd.Series(res.resid,    index=work.index),
        }

    colors = ['#2c3e50', '#e67e22', '#27ae60', '#c0392b']
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, (name, data), color in zip(axes, components.items(), colors):
        ax.plot(data.index, data.values, linewidth=0.7, color=color)
        ax.set_ylabel(name)
        if name == 'Residual':
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

except Exception as e:
    st.error(f'Decomposition failed: {e}')

# ── Stationarity tests ────────────────────────────────────────
st.subheader('Stationarity Tests')
col1, col2 = st.columns(2)

with col1:
    st.markdown('**Levels**')
    r = test_stationarity(work)
    st.write(f"ADF p = {r['adf_p']} | KPSS p = {r['kpss_p']}")
    st.write(f"Verdict: **{r['verdict'].upper()}**")

with col2:
    st.markdown('**First differences**')
    r2 = test_stationarity(work.diff().dropna())
    st.write(f"ADF p = {r2['adf_p']} | KPSS p = {r2['kpss_p']}")
    st.write(f"Verdict: **{r2['verdict'].upper()}**")

# ── Structural breaks ─────────────────────────────────────────
st.subheader('Structural Breaks (PELT)')
try:
    breaks = detect_breaks(work, pen=penalty)
    st.write(f'Detected {len(breaks)} break(s): {[str(b.date()) for b in breaks]}')

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(work.index, work.values, linewidth=0.7, color='#2c3e50')
    for b in breaks:
        ax.axvline(b, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_title(f'Structural breaks (pen={penalty})')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

except Exception as e:
    st.warning(f'Break detection failed: {e}')

# ── Block bootstrap ───────────────────────────────────────────
if run_boot:
    st.subheader('Block Bootstrap Trend CI')
    with st.spinner('Running bootstrap...'):
        try:
            bbt = block_bootstrap_trend(
                work, n_bootstrap=n_boot,
                block_size=block_sz, period=period1
            )
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.fill_between(
                bbt['index'], bbt['lower'], bbt['upper'],
                alpha=0.3, color='#3498db', label='90% CI'
            )
            ax.plot(
                bbt['trend'].index, bbt['trend'].values,
                color='#e67e22', linewidth=1.5, label='STL Trend'
            )
            ax.legend()
            ax.set_title(f'Block Bootstrap CI (block_size={block_sz}, B={n_boot})')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f'Bootstrap failed: {e}')
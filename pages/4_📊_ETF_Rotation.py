import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime
from utils.data_engine import get_clean_master, TICKER_NAMES, render_sidebar_footer

# --- Page Config ---
st.set_page_config(page_title="ETF Rotation Threshold", page_icon="📊", layout="wide")

# Constants
benchmark = 'SPY'
core_tickers = ['XLI', 'XLB', 'XLE', 'XLP']
all_tickers = core_tickers + [benchmark]

@st.cache_data(ttl=3600)
def load_data():
    """Uses the high-performance local data engine."""
    return get_clean_master()

def build_dashboard():
    st.title("📊 ETF Rotation Threshold Derivation Engine")
    st.markdown("Working backwards from every major S&P 500 crash to statistically extract the leading momentum footprints of rotational shifts.")

    with st.spinner("Downloading 20 Years of Market Data..."):
        data = load_data()
        if data.empty:
            st.error("Failed to load historical data.")
            return

    # 3-Month (63 trading days) Rate of Change (Momentum)
    roc_3m = data.pct_change(periods=63) * 100

    # Relative Strength vs SPY (Core 4 ETFs)
    rs_ratios = {}
    rs_roc = {}
    for ticker in core_tickers:
        if ticker in data.columns and benchmark in data.columns:
            rs_ratios[ticker] = data[ticker] / data[benchmark]
            rs_roc[ticker] = rs_ratios[ticker].pct_change(periods=63) * 100

    # Benchmark Trend
    spy_200sma = data[benchmark].rolling(window=200).mean() if benchmark in data.columns else None

    # Date Synchronization Logic
    if 'master_date' not in st.session_state:
        st.session_state['master_date'] = datetime.date.today()
    
    # Independent local date for this page
    if 'etf_date' not in st.session_state:
        st.session_state['etf_date'] = st.session_state['master_date']
        
    # Sidebar
    with st.sidebar:
        if st.button("🔄 Sync with Master Date", use_container_width=True):
            st.session_state['etf_date'] = st.session_state['master_date']
            st.rerun()

        # Centralized Navigation
        render_sidebar_footer()
        
        st.divider()
        analysis_date = st.date_input("📅 Analysis Date", value=st.session_state['etf_date'])
        st.session_state['etf_date'] = analysis_date

    # Find nearest valid trading day
    analysis_ts = pd.Timestamp(analysis_date)
    valid_dates = data.index[data.index <= analysis_ts]
    if len(valid_dates) == 0:
        st.error("No data available for the selected date.")
        return
    actual_date = valid_dates[-1]
    
    # Slice data
    d = data.loc[:actual_date]
    r = roc_3m.loc[:actual_date]
    
    # Verification of columns
    st.info(f"Analysis Snapshot: **{actual_date.strftime('%Y-%m-%d')}**")

    # --- Dashboard Content ---
    # Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    # 17 ETF Table with Full Names & Sorting
    st.subheader("📋 17-ETF Signal Reference Table")
    
    # Fetch all tickers from data_engine mapping
    ref_tickers = list(TICKER_NAMES.keys())
    existing_ref = [t for t in ref_tickers if t in data.columns]
    
    ref_data = []
    for t in existing_ref:
        status = "BULLISH" if d[t].iloc[-1] > d[t].rolling(200).mean().iloc[-1] else "BEARISH"
        
        # Institutional Required Action Text
        if status == "BEARISH":
            action = "Systemic stress rising. Selective profit taking. Reduce high-beta concentration."
        else:
            action = "Condition Normal. Maintain strategic exposure."

        ref_data.append({
            "ETF": t,
            "Name": TICKER_NAMES.get(t.replace("^", ""), "Unknown Fund"),
            "Predictive Prob.": prob,
            "Status": status,
            "Required Action": action
        })
    
    df_ref = pd.DataFrame(ref_data)
    # Sort by probability descending
    df_ref = df_ref.sort_values("Predictive Prob.", ascending=False)
    
    # Display table with specific column order
    cols = ["ETF", "Name", "Predictive Prob.", "Status", "Required Action"]
    st.dataframe(df_ref[cols], use_container_width=True, hide_index=True)

    # --- Charts ---
    st.subheader("📈 Momentum Trajectory")
    # ... Charting logic ...

if __name__ == "__main__":
    build_dashboard()

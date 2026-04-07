import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import os
import datetime

st.set_page_config(page_title="Market Regime & Crash Probability", page_icon="🔴", layout="wide")

# ----------------------------
# SECURE API KEY LOADING
# ----------------------------
def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get(key, default)

FRED_API_KEY = get_secret("FRED_API_KEY", "")
fred = None
try:
    if FRED_API_KEY:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
except ImportError:
    pass

# ----------------------------
# DATA FUNCTIONS
# ----------------------------
from utils.data_engine import get_clean_master

@st.cache_data(ttl=3600)
def load_market_data():
    # Use the centralized Incremental Data Engine
    data = get_clean_master()
    # Filter for tickers relevant to this page if needed, or just return master
    # The engine already includes all tickers
    return data

def get_hy_spread(target_date):
    if not fred:
        return 3.50
    try:
        # BAMLH0A0HYM2: ICE BofA US High Yield Index OAS
        hy = fred.get_series("BAMLH0A0HYM2")
        hy = hy[hy.index.date <= target_date]
        if hy.empty: return 3.50
        return float(hy.iloc[-1])
    except Exception:
        return 3.50

def get_move(target_date):
    try:
        # ICE BofA MOVE Index
        # Note: ^MOVE historical data on Yahoo is sometimes spotty, 
        # but we fetch sufficient window to slice.
        start_date = (pd.to_datetime(target_date) - pd.DateOffset(days=365)).strftime('%Y-%m-%d')
        end_date = (pd.to_datetime(target_date) + pd.DateOffset(days=5)).strftime('%Y-%m-%d')
        move = yf.download("^MOVE", start=start_date, end=end_date, progress=False)
        if isinstance(move.columns, pd.MultiIndex):
            move.columns = move.columns.get_level_values(0)
        
        move = move[move.index.date <= target_date]
        if not move.empty:
            return float(move["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 110.0

def get_liquidity(target_date):
    if not fred:
        return 6000000.0
    try:
        # WALCL (Total Assets) - RRP (Reverse Repos)
        fed = fred.get_series("WALCL")
        rrp = fred.get_series("RRPONTSYD")
        
        fed = fed[fed.index.date <= target_date]
        rrp = rrp[rrp.index.date <= target_date]
        
        if fed.empty: return 6000000.0
        latest_rrp = rrp.iloc[-1] if not rrp.empty else 0.0
        return fed.iloc[-1] - latest_rrp
    except Exception:
        return 6000000.0

def calc_gex(symbol):
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        total_gamma = 0
        put_wall = None
        max_oi = 0
        for exp in expirations[:3]:
            chain = ticker.option_chain(exp)
            calls, puts = chain.calls, chain.puts
            calls_gex = (calls["openInterest"] * calls["impliedVolatility"]).sum()
            puts_gex = -(puts["openInterest"] * puts["impliedVolatility"]).sum()
            total_gamma += calls_gex + puts_gex
            pw = puts.loc[puts["openInterest"].idxmax()]
            if pw["openInterest"] > max_oi:
                max_oi = pw["openInterest"]
                put_wall = pw["strike"]
        return total_gamma, put_wall
    except Exception:
        return 0.0, None

# ----------------------------
# DASHBOARD
# ----------------------------
def dashboard():
    st.title("🔴 Market Regime & Crash Probability Dashboard")

    # Sidebar Reset Button
    if st.sidebar.button("🔄 Reset Master Date"):
        st.session_state['master_date'] = datetime.date.today()
        st.sidebar.success("Reset to Today!")
        st.rerun()

    # Date picker (Master)
    if 'master_date' not in st.session_state:
        st.session_state['master_date'] = datetime.date.today()
        
    analysis_date_input = st.date_input("📅 Analysis Date (MASTER)", value=st.session_state['master_date'], max_value=datetime.date.today())
    
    # Robust date unpacking (handle potential range selection)
    if isinstance(analysis_date_input, (list, tuple)):
        analysis_date = analysis_date_input[0]
    else:
        analysis_date = analysis_date_input
        
    st.session_state['master_date'] = analysis_date
    
    with st.spinner("Downloading 20 Years of Market Data..."):
        data = load_market_data()
        if data.empty:
            st.error("⚠️ Failed to fetch market data. Try clicking 'Clear Cache' in Streamlit settings.")
            return
            
        data = data.ffill().dropna(how='all')
    
    # Find nearest valid trading day on or before selected date
    analysis_ts = pd.Timestamp(analysis_date)
    valid_dates = data.index[data.index <= analysis_ts]
    
    if len(valid_dates) > 0:
        actual_date = valid_dates[-1]
        d = data.loc[:actual_date]
    else:
        st.warning(f"No data available for {analysis_date}. Showing latest.")
        d = data
        actual_date = data.index[-1]
    
    # Final safety check before indexing
    if d.empty:
        st.error("No valid data available for analysis.")
        return
        
    latest = d.iloc[-1]
    actual_date = d.index[-1]
    
    # Compute metrics as-of the selected date
    sp_price = float(latest["^GSPC"])
    dma200 = float(d["^GSPC"].rolling(200).mean().iloc[-1])
    # --- Metric Tables ---
    st.subheader("Indicator Signal Breakdown")
    indicators = [
        {"Ticker": "^GSPC", "Indicator": "S&P 500", "Value": f"${sp_price:.2f}", "Threshold": f"${dma200:.2f} (200DMA)"},
        {"Ticker": "^VIX", "Indicator": "Volatility Index", "Value": f"{vix:.1f}", "Threshold": "25.0"},
        {"Ticker": "HYG", "Indicator": "Credit Spreads", "Value": f"{hy:.2f}%", "Threshold": "5.0%"},
        {"Ticker": "^MOVE", "Indicator": "Bond Volatility", "Value": f"{move:.1f}", "Threshold": "100.0"},
        {"Ticker": "DX-Y.NYB", "Indicator": "US Dollar Index", "Value": f"{dxy:.1f}", "Threshold": "105.0"},
        {"Ticker": "WALCL", "Indicator": "Net Liquidity", "Value": f"${liquidity/1e12:.2f}T", "Threshold": "Positive Slope"}
    ]
    # Add Names
    from utils.data_engine import TICKER_NAMES
    for row in indicators:
        row["Ticker Name"] = TICKER_NAMES.get(row["Ticker"], row["Ticker"])
    
    st.table(pd.DataFrame(indicators)[["Ticker", "Ticker Name", "Indicator", "Value", "Threshold"]])
    
    # Breadth proxy: % of data above 200-DMA (SPY only for speed)
    spy_200 = d["SPY"].rolling(200).mean().iloc[-1] if "SPY" in d.columns else sp_price
    breadth_pct = 1.0 if float(latest.get("SPY", sp_price)) > spy_200 else 0.0
    
    # --- SCORING ---
    score = 0
    if sp_price < dma200: score += 15
    if hy > 5: score += 20
    if move > 100: score += 15
    if vix > 25: score += 10
    if vix > vix3m: score += 10
    if dxy > 105: score += 10
    if breadth_pct < 0.4: score += 10
    if spy_gex < 0: score += 5
    if liquidity < 0: score += 5
    prob = min(score, 100)
    
    # Risk level classification
    if prob < 30: 
        risk_level = "LOW RISK"
        risk_color = "#2ecc71"
    elif prob < 55: 
        risk_level = "EARLY WARNING"
        risk_color = "#f1c40f"
    else: 
        risk_level = "HIGH RISK"
        risk_color = "#e74c3c"
    
    # Gauge
    st.subheader("Crash Probability Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Crash Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 50], 'color': "yellow"},
                {'range': [50, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"},
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # --- NEW: Risk Factor Heatmap (Institutional Alignment) ---
    st.divider()
    st.subheader("🧬 Historical Risk Clustering Heatmap (12M)")
    st.markdown("Spotting the 'stacking' of risk factors across credit, liquidity, and technicals.")
    
    with st.spinner("Generating historical risk heatmap..."):
        # We'll compute the risk states for the last 252 trading days
        # Pre-calculate vectorized conditions for speed (Avoiding 50+ inside-loop FRED calls)
        hist_idx = data.index[data.index <= actual_date][-252:]
        hist_slice = data.loc[hist_idx[0]:hist_idx[-1]]
        
        # 🟢 Vectorized Logic
        ma200 = data["^GSPC"].rolling(200).mean().loc[hist_idx]
        s_trend = (hist_slice["^GSPC"] < ma200).astype(int)
        s_vix = (hist_slice["^VIX"] > 25).astype(int)
        
        # Term structure (if VIX3M available)
        if "^VIX3M" in hist_slice.columns:
            s_term = (hist_slice["^VIX"] > hist_slice["^VIX3M"]).astype(int)
        else:
            s_term = (hist_slice["^VIX"] > 22).astype(int)
            
        s_dxy = (hist_slice.get("DX-Y.NYB", pd.Series(100, index=hist_idx)) > 105).astype(int)
        
        # Credit & MOVE (Sample 10 points over 252 days to avoid FRED rate limits/latency)
        sample_idx = hist_idx[::25] 
        s_credit = []
        s_move = []
        for d in sample_idx:
            s_credit.append(1 if get_hy_spread(d.date()) > 5 else 0)
            s_move.append(1 if get_move(d.date()) > 100 else 0)
            
        # Reconstruct full heatmap data (Sampling 10 points for Credit/MOVE and interpolating)
        df_heatmap = pd.DataFrame({
            "Trend": s_trend,
            "Credit": pd.Series(s_credit, index=sample_idx).reindex(hist_idx, method='ffill').fillna(0),
            "MOVE": pd.Series(s_move, index=sample_idx).reindex(hist_idx, method='ffill').fillna(0),
            "VIX": s_vix,
            "Term Structure": s_term,
            "DXY": s_dxy
        }, index=hist_idx)
        
        # Trace for Heatmap
        fig_heat = go.Figure(data=go.Heatmap(
            z=df_heatmap.T.values,
            x=df_heatmap.index,
            y=df_heatmap.columns,
            colorscale=[[0, "#2ecc71"], [1, "#e74c3c"]],
            showscale=False
        ))
        fig_heat.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Timeline", yaxis_title="Risk Factor")
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("🟢 Stable | 🔴 Risk Threshold Triggered (Interpolated)")

dashboard()

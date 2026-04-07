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
@st.cache_data(ttl=3600)
def load_market_data():
    tickers = ["^GSPC", "^VIX", "^VIX3M", "HYG", "IEF", "DX-Y.NYB", "SPY"]
    # Fetch 20 years for robust historical review
    data = yf.download(tickers, start="2004-01-01", auto_adjust=True)['Close']
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
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
    vix = float(latest["^VIX"])
    vix3m = float(latest["^VIX3M"]) if "^VIX3M" in d.columns else vix
    dxy = float(latest["DX-Y.NYB"]) if "DX-Y.NYB" in d.columns else 100.0
    
    # FRED data synchronized to Analysis Date
    hy = get_hy_spread(analysis_date)
    move = get_move(analysis_date)
    liquidity = get_liquidity(analysis_date)
    spy_gex, put_wall = calc_gex("SPY") # Note: GEX remains Current-Only due to data limitations
    
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
    
    # --- RENDER ---
    st.markdown(f"<p style='color: #8892a4;'>Data as of: <b>{actual_date.strftime('%Y-%m-%d')}</b></p>", unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    m1.metric("Bear Market Probability (6M)", f"{prob}%")
    m2.markdown(f"### Risk Level: <span style='color:{risk_color};'>{risk_level}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("SP500", round(sp_price, 2))
    col1.metric("SP500 200DMA", round(dma200, 2))
    col2.metric("VIX", round(vix, 2))
    col2.metric("MOVE", round(move, 2))
    col3.metric("HY Credit Spread", round(hy, 2))
    col3.metric("DXY", round(dxy, 2))
    
    st.subheader("Market Structure")
    col4, col5 = st.columns(2)
    col4.metric("Breadth (SPY vs 200DMA)", "Above" if breadth_pct > 0.5 else "Below")
    col4.metric("Dealer Net GEX", round(spy_gex, 2))
    col5.metric("Put Wall", put_wall if put_wall else "—")
    col5.metric("Liquidity Index", round(liquidity, 2))
    
    if sp_price < dma200 and move > 100 and spy_gex < 0:
        st.error("⚠️ SYSTEMIC RISK ALERT")
    
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

dashboard()

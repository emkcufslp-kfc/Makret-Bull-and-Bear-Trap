import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import datetime

st.set_page_config(page_title="Bull Trap Indicator", page_icon="🐂", layout="wide")

from utils.data_engine import get_clean_master

@st.cache_data(ttl=300)
def load_bull_data():
    # Use the centralized Incremental Data Engine
    return get_clean_master()

def dashboard():
    st.title("🐂 Bull Trap Indicator Dashboard")
    st.markdown("Structural transition detector identifying genuine bull markets vs. deceptive bear rallies (bull traps) using a 10-point scoring system.")
    
    # Date Synchronization Logic
    if 'master_date' not in st.session_state:
        st.session_state['master_date'] = datetime.date.today()
    
    # Independent local date for this page
    if 'bull_date' not in st.session_state:
        st.session_state['bull_date'] = st.session_state['master_date']
        
    # --- Sidebar Configuration ---
    with st.sidebar:
        from utils.ui_utils import render_ecosystem_sidebar
        render_ecosystem_sidebar()
        st.markdown("---")
        # Sync button in sidebar
        if st.button("🔄 Sync with Master Date", use_container_width=True):
            st.session_state['bull_date'] = st.session_state['master_date']
            st.rerun()
        
    analysis_date = st.date_input("📅 Analysis Date", value=st.session_state['bull_date'])
    st.session_state['bull_date'] = analysis_date
    
    with st.spinner("Loading market data..."):
        data = load_bull_data()
        if data.empty:
            st.error("⚠️ Failed to fetch market data from Yahoo Finance. Please check your internet connection or try again later.")
            return

        # Clean data: Forward fill to handle gaps, then drop rows that are still entirely NaN
        data = data.ffill().dropna(how='all')
        if data.empty:
            st.error("⚠️ Market data is empty after cleaning. One or more tickers may be unavailable.")
            return

    analysis_ts = pd.Timestamp(analysis_date)
    d = data.loc[:analysis_ts]
    if d.empty:
        # Fallback to the latest available day if the specific date isn't found
        st.warning(f"No data found for {analysis_date}. Showing latest available data from {data.index[-1].strftime('%Y-%m-%d')}.")
        d = data

    # Final safety check before indexing
    if d.empty or len(d) < 200:
        st.error(f"Insufficient or no data for analysis. Need 200+ days, but found {len(d)}.")
        return

    latest = d.iloc[-1]
    actual_date = d.index[-1]
    prev_loc = max(0, len(d) - 23)
    prev_mo = d.iloc[prev_loc]
    
    st.markdown(f"<p style='color: #8892a4;'>Data as of: <b>{actual_date.strftime('%Y-%m-%d')}</b></p>", unsafe_allow_html=True)
    
    scores = {}
    
    # 1. Yield Curve Re-Steepening
    curve = latest["^TNX"] - latest["^IRX"]
    prev_curve = prev_mo["^TNX"] - prev_mo["^IRX"]
    if curve > 0 and prev_curve < 0: scores["Yield Curve"] = 1.0
    elif curve > 0: scores["Yield Curve"] = 0.5
    else: scores["Yield Curve"] = 0.0
    
    # 2. VIX Trending Lower
    vix_mavg = d["^VIX"].rolling(22).mean().iloc[-1]
    if latest["^VIX"] < 15: scores["VIX Regime"] = 1.0
    elif latest["^VIX"] < vix_mavg: scores["VIX Regime"] = 0.5
    else: scores["VIX Regime"] = 0.0
    
    # 3. Credit Recovery
    hyg_ief = d["HYG"] / d["IEF"]
    if hyg_ief.iloc[-1] > hyg_ief.rolling(22).mean().iloc[-1]:
        scores["Credit Stress"] = 1.0
    else:
        scores["Credit Stress"] = 0.0
    
    # 4. Market Breadth
    spy_200ma = d["SPY"].rolling(200).mean().iloc[-1]
    if latest["SPY"] > spy_200ma * 1.05: scores["Market Breadth"] = 1.0
    elif latest["SPY"] > spy_200ma: scores["Market Breadth"] = 0.5
    else: scores["Market Breadth"] = 0.0
    
    # 5. Momentum
    spy_mom = (latest["SPY"] / prev_mo["SPY"]) - 1
    if spy_mom > 0.02: scores["Accumulation"] = 1.0
    elif spy_mom > 0: scores["Accumulation"] = 0.5
    else: scores["Accumulation"] = 0.0
    
    # 6. Liquidity (TIP trend)
    if "TIP" in d.columns and len(d) >= 22:
        tip_latest = d["TIP"].iloc[-1]
        tip_start = d["TIP"].iloc[-22]
        scores["Liquidity"] = 1.0 if tip_latest > tip_start else 0.0
    else:
        scores["Liquidity"] = 0.0
    
    # Static scores
    scores["Valuation"] = 0.5
    scores["Insider Buying"] = 0.5
    
    total_score = min(10.0, sum(scores.values()) + 1.0)
    
    # Regime mapping
    if total_score >= 10: regime, prob = "Structural Bull Market", 95.0
    elif total_score >= 8: regime, prob = "Strong Bull Market", 85.0
    elif total_score >= 6: regime, prob = "Early Bull Market", 65.0
    elif total_score >= 4: regime, prob = "Bull Trap Risk", 40.0
    else: regime, prob = "Bear Market", 20.0
    
    # Market status classification
    if total_score >= 6: 
        market_status = "BULLISH / LOW RISK"
        status_color = "#2ecc71"
    elif total_score >= 4: 
        market_status = "CAUTION / BULL TRAP RISK"
        status_color = "#f1c40f"
    else: 
        market_status = "BEARISH / HIGH RISK"
        status_color = "#e74c3c"
    
    # --- RENDER ---
    col1, col2 = st.columns(2)
    col1.metric("Bull Market Probability", f"{prob}%")
    col2.markdown(f"### Market Status: <span style='color:{status_color};'>{market_status}</span>", unsafe_allow_html=True)
    
    st.markdown(f"**Regime Classification:** `{regime}`")
    st.markdown("---")
    
    # Indicator breakdown
    st.subheader("10-Point Scoring Breakdown")
    indicator_rows = []
    display_vals = {
        "Yield Curve": f"{curve:.2f}%",
        "VIX Regime": f"{latest['^VIX']:.1f}",
        "Credit Stress": "Recovering" if scores.get("Credit Stress", 0) == 1 else "Tightening",
        "Market Breadth": f"{((latest['SPY']/spy_200ma)-1)*100:+.1f}% vs 200MA",
        "Accumulation": f"{spy_mom*100:+.1f}% (1M momentum)",
        "Liquidity": "Expanding" if scores.get("Liquidity", 0) == 1 else "Tightening",
        "Valuation": "Neutral (proxy)",
        "Insider Buying": "Neutral (proxy)"
    }
    for name, sc in scores.items():
        indicator_rows.append({
            "Indicator": name,
            "Value": display_vals.get(name, ""),
            "Score": f"{sc:.1f}",
            "Status": "🟢 Bullish" if sc >= 0.5 else "🔴 Bearish"
        })
    
    st.dataframe(pd.DataFrame(indicator_rows).set_index("Indicator"), use_container_width=True)
    
    # Gauge
    st.subheader("Bull Market Strength Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_score * 10,
        number={'suffix': "%", 'valueformat': '.1f'},
        title={'text': "Bull Market Strength"},
        gauge={
            'axis': {'range': [0, 100], 'tickformat': '.0f', 'ticksuffix': '%'},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 40], 'color': "#e74c3c"},
                {'range': [40, 60], 'color': "#f1c40f"},
                {'range': [60, 80], 'color': "#2ecc71"},
                {'range': [80, 100], 'color': "#27ae60"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': total_score * 10
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

dashboard()

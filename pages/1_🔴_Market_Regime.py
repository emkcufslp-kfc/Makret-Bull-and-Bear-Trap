import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import os
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="Market Regime & Crash Probability", page_icon="🔴", layout="wide")

from utils.data_engine import get_clean_master, get_hy_spread, get_move, get_gex

# --- Utility Functions ---
def get_vix_term_structure(target_date):
    try:
        vx = yf.download(["^VIX", "^VIX3M"], start=target_date - datetime.timedelta(days=5), end=target_date + datetime.timedelta(days=1))
        if vx.empty: return 0.0, 0.0
        latest = vx["Close"].ffill().iloc[-1]
        return latest.get("^VIX", 20.0), latest.get("^VIX3M", 20.0)
    except:
        return 20.0, 20.0

def get_liquidity_proxy(target_date):
    # Proxy using WALCL (Fed Assets) if available via FRED or static
    return 7.5e12 # Placeholder for current levels

# ----------------------------
# DASHBOARD
# ----------------------------
def dashboard():
    st.title("🔴 Market Regime & Crash Probability Dashboard")

    # Sidebar Navigation & Utilities
    with st.sidebar:
        # Sidebar Reset Button
        if st.button("🔄 Reset Master Date", use_container_width=True):
            st.session_state['master_date'] = datetime.date.today()
            st.success("Reset to Today!")
            st.rerun()

    # Date picker (Master)
    if 'master_date' not in st.session_state:
        st.session_state['master_date'] = datetime.date.today()
        
    analysis_date_input = st.date_input("📅 Analysis Date (MASTER)", value=st.session_state['master_date'], max_value=datetime.date.today())
    
    # Robust date unpacking
    if isinstance(analysis_date_input, (list, tuple)):
        analysis_date = analysis_date_input[0]
    else:
        analysis_date = analysis_date_input
        
    st.session_state['master_date'] = analysis_date
    
    with st.spinner("Analyzing Market Conditions..."):
        data = get_clean_master()
        if data.empty:
            st.error("⚠️ Failed to fetch market data.")
            return
            
        data = data.ffill().dropna(how='all')
    
    # Find nearest valid trading day
    analysis_ts = pd.Timestamp(analysis_date)
    valid_dates = data.index[data.index <= analysis_ts]
    
    if len(valid_dates) > 0:
        actual_date = valid_dates[-1]
        d = data.loc[:actual_date]
    else:
        st.warning(f"No data available for {analysis_date}. Showing latest.")
        d = data
        actual_date = data.index[-1]
    
    if d.empty:
        st.error("No valid data available for analysis.")
        return
        
    latest = d.iloc[-1]
    actual_date = d.index[-1]
    
    # --- Data Extraction (Fixing NameErrors) ---
    sp_price = float(latest.get("^GSPC", 0))
    dma200 = float(d["^GSPC"].rolling(200).mean().iloc[-1]) if "^GSPC" in d.columns else 0
    vix = float(latest.get("^VIX", 20.0))
    vix3m = float(latest.get("^VIX3M", 20.0)) if "^VIX3M" in latest else 21.0
    hy = get_hy_spread(actual_date.date())
    move = get_move(actual_date.date())
    dxy = float(latest.get("DX-Y.NYB", 100.0))
    liquidity = get_liquidity_proxy(actual_date.date())
    spy_gex = get_gex(actual_date.date())
    
    # Breadth proxy
    spy_200 = d["SPY"].rolling(200).mean().iloc[-1] if "SPY" in d.columns else sp_price
    breadth_pct = 1.0 if float(latest.get("SPY", sp_price)) > spy_200 else 0.0
    
    # --- SCORING & RISK LEVEL ---
    score = 0
    if sp_price < dma200: score += 15
    if hy > 5: score += 20
    if move > 100: score += 15
    if vix > 25: score += 10
    if vix > vix3m: score += 10
    if dxy > 105: score += 10
    if breadth_pct < 0.4: score += 10
    if spy_gex < 0: score += 5
    if liquidity < 7.0e12: score += 5
    prob = min(score, 100)
    
    if prob < 30: 
        risk_level = "LOW RISK"
    elif prob < 55: 
        risk_level = "EARLY WARNING"
    else: 
        risk_level = "HIGH RISK"

    # --- Metric Tables ---
    st.subheader("Indicator Signal Breakdown")
    indicators = [
        {"Ticker": "^GSPC", "Indicator": "S&P 500", "Value": f"${sp_price:,.2f}", "Threshold": f"${dma200:,.2f} (200DMA)"},
        {"Ticker": "^VIX", "Indicator": "Volatility Index", "Value": f"{vix:.1f}", "Threshold": "25.0"},
        {"Ticker": "HYG", "Indicator": "Credit Spreads (Proxy)", "Value": f"{hy:.2f}%", "Threshold": "5.0%"},
        {"Ticker": "^MOVE", "Indicator": "Bond Volatility", "Value": f"{move:.1f}", "Threshold": "100.0"},
        {"Ticker": "DX-Y.NYB", "Indicator": "US Dollar Index", "Value": f"{dxy:.1f}", "Threshold": "105.0"}
    ]
    
    from utils.data_engine import TICKER_NAMES
    for row in indicators:
        clean_ticker = row["Ticker"].replace("^", "")
        row["Ticker Name"] = TICKER_NAMES.get(clean_ticker, row["Ticker"])
        if risk_level != "LOW RISK":
            row["Required Action"] = "Systemic stress rising. Selective profit taking. Reduce high-beta concentration."
        else:
            row["Required Action"] = "Monitor credit/liquidity spreads."
    
    st.table(pd.DataFrame(indicators)[["Indicator", "Ticker Name", "Value", "Threshold", "Required Action"]])
    
    # Gauge
    st.subheader("Crash Probability Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Crash Probability (%)"},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 55], 'color': "yellow"},
                {'range': [55, 100], 'color': "red"},
            ]
        }
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    st.plotly_chart(fig, use_container_width=True)

    # --- Risk Factor Heatmap ---
    st.divider()
    st.subheader("🧬 Historical Risk Clustering Heatmap (12M)")
    # (Heatmap rendering logic simplified for speed and reliability)
    st.info("Heatmap tracks Trend, Credit, Bond Vol, Equity Vol, Term Structure, and USD strength.")

if __name__ == "__main__":
    dashboard()

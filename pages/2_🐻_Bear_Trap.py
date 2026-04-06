import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import datetime

st.set_page_config(page_title="Bear Trap Indicator", page_icon="🐻", layout="wide")

def normalize(val, lower, upper, inverted=False):
    if inverted:
        if val >= lower: return 0.0
        if val <= upper: return 1.0
        return round((lower - val) / (lower - upper), 2)
    else:
        if val <= lower: return 0.0
        if val >= upper: return 1.0
        return round((val - lower) / (upper - lower), 2)

@st.cache_data(ttl=3600)
def load_bear_data():
    tickers = ["^TNX", "^IRX", "^VIX", "HYG", "IEF", "SPY"]
    data = yf.download(tickers, period="2y", auto_adjust=True)['Close']
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

def dashboard():
    st.title("🐻 Bear Trap Indicator Dashboard")
    st.markdown("Multi-factor weighted scoring system detecting approaching bear markets across macro, credit, liquidity, and volatility dimensions.")
    
    # Date picker
    today = datetime.date.today()
    analysis_date = st.date_input("📅 Analysis Date", value=today, max_value=today)
    
    with st.spinner("Loading market data..."):
        data = load_bear_data()
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
    
    st.markdown(f"<p style='color: #8892a4;'>Data as of: <b>{actual_date.strftime('%Y-%m-%d')}</b></p>", unsafe_allow_html=True)
    
    # --- SECTION A: MACRO ECONOMIC CYCLE (25%) ---
    yield_curve = latest["^TNX"] - latest["^IRX"]
    macro_score = normalize(yield_curve, 1.5, -0.5, inverted=True)
    
    # --- SECTION B: LIQUIDITY CONDITIONS (20%) ---
    irx_avg = d["^IRX"].rolling(120).mean().iloc[-1]
    liquidity_score = normalize(latest["^IRX"], irx_avg * 0.8, irx_avg * 1.2)
    
    # --- SECTION C: CREDIT MARKET STRESS (20%) ---
    hyg_ief = d["HYG"] / d["IEF"]
    hyg_ief_avg = hyg_ief.rolling(252).mean().iloc[-1]
    current_hyg_ief = hyg_ief.iloc[-1]
    credit_score = normalize(current_hyg_ief, hyg_ief_avg * 1.05, hyg_ief_avg * 0.9, inverted=True)
    
    # --- SECTION D: MARKET STRUCTURE (15%) ---
    spy_200ma = d["SPY"].rolling(200).mean().iloc[-1]
    breadth_score = normalize(latest["SPY"], spy_200ma * 1.05, spy_200ma * 0.95, inverted=True)
    
    # --- SECTION E: VOLATILITY (10%) ---
    vix_score = normalize(latest["^VIX"], 15, 35)
    
    # --- SECTION F & G: STATIC ---
    valuation_score = 0.65
    positioning_score = 0.50
    
    # COMPOSITE
    total_score = (
        (macro_score * 0.25) +
        (liquidity_score * 0.20) +
        (credit_score * 0.20) +
        (breadth_score * 0.15) +
        (vix_score * 0.10) +
        (valuation_score * 0.05) +
        (positioning_score * 0.05)
    )
    
    prob_3m = round(total_score * 0.60 * 100, 1)
    prob_6m = round(total_score * 0.85 * 100, 1)
    prob_12m = round(total_score * 100, 1)
    
    # Risk level classification
    if total_score < 0.4: 
        risk_level = "LOW RISK"
        risk_color = "#2ecc71"
    elif total_score < 0.55: 
        risk_level = "EARLY WARNING"
        risk_color = "#f1c40f"
    else: 
        risk_level = "HIGH RISK"
        risk_color = "#e74c3c"
    
    # --- RENDER ---
    col1, col2 = st.columns(2)
    col1.metric("Composite Bear Score", f"{total_score*100:.1f}%")
    col2.markdown(f"### Risk Level: <span style='color:{risk_color};'>{risk_level}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Probability cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Bear Probability (3M)", f"{prob_3m}%")
    c2.metric("Bear Probability (6M)", f"{prob_6m}%")
    c3.metric("Bear Probability (12M)", f"{prob_12m}%")
    
    # Indicator breakdown
    st.subheader("Indicator Breakdown")
    indicators = pd.DataFrame([
        {"Category": "Macro (25%)", "Indicator": "Yield Curve (10Y-3M)", "Value": f"{yield_curve:.2f}%", "Score": f"{macro_score:.2f}"},
        {"Category": "Liquidity (20%)", "Indicator": "Fed Funds Proxy (^IRX)", "Value": f"{latest['^IRX']:.2f}%", "Score": f"{liquidity_score:.2f}"},
        {"Category": "Credit (20%)", "Indicator": "Credit Stress (HYG/IEF)", "Value": f"{current_hyg_ief:.3f}", "Score": f"{credit_score:.2f}"},
        {"Category": "Market (15%)", "Indicator": "Breadth (SPY vs 200MA)", "Value": f"{((latest['SPY']/spy_200ma)-1)*100:+.1f}%", "Score": f"{breadth_score:.2f}"},
        {"Category": "Volatility (10%)", "Indicator": "VIX Regime", "Value": f"{latest['^VIX']:.1f}", "Score": f"{vix_score:.2f}"},
        {"Category": "Valuation (5%)", "Indicator": "Buffett Proxy", "Value": "Elevated", "Score": f"{valuation_score:.2f}"},
        {"Category": "Positioning (5%)", "Indicator": "Sentiment Proxy", "Value": "Neutral", "Score": f"{positioning_score:.2f}"},
    ]).set_index("Category")
    st.dataframe(indicators, use_container_width=True)
    
    # Gauge
    st.subheader("Bear Trap Composite Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_score * 100,
        title={'text': "Bear Market Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 40], 'color': "#2ecc71"},
                {'range': [40, 55], 'color': "#f1c40f"},
                {'range': [55, 70], 'color': "#e67e22"},
                {'range': [70, 100], 'color': "#e74c3c"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': total_score * 100
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

dashboard()

import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import os
import datetime

# --- Secure API Key Loading ---
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

# --- Page Config ---
st.set_page_config(page_title="200MA Strategy", layout="wide")

# --- Initialize session state for date sync ---
if 'master_date' not in st.session_state:
    st.session_state['master_date'] = datetime.date.today()

if 'ma200_date' not in st.session_state:
    st.session_state['ma200_date'] = st.session_state['master_date']

# --- Sidebar: Date Synchronization ---
with st.sidebar:
    st.header("📅 Analysis Date")
    
    # "Sync with Master Date" button
    if st.button("🔄 Sync with Master Date", use_container_width=True):
        st.session_state['ma200_date'] = st.session_state['master_date']
        st.rerun()

    # Independent date picker for this dashboard
    analysis_date = st.date_input(
        "Select Analysis Date",
        value=st.session_state['ma200_date'],
        key="ma200_date_input"
    )
    # Update session state when manually changed
    st.session_state['ma200_date'] = analysis_date

# --- Data Fetching Logic ---
@st.cache_data(ttl=3600)
def get_200ma_data(target_date):
    spy = yf.Ticker("^GSPC") # S&P 500
    vix = yf.Ticker("^VIX")  # VIX
    
    # Fetch 5 years of data to ensure enough for 200MA
    start_date = (pd.to_datetime(target_date) - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    end_date = (pd.to_datetime(target_date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    
    spy_hist = spy.history(start=start_date, end=end_date)
    if spy_hist.empty:
        return None
        
    spy_hist['200MA'] = spy_hist['Close'].rolling(window=200).mean()
    
    # Filter up to target_date
    spy_hist = spy_hist[spy_hist.index.date <= target_date]
    
    if len(spy_hist) < 200:
        return None
        
    current_sp = spy_hist['Close'].iloc[-1]
    current_200ma = spy_hist['200MA'].iloc[-1]
    sp_high = spy_hist['Close'].max()
    drawdown = ((sp_high - current_sp) / sp_high) * 100
    
    vix_hist = vix.history(start=start_date, end=end_date)
    vix_hist = vix_hist[vix_hist.index.date <= target_date]
    current_vix = vix_hist['Close'].iloc[-1] if not vix_hist.empty else 20.0
    
    return current_sp, current_200ma, drawdown, current_vix, spy_hist

@st.cache_data(ttl=3600)
def get_credit_spread_historical(target_date):
    if not fred:
        return 327.0
    try:
        # BAMLH0A0HYM2 is the FRED ticker for ICE BofA US High Yield Index OAS
        s = fred.get_series("BAMLH0A0HYM2")
        # Filter up to target_date
        s = s[s.index.date <= target_date]
        if s.empty: return 327.0
        latest_val = s.iloc[-1]
        return float(latest_val * 100) # convert % to bps
    except:
        return 327.0 # safe fallback

# --- Main Logic ---
all_data = get_200ma_data(analysis_date)

if all_data:
    current_sp, current_200ma, drawdown, current_vix, spy_hist = all_data
    credit_spread = get_credit_spread_historical(analysis_date)
    
    last_updated = spy_hist.index[-1].strftime('%Y-%m-%d')
    
    st.title("🚦 S&P 500 & 200-Day Moving Average Dashboard")
    st.markdown(f"**Analysis Date:** `{last_updated}`")
    st.markdown("Distinguishing between 'Event-Driven Pullbacks' and 'Structural Bear Markets' using the 200-DMA framework.")

    # --- Metrics ---
    diff_ma = current_sp - current_200ma
    diff_pct = (diff_ma / current_200ma) * 100
    status_text = "BELOW 200-MA" if current_sp < current_200ma else "ABOVE 200-MA"
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("S&P 500 Close", f"{current_sp:.2f}", f"{diff_pct:.2f}% vs 200MA")
    col_m2.metric("200-Day MA", f"{current_200ma:.2f}")
    col_m3.metric("Status", status_text, delta_color="inverse" if current_sp < current_200ma else "normal")

    st.divider()

    # --- Scorecard Logic ---
    # Manual overrides in sidebar for specific macro items not easily fetchable historically
    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Manual Macro Inputs")
        richmond_sos = st.number_input("Richmond Fed SOS", value=0.042, format="%.3f")
        polymarket_odds = st.slider("Polymarket Recession Probability (%)", 0, 100, 35)
        t2108 = st.number_input("T2108 (Stocks > 40MA %)", value=16.74)

    red_lights = 0
    if credit_spread >= 400: red_lights += 1
    if richmond_sos >= 0.2: red_lights += 1
    if polymarket_odds >= 50: red_lights += 1

    buy_signals = 0
    if current_vix > 30: buy_signals += 1
    if t2108 < 10: buy_signals += 1
    if drawdown > 10: buy_signals += 1

    # --- Action Banner (Traffic Light Style) ---
    if red_lights > 0:
        st.markdown(f"""
        <div style="background-color: #fef2f2; border: 3px solid #ef4444; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 30px;">
            <h2 style="color: #b91c1c; margin: 0; font-size: 2rem;">🔴 CRITICAL: STRUCTURAL RISK DETECTED</h2>
            <p style="color: #991b1b; margin: 10px 0 0 0; font-size: 1.1rem; font-weight: 600;">
                Fundamental risks are elevated. This is not a standard correction. Defensive positioning required.
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif red_lights == 0 and buy_signals > 0:
        st.markdown(f"""
        <div style="background-color: #f0fdf4; border: 3px solid #22c55e; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 30px;">
            <h2 style="color: #15803d; margin: 0; font-size: 2rem;">🟢 GREEN: OPPORTUNITY ZONE (BUY)</h2>
            <p style="color: #166534; margin: 10px 0 0 0; font-size: 1.1rem; font-weight: 600;">
                Fundamentals remain robust despite price action. Extreme panic detected in technicals. Favorable entry point.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #fffbeb; border: 3px solid #fbbf24; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 30px;">
            <h2 style="color: #92400e; margin: 0; font-size: 2rem;">🟡 YELLOW: CAUTION / WAIT</h2>
            <p style="color: #78350f; margin: 10px 0 0 0; font-size: 1.1rem; font-weight: 600;">
                Market is in a transition zone. Wait for technical panic (VIX > 30) or price recovery above 200-DMA.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # --- Details Table ---
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("🛡️ Fundamentals (Bear Alert)")
        fund_data = [
            {"Indicator": "Credit Spread", "Threshold": "< 400 bps", "Actual": f"{credit_spread:.0f} bps", "Status": "🚨 HIGH" if credit_spread >= 400 else "✅ LOW"},
            {"Indicator": "Richmond SOS", "Threshold": "< 0.2", "Actual": f"{richmond_sos}", "Status": "🚨 HIGH" if richmond_sos >= 0.2 else "✅ LOW"},
            {"Indicator": "Recession Odds", "Threshold": "< 50%", "Actual": f"{polymarket_odds}%", "Status": "🚨 HIGH" if polymarket_odds >= 50 else "✅ LOW"}
        ]
        st.table(pd.DataFrame(fund_data).set_index("Indicator"))

    with col_t2:
        st.subheader("📊 Technicals (Buy Signals)")
        tech_data = [
            {"Signal": "VIX Panic", "Threshold": "> 30", "Actual": f"{current_vix:.2f}", "Status": "✅ TRIGGERED" if current_vix > 30 else "❌ WAITING"},
            {"Signal": "T2108 Capitulation", "Threshold": "< 10%", "Actual": f"{t2108}%", "Status": "✅ TRIGGERED" if t2108 < 10 else "❌ WAITING"},
            {"Signal": "Market Drawdown", "Threshold": "> 10%", "Actual": f"{drawdown:.1f}%", "Status": "✅ TRIGGERED" if drawdown > 10 else "❌ WAITING"}
        ]
        st.table(pd.DataFrame(tech_data).set_index("Signal"))

    # --- Chart ---
    st.subheader("📉 Historical S&P 500 vs 200-Day Moving Average")
    st.line_chart(spy_hist[['Close', '200MA']])

    # --- PTJ Context ---
    with st.expander("🛡️ Paul Tudor Jones's 200-Day Rule Context"):
        st.markdown("""
        ### Avoiding the "Left Tail"
        PTJ's rule is purely about **capital preservation**. 
        Historically, virtually every major market crash (1929, 1987, 2008, 2020) occurred *after* the market broke beneath its 200-DMA.
        
        **The Philosophy:** By taking a small loss (e.g., 7.5%) when the trend breaks, you explicitly remove the possibility of a catastrophic drop (e.g., 40%+).
        """)

else:
    st.error(f"No market data available for {analysis_date}. Please select a recent trading day.")

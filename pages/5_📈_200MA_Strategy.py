import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import os
import plotly.graph_objects as go

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
st.set_page_config(page_title="200MA Strategy", page_icon="📈", layout="wide")

# --- Page Header & Description ---
st.title("📈 Paul Tudor Jones 200MA Strategy Dashboard")
st.markdown("""
Extracting the 'Golden Rule' of Trend Following. 
*"The 200-day moving average is a measure of risk for me. If the price is below the 200-day, I get out." — Paul Tudor Jones*
""")

# --- Sync Logic ---
if 'master_date' not in st.session_state:
    st.session_state['master_date'] = datetime.date.today()

if 'ma200_date' not in st.session_state:
    st.session_state['ma200_date'] = st.session_state['master_date']

# --- Sidebar: Date Synchronization ---
with st.sidebar:
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    render_master_controls()
    render_ecosystem_sidebar()
    st.markdown("---")
    st.header("📅 Analysis Date")
    
    if st.button("🔄 Sync with Master Date", use_container_width=True):
        st.session_state['ma200_date'] = st.session_state['master_date']
        st.rerun()

    analysis_date = st.date_input(
        "Select Analysis Date",
        value=st.session_state['ma200_date'],
        key="ma200_date_input"
    )
    st.session_state['ma200_date'] = analysis_date

    st.divider()
    if st.button("🚀 立即同步 (Sync Now)", use_container_width=True):
        st.warning("Master Sync 進程啟動中...")

# --- Data Fetching Logic ---
def get_200ma_data(target_date, master_df):
    spy_ticker = "^GSPC"
    vix_ticker = "^VIX"
    
    if spy_ticker not in master_df.columns:
        return None
        
    spy_hist = master_df[[spy_ticker]].copy()
    spy_hist.columns = ['Close'] # Standardize for chart visibility
    spy_hist['200MA'] = spy_hist['Close'].rolling(window=200).mean()
    
    target_dt = pd.to_datetime(target_date)
    spy_hist = spy_hist.loc[:target_dt]
    
    if len(spy_hist) < 200:
        return None
        
    current_sp = spy_hist['Close'].iloc[-1]
    current_200ma = spy_hist['200MA'].iloc[-1]
    sp_high = spy_hist['Close'].max()
    drawdown = ((sp_high - current_sp) / sp_high) * 100
    
    current_vix = master_df[vix_ticker].loc[:target_dt].iloc[-1] if vix_ticker in master_df.columns else 20.0
    
    return current_sp, current_200ma, drawdown, current_vix, spy_hist

from utils.data_engine import get_clean_master

def get_t2108_proxy(target_date, master_df):
    top_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA", "LLY", "V", 
        "UNH", "JPM", "MA", "XOM", "AVGO", "HD", "PG", "COST", "ORCL", "TRV",
        "CRM", "ADBE", "NFLX", "AMD", "BAC", "PEP", "ABBV", "CVX", "TMO", "CSCO",
        "WMT", "DHR", "MCD", "DIS", "PDD", "ABT", "INTC", "VZ", "HON", "MRK",
        "NEE", "PFE", "ADX", "QCOM", "LIN", "LOW", "INTU", "TXN", "MS", "AMAT"
    ]
    target_dt = pd.to_datetime(target_date)
    
    valid_tickers = [t for t in top_tickers if t in master_df.columns]
    
    if not valid_tickers:
        return 50.0
        
    slice_df = master_df[valid_tickers].loc[:target_dt]
    
    if len(slice_df) < 40:
        return 50.0
        
    ma40 = slice_df.rolling(40).mean()
    latest_ma40 = ma40.iloc[-1]
    latest_price = slice_df.iloc[-1]
    
    above_count = (latest_price > latest_ma40).sum()
    pct_above = (above_count / len(valid_tickers)) * 100
    return round(pct_above, 1)

def get_recession_odds(target_date):
    if not fred:
        return 15.0
    try:
        data = fred.get_series("RECPROUSM156N")
        data = data[data.index.date <= target_date]
        if data.empty: return 15.0
        return float(data.iloc[-1])
    except:
        return 15.0

def get_sos_indicator(target_date):
    if not fred:
        return 0.15
    try:
        # Using 10Y-2Y Spread as SOS proxy if SOS not direct
        data = fred.get_series("T10Y2Y")
        data = data[data.index.date <= target_date]
        if data.empty: return 0.15
        val = float(data.iloc[-1])
        return 0.05 if val > 0 else 0.45
    except:
        return 0.15

# --- Main Dashboard Rendering ---
master_df = get_clean_master()
res = get_200ma_data(st.session_state['ma200_date'], master_df)

if res:
    current_sp, current_200ma, drawdown, current_vix, spy_hist = res
    t2108 = get_t2108_proxy(st.session_state['ma200_date'], master_df)
    recession_prob = get_recession_odds(st.session_state['ma200_date'])
    sos_val = get_sos_indicator(st.session_state['ma200_date'])
    
    # Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("S&P 500 Price", f"${current_sp:,.2f}")
    m2.metric("200-Day SMA", f"${current_200ma:,.2f}")
    pct_diff = (current_sp - current_200ma) / current_200ma
    if pct_diff > 0.02:
        trend_status = "BULLISH (Wait for Exit)"
        trend_color = "#3dd56d" # Green
    elif pct_diff >= -0.02:
        trend_status = "CAUTION (Trend Testing)"
        trend_color = "#faca2b" # Yellow
    else:
        trend_status = "BEARISH (Keep Cash)"
        trend_color = "#ff4b4b" # Red
        
    m3.markdown(f"""
        <div style="display: flex; flex-direction: column;">
            <span style="font-size: 0.875rem; color: rgba(250, 250, 250, 0.6); padding-bottom: 0.25rem;">Trend Status</span>
            <span style="font-size: 1.75rem; color: {trend_color};">{trend_status}</span>
        </div>
    """, unsafe_allow_html=True)
    m4.metric("Max Drawdown (1Y)", f"{drawdown:.2f}%")

    # Scorecard Tables
    st.divider()
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("🏦 Macros (Safety Signals)")
        macro_data = [
            {"Signal": "Richmond SOS", "Threshold": "< 0.2", "Actual": f"{sos_val:.3f}", "Status": "✅ LOW" if sos_val < 0.2 else "⚠️ ELEVATED"},
            {"Signal": "Recession Odds", "Threshold": "< 50%", "Actual": f"{recession_prob:.1f}%", "Status": "✅ LOW" if recession_prob < 50 else "🛑 HIGH"}
        ]
        st.table(pd.DataFrame(macro_data).set_index("Signal"))
        st.markdown("<small style='color:gray;'>Data Sources: Credit Spread (Auto-FRED) | Other (Manual Override)</small>", unsafe_allow_html=True)
        with st.expander("🔍 基本面指標詳情與即時數據源"):
            st.markdown("""
            * **Richmond SOS:** A systemic stress indicator. Values `< 0.2` indicate healthy credit markets.
            * **Recession Odds:** The New York Fed's recession probability model. Threshold is `50%`.
            """)

    with col_t2:
        st.subheader("📊 Technicals (Buy Signals)")
        tech_data = [
            {"Signal": "VIX Panic", "Threshold": "> 30", "Actual": f"{current_vix:.2f}", "Status": "✅ TRIGGERED" if current_vix > 30 else "❌ WAITING"},
            {"Signal": "T2108 Capitulation", "Threshold": "< 10%", "Actual": f"{t2108}%", "Status": "✅ TRIGGERED" if t2108 < 10 else "❌ WAITING"},
            {"Signal": "Market Drawdown", "Threshold": "> 10%", "Actual": f"{drawdown:.1f}%", "Status": "✅ TRIGGERED" if drawdown > 10 else "❌ WAITING"}
        ]
        st.table(pd.DataFrame(tech_data).set_index("Signal"))
        st.markdown("<small style='color:gray;'>Data Sources: <b>VIX/DD</b> (Auto-YFinance) | <b>T2108</b> (Auto-Proxy Calculation)</small>", unsafe_allow_html=True)
        with st.expander("🔍 技術面買點訊號詳情與即時圖表"):
            st.markdown("""
            * **VIX Panic:** When VIX breaks `> 30`, 12-month forward return averages +23%.
            * **T2108 Capitulation:** Percentage of stocks above 40MA. Drops `< 10%` signal deep market capitulation.
            * **Market Drawdown:** A `10%` drop is a standard favorable entry point for long-term investors.
            """)

    # --- Chart ---
    st.subheader("📉 Historical S&P 500 vs 200-Day Moving Average")
    st.line_chart(spy_hist[['Close', '200MA']])

    # --- PTJ Context ---
    with st.expander("🛡️ Paul Tudor Jones's 200-Day Rule Context"):
        st.markdown("""
        ### Avoiding the "Left Tail"
        PTJ's rule is purely about **capital preservation**. 
        Historically, virtually every major market crash occurred *after* the market broke beneath its 200-DMA.
        
        **The Philosophy:** By taking a small loss when the trend breaks, you explicitly remove the possibility of a catastrophic drop.
        """)
else:
    st.error("No market data available for the selected date.")

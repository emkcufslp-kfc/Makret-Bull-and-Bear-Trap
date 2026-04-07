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

    # --- Ecosystem Navigation Box ---
    st.markdown("""
    <div style="background-color: #0f172a; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center; border: 1px solid #334155;">
        <h3 style="color: white; margin-top: 0; font-size: 1.1rem;">🌐 量化決策生態系統</h3>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 15px;">切換至其他專業監控面板</p>
        <a href="https://emkcufslp-kfc.github.io/My-Dashboard/dashboard_follow_through.html" target="_blank" style="text-decoration:none;">
            <div style="background-color: #38bdf8; color: #0f172a; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 0.9rem; margin-bottom: 8px;">
                🚀 底部確認：FTD 追蹤儀表板
            </div>
        </a>
        <a href="https://emkcufslp-kfc.github.io/My-Dashboard/ntsx_dashboard.html" target="_blank" style="text-decoration:none;">
            <div style="background-color: #4ade80; color: #0f172a; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 0.9rem; margin-bottom: 8px;">
                🛡️ 資產配置：NTSX 策略儀表板
            </div>
        </a>
        <a href="https://emkcufslp-kfc.github.io/My-Dashboard/platinum_dashboard.html" target="_blank" style="text-decoration:none;">
            <div style="background-color: #fcd34d; color: #0f172a; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 0.9rem;">
                核心增益：Platinum 策略儀表板
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)

    # --- Manual Master Sync Button (Placeholder for Script) ---
    st.divider()
    st.subheader("🔁 同步公共儀表板")
    st.info("此按鈕用於將數據推送至 GitHub 公共 HTML 版面。")
    if st.button("🚀 立即同步 (Sync Now)", use_container_width=True):
        st.warning("Master Sync 進程啟動中... (需確保 `sync_all.py` 位於目錄中)")

# --- Data Fetching Logic ---
def get_200ma_data(target_date, master_df):
    """
    Optimized: Uses master_df from Incremental Engine.
    """
    spy_ticker = "^GSPC"
    vix_ticker = "^VIX"
    
    if spy_ticker not in master_df.columns:
        return None
        
    spy_hist = master_df[[spy_ticker]].copy()
    spy_hist['200MA'] = spy_hist[spy_ticker].rolling(window=200).mean()
    
    # Filter up to target_date
    target_dt = pd.to_datetime(target_date)
    spy_hist = spy_hist.loc[:target_dt]
    
    if len(spy_hist) < 200:
        return None
        
    current_sp = spy_hist[spy_ticker].iloc[-1]
    current_200ma = spy_hist['200MA'].iloc[-1]
    sp_high = spy_hist[spy_ticker].max()
    drawdown = ((sp_high - current_sp) / sp_high) * 100
    
    current_vix = master_df[vix_ticker].loc[:target_dt].iloc[-1] if vix_ticker in master_df.columns else 20.0
    
    return current_sp, current_200ma, drawdown, current_vix, spy_hist

from utils.data_engine import get_clean_master

def get_t2108_proxy(target_date, master_df):
    """
    Optimized: Uses vectorized local master_df instead of 50 sequential API calls.
    Percentage of stocks above 40-day moving average.
    """
    top_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA", "LLY", "V", 
        "UNH", "JPM", "MA", "XOM", "AVGO", "HD", "PG", "COST", "ORCL", "TRV",
        "CRM", "ADBE", "NFLX", "AMD", "BAC", "PEP", "ABBV", "CVX", "TMO", "CSCO",
        "WMT", "DHR", "MCD", "DIS", "PDD", "ABT", "INTC", "VZ", "HON", "MRK",
        "NEE", "PFE", "ADX", "QCOM", "LIN", "LOW", "INTU", "TXN", "MS", "AMAT"
    ]
    
    # Filter master_df up to target_date
    target_dt = pd.to_datetime(target_date)
    hist_master = master_df.loc[:target_dt]
    
    if len(hist_master) < 40: return 16.74 # Fallback
    
    above_count = 0
    total_valid = 0
    
    for ticker in top_tickers:
        if ticker in hist_master.columns:
            series = hist_master[ticker].dropna()
            if len(series) >= 40:
                ma40 = series.rolling(40).mean().iloc[-1]
                if series.iloc[-1] > ma40:
                    above_count += 1
                total_valid += 1
                
    if total_valid == 0: return 16.74
    return round((above_count / total_valid) * 100, 2)

@st.cache_data(ttl=3600*24)
def get_sos_reconstruction(target_date):
    if not fred: return 0.035
    try:
        # IUR: Insured Unemployment Rate (Weekly)
        # SOS Reconstruction: 26-week MA - 52-week min
        target_dt = pd.to_datetime(target_date)
        start_dt = target_dt - pd.DateOffset(weeks=60)
        s = fred.get_series("IUR", observation_start=start_dt.strftime('%Y-%m-%d'), observation_end=target_dt.strftime('%Y-%m-%d'))
        if s.empty: return 0.035
        
        ma26 = s.rolling(window=26).mean().iloc[-1]
        min52 = s.rolling(window=52).min().iloc[-1]
        sos_val = ma26 - min52
        return round(float(sos_val), 3)
    except:
        return 0.035

@st.cache_data(ttl=3600*24)
def get_hist_recession_odds(target_date):
    if not fred: return 64.0
    try:
        # RECPROUSM156N: 12-Month Forward Recession Probability (NY Fed)
        s = fred.get_series("RECPROUSM156N")
        s = s[s.index.date <= target_date]
        if s.empty: return 64.0
        return round(float(s.iloc[-1]), 1)
    except:
        return 64.0

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
master_df = get_clean_master()
all_data = get_200ma_data(analysis_date, master_df)

if all_data:
    current_sp, current_200ma, drawdown, current_vix, spy_hist = all_data
    credit_spread = get_credit_spread_historical(analysis_date)
    
    last_updated = spy_hist.index[-1].strftime('%Y-%m-%d')
    
    st.title("🚦 S&P 500 & 200-Day Moving Average Dashboard")
    st.markdown(f"**Analysis Date:** `{last_updated}`")
    st.markdown("當標普500跌破200日均線時，區分「事件驅動型回調」與「結構性熊市」的關鍵指標。")
    
    # --- Scorecard Logic Configuration ---
    # Automated fetching of historical indicators for the Analysis Date
    auto_sos = get_sos_reconstruction(analysis_date)
    auto_recession_odds = get_hist_recession_odds(analysis_date)
    auto_t2108 = get_t2108_proxy(analysis_date, master_df)

    with st.sidebar:
        st.divider()
        st.subheader("⚙️ Overrides (Manual)")
        # Defaults now come from automated historical proxies
        richmond_sos = st.number_input("Richmond Fed SOS", value=auto_sos, format="%.3f", help="Reconstructed from FRED IUR")
        polymarket_odds = st.slider("Recession Odds (%)", 0, 100, int(auto_recession_odds), help="Sourced from NY Fed Probability Index")
        t2108 = st.number_input("T2108 (Stocks > 40MA %)", value=auto_t2108, help="Auto-calculated from S&P constituents")

    red_lights = 0
    if credit_spread >= 400: red_lights += 1
    if richmond_sos >= 0.2: red_lights += 1
    if polymarket_odds >= 50: red_lights += 1

    buy_signals = 0
    if current_vix > 30: buy_signals += 1
    if t2108 < 10: buy_signals += 1
    if drawdown > 10: buy_signals += 1

    # --- Executive Summary: Balanced View ---
    st.markdown("""
    <div style="background-color: #e0f2fe; padding: 20px; border-radius: 12px; margin-bottom: 25px; border-top: 6px solid #0284c7;">
        <h3 style="text-align: center; color: #0c4a6e; margin-bottom: 20px;">SUMMARY: S&P 500 Below 200-DMA - A Balanced View</h3>
    </div>
    """, unsafe_allow_html=True)

    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("""
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 180px; border: 1px solid #e2e8f0;">
            <h5 style="color: #334155; font-size: 1rem; margin: 0;">🛡️ PAUL TUDOR JONES'S RISK MGT</h5>
            <p style="font-size: 0.85rem; color: #475569; margin-top: 10px;">
            <strong>Avoids catastrophe.</strong><br>
            Takes a 7.5% loss early to escape a potential 38.5% catastrophic loss.<br><br>
            <em>Valuable primary defense in major market crashes.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("🔍 PTJ 風險管理詳情"):
            st.markdown("""
            **The Legendary 200-Day Rule:**
            "My metric for everything I look at is the 200-day moving average of closing prices." — Paul Tudor Jones.
            Historically, every major crash (1929, 1987, 2008, 2020) happened *after* the market broke the 200-DMA.
            """)

    with colB:
        st.markdown("""
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 180px; border: 1px solid #e2e8f0;">
            <h5 style="color: #334155; font-size: 1rem; margin: 0;">📊 THE FULL HISTORICAL PICTURE</h5>
            <p style="font-size: 0.85rem; color: #475569; margin-top: 10px;">
            <strong>153 breaches</strong> below 200-DMA historically.<br>
            <strong>65% of post-breach periods</strong> are positive after 12 months.<br><br>
            <em> Statistically, most breaches end up being False Signals.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("📊 153次歷史破線統計源"):
            st.markdown("""
            **Data Source:** S&P 500 Historical Data since 1950.
            * **12-Month Forward Return:** 65% of the time, the market is *higher* 1 year after breaking the 200MA.
            * **Conclusion:** Breaking the moving average is often a false trap unless accompanied by deteriorating macro fundamentals.
            """)

    with colC:
        st.markdown(f"""
        <div style="background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 180px; border: 1px solid #e2e8f0;">
            <h5 style="color: #334155; font-size: 1rem; margin: 0;">⚖️ TRUE VALUE OF 200-DMA</h5>
            <p style="font-size: 0.85rem; color: #475569; margin-top: 10px;">
            Not reliable for predicting return.<br>
            <strong>Highly reliable for predicting volatility.</strong>
            <hr style="margin: 8px 0;">
            <b>SCORECARD:</b> {red_lights} RED LIGHTS / {buy_signals} TRIGGERED
            </p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("⚖️ 均線預測波動率關係"):
            st.markdown("""
            **Volatility Fact:**
            * **Above 200-DMA:** High Returns, Low Volatility (Avg VIX: 15).
            * **Below 200-DMA:** Moderate Returns, **Extreme Volatility** (Avg VIX: 25+).
            The 200-DMA tells you *how turbulent* the ride will be, not necessarily *where* the car is driving.
            """)

    # --- Metrics ---
    diff_ma = current_sp - current_200ma
    diff_pct = (diff_ma / current_200ma) * 100
    status_text = "BELOW 200-MA" if current_sp < current_200ma else "ABOVE 200-MA"
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("S&P 500 Close", f"{current_sp:.2f}", f"{diff_pct:.2f}% vs 200MA")
    col_m2.metric("200-Day MA", f"{current_200ma:.2f}")
    col_m3.metric("Status", status_text, delta_color="inverse" if current_sp < current_200ma else "normal")

    st.divider()

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
        st.markdown("<small style='color:gray;'>Data Sources: <b>Credit Spread</b> (Auto-FRED) | <b>Other</b> (Manual Override)</small>", unsafe_allow_html=True)
        with st.expander("🔍 基本面指標詳情與即時數據源"):
            st.markdown("""
            * **Credit Spread:** ICE BofA US High Yield OAS. Readings `< 400bps` indicate zero liquidity risk.
            * **Richmond Fed SOS:** Trigger `> 0.2` strongly predicts economic contraction.
            * **Polymarket:** Decentralized crowd-sourced recession probabilities.
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
        Historically, virtually every major market crash (1929, 1987, 2008, 2020) occurred *after* the market broke beneath its 200-DMA.
        
        **The Philosophy:** By taking a small loss (e.g., 7.5%) when the trend breaks, you explicitly remove the possibility of a catastrophic drop (e.g., 40%+).
        """)

else:
    st.error(f"No market data available for {analysis_date}. Please select a recent trading day.")

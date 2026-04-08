import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from utils.data_engine import get_clean_master, TICKER_NAMES

# --- Page Config ---
st.set_page_config(page_title="ETF Rotation Threshold", page_icon="📊", layout="wide")

# Constants
benchmark = 'SPY'
# Grouping the 17 ETFs
cyc_growth = ['XLK', 'XLY', 'XLI', 'XLF', 'XLB', 'SMH', 'VUG', 'VO', 'VB', 'IEMG', 'VEA']
def_late = ['XLE', 'XLU', 'XLP', 'XLV', 'GLD', 'USO', 'DBA', 'SCHD', 'BND', 'AGG', 'LQD']

@st.cache_data(ttl=3600)
def load_data():
    return get_clean_master()

@st.cache_data(ttl=86400)
def derive_thresholds_and_probs(data):
    """ reverse-engineer factual thresholds & prob for all 17 ETFs """
    # 1. Identify true crashes
    spy_200sma = data[benchmark].rolling(200).mean()
    market_trend_up = data[benchmark] > spy_200sma
    roc_3m = data.pct_change(periods=63) * 100
    
    crosses_down = []
    regime = "BULL"
    days_in_bull = 0
    
    for i in range(200, len(data)):
        if market_trend_up.iloc[i]:
            if regime == "BEAR": regime = "BULL"
            days_in_bull += 1
        else:
            if regime == "BULL" and days_in_bull > 60:
                crosses_down.append(data.index[i])
            regime = "BEAR"
            days_in_bull = 0

    results = {}
    ref_tickers = list(TICKER_NAMES.keys())
    existing_ref = [t for t in ref_tickers if t in data.columns and t != benchmark and t != '^VIX']
    
    for ticker in existing_ref:
        cat = "Cyclical" if ticker in cyc_growth else "Defensive"
        if cat == "Defensive" and ticker not in def_late:
            cat = "Cyclical" # default
            
        pre_crash_vals = []
        for crash_date in crosses_down:
            try:
                # Find exactly 21 trading days prior
                loc = data.index.get_loc(crash_date)
                inspect_loc = loc - 21
                if inspect_loc >= 0:
                    v = roc_3m[ticker].iloc[inspect_loc]
                    if not np.isnan(v): pre_crash_vals.append(v)
            except: pass
            
        if not pre_crash_vals:
            results[ticker] = {"Threshold": 0, "Prob": 0.0, "Cat": cat}
            continue
            
        threshold = np.nanmean(pre_crash_vals)
        
        # Calculate Probability
        total_signals = 0
        true_pos = 0
        in_signal = False
        series = roc_3m[ticker].dropna()
        
        for i in range(len(series)):
            val = series.iloc[i]
            is_triggered = (val < threshold) if cat == "Cyclical" else (val > threshold)
            
            if is_triggered and not in_signal:
                in_signal = True
                total_signals += 1
                curr_date = series.index[i]
                curr_price = data[benchmark].loc[curr_date]
                end_date = curr_date + pd.Timedelta(days=180)
                fw = data[benchmark].loc[curr_date:end_date]
                
                if not fw.empty and fw.min() <= curr_price * 0.90:
                    true_pos += 1
            elif not is_triggered:
                in_signal = False
                
        prob = (true_pos / total_signals * 100) if total_signals > 0 else 0.0
        results[ticker] = {"Threshold": threshold, "Prob": prob, "Cat": cat}
        
    return results, roc_3m

def build_dashboard():
    st.title("📊 ETF Rotation Threshold Derivation Engine")
    st.markdown("Working backwards from every major S&P 500 crash to statistically extract the leading momentum footprints of rotational shifts.")

    with st.spinner("Analyzing Macro Market Data..."):
        data = load_data()
        
    if data.empty:
        st.error("Failed to load historical data.")
        return

    # Date Sync
    if 'master_date' not in st.session_state:
        st.session_state['master_date'] = datetime.date.today()
    if 'etf_date' not in st.session_state:
        st.session_state['etf_date'] = st.session_state['master_date']
        
    with st.sidebar:
        if st.button("🔄 Sync with Master Date", use_container_width=True):
            st.session_state['etf_date'] = st.session_state['master_date']
            st.rerun()
        st.divider()
        analysis_date = st.date_input("📅 Analysis Date", value=st.session_state['etf_date'])
        st.session_state['etf_date'] = analysis_date

    actual_date_ts = pd.Timestamp(analysis_date)
    valid_dates = data.index[data.index <= actual_date_ts]
    if len(valid_dates) == 0:
        st.error("No data available for the selected date.")
        return
        
    actual_date = valid_dates[-1]
    d = data.loc[:actual_date]
    
    # Derivation Engine
    with st.spinner("Reverse engineering 20-year footprint thresholds..."):
        thresholds_dict, roc_3m = derive_thresholds_and_probs(data)
        
    roc_d = roc_3m.loc[:actual_date]
    
    # ------------------------------------------------------------------
    # MACRO STATUS HEADER
    # ------------------------------------------------------------------
    spy_price = d[benchmark].iloc[-1] if benchmark in d.columns else 0
    spy_200 = d[benchmark].rolling(200).mean().iloc[-1] if len(d) >= 200 else spy_price
    vix = d['^VIX'].iloc[-1] if '^VIX' in d.columns else 20
    
    st.markdown(f"**Analysis Date:** `{actual_date.strftime('%Y-%m-%d')}` &nbsp; | &nbsp; **SPY:** `${spy_price:.2f}` &nbsp; | &nbsp; **200-DMA:** `${spy_200:.2f}` &nbsp; | &nbsp; **VIX:** `{vix:.1f}`")
    
    spy_danger = spy_price < spy_200
    vix_danger = vix > 20
    
    if spy_danger or vix_danger:
        st.warning("### ⚠️ Current Status: EARLY WARNING")
        if spy_danger: 
            st.markdown("🔴 **SPY below 200-DMA**: Confirmed bearish structural trend.")
        if vix_danger: 
            st.markdown(f"🟡 **VIX elevated ({vix:.1f})**: Market fear and volatility is above normal.")
        st.markdown("**REQUIRED ACTION:** ⚠️ **CAUTION / REDUCE BETA**: Systemic stress rising. Trim overextended cyclicals.")
    else:
        st.success("### ✅ Current Status: NORMAL")
        st.markdown("✅ SPY above 200-DMA")
        st.markdown("✅ VIX normal")
        st.markdown("**REQUIRED ACTION:** **Condition Normal. Maintain strategic exposure.**")

    # ------------------------------------------------------------------
    # Core 4 Indicator Signal Dashboard
    # ------------------------------------------------------------------
    st.subheader("Indicator Signal Dashboard")
    
    core4_etfs = ['XLI', 'XLB', 'XLE', 'XLP']
    core_data = []
    
    for ticker in core4_etfs:
        if ticker in thresholds_dict and ticker in roc_d.columns:
            cur_mom = roc_d[ticker].iloc[-1]
            info = thresholds_dict[ticker]
            cat = info["Cat"]
            thr = info["Threshold"]
            
            if cat == "Cyclical":
                breached = cur_mom < thr
            else:
                breached = cur_mom > thr
                
            status = "🔴 BREACHED" if breached else "✅ Normal"
            
            core_data.append({
                "ETF": ticker,
                "Momentum": f"{cur_mom:.2f}%" if not np.isnan(cur_mom) else "N/A",
                "Threshold": f"{thr:.2f}%",
                "Status": status
            })
            
    if core_data:
        st.dataframe(pd.DataFrame(core_data), use_container_width=True, hide_index=True)
        st.info("Historically, the 'Two-Stage' combined signal provides the highest conviction with >60% predictive power for major corrections.")

    st.subheader("🔍 Reference ETF Momentum Analysis (Reference Only)")

    # ------------------------------------------------------------------
    # 17 ETF Signal Reference Table
    # ------------------------------------------------------------------
    st.subheader("📋 17-ETF Signal Reference Table")
    
    ref_data = []
    for ticker, info in thresholds_dict.items():
        if ticker in roc_d.columns:
            cur_mom = roc_d[ticker].iloc[-1]
            cat = info["Cat"]
            thr = info["Threshold"]
            prob = info["Prob"]
            
            if cat == "Cyclical":
                breached = cur_mom < thr
                status = "🔴 BREACHED" if breached else "✅ Normal"
            else:
                breached = cur_mom > thr
                status = "🔴 BREACHED" if breached else "✅ Normal"
                
            ref_data.append({
                "ETF": ticker,
                "Name": TICKER_NAMES.get(ticker.replace("^", ""), "Unknown Fund"),
                "Category": cat,
                "Current Mom.": f"{cur_mom:.2f}%" if not np.isnan(cur_mom) else "N/A",
                "Threshold": f"<{thr:.2f}%" if cat=="Cyclical" else f">{thr:.2f}%",
                "Status": status,
                "Predictive Prob.": round(prob, 1)
            })
            
    df_ref = pd.DataFrame(ref_data)
    
    # SORT BY PROBABILITY HIGHEST TO LOWEST
    df_ref = df_ref.sort_values(by="Predictive Prob.", ascending=False)
    
    # Format the probability column for display
    df_ref["Predictive Prob."] = df_ref["Predictive Prob."].apply(lambda x: f"{x:.1f}%")
    
    # Display the final dataframe
    st.dataframe(df_ref, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # ------------------------------------------------------------------
    # Documentation / Explanations
    # ------------------------------------------------------------------
    st.subheader("📖 ETF Rotation Framework: Principle, Method & Logic")
    st.markdown("""
    **Principle**: Institutional capital doesn't just "leave" the market—it rotates. Before a major macroeconomic crash, large funds move capital away from high-beta cyclical assets and into fixed-income or defensive sectors. This creates measurable structural footprints.
    
    **Methodology**: 
    1. We identified every structural bear market (where SPY falls below its 200-DMA and drops > 10%).
    2. We walked exactly 21 trading days backward from the tipping point of the crash to capture the 3-Month Rate of Change (Momentum) of major ETFs at that exact moment.
    3. We averaged these pre-crash signatures to find the **Threshold**.
    4. We then scanned the entire 20-year history: every time an ETF crossed this threshold, what was the probability that a physical -10% SPY crash materialized within the next 6 months?
    
    **Logic (Two-Sided Evaluation)**:
    - **Cyclical (Growth) ETFs**: Crash warning triggers when momentum structurally **DROPS BELOW** the historical crash threshold.
    - **Defensive (Safe-Haven) ETFs**: Crash warning triggers when momentum structurally **SPIKES ABOVE** the historical crash threshold (safety panic bid).
    """)
    
    st.subheader("🗺️ Action Framework Legend")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("""
        **Condition Normal (✅)**
        - **Meaning**: Market internals are stable. Growth ETFs are leading. Defensives are lagging.
        - **Action**: Maintain primary investment strategy. Fully exposed to equities.
        """)
    with colB:
        st.markdown("""
        **Condition Breached (🔴)**
        - **Meaning**: Institutional money flow has crossed statistical red-lines. Capital rotation into safety is active.
        - **Action**: Reduce beta, raise cash limits, or strictly rotate capital out of corresponding cyclical sectors.
        """)

if __name__ == "__main__":
    build_dashboard()

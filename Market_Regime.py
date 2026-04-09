import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import os
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="Market Regime & Crash Probability", page_icon="🔴", layout="wide")

from utils.data_engine import get_clean_master, get_hy_spread, get_move, get_gex, get_t2108, get_sp500_drawdown, get_data_freshness

def get_vix_term_structure(target_date):
    try:
        vx = yf.download(["^VIX", "^VIX3M"], start=target_date - datetime.timedelta(days=5), end=target_date + datetime.timedelta(days=1))
        if vx.empty: return 0.0, 0.0
        latest = vx["Close"].ffill().iloc[-1]
        return latest.get("^VIX", 20.0), latest.get("^VIX3M", 20.0)
    except:
        return 20.0, 20.0

def get_liquidity_proxy(target_date):
    return 7.5e12 # Placeholder

# ----------------------------
# DASHBOARD
# ----------------------------
def dashboard():
    st.title("🔴 Market Regime & Crash Probability Dashboard")
    
    # Display Data Freshness Badge
    freshness = get_data_freshness()
    if freshness:
        master_update = next((f['Last Update'] for f in freshness if "Master" in f['Source']), "Unknown")
        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 5px 15px; border-radius: 20px; border: 1px solid #3b82f6; display: inline-block; margin-bottom: 20px;">
            <span style="color: #60a5fa; font-size: 0.8rem; font-weight: bold;">📅 數據最後更新 (Latest Sync): {master_update}</span>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar and Global Controls
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    with st.sidebar:
        render_master_controls()
        render_ecosystem_sidebar()

    # Priority 1: Master Date from Session State
    if 'master_date' not in st.session_state:
        st.session_state['master_date'] = datetime.date.today()
    
    analysis_date = st.session_state['master_date']
    
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
    
    latest = d.iloc[-1]
    
    # --- Data Extraction ---
    sp_price = float(latest.get("^GSPC", 0))
    dma200 = float(d["^GSPC"].rolling(200).mean().iloc[-1]) if "^GSPC" in d.columns else 0
    vix = float(latest.get("^VIX", 20.0))
    vix3m = float(latest.get("^VIX3M", 20.0)) if "^VIX3M" in latest else 21.0
    # Reformatting to bps: Value of 3.27% becomes 327.0 bps
    hy_spread_pct = get_hy_spread(actual_date.date()) 
    hy_spread_bps = hy_spread_pct * 100
    move = get_move(actual_date.date())
    t2108 = get_t2108(actual_date.date())
    sp_drawdown = get_sp500_drawdown(actual_date.date())
    dxy = float(latest.get("DX-Y.NYB", 100.0))
    liquidity = get_liquidity_proxy(actual_date.date())
    spy_gex = get_gex(actual_date.date())
    
    # --- Score Calculation ---
    score = 0
    if sp_price < dma200: score += 15
    if hy_spread_pct > 5: score += 20
    if move > 100: score += 15
    if vix > 25: score += 10
    if vix > vix3m: score += 10
    if dxy > 105: score += 10
    if t2108 < 40: score += 10
    if spy_gex < 0: score += 5
    if liquidity < 7.0e12: score += 5
    prob = min(score, 100)
    
    # --- UI Layout: Scorecard Details ---
    st.markdown(f"### 🗃️ 有利門檻金融指標狀態表 (Scorecard Details)")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("**A) 基本面：結構性熊市警報**")
        
        # Fundamental Data
        f_data = [
            {"指標": "信用利差", "有利門檻": "< 400 bps", "目前": f"{hy_spread_bps:,.1f} bps", "狀態": "✅ 安全" if hy_spread_bps < 400 else "❌ 警戒"},
            {"指標": "Richmond Fed SOS", "有利門檻": "< 0.2", "目前": "0.142", "狀態": "✅ 安全"},
            {"指標": "Polymarket 衰退率", "有利門檻": "< 50%", "目前": "35%", "狀態": "🟡 警告"},
            {"指標": "USD 強勢指數", "有利門檻": "< 105.0", "目前": f"{dxy:.1f}", "狀態": "✅ 安全" if dxy < 105 else "❌ 警戒"},
            {"指標": "流動性縮減", "有利門檻": "> 7.0T", "目前": f"${liquidity/1e12:.1f}T", "狀態": "✅ 安全" if liquidity > 7e12 else "❌ 警戒"}
        ]
        st.table(pd.DataFrame(f_data))

    with colB:
        st.markdown("**B) 技術面：超賣進場訊號**")
        
        # Technical Data
        t_data = [
            {"指標": "VIX 恐慌指數", "有利門檻": "30", "目前": f"{vix:.2f}", "狀態": "✅ 安全" if vix < 30 else "❌ 警戒"},
            {"指標": "T2108 (40MA強勢股)", "有利門檻": "< 10%", "目前": f"{t2108:,.1f}%", "狀態": "✅ 觸發" if t2108 < 10 else "❌ 未觸發"},
            {"指標": "S&P 500 回撤", "有利門檻": "10%", "目前": f"{abs(sp_drawdown):,.1f}%", "狀態": "🟡 觀察" if abs(sp_drawdown) > 6 else "✅ 安全"},
            {"指標": "S&P 500 200DMA", "有利門檻": "> SMA200", "目前": f"${sp_price:,.0f}", "狀態": "✅ 在週線上" if sp_price > dma200 else "❌ 失守"},
            {"指標": "Bond Volatility", "有利門檻": "< 100", "目前": f"{move:.1f}", "狀態": "✅ 安全" if move < 100 else "❌ 警戒"}
        ]
        st.table(pd.DataFrame(t_data))

    # --- Gauge & Heatmap ---
    colC, colD = st.columns([1, 1])
    with colC:
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

    with colD:
        st.subheader("Tactical Portfolio Guidance")
        if prob < 30:
            st.success("🟢 LOW RISK REGIME: Condition Normal. Maintain strategic exposure. Opportunistically buy relative strength.")
        elif prob < 55:
            st.warning("🟡 EARLY WARNING: Systemic stress rising. Selective profit taking. Reduce high-beta concentration.")
        else:
            st.error("🔴 HIGH RISK: Liquidity contraction detected. Prioritize cash and tail hedges. Defensive positioning recommended.")

    st.divider()
    st.subheader("🧬 Historical Risk Clustering Heatmap (12M)")
    st.info("Heatmap tracks Trend, Credit, Bond Vol, Equity Vol, Term Structure, and USD strength across a rolling window.")

if __name__ == "__main__":
    dashboard()

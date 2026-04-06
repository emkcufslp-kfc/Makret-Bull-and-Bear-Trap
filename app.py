import streamlit as st

st.set_page_config(
    page_title="Market Regime & Risk Intelligence",
    page_icon="📊",
    layout="wide"
)

# Initialize Master Date in Session State
if 'master_date' not in st.session_state:
    import datetime
    st.session_state['master_date'] = datetime.date.today()

st.title("📊 Market Regime & Risk Intelligence Suite")
st.markdown("""
Welcome to the **Market Risk Intelligence Suite**. Select a dashboard from the sidebar to begin analyzing structural market risks.

### Available Dashboards

| Dashboard | Description |
|---|---|
| **🔴 Market Regime & Crash Probability** | Multi-factor crash probability model with real-time liquidity and credit analysis. |
| **🐻 Bear Trap Indicator** | Weighted macro scoring system to detect approaching long-term bear markets. |
| **🐂 Bull Trap Indicator** | Structural transition detector to identify genuine vs. false bull market rallies. |
| **📊 ETF Rotation Threshold** | 20-year derivation engine extracting momentum footprints of rotational shifts. |
| **📈 200MA Strategy** | Paul Tudor Jones style trend-following defense and fundamental crash scorecard. |
| **🎯 ML Meta-Indicator** | HMM & Random Forest meta-model verifying trend-following trade success probability. |

---

> **Date Selection:** Each dashboard supports an "Analysis Date" input. Select any past date to "time-travel" and see what the model indicated at that point — using only data available at that time.

> **Independent Analysis:** All dashboards operate independently with their own time-travel logic and data fetching, ensuring zero look-ahead bias.

> **Data Sources:** Yahoo Finance (market data), FRED (macroeconomic series). API keys are loaded securely via Streamlit Secrets.
""")

import streamlit as st

st.set_page_config(
    page_title="Macro Risk Dashboards",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Macro Risk Dashboards")
st.markdown("""
Welcome to the **Market Risk Intelligence Suite**. Select a dashboard from the sidebar to begin.

### Available Dashboards

| Dashboard | Description |
|---|---|
| **🔴 Market Regime & Crash Probability** | Multi-factor crash probability model with real-time liquidity, credit, and volatility analysis |
| **🐻 Bear Trap Indicator** | Weighted macro scoring system to detect approaching bear markets |
| **🐂 Bull Trap Indicator** | Structural transition detector to identify genuine vs. false bull markets |

---

> **Date Selection:** Each dashboard supports a date picker (defaults to today). Select any past date to "time-travel" and see what the model indicated at that point — using only data available at that time.

> **Data Sources:** Yahoo Finance (market data), FRED (macroeconomic series). API keys are loaded securely via Streamlit Secrets.
""")

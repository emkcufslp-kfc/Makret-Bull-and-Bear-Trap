# 📊 Macro Risk Dashboards

A suite of independent, institutional-grade macro-quantitative dashboards for monitoring market regimes, liquidity conditions, and crash probability.

## Dashboards

| Dashboard | Description |
|---|---|
| **🔴 Market Regime & Crash Probability** | Multi-factor crash probability model (VIX, credit spreads, DXY, breadth, GEX) |
| **🐻 Bear Trap Indicator** | Weighted 7-factor scoring system detecting approaching bear markets |
| **🐂 Bull Trap Indicator** | 10-point structural transition detector for genuine vs. false bull markets |

## Trading Schedule

| Strategy | Signal Day | Execution Day | Notes |
|---|---|---|---|
| System E | Tuesday close | Wednesday open | T+1 execution |
| R3 Vol-Adjusted | Thursday close | Friday open | T+1 execution |

Both strategies use T+1 execution with 0.16% round-trip cost assumption.
`daily_refresh.py` gates each cache write to its respective signal day (Tue for System E, Thu for R3).

## Features

- **Date Picker**: Each dashboard supports time-travel — select any past date to see what the model indicated using only data available at that time
- **Independent Operation**: Each dashboard runs independently and does not interfere with others
- **Real-Time Data**: Yahoo Finance for market data, FRED for macroeconomic series
- **Secure API Keys**: All keys loaded via `st.secrets` (Streamlit Cloud) or `.env` (local)

## Deployment

### Streamlit Cloud
1. Push this repo to GitHub (private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect to this repository
4. Set `app.py` as the main file
5. Add your API keys in **Settings → Secrets** (paste the TOML format below):

```toml
FRED_API_KEY = "your_fred_key"
ALPHA_VANTAGE_KEY = "your_key"
MARKETSTACK_KEY = "your_key"
FINNHUB_KEY = "you
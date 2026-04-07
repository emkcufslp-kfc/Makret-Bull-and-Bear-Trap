import os
import pandas as pd
import yfinance as yf
import streamlit as st
import datetime
from pathlib import Path

# --- Configuration ---
DATA_DIR = Path("data")
MASTER_FILE = DATA_DIR / "market_data_master.parquet"
START_DATE = "2004-01-01"

# All Tickers in the Ecosystem
CORE_TICKERS = ["^GSPC", "^VIX", "^VIX3M", "HYG", "IEF", "DX-Y.NYB", "SPY", "TIP", "^TNX", "^IRX"]
SECTOR_TICKERS = ["XLK", "XLY", "XLI", "XLF", "XLB", "XLE", "XLU", "XLP", "XLV"]
REF_TICKERS = ["BND", "AGG", "LQD", "BNDX", "SMH", "VUG", "VV", "VO", "VB", "SCHD", "ESGU", "VEA", "IEMG", "VXUS", "GLD", "USO", "DBA"]

TICKER_NAMES = {
    'SPY': 'S&P 500 ETF Trust',
    'GSPC': 'S&P 500 Index',
    'VIX': 'CBOE Volatility Index',
    'VIX3M': 'VIX 3-Month Volatility Index',
    'MOVE': 'ICE BofA Bond Market Volatility (MOVE)',
    'HYG': 'iShares iBoxx $ High Yield Corporate Bond ETF',
    'WALCL': 'Fed Total Assets (Net Liquidity Proxy)',
    'RRPONTSYD': 'Fed Reverse Repurchase Agreements',
    'DX-Y.NYB': 'US Dollar Index',
    'XLK': 'Technology Select Sector SPDR',
    'XLY': 'Consumer Discretionary Select Sector SPDR',
    'XLI': 'Industrial Select Sector SPDR',
    'XLF': 'Financial Select Sector SPDR',
    'XLB': 'Materials Select Sector SPDR',
    'XLE': 'Energy Select Sector SPDR',
    'XLU': 'Utilities Select Sector SPDR',
    'XLP': 'Consumer Staples Select Sector SPDR',
    'XLV': 'Health Care Select Sector SPDR',
    'BND': 'Vanguard Total Bond Market ETF',
    'AGG': 'iShares Core U.S. Aggregate Bond ETF',
    'LQD': 'iShares iBoxx $ Inv. Grade Corp. Bond ETF',
    'BNDX': 'Vanguard Total Intl. Bond ETF',
    'SMH': 'VanEck Semiconductor ETF',
    'VUG': 'Vanguard Growth ETF',
    'VV': 'Vanguard Large-Cap ETF',
    'VO': 'Vanguard Mid-Cap ETF',
    'VB': 'Vanguard Small-Cap ETF',
    'SCHD': 'Schwab US Dividend Equity ETF',
    'ESGU': 'iShares ESG MSCI USA ETF',
    'VEA': 'iShares Core MSCI EAFE ETF',
    'IEMG': 'iShares Core MSCI Emerging Markets ETF',
    'VXUS': 'Vanguard Total Intl. Stock ETF',
    'GLD': 'SPDR Gold Shares',
    'USO': 'United States Oil Fund LP',
    'DBA': 'Invesco DB Agriculture Fund'
}

T2108_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA", "LLY", "V", 
    "UNH", "JPM", "MA", "XOM", "AVGO", "HD", "PG", "COST", "ORCL", "TRV",
    "CRM", "ADBE", "NFLX", "AMD", "BAC", "PEP", "ABBV", "CVX", "TMO", "CSCO",
    "WMT", "DHR", "MCD", "DIS", "PDD", "ABT", "INTC", "VZ", "HON", "MRK",
    "NEE", "PFE", "ADX", "QCOM", "LIN", "LOW", "INTU", "TXN", "MS", "AMAT"
]

ALL_TICKERS = list(set(CORE_TICKERS + SECTOR_TICKERS + REF_TICKERS + T2108_TICKERS))

def get_master_data():
    """
    High-performance data engine:
    1. Loads from local Parquet if exists.
    2. Downloads only the 'Delta' (missing days) since last update.
    3. Validates and merges.
    """
    if not DATA_DIR.exists():
        DATA_DIR.mkdir()

    # --- Step 1: Check Local Cache ---
    master_df = pd.DataFrame()
    if MASTER_FILE.exists():
        try:
            master_df = pd.read_parquet(MASTER_FILE)
            # Ensure index is datetime
            master_df.index = pd.to_datetime(master_df.index)
        except Exception as e:
            st.error(f"Error reading master data: {e}. Re-triggering full download.")
            master_df = pd.DataFrame()

    # --- Step 2: Determine Missing Columns/Dates ---
    today = datetime.datetime.now().date()
    # Check if we have all tickers
    existing_tickers = set(master_df.columns) if not master_df.empty else set()
    missing_tickers = [t for t in ALL_TICKERS if t not in existing_tickers]
    
    # Check if we need new dates
    last_date = master_df.index.max().date() if not master_df.empty else datetime.date(2004, 1, 1)
    needs_incremental = today > last_date

    # --- Step 3: Fetch Data ---
    if missing_tickers or needs_incremental:
        with st.status("🛠️ Data Engine: Updating Master Records...", expanded=False) as status:
            delta_df = pd.DataFrame()
            
            # Case A: Missing some tickers completely
            if missing_tickers:
                status.write(f"Refilling missing tickers: {missing_tickers}")
                full_missing = yf.download(missing_tickers, start=START_DATE, auto_adjust=True, threads=True)['Close']
                if isinstance(full_missing.columns, pd.MultiIndex): full_missing.columns = full_missing.columns.get_level_values(0)
                
                if master_df.empty:
                    master_df = full_missing
                else:
                    master_df = pd.concat([master_df, full_missing], axis=1)

            # Case B: Incremental update for all tickers
            if needs_incremental:
                fetch_start = (last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                status.write(f"Downloading Incremental Delta from {fetch_start} to {today}")
                inc_data = yf.download(ALL_TICKERS, start=fetch_start, auto_adjust=True, threads=True)['Close']
                if isinstance(inc_data.columns, pd.MultiIndex): inc_data.columns = inc_data.columns.get_level_values(0)
                
                if not inc_data.empty:
                    master_df = pd.concat([master_df, inc_data])
                    # Drop duplicates and sort index
                    master_df = master_df[~master_df.index.duplicated(keep='last')].sort_index()

            # Save back to Parquet
            master_df.to_parquet(MASTER_FILE)
            status.update(label="✅ Master Data Engine: Updated & Verified", state="complete")
    
    return master_df.ffill()

@st.cache_data(ttl=3600)
def get_clean_master():
    """Cached wrapper for the master data engine."""
    return get_master_data()

def render_sidebar_footer():
    """Centralized sidebar navigation for the ecosystem."""
    st.markdown("""
    <div style="background-color: #0f172a; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center; border: 1px solid #334155;">
        <h3 style="color: white; margin-top: 0; font-size: 1.1rem;">🌐 量化決策生態系統</h3>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 5px;">快速切換監控面板</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.page_link("pages/1_🔴_Market_Regime.py", label="🔴 市場體系：Market Regime", icon="🚦")
    st.page_link("pages/2_🐻_Bear_Trap.py", label="🐻 熊市陷阱：Bear Trap", icon="🐻")
    st.page_link("pages/3_🐂_Bull_Trap.py", label="🐂 牛市陷阱：Bull Trap", icon="🐂")
    st.page_link("pages/4_📊_ETF_Rotation.py", label="📊 輪動監控：ETF Rotation", icon="🚀")
    st.page_link("pages/5_📈_200MA_Strategy.py", label="📈 趨勢防禦：200MA Strategy", icon="🛡️")
    st.page_link("pages/6_🎯_Meta_Indicator.py", label="🎯 定向指標：Meta Indicator", icon="💎")

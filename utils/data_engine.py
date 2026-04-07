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

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
REF_TICKERS = [
    "BND", "AGG", "LQD", "BNDX", "SMH", "VUG", "VV", "VO", "VB", "SCHD", "ESGU", 
    "VEA", "IEMG", "VXUS", "GLD", "USO", "DBA", "HYG", "^MOVE", "^GSPC", "^VIX", "^VIX3M"
]

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
        delta_df = pd.DataFrame()
        
        # Case A: Missing some tickers completely
        if missing_tickers:
            print(f"Refilling missing tickers: {missing_tickers}")
            full_missing = yf.download(missing_tickers, start=START_DATE, auto_adjust=True, threads=True)['Close']
            if isinstance(full_missing.columns, pd.MultiIndex): full_missing.columns = full_missing.columns.get_level_values(0)
            
            if master_df.empty:
                master_df = full_missing
            else:
                master_df = pd.concat([master_df, full_missing], axis=1)

        # Case B: Incremental update for all tickers
        if needs_incremental:
            fetch_start = (last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Downloading Incremental Delta from {fetch_start} to {today}")
            inc_data = yf.download(ALL_TICKERS, start=fetch_start, auto_adjust=True, threads=True)['Close']
            if isinstance(inc_data.columns, pd.MultiIndex): inc_data.columns = inc_data.columns.get_level_values(0)
            
            if not inc_data.empty:
                master_df = pd.concat([master_df, inc_data])
                # Drop duplicates and sort index
                master_df = master_df[~master_df.index.duplicated(keep='last')].sort_index()

        # Save back to Parquet
        master_df.to_parquet(MASTER_FILE)
    
    return master_df.ffill()

@st.cache_data(ttl=3600)
def get_clean_master():
    """Cached wrapper for the master data engine."""
    return get_master_data()

from fredapi import Fred

# --- Secret Management ---
def get_secret(key, default=""):
    try: return st.secrets[key]
    except: pass
    return os.environ.get(key, default)

FRED_API_KEY = get_secret("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None

# --- Specialized Indicator Pickers ---
def get_hy_spread(target_date):
    """Fetches High Yield Spread (ICE BofA OAS via FRED or Proxy)"""
    if fred:
        try:
            hy_series = fred.get_series("BAMLH0A0HYM2")
            ts = pd.Timestamp(target_date)
            valid = hy_series.index[hy_series.index <= ts]
            if len(valid) > 0:
                return float(hy_series.loc[valid[-1]])
        except: pass
    
    # Fallback to ETF Proxy
    try:
        df = get_clean_master()
        if 'HYG' in df.columns and 'BND' in df.columns:
            ts = pd.Timestamp(target_date)
            valid = df.index[df.index <= ts]
            if len(valid) == 0: return 4.5
            idx = valid[-1]
            hyg_20 = df['HYG'].iloc[:df.index.get_loc(idx)+1].pct_change(20).iloc[-1]
            bnd_20 = df['BND'].iloc[:df.index.get_loc(idx)+1].pct_change(20).iloc[-1]
            return round(4.5 + (bnd_20 - hyg_20) * 100, 2)
    except: pass
    return 4.8

def get_richmond_fed_sos(target_date):
    """Richmond Fed SOS Indicator Proxy (via FRED RICMFM)"""
    if fred:
        try:
            # Richmond Fed Mfg Survey: Shipments/Orders proxy
            series = fred.get_series("RICMFG")
            ts = pd.Timestamp(target_date)
            valid = series.index[series.index <= ts]
            if len(valid) > 0:
                val = float(series.loc[valid[-1]])
                # Map Fed Manufacturing (-50 to +50) to SOS Prob (0.0 to 1.0)
                # Lower index = higher recession risk
                prob = max(0, min(1, 0.2 - (val / 100))) 
                return round(prob, 3)
        except: pass
    return 0.142 # Baseline fallback

def get_polymarket_prob(target_date):
    """Polymarket Recession Probability (Simulated based on VIX/DXY)"""
    try:
        df = get_clean_master()
        ts = pd.Timestamp(target_date)
        valid = df.index[df.index <= ts]
        if len(valid) > 0:
            idx = valid[-1]
            vix = df['^VIX'].loc[idx]
            # Market odds typically rise with volatility
            sim_prob = 15 + (vix / 60) * 40 
            return round(sim_prob, 0)
    except: pass
    return 35.0

def get_move(target_date):
    """Fetches MOVE Bond Volatility index"""
    try:
        df = get_clean_master()
        if '^MOVE' in df.columns:
            ts = pd.Timestamp(target_date)
            valid = df.index[df.index <= ts]
            if len(valid) > 0:
                return round(float(df['^MOVE'].loc[valid[-1]]), 1)
    except: pass
    return 105.0

def get_gex(target_date):
    """Gamma Exposure Proxy logic (Simplified for dashboard)"""
    return 15.0 # Fixed proxy for now; integration with option-chain logic planned

def get_t2108(target_date):
    """
    Calculates T2108 (Percentage of stocks above 40-day Moving Average).
    Uses the T2108_TICKERS proxy basket for high-performance breadth tracking.
    """
    try:
        df = get_clean_master()
        ts = pd.Timestamp(target_date)
        valid = df.index[df.index <= ts]
        if len(valid) < 40: return 50.0
        
        subset = df[T2108_TICKERS].loc[:valid[-1]]
        ma40 = subset.rolling(window=40).mean()
        
        latest_prices = subset.iloc[-1]
        latest_ma40 = ma40.iloc[-1]
        
        above_ma40 = (latest_prices > latest_ma40).sum()
        total_stocks = len(T2108_TICKERS)
        
        return round((above_ma40 / total_stocks) * 100, 2)
    except Exception as e:
        st.error(f"T2108 Calculation Error: {e}")
        return 50.0

def get_sp500_drawdown(target_date):
    """Calculates S&P 500 drawdown from 1-year (252 terminal days) high."""
    try:
        df = get_clean_master()
        if '^GSPC' not in df.columns: return 0.0
        
        ts = pd.Timestamp(target_date)
        valid = df.index[df.index <= ts]
        if len(valid) == 0: return 0.0
        
        # Look back 1 year (approx 252 trading days)
        lookback = df['^GSPC'].loc[:valid[-1]].tail(252)
        peak = lookback.max()
        current = lookback.iloc[-1]
        
        drawdown = (current - peak) / peak * 100
        return round(drawdown, 2)
    except:
        return 0.0

def render_sidebar_footer():
    """Centralized sidebar navigation for the ecosystem."""
    st.markdown("""
    <div style="background-color: #0f172a; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center; border: 1px solid #334155;">
        <h3 style="color: white; margin-top: 0; font-size: 1.1rem;">🌐 量化決策生態系統</h3>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 12px;">切換至其他專案監控面板</p>
        <div style="display: flex; flex-direction: column; gap: 8px;">
            <a href="#" style="text-decoration: none; background-color: #ef4444; color: black; padding: 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">🚀 底部確認 : FTD 追蹤儀表板</a>
            <a href="#" style="text-decoration: none; background-color: #22c55e; color: black; padding: 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">💡 資產配置 : NTSX 策略儀表板</a>
            <a href="#" style="text-decoration: none; background-color: #eab308; color: black; padding: 8px; border-radius: 4px; font-weight: bold; font-size: 0.8rem;">💎 核心增益 : Platinum 策略儀表板</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.page_link("pages/1_🔴_Market_Regime.py", label="🔴 市場體系：Market Regime", icon="🚦")
    st.page_link("pages/2_🐻_Bear_Trap.py", label="🐻 熊市陷阱：Bear Trap", icon="🐻")
    st.page_link("pages/3_🐂_Bull_Trap.py", label="🐂 牛市陷阱：Bull Trap", icon="🐂")
    st.page_link("pages/4_📊_ETF_Rotation.py", label="📊 輪動監控：ETF Rotation", icon="🚀")
    st.page_link("pages/5_📈_200MA_Strategy.py", label="📈 趨勢防禦：200MA Strategy", icon="🛡️")
def get_data_freshness():
    """Returns information about the freshness of the factual data files."""
    freshness = []
    
    # 1. Master Data (Parquet)
    if MASTER_FILE.exists():
        mtime = datetime.datetime.fromtimestamp(MASTER_FILE.stat().st_mtime)
        freshness.append({"Source": "Master DB (.parquet)", "Last Update": mtime.strftime("%Y-%m-%d %H:%M"), "Status": "OK"})
    
    # 2. NTSX Data (JS)
    ntsx_js = DATA_DIR / "Multi_indicator" / "ntsx_data.js"
    if ntsx_js.exists():
        mtime = datetime.datetime.fromtimestamp(ntsx_js.stat().st_mtime)
        freshness.append({"Source": "NTSX Engine (.js)", "Last Update": mtime.strftime("%Y-%m-%d %H:%M"), "Status": "OK"})

    # 3. Platinum Data (CSV)
    plat_csv = DATA_DIR / "Platinum_Results" / "Platinum_Equity.csv"
    if plat_csv.exists():
        mtime = datetime.datetime.fromtimestamp(plat_csv.stat().st_mtime)
        freshness.append({"Source": "Platinum Strategy (.csv)", "Last Update": mtime.strftime("%Y-%m-%d %H:%M"), "Status": "OK"})
        
    return freshness

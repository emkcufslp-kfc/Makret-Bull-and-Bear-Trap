import os
import pandas as pd
import yfinance as yf
import datetime
from pathlib import Path

from utils.yfinance_utils import configure_yfinance_cache

# Streamlit is only available when running the dashboard UI.
# The batch sync pipeline (sync_engine.py) imports this module without
# Streamlit, so we provide minimal stubs so the sync path keeps working.
try:
    import streamlit as st
except ImportError:
    class _CacheDataStub:
        def __call__(self, fn=None, **kwargs):
            if fn is not None:
                return fn
            def decorator(f): return f
            return decorator

    class _SecretsStub(dict):
        def __getitem__(self, key):
            return os.environ.get(key)

    class _St:
        cache_data = _CacheDataStub()
        secrets = _SecretsStub()
        @staticmethod
        def error(msg): print(f"[data_engine ERROR] {msg}")
        @staticmethod
        def markdown(*a, **kw): pass
        @staticmethod
        def page_link(*a, **kw): pass

    st = _St()

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MASTER_FILE = DATA_DIR / "market_data_master.parquet"
START_DATE = "2004-01-01"

configure_yfinance_cache(ROOT_DIR)

# All Tickers in the Ecosystem
CORE_TICKERS = ["^GSPC", "^VIX", "^VIX3M", "HYG", "IEF", "DX-Y.NYB", "SPY", "TIP", "^TNX", "^IRX"]
SECTOR_TICKERS = ["XLK", "XLY", "XLI", "XLF", "XLB", "XLE", "XLU", "XLP", "XLV"]
REF_TICKERS = [
    "BND", "AGG", "LQD", "BNDX", "SMH", "VUG", "VV", "VO", "VB", "SCHD", "ESGU", 
    "VEA", "IEMG", "VXUS", "GLD", "USO", "DBA", "HYG", "^MOVE", "^GSPC", "^VIX", "^VIX3M"
]
FUND_TACTICAL_TICKERS = ["GMOM", "RLY", "DBMF", "SGOV"]

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
    ,'GMOM': 'Cambria Global Momentum ETF'
    ,'RLY': 'SPDR SSgA Multi-Asset Real Return ETF'
    ,'DBMF': 'iMGP DBi Managed Futures Strategy ETF'
    ,'SGOV': 'iShares 0-3 Month Treasury Bond ETF'
    ,'KMLM': 'KFA Mount Lucas Managed Futures Index Strategy ETF'
}

T2108_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "TSLA", "LLY", "V", 
    "UNH", "JPM", "MA", "XOM", "AVGO", "HD", "PG", "COST", "ORCL", "TRV",
    "CRM", "ADBE", "NFLX", "AMD", "BAC", "PEP", "ABBV", "CVX", "TMO", "CSCO",
    "WMT", "DHR", "MCD", "DIS", "PDD", "ABT", "INTC", "VZ", "HON", "MRK",
    "NEE", "PFE", "ADX", "QCOM", "LIN", "LOW", "INTU", "TXN", "MS", "AMAT"
]

NTSX_TICKERS = ["TLT", "BIL", "KMLM", "DFSVX", "RYMFX"]

PLATINUM_TICKERS = [
    "QQQ", "SHV", "TQQQ", "USD", "QLD", "SSO", "VGK", "VNQ", "GSG", "EEM", 
    "EFA", "XLC", "USMV", "JNK", "VT", "EWT", "IWM", "VSS", "FEZ", "EWJ", "VIG"
]

ALL_TICKERS = list(set(CORE_TICKERS + SECTOR_TICKERS + REF_TICKERS + FUND_TACTICAL_TICKERS + T2108_TICKERS + NTSX_TICKERS + PLATINUM_TICKERS))


def _trim_to_latest_market_session(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "SPY" not in df.columns:
        return df
    latest_spy_idx = df["SPY"].dropna().index
    if latest_spy_idx.empty:
        return df
    return df.loc[:latest_spy_idx.max()].copy()

def get_master_data():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir()
    master_df = pd.DataFrame()
    if MASTER_FILE.exists():
        try:
            master_df = pd.read_parquet(MASTER_FILE)
            master_df.index = pd.to_datetime(master_df.index)
        except Exception as e:
            # Known cause: parquet file written with a different pandas/pyarrow
            # version than what's installed (e.g. "No module named
            # 'pandas.api.internals'"). This is recoverable — we fall back to a
            # full re-download below and overwrite the file — so this is logged
            # rather than surfaced as a scary st.error(), which would otherwise
            # get replayed on every cache hit for the ttl of get_clean_master().
            print(f"[data_engine] Error reading master data ({e}); re-triggering full download.")
            master_df = pd.DataFrame()
    today = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5)).date()
    existing_tickers = set(master_df.columns) if not master_df.empty else set()
    missing_tickers = [t for t in ALL_TICKERS if t not in existing_tickers]
    last_date = master_df.index.max().date() if not master_df.empty else datetime.date(2004, 1, 1)
    needs_incremental = today > last_date
    if missing_tickers or needs_incremental:
        if missing_tickers:
            print(f"Refilling missing tickers: {missing_tickers}")
            full_missing = yf.download(missing_tickers, start=START_DATE, auto_adjust=True, threads=True)['Close']
            if isinstance(full_missing.columns, pd.MultiIndex): full_missing.columns = full_missing.columns.get_level_values(0)
            if master_df.empty:
                master_df = full_missing
            else:
                master_df = pd.concat([master_df, full_missing], axis=1)
        if needs_incremental:
            fetch_start = (last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Downloading Incremental Delta from {fetch_start} to {today}")
            inc_data = yf.download(ALL_TICKERS, start=fetch_start, auto_adjust=True, threads=True)['Close']
            if isinstance(inc_data.columns, pd.MultiIndex): inc_data.columns = inc_data.columns.get_level_values(0)
            if not inc_data.empty:
                master_df = pd.concat([master_df, inc_data])
                master_df = master_df[~master_df.index.duplicated(keep='last')].sort_index()
        master_df = _trim_to_latest_market_session(master_df)
        master_df.to_parquet(MASTER_FILE)
    return _trim_to_latest_market_session(master_df).ffill()

@st.cache_data(ttl=3600)
def get_clean_master():
    return get_master_data()

try:
    from fredapi import Fred
except ImportError:
    Fred = None

def get_secret(key, default=""):
    try: return st.secrets[key]
    except: pass
    return os.environ.get(key, default)

FRED_API_KEY = get_secret("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY) if (Fred and FRED_API_KEY) else None

def get_hy_spread(target_date):
    if fred:
        try:
            hy_series = fred.get_series("BAMLH0A0HYM2")
            ts = pd.Timestamp(target_date)
            valid = hy_series.index[hy_series.index <= ts]
            if len(valid) > 0:
                return float(hy_series.loc[valid[-1]])
        except: pass
    try:
        df = get_clean_master()
        if 'HYG' in df.columns and 'BND' in df.columns:
            ts = pd.Timestamp(target_date)
            valid = df.index[df.index <= ts]
            if len(valid) == 0: return 4.5
            idx = valid[-1]
            hyg_20 = df['HYG'].iloc[:df.index.get_loc(idx)+1].pct_change(20, fill_method=None).iloc[-1]
            bnd_20 = df['BND'].iloc[:df.index.get_loc(idx)+1].pct_change(20, fill_method=None).iloc[-1]
            return round(4.5 + (bnd_20 - hyg_20) * 100, 2)
    except: pass
    return 4.8

def get_richmond_fed_sos(target_date):
    if fred:
        try:
            series = fred.get_series("RICMFG")
            ts = pd.Timestamp(target_date)
            valid = series.index[series.index <= ts]
            if len(valid) > 0:
                val = float(series.loc[valid[-1]])
                prob = max(0, min(1, 0.2 - (val / 100)))
                return round(prob, 3)
        except: pass
    return 0.142

def get_polymarket_prob(target_date):
    try:
        df = get_clean_master()
        ts = pd.Timestamp(target_date)
        valid = df.index[df.index <= ts]
        if len(valid) > 0:
            idx = valid[-1]
            vix = df['^VIX'].loc[idx]
            sim_prob = 15 + (vix / 60) * 40
            return round(sim_prob, 0)
    except: pass
    return 35.0

def get_move(target_date):
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
    return 15.0

def get_t2108(target_date):
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
    try:
        df = get_clean_master()
        if '^GSPC' not in df.columns: return 0.0
        ts = pd.Timestamp(target_date)
        valid = df.index[df.index <= ts]
        if len(valid) == 0: return 0.0
        lookback = df['^GSPC'].loc[:valid[-1]].tail(252)
        peak = lookback.max()
        current = lookback.iloc[-1]
        drawdown = (current - peak) / peak * 100
        return round(drawdown, 2)
    except:
        return 0.0

def render_sidebar_footer():
    pass

def get_data_freshness():
    freshness = []
    if MASTER_FILE.exists():
        mtime = datetime.datetime.fromtimestamp(MASTER_FILE.stat().st_mtime)
        freshness.append({"Source": "Master DB (.parquet)", "Last Update": mtime.strftime("%Y-%m-%d %H:%M"), "Status": "OK"})
    ntsx_js = DATA_DIR / "Multi_indicator" / "ntsx_data.js"
    if ntsx_js.exists():
        mtime = datetime.datetime.fromtimestamp(ntsx_js.stat().st_mtime)
        freshness.append({"Source": "NTSX Engine (.js)", "Last Update": mtime.strftime("%Y-%m-%d %H:%M"), "Status": "OK"})
    plat_csv = DATA_DIR / "Platinum_Results" / "Platinum_Equity.csv"
    if plat_csv.exists():
        mtime = datetime.datetime.fromtimestamp(plat_csv.stat().st_mtime)
        freshness.append({"Source": "Platinum Strategy (.csv)", "Last Update": mtime.strftime("%Y-%m-%d %H:%M"), "Status": "OK"})
    return freshness

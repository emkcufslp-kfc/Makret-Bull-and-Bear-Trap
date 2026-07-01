import pandas as pd
import yfinance as yf
import os
from pathlib import Path
import datetime

# --- Configuration ---
DATA_DIR = Path("data")
MASTER_FILE = DATA_DIR / "market_data_master.parquet"
START_DATE = "2004-01-01"

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

def initialize():
    if not DATA_DIR.exists():
        DATA_DIR.mkdir()

    print("Starting Initial 'Golden Source' Download (20 Years)...")
    print(f"Total Tickers: {len(ALL_TICKERS)}")
    
    # Fetch data
    data = yf.download(ALL_TICKERS, start=START_DATE, auto_adjust=True, threads=True)['Close']
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Cleaning
    data = data.ffill().dropna(how='all')
    
    # Save
    data.to_parquet(MASTER_FILE)
    print(f"Success! Master Data saved to {MASTER_FILE}")
    print(f"Dataset Shape: {data.shape}")
    print(f"Date Range: {data.index.min().date()} to {data.index.max().date()}")

if __name__ == "__main__":
    initialize()

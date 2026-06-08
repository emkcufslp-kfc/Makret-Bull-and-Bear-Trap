import yfinance as yf
import pandas as pd
import json
import os
import sys
from pathlib import Path

# Get path relative to this script's location (backend/ folder)
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = Path(CURR_DIR).parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.yfinance_utils import configure_yfinance_cache

configure_yfinance_cache(REPO_ROOT)

# Output to root/data/Multi_indicator/
OUT_DIR = os.path.abspath(os.path.join(CURR_DIR, "../data/Multi_indicator"))
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "spy_data.js")

try:
    print("Downloading SPY data for FTD Dashboard (Max history)...")
    df = yf.download("SPY", period="max", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
except Exception as e:
    print(f"yfinance download failed: {e}. Falling back to local master DB.")
    try:
        from utils.data_engine import get_master_data
        master_df = get_master_data()
        if 'SPY' in master_df.columns:
            df = pd.DataFrame({
                'Open': master_df['SPY'],
                'High': master_df['SPY'],
                'Low': master_df['SPY'],
                'Close': master_df['SPY'],
                'Volume': 0.0
            })
            df.dropna(inplace=True)
        else:
            df = pd.DataFrame()
    except Exception as inner_e:
        print(f"Failed to load master DB: {inner_e}")
        df = pd.DataFrame()

data = []
for index, row in df.iterrows():
    date_str = index.strftime('%Y-%m-%d')
    data.append({
        "ticker": "SPY",
        "date": date_str,
        "open": float(row["Open"]),
        "high": float(row["High"]),
        "low": float(row["Low"]),
        "close": float(row["Close"]),
        "volume": float(row["Volume"])
    })

js_content = "const spyHistoricalData = " + json.dumps(data) + ";"

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(js_content)

print(f"Successfully wrote {len(data)} rows to {OUT_PATH}!")

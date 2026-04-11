import yfinance as yf
import pandas as pd
import json
import os

# Get path relative to this script's location (backend/ folder)
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
# Output to root/data/Multi_indicator/
OUT_DIR = os.path.abspath(os.path.join(CURR_DIR, "../data/Multi_indicator"))
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "spy_data.js")

print("Downloading SPY data for FTD Dashboard (Max history)...")
df = yf.download("SPY", period="max", interval="1d", progress=False, auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df.dropna(inplace=True)

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

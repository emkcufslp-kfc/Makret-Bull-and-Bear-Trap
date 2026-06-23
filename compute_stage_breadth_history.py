"""
compute_stage_breadth_history.py
One-time script: download 20 years of OHLCV for the top-100 tickers,
run the Market Stage engine on each, and save daily stage breadth
to .tmp/market_stage_validation/output/stage_breadth_history.csv

Run from the project root:
    python compute_stage_breadth_history.py

Output file is then loaded automatically by the Early Warning dashboard.
Estimated time: 3–6 minutes depending on internet speed.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

ROOT_DIR   = Path(__file__).resolve().parent
MODEL_DIR  = ROOT_DIR / "market_stage_model"
OUTPUT_DIR = ROOT_DIR / ".tmp" / "market_stage_validation" / "output"
COV_PATH   = OUTPUT_DIR / "data_coverage.csv"
OUT_PATH   = OUTPUT_DIR / "stage_breadth_history.csv"

sys.path.insert(0, str(MODEL_DIR))
from engine import compute_market_stages  # noqa: E402

START_DATE = "2003-01-01"   # warm-up for 34-bar VWMA needs ~34 trading days before first valid bar
END_DATE   = None           # None = download through today
MIN_BARS   = 60             # skip tickers with fewer than this many rows

# ── Load ticker list ───────────────────────────────────────────────────────────
if not COV_PATH.exists():
    print(f"ERROR: coverage file not found at {COV_PATH}")
    sys.exit(1)

cov     = pd.read_csv(COV_PATH)
tickers = cov["Ticker"].dropna().str.upper().unique().tolist()
print(f"Tickers to process: {len(tickers)}")
print(f"Downloading OHLCV from {START_DATE} …")

# ── Download in a single batch call (faster than one-by-one) ──────────────────
raw = yf.download(
    tickers,
    start=START_DATE,
    interval="1d",
    auto_adjust=True,
    progress=True,
    threads=True,
)

if raw.empty:
    print("ERROR: yfinance returned no data.")
    sys.exit(1)

# yfinance returns MultiIndex (Price, Ticker) columns when multiple tickers
if isinstance(raw.columns, pd.MultiIndex):
    price_fields = raw.columns.get_level_values(0).unique().tolist()
    print(f"Downloaded fields: {price_fields}")
else:
    print("Single-ticker format — wrapping.")
    raw.columns = pd.MultiIndex.from_product([raw.columns, [tickers[0]]])

# ── Compute stages and accumulate daily breadth ────────────────────────────────
all_stages: list[pd.DataFrame] = []
failed = []

for i, ticker in enumerate(tickers, 1):
    try:
        # Extract single-ticker OHLCV
        if isinstance(raw.columns, pd.MultiIndex):
            df_t = raw.xs(ticker, axis=1, level=1).copy()
        else:
            df_t = raw.copy()

        df_t = df_t.dropna(subset=["Close", "Volume"])
        if len(df_t) < MIN_BARS:
            failed.append(f"{ticker} (only {len(df_t)} rows)")
            continue

        # Run stage engine
        staged = compute_market_stages(df_t)
        stage_col = staged[["Stage"]].copy()
        stage_col.columns = ["Stage"]
        stage_col["Ticker"] = ticker
        all_stages.append(stage_col)

        if i % 10 == 0 or i == len(tickers):
            print(f"  [{i}/{len(tickers)}] {ticker} — {len(stage_col)} bars ✓")

    except Exception as exc:
        failed.append(f"{ticker} ({exc})")

if failed:
    print(f"\nSkipped {len(failed)} tickers: {', '.join(failed[:20])}")

print(f"\nSuccessfully processed {len(all_stages)} tickers.")

# ── Aggregate to daily breadth ─────────────────────────────────────────────────
combined = pd.concat(all_stages)
combined.index = pd.to_datetime(combined.index)
combined = combined[combined["Stage"] != "Insufficient History"]

breadth = (
    combined.groupby(combined.index)["Stage"]
    .value_counts(normalize=True)
    .unstack(fill_value=0) * 100
)

# Ensure all four stage columns exist
for col in ["Acceleration", "Accumulation", "Distribution", "Deceleration"]:
    if col not in breadth.columns:
        breadth[col] = 0.0

breadth = breadth[["Acceleration", "Accumulation", "Distribution", "Deceleration"]]
breadth.index.name = "date"
breadth = breadth.sort_index()

# ── Save ───────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
breadth.to_csv(OUT_PATH)
print(f"\nSaved {len(breadth):,} daily rows → {OUT_PATH}")
print(f"Date range: {breadth.index[0].date()} → {breadth.index[-1].date()}")
print("\nSample (last 5 rows):")
print(breadth.tail())
print("\nDone. Refresh the Early Warning dashboard to see the full history.")

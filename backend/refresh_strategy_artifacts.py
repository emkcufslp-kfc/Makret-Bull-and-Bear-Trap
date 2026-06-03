from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.strategies.ensemble_top100_engine import START_DATE, TOP100_ETFS
from utils.data_engine import get_clean_master
from utils.model_change_monitor import build_warning_mode_events

EXPORTS_DIR = ROOT_DIR / "exports"
STRATEGY_DATA_DIR = ROOT_DIR / "backend" / "strategies" / "data"
WORKSHEET_NAME = "model_change_worksheet_full_history.csv"
TOP100_PRICE_NAME = "top100_etf_prices.csv"


def _today_us_eastern() -> dt.date:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=5)).date()


def refresh_model_change_worksheet() -> Path:
    STRATEGY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    master = get_clean_master().ffill().dropna(how="all")
    if master.empty:
        raise RuntimeError("Master dataset is empty; cannot build model change worksheet.")

    end_date = min(master.index.max().date(), _today_us_eastern())
    worksheet = build_warning_mode_events(end_date)
    if worksheet.empty:
        raise RuntimeError("Model change worksheet build returned no rows.")

    bundled_path = STRATEGY_DATA_DIR / WORKSHEET_NAME
    export_path = EXPORTS_DIR / WORKSHEET_NAME
    worksheet.to_csv(bundled_path, index=False)
    worksheet.to_csv(export_path, index=False)
    return bundled_path


def refresh_top100_price_cache() -> Path:
    STRATEGY_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STRATEGY_DATA_DIR / TOP100_PRICE_NAME
    today = _today_us_eastern()

    tickers = list(dict.fromkeys(TOP100_ETFS + ["SPY"]))
    existing = pd.DataFrame()
    if out_path.exists():
        existing = pd.read_csv(out_path, index_col=0, parse_dates=True).sort_index()
        existing.index = pd.to_datetime(existing.index)

    if existing.empty:
        fetch_start = START_DATE.strftime("%Y-%m-%d")
    else:
        fetch_start = (existing.index.max().date() + dt.timedelta(days=1)).strftime("%Y-%m-%d")

    fresh = pd.DataFrame()
    if pd.Timestamp(fetch_start).date() <= today:
        fresh = yf.download(tickers, start=fetch_start, auto_adjust=True, progress=False)["Close"]
        if isinstance(fresh.columns, pd.MultiIndex):
            fresh.columns = fresh.columns.get_level_values(0)
        fresh = fresh.loc[:, ~fresh.columns.duplicated()].sort_index()

    if existing.empty and fresh.empty:
        raise RuntimeError("Unable to build top-100 ETF price cache from yfinance.")

    combined = existing if fresh.empty else pd.concat([existing, fresh])
    combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index().ffill()
    keep = [ticker for ticker in tickers if ticker in combined.columns]
    combined = combined[keep]
    combined.to_csv(out_path)
    return out_path


def refresh_all_strategy_artifacts() -> list[Path]:
    return [
        refresh_model_change_worksheet(),
        refresh_top100_price_cache(),
    ]


if __name__ == "__main__":
    for artifact in refresh_all_strategy_artifacts():
        print(f"updated {artifact}")

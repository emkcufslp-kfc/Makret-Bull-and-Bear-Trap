"""
backend/refresh_market_stage_data.py
=====================================
Rebuilds the Market Stage / Early Warning data (stage breadth, forward
returns, ticker coverage) AND the market_stage_model top-100 phase-shift
scan snapshot — all from a static top-100 universe + yfinance, with no
dependency on the external `D:\\Codex projects\\US-data` loader (which only
exists on one Windows machine and is unreachable from GitHub Actions).

Universe: data/universe_top100.csv (Ticker, Name, Sector) — a large/mega-cap
snapshot. Composition is static (doesn't auto-track market-cap rank changes);
refresh it manually/occasionally if membership needs updating. Prices and
stage classifications for those 100 tickers refresh every run.

Outputs:
  .tmp/market_stage_validation/output/data_coverage.csv
  .tmp/market_stage_validation/output/stage_breadth_history.csv
  .tmp/market_stage_validation/output/stage_forward_returns.csv
  .tmp/market_stage_validation/output/stage_forward_summary.csv
  market_stage_model/top100_scan_snapshot.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "market_stage_model"
OUT_DIR = ROOT_DIR / ".tmp" / "market_stage_validation" / "output"
UNIVERSE_PATH = ROOT_DIR / "data" / "universe_top100.csv"
SNAPSHOT_PATH = MODEL_DIR / "top100_scan_snapshot.json"

START_DATE = "2003-01-01"
STAGES = ["Acceleration", "Accumulation", "Distribution", "Deceleration"]
MIN_BARS = 60

sys.path.insert(0, str(MODEL_DIR))
from engine import compute_market_stages, scan_phase_shifts  # noqa: E402


def scan_both_shifts(scan_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Same 'Both Shifts' combination market_stage_model/app.py uses, without
    importing app.py itself (which pulls in streamlit at module scope)."""
    frames = [
        scan_phase_shifts(scan_data, "Acceleration Shift"),
        scan_phase_shifts(scan_data, "Deceleration Shift"),
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"Universe file missing: {UNIVERSE_PATH}")
    uni = pd.read_csv(UNIVERSE_PATH)
    uni["Ticker"] = uni["Ticker"].str.upper()
    return uni


def download_ohlcv(tickers: list[str]) -> pd.DataFrame:
    _log(f"Downloading {len(tickers)} tickers from {START_DATE} …")
    t0 = time.time()
    raw = yf.download(tickers, start=START_DATE, interval="1d",
                       auto_adjust=True, progress=False, threads=True)
    _log(f"Download done: {raw.shape} ({time.time()-t0:.0f}s)")
    return raw


def per_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            df_t = raw.xs(ticker, axis=1, level=1).copy()
        except KeyError:
            return pd.DataFrame()
    else:
        df_t = raw.copy()
    return df_t.dropna(subset=["Close", "Volume"])


def refresh_market_stage_data() -> dict:
    uni = load_universe()
    tickers = uni["Ticker"].tolist()
    name_map = dict(zip(uni["Ticker"], uni["Name"]))
    sector_map = dict(zip(uni["Ticker"], uni["Sector"]))

    raw = download_ohlcv(tickers)
    if raw.empty:
        raise RuntimeError("yfinance returned no data for the top-100 universe.")

    coverage_rows = []
    stage_frames = []
    forward_rows = []
    scan_data: dict[str, pd.DataFrame] = {}
    missing = []

    for ticker in tickers:
        df_t = per_ticker_frame(raw, ticker)
        if len(df_t) < MIN_BARS:
            missing.append(ticker)
            continue

        processed = compute_market_stages(df_t)
        if processed.empty:
            missing.append(ticker)
            continue

        coverage_rows.append({
            "Ticker": ticker, "Name": name_map.get(ticker, ""),
            "Sector": sector_map.get(ticker, ""), "Rows": len(processed),
            "Start": processed.index[0].date().isoformat(),
            "End": processed.index[-1].date().isoformat(),
            "Latest Stage": processed["Stage"].iloc[-1],
        })

        sc = processed[["Stage"]].copy()
        sc["Ticker"] = ticker
        stage_frames.append(sc)

        scan_data[ticker] = df_t.tail(260)  # enough history for the scan (needs ~35 bars)

        for idx_dt, row in processed.iterrows():
            stage = row.get("Stage", "")
            if stage in STAGES:
                pos = processed.index.get_loc(idx_dt)
                for h in [1, 5, 10, 20]:
                    if pos + h < len(processed):
                        ret = processed["Close"].iloc[pos + h] / processed["Close"].iloc[pos] - 1.0
                        forward_rows.append({
                            "Ticker": ticker, "Sector": sector_map.get(ticker, ""),
                            "Signal Date": idx_dt.date().isoformat(), "Horizon Days": h,
                            "Stage": stage, "Return": float(ret),
                        })

    if not stage_frames:
        raise RuntimeError("No tickers produced usable stage data.")

    # ── Stage breadth history ──────────────────────────────────────────
    combined = pd.concat(stage_frames)
    combined.index = pd.to_datetime(combined.index)
    combined = combined[combined["Stage"] != "Insufficient History"]
    breadth = (
        combined.groupby(combined.index)["Stage"]
        .value_counts(normalize=True).unstack(fill_value=0) * 100
    )
    for col in STAGES:
        if col not in breadth.columns:
            breadth[col] = 0.0
    breadth = breadth[STAGES]
    breadth.index.name = "date"
    breadth = breadth.sort_index()

    # ── Forward returns + summary ──────────────────────────────────────
    forward = pd.DataFrame(forward_rows)
    stage_summary = (
        forward.groupby(["Stage", "Horizon Days"]).agg(
            Observations=("Return", "size"), Mean_Return=("Return", "mean"),
            Median_Return=("Return", "median"), Win_Rate=("Return", lambda s: (s > 0).mean())
        ).reset_index() if not forward.empty else pd.DataFrame()
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv(OUT_DIR / "data_coverage.csv", index=False)
    breadth.to_csv(OUT_DIR / "stage_breadth_history.csv")
    forward.to_csv(OUT_DIR / "stage_forward_returns.csv", index=False)
    stage_summary.to_csv(OUT_DIR / "stage_forward_summary.csv", index=False)

    _log(f"Stage breadth: {len(breadth)} rows ({breadth.index[0].date()} -> {breadth.index[-1].date()})")
    _log(f"Coverage: {len(coverage_rows)} tickers ok, {len(missing)} skipped")

    # ── Top-100 phase-shift scan snapshot (reuses the same downloaded data) ──
    scan_results = scan_both_shifts(scan_data)
    if not scan_results.empty:
        scan_results = scan_results.copy()
        scan_results["Sector"] = scan_results["Ticker"].map(sector_map)
    as_of = None if scan_results.empty else str(scan_results["As Of"].max())

    snapshot = {
        "generated_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
        "source_dir": "yfinance (static universe: data/universe_top100.csv)",
        "source_universe": str(UNIVERSE_PATH.relative_to(ROOT_DIR)),
        "scan_mode": "Both Shifts",
        "universe_method": "Static top-100 large/mega-cap snapshot, priced live via yfinance",
        "universe_count": int(len(uni)),
        "loaded_count": int(len(scan_data)),
        "missing_count": int(len(missing)),
        "missing_tickers": missing,
        "as_of": as_of,
        "results": scan_results.to_dict("records"),
    }
    SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    _log(f"Wrote {SNAPSHOT_PATH} ({len(scan_results)} phase-shift hits)")

    return {
        "coverage_count": len(coverage_rows),
        "missing_count": len(missing),
        "breadth_rows": len(breadth),
        "breadth_end": str(breadth.index[-1].date()),
        "scan_hits": len(scan_results),
    }


if __name__ == "__main__":
    result = refresh_market_stage_data()
    print(json.dumps(result, indent=2))

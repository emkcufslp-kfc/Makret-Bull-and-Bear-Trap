from __future__ import annotations

import datetime as dt
import html
import json
import math
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from engine import compute_market_stages, latest_stage_snapshot, scan_phase_shifts

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 on some Streamlit deployments.
    tomllib = None


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
CACHE_DIR = APP_DIR / ".cache" / "yfinance"
DEFAULT_US_DATA_DIR = Path(r"D:\Codex projects\US-data")
TOP100_SCAN_SNAPSHOT_FILE = APP_DIR / "top100_scan_snapshot.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
try:
    yf.cache.set_cache_location(str(CACHE_DIR))
except Exception:
    pass


STAGE_STYLE = {
    "Acceleration": ("#16a34a", "Bullish stack"),
    "Accumulation": ("#ca8a04", "Base building"),
    "Distribution": ("#f97316", "Pressure building"),
    "Deceleration": ("#dc2626", "Bearish stack"),
    "Insufficient History": ("#64748b", "Need more bars"),
}

POLYGON_BASE_URL = "https://api.polygon.io"
WARNING_RYG = {
    "Critical": "Red",
    "Elevated": "Red",
    "Guarded": "Yellow",
    "Stable": "Green",
}
MARKET_SUMMARY_RYG = {
    "HIGH RISK": "Red",
    "EARLY WARNING": "Yellow",
    "LOW RISK REGIME": "Green",
}
RYG_DOT_COUNT = {"Green": 1, "Yellow": 2, "Red": 3}
RYG_DOT_COLOR = {"Green": "#22c55e", "Yellow": "#f59e0b", "Red": "#ef4444"}
RYG_COUNT_TEMPLATE = {"Red": 0, "Yellow": 0, "Green": 0, "Total": 0}


st.set_page_config(page_title="Market Stage Model", page_icon="MS", layout="wide")


def drop_unconfirmed_daily_bar(data: pd.DataFrame, include_partial: bool) -> pd.DataFrame:
    if include_partial or data.empty:
        return data

    now_et = dt.datetime.now(ZoneInfo("America/New_York"))
    last_date = pd.Timestamp(data.index[-1]).date()
    today_et = now_et.date()
    market_close_buffer = dt.time(16, 15)
    if last_date >= today_et and now_et.time() < market_close_buffer:
        return data.iloc[:-1]
    return data


def flatten_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance MultiIndex columns into plain OHLCV-style columns."""
    if data is None or data.empty:
        return pd.DataFrame()
    out = data.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]
    return out


def close_series(data: pd.DataFrame) -> pd.Series:
    """Return a numeric close series even when yfinance returns duplicate-shaped columns."""
    if data is None or data.empty or "Close" not in data.columns:
        return pd.Series(dtype="float64")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.to_numeric(close, errors="coerce").dropna()


def _ensure_repo_import_path() -> None:
    root_str = str(ROOT_DIR)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


@st.cache_data(ttl=1800, show_spinner=False)
def warning_ryg_for_date(date_text: str) -> str:
    _ensure_repo_import_path()
    try:
        from utils.warning_dashboard import build_warning_dashboard
    except Exception:
        return "Unavailable"

    try:
        payload = build_warning_dashboard(pd.Timestamp(date_text).date())
    except Exception:
        return "Unavailable"
    if not payload:
        return "Unavailable"

    return WARNING_RYG.get(str(payload.get("warning_level", "")), "Unavailable")


def _safe_float(series: pd.Series, key: str, default: float = 0.0) -> float:
    value = series.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


@st.cache_data(ttl=1800, show_spinner=False)
def market_summary_ryg_for_date(date_text: str) -> str:
    return market_summary_ryg_lookup((date_text,)).get(date_text, "Unavailable")


def render_ryg_dots(status: str) -> str:
    normalized = str(status)
    count = RYG_DOT_COUNT.get(normalized, 0)
    color = RYG_DOT_COLOR.get(normalized, "#64748b")
    label = html.escape(normalized if count else "Unavailable")
    dot_html = "".join(
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:999px;background:{color};box-shadow:0 0 8px {color};"></span>'
        for _ in range(count)
    )
    count_text = "dot" if count == 1 else "dots"
    return (
        '<span style="display:inline-flex;align-items:center;gap:6px;white-space:nowrap;">'
        f'<span style="display:inline-flex;gap:3px;min-width:30px;">{dot_html}</span>'
        f'<span style="color:{color};font-weight:800;">{count} {count_text}</span>'
        f'<span style="color:#cbd5e1;">{label}</span>'
        "</span>"
    )


def _empty_ryg_counts() -> dict[str, int]:
    return dict(RYG_COUNT_TEMPLATE)


def _count_ryg_values(values: list[str]) -> dict[str, int]:
    counts = _empty_ryg_counts()
    for value in values:
        if value in {"Red", "Yellow", "Green"}:
            counts[value] += 1
            counts["Total"] += 1
    return counts


def _summary_status_to_ryg(status: str) -> str | None:
    text = str(status).upper()
    if not text:
        return None

    if "HIGH RISK" in text or "BEARISH" in text or "LOW CONFIDENCE" in text or "RISK OFF" in text or "CRISIS" in text:
        return "Red"
    if "EARLY WARNING" in text or "CAUTION" in text or "NEUTRAL" in text or "REBALANCE" in text:
        return "Yellow"
    if "LOW RISK" in text or "BULLISH" in text or "NORMAL" in text or "HIGH CONFIDENCE" in text or "HOLD" in text:
        return "Green"
    return None


def _combined_execution_to_ryg(execution_status: str) -> str | None:
    text = str(execution_status).upper()
    if "REBALANCE" in text:
        return "Yellow"
    if "HOLD" in text:
        return "Green"
    return _summary_status_to_ryg(text)


@st.cache_data(ttl=1800, show_spinner=False)
def warning_ryg_counts_lookup(date_texts: tuple[str, ...]) -> dict[str, dict[str, int]]:
    _ensure_repo_import_path()
    try:
        from utils.data_engine import get_clean_master
        from utils.warning_dashboard import _build_indicator_rows
    except Exception:
        return {date_text: _empty_ryg_counts() for date_text in date_texts}

    try:
        master = get_clean_master().ffill().dropna(how="all")
    except Exception:
        return {date_text: _empty_ryg_counts() for date_text in date_texts}
    if master.empty:
        return {date_text: _empty_ryg_counts() for date_text in date_texts}

    out: dict[str, dict[str, int]] = {}
    status_map = {"Warning": "Red", "Watch": "Yellow", "Normal": "Green"}
    for date_text in date_texts:
        try:
            valid_dates = master.index[master.index <= pd.Timestamp(date_text)]
            if len(valid_dates) == 0:
                out[date_text] = _empty_ryg_counts()
                continue

            actual_date = valid_dates[-1]
            data = master.loc[:actual_date].copy()
            if len(data) < 220:
                out[date_text] = _empty_ryg_counts()
                continue

            indicators = _build_indicator_rows(data, actual_date)
            out[date_text] = _count_ryg_values([status_map.get(str(row.status), "") for row in indicators])
        except Exception:
            out[date_text] = _empty_ryg_counts()
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def market_summary_ryg_counts_lookup(date_texts: tuple[str, ...]) -> dict[str, dict[str, int]]:
    _ensure_repo_import_path()
    try:
        from backend.strategies.combined_macro_rwra import compute_combined_snapshot
        from utils.data_engine import get_clean_master
        from utils.model_change_monitor import (
            calc_200ma_strategy,
            calc_bear_trap,
            calc_bull_trap,
            calc_etf_rotation,
            calc_market_regime,
            calc_meta_indicator,
        )
    except Exception:
        return {date_text: _empty_ryg_counts() for date_text in date_texts}

    try:
        master = get_clean_master().ffill().dropna(how="all")
    except Exception:
        return {date_text: _empty_ryg_counts() for date_text in date_texts}
    if master.empty:
        return {date_text: _empty_ryg_counts() for date_text in date_texts}

    out: dict[str, dict[str, int]] = {}
    for date_text in date_texts:
        try:
            valid_dates = master.index[master.index <= pd.Timestamp(date_text)]
            if len(valid_dates) == 0:
                out[date_text] = _empty_ryg_counts()
                continue

            actual_date = valid_dates[-1]
            data = master.loc[:actual_date].copy()
            if len(data) < 260:
                out[date_text] = _empty_ryg_counts()
                continue

            combined = compute_combined_snapshot(actual_date.date())
            colors = [
                _summary_status_to_ryg(calc_market_regime(data)["status"]),
                _summary_status_to_ryg(calc_bear_trap(data)["risk_level"]),
                _summary_status_to_ryg(calc_bull_trap(data)["market_status"]),
                _summary_status_to_ryg(calc_etf_rotation(data)["status"]),
                _summary_status_to_ryg(calc_200ma_strategy(data)["trend_status"]),
                _summary_status_to_ryg(calc_meta_indicator(data, actual_date.date())["status"]),
                _combined_execution_to_ryg(combined.execution_status),
            ]
            out[date_text] = _count_ryg_values([color for color in colors if color is not None])
        except Exception:
            out[date_text] = _empty_ryg_counts()
    return out


def render_ryg_distribution(value: object) -> str:
    counts = value if isinstance(value, dict) else _empty_ryg_counts()
    groups = []
    for label in ["Red", "Yellow", "Green"]:
        count = int(counts.get(label, 0))
        color = RYG_DOT_COLOR[label]
        dot_html = "".join(
            f'<span style="display:inline-block;width:7px;height:7px;border-radius:999px;background:{color};box-shadow:0 0 7px {color};"></span>'
            for _ in range(count)
        )
        groups.append(
            '<span style="display:inline-flex;align-items:center;gap:4px;min-width:58px;">'
            f'<span style="display:inline-flex;gap:2px;min-width:24px;">{dot_html}</span>'
            f'<span style="color:{color};font-weight:800;">{label[0]} {count}</span>'
            "</span>"
        )
    total = int(counts.get("Total", 0))
    return (
        '<span style="display:inline-flex;align-items:center;gap:7px;white-space:nowrap;">'
        f"{''.join(groups)}"
        f'<span style="color:#94a3b8;">/{total}</span>'
        "</span>"
    )


def render_history_table(display_history: pd.DataFrame) -> str:
    headers = "".join(
        f'<th style="padding:9px 10px;text-align:left;border-bottom:1px solid #334155;color:#cbd5e1;font-size:12px;">{html.escape(str(col))}</th>'
        for col in display_history.columns
    )
    rows = []
    numeric_cols = {"Close", "VMA 21", "Volume"}
    for _, row in display_history.iterrows():
        cells = []
        for col in display_history.columns:
            value = row[col]
            align = "right" if col in numeric_cols else "left"
            if col in {"Warning RYG", "Summary RYG"}:
                cell_value = render_ryg_distribution(value)
            else:
                cell_value = html.escape(str(value))
            cells.append(
                f'<td style="padding:8px 10px;text-align:{align};border-bottom:1px solid rgba(51,65,85,0.55);color:#e2e8f0;font-size:12px;vertical-align:middle;">{cell_value}</td>'
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
    <div style="overflow-x:auto;border:1px solid rgba(51,65,85,0.9);border-radius:8px;background:#0f172a;">
        <table style="border-collapse:collapse;width:100%;min-width:1120px;">
            <thead style="background:#111827;"><tr>{headers}</tr></thead>
            <tbody>{''.join(rows)}</tbody>
        </table>
    </div>
    """


@st.cache_data(ttl=1800, show_spinner=False)
def warning_ryg_lookup(date_texts: tuple[str, ...]) -> dict[str, str]:
    _ensure_repo_import_path()
    try:
        from utils.data_engine import get_clean_master
        from utils.warning_dashboard import _build_indicator_rows
    except Exception:
        return {date_text: "Unavailable" for date_text in date_texts}

    try:
        master = get_clean_master().ffill().dropna(how="all")
    except Exception:
        return {date_text: "Unavailable" for date_text in date_texts}
    if master.empty:
        return {date_text: "Unavailable" for date_text in date_texts}

    out: dict[str, str] = {}
    for date_text in date_texts:
        try:
            valid_dates = master.index[master.index <= pd.Timestamp(date_text)]
            if len(valid_dates) == 0:
                out[date_text] = "Unavailable"
                continue

            actual_date = valid_dates[-1]
            data = master.loc[:actual_date].copy()
            if len(data) < 220:
                out[date_text] = "Unavailable"
                continue

            indicators = _build_indicator_rows(data, actual_date)
            if not indicators:
                out[date_text] = "Unavailable"
                continue

            scores = [int(row.score) for row in indicators]
            statuses = [str(row.status) for row in indicators]
            total_score = sum(scores)
            warning_count = statuses.count("Warning")
            watch_count = statuses.count("Watch")
            if total_score >= 11 or warning_count >= 3:
                level = "Critical"
            elif total_score >= 7 or warning_count >= 2:
                level = "Elevated"
            elif total_score >= 4 or watch_count >= 3:
                level = "Guarded"
            else:
                level = "Stable"
            out[date_text] = WARNING_RYG.get(level, "Unavailable")
        except Exception:
            out[date_text] = "Unavailable"
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def market_summary_ryg_lookup(date_texts: tuple[str, ...]) -> dict[str, str]:
    _ensure_repo_import_path()
    try:
        from utils.data_engine import get_clean_master, get_gex, get_hy_spread, get_move, get_t2108
    except Exception:
        return {date_text: "Unavailable" for date_text in date_texts}

    try:
        master = get_clean_master().ffill().dropna(how="all")
    except Exception:
        return {date_text: "Unavailable" for date_text in date_texts}
    if master.empty:
        return {date_text: "Unavailable" for date_text in date_texts}

    out: dict[str, str] = {}
    for date_text in date_texts:
        try:
            valid_dates = master.index[master.index <= pd.Timestamp(date_text)]
            if len(valid_dates) == 0:
                out[date_text] = "Unavailable"
                continue

            actual_date = valid_dates[-1]
            d = master.loc[:actual_date].copy()
            if len(d) < 200 or "^GSPC" not in d.columns:
                out[date_text] = "Unavailable"
                continue

            latest = d.iloc[-1]
            sp_price = _safe_float(latest, "^GSPC")
            dma200 = float(d["^GSPC"].rolling(200).mean().iloc[-1])
            vix = _safe_float(latest, "^VIX", 20.0)
            vix3m = _safe_float(latest, "^VIX3M", 21.0)
            hy_spread_pct = get_hy_spread(actual_date.date())
            move = get_move(actual_date.date())
            t2108 = get_t2108(actual_date.date())
            dxy = _safe_float(latest, "DX-Y.NYB", 100.0)
            liquidity = 7.5e12
            spy_gex = get_gex(actual_date.date())

            score = 0
            if sp_price < dma200:
                score += 15
            if hy_spread_pct > 5:
                score += 20
            if move > 100:
                score += 15
            if vix > 25:
                score += 10
            if vix > vix3m:
                score += 10
            if dxy > 105:
                score += 10
            if t2108 < 40:
                score += 10
            if spy_gex < 0:
                score += 5
            if liquidity < 7.0e12:
                score += 5

            probability = min(score, 100)
            if probability < 30:
                status = "LOW RISK REGIME"
            elif probability < 55:
                status = "EARLY WARNING"
            else:
                status = "HIGH RISK"
            out[date_text] = MARKET_SUMMARY_RYG.get(status, "Unavailable")
        except Exception:
            out[date_text] = "Unavailable"
    return out


@st.cache_data(ttl=300, show_spinner=False)
def load_price_history(ticker: str, days: int, include_partial: bool) -> pd.DataFrame:
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=days)
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame()
    data = flatten_yfinance_columns(data)
    return drop_unconfirmed_daily_bar(data, include_partial)


@st.cache_data(ttl=86400, show_spinner=False)
def load_ticker_display_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        return ticker.upper()

    for key in ("longName", "shortName", "displayName"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ticker.upper()


def get_history(ticker: str, days: int, include_partial: bool) -> tuple[pd.DataFrame, str]:
    data = load_price_history(ticker, days, include_partial)
    if not data.empty:
        return data, "yfinance historical OHLCV"
    return pd.DataFrame(), "Unavailable"


def get_us_data_dir_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.getenv("US_DATA_DIR", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    try:
        secret_path = str(st.secrets.get("US_DATA_DIR", "")).strip()
    except Exception:
        secret_path = ""
    if secret_path:
        candidates.append(Path(secret_path))

    candidates.extend(
        [
            DEFAULT_US_DATA_DIR,
            ROOT_DIR / "US-data",
            ROOT_DIR / "data" / "US-data",
            ROOT_DIR / "data" / "us_data",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def get_us_data_dir() -> Path | None:
    for candidate in get_us_data_dir_candidates():
        if (candidate / "universe.json").exists() and (candidate / "ohlcv").exists():
            return candidate
    return None


def describe_us_data_search_paths() -> str:
    return "; ".join(str(path) for path in get_us_data_dir_candidates())


@st.cache_data(ttl=1800, show_spinner=False)
def load_us_data_universe() -> pd.DataFrame:
    us_data_dir = get_us_data_dir()
    if us_data_dir is None:
        return pd.DataFrame(columns=["Ticker", "Name", "Sector", "Market Cap"])
    universe_file = us_data_dir / "universe.json"

    try:
        payload = json.loads(universe_file.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Name", "Sector", "Market Cap"])

    rows = []
    for item in payload.get("stocks", []):
        ticker = str(item.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        market_cap = item.get("market_cap", 0)
        try:
            market_cap_value = float(market_cap)
        except Exception:
            market_cap_value = 0.0
        if math.isnan(market_cap_value):
            market_cap_value = 0.0
        rows.append(
            {
                "Ticker": ticker,
                "Name": str(item.get("name", "")).strip(),
                "Sector": str(item.get("sector", "Unknown")).strip() or "Unknown",
                "Market Cap": market_cap_value,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Ticker", "Name", "Sector", "Market Cap"])
    return pd.DataFrame(rows).drop_duplicates("Ticker").sort_values("Market Cap", ascending=False).reset_index(drop=True)


def load_us_data_top100_universe() -> pd.DataFrame:
    universe = load_us_data_universe()
    if universe.empty:
        return universe
    us_data_dir = get_us_data_dir()
    if us_data_dir is None:
        return universe.iloc[0:0].copy()
    ohlcv_dir = us_data_dir / "ohlcv"
    available = universe[universe["Ticker"].map(lambda ticker: (ohlcv_dir / f"{ticker}.json").exists())]
    return available.head(100).reset_index(drop=True)


def sector_lookup_from_us_data() -> dict[str, str]:
    universe = load_us_data_universe()
    if universe.empty:
        return {}
    return dict(zip(universe["Ticker"], universe["Sector"]))


@st.cache_data(ttl=1800, show_spinner=False)
def load_us_data_price_history(ticker: str, days: int, include_partial: bool) -> pd.DataFrame:
    us_data_dir = get_us_data_dir()
    if us_data_dir is None:
        return pd.DataFrame()
    path = us_data_dir / "ohlcv" / f"{ticker.upper()}.json"
    if not path.exists():
        return pd.DataFrame()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return pd.DataFrame()

    records = payload.get("data", [])
    if not records:
        return pd.DataFrame()
    data = pd.DataFrame(records)
    if data.empty or "date" not in data.columns:
        return pd.DataFrame()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"]).set_index("date").sort_index()
    data = data.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(data.columns):
        return pd.DataFrame()
    data = data[required].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close", "Volume"])
    if days > 0 and not data.empty:
        start_date = data.index[-1] - pd.Timedelta(days=days)
        data = data.loc[data.index >= start_date]
    return drop_unconfirmed_daily_bar(data, include_partial)


def scan_phase_shifts_for_modes(scan_data: dict[str, pd.DataFrame], scan_mode: str) -> pd.DataFrame:
    if scan_mode == "Both Shifts":
        frames = [
            scan_phase_shifts(scan_data, "Acceleration Shift"),
            scan_phase_shifts(scan_data, "Deceleration Shift"),
        ]
        frames = [frame for frame in frames if not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return scan_phase_shifts(scan_data, scan_mode)


def attach_scan_sectors(results: pd.DataFrame, sector_by_ticker: dict[str, str]) -> pd.DataFrame:
    if results.empty:
        return results
    out = results.copy()
    out.insert(1, "Sector", out["Ticker"].map(sector_by_ticker).fillna("Unknown"))
    return out


@st.cache_data(ttl=300, show_spinner=False)
def load_top100_scan_snapshot() -> dict[str, object]:
    if not TOP100_SCAN_SNAPSHOT_FILE.exists():
        return {}
    try:
        payload = json.loads(TOP100_SCAN_SNAPSHOT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def top100_scan_snapshot_frame(scan_mode: str) -> tuple[pd.DataFrame, dict[str, object]]:
    payload = load_top100_scan_snapshot()
    results = payload.get("results", [])
    if not isinstance(results, list):
        return pd.DataFrame(), payload

    frame = pd.DataFrame(results)
    if frame.empty or "Phase Shift" not in frame.columns:
        return pd.DataFrame(), payload

    if scan_mode == "Acceleration Shift":
        frame = frame[frame["Phase Shift"] == "Entry Into Acceleration"]
    elif scan_mode == "Deceleration Shift":
        frame = frame[frame["Phase Shift"] == "Entry Into Deceleration"]
    return frame.reset_index(drop=True), payload


def get_polygon_api_key() -> str:
    env_key = os.getenv("POLYGON_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        secret_key = st.secrets.get("POLYGON_API_KEY", "")
    except Exception:
        secret_key = ""
    if secret_key:
        return str(secret_key).strip()

    local_secrets = APP_DIR / ".streamlit" / "secrets.toml"
    if tomllib is not None and local_secrets.exists():
        try:
            with local_secrets.open("rb") as handle:
                local_key = tomllib.load(handle).get("POLYGON_API_KEY", "")
        except Exception:
            local_key = ""
        if local_key:
            return str(local_key).strip()
    return ""


def polygon_get_json(path_or_url: str, params: dict[str, object], api_key: str) -> dict[str, object]:
    url = path_or_url if path_or_url.startswith("http") else f"{POLYGON_BASE_URL}{path_or_url}"
    query = dict(params)
    query["apiKey"] = api_key
    response = requests.get(url, params=query, timeout=30)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=86400, show_spinner=False)
def polygon_nyse_tickers(target_date: str, api_key: str) -> set[str]:
    tickers: set[str] = set()
    url = f"{POLYGON_BASE_URL}/v3/reference/tickers"
    params: dict[str, object] = {
        "market": "stocks",
        "exchange": "XNYS",
        "active": "true",
        "date": target_date,
        "limit": 1000,
        "sort": "ticker",
    }

    while url:
        payload = polygon_get_json(url, params, api_key)
        for item in payload.get("results", []) or []:
            ticker = item.get("ticker")
            if isinstance(ticker, str) and ticker:
                tickers.add(ticker)
        url = str(payload.get("next_url") or "")
        params = {}

    return tickers


@st.cache_data(ttl=86400, show_spinner=False)
def polygon_grouped_daily(target_date: str, api_key: str) -> pd.DataFrame:
    payload = polygon_get_json(
        f"/v2/aggs/grouped/locale/us/market/stocks/{target_date}",
        {"adjusted": "false", "include_otc": "false"},
        api_key,
    )
    rows = payload.get("results", []) or []
    if not rows:
        return pd.DataFrame(columns=["ticker", "open", "close", "volume"])

    data = pd.DataFrame(rows)
    data = data.rename(columns={"T": "ticker", "o": "open", "c": "close", "v": "volume"})
    return data[["ticker", "open", "close", "volume"]].dropna(subset=["ticker", "close", "volume"])


def latest_completed_polygon_date(include_partial: bool) -> dt.date:
    today_et = dt.datetime.now(ZoneInfo("America/New_York")).date()
    now_et = dt.datetime.now(ZoneInfo("America/New_York"))
    if include_partial and now_et.time() >= dt.time(9, 45):
        return today_et
    candidate = today_et - dt.timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= dt.timedelta(days=1)
    return candidate


def previous_polygon_session(current_date: dt.date, api_key: str) -> dt.date | None:
    candidate = current_date - dt.timedelta(days=1)
    for _ in range(10):
        if candidate.weekday() < 5:
            grouped = polygon_grouped_daily(candidate.isoformat(), api_key)
            if not grouped.empty:
                return candidate
        candidate -= dt.timedelta(days=1)
    return None


@st.cache_data(ttl=900, show_spinner=False)
def compute_polygon_nyse_trin(target_date: str, api_key: str) -> dict[str, object]:
    current_date = dt.date.fromisoformat(target_date)
    previous_date = previous_polygon_session(current_date, api_key)
    if previous_date is None:
        return {"available": False, "reason": "No previous Polygon trading session found."}

    nyse_tickers = polygon_nyse_tickers(target_date, api_key)
    current = polygon_grouped_daily(target_date, api_key)
    previous = polygon_grouped_daily(previous_date.isoformat(), api_key)
    if current.empty or previous.empty or not nyse_tickers:
        return {"available": False, "reason": "Polygon returned incomplete grouped daily/reference data."}

    current = current[current["ticker"].isin(nyse_tickers)].set_index("ticker")
    previous = previous[previous["ticker"].isin(nyse_tickers)].set_index("ticker")
    joined = current.join(previous[["close"]].rename(columns={"close": "previous_close"}), how="inner")
    joined = joined.dropna(subset=["close", "previous_close", "volume"])
    if joined.empty:
        return {"available": False, "reason": "No overlapping NYSE tickers between current and previous sessions."}

    advancing = joined["close"] > joined["previous_close"]
    declining = joined["close"] < joined["previous_close"]
    adv_issues = int(advancing.sum())
    dec_issues = int(declining.sum())
    adv_volume = float(joined.loc[advancing, "volume"].sum())
    dec_volume = float(joined.loc[declining, "volume"].sum())

    if dec_issues == 0 or adv_issues == 0 or dec_volume == 0 or adv_volume == 0:
        return {"available": False, "reason": "TRIN denominator is zero for this session."}

    trin = (adv_issues / dec_issues) / (adv_volume / dec_volume)
    return {
        "available": True,
        "date": target_date,
        "previous_date": previous_date.isoformat(),
        "trin": trin,
        "adv_issues": adv_issues,
        "dec_issues": dec_issues,
        "adv_volume": adv_volume,
        "dec_volume": dec_volume,
        "universe_size": int(len(joined)),
    }


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #08111f 0%, #0b1220 62%, #0b1220 100%);
        }
        .block-container {
            max-width: 1420px;
            padding-top: 1.1rem;
            padding-bottom: 2rem;
        }
        .stage-hero {
            border: 1px solid rgba(51,65,85,0.9);
            background: #111c2f;
            border-radius: 8px;
            padding: 22px 26px;
            margin: 12px 0 20px 0;
        }
        .stage-badge {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 6px;
            font-weight: 800;
            color: white;
            margin-left: 12px;
        }
        .stage-note {
            color: #94a3b8;
            margin-left: 12px;
            font-size: 1rem;
        }
        div[data-testid="stMetric"] {
            background: #101827;
            border: 1px solid rgba(51,65,85,0.9);
            border-radius: 8px;
            padding: 14px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stage_hero_html(ticker: str, display_name: str, stage: str, as_of: pd.Timestamp, source: str) -> str:
    color, note = STAGE_STYLE.get(stage, ("#64748b", "Unknown"))
    ticker_title = ticker.upper()
    if display_name and display_name.upper() != ticker.upper():
        ticker_title = f"{ticker.upper()} - {display_name}"
    return (
        "<div class='stage-hero'>"
        f"<div style='font-size:1.55rem;font-weight:800;color:#e5e7eb;margin-bottom:10px;'>{ticker_title}</div>"
        "<span style='font-size:1.05rem;font-weight:800;color:#cbd5e1;'>Current Market Stage</span>"
        f"<span class='stage-badge' style='background:{color}'>{stage}</span>"
        f"<span class='stage-note'>{note} | As of {as_of.date().isoformat()} | Source: {source}</span>"
        "</div>"
    )


def render_stage_chart(processed: pd.DataFrame, ticker: str) -> None:
    tail = processed.tail(140)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=tail.index,
            open=tail["Open"],
            high=tail["High"],
            low=tail["Low"],
            close=tail["Close"],
            name=ticker,
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
        )
    )
    fig.add_trace(go.Scatter(x=tail.index, y=tail["VWMA_8"], mode="lines", name="VWMA 8", line=dict(color="#22c55e", width=1.8)))
    fig.add_trace(go.Scatter(x=tail.index, y=tail["VWMA_21"], mode="lines", name="VWMA 21", line=dict(color="#38bdf8", width=1.8)))
    fig.add_trace(go.Scatter(x=tail.index, y=tail["VWMA_34"], mode="lines", name="VWMA 34", line=dict(color="#f97316", width=1.8)))
    fig.add_trace(go.Scatter(x=tail.index, y=tail["VMA_21"], mode="lines", name="VMA 21", line=dict(color="#eab308", width=2.2, dash="dot")))
    fig.update_layout(
        height=440,
        margin=dict(l=8, r=8, t=12, b=8),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", y=1.03, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_snapshot(ticker: str, display_name: str, data: pd.DataFrame, source: str) -> None:
    processed = compute_market_stages(data)
    snapshot = latest_stage_snapshot(ticker, data)
    st.markdown(stage_hero_html(ticker, display_name, snapshot.stage, snapshot.as_of, source), unsafe_allow_html=True)

    volume_delta = "n/a" if np.isnan(snapshot.volume_delta_20d) else f"{snapshot.volume_delta_20d:+.1f}%"
    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;margin-bottom:18px;">
            <div style="background:#101827;border:1px solid rgba(51,65,85,0.9);border-radius:8px;padding:14px 16px;">
                <div style="font-size:.86rem;color:#94a3b8;font-weight:700;">Close</div>
                <div style="font-size:1.75rem;color:#f8fafc;font-weight:800;">${snapshot.close:,.2f}</div>
            </div>
            <div style="background:#101827;border:1px solid rgba(51,65,85,0.9);border-radius:8px;padding:14px 16px;">
                <div style="font-size:.86rem;color:#94a3b8;font-weight:700;">Signal</div>
                <div style="font-size:1.75rem;color:#f8fafc;font-weight:800;">{snapshot.transition_signal}</div>
            </div>
            <div style="background:#101827;border:1px solid rgba(51,65,85,0.9);border-radius:8px;padding:14px 16px;">
                <div style="font-size:.86rem;color:#94a3b8;font-weight:700;">VWMA 8 / 21 / 34</div>
                <div style="font-size:1.18rem;color:#f8fafc;font-weight:800;">{snapshot.vwma_8:,.2f} / {snapshot.vwma_21:,.2f} / {snapshot.vwma_34:,.2f}</div>
            </div>
            <div style="background:#101827;border:1px solid rgba(51,65,85,0.9);border-radius:8px;padding:14px 16px;">
                <div style="font-size:.86rem;color:#94a3b8;font-weight:700;">VMA 21</div>
                <div style="font-size:1.75rem;color:#f8fafc;font-weight:800;">{snapshot.vma_21:,.2f}</div>
            </div>
            <div style="background:#101827;border:1px solid rgba(51,65,85,0.9);border-radius:8px;padding:14px 16px;">
                <div style="font-size:.86rem;color:#94a3b8;font-weight:700;">Volume vs 20D</div>
                <div style="font-size:1.75rem;color:#f8fafc;font-weight:800;">{volume_delta}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_stage_chart(processed, ticker)
    render_stage_history(processed)


def build_last_month_stage_history(processed: pd.DataFrame) -> pd.DataFrame:
    if processed.empty:
        return pd.DataFrame()

    end_date = pd.Timestamp(processed.index[-1])
    start_date = end_date - pd.DateOffset(months=1)
    history = processed.loc[processed.index >= start_date].copy()
    if history.empty:
        return history

    history["Date"] = history.index.date.astype(str)
    history["Signal"] = history["Transition_Signal"].str.replace("Entering ", "Enter ", regex=False)
    history["VWMA Stack"] = history.apply(
        lambda row: f"{row['VWMA_8']:.2f} / {row['VWMA_21']:.2f} / {row['VWMA_34']:.2f}",
        axis=1,
    )
    history_dates = tuple(history["Date"].tolist())
    warning_lookup = warning_ryg_counts_lookup(history_dates)
    market_summary_lookup = market_summary_ryg_counts_lookup(history_dates)
    history["Warning RYG"] = history["Date"].map(lambda date_text: warning_lookup.get(date_text, _empty_ryg_counts()))
    history["Summary RYG"] = history["Date"].map(lambda date_text: market_summary_lookup.get(date_text, _empty_ryg_counts()))
    return history[
        [
            "Date",
            "Close",
            "Stage",
            "Signal",
            "Warning RYG",
            "Summary RYG",
            "VWMA Stack",
            "VMA_21",
            "Volume",
        ]
    ].rename(columns={"VMA_21": "VMA 21"})


def render_stage_history(processed: pd.DataFrame) -> None:
    history = build_last_month_stage_history(processed)
    if history.empty:
        st.info("No completed bars are available for the last 1 month stage history.")
        return

    start_date = history["Date"].iloc[0]
    end_date = history["Date"].iloc[-1]
    st.divider()
    st.subheader("Last 1 Month Stage History")
    st.caption(
        f"Real yfinance daily OHLCV only. Stages are computed with the full loaded history, then displayed from {start_date} to {end_date}."
    )
    st.caption(
        "RYG columns show daily component counts. Summary counts the 7 Market Summary modules; "
        "Warning counts the 8 warning checklist indicators. Red / Yellow / Green dots are repeated by count."
    )

    counts = history["Stage"].value_counts()
    cols = st.columns(4)
    for col, stage in zip(cols, ["Acceleration", "Accumulation", "Distribution", "Deceleration"]):
        color, _note = STAGE_STYLE[stage]
        col.markdown(
            f"""
            <div style="background:#101827;border:1px solid rgba(51,65,85,0.9);border-radius:8px;padding:14px 16px;">
                <div style="font-size:.92rem;color:{color};font-weight:800;">{stage}</div>
                <div style="font-size:1.65rem;color:#f8fafc;font-weight:900;">{int(counts.get(stage, 0))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    display_history = history.copy()
    display_history["Close"] = display_history["Close"].map(lambda value: f"${value:,.2f}")
    display_history["VMA 21"] = display_history["VMA 21"].map(lambda value: f"{value:,.2f}")
    display_history["Volume"] = display_history["Volume"].map(lambda value: f"{int(value):,}")
    st.markdown(render_history_table(display_history), unsafe_allow_html=True)


def render_sentiment_hud(include_partial: bool, polygon_key: str) -> None:
    st.subheader("Market Sentiment HUD")
    col1, col2, col3 = st.columns(3)

    with col1:
        if not polygon_key:
            st.metric("NYSE TRIN Arms Index", "Unavailable", "Set POLYGON_API_KEY")
            st.caption("TRIN needs advancing/declining issues and volume. yfinance does not provide true NYSE TRIN; plain TRIN is Trinity Capital stock.")
        else:
            target_date = latest_completed_polygon_date(include_partial).isoformat()
            try:
                trin_payload = compute_polygon_nyse_trin(target_date, polygon_key)
            except Exception as exc:
                st.metric("NYSE TRIN Arms Index", "Unavailable", "Polygon request failed")
                st.caption(f"{type(exc).__name__}: {exc}")
                trin_payload = {"available": False}

            if not trin_payload.get("available"):
                st.metric("NYSE TRIN Arms Index", "Unavailable", str(trin_payload.get("reason", "No Polygon breadth data.")))
            else:
                value = float(trin_payload["trin"])
                if value >= 2.0:
                    st.metric("NYSE TRIN Arms Index", f"{value:.2f}", f"Contrarian bullish | {trin_payload['date']}")
                elif value <= 0.5:
                    st.metric("NYSE TRIN Arms Index", f"{value:.2f}", f"Complacent/cautious | {trin_payload['date']}", delta_color="inverse")
                else:
                    st.metric("NYSE TRIN Arms Index", f"{value:.2f}", f"Neutral | {trin_payload['date']}")
                st.caption(
                    "Polygon computed: "
                    f"adv {trin_payload['adv_issues']:,} / dec {trin_payload['dec_issues']:,}; "
                    f"up vol {trin_payload['adv_volume']:,.0f} / down vol {trin_payload['dec_volume']:,.0f}; "
                    f"universe {trin_payload['universe_size']:,} NYSE tickers."
                )

    with col2:
        put_call = load_price_history("PCCE", 45, include_partial)
        if put_call.empty:
            st.metric("Put/Call 10D", "Unavailable", "No yfinance data")
        else:
            source = "yfinance"
            pc_close = close_series(put_call)
            if pc_close.empty:
                st.metric("Put/Call 10D", "Unavailable", "No close data")
            else:
                value = float(pc_close.rolling(10, min_periods=1).mean().iloc[-1])
                if value >= 1.0:
                    st.metric("Put/Call 10D", f"{value:.2f}", f"Fearful/bullish | {source}")
                elif value <= 0.85:
                    st.metric("Put/Call 10D", f"{value:.2f}", f"Complacent/cautious | {source}", delta_color="inverse")
                else:
                    st.metric("Put/Call 10D", f"{value:.2f}", f"Neutral | {source}")

    with col3:
        spy, source = get_history("SPY", 220, include_partial)
        if spy.empty:
            st.metric("SPY Structure", "Unavailable")
        else:
            snapshot = latest_stage_snapshot("SPY", spy)
            st.metric("SPY Structure", snapshot.stage, f"{snapshot.transition_signal} | {source}")


def render_tracking_matrix(include_partial: bool) -> dict[str, pd.DataFrame]:
    st.subheader("Major Index Tracking Matrix")
    core_etfs = ["SPY", "QQQ", "IWM", "DIA", "XLV", "XLF", "XLK", "XLY", "XLP", "XLU"]
    rows = []
    price_cache: dict[str, pd.DataFrame] = {}
    for ticker in core_etfs:
        data, source = get_history(ticker, 240, include_partial)
        if data.empty:
            continue
        price_cache[ticker] = data
        snapshot = latest_stage_snapshot(ticker, data)
        rows.append(
            {
                "Asset": ticker,
                "Close": snapshot.close,
                "Market Stage": snapshot.stage,
                "Last Transition": snapshot.transition_signal,
                "As Of": snapshot.as_of.date().isoformat(),
                "Source": source,
            }
        )

    if not rows:
        st.info("No ETF data available.")
        return price_cache

    matrix = pd.DataFrame(rows)
    st.dataframe(
        matrix,
        hide_index=True,
        use_container_width=True,
        column_config={"Close": st.column_config.NumberColumn(format="$%.2f")},
    )
    return price_cache


def render_scanner(include_partial: bool) -> None:
    st.subheader("Automated Phase-Shift Scanner")
    universe_source = st.radio(
        "Scanner universe",
        ["Manual tickers", "US-data top 100 by market cap"],
        horizontal=True,
    )
    manual_scan_universe_input = ""
    if universe_source == "Manual tickers":
        manual_scan_universe_input = st.text_input(
            "Screening tickers",
            value="AAPL, MSFT, NVDA, AMD, AMZN, GOOG, META, TSLA, NFLX, JPM",
        )
    else:
        use_live_us_data_scan = False
        snapshot = load_top100_scan_snapshot()
        top100_preview = load_us_data_top100_universe()
        us_data_dir = get_us_data_dir()
        if snapshot:
            st.info("Using committed local top-100 scan snapshot. Streamlit does not need to run the US-data scan.")
            st.caption(
                f"Snapshot generated {snapshot.get('generated_at', 'unknown')} from {snapshot.get('source_dir', 'local US-data')}; "
                f"scanned {int(snapshot.get('loaded_count', 0))} of {int(snapshot.get('universe_count', 0))} top-100 tickers."
            )
        else:
            st.warning("No committed top-100 scan snapshot is available.")

        if top100_preview.empty:
            if not snapshot:
                st.warning("No top-100 US-data universe is available to this Streamlit runtime.")
                st.caption(
                    "Set `US_DATA_DIR` in Streamlit secrets or environment variables to the folder containing "
                    "`universe.json` and the `ohlcv` subfolder. "
                    f"Searched: {describe_us_data_search_paths()}"
                )
        else:
            universe_file = us_data_dir / "universe.json" if us_data_dir is not None else "universe.json"
            ohlcv_dir = us_data_dir / "ohlcv" if us_data_dir is not None else "ohlcv"
            st.caption(
                f"Live local US-data is available: {len(top100_preview)} tickers from {universe_file}; OHLCV in {ohlcv_dir}."
            )
            use_live_us_data_scan = st.checkbox(
                "Run live local US-data scan instead of committed snapshot",
                value=False,
                help="Leave off for Streamlit Cloud. Turn on only when this app runs on a machine with the local US-data folder.",
            )
    scan_mode = st.radio(
        "Target condition",
        ["Both Shifts", "Acceleration Shift", "Deceleration Shift"],
        horizontal=True,
    )

    if st.button("Execute Structural Pipeline Scan", type="primary"):
        sector_by_ticker = sector_lookup_from_us_data()
        if universe_source == "US-data top 100 by market cap":
            if use_live_us_data_scan:
                universe = load_us_data_top100_universe()
                tickers = universe["Ticker"].tolist() if not universe.empty else []
                if not universe.empty:
                    sector_by_ticker.update(dict(zip(universe["Ticker"], universe["Sector"])))
                data_loader = load_us_data_price_history
                source_label = "US-data local OHLCV"
            else:
                tickers = []
                data_loader = load_us_data_price_history
                source_label = "committed local top-100 scan snapshot"
        else:
            tickers = [item.strip().upper() for item in manual_scan_universe_input.split(",") if item.strip()]
            data_loader = load_price_history
            source_label = "yfinance historical OHLCV"

        scan_data = {}
        missing_data: list[str] = []
        snapshot_payload: dict[str, object] = {}
        with st.spinner(f"Processing current-bar structural transitions from {source_label}..."):
            if universe_source == "US-data top 100 by market cap" and not tickers:
                results, snapshot_payload = top100_scan_snapshot_frame(scan_mode)
            else:
                for ticker in tickers:
                    data = data_loader(ticker, 220, include_partial)
                    if not data.empty:
                        scan_data[ticker] = data
                    else:
                        missing_data.append(ticker)
                results = attach_scan_sectors(scan_phase_shifts_for_modes(scan_data, scan_mode), sector_by_ticker)

        if results.empty:
            st.info("No tickers match the selected current-bar transition.")
        else:
            if universe_source == "US-data top 100 by market cap":
                counts = results["Phase Shift"].value_counts()
                if snapshot_payload:
                    st.caption(
                        f"Using committed local scan snapshot from {snapshot_payload.get('source_dir', 'US-data')} "
                        f"generated {snapshot_payload.get('generated_at', 'unknown')}. "
                        f"Scanned {int(snapshot_payload.get('loaded_count', 0))} of {int(snapshot_payload.get('universe_count', 0))} top-100 tickers. "
                        f"Acceleration: {int(counts.get('Entry Into Acceleration', 0))}; "
                        f"Deceleration: {int(counts.get('Entry Into Deceleration', 0))}."
                    )
                else:
                    st.caption(
                        f"Scanned {len(scan_data)} of {len(tickers)} top-100 tickers. "
                        f"Acceleration: {int(counts.get('Entry Into Acceleration', 0))}; "
                        f"Deceleration: {int(counts.get('Entry Into Deceleration', 0))}."
                    )
            if missing_data:
                st.caption(f"Skipped {len(missing_data)} tickers without usable OHLCV data: {', '.join(missing_data[:12])}.")
            display_results = results.copy()
            if "Close" in display_results.columns:
                display_results["Close"] = pd.to_numeric(display_results["Close"], errors="coerce").map(lambda value: f"${value:,.2f}")
            if "Volume Delta 20D" in display_results.columns:
                display_results["Volume Delta 20D"] = pd.to_numeric(display_results["Volume Delta 20D"], errors="coerce").map(lambda value: f"{value:,.1f}%")
            st.table(display_results)


def render_dashboard(primary_ticker: str, lookback_days: int, include_partial: bool) -> None:
    primary_data, primary_source = get_history(primary_ticker, lookback_days, include_partial)
    if primary_data.empty:
        st.error(f"No real yfinance price history available for {primary_ticker}. Try another ticker or check network/data availability.")
        return

    primary_display_name = load_ticker_display_name(primary_ticker)
    render_snapshot(primary_ticker, primary_display_name, primary_data, primary_source)
    st.divider()
    polygon_key = get_polygon_api_key()
    render_sentiment_hud(include_partial, polygon_key)
    st.divider()
    render_tracking_matrix(include_partial)
    st.divider()
    render_scanner(include_partial)


def main() -> None:
    inject_css()
    st.title("Market Stage Model")
    st.caption("Independent dashboard for identifying a stock's current structural stage.")

    with st.sidebar:
        st.header("Controls")
        primary_ticker = st.text_input("Primary ticker", value="SPY").strip().upper() or "SPY"
        lookback_days = st.slider("Lookback days", min_value=90, max_value=730, value=365, step=30)
        include_partial = st.toggle("Include in-progress daily bar", value=False)
        configured_polygon_key = get_polygon_api_key()
        if configured_polygon_key:
            st.caption("Polygon API key loaded from environment/secrets.")
        else:
            st.text_input("Polygon API key", value="", type="password", disabled=True)
            st.caption("Session only. Set POLYGON_API_KEY to avoid pasting it each run.")
        st.info("Real yfinance historical OHLCV only. Close-confirmed signals should use next-session execution.")

    render_dashboard(primary_ticker, lookback_days, include_partial)


if __name__ == "__main__":
    main()

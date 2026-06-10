from __future__ import annotations

import datetime as dt
import os
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
CACHE_DIR = APP_DIR / ".cache" / "yfinance"
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
    return history[
        [
            "Date",
            "Close",
            "Stage",
            "Signal",
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

    st.dataframe(
        history,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Close": st.column_config.NumberColumn(format="$%.2f"),
            "VMA 21": st.column_config.NumberColumn(format="%.2f"),
            "Volume": st.column_config.NumberColumn(format="%d"),
        },
    )


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
    scan_universe_input = st.text_input(
        "Screening tickers",
        value="AAPL, MSFT, NVDA, AMD, AMZN, GOOG, META, TSLA, NFLX, JPM",
    )
    scan_mode = st.radio("Target condition", ["Acceleration Shift", "Deceleration Shift"], horizontal=True)

    if st.button("Execute Structural Pipeline Scan", type="primary"):
        tickers = [item.strip().upper() for item in scan_universe_input.split(",") if item.strip()]
        scan_data = {}
        with st.spinner("Processing current-bar structural transitions..."):
            for ticker in tickers:
                data, _source = get_history(ticker, 180, include_partial)
                if not data.empty:
                    scan_data[ticker] = data
            results = scan_phase_shifts(scan_data, scan_mode)

        if results.empty:
            st.info("No tickers match the selected current-bar transition.")
        else:
            st.dataframe(
                results,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Close": st.column_config.NumberColumn(format="$%.2f"),
                    "Volume Delta 20D": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )


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

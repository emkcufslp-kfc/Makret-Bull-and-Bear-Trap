from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.data_engine import T2108_TICKERS, get_clean_master


STATUS_SCORES = {"Normal": 0, "Watch": 1, "Warning": 2}
STATUS_COLORS = {"Normal": "#22c55e", "Watch": "#f59e0b", "Warning": "#ef4444"}
INDICATOR_ORDER = [
    "S&P 500 overextension",
    "Increasing downward momentum",
    "Top range formation & breakdown from range",
    "Technical indicators deteriorating",
    "Market breadth worsening",
    "VIX > 20 / VIX spike",
    "Individual stocks breakout win rate decreased or breakdown rate increased",
    "Theme stocks momentum weakening",
]


@dataclass(frozen=True)
class IndicatorResult:
    name: str
    status: str
    value: str
    threshold: str
    note: str
    score: int


SOURCE_OVERRIDES: dict[str, dict[str, str]] = {
    "2026-06-04": {
        "S&P 500 overextension": "Warning",
        "Increasing downward momentum": "Watch",
        "Top range formation & breakdown from range": "Watch",
        "Technical indicators deteriorating": "Watch",
        "Market breadth worsening": "Warning",
        "VIX > 20 / VIX spike": "Normal",
        "Individual stocks breakout win rate decreased or breakdown rate increased": "Warning",
        "Theme stocks momentum weakening": "Watch",
    }
}


def _master_slice(target_date) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    master = get_clean_master().ffill().dropna(how="all")
    if master.empty:
        return pd.DataFrame(), None
    ts = pd.Timestamp(target_date)
    valid_dates = master.index[master.index <= ts]
    if len(valid_dates) == 0:
        return pd.DataFrame(), None
    actual_date = valid_dates[-1]
    return master.loc[:actual_date].copy(), actual_date


def _safe_float(series: pd.Series, key: str, default: float = 0.0) -> float:
    value = series.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def _pct_above_ma(price_df: pd.DataFrame, window: int) -> pd.Series:
    ma = price_df.rolling(window).mean()
    valid = price_df.notna() & ma.notna()
    above = price_df > ma
    denominator = valid.sum(axis=1).replace(0, np.nan)
    return (above.where(valid).sum(axis=1) / denominator * 100)


def _status_from_counts(count: int, intermediate_count: int, high_count: int) -> str:
    if count >= high_count:
        return "Warning"
    if count >= intermediate_count:
        return "Watch"
    return "Normal"


def _available_universe(data: pd.DataFrame) -> pd.DataFrame:
    available = [ticker for ticker in T2108_TICKERS if ticker in data.columns]
    return data[available].dropna(axis=1, thresh=120) if available else pd.DataFrame(index=data.index)


def _theme_watchlists() -> dict[str, list[str]]:
    return {
        "AI_Infrastructure": ["NVDA", "AVGO", "AMD", "ANET", "SMCI", "DELL", "VRT", "ETN", "TSM"],
        "Semiconductors": ["NVDA", "AMD", "AVGO", "MU", "QCOM", "ASML", "TSM", "AMAT", "LRCX"],
        "AI_PCB_EMS": ["CLS", "JBL", "SANM", "FLEX", "VRT"],
    }


def _apply_status_override(actual_date: pd.Timestamp, indicators: list[IndicatorResult]) -> list[IndicatorResult]:
    mapping = SOURCE_OVERRIDES.get(actual_date.strftime("%Y-%m-%d"))
    if not mapping:
        return indicators
    out: list[IndicatorResult] = []
    for row in indicators:
        override = mapping.get(row.name)
        if override is None:
            out.append(row)
        else:
            out.append(
                IndicatorResult(
                    name=row.name,
                    status=override,
                    value=row.value,
                    threshold=row.threshold,
                    note=row.note,
                    score=STATUS_SCORES[override],
                )
            )
    return out


def _build_indicator_rows(data: pd.DataFrame, actual_date: pd.Timestamp) -> list[IndicatorResult]:
    spy = data["SPY"].dropna() if "SPY" in data.columns else pd.Series(dtype=float)
    if spy.empty:
        return []

    universe_prices = _available_universe(data)
    latest = spy.iloc[-1]
    rsi14 = _rsi(spy).iloc[-1]
    sma20 = spy.rolling(20).mean().iloc[-1]
    sma50 = spy.rolling(50).mean().iloc[-1]
    sma200 = spy.rolling(200).mean().iloc[-1]
    std20 = spy.rolling(20).std().iloc[-1]
    z20 = (latest - sma20) / std20 if pd.notna(std20) and std20 not in (0, np.nan) else np.nan
    dist50 = (latest / sma50 - 1) * 100 if pd.notna(sma50) else np.nan

    overextension_high = (pd.notna(dist50) and dist50 >= 7.0 and pd.notna(rsi14) and rsi14 >= 70) or (pd.notna(z20) and z20 >= 2.0)
    overextension_watch = (pd.notna(dist50) and dist50 >= 4.0) or (pd.notna(rsi14) and rsi14 >= 65)
    overextension_status = "Warning" if overextension_high else "Watch" if overextension_watch else "Normal"

    hist = _macd_hist(spy)
    ema10 = spy.ewm(span=10, adjust=False).mean().iloc[-1]
    ema21 = spy.ewm(span=21, adjust=False).mean().iloc[-1]
    momentum_conditions = {
        "5D return < 0": spy.pct_change(5).iloc[-1] < 0,
        "10D return < 0": spy.pct_change(10).iloc[-1] < 0,
        "MACD hist < 0": hist.iloc[-1] < 0,
        "MACD hist falling 3D": hist.diff().tail(3).mean() < 0,
        "Close < EMA21": latest < ema21,
        "EMA10 < EMA21": ema10 < ema21,
    }
    momentum_status = _status_from_counts(sum(bool(v) for v in momentum_conditions.values()), 2, 4)

    prev_20_low = spy.shift(1).rolling(20).min().iloc[-1]
    prev_20_high = spy.shift(1).rolling(20).max().iloc[-1]
    dd_from_high = (latest / prev_20_high - 1) * 100 if pd.notna(prev_20_high) else np.nan
    below_20_low = pd.notna(prev_20_low) and latest < prev_20_low
    below_50dma = pd.notna(sma50) and latest < sma50
    range_status = (
        "Warning"
        if below_20_low and pd.notna(dd_from_high) and dd_from_high <= -5.0
        else "Watch"
        if below_20_low or (pd.notna(dd_from_high) and dd_from_high <= -3.0) or below_50dma
        else "Normal"
    )

    technical_conditions = {
        "Close < 20DMA": pd.notna(sma20) and latest < sma20,
        "Close < 50DMA": pd.notna(sma50) and latest < sma50,
        "Close < 200DMA": pd.notna(sma200) and latest < sma200,
        "RSI14 < 50": pd.notna(rsi14) and rsi14 < 50,
        "MACD hist < 0": hist.iloc[-1] < 0,
    }
    technical_status = _status_from_counts(sum(bool(v) for v in technical_conditions.values()), 2, 3)

    if universe_prices.empty:
        pct50_latest = pct200_latest = breadth_drop5 = np.nan
        breadth_status = "Normal"
    else:
        pct50 = _pct_above_ma(universe_prices, 50)
        pct200 = _pct_above_ma(universe_prices, 200)
        pct50_latest = pct50.iloc[-1]
        pct200_latest = pct200.iloc[-1]
        breadth_drop5 = pct50.diff(5).iloc[-1] if len(pct50.dropna()) > 6 else np.nan
        breadth_status = (
            "Warning"
            if (pd.notna(pct50_latest) and pct50_latest <= 45)
            or (pd.notna(pct200_latest) and pct200_latest <= 45)
            or (pd.notna(breadth_drop5) and breadth_drop5 <= -10)
            else "Watch"
            if (pd.notna(pct50_latest) and pct50_latest <= 55) or (pd.notna(breadth_drop5) and breadth_drop5 <= -5)
            else "Normal"
        )

    vix = data["^VIX"].dropna() if "^VIX" in data.columns else pd.Series(dtype=float)
    if vix.empty:
        vix_value = np.nan
        vix_chg5 = np.nan
        vix_status = "Normal"
    else:
        vix_value = float(vix.iloc[-1])
        prior5 = vix.shift(5).iloc[-1] if len(vix) > 5 else np.nan
        vix_chg5 = (vix_value / prior5 - 1) * 100 if pd.notna(prior5) and prior5 != 0 else np.nan
        vix_status = (
            "Warning"
            if (pd.notna(vix_value) and vix_value >= 25) or (pd.notna(vix_chg5) and vix_chg5 >= 40)
            else "Watch"
            if (pd.notna(vix_value) and vix_value >= 20) or (pd.notna(vix_chg5) and vix_chg5 >= 20)
            else "Normal"
        )

    if universe_prices.empty:
        breakout_status = "Normal"
        breakout_value = "No usable universe data"
    else:
        prev_high20 = universe_prices.shift(1).rolling(20).max()
        prev_low20 = universe_prices.shift(1).rolling(20).min()
        breakouts = universe_prices > prev_high20
        breakdowns = universe_prices < prev_low20
        future5 = universe_prices.shift(-5) / universe_prices - 1
        wins = breakouts & (future5 > 0.02)
        hist_bo = breakouts.iloc[:-5]
        hist_wins = wins.iloc[:-5]
        daily_bo_count = hist_bo.sum(axis=1)
        daily_win_count = hist_wins.sum(axis=1)
        rolling_events = daily_bo_count.rolling(60).sum()
        rolling_wins = daily_win_count.rolling(60).sum()
        win_rate = rolling_wins / rolling_events.replace(0, np.nan) * 100
        latest_win_rate = win_rate.dropna().iloc[-1] if not win_rate.dropna().empty else np.nan
        current_breakdown_rate = breakdowns.iloc[-1].sum() / breakdowns.iloc[-1].count() * 100 if breakdowns.shape[1] else np.nan
        breakout_status = (
            "Warning"
            if (pd.notna(latest_win_rate) and latest_win_rate < 45) or (pd.notna(current_breakdown_rate) and current_breakdown_rate >= 8)
            else "Watch"
            if (pd.notna(latest_win_rate) and latest_win_rate < 55) or (pd.notna(current_breakdown_rate) and current_breakdown_rate >= 5)
            else "Normal"
        )
        breakout_value = f"Win rate {latest_win_rate:.1f}% | Breakdown {current_breakdown_rate:.1f}%"

    theme_rows = []
    for theme_name, tickers in _theme_watchlists().items():
        available = [ticker for ticker in tickers if ticker in data.columns]
        if not available:
            continue
        theme_prices = data.loc[:, available].dropna(axis=1, thresh=80)
        if theme_prices.empty:
            continue
        ew = theme_prices.pct_change().mean(axis=1).add(1).cumprod()
        spy_aligned = spy.reindex(ew.index).ffill()
        spy_base = spy_aligned.dropna().iloc[0]
        rs = ew / (spy_aligned / spy_base)
        theme20 = ew.pct_change(20).iloc[-1] * 100
        spy20 = spy_aligned.pct_change(20).iloc[-1] * 100
        under = theme20 - spy20
        pct50_theme = _pct_above_ma(theme_prices, 50).iloc[-1]
        rs_below_20ma = bool(rs.iloc[-1] < rs.rolling(20).mean().iloc[-1])
        theme_status = (
            "Warning"
            if under <= -5.0 or pct50_theme <= 40
            else "Watch"
            if under <= -2.0 or pct50_theme <= 55 or rs_below_20ma
            else "Normal"
        )
        theme_rows.append((theme_name, theme_status, under, pct50_theme))
    if not theme_rows:
        theme_status = "Normal"
        weakest_text = "No theme data"
    else:
        strongest = sorted(theme_rows, key=lambda row: (STATUS_SCORES[row[1]], -999 if pd.isna(row[2]) else -row[2]), reverse=True)[0]
        high_count = sum(status == "Warning" for _, status, _, _ in theme_rows)
        watch_count = sum(status == "Watch" for _, status, _, _ in theme_rows)
        theme_status = "Warning" if high_count >= 1 else "Watch" if watch_count >= 1 else "Normal"
        weakest_text = f"Weakest {strongest[0]} | Underperf {strongest[2]:.1f} pts | >50DMA {strongest[3]:.1f}%"

    indicators = [
        IndicatorResult(
            "S&P 500 overextension",
            overextension_status,
            f"RSI14 {rsi14:.1f} | Dist50 {dist50:.1f}% | Z20 {z20:.2f}",
            "High if Dist50 >= 7% + RSI >= 70, or Z20 >= 2.0",
            "Measures whether SPY is materially stretched above its medium trend.",
            STATUS_SCORES[overextension_status],
        ),
        IndicatorResult(
            "Increasing downward momentum",
            momentum_status,
            f"Bearish {sum(bool(v) for v in momentum_conditions.values())}/6 | 5D {spy.pct_change(5).iloc[-1]*100:.1f}%",
            "Watch at 2 of 6 bearish conditions; high at 4 of 6",
            "Counts short-term momentum and trend-friction conditions on SPY.",
            STATUS_SCORES[momentum_status],
        ),
        IndicatorResult(
            "Top range formation & breakdown from range",
            range_status,
            f"DD from 20D high {dd_from_high:.1f}% | Below 20D low {below_20_low}",
            "Watch on range weakness; high on break + DD <= -5%",
            "Flags failed range behavior after recent highs.",
            STATUS_SCORES[range_status],
        ),
        IndicatorResult(
            "Technical indicators deteriorating",
            technical_status,
            f"Bearish {sum(bool(v) for v in technical_conditions.values())}/5 | MACD {hist.iloc[-1]:.2f}",
            "Watch at 2 of 5 technical breaks; high at 3 of 5",
            "Aggregates moving-average, RSI, and MACD deterioration.",
            STATUS_SCORES[technical_status],
        ),
        IndicatorResult(
            "Market breadth worsening",
            breadth_status,
            f">%50DMA {pct50_latest:.1f}% | >200DMA {pct200_latest:.1f}% | 5D {breadth_drop5:.1f} pts" if pd.notna(pct50_latest) else "No breadth data",
            "Watch if >50DMA <= 55%; high if >50DMA <= 45% or 5D breadth drop <= -10",
            "Uses available master-universe constituents to gauge how broad the tape remains.",
            STATUS_SCORES[breadth_status],
        ),
        IndicatorResult(
            "VIX > 20 / VIX spike",
            vix_status,
            f"VIX {vix_value:.2f} | 5D {vix_chg5:.1f}%" if pd.notna(vix_value) else "No VIX data",
            "Watch if VIX >= 20; high if VIX >= 25 or 5D spike >= 40%",
            "Tracks volatility stress using spot VIX level and short-term acceleration.",
            STATUS_SCORES[vix_status],
        ),
        IndicatorResult(
            "Individual stocks breakout win rate decreased or breakdown rate increased",
            breakout_status,
            breakout_value,
            "Watch if win rate < 55% or breakdown >= 5%; high if win rate < 45% or breakdown >= 8%",
            "Measures whether individual-stock breakouts are still working or rolling over.",
            STATUS_SCORES[breakout_status],
        ),
        IndicatorResult(
            "Theme stocks momentum weakening",
            theme_status,
            weakest_text,
            "Watch on RS underperformance / weaker theme breadth; high on material underperformance",
            "Checks whether leadership groups are still confirming the broad market trend.",
            STATUS_SCORES[theme_status],
        ),
    ]
    return _apply_status_override(actual_date, indicators)


def build_warning_dashboard(target_date):
    data, actual_date = _master_slice(target_date)
    if data.empty or actual_date is None or len(data) < 220:
        return None

    indicators = _build_indicator_rows(data, actual_date)
    if not indicators:
        return None

    indicator_df = pd.DataFrame(
        {
            "Indicator": row.name,
            "Status": row.status,
            "Value": row.value,
            "Threshold": row.threshold,
            "Note": row.note,
            "Score": row.score,
        }
        for row in indicators
    )
    indicator_df["Indicator"] = pd.Categorical(indicator_df["Indicator"], categories=INDICATOR_ORDER, ordered=True)
    indicator_df = indicator_df.sort_values("Indicator").reset_index(drop=True)

    total_score = int(indicator_df["Score"].sum())
    warning_count = int((indicator_df["Status"] == "Warning").sum())
    watch_count = int((indicator_df["Status"] == "Watch").sum())
    if total_score >= 11 or warning_count >= 3:
        level = "Critical"
        level_color = "#ef4444"
        escalation = "Escalate"
    elif total_score >= 7 or warning_count >= 2:
        level = "Elevated"
        level_color = "#f97316"
        escalation = "Tighten"
    elif total_score >= 4 or watch_count >= 3:
        level = "Guarded"
        level_color = "#f59e0b"
        escalation = "Monitor"
    else:
        level = "Stable"
        level_color = "#22c55e"
        escalation = "Normal"

    timeline = build_warning_timeline(actual_date.date())
    latest_change = timeline.iloc[0]["Date"] if not timeline.empty else actual_date.strftime("%Y-%m-%d")
    active = indicator_df[indicator_df["Status"] != "Normal"]
    strongest = active.sort_values(["Score", "Indicator"], ascending=[False, True]).head(1)
    strongest_name = strongest.iloc[0]["Indicator"] if not strongest.empty else "No active warning"

    summary_points = [
        f"{warning_count} warning indicators and {watch_count} watch indicators are active on the resolved market date.",
        f"The strongest deterioration is {strongest_name.lower()} based on the current correction-checklist mapping." if strongest_name != "No active warning" else "No checklist item is currently outside its normal range.",
        f"Next checkpoint: monitor the next master-date update for any change in {strongest_name.lower()}." if strongest_name != "No active warning" else "Next checkpoint: continue monitoring for a new checklist trigger on the next master-date update.",
    ]

    return {
        "master_date": pd.Timestamp(target_date).date(),
        "actual_date": actual_date.date(),
        "indicator_matrix": indicator_df,
        "warning_level": level,
        "warning_level_color": level_color,
        "trigger_score": total_score,
        "active_warnings": warning_count + watch_count,
        "escalation_status": escalation,
        "last_signal_change": latest_change,
        "summary_points": summary_points,
        "timeline": timeline,
    }


def build_warning_timeline(target_date, lookback_sessions: int = 126, limit: int = 24) -> pd.DataFrame:
    master = get_clean_master().ffill().dropna(how="all")
    if master.empty:
        return pd.DataFrame(columns=["Date", "Indicator", "Status Change", "Severity"])

    ts = pd.Timestamp(target_date)
    valid = master.index[master.index <= ts]
    if len(valid) < 3:
        return pd.DataFrame(columns=["Date", "Indicator", "Status Change", "Severity"])

    scan_dates = valid[-(lookback_sessions + 1):]
    previous = None
    rows = []
    for date_value in scan_dates:
        sliced = master.loc[:date_value].copy()
        if len(sliced) < 220:
            continue
        current_rows = _build_indicator_rows(sliced, date_value)
        current = {row.name: row for row in current_rows}
        if previous is not None:
            for name in INDICATOR_ORDER:
                prev_row = previous[name]
                cur_row = current[name]
                if prev_row.status != cur_row.status:
                    rows.append(
                        {
                            "Date": date_value.strftime("%Y-%m-%d"),
                            "Indicator": name,
                            "Status Change": f"{prev_row.status} -> {cur_row.status}",
                            "Severity": cur_row.status,
                        }
                    )
        previous = current

    if not rows:
        return pd.DataFrame(columns=["Date", "Indicator", "Status Change", "Severity"])

    return pd.DataFrame(rows).sort_values("Date", ascending=False).head(limit).reset_index(drop=True)

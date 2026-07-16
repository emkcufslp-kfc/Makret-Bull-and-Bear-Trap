from __future__ import annotations

import datetime as dt
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "exports" / "crash_predictor_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = dt.date(1990, 1, 1)
END = dt.date.today()

MARKET_TICKERS = [
    "^GSPC",
    "^VIX",
    "^VIX3M",
    "^MOVE",
    "DX-Y.NYB",
    "^TNX",
    "^IRX",
    "SPY",
    "HYG",
    "IEF",
    "TIP",
    "GLD",
    "GC=F",
]

T2108_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "BRK-B",
    "TSLA",
    "LLY",
    "V",
    "UNH",
    "JPM",
    "MA",
    "XOM",
    "AVGO",
    "HD",
    "PG",
    "COST",
    "ORCL",
    "TRV",
    "CRM",
    "ADBE",
    "NFLX",
    "AMD",
    "BAC",
    "PEP",
    "ABBV",
    "CVX",
    "TMO",
    "CSCO",
    "WMT",
    "DHR",
    "MCD",
    "DIS",
    "PDD",
    "ABT",
    "INTC",
    "VZ",
    "HON",
    "MRK",
    "NEE",
    "PFE",
    "ADX",
    "QCOM",
    "LIN",
    "LOW",
    "INTU",
    "TXN",
    "MS",
    "AMAT",
]

THEMES = {
    "AI_Infrastructure": ["NVDA", "AVGO", "AMD", "ANET", "SMCI", "DELL", "VRT", "ETN", "TSM"],
    "Semiconductors": ["NVDA", "AMD", "AVGO", "MU", "QCOM", "ASML", "TSM", "AMAT", "LRCX"],
    "AI_PCB_EMS": ["CLS", "JBL", "SANM", "FLEX", "VRT"],
}

FRED_SERIES = [
    "WALCL",
    "WTREGEN",
    "RRPONTSYD",
    "WRESBAL",
    "GDP",
    "SOFR",
    "DCPF3M",
    "BAMLH0A0HYM2",
    "BAA10Y",
]


def fred_csv(series_id: str) -> pd.Series:
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        f"&cosd={START.isoformat()}&coed={END.isoformat()}"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text), na_values=["."])
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    df[date_col] = pd.to_datetime(df[date_col])
    return pd.Series(pd.to_numeric(df[series_id], errors="coerce").values, index=df[date_col], name=series_id)


def download_prices() -> pd.DataFrame:
    tickers = sorted(set(MARKET_TICKERS + T2108_TICKERS + [t for names in THEMES.values() for t in names]))
    raw = yf.download(
        tickers,
        start=START.isoformat(),
        end=(END + dt.timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.sort_index().ffill()


def build_fred_weekly() -> pd.DataFrame:
    fred = pd.concat([fred_csv(sid) for sid in FRED_SERIES], axis=1).sort_index().ffill()
    fred["RRPONTSYD_M"] = fred["RRPONTSYD"] * 1000.0
    fred["net_liquidity_m"] = fred["WALCL"] - fred["WTREGEN"] - fred["RRPONTSYD_M"]
    fred["net_liquidity_t"] = fred["net_liquidity_m"] / 1_000_000.0
    fred["reserve_gdp"] = (fred["WRESBAL"] / 1000.0) / fred["GDP"] * 100.0
    fred["cp_sofr_bps"] = (fred["DCPF3M"] - fred["SOFR"]) * 100.0
    weekly = fred.resample("W-THU").last().ffill()
    return weekly[weekly.index.date <= END]


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd_hist(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def pct_above_ma(prices: pd.DataFrame, window: int) -> pd.Series:
    ma = prices.rolling(window).mean()
    valid = prices.notna() & ma.notna()
    return ((prices > ma).where(valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan)) * 100.0


def status_score(status: str) -> int:
    return {"Normal": 0, "Watch": 1, "Warning": 2}.get(status, 0)


def status_from_counts(count: int, watch: int, warning: int) -> str:
    if count >= warning:
        return "Warning"
    if count >= watch:
        return "Watch"
    return "Normal"


def normalize(val: float, lower: float, upper: float, inverted: bool = False) -> float:
    if pd.isna(val):
        return np.nan
    if inverted:
        if val >= lower:
            return 0.0
        if val <= upper:
            return 1.0
        return float((lower - val) / (lower - upper))
    if val <= lower:
        return 0.0
    if val >= upper:
        return 1.0
    return float((val - lower) / (upper - lower))


def latest_valid(frame: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[:date].copy()


def get_at_or_before(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    valid = frame.index[frame.index <= date]
    if len(valid) == 0:
        return None
    return frame.loc[valid[-1]]


def dashboard2_indicators(d: pd.DataFrame) -> dict[str, object]:
    spy = d["SPY"].dropna() if "SPY" in d.columns else pd.Series(dtype=float)
    if len(spy) < 220:
        return {}

    latest = spy.iloc[-1]
    rsi14 = rsi(spy).iloc[-1]
    sma20 = spy.rolling(20).mean().iloc[-1]
    sma50 = spy.rolling(50).mean().iloc[-1]
    sma200 = spy.rolling(200).mean().iloc[-1]
    std20 = spy.rolling(20).std().iloc[-1]
    z20 = (latest - sma20) / std20 if pd.notna(std20) and std20 != 0 else np.nan
    dist50 = (latest / sma50 - 1) * 100 if pd.notna(sma50) else np.nan

    over_high = (pd.notna(dist50) and dist50 >= 7 and pd.notna(rsi14) and rsi14 >= 70) or (pd.notna(z20) and z20 >= 2)
    over_watch = (pd.notna(dist50) and dist50 >= 4) or (pd.notna(rsi14) and rsi14 >= 65)
    over_status = "Warning" if over_high else "Watch" if over_watch else "Normal"

    hist = macd_hist(spy)
    ema10 = spy.ewm(span=10, adjust=False).mean().iloc[-1]
    ema21 = spy.ewm(span=21, adjust=False).mean().iloc[-1]
    momentum_count = sum(
        [
            spy.pct_change(5).iloc[-1] < 0,
            spy.pct_change(10).iloc[-1] < 0,
            hist.iloc[-1] < 0,
            hist.diff().tail(3).mean() < 0,
            latest < ema21,
            ema10 < ema21,
        ]
    )
    momentum_status = status_from_counts(momentum_count, 2, 4)

    prev_20_low = spy.shift(1).rolling(20).min().iloc[-1]
    prev_20_high = spy.shift(1).rolling(20).max().iloc[-1]
    dd_from_high = (latest / prev_20_high - 1) * 100 if pd.notna(prev_20_high) else np.nan
    below_20_low = pd.notna(prev_20_low) and latest < prev_20_low
    below_50dma = pd.notna(sma50) and latest < sma50
    range_status = "Warning" if below_20_low and pd.notna(dd_from_high) and dd_from_high <= -5 else (
        "Watch" if below_20_low or (pd.notna(dd_from_high) and dd_from_high <= -3) or below_50dma else "Normal"
    )

    technical_count = sum([latest < sma20, latest < sma50, latest < sma200, rsi14 < 50, hist.iloc[-1] < 0])
    technical_status = status_from_counts(technical_count, 2, 3)

    available = [t for t in T2108_TICKERS if t in d.columns]
    universe = d[available].dropna(axis=1, thresh=120) if available else pd.DataFrame(index=d.index)
    if universe.empty:
        pct50_latest = pct200_latest = breadth_drop5 = np.nan
        breadth_status = "Normal"
        breakdown_rate = breakout_win_rate = np.nan
    else:
        pct50 = pct_above_ma(universe, 50)
        pct200 = pct_above_ma(universe, 200)
        pct50_latest = pct50.iloc[-1]
        pct200_latest = pct200.iloc[-1]
        breadth_drop5 = pct50.diff(5).iloc[-1]
        breadth_status = "Warning" if (
            (pd.notna(pct50_latest) and pct50_latest <= 45)
            or (pd.notna(pct200_latest) and pct200_latest <= 45)
            or (pd.notna(breadth_drop5) and breadth_drop5 <= -10)
        ) else "Watch" if (
            (pd.notna(pct50_latest) and pct50_latest <= 55) or (pd.notna(breadth_drop5) and breadth_drop5 <= -5)
        ) else "Normal"

        prev_high20 = universe.shift(1).rolling(20).max()
        prev_low20 = universe.shift(1).rolling(20).min()
        breakouts = universe > prev_high20
        breakdowns = universe < prev_low20
        future5 = universe.shift(-5) / universe - 1
        wins = breakouts & (future5 > 0.02)
        hist_bo = breakouts.iloc[:-5]
        hist_wins = wins.iloc[:-5]
        events = hist_bo.sum(axis=1).rolling(60).sum()
        win_events = hist_wins.sum(axis=1).rolling(60).sum()
        win_rate = win_events / events.replace(0, np.nan) * 100
        breakout_win_rate = win_rate.dropna().iloc[-1] if not win_rate.dropna().empty else np.nan
        breakdown_rate = breakdowns.iloc[-1].sum() / breakdowns.iloc[-1].count() * 100 if breakdowns.shape[1] else np.nan
    breakout_status = "Warning" if (
        (pd.notna(breakout_win_rate) and breakout_win_rate < 45) or (pd.notna(breakdown_rate) and breakdown_rate >= 8)
    ) else "Watch" if (
        (pd.notna(breakout_win_rate) and breakout_win_rate < 55) or (pd.notna(breakdown_rate) and breakdown_rate >= 5)
    ) else "Normal"

    vix = d["^VIX"].dropna() if "^VIX" in d.columns else pd.Series(dtype=float)
    if vix.empty:
        vix_value = vix_chg5 = np.nan
        vix_status = "Normal"
    else:
        vix_value = float(vix.iloc[-1])
        prior5 = vix.shift(5).iloc[-1] if len(vix) > 5 else np.nan
        vix_chg5 = (vix_value / prior5 - 1) * 100 if pd.notna(prior5) and prior5 != 0 else np.nan
        vix_status = "Warning" if (vix_value >= 25 or (pd.notna(vix_chg5) and vix_chg5 >= 40)) else (
            "Watch" if (vix_value >= 20 or (pd.notna(vix_chg5) and vix_chg5 >= 20)) else "Normal"
        )

    theme_statuses = []
    theme_under = []
    theme_pct50 = []
    for tickers in THEMES.values():
        names = [t for t in tickers if t in d.columns]
        theme = d[names].dropna(axis=1, thresh=80) if names else pd.DataFrame(index=d.index)
        if theme.empty:
            continue
        ew = theme.pct_change().mean(axis=1).add(1).cumprod()
        aligned = spy.reindex(ew.index).ffill()
        if aligned.dropna().empty:
            continue
        rs = ew / (aligned / aligned.dropna().iloc[0])
        under = ew.pct_change(20).iloc[-1] * 100 - aligned.pct_change(20).iloc[-1] * 100
        p50 = pct_above_ma(theme, 50).iloc[-1]
        rs_below = bool(rs.iloc[-1] < rs.rolling(20).mean().iloc[-1])
        status = "Warning" if under <= -5 or p50 <= 40 else "Watch" if under <= -2 or p50 <= 55 or rs_below else "Normal"
        theme_statuses.append(status)
        theme_under.append(under)
        theme_pct50.append(p50)
    theme_status = "Warning" if "Warning" in theme_statuses else "Watch" if "Watch" in theme_statuses else "Normal"

    statuses = {
        "d2_sp500_overextension_status": over_status,
        "d2_downward_momentum_status": momentum_status,
        "d2_range_breakdown_status": range_status,
        "d2_technical_deterioration_status": technical_status,
        "d2_breadth_worsening_status": breadth_status,
        "d2_vix_spike_status": vix_status,
        "d2_breakout_failure_status": breakout_status,
        "d2_theme_momentum_status": theme_status,
    }
    scores = {k.replace("_status", "_score"): status_score(v) for k, v in statuses.items()}
    total = sum(scores.values())
    return {
        **statuses,
        **scores,
        "d2_score": total,
        "d2_warning_count": sum(v == "Warning" for v in statuses.values()),
        "d2_watch_count": sum(v == "Watch" for v in statuses.values()),
        "d2_level": "Critical" if total >= 11 or sum(v == "Warning" for v in statuses.values()) >= 3 else "Elevated" if total >= 7 or sum(v == "Warning" for v in statuses.values()) >= 2 else "Guarded" if total >= 4 or sum(v == "Watch" for v in statuses.values()) >= 3 else "Stable",
        "d2_rsi14": rsi14,
        "d2_dist50": dist50,
        "d2_z20": z20,
        "d2_momentum_count": momentum_count,
        "d2_technical_count": technical_count,
        "d2_pct_above_50dma": pct50_latest,
        "d2_pct_above_200dma": pct200_latest,
        "d2_breadth_drop5": breadth_drop5,
        "d2_vix": vix_value,
        "d2_vix_chg5": vix_chg5,
        "d2_breakout_win_rate": breakout_win_rate,
        "d2_breakdown_rate": breakdown_rate,
        "d2_theme_worst_underperf": np.nanmin(theme_under) if theme_under else np.nan,
        "d2_theme_worst_pct50": np.nanmin(theme_pct50) if theme_pct50 else np.nan,
    }


def build_feature_table(prices: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    weekly_sp = prices[["^GSPC"]].dropna().resample("W-THU").last().dropna()
    weekly_sp = weekly_sp[weekly_sp.index.date <= END]
    rows = []
    for date in weekly_sp.index:
        d = latest_valid(prices, date)
        sp_hist = d["^GSPC"].dropna()
        if len(sp_hist) < 260:
            continue
        latest = d.iloc[-1]
        sp = float(sp_hist.iloc[-1])
        spy = float(latest.get("SPY", np.nan))
        sma200 = float(sp_hist.rolling(200).mean().iloc[-1])
        vix = float(latest.get("^VIX", np.nan))
        vix3m = float(latest.get("^VIX3M", np.nan))
        move = float(latest.get("^MOVE", np.nan))
        dxy = float(latest.get("DX-Y.NYB", np.nan))
        tnx = float(latest.get("^TNX", np.nan))
        irx = float(latest.get("^IRX", np.nan))
        row_fred = get_at_or_before(fred, date)
        if row_fred is None:
            continue
        hy = float(row_fred.get("BAMLH0A0HYM2", np.nan))
        baa10y = float(row_fred.get("BAA10Y", np.nan))
        if pd.notna(hy):
            credit_stress = bool(hy > 5)
            credit_spread_for_model = hy
            credit_spread_source = "HY_OAS"
        else:
            credit_stress = bool(pd.notna(baa10y) and baa10y > 3)
            credit_spread_for_model = baa10y
            credit_spread_source = "BAA10Y"
        net_liq = float(row_fred.get("net_liquidity_t", np.nan))
        reserve = float(row_fred.get("reserve_gdp", np.nan))
        cp = float(row_fred.get("cp_sofr_bps", np.nan))

        net_liq_13 = net_liq - float(fred["net_liquidity_t"].loc[:date].iloc[-14]) if len(fred.loc[:date]) >= 14 else np.nan
        net_liq_26 = net_liq - float(fred["net_liquidity_t"].loc[:date].iloc[-27]) if len(fred.loc[:date]) >= 27 else np.nan
        sp_26 = sp / float(weekly_sp["^GSPC"].loc[:date].iloc[-27]) - 1 if len(weekly_sp.loc[:date]) >= 27 else np.nan

        universe = d[[t for t in T2108_TICKERS if t in d.columns]].dropna(axis=1, thresh=120)
        t2108 = np.nan
        if not universe.empty:
            t2108 = float((universe.iloc[-1] > universe.rolling(40).mean().iloc[-1]).mean() * 100)

        d1_market_score = 0
        d1_market_score += 15 if sp < sma200 else 0
        d1_market_score += 20 if credit_stress else 0
        d1_market_score += 15 if pd.notna(move) and move > 100 else 0
        d1_market_score += 10 if pd.notna(vix) and vix > 25 else 0
        d1_market_score += 10 if pd.notna(vix) and pd.notna(vix3m) and vix > vix3m else 0
        d1_market_score += 10 if pd.notna(dxy) and dxy > 105 else 0
        d1_market_score += 10 if pd.notna(t2108) and t2108 < 40 else 0
        d1_market_score += 5 if pd.notna(net_liq) and net_liq < 7 else 0

        yield_curve = tnx - irx if pd.notna(tnx) and pd.notna(irx) else np.nan
        irx_avg = d["^IRX"].rolling(120).mean().iloc[-1] if "^IRX" in d else np.nan
        hyg_ief = d["HYG"] / d["IEF"] if {"HYG", "IEF"}.issubset(d.columns) else pd.Series(dtype=float)
        hyg_ief_avg = hyg_ief.rolling(252).mean().iloc[-1] if not hyg_ief.empty else np.nan
        spy_sma200 = d["SPY"].rolling(200).mean().iloc[-1] if "SPY" in d else np.nan
        macro_score = normalize(yield_curve, 1.5, -0.5, inverted=True)
        liquidity_score = normalize(irx, irx_avg * 0.8, irx_avg * 1.2)
        credit_score = normalize(hyg_ief.iloc[-1] if not hyg_ief.empty else np.nan, hyg_ief_avg * 1.05, hyg_ief_avg * 0.9, inverted=True)
        breadth_score = normalize(spy, spy_sma200 * 1.05, spy_sma200 * 0.95, inverted=True)
        vix_score = normalize(vix, 15, 35)
        d1_bear_prob12 = (
            macro_score * 0.25
            + liquidity_score * 0.20
            + credit_score * 0.20
            + breadth_score * 0.15
            + vix_score * 0.10
            + 0.65 * 0.05
            + 0.50 * 0.05
        ) * 100 if all(pd.notna(x) for x in [macro_score, liquidity_score, credit_score, breadth_score, vix_score]) else np.nan

        prev_mo = d.iloc[max(0, len(d) - 23)]
        bull_yield_curve = 1.0 if pd.notna(yield_curve) and pd.notna(prev_mo.get("^TNX", np.nan)) and yield_curve > 0 and (prev_mo.get("^TNX", np.nan) - prev_mo.get("^IRX", np.nan)) < 0 else (0.5 if pd.notna(yield_curve) and yield_curve > 0 else 0.0)
        bull_vix = 1.0 if pd.notna(vix) and vix < 15 else (0.5 if pd.notna(vix) and vix < d["^VIX"].rolling(22).mean().iloc[-1] else 0.0)
        bull_credit = 1.0 if not hyg_ief.empty and hyg_ief.iloc[-1] > hyg_ief.rolling(22).mean().iloc[-1] else 0.0
        bull_breadth = 1.0 if pd.notna(spy) and pd.notna(spy_sma200) and spy > spy_sma200 * 1.05 else (0.5 if pd.notna(spy) and pd.notna(spy_sma200) and spy > spy_sma200 else 0.0)
        bull_accum = 1.0 if "SPY" in d.columns and len(d) >= 23 and spy / float(prev_mo.get("SPY", np.nan)) - 1 > 0.02 else (0.5 if "SPY" in d.columns and len(d) >= 23 and spy / float(prev_mo.get("SPY", np.nan)) - 1 > 0 else 0.0)
        bull_tip = 1.0 if "TIP" in d.columns and len(d) >= 22 and d["TIP"].iloc[-1] > d["TIP"].iloc[-22] else 0.0
        d1_bull_score = min(10.0, bull_yield_curve + bull_vix + bull_credit + bull_breadth + bull_accum + bull_tip + 0.5 + 0.5 + 1.0)
        d1_bull_prob = 95.0 if d1_bull_score >= 10 else 85.0 if d1_bull_score >= 8 else 65.0 if d1_bull_score >= 6 else 40.0 if d1_bull_score >= 4 else 20.0

        d1_etf_warning = bool((pd.notna(spy) and pd.notna(spy_sma200) and spy < spy_sma200) or (pd.notna(vix) and vix > 20))
        d1_200ma_diff = sp / sma200 - 1
        d1_200ma_state = "Bullish" if d1_200ma_diff > 0.02 else "Caution" if d1_200ma_diff >= -0.02 else "Bearish"

        bearish_signals = 0
        bearish_signals += 1 if pd.notna(yield_curve) and yield_curve < 0 else 0
        bearish_signals += 1 if not hyg_ief.empty and hyg_ief.iloc[-1] < hyg_ief.rolling(252).mean().iloc[-1] else 0
        bearish_signals += 1 if pd.notna(vix) and vix > 20 else 0
        bearish_signals += 1 if sp < sma200 else 0
        bearish_signals += 1 if pd.notna(t2108) and t2108 < 40 else 0
        rwra_bull = {0: 70.0, 1: 50.0, 2: 30.0, 3: 10.0, 4: 5.0, 5: 0.0}[bearish_signals]
        rwra_bear = {0: 8.0, 1: 15.0, 2: 20.0, 3: 40.0, 4: 50.0, 5: 35.0}[bearish_signals]
        rwra_crisis = {0: 2.0, 1: 5.0, 2: 10.0, 3: 15.0, 4: 30.0, 5: 60.0}[bearish_signals]
        if pd.notna(vix) and vix > 35:
            rwra_bull, rwra_bear, rwra_crisis = 0.0, 0.0, 100.0

        liq_score = 0
        liq_score += 35 if pd.notna(reserve) and reserve < 10 else 20 if pd.notna(reserve) and reserve < 12 else 0
        liq_score += 35 if pd.notna(cp) and cp > 50 else 20 if pd.notna(cp) and cp > 25 else 0
        liq_score += 15 if pd.notna(net_liq_13) and net_liq_13 < 0 else 0
        liq_score += 15 if pd.notna(sp_26) and sp_26 > 0 and pd.notna(net_liq_26) and net_liq_26 < 0 else 0

        row = {
            "date": date,
            "sp500": sp,
            "gold": float(latest.get("GC=F", np.nan)),
            "net_liquidity_t": net_liq,
            "net_liquidity_13w_chg": net_liq_13,
            "net_liquidity_26w_chg": net_liq_26,
            "reserve_gdp": reserve,
            "cp_sofr_bps": cp,
            "hy_spread": hy,
            "baa10y_spread": baa10y,
            "credit_spread_for_model": credit_spread_for_model,
            "credit_spread_source": credit_spread_source,
            "credit_stress": credit_stress,
            "vix": vix,
            "vix3m": vix3m,
            "vix_term_inverted": bool(pd.notna(vix) and pd.notna(vix3m) and vix > vix3m),
            "move": move,
            "dxy": dxy,
            "tnx_irx_curve": yield_curve,
            "t2108": t2108,
            "sp_below_200dma": bool(sp < sma200),
            "sp_200dma_diff": d1_200ma_diff,
            "liquidity_score": min(liq_score, 100),
            "liquidity_divergence": bool(pd.notna(sp_26) and sp_26 > 0 and pd.notna(net_liq_26) and net_liq_26 < 0 and pd.notna(net_liq_13) and net_liq_13 < 0),
            "d1_market_regime_score": d1_market_score,
            "d1_bear_prob12": d1_bear_prob12,
            "d1_bull_prob": d1_bull_prob,
            "d1_bull_score": d1_bull_score,
            "d1_etf_rotation_warning": d1_etf_warning,
            "d1_200ma_state": d1_200ma_state,
            "d1_rwra_bearish_count": bearish_signals,
            "d1_rwra_bull_prob": rwra_bull,
            "d1_rwra_bear_prob": rwra_bear,
            "d1_rwra_crisis_prob": rwra_crisis,
            "d1_gex_available": False,
            "d1_market_pulse_available": False,
            "d1_ml_meta_included": False,
        }
        row.update(dashboard2_indicators(d))
        rows.append(row)
    features = pd.DataFrame(rows).set_index("date").sort_index()
    return features


def add_outcomes(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for horizon in [4, 8, 13, 26]:
        fwd_min = pd.Series(index=out.index, dtype=float)
        values = out["sp500"].to_numpy()
        for i in range(len(out)):
            future = values[i + 1 : i + 1 + horizon]
            if len(future) == horizon:
                fwd_min.iloc[i] = np.nanmin(future)
        out[f"fwd_max_dd_{horizon}w"] = fwd_min / out["sp500"] - 1.0
        for threshold in [0.05, 0.10, 0.15, 0.20]:
            out[f"drop_{int(threshold * 100)}_{horizon}w"] = out[f"fwd_max_dd_{horizon}w"] <= -threshold
    return out


def add_composite_indicator(features: pd.DataFrame) -> pd.DataFrame:
    """Build a walk-forward risk indicator from D1/D2/D3 features.

    The target is a 10% forward drawdown within 13 weeks. Each prediction is
    trained only on prior weekly observations, then converted to a 0-100 score.
    """
    out = features.copy()
    feature_cols = [
        "liquidity_score",
        "net_liquidity_13w_chg",
        "net_liquidity_26w_chg",
        "reserve_gdp",
        "cp_sofr_bps",
        "credit_spread_for_model",
        "credit_stress",
        "vix",
        "vix_term_inverted",
        "move",
        "dxy",
        "t2108",
        "sp_below_200dma",
        "sp_200dma_diff",
        "d1_market_regime_score",
        "d1_bear_prob12",
        "d1_bull_prob",
        "d1_etf_rotation_warning",
        "d1_rwra_bearish_count",
        "d1_rwra_crisis_prob",
        "d2_score",
        "d2_warning_count",
        "d2_watch_count",
        "d2_sp500_overextension_score",
        "d2_downward_momentum_score",
        "d2_range_breakdown_score",
        "d2_technical_deterioration_score",
        "d2_breadth_worsening_score",
        "d2_vix_spike_score",
        "d2_breakout_failure_score",
        "d2_theme_momentum_score",
    ]
    for col in feature_cols:
        if col not in out.columns:
            out[col] = np.nan

    scores = pd.Series(np.nan, index=out.index, dtype=float)
    min_train = 260
    model = None
    medians = None
    for i in range(min_train, len(out)):
        if model is None or i % 4 == 0:
            train = out.iloc[:i].copy()
            train = train.dropna(subset=["drop_10_13w"])
            if len(train) < min_train or train["drop_10_13w"].nunique() < 2:
                continue
            x_train = train[feature_cols].copy()
            x_train = x_train.replace([np.inf, -np.inf], np.nan)
            medians = x_train.median(numeric_only=True)
            x_train = x_train.fillna(medians).fillna(0)

            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                ),
            )
            model.fit(x_train, train["drop_10_13w"].astype(bool))

        x_now = out.iloc[[i]][feature_cols].replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
        scores.iloc[i] = float(model.predict_proba(x_now)[0, 1] * 100.0)

    out["composite_risk_score"] = scores
    out["composite_risk_state"] = "Unavailable"
    out.loc[out["composite_risk_score"] < 30, "composite_risk_state"] = "Normal"
    out.loc[(out["composite_risk_score"] >= 30) & (out["composite_risk_score"] < 55), "composite_risk_state"] = "Watch"
    out.loc[(out["composite_risk_score"] >= 55) & (out["composite_risk_score"] < 75), "composite_risk_state"] = "Warning"
    out.loc[out["composite_risk_score"] >= 75, "composite_risk_state"] = "Stress"
    return out


def score_models(features: pd.DataFrame) -> pd.DataFrame:
    x = features.copy()
    # Tiered models: early warning -> correction -> crash.
    x["model_liquidity_only"] = x["liquidity_divergence"].fillna(False)
    x["model_correction_watch"] = x["model_liquidity_only"] & (
        (x["d2_score"] >= 5) | (x["d1_market_regime_score"] >= 30) | (x["d1_etf_rotation_warning"])
    )
    x["model_confirmed_drop"] = (
        (x["d2_score"] >= 7)
        & ((x["d1_market_regime_score"] >= 40) | (x["sp_below_200dma"]) | (x["credit_stress"]) | (x["vix"] > 25))
    )
    x["model_crash_watch"] = (
        (x["d2_score"] >= 8)
        & ((x["credit_stress"]) | (x["vix"] > 30) | (x["sp_below_200dma"]) | (x["cp_sofr_bps"] > 50))
        & ((x["d1_market_regime_score"] >= 45) | (x["d1_rwra_crisis_prob"] >= 30) | (x["liquidity_score"] >= 50))
    )
    x["model_crash_confirmed"] = (
        (x["d2_score"] >= 10)
        & (x["d1_market_regime_score"] >= 55)
        & ((x["credit_stress"]) | (x["vix"] > 30) | (x["cp_sofr_bps"] > 50))
        & (x["sp_below_200dma"])
    )
    return x


def evaluate_signal(data: pd.DataFrame, signal: str, outcome: str) -> dict[str, object]:
    valid = data.dropna(subset=[outcome]).copy()
    sig = valid[signal].fillna(False)
    n = int(sig.sum())
    base = float(valid[outcome].mean())
    if n == 0:
        return {"signal": signal, "outcome": outcome, "n": 0, "base_rate": base, "hit_rate": np.nan, "lift": np.nan}
    hit = float(valid.loc[sig, outcome].mean())
    return {
        "signal": signal,
        "outcome": outcome,
        "n": n,
        "base_rate": base,
        "hit_rate": hit,
        "lift": hit / base if base else np.nan,
        "avg_fwd_dd_26w": float(valid.loc[sig, "fwd_max_dd_26w"].mean()),
        "median_fwd_dd_26w": float(valid.loc[sig, "fwd_max_dd_26w"].median()),
        "avg_fwd_dd_26w": float(valid.loc[sig, "fwd_max_dd_26w"].mean()),
    }


def evaluate_composite_bins(data: pd.DataFrame) -> pd.DataFrame:
    valid = data.dropna(subset=["composite_risk_score", "drop_10_13w", "drop_10_26w"]).copy()
    bins = [0, 30, 55, 75, 101]
    labels = ["Normal", "Watch", "Warning", "Stress"]
    valid["risk_bin"] = pd.cut(valid["composite_risk_score"], bins=bins, labels=labels, right=False)
    rows = []
    for label in labels:
        subset = valid[valid["risk_bin"].eq(label)]
        if subset.empty:
            continue
        row = {
            "risk_bin": label,
            "n": len(subset),
            "avg_score": subset["composite_risk_score"].mean(),
        }
        for horizon in [4, 8, 13, 26]:
            row[f"drop5_{horizon}w"] = subset[f"drop_5_{horizon}w"].mean()
            row[f"drop10_{horizon}w"] = subset[f"drop_10_{horizon}w"].mean()
            row[f"drop15_{horizon}w"] = subset[f"drop_15_{horizon}w"].mean()
            row[f"drop20_{horizon}w"] = subset[f"drop_20_{horizon}w"].mean()
            row[f"avg_dd_{horizon}w"] = subset[f"fwd_max_dd_{horizon}w"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def build_daily_confirmation(prices: pd.DataFrame, weekly_features: pd.DataFrame) -> pd.DataFrame:
    daily = prices.copy().sort_index().ffill()
    if "^GSPC" not in daily.columns:
        return pd.DataFrame()

    sp = daily["^GSPC"].dropna()
    out = pd.DataFrame(index=sp.index)
    out["sp500"] = sp
    out["sma20"] = sp.rolling(20).mean()
    out["sma50"] = sp.rolling(50).mean()
    out["sma200"] = sp.rolling(200).mean()
    out["ret5"] = sp.pct_change(5)
    out["ret20"] = sp.pct_change(20)
    out["drawdown20"] = sp / sp.rolling(20).max() - 1.0
    out["drawdown60"] = sp / sp.rolling(60).max() - 1.0
    out["rsi14"] = rsi(sp)

    vix = daily["^VIX"].reindex(out.index).ffill() if "^VIX" in daily else pd.Series(np.nan, index=out.index)
    vix3m = daily["^VIX3M"].reindex(out.index).ffill() if "^VIX3M" in daily else pd.Series(np.nan, index=out.index)
    move = daily["^MOVE"].reindex(out.index).ffill() if "^MOVE" in daily else pd.Series(np.nan, index=out.index)
    dxy = daily["DX-Y.NYB"].reindex(out.index).ffill() if "DX-Y.NYB" in daily else pd.Series(np.nan, index=out.index)
    out["vix"] = vix
    out["vix_chg5"] = vix / vix.shift(5) - 1.0
    out["vix_term_inverted"] = vix > vix3m
    out["move"] = move
    out["dxy_ret20"] = dxy.pct_change(20)

    if {"HYG", "IEF"}.issubset(daily.columns):
        hyg_ief = (daily["HYG"] / daily["IEF"]).reindex(out.index).ffill()
        out["hyg_ief_ret20"] = hyg_ief.pct_change(20)
        out["hyg_ief_below_50dma"] = hyg_ief < hyg_ief.rolling(50).mean()
    else:
        out["hyg_ief_ret20"] = np.nan
        out["hyg_ief_below_50dma"] = False

    available = [t for t in T2108_TICKERS if t in daily.columns]
    universe = daily[available].dropna(axis=1, thresh=220) if available else pd.DataFrame(index=daily.index)
    if universe.empty:
        out["pct_above_50dma"] = np.nan
        out["pct_above_200dma"] = np.nan
        out["breadth_drop5"] = np.nan
    else:
        pct50 = pct_above_ma(universe, 50).reindex(out.index)
        pct200 = pct_above_ma(universe, 200).reindex(out.index)
        out["pct_above_50dma"] = pct50
        out["pct_above_200dma"] = pct200
        out["breadth_drop5"] = pct50.diff(5)

    score = pd.Series(0.0, index=out.index)
    score += np.where(out["sp500"] < out["sma200"], 18, 0)
    score += np.where(out["sma20"] < out["sma50"], 10, 0)
    score += np.where(out["ret20"] < -0.03, 12, np.where(out["ret20"] < 0, 6, 0))
    score += np.where(out["drawdown20"] < -0.05, 10, np.where(out["drawdown20"] < -0.03, 5, 0))
    score += np.where(out["vix"] >= 25, 14, np.where(out["vix"] >= 20, 8, 0))
    score += np.where(out["vix_chg5"] >= 0.40, 10, np.where(out["vix_chg5"] >= 0.20, 5, 0))
    score += np.where(out["vix_term_inverted"].fillna(False), 8, 0)
    score += np.where((out["hyg_ief_ret20"] < -0.02) | out["hyg_ief_below_50dma"].fillna(False), 10, 0)
    score += np.where(out["pct_above_50dma"] <= 45, 10, np.where(out["pct_above_50dma"] <= 55, 5, 0))
    score += np.where(out["breadth_drop5"] <= -10, 8, np.where(out["breadth_drop5"] <= -5, 4, 0))
    score += np.where(out["move"] > 100, 5, 0)
    score += np.where(out["dxy_ret20"] > 0.02, 5, 0)
    out["daily_confirmation_score"] = score.clip(0, 100)
    out["daily_confirmation_state"] = "Clear"
    out.loc[out["daily_confirmation_score"] >= 30, "daily_confirmation_state"] = "Confirming"
    out.loc[out["daily_confirmation_score"] >= 55, "daily_confirmation_state"] = "High Confirmation"
    out.index.name = "date"

    weekly = weekly_features[
        [
            "sp500",
            "composite_risk_score",
            "composite_risk_state",
            "d1_market_regime_score",
            "d2_score",
            "liquidity_score",
        ]
    ].copy()
    weekly = weekly.rename(columns={"sp500": "weekly_sp500"})
    merged = pd.merge_asof(
        out.reset_index().sort_values("date"),
        weekly.reset_index().rename(columns={"date": "weekly_date"}).sort_values("weekly_date"),
        left_on="date",
        right_on="weekly_date",
        direction="backward",
    ).set_index("date")

    for days, label in [(20, "4w"), (40, "8w"), (65, "13w"), (130, "26w")]:
        fwd_min = pd.Series(index=merged.index, dtype=float)
        values = merged["sp500"].to_numpy()
        for i in range(len(values)):
            future = values[i + 1 : i + 1 + days]
            if len(future) == days:
                fwd_min.iloc[i] = np.nanmin(future)
        merged[f"daily_fwd_max_dd_{label}"] = fwd_min / merged["sp500"] - 1.0
        for threshold in [0.05, 0.10, 0.15, 0.20]:
            merged[f"daily_drop_{int(threshold * 100)}_{label}"] = merged[f"daily_fwd_max_dd_{label}"] <= -threshold
    return merged.dropna(subset=["composite_risk_state", "daily_confirmation_score"])


def evaluate_daily_confirmation(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    rows = []
    valid = daily.dropna(subset=["daily_drop_10_13w", "daily_drop_20_26w"]).copy()
    for weekly_state in ["Normal", "Watch", "Warning", "Stress"]:
        for daily_state in ["Clear", "Confirming", "High Confirmation"]:
            subset = valid[
                valid["composite_risk_state"].eq(weekly_state)
                & valid["daily_confirmation_state"].eq(daily_state)
            ]
            if subset.empty:
                continue
            rows.append(
                {
                    "weekly_state": weekly_state,
                    "daily_confirmation_state": daily_state,
                    "n": len(subset),
                    "avg_weekly_score": subset["composite_risk_score"].mean(),
                    "avg_daily_score": subset["daily_confirmation_score"].mean(),
                    "drop5_4w": subset["daily_drop_5_4w"].mean(),
                    "drop10_8w": subset["daily_drop_10_8w"].mean(),
                    "drop10_13w": subset["daily_drop_10_13w"].mean(),
                    "drop20_26w": subset["daily_drop_20_26w"].mean(),
                    "avg_dd_13w": subset["daily_fwd_max_dd_13w"].mean(),
                    "avg_dd_26w": subset["daily_fwd_max_dd_26w"].mean(),
                }
            )
    return pd.DataFrame(rows)


def grid_search(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = data.dropna(subset=["drop_20_26w"]).copy()
    for d1 in [20, 30, 40, 50, 60, 70]:
        for d2 in [3, 5, 7, 9, 11]:
            for liq in [0, 30, 50, 70]:
                sig = (
                    (valid["d1_market_regime_score"] >= d1)
                    & (valid["d2_score"] >= d2)
                    & (valid["liquidity_score"] >= liq)
                )
                if sig.sum() < 5:
                    continue
                rows.append(
                    {
                        "d1_min": d1,
                        "d2_min": d2,
                        "liquidity_min": liq,
                        "n": int(sig.sum()),
                        "drop5_4w": float(valid.loc[sig, "drop_5_4w"].mean()),
                        "drop10_8w": float(valid.loc[sig, "drop_10_8w"].mean()),
                        "drop10_13w": float(valid.loc[sig, "drop_10_13w"].mean()),
                        "drop10_26w": float(valid.loc[sig, "drop_10_26w"].mean()),
                        "drop15_26w": float(valid.loc[sig, "drop_15_26w"].mean()),
                        "drop20_26w": float(valid.loc[sig, "drop_20_26w"].mean()),
                        "avg_fwd_dd_26w": float(valid.loc[sig, "fwd_max_dd_26w"].mean()),
                        "median_fwd_dd_26w": float(valid.loc[sig, "fwd_max_dd_26w"].median()),
                    }
                )
    return pd.DataFrame(rows).sort_values(["drop20_26w", "drop10_26w", "avg_fwd_dd_26w"], ascending=[False, False, True])


def main() -> None:
    prices = download_prices()
    fred = build_fred_weekly()
    features = build_feature_table(prices, fred)
    features = add_outcomes(features)
    features = add_composite_indicator(features)
    features = score_models(features)

    features.to_csv(OUT_DIR / "weekly_feature_outcomes.csv")
    daily_confirmation = build_daily_confirmation(prices, features)
    daily_confirmation.to_csv(OUT_DIR / "daily_confirmation.csv")
    daily_confirmation_eval = evaluate_daily_confirmation(daily_confirmation)
    daily_confirmation_eval.to_csv(OUT_DIR / "daily_confirmation_evaluation.csv", index=False)

    signals = [
        "model_liquidity_only",
        "model_correction_watch",
        "model_confirmed_drop",
        "model_crash_watch",
        "model_crash_confirmed",
    ]
    outcomes = [
        "drop_5_4w",
        "drop_10_4w",
        "drop_5_8w",
        "drop_10_8w",
        "drop_5_13w",
        "drop_10_13w",
        "drop_10_26w",
        "drop_15_26w",
        "drop_20_26w",
    ]
    evals = pd.DataFrame([evaluate_signal(features, sig, out) for sig in signals for out in outcomes])
    evals.to_csv(OUT_DIR / "signal_evaluation.csv", index=False)

    composite_bins = evaluate_composite_bins(features)
    composite_bins.to_csv(OUT_DIR / "composite_risk_bins.csv", index=False)

    grid = grid_search(features)
    grid.to_csv(OUT_DIR / "threshold_grid.csv", index=False)

    latest = features.iloc[-1].to_frame("latest")
    latest.to_csv(OUT_DIR / "latest_signal_snapshot.csv")

    coverage = pd.DataFrame(
        {
            "first_valid": features.apply(lambda s: s.first_valid_index()),
            "last_valid": features.apply(lambda s: s.last_valid_index()),
            "non_null": features.notna().sum(),
            "rows": len(features),
        }
    )
    coverage.to_csv(OUT_DIR / "data_coverage.csv")

    print(f"features: {features.shape}, {features.index.min().date()} -> {features.index.max().date()}")
    print(f"exports: {OUT_DIR}")
    print("\nSignal evaluation:")
    print(evals.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nComposite risk bins:")
    print(composite_bins.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nDaily confirmation evaluation:")
    print(daily_confirmation_eval.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nTop threshold grid:")
    print(grid.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nLatest key fields:")
    key_fields = [
        "sp500",
        "net_liquidity_t",
        "net_liquidity_13w_chg",
        "net_liquidity_26w_chg",
        "liquidity_score",
        "reserve_gdp",
        "cp_sofr_bps",
        "credit_spread_for_model",
        "credit_spread_source",
        "credit_stress",
        "d1_market_regime_score",
        "d1_bear_prob12",
        "d1_bull_prob",
        "d1_etf_rotation_warning",
        "d1_200ma_state",
        "d1_rwra_bearish_count",
        "d2_score",
        "d2_level",
        "d2_warning_count",
        "d2_watch_count",
        "composite_risk_score",
        "composite_risk_state",
        "model_liquidity_only",
        "model_correction_watch",
        "model_confirmed_drop",
        "model_crash_watch",
        "model_crash_confirmed",
    ]
    print(features[key_fields].tail(1).T.to_string())


if __name__ == "__main__":
    main()

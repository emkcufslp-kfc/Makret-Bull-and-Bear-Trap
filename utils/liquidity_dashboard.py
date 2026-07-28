from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests
import yfinance as yf


FRED_SERIES = {
    "WALCL": "Fed Total Assets",
    "WTREGEN": "Treasury General Account",
    "RRPONTSYD": "Overnight Reverse Repo",
    "WRESBAL": "Reserve Balances",
    "GDP": "Gross Domestic Product",
    "SOFR": "SOFR",
    "DCPF3M": "3M AA Financial Commercial Paper",
}


@dataclass(frozen=True)
class LiquidityThresholds:
    reserve_watch_pct: float = 12.0
    reserve_stress_pct: float = 10.0
    cp_sofr_watch_bps: float = 25.0
    cp_sofr_stress_bps: float = 50.0
    liquidity_trend_weeks: int = 13
    divergence_weeks: int = 26


def _fred_csv_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_fred_series(series_id: str, start: dt.date, end: dt.date) -> pd.Series:
    response = requests.get(_fred_csv_url(series_id), timeout=30)
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text), na_values=["."])
    date_col = "observation_date" if "observation_date" in data.columns else "DATE"
    data[date_col] = pd.to_datetime(data[date_col])
    values = pd.to_numeric(data[series_id], errors="coerce")
    series = pd.Series(values.values, index=data[date_col], name=series_id).sort_index()
    return series.loc[(series.index.date >= start) & (series.index.date <= end)]


def fetch_fred_data(start: dt.date, end: dt.date) -> pd.DataFrame:
    parts = [fetch_fred_series(series_id, start, end) for series_id in FRED_SERIES]
    return pd.concat(parts, axis=1).sort_index().ffill()


def fetch_market_data(start: dt.date, end: dt.date) -> pd.DataFrame:
    raw = yf.download(["^GSPC", "GC=F"], start=start, end=end + dt.timedelta(days=1), auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame(columns=["SP_500", "Gold"], index=pd.DatetimeIndex([], name="Date"))

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.rename(columns={"^GSPC": "SP_500", "GC=F": "Gold"})
    return prices[["SP_500", "Gold"]].dropna(how="all")


def build_liquidity_dashboard(
    years: int = 10,
    end_date: dt.date | None = None,
    thresholds: LiquidityThresholds | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    thresholds = thresholds or LiquidityThresholds()
    end = end_date or dt.date.today()
    start = end - dt.timedelta(days=round(years * 365.25))

    fred = fetch_fred_data(start, end)
    market = fetch_market_data(start, end)

    fred["RRPONTSYD_M"] = fred["RRPONTSYD"] * 1000.0
    fred["Net_Liquidity"] = fred["WALCL"] - fred["WTREGEN"] - fred["RRPONTSYD_M"]
    fred["Reserve_to_GDP"] = (fred["WRESBAL"] / 1000.0) / fred["GDP"] * 100.0
    fred["CP_SOFR_Spread_bps"] = (fred["DCPF3M"] - fred["SOFR"]) * 100.0

    weekly_fred = fred.resample("W-THU").last()
    weekly_market = market.resample("W-THU").last()
    dashboard = pd.concat(
        [
            weekly_fred[
                [
                    "WALCL",
                    "WTREGEN",
                    "RRPONTSYD",
                    "RRPONTSYD_M",
                    "Net_Liquidity",
                    "WRESBAL",
                    "GDP",
                    "Reserve_to_GDP",
                    "CP_SOFR_Spread_bps",
                ]
            ],
            weekly_market,
        ],
        axis=1,
    ).ffill()

    dashboard = dashboard[dashboard.index.date <= end]
    dashboard = dashboard.dropna(subset=["Net_Liquidity", "SP_500", "Gold"])
    dashboard["Net_Liquidity_T"] = dashboard["Net_Liquidity"] / 1_000_000.0
    dashboard["Liquidity_13W_Change_T"] = dashboard["Net_Liquidity_T"].diff(thresholds.liquidity_trend_weeks)
    dashboard["Liquidity_26W_Change_T"] = dashboard["Net_Liquidity_T"].diff(thresholds.divergence_weeks)
    dashboard["SP_500_26W_Return"] = dashboard["SP_500"].pct_change(thresholds.divergence_weeks) * 100.0
    dashboard["Gold_26W_Return"] = dashboard["Gold"].pct_change(thresholds.divergence_weeks) * 100.0
    dashboard["Liquidity_Weekly_Change"] = dashboard["Net_Liquidity_T"].diff()
    dashboard["SP_500_Weekly_Return"] = dashboard["SP_500"].pct_change()
    dashboard["Gold_Weekly_Return"] = dashboard["Gold"].pct_change()
    dashboard["Liquidity_SP_500_Corr_26W"] = dashboard["Liquidity_Weekly_Change"].rolling(26).corr(
        dashboard["SP_500_Weekly_Return"]
    )
    dashboard["Liquidity_Gold_Corr_26W"] = dashboard["Liquidity_Weekly_Change"].rolling(26).corr(
        dashboard["Gold_Weekly_Return"]
    )
    dashboard["SP_500_Indexed"] = dashboard["SP_500"] / dashboard["SP_500"].iloc[0] * 100.0
    dashboard["Gold_Indexed"] = dashboard["Gold"] / dashboard["Gold"].iloc[0] * 100.0
    dashboard["Liquidity_Indexed"] = dashboard["Net_Liquidity_T"] / dashboard["Net_Liquidity_T"].iloc[0] * 100.0
    dashboard = add_historical_signals(dashboard, thresholds)

    summary = summarize_liquidity(dashboard, thresholds)
    return dashboard, summary


def add_historical_signals(data: pd.DataFrame, thresholds: LiquidityThresholds) -> pd.DataFrame:
    result = data.copy()
    risk_score = pd.Series(0, index=result.index, dtype=float)

    reserve = result["Reserve_to_GDP"]
    cp_spread = result["CP_SOFR_Spread_bps"]
    liq_13w = result["Liquidity_13W_Change_T"]
    liq_26w = result["Liquidity_26W_Change_T"]
    sp_26w = result["SP_500_26W_Return"]

    risk_score += (reserve < thresholds.reserve_stress_pct).fillna(False) * 35
    risk_score += ((reserve >= thresholds.reserve_stress_pct) & (reserve < thresholds.reserve_watch_pct)).fillna(False) * 20
    risk_score += (cp_spread > thresholds.cp_sofr_stress_bps).fillna(False) * 35
    risk_score += ((cp_spread <= thresholds.cp_sofr_stress_bps) & (cp_spread > thresholds.cp_sofr_watch_bps)).fillna(False) * 20
    risk_score += (liq_13w < 0).fillna(False) * 15
    risk_score += ((sp_26w > 0) & (liq_26w < 0)).fillna(False) * 15

    result["Risk_Score"] = risk_score.clip(upper=100)
    result["Liquidity_State"] = "Supportive"
    result.loc[result["Risk_Score"] >= 30, "Liquidity_State"] = "Watch"
    result.loc[result["Risk_Score"] >= 60, "Liquidity_State"] = "Stress"
    result["Possible_Turning_Point"] = (
        (result["SP_500_26W_Return"] > 0)
        & (result["Liquidity_26W_Change_T"] < 0)
        & (result["Liquidity_13W_Change_T"] < 0)
    ).fillna(False)
    result["Turning_Point_Start"] = result["Possible_Turning_Point"] & ~result["Possible_Turning_Point"].shift(
        fill_value=False
    )
    return result


def summarize_liquidity(data: pd.DataFrame, thresholds: LiquidityThresholds) -> dict[str, object]:
    if data.empty:
        return {"status": "No data", "risk_score": 0, "messages": []}

    latest = data.iloc[-1]
    risk_score = 0
    messages: list[str] = []

    reserve_ratio = float(latest.get("Reserve_to_GDP", float("nan")))
    cp_spread = float(latest.get("CP_SOFR_Spread_bps", float("nan")))
    liq_13w = float(latest.get("Liquidity_13W_Change_T", 0.0))
    liq_26w = float(latest.get("Liquidity_26W_Change_T", 0.0))
    sp_26w = float(latest.get("SP_500_26W_Return", 0.0))

    if pd.notna(reserve_ratio) and reserve_ratio < thresholds.reserve_stress_pct:
        risk_score += 35
        messages.append("Bank reserves are below the stress threshold versus GDP.")
    elif pd.notna(reserve_ratio) and reserve_ratio < thresholds.reserve_watch_pct:
        risk_score += 20
        messages.append("Bank reserves are in the watch zone versus GDP.")

    if pd.notna(cp_spread) and cp_spread > thresholds.cp_sofr_stress_bps:
        risk_score += 35
        messages.append("Commercial paper funding is materially wider than SOFR.")
    elif pd.notna(cp_spread) and cp_spread > thresholds.cp_sofr_watch_bps:
        risk_score += 20
        messages.append("Commercial paper funding spread is elevated.")

    if liq_13w < 0:
        risk_score += 15
        messages.append("Net liquidity has contracted over the last 13 weeks.")

    divergence = sp_26w > 0 and liq_26w < 0
    if divergence:
        risk_score += 15
        messages.append("S&P 500 is rising while net liquidity is falling over 26 weeks.")

    if risk_score >= 60:
        status = "Stress"
    elif risk_score >= 30:
        status = "Watch"
    else:
        status = "Supportive"

    if not messages:
        messages.append("Liquidity plumbing is not flashing a major stress warning.")

    return {
        "as_of": data.index[-1].date(),
        "status": status,
        "risk_score": min(risk_score, 100),
        "messages": messages,
        "net_liquidity_t": float(latest["Net_Liquidity_T"]),
        "liquidity_13w_change_t": liq_13w,
        "reserve_to_gdp": reserve_ratio,
        "cp_sofr_spread_bps": cp_spread,
        "sp_500_26w_return": sp_26w,
        "gold_26w_return": float(latest.get("Gold_26W_Return", 0.0)),
        "sp_corr_26w": float(latest.get("Liquidity_SP_500_Corr_26W", float("nan"))),
        "gold_corr_26w": float(latest.get("Liquidity_Gold_Corr_26W", float("nan"))),
        "divergence": divergence,
    }

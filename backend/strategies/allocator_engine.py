from __future__ import annotations

import ast
import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
EXPORTS_DIR = ROOT_DIR / "exports"
BUNDLED_DATA_DIR = Path(__file__).resolve().parent / "data"
WORKSHEET_CSV = EXPORTS_DIR / "model_change_worksheet_full_history.csv"
BUNDLED_WORKSHEET_CSV = BUNDLED_DATA_DIR / "model_change_worksheet_full_history.csv"
INITIAL_CAPITAL = 100000.0
COST_BPS = 5.0
STATE_WEIGHTS = {
    "normal": {"SPY": 0.50, "QQQ": 0.40, "GLD": 0.10},
    "warning": {"SPY": 0.30, "QQQ": 0.00, "GLD": 0.70},
    "severe": {"SPY": 0.20, "QQQ": 0.00, "GLD": 0.80},
}
DISPLAY_NAMES = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust",
    "GLD": "SPDR Gold Shares",
}

import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.data_engine import get_clean_master
from utils.model_change_monitor import ALL_MODELS, get_model_status_row, status_shift_count


@dataclass(frozen=True)
class AllocatorResult:
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    state_frame: pd.DataFrame
    weights_frame: pd.DataFrame
    target_frame: pd.DataFrame
    trade_log: pd.DataFrame
    metrics: dict[str, float]


def _load_price_frame() -> pd.DataFrame:
    master = get_clean_master().ffill()
    prices = master[["SPY", "QQQ", "GLD"]].dropna().copy()
    prices = prices[prices.index >= pd.Timestamp("2006-05-30")].sort_index()
    prices["month_key"] = prices.index.to_period("M").astype(str)
    prices = prices.reset_index().rename(columns={"index": "Date"})
    first_by_month = prices.groupby("month_key")["Date"].min().rename("first_trading_day")
    prices = prices.merge(first_by_month, left_on="month_key", right_index=True, how="left")
    prices["is_month_start"] = prices["Date"] == prices["first_trading_day"]
    return prices.drop(columns=["first_trading_day"]).set_index("Date")


@lru_cache(maxsize=1)
def _build_event_frame_fallback() -> pd.DataFrame:
    master = get_clean_master().ffill().dropna(how="all")
    if master.empty:
        return pd.DataFrame()

    rows = []
    previous_snapshot = None
    previous_shift_count = 0
    previous_warning_mode = False

    for ts in master.index:
        current_snapshot = get_model_status_row(ts.date())
        if current_snapshot is None:
            continue

        if previous_snapshot is None:
            previous_snapshot = current_snapshot
            continue

        shift_count, changed_models = status_shift_count(previous_snapshot, current_snapshot)
        current_warning_mode = shift_count >= 3

        if current_warning_mode != previous_warning_mode:
            event = {
                "Date": pd.Timestamp(current_snapshot["Date"]),
                "SPY Price": current_snapshot["SPY Price"],
                "Prev Shift Count": previous_shift_count,
                "New Shift Count": shift_count,
                "Warning Mode Change": "Triggered" if current_warning_mode else "Cleared",
                "Changed Core Models": ", ".join(changed_models) if changed_models else "None",
            }
            for model in ALL_MODELS:
                event[model] = current_snapshot.get(model)
            rows.append(event)

        previous_snapshot = current_snapshot
        previous_shift_count = shift_count
        previous_warning_mode = current_warning_mode

    return pd.DataFrame(rows)


def _load_event_frame() -> pd.DataFrame:
    worksheet_path = WORKSHEET_CSV if WORKSHEET_CSV.exists() else BUNDLED_WORKSHEET_CSV
    if worksheet_path.exists():
        events = pd.read_csv(worksheet_path, parse_dates=["Date"]).sort_values("Date")
    else:
        events = _build_event_frame_fallback().sort_values("Date")
        if events.empty:
            raise FileNotFoundError(
                f"Missing worksheet export/bundled worksheet and unable to rebuild warning events in memory: {WORKSHEET_CSV}"
            )
    cols = [
        "Date",
        "Warning Mode Change",
        "Market Regime",
        "Bear Trap",
        "Bull Trap",
        "ETF Rotation",
        "200MA Strategy",
        "ML Meta-Indicator",
        "Combined Macro + RWRA",
        "Market Pulse",
    ]
    return events[cols].rename(columns={"Warning Mode Change": "event_type"})


def _build_daily_signal_frame(prices: pd.DataFrame) -> pd.DataFrame:
    events = _load_event_frame()
    df = prices.reset_index().rename(columns={"index": "Date"}).merge(events, on="Date", how="left")

    warning_mode = False
    latest_snapshot = {
        "Market Regime": "LOW RISK REGIME",
        "Bear Trap": "LOW RISK",
        "Bull Trap": "BULLISH / LOW RISK",
        "ETF Rotation": "NORMAL",
        "200MA Strategy": "BULLISH (Wait for Exit)",
        "ML Meta-Indicator": "NEUTRAL / CAUTION",
        "Combined Macro + RWRA": "STRONG BULL | HOLD",
        "Market Pulse": "CAUTION / MIXED",
    }

    rows: list[dict[str, object]] = []
    for raw in df.to_dict("records"):
        row = dict(raw)
        row["warning_mode_before_event"] = warning_mode
        event_type = row.get("event_type")
        if pd.notna(event_type):
            for key in latest_snapshot:
                val = row.get(key)
                if pd.notna(val):
                    latest_snapshot[key] = val
            if event_type == "Triggered":
                warning_mode = True
            elif event_type == "Cleared":
                warning_mode = False
        row["warning_mode_after_event"] = warning_mode
        for key, val in latest_snapshot.items():
            row[key] = val
        rows.append(row)

    out = pd.DataFrame(rows).set_index("Date")
    out["combined_guardrail"] = out["Combined Macro + RWRA"].map(
        lambda v: str(v).split("|")[0].strip() if isinstance(v, str) else "UNKNOWN"
    )
    return out


def _state_name(snapshot: pd.Series) -> str:
    severe = (
        str(snapshot["Market Regime"]) == "HIGH RISK"
        or str(snapshot["Bull Trap"]) == "BEARISH / HIGH RISK"
        or str(snapshot["combined_guardrail"]) in {"RISK OFF", "CRISIS"}
        or "BEARISH" in str(snapshot["200MA Strategy"]).upper()
    )
    if severe:
        return "severe"
    if bool(snapshot["warning_mode_after_event"]):
        return "warning"
    return "normal"


def _annual_returns(ret: pd.Series) -> pd.Series:
    return (1 + ret).resample("YE").prod() - 1


def _monthly_returns(ret: pd.Series) -> pd.Series:
    return (1 + ret).resample("ME").prod() - 1


def _rolling_histogram(series: pd.Series, bins: int = 11) -> tuple[list[float], list[int]]:
    clean = pd.Series(series).dropna()
    if clean.empty:
        return [0.0] * bins, [0] * bins
    counts, edges = np.histogram(clean, bins=bins)
    return [float(v) for v in edges[:-1]], [int(v) for v in counts]


def _start_sensitivity(equity: pd.Series) -> tuple[list[float], list[float], str, str]:
    start_points: list[tuple[str, float, float]] = []
    for start_dt in equity.resample("MS").first().index:
        sub = equity[equity.index >= start_dt]
        if len(sub) < 126:
            continue
        years = max((sub.index[-1] - sub.index[0]).days / 365.25, 1 / 12)
        cagr = ((sub.iloc[-1] / sub.iloc[0]) ** (1 / years) - 1) * 100
        drawdown = ((sub / sub.cummax()) - 1).min() * 100
        start_points.append((start_dt.strftime("%Y-%m"), float(drawdown), float(cagr)))
    if not start_points:
        return [], [], "-", "-"
    best = max(start_points, key=lambda item: item[2])
    worst = min(start_points, key=lambda item: item[2])
    return (
        [pt[1] for pt in start_points],
        [pt[2] for pt in start_points],
        f"{best[0]} (CAGR {best[2]:.1f}%)",
        f"{worst[0]} (CAGR {worst[2]:.1f}%)",
    )


def _slice_metrics(equity: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    allocator_ret = equity.pct_change().fillna(0.0)
    benchmark_ret = benchmark.pct_change().fillna(0.0)
    years = len(allocator_ret.dropna()) / 252
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vol = allocator_ret.std() * np.sqrt(252)
    sharpe = ((allocator_ret.mean() - 0.02 / 252) / allocator_ret.std()) * np.sqrt(252) if allocator_ret.std() > 0 else 0.0
    downside = allocator_ret[allocator_ret < 0].std() * np.sqrt(252)
    sortino = ((allocator_ret.mean() - 0.02 / 252) * 252 / downside) if downside and not np.isnan(downside) else 0.0
    drawdown = (equity / equity.cummax()) - 1

    benchmark_vol = benchmark_ret.std() * np.sqrt(252)
    benchmark_sharpe = ((benchmark_ret.mean() - 0.02 / 252) / benchmark_ret.std()) * np.sqrt(252) if benchmark_ret.std() > 0 else 0.0
    benchmark_downside = benchmark_ret[benchmark_ret < 0].std() * np.sqrt(252)
    benchmark_sortino = ((benchmark_ret.mean() - 0.02 / 252) * 252 / benchmark_downside) if benchmark_downside and not np.isnan(benchmark_downside) else 0.0
    benchmark_drawdown = (benchmark / benchmark.cummax()) - 1
    calmar = cagr / abs(drawdown.min()) if drawdown.min() < 0 else 0.0
    benchmark_cagr = (((benchmark.iloc[-1] / benchmark.iloc[0]) ** (1 / years)) - 1) if years > 0 else 0.0
    benchmark_calmar = benchmark_cagr / abs(benchmark_drawdown.min()) if benchmark_drawdown.min() < 0 else 0.0
    return {
        "start": equity.index[0],
        "end": equity.index[-1],
        "years": round(years, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "vol_pct": round(vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(drawdown.min() * 100, 2),
        "benchmark_cagr_pct": round(benchmark_cagr * 100, 2),
        "benchmark_max_dd_pct": round(benchmark_drawdown.min() * 100, 2),
        "benchmark_vol_pct": round(benchmark_vol * 100, 2),
        "benchmark_sharpe": round(benchmark_sharpe, 3),
        "benchmark_sortino": round(benchmark_sortino, 3),
        "benchmark_calmar": round(benchmark_calmar, 3),
        "calmar": round(calmar, 3),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "benchmark_final_equity": round(float(benchmark.iloc[-1]), 2),
    }


def run_allocator_backtest() -> AllocatorResult:
    prices = _load_price_frame()
    signal_frame = _build_daily_signal_frame(prices)
    df = prices.join(signal_frame.drop(columns=["SPY", "QQQ", "GLD", "month_key", "is_month_start"], errors="ignore"), how="left")
    df = df.loc[~df.index.duplicated(keep="last")].copy()
    df["state"] = df.apply(_state_name, axis=1)

    holdings = {asset: 0.0 for asset in DISPLAY_NAMES}
    cash = INITIAL_CAPITAL * (1 - COST_BPS / 10000.0)
    prev_state: str | None = None

    equity_rows: list[tuple[pd.Timestamp, float]] = []
    benchmark_value = INITIAL_CAPITAL * (1 - COST_BPS / 10000.0)
    benchmark_rows: list[tuple[pd.Timestamp, float]] = []
    benchmark_shares = benchmark_value / float(df.iloc[0]["SPY"])

    actual_weights_rows: list[tuple[pd.Timestamp, dict[str, float]]] = []
    target_rows: list[tuple[pd.Timestamp, dict[str, float]]] = []
    trade_rows: list[dict[str, object]] = []
    total_turnover = 1.0
    trade_count = 1

    for date, row in df.iterrows():
        px = {asset: float(row[asset]) for asset in DISPLAY_NAMES}
        state = str(row["state"])
        rebalance = prev_state is None or state != prev_state or bool(row["is_month_start"])
        current_value = cash + sum(holdings[a] * px[a] for a in holdings)

        if rebalance:
            current_values = {a: holdings[a] * px[a] for a in holdings}
            current_weights = {
                a: (current_values[a] / current_value if current_value > 0 else 0.0) for a in holdings
            }
            target_weights = STATE_WEIGHTS[state]
            turnover = sum(abs(target_weights[a] - current_weights[a]) for a in holdings)
            cost = current_value * (COST_BPS / 10000.0) * (turnover / 2.0)
            current_value = max(0.0, current_value - cost)
            post_trade_values = {}
            for asset in holdings:
                holdings[asset] = (current_value * target_weights[asset]) / px[asset]
                post_trade_values[asset] = holdings[asset] * px[asset]
            cash = 0.0
            total_turnover += turnover / 2.0
            trade_count += 1 if prev_state is not None else 0
            diff = {
                asset: target_weights[asset] - current_weights[asset] for asset in holdings
            }
            biggest_add = max(diff.items(), key=lambda item: item[1])
            biggest_cut = min(diff.items(), key=lambda item: item[1])
            trade_rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "regime": state.upper(),
                    "add": f"{biggest_add[0]} {biggest_add[1] * 100:+.1f}%",
                    "cut": f"{biggest_cut[0]} {biggest_cut[1] * 100:+.1f}%",
                    "turnover": f"{(turnover / 2.0) * 100:.1f}%",
                    "mix": " / ".join([f"{a} {target_weights[a] * 100:.0f}%" for a in holdings]),
                    "event": "Month Start" if bool(row["is_month_start"]) and prev_state == state else ("State Shift" if prev_state is not None and state != prev_state else "Initialize"),
                }
            )
            actual_weights = {a: (post_trade_values[a] / current_value if current_value > 0 else 0.0) for a in holdings}
        else:
            post_trade_values = {a: holdings[a] * px[a] for a in holdings}
            actual_weights = {a: (post_trade_values[a] / current_value if current_value > 0 else 0.0) for a in holdings}
            target_weights = STATE_WEIGHTS[state]

        portfolio_value = cash + sum(holdings[a] * px[a] for a in holdings)
        benchmark_value = benchmark_shares * px["SPY"]
        equity_rows.append((date, portfolio_value))
        benchmark_rows.append((date, benchmark_value))
        actual_weights_rows.append((date, actual_weights))
        target_rows.append((date, target_weights))
        prev_state = state

    equity = pd.Series({dt_idx: value for dt_idx, value in equity_rows}, name="Allocator_Equity")
    benchmark = pd.Series({dt_idx: value for dt_idx, value in benchmark_rows}, name="SPY_Equity")
    weights_frame = pd.DataFrame({dt_idx: vals for dt_idx, vals in actual_weights_rows}).T
    target_frame = pd.DataFrame({dt_idx: vals for dt_idx, vals in target_rows}).T
    trade_log = pd.DataFrame(trade_rows)

    metrics = _slice_metrics(equity, benchmark)
    metrics["trade_count"] = int(trade_count)
    metrics["turnover_x"] = round(total_turnover, 2)

    return AllocatorResult(
        equity_curve=equity,
        benchmark_curve=benchmark,
        state_frame=df,
        weights_frame=weights_frame,
        target_frame=target_frame,
        trade_log=trade_log,
        metrics=metrics,
    )


def _heatmap_yearly(equity: pd.Series, benchmark: pd.Series) -> list[dict[str, str]]:
    ret = equity.resample("YE").last().pct_change().dropna()
    bench = benchmark.resample("YE").last().pct_change().dropna()
    dd = ((equity / equity.cummax()) - 1).groupby(equity.index.year).min()
    bench_dd = ((benchmark / benchmark.cummax()) - 1).groupby(benchmark.index.year).min()
    rows = []
    for dt_idx, value in ret.items():
        year = dt_idx.year
        bench_val = float(bench.get(dt_idx, 0.0))
        rows.append(
            {
                "year": str(year),
                "portRet": f"{value * 100:+.1f}%",
                "spyRet": f"{bench_val * 100:+.1f}%",
                "portDD": f"{dd.get(year, 0.0) * 100:.1f}%",
                "spyDD": f"{bench_dd.get(year, 0.0) * 100:.1f}%",
                "winner": "Portfolio" if value > bench_val else "SPY",
            }
        )
    return rows[::-1]


def _parse_mix(text: str) -> dict[str, float]:
    parsed = {}
    for part in text.split("/"):
        tokens = part.strip().split()
        if len(tokens) == 2 and tokens[1].endswith("%"):
            parsed[tokens[0]] = float(tokens[1][:-1])
    return parsed


def load_allocator_json(sel_dt: pd.Timestamp) -> dict[str, object] | None:
    result = run_allocator_backtest()
    eq = result.equity_curve[result.equity_curve.index <= sel_dt]
    bench = result.benchmark_curve[result.benchmark_curve.index <= sel_dt]
    states = result.state_frame[result.state_frame.index <= sel_dt]
    weights = result.weights_frame[result.weights_frame.index <= sel_dt]
    targets = result.target_frame[result.target_frame.index <= sel_dt]
    history = result.trade_log[pd.to_datetime(result.trade_log["date"]) <= sel_dt].copy()

    if eq.empty or states.empty or weights.empty or targets.empty:
        return None

    slice_metrics = _slice_metrics(eq, bench)

    latest_date = eq.index[-1]
    latest_prices = states.loc[latest_date, ["SPY", "QQQ", "GLD"]]
    prev_prices = states.loc[:latest_date, ["SPY", "QQQ", "GLD"]].iloc[-2] if len(states.loc[:latest_date]) > 1 else latest_prices
    current_weights = weights.loc[latest_date].fillna(0.0)
    target_weights = targets.loc[latest_date].fillna(0.0)
    state = str(states.loc[latest_date, "state"])
    combined = str(states.loc[latest_date, "Combined Macro + RWRA"])
    guardrail = str(states.loc[latest_date, "combined_guardrail"])
    warning_mode = bool(states.loc[latest_date, "warning_mode_after_event"])
    risk_note = (
        f"As of {latest_date.strftime('%Y-%m-%d')}, defensive regime is active. Cut QQQ exposure and rotate aggressively into GLD."
        if state == "severe"
        else f"As of {latest_date.strftime('%Y-%m-%d')}, warning mode is active. Hold QQQ at zero and keep a heavy GLD cushion."
        if state == "warning"
        else f"As of {latest_date.strftime('%Y-%m-%d')}, normal regime is active. Maintain the growth blend and rebalance only to target."
    )

    assets = []
    for asset in ["SPY", "QQQ", "GLD"]:
        px = float(latest_prices[asset])
        prev_px = float(prev_prices[asset])
        daily_chg = ((px / prev_px) - 1) * 100 if prev_px else 0.0
        tgt_pct = float(target_weights[asset] * 100)
        cur_pct = float(current_weights[asset] * 100)
        assets.append(
            {
                "symbol": asset,
                "name": DISPLAY_NAMES[asset],
                "price": px,
                "change": daily_chg,
                "target": round(tgt_pct, 1),
                "current": round(cur_pct, 1),
                "lower": max(0.0, round(tgt_pct - 10.0, 1)),
                "upper": min(100.0, round(tgt_pct + 10.0, 1)),
            }
        )

    rolling_1y = eq.pct_change(252)
    mc_values, mc_counts = _rolling_histogram(rolling_1y * 100)
    sc_dd, sc_cagr, best_era, worst_era = _start_sensitivity(eq)

    action_items = []
    for asset in assets:
        drift = asset["current"] - asset["target"]
        dollar_drift = INITIAL_CAPITAL * abs(drift) / 100
        if abs(drift) < 1.0:
            continue
        if drift > 0:
            action_items.append(
                {
                    "tone": "sell",
                    "label": f"Trim {asset['symbol']}",
                    "detail": f"As of {latest_date.strftime('%Y-%m-%d')}: current {asset['current']:.1f}% vs target {asset['target']:.1f}% ({dollar_drift:,.0f} USD overweight).",
                }
            )
        else:
            action_items.append(
                {
                    "tone": "buy",
                    "label": f"Add {asset['symbol']}",
                    "detail": f"As of {latest_date.strftime('%Y-%m-%d')}: current {asset['current']:.1f}% vs target {asset['target']:.1f}% ({dollar_drift:,.0f} USD underweight).",
                }
            )
    if not action_items:
        action_items.append(
            {
                "tone": "hold",
                "label": "Hold targets",
                "detail": f"As of {latest_date.strftime('%Y-%m-%d')}, all sleeves are already close to target. No rebalance trade is required for this PIT snapshot.",
            }
        )

    sliced_turnover = history["turnover"].str.rstrip("%").astype(float).sum() / 100.0 if not history.empty else 0.0
    display_history = history.sort_values("date", ascending=False).head(24).copy()
    display_history["weights"] = display_history["mix"]
    display_history["source"] = display_history["cut"].str.split().str[0]
    display_history["destination"] = display_history["add"].str.split().str[0]
    display_history["change"] = display_history["turnover"]
    display_history["rationale"] = display_history["event"] + " | " + display_history["regime"].str.title()

    monthly = _monthly_returns(eq.pct_change().dropna())
    bench_monthly = _monthly_returns(bench.pct_change().dropna())
    monthly_win = ((monthly > bench_monthly).mean() * 100) if not monthly.empty else 0.0

    return {
        "kpi": {
            "cagr": f"{slice_metrics['cagr_pct']:.2f}%",
            "cagrBench": f"vs SPY: {slice_metrics['benchmark_cagr_pct']:.2f}%",
            "maxdd": f"{slice_metrics['max_dd_pct']:.2f}%",
            "maxddBench": f"vs SPY: {slice_metrics['benchmark_max_dd_pct']:.2f}%",
            "sharpe": f"{slice_metrics['sharpe']:.3f}",
            "sharpeBench": f"vs SPY: {slice_metrics['benchmark_sharpe']:.3f}",
            "ratios": f"{slice_metrics['calmar']:.3f} / {slice_metrics['sortino']:.3f}",
            "ratiosBench": f"vs SPY: {slice_metrics['benchmark_calmar']:.3f} / {slice_metrics['benchmark_sortino']:.3f}",
            "signal": "REBALANCE" if action_items[0]["tone"] != "hold" else "HOLD",
            "signalBg": "bg-orange-500/15 text-orange-300 border border-orange-500/40"
            if action_items[0]["tone"] != "hold"
            else "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30",
        },
        "regime": f"{state.upper()} | {guardrail} | {'WARNING MODE ON' if warning_mode else 'WARNING MODE OFF'}",
        "assets": assets,
        "yearly": _heatmap_yearly(eq, bench),
        "history": display_history.to_dict(orient="records"),
        "chartDates": eq.index.strftime("%Y-%m-%d").tolist()[::5],
        "chartPort": eq.tolist()[::5],
        "chartSpy": bench.tolist()[::5],
        "chartDDPort": [float(v * 100) for v in ((eq / eq.cummax()) - 1).tolist()[::5]],
        "chartDDSpy": [float(v * 100) for v in ((bench / bench.cummax()) - 1).tolist()[::5]],
        "mcValues": mc_values,
        "mcCounts": mc_counts,
        "scDD": sc_dd,
        "scCAGR": sc_cagr,
        "meta": {
            "backtestPeriod": f"{eq.index[0].strftime('%b %Y')} - {eq.index[-1].strftime('%b %Y')}",
            "executionDescription": f"As-of {latest_date.strftime('%Y-%m-%d')} PIT instruction set. The allocator rotates between SPY, QQQ, and GLD using Normal / Warning / Severe weights with 5 bps turnover cost; all actions below are for the selected Master date snapshot only.",
            "lastActionDate": display_history.iloc[0]["date"] if not display_history.empty else latest_date.strftime("%Y-%m-%d"),
            "monthlyWinRate": f"{monthly_win:.1f}%",
            "bestEra": best_era,
            "worstEra": worst_era,
            "sleeves": [
                {"symbol": "SPY", "name": "Core equity beta", "weight": round(float(target_weights["SPY"] * 100), 1)},
                {"symbol": "QQQ", "name": "Growth accelerator", "weight": round(float(target_weights["QQQ"] * 100), 1)},
                {"symbol": "GLD", "name": "Defensive shock absorber", "weight": round(float(target_weights["GLD"] * 100), 1)},
            ],
            "actionItems": action_items,
            "stateWeights": [
                {"state": "Normal", "mix": "SPY 50% / QQQ 40% / GLD 10%"},
                {"state": "Warning", "mix": "SPY 30% / QQQ 0% / GLD 70%"},
                {"state": "Severe", "mix": "SPY 20% / QQQ 0% / GLD 80%"},
            ],
            "currentState": state.upper(),
            "warningMode": "ON" if warning_mode else "OFF",
            "guardrail": guardrail,
            "riskNote": risk_note,
            "combinedSignal": combined,
            "tradeCount": int(len(history)),
            "turnoverX": round(float(sliced_turnover), 2),
            "benchmarkFinalEquity": slice_metrics["benchmark_final_equity"],
            "allocatorFinalEquity": slice_metrics["final_equity"],
            "allocatorVsSpy": round(slice_metrics["final_equity"] - slice_metrics["benchmark_final_equity"], 2),
        },
    }

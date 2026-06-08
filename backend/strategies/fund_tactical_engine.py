from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.data_engine import get_clean_master


warnings.filterwarnings("ignore")


@dataclass
class Config:
    tickers: List[str]
    benchmark: str = "SPY"
    start_date: str = "2019-01-01"
    end_date: str | None = None
    rebalance_frequency: str = "W-FRI"
    trend_window: int = 200
    momentum_window: int = 126
    short_momentum_window: int = 63
    trading_cost: float = 0.0010
    slippage: float = 0.0005
    initial_capital: float = 100000.0
    annual_risk_free_rate: float = 0.02
    turnover_threshold: float = 0.03


CONFIG = Config(tickers=["SPY", "QQQ", "GMOM", "RLY", "DBMF", "SGOV"])

MAX_CAPS = {"SPY": 0.60, "QQQ": 0.35, "GMOM": 0.40, "RLY": 0.35, "DBMF": 0.35, "SGOV": 0.60}

BASE = {
    "US_BULL": {"SPY": 0.45, "QQQ": 0.25, "GMOM": 0.15, "RLY": 0.10, "DBMF": 0.05, "SGOV": 0.00},
    "BROAD_BULL": {"SPY": 0.35, "QQQ": 0.10, "GMOM": 0.25, "RLY": 0.15, "DBMF": 0.10, "SGOV": 0.05},
    "INFLATION": {"SPY": 0.25, "QQQ": 0.05, "GMOM": 0.20, "RLY": 0.30, "DBMF": 0.15, "SGOV": 0.05},
    "RISK_OFF": {"SPY": 0.10, "QQQ": 0.00, "GMOM": 0.20, "RLY": 0.15, "DBMF": 0.25, "SGOV": 0.30},
    "MIXED": {"SPY": 0.30, "QQQ": 0.10, "GMOM": 0.25, "RLY": 0.20, "DBMF": 0.10, "SGOV": 0.05},
}


def normalize(weights: Dict[str, float]) -> Dict[str, float]:
    clean = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(clean.values())
    return {"SGOV": 1.0} if total <= 0 else {k: v / total for k, v in clean.items()}


def apply_caps(weights: Dict[str, float]) -> Dict[str, float]:
    clean = normalize(weights)
    excess = 0.0
    capped: Dict[str, float] = {}
    for ticker, value in clean.items():
        cap = MAX_CAPS.get(ticker, 1.0)
        if value > cap:
            capped[ticker] = cap
            excess += value - cap
        else:
            capped[ticker] = value
    capped["SGOV"] = capped.get("SGOV", 0.0) + excess
    return normalize(capped)


def load_prices(cfg: Config) -> pd.DataFrame:
    master = get_clean_master()
    missing = [ticker for ticker in cfg.tickers if ticker not in master.columns]
    if missing:
        raise RuntimeError(f"Shared master dataset is missing required F-TAA tickers: {missing}")

    prices = master.loc[cfg.start_date:cfg.end_date, cfg.tickers].copy()
    prices = prices.ffill().dropna(how="all").dropna(axis=0, how="any")
    if prices.empty:
        raise RuntimeError("No valid F-TAA prices available from the shared master dataset.")
    return prices


def indicators(prices: pd.DataFrame, cfg: Config):
    return {
        "ma": prices.rolling(cfg.trend_window).mean(),
        "mom": prices.pct_change(cfg.momentum_window),
        "ret": prices.pct_change().fillna(0.0),
    }


def above(ticker: str, day: pd.Timestamp, prices: pd.DataFrame, ind) -> bool:
    return ticker in prices.columns and pd.notna(ind["ma"].loc[day, ticker]) and prices.loc[day, ticker] > ind["ma"].loc[day, ticker]


def momentum(ticker: str, day: pd.Timestamp, ind) -> float:
    if ticker not in ind["mom"].columns or pd.isna(ind["mom"].loc[day, ticker]):
        return np.nan
    return float(ind["mom"].loc[day, ticker])


def classify(day: pd.Timestamp, prices: pd.DataFrame, ind) -> str:
    spy_bull = above("SPY", day, prices, ind)
    qqq_bull = above("QQQ", day, prices, ind)
    gmom_bull = above("GMOM", day, prices, ind)
    rly_bull = above("RLY", day, prices, ind)
    qqq_leads = momentum("QQQ", day, ind) > momentum("SPY", day, ind)
    rly_leads = momentum("RLY", day, ind) > momentum("SPY", day, ind)
    dbmf_leads = momentum("DBMF", day, ind) > momentum("SPY", day, ind)

    if spy_bull and qqq_bull and qqq_leads:
        return "US_BULL"
    if spy_bull and gmom_bull and not qqq_leads:
        return "BROAD_BULL"
    if rly_bull and (rly_leads or dbmf_leads):
        return "INFLATION"
    if (not spy_bull) and (not qqq_bull):
        return "RISK_OFF"
    return "MIXED"


def overlay(day: pd.Timestamp, weights: Dict[str, float], ind) -> Dict[str, float]:
    candidates = [ticker for ticker in ["SPY", "QQQ", "GMOM", "RLY", "DBMF"] if ticker in weights]
    scores = {ticker: momentum(ticker, day, ind) for ticker in candidates}
    scores = {ticker: score for ticker, score in scores.items() if pd.notna(score)}
    if len(scores) < 3:
        return weights

    ranked = sorted(scores, key=scores.get, reverse=True)
    strongest, second, weakest = ranked[0], ranked[1], ranked[-1]
    tilted = weights.copy()
    tilt = 0.08
    tilted[strongest] = tilted.get(strongest, 0.0) + 0.05
    tilted[second] = tilted.get(second, 0.0) + 0.03
    take = min(tilted.get(weakest, 0.0), tilt)
    tilted[weakest] = tilted.get(weakest, 0.0) - take
    remain = tilt - take

    if remain > 0:
        donors = [ticker for ticker in ["SPY", "GMOM", "RLY"] if ticker not in [strongest, second] and tilted.get(ticker, 0.0) > 0]
        donor_sum = sum(tilted[ticker] for ticker in donors)
        if donor_sum > 0:
            for ticker in donors:
                tilted[ticker] -= remain * tilted[ticker] / donor_sum
        else:
            tilted["SGOV"] = max(0.0, tilted.get("SGOV", 0.0) - remain)

    return normalize(tilted)


def bear_guard(day: pd.Timestamp, weights: Dict[str, float], prices: pd.DataFrame, ind) -> Dict[str, float]:
    if not above("SPY", day, prices, ind) and not above("QQQ", day, prices, ind):
        guarded = weights.copy()
        equity = guarded.get("SPY", 0.0) + guarded.get("QQQ", 0.0)
        if equity > 0.15:
            cut = equity - 0.15
            guarded["SPY"] = guarded.get("SPY", 0.0) * 0.15 / equity
            guarded["QQQ"] = guarded.get("QQQ", 0.0) * 0.15 / equity
            guarded["SGOV"] = guarded.get("SGOV", 0.0) + cut
        if guarded.get("SGOV", 0.0) < 0.25:
            need = 0.25 - guarded.get("SGOV", 0.0)
            for donor in ["SPY", "QQQ", "RLY", "GMOM"]:
                take = min(guarded.get(donor, 0.0), need)
                guarded[donor] = guarded.get(donor, 0.0) - take
                guarded["SGOV"] = guarded.get("SGOV", 0.0) + take
                need -= take
                if need <= 0:
                    break
        if guarded.get("DBMF", 0.0) < 0.20:
            need = 0.20 - guarded.get("DBMF", 0.0)
            for donor in ["SPY", "QQQ", "RLY", "GMOM"]:
                take = min(guarded.get(donor, 0.0), need)
                guarded[donor] = guarded.get(donor, 0.0) - take
                guarded["DBMF"] = guarded.get("DBMF", 0.0) + take
                need -= take
                if need <= 0:
                    break
        return normalize(guarded)
    return normalize(weights)


def rebalance_dates(prices: pd.DataFrame, freq: str):
    return prices.groupby(pd.Grouper(freq=freq)).tail(1).index


def generate_weights(prices: pd.DataFrame, cfg: Config):
    ind = indicators(prices, cfg)
    dates = rebalance_dates(prices, cfg.rebalance_frequency)
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    regimes = pd.Series(index=prices.index, dtype="object")
    last = {ticker: 0.0 for ticker in prices.columns}
    if "SGOV" in prices.columns:
        last["SGOV"] = 1.0
    warmup = prices.index[min(cfg.trend_window, len(prices.index) - 1)]

    for day in dates:
        if day < warmup:
            regime, current = "WARMUP", last.copy()
        else:
            regime = classify(day, prices, ind)
            current = {ticker: value for ticker, value in BASE[regime].items() if ticker in prices.columns}
            current = overlay(day, current, ind)
            current = bear_guard(day, current, prices, ind)
            current = apply_caps(current)
            old = pd.Series(last).reindex(prices.columns).fillna(0.0)
            new = pd.Series(current).reindex(prices.columns).fillna(0.0)
            if (new - old).abs().max() < cfg.turnover_threshold:
                current = last.copy()
        last = normalize(current)
        weights.loc[day, list(last.keys())] = pd.Series(last)
        regimes.loc[day] = regime

    weights = weights.ffill().fillna(0.0).shift(1).fillna(0.0)
    regimes = regimes.ffill()
    return weights, regimes


def performance_metrics(ret: pd.Series, equity: pd.Series, rf: float):
    clean_ret = ret.dropna()
    clean_equity = equity.loc[clean_ret.index]
    years = len(clean_ret) / 252
    cagr = (clean_equity.iloc[-1] / clean_equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = clean_ret.std() * np.sqrt(252)
    sharpe = ((clean_ret.mean() - rf / 252) / clean_ret.std()) * np.sqrt(252) if clean_ret.std() > 0 else np.nan
    downside = clean_ret[clean_ret < 0].std() * np.sqrt(252)
    sortino = ((clean_ret.mean() - rf / 252) * 252 / downside) if downside and not np.isnan(downside) else np.nan
    dd = clean_equity / clean_equity.cummax() - 1
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max_Drawdown": maxdd,
        "Calmar": calmar,
        "Final_Value": clean_equity.iloc[-1],
    }


def annual_returns(ret: pd.Series):
    return (1 + ret).resample("YE").prod() - 1


def monthly_returns(ret: pd.Series):
    return (1 + ret).resample("ME").prod() - 1


def rolling_win_rate(lhs: pd.Series, rhs: pd.Series, days: int):
    lhs_roll = (1 + lhs).rolling(days).apply(np.prod, raw=True) - 1
    rhs_roll = (1 + rhs).rolling(days).apply(np.prod, raw=True) - 1
    joined = pd.concat([lhs_roll, rhs_roll], axis=1).dropna()
    return np.nan if joined.empty else float((joined.iloc[:, 0] > joined.iloc[:, 1]).mean())


def run(cfg: Config):
    prices = load_prices(cfg)
    weights, regimes = generate_weights(prices, cfg)
    ret = prices.pct_change().fillna(0.0)
    gross = (weights * ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turnover * (cfg.trading_cost + cfg.slippage)
    spy = ret[cfg.benchmark]
    fixed_weights = normalize({k: v for k, v in {"SPY": 0.35, "GMOM": 0.30, "RLY": 0.20, "DBMF": 0.15}.items() if k in prices.columns})
    fixed_ret = sum(ret[ticker] * value for ticker, value in fixed_weights.items())
    results = pd.DataFrame(
        {
            "Strategy_Return": net,
            "SPY_Return": spy,
            "Fixed_Fund_Return": fixed_ret,
            "Strategy_Equity": (1 + net).cumprod() * cfg.initial_capital,
            "SPY_Equity": (1 + spy).cumprod() * cfg.initial_capital,
            "Fixed_Fund_Equity": (1 + fixed_ret).cumprod() * cfg.initial_capital,
            "Turnover": turnover,
        }
    )
    return prices, weights, regimes, results


def save(cfg: Config, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    prices, weights, regimes, res = run(cfg)
    perf = pd.DataFrame(
        {
            "Dynamic_Strategy": performance_metrics(res.Strategy_Return, res.Strategy_Equity, cfg.annual_risk_free_rate),
            "SPY": performance_metrics(res.SPY_Return, res.SPY_Equity, cfg.annual_risk_free_rate),
            "Fixed_Fund": performance_metrics(res.Fixed_Fund_Return, res.Fixed_Fund_Equity, cfg.annual_risk_free_rate),
        }
    )
    annual = pd.DataFrame({"Dynamic_Strategy": annual_returns(res.Strategy_Return), "SPY": annual_returns(res.SPY_Return), "Fixed_Fund": annual_returns(res.Fixed_Fund_Return)})
    annual["Dynamic_vs_SPY"] = annual.Dynamic_Strategy - annual.SPY
    annual["Fixed_vs_SPY"] = annual.Fixed_Fund - annual.SPY
    monthly = pd.DataFrame({"Dynamic_Strategy": monthly_returns(res.Strategy_Return), "SPY": monthly_returns(res.SPY_Return), "Fixed_Fund": monthly_returns(res.Fixed_Fund_Return)})
    monthly["Dynamic_vs_SPY"] = monthly.Dynamic_Strategy - monthly.SPY
    wins = pd.DataFrame(
        {
            "Metric": [
                "Calendar Year Win Rate vs SPY",
                "Monthly Win Rate vs SPY",
                "Rolling 1Y Win Rate vs SPY",
                "Rolling 3Y Win Rate vs SPY",
                "Rolling 5Y Win Rate vs SPY",
                "Average Monthly Turnover",
            ],
            "Dynamic_Strategy": [
                (annual.Dynamic_Strategy > annual.SPY).mean(),
                (monthly.Dynamic_Strategy > monthly.SPY).mean(),
                rolling_win_rate(res.Strategy_Return, res.SPY_Return, 252),
                rolling_win_rate(res.Strategy_Return, res.SPY_Return, 252 * 3),
                rolling_win_rate(res.Strategy_Return, res.SPY_Return, 252 * 5),
                res.Turnover.resample("ME").sum().mean(),
            ],
        }
    )

    perf.to_csv(outdir / "performance_summary.csv")
    annual.to_csv(outdir / "annual_returns.csv")
    monthly.to_csv(outdir / "monthly_returns.csv")
    res.to_csv(outdir / "equity_curve.csv")
    weights.to_csv(outdir / "weights.csv")
    regimes.rename("Regime").to_csv(outdir / "regimes.csv")
    wins.to_csv(outdir / "win_stats.csv", index=False)

    plt.figure(figsize=(12, 6))
    for col in ["Strategy_Equity", "SPY_Equity", "Fixed_Fund_Equity"]:
        plt.plot(res.index, res[col], label=col)
    plt.title("Equity Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "equity_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    for col in ["Strategy_Equity", "SPY_Equity", "Fixed_Fund_Equity"]:
        dd = res[col] / res[col].cummax() - 1
        plt.plot(res.index, dd, label=col.replace("_Equity", ""))
    plt.title("Drawdown")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "drawdown.png", dpi=150)
    plt.close()

    return prices, weights, regimes, res, perf

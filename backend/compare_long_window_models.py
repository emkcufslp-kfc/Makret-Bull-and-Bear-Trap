import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "Strategy_Comparisons"
RF = 0.02

sys.path.append(str(REPO_ROOT / "backend" / "strategies"))
import fund_tactical_engine as ftaa
from utils.data_engine import get_clean_master


def annual_returns(ret: pd.Series):
    return (1 + ret).resample("YE").prod() - 1


def monthly_returns(ret: pd.Series):
    return (1 + ret).resample("ME").prod() - 1


def metrics(name: str, equity: pd.Series, benchmark_ret: pd.Series):
    ret = equity.pct_change().dropna()
    benchmark_ret = benchmark_ret.loc[ret.index]
    years = len(ret) / 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * np.sqrt(252)
    sharpe = ((ret.mean() - RF / 252) / ret.std()) * np.sqrt(252) if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std() * np.sqrt(252)
    sortino = ((ret.mean() - RF / 252) * 252 / downside) if downside and not np.isnan(downside) else np.nan
    dd = equity / equity.cummax() - 1
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    yearly = annual_returns(ret)
    monthly = monthly_returns(ret)
    bench_monthly = monthly_returns(benchmark_ret)

    return {
        "Strategy": name,
        "Start": equity.index[0].strftime("%Y-%m-%d"),
        "End": equity.index[-1].strftime("%Y-%m-%d"),
        "Years": round(years, 2),
        "Total Return %": round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
        "CAGR %": round(cagr * 100, 2),
        "Volatility %": round(vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Max Drawdown %": round(maxdd * 100, 2),
        "Calmar": round(calmar, 3),
        "Positive Months %": round((monthly > 0).mean() * 100, 2),
        "Monthly Win vs SPY %": round((monthly > bench_monthly).mean() * 100, 2),
        "Best Year %": round(yearly.max() * 100, 2),
        "Worst Year %": round(yearly.min() * 100, 2),
        "Final Equity": round(equity.iloc[-1], 2),
    }


def parse_ntsx():
    text = (DATA_DIR / "Multi_indicator" / "ntsx_data.js").read_text(encoding="utf-8", errors="replace")
    rows = json.loads(re.search(r"const NTSX_EQUITY\s*=\s*(.*?);\n(?=const|$)", text, re.S).group(1))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    port = df["port"].astype(float)
    spy = df["spy"].astype(float)
    return port / port.iloc[0] * 100000.0, spy / spy.iloc[0] * 100000.0


def build_ftaa_stitched():
    master = get_clean_master()
    proxy = pd.DataFrame(index=master.index)
    proxy["SPY"] = master["SPY"]
    proxy["QQQ"] = master["QQQ"]
    proxy["GMOM"] = master["EFA"]
    proxy["RLY"] = pd.concat([master["TIP"], master["VNQ"], master["GLD"], master["GSG"]], axis=1).mean(axis=1)
    proxy["DBMF"] = master["RYMFX"]
    proxy["SGOV"] = master["BIL"]
    proxy = proxy.loc["2006-01-01":"2020-05-29"].ffill().dropna(how="any")

    proxy_cfg = ftaa.Config(
        tickers=ftaa.CONFIG.tickers,
        benchmark=ftaa.CONFIG.benchmark,
        start_date=str(proxy.index[0].date()),
        end_date=str(proxy.index[-1].date()),
        rebalance_frequency="W-FRI",
        trend_window=ftaa.CONFIG.trend_window,
        momentum_window=ftaa.CONFIG.momentum_window,
        short_momentum_window=ftaa.CONFIG.short_momentum_window,
        trading_cost=ftaa.CONFIG.trading_cost,
        slippage=ftaa.CONFIG.slippage,
        initial_capital=100000.0,
        annual_risk_free_rate=ftaa.CONFIG.annual_risk_free_rate,
        turnover_threshold=ftaa.CONFIG.turnover_threshold,
    )
    proxy_weights, _ = ftaa.generate_weights(proxy, proxy_cfg)
    proxy_ret = proxy.pct_change().fillna(0.0)
    proxy_net = (proxy_weights * proxy_ret).sum(axis=1) - proxy_weights.diff().abs().sum(axis=1).fillna(0.0) * (proxy_cfg.trading_cost + proxy_cfg.slippage)
    proxy_eq = (1 + proxy_net).cumprod() * 100000.0

    _, _, _, actual_res = ftaa.run(ftaa.CONFIG)
    actual_eq = actual_res["Strategy_Equity"] / actual_res["Strategy_Equity"].iloc[0] * proxy_eq.iloc[-1]
    return pd.concat([proxy_eq, actual_eq.loc[actual_eq.index > proxy_eq.index[-1]]])


def build_report():
    ntsx_eq, spy_eq = parse_ntsx()
    plat = pd.read_csv(DATA_DIR / "Platinum_Results" / "Platinum_Equity.csv", index_col=0, parse_dates=True)["Platinum_Equity"]
    plat_eq = plat / plat.iloc[0] * 100000.0
    ftaa_eq = build_ftaa_stitched()

    combined = pd.concat(
        [
            ntsx_eq.rename("NTSX"),
            plat_eq.rename("Platinum Proxy-Aware"),
            ftaa_eq.rename("F-TAA Stitched"),
            spy_eq.rename("SPY"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    bench_ret = combined["SPY"].pct_change().dropna()
    rows = [metrics(col, combined[col], bench_ret) for col in combined.columns]
    report = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_DIR / "long_window_strategy_metrics.csv", index=False)
    combined.to_csv(OUTPUT_DIR / "long_window_equity_curves.csv")
    return combined.index[0], combined.index[-1], report


if __name__ == "__main__":
    start, end, report = build_report()
    print(f"Longest valid common window: {start.date()} -> {end.date()}")
    print(report.to_string(index=False))

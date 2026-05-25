from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "Strategy_Comparisons"
RF = 0.02
INITIAL_CAPITAL = 100000.0
STRATEGIES = ["NTSX", "Platinum Proxy-Aware", "F-TAA Stitched"]


def annual_returns(ret: pd.Series):
    return (1 + ret).resample("YE").prod() - 1


def monthly_returns(ret: pd.Series):
    return (1 + ret).resample("ME").prod() - 1


def calculate_metrics(name: str, equity: pd.Series):
    ret = equity.pct_change().dropna()
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
    return {
        "Portfolio": name,
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
        "Best Year %": round(yearly.max() * 100, 2),
        "Worst Year %": round(yearly.min() * 100, 2),
        "Final Equity": round(equity.iloc[-1], 2),
    }


def simulate_rebalanced_portfolio(returns: pd.DataFrame, weights: np.ndarray, rebalance_freq="ME"):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    rebalance_dates = returns.groupby(pd.Grouper(freq=rebalance_freq)).tail(1).index
    allocations = weights * INITIAL_CAPITAL
    equity = []

    for day, row in returns.iterrows():
        allocations = allocations * (1 + row.values)
        total_value = allocations.sum()
        equity.append(total_value)
        if day in rebalance_dates:
            allocations = weights * total_value

    return pd.Series(equity, index=returns.index, name="Combined_Equity")


def grid_search(returns: pd.DataFrame, step=0.02):
    rows = []
    for w0 in np.arange(0, 1 + step / 2, step):
        for w1 in np.arange(0, 1 - w0 + step / 2, step):
            w2 = 1 - w0 - w1
            if w2 < -1e-9:
                continue
            weights = np.array([w0, w1, max(0.0, w2)])
            equity = simulate_rebalanced_portfolio(returns, weights)
            metrics = calculate_metrics("grid", equity)
            metrics["NTSX %"] = round(weights[0] * 100, 1)
            metrics["Platinum %"] = round(weights[1] * 100, 1)
            metrics["F-TAA %"] = round(weights[2] * 100, 1)
            rows.append(metrics)
    return pd.DataFrame(rows)


def build_report():
    equities = pd.read_csv(DATA_DIR / "long_window_equity_curves.csv", index_col=0, parse_dates=True)
    strategy_equities = equities[STRATEGIES].copy()
    returns = strategy_equities.pct_change().dropna()

    search = grid_search(returns, step=0.02)
    best_cagr = search.sort_values(["CAGR %", "Max Drawdown %"], ascending=[False, False]).iloc[0]
    best_dd = search.sort_values(["Max Drawdown %", "CAGR %"], ascending=[False, False]).iloc[0]
    best_calmar = search.sort_values(["Calmar", "CAGR %"], ascending=[False, False]).iloc[0]
    best_sharpe = search.sort_values(["Sharpe", "CAGR %"], ascending=[False, False]).iloc[0]

    top_calmar = search.sort_values(["Calmar", "CAGR %"], ascending=[False, False]).head(20)
    search.to_csv(DATA_DIR / "strategy_mix_grid_search.csv", index=False)
    top_calmar.to_csv(DATA_DIR / "strategy_mix_top_calmar.csv", index=False)

    chosen_weights = np.array([best_calmar["NTSX %"], best_calmar["Platinum %"], best_calmar["F-TAA %"]]) / 100.0
    chosen_equity = simulate_rebalanced_portfolio(returns, chosen_weights)
    chosen_equity.to_csv(DATA_DIR / "recommended_strategy_mix_equity.csv", header=True)

    summary = pd.DataFrame(
        [
            best_cagr.to_dict(),
            best_dd.to_dict(),
            best_calmar.to_dict(),
            best_sharpe.to_dict(),
        ],
        index=["best_cagr", "best_dd", "best_calmar", "best_sharpe"],
    )
    summary.to_csv(DATA_DIR / "strategy_mix_key_portfolios.csv")

    return {
        "best_cagr": best_cagr,
        "best_dd": best_dd,
        "best_calmar": best_calmar,
        "best_sharpe": best_sharpe,
    }


if __name__ == "__main__":
    result = build_report()
    for key, row in result.items():
        print(f"\n{key.upper()}")
        print(row.to_string())

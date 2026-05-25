import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "Strategy_Comparisons"
RF = 0.02


def _parse_ntsx_series():
    js_path = DATA_DIR / "Multi_indicator" / "ntsx_data.js"
    text = js_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"const NTSX_EQUITY\s*=\s*(.*?);\n(?=const|$)", text, re.S)
    if not match:
        raise RuntimeError("Unable to parse NTSX_EQUITY from ntsx_data.js")
    rows = json.loads(match.group(1))
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["port"].astype(float), df["spy"].astype(float)


def _load_strategy_equities():
    ntsx_port, ntsx_spy = _parse_ntsx_series()

    platinum = pd.read_csv(DATA_DIR / "Platinum_Results" / "Platinum_Equity.csv", index_col=0, parse_dates=True)
    fund = pd.read_csv(DATA_DIR / "Fund_Tactical_Results" / "Fund_Tactical_equity_curve.csv", index_col=0, parse_dates=True)

    series = {
        "NTSX": ntsx_port.rename("NTSX"),
        "Platinum": platinum["Platinum_Equity"].rename("Platinum"),
        "F-TAA Weekly": fund["Strategy_Equity"].rename("F-TAA Weekly"),
        "SPY": ntsx_spy.rename("SPY"),
    }
    combined = pd.concat(series.values(), axis=1, join="inner").dropna()
    combined = combined.apply(lambda col: col / col.iloc[0] * 100000.0)
    return combined


def _annual_returns(ret: pd.Series):
    return (1 + ret).resample("YE").prod() - 1


def _monthly_returns(ret: pd.Series):
    return (1 + ret).resample("ME").prod() - 1


def _metrics(name: str, equity: pd.Series, benchmark_ret: pd.Series):
    ret = equity.pct_change().dropna()
    benchmark_ret = benchmark_ret.loc[ret.index]
    years = len(ret) / 252
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = ret.std() * np.sqrt(252)
    sharpe = ((ret.mean() - RF / 252) / ret.std()) * np.sqrt(252) if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std() * np.sqrt(252)
    sortino = ((ret.mean() - RF / 252) * 252 / downside) if downside and not np.isnan(downside) else np.nan
    dd = equity / equity.cummax() - 1
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    yearly = _annual_returns(ret)
    monthly = _monthly_returns(ret)
    benchmark_monthly = _monthly_returns(benchmark_ret)

    return {
        "Strategy": name,
        "Start": equity.index[0].strftime("%Y-%m-%d"),
        "End": equity.index[-1].strftime("%Y-%m-%d"),
        "Years": round(years, 2),
        "Total Return %": round(total_return * 100, 2),
        "CAGR %": round(cagr * 100, 2),
        "Volatility %": round(vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Max Drawdown %": round(maxdd * 100, 2),
        "Calmar": round(calmar, 3),
        "Positive Months %": round((monthly > 0).mean() * 100, 2),
        "Monthly Win vs SPY %": round((monthly > benchmark_monthly).mean() * 100, 2),
        "Best Year %": round(yearly.max() * 100, 2),
        "Worst Year %": round(yearly.min() * 100, 2),
        "Final Equity": round(equity.iloc[-1], 2),
    }


def build_report():
    combined = _load_strategy_equities()
    common_start = combined.index[0]
    common_end = combined.index[-1]

    benchmark = combined["SPY"]
    benchmark_ret = benchmark.pct_change().dropna()

    rows = []
    for column in ["NTSX", "Platinum", "F-TAA Weekly", "SPY"]:
        rows.append(_metrics(column, combined[column], benchmark_ret))

    report = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_DIR / "common_period_strategy_metrics.csv", index=False)
    combined.to_csv(OUTPUT_DIR / "common_period_equity_curves.csv")
    return common_start, common_end, report


if __name__ == "__main__":
    start, end, report = build_report()
    print(f"Common comparison period: {start.date()} -> {end.date()}")
    print(report.to_string(index=False))

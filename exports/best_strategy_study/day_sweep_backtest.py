"""
Signal-Day × Execution-Lag Sweep — System E / R3 / E80-R3 Blend
================================================================
Tests every weekly schedule combination on 22.5y of real daily data:
  signal day  ∈ {Mon, Tue, Wed, Thu, Fri}   (signal locked at that close)
  exec lag    ∈ {+1, +2, +3 trading days}   (enter at that day's close)
  hold        : until next week's execution (weekly roll)

e.g. "trigger Tue, close next Mon" = signal Tue + lag 3 (exec Fri→ hold to next Fri)
     vs signal Tue + lag 1 (exec Wed, the house standard), etc.

Realistic costs: 0.08% per side × turnover (= 0.16% round trip), T+1+ execution
by construction (exec strictly after signal), no look-ahead.

Inputs (read-only):
  exports/trap_regime_backtest/system_e_nq100_prices.csv   (daily, 2004–2026)
  exports/trap_regime_backtest/raw_data.csv                (daily macro)

Output:
  exports/best_strategy_study/day_sweep_results.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRICES_CSV = ROOT / "exports" / "trap_regime_backtest" / "system_e_nq100_prices.csv"
RAW_CSV = ROOT / "exports" / "trap_regime_backtest" / "raw_data.csv"

COST_SIDE = 0.0008          # 0.08% per side => 0.16% round trip
RF = 0.02
DAYS = ["MON", "TUE", "WED", "THU", "FRI"]
DAY_LABEL = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

BULL_SPY, BULL_STK = 0.35, 0.217
BEAR_SPY, BEAR_IEF = 0.10, 0.90
BREADTH_MIN = 0.40


def metrics(eq: pd.Series, name: str) -> dict:
    ret = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(52)
    dn = ret[ret < 0].std() * np.sqrt(52)
    dd = (eq / eq.cummax() - 1).min()
    yearly = (1 + ret).groupby(ret.index.year).prod() - 1
    return {
        "Combo": name, "CAGR_%": round(cagr * 100, 2), "Max_DD_%": round(dd * 100, 2),
        "Calmar": round(cagr / abs(dd), 3) if dd else np.nan,
        "Sharpe": round((ret.mean() * 52 - RF) / vol, 3) if vol else np.nan,
        "Sortino": round((ret.mean() * 52 - RF) / dn, 3) if dn else np.nan,
        "Vol_%": round(vol * 100, 1), "Win_Rate_%": round((ret > 0).mean() * 100, 1),
        "Best_Year_%": round(yearly.max() * 100, 1), "Worst_Year_%": round(yearly.min() * 100, 1),
        "Final_$": round(float(eq.iloc[-1] / eq.iloc[0] * 100000)),
    }


def load_inputs():
    px = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True).sort_index().ffill()
    raw = pd.read_csv(RAW_CSV, index_col=0, parse_dates=True).sort_index().ffill()
    if "SPY" not in px.columns:
        px["SPY"] = raw["SPY"]
    if "IEF" not in px.columns:
        px["IEF"] = raw["IEF"]
    return px, raw


def r3_daily_scores(raw: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Vectorized daily Bull Trap (0-10) and Bear Trap (0-1) scores (pages/2+3 logic)."""
    curve = raw["^TNX"] - raw["^IRX"]
    curve_prev = curve.shift(22)
    ys = np.where((curve > 0) & (curve_prev < 0), 1.0, np.where(curve > 0, 0.5, 0.0))

    vix_ma22 = raw["^VIX"].rolling(22).mean()
    vs = np.where(raw["^VIX"] < 15, 1.0, np.where(raw["^VIX"] < vix_ma22, 0.5, 0.0))

    hyg_ief = raw["HYG"] / raw["IEF"]
    cs = (hyg_ief > hyg_ief.rolling(22).mean()).astype(float)

    spy_ma200 = raw["SPY"].rolling(200).mean()
    bs = np.where(raw["SPY"] > spy_ma200 * 1.05, 1.0, np.where(raw["SPY"] > spy_ma200, 0.5, 0.0))

    mom22 = raw["SPY"] / raw["SPY"].shift(22) - 1
    acc = np.where(mom22 > 0.02, 1.0, np.where(mom22 > 0, 0.5, 0.0))

    tip = raw["TIP"] / raw["TIP"].shift(22) - 1
    liq = (tip > 0).astype(float)

    bull = pd.Series(np.minimum(10.0, ys + vs + cs.values + bs + acc + liq.values + 2.0),
                     index=raw.index)

    def norm(val, lower, upper, inverted=False):
        if inverted:
            return ((lower - val) / (lower - upper)).clip(0, 1)
        return ((val - lower) / (upper - lower)).clip(0, 1)

    irx_ma120 = raw["^IRX"].rolling(120).mean()
    hi_ma252 = hyg_ief.rolling(252).mean()
    bear = (
        norm(curve, 1.5, -0.5, True) * 0.25
        + norm(raw["^IRX"], irx_ma120 * 0.8, irx_ma120 * 1.2) * 0.20
        + norm(hyg_ief, hi_ma252 * 1.05, hi_ma252 * 0.90, True) * 0.20
        + norm(raw["SPY"], spy_ma200 * 1.05, spy_ma200 * 0.95, True) * 0.15
        + norm(raw["^VIX"], 15, 35) * 0.10
        + 0.65 * 0.05 + 0.50 * 0.05
    )
    return bull, bear


def run_schedule(px: pd.DataFrame, raw: pd.DataFrame, sd: int, lag: int,
                 bull: pd.Series, bear: pd.Series):
    """Return (eq_sysE, eq_R3, eq_blend) for signal day sd (0=Mon) and exec lag."""
    stocks = [c for c in px.columns if c not in ("SPY", "IEF")]
    w = px.resample(f"W-{DAYS[sd]}").last().dropna(how="all")
    # keep only weeks whose stamp exists in daily index (skip holidays gracefully)
    didx = px.index
    pos = didx.searchsorted(w.index, side="right") - 1
    ok = pos >= 0
    w, pos = w.iloc[ok], pos[ok]

    # Momentum on this weekly grid (44w - 4w skip-month)
    stk_w = w[stocks]
    mom = stk_w / stk_w.shift(44) - 1 - (stk_w / stk_w.shift(4) - 1)

    # Daily gates
    spy_ma = px["SPY"].rolling(200).mean()
    nq_ew = px[stocks].mean(axis=1)
    nq_ma = nq_ew.rolling(200).mean()

    # R3 weekly vol (this grid)
    spy_w_ret = w["SPY"].pct_change()
    vol4 = spy_w_ret.rolling(4).std() * np.sqrt(52)

    P = px.to_numpy()
    col = {c: i for i, c in enumerate(px.columns)}
    n_days = len(didx)

    eqs = {"E": [1.0], "R": [1.0], "B": [1.0]}
    prev_w = {"E": {}, "R": {}, "B": {}}
    dates_out = [w.index[45]]

    for t in range(45, len(w) - 1):
        sig_i = pos[t]
        exec_i = sig_i + lag
        next_exec_i = pos[t + 1] + lag
        if next_exec_i >= n_days or exec_i >= n_days:
            break

        row = mom.iloc[t].dropna()
        breadth = float((row > 0).mean()) if len(row) else 0.0
        top3 = list(row.nlargest(3).index) if len(row) >= 5 else []
        d_sig = didx[sig_i]
        bull_gate = (px["SPY"].iloc[sig_i] > spy_ma.iloc[sig_i]
                     and nq_ew.iloc[sig_i] > nq_ma.iloc[sig_i])

        # System E weights
        if bull_gate and breadth >= BREADTH_MIN and len(top3) == 3:
            we = {"SPY": BULL_SPY, **{s: BULL_STK for s in top3}}
        else:
            we = {"SPY": BEAR_SPY, "IEF": BEAR_IEF}

        # R3 weights
        bt = float(bull.asof(d_sig)) if not np.isnan(bull.asof(d_sig)) else 2.5
        br = float(bear.asof(d_sig)) if not np.isnan(bear.asof(d_sig)) else 0.5
        v = vol4.iloc[t]
        vs_ = (v / 0.12) if pd.notna(v) and v > 0 else 1.0
        net = (bt / 10.0) * (1.0 - br)
        deploy = float(np.clip(0.10 + net * min(1.5, 1.0 / vs_) * 0.90, 0.10, 1.00))
        if breadth >= BREADTH_MIN and len(top3) == 3:
            wr = {"SPY": BULL_SPY * deploy, **{s: BULL_STK * deploy for s in top3}}
        else:
            wr = {"SPY": deploy}
        wr["IEF"] = wr.get("IEF", 0.0) + (1.0 - deploy)

        # Blend 80/20
        wb = {}
        for k, v_ in we.items():
            wb[k] = wb.get(k, 0.0) + 0.8 * v_
        for k, v_ in wr.items():
            wb[k] = wb.get(k, 0.0) + 0.2 * v_

        for key, wt in (("E", we), ("R", wr), ("B", wb)):
            growth = 0.0
            for tkr, x in wt.items():
                j = col[tkr]
                p0, p1 = P[exec_i, j], P[next_exec_i, j]
                growth += x * (p1 / p0 if p0 > 0 and not np.isnan(p0) and not np.isnan(p1) else 1.0)
            turnover = sum(abs(wt.get(k, 0.0) - prev_w[key].get(k, 0.0))
                           for k in set(wt) | set(prev_w[key]))
            eqs[key].append(eqs[key][-1] * growth * (1 - COST_SIDE * turnover))
            prev_w[key] = wt
        dates_out.append(w.index[t + 1])

    idx = pd.DatetimeIndex(dates_out)
    return (pd.Series(eqs["E"], index=idx), pd.Series(eqs["R"], index=idx),
            pd.Series(eqs["B"], index=idx))


def main() -> None:
    px, raw = load_inputs()
    bull, bear = r3_daily_scores(raw)

    rows = []
    for sd in range(5):
        for lag in (1, 2, 3):
            eq_e, eq_r, eq_b = run_schedule(px, raw, sd, lag, bull, bear)
            sched = f"sig {DAY_LABEL[sd]} close, exec +{lag}d"
            for name, eq in [(f"System E | {sched}", eq_e),
                             (f"R3 Vol-Adj | {sched}", eq_r),
                             (f"E80/R3 Blend | {sched}", eq_b)]:
                m = metrics(eq, name)
                m.update({"Strategy": name.split(" | ")[0], "Signal_Day": DAY_LABEL[sd],
                          "Exec_Lag_d": lag})
                rows.append(m)
            print(f"done {sched}", flush=True)

    # SPY baseline
    spy = px["SPY"].resample("W-FRI").last().dropna()
    m = metrics(spy, "SPY Buy & Hold")
    m.update({"Strategy": "SPY", "Signal_Day": "—", "Exec_Lag_d": 0})
    rows.append(m)

    df = pd.DataFrame(rows).sort_values("Calmar", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    df.to_csv(OUT / "day_sweep_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()

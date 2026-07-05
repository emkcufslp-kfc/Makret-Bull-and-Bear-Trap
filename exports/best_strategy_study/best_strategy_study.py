"""
Best Strategy Study — Calmar-Optimal Combination
=================================================
Reads EXISTING equity curves and trap scores (nothing recomputed, nothing
overwritten) and searches for the combination with the best CAGR/MaxDD
trade-off that beats SPY on both axes.

Inputs (read-only):
  exports/trap_regime_backtest/risk_allocation/risk_allocation_equity_curves.csv
  exports/trap_regime_backtest/system_e_weekly_signals.csv
  data/Strategy_Comparisons/long_window_equity_curves.csv

Candidates:
  A. Static weekly-rebalanced blends of Sys E / R3 / R7  (2004–2026, 22.5y)
  B. Static blends adding Platinum Proxy-Aware           (2007–2026, 18.5y)
  C. Regime-switched: Sys E <-> defensive sleeve, weight = f(Bear Trap score)
  D. Vol-target overlay (15% ann.) on the Calmar-best of A–C

Costs: 5 bps per unit of sleeve turnover (house standard).
No look-ahead: regime weights use the PRIOR week's Bear Trap score.

Outputs → exports/best_strategy_study/
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RA_CSV = ROOT / "exports" / "trap_regime_backtest" / "risk_allocation" / "risk_allocation_equity_curves.csv"
SIG_CSV = ROOT / "exports" / "trap_regime_backtest" / "system_e_weekly_signals.csv"
LW_CSV = ROOT / "data" / "Strategy_Comparisons" / "long_window_equity_curves.csv"

RF = 0.02
COST = 0.0005          # 5 bps per unit turnover
INIT = 100_000.0


# ── Metrics ────────────────────────────────────────────────────────────────────
def metrics(eq: pd.Series, spy_ret: pd.Series, name: str) -> dict:
    eq = eq.dropna()
    ret = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(52)
    sharpe = (ret.mean() * 52 - RF) / vol if vol > 0 else np.nan
    dn = ret[ret < 0].std() * np.sqrt(52)
    sortino = (ret.mean() * 52 - RF) / dn if dn > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd != 0 else np.nan
    s = spy_ret.reindex(ret.index).fillna(0)
    beta = ret.cov(s) / s.var() if s.var() > 0 else np.nan
    alpha = (ret.mean() - beta * s.mean()) * 52
    yearly = (1 + ret).groupby(ret.index.year).prod() - 1
    return {
        "Strategy": name, "Years": round(years, 1),
        "CAGR_%": round(cagr * 100, 2), "Max_DD_%": round(dd * 100, 2),
        "Calmar": round(calmar, 3), "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3), "Vol_%": round(vol * 100, 1),
        "Beta": round(beta, 3), "Alpha_%": round(alpha * 100, 2),
        "Best_Year_%": round(yearly.max() * 100, 1),
        "Worst_Year_%": round(yearly.min() * 100, 1),
        "Final_$": round(float(eq.iloc[-1] / eq.iloc[0] * INIT)),
    }


def wlabel(weights: dict[str, float]) -> str:
    parts = []
    for k, v in weights.items():
        if v > 0:
            parts.append(k.split("_")[0] + str(int(round(v * 100))))
    return "/".join(parts)


def blend_equity(rets: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weekly-rebalanced blend, vectorized, with turnover cost."""
    cols = list(weights.keys())
    R = rets[cols].fillna(0.0).to_numpy()
    w = np.array([weights[c] for c in cols])
    rp = R @ w
    growth = 1.0 + rp
    growth[growth == 0] = 1e-9
    drift = (w * (1.0 + R)) / growth[:, None]
    turnover = np.abs(drift - w).sum(axis=1)
    net = rp - COST * turnover
    eq = np.cumprod(1.0 + net)
    eq[0] = 1.0
    return pd.Series(eq, index=rets.index)


def grid_search(rets: pd.DataFrame, sleeves: list[str], spy_ret: pd.Series,
                step: float = 0.05) -> pd.DataFrame:
    n = len(sleeves)
    ticks = int(round(1 / step))
    rows = []
    for combo in product(range(ticks + 1), repeat=n - 1):
        if sum(combo) > ticks:
            continue
        ws = [c * step for c in combo] + [1 - sum(combo) * step]
        weights = dict(zip(sleeves, ws))
        eq = blend_equity(rets, weights)
        m = metrics(eq, spy_ret, "grid")
        m.update({f"w_{s}": round(w * 100) for s, w in weights.items()})
        rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Load (read-only) ──────────────────────────────────────────────────────
    ra = pd.read_csv(RA_CSV, index_col="Date", parse_dates=True).sort_index()
    sig = pd.read_csv(SIG_CSV, index_col="Date", parse_dates=True).sort_index()
    lw = pd.read_csv(LW_CSV, index_col=0, parse_dates=True).sort_index()

    ra_ret = ra.pct_change().fillna(0.0)
    spy_ret = ra_ret["Buy_Hold_SPY"]

    results: list[dict] = []
    curves: dict[str, pd.Series] = {}

    # Baselines
    for col, label in [("Buy_Hold_SPY", "SPY Buy & Hold"),
                       ("Sys_E_Original", "System E Original"),
                       ("R3_Vol_Adj", "R3 Vol-Adjusted"),
                       ("R7_Full_Kelly", "R7 Full Kelly")]:
        results.append(metrics(ra[col], spy_ret, label))
        curves[label] = ra[col] / ra[col].iloc[0]

    # ── A. Static blends Sys E / R3 / R7 (full 2004–2026 window) ─────────────
    sleeves_a = ["Sys_E_Original", "R3_Vol_Adj", "R7_Full_Kelly"]
    grid_a = grid_search(ra_ret, sleeves_a, spy_ret)
    grid_a.to_csv(OUT / "grid_A_sysE_R3_R7.csv", index=False)
    best_a = grid_a.loc[grid_a["Calmar"].idxmax()].to_dict()
    wa = {s: best_a[f"w_{s}"] / 100 for s in sleeves_a}
    eq_a = blend_equity(ra_ret, wa) * INIT / INIT
    ma = metrics(eq_a, spy_ret, "A: Calmar blend " + wlabel(wa))
    results.append(ma)
    curves[ma["Strategy"]] = eq_a

    # ── B. Add Platinum (common window 2007-12 → 2026-05) ────────────────────
    plat_w = lw["Platinum Proxy-Aware"].resample("W-FRI").last().dropna()
    common = ra.index.intersection(plat_w.index)
    rb = ra.loc[common].pct_change().fillna(0.0)
    rb["Platinum"] = plat_w.reindex(common).pct_change().fillna(0.0)
    spy_b = rb["Buy_Hold_SPY"]
    sleeves_b = ["Sys_E_Original", "R7_Full_Kelly", "Platinum"]
    grid_b = grid_search(rb, sleeves_b, spy_b)
    grid_b.to_csv(OUT / "grid_B_sysE_R7_platinum.csv", index=False)
    best_b = grid_b.loc[grid_b["Calmar"].idxmax()].to_dict()
    wb = {s: best_b[f"w_{s}"] / 100 for s in sleeves_b}
    eq_b = blend_equity(rb, wb)
    mb = metrics(eq_b, spy_b, "B: +Platinum blend " + wlabel(wb))
    results.append(mb)

    # ── C. Regime-switched Sys E <-> R7 by Bear Trap score (no look-ahead) ───
    bear = sig["BearTrap_Score"].reindex(ra.index).ffill().fillna(0.5)
    w_off = (1.0 - bear).clip(0.0, 1.0).shift(1).fillna(0.5)   # prior week's score
    r_off = ra_ret["Sys_E_Original"]
    r_def = ra_ret["R7_Full_Kelly"]
    port_ret = w_off * r_off + (1 - w_off) * r_def
    turnover = w_off.diff().abs().fillna(0.0) * 2
    port_ret -= COST * turnover
    eq_c = (1 + port_ret).cumprod()
    mc = metrics(eq_c, spy_ret, "C: Regime-switch SysE<->R7 (BearScore)")
    results.append(mc)
    curves[mc["Strategy"]] = eq_c

    # ── D. Vol-target overlay (15%) on best of A–C ────────────────────────────
    cands = {ma["Strategy"]: eq_a, mb["Strategy"]: eq_b, mc["Strategy"]: eq_c}
    best_name = max((m for m in results if m["Strategy"] in cands), key=lambda m: m["Calmar"])["Strategy"]
    base_eq = cands[best_name]
    base_ret = base_eq.pct_change().fillna(0.0)
    realized = base_ret.rolling(8).std() * np.sqrt(52)
    expo = (0.15 / realized).clip(upper=1.0).shift(1).fillna(1.0)
    d_ret = expo * base_ret - COST * expo.diff().abs().fillna(0.0)
    eq_d = (1 + d_ret).cumprod()
    spy_for_d = spy_ret if base_eq.index.equals(ra.index) else rb["Buy_Hold_SPY"]
    md = metrics(eq_d, spy_for_d, f"D: 15% vol-target on [{best_name[:1]}]")
    results.append(md)
    curves[md["Strategy"]] = eq_d

    # ── Save & report ─────────────────────────────────────────────────────────
    perf = pd.DataFrame(results)
    perf.to_csv(OUT / "performance_summary.csv", index=False)
    pd.DataFrame({k: v for k, v in curves.items()}).to_csv(OUT / "equity_curves.csv")

    winner = max(results, key=lambda m: m["Calmar"])
    (OUT / "winner.json").write_text(json.dumps(winner, indent=2))

    pd.set_option("display.width", 250)
    print(perf.to_string(index=False))
    print("\nWINNER (Calmar-optimal):", winner["Strategy"])


if __name__ == "__main__":
    main()

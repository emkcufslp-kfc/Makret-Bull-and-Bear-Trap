"""
Hybrid Regime-Switching Backtest: Strategy A (Bull) <-> Strategy D (Bear)
=========================================================================
Goal:
  - Bull + low-vol  -> Strategy A (80% Sys_E / 20% R3, static blend)
  - Bear / high-vol -> Strategy D (15% annualised vol-target on A's returns)

Regime signals tested (all shifted 1 week to avoid look-ahead):
  1. BearTrap_Score         (0-1 weekly score, system_e_weekly_signals.csv)
  2. VIX                    (from system_e_weekly_signals.csv)
  3. composite_risk_score   (0-100, weekly_feature_outcomes.csv)
  4. composite_risk_state   (Normal/Watch/Warning/Stress)
  5. liquidity_score        (0-100, higher = more stressed)
  6. d1_market_regime_score (0-100 bear probability composite)
  7. d2_score               (0-14 technical deterioration)
  8. Realized vol of A      (trailing 8-week annualised, endogenous)
  9. Composite signals      (combinations / majority vote)

Costs: 5 bps per unit turnover (house standard).
No look-ahead: all signals use prior-week values (shift(1)).
Window: 2004-01-02 to 2026-07-03

Outputs -> exports/best_strategy_study/
  hybrid_regime_backtest.csv   - weekly equity curves for all variants
  hybrid_results.json          - performance metrics comparison table
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent

RA_CSV  = ROOT / "exports" / "trap_regime_backtest" / "risk_allocation" / "risk_allocation_equity_curves.csv"
SIG_CSV = ROOT / "exports" / "trap_regime_backtest" / "system_e_weekly_signals.csv"
WFO_CSV = ROOT / "exports" / "crash_predictor_study" / "weekly_feature_outcomes.csv"

RF         = 0.02
COST       = 0.0005    # 5 bps per unit turnover
VOL_TARGET = 0.15
INIT       = 100_000.0

# ── Performance metrics ────────────────────────────────────────────────────────
def metrics(eq: pd.Series, spy_ret: pd.Series, name: str) -> dict:
    eq = eq.dropna()
    ret = eq.pct_change().dropna()
    years  = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr   = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    vol    = ret.std() * np.sqrt(52)
    sharpe = (ret.mean() * 52 - RF) / vol if vol > 0 else np.nan
    dn     = ret[ret < 0].std() * np.sqrt(52)
    sortino = (ret.mean() * 52 - RF) / dn if dn > 0 else np.nan
    dd     = (eq / eq.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd != 0 else np.nan
    s      = spy_ret.reindex(ret.index).fillna(0)
    beta   = ret.cov(s) / s.var() if s.var() > 0 else np.nan
    alpha  = (ret.mean() - beta * s.mean()) * 52
    yearly = (1 + ret).groupby(ret.index.year).prod() - 1
    return {
        "Strategy":     name,
        "Years":        round(years,   1),
        "CAGR_%":       round(cagr  * 100, 2),
        "Max_DD_%":     round(dd    * 100, 2),
        "Calmar":       round(calmar,   3),
        "Sharpe":       round(sharpe,   3),
        "Sortino":      round(sortino,  3),
        "Vol_%":        round(vol   * 100, 1),
        "Beta":         round(beta,     3),
        "Alpha_%":      round(alpha * 100, 2),
        "Best_Year_%":  round(yearly.max() * 100, 1),
        "Worst_Year_%": round(yearly.min() * 100, 1),
        "Final_$":      round(float(eq.iloc[-1] / eq.iloc[0] * INIT)),
    }


# ── Strategy A: static blend 80% SysE / 20% R3 ────────────────────────────────
def build_strategy_a(ra_ret: pd.DataFrame) -> pd.Series:
    W_SYSE = 0.80
    W_R3   = 0.20
    r_syse = ra_ret["Sys_E_Original"]
    r_r3   = ra_ret["R3_Vol_Adj"]
    rp = W_SYSE * r_syse + W_R3 * r_r3
    w  = np.array([W_SYSE, W_R3])
    R  = ra_ret[["Sys_E_Original", "R3_Vol_Adj"]].to_numpy()
    drift    = (w * (1 + R)) / (1 + rp.to_numpy())[:, None]
    turnover = np.abs(drift - w).sum(axis=1)
    net = rp - COST * turnover
    return (1 + net).cumprod()


# ── Strategy D: 15% vol-target on A ───────────────────────────────────────────
def build_strategy_d(a_ret: pd.Series) -> tuple[pd.Series, pd.Series]:
    realized = a_ret.rolling(8).std() * np.sqrt(52)
    expo     = (VOL_TARGET / realized).clip(upper=1.0).shift(1).fillna(1.0)
    d_ret    = expo * a_ret - COST * expo.diff().abs().fillna(0.0)
    return (1 + d_ret).cumprod(), expo


# ── Hybrid: switch between A (bull) and D (bear) ──────────────────────────────
def build_hybrid(a_ret: pd.Series, expo_d: pd.Series,
                 bear_flag: pd.Series, name: str,
                 spy_ret: pd.Series) -> tuple[dict, pd.Series]:
    """
    bear_flag: boolean Series, already shifted by 1 week.
    Bull weeks -> full exposure to A (expo=1.0).
    Bear weeks -> vol-targeted exposure from expo_d.
    """
    bear   = bear_flag.reindex(a_ret.index).fillna(False)
    e_bull = pd.Series(1.0, index=a_ret.index)
    e_bear = expo_d.reindex(a_ret.index).fillna(1.0)
    expo_h = e_bear.where(bear, e_bull)

    expo_change = expo_h.diff().abs().fillna(0.0)
    h_ret = expo_h * a_ret - COST * expo_change
    eq_h  = (1 + h_ret).cumprod()

    pct_bear = bear.mean()
    n_switch = int((bear.astype(int).diff().abs().fillna(0) > 0).sum())

    m = metrics(eq_h, spy_ret, name)
    m["Pct_Bear_%"] = round(pct_bear * 100, 1)
    m["N_Switches"]  = n_switch
    return m, eq_h


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    ra  = pd.read_csv(RA_CSV,  index_col="Date", parse_dates=True).sort_index()
    sig = pd.read_csv(SIG_CSV, index_col="Date", parse_dates=True).sort_index()
    wfo = pd.read_csv(WFO_CSV, index_col="date", parse_dates=True).sort_index()

    ra_ret  = ra.pct_change().fillna(0.0)
    spy_ret = ra_ret["Buy_Hold_SPY"]

    # Align WFO (Thursday) -> RA (Friday) by shifting index +1 day, then ffill
    wfo.index = wfo.index + pd.Timedelta(days=1)
    wfo_a = wfo.reindex(ra.index, method="ffill")

    # ── Base strategies ────────────────────────────────────────────────────────
    eq_a  = build_strategy_a(ra_ret)
    a_ret = eq_a.pct_change().fillna(0.0)
    eq_d, expo_d = build_strategy_d(a_ret)

    results = []
    curves  = {}

    # Baselines (no bear-regime fields)
    for col, label in [("Buy_Hold_SPY", "SPY Buy & Hold"),
                       ("Sys_E_Original", "System E Original")]:
        m = metrics(ra[col] / ra[col].iloc[0], spy_ret, label)
        m["Pct_Bear_%"] = None
        m["N_Switches"]  = None
        results.append(m)
        curves[label] = ra[col] / ra[col].iloc[0]

    m_a = metrics(eq_a, spy_ret, "A: 80%SysE/20%R3 static")
    m_a["Pct_Bear_%"] = 0.0
    m_a["N_Switches"]  = 0
    results.append(m_a)
    curves["A: 80%SysE/20%R3 static"] = eq_a

    m_d = metrics(eq_d, spy_ret, "D: 15%vol-target on A (always)")
    m_d["Pct_Bear_%"] = 100.0
    m_d["N_Switches"]  = 0
    results.append(m_d)
    curves["D: 15%vol-target on A (always)"] = eq_d

    # ── Indicator 1: BearTrap_Score ────────────────────────────────────────────
    bear_score = sig["BearTrap_Score"].reindex(ra.index).ffill().fillna(0.0)
    for thresh in [0.35, 0.40, 0.45, 0.50]:
        flag = bear_score.shift(1) >= thresh
        name = f"H1_BearTrap>={thresh:.2f}"
        m, eq = build_hybrid(a_ret, expo_d, flag, name, spy_ret)
        results.append(m); curves[name] = eq

    # ── Indicator 2: VIX ──────────────────────────────────────────────────────
    vix = sig["VIX"].reindex(ra.index).ffill().fillna(20.0)
    for thresh in [20, 22, 25, 28, 32]:
        flag = vix.shift(1) >= thresh
        name = f"H2_VIX>={thresh}"
        m, eq = build_hybrid(a_ret, expo_d, flag, name, spy_ret)
        results.append(m); curves[name] = eq

    # ── Indicator 3: composite_risk_score (D3 liquidity) ─────────────────────
    crs = wfo_a["composite_risk_score"].fillna(0.0)
    for thresh in [30, 40, 50, 60, 70]:
        flag = crs.shift(1) >= thresh
        name = f"H3_CompRisk>={thresh}"
        m, eq = build_hybrid(a_ret, expo_d, flag, name, spy_ret)
        results.append(m); curves[name] = eq

    # ── Indicator 4: composite_risk_state ─────────────────────────────────────
    crs_state = wfo_a["composite_risk_state"].fillna("Normal")
    for states, label_s in [
        (["Stress"],                    "Stress_only"),
        (["Warning", "Stress"],         "Warn+Stress"),
        (["Watch", "Warning", "Stress"],"Watch+Warn+Stress"),
    ]:
        # shift(1) on categorical: map to bool first
        flag = crs_state.isin(states).shift(1).fillna(False).astype(bool)
        name = f"H4_State={label_s}"
        m, eq = build_hybrid(a_ret, expo_d, flag, name, spy_ret)
        results.append(m); curves[name] = eq

    # ── Indicator 5: liquidity_score (higher = more stress) ───────────────────
    liq = wfo_a["liquidity_score"].fillna(0.0)
    for thresh in [30, 40, 50, 60, 70]:
        flag = liq.shift(1) >= thresh
        name = f"H5_Liq>={thresh}"
        m, eq = build_hybrid(a_ret, expo_d, flag, name, spy_ret)
        results.append(m); curves[name] = eq

    # ── Indicator 6: d1_market_regime_score (bear probability 0-100) ──────────
    d1 = wfo_a["d1_market_regime_score"].fillna(0.0)
    for thresh in [20, 30, 40, 50, 60]:
        flag = d1.shift(1) >= thresh
        name = f"H6_D1>={thresh}"
        m, eq = build_hybrid(a_ret, expo_d, flag, name, spy_ret)
        results.append(m); curves[name] = eq

    # ── Indicator 7: d2_score (technical deterioration, 0-14) ─────────────────
    d2 = wfo_a["d2_score"].fillna(0.0)
    for thresh in [7, 9, 11, 13]:
        flag = d2.shift(1) >= thresh
        name = f"H7_D2>={thresh}"
        m, eq = build_hybrid(a_ret, expo_d, flag, name, spy_ret)
        results.append(m); curves[name] = eq

    # ── Indicator 8: Realized vol of Strategy A (endogenous) ──────────────────
    a_rvol = a_ret.rolling(8).std() * np.sqrt(52)
    for thresh in [0.15, 0.18, 0.20, 0.25, 0.30]:
        flag = a_rvol.shift(1) >= thresh
        name = f"H8_Avol>={int(thresh*100)}pct"
        m, eq = build_hybrid(a_ret, expo_d, flag, name, spy_ret)
        results.append(m); curves[name] = eq

    spy_rvol = spy_ret.rolling(8).std() * np.sqrt(52)
    for thresh in [0.15, 0.18, 0.20, 0.25]:
        flag = spy_rvol.shift(1) >= thresh
        name = f"H8b_SPYvol>={int(thresh*100)}pct"
        m, eq = build_hybrid(a_ret, expo_d, flag, name, spy_ret)
        results.append(m); curves[name] = eq

    # ── Indicator 9: Composite (majority-vote combinations) ───────────────────
    sub = pd.DataFrame(index=ra.index)
    sub["beartrap"]  = bear_score.shift(1) >= 0.40
    sub["vix"]       = vix.shift(1) >= 22
    sub["comp_risk"] = crs.shift(1) >= 50
    sub["liq"]       = liq.shift(1) >= 40
    sub["d1"]        = d1.shift(1) >= 30
    sub["d2"]        = d2.shift(1) >= 9
    sub["avol"]      = a_rvol.shift(1) >= 0.18

    for k in [2, 3, 4]:
        flag = (sub.sum(axis=1) >= k)
        name = f"H9_Composite>={k}of7"
        m, eq = build_hybrid(a_ret, expo_d, flag, name, spy_ret)
        results.append(m); curves[name] = eq

    # BearTrap OR (VIX AND CompRisk)
    flag = sub["beartrap"] | (sub["vix"] & sub["comp_risk"])
    m, eq = build_hybrid(a_ret, expo_d, flag, "H9b_BearTrap|VIX&Risk", spy_ret)
    results.append(m); curves["H9b_BearTrap|VIX&Risk"] = eq

    # BearTrap AND (VIX OR CompRisk)  -- narrower trigger
    flag = sub["beartrap"] & (sub["vix"] | sub["comp_risk"])
    m, eq = build_hybrid(a_ret, expo_d, flag, "H9c_BearTrap&(VIX|Risk)", spy_ret)
    results.append(m); curves["H9c_BearTrap&(VIX|Risk)"] = eq

    # BearTrap AND VIX AND (CompRisk OR Liq)
    flag = sub["beartrap"] & sub["vix"] & (sub["comp_risk"] | sub["liq"])
    m, eq = build_hybrid(a_ret, expo_d, flag, "H9d_BearTrap&VIX&(Risk|Liq)", spy_ret)
    results.append(m); curves["H9d_BearTrap&VIX&(Risk|Liq)"] = eq

    # ── Save outputs ───────────────────────────────────────────────────────────
    ec_df = pd.DataFrame(curves)
    ec_df.to_csv(OUT / "hybrid_regime_backtest.csv")

    perf = pd.DataFrame(results).sort_values("Calmar", ascending=False).reset_index(drop=True)

    hybrids = [r for r in results if r["Strategy"].startswith("H")]
    hybrids_sorted = sorted(hybrids, key=lambda x: x["Calmar"], reverse=True)

    summary = {
        "generated": pd.Timestamp.now().isoformat(),
        "description": "Hybrid A/D regime-switching -- all candidate indicators tested",
        "baselines": {
            r["Strategy"]: {k: v for k, v in r.items() if k != "Strategy"}
            for r in results if not r["Strategy"].startswith("H")
        },
        "best_hybrid": hybrids_sorted[0] if hybrids_sorted else {},
        "top5_hybrids": hybrids_sorted[:5],
        "all_hybrids_ranked": hybrids_sorted,
    }
    (OUT / "hybrid_results.json").write_text(json.dumps(summary, indent=2, default=str))

    # ── Print report ───────────────────────────────────────────────────────────
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 120)

    SEP = "=" * 100
    print(f"\n{SEP}")
    print("HYBRID REGIME-SWITCHING BACKTEST  |  A (Bull) <-> D (Bear)  |  2004-2026")
    print(SEP)

    base_df   = perf[~perf["Strategy"].str.startswith("H")]
    hybrid_df = perf[perf["Strategy"].str.startswith("H")]

    cols = ["Strategy", "CAGR_%", "Max_DD_%", "Calmar", "Sharpe",
            "Sortino", "Vol_%", "Pct_Bear_%", "N_Switches"]

    print("\n-- BASELINES --")
    print(base_df[cols].to_string(index=False))

    print("\n-- TOP-25 HYBRID VARIANTS (sorted by Calmar) --")
    print(hybrid_df.head(25)[cols].to_string(index=False))

    print(f"\n-- BEST HYBRID: {hybrids_sorted[0]['Strategy']} --")
    best = hybrids_sorted[0]
    for k, v in best.items():
        print(f"  {k:22s}: {v}")

    print(f"\nSaved to {OUT}/")
    print("  hybrid_regime_backtest.csv")
    print("  hybrid_results.json")


if __name__ == "__main__":
    main()

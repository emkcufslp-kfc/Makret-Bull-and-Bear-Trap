"""
backend/strategies/blend_e_r3_engine.py
========================================
E80/R3 Blend — Calmar-optimal combination found by exports/best_strategy_study/
  80% System E (NQ100 top-3 momentum, dual 200d MA + breadth gate)
  20% R3 Vol-Adjusted (continuous trap-score + vol sizing)

Realistic backtest assumptions
------------------------------
- Underlying sleeve equity curves already embed T+1 execution
  (Sys E: Tue signal → Wed open; R3: Thu signal → Fri open) and the house
  0.16% round-trip cost per position change (see README).
- The blend layer rebalances back to 80/20 weekly and pays an additional
  BLEND_COST (10 bps) per unit of turnover — covering commission + spread
  on the drift-correction trades.
- No look-ahead: weights are corrected using the week's realized drift only.

Public API
----------
  BLEND_WEIGHTS                              dict[str, float]
  build_blend_curve()      -> pd.DataFrame   (Blend, Sys E, R3, SPY equity)
  blend_metrics(eq, spy)   -> dict
  compute_blend_signal(df) -> dict           (live combined allocation)
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RA_CSV = ROOT / "exports" / "trap_regime_backtest" / "risk_allocation" / "risk_allocation_equity_curves.csv"
R3_CACHE = ROOT / "data" / "r3_signal_cache.json"

BLEND_WEIGHTS = {"Sys_E_Original": 0.80, "R3_Vol_Adj": 0.20}
BLEND_COST = 0.0010          # 10 bps per unit turnover (commission + spread)
RF = 0.02
INIT = 100_000.0

# R3 allocation constants (mirror pages/15 — do not modify originals)
R3_BULL_SPY = 0.350
R3_BULL_STK = 0.217


# ── Backtest ──────────────────────────────────────────────────────────────────
def build_blend_curve() -> pd.DataFrame:
    """Weekly-rebalanced 80/20 blend from the existing sleeve curves."""
    ra = pd.read_csv(RA_CSV, index_col="Date", parse_dates=True).sort_index()
    rets = ra[list(BLEND_WEIGHTS)].pct_change().fillna(0.0)
    R = rets.to_numpy()
    w = np.array([BLEND_WEIGHTS[c] for c in rets.columns])

    rp = R @ w
    growth = 1.0 + rp
    growth[growth == 0] = 1e-9
    drift = (w * (1.0 + R)) / growth[:, None]
    turnover = np.abs(drift - w).sum(axis=1)
    net = rp - BLEND_COST * turnover
    eq = np.cumprod(1.0 + net)
    eq[0] = 1.0

    out = pd.DataFrame(index=ra.index)
    out["Blend_E80_R3"] = eq * INIT
    out["Sys_E_Original"] = ra["Sys_E_Original"] / ra["Sys_E_Original"].iloc[0] * INIT
    out["R3_Vol_Adj"] = ra["R3_Vol_Adj"] / ra["R3_Vol_Adj"].iloc[0] * INIT
    out["Buy_Hold_SPY"] = ra["Buy_Hold_SPY"] / ra["Buy_Hold_SPY"].iloc[0] * INIT
    return out


def blend_metrics(eq: pd.Series, spy: pd.Series) -> dict:
    eq, spy = eq.dropna(), spy.dropna()
    ret = eq.pct_change().dropna()
    sret = spy.pct_change().reindex(ret.index).fillna(0)
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(52)
    dn = ret[ret < 0].std() * np.sqrt(52)
    dd = (eq / eq.cummax() - 1).min()
    beta = ret.cov(sret) / sret.var() if sret.var() > 0 else np.nan
    yearly = (1 + ret).groupby(ret.index.year).prod() - 1
    return {
        "CAGR_%": round(cagr * 100, 2),
        "Max_DD_%": round(dd * 100, 2),
        "Calmar": round(cagr / abs(dd), 3) if dd else np.nan,
        "Sharpe": round((ret.mean() * 52 - RF) / vol, 3) if vol else np.nan,
        "Sortino": round((ret.mean() * 52 - RF) / dn, 3) if dn else np.nan,
        "Vol_%": round(vol * 100, 1),
        "Beta": round(beta, 3),
        "Alpha_%": round((ret.mean() - beta * sret.mean()) * 52 * 100, 2),
        "Win_Rate_%": round((ret > 0).mean() * 100, 1),
        "Best_Year_%": round(yearly.max() * 100, 1),
        "Worst_Year_%": round(yearly.min() * 100, 1),
        "Final_$": round(float(eq.iloc[-1])),
    }


# ── Live signal ───────────────────────────────────────────────────────────────
def _r3_deploy_from_cache() -> dict | None:
    if R3_CACHE.exists():
        try:
            return json.loads(R3_CACHE.read_text())
        except Exception:
            return None
    return None


def compute_blend_signal(df: pd.DataFrame) -> dict:
    """
    Combined live allocation: 0.8 × System E allocation + 0.2 × R3 allocation.

    R3 leg: deploy% into (35% SPY + 21.7% × top-3), remainder in IEF.
    Uses data/r3_signal_cache.json (Thursday snapshot) for the R3 deploy;
    falls back to 50% deploy if unavailable.
    """
    from strategy_e.engine import compute_system_e_signal

    sig_e = compute_system_e_signal(df)
    r3 = _r3_deploy_from_cache() or {}
    deploy = float(r3.get("deploy_pct", 50.0)) / 100.0
    r3_top3 = r3.get("top3_picks") or sig_e.get("top3", [])

    # R3 leg allocation
    alloc_r3: dict[str, float] = {}
    if len(r3_top3) >= 3:
        alloc_r3["SPY"] = R3_BULL_SPY * deploy
        for t in r3_top3[:3]:
            alloc_r3[t] = alloc_r3.get(t, 0.0) + R3_BULL_STK * deploy
    else:
        alloc_r3["SPY"] = deploy
    alloc_r3["IEF"] = alloc_r3.get("IEF", 0.0) + (1.0 - deploy)

    # Combine 80/20
    combined: dict[str, float] = {}
    for t, wt in (sig_e.get("allocation") or {}).items():
        combined[t] = combined.get(t, 0.0) + 0.80 * wt
    for t, wt in alloc_r3.items():
        combined[t] = combined.get(t, 0.0) + 0.20 * wt
    combined = {t: round(wt, 4) for t, wt in sorted(combined.items(), key=lambda x: -x[1]) if wt > 0.0005}

    return {
        "sys_e": sig_e,
        "r3": r3,
        "r3_deploy_pct": round(deploy * 100, 1),
        "allocation": combined,
        "as_of": dt.date.today().isoformat(),
    }

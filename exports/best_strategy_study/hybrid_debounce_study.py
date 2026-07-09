"""
Hybrid Debounce Study — Asymmetric Entry/Exit for H8b
======================================================
Extends the H8b result (SPY 8-week realized vol >= 15% → switch to Strategy D)
with asymmetric debouncing:
  - Entry into D mode: fast (1 week) — triggered immediately when vol >= threshold
  - Exit back to A:  slow (N consecutive weeks below threshold required)

Grid tested:
  vol thresholds:  12%, 15%, 18%, 20%
  exit_weeks:      1 (no debounce), 2, 3, 4
  → 16 combinations

Also tests a composite signal:
  enter D if (SPY vol >= 15%) OR (d2_warning_count >= 2)
  with exit debounce N = 1, 2, 3, 4
  → 4 additional variants

Outputs → exports/best_strategy_study/
  hybrid_debounce_results.csv   - full metrics for all variants
  hybrid_results.json           - updated with debounce winner if better than H8b
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent

RA_CSV   = ROOT / "exports" / "trap_regime_backtest" / "risk_allocation" / "risk_allocation_equity_curves.csv"
WFO_CSV  = ROOT / "exports" / "crash_predictor_study" / "weekly_feature_outcomes.csv"

RF         = 0.02
COST       = 0.0005    # 5 bps per unit turnover
VOL_TARGET = 0.15
INIT       = 100_000.0


# ── Performance metrics ────────────────────────────────────────────────────────
def metrics(eq: pd.Series, spy_ret: pd.Series, name: str, n_switches: int = 0,
            pct_defensive: float = 0.0) -> dict:
    eq = eq.dropna()
    ret = eq.pct_change().dropna()
    years   = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr    = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    vol     = ret.std() * np.sqrt(52)
    sharpe  = (ret.mean() * 52 - RF) / vol if vol > 0 else np.nan
    dn      = ret[ret < 0].std() * np.sqrt(52)
    sortino = (ret.mean() * 52 - RF) / dn if dn > 0 else np.nan
    dd      = (eq / eq.cummax() - 1).min()
    calmar  = cagr / abs(dd) if dd != 0 else np.nan
    s       = spy_ret.reindex(ret.index).fillna(0)
    beta    = ret.cov(s) / s.var() if s.var() > 0 else np.nan
    alpha   = (ret.mean() - beta * s.mean()) * 52
    yearly  = (1 + ret).groupby(ret.index.year).prod() - 1
    return {
        "Strategy":     name,
        "Years":        round(years, 1),
        "CAGR_%":       round(cagr   * 100, 2),
        "Max_DD_%":     round(dd     * 100, 2),
        "Calmar":       round(calmar,    3),
        "Sharpe":       round(sharpe,    3),
        "Sortino":      round(sortino,   3),
        "Vol_%":        round(vol    * 100, 1),
        "Beta":         round(beta,      3),
        "Alpha_%":      round(alpha  * 100, 2),
        "Best_Year_%":  round(yearly.max() * 100, 1),
        "Worst_Year_%": round(yearly.min() * 100, 1),
        "Final_$":      round(float(eq.iloc[-1] / eq.iloc[0] * INIT)),
        "N_Switches":   n_switches,
        "Pct_Defensive_%": round(pct_defensive * 100, 1),
    }


# ── Strategy A: 80% SysE / 20% R3 ─────────────────────────────────────────────
def build_strategy_a(ra_ret: pd.DataFrame) -> pd.Series:
    W_SYSE, W_R3 = 0.80, 0.20
    rp = W_SYSE * ra_ret["Sys_E_Original"] + W_R3 * ra_ret["R3_Vol_Adj"]
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


# ── Debounced asymmetric signal ────────────────────────────────────────────────
def debounced_bear_flag(raw_signal: pd.Series, exit_weeks: int) -> pd.Series:
    """
    raw_signal: boolean Series (True = defensive/bear condition is active)
    Entry into defensive: immediate (1 week) — same as raw_signal
    Exit back to A: requires N consecutive weeks of raw_signal == False

    Returns boolean Series (True = in defensive/D mode)
    Already shifted by 1 week (no look-ahead).
    """
    raw = raw_signal.values.astype(bool)
    n = len(raw)
    state = np.zeros(n, dtype=bool)

    if exit_weeks <= 1:
        # No debounce — symmetric
        return pd.Series(raw, index=raw_signal.index)

    # Stateful debounce
    in_defensive = False
    consecutive_clear = 0
    for i in range(n):
        if raw[i]:
            # Enter or stay in defensive
            in_defensive = True
            consecutive_clear = 0
        else:
            if in_defensive:
                consecutive_clear += 1
                if consecutive_clear >= exit_weeks:
                    in_defensive = False
                    consecutive_clear = 0
            else:
                consecutive_clear = 0
        state[i] = in_defensive

    return pd.Series(state, index=raw_signal.index)


# ── Hybrid: switch between A and D using debounced flag ───────────────────────
def build_hybrid_debounced(a_ret: pd.Series, expo_d: pd.Series,
                            bear_flag_shifted: pd.Series,
                            spy_ret: pd.Series, name: str) -> tuple[dict, pd.Series]:
    """
    bear_flag_shifted: already shifted by 1 week (no look-ahead).
    """
    bear   = bear_flag_shifted.reindex(a_ret.index).fillna(False)
    e_bull = pd.Series(1.0, index=a_ret.index)
    e_bear = expo_d.reindex(a_ret.index).fillna(1.0)
    expo_h = e_bear.where(bear, e_bull)

    expo_change = expo_h.diff().abs().fillna(0.0)
    h_ret = expo_h * a_ret - COST * expo_change
    eq_h  = (1 + h_ret).cumprod()

    pct_bear = bear.mean()
    n_switch = int((bear.astype(int).diff().abs().fillna(0) > 0).sum())

    m = metrics(eq_h, spy_ret, name, n_switch, pct_bear)
    return m, eq_h


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    ra  = pd.read_csv(RA_CSV, index_col="Date", parse_dates=True).sort_index()
    wfo = pd.read_csv(WFO_CSV, index_col="date", parse_dates=True).sort_index()

    ra_ret  = ra.pct_change().fillna(0.0)
    spy_ret = ra_ret["Buy_Hold_SPY"]

    # Align WFO (Thursday signal) to RA (Friday prices): shift +1 day, ffill
    wfo.index = wfo.index + pd.Timedelta(days=1)
    wfo_a = wfo.reindex(ra.index, method="ffill")

    # ── Build base strategies ──────────────────────────────────────────────────
    eq_a  = build_strategy_a(ra_ret)
    a_ret = eq_a.pct_change().fillna(0.0)
    eq_d, expo_d = build_strategy_d(a_ret)

    # ── SPY 8-week realized vol (raw, unshifted) ──────────────────────────────
    spy_rvol = spy_ret.rolling(8).std() * np.sqrt(52)

    # ── d2_warning_count for composite signal ─────────────────────────────────
    # d2_warning_count: number of technical warning indicators currently firing (0-7)
    warning_cnt = wfo_a["d2_warning_count"].fillna(0.0)

    results: list[dict] = []
    curves:  dict[str, pd.Series] = {}

    # ── Reference: pure A, pure D, SPY ────────────────────────────────────────
    m_spy = metrics(ra["Buy_Hold_SPY"] / ra["Buy_Hold_SPY"].iloc[0], spy_ret, "SPY Buy & Hold")
    m_spy["N_Switches"] = 0; m_spy["Pct_Defensive_%"] = 0.0
    results.append(m_spy)
    curves["SPY Buy & Hold"] = ra["Buy_Hold_SPY"] / ra["Buy_Hold_SPY"].iloc[0]

    m_a = metrics(eq_a, spy_ret, "A: 80%SysE/20%R3 static")
    m_a["N_Switches"] = 0; m_a["Pct_Defensive_%"] = 0.0
    results.append(m_a)
    curves["A: 80%SysE/20%R3 static"] = eq_a

    m_d = metrics(eq_d, spy_ret, "D: 15%vol-target on A (always)")
    m_d["N_Switches"] = 0; m_d["Pct_Defensive_%"] = 100.0
    results.append(m_d)
    curves["D: 15%vol-target on A (always)"] = eq_d

    # ── Baseline H8b (symmetric, no debounce, 15% threshold) ──────────────────
    raw_15 = spy_rvol >= 0.15
    flag_15_nd = raw_15.shift(1).fillna(False).astype(bool)
    m_h8b, eq_h8b = build_hybrid_debounced(a_ret, expo_d, flag_15_nd, spy_ret,
                                            "H8b_baseline (SPYvol>=15, exit=1w)")
    results.append(m_h8b)
    curves[m_h8b["Strategy"]] = eq_h8b

    # ── Grid: vol thresholds × exit_weeks ─────────────────────────────────────
    vol_thresholds = [0.12, 0.15, 0.18, 0.20]
    exit_weeks_list = [1, 2, 3, 4]

    for thresh in vol_thresholds:
        raw_flag = spy_rvol >= thresh          # unshifted
        for exit_w in exit_weeks_list:
            # Apply debounce on raw (unshifted) signal, then shift 1 to avoid look-ahead
            debounced = debounced_bear_flag(raw_flag, exit_w)
            flag_shifted = debounced.shift(1).fillna(False).astype(bool)

            name = f"Deb_SPYvol{int(thresh*100)}pct_exit{exit_w}w"
            m, eq = build_hybrid_debounced(a_ret, expo_d, flag_shifted, spy_ret, name)

            # Extra metadata for the grid
            m["Vol_Thresh_%"] = int(thresh * 100)
            m["Exit_Weeks"]   = exit_w

            results.append(m)
            curves[name] = eq
            print(f"  {name:45s}  Calmar={m['Calmar']:.3f}  CAGR={m['CAGR_%']:.1f}%  DD={m['Max_DD_%']:.1f}%  N_sw={m['N_Switches']}")

    # ── Composite signal: SPYvol>=15% OR warning_cnt>=2 ──────────────────────
    raw_composite = (spy_rvol >= 0.15) | (warning_cnt >= 2)
    for exit_w in exit_weeks_list:
        debounced = debounced_bear_flag(raw_composite, exit_w)
        flag_shifted = debounced.shift(1).fillna(False).astype(bool)

        name = f"Comp_SPYvol15_OR_Warn2_exit{exit_w}w"
        m, eq = build_hybrid_debounced(a_ret, expo_d, flag_shifted, spy_ret, name)
        m["Vol_Thresh_%"] = 15
        m["Exit_Weeks"]   = exit_w
        m["Signal_Type"]  = "Composite"

        results.append(m)
        curves[name] = eq
        print(f"  {name:45s}  Calmar={m['Calmar']:.3f}  CAGR={m['CAGR_%']:.1f}%  DD={m['Max_DD_%']:.1f}%  N_sw={m['N_Switches']}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)

    # Ensure Signal_Type column exists for baselines/grid entries
    if "Signal_Type" not in df_results.columns:
        df_results["Signal_Type"] = "SPY_Vol"
    df_results["Signal_Type"] = df_results["Signal_Type"].fillna("SPY_Vol")

    df_results.to_csv(OUT / "hybrid_debounce_results.csv", index=False)
    print(f"\nSaved → {OUT / 'hybrid_debounce_results.csv'}")

    # ── Identify best debounce variant (exclude baselines) ────────────────────
    grid_results = df_results[df_results["Strategy"].str.startswith(("Deb_", "Comp_"))]
    if not grid_results.empty:
        best_idx  = grid_results["Calmar"].idxmax()
        best_row  = grid_results.loc[best_idx].to_dict()
    else:
        best_row = {}

    # ── Update hybrid_results.json ─────────────────────────────────────────────
    json_path = OUT / "hybrid_results.json"
    if json_path.exists():
        existing = json.loads(json_path.read_text())
    else:
        existing = {}

    existing["debounce_study"] = {
        "generated":     pd.Timestamp.now().isoformat(),
        "description":   "Asymmetric debounced variants of H8b — fast entry, slow exit",
        "grid":          "vol_thresh=[12,15,18,20]% × exit_weeks=[1,2,3,4] + composite signal",
        "best_debounced": best_row,
        "h8b_baseline_calmar": m_h8b["Calmar"],
        "improvement_over_h8b": round(
            (best_row.get("Calmar", 0) - m_h8b["Calmar"]) / abs(m_h8b["Calmar"]) * 100, 2
        ) if best_row else 0.0,
    }
    json_path.write_text(json.dumps(existing, indent=2, default=str))
    print(f"Updated → {json_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    SEP = "=" * 100
    print(f"\n{SEP}")
    print("HYBRID DEBOUNCE STUDY  |  SPY 8-week vol signal, asymmetric exit debounce")
    print(SEP)

    pd.set_option("display.width", 220)
    show_cols = ["Strategy", "CAGR_%", "Max_DD_%", "Calmar", "Sharpe",
                 "Sortino", "Pct_Defensive_%", "N_Switches"]

    print("\n-- BASELINES --")
    baselines = df_results[df_results["Strategy"].str.startswith(
        ("SPY", "A:", "D:", "H8b_baseline"))]
    print(baselines[show_cols].to_string(index=False))

    print("\n-- GRID: vol threshold × exit_weeks (sorted by Calmar) --")
    grid_df = grid_results[grid_results["Strategy"].str.startswith("Deb_")].copy()
    grid_df = grid_df.sort_values("Calmar", ascending=False)
    print(grid_df[show_cols + ["Vol_Thresh_%", "Exit_Weeks"]].to_string(index=False))

    print("\n-- COMPOSITE SIGNAL --")
    comp_df = grid_results[grid_results["Strategy"].str.startswith("Comp_")].copy()
    comp_df = comp_df.sort_values("Calmar", ascending=False)
    print(comp_df[show_cols].to_string(index=False))

    print(f"\n{'--' * 50}")
    if best_row:
        print(f"BEST DEBOUNCED VARIANT: {best_row['Strategy']}")
        print(f"  Calmar={best_row['Calmar']:.3f}  CAGR={best_row['CAGR_%']:.1f}%  "
              f"MaxDD={best_row['Max_DD_%']:.1f}%  Sharpe={best_row['Sharpe']:.3f}")
        print(f"  vs H8b baseline: Calmar={m_h8b['Calmar']:.3f}")
        print(f"  Improvement: {existing['debounce_study']['improvement_over_h8b']:.1f}%")
    print(f"\nSaved: {OUT}/hybrid_debounce_results.csv")
    print(f"Updated: {OUT}/hybrid_results.json")


if __name__ == "__main__":
    main()

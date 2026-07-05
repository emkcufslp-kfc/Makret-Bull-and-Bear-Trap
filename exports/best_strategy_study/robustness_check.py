"""
Robustness Check — Schedule Sweep Sub-Period Stability
=======================================================
Question: is "Fri signal + exec +2d" a real edge or a lucky cell?

Method:
  1. Rebuild all 15 schedule equity curves (per strategy) from daily data.
  2. Score each combo in FOUR independent sub-periods:
       P1 2004–2009  (GFC cycle)      P2 2010–2015  (QE bull)
       P3 2016–2020  (late cycle + COVID)   P4 2021–2026  (inflation cycle)
     plus the two halves (2004–2015 / 2015–2026).
  3. Rank combos by Calmar within each period; report mean rank, worst rank,
     rank of the full-period winner, and how often each combo lands top-3.

A robust schedule should rank near the top in MOST periods, not just overall.

Output: robustness_results.csv + console summary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from day_sweep_backtest import DAY_LABEL, load_inputs, r3_daily_scores, run_schedule

OUT = Path(__file__).resolve().parent
RF = 0.02

PERIODS = {
    "P1_2004_2009": ("2004-01-01", "2009-12-31"),
    "P2_2010_2015": ("2010-01-01", "2015-12-31"),
    "P3_2016_2020": ("2016-01-01", "2020-12-31"),
    "P4_2021_2026": ("2021-01-01", "2026-12-31"),
    "H1_2004_2015": ("2004-01-01", "2015-06-30"),
    "H2_2015_2026": ("2015-07-01", "2026-12-31"),
    "FULL": ("2004-01-01", "2026-12-31"),
}


def calmar(eq: pd.Series) -> float:
    eq = eq.dropna()
    if len(eq) < 30:
        return np.nan
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    dd = (eq / eq.cummax() - 1).min()
    return cagr / abs(dd) if dd != 0 else np.nan


def cagr_dd(eq: pd.Series) -> tuple[float, float]:
    eq = eq.dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    c = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    d = (eq / eq.cummax() - 1).min()
    return c * 100, d * 100


def main() -> None:
    px, raw = load_inputs()
    bull, bear = r3_daily_scores(raw)

    curves: dict[str, dict[str, pd.Series]] = {"System E": {}, "R3 Vol-Adj": {}, "E80/R3 Blend": {}}
    for sd in range(5):
        for lag in (1, 2, 3):
            key = f"{DAY_LABEL[sd]} +{lag}d"
            eq_e, eq_r, eq_b = run_schedule(px, raw, sd, lag, bull, bear)
            curves["System E"][key] = eq_e
            curves["R3 Vol-Adj"][key] = eq_r
            curves["E80/R3 Blend"][key] = eq_b
            print(f"built {key}", flush=True)

    all_rows = []
    for strat, combo_curves in curves.items():
        # Calmar per period per combo
        per = pd.DataFrame({
            pname: {
                combo: calmar(eq.loc[p0:p1])
                for combo, eq in combo_curves.items()
            }
            for pname, (p0, p1) in PERIODS.items()
        })
        ranks = per.rank(ascending=False)
        sub_cols = ["P1_2004_2009", "P2_2010_2015", "P3_2016_2020", "P4_2021_2026"]
        for combo in per.index:
            full_eq = combo_curves[combo]
            c_full, d_full = cagr_dd(full_eq)
            r = {
                "Strategy": strat, "Schedule": combo,
                "Full_CAGR_%": round(c_full, 2), "Full_MaxDD_%": round(d_full, 2),
                "Full_Calmar": round(per.loc[combo, "FULL"], 3),
                "Full_Rank": int(ranks.loc[combo, "FULL"]),
                "Mean_Rank_4P": round(float(ranks.loc[combo, sub_cols].mean()), 1),
                "Worst_Rank_4P": int(ranks.loc[combo, sub_cols].max()),
                "Top3_Count_4P": int((ranks.loc[combo, sub_cols] <= 3).sum()),
                "Rank_H1": int(ranks.loc[combo, "H1_2004_2015"]),
                "Rank_H2": int(ranks.loc[combo, "H2_2015_2026"]),
            }
            for pcol in sub_cols:
                r[f"Calmar_{pcol[:2]}"] = round(per.loc[combo, pcol], 2)
            all_rows.append(r)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["Strategy", "Mean_Rank_4P"]).reset_index(drop=True)
    df.to_csv(OUT / "robustness_results.csv", index=False)

    pd.set_option("display.width", 250)
    for strat in curves:
        sub = df[df["Strategy"] == strat].head(5)
        print(f"\n=== {strat} — most robust schedules (by mean rank across 4 sub-periods) ===")
        print(sub[["Schedule", "Full_Calmar", "Full_Rank", "Mean_Rank_4P", "Worst_Rank_4P",
                   "Top3_Count_4P", "Rank_H1", "Rank_H2"]].to_string(index=False))
        # where does the full-period winner sit?
        w = df[(df["Strategy"] == strat) & (df["Full_Rank"] == 1)].iloc[0]
        print(f"Full-period winner: {w['Schedule']} -> mean rank {w['Mean_Rank_4P']}, "
              f"worst {w['Worst_Rank_4P']}, top-3 in {w['Top3_Count_4P']}/4 sub-periods, "
              f"H1 rank {w['Rank_H1']}, H2 rank {w['Rank_H2']}")


if __name__ == "__main__":
    main()

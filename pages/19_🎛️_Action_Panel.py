"""
pages/19_🎛️_Action_Panel.py
=============================
Action Panel — today's action for every strategy in the ecosystem,
plus a cross-strategy comparison chart and ranking.

Sections:
  1. Today's Actions (all strategies: signal, allocation, next execution)
  2. Comparison Chart (normalized equity curves, common window, log scale)
  3. Ranking (Calmar-ranked metrics table)
"""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Action Panel", page_icon="🎛️", layout="wide")

ROOT = Path(__file__).resolve().parents[1]

from utils.data_engine import get_clean_master
from backend.strategies.blend_e_r3_engine import blend_metrics, build_blend_curve, compute_blend_signal


@st.cache_data(ttl=3600)
def load_data():
    return get_clean_master()


@st.cache_data(ttl=86400)
def load_blend_curves():
    return build_blend_curve()


@st.cache_data(ttl=86400)
def load_long_window():
    p = ROOT / "data" / "Strategy_Comparisons" / "long_window_equity_curves.csv"
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def latest_weights(csv_path: str):
    p = ROOT / csv_path
    if not p.exists():
        return {}
    df = pd.read_csv(p, index_col="Date", parse_dates=True).sort_index()
    row = df.iloc[-1].fillna(0.0)
    return {t: round(float(w) * 100, 1) for t, w in row.items() if float(w) > 0.001}


def fmt_alloc(alloc: dict, pct_already: bool = False) -> str:
    if not alloc:
        return "—"
    items = sorted(alloc.items(), key=lambda x: -x[1])
    return ", ".join(f"{t} {w if pct_already else round(w * 100, 1)}%" for t, w in items[:6])


def next_weekday(target: int) -> dt.date:
    today = dt.date.today()
    return today + dt.timedelta(days=(target - today.weekday()) % 7)


with st.sidebar:
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    render_master_controls()
    render_ecosystem_sidebar()


st.title("🎛️ Action Panel")
st.markdown("One screen: what every strategy says to do **today**, how they compare, and how they rank.")

data = load_data()
if data is None or data.empty:
    st.error("⚠️ Failed to load market data.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TODAY'S ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📋 Section 1 — Today's Actions")
today_name = dt.date.today().strftime("%A, %b %d %Y")
st.caption(f"As of {today_name}. All strategies use T+1 execution — signals lock at the close, trades go in at the next open.")

rows = []

# System E + R3 + Blend (live)
try:
    bsig = compute_blend_signal(data)
    sig_e = bsig["sys_e"]
    r3 = bsig["r3"] or {}

    rows.append({
        "Strategy": "🚂 System E",
        "Signal → Exec": "Tue close → Wed open",
        "Current State": f"{sig_e.get('regime', '?')} (breadth {sig_e.get('breadth_pct', 0):.0f}%)",
        "Target Allocation": fmt_alloc(sig_e.get("allocation", {})),
        "Next Execution": next_weekday(2).strftime("%a %b %d"),
    })
    rows.append({
        "Strategy": "📊 R3 Vol-Adjusted",
        "Signal → Exec": "Thu close → Fri open",
        "Current State": f"{r3.get('regime', '?')} ({r3.get('deploy_pct', '—')}% deployed)",
        "Target Allocation": (
            f"Deploy {r3.get('deploy_pct', '—')}% → SPY 35% + top-3 ({', '.join(r3.get('top3_picks', [])[:3])}), rest IEF"
            if r3 else "Run daily_refresh (Thu) to update"
        ),
        "Next Execution": next_weekday(4).strftime("%a %b %d"),
    })
    rows.append({
        "Strategy": "🏆 E80/R3 Blend",
        "Signal → Exec": "Legs: Tue→Wed & Thu→Fri",
        "Current State": f"E: {sig_e.get('regime', '?')} · R3 deploy {bsig['r3_deploy_pct']:.0f}%",
        "Target Allocation": fmt_alloc(bsig.get("allocation", {})),
        "Next Execution": f"{next_weekday(2).strftime('%a')} + {next_weekday(4).strftime('%a')}",
    })
except Exception as exc:
    st.warning(f"Live Sys E / R3 / Blend signals unavailable: {exc}")

# 200MA strategy (live, simple)
try:
    spy = data["SPY"].dropna()
    ma200 = spy.rolling(200).mean().iloc[-1]
    above = spy.iloc[-1] > ma200
    rows.append({
        "Strategy": "📈 200MA Strategy",
        "Signal → Exec": "Daily close → next open",
        "Current State": f"SPY {'ABOVE' if above else 'BELOW'} 200d MA ({spy.iloc[-1]:,.0f} vs {ma200:,.0f})",
        "Target Allocation": "SPY 100%" if above else "Cash / T-Bills 100%",
        "Next Execution": "Next session open",
    })
except Exception:
    pass

# Platinum + F-TAA (latest engine weights)
plat_w = latest_weights("data/Platinum_Results/Platinum_Weights.csv")
if plat_w:
    rows.append({
        "Strategy": "💎 Platinum",
        "Signal → Exec": "Daily model → next open",
        "Current State": f"{len(plat_w)} positions",
        "Target Allocation": fmt_alloc(plat_w, pct_already=True),
        "Next Execution": "Next session open",
    })
fta_w = latest_weights("data/Fund_Tactical_Results/Fund_Tactical_weights.csv")
if fta_w:
    rows.append({
        "Strategy": "🧭 F-TAA",
        "Signal → Exec": "Weekly model → next open",
        "Current State": f"{len(fta_w)} positions",
        "Target Allocation": fmt_alloc(fta_w, pct_already=True),
        "Next Execution": "Next session open",
    })

rows.append({
    "Strategy": "🛡️ NTSX",
    "Signal → Exec": "Multi-indicator gate",
    "Current State": "See Strategies Dashboard",
    "Target Allocation": "NTSX (90/60) per gate",
    "Next Execution": "—",
})

st.dataframe(pd.DataFrame(rows).set_index("Strategy"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COMPARISON CHART
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📈 Section 2 — Strategy Comparison (common window, $100k start, log scale)")

try:
    bc = load_blend_curves()          # weekly, 2004+
    lw = load_long_window()           # daily, 2007-12+

    curves = {
        "E80/R3 Blend": bc["Blend_E80_R3"],
        "System E": bc["Sys_E_Original"],
        "R3 Vol-Adjusted": bc["R3_Vol_Adj"],
        "SPY Buy & Hold": bc["Buy_Hold_SPY"],
    }
    if not lw.empty:
        for col, label in [("Platinum Proxy-Aware", "Platinum"), ("NTSX", "NTSX"), ("F-TAA Stitched", "F-TAA")]:
            if col in lw.columns:
                curves[label] = lw[col].resample("W-FRI").last().dropna()

    # Common window
    start = max(s.dropna().index.min() for s in curves.values())
    norm = {}
    for label, s in curves.items():
        s = s[s.index >= start].dropna()
        norm[label] = s / s.iloc[0] * 100_000

    COLORS = {
        "E80/R3 Blend": ("#a855f7", 3), "System E": ("#f59e0b", 1.5),
        "R3 Vol-Adjusted": ("#3b82f6", 1.5), "SPY Buy & Hold": ("#94a3b8", 1.5),
        "Platinum": ("#22c55e", 1.5), "NTSX": ("#e879f9", 1.2), "F-TAA": ("#f43f5e", 1.2),
    }
    fig = go.Figure()
    for label, s in norm.items():
        color, width = COLORS.get(label, ("#e2e8f0", 1))
        fig.add_trace(go.Scatter(
            x=s.index, y=s, name=label, line=dict(color=color, width=width),
            hovertemplate="%{x|%Y-%m-%d}: $%{y:,.0f}<extra>" + label + "</extra>",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a", font_color="#e2e8f0",
        yaxis_type="log", yaxis_title="Portfolio Value (log)",
        hovermode="x unified", height=480,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=20, l=70, r=20),
    )
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b", tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Common window: {start.date()} → present. Curves embed each strategy's own cost/T+1 assumptions.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — RANKING
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🏅 Section 3 — Ranking (common window, ranked by Calmar)")

    spy_common = norm.get("SPY Buy & Hold")
    rank_rows = []
    for label, s in norm.items():
        m = blend_metrics(s, spy_common)
        m["Strategy"] = label
        rank_rows.append(m)
    rank_df = pd.DataFrame(rank_rows).set_index("Strategy")
    rank_df = rank_df.sort_values("Calmar", ascending=False)
    rank_df.insert(0, "Rank", range(1, len(rank_df) + 1))
    st.dataframe(
        rank_df.style.background_gradient(subset=["Calmar", "CAGR_%", "Sharpe"], cmap="Greens")
                     .background_gradient(subset=["Max_DD_%"], cmap="Reds_r"),
        use_container_width=True,
    )

    best = rank_df.index[0]
    st.success(f"🏆 Current leader by Calmar: **{best}** "
               f"(CAGR {rank_df.loc[best, 'CAGR_%']:.1f}%, Max DD {rank_df.loc[best, 'Max_DD_%']:.1f}%)")

except Exception as exc:
    st.error(f"Comparison/ranking unavailable: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SCHEDULE STUDY & ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📆 Section 4 — Signal-Day Schedule Study")

SWEEP_CSV = ROOT / "exports" / "best_strategy_study" / "day_sweep_results.csv"
ROBUST_CSV = ROOT / "exports" / "best_strategy_study" / "robustness_results.csv"

try:
    sweep = pd.read_csv(SWEEP_CSV)
    st.caption(
        "All 15 weekly schedules (signal day Mon–Fri × execution +1/+2/+3 trading days) tested per "
        "strategy on 22.5y of daily data — 0.16% round-trip costs, execution strictly after the signal close."
    )

    tab1, tab2 = st.tabs(["🏁 Full-Period Ranking", "🧪 Robustness (sub-periods)"])

    with tab1:
        show = sweep[["Rank", "Combo", "Strategy", "Signal_Day", "Exec_Lag_d",
                      "CAGR_%", "Max_DD_%", "Calmar", "Sharpe", "Final_$"]].head(20)
        st.dataframe(
            show.set_index("Rank").style
                .background_gradient(subset=["Calmar", "CAGR_%"], cmap="Greens")
                .background_gradient(subset=["Max_DD_%"], cmap="Reds_r")
                .format({"Final_$": "${:,.0f}"}),
            use_container_width=True, height=420,
        )

    with tab2:
        if ROBUST_CSV.exists():
            rob = pd.read_csv(ROBUST_CSV)
            st.dataframe(
                rob[["Strategy", "Schedule", "Full_Calmar", "Full_Rank", "Mean_Rank_4P",
                     "Worst_Rank_4P", "Top3_Count_4P", "Rank_H1", "Rank_H2"]]
                .style.background_gradient(subset=["Mean_Rank_4P", "Worst_Rank_4P"], cmap="RdYlGn_r"),
                use_container_width=True, height=420,
            )
        else:
            st.info("Run exports/best_strategy_study/robustness_check.py to generate sub-period ranks.")

    st.warning(
        "🧪 **Robustness verdict**: the full-period winner (Fri signal, exec +2d) earns its rank almost "
        "entirely in 2015–2026 (rank #1 in H2 but ~#10 of 15 in H1) — partly period luck. **Mon +1d** is the "
        "most consistent schedule (same mean sub-period rank, best worst-case). Schedule choice is "
        "second-order: every schedule of every strategy beats SPY in every sub-period. The live cadence "
        "stays **Tue→Wed (System E) / Thu→Fri (R3)** until stronger evidence emerges."
    )
except Exception as exc:
    st.info(f"Schedule study data unavailable: {exc}")

st.markdown(
    "<hr style='border-color:#1e293b;margin-top:30px'>"
    "<p style='color:#475569;font-size:0.8rem;text-align:center'>"
    "All figures from backtests on real historical data with T+1 execution and realistic costs "
    "(0.16% round-trip on sleeve trades; 10 bps blend rebalancing). "
    "Past performance does not guarantee future results.</p>",
    unsafe_allow_html=True,
)

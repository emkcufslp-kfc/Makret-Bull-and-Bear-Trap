"""
pages/18_🏆_E80_R3_Blend_Strategy.py
=====================================
E80/R3 Blend — Calmar-optimal strategy from exports/best_strategy_study/
80% System E + 20% R3 Vol-Adjusted, weekly rebalanced.

Sections:
  1. Live Combined Allocation
  2. 22-Year Backtest (realistic: sleeve T+1 + 0.16% RT cost + 10bps blend turnover)
  3. How the Blend Works
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="E80/R3 Blend Strategy", page_icon="🏆", layout="wide")

from utils.data_engine import get_clean_master
from backend.strategies.blend_e_r3_engine import (
    BLEND_WEIGHTS,
    blend_metrics,
    build_blend_curve,
    compute_blend_signal,
)


@st.cache_data(ttl=3600)
def load_data():
    return get_clean_master()


@st.cache_data(ttl=86400)
def load_curves():
    return build_blend_curve()


with st.sidebar:
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    render_master_controls()
    render_ecosystem_sidebar()


st.title("🏆 E80/R3 Blend Strategy")
st.markdown(
    "The Calmar-optimal combination from the strategy study: **80% System E + 20% R3 Vol-Adjusted**, "
    "rebalanced weekly. Beats pure System E on drawdown and Calmar while keeping ~92% of its CAGR."
)
st.caption(
    "Realistic assumptions: sleeves embed T+1 execution (Sys E Tue→Wed, R3 Thu→Fri) and 0.16% "
    "round-trip position costs; the blend layer pays an extra 10 bps per unit of rebalancing turnover."
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LIVE COMBINED ALLOCATION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📡 Section 1 — Live Combined Allocation")

data = load_data()
if data is None or data.empty:
    st.error("⚠️ Failed to load market data.")
    st.stop()

sig = compute_blend_signal(data)
sig_e = sig["sys_e"]
regime = sig_e.get("regime", "UNKNOWN")

colour = {"BULL": "#2ecc71", "BEAR": "#e74c3c"}.get(regime, "#94a3b8")
bg = {"BULL": "#0d2e1a", "BEAR": "#2e0d0d"}.get(regime, "#1e293b")
st.markdown(
    f"""
    <div style="background:{bg};border:2px solid {colour};border-radius:12px;
                padding:18px;text-align:center;margin-bottom:16px;">
        <div style="font-size:2.2rem;font-weight:800;color:{colour};">
            System E: {regime} &nbsp;·&nbsp; R3 Deploy: {sig['r3_deploy_pct']:.0f}%
        </div>
        <div style="color:#94a3b8;margin-top:6px;font-size:0.9rem;">
            Blend weights: 80% System E / 20% R3 &nbsp;|&nbsp;
            Rebalance: weekly (legs execute Wed / Fri opens, T+1)
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns([1, 1])
with c1:
    st.markdown("#### Combined Target Allocation")
    alloc = sig["allocation"]
    if alloc:
        rows = [{"Ticker": t, "Weight %": round(w * 100, 1)} for t, w in alloc.items()]
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)
    else:
        st.info("Allocation unavailable.")
with c2:
    st.markdown("#### Allocation Pie")
    if alloc:
        palette = {"SPY": "#3b82f6", "IEF": "#64748b", "CASH": "#334155"}
        fig_pie = go.Figure(go.Pie(
            labels=list(alloc.keys()),
            values=[w * 100 for w in alloc.values()],
            marker_colors=[palette.get(t, "#22c55e") for t in alloc],
            hole=0.45, textinfo="label+percent", textfont_size=11,
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10), height=280,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🏆 Section 2 — Backtest (Jan 2004 → Present, real data)")

try:
    curves = load_curves()
    m_bl = blend_metrics(curves["Blend_E80_R3"], curves["Buy_Hold_SPY"])
    m_e = blend_metrics(curves["Sys_E_Original"], curves["Buy_Hold_SPY"])
    m_r3 = blend_metrics(curves["R3_Vol_Adj"], curves["Buy_Hold_SPY"])
    m_spy = blend_metrics(curves["Buy_Hold_SPY"], curves["Buy_Hold_SPY"])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Blend CAGR", f"{m_bl['CAGR_%']:.1f}%", delta=f"{m_bl['CAGR_%'] - m_spy['CAGR_%']:+.1f}% vs SPY")
    k2.metric("Max Drawdown", f"{m_bl['Max_DD_%']:.1f}%", delta=f"{m_bl['Max_DD_%'] - m_e['Max_DD_%']:+.1f}% vs Sys E",
              delta_color="inverse")
    k3.metric("Calmar", f"{m_bl['Calmar']:.2f}")
    k4.metric("Sharpe", f"{m_bl['Sharpe']:.2f}")
    k5.metric("$100k grew to", f"${m_bl['Final_$']:,.0f}")

    SHOW = {
        "Blend_E80_R3": ("E80/R3 Blend", "#a855f7", 3),
        "Sys_E_Original": ("System E", "#f59e0b", 1.5),
        "R3_Vol_Adj": ("R3 Vol-Adjusted", "#3b82f6", 1.5),
        "Buy_Hold_SPY": ("SPY Buy & Hold", "#94a3b8", 1.5),
    }
    fig = go.Figure()
    for col, (label, color, width) in SHOW.items():
        fig.add_trace(go.Scatter(
            x=curves.index, y=curves[col], name=label,
            line=dict(color=color, width=width),
            hovertemplate="%{x|%Y-%m-%d}: $%{y:,.0f}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a", font_color="#e2e8f0",
        yaxis_type="log", yaxis_title="Portfolio Value (log, $100k start)",
        hovermode="x unified", height=440,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=20, l=70, r=20),
    )
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b", tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Metrics")
    st.dataframe(
        pd.DataFrame(
            [m_bl, m_e, m_r3, m_spy],
            index=["E80/R3 Blend", "System E", "R3 Vol-Adjusted", "SPY Buy & Hold"],
        ),
        use_container_width=True,
    )

    # Drawdown chart
    st.markdown("#### Drawdown")
    fig_dd = go.Figure()
    for col, (label, color, width) in SHOW.items():
        dd = curves[col] / curves[col].cummax() - 1
        fig_dd.add_trace(go.Scatter(
            x=curves.index, y=dd * 100, name=label,
            line=dict(color=color, width=width),
        ))
    fig_dd.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a", font_color="#e2e8f0",
        yaxis_title="Drawdown %", hovermode="x unified", height=300,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=20, l=60, r=20),
    )
    fig_dd.update_xaxes(gridcolor="#1e293b")
    fig_dd.update_yaxes(gridcolor="#1e293b", ticksuffix="%")
    st.plotly_chart(fig_dd, use_container_width=True)

except Exception as exc:
    st.error(f"Could not build backtest curves: {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🧠 Section 3 — How the Blend Works")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        """
**Why 80/20?**

A grid search over all Sys E / R3 / R7 weight combinations (5% steps, 22.5 years of
weekly data) found 80% System E + 20% R3 maximizes **Calmar** (CAGR ÷ Max DD).
The optimum is robust: neighbouring blends (80/15/5, 80/10/10) score nearly identically.

**What each sleeve contributes**

- *System E (80%)* — the return engine. Binary all-in/all-out NQ100 momentum.
- *R3 (20%)* — the shock absorber. Continuously sized 10–100% by Bull/Bear Trap
  scores and realized volatility; averages only ~44% deployed.
        """
    )
with col_b:
    st.markdown(
        """
**Execution playbook**

```
Tue close : System E signal locks   → execute Wed open (80% sleeve)
Thu close : R3 deploy % locks       → execute Fri open (20% sleeve)
Weekly    : rebalance sleeves to 80/20 (drift correction)
```

**Cost model (backtest)**

- Sleeves: 0.16% round-trip per position change, T+1 execution
- Blend layer: 10 bps × rebalancing turnover (commission + spread)

**Schedule note**: a 15-schedule sweep found Fri-signal variants scored best
over the full period, but robustness testing showed that edge concentrates in
2015–2026 — the live Tue/Thu cadence stays (see Action Panel → Section 4).

**Caveat**: both sleeves share the NQ100 momentum engine — 2022-type years
still hurt (worst year ≈ −21%). Past performance ≠ future results.
        """
    )

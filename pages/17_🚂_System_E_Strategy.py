"""
pages/17_🚂_System_E_Strategy.py
=================================
System E — NQ100 Momentum Strategy (dedicated page)

Sections:
  1. Live Signal (regime banner, gates, top-3 picks, allocation pie)
  2. Momentum Leaderboard (top-10 NQ100 momentum scores)
  3. 20-Year Backtest (System E vs SPY Buy & Hold, live-computed)
  4. How System E Works
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="System E Strategy", page_icon="🚂", layout="wide")

from utils.data_engine import get_clean_master
from strategy_e.engine import (
    backtest_system_e,
    compute_system_e_signal,
    get_next_execution_date,
)


@st.cache_data(ttl=3600)
def load_data():
    return get_clean_master()


@st.cache_data(ttl=3600)
def run_backtest(df_key: str, _df: pd.DataFrame):
    # df_key busts the cache when the data slice changes
    return backtest_system_e(_df)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    render_master_controls()
    render_ecosystem_sidebar()


def _metrics_from_equity(eq: pd.Series) -> dict:
    ret = eq.pct_change().dropna()
    if ret.empty:
        return {}
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    ann_vol = ret.std() * np.sqrt(52)
    sharpe = (ret.mean() * 52 - 0.02) / ann_vol if ann_vol > 0 else 0.0
    downside = ret[ret < 0].std() * np.sqrt(52)
    sortino = (ret.mean() * 52 - 0.02) / downside if downside > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    return {
        "CAGR %": round(cagr * 100, 1),
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "Max DD %": round(dd * 100, 1),
        "Ann. Vol %": round(ann_vol * 100, 1),
    }


def dashboard():
    from utils.ui_utils import resolve_master_date_slice

    st.title("🚂 System E — NQ100 Momentum Strategy")
    st.markdown(
        "Weekly NQ100 momentum rotation: top-3 skip-month momentum picks + SPY core, "
        "gated by a dual 200-day MA regime filter and a 40% breadth threshold. "
        "Defensive legs into IEF bonds when the gate closes."
    )

    analysis_date = st.session_state["master_date"]

    with st.spinner("Loading market data…"):
        data = load_data()
        if data is None or data.empty:
            st.error("⚠️ Failed to load market data.")
            return
        d, actual_date = resolve_master_date_slice(data, analysis_date)
        if d.empty or actual_date is None:
            st.error("No data available for the selected date.")
            return

    if len(d) < 252:
        st.error(f"Insufficient data for analysis ({len(d)} rows, need 252+).")
        return

    st.markdown(
        f"<p style='color:#8892a4;'>Master date: <b>{analysis_date.strftime('%Y-%m-%d')}</b> "
        f"| Resolved data date: <b>{actual_date.strftime('%Y-%m-%d')}</b></p>",
        unsafe_allow_html=True,
    )

    sig = compute_system_e_signal(d)
    regime = sig["regime"]

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — LIVE SIGNAL
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📡 Section 1 — Live Signal")
    st.caption("⏰ Signal: Tuesday close  |  Execute: Wednesday open  (T+1 execution)")

    if regime == "BULL":
        colour, bg, label = "#2ecc71", "#0d2e1a", "BULL — Fully Deployed"
    elif regime == "BEAR":
        colour, bg, label = "#e74c3c", "#2e0d0d", "BEAR — Defensive (10% SPY / 90% IEF)"
    else:
        colour, bg, label = "#94a3b8", "#1e293b", "UNKNOWN — Insufficient Data"

    next_exec = get_next_execution_date(actual_date)
    st.markdown(
        f"""
        <div style="background:{bg};border:2px solid {colour};border-radius:12px;
                    padding:20px;text-align:center;margin-bottom:20px;">
            <div style="font-size:3rem;font-weight:800;color:{colour};">{regime}</div>
            <div style="font-size:1.2rem;font-weight:600;color:{colour};">{label}</div>
            <div style="color:#94a3b8;margin-top:6px;font-size:0.9rem;">
                {actual_date.strftime('%Y-%m-%d')} signal &nbsp;|&nbsp;
                ⏰ Next execution: {next_exec.strftime('%A %Y-%m-%d')} at the open
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Gate metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "SPY vs 200d MA",
        f"{sig['spy_price']:,.0f} / {sig['spy_200ma']:,.0f}",
        delta=f"{(sig['spy_price'] / sig['spy_200ma'] - 1) * 100:+.1f}%" if sig["spy_200ma"] else None,
    )
    c2.metric(
        "NQ100 EW vs 200d MA",
        f"{sig['nq_ew_price']:,.1f} / {sig['nq_ew_200ma']:,.1f}",
        delta=f"{(sig['nq_ew_price'] / sig['nq_ew_200ma'] - 1) * 100:+.1f}%" if sig["nq_ew_200ma"] else None,
    )
    c3.metric("Breadth (positive momentum)", f"{sig['breadth_pct']:.1f}%")
    c4.metric("Dual MA Gate", "✅ OPEN" if sig["bull_gate"] else "❌ CLOSED")
    c5.metric("Breadth Gate (≥40%)", "✅ OPEN" if sig["breadth_gate"] else "❌ CLOSED")

    st.markdown("&nbsp;")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("#### Top-3 NQ100 Momentum Picks")
        if regime == "BULL" and sig["top3"]:
            for i, tkr in enumerate(sig["top3"], 1):
                score = sig["mom_scores"].get(tkr, 0.0)
                st.markdown(
                    f"<div style='background:#1e293b;padding:10px 16px;border-radius:8px;"
                    f"margin-bottom:8px;border-left:4px solid #3b82f6;'>"
                    f"<span style='font-size:1.1rem;font-weight:700;color:#60a5fa;'>#{i} {tkr}</span>"
                    f"<span style='float:right;color:#94a3b8;'>21.7% weight &nbsp;|&nbsp; "
                    f"momentum {score * 100:+.1f}%</span></div>",
                    unsafe_allow_html=True,
                )
        elif regime == "BEAR":
            st.warning("Regime gate closed — momentum picks suspended, defensive allocation active.")
        else:
            st.info("Signal unavailable for this date.")

    with col_r:
        st.markdown("#### Allocation Breakdown")
        alloc = sig.get("allocation", {})
        if alloc:
            palette = {
                "SPY": "#3b82f6", "IEF": "#64748b", "CASH": "#334155",
            }
            labels, values, colors = [], [], []
            for tkr, wt in alloc.items():
                labels.append(tkr)
                values.append(wt * 100)
                colors.append(palette.get(tkr, "#22c55e"))
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values, marker_colors=colors,
                hole=0.45, textinfo="label+percent", textfont_size=11,
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10), height=260,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No allocation available.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — MOMENTUM LEADERBOARD
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🏁 Section 2 — NQ100 Momentum Leaderboard (Top 10)")
    st.caption("Score = 44-week total return − 4-week total return (skip-month momentum).")

    mom = sig.get("mom_scores", {})
    if mom:
        lead = pd.DataFrame(
            [(t, v * 100) for t, v in list(mom.items())[:10]],
            columns=["Ticker", "Momentum %"],
        )
        lead.index = range(1, len(lead) + 1)
        fig_bar = go.Figure(go.Bar(
            x=lead["Momentum %"], y=lead["Ticker"], orientation="h",
            marker_color=["#22c55e" if t in sig["top3"] else "#3b82f6" for t in lead["Ticker"]],
            text=[f"{v:+.1f}%" for v in lead["Momentum %"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
            font_color="#e2e8f0", height=380,
            yaxis=dict(autorange="reversed"),
            xaxis_title="Momentum Score (%)",
            margin=dict(t=20, b=20, l=60, r=40),
        )
        fig_bar.update_xaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Momentum scores unavailable for this date.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — 20-YEAR BACKTEST
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🏆 Section 3 — Backtest: System E vs SPY Buy & Hold")

    with st.spinner("Running System E backtest…"):
        bt = run_backtest(str(actual_date.date()), d)

    if bt is None or bt.empty:
        st.warning("Not enough history to run the backtest.")
    else:
        me = _metrics_from_equity(bt["System_E"])
        mb = _metrics_from_equity(bt["Buy_Hold_SPY"])
        if me:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("System E CAGR", f"{me['CAGR %']:.1f}%",
                      delta=f"{me['CAGR %'] - mb.get('CAGR %', 0):+.1f}% vs SPY")
            m2.metric("Sharpe", f"{me['Sharpe']:.2f}")
            m3.metric("Sortino", f"{me['Sortino']:.2f}")
            m4.metric("Max Drawdown", f"{me['Max DD %']:.1f}%")
            m5.metric("Ann. Volatility", f"{me['Ann. Vol %']:.1f}%")

        fig3 = go.Figure()
        for col, label, color, width in [
            ("System_E", "System E", "#f59e0b", 3),
            ("Buy_Hold_SPY", "SPY Buy & Hold", "#94a3b8", 1.5),
        ]:
            if col in bt.columns:
                fig3.add_trace(go.Scatter(
                    x=bt.index, y=bt[col], name=label,
                    line=dict(color=color, width=width),
                    hovertemplate="%{x|%Y-%m-%d}: $%{y:,.0f}<extra></extra>",
                ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
            font_color="#e2e8f0", yaxis_type="log",
            yaxis_title="Portfolio Value (log scale, $100k start)",
            hovermode="x unified",
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
            height=420, margin=dict(t=40, b=20, l=70, r=20),
        )
        fig3.update_xaxes(gridcolor="#1e293b")
        fig3.update_yaxes(gridcolor="#1e293b", tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Metrics Comparison")
        st.dataframe(
            pd.DataFrame([me, mb], index=["System E", "SPY Buy & Hold"]),
            use_container_width=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — HOW IT WORKS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🧠 Section 4 — How System E Works")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
**Universe & Momentum**

System E scans the NQ100 universe (91-stock list) weekly. Each stock is ranked by
*skip-month momentum*: 44-week total return minus 4-week total return. Skipping the most
recent month avoids buying into short-term mean-reversion.

**Regime Gate (dual 200d MA)**

The strategy only deploys into equities when **both** SPY and the NQ100 equal-weight index
trade above their 200-day moving averages. If either breaks below, the regime flips to BEAR.

**Breadth Gate**

At least 40% of the NQ100 universe must show positive momentum. A narrow rally led by a
handful of names does not qualify — breadth failures force the defensive allocation.
            """
        )
    with col_b:
        st.markdown(
            """
**Allocations**

```
BULL : 35.0% SPY + 21.7% × top-3 momentum picks  (≈ 100%)
BEAR : 10.0% SPY + 90.0% IEF (7–10y Treasuries)
```

**Rebalance Schedule**

Signal computed at **Tuesday close**; positions executed at **Wednesday open** (T+1).
The Tuesday/Wednesday schedule came out of backtest optimisation and is enforced in
`daily_refresh.py`, which only writes the System E cache on its signal day.

**Related pages**

- *R3 Vol-Adjusted Strategy* — continuous position sizing built on the same NQ100 engine
- *System E vs R3 Comparison* — side-by-side signals, equity curves, and metrics
            """
        )


dashboard()

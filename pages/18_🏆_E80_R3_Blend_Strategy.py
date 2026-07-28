"""
pages/18_🏆_E80_R3_Blend_Strategy.py
=====================================
E80/R3 Blend — strategy dashboard + Live Trading cockpit, combined via tabs.

Tab "Strategy & Backtest": Calmar-optimal 80% System E + 20% R3 Vol-Adjusted
  blend, weekly rebalanced. Sections: Live Combined Allocation, 22-Year
  Backtest, How the Blend Works.

Tab "Live Trading": real paper-trading ledger since 2026-07-01, $10,000 start.
  Sections: Action Panel, Portfolio & Risk, Trading Log, Ledger.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="E80/R3 Blend Strategy", page_icon="🏆", layout="wide")

ROOT = Path(__file__).resolve().parent.parent

from utils.data_engine import get_clean_master
from backend.strategies.blend_e_r3_engine import (
    BLEND_WEIGHTS,
    blend_metrics,
    build_blend_curve,
    compute_blend_signal,
)
from backend.strategies.blend_live_ledger import (
    START, START_CAPITAL, build_live_state, _combine, _r3_weights,
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
    "rebalanced weekly. Beats pure System E on drawdown and Calmar while keeping ~92% of its CAGR. "
    "**Strategy & Backtest** below shows the 22-year history; **Live Trading** tracks a real paper "
    "book since 2026-07-01."
)
st.caption(
    "Realistic assumptions: sleeves embed T+1 execution (Sys E Tue→Wed, R3 Thu→Fri) and 0.16% "
    "round-trip position costs; the blend layer pays an extra 10 bps per unit of rebalancing turnover."
)

# ── Load market data once, shared by both tabs ─────────────────────────────────
data = load_data()
if data is None or data.empty:
    st.error("⚠️ Failed to load market data.")
    st.stop()

tab_strategy, tab_live = st.tabs(["📊 Strategy & Backtest", "💼 Live Trading"])

with tab_strategy:
    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 1 — LIVE COMBINED ALLOCATION
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📡 Section 1 — Live Combined Allocation")

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


with tab_live:
    st.markdown(
        f"Real-money tracking of the **E80/R3 Blend** — started **{START.date()}** "
        f"with **${START_CAPITAL:,.0f}**. Signals: Sys E Tue close → Wed, R3 Thu close → Fri (T+1)."
    )
    st.caption(
        "Fills approximated at execution-day close (master data stores closes). "
        "Costs 0.08%/side. Stops are indicative 2×ATR(14) trailing guides — "
        "the system itself exits on signals, not stops."
    )

    state = build_live_state(data)
    summ = state["summary"]
    trades = state["trades"]
    ledger = state["ledger"]
    positions = state["positions"]
    pending = state["pending"]

    # ── Effective target (apply pending signals on top of executed state) ─────────
    w_e_eff, w_r3_eff = dict(state["w_e"]), dict(state["w_r3"])
    for p in pending:
        if p["sleeve"] == "R3":
            try:
                snap = json.loads((ROOT / "data" / "r3_signal_cache.json").read_text())
                w_r3_eff = _r3_weights(snap, set(data.columns))
            except Exception:
                pass
        else:  # SYS_E
            from strategy_e.engine import compute_system_e_signal
            sig_p = compute_system_e_signal(data)
            if sig_p.get("allocation"):
                w_e_eff = sig_p["allocation"]
    target_eff = _combine(w_e_eff, w_r3_eff)

    sig_e = state["sys_e_sig"]
    regime = sig_e.get("regime", "UNKNOWN")
    try:
        r3_snap = json.loads((ROOT / "data" / "r3_signal_cache.json").read_text())
        r3_deploy = float(r3_snap.get("deploy_pct", 50.0))
    except Exception:
        r3_snap, r3_deploy = {}, 50.0

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 1 — ACTION PANEL
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🎯 Section 1 — Action Panel")

    colour = {"BULL": "#2ecc71", "BEAR": "#e74c3c"}.get(regime, "#94a3b8")
    bg = {"BULL": "#0d2e1a", "BEAR": "#2e0d0d"}.get(regime, "#1e293b")
    st.markdown(
        f"""
        <div style="background:{bg};border:2px solid {colour};border-radius:12px;
                    padding:16px;text-align:center;margin-bottom:14px;">
            <div style="font-size:2rem;font-weight:800;color:{colour};">
                System E (80%): {regime} &nbsp;·&nbsp; R3 (20%) Deploy: {r3_deploy:.0f}%
            </div>
            <div style="color:#94a3b8;margin-top:6px;font-size:0.9rem;">
                As of {summ['as_of']} &nbsp;|&nbsp; equity exposure {summ['equity_exposure_pct']:.0f}%
                &nbsp;|&nbsp; next signals: Sys E Tue close, R3 Thu close
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if pending:
        for p in pending:
            st.warning(
                f"⏳ **Pending execution — {p['sleeve'].replace('_', ' ')}** "
                f"(signal {p['signal_date']}): {p['note']}. "
                f"Execute at the **next market open**; actions below already include it."
            )
    else:
        st.success("✅ No pending executions — portfolio is in line with the latest signals.")

    # Today's actions: current holdings vs effective target
    last_day = data.index[-1]
    pv = summ["equity"]
    shares_now = state["shares"]
    rows = []
    for t in sorted(set(shares_now) | {k for k in target_eff if k != "CASH"}):
        if t not in data.columns:
            continue
        p_last = float(data[t].iloc[-1])
        cur_sh = shares_now.get(t, 0.0)
        cur_val = cur_sh * p_last
        tgt_w = target_eff.get(t, 0.0)
        tgt_val = tgt_w * pv
        dv = tgt_val - cur_val
        if abs(dv) < max(1.0, 0.002 * pv):
            action, icon = "HOLD", "⏸️"
        elif dv > 0:
            action, icon = "BUY", "🟢"
        else:
            action, icon = "SELL", "🔴"
        rows.append({
            "Action": f"{icon} {action}",
            "Ticker": t,
            "Target %": round(tgt_w * 100, 1),
            "Now %": round(cur_val / pv * 100, 1),
            "Δ Shares": round(dv / p_last, 3),
            "Δ Value $": round(dv, 0),
            "Ref Price": round(p_last, 2),
        })
    act_df = pd.DataFrame(rows).sort_values("Δ Value $")
    n_act = (~act_df["Action"].str.contains("HOLD")).sum()

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"#### Today's Actions ({n_act} trade{'s' if n_act != 1 else ''} needed)")
        st.dataframe(act_df.set_index("Ticker"), use_container_width=True)
        st.caption("Δ = trade needed to reach the effective target (executed + pending signals) "
                   "at last close. Sized on current equity.")
    with c2:
        st.markdown("#### Effective Target Allocation")
        fig_pie = go.Figure(go.Pie(
            labels=list(target_eff.keys()),
            values=[w * 100 for w in target_eff.values()],
            marker_colors=["#3b82f6" if t == "SPY" else "#64748b" if t in ("IEF", "CASH")
                           else "#22c55e" for t in target_eff],
            hole=0.45, textinfo="label+percent", textfont_size=11,
        ))
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                              showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 2 — PORTFOLIO, RISK & FLOATING P&L
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📊 Section 2 — Portfolio & Risk (daily refresh)")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Equity", f"${summ['equity']:,.0f}",
              delta=f"{summ['total_pnl_pct']:+.2f}% since start")
    k2.metric("Floating P&L (day)", f"${summ['daily_pnl']:+,.0f}")
    k3.metric("Total P&L", f"${summ['total_pnl']:+,.0f}")
    k4.metric("Cash", f"${summ['cash']:,.0f}")
    k5.metric("Max Drawdown", f"{summ['max_dd_pct']:.2f}%")
    k6.metric("Costs Paid", f"${summ['costs_paid']:,.2f}")

    if not positions.empty:
        st.markdown("#### Open Positions — entry, stop & floating P&L")
        pos_show = positions.rename(columns={
            "Target_W_%": "Target %", "Actual_W_%": "Now %", "Avg_Entry": "Avg Entry",
            "Last_Price": "Last", "Market_Value": "Mkt Value $",
            "Float_PnL_$": "Float P&L $", "Float_PnL_%": "Float P&L %",
            "Stop_(2xATR14)": "Stop (2×ATR)", "Stop_Dist_%": "Stop Dist %",
            "Entry_Date": "Entry Date",
        }).set_index("Ticker")
        st.dataframe(
            pos_show.style.map(
                lambda v: "color:#2ecc71" if isinstance(v, (int, float)) and v > 0
                else "color:#e74c3c" if isinstance(v, (int, float)) and v < 0 else "",
                subset=["Float P&L $", "Float P&L %"],
            ),
            use_container_width=True,
        )
        st.caption("Stop = trailing (highest close since entry − 2×ATR-proxy(14)). "
                   "Indicative risk guide only — not a system rule.")
    else:
        st.info("No open positions.")

    if not ledger.empty:
        eq = ledger.set_index("Date")["Total_Equity"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Equity",
                                 line=dict(color="#a855f7", width=2.5),
                                 hovertemplate="%{x|%Y-%m-%d}: $%{y:,.0f}<extra></extra>"))
        fig.add_hline(y=START_CAPITAL, line_dash="dot", line_color="#64748b",
                      annotation_text=f"start ${START_CAPITAL:,.0f}")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
                          font_color="#e2e8f0", height=300, yaxis_title="Equity ($)",
                          margin=dict(t=20, b=20, l=60, r=20))
        fig.update_xaxes(gridcolor="#1e293b")
        fig.update_yaxes(gridcolor="#1e293b", tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 3 — TRADING LOG
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📒 Section 3 — Trading Log")

    if not trades.empty:
        tl = trades.copy()
        tl["Date"] = pd.to_datetime(tl["Date"]).dt.date
        st.dataframe(tl.sort_values(["Date", "Action"], ascending=[False, True])
                     .set_index("Date"), use_container_width=True)
        st.download_button("⬇️ Download trades.csv", tl.to_csv(index=False),
                           "trades.csv", "text/csv")
        st.caption(f"{len(tl)} trades · realized P&L ${summ['realized_pnl']:+,.2f} · "
                   f"total costs ${summ['costs_paid']:,.2f}. "
                   "Also saved to data/live_trading/trades.csv.")
    else:
        st.info("No trades executed yet.")

    # ═══════════════════════════════════════════════════════════════════════════════
    # SECTION 4 — LEDGER
    # ═══════════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📚 Section 4 — Ledger (daily statement)")

    if not ledger.empty:
        lg = ledger.copy()
        lg["Date"] = pd.to_datetime(lg["Date"]).dt.date
        st.dataframe(
            lg.rename(columns={
                "Cash_Flow": "Cash Flow $", "Cash_Balance": "Cash $",
                "Positions_Value": "Positions $", "Total_Equity": "Equity $",
                "Daily_PnL": "Daily P&L $", "Total_PnL": "Total P&L $",
            }).sort_values("Date", ascending=False).set_index("Date"),
            use_container_width=True,
        )
        st.download_button("⬇️ Download ledger.csv", lg.to_csv(index=False),
                           "ledger.csv", "text/csv")
        st.caption("Marked to market at each day's close since 2026-07-01. "
                   "Also saved to data/live_trading/ledger.csv.")
    else:
        st.info("Ledger empty — no market days since start yet.")


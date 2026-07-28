"""
pages/22_🔀_Hybrid_AD_Live_Trading.py
========================================
Hybrid A/D — LIVE trading cockpit (paper ledger since 2026-07-01, $10,000).

Combines the E80/R3 blend target with a vol-target exposure overlay: full
exposure in Strategy A (bull/low-vol), scaled exposure in Strategy D
(bear/high-vol, SPY 8wk vol >= 15% for 2 consecutive weeks).

Sections:
  1. Action Panel   — regime badge, sleeve signals, today's BUY/HOLD/SELL
  2. Portfolio      — positions, entry/stop/floating P&L, risk, equity curve
  3. Trading Log    — every executed trade
  4. Ledger         — daily cash & equity statement
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Hybrid A/D Live Trading", page_icon="🔀", layout="wide")

ROOT = Path(__file__).resolve().parents[1]

from utils.data_engine import get_clean_master
from backend.strategies.hybrid_ad_live_ledger import (
    START, START_CAPITAL, VOL_THRESHOLD, build_hybrid_live_state,
)


@st.cache_data(ttl=3600)
def load_data():
    return get_clean_master()


with st.sidebar:
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    render_master_controls()
    render_ecosystem_sidebar()


st.title("🔀 Hybrid A/D Live Trading")
st.markdown(
    f"Real-money tracking of the **Hybrid A/D** strategy — started **{START.date()}** "
    f"with **${START_CAPITAL:,.0f}**. Full exposure to the 80% Sys E / 20% R3 blend in "
    f"**Strategy A** (bull/low-vol); scaled exposure in **Strategy D** once SPY 8-week "
    f"realized vol is ≥ {VOL_THRESHOLD*100:.0f}% for 2 consecutive Friday closes."
)
st.caption(
    "Signals: Sys E Tue close → Wed, R3 Thu close → Fri, vol regime Fri close → Mon (T+1). "
    "Fills approximated at execution-day close. Costs 0.08%/side. Stops are indicative "
    "2×ATR(14) trailing guides — the system itself exits on signals, not stops."
)

data = load_data()
if data is None or data.empty:
    st.error("⚠️ Failed to load market data.")
    st.stop()

state = build_hybrid_live_state(data)
summ = state["summary"]
trades = state["trades"]
ledger = state["ledger"]
positions = state["positions"]
pending = state["pending"]
target = state["target"]
expo = state["expo"]
in_defensive = summ.get("in_defensive", False)
vol_now = summ.get("vol_now", np.nan)
vol_prev = summ.get("vol_prev", np.nan)

sig_e = state["sys_e_sig"]
regime = sig_e.get("regime", "UNKNOWN")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — ACTION PANEL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🎯 Section 1 — Action Panel")

mode_colour = "#f59e0b" if in_defensive else "#2ecc71"
mode_bg = "#2e2410" if in_defensive else "#0d2e1a"
mode_label = "STRATEGY D — VOL-TARGET ACTIVE" if in_defensive else "STRATEGY A — FULL EXPOSURE"
vol_str = f"{vol_now*100:.1f}%" if not np.isnan(vol_now) else "—"
vol_prev_str = f"{vol_prev*100:.1f}%" if not np.isnan(vol_prev) else "—"

st.markdown(
    f"""
    <div style="background:{mode_bg};border:2px solid {mode_colour};border-radius:12px;
                padding:16px;text-align:center;margin-bottom:14px;">
        <div style="font-size:1.7rem;font-weight:800;color:{mode_colour};">
            {mode_label}
        </div>
        <div style="color:#94a3b8;margin-top:6px;font-size:0.9rem;">
            SPY 8wk vol {vol_str} (prior wk {vol_prev_str}) vs {VOL_THRESHOLD*100:.0f}% threshold
            &nbsp;|&nbsp; exposure ×{expo:.2f}
            &nbsp;|&nbsp; Sys E (80%): {regime} · R3 (20%) deploy present in blend
        </div>
        <div style="color:#64748b;margin-top:4px;font-size:0.8rem;">
            As of {summ['as_of']} · next checks: Sys E Tue close, R3 Thu close, vol regime Fri close
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
            f"Executes at the **next market open**; actions below already include it."
        )
else:
    st.success("✅ No pending executions — portfolio is in line with the latest signals.")

# Today's actions: current holdings vs effective target
pv = summ["equity"]
shares_now = state["shares"]
rows = []
for t in sorted(set(shares_now) | {k for k in target if k != "CASH"}):
    if t not in data.columns:
        continue
    p_last = float(data[t].iloc[-1])
    cur_sh = shares_now.get(t, 0.0)
    cur_val = cur_sh * p_last
    tgt_w = target.get(t, 0.0)
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
        "Now %": round(cur_val / pv * 100, 1) if pv else 0.0,
        "Δ Shares": round(dv / p_last, 3),
        "Δ Value $": round(dv, 0),
        "Ref Price": round(p_last, 2),
    })
act_df = pd.DataFrame(rows).sort_values("Δ Value $") if rows else pd.DataFrame(
    columns=["Action", "Ticker", "Target %", "Now %", "Δ Shares", "Δ Value $", "Ref Price"])
n_act = int((~act_df["Action"].str.contains("HOLD")).sum()) if not act_df.empty else 0

c1, c2 = st.columns([3, 2])
with c1:
    st.markdown(f"#### Today's Actions ({n_act} trade{'s' if n_act != 1 else ''} needed)")
    if not act_df.empty:
        st.dataframe(act_df.set_index("Ticker"), use_container_width=True)
    else:
        st.info("No target allocation yet.")
    st.caption(f"Δ = trade needed to reach the current target (blend × exposure ×{expo:.2f}) "
               "at last close. Sized on current equity. Cash target: "
               f"{summ.get('cash_target_pct', 0):.1f}% (de-leverage overlay, not moved to bonds).")
with c2:
    st.markdown("#### Current Target Allocation")
    if target:
        labels = list(target.keys())
        values = [w * 100 for w in target.values()]
        cash_pct = max(0.0, 100.0 - sum(values))
        if cash_pct > 0.5:
            labels = labels + ["CASH"]
            values = values + [cash_pct]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            marker_colors=["#3b82f6" if t == "SPY" else "#64748b" if t in ("IEF", "CASH")
                           else "#22c55e" for t in labels],
            hole=0.45, textinfo="label+percent", textfont_size=11,
        ))
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                              showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No target allocation yet.")

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
k6.metric("Exposure Scalar", f"×{expo:.2f}", delta="DEFENSIVE" if in_defensive else "FULL",
          delta_color="inverse" if in_defensive else "off")

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
                             line=dict(color="#38bdf8", width=2.5),
                             hovertemplate="%{x|%Y-%m-%d}: $%{y:,.0f}<extra></extra>"))
    fig.add_hline(y=START_CAPITAL, line_dash="dot", line_color="#64748b",
                  annotation_text=f"start ${START_CAPITAL:,.0f}")

    # Shade defensive periods
    reg_hist = state.get("regime_history") or []
    if reg_hist:
        reg_df = pd.DataFrame(reg_hist).set_index("Date")
        bear_start = None
        dts = list(reg_df.index) + [eq.index[-1]]
        flags = list(reg_df["In_Defensive"]) + [reg_df["In_Defensive"].iloc[-1] if len(reg_df) else False]
        for dtx, is_def in zip(dts, flags):
            if is_def and bear_start is None:
                bear_start = dtx
            elif not is_def and bear_start is not None:
                fig.add_vrect(x0=bear_start, x1=dtx, fillcolor="rgba(245,158,11,0.12)",
                             line_width=0, layer="below")
                bear_start = None
        if bear_start is not None:
            fig.add_vrect(x0=bear_start, x1=eq.index[-1], fillcolor="rgba(245,158,11,0.12)",
                         line_width=0, layer="below")

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
                      font_color="#e2e8f0", height=300, yaxis_title="Equity ($)",
                      margin=dict(t=20, b=20, l=60, r=20))
    fig.update_xaxes(gridcolor="#1e293b")
    fig.update_yaxes(gridcolor="#1e293b", tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟡 Shaded = Strategy D (vol-target) active periods.")

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
                       "hybrid_ad_trades.csv", "text/csv")
    st.caption(f"{len(tl)} trades · realized P&L ${summ['realized_pnl']:+,.2f} · "
               f"total costs ${summ['costs_paid']:,.2f}. "
               "Also saved to data/live_trading_hybrid_ad/trades.csv.")
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
                       "hybrid_ad_ledger.csv", "text/csv")
    st.caption("Marked to market at each day's close since 2026-07-01. "
               "Also saved to data/live_trading_hybrid_ad/ledger.csv.")
else:
    st.info("Ledger empty — no market days since start yet.")

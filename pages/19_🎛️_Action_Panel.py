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
from backend.strategies.blend_live_ledger import build_live_state
from backend.strategies.allocator_engine import load_allocator_json
from backend.strategies.ensemble_top100_engine import load_ensemble_top100_json


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


# ── Priority tiers (drive sort order + the "Urgency" badge) ───────────────────
# 0 = Act now (real position drift vs a live/tracked book)
# 1 = Pending execution (signal already locked, trade goes in at next open)
# 2 = Model rebalance / regime switch (target changed, but not diffed vs a live book)
# 3 = Monitor only (static weights, indicator-only, or no live signal exposed)
P_ACT, P_PENDING, P_REBAL, P_MONITOR = 0, 1, 2, 3

ACTION_BADGE = {
    "BUY": "🟢 BUY",
    "SELL": "🔴 SELL",
    "TRIM": "🟡 TRIM",
    "ADD": "🟢 ADD",
    "HOLD": "⚪ HOLD",
    "DEFENSIVE": "🛡️ DEFENSIVE",
    "REBALANCE": "🔵 REBALANCE",
    "PENDING": "⏳ PENDING",
    "INFO": "◽ INFO",
}


def action_row(strategy, action, ticker, size, reason, risk, timing, priority):
    return {
        "Pri": priority,
        "Strategy": strategy,
        "Action": ACTION_BADGE.get(action, action),
        "Ticker / Allocation": ticker,
        "Size / Weight": size,
        "Reason": reason,
        "Risk": risk,
        "Signal → Exec": timing,
    }


@st.cache_data(ttl=1800)
def load_allocator_live(sel_dt):
    try:
        return load_allocator_json(sel_dt)
    except Exception:
        return None


@st.cache_data(ttl=1800)
def load_ensemble_live(sel_dt):
    try:
        return load_ensemble_top100_json(sel_dt)
    except Exception:
        return None


@st.cache_data(ttl=1800)
def load_live_blend_state(_df):
    return build_live_state(_df)


def _fallback_row(name: str, exc: Exception) -> dict:
    """Guarantee a strategy still shows up in the panel even when its signal
    computation throws or its data isn't available. Placed at Monitor
    priority (bottom of the urgency ranking) since we don't know the real
    urgency — but the strategy is never silently dropped from the table."""
    return action_row(
        name, "INFO", "—", "—",
        f"Signal temporarily unavailable ({exc})",
        "—", "—", P_MONITOR,
    )


def build_action_rows(data: pd.DataFrame) -> list[dict]:
    """Aggregate today's actionable item(s) from every strategy in the project
    into one flat list of rows. Every strategy in the ecosystem contributes
    exactly one or more rows, always — if a signal computation fails or its
    data isn't available, it still appears as an INFO row (via _fallback_row)
    rather than disappearing from the table. Failures in one strategy never
    block the others. Rows are ranked by urgency/time-to-action (Pri: 0=Act
    now … 3=Monitor only), so the busiest strategies always float to the top."""
    rows: list[dict] = []
    sel_dt = pd.Timestamp(dt.date.today())

    # ── 1) E80/R3 Live Trading — the only strategy with a real tracked book,
    #        so this is the only place true BUY/SELL/TRIM drift comes from. ──
    try:
        state = load_live_blend_state(data)
        positions = state.get("positions")
        summary = state.get("summary", {})
        sig_e = state.get("sys_e_sig", {}) or {}
        if positions is not None and not positions.empty:
            for _, p in positions.iterrows():
                tgt, act = float(p["Target_W_%"]), float(p["Actual_W_%"])
                drift = tgt - act
                if abs(drift) < 2.0:
                    action, pri = "HOLD", P_MONITOR
                elif drift > 0:
                    action, pri = "ADD", P_ACT
                else:
                    action, pri = "TRIM", P_ACT
                rows.append(action_row(
                    "💼 E80/R3 Live", action, p["Ticker"],
                    f"{act:.1f}% → target {tgt:.1f}% ({drift:+.1f}pp)",
                    f"{sig_e.get('regime', '?')} · equity exposure {summary.get('equity_exposure_pct', 0):.0f}%",
                    f"Stop {p['Stop_(2xATR14)']:.2f} ({p['Stop_Dist_%']:.1f}% away)",
                    "Live book (T+1 executed)", pri,
                ))
        for p in state.get("pending", []):
            rows.append(action_row(
                "💼 E80/R3 Live", "PENDING", p["sleeve"].replace("_", " "),
                "—", p["note"], "Executes at next open — size unconfirmed until fill",
                f"Signal {p['signal_date']} → next open", P_PENDING,
            ))
        if not state.get("pending") and (positions is None or positions.empty):
            rows.append(action_row(
                "💼 E80/R3 Live", "INFO", "No open positions",
                "—", "Ledger has no live trades yet", "—", "—", P_MONITOR,
            ))
    except Exception as exc:
        st.caption(f"⚠️ E80/R3 Live Trading signal unavailable: {exc}")
        rows.append(_fallback_row("💼 E80/R3 Live", exc))

    # ── 2) SPY+QQQ+GLD Allocator — engine already emits buy/sell action_items ─
    try:
        alloc = load_allocator_live(sel_dt)
        if alloc:
            items = alloc.get("action_items") or []
            risk_note = alloc.get("risk_note", "") or alloc.get("state", "")
            if items:
                for it in items:
                    action = "BUY" if it.get("tone") == "buy" else "SELL"
                    rows.append(action_row(
                        "🎯 SPY+QQQ+GLD Allocator", action, it.get("label", "—"),
                        it.get("detail", "—"), alloc.get("combined", risk_note),
                        risk_note or "—", "Model rebalance (daily close)", P_ACT,
                    ))
            else:
                rows.append(action_row(
                    "🎯 SPY+QQQ+GLD Allocator", "HOLD", "On target",
                    fmt_alloc({a["symbol"]: a["current"] / 100 for a in alloc.get("assets", [])}, pct_already=False),
                    alloc.get("combined", "—"), risk_note or "—",
                    "Model rebalance (daily close)", P_MONITOR,
                ))
    except Exception as exc:
        st.caption(f"⚠️ Allocator signal unavailable: {exc}")
        rows.append(_fallback_row("🎯 SPY+QQQ+GLD Allocator", exc))

    # ── 3) Ensemble Top-100 ETF — same action_items pattern ───────────────────
    try:
        ens = load_ensemble_live(sel_dt)
        if ens:
            items = ens.get("action_items") or []
            if items:
                for it in items:
                    action = "BUY" if it.get("tone") == "buy" else "SELL"
                    rows.append(action_row(
                        "🧩 Ensemble Top-100 ETF", action, it.get("label", "—"),
                        it.get("detail", "—"), "Rank/momentum rotation",
                        "—", "Model rebalance (weekly)", P_ACT,
                    ))
            else:
                rows.append(action_row(
                    "🧩 Ensemble Top-100 ETF", "HOLD", "On target",
                    fmt_alloc({a["symbol"]: a["current"] / 100 for a in ens.get("assets", [])}, pct_already=False),
                    "Rank/momentum rotation", "—",
                    "Model rebalance (weekly)", P_MONITOR,
                ))
    except Exception as exc:
        st.caption(f"⚠️ Ensemble Top-100 signal unavailable: {exc}")
        rows.append(_fallback_row("🧩 Ensemble Top-100 ETF", exc))

    # ── 4) System E (standalone model target — not diffed vs a live book) ─────
    try:
        bsig = compute_blend_signal(data)
        sig_e = bsig["sys_e"]
        r3 = bsig["r3"] or {}
        regime = sig_e.get("regime", "?")
        rows.append(action_row(
            "🚂 System E", "DEFENSIVE" if regime == "BEAR" else "REBALANCE",
            fmt_alloc(sig_e.get("allocation", {})),
            f"Breadth {sig_e.get('breadth_pct', 0):.0f}%",
            f"{regime} · bull gate {sig_e.get('bull_gate')} · breadth gate {sig_e.get('breadth_gate')}",
            "Binary all-in/all-out — no partial sizing",
            f"Tue close → {next_weekday(2).strftime('%a %b %d')} open", P_REBAL,
        ))

        # ── 5) R3 Vol-Adjusted (from cached Thursday deploy snapshot) ─────────
        deploy = float(r3.get("deploy_pct", 0)) if r3 else 0.0
        if deploy >= 70:
            r3_action, r3_pri = "ADD", P_REBAL
        elif deploy >= 40:
            r3_action, r3_pri = "TRIM", P_REBAL
        else:
            r3_action, r3_pri = "DEFENSIVE", P_REBAL
        rows.append(action_row(
            "📊 R3 Vol-Adjusted", r3_action if r3 else "INFO",
            f"SPY 35% + top-3 ({', '.join(r3.get('top3_picks', [])[:3])})" if r3 else "No cache",
            f"Deploy {deploy:.0f}%, rest IEF" if r3 else "Run daily_refresh (Thu) to update",
            f"Trap-score sizing, continuous 10–100% deploy",
            "Continuously sized — lower risk than binary Sys E",
            f"Thu close → {next_weekday(4).strftime('%a %b %d')} open", r3_pri if r3 else P_MONITOR,
        ))

        # ── 6) E80/R3 Blend (theoretical 80/20 model target, separate from the
        #        live-tracked book in row #1) ──────────────────────────────────
        rows.append(action_row(
            "🏆 E80/R3 Blend (Model)", "REBALANCE",
            fmt_alloc(bsig.get("allocation", {})),
            "80% Sys E / 20% R3 combined",
            f"Sys E {regime} · R3 deploy {bsig['r3_deploy_pct']:.0f}%",
            "See 💼 E80/R3 Live for the actual tracked book",
            f"{next_weekday(2).strftime('%a')} + {next_weekday(4).strftime('%a')} legs", P_REBAL,
        ))
    except Exception as exc:
        st.caption(f"⚠️ System E / R3 / Blend model signals unavailable: {exc}")
        # Fall back individually so one shared failure doesn't silently drop
        # three strategies from the panel at once.
        rows.append(_fallback_row("🚂 System E", exc))
        rows.append(_fallback_row("📊 R3 Vol-Adjusted", exc))
        rows.append(_fallback_row("🏆 E80/R3 Blend (Model)", exc))

    # ── 7) Hybrid A/D — vol-regime mode switch ─────────────────────────────────
    try:
        VOL_THRESHOLD, VOL_WEEKS = 0.15, 8
        spy_weekly = data["SPY"].resample("W-FRI").last().dropna()
        spy_ret = spy_weekly.pct_change().dropna()
        if len(spy_ret) >= VOL_WEEKS:
            vol = float(spy_ret.rolling(VOL_WEEKS).std().iloc[-1] * np.sqrt(52))
            in_defensive = vol >= VOL_THRESHOLD
            rows.append(action_row(
                "🔀 Hybrid A/D", "DEFENSIVE" if in_defensive else "HOLD",
                "Strategy D (vol-target overlay)" if in_defensive else "Strategy A (full 80/20 exposure)",
                "—",
                f"SPY 8wk realized vol {vol * 100:.1f}% vs {VOL_THRESHOLD * 100:.0f}% threshold",
                "Mode switch only — reduces sizing, doesn't change tickers",
                "Weekly (Friday close check)", P_REBAL if in_defensive else P_MONITOR,
            ))
        else:
            # Not enough weekly history to compute realized vol yet — still
            # show the strategy, just flagged as insufficient data rather
            # than silently omitted.
            rows.append(action_row(
                "🔀 Hybrid A/D", "INFO", "—", "—",
                f"Insufficient weekly history ({len(spy_ret)} < {VOL_WEEKS} weeks) to compute vol regime",
                "—", "Weekly (Friday close check)", P_MONITOR,
            ))
    except Exception as exc:
        st.caption(f"⚠️ Hybrid A/D signal unavailable: {exc}")
        rows.append(_fallback_row("🔀 Hybrid A/D", exc))

    # ── 8) 200MA Strategy — classic trend rule (index proxy) ───────────────────
    try:
        spy = data["SPY"].dropna()
        ma200 = spy.rolling(200).mean().iloc[-1]
        above = spy.iloc[-1] > ma200
        dd = (spy.iloc[-1] / spy.cummax().iloc[-1] - 1) * 100
        rows.append(action_row(
            "📈 200MA Strategy", "HOLD" if above else "DEFENSIVE",
            "SPY 100%" if above else "Cash / T-Bills 100%",
            "—",
            f"SPY {'ABOVE' if above else 'BELOW'} 200d MA ({spy.iloc[-1]:,.0f} vs {ma200:,.0f})",
            f"{dd:.1f}% off all-time high",
            "Daily close → next open", P_MONITOR if above else P_REBAL,
        ))
    except Exception as exc:
        rows.append(_fallback_row("📈 200MA Strategy", exc))

    # ── 9) Platinum / F-TAA — latest model weights (informational; no live
    #        drift tracking exists yet for these two, see Strategies Dashboard) ─
    plat_w = latest_weights("data/Platinum_Results/Platinum_Weights.csv")
    if plat_w:
        rows.append(action_row(
            "💎 Platinum", "INFO", fmt_alloc(plat_w, pct_already=True),
            f"{len(plat_w)} positions", "Golden Ratio (75%) + Ah_Pig (25%) sleeves",
            "No live drift tracking — see Strategies Dashboard",
            "Daily model → next open", P_MONITOR,
        ))
    else:
        rows.append(action_row(
            "💎 Platinum", "INFO", "—", "—",
            "Platinum_Weights.csv not found — run daily_refresh.py to populate",
            "—", "Daily model → next open", P_MONITOR,
        ))
    fta_w = latest_weights("data/Fund_Tactical_Results/Fund_Tactical_weights.csv")
    if fta_w:
        rows.append(action_row(
            "🧭 F-TAA", "INFO", fmt_alloc(fta_w, pct_already=True),
            f"{len(fta_w)} positions", "Weekly regime-tilted tactical rotation",
            "No live drift tracking — see Strategies Dashboard",
            "Weekly model → next open", P_MONITOR,
        ))
    else:
        rows.append(action_row(
            "🧭 F-TAA", "INFO", "—", "—",
            "Fund_Tactical_weights.csv not found — run daily_refresh.py to populate",
            "—", "Weekly model → next open", P_MONITOR,
        ))

    # ── 10) NTSX — fixed target with rebalance bands (no callable live signal
    #         yet; see backend/strategies/ntsx_engine.py) ─────────────────────
    rows.append(action_row(
        "🛡️ NTSX", "INFO", "NTSX 55% / AVWS 12% / KMLM 33%",
        "Rebalance on band breach", "90/60 balanced core, multi-indicator gate",
        "No live drift tracking — see Strategies Dashboard",
        "Band-triggered", P_MONITOR,
    ))

    # ── 11) Best Mix — static optimizer weights, monthly rebalance cadence ────
    try:
        wpath = ROOT / "data" / "Strategy_Comparisons" / "strategy_mix_key_portfolios.csv"
        if wpath.exists():
            wdf = pd.read_csv(wpath, index_col=0)
            w = wdf.loc["best_calmar"]
            today = dt.date.today()
            last_day_of_month = (pd.Timestamp(today) + pd.offsets.MonthEnd(0)).date()
            due = (last_day_of_month - today).days <= 2
            rows.append(action_row(
                "🥇 Best Mix", "REBALANCE" if due else "HOLD",
                f"NTSX {w['NTSX %']:.0f}% / Platinum {w['Platinum %']:.0f}% / F-TAA {w['F-TAA %']:.0f}%",
                "Fixed weights (Calmar-optimal grid search)",
                "Static blend — no daily regime signal",
                "Not diffed vs a live book",
                "Monthly rebalance (month-end)", P_REBAL if due else P_MONITOR,
            ))
        else:
            rows.append(action_row(
                "🥇 Best Mix", "INFO", "—", "—",
                "strategy_mix_key_portfolios.csv not found — run daily_refresh.py to populate",
                "—", "Monthly rebalance (month-end)", P_MONITOR,
            ))
    except Exception as exc:
        st.caption(f"⚠️ Best Mix row unavailable: {exc}")
        rows.append(_fallback_row("🥇 Best Mix", exc))

    # ── 12) No machine-readable live signal exists yet for these — informational
    #         placeholders so the panel is a complete strategy inventory ────────
    rows.append(action_row(
        "🚀 Market Pulse / FTD", "INFO", "See page",
        "—", "Static HTML dashboard embed — no Python-computable signal yet",
        "—", "—", P_MONITOR,
    ))
    rows.append(action_row(
        "🧭 Market Stage Model", "INFO", "See page",
        "—", "Per-ticker stage classifier (Accumulation/Distribution/etc.) — "
             "needs OHLCV, not wired into this panel yet",
        "—", "—", P_MONITOR,
    ))

    return rows


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
st.subheader("📋 Section 1 — Today's Actions (every strategy in the project)")
today_name = dt.date.today().strftime("%A, %b %d %Y")
st.caption(
    f"As of {today_name}. Sorted by urgency: 🔴 **Act now** (real drift on a live-tracked book) → "
    "⏳ **Pending** (signal locked, trade goes in at next open) → 🔵 **Model rebalance** (target changed but "
    "not diffed vs a live book) → ◽ **Monitor** (static / informational). All timings are T+1 — "
    "signals lock at the close, trades execute at the next open."
)

action_rows = build_action_rows(data)
panel_df = pd.DataFrame(action_rows).sort_values("Pri").reset_index(drop=True)

urgency_label = {P_ACT: "🔴 Act now", P_PENDING: "⏳ Pending", P_REBAL: "🔵 Model rebalance", P_MONITOR: "◽ Monitor"}
panel_df.insert(0, "Urgency", panel_df["Pri"].map(urgency_label))
panel_df = panel_df.drop(columns=["Pri"])

n_act = int((panel_df["Urgency"] == "🔴 Act now").sum())
n_pending = int((panel_df["Urgency"] == "⏳ Pending").sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("🔴 Act now", n_act)
c2.metric("⏳ Pending", n_pending)
c3.metric("🔵 Model rebalance", int((panel_df["Urgency"] == "🔵 Model rebalance").sum()))
c4.metric("◽ Monitor only", int((panel_df["Urgency"] == "◽ Monitor").sum()))

if n_act:
    st.warning(f"⚡ **{n_act} item{'s' if n_act != 1 else ''} need real trades today** — see 🔴 Act now rows below.")
else:
    st.success("✅ No live-book drift beyond threshold — nothing forces a trade right now.")

urgency_filter = st.multiselect(
    "Filter by urgency",
    options=list(urgency_label.values()),
    default=list(urgency_label.values()),
)
strategy_filter = st.multiselect(
    "Filter by strategy",
    options=sorted(panel_df["Strategy"].unique().tolist()),
    default=sorted(panel_df["Strategy"].unique().tolist()),
)
shown = panel_df[panel_df["Urgency"].isin(urgency_filter) & panel_df["Strategy"].isin(strategy_filter)]

st.dataframe(
    shown.set_index("Urgency"),
    use_container_width=True,
    height=min(560, 46 + 35 * max(1, len(shown))),
)
st.caption(
    "**Ticker / Allocation** = what to hold/trade. **Size / Weight** = target weight or drift magnitude. "
    "**Reason** = the regime/score driving the call. **Risk** = stop distance, gate status, or a caveat "
    "worth knowing before acting. Rows marked ◽ INFO have no machine-computable live signal yet — see the "
    "linked page for the current read."
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COMPARISON CHART
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📈 Section 2 — Strategy Comparison (common window, $100k start, log scale)")

MISSING_FROM_COMPARISON: list[str] = []

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
            else:
                MISSING_FROM_COMPARISON.append(label)
    else:
        MISSING_FROM_COMPARISON += ["Platinum", "NTSX", "F-TAA"]

    # Hybrid A/D — the live H8b_SPYvol>=15pct column from the debounce study
    # matches the exact rule used in Section 1's live signal (15% vol threshold).
    try:
        hybrid_csv = ROOT / "exports" / "best_strategy_study" / "hybrid_regime_backtest.csv"
        if hybrid_csv.exists():
            hdf = pd.read_csv(hybrid_csv, index_col="Date", parse_dates=True).sort_index()
            if "H8b_SPYvol>=15pct" in hdf.columns:
                curves["Hybrid A/D"] = hdf["H8b_SPYvol>=15pct"].resample("W-FRI").last().dropna()
            else:
                MISSING_FROM_COMPARISON.append("Hybrid A/D")
        else:
            MISSING_FROM_COMPARISON.append("Hybrid A/D")
    except Exception:
        MISSING_FROM_COMPARISON.append("Hybrid A/D")

    # Best Mix — static Calmar-optimal blend of NTSX/Platinum/F-TAA
    try:
        bestmix_csv = ROOT / "data" / "Strategy_Comparisons" / "recommended_strategy_mix_equity.csv"
        if bestmix_csv.exists():
            bmdf = pd.read_csv(bestmix_csv, index_col=0, parse_dates=True).sort_index()
            if "Combined_Equity" in bmdf.columns:
                curves["Best Mix"] = bmdf["Combined_Equity"].resample("W-FRI").last().dropna()
            else:
                MISSING_FROM_COMPARISON.append("Best Mix")
        else:
            MISSING_FROM_COMPARISON.append("Best Mix")
    except Exception:
        MISSING_FROM_COMPARISON.append("Best Mix")

    # 200MA Strategy — computed inline: hold SPY above the 200d MA, else BIL
    # (T-bill proxy) below it. No stored backtest file exists for this rule yet.
    try:
        if "SPY" in data.columns and "BIL" in data.columns:
            spy_px = data["SPY"].dropna()
            bil_px = data["BIL"].dropna()
            ma200 = spy_px.rolling(200).mean()
            above = (spy_px > ma200).reindex(spy_px.index).fillna(False)
            spy_ret = spy_px.pct_change().fillna(0.0)
            bil_ret = bil_px.pct_change().reindex(spy_px.index).fillna(0.0)
            strat_ret = spy_ret.where(above, bil_ret)
            equity_200ma = (1 + strat_ret).cumprod()
            curves["200MA Strategy"] = equity_200ma.resample("W-FRI").last().dropna()
        else:
            MISSING_FROM_COMPARISON.append("200MA Strategy")
    except Exception:
        MISSING_FROM_COMPARISON.append("200MA Strategy")

    # These strategies don't have a long-run backtest equity curve stored in the
    # repo yet — they're live/rotation models (or, for Market Pulse/FTD, a static
    # embed) rather than a single-line historical series. Flagged explicitly
    # instead of just quietly missing from the chart.
    MISSING_FROM_COMPARISON += [
        "SPY+QQQ+GLD Allocator", "Ensemble Top-100 ETF", "E80/R3 Live", "Market Pulse / FTD",
    ]

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
        "Hybrid A/D": ("#38bdf8", 1.5), "Best Mix": ("#facc15", 1.5), "200MA Strategy": ("#fb7185", 1.2),
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
    if MISSING_FROM_COMPARISON:
        st.caption(
            "◽ Not plotted (no stored long-run backtest curve yet): "
            + ", ".join(sorted(set(MISSING_FROM_COMPARISON)))
            + ". See Section 1 for their live signal."
        )

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

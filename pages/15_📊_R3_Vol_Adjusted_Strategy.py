import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import json
from pathlib import Path

st.set_page_config(page_title="R3 Vol-Adjusted Strategy", page_icon="📊", layout="wide")

# ── Data loaders ──────────────────────────────────────────────────────────────
from utils.data_engine import get_clean_master

@st.cache_data(ttl=3600)
def load_data():
    return get_clean_master()

@st.cache_data(ttl=3600)
def load_weekly_signals():
    sig_path = Path(__file__).parent.parent / "exports" / "trap_regime_backtest" / "system_e_weekly_signals.csv"
    if sig_path.exists():
        df = pd.read_csv(sig_path, index_col="Date", parse_dates=True)
        df.sort_index(inplace=True)
        return df
    return pd.DataFrame()

@st.cache_data(ttl=86400)
def load_backtest_data():
    base = Path(__file__).parent.parent / "exports" / "trap_regime_backtest" / "risk_allocation"
    eq   = pd.read_csv(base / "risk_allocation_equity_curves.csv",  index_col="Date", parse_dates=True)
    perf = pd.read_csv(base / "risk_allocation_performance.csv")
    sub  = pd.read_csv(base / "risk_allocation_subperiods.csv")
    return eq, perf, sub

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    render_master_controls()
    render_ecosystem_sidebar()

# ── Score computation helpers ─────────────────────────────────────────────────

def compute_bull_trap_score(d: pd.DataFrame) -> float:
    """Exact logic from pages/3_🐂_Bull_Trap.py"""
    if len(d) < 200:
        return 2.5
    latest  = d.iloc[-1]
    prev_mo = d.iloc[max(0, len(d) - 23)]

    # 1. Yield curve re-steepening
    curve      = latest["^TNX"] - latest["^IRX"]
    prev_curve = prev_mo["^TNX"] - prev_mo["^IRX"]
    if curve > 0 and prev_curve < 0:
        ys = 1.0
    elif curve > 0:
        ys = 0.5
    else:
        ys = 0.0

    # 2. VIX regime
    vix_ma22 = d["^VIX"].rolling(22).mean().iloc[-1]
    if latest["^VIX"] < 15:
        vs = 1.0
    elif latest["^VIX"] < vix_ma22:
        vs = 0.5
    else:
        vs = 0.0

    # 3. Credit recovery (HYG/IEF vs 22d MA)
    hyg_ief      = d["HYG"] / d["IEF"]
    hyg_ief_ma22 = hyg_ief.rolling(22).mean().iloc[-1]
    cs = 1.0 if hyg_ief.iloc[-1] > hyg_ief_ma22 else 0.0

    # 4. Market breadth (SPY vs 200d MA)
    spy_200ma = d["SPY"].rolling(200).mean().iloc[-1]
    if latest["SPY"] > spy_200ma * 1.05:
        bs = 1.0
    elif latest["SPY"] > spy_200ma:
        bs = 0.5
    else:
        bs = 0.0

    # 5. Accumulation (22d momentum)
    spy_mom22 = (latest["SPY"] / prev_mo["SPY"]) - 1
    if spy_mom22 > 0.02:
        acc = 1.0
    elif spy_mom22 > 0:
        acc = 0.5
    else:
        acc = 0.0

    # 6. Liquidity (TIP trend)
    if "TIP" in d.columns and len(d) >= 22:
        tip_trend = d["TIP"].iloc[-1] / d["TIP"].iloc[-22] - 1
        liq = 1.0 if tip_trend > 0 else 0.0
    else:
        liq = 0.5

    # Static scores (valuation + insider = 0.5 + 0.5 = 1.0 base)
    total = min(10.0, ys + vs + cs + bs + acc + liq + 0.5 + 0.5 + 1.0)
    return round(total, 2)


def _normalize(val, lower, upper, inverted=False):
    """Exact normalize from pages/2_🐻_Bear_Trap.py"""
    if inverted:
        if val >= lower: return 0.0
        if val <= upper: return 1.0
        return round((lower - val) / (lower - upper), 4)
    else:
        if val <= lower: return 0.0
        if val >= upper: return 1.0
        return round((val - lower) / (upper - lower), 4)


def compute_bear_trap_score(d: pd.DataFrame) -> float:
    """Exact logic from pages/2_🐻_Bear_Trap.py"""
    if len(d) < 252:
        return 0.5
    latest = d.iloc[-1]

    curve        = latest["^TNX"] - latest["^IRX"]
    irx_ma120    = d["^IRX"].rolling(120).mean().iloc[-1]
    hyg_ief      = (d["HYG"] / d["IEF"])
    hyg_ief_ma252= hyg_ief.rolling(252).mean().iloc[-1]
    spy_200ma    = d["SPY"].rolling(200).mean().iloc[-1]

    macro_score      = _normalize(curve,           1.5,               -0.5,               inverted=True)
    liquidity_score  = _normalize(latest["^IRX"],  irx_ma120 * 0.8,   irx_ma120 * 1.2,   inverted=False)
    credit_score     = _normalize(hyg_ief.iloc[-1],hyg_ief_ma252*1.05,hyg_ief_ma252*0.90, inverted=True)
    breadth_score    = _normalize(latest["SPY"],   spy_200ma * 1.05,  spy_200ma * 0.95,   inverted=True)
    vix_score        = _normalize(latest["^VIX"],  15,                35,                 inverted=False)

    total = (
        macro_score     * 0.25 +
        liquidity_score * 0.20 +
        credit_score    * 0.20 +
        breadth_score   * 0.15 +
        vix_score       * 0.10 +
        0.65            * 0.05 +
        0.50            * 0.05
    )
    return round(total, 4)


def compute_r3_deploy(bt_score: float, bear_score: float, spy_weekly_returns: pd.Series) -> dict:
    """R3 deployment calculation."""
    TARGET_VOL    = 0.12
    bull_weight   = bt_score / 10.0
    bear_penalty  = bear_score
    net_confidence = bull_weight * (1.0 - bear_penalty)

    # SPY 4-week realised vol (annualised)
    last4 = spy_weekly_returns.dropna().tail(4)
    if len(last4) >= 2:
        spy_4wk_vol = last4.std() * np.sqrt(52)
    else:
        spy_4wk_vol = TARGET_VOL

    vol_scalar = spy_4wk_vol / TARGET_VOL
    vol_adj    = net_confidence * min(1.5, 1.0 / vol_scalar) if vol_scalar > 0 else net_confidence
    deploy     = 0.10 + vol_adj * 0.90
    deploy     = float(np.clip(deploy, 0.10, 1.00))

    return {
        "bt_score":        bt_score,
        "bear_score":      bear_score,
        "net_confidence":  round(net_confidence, 4),
        "vol_scalar":      round(vol_scalar, 4),
        "spy_4wk_vol":     round(spy_4wk_vol, 4),
        "deploy":          round(deploy, 4),
        "deploy_pct":      round(deploy * 100, 1),
    }


def compute_historical_signals(data: pd.DataFrame, weeks: int = 52) -> pd.DataFrame:
    """Compute weekly bt_score, bear_score, deploy% over the last N weeks."""
    # Resample to weekly (Friday close)
    w = data.resample("W-FRI").last().ffill()
    w = w.dropna(how="all")

    # SPY weekly returns for vol
    spy_ret_w = w["SPY"].pct_change()

    rows = []
    for i in range(max(260, len(w) - weeks), len(w)):
        d_daily = data.loc[:w.index[i]].copy()
        if len(d_daily) < 252:
            continue
        bt  = compute_bull_trap_score(d_daily)
        btr = compute_bear_trap_score(d_daily)
        sig = compute_r3_deploy(bt, btr, spy_ret_w.iloc[max(0, i-4):i])
        rows.append({"Date": w.index[i], "bt_score": bt, "bear_score": btr,
                     "deploy_pct": sig["deploy_pct"]})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Date").tail(weeks)


# ── Main dashboard ────────────────────────────────────────────────────────────

def dashboard():
    from utils.ui_utils import resolve_master_date_slice

    st.title("📊 R3 Vol-Adjusted Strategy")
    st.markdown(
        "Dynamic risk allocation combining Bull Trap momentum scoring, Bear Trap regime penalty, "
        "and real-time SPY volatility scaling to size NQ100 top-3 equity exposure."
    )

    analysis_date = st.session_state["master_date"]

    with st.spinner("Loading market data…"):
        data = load_data()
        if data.empty:
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

    # ── Load pre-computed weekly signals for NQ100 picks ─────────────────────
    weekly_sig = load_weekly_signals()

    # ── Compute live scores ───────────────────────────────────────────────────
    bt_score   = compute_bull_trap_score(d)
    bear_score = compute_bear_trap_score(d)

    # SPY weekly returns for vol calculation
    spy_w   = d["SPY"].resample("W-FRI").last()
    spy_ret = spy_w.pct_change()
    sig     = compute_r3_deploy(bt_score, bear_score, spy_ret)

    deploy_pct     = sig["deploy_pct"]
    net_confidence = sig["net_confidence"]
    vol_scalar     = sig["vol_scalar"]
    spy_4wk_vol    = sig["spy_4wk_vol"]

    # Regime label & colour
    if deploy_pct >= 70:
        regime, regime_colour, regime_bg = "FULL DEPLOYMENT", "#2ecc71", "#0d2e1a"
    elif deploy_pct >= 40:
        regime, regime_colour, regime_bg = "CAUTION",         "#f1c40f", "#2e2a00"
    else:
        regime, regime_colour, regime_bg = "DEFENSIVE",       "#e74c3c", "#2e0d0d"

    # ── NQ100 top-3 picks ─────────────────────────────────────────────────────
    BREADTH_MIN  = 40.0
    BULL_SPY     = 0.350
    BULL_STK     = 0.217   # per stock; 3 × 21.7% = 65.1% stocks
    BOND_FRAC    = 1.0 - BULL_SPY - BULL_STK * 3  # 0% cash — 100% of deployed

    if not weekly_sig.empty:
        # Use latest signal on or before the analysis date
        available = weekly_sig[weekly_sig.index <= pd.Timestamp(analysis_date)]
        if not available.empty:
            latest_sig  = available.iloc[-1]
            breadth_pct = float(latest_sig.get("breadth_pct", 50))
            top3_raw    = str(latest_sig.get("top3_picks", ""))
            top3_picks  = [x.strip().strip('"') for x in top3_raw.split(",") if x.strip()] if top3_raw else []
        else:
            breadth_pct, top3_picks = 50.0, []
    else:
        breadth_pct, top3_picks = 50.0, []

    breadth_ok = breadth_pct >= BREADTH_MIN and len(top3_picks) >= 3

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — LIVE SIGNAL
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📡 Section 1 — Live Signal")

    # Big regime banner
    st.markdown(
        f"""
        <div style="background:{regime_bg};border:2px solid {regime_colour};border-radius:12px;
                    padding:20px;text-align:center;margin-bottom:20px;">
            <div style="font-size:2.2rem;font-weight:700;color:{regime_colour};">{regime}</div>
            <div style="font-size:3.5rem;font-weight:800;color:{regime_colour};">{deploy_pct:.1f}% Deployed</div>
            <div style="color:#94a3b8;margin-top:6px;font-size:0.9rem;">
                {actual_date.strftime('%Y-%m-%d')} signal
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Score metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Bull Trap Score",    f"{bt_score:.1f} / 10")
    c2.metric("Bear Score",         f"{bear_score*100:.1f}%")
    c3.metric("Net Confidence",     f"{net_confidence*100:.1f}%")
    c4.metric("SPY 4-wk Vol (ann)", f"{spy_4wk_vol*100:.1f}%")
    c5.metric("Vol Scalar",         f"{vol_scalar:.2f}×")

    st.markdown("&nbsp;")

    # NQ100 picks and allocation breakdown
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("#### Top-3 NQ100 Momentum Picks")
        if not breadth_ok:
            st.warning(
                f"⚠️ Breadth filter not met ({breadth_pct:.1f}% positive momentum < 40% threshold). "
                "SPY-only deployment in effect — NQ100 picks suspended."
            )
        if top3_picks:
            for i, tkr in enumerate(top3_picks[:3], 1):
                alloc = BULL_STK * deploy_pct
                st.markdown(
                    f"<div style='background:#1e293b;padding:10px 16px;border-radius:8px;"
                    f"margin-bottom:8px;border-left:4px solid #3b82f6;'>"
                    f"<span style='font-size:1.1rem;font-weight:700;color:#60a5fa;'>#{i} {tkr}</span>"
                    f"<span style='float:right;color:#94a3b8;'>{alloc:.1f}% of portfolio</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Top-3 picks unavailable for this date.")

    with col_r:
        st.markdown("#### Allocation Breakdown")
        if breadth_ok:
            spy_alloc   = BULL_SPY * deploy_pct
            stk_alloc   = BULL_STK * deploy_pct
            ief_alloc   = 100.0 - deploy_pct
            alloc_items = [
                ("SPY (35% of equity)", spy_alloc,   "#3b82f6"),
                (f"{top3_picks[0] if top3_picks else 'Stock 1'} (21.7%)", stk_alloc, "#22c55e"),
                (f"{top3_picks[1] if top3_picks else 'Stock 2'} (21.7%)", stk_alloc, "#22c55e"),
                (f"{top3_picks[2] if top3_picks else 'Stock 3'} (21.7%)", stk_alloc, "#22c55e"),
                ("IEF (bond buffer)",   ief_alloc,   "#64748b"),
            ]
        else:
            spy_alloc = deploy_pct
            ief_alloc = 100.0 - deploy_pct
            alloc_items = [
                ("SPY (full equity, breadth fail)", spy_alloc, "#f59e0b"),
                ("IEF (bond buffer)",               ief_alloc, "#64748b"),
            ]

        fig_pie = go.Figure(go.Pie(
            labels=[a[0] for a in alloc_items],
            values=[a[1] for a in alloc_items],
            marker_colors=[a[2] for a in alloc_items],
            hole=0.45,
            textinfo="label+percent",
            textfont_size=11,
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=260,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — HISTORICAL SIGNAL CHART
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("📈 Section 2 — Historical Signal (Last 52 Weeks)")

    with st.spinner("Computing historical signals…"):
        hist = compute_historical_signals(d, weeks=52)

    if hist.empty:
        st.warning("Not enough history to compute the 52-week signal chart.")
    else:
        fig2 = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.55, 0.45],
            vertical_spacing=0.06,
            subplot_titles=("Deploy % with Regime Shading", "Bull Trap Score & Bear Score"),
        )

        # — Regime shading on top panel —
        dates     = hist.index.tolist()
        deploy_v  = hist["deploy_pct"].tolist()

        # colour each week by regime
        for i in range(len(dates)):
            dp = deploy_v[i]
            if dp >= 70:   c = "rgba(46,204,113,0.10)"
            elif dp >= 40: c = "rgba(241,196,15,0.10)"
            else:          c = "rgba(231,76,60,0.10)"
            x0 = dates[i]
            x1 = dates[i+1] if i+1 < len(dates) else dates[i] + pd.Timedelta(days=7)
            fig2.add_vrect(x0=x0, x1=x1, fillcolor=c, line_width=0, row=1, col=1)

        # deploy line
        fig2.add_trace(go.Scatter(
            x=hist.index, y=hist["deploy_pct"],
            name="Deploy %", line=dict(color="#3b82f6", width=2),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.10)",
        ), row=1, col=1)

        # reference lines at 40% and 70%
        for level, color, label in [(70, "#2ecc71", "70% (Full)"), (40, "#f1c40f", "40% (Caution)")]:
            fig2.add_hline(y=level, line_dash="dash", line_color=color,
                           annotation_text=label, annotation_position="top left",
                           row=1, col=1)

        # — Bull Trap Score (left axis scaled to %) —
        fig2.add_trace(go.Scatter(
            x=hist.index, y=hist["bt_score"] * 10,   # scale 0-10 → 0-100%
            name="Bull Trap Score (×10)", line=dict(color="#22c55e", width=1.5, dash="dot"),
        ), row=2, col=1)

        # — Bear Score (already 0-1, multiply by 100) —
        fig2.add_trace(go.Scatter(
            x=hist.index, y=hist["bear_score"] * 100,
            name="Bear Score (%)", line=dict(color="#ef4444", width=1.5),
        ), row=2, col=1)

        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
            height=520,
            hovermode="x unified",
            margin=dict(t=60, b=20, l=60, r=20),
        )
        fig2.update_xaxes(gridcolor="#1e293b", showgrid=True)
        fig2.update_yaxes(gridcolor="#1e293b", showgrid=True)
        fig2.update_yaxes(title_text="Deploy %",       ticksuffix="%", row=1, col=1)
        fig2.update_yaxes(title_text="Score (normalised %)", ticksuffix="%", row=2, col=1)

        st.plotly_chart(fig2, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — 20-YEAR BACKTEST RESULTS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🏆 Section 3 — 20-Year Backtest Results (2004–Present)")

    try:
        eq_curves, perf_df, sub_df = load_backtest_data()

        # ── Key metric callouts ────────────────────────────────────────────────
        r3_row = perf_df[perf_df["Strategy"] == "R3_Vol_Adj"]
        spy_row = perf_df[perf_df["Strategy"] == "Buy_Hold_SPY"]
        syse_row = perf_df[perf_df["Strategy"] == "Sys_E_Original"]

        if not r3_row.empty:
            r3 = r3_row.iloc[0]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("R3 CAGR",         f"{r3.get('CAGR_%', 18.3):.1f}%",
                      delta=f"{r3.get('CAGR_%',18.3) - (spy_row.iloc[0].get('CAGR_%',10.8) if not spy_row.empty else 10.8):.1f}% vs SPY")
            m2.metric("Sharpe Ratio",     f"{r3.get('Sharpe', 1.248):.3f}")
            m3.metric("Max Drawdown",     f"{r3.get('Max_DD_%', -25.4):.1f}%")
            m4.metric("Avg Deploy",       f"{r3.get('AvgDeploy_%', 44):.1f}%")
            m5.metric("Win Rate",         f"{r3.get('Win_Rate_%', 59.2):.1f}%")

        st.markdown("&nbsp;")

        # ── Equity curve chart (log scale) ────────────────────────────────────
        SHOW_STRATS = {
            "R3_Vol_Adj":    ("R3 Vol-Adjusted",   "#3b82f6", 3),
            "Buy_Hold_SPY":  ("SPY Buy & Hold",     "#94a3b8", 1.5),
            "Sys_E_Original":("System E Original",  "#f59e0b", 1.5),
        }

        fig3 = go.Figure()
        for col, (label, color, width) in SHOW_STRATS.items():
            if col in eq_curves.columns:
                fig3.add_trace(go.Scatter(
                    x=eq_curves.index, y=eq_curves[col],
                    name=label, line=dict(color=color, width=width),
                    hovertemplate="%{x|%Y-%m-%d}: $%{y:,.0f}<extra></extra>",
                ))

        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0f172a",
            font_color="#e2e8f0",
            yaxis_type="log",
            yaxis_title="Portfolio Value (log scale, $100k start)",
            xaxis_title="",
            hovermode="x unified",
            legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02),
            height=420,
            margin=dict(t=40, b=20, l=70, r=20),
        )
        fig3.update_xaxes(gridcolor="#1e293b")
        fig3.update_yaxes(gridcolor="#1e293b", tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig3, use_container_width=True)

        # ── Performance table ─────────────────────────────────────────────────
        st.markdown("#### Strategy Comparison")
        DISPLAY_COLS = {
            "Strategy":       "Strategy",
            "CAGR_%":         "CAGR %",
            "Sharpe":         "Sharpe",
            "Sortino":        "Sortino",
            "Max_DD_%":       "Max DD %",
            "Win_Rate_%":     "Win Rate %",
            "AvgDeploy_%":    "Avg Deploy %",
            "Best_Year_%":    "Best Year %",
            "Worst_Year_%":   "Worst Year %",
        }
        show_strats_list = list(SHOW_STRATS.keys())
        table_df = perf_df[perf_df["Strategy"].isin(show_strats_list)].copy()
        table_df["Strategy"] = table_df["Strategy"].map(
            {k: v[0] for k, v in SHOW_STRATS.items()}
        )
        display_cols = [c for c in DISPLAY_COLS if c in table_df.columns]
        table_display = table_df[display_cols].rename(columns=DISPLAY_COLS).set_index("Strategy")
        st.dataframe(table_display, use_container_width=True)

        # ── Sub-period table ──────────────────────────────────────────────────
        st.markdown("#### Sub-Period Returns")
        sub_display_cols = ["Period"] + [c for c in SHOW_STRATS if c in sub_df.columns]
        if "Period" in sub_df.columns:
            sub_show = sub_df[sub_display_cols].copy()
            sub_show = sub_show.rename(columns={k: v[0] for k, v in SHOW_STRATS.items()})
            sub_show = sub_show.set_index("Period")
            # Format as percentages
            st.dataframe(
                sub_show.style.format("{:.1f}%")
                              .background_gradient(axis=None, cmap="RdYlGn", vmin=-40, vmax=150),
                use_container_width=True,
            )

    except Exception as exc:
        st.error(f"Could not load backtest results: {exc}")
        st.info("Expected files in `exports/trap_regime_backtest/risk_allocation/`.")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — HOW IT WORKS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🧠 Section 4 — How R3 Works")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(
            """
**Bull Trap Score (0–10)**

The Bull Trap score from Page 3 measures structural market health across six live indicators:
Yield Curve re-steepening, VIX regime, Credit recovery (HYG/IEF), Market Breadth (SPY vs 200d MA),
Momentum accumulation, and Liquidity (TIP trend). A score of 10 signals a confirmed bull market;
below 4 indicates elevated bull-trap risk.

**Bear Trap Score (0–1)**

The Bear Trap composite from Page 2 weights five macro risk factors:
Yield Curve inversion (25%), Interest Rate pressure (20%), Credit Stress (20%),
Market Structure (15%), and Volatility (10%). High scores signal recession/bear risk.

**Why two scores?**

The Bull Trap score captures *upside momentum* — it rises when conditions are constructive.
The Bear Trap score captures *downside risk* — it rises when structural cracks appear.
Combining them prevents deploying aggressively when the bull signal is moderately positive
but macro risks are elevated.
            """
        )

    with col_b:
        st.markdown(
            """
**Volatility Adjustment**

```
bull_weight    = bt_score / 10         # 0 → 1
bear_penalty   = bear_score            # 0 → 1
net_confidence = bull_weight × (1 − bear_penalty)

spy_4wk_vol    = std(SPY weekly returns, 4 wks) × √52
vol_scalar     = spy_4wk_vol / 12%  (target vol)

vol_adj = net_confidence × min(1.5, 1 / vol_scalar)
deploy  = clip(10% + vol_adj × 90%, 10%, 100%)
```

**When vol is HIGH** (vol_scalar > 1): we scale *down* exposure — high realised volatility
means each dollar of risk costs more, so we buy less equity to stay near the 12% annual
volatility target.

**When vol is LOW** (vol_scalar < 1): we scale *up* (capped at 1.5× the base), letting
quiet markets run further.

**Why this beats binary on/off:**
Static systems flip from 100% to 0% and suffer from both whipsaws and missed recovery rallies.
R3's continuous dial — blending regime confidence with real-time volatility — reduces unnecessary
trading and keeps drawdowns tighter during regime uncertainty.
            """
        )


dashboard()

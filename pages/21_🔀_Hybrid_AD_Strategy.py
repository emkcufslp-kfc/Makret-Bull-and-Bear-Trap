"""
pages/21_🔀_Hybrid_AD_Strategy.py
===================================
Hybrid A/D — Regime Strategy dashboard + Live Trading cockpit, combined via tabs.

Tab "Strategy & Backtest": 22-year historical backtest of the H8b strategy
  (switch to Strategy D when SPY 8-week realized vol >= 15%) and its debounced
  variants. Sections: Live Signal Today, Equity Curve, Performance Metrics,
  Regime History, Year-by-Year Returns, Debounce Grid Results.

Tab "Live Trading": real paper-trading ledger since 2026-07-01, $10,000 start.
  Sections: Action Panel, Portfolio & Risk, Trading Log, Ledger.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid A/D Strategy",
    page_icon="🔀",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent.parent

from utils.data_engine import get_clean_master
from backend.strategies.hybrid_ad_live_ledger import (
    START, START_CAPITAL, VOL_THRESHOLD, build_hybrid_live_state,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    render_master_controls()
    render_ecosystem_sidebar()

# ── Style ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.badge-strategy-a  { background:#10b981; color:#fff; padding:8px 22px; border-radius:24px;
                     font-weight:700; font-size:1.35rem; display:inline-block;
                     letter-spacing:0.5px; }
.badge-strategy-d  { background:#f59e0b; color:#000; padding:8px 22px; border-radius:24px;
                     font-weight:700; font-size:1.35rem; display:inline-block;
                     letter-spacing:0.5px; }
.badge-unknown     { background:#6b7280; color:#fff; padding:8px 22px; border-radius:24px;
                     font-weight:700; font-size:1.35rem; display:inline-block; }
.section-hdr       { color:#e2e8f0; font-size:1.3rem; font-weight:700;
                     border-left:4px solid #f59e0b; padding-left:10px; margin:24px 0 12px; }
.vol-gauge-wrap    { background:#0f172a; border:1px solid #334155; border-radius:10px;
                     padding:16px 20px; margin:8px 0; }
.info-card         { background:#1e293b; border:1px solid #334155; border-radius:8px;
                     padding:12px 16px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🔀 Hybrid A/D Strategy")
st.caption(
    "Switches between **Strategy A** (80% SysE / 20% R3, full exposure) and "
    "**Strategy D** (15% vol-target overlay on A) based on SPY 8-week realized volatility. "
    "**Production signal: H8b + 2-week debounce** — enter D only after SPY vol ≥ 15% for "
    "2 consecutive weeks (fewer whipsaws), exit immediately once vol drops back below threshold. "
    "Calmar 1.404 vs 1.268 for the raw 1-week signal, with 62 switches vs 72 over 22.5 years. "
    "**Strategy & Backtest** below shows 22 years of history; **Live Trading** tracks a real "
    "paper book since 2026-07-01."
)

# ── Paths ─────────────────────────────────────────────────────────────────────
STUDY_DIR     = ROOT / "exports" / "best_strategy_study"
RA_CSV        = ROOT / "exports" / "trap_regime_backtest" / "risk_allocation" / "risk_allocation_equity_curves.csv"
HYBRID_CSV    = STUDY_DIR / "hybrid_regime_backtest.csv"
DEBOUNCE_CSV  = STUDY_DIR / "hybrid_debounce_results.csv"
RESULTS_JSON  = STUDY_DIR / "hybrid_results.json"

# Column name for H8b in hybrid_regime_backtest.csv — production signal uses the
# 2-week entry debounce (Calmar 1.404 vs 1.268 for the raw 1-week signal).
# Falls back to the raw column if the debounce variant hasn't been generated yet.
H8B_COL_DEB   = "H8b_SPYvol>=15pct_deb2w"
H8B_COL_RAW   = "H8b_SPYvol>=15pct"
VOL_WINDOW    = 8      # 8 weeks
DEBOUNCE_WEEKS = 2     # consecutive weeks above threshold required to enter Strategy D


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_market_data():
    return get_clean_master()


@st.cache_data(ttl=86400)
def load_hybrid_curves():
    """Load hybrid equity curves. Returns (DataFrame or None, error_msg)."""
    if not HYBRID_CSV.exists():
        return None, f"File not found: {HYBRID_CSV}"
    try:
        df = pd.read_csv(HYBRID_CSV, index_col=0, parse_dates=True).sort_index()
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=86400)
def load_ra_curves():
    """Load risk-allocation equity curves (baseline A, D, SPY)."""
    if not RA_CSV.exists():
        return None
    return pd.read_csv(RA_CSV, index_col="Date", parse_dates=True).sort_index()


@st.cache_data(ttl=86400)
def load_debounce_results():
    if not DEBOUNCE_CSV.exists():
        return None
    return pd.read_csv(DEBOUNCE_CSV)


@st.cache_data(ttl=86400)
def load_results_json():
    if not RESULTS_JSON.exists():
        return {}
    try:
        return json.loads(RESULTS_JSON.read_text())
    except Exception:
        return {}


@st.cache_data(ttl=1800)
def load_hybrid_live_state(_df):
    return build_hybrid_live_state(_df)


# ── Compute live SPY realized vol (with 2-week entry debounce) ────────────────
def compute_spy_vol(df: pd.DataFrame, weeks: int = 8) -> tuple[float, float, bool, float, bool]:
    """Returns (current_vol, threshold, in_defensive, prior_vol, prior_breach).

    in_defensive uses the production rule: current AND prior week's vol must both
    be >= threshold (2-week entry confirmation). Exit is immediate — as soon as
    current vol drops below threshold, in_defensive is False regardless of history.
    """
    spy_weekly = df["SPY"].resample("W-FRI").last().dropna()
    spy_ret    = spy_weekly.pct_change().dropna()
    if len(spy_ret) < weeks + 1:
        return np.nan, VOL_THRESHOLD, False, np.nan, False
    rvol = spy_ret.rolling(weeks).std() * np.sqrt(52)
    realized  = float(rvol.iloc[-1])
    prior_vol = float(rvol.iloc[-2])
    breach_now   = realized  >= VOL_THRESHOLD
    breach_prior = prior_vol >= VOL_THRESHOLD
    in_defensive = breach_now and breach_prior   # 2-week confirmation to enter
    return realized, VOL_THRESHOLD, in_defensive, prior_vol, breach_prior


# ── Metrics helper (used in Strategy & Backtest tab) ───────────────────────────
RF_RATE = 0.02

def _metrics_from_series(series: pd.Series, label: str) -> dict:
    s  = series.dropna()
    r  = s.pct_change().dropna()
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    cagr    = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    vol     = r.std() * np.sqrt(52)
    sharpe  = (r.mean() * 52 - RF_RATE) / vol if vol > 0 else np.nan
    dn      = r[r < 0].std() * np.sqrt(52)
    sortino = (r.mean() * 52 - RF_RATE) / dn if dn > 0 else np.nan
    dd      = (s / s.cummax() - 1).min()
    calmar  = cagr / abs(dd) if dd != 0 else np.nan
    return {
        "Strategy": label,
        "CAGR":     f"{cagr*100:.1f}%",
        "Max DD":   f"{dd*100:.1f}%",
        "Calmar":   f"{calmar:.3f}",
        "Sharpe":   f"{sharpe:.3f}",
        "Sortino":  f"{sortino:.3f}",
    }


# ── Load market data once, shared by both tabs ─────────────────────────────────
data = load_market_data()
if data is None or data.empty:
    st.error("⚠️ Failed to load market data.")
    st.stop()

tab_strategy, tab_live = st.tabs(["📊 Strategy & Backtest", "💼 Live Trading"])

with tab_strategy:
    # ════════════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Live Signal Today
    # ════════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">📡 Section 1 — Live Signal Today</div>',
                unsafe_allow_html=True)

    try:
        spy_vol, threshold, in_defensive, prior_vol, breach_prior = compute_spy_vol(data)
        live_ok = True
    except Exception as e:
        spy_vol, threshold, in_defensive, prior_vol, breach_prior = np.nan, VOL_THRESHOLD, False, np.nan, False
        live_ok = False
        st.warning(f"Could not compute live vol: {e}")

    col_badge, col_gauge, col_info = st.columns([2, 2, 3])

    with col_badge:
        st.markdown("**Current Mode**")
        if np.isnan(spy_vol):
            st.markdown('<span class="badge-unknown">MODE: UNKNOWN</span>', unsafe_allow_html=True)
        elif in_defensive:
            st.markdown(
                '<span class="badge-strategy-d">MODE: STRATEGY D<br>'
                '<small style="font-size:0.85rem;font-weight:400">'
                'Bear / Vol-Target Active</small></span>',
                unsafe_allow_html=True
            )
            st.markdown("")
            st.info("🛡️ **Strategy D is active.** Vol-target overlay reduces exposure "
                    "(2-week confirmation met — both this week and prior week's vol were ≥15%). "
                    "Strategy A resumes immediately once vol drops below 15%.")
        else:
            st.markdown(
                '<span class="badge-strategy-a">MODE: STRATEGY A<br>'
                '<small style="font-size:0.85rem;font-weight:400">'
                'Bull / Low-Vol — Full Exposure</small></span>',
                unsafe_allow_html=True
            )
            st.markdown("")
            if not np.isnan(spy_vol) and spy_vol >= VOL_THRESHOLD and not breach_prior:
                st.warning(f"⏳ **Watching.** SPY vol {spy_vol*100:.1f}% is above the 15% threshold this week, "
                           f"but prior week ({prior_vol*100:.1f}%) was not — needs 2 consecutive weeks to "
                           f"switch to Strategy D. Currently still on Strategy A.")
            else:
                st.success("✅ **Strategy A is active.** Full 80% SysE / 20% R3 deployment. "
                           "No vol-target overlay needed.")

    with col_gauge:
        st.markdown("**SPY 8-Week Realized Vol**")
        if not np.isnan(spy_vol):
            vol_pct = spy_vol * 100
            thresh_pct = threshold * 100

            # Gauge figure
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=vol_pct,
                delta={"reference": thresh_pct, "suffix": "%",
                       "increasing": {"color": "#ef4444"},
                       "decreasing": {"color": "#10b981"}},
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 50], "ticksuffix": "%",
                             "tickcolor": "#94a3b8", "tickfont": {"color": "#94a3b8"}},
                    "bar": {"color": "#ef4444" if in_defensive else "#10b981", "thickness": 0.3},
                    "bgcolor": "#1e293b",
                    "borderwidth": 1,
                    "bordercolor": "#334155",
                    "steps": [
                        {"range": [0, thresh_pct],  "color": "#0f172a"},
                        {"range": [thresh_pct, 50], "color": "#1e293b"},
                    ],
                    "threshold": {
                        "line": {"color": "#f59e0b", "width": 3},
                        "thickness": 0.85,
                        "value": thresh_pct,
                    },
                },
                title={"text": f"vs {thresh_pct:.0f}% threshold", "font": {"color": "#94a3b8", "size": 13}},
            ))
            fig_g.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                height=200,
                margin=dict(l=20, r=20, t=30, b=10),
                font={"color": "#e2e8f0"},
            )
            st.plotly_chart(fig_g, use_container_width=True)
        else:
            st.markdown('<div class="info-card">SPY vol data unavailable</div>',
                        unsafe_allow_html=True)

    with col_info:
        st.markdown("**Strategy Logic**")
        st.markdown("""
    <div class="info-card">
    <b>Entry → Strategy D (defensive)</b><br>
    SPY 8-week realized vol ≥ <b>15%</b> annualized for <b>2 consecutive weeks</b> — the
    confirmation reduces whipsaw switching (62 vs 72 switches over 22.5y).<br><br>
    <b>Exit → Strategy A (bull / full exposure)</b><br>
    Immediate — reverts to full A as soon as vol drops back below 15%, no exit debounce.<br><br>
    <b>Why 2-week entry / 1-week exit:</b> asymmetric confirmation catches genuine vol
    regime shifts while still exiting fast once the storm passes. Calmar 1.404 vs 1.268
    for the raw (no-debounce) signal; production choice, replacing the earlier 1-week rule.
    </div>
    """, unsafe_allow_html=True)

        # Show best debounced result
        json_data = load_results_json()
        deb_study = json_data.get("debounce_study", {})
        best_deb  = deb_study.get("best_debounced", {})
        if best_deb:
            st.markdown(f"""
    <div class="info-card">
    🏆 <b>Best debounced variant:</b> <code>{best_deb.get('Strategy','N/A')}</code><br>
    Calmar={best_deb.get('Calmar','?')}  |  CAGR={best_deb.get('CAGR_%','?')}%  |  MaxDD={best_deb.get('Max_DD_%','?')}%
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ════════════════════════════════════════════════════════════════════════════════
    # Load curve data
    # ════════════════════════════════════════════════════════════════════════════════
    hybrid_df, hybrid_err = load_hybrid_curves()
    ra_df = load_ra_curves()

    # Resolve which H8b column is available — prefer the 2-week debounced production
    # signal, fall back to the raw 1-week variant if the debounce columns haven't been
    # generated yet (keeps the page from breaking on stale data).
    if hybrid_df is not None and H8B_COL_DEB in hybrid_df.columns:
        H8B_COL = H8B_COL_DEB
        H8B_LABEL = "H8b + 2wk debounce — Hybrid A/D (SPY vol 15%)"
    else:
        H8B_COL = H8B_COL_RAW
        H8B_LABEL = "H8b (raw 1wk) — Hybrid A/D (SPY vol 15%)"

    # Build named curve dict
    CURVE_DEFS = {
        H8B_LABEL:                          (hybrid_df, H8B_COL),
        "A: 80%SysE/20%R3 static":        (hybrid_df, "A: 80%SysE/20%R3 static"),
        "D: 15% vol-target on A":          (hybrid_df, "D: 15%vol-target on A (always)"),
        "SPY Buy & Hold":                   (hybrid_df, "SPY Buy & Hold"),
    }

    COLORS = {
        H8B_LABEL: "#f59e0b",
        "A: 80%SysE/20%R3 static":        "#6366f1",
        "D: 15% vol-target on A":          "#10b981",
        "SPY Buy & Hold":                   "#94a3b8",
    }
    DASHES = {
        H8B_LABEL: "solid",
        "A: 80%SysE/20%R3 static":        "dot",
        "D: 15% vol-target on A":          "dash",
        "SPY Buy & Hold":                   "longdash",
    }

    curves: dict[str, pd.Series] = {}

    if hybrid_df is not None:
        for label, (df, col) in CURVE_DEFS.items():
            if df is not None and col in df.columns:
                s = df[col].dropna()
                curves[label] = s / s.iloc[0]   # normalise to 1.0

        # Regime flag: in defensive when H8b != A
        if H8B_COL in hybrid_df.columns and "A: 80%SysE/20%R3 static" in hybrid_df.columns:
            diff_flag = (hybrid_df[H8B_COL] - hybrid_df["A: 80%SysE/20%R3 static"]).abs()
            in_d_mode = diff_flag > 1e-8
        else:
            in_d_mode = pd.Series(dtype=bool)
    else:
        in_d_mode = pd.Series(dtype=bool)

    data_missing = len(curves) == 0

    if data_missing:
        st.warning(
            f"⚠️ Hybrid equity curve data is unavailable. "
            f"{'Error: ' + hybrid_err if hybrid_err else 'Run hybrid_regime_backtest.py first.'}"
        )

    # ════════════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Equity Curve
    # ════════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">📈 Section 2 — Equity Curve (2004–Present)</div>',
                unsafe_allow_html=True)

    if not data_missing:
        fig_eq = go.Figure()

        # Shade periods when hybrid is in Strategy D mode
        if len(in_d_mode) > 0:
            bear_start = None
            for dt, is_bear in in_d_mode.items():
                if is_bear and bear_start is None:
                    bear_start = dt
                elif not is_bear and bear_start is not None:
                    fig_eq.add_vrect(x0=bear_start, x1=dt,
                                     fillcolor="rgba(245,158,11,0.09)", line_width=0, layer="below")
                    bear_start = None
            if bear_start is not None:
                fig_eq.add_vrect(x0=bear_start, x1=in_d_mode.index[-1],
                                 fillcolor="rgba(245,158,11,0.09)", line_width=0, layer="below")

        for label, series in curves.items():
            width  = 2.5 if label == H8B_LABEL else 1.5
            fig_eq.add_trace(go.Scatter(
                x=series.index, y=series,
                name=label,
                line=dict(color=COLORS[label], width=width, dash=DASHES[label]),
                hovertemplate=f"<b>{label}</b><br>%{{x|%Y-%m-%d}}<br>"
                              f"Value: %{{y:.2f}}x<extra></extra>",
            ))

        fig_eq.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            yaxis_type="log",
            yaxis_title="Portfolio Value (log scale, normalised to 1.0)",
            xaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=480,
            margin=dict(l=60, r=20, t=40, b=40),
            hovermode="x unified",
        )
        fig_eq.update_xaxes(showgrid=True, gridcolor="#1e293b")
        fig_eq.update_yaxes(showgrid=True, gridcolor="#1e293b")

        st.plotly_chart(fig_eq, use_container_width=True)
        st.caption(
            "🟡 Shaded regions = periods when Hybrid is in **Strategy D** (vol-target) mode. "
            "Log scale. Starting value normalised to 1.0 ($100,000)."
        )
    else:
        st.info("Equity curve unavailable — data not loaded.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Performance Metrics
    # ════════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">📊 Section 3 — Performance Metrics</div>',
                unsafe_allow_html=True)

    # Inline compute metrics from curves
    RF_RATE = 0.02

    def _metrics_from_series(series: pd.Series, label: str) -> dict:
        s  = series.dropna()
        r  = s.pct_change().dropna()
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        cagr    = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
        vol     = r.std() * np.sqrt(52)
        sharpe  = (r.mean() * 52 - RF_RATE) / vol if vol > 0 else np.nan
        dn      = r[r < 0].std() * np.sqrt(52)
        sortino = (r.mean() * 52 - RF_RATE) / dn if dn > 0 else np.nan
        dd      = (s / s.cummax() - 1).min()
        calmar  = cagr / abs(dd) if dd != 0 else np.nan
        return {
            "Strategy": label,
            "CAGR":     f"{cagr*100:.1f}%",
            "Max DD":   f"{dd*100:.1f}%",
            "Calmar":   f"{calmar:.3f}",
            "Sharpe":   f"{sharpe:.3f}",
            "Sortino":  f"{sortino:.3f}",
        }

    if not data_missing:
        met_rows = [_metrics_from_series(s, l) for l, s in curves.items()]
        met_df   = pd.DataFrame(met_rows)

        def _style_metrics(row):
            styles = []
            label  = row["Strategy"]
            color  = COLORS.get(label, "#e2e8f0")
            for col in row.index:
                if col == "Strategy":
                    styles.append(f"color:{color};font-weight:700")
                elif col == "Max DD":
                    styles.append("color:#ef4444")
                else:
                    styles.append("color:#e2e8f0")
            return styles

        styled = met_df.style.apply(_style_metrics, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Callout: H8b vs pure A
        if H8B_LABEL in curves and "A: 80%SysE/20%R3 static" in curves:
            h8b_s = curves[H8B_LABEL]
            a_s   = curves["A: 80%SysE/20%R3 static"]
            h8b_dd = ((h8b_s / h8b_s.cummax()) - 1).min() * 100
            a_dd   = ((a_s   / a_s.cummax())   - 1).min() * 100
            dd_improvement = a_dd - h8b_dd   # more negative means deeper drawdown for A

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.metric(
                    "H8b Max Drawdown",
                    f"{h8b_dd:.1f}%",
                    delta=f"{dd_improvement:+.1f}% vs pure A",
                    delta_color="normal" if dd_improvement > 0 else "inverse",
                    help="H8b (2wk debounce) improves max drawdown by spending ~31% of time in Strategy D"
                )
            with col_c2:
                pct_d = in_d_mode.mean() * 100 if len(in_d_mode) > 0 else 0.0
                st.metric(
                    "Time in Strategy D",
                    f"{pct_d:.1f}%",
                    delta="defensive periods",
                    delta_color="off",
                )
    else:
        st.info("Metrics unavailable — equity curve data not loaded.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Regime History
    # ════════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">🗓️ Section 4 — Regime History (A vs D)</div>',
                unsafe_allow_html=True)

    if not data_missing and len(in_d_mode) > 0:
        # Annual % of weeks in Strategy D
        d_mode_int = in_d_mode.astype(int)
        annual_d   = d_mode_int.groupby(d_mode_int.index.year).mean() * 100

        fig_reg = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.35, 0.65],
            vertical_spacing=0.04,
            subplot_titles=["% Weeks in Strategy D per Year", "SPY 8-Week Vol vs 15% Threshold"],
        )

        # Bar: % weeks in D per year
        fig_reg.add_trace(go.Bar(
            x=annual_d.index, y=annual_d.values,
            name="% Weeks in Strategy D",
            marker_color=[
                "#ef4444" if v >= 50 else ("#f59e0b" if v >= 20 else "#10b981")
                for v in annual_d.values
            ],
            hovertemplate="Year %{x}<br>%{y:.0f}% of weeks in Strategy D<extra></extra>",
        ), row=1, col=1)

        # SPY vol line
        if ra_df is not None:
            spy_ret_w  = ra_df["Buy_Hold_SPY"].pct_change().fillna(0.0)
            spy_rvol_w = spy_ret_w.rolling(8).std() * np.sqrt(52) * 100  # in %
            fig_reg.add_trace(go.Scatter(
                x=spy_rvol_w.index, y=spy_rvol_w,
                name="SPY 8-Week Realized Vol",
                line=dict(color="#6366f1", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.13)",
                hovertemplate="%{x|%Y-%m-%d}<br>SPY 8w vol: %{y:.1f}%<extra></extra>",
            ), row=2, col=1)

            # Threshold line
            fig_reg.add_hline(
                y=15, row=2, col=1,
                line_color="#f59e0b", line_dash="dash", line_width=1.5,
                annotation_text="15% threshold",
                annotation_position="bottom right",
                annotation_font_color="#f59e0b",
            )

            # Shade regime-D periods on vol plot
            if len(in_d_mode) > 0:
                bear_start = None
                for dt, is_d in in_d_mode.items():
                    if is_d and bear_start is None:
                        bear_start = dt
                    elif not is_d and bear_start is not None:
                        fig_reg.add_vrect(x0=bear_start, x1=dt, row=2, col=1,
                                          fillcolor="rgba(245,158,11,0.08)", line_width=0, layer="below")
                        bear_start = None
                if bear_start is not None:
                    fig_reg.add_vrect(x0=bear_start, x1=in_d_mode.index[-1], row=2, col=1,
                                      fillcolor="rgba(245,158,11,0.08)", line_width=0, layer="below")

        fig_reg.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            height=520,
            margin=dict(l=60, r=20, t=60, b=40),
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig_reg.update_xaxes(showgrid=True, gridcolor="#1e293b")
        fig_reg.update_yaxes(showgrid=True, gridcolor="#1e293b")
        fig_reg.update_yaxes(ticksuffix="%", row=1, col=1)
        fig_reg.update_yaxes(title="Vol (%)", ticksuffix="%", row=2, col=1)

        st.plotly_chart(fig_reg, use_container_width=True)
        st.caption(
            "🟡 Shaded = Strategy D active. Yellow = ≥50% of year in D. "
            "Orange = 20–50%. Green = <20%."
        )
    else:
        st.info("Regime history unavailable — hybrid curve data not loaded.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Year-by-Year Returns
    # ════════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">📅 Section 5 — Year-by-Year Returns</div>',
                unsafe_allow_html=True)

    if not data_missing:
        yearly: dict[str, pd.Series] = {}

        for label, series in curves.items():
            r = series.pct_change().dropna()
            yearly[label] = (1 + r).groupby(r.index.year).prod() - 1

        # Align years
        all_years = sorted(set().union(*[s.index for s in yearly.values()]))

        fig_yr = go.Figure()
        bar_colors = {
            H8B_LABEL: "#f59e0b",
            "A: 80%SysE/20%R3 static":        "#6366f1",
            "D: 15% vol-target on A":          "#10b981",
            "SPY Buy & Hold":                   "#64748b",
        }

        for label, yseries in yearly.items():
            vals = [yseries.get(yr, np.nan) * 100 for yr in all_years]
            fig_yr.add_trace(go.Bar(
                name=label,
                x=all_years,
                y=vals,
                marker_color=bar_colors.get(label, "#94a3b8"),
                hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
            ))

        fig_yr.add_hline(y=0, line_color="#475569", line_width=1)

        fig_yr.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            barmode="group",
            bargap=0.15,
            bargroupgap=0.05,
            yaxis_title="Annual Return (%)",
            yaxis_ticksuffix="%",
            xaxis_title="Year",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=450,
            margin=dict(l=60, r=20, t=40, b=40),
            hovermode="x unified",
        )
        fig_yr.update_xaxes(showgrid=False)
        fig_yr.update_yaxes(showgrid=True, gridcolor="#1e293b")

        st.plotly_chart(fig_yr, use_container_width=True)
        st.caption(
            "🟡 Hybrid A/D (H8b)  |  🟣 Strategy A (pure)  |  "
            "🟢 Strategy D (always)  |  ⬛ SPY Buy & Hold"
        )
    else:
        st.info("Year-by-year returns unavailable — equity curve data not loaded.")

    st.divider()

    # ════════════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Debounce Grid Results
    # ════════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-hdr">🔧 Section 6 — Debounce Grid Results</div>',
                unsafe_allow_html=True)
    st.caption(
        "Two debounce methodologies compared: **[EntryDeb]** requires N consecutive weeks "
        "*above* threshold before switching to D (from the extended hybrid backtest); "
        "**[ExitDeb]** requires N consecutive weeks *below* threshold before reverting to A "
        "(from `hybrid_debounce_study.py`, the originally-scripted grid). Entry-debounce wins: "
        "best Calmar 1.404 vs 1.279 for the best exit-debounce variant."
    )

    deb_df = load_debounce_results()
    if deb_df is not None:
        # Filter to grid/comparison variants only — both methodologies use bracket prefixes
        grid_mask = deb_df["Strategy"].str.startswith(("[EntryDeb]", "[ExitDeb]"))
        grid_df   = deb_df[grid_mask].copy()

        if not grid_df.empty:
            show_cols = [
                "Strategy", "Methodology", "Vol_Thresh_%", "Exit_Weeks",
                "CAGR_%", "Max_DD_%", "Calmar", "Sharpe", "Sortino",
                "Pct_Defensive_%", "N_Switches"
            ]
            available = [c for c in show_cols if c in grid_df.columns]
            grid_show = grid_df[available].sort_values("Calmar", ascending=False).reset_index(drop=True)

            def highlight_top(row):
                if row.name == 0:
                    return ["background-color: #1e3a1e; color: #86efac; font-weight:700"] * len(row)
                return [""] * len(row)

            styled_grid = grid_show.style.apply(highlight_top, axis=1)
            st.dataframe(styled_grid, use_container_width=True, hide_index=True)

            best_row = grid_show.iloc[0]
            # Compare against the raw (no-debounce) H8b baseline, whichever labeling is present
            h8b_mask = deb_df["Strategy"].str.contains("H8b_baseline", na=False)
            h8b_calmar = deb_df.loc[h8b_mask, "Calmar"].values
            if len(h8b_calmar) > 0:
                improve = (best_row["Calmar"] - h8b_calmar[0]) / abs(h8b_calmar[0]) * 100
                st.caption(
                    f"🏆 Best overall: **{best_row['Strategy']}** ({best_row.get('Methodology','?')}) — "
                    f"Calmar {best_row['Calmar']:.3f}, "
                    f"CAGR {best_row['CAGR_%']:.1f}%, MaxDD {best_row['Max_DD_%']:.1f}% "
                    f"({improve:+.1f}% Calmar improvement vs raw H8b baseline, no debounce)"
                )
        else:
            st.info("No grid results found in debounce CSV.")
    else:
        st.info(
            "Debounce results not found. Run `exports/best_strategy_study/hybrid_debounce_study.py` "
            "to generate `hybrid_debounce_results.csv`."
        )

    # ── Footer ─────────────────────────────────────────────────────────────────────
    st.markdown("""
    ---
    <div style="color:#475569;font-size:0.8rem;text-align:center">
    Hybrid A/D Strategy · H8b + 2wk entry debounce (SPY 8-week vol ≥ 15% for 2 consecutive weeks → Strategy D) ·
    Backtest 2004–2026 · 5 bps/unit turnover cost · No look-ahead bias (signal shift=1 week)
    </div>
    """, unsafe_allow_html=True)

with tab_live:
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

    state = load_hybrid_live_state(data)
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

    # ── One-line plain-English verdict — is this a regime change or routine drift? ─
    if not act_df.empty:
        buy_val = float(act_df.loc[act_df["Action"].str.contains("BUY"), "Δ Value $"].sum())
        sell_val = float(-act_df.loc[act_df["Action"].str.contains("SELL"), "Δ Value $"].sum())
    else:
        buy_val = sell_val = 0.0
    net_flow = buy_val - sell_val   # positive = net buying (rising net exposure)

    if n_act == 0:
        verdict, v_icon, v_colour = "HOLD — no trades needed, book matches target.", "✅", "#10b981"
    elif in_defensive:
        verdict = (f"DEFENSIVE REBALANCE — vol threshold triggered, scaling exposure to ×{expo:.2f}. "
                  f"{n_act} trade{'s' if n_act != 1 else ''} to reach the reduced target.")
        v_icon, v_colour = "🛡️", "#f59e0b"
    elif abs(net_flow) < 0.01 * pv:
        verdict = (f"REBALANCE ONLY — {n_act} trade{'s' if n_act != 1 else ''} to realign with this "
                  f"week's signal update (weights shifted within Strategy A, no regime change, no "
                  f"net change in overall exposure).")
        v_icon, v_colour = "🔄", "#38bdf8"
    elif net_flow > 0:
        verdict = f"INCREASING EXPOSURE — net buying ${net_flow:,.0f} across {n_act} trades."
        v_icon, v_colour = "🟢", "#22c55e"
    else:
        verdict = f"REDUCING EXPOSURE — net selling ${abs(net_flow):,.0f} across {n_act} trades."
        v_icon, v_colour = "🔴", "#ef4444"

    st.markdown(
        f"""<div style="border-left:4px solid {v_colour};background:#0f172a;border-radius:6px;
                    padding:10px 16px;margin-bottom:14px;">
            <span style="font-size:1.1rem;font-weight:700;color:{v_colour}">{v_icon} {verdict}</span>
        </div>""",
        unsafe_allow_html=True,
    )

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


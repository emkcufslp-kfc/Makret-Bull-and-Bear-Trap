"""
pages/21_🔀_Hybrid_AD_Strategy.py
===================================
Hybrid A/D Regime Strategy Dashboard

Shows the H8b strategy (switch to Strategy D when SPY 8-week realized vol ≥ 15%)
and its debounced variants from the hybrid_debounce_study.

Sections:
  1. Live Signal Today   — current mode (Strategy A or D), SPY vol vs threshold
  2. Equity Curve        — hybrid vs pure A, pure D, SPY (log scale)
  3. Performance Metrics — table: CAGR, Max DD, Calmar, Sharpe, Sortino
  4. Regime History      — shaded area showing A vs D periods over time
  5. Year-by-Year Returns — grouped bar chart hybrid vs A vs D vs SPY
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
    page_title="Hybrid A/D Regime Strategy",
    page_icon="🔀",
    layout="wide",
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

st.title("🔀 Hybrid A/D Regime Strategy")
st.caption(
    "Switches between **Strategy A** (80% SysE / 20% R3, full exposure) and "
    "**Strategy D** (15% vol-target overlay on A) based on SPY 8-week realized volatility. "
    "Best variant: **H8b** — enter D when SPY vol ≥ 15%, exit when vol drops back below threshold."
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent.parent
STUDY_DIR     = ROOT / "exports" / "best_strategy_study"
RA_CSV        = ROOT / "exports" / "trap_regime_backtest" / "risk_allocation" / "risk_allocation_equity_curves.csv"
HYBRID_CSV    = STUDY_DIR / "hybrid_regime_backtest.csv"
DEBOUNCE_CSV  = STUDY_DIR / "hybrid_debounce_results.csv"
RESULTS_JSON  = STUDY_DIR / "hybrid_results.json"

# Column name for H8b in hybrid_regime_backtest.csv
H8B_COL       = "H8b_SPYvol>=15pct"
VOL_THRESHOLD = 0.15   # 15%
VOL_WINDOW    = 8      # 8 weeks


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_market_data():
    from utils.data_engine import get_clean_master
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


# ── Compute live SPY realized vol ─────────────────────────────────────────────
def compute_spy_vol(df: pd.DataFrame, weeks: int = 8) -> tuple[float, float, bool]:
    """Returns (current_vol, threshold, in_defensive)."""
    spy_weekly = df["SPY"].resample("W-FRI").last().dropna()
    spy_ret    = spy_weekly.pct_change().dropna()
    if len(spy_ret) < weeks:
        return np.nan, VOL_THRESHOLD, False
    realized = spy_ret.rolling(weeks).std().iloc[-1] * np.sqrt(52)
    return float(realized), VOL_THRESHOLD, float(realized) >= VOL_THRESHOLD


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Live Signal Today
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">📡 Section 1 — Live Signal Today</div>',
            unsafe_allow_html=True)

try:
    market_df = load_market_data()
    spy_vol, threshold, in_defensive = compute_spy_vol(market_df)
    live_ok = True
except Exception as e:
    spy_vol, threshold, in_defensive = np.nan, VOL_THRESHOLD, False
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
        st.info("🛡️ **Strategy D is active.** Vol-target overlay reduces exposure. "
                "Strategy A resumes once SPY 8-week vol drops below 15%.")
    else:
        st.markdown(
            '<span class="badge-strategy-a">MODE: STRATEGY A<br>'
            '<small style="font-size:0.85rem;font-weight:400">'
            'Bull / Low-Vol — Full Exposure</small></span>',
            unsafe_allow_html=True
        )
        st.markdown("")
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
SPY 8-week realized vol ≥ <b>15%</b> annualized — triggered <i>immediately</i>.<br><br>
<b>Exit → Strategy A (bull / full exposure)</b><br>
SPY vol drops back below 15% threshold (1-week, symmetric for H8b).<br><br>
<b>Debounced variants</b> (exit after N consecutive weeks below threshold)<br>
reduce whipsaw switching at the cost of slightly lower Calmar.
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

# Build named curve dict
CURVE_DEFS = {
    "H8b — Hybrid A/D (SPY vol 15%)": (hybrid_df, H8B_COL),
    "A: 80%SysE/20%R3 static":        (hybrid_df, "A: 80%SysE/20%R3 static"),
    "D: 15% vol-target on A":          (hybrid_df, "D: 15%vol-target on A (always)"),
    "SPY Buy & Hold":                   (hybrid_df, "SPY Buy & Hold"),
}

COLORS = {
    "H8b — Hybrid A/D (SPY vol 15%)": "#f59e0b",
    "A: 80%SysE/20%R3 static":        "#6366f1",
    "D: 15% vol-target on A":          "#10b981",
    "SPY Buy & Hold":                   "#94a3b8",
}
DASHES = {
    "H8b — Hybrid A/D (SPY vol 15%)": "solid",
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
        width  = 2.5 if label == "H8b — Hybrid A/D (SPY vol 15%)" else 1.5
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
    if "H8b — Hybrid A/D (SPY vol 15%)" in curves and "A: 80%SysE/20%R3 static" in curves:
        h8b_s = curves["H8b — Hybrid A/D (SPY vol 15%)"]
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
                help="H8b improves max drawdown by spending ~34% of time in Strategy D"
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
        "H8b — Hybrid A/D (SPY vol 15%)": "#f59e0b",
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
    "Results from `hybrid_debounce_study.py` — 16-combo grid of vol threshold × exit debounce weeks. "
    "Lower exit weeks = faster re-entry to Strategy A. Higher = fewer whipsaw switches."
)

deb_df = load_debounce_results()
if deb_df is not None:
    # Filter to grid variants only
    grid_mask = deb_df["Strategy"].str.startswith(("Deb_", "Comp_"))
    grid_df   = deb_df[grid_mask].copy()

    if not grid_df.empty:
        show_cols = [
            "Strategy", "Vol_Thresh_%", "Exit_Weeks",
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
        h8b_calmar = deb_df[deb_df["Strategy"].str.contains("H8b_baseline")]["Calmar"].values
        if len(h8b_calmar) > 0:
            improve = (best_row["Calmar"] - h8b_calmar[0]) / abs(h8b_calmar[0]) * 100
            st.caption(
                f"🏆 Best: **{best_row['Strategy']}** — "
                f"Calmar {best_row['Calmar']:.3f}, "
                f"CAGR {best_row['CAGR_%']:.1f}%, MaxDD {best_row['Max_DD_%']:.1f}% "
                f"({improve:+.1f}% Calmar improvement vs H8b baseline)"
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
Hybrid A/D Strategy · H8b (SPY 8-week vol ≥ 15% → Strategy D) ·
Backtest 2004–2026 · 5 bps/unit turnover cost · No look-ahead bias (signal shift=1 week)
</div>
""", unsafe_allow_html=True)

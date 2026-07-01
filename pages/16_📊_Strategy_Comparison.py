"""
pages/16_📊_Strategy_Comparison.py
====================================
Strategy Comparison — System E vs R3 Vol-Adjusted

Sections:
  1. Live Signals Today (side-by-side)
  2. Performance Comparison (20-year equity curves, log scale)
  3. Metrics Table (CAGR, Sharpe, Sortino, Max DD, Calmar, Beta, Alpha, …)
  4. Year-by-Year Returns (grouped bar chart)
  5. When to Use Which (narrative explainer)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Strategy Comparison — E vs R3",
    page_icon="⚔️",
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
.badge-bull  { background:#10b981; color:#fff; padding:4px 14px; border-radius:20px;
               font-weight:700; font-size:1.1rem; display:inline-block; }
.badge-bear  { background:#ef4444; color:#fff; padding:4px 14px; border-radius:20px;
               font-weight:700; font-size:1.1rem; display:inline-block; }
.badge-caution { background:#f59e0b; color:#000; padding:4px 14px; border-radius:20px;
               font-weight:700; font-size:1.1rem; display:inline-block; }
.badge-unknown { background:#6b7280; color:#fff; padding:4px 14px; border-radius:20px;
               font-weight:700; font-size:1.1rem; display:inline-block; }
.section-hdr { color:#e2e8f0; font-size:1.3rem; font-weight:700;
               border-left:4px solid #6366f1; padding-left:10px; margin:24px 0 12px; }
.alloc-row   { display:flex; align-items:center; gap:8px; margin:4px 0; }
.alloc-bar   { height:12px; border-radius:6px; }
.ticker-chip { background:#1e293b; color:#93c5fd; padding:2px 10px;
               border-radius:12px; font-size:0.85rem; font-family:monospace;
               border:1px solid #334155; display:inline-block; margin:2px; }
</style>
""", unsafe_allow_html=True)

st.title("⚔️ Strategy Comparison — System E vs R3 Vol-Adjusted")
st.caption("Both strategies use the same NQ100 momentum universe and Bull/Bear Trap regime signals.")

# ── Data loaders ──────────────────────────────────────────────────────────────
from utils.data_engine import get_clean_master

BACKTEST_DIR = Path(__file__).parent.parent / "exports" / "trap_regime_backtest" / "risk_allocation"

@st.cache_data(ttl=3600)
def load_market_data():
    return get_clean_master()

@st.cache_data(ttl=86400)
def load_backtest_data():
    eq   = pd.read_csv(BACKTEST_DIR / "risk_allocation_equity_curves.csv",
                       index_col="Date", parse_dates=True)
    perf = pd.read_csv(BACKTEST_DIR / "risk_allocation_performance.csv")
    ann  = pd.read_csv(BACKTEST_DIR / "risk_allocation_annual.csv")
    return eq, perf, ann

# ── R3 live signal helpers (identical to page 15) ─────────────────────────────
def compute_bull_trap_score(d: pd.DataFrame) -> float:
    if len(d) < 200:
        return 2.5
    latest  = d.iloc[-1]
    prev_mo = d.iloc[max(0, len(d) - 23)]
    curve      = latest["^TNX"] - latest["^IRX"]
    prev_curve = prev_mo["^TNX"] - prev_mo["^IRX"]
    ys = 1.0 if (curve > 0 and prev_curve < 0) else (0.5 if curve > 0 else 0.0)
    vix_ma22 = d["^VIX"].rolling(22).mean().iloc[-1]
    vs = 1.0 if latest["^VIX"] < 15 else (0.5 if latest["^VIX"] < vix_ma22 else 0.0)
    hyg_ief      = d["HYG"] / d["IEF"]
    hyg_ief_ma22 = hyg_ief.rolling(22).mean().iloc[-1]
    cs = 1.0 if hyg_ief.iloc[-1] > hyg_ief_ma22 else 0.0
    spy_200ma = d["SPY"].rolling(200).mean().iloc[-1]
    bs = 1.0 if latest["SPY"] > spy_200ma * 1.05 else (0.5 if latest["SPY"] > spy_200ma else 0.0)
    spy_mom22 = latest["SPY"] / prev_mo["SPY"] - 1
    acc = 1.0 if spy_mom22 > 0.02 else (0.5 if spy_mom22 > 0 else 0.0)
    liq = 1.0 if ("TIP" in d.columns and len(d) >= 22 and d["TIP"].iloc[-1] / d["TIP"].iloc[-22] - 1 > 0) else (0.5 if "TIP" not in d.columns else 0.0)
    return round(min(10.0, ys + vs + cs + bs + acc + liq + 0.5 + 0.5 + 1.0), 2)

def _norm(val, lower, upper, inverted=False):
    if inverted:
        if val >= lower: return 0.0
        if val <= upper: return 1.0
        return round((lower - val) / (lower - upper), 4)
    else:
        if val <= lower: return 0.0
        if val >= upper: return 1.0
        return round((val - lower) / (upper - lower), 4)

def compute_bear_trap_score(d: pd.DataFrame) -> float:
    if len(d) < 252:
        return 0.5
    latest = d.iloc[-1]
    curve         = latest["^TNX"] - latest["^IRX"]
    irx_ma120     = d["^IRX"].rolling(120).mean().iloc[-1]
    hyg_ief       = d["HYG"] / d["IEF"]
    hyg_ief_ma252 = hyg_ief.rolling(252).mean().iloc[-1]
    spy_200ma     = d["SPY"].rolling(200).mean().iloc[-1]
    return round(
        _norm(curve,           1.5,               -0.5,               inverted=True) * 0.25 +
        _norm(latest["^IRX"],  irx_ma120 * 0.8,   irx_ma120 * 1.2)                  * 0.20 +
        _norm(hyg_ief.iloc[-1],hyg_ief_ma252*1.05,hyg_ief_ma252*0.90,inverted=True) * 0.20 +
        _norm(latest["SPY"],   spy_200ma * 1.05,  spy_200ma * 0.95,  inverted=True) * 0.15 +
        _norm(latest["^VIX"],  15, 35)                                                * 0.10 +
        0.65 * 0.05 + 0.50 * 0.05,
        4
    )

def compute_r3_deploy(bt: float, bear: float, spy_weekly_rets: pd.Series) -> dict:
    TARGET_VOL    = 0.12
    net_conf      = (bt / 10.0) * (1.0 - bear)
    last4         = spy_weekly_rets.dropna().tail(4)
    spy_vol       = last4.std() * np.sqrt(52) if len(last4) >= 2 else TARGET_VOL
    vol_scalar    = spy_vol / TARGET_VOL
    vol_adj       = net_conf * min(1.5, 1.0 / vol_scalar) if vol_scalar > 0 else net_conf
    deploy        = float(np.clip(0.10 + vol_adj * 0.90, 0.10, 1.00))
    return {"bt_score": bt, "bear_score": bear, "deploy": deploy,
            "deploy_pct": round(deploy * 100, 1), "spy_vol": round(spy_vol, 4)}

# ── Compute top-3 NQ100 picks from live data (shared by both strategies) ───────
from strategy_e.engine import NQ100_TICKERS, compute_system_e_signal

def compute_r3_top3(df: pd.DataFrame) -> list[str]:
    """Same momentum ranking as System E — R3 uses the same picks, different weights."""
    nq_tickers = [t for t in NQ100_TICKERS if t in df.columns]
    if not nq_tickers or len(df) < 46 * 5:
        return []
    w     = df[nq_tickers].resample("W-FRI").last().ffill()
    if len(w) < 46:
        return []
    mom44 = w.iloc[-1] / w.iloc[-45] - 1
    mom4  = w.iloc[-1] / w.iloc[-5]  - 1
    score = (mom44 - mom4).dropna()
    return list(score.nlargest(3).index) if len(score) >= 5 else []

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Live Signals Today
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">📡 Section 1 — Live Signals Today</div>', unsafe_allow_html=True)

with st.spinner("Computing live signals from market data …"):
    df = load_market_data()

if df is None or df.empty:
    st.error("Market data unavailable. Please refresh data from the Home page.")
    st.stop()

sig_e  = compute_system_e_signal(df)
bt_val = compute_bull_trap_score(df)
bear_val = compute_bear_trap_score(df)
spy_w_ret = df["SPY"].resample("W-FRI").last().pct_change()
r3     = compute_r3_deploy(bt_val, bear_val, spy_w_ret)
r3_top3 = compute_r3_top3(df)

col_e, col_r3 = st.columns(2)

# ── System E ─────────────────────────────────────────────────────────────────
with col_e:
    st.subheader("🤖 System E — Dual 200d MA Gate")

    regime = sig_e["regime"]
    badge_cls = "badge-bull" if regime == "BULL" else ("badge-bear" if regime == "BEAR" else "badge-unknown")
    st.markdown(f'<span class="{badge_cls}">{regime}</span>', unsafe_allow_html=True)
    st.markdown("")

    c1, c2 = st.columns(2)
    c1.metric("SPY Price",   f"${sig_e['spy_price']:,.2f}")
    c1.metric("SPY 200d MA", f"${sig_e['spy_200ma']:,.2f}",
              delta="✓ Above" if sig_e["spy_price"] > sig_e["spy_200ma"] else "✗ Below",
              delta_color="normal" if sig_e["spy_price"] > sig_e["spy_200ma"] else "inverse")
    c2.metric("Breadth",     f"{sig_e['breadth_pct']}%",
              delta="✓ ≥40%" if sig_e["breadth_gate"] else "✗ <40%",
              delta_color="normal" if sig_e["breadth_gate"] else "inverse")
    c2.metric("Bull Gate",   "✅ OPEN" if sig_e["bull_gate"] else "🔴 CLOSED")

    st.markdown("**Top-3 Momentum Picks**")
    mom = sig_e.get("mom_scores", {})
    for t in sig_e["top3"]:
        sc = mom.get(t, 0.0)
        st.markdown(f'<span class="ticker-chip">{t}</span> &nbsp; score: {sc:+.3f}', unsafe_allow_html=True)

    st.markdown("**Current Allocation**")
    alloc_e = sig_e.get("allocation", {})
    for ticker, wt in alloc_e.items():
        pct = wt * 100
        color = "#6366f1" if ticker in sig_e["top3"] else ("#3b82f6" if ticker == "SPY" else "#94a3b8")
        st.markdown(
            f'<div class="alloc-row">'
            f'<span style="width:52px;font-size:0.8rem;font-family:monospace;color:#e2e8f0">{ticker}</span>'
            f'<div class="alloc-bar" style="width:{int(pct*2.5)}px;background:{color}"></div>'
            f'<span style="font-size:0.85rem;color:#94a3b8">{pct:.1f}%</span>'
            f'</div>', unsafe_allow_html=True
        )

# ── R3 Vol-Adjusted ───────────────────────────────────────────────────────────
with col_r3:
    st.subheader("📊 R3 Vol-Adjusted — Trap Regime Gate")

    deploy_pct = r3["deploy_pct"]
    if deploy_pct >= 75:
        r3_badge, r3_cls = "HIGH DEPLOY", "badge-bull"
    elif deploy_pct >= 40:
        r3_badge, r3_cls = "MODERATE", "badge-caution"
    else:
        r3_badge, r3_cls = "LOW DEPLOY", "badge-bear"

    st.markdown(f'<span class="{r3_cls}">{r3_badge} ({deploy_pct}%)</span>', unsafe_allow_html=True)
    st.markdown("")

    c1, c2 = st.columns(2)
    c1.metric("Bull Trap Score", f"{bt_val:.2f} / 10",
              delta="Bullish" if bt_val >= 6 else ("Neutral" if bt_val >= 4 else "Bearish"),
              delta_color="normal" if bt_val >= 6 else ("off" if bt_val >= 4 else "inverse"))
    c2.metric("Bear Trap Score", f"{bear_val:.3f}",
              delta="Low Risk" if bear_val < 0.45 else ("Elevated" if bear_val < 0.60 else "High Risk"),
              delta_color="normal" if bear_val < 0.45 else ("off" if bear_val < 0.60 else "inverse"))
    c1.metric("Deploy %",        f"{deploy_pct}%")
    c2.metric("SPY 4w Vol",      f"{r3['spy_vol']*100:.1f}%")

    st.markdown("**Top-3 Momentum Picks** (same universe)")
    for t in r3_top3:
        sc = sig_e.get("mom_scores", {}).get(t, 0.0)
        st.markdown(f'<span class="ticker-chip">{t}</span> &nbsp; score: {sc:+.3f}', unsafe_allow_html=True)

    st.markdown("**Implied Allocation** (vol-scaled)")
    deploy = r3["deploy"] / 3
    stock_wt = deploy if r3_top3 else 0.0
    spy_wt   = max(0.10, r3["deploy"] - stock_wt * len(r3_top3))
    cash_wt  = max(0.0, 1.0 - spy_wt - stock_wt * len(r3_top3))
    r3_alloc = {}
    if spy_wt > 0:
        r3_alloc["SPY"] = spy_wt
    for t in r3_top3:
        r3_alloc[t] = stock_wt
    if cash_wt > 0.01:
        r3_alloc["CASH"] = cash_wt

    for ticker, wt in r3_alloc.items():
        pct = wt * 100
        color = "#10b981" if ticker in r3_top3 else ("#3b82f6" if ticker == "SPY" else "#475569")
        st.markdown(
            f'<div class="alloc-row">'
            f'<span style="width:52px;font-size:0.8rem;font-family:monospace;color:#e2e8f0">{ticker}</span>'
            f'<div class="alloc-bar" style="width:{int(pct*2.5)}px;background:{color}"></div>'
            f'<span style="font-size:0.85rem;color:#94a3b8">{pct:.1f}%</span>'
            f'</div>', unsafe_allow_html=True
        )

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Performance Comparison (20-year equity curves)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">📈 Section 2 — 20-Year Performance Comparison</div>', unsafe_allow_html=True)

try:
    eq_df, perf_df, ann_df = load_backtest_data()
except Exception as e:
    st.error(f"Could not load backtest data: {e}")
    st.stop()

# Pull the three curves we want
curves = {}
if "Sys_E_Original" in eq_df.columns:
    curves["System E (Original)"] = eq_df["Sys_E_Original"]
if "R3_Vol_Adj" in eq_df.columns:
    curves["R3 Vol-Adjusted"] = eq_df["R3_Vol_Adj"]
if "Buy_Hold_SPY" in eq_df.columns:
    curves["SPY Buy & Hold"] = eq_df["Buy_Hold_SPY"]

COLORS = {
    "System E (Original)": "#6366f1",
    "R3 Vol-Adjusted":     "#10b981",
    "SPY Buy & Hold":      "#94a3b8",
}
DASHES = {
    "System E (Original)": "solid",
    "R3 Vol-Adjusted":     "solid",
    "SPY Buy & Hold":      "dash",
}

fig_eq = go.Figure()

for name, series in curves.items():
    s = series.dropna()
    fig_eq.add_trace(go.Scatter(
        x=s.index, y=s,
        name=name,
        line=dict(color=COLORS[name], width=2.5 if name != "SPY Buy & Hold" else 1.5,
                  dash=DASHES[name]),
        hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>$%{{y:,.0f}}<extra></extra>",
    ))

# Shaded drawdown for System E
if "System E (Original)" in curves:
    s = curves["System E (Original)"].dropna()
    rolling_max = s.cummax()
    dd = (s - rolling_max) / rolling_max
    bear_mask = dd < -0.10
    start = None
    for date, is_bear in bear_mask.items():
        if is_bear and start is None:
            start = date
        elif not is_bear and start is not None:
            fig_eq.add_vrect(x0=start, x1=date,
                             fillcolor="#ef444430", line_width=0, layer="below")
            start = None
    if start is not None:
        fig_eq.add_vrect(x0=start, x1=s.index[-1],
                         fillcolor="#ef444430", line_width=0, layer="below")

fig_eq.update_layout(
    template="plotly_dark",
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    yaxis_type="log",
    yaxis_title="Portfolio Value (log scale, $)",
    xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=480,
    margin=dict(l=60, r=20, t=40, b=40),
    hovermode="x unified",
)
fig_eq.update_xaxes(showgrid=True, gridcolor="#1e293b")
fig_eq.update_yaxes(showgrid=True, gridcolor="#1e293b")

st.plotly_chart(fig_eq, use_container_width=True)
st.caption("Shaded regions = System E drawdowns >10%. Log scale. Starting value $100,000.")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Metrics Table
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">📊 Section 3 — Performance Metrics</div>', unsafe_allow_html=True)

STRATEGY_MAP = {
    "Buy_Hold_SPY":  "SPY Buy & Hold",
    "Sys_E_Original":"System E (Original)",
    "R3_Vol_Adj":    "R3 Vol-Adjusted",
}
SHOW_STRATEGIES = ["Sys_E_Original", "R3_Vol_Adj", "Buy_Hold_SPY"]
METRIC_COLS = [
    ("CAGR_%",       "CAGR",       "{:.1f}%",  False),
    ("Sharpe",       "Sharpe",     "{:.3f}",   False),
    ("Sortino",      "Sortino",    "{:.3f}",   False),
    ("Max_DD_%",     "Max DD",     "{:.1f}%",  True),
    ("Calmar",       "Calmar",     "{:.3f}",   False),
    ("Win_Rate_%",   "Win Rate",   "{:.1f}%",  False),
    ("Beta",         "Beta",       "{:.3f}",   True),
    ("Alpha_%",      "Alpha/yr",   "{:.2f}%",  False),
    ("AvgDeploy_%",  "Avg Deploy", "{:.1f}%",  False),
    ("Best_Year_%",  "Best Year",  "{:.1f}%",  False),
    ("Worst_Year_%", "Worst Year", "{:.1f}%",  True),
]

perf_filtered = perf_df[perf_df["Strategy"].isin(SHOW_STRATEGIES)].copy()
perf_filtered["_order"] = perf_filtered["Strategy"].map({s: i for i, s in enumerate(SHOW_STRATEGIES)})
perf_filtered = perf_filtered.sort_values("_order")

# Render as styled metric cards
n = len(perf_filtered)
cols_m = st.columns(n)
for ci, (_, row) in enumerate(perf_filtered.iterrows()):
    strat = STRATEGY_MAP.get(row["Strategy"], row["Strategy"])
    color = {"System E (Original)": "#6366f1", "R3 Vol-Adjusted": "#10b981",
             "SPY Buy & Hold": "#94a3b8"}.get(strat, "#e2e8f0")
    with cols_m[ci]:
        st.markdown(f"<h4 style='color:{color};margin-bottom:8px'>{strat}</h4>", unsafe_allow_html=True)
        for col, label, fmt, lower_is_neutral in METRIC_COLS:
            if col not in row.index:
                continue
            val = row[col]
            if pd.isna(val) or val == "":
                continue
            try:
                val_f = float(val)
                display = fmt.format(val_f)
            except (ValueError, TypeError):
                display = str(val)
            st.metric(label, display)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Year-by-Year Returns
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">📅 Section 4 — Year-by-Year Returns</div>', unsafe_allow_html=True)

if "Year" in ann_df.columns:
    ann_df = ann_df.set_index("Year")

show_cols = {}
if "Sys_E_Original" in ann_df.columns:
    show_cols["System E (Original)"] = ann_df["Sys_E_Original"]
if "R3_Vol_Adj" in ann_df.columns:
    show_cols["R3 Vol-Adjusted"] = ann_df["R3_Vol_Adj"]
if "Buy_Hold_SPY" in ann_df.columns:
    show_cols["SPY Buy & Hold"] = ann_df["Buy_Hold_SPY"]

if show_cols:
    years = ann_df.index.tolist()
    fig_bar = go.Figure()

    BAR_COLORS = {
        "System E (Original)": "#6366f1",
        "R3 Vol-Adjusted":     "#10b981",
        "SPY Buy & Hold":      "#94a3b8",
    }

    for name, series in show_cols.items():
        vals = series.reindex(years).fillna(0).tolist()
        bar_colors = [
            BAR_COLORS[name] if v >= 0 else "#ef4444"
            for v in vals
        ]
        fig_bar.add_trace(go.Bar(
            name=name,
            x=[str(y) for y in years],
            y=vals,
            marker_color=BAR_COLORS[name],
            opacity=0.85,
            hovertemplate=f"<b>{name}</b> %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        barmode="group",
        yaxis_title="Annual Return (%)",
        xaxis_title="Year",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=420,
        margin=dict(l=60, r=20, t=40, b=40),
        hovermode="x unified",
    )
    fig_bar.update_xaxes(showgrid=False, gridcolor="#1e293b", tickangle=-45)
    fig_bar.update_yaxes(showgrid=True, gridcolor="#1e293b", zeroline=True,
                         zerolinecolor="#475569", zerolinewidth=1.5)
    fig_bar.add_hline(y=0, line_color="#475569", line_width=1)

    st.plotly_chart(fig_bar, use_container_width=True)

    # Compact table
    with st.expander("📋 Annual Returns Table"):
        display_df = pd.DataFrame(show_cols).round(1)
        display_df.index.name = "Year"
        display_df.columns = [c.replace(" (Original)", "") for c in display_df.columns]

        def color_return(v):
            try:
                fv = float(v)
                if fv > 20:   return "color:#10b981;font-weight:700"
                if fv > 0:    return "color:#6ee7b7"
                if fv > -10:  return "color:#fca5a5"
                return "color:#ef4444;font-weight:700"
            except:
                return ""

        st.dataframe(
            display_df.style.applymap(color_return),
            use_container_width=True,
        )

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — When to Use Which
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-hdr">🧠 Section 5 — When to Use Which</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
<div style='background:#1e1b4b;border:1px solid #4f46e5;border-radius:12px;padding:20px;'>
<h4 style='color:#818cf8;margin-top:0'>🤖 System E — Maximum Compounding</h4>

**Best for:** Long-term wealth building, investors with high risk tolerance and long horizons.

**Key traits:**
- Fully deployed (100%) during BULL regimes
- NQ100 top-3 momentum picks at 21.7% each = concentrated risk, outsized returns
- Dual 200d MA gate is **binary** — either all-in or defensive
- CAGR ~31.7%, Sharpe 1.21 — best absolute returns in the ecosystem
- Max drawdown –26.8% (comparable to R3 despite much higher returns)
- Beta 0.53 — leveraged exposure to tech/growth trends

**Use when:** You want maximum compounding and can stomach volatile years.
You are investing capital you won't need for 5+ years.

</div>
""", unsafe_allow_html=True)

with col_b:
    st.markdown("""
<div style='background:#052e16;border:1px solid #16a34a;border-radius:12px;padding:20px;'>
<h4 style='color:#34d399;margin-top:0'>📊 R3 Vol-Adjusted — Smooth Compounding</h4>

**Best for:** Risk-conscious investors, leveraged accounts, and portfolios where drawdown matters more than max return.

**Key traits:**
- Deployment scales continuously 10–100% based on trap scores + realised vol
- Never goes fully binary — always some SPY exposure
- Same NQ100 top-3 picks, but vol-normalised weighting
- CAGR ~18.3%, Sharpe 1.25 — **highest Sharpe in the ecosystem**
- Max drawdown –25.4% — nearly identical to System E but at half the beta
- Beta 0.28, Alpha 14.3% — true alpha generation with low market correlation
- Avg deployment only 44% — capital-efficient for leveraged accounts

**Use when:** You want consistent compounding with lower vol. Ideal for
margin accounts, institutional mandates, or when you want to sleep at night.

</div>
""", unsafe_allow_html=True)

st.markdown("""
---
### 🔄 Common Foundation
Both strategies share the same core ingredients:
- **Universe:** NQ100 top momentum stocks (91-ticker list)
- **Signal:** 44-week minus 4-week skip-month momentum ranking
- **Breadth Filter:** ≥40% of NQ100 must have positive momentum to deploy
- **Regime:** Bull/Bear Trap composite scoring (SPY + NQ100 EW vs. 200d MA for System E;
  multi-factor trap scoring for R3)

The difference is purely in **how aggressively** the signal is acted on:
System E goes all-in; R3 scales position size to risk conditions.
""")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e293b;margin-top:40px'>
<p style='color:#475569;font-size:0.8rem;text-align:center'>
Backtest period: Jan 2004 – present · $100,000 initial capital · Weekly rebalancing ·
No transaction costs or slippage · Past performance does not guarantee future results.
</p>
""", unsafe_allow_html=True)

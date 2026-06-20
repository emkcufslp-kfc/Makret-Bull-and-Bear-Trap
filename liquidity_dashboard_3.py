from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils.liquidity_dashboard import LiquidityThresholds, add_historical_signals, build_liquidity_dashboard


ROOT_DIR = Path(__file__).parent
CRASH_STUDY_DIR = ROOT_DIR / "exports" / "crash_predictor_study"

st.set_page_config(page_title="Liquidity Dashboard 3", page_icon="LQ", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] { display: none !important; }
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(245,158,11,0.08), transparent 28%),
            linear-gradient(180deg, #08111e 0%, #0b1423 58%, #0d1627 100%);
    }
    .block-container {
        max-width: 1420px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid rgba(71,85,105,0.45);
    }
    div[data-testid="stMetric"] {
        background: transparent;
    }
    div[data-testid="stMetric"] label {
        color: #cbd5e1 !important;
        font-weight: 800 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 18px;
        border-bottom: 1px solid rgba(71,85,105,0.55);
    }
    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1;
        font-weight: 800;
    }
    .stTabs [aria-selected="true"] {
        color: #ff5c57 !important;
        border-bottom-color: #ff5c57 !important;
    }
    .lq-panel {
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(14,25,42,0.98), rgba(10,19,34,0.92));
        border: 1px solid rgba(71,85,105,0.52);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 18px 36px rgba(2,6,23,0.18);
    }
    .lq-status-ledger {
        border: 1px solid rgba(245,158,11,0.5);
        border-radius: 14px;
        padding: 14px 16px;
        background: rgba(245,158,11,0.12);
        margin: 12px 0 14px 0;
    }
    .lq-status-ledger-title {
        color: #fbbf24;
        font-size: 0.82rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .lq-status-ledger-body {
        color: #f8fafc;
        line-height: 1.5;
    }
    .lq-pill {
        display:inline-flex;
        align-items:center;
        border-radius:999px;
        padding: 3px 9px;
        border:1px solid rgba(148,163,184,0.35);
        font-size:0.78rem;
        font-weight:900;
        margin-right:6px;
        white-space:nowrap;
    }
    .lq-mini-grid {
        display:grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 12px;
        margin: 10px 0 14px 0;
    }
    .lq-mini-card {
        background: rgba(2,6,23,0.30);
        border: 1px solid rgba(51,65,85,0.55);
        border-radius: 14px;
        padding: 12px 14px;
        min-height: 86px;
    }
    .lq-mini-label {
        color:#cbd5e1;
        font-size:0.82rem;
        font-weight:800;
        margin-bottom: 8px;
    }
    .lq-mini-value {
        font-size:1.9rem;
        line-height:1;
        font-weight:900;
    }
    @media (max-width: 1200px) {
        .lq-mini-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600, show_spinner=False)
def load_liquidity_data(
    years: int,
    end_date: dt.date,
    reserve_watch: float,
    reserve_stress: float,
    cp_watch: float,
    cp_stress: float,
    cache_version: int = 2,
):
    thresholds = LiquidityThresholds(
        reserve_watch_pct=reserve_watch,
        reserve_stress_pct=reserve_stress,
        cp_sofr_watch_bps=cp_watch,
        cp_sofr_stress_bps=cp_stress,
    )
    return build_liquidity_dashboard(years=years, end_date=end_date, thresholds=thresholds)


@st.cache_data(ttl=1800, show_spinner=False)
def load_crash_study(cache_version: int = 1):
    features_path = CRASH_STUDY_DIR / "weekly_feature_outcomes.csv"
    eval_path = CRASH_STUDY_DIR / "signal_evaluation.csv"
    grid_path = CRASH_STUDY_DIR / "threshold_grid.csv"
    bins_path = CRASH_STUDY_DIR / "composite_risk_bins.csv"
    daily_path = CRASH_STUDY_DIR / "daily_confirmation.csv"
    daily_eval_path = CRASH_STUDY_DIR / "daily_confirmation_evaluation.csv"
    if (
        not features_path.exists()
        or not eval_path.exists()
        or not grid_path.exists()
        or not bins_path.exists()
        or not daily_path.exists()
        or not daily_eval_path.exists()
    ):
        return None, None, None, None, None, None

    features = pd.read_csv(features_path, parse_dates=["date"]).set_index("date").sort_index()
    evals = pd.read_csv(eval_path)
    grid = pd.read_csv(grid_path)
    bins = pd.read_csv(bins_path)
    daily = pd.read_csv(daily_path, parse_dates=["date", "weekly_date"]).set_index("date").sort_index()
    daily_eval = pd.read_csv(daily_eval_path)
    return features, evals, grid, bins, daily, daily_eval


def status_color(status: str) -> str:
    return {
        "Supportive": "#22c55e",
        "Normal": "#22c55e",
        "Watch": "#f59e0b",
        "Warning": "#ff5c57",
        "Stress": "#ff5c57",
    }.get(status, "#94a3b8")


def status_rank(status: str) -> int:
    return {"Normal": 0, "Watch": 1, "Warning": 2, "Stress": 3}.get(str(status), 0)


def status_band_text(status: str) -> str:
    return {
        "Normal": "0-29: risk near normal historical drawdown odds.",
        "Watch": "30-54: elevated correction odds; monitor confirmation.",
        "Warning": "55-74: high-risk mix; D1/D2/D3 are confirming deterioration.",
        "Stress": "75-100: severe risk mix; historically the largest drawdown odds.",
    }.get(str(status), "Unavailable")


def d1_metric_color(metric: str, value) -> str:
    if pd.isna(value) if isinstance(value, float) else False:
        return "#94a3b8"
    if metric == "market_score":
        value = float(value)
        return "#22c55e" if value < 30 else "#f59e0b" if value < 55 else "#ff5c57"
    if metric == "bear_prob":
        value = float(value)
        return "#22c55e" if value < 40 else "#f59e0b" if value < 55 else "#ff5c57"
    if metric == "bull_prob":
        value = float(value)
        return "#22c55e" if value >= 60 else "#f59e0b" if value >= 40 else "#ff5c57"
    if metric == "ma_state":
        text = str(value).lower()
        return "#22c55e" if "bull" in text else "#f59e0b" if "caution" in text else "#ff5c57"
    if metric == "rwra_count":
        value = float(value)
        return "#22c55e" if value <= 1 else "#f59e0b" if value <= 3 else "#ff5c57"
    return "#94a3b8"


def d2_metric_color(metric: str, value) -> str:
    if metric == "level":
        return {"Stable": "#22c55e", "Guarded": "#f59e0b", "Elevated": "#ff5c57", "Critical": "#ff5c57"}.get(str(value), "#94a3b8")
    if metric == "score":
        value = float(value)
        return "#22c55e" if value < 4 else "#f59e0b" if value < 7 else "#ff5c57"
    if metric == "warnings":
        return "#22c55e" if float(value) == 0 else "#ff5c57"
    if metric == "watches":
        return "#22c55e" if float(value) == 0 else "#f59e0b"
    if metric == "credit":
        return "#ff5c57" if bool(value) else "#22c55e"
    return "#94a3b8"


def metric_card_html(title: str, value: str, color: str) -> str:
    return (
        "<div class='lq-mini-card'>"
        f"<div class='lq-mini-label'>{title}</div>"
        f"<div class='lq-mini-value' style='color:{color};'>{value}</div>"
        "</div>"
    )


def fmt(value: float, suffix: str = "", digits: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{digits}f}{suffix}"


def pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:,.{digits}f}%"


def bool_label(value) -> str:
    return "Active" if bool(value) else "Inactive"


def relevant_indicators(row: pd.Series) -> str:
    signals: list[str] = []

    d1_score = row.get("d1_market_regime_score", float("nan"))
    if pd.notna(d1_score) and float(d1_score) >= 30:
        signals.append(f"D1 regime {fmt(d1_score, '', 0)}")
    if bool(row.get("d1_etf_rotation_warning", False)):
        signals.append("D1 ETF rotation warning")
    if str(row.get("d1_200ma_state", "")).lower() not in {"", "bullish", "nan"}:
        signals.append(f"D1 200MA {row.get('d1_200ma_state')}")
    rwra_count = row.get("d1_rwra_bearish_count", 0)
    if pd.notna(rwra_count) and float(rwra_count) > 0:
        signals.append(f"D1 RWRA bearish {fmt(rwra_count, '', 0)}")

    d2_pairs = [
        ("Overextension", "d2_sp500_overextension_status"),
        ("Momentum", "d2_downward_momentum_status"),
        ("Range", "d2_range_breakdown_status"),
        ("Technical", "d2_technical_deterioration_status"),
        ("Breadth", "d2_breadth_worsening_status"),
        ("VIX", "d2_vix_spike_status"),
        ("Breakout", "d2_breakout_failure_status"),
        ("Theme", "d2_theme_momentum_status"),
    ]
    for label, col in d2_pairs:
        status = str(row.get(col, "Normal"))
        if status in {"Watch", "Warning"}:
            signals.append(f"D2 {label} {status}")

    liquidity_score = row.get("liquidity_score", float("nan"))
    if pd.notna(liquidity_score) and float(liquidity_score) >= 30:
        signals.append(f"D3 liquidity {fmt(liquidity_score, '', 0)}")
    reserve_gdp = row.get("reserve_gdp", float("nan"))
    if pd.notna(reserve_gdp) and float(reserve_gdp) < 12:
        signals.append(f"D3 reserves/GDP {fmt(reserve_gdp, '%', 1)}")
    cp_sofr = row.get("cp_sofr_bps", float("nan"))
    if pd.notna(cp_sofr) and float(cp_sofr) >= 25:
        signals.append(f"D3 CP-SOFR {fmt(cp_sofr, ' bps', 0)}")
    if bool(row.get("credit_stress", False)):
        signals.append("D3 credit stress")

    return "; ".join(signals[:8]) if signals else "No major non-normal driver"


def study_as_of(features: pd.DataFrame, as_of_date: dt.date) -> pd.DataFrame:
    return features.loc[features.index <= pd.Timestamp(as_of_date)].copy()


def daily_as_of(daily: pd.DataFrame, as_of_date: dt.date) -> pd.DataFrame:
    return daily.loc[daily.index <= pd.Timestamp(as_of_date)].copy()


def baseline_probability(features: pd.DataFrame, outcome: str) -> float:
    if outcome not in features.columns:
        return float("nan")
    return float(features[outcome].dropna().mean())


def pit_probability(features: pd.DataFrame, as_of_date: dt.date, state: str, threshold: int, horizon: int) -> tuple[float, float, int]:
    outcome = f"drop_{threshold}_{horizon}w"
    if outcome not in features.columns:
        return float("nan"), float("nan"), 0

    cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(weeks=horizon)
    valid = features.loc[features.index <= cutoff].dropna(subset=[outcome, "composite_risk_state"])
    if valid.empty:
        return float("nan"), float("nan"), 0

    same_state = valid[valid["composite_risk_state"].eq(state)]
    if same_state.empty:
        return float("nan"), float(valid[outcome].mean()), 0
    return float(same_state[outcome].mean()), float(valid[outcome].mean()), int(len(same_state))


def composite_probability_table(features: pd.DataFrame, as_of_date: dt.date, state: str) -> pd.DataFrame:
    rows = []
    for horizon in [4, 8, 13, 26]:
        item = {"Horizon": f"{horizon}W"}
        samples = []
        for threshold in [5, 10, 15, 20]:
            value, base, n = pit_probability(features, as_of_date, state, threshold, horizon)
            item[f">={threshold}% Drop"] = pct(value)
            item[f">={threshold}% vs Base"] = f"{(value - base) * 100:+.1f} pts" if pd.notna(value) and pd.notna(base) else "n/a"
            samples.append(n)
        dd_col = f"fwd_max_dd_{horizon}w"
        cutoff = pd.Timestamp(as_of_date) - pd.Timedelta(weeks=horizon)
        valid_dd = features.loc[features.index <= cutoff].dropna(subset=[dd_col, "composite_risk_state"])
        state_dd = valid_dd[valid_dd["composite_risk_state"].eq(state)]
        item["Avg Forward Drawdown"] = pct(float(state_dd[dd_col].mean())) if not state_dd.empty else "n/a"
        item["Completed State Samples"] = max(samples) if samples else 0
        rows.append(item)
    return pd.DataFrame(rows)


def daily_confirmation_table(daily_eval: pd.DataFrame, weekly_state: str, daily_state: str) -> pd.DataFrame:
    row = daily_eval[
        daily_eval["weekly_state"].eq(weekly_state)
        & daily_eval["daily_confirmation_state"].eq(daily_state)
    ]
    if row.empty:
        return pd.DataFrame(
            [{"Window": "No matching history", "Probability": "n/a", "Avg Drawdown": "n/a", "Samples": 0}]
        )
    latest = row.iloc[0]
    return pd.DataFrame(
        [
            {"Window": "4W >=5% drop", "Probability": pct(latest["drop5_4w"]), "Avg Drawdown": "n/a", "Samples": int(latest["n"])},
            {"Window": "8W >=10% drop", "Probability": pct(latest["drop10_8w"]), "Avg Drawdown": "n/a", "Samples": int(latest["n"])},
            {"Window": "13W >=10% drop", "Probability": pct(latest["drop10_13w"]), "Avg Drawdown": pct(latest["avg_dd_13w"]), "Samples": int(latest["n"])},
            {"Window": "26W >=20% drop", "Probability": pct(latest["drop20_26w"]), "Avg Drawdown": pct(latest["avg_dd_26w"]), "Samples": int(latest["n"])},
        ]
    )


def render_daily_confirmation_panel(daily_view: pd.DataFrame, daily_eval: pd.DataFrame) -> None:
    if daily_view.empty:
        st.warning("No daily confirmation rows are available for the selected as-of date.")
        return
    latest = daily_view.iloc[-1]
    weekly_state = str(latest.get("composite_risk_state", "Unavailable"))
    daily_state = str(latest.get("daily_confirmation_state", "Unavailable"))
    daily_score = latest.get("daily_confirmation_score", float("nan"))
    color = "#22c55e" if daily_state == "Clear" else "#f59e0b" if daily_state == "Confirming" else "#ff5c57"
    st.markdown(
        f"""
        <div class="lq-status-ledger" style="border-color:{color}66;background:{color}12;">
            <div class="lq-status-ledger-title" style="color:{color};">DAILY CONFIRMATION OVERLAY</div>
            <div class="lq-status-ledger-body">
                <strong>Daily state as of {daily_view.index[-1].date()}:</strong> {daily_state} / {fmt(daily_score, '', 1)}/100
                <br>
                <strong>Weekly anchor:</strong> {weekly_state} from {pd.Timestamp(latest.get("weekly_date")).date()}
                <br>
                This layer does not replace weekly D3 probability. It checks whether daily price, volatility, breadth, credit, rates, and dollar pressure confirm the weekly risk state.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(daily_confirmation_table(daily_eval, weekly_state, daily_state), use_container_width=True, hide_index=True)


def render_probability_panel(features: pd.DataFrame, evals: pd.DataFrame, grid: pd.DataFrame, bins: pd.DataFrame, as_of_date: dt.date) -> None:
    latest = features.iloc[-1]
    state = str(latest.get("composite_risk_state", "Unavailable"))
    score = latest.get("composite_risk_score", float("nan"))
    probability_table = composite_probability_table(features, as_of_date, state)
    best = grid.iloc[0] if not grid.empty else pd.Series(dtype=object)
    severity_color = status_color(state)
    current_drivers = relevant_indicators(latest)

    st.subheader("Mapped Crash / Drop Probability")
    st.markdown(
        f"""
        <div class="lq-panel" style="border-color:{severity_color}66; padding:14px 16px; background:{severity_color}12; margin-bottom:12px;">
            <div style="font-size:0.78rem; color:#cbd5e1; font-weight:700;">COMPOSITE D1 / D2 / D3 RISK STATE AS OF {features.index[-1].date()}</div>
            <div style="font-size:1.45rem; color:{severity_color}; font-weight:900;">{state} / {fmt(score, '', 1)}/100</div>
            <div style="color:#dbeafe; margin-top:4px;">
                Probability is mapped from the walk-forward weekly backtest. The indicator is learned from Dashboard 3 liquidity,
                Dashboard 1 macro/trend context, and all Dashboard 2 warning indicators instead of being manually added together.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="lq-status-ledger">
            <div class="lq-status-ledger-title">STATUS LEDGER</div>
            <div class="lq-status-ledger-body">
                <span class="lq-pill" style="color:#22c55e;border-color:#22c55e66;background:#22c55e14;">Normal 0-29</span>
                <span class="lq-pill" style="color:#f59e0b;border-color:#f59e0b66;background:#f59e0b14;">Watch 30-54</span>
                <span class="lq-pill" style="color:#ff5c57;border-color:#ff5c5766;background:#ff5c5714;">Warning 55-74</span>
                <span class="lq-pill" style="color:#ff5c57;border-color:#ff5c5766;background:#ff5c5722;">Stress 75-100</span>
                <br>
                <strong>Current read:</strong> {status_band_text(state)}
                <br>
                <strong>Relevant indicators this week:</strong> {current_drivers}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Mapped horizons: 4W, 8W, 13W, and 26W forward maximum drawdown windows. "
        "The probability table uses only completed historical windows available by the selected as-of date."
    )
    st.dataframe(probability_table, use_container_width=True, hide_index=True)

    st.caption(
        f"Study sample visible through selected as-of: {features.index.min().date()} to {features.index.max().date()}, "
        f"{len(features):,} weekly observations. Probabilities use full forward windows only."
    )

    d1_cards = [
        metric_card_html("Dashboard 1 Score", fmt(latest["d1_market_regime_score"], "", 0), d1_metric_color("market_score", latest["d1_market_regime_score"])),
        metric_card_html("Bear 12M", fmt(latest["d1_bear_prob12"], "%", 1), d1_metric_color("bear_prob", latest["d1_bear_prob12"])),
        metric_card_html("Bull Prob", fmt(latest["d1_bull_prob"], "%", 1), d1_metric_color("bull_prob", latest["d1_bull_prob"])),
        metric_card_html("200MA State", str(latest["d1_200ma_state"]), d1_metric_color("ma_state", latest["d1_200ma_state"])),
        metric_card_html("RWRA Bearish Count", fmt(latest["d1_rwra_bearish_count"], "", 0), d1_metric_color("rwra_count", latest["d1_rwra_bearish_count"])),
    ]
    st.markdown("<div class='lq-mini-grid'>" + "".join(d1_cards) + "</div>", unsafe_allow_html=True)

    d2_cards = [
        metric_card_html("Dashboard 2 Score", fmt(latest["d2_score"], "", 0), d2_metric_color("score", latest["d2_score"])),
        metric_card_html("Dashboard 2 Level", str(latest["d2_level"]), d2_metric_color("level", latest["d2_level"])),
        metric_card_html("D2 Warnings", fmt(latest["d2_warning_count"], "", 0), d2_metric_color("warnings", latest["d2_warning_count"])),
        metric_card_html("D2 Watches", fmt(latest["d2_watch_count"], "", 0), d2_metric_color("watches", latest["d2_watch_count"])),
        metric_card_html("Credit Stress", bool_label(latest["credit_stress"]), d2_metric_color("credit", latest["credit_stress"])),
    ]
    st.markdown("<div class='lq-mini-grid'>" + "".join(d2_cards) + "</div>", unsafe_allow_html=True)

    st.markdown("**Dashboard 2 Weekly Indicator Map**")
    d2_status = pd.DataFrame(
        [
            ("S&P 500 overextension", latest["d2_sp500_overextension_status"]),
            ("Increasing downward momentum", latest["d2_downward_momentum_status"]),
            ("Top range formation / breakdown", latest["d2_range_breakdown_status"]),
            ("Technical indicators deteriorating", latest["d2_technical_deterioration_status"]),
            ("Market breadth worsening", latest["d2_breadth_worsening_status"]),
            ("VIX > 20 / VIX spike", latest["d2_vix_spike_status"]),
            ("Breakout win rate / breakdown rate", latest["d2_breakout_failure_status"]),
            ("Theme stocks momentum weakening", latest["d2_theme_momentum_status"]),
        ],
        columns=["Indicator", "Current Weekly Status"],
    )
    d2_status["Relevant This Week"] = d2_status["Current Weekly Status"].map(lambda value: "Yes" if value in {"Watch", "Warning"} else "No")
    st.dataframe(d2_status.style.map(lambda value: f"color:{status_color(value)}; font-weight:900;" if value in {"Normal", "Watch", "Warning"} else "", subset=["Current Weekly Status"]), use_container_width=True, hide_index=True)

    st.markdown("**How to Read the Current Indicator**")
    if state == "Normal":
        st.success(
            "Normal means the current D1/D2/D3 mix historically behaved close to base-rate drawdown risk."
        )
    elif state == "Watch":
        st.warning(
            "Watch means the current D1/D2/D3 mix historically raised correction odds, but it is not a standalone crash call. "
            "Look for escalation to Warning/Stress plus weakening price trend before treating it as a serious market-drop signal."
        )
    else:
        st.error(
            "Warning/Stress means the current D1/D2/D3 mix historically had materially higher 10-20% drawdown odds."
        )

    if not best.empty:
        st.caption(
            "Best historical 20% drawdown grid in the study: "
            f"D1 >= {best['d1_min']:.0f}, D2 >= {best['d2_min']:.0f}, Liquidity >= {best['liquidity_min']:.0f}; "
            f"20% / 26W hit rate {best['drop20_26w'] * 100:.1f}% across {best['n']:.0f} signals."
        )


def make_composite_risk_chart(features: pd.DataFrame) -> go.Figure:
    plot = features.tail(520).copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    risk_colors = plot["composite_risk_state"].map(status_color).fillna("#94a3b8")
    fig.add_trace(
        go.Bar(
            x=plot.index,
            y=plot["composite_risk_score"],
            name="Composite Risk Bars",
            marker=dict(color=risk_colors, line=dict(width=0)),
            opacity=0.42,
            hovertemplate="%{x|%Y-%m-%d}<br>Composite risk: %{y:.1f}/100<br>State: %{customdata}<extra></extra>",
            customdata=plot["composite_risk_state"],
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=plot.index, y=plot["sp500"], name="S&P 500", line=dict(color="#22c55e", width=2.5)),
        secondary_y=False,
    )
    for state, label, color in [("Warning", "Warning", "#ff5c57"), ("Stress", "Stress", "#ff5c57")]:
        points = plot[plot["composite_risk_state"].eq(state)]
        if not points.empty:
            fig.add_trace(
                go.Scatter(
                    x=points.index,
                    y=points["sp500"],
                    name=label,
                    mode="markers",
                    marker=dict(size=8, color=color, symbol="diamond", line=dict(color="#f8fafc", width=0.6)),
                ),
                secondary_y=False,
            )
    for threshold, label, color in [(30, "Watch", "#f59e0b"), (55, "Warning", "#ff5c57"), (75, "Stress", "#ff5c57")]:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color=color,
            opacity=0.55,
            annotation_text=label,
            annotation_position="right",
            secondary_y=True,
        )
    fig.update_layout(
        title="Composite D1/D2/D3 Risk Bars vs S&P 500",
        height=460,
        hovermode="x unified",
        plot_bgcolor="#0b1423",
        paper_bgcolor="#0b1423",
        font=dict(color="#f8fafc"),
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(l=10, r=10, t=42, b=10),
        bargap=0,
    )
    fig.update_xaxes(gridcolor="rgba(71,85,105,0.35)", zeroline=False)
    fig.update_yaxes(title_text="S&P 500", secondary_y=False)
    fig.update_yaxes(title_text="Composite risk score", range=[0, 100], secondary_y=True, gridcolor="rgba(71,85,105,0.28)")
    return fig


def weekly_risk_probability_map(features: pd.DataFrame, as_of_date: dt.date) -> pd.DataFrame:
    rows = []
    for date, row in features.iterrows():
        state = str(row.get("composite_risk_state", "Unavailable"))
        p5_4w, _, n5_4w = pit_probability(features, date.date(), state, 5, 4)
        p10_8w, _, n10_8w = pit_probability(features, date.date(), state, 10, 8)
        p10_13w, _, n10_13w = pit_probability(features, date.date(), state, 10, 13)
        p20_26w, _, n20_26w = pit_probability(features, date.date(), state, 20, 26)
        rows.append(
            {
                "Date": date.date(),
                "S&P 500": row.get("sp500"),
                "Composite Risk": row.get("composite_risk_score"),
                "Risk State": state,
                "Relevant Indicator": relevant_indicators(row),
                "P >=5% Drop 4W": p5_4w,
                "P >=10% Drop 8W": p10_8w,
                "P >=10% Drop 13W": p10_13w,
                "P >=20% Drop 26W": p20_26w,
                "D1 Score": row.get("d1_market_regime_score"),
                "D2 Score": row.get("d2_score"),
                "Liquidity Score": row.get("liquidity_score"),
                "Calibration Samples": max(n5_4w, n10_8w, n10_13w, n20_26w),
            }
        )
    mapped = pd.DataFrame(rows)
    if mapped.empty:
        return mapped
    pct_cols = ["P >=5% Drop 4W", "P >=10% Drop 8W", "P >=10% Drop 13W", "P >=20% Drop 26W"]
    mapped[pct_cols] = mapped[pct_cols].applymap(lambda value: pct(value) if pd.notna(value) else "n/a")
    mapped["S&P 500"] = mapped["S&P 500"].map(lambda value: fmt(value, "", 1))
    mapped["Composite Risk"] = mapped["Composite Risk"].map(lambda value: fmt(value, "", 1))
    return mapped.iloc[::-1].reset_index(drop=True)


def style_risk_ledger(data: pd.DataFrame):
    def color_state(value: str) -> str:
        color = status_color(str(value))
        return f"color: {color}; font-weight: 900; background-color: {color}14;"

    return data.style.map(color_state, subset=["Risk State"])


def render_logic_tab() -> None:
    st.markdown(
        """
        ### Composite Indicator Logic

        The combined D1/D2/D3 indicator is not a simple addition of scores. It is a weekly walk-forward classifier trained to estimate whether the S&P 500 will suffer a **10% or larger drawdown within the next 13 weeks**.

        Each historical score is trained only on prior weekly observations, then scored for the next week. That keeps the dashboard from learning from the future. The model uses:

        - **Dashboard 3:** net liquidity, 13W/26W liquidity change, reserves/GDP, CP-SOFR, credit stress, VIX term structure, MOVE, dollar pressure.
        - **Dashboard 1:** market regime score, bear/bull probabilities, ETF rotation warning, 200-day trend state, RWRA-style bearish count.
        - **Dashboard 2:** all eight warning indicators, including overextension, downward momentum, range breakdown, technical deterioration, breadth worsening, VIX spike, breakout failure, and theme momentum.

        Color alignment follows the same risk language as D1/D2:

        - **Normal / green:** risk score below 30.
        - **Watch / amber:** 30 to 55.
        - **Warning / orange:** 55 to 75.
        - **Stress / red:** 75 or higher.

        GEX and Market Pulse are not included because they do not have complete weekly history in the local study. The existing Dashboard 1 ML meta signal is also excluded because the current implementation is trained on future labels, which would contaminate this backtest.

        ### Daily Confirmation Overlay

        The daily layer does **not** replace the weekly D3 probability model. Weekly D3 remains the main probability engine because liquidity data is partly weekly or lower frequency. The daily layer is a timing/confirmation check that uses daily market internals:

        - price trend versus 20/50/200-day moving averages
        - 5D/20D momentum and recent drawdown
        - VIX level, VIX spike, and VIX term inversion
        - HYG/IEF credit pressure
        - breadth deterioration across the equity universe
        - MOVE and dollar pressure

        The interpretation is: **Weekly D3 says whether the environment is risky; daily confirmation says whether the market is starting to trade like that risk matters now.**
        """
    )


def render_backtest_tab(
    features: pd.DataFrame,
    evals: pd.DataFrame,
    grid: pd.DataFrame,
    bins: pd.DataFrame,
    as_of_date: dt.date,
    daily_view: pd.DataFrame | None = None,
    daily_eval: pd.DataFrame | None = None,
) -> None:
    st.markdown("### Backtest Result")
    st.write(
        "The strongest conclusion is that liquidity alone is noisy. Predictability improves when liquidity stress aligns with trend, breadth, volatility, credit, and Dashboard 2 deterioration."
    )
    st.markdown("**Weekly Composite Risk and PIT Probability Map**")
    st.caption(
        "Each row uses the composite risk state from that week and maps it to probabilities using only earlier completed forward windows. "
        f"The table is filtered through the selected as-of date: {as_of_date}."
    )
    weekly_ledger = weekly_risk_probability_map(features, as_of_date)
    st.dataframe(weekly_ledger, use_container_width=True, hide_index=True)
    if daily_view is not None and daily_eval is not None and not daily_view.empty:
        st.markdown("**Daily Confirmation Ledger**")
        daily_cols = [
            "sp500",
            "weekly_date",
            "composite_risk_state",
            "composite_risk_score",
            "daily_confirmation_state",
            "daily_confirmation_score",
            "vix",
            "ret20",
            "pct_above_50dma",
            "hyg_ief_ret20",
            "daily_drop_10_13w",
            "daily_drop_20_26w",
        ]
        st.dataframe(daily_view[daily_cols].tail(120).iloc[::-1], use_container_width=True)
        st.markdown("**Daily Confirmation Conditional Evaluation**")
        st.dataframe(daily_eval, use_container_width=True, hide_index=True)
    st.markdown("**Full-sample composite risk bins, reference only**")
    st.caption("This export-level table uses the completed historical sample and is not the selected-date PIT calibration.")
    st.dataframe(bins, use_container_width=True, hide_index=True)
    st.markdown("**Legacy tier diagnostics, kept for comparison**")
    st.dataframe(evals, use_container_width=True, hide_index=True)
    st.markdown("**Best threshold combinations from the grid search**")
    st.dataframe(grid.head(25), use_container_width=True, hide_index=True)


def render_header(summary: dict[str, object]) -> None:
    color = status_color(str(summary["status"]))
    st.markdown(
        f"""
        <div class="lq-panel" style="padding:20px 22px;">
            <div style="display:flex; justify-content:space-between; gap:18px; flex-wrap:wrap; align-items:flex-start;">
                <div>
                    <div style="font-size:0.78rem; color:#93c5fd; font-weight:900; letter-spacing:0.12em;">INDEPENDENT DASHBOARD 3</div>
                    <div style="font-size:2.35rem; line-height:1.05; font-weight:900; color:#f8fafc; letter-spacing:-0.03em;">Liquidity Dashboard</div>
                    <div style="color:#d1d5db; margin-top:10px; max-width:920px; line-height:1.55;">
                        Liquidity means the tradable dollar flow left after Treasury cash and reverse repo parking absorb reserves:
                        <strong>Fed Total Assets - TGA - ON RRP</strong>, with all source series normalized to the same units.
                        This dashboard compares that flow with stocks, gold,
                        bank-reserve scarcity, and unsecured funding stress.
                    </div>
                </div>
                <div style="min-width:170px; text-align:center; border:1px solid {color}66; border-radius:10px; padding:14px 16px; background:{color}14;">
                    <div style="font-size:0.75rem; color:#cbd5e1; font-weight:700;">CURRENT STATE</div>
                    <div style="font-size:1.45rem; color:{color}; font-weight:900;">{summary["status"]}</div>
                    <div style="font-size:0.85rem; color:#e2e8f0;">Risk score {summary["risk_score"]}/100</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_overlay_chart(data: pd.DataFrame) -> go.Figure:
    if "Liquidity_State" not in data.columns or "Turning_Point_Start" not in data.columns:
        data = add_historical_signals(data, thresholds)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    state_style = {
        "Watch": ("rgba(245, 158, 11, 0.16)", "Watch"),
        "Stress": ("rgba(239, 68, 68, 0.18)", "Stress"),
    }
    for state, (color, label) in state_style.items():
        mask = data["Liquidity_State"].eq(state)
        starts = data.index[mask & ~mask.shift(fill_value=False)]
        ends = data.index[mask & ~mask.shift(-1, fill_value=False)]
        for i, (start, end) in enumerate(zip(starts, ends)):
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=color,
                line_width=0,
                layer="below",
                annotation_text=label if i == 0 else None,
                annotation_position="top left",
            )

    fig.add_trace(
        go.Scatter(x=data.index, y=data["Net_Liquidity_T"], name="Net Liquidity ($T)", line=dict(color="#38bdf8", width=3)),
        secondary_y=False,
    )
    fig.add_trace(go.Scatter(x=data.index, y=data["SP_500"], name="S&P 500", line=dict(color="#22c55e", width=2)), secondary_y=True)
    fig.add_trace(go.Scatter(x=data.index, y=data["Gold"], name="Gold Futures", line=dict(color="#f59e0b", width=2)), secondary_y=True)
    turn_points = data[data["Turning_Point_Start"]].copy()
    if not turn_points.empty:
        fig.add_trace(
            go.Scatter(
                x=turn_points.index,
                y=turn_points["SP_500"],
                name="Possible Stock Turning Point",
                mode="markers",
                marker=dict(symbol="diamond", size=10, color="#ef4444", line=dict(color="#fecaca", width=1)),
                customdata=turn_points[["Risk_Score", "Liquidity_26W_Change_T", "SP_500_26W_Return"]],
                hovertemplate=(
                    "Possible turning point<br>"
                    "%{x|%Y-%m-%d}<br>"
                    "S&P 500: %{y:,.0f}<br>"
                    "Risk score: %{customdata[0]:.0f}<br>"
                    "26W liquidity change: %{customdata[1]:+.2f}T<br>"
                    "26W S&P return: %{customdata[2]:+.1f}%<extra></extra>"
                ),
            ),
            secondary_y=True,
        )
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=42, b=10),
        title="Net Liquidity vs S&P 500 and Gold with Watch, Stress, and Turning-Point Signals",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        template="plotly_dark",
    )
    fig.update_yaxes(title_text="Net Liquidity, USD trillions", secondary_y=False)
    fig.update_yaxes(title_text="Asset price", secondary_y=True)
    return fig


def make_indexed_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Liquidity_Indexed"], name="Net Liquidity", line=dict(color="#38bdf8", width=3)))
    fig.add_trace(go.Scatter(x=data.index, y=data["SP_500_Indexed"], name="S&P 500", line=dict(color="#22c55e", width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data["Gold_Indexed"], name="Gold", line=dict(color="#f59e0b", width=2)))
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=42, b=10), title="Indexed Relationship, First Point = 100", hovermode="x unified", template="plotly_dark")
    fig.update_yaxes(title_text="Indexed value")
    return fig


def make_tripwire_chart(data: pd.DataFrame, thresholds: LiquidityThresholds) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=data.index, y=data["Reserve_to_GDP"], name="Reserves / GDP", line=dict(color="#a78bfa", width=2)), row=1, col=1)
    fig.add_hline(y=thresholds.reserve_watch_pct, line_dash="dash", line_color="#f59e0b", row=1, col=1)
    fig.add_hline(y=thresholds.reserve_stress_pct, line_dash="dash", line_color="#ef4444", row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["CP_SOFR_Spread_bps"], name="CP - SOFR", line=dict(color="#fb7185", width=2)), row=2, col=1)
    fig.add_hline(y=thresholds.cp_sofr_watch_bps, line_dash="dash", line_color="#f59e0b", row=2, col=1)
    fig.add_hline(y=thresholds.cp_sofr_stress_bps, line_dash="dash", line_color="#ef4444", row=2, col=1)
    fig.update_layout(height=500, margin=dict(l=10, r=10, t=42, b=10), title="Tripwires: Scarce Reserves and Funding Stress", hovermode="x unified", template="plotly_dark")
    fig.update_yaxes(title_text="Reserve / GDP %", row=1, col=1)
    fig.update_yaxes(title_text="Spread, bps", row=2, col=1)
    return fig


def make_component_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["WALCL"] / 1_000_000, name="Fed Assets", line=dict(color="#38bdf8")))
    fig.add_trace(go.Scatter(x=data.index, y=data["WTREGEN"] / 1_000_000, name="TGA", line=dict(color="#f97316")))
    fig.add_trace(go.Scatter(x=data.index, y=data["RRPONTSYD"] / 1000, name="ON RRP", line=dict(color="#eab308")))
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=42, b=10), title="Net Liquidity Components", hovermode="x unified", template="plotly_dark")
    fig.update_yaxes(title_text="USD trillions")
    return fig


with st.sidebar:
    st.header("Liquidity Settings")
    years = st.slider("History window", min_value=3, max_value=20, value=10, step=1)
    end_date = st.date_input("As-of date", value=dt.date.today(), max_value=dt.date.today())
    reserve_watch = st.number_input("Reserve/GDP watch %", value=12.0, min_value=0.0, max_value=30.0, step=0.5)
    reserve_stress = st.number_input("Reserve/GDP stress %", value=10.0, min_value=0.0, max_value=30.0, step=0.5)
    cp_watch = st.number_input("CP-SOFR watch bps", value=25.0, min_value=0.0, max_value=300.0, step=5.0)
    cp_stress = st.number_input("CP-SOFR stress bps", value=50.0, min_value=0.0, max_value=300.0, step=5.0)

thresholds = LiquidityThresholds(reserve_watch_pct=reserve_watch, reserve_stress_pct=reserve_stress, cp_sofr_watch_bps=cp_watch, cp_sofr_stress_bps=cp_stress)

try:
    with st.spinner("Fetching FRED and market data..."):
        dashboard, summary = load_liquidity_data(
            years,
            end_date,
            reserve_watch,
            reserve_stress,
            cp_watch,
            cp_stress,
            cache_version=2,
        )
except Exception as exc:
    st.error(f"Could not load liquidity dashboard data: {exc}")
    st.stop()

if dashboard.empty:
    st.error("No liquidity data was returned for the selected window.")
    st.stop()

render_header(summary)
st.caption(f"Data resolved through {summary['as_of']}. FRED values are point-in-time observations resampled to Thursdays; market data is weekly close.")

metric_cols = st.columns(6)
metric_cols[0].metric("Net Liquidity", f"${summary['net_liquidity_t']:,.2f}T", f"{summary['liquidity_13w_change_t']:+.2f}T / 13W")
metric_cols[1].metric("Reserve / GDP", fmt(summary["reserve_to_gdp"], "%", 2))
metric_cols[2].metric("CP - SOFR", fmt(summary["cp_sofr_spread_bps"], " bps", 1))
metric_cols[3].metric("S&P 500 26W", fmt(summary["sp_500_26w_return"], "%", 1))
metric_cols[4].metric("Gold 26W", fmt(summary["gold_26w_return"], "%", 1))
metric_cols[5].metric("LQ/SPX Corr", fmt(summary["sp_corr_26w"], "", 2))

for message in summary["messages"]:
    if summary["status"] == "Stress":
        st.error(message)
    elif summary["status"] == "Watch":
        st.warning(message)
    else:
        st.success(message)

study_features, study_evals, study_grid, study_bins, daily_confirmation, daily_confirmation_eval = load_crash_study(cache_version=3)
if (
    study_features is None
    or study_evals is None
    or study_grid is None
    or study_bins is None
    or daily_confirmation is None
    or daily_confirmation_eval is None
):
    st.warning(
        "Crash/drop probability study exports are missing. Run `python backend/analyze_crash_predictors.py` "
        "to map Dashboard 1 and Dashboard 2 indicators into Dashboard 3."
    )
else:
    study_view = study_as_of(study_features, end_date)
    daily_view = daily_as_of(daily_confirmation, end_date)
    if study_view.empty:
        st.warning("No composite risk study rows are available on or before the selected as-of date.")
        st.stop()

    composite_tab, logic_tab, backtest_tab = st.tabs(["Composite Risk Indicator", "Logic", "Backtest Result"])
    with composite_tab:
        render_probability_panel(study_view, study_evals, study_grid, study_bins, end_date)
        render_daily_confirmation_panel(daily_view, daily_confirmation_eval)
        st.plotly_chart(make_composite_risk_chart(study_view), use_container_width=True)
    with logic_tab:
        render_logic_tab()
    with backtest_tab:
        render_backtest_tab(study_view, study_evals, study_grid, study_bins, end_date, daily_view, daily_confirmation_eval)

st.markdown(
    """
    **Liquidity Chart Signals:** orange shaded zones = **Watch**, red shaded zones = **Stress**,
    red diamonds = **Possible Turning Point for Stock Market** where stocks rose while liquidity rolled over.
    """
)
st.plotly_chart(make_overlay_chart(dashboard), use_container_width=True)

left, right = st.columns(2)
with left:
    st.plotly_chart(make_indexed_chart(dashboard), use_container_width=True)
with right:
    st.plotly_chart(make_component_chart(dashboard), use_container_width=True)

st.plotly_chart(make_tripwire_chart(dashboard, thresholds), use_container_width=True)

st.subheader("Relationship Readout")
relationship = pd.DataFrame(
    [
        {"Measure": "26W Liquidity vs S&P correlation", "Latest": fmt(summary["sp_corr_26w"], "", 2), "Meaning": "Positive means stocks and liquidity are moving together; negative means stocks are fighting the liquidity impulse."},
        {"Measure": "26W Liquidity vs Gold correlation", "Latest": fmt(summary["gold_corr_26w"], "", 2), "Meaning": "Gold can rise with liquidity expansion or with stress; this helps separate monetary tailwind from safe-haven behavior."},
        {"Measure": "Divergence flag", "Latest": "Active" if summary["divergence"] else "Inactive", "Meaning": "Active means the S&P 500 rose over 26 weeks while net liquidity contracted, a momentum-over-plumbing warning."},
        {"Measure": "Reserve scarcity", "Latest": fmt(summary["reserve_to_gdp"], "%", 2), "Meaning": "Below the watch/stress thresholds, bank balance-sheet liquidity may become less abundant."},
        {"Measure": "Unsecured funding stress", "Latest": fmt(summary["cp_sofr_spread_bps"], " bps", 1), "Meaning": "A rising CP-SOFR spread implies corporations are paying up for unsecured short-term funding."},
    ]
)
st.dataframe(relationship, use_container_width=True, hide_index=True)

with st.expander("Latest weekly observations"):
    display_cols = [
        "Net_Liquidity_T",
        "WALCL",
        "WTREGEN",
        "RRPONTSYD",
        "Reserve_to_GDP",
        "CP_SOFR_Spread_bps",
        "SP_500",
        "Gold",
        "Liquidity_13W_Change_T",
        "SP_500_26W_Return",
        "Gold_26W_Return",
    ]
    st.dataframe(dashboard[display_cols].tail(20), use_container_width=True)

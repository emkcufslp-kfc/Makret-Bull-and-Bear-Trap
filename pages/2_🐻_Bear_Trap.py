import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Bear Trap Indicator", page_icon="🐻", layout="wide")

from utils.data_engine import get_clean_master
from utils.bear_trap_engine import compute_score, load_calibration, calibrated_probability

DEFAULT_ELEVATED, DEFAULT_WARNING = 20, 40  # used only if the calibration cache hasn't been built yet

@st.cache_data(ttl=3600)
def load_bear_data():
    # Use the centralized Incremental Data Engine
    return get_clean_master()

# --- Sidebar & Master Controls ---
with st.sidebar:
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    render_master_controls()
    render_ecosystem_sidebar()

def dashboard():
    from utils.ui_utils import resolve_master_date_slice

    analysis_date = st.session_state['master_date']

    with st.spinner("Loading market data..."):
        data = load_bear_data()
        if data.empty:
            st.error("⚠️ Failed to fetch market data from Yahoo Finance. Please check your internet connection or try again later.")
            return

        d, actual_date = resolve_master_date_slice(data, analysis_date)
        if d.empty or actual_date is None:
            st.error("No market data is available on or before the selected master date.")
            return

    # Final safety check before indexing
    if d.empty or len(d) < 200:
        st.error(f"Insufficient or no data for analysis. Need 200+ days, but found {len(d)}.")
        return

    latest = d.iloc[-1]

    st.markdown(
        f"<p style='color: #8892a4;'>Master date: <b>{analysis_date.strftime('%Y-%m-%d')}</b> | Resolved data date: <b>{actual_date.strftime('%Y-%m-%d')}</b></p>",
        unsafe_allow_html=True,
    )

    # --- Score Calculation: 5 real components only (Valuation/Positioning were
    # hardcoded stubs -- 0.65 and 0.50, never computed from data -- dropped
    # rather than faked; see utils/bear_trap_engine.py). Raw score is then
    # mapped through walk-forward isotonic calibrations, one per horizon
    # (data/bear_trap_calibration.json, rebuilt daily by daily_refresh.py
    # Step 8 / backend/calibrate_bear_trap.py). ---
    yield_curve = latest["^TNX"] - latest["^IRX"]
    irx_avg = d["^IRX"].rolling(120).mean().iloc[-1]
    hyg_ief = d["HYG"] / d["IEF"]
    hyg_ief_avg = hyg_ief.rolling(252).mean().iloc[-1]
    current_hyg_ief = hyg_ief.iloc[-1]
    spy_200ma = d["SPY"].rolling(200).mean().iloc[-1]

    scored = compute_score(
        yield_curve=yield_curve, irx=latest["^IRX"], irx_avg120=irx_avg,
        hyg_ief=current_hyg_ief, hyg_ief_avg252=hyg_ief_avg,
        spy=latest["SPY"], spy_200ma=spy_200ma, vix=latest["^VIX"],
    )
    raw_score = scored["raw_score"]

    calibration = load_calibration()
    prob_3m = calibrated_probability(raw_score, calibration, "3m")
    prob_6m = calibrated_probability(raw_score, calibration, "6m")
    prob_12m = calibrated_probability(raw_score, calibration, "12m")
    thresholds = (calibration or {}).get("thresholds", {})
    elevated_th = thresholds.get("elevated", DEFAULT_ELEVATED)
    warning_th = thresholds.get("warning", DEFAULT_WARNING)
    ceiling = (calibration or {}).get("horizons", {}).get("3m", {}).get("max_observed_calibrated_probability")

    # Risk level classification -- data-derived thresholds, not assumed
    if prob_3m < elevated_th:
        risk_level, risk_color = "LOW RISK", "#2ecc71"
    elif prob_3m < warning_th:
        risk_level, risk_color = "EARLY WARNING", "#f1c40f"
    else:
        risk_level, risk_color = "HIGH RISK", "#e74c3c"

    # --- RENDER ---
    col1, col2 = st.columns(2)
    col1.metric("Raw Composite Score", f"{raw_score:.1f} / 100")
    col2.markdown(f"### Risk Level: <span style='color:{risk_color};'>{risk_level}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Probability cards -- now genuinely calibrated per horizon, not raw_score * an arbitrary multiplier
    c1, c2, c3 = st.columns(3)
    c1.metric("Bear Probability (3M, calibrated)", f"{prob_3m}%")
    c2.metric("Bear Probability (6M, calibrated)", f"{prob_6m}%")
    c3.metric("Bear Probability (12M, calibrated)", f"{prob_12m}%")
    st.caption(
        "Each is a separate walk-forward isotonic calibration mapping the raw score to the actual historical "
        "frequency of a ≥10% S&P 500 drawdown within that horizon -- not one score scaled by an arbitrary "
        "multiplier (the previous 0.60 / 0.85 / 1.0 factors had no historical basis)."
    )

    # Indicator breakdown
    st.subheader("Indicator Breakdown")
    rows = {r["component"]: r for r in scored["detail"]}
    indicators = pd.DataFrame([
        {"Category": f"Macro ({rows['macro']['weight_pct']}%)", "Indicator": "Yield Curve (10Y-3M)", "Value": f"{yield_curve:.2f}%", "Score": f"{rows['macro']['normalized_value']:.2f}"},
        {"Category": f"Liquidity ({rows['liquidity']['weight_pct']}%)", "Indicator": "Fed Funds Proxy (^IRX)", "Value": f"{latest['^IRX']:.2f}%", "Score": f"{rows['liquidity']['normalized_value']:.2f}"},
        {"Category": f"Credit ({rows['credit']['weight_pct']}%)", "Indicator": "Credit Stress (HYG/IEF)", "Value": f"{current_hyg_ief:.3f}", "Score": f"{rows['credit']['normalized_value']:.2f}"},
        {"Category": f"Market ({rows['breadth']['weight_pct']}%)", "Indicator": "Breadth (SPY vs 200MA)", "Value": f"{((latest['SPY']/spy_200ma)-1)*100:+.1f}%", "Score": f"{rows['breadth']['normalized_value']:.2f}"},
        {"Category": f"Volatility ({rows['vix']['weight_pct']}%)", "Indicator": "VIX Regime", "Value": f"{latest['^VIX']:.1f}", "Score": f"{rows['vix']['normalized_value']:.2f}"},
    ]).set_index("Category")
    st.dataframe(indicators, width="stretch")
    st.caption(
        "The original rubric also carried a 'Valuation' (Buffett Proxy) and 'Positioning' (Sentiment Proxy) "
        "leg worth 10% combined -- both were hardcoded constants (0.65, 0.50) that never varied with real "
        "data, always displayed as 'Elevated'/'Neutral'. They've been removed rather than faked; the "
        "remaining 5 weights are renormalized to sum to 100%."
    )

    # Gauge
    st.subheader("Bear Trap Composite Gauge (calibrated, 3-month)")
    gauge_max = max(50, int(ceiling) + 20 if ceiling else 50)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_3m,
        number={'suffix': "%", 'valueformat': '.1f'},
        title={'text': "P(≥10% SPX drawdown within 3mo)"},
        gauge={
            'axis': {'range': [0, gauge_max], 'tickformat': '.0f', 'ticksuffix': '%'},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, elevated_th], 'color': "#2ecc71"},
                {'range': [elevated_th, warning_th], 'color': "#f1c40f"},
                {'range': [warning_th, gauge_max], 'color': "#e74c3c"},
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': prob_3m}
        }
    ))
    st.plotly_chart(fig, width="stretch")

    if calibration:
        wf = calibration["horizons"]["3m"]["walkforward_diagnostics"]
        beats_base_rate = wf["brier_walkforward_calibrated"] < wf["brier_always_base_rate"]
        st.caption(
            f"Walk-forward calibrated on {calibration.get('history_start')} → {calibration.get('history_end')} "
            f"({calibration.get('horizons', {}).get('3m', {}).get('n_observations')} trading days). Historically "
            f"the 3-month reading has never exceeded {ceiling:.1f}% -- an honest empirical ceiling, not a display cap."
        )
        if not beats_base_rate:
            st.warning(
                f"⚠️ Out-of-sample honesty check: even after walk-forward calibration, this composite's Brier "
                f"score ({wf['brier_walkforward_calibrated']}) is **worse** than just always guessing the "
                f"historical base rate ({wf['brier_always_base_rate']}). Treat the number above as a bounded, "
                "historically-grounded description of current conditions, not a proven forecast -- this "
                "composite has not demonstrated real predictive skill on a true out-of-sample basis."
            )
    else:
        st.warning(
            "No calibration cache found yet (data/bear_trap_calibration.json) — showing a naive placeholder "
            "until daily_refresh.py Step 8 runs. Run `python backend/calibrate_bear_trap.py` to generate it."
        )

    # --- Historical mapping: same three views as pages/1_Market_Regime.py ---
    st.divider()
    st.subheader("📜 Historical Bear-Trap-Score Map")
    st.caption(
        "This is exactly how the 3-month gauge above is derived. Each row buckets the raw composite score to "
        "the nearest 5 points. **Actual Historical Hit Rate** is the real, unsmoothed frequency: of all days "
        "that scored in this bucket, what % were followed by a ≥10% S&P 500 drawdown within 3 months. "
        "**Calibrated Probability** is that same data run through isotonic regression to stay monotonic while "
        "removing sampling noise."
    )
    if calibration and calibration.get("history_table"):
        hist_df = pd.DataFrame(calibration["history_table"]).rename(columns={
            "raw_score_bucket": "Raw Score (bucketed to nearest 5)",
            "n_days": "Historical Days in this Bucket",
            "actual_hit_rate_pct": "Actual Historical Hit Rate (%)",
            "calibrated_probability_pct": "Calibrated Probability (%, 3M)",
        })
        today_bucket = round(raw_score / 5) * 5

        def _highlight_today(row):
            style = "background-color: #1e3a5f; font-weight: bold;" if row["Raw Score (bucketed to nearest 5)"] == today_bucket else ""
            return [style] * len(row)

        st.dataframe(
            hist_df.style.apply(_highlight_today, axis=1).format({
                "Actual Historical Hit Rate (%)": "{:.1f}",
                "Calibrated Probability (%, 3M)": "{:.1f}",
            }),
            width="stretch", hide_index=True,
        )
        st.caption(f"Highlighted row = today's bucket (raw score {raw_score:.1f} → bucket {today_bucket:.0f}).")
    else:
        st.info("Historical map not available yet — run `python backend/calibrate_bear_trap.py`.")

    st.divider()
    st.subheader("🎯 \"If the score is above X...\" — direct historical odds")
    st.caption(
        "If the score stays at or above a given level, how often has a ≥10% S&P 500 drawdown actually "
        "followed within 3 / 6 / 12 months? Plain historical conditional frequency over the backtest window "
        "-- not walk-forward, not smoothed -- a description of the past, not a live forecast."
    )
    if calibration and calibration.get("threshold_horizon_table"):
        th_df = pd.DataFrame(calibration["threshold_horizon_table"])
        pivot = th_df.pivot(index="score_threshold", columns="horizon_label", values="probability_pct")
        pivot = pivot[["3 Months", "6 Months", "12 Months"]]
        pivot.index.name = "If score ≥"
        pivot = pivot.reset_index()

        active_threshold = max((t for t in pivot["If score ≥"] if t <= raw_score), default=pivot["If score ≥"].min())

        def _highlight_active_row(row):
            style = "background-color: #1e3a5f; font-weight: bold;" if row["If score ≥"] == active_threshold else ""
            return [style] * len(row)

        st.dataframe(
            pivot.style.apply(_highlight_active_row, axis=1).format(
                {c: "{:.1f}%" for c in pivot.columns if c != "If score ≥"}
            ),
            width="stretch", hide_index=True,
        )
        st.caption(f"Highlighted row = the band today's score ({raw_score:.1f}) currently falls into.")
    else:
        st.info("Table not available yet — run `python backend/calibrate_bear_trap.py`.")

    st.divider()
    st.subheader("📅 Every real historical date the score reached Warning level")
    if calibration and calibration.get("historical_episodes"):
        episodes = calibration["historical_episodes"]
        ep_df = pd.DataFrame(episodes).rename(columns={
            "start_date": "Start Date", "end_date": "End Date", "trading_days": "Trading Days",
            "peak_raw_score": "Peak Raw Score", "fwd_3m_drawdown_from_start_pct": "Actual 3-Month Forward Move (%)",
            "crash_followed": "Crash Followed (≥10% in 3mo)",
        })
        n_hits = sum(e["crash_followed"] for e in episodes)
        st.caption(
            f"Every contiguous run of days where the raw score reached {calibration.get('warning_raw_threshold')}+ "
            f"(this calibration's Warning-tier raw score) since {calibration.get('history_start')} — "
            f"{n_hits} of {len(episodes)} such episodes ({n_hits/len(episodes)*100:.0f}%) were actually followed "
            f"by a ≥10% drawdown within 3 months of the date the episode started. The rest were false alarms."
        )
        st.dataframe(
            ep_df.style.format({"Actual 3-Month Forward Move (%)": "{:.1f}"}).map(
                lambda v: "color: #ef4444; font-weight: bold;" if v is True else ("color: #94a3b8;" if v is False else ""),
                subset=["Crash Followed (≥10% in 3mo)"],
            ),
            width="stretch", hide_index=True,
        )
    else:
        st.info("Episode log not available yet — run `python backend/calibrate_bear_trap.py`.")

dashboard()

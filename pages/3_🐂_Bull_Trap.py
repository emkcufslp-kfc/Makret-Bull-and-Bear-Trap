import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Bull Trap Indicator", page_icon="🐂", layout="wide")

from utils.data_engine import get_clean_master
from utils.bull_trap_engine import compute_score, load_calibration, calibrated_probability

DEFAULT_CONFIRMED, DEFAULT_STRONG = 60, 75  # used only if the calibration cache hasn't been built yet

@st.cache_data(ttl=300)
def load_bull_data():
    # Use the centralized Incremental Data Engine
    return get_clean_master()

# --- Sidebar & Master Controls ---
with st.sidebar:
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    render_master_controls()
    render_ecosystem_sidebar()

def dashboard():
    from utils.ui_utils import resolve_master_date_slice

    st.title("🐂 Bull Trap Indicator Dashboard")
    st.markdown("Structural transition detector distinguishing genuine bull markets from deceptive bear rallies (bull traps), using a 6-component composite calibrated against 20 years of real market history.")

    analysis_date = st.session_state['master_date']

    with st.spinner("Loading market data..."):
        data = load_bull_data()
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
    prev_loc = max(0, len(d) - 23)
    prev_mo = d.iloc[prev_loc]

    st.markdown(
        f"<p style='color: #8892a4;'>Master date: <b>{analysis_date.strftime('%Y-%m-%d')}</b> | Resolved data date: <b>{actual_date.strftime('%Y-%m-%d')}</b></p>",
        unsafe_allow_html=True,
    )

    # --- Score calculation: 6 real components only (Valuation/Insider Buying were
    # hardcoded stubs -- 0.5 each, never computed from data -- plus a +1.0 baseline
    # that let the old score read as high as 8/10 with every real input at zero.
    # Dropped rather than faked; see utils/bull_trap_engine.py. Raw score is then
    # mapped through walk-forward isotonic calibrations, one per horizon
    # (data/bull_trap_calibration.json, rebuilt daily by daily_refresh.py Step 9 /
    # backend/calibrate_bull_trap.py). ---
    curve = latest["^TNX"] - latest["^IRX"]
    prev_curve = prev_mo["^TNX"] - prev_mo["^IRX"]
    vix_mavg = d["^VIX"].rolling(22).mean().iloc[-1]
    hyg_ief_series = d["HYG"] / d["IEF"]
    hyg_ief = hyg_ief_series.iloc[-1]
    hyg_ief_mavg = hyg_ief_series.rolling(22).mean().iloc[-1]
    spy_200ma = d["SPY"].rolling(200).mean().iloc[-1]
    spy_mom = (latest["SPY"] / prev_mo["SPY"]) - 1
    tip_now = d["TIP"].iloc[-1] if "TIP" in d.columns else np.nan
    tip_start = d["TIP"].iloc[-22] if "TIP" in d.columns and len(d) >= 22 else np.nan

    scored = compute_score(
        curve=curve, prev_curve=prev_curve, vix=latest["^VIX"], vix_mavg=vix_mavg,
        hyg_ief=hyg_ief, hyg_ief_mavg=hyg_ief_mavg, spy=latest["SPY"], spy_200ma=spy_200ma,
        spy_mom=spy_mom, tip_now=tip_now, tip_start=tip_start,
    )
    raw_score = scored["raw_score"]

    calibration = load_calibration()
    prob_3m = calibrated_probability(raw_score, calibration, "3m")
    prob_6m = calibrated_probability(raw_score, calibration, "6m")
    prob_12m = calibrated_probability(raw_score, calibration, "12m")
    thresholds = (calibration or {}).get("thresholds", {})
    confirmed_th = thresholds.get("confirmed", DEFAULT_CONFIRMED)
    strong_th = thresholds.get("strong", DEFAULT_STRONG)
    floor = (calibration or {}).get("horizons", {}).get("6m", {}).get("min_observed_calibrated_probability")
    ceiling = (calibration or {}).get("horizons", {}).get("6m", {}).get("max_observed_calibrated_probability")

    # Regime classification -- against the calibrated 6-month probability, using
    # data-derived thresholds (not the raw score directly, and not assumed cutoffs)
    if prob_6m < confirmed_th:
        regime, status_color = "Bull Trap Risk / Low Confidence", "#e74c3c"
    elif prob_6m < strong_th:
        regime, status_color = "Early Bull Market", "#f1c40f"
    else:
        regime, status_color = "Confirmed Bull Market", "#2ecc71"

    # --- RENDER ---
    col1, col2 = st.columns(2)
    col1.metric("Raw Composite Score", f"{raw_score:.1f} / 100")
    col2.markdown(f"### Regime: <span style='color:{status_color};'>{regime}</span>", unsafe_allow_html=True)

    st.markdown("---")

    # Probability cards -- genuinely calibrated per horizon against real forward
    # S&P 500 returns, not five hand-picked anchors keyed to a score band
    c1, c2, c3 = st.columns(3)
    c1.metric("Bull Continuation Probability (3M, calibrated)", f"{prob_3m}%")
    c2.metric("Bull Continuation Probability (6M, calibrated)", f"{prob_6m}%")
    c3.metric("Bull Continuation Probability (12M, calibrated)", f"{prob_12m}%")
    st.caption(
        "Each is a separate walk-forward isotonic calibration mapping the raw score to the actual historical "
        "frequency that the S&P 500's total return over that horizon was positive -- not one of five hand-picked "
        "anchors (the previous 20/40/65/85/95% bands had no historical basis, and the score could never "
        "mathematically exceed 8.0/10 under the old formula)."
    )

    # Indicator breakdown
    st.subheader("Indicator Breakdown")
    rows = {r["component"]: r for r in scored["detail"]}
    indicators = pd.DataFrame([
        {"Category": f"Rates ({rows['yield_curve']['weight_pct']}%)", "Indicator": "Yield Curve Re-Steepening (10Y-3M)", "Value": f"{curve:.2f}%", "Score": f"{rows['yield_curve']['tier_value']:.2f}"},
        {"Category": f"Volatility ({rows['vix']['weight_pct']}%)", "Indicator": "VIX Regime", "Value": f"{latest['^VIX']:.1f}", "Score": f"{rows['vix']['tier_value']:.2f}"},
        {"Category": f"Credit ({rows['credit']['weight_pct']}%)", "Indicator": "Credit Recovery (HYG/IEF)", "Value": f"{hyg_ief:.3f}", "Score": f"{rows['credit']['tier_value']:.2f}"},
        {"Category": f"Market ({rows['breadth']['weight_pct']}%)", "Indicator": "Market Breadth (SPY vs 200MA)", "Value": f"{((latest['SPY']/spy_200ma)-1)*100:+.1f}%", "Score": f"{rows['breadth']['tier_value']:.2f}"},
        {"Category": f"Momentum ({rows['momentum']['weight_pct']}%)", "Indicator": "Accumulation (1M momentum)", "Value": f"{spy_mom*100:+.1f}%", "Score": f"{rows['momentum']['tier_value']:.2f}"},
        {"Category": f"Liquidity ({rows['liquidity']['weight_pct']}%)", "Indicator": "Liquidity (TIP trend)", "Value": "Expanding" if rows['liquidity']['tier_value'] == 1 else "Tightening", "Score": f"{rows['liquidity']['tier_value']:.2f}"},
    ]).set_index("Category")
    st.dataframe(indicators, width="stretch")
    st.caption(
        "The original rubric also carried a 'Valuation' and 'Insider Buying' leg worth 1 point combined, plus a "
        "flat +1.0 baseline added to every reading -- all three were hardcoded constants that never varied with "
        "real data, and together they let the score reach 8.0/10 even when every real indicator below was at "
        "zero. They've been removed rather than faked; the remaining 6 weights are renormalized to sum to 100%."
    )

    # Gauge -- color bands tied to the same calibrated thresholds driving the
    # regime label above, not fixed cosmetic cutoffs. True range is 0-100%
    # (an achievable ceiling, unlike the old score's mathematical cap at 80%).
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_6m,
        number={'suffix': "%", 'valueformat': '.1f'},
        title={'text': "P(S&P 500 total return positive within 6mo)"},
        gauge={
            'axis': {'range': [0, 100], 'tickformat': '.0f', 'ticksuffix': '%'},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, confirmed_th], 'color': "#e74c3c"},
                {'range': [confirmed_th, strong_th], 'color': "#f1c40f"},
                {'range': [strong_th, 100], 'color': "#2ecc71"},
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': prob_6m}
        }
    ))
    st.plotly_chart(fig, width="stretch")

    if calibration:
        wf = calibration["horizons"]["6m"]["walkforward_diagnostics"]
        beats_base_rate = wf["brier_walkforward_calibrated"] < wf["brier_always_base_rate"]
        st.caption(
            f"Walk-forward calibrated on {calibration.get('history_start')} → {calibration.get('history_end')} "
            f"({calibration.get('horizons', {}).get('6m', {}).get('n_observations')} trading days, limited by HYG's "
            f"2007 inception). Historically the 6-month reading has ranged {floor:.1f}%–{ceiling:.1f}% -- an "
            "honest empirical range, not a display cap."
        )
        if not beats_base_rate:
            st.warning(
                f"⚠️ Out-of-sample honesty check: even after walk-forward calibration, this composite's Brier "
                f"score ({wf['brier_walkforward_calibrated']}) is **worse** than just always guessing the "
                f"historical base rate ({wf['brier_always_base_rate']}) that the S&P 500 was up over 6 months "
                f"({wf['base_rate_pct']}% of the time, unconditionally). Treat the numbers above as a bounded, "
                "historically-grounded description of current trend conditions, not a proven forecast -- this "
                "composite has not demonstrated real predictive skill on a true out-of-sample basis. Its six "
                "inputs are all momentum/coincident signals (they confirm a trend already underway), which is "
                "exactly why they tend to read most bullish right before real bull-trap peaks and most bearish "
                "right at real capitulation bottoms -- see the episode log below."
            )
    else:
        st.warning(
            "No calibration cache found yet (data/bull_trap_calibration.json) — showing a naive placeholder "
            "until daily_refresh.py Step 9 runs. Run `python backend/calibrate_bull_trap.py` to generate it."
        )

    # --- Historical mapping ---
    st.divider()
    st.subheader("📜 Historical Bull-Market-Score Map")
    st.caption(
        "This is exactly how the 6-month gauge above is derived. Each row buckets the raw composite score to "
        "the nearest 5 points. **Actual Historical Hit Rate** is the real, unsmoothed frequency: of all days "
        "that scored in this bucket, what % were followed by a positive S&P 500 total return within 6 months. "
        "**Calibrated Probability** is that same data run through isotonic regression to stay monotonic while "
        "removing sampling noise."
    )
    if calibration and calibration.get("history_table"):
        hist_df = pd.DataFrame(calibration["history_table"]).rename(columns={
            "raw_score_bucket": "Raw Score (bucketed to nearest 5)",
            "n_days": "Historical Days in this Bucket",
            "actual_hit_rate_pct": "Actual Historical Hit Rate (%)",
            "calibrated_probability_pct": "Calibrated Probability (%, 6M)",
        })
        today_bucket = round(raw_score / 5) * 5

        def _highlight_today(row):
            style = "background-color: #1e3a5f; font-weight: bold;" if row["Raw Score (bucketed to nearest 5)"] == today_bucket else ""
            return [style] * len(row)

        st.dataframe(
            hist_df.style.apply(_highlight_today, axis=1).format({
                "Actual Historical Hit Rate (%)": "{:.1f}",
                "Calibrated Probability (%, 6M)": "{:.1f}",
            }),
            width="stretch", hide_index=True,
        )
        st.caption(f"Highlighted row = today's bucket (raw score {raw_score:.1f} → bucket {today_bucket:.0f}).")
    else:
        st.info("Historical map not available yet — run `python backend/calibrate_bull_trap.py`.")

    st.divider()
    st.subheader("🎯 \"If the score is above X...\" — direct historical odds")
    st.caption(
        "If the score stays at or above a given level, how often was the S&P 500's total return actually "
        "positive within 3 / 6 / 12 months? Plain historical conditional frequency over the backtest window "
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
        st.info("Table not available yet — run `python backend/calibrate_bull_trap.py`.")

    st.divider()
    st.subheader("📅 Bull Trap Episode Log — every high-conviction bullish call, checked against reality")
    if calibration and calibration.get("bull_trap_episodes"):
        episodes = calibration["bull_trap_episodes"]
        summary = calibration.get("bull_trap_episode_summary", {})
        ep_df = pd.DataFrame(episodes).rename(columns={
            "start_date": "Start Date", "end_date": "End Date", "trading_days": "Trading Days",
            "peak_raw_score": "Peak Raw Score", "fwd_3m_drawdown_from_start_pct": "Actual 3-Month Forward Move (%)",
            "trap_followed": "Turned Out to Be a Trap (≥10% drop in 3mo)",
        })
        st.caption(
            f"Every contiguous run of days where the raw score reached {calibration.get('strong_raw_threshold')}+ "
            f"(this calibration's 'strong' bullish conviction level) since {calibration.get('history_start')} — "
            f"{summary.get('n_traps')} of {summary.get('n_episodes')} such episodes "
            f"({summary.get('trap_rate_pct')}%) were actually followed by a ≥10% S&P 500 drawdown within 3 "
            f"months of the episode starting. For comparison, an *unconditional* random day over the same "
            f"period saw a ≥10% drawdown within 3 months {summary.get('unconditional_base_trap_rate_pct')}% of "
            "the time — so a high-conviction reading has historically been somewhat *less* trap-prone than an "
            "average day, not more, even though (per the honesty check above) it isn't a reliable forecast either."
        )
        st.dataframe(
            ep_df.style.format({"Actual 3-Month Forward Move (%)": "{:.1f}"}).map(
                lambda v: "color: #ef4444; font-weight: bold;" if v is True else ("color: #94a3b8;" if v is False else ""),
                subset=["Turned Out to Be a Trap (≥10% drop in 3mo)"],
            ),
            width="stretch", hide_index=True,
        )
    else:
        st.info("Episode log not available yet — run `python backend/calibrate_bull_trap.py`.")

dashboard()

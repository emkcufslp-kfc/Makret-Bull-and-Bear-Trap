import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="Market Regime & Crash Probability", page_icon="🔴", layout="wide")

from utils.data_engine import (
    get_clean_master, get_hy_spread, get_move, get_t2108, get_sp500_drawdown,
    get_data_freshness, get_richmond_fed_sos, get_polymarket_prob,
)
from utils.ui_utils import get_latest_master_data_date
from utils.market_regime_engine import compute_score, load_calibration, calibrated_probability

DEFAULT_ELEVATED, DEFAULT_WARNING = 20, 40  # used only if the calibration cache hasn't been built yet

# ----------------------------
# DASHBOARD
# ----------------------------
def dashboard():
    st.title("🔴 Market Regime & Crash Probability Dashboard")

    # Display Data Freshness Badge
    freshness = get_data_freshness()
    if freshness:
        master_update = next((f['Last Update'] for f in freshness if "Master" in f['Source']), "Unknown")
        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 5px 15px; border-radius: 20px; border: 1px solid #3b82f6; display: inline-block; margin-bottom: 20px;">
            <span style="color: #60a5fa; font-size: 0.8rem; font-weight: bold;">📅 數據最後更新 (Latest Sync): {master_update}</span>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar and Global Controls
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    with st.sidebar:
        render_master_controls()
        render_ecosystem_sidebar()

    # Priority 1: Master Date from Session State
    if 'master_date' not in st.session_state:
        st.session_state['master_date'] = get_latest_master_data_date()

    analysis_date = st.session_state['master_date']

    with st.spinner("Analyzing Market Conditions..."):
        data = get_clean_master()
        if data.empty:
            st.error("⚠️ Failed to fetch market data.")
            return

        data = data.ffill().dropna(how='all')

    # Find nearest valid trading day
    analysis_ts = pd.Timestamp(analysis_date)
    valid_dates = data.index[data.index <= analysis_ts]

    if len(valid_dates) > 0:
        actual_date = valid_dates[-1]
        d = data.loc[:actual_date]
    else:
        st.warning(f"No data available for {analysis_date}. Showing latest.")
        d = data
        actual_date = data.index[-1]

    st.caption(
        f"Master date: {analysis_date.strftime('%Y-%m-%d')} | Resolved data date: {actual_date.strftime('%Y-%m-%d')}"
    )

    latest = d.iloc[-1]

    # --- Data Extraction ---
    sp_price = float(latest.get("^GSPC", 0))
    dma200 = float(d["^GSPC"].rolling(200).mean().iloc[-1]) if "^GSPC" in d.columns else 0
    vix = float(latest.get("^VIX", 20.0))
    vix3m = float(latest.get("^VIX3M", 20.0)) if "^VIX3M" in latest else 21.0
    # Reformatting to bps: Value of 3.27% becomes 327.0 bps
    hy_spread_pct = get_hy_spread(actual_date.date())
    hy_spread_bps = hy_spread_pct * 100
    move = get_move(actual_date.date())
    t2108 = get_t2108(actual_date.date())
    sp_drawdown = get_sp500_drawdown(actual_date.date())
    dxy = float(latest.get("DX-Y.NYB", 100.0))
    richmond_sos = get_richmond_fed_sos(actual_date.date())
    polymarket_prob = get_polymarket_prob(actual_date.date())

    # --- Score Calculation: raw rule-based score, then mapped through a
    # walk-forward isotonic calibration (data/market_regime_calibration.json,
    # rebuilt daily by daily_refresh.py Step 7 / backend/calibrate_market_regime.py)
    # so the displayed number is an empirically honest probability rather
    # than a raw point count. See utils/market_regime_engine.py. ---
    scored = compute_score(sp_price, dma200, hy_spread_pct, move, vix, vix3m, dxy, t2108)
    raw_score = scored["raw_score"]
    calibration = load_calibration()
    prob = calibrated_probability(raw_score, calibration)
    thresholds = (calibration or {}).get("thresholds", {})
    elevated_th = thresholds.get("elevated", DEFAULT_ELEVATED)
    warning_th = thresholds.get("warning", DEFAULT_WARNING)
    ceiling = (calibration or {}).get("max_observed_calibrated_probability")

    # --- UI Layout: Scorecard Details ---
    st.markdown(f"### 🗃️ 有利門檻金融指標狀態表 (Scorecard Details)")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**A) 基本面：結構性熊市警報**")

        # Fundamental Data
        f_data = [
            {"指標": "信用利差", "有利門檻": "< 400 bps", "目前": f"{hy_spread_bps:,.1f} bps", "狀態": "✅ 安全" if hy_spread_bps < 400 else "❌ 警戒"},
            {"指標": "Richmond Fed SOS", "有利門檻": "< 0.2", "目前": f"{richmond_sos:.3f}", "狀態": "✅ 安全" if richmond_sos < 0.2 else "❌ 警戒"},
            {"指標": "Polymarket 衰退率", "有利門檻": "< 50%", "目前": f"{polymarket_prob:.0f}%", "狀態": "✅ 安全" if polymarket_prob < 50 else "🟡 警告"},
            {"指標": "USD 強勢指數", "有利門檻": "< 105.0", "目前": f"{dxy:.1f}", "狀態": "✅ 安全" if dxy < 105 else "❌ 警戒"},
        ]
        st.table(pd.DataFrame(f_data))

    with colB:
        st.markdown("**B) 技術面：超賣進場訊號**")

        # Technical Data
        t_data = [
            {"指標": "VIX 恐慌指數", "有利門檻": "30", "目前": f"{vix:.2f}", "狀態": "✅ 安全" if vix < 30 else "❌ 警戒"},
            {"指標": "T2108 (40MA強勢股)", "有利門檻": "< 10%", "目前": f"{t2108:,.1f}%", "狀態": "✅ 觸發" if t2108 < 10 else "❌ 未觸發"},
            {"指標": "S&P 500 回撤", "有利門檻": "10%", "目前": f"{abs(sp_drawdown):,.1f}%", "狀態": "🟡 觀察" if abs(sp_drawdown) > 6 else "✅ 安全"},
            {"指標": "S&P 500 200DMA", "有利門檻": "> SMA200", "目前": f"${sp_price:,.0f}", "狀態": "✅ 在週線上" if sp_price > dma200 else "❌ 失守"},
            {"指標": "Bond Volatility", "有利門檻": "< 100", "目前": f"{move:.1f}", "狀態": "✅ 安全" if move < 100 else "❌ 警戒"}
        ]
        st.table(pd.DataFrame(t_data))

    with st.expander("Scorecard rule breakdown (which of the 7 rules fired today)"):
        rule_df = pd.DataFrame(scored["detail"])
        rule_df["fired"] = rule_df["fired"].map({True: "🔴 fired", False: "— clear"})
        st.table(rule_df.rename(columns={"rule": "Rule", "weight": "Weight", "fired": "Status"}))
        st.caption(f"Raw score: {raw_score} / {scored['max_score']} → calibrated probability: {prob:.1f}%")

    # --- Gauge & Heatmap ---
    colC, colD = st.columns([1, 1])
    with colC:
        st.subheader("Crash Probability Gauge (calibrated)")
        gauge_max = max(100, int(ceiling) + 20 if ceiling else 100)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': "P(≥10% SPX drawdown within 3mo)"},
            gauge={
                'axis': {'range': [0, gauge_max], 'tickcolor': "white"},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, elevated_th], 'color': "green"},
                    {'range': [elevated_th, warning_th], 'color': "yellow"},
                    {'range': [warning_th, gauge_max], 'color': "red"},
                ]
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        st.plotly_chart(fig, use_container_width=True)
        if calibration:
            st.caption(
                f"Walk-forward calibrated on {calibration.get('history_start')} → {calibration.get('history_end')} "
                f"({calibration.get('n_observations')} trading days). Historically this reading has never exceeded "
                f"{ceiling:.0f}% — that's an honest empirical ceiling, not a display cap. "
                f"HY spread source: {calibration.get('hy_spread_source', 'n/a')}."
            )
        else:
            st.warning(
                "No calibration cache found yet (data/market_regime_calibration.json) — showing "
                "raw_score/max_score as a rough placeholder until daily_refresh.py Step 7 runs. "
                "Run `python backend/calibrate_market_regime.py` to generate it."
            )

    with colD:
        st.subheader("Tactical Portfolio Guidance")
        if prob < elevated_th:
            st.success(f"🟢 LOW RISK REGIME: Condition Normal (< {elevated_th}%). Maintain strategic exposure. Opportunistically buy relative strength.")
        elif prob < warning_th:
            st.warning(f"🟡 ELEVATED ({elevated_th}–{warning_th}%): Systemic stress rising. Selective profit taking. Reduce high-beta concentration.")
        else:
            st.error(f"🔴 WARNING (≥ {warning_th}%): Elevated drawdown risk historically. Prioritize cash and tail hedges. Defensive positioning recommended.")
        st.caption(
            "Thresholds are data-derived, not assumed: 'Elevated' is where the alarm's hit-rate on 20 years of "
            "history starts meaningfully beating the base rate; 'Warning' is where it becomes high-conviction "
            "(fewer, more reliable flags). Even at Warning, most of this scorecard's inputs are coincident/lagging "
            "(price already below its 200DMA, VIX already elevated) — it has historically missed sudden, "
            "news-driven shocks (e.g. Feb 2020, early 2025) until the decline was already underway."
        )

    st.divider()
    st.subheader("📜 Historical Crash-Probability Map")
    st.caption(
        "This is exactly how the gauge above is derived — no hidden model. Every row is a raw score this "
        "scorecard actually produced somewhere in the last 20 years. **Actual Historical Hit Rate** is the "
        "real, unsmoothed frequency: of all days that scored this level, what % were followed by a ≥10% "
        "S&P 500 drawdown within the next 3 months. **Calibrated Probability** is that same data run through "
        "isotonic regression to remove sampling noise while staying monotonic (never decreasing as the score "
        "rises) — that's why nearby scores sometimes share one calibrated value even though their raw hit "
        "rates bounce around (e.g. small buckets like score 45 can read *lower* than score 40 just from noise; "
        "the calibration pools them rather than taking that literally)."
    )
    if calibration and calibration.get("history_table"):
        hist_df = pd.DataFrame(calibration["history_table"]).rename(columns={
            "raw_score": "Raw Score",
            "n_days": "Historical Days at this Score",
            "actual_hit_rate_pct": "Actual Historical Hit Rate (%)",
            "calibrated_probability_pct": "Calibrated Probability (%)",
        })

        def _highlight_today(row):
            is_today = row["Raw Score"] == raw_score
            style = "background-color: #1e3a5f; font-weight: bold;" if is_today else ""
            return [style] * len(row)

        st.dataframe(
            hist_df.style.apply(_highlight_today, axis=1).format({
                "Actual Historical Hit Rate (%)": "{:.1f}",
                "Calibrated Probability (%)": "{:.1f}",
            }),
            width="stretch", hide_index=True,
        )
        st.caption(f"Highlighted row = today's raw score ({raw_score} / {scored['max_score']}).")
    else:
        st.info("Historical map not available yet — run `python backend/calibrate_market_regime.py`.")

    st.divider()
    st.subheader("🎯 \"If the score is above X...\" — direct historical odds")
    st.caption(
        "Straight answer to the obvious question: if the score stays at or above a given level, how often has "
        "a ≥10% S&P 500 drawdown actually followed within 1 / 3 / 6 months? This is a plain historical "
        "conditional frequency over the last 20 years — not walk-forward, not smoothed — so treat it as a "
        "description of the past, not a live forecast."
    )
    if calibration and calibration.get("threshold_horizon_table"):
        th_df = pd.DataFrame(calibration["threshold_horizon_table"])
        pivot = th_df.pivot(index="score_threshold", columns="horizon_label", values="probability_pct")
        pivot = pivot[[label for _, label in [(21, "1 Month"), (63, "3 Months"), (126, "6 Months")]]]
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
        st.caption(f"Highlighted row = the band today's score ({raw_score}) currently falls into.")
    else:
        st.info("Table not available yet — run `python backend/calibrate_market_regime.py`.")

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
            f"by a ≥10% drawdown within 3 months of the date the episode started. The rest were false alarms. "
            "Sort/search the table to find specific dates (e.g. 2007, 2022)."
        )
        st.dataframe(
            ep_df.style.format({"Actual 3-Month Forward Move (%)": "{:.1f}"}).map(
                lambda v: "color: #ef4444; font-weight: bold;" if v is True else ("color: #94a3b8;" if v is False else ""),
                subset=["Crash Followed (≥10% in 3mo)"],
            ),
            width="stretch", hide_index=True,
        )
    else:
        st.info("Episode log not available yet — run `python backend/calibrate_market_regime.py`.")

    st.divider()
    st.subheader("🧬 Historical Risk Clustering Heatmap (12M)")
    st.info("Heatmap tracks Trend, Credit, Bond Vol, Equity Vol, Term Structure, and USD strength across a rolling window.")

dashboard()

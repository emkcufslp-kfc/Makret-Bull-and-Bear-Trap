import pandas as pd
import streamlit as st

from utils.model_change_monitor import ALL_MODELS, build_warning_mode_events
from utils.ui_utils import get_latest_master_data_date, render_ecosystem_sidebar, render_master_controls


st.set_page_config(page_title="Model Change Monitor Worksheet", page_icon="MC", layout="wide")


def badge(text, color):
    return f"<span class='mc-pill' style='background:{color}1a;color:{color};border-color:{color}55;'>{text}</span>"


def color_warning(value):
    return "#ef4444" if value == "Triggered" else "#22c55e"


def status_color_for_text(text):
    if not text:
        return "#94a3b8"

    exact_palette = {
        "HIGH RISK": "#ef4444",
        "EARLY WARNING": "#f59e0b",
        "LOW RISK": "#22c55e",
        "LOW RISK REGIME": "#22c55e",
        "BULLISH / LOW RISK": "#22c55e",
        "CAUTION / BULL TRAP RISK": "#f59e0b",
        "BEARISH / HIGH RISK": "#ef4444",
        "NORMAL": "#22c55e",
        "BULLISH (Wait for Exit)": "#22c55e",
        "CAUTION (Trend Testing)": "#f59e0b",
        "BEARISH (Keep Cash)": "#ef4444",
        "HIGH CONFIDENCE (BUY/HOLD)": "#22c55e",
        "NEUTRAL / CAUTION": "#f59e0b",
        "LOW CONFIDENCE (REDUCE)": "#ef4444",
        "STRONGLY BULLISH": "#22c55e",
        "CAUTION / MIXED": "#f59e0b",
        "DEFENSIVE": "#ef4444",
        "STRONG BULL": "#22c55e",
        "BULL": "#38bdf8",
        "NEUTRAL": "#fbbf24",
        "RISK OFF": "#f97316",
        "CRISIS": "#ef4444",
        "HOLD": "#22c55e",
        "REBALANCE": "#f97316",
    }
    if text in exact_palette:
        return exact_palette[text]

    if " | " in text:
        parts = [part.strip() for part in text.split("|") if part.strip()]
        # Match the dashboard card logic: emphasize execution status first, then guardrail.
        for part in reversed(parts):
            color = exact_palette.get(part)
            if color:
                return color

    if "BULL" in text and "TRAP" not in text:
        return "#22c55e" if "STRONG" in text else "#38bdf8"
    if "WARNING" in text or "CAUTION" in text:
        return "#f59e0b"
    if "RISK" in text or "CRISIS" in text or "REDUCE" in text or "BEARISH" in text:
        return "#ef4444"
    return "#94a3b8"


def render_status_cell(text):
    return badge(text, status_color_for_text(text))


def render_page():
    with st.sidebar:
        render_master_controls()
        render_ecosystem_sidebar()

    end_date = st.session_state.get("master_date", get_latest_master_data_date())
    events = build_warning_mode_events(end_date)

    st.markdown(
        """
        <style>
        .mc-title {font-size: 2.4rem; font-weight: 900; color:#f8fafc; margin-bottom: 0.15rem;}
        .mc-subtitle {font-size: 0.98rem; color:#94a3b8; margin-bottom: 1rem;}
        .mc-banner {
            background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(15,23,42,0.88));
            border: 1px solid rgba(56, 189, 248, 0.28);
            border-radius: 18px;
            padding: 18px;
            margin-bottom: 14px;
        }
        .mc-banner-grid {display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 12px;}
        .mc-banner-item {
            background: rgba(2, 6, 23, 0.28);
            border: 1px solid rgba(51,65,85,0.45);
            border-radius: 14px;
            padding: 12px;
        }
        .mc-banner-label {font-size: 0.8rem; color:#94a3b8; margin-bottom: 0.25rem;}
        .mc-banner-value {font-size: 1.35rem; font-weight: 800; color:#f8fafc;}
        .mc-table-shell {
            background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(15,23,42,0.88));
            border: 1px solid rgba(71,85,105,0.45);
            border-radius: 18px;
            padding: 18px;
        }
        .mc-table-title {font-size: 1.35rem; font-weight: 800; color:#f8fafc; margin-bottom: 0.35rem;}
        .mc-table-note {font-size: 0.88rem; color:#94a3b8; margin-bottom: 1rem;}
        .mc-pill {
            display:inline-flex; align-items:center; justify-content:center;
            padding: 0.25rem 0.6rem; border-radius:999px; border:1px solid;
            font-size:0.74rem; font-weight:700; letter-spacing:0.04em;
            white-space: nowrap;
        }
        div[data-testid="stDataFrame"] div[role="table"] {font-size: 0.84rem;}
        @media (max-width: 1200px) {.mc-banner-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="mc-title">Model Change Monitor Worksheet</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="mc-subtitle">Last 5 Years through {end_date.strftime("%Y-%m-%d")} | Warning Mode Threshold = 3 core-model shifts | Market Pulse shown for context but excluded from alert count.</div>',
        unsafe_allow_html=True,
    )

    event_count = len(events)
    trigger_count = int((events["Warning Mode Change"] == "Triggered").sum()) if not events.empty else 0
    clear_count = int((events["Warning Mode Change"] == "Cleared").sum()) if not events.empty else 0
    latest_event = events.iloc[-1]["Date"] if not events.empty else "No event"

    st.markdown(
        f"""
        <div class="mc-banner">
            <div><strong style="color:#f8fafc;">Worksheet Rule</strong> <span style="color:#94a3b8;">This event log displays all 8 model statuses, but only counts status shifts across the 7 core models. Rows appear only when warning mode changes from &lt;3 to &gt;=3 shifts or from &gt;=3 to &lt;3 shifts.</span></div>
            <div class="mc-banner-grid">
                <div class="mc-banner-item"><div class="mc-banner-label">8 Model Statuses Displayed</div><div class="mc-banner-value">8</div></div>
                <div class="mc-banner-item"><div class="mc-banner-label">7 Core Models Counted</div><div class="mc-banner-value">7</div></div>
                <div class="mc-banner-item"><div class="mc-banner-label">Threshold</div><div class="mc-banner-value">3 Shifts</div></div>
                <div class="mc-banner-item"><div class="mc-banner-label">Latest Event</div><div class="mc-banner-value">{latest_event}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if events.empty:
        st.info("No warning-mode transitions were detected in the last 5 years using the current master-date horizon.")
        return

    st.markdown(
        f"""
        <div class="mc-table-shell">
            <div class="mc-table-title">Warning Mode Event Log</div>
            <div class="mc-table-note">{event_count} event rows found | Triggered: {trigger_count} | Cleared: {clear_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    display = events.copy()
    display["Warning Mode Change"] = display["Warning Mode Change"].apply(lambda v: badge(v, color_warning(v)))
    for model in ALL_MODELS:
        display[model] = display[model].apply(render_status_cell)

    ordered_cols = [
        "Date",
        "SPY Price",
        "Prev Shift Count",
        "New Shift Count",
        "Warning Mode Change",
        "Changed Core Models",
        "Market Regime",
        "Bear Trap",
        "Bull Trap",
        "ETF Rotation",
        "200MA Strategy",
        "ML Meta-Indicator",
        "Combined Macro + RWRA",
        "Market Pulse",
    ]
    st.markdown('<div style="margin-top: -16px;"></div>', unsafe_allow_html=True)
    st.write(
        display[ordered_cols].to_html(index=False, escape=False),
        unsafe_allow_html=True,
    )
    st.caption("Event rows appear only when the 7-core-model shift count crosses the warning threshold of 3.")


render_page()

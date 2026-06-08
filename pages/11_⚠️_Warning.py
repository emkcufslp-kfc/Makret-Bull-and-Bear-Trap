from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from utils.ui_utils import get_latest_master_data_date, render_ecosystem_sidebar, render_master_controls
from utils.warning_dashboard import build_warning_dashboard


st.set_page_config(page_title="Warning", page_icon="⚠️", layout="wide")


STATUS_STYLE = {
    "Warning": ("HIGH", "#ff5c57"),
    "Watch": ("ELEVATED", "#f59e0b"),
    "Normal": ("NORMAL", "#22c55e"),
}

LEVEL_STYLE = {
    "Critical": ("HIGH", "#ff5c57", "Elevated Risk Conditions"),
    "Elevated": ("HIGH", "#ff5c57", "Elevated Risk Conditions"),
    "Guarded": ("ELEVATED", "#f59e0b", "Pressure Building"),
    "Stable": ("NORMAL", "#22c55e", "Contained Risk Conditions"),
}

ESCALATION_STYLE = {
    "Escalate": ("RISING", "#f59e0b", "Momentum Deteriorating"),
    "Tighten": ("RISING", "#f59e0b", "Momentum Deteriorating"),
    "Monitor": ("WATCH", "#fbbf24", "Monitor Closely"),
    "Normal": ("NORMAL", "#22c55e", "Stable Conditions"),
}

INDICATOR_ICONS = {
    "Liquidity Stress": "◔",
    "Breadth Failure": "◎",
    "Volatility Expansion": "≈",
    "Credit Risk Drift": "⬡",
    "Trend Fracture": "↘",
    "Leadership Breakdown": "♛",
    "Correlation Spike": "⇄",
    "Defensive Rotation": "⬢",
}


@st.cache_data(ttl=1800)
def load_warning_payload(target_date: date):
    return build_warning_dashboard(target_date)


def pill_html(status: str) -> str:
    label, color = STATUS_STYLE.get(status, (status.upper(), "#94a3b8"))
    return (
        "<span class='warn-pill' "
        f"style='color:{color}; border-color:{color}66; background:{color}14;'>"
        f"{label}</span>"
    )


def timeline_change_label(text: str) -> str:
    if "Normal -> Watch" in text:
        return "↑ Low → Elevated"
    if "Watch -> Warning" in text:
        return "↑ Elevated → High"
    if "Normal -> Warning" in text:
        return "↑ Low → High"
    if "Watch -> Normal" in text:
        return "↓ Elevated → Low"
    if "Warning -> Watch" in text:
        return "↓ High → Elevated"
    return text.replace("Normal", "Low").replace("Watch", "Elevated").replace("Warning", "High")


def days_ago_label(actual_date: date, value: str) -> str:
    try:
        diff = (actual_date - pd.Timestamp(value).date()).days
    except Exception:
        return ""
    if diff <= 0:
        return "Today"
    if diff == 1:
        return "1 Day Ago"
    return f"{diff} Days Ago"


def render_page():
    with st.sidebar:
        render_master_controls()
        render_ecosystem_sidebar()

    master_date = st.session_state.get("master_date", get_latest_master_data_date())
    payload = load_warning_payload(master_date)

    st.markdown(
        """
        <style>
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
        .warn-topbar {
            display:flex;
            justify-content:space-between;
            align-items:flex-start;
            gap:16px;
            margin-bottom: 0.85rem;
        }
        .warn-title {
            font-size: 3.1rem;
            font-weight: 900;
            color: #f8fafc;
            line-height: 1;
            letter-spacing: -0.04em;
            margin-bottom: 0.55rem;
        }
        .warn-subtitle {
            font-size: 0.98rem;
            color: #d1d5db;
            display:flex;
            flex-wrap:wrap;
            gap:10px;
            align-items:center;
        }
        .warn-subtitle strong {
            color: #f59e0b;
            font-weight: 800;
        }
        .warn-subtitle .muted {
            color: #cbd5e1;
        }
        .warn-export {
            display:inline-flex;
            align-items:center;
            gap:10px;
            border:1px solid rgba(148,163,184,0.35);
            border-radius:14px;
            padding:12px 16px;
            color:#f8fafc;
            background:rgba(15,23,42,0.45);
            font-size:1rem;
            font-weight:700;
        }
        .warn-cards {
            display:grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 14px;
            margin-bottom: 16px;
        }
        .warn-card {
            position:relative;
            min-height: 134px;
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(14,25,42,0.98), rgba(10,19,34,0.92));
            border:1px solid rgba(71,85,105,0.5);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 18px 36px rgba(2,6,23,0.18);
            padding: 18px 18px 16px 18px;
            overflow:hidden;
        }
        .warn-card-label {
            font-size: 0.86rem;
            color:#cbd5e1;
            margin-bottom: 0.8rem;
            font-weight:700;
        }
        .warn-card-value {
            font-size: 1.95rem;
            line-height:1;
            font-weight: 900;
            margin-bottom: 0.65rem;
        }
        .warn-card-note {
            font-size: 0.82rem;
            color:#d1d5db;
            line-height:1.35;
        }
        .warn-card-subnote {
            font-size: 0.8rem;
            color:#94a3b8;
            margin-top: 0.2rem;
        }
        .warn-card-art {
            position:absolute;
            right:18px;
            top:18px;
            font-size: 3rem;
            line-height:1;
            opacity: 0.95;
        }
        .warn-card-art.spark {
            top:auto;
            bottom:12px;
            font-size:2.3rem;
            letter-spacing:-0.08em;
        }
        .warn-main {
            display:grid;
            grid-template-columns: minmax(0, 1.9fr) minmax(360px, 1fr);
            gap: 16px;
            margin-bottom: 16px;
        }
        .warn-panel {
            border-radius: 16px;
            background: linear-gradient(180deg, rgba(14,25,42,0.98), rgba(10,19,34,0.92));
            border:1px solid rgba(71,85,105,0.5);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.03), 0 18px 36px rgba(2,6,23,0.18);
            padding: 14px 16px 16px 16px;
        }
        .warn-panel-header {
            display:flex;
            align-items:center;
            gap:10px;
            color:#f8fafc;
            font-size: 1.02rem;
            font-weight: 800;
            margin-bottom: 12px;
        }
        .warn-panel-icon {
            color:#cbd5e1;
            font-size: 1.15rem;
        }
        .warn-table {
            width:100%;
            border-collapse: separate;
            border-spacing: 0;
            overflow:hidden;
            border:1px solid rgba(51,65,85,0.6);
            border-radius: 10px;
        }
        .warn-table th {
            background: rgba(20,29,44,0.96);
            color: #d1d5db;
            font-size: 0.75rem;
            font-weight:700;
            text-align:left;
            padding: 10px 12px;
            border-right:1px solid rgba(51,65,85,0.55);
            border-bottom:1px solid rgba(51,65,85,0.6);
        }
        .warn-table td {
            color: #e5e7eb;
            font-size: 0.86rem;
            padding: 10px 12px;
            border-right:1px solid rgba(51,65,85,0.4);
            border-bottom:1px solid rgba(51,65,85,0.45);
            vertical-align:top;
        }
        .warn-table tr:last-child td { border-bottom:none; }
        .warn-table th:last-child, .warn-table td:last-child { border-right:none; }
        .warn-table tbody tr:nth-child(even) td { background: rgba(10,18,31,0.45); }
        .warn-pill {
            display:inline-flex;
            align-items:center;
            justify-content:center;
            min-width: 88px;
            padding: 0.26rem 0.7rem;
            border-radius: 7px;
            border:1px solid;
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .warn-indicator {
            display:flex;
            align-items:center;
            gap:10px;
            font-weight:700;
        }
        .warn-indicator-icon {
            color:#60a5fa;
            width:16px;
            display:inline-flex;
            justify-content:center;
            font-size: 1rem;
        }
        .warn-summary-group {
            padding: 12px 2px 12px 2px;
            border-top:1px solid rgba(51,65,85,0.5);
        }
        .warn-summary-group:first-of-type { border-top:none; padding-top: 2px; }
        .warn-summary-title {
            display:flex;
            align-items:center;
            gap:10px;
            color:#f8fafc;
            font-size: 0.98rem;
            font-weight:800;
            margin-bottom: 8px;
        }
        .warn-summary-title .icon {
            color:#f59e0b;
            font-size:1.05rem;
            width:20px;
            text-align:center;
        }
        .warn-summary-list {
            margin:0;
            padding-left: 22px;
            color:#d1d5db;
        }
        .warn-summary-list li {
            margin: 5px 0;
            line-height:1.45;
            font-size:0.88rem;
        }
        .warn-timeline-header {
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:10px;
            margin-bottom: 12px;
        }
        .warn-history-btn {
            display:inline-flex;
            align-items:center;
            gap:10px;
            border:1px solid rgba(148,163,184,0.35);
            border-radius:12px;
            padding:10px 14px;
            color:#e5e7eb;
            background:rgba(15,23,42,0.35);
            font-size:0.9rem;
            font-weight:700;
        }
        @media (max-width: 1300px) {
            .warn-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .warn-main { grid-template-columns: 1fr; }
        }
        @media (max-width: 800px) {
            .warn-topbar { flex-direction:column; }
            .warn-cards { grid-template-columns: 1fr; }
            .warn-title { font-size: 2.4rem; }
            .warn-subtitle { line-height:1.5; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if payload is None:
        st.error("Insufficient master data is available to build the Warning module for the selected master date.")
        return

    level_label, level_color, level_note = LEVEL_STYLE.get(payload["warning_level"], ("NORMAL", "#22c55e", "Contained Risk Conditions"))
    esc_label, esc_color, esc_note = ESCALATION_STYLE.get(payload["escalation_status"], ("NORMAL", "#22c55e", "Stable Conditions"))
    indicator_df = payload["indicator_matrix"].copy()
    active_df = indicator_df[indicator_df["Status"] != "Normal"].copy()
    actual_date = payload["actual_date"]
    last_change_days = days_ago_label(actual_date, payload["last_signal_change"])
    trigger_score_pct = min(100, max(0, payload["trigger_score"] * 12))

    top_flashing = active_df["Indicator"].head(3).tolist()
    more_flashing = active_df["Indicator"].iloc[3:].tolist()
    strongest = active_df.sort_values(["Score", "Indicator"], ascending=[False, True]).head(3)

    subtitle = (
        f"<div class='warn-subtitle'>"
        f"<span>Master Date: <strong>{payload['master_date']:%Y-%m-%d}</strong></span>"
        f"<span>|</span>"
        f"<span>Actual Data Date: <strong>{payload['actual_date']:%Y-%m-%d}</strong></span>"
        f"<span>|</span>"
        f"<span class='muted'>New Warning Indicators Dashboard</span>"
        f"</div>"
    )

    header_html = f"""
    <div class="warn-topbar">
        <div>
            <div class="warn-title">Warning</div>
            {subtitle}
        </div>
        <div class="warn-export">⇩ Export ▾</div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    cards_html = f"""
    <div class="warn-cards">
        <div class="warn-card">
            <div class="warn-card-label">Warning Level</div>
            <div class="warn-card-value" style="color:{level_color};">{level_label}</div>
            <div class="warn-card-note">{level_note}</div>
            <div class="warn-card-art" style="color:{level_color};">◔</div>
        </div>
        <div class="warn-card">
            <div class="warn-card-label">Trigger Score</div>
            <div class="warn-card-value" style="color:#ff6b5f;">{trigger_score_pct}<span style="font-size:0.56em;color:#cbd5e1;"> /100</span></div>
            <div class="warn-card-note">{level_label.title()} (≥ 65)</div>
            <div class="warn-card-art spark" style="color:#ff6b5f;">▁▃▂▄▃▅▃▅</div>
        </div>
        <div class="warn-card">
            <div class="warn-card-label">Active Warnings</div>
            <div class="warn-card-value" style="color:#ff6b5f;">{payload['active_warnings']}<span style="font-size:0.56em;color:#cbd5e1;"> / 8</span></div>
            <div class="warn-card-note">Indicators Flashing</div>
            <div class="warn-card-art" style="color:#ff6b5f;">⚠</div>
        </div>
        <div class="warn-card">
            <div class="warn-card-label">Escalation Status</div>
            <div class="warn-card-value" style="color:{esc_color};">{esc_label}</div>
            <div class="warn-card-note">{esc_note}</div>
            <div class="warn-card-art" style="color:{esc_color};">⇱</div>
        </div>
        <div class="warn-card">
            <div class="warn-card-label">Last Signal Change</div>
            <div class="warn-card-value" style="color:#ff6b5f;font-size:1.75rem;">{payload['last_signal_change']}</div>
            <div class="warn-card-note">{last_change_days}</div>
            <div class="warn-card-art" style="color:#ff6b5f;">⌲</div>
        </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

    matrix_rows = []
    for row in indicator_df.itertuples(index=False):
        matrix_rows.append(
            "<tr>"
            f"<td><div class='warn-indicator'><span class='warn-indicator-icon'>{INDICATOR_ICONS.get(row.Indicator, '•')}</span><span>{row.Indicator}</span></div></td>"
            f"<td>{pill_html(row.Status)}</td>"
            f"<td>{row.Value}</td>"
            f"<td>{row.Threshold}</td>"
            f"<td>{row.Note}</td>"
            "</tr>"
        )
    matrix_html = (
        "<table class='warn-table'><thead><tr>"
        "<th>Indicator</th><th>Status</th><th>Current Value</th><th>Threshold</th><th>Note</th>"
        "</tr></thead><tbody>"
        + "".join(matrix_rows)
        + "</tbody></table>"
    )

    flashing_items = []
    if top_flashing:
        flashing_items.append("<li>" + ", ".join(top_flashing) + "</li>")
    if more_flashing:
        flashing_items.append("<li>" + ", ".join(more_flashing) + "</li>")
    flashing_items.append(f"<li>{payload['active_warnings']} out of 8 indicators are at Elevated or High</li>")

    strongest_items = []
    if strongest.empty:
        strongest_items.append("<li>No major deterioration is active right now</li>")
    else:
        for row in strongest.itertuples(index=False):
            strongest_items.append(f"<li>{row.Indicator} ({row.Value})</li>")

    next_checkpoint_items = [
        f"<li>Monitor into {(pd.Timestamp(actual_date) + pd.Timedelta(days=5)).date()} (weekly checkpoint)</li>",
        "<li>Watch liquidity, breadth, and credit trend together</li>",
    ]

    summary_html = f"""
    <div class="warn-panel">
        <div class="warn-panel-header"><span class="warn-panel-icon">☷</span><span>Warning Summary</span></div>
        <div class="warn-summary-group">
            <div class="warn-summary-title"><span class="icon">⚠</span><span>What's Flashing Now</span></div>
            <ul class="warn-summary-list">{''.join(flashing_items)}</ul>
        </div>
        <div class="warn-summary-group">
            <div class="warn-summary-title"><span class="icon" style="color:#ff6b5f;">⌁</span><span>Strongest Deterioration</span></div>
            <ul class="warn-summary-list">{''.join(strongest_items)}</ul>
        </div>
        <div class="warn-summary-group">
            <div class="warn-summary-title"><span class="icon" style="color:#fbbf24;">◔</span><span>Next Checkpoint</span></div>
            <ul class="warn-summary-list">{''.join(next_checkpoint_items)}</ul>
        </div>
    </div>
    """

    st.markdown("<div class='warn-main'>", unsafe_allow_html=True)
    left_col, right_col = st.columns([1.9, 1.0], gap="large")
    with left_col:
        st.markdown(
            "<div class='warn-panel'><div class='warn-panel-header'><span class='warn-panel-icon'>⊞</span><span>Warning Indicator Matrix</span></div>"
            + matrix_html
            + "</div>",
            unsafe_allow_html=True,
        )
    with right_col:
        st.markdown(summary_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    timeline_df = payload["timeline"].copy()
    timeline_rows = []
    for row in timeline_df.head(8).itertuples(index=False):
        label, sev_color = STATUS_STYLE.get(row.Severity, (row.Severity.upper(), "#94a3b8"))
        timeline_rows.append(
            "<tr>"
            f"<td>{row.Date}</td>"
            f"<td>{row.Indicator}</td>"
            f"<td>{timeline_change_label(row._2 if hasattr(row, '_2') else row[2]) if False else timeline_change_label(row[2])}</td>"
            f"<td><span class='warn-pill' style='color:{sev_color}; border-color:{sev_color}66; background:{sev_color}14;'>{label}</span></td>"
            f"<td>{next((note for note in indicator_df.loc[indicator_df['Indicator'] == row.Indicator, 'Note'].tolist()), '')}</td>"
            "</tr>"
        )
    timeline_html = (
        "<table class='warn-table'><thead><tr>"
        "<th>Date</th><th>Indicator</th><th>Status Change</th><th>Severity</th><th>Notes</th>"
        "</tr></thead><tbody>"
        + "".join(timeline_rows)
        + "</tbody></table>"
    )

    bottom_html = (
        "<div class='warn-panel'>"
        "<div class='warn-timeline-header'>"
        "<div class='warn-panel-header' style='margin-bottom:0;'><span class='warn-panel-icon'>◷</span><span>Recent Warning Timeline</span></div>"
        "<div class='warn-history-btn'>View All History ›</div>"
        "</div>"
        + timeline_html
        + "</div>"
    )
    st.markdown(bottom_html, unsafe_allow_html=True)


render_page()

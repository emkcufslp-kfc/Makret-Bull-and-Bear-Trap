import datetime
import subprocess
import sys
from pathlib import Path

import streamlit as st

from utils.data_engine import get_data_freshness


def render_master_controls():
    """Centralized master date and synchronization controls."""
    if "master_date" not in st.session_state:
        st.session_state["master_date"] = datetime.date.today()

    st.sidebar.markdown(
        """
        <div style="background-color:#1e293b;padding:10px;border-radius:8px;border:1px solid #3b82f6;margin-bottom:20px;">
            <h4 style="color:#60a5fa;margin:0;font-size:0.9rem;text-align:center;">Master Control Center</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    new_date = st.sidebar.date_input(
        "Master Date",
        value=st.session_state["master_date"],
        max_value=datetime.date.today(),
        help="Synchronize every dashboard to the same analysis date.",
    )
    if new_date != st.session_state["master_date"]:
        st.session_state["master_date"] = new_date
        st.rerun()

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Reset", use_container_width=True):
        st.session_state["master_date"] = datetime.date.today()
        st.rerun()

    if col2.button("Refresh", use_container_width=True):
        try:
            with st.status("Running unified data sync...", expanded=True) as status:
                root_path = Path(__file__).parent.parent
                backend_sync = root_path / "backend" / "sync_engine.py"
                if not backend_sync.exists():
                    status.update(label="Sync engine not found.", state="error", expanded=True)
                    st.error("Missing backend/sync_engine.py")
                else:
                    st.write("Launching backend sync engine...")
                    subprocess.run([sys.executable, str(backend_sync)], check=True)
                    st.info("Local datasets refreshed. Push to GitHub if you want Streamlit Cloud to use the new files.")
                    status.update(label="Sync complete", state="complete", expanded=False)
                    st.toast("Data refreshed successfully.")
        except Exception as exc:
            st.error(f"Sync failed: {exc}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Data Freshness")
    freshness_data = get_data_freshness()

    sync_gap_detected = False
    for item in freshness_data:
        try:
            file_date = datetime.datetime.strptime(item["Last Update"], "%Y-%m-%d %H:%M").date()
            if st.session_state["master_date"] > file_date:
                sync_gap_detected = True
        except Exception:
            pass

        color = "#22c55e" if "OK" in item["Status"] else "#ef4444"
        st.sidebar.markdown(
            f"""
            <div style="font-size:0.8rem;margin-bottom:5px;">
                <span style="color:#94a3b8;">{item['Source']}:</span><br/>
                <span style="color:{color};font-weight:bold;">{item['Last Update']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if sync_gap_detected:
        st.sidebar.warning(
            f"Master date {st.session_state['master_date']} is ahead of one or more local datasets. Consider running Refresh."
        )

    st.sidebar.markdown("---")


def render_ecosystem_sidebar():
    """Unified sidebar navigation for the market dashboard ecosystem."""
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] { display: none !important; }
            section[data-testid="stSidebar"] ul { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background-color:#0f172a;padding:10px;border-radius:8px;margin-top:10px;text-align:center;border:1px solid #334155;">
            <h3 style="color:white;margin-top:0;font-size:1rem;">Market Dashboard Modules</h3>
        </div>
        <div style="margin-top:15px;"></div>
        """,
        unsafe_allow_html=True,
    )

    st.page_link("pages/0_📋_Market_Summary.py", label="Market Summary Dashboard", icon="📋")
    st.page_link("pages/1_🔴_Market_Regime.py", label="Market Regime", icon="🔴")
    st.page_link("pages/2_🐻_Bear_Trap.py", label="Bear Trap", icon="🐻")
    st.page_link("pages/3_🐂_Bull_Trap.py", label="Bull Trap", icon="🐂")
    st.page_link("pages/4_📊_ETF_Rotation.py", label="ETF Rotation", icon="📊")
    st.page_link("pages/5_📈_200MA_Strategy.py", label="200MA Strategy", icon="📈")
    st.page_link("pages/6_🎯_Meta_Indicator.py", label="Meta Indicator", icon="🎯")
    st.page_link("pages/10_📒_Model_Change_Worksheet.py", label="Model Change Worksheet", icon="📒")

    st.markdown(
        """
        <div style="background-color:#1e293b;padding:15px;border-radius:10px;margin-top:20px;text-align:center;border:1px solid #475569;">
            <p style="color:#94a3b8;font-size:0.85rem;margin-bottom:12px;font-weight:bold;">Strategy Suite</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.page_link("pages/7_🚀_FTD_Strategy.py", label="Market Pulse / FTD Strategy", icon="🚀")
    st.page_link("pages/8_📊_Strategies_Dashboard.py", label="Strategies Dashboard", icon="📊")

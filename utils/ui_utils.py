import os
import subprocess
import shutil
import sys
from pathlib import Path

from utils.data_engine import get_data_freshness

def render_master_controls():
    """Centralized master date and system synchronization controls."""
    # Initialization
    if 'master_date' not in st.session_state:
        st.session_state['master_date'] = datetime.date.today()
    
    st.sidebar.markdown("""
<div style="background-color: #1e293b; padding: 10px; border-radius: 8px; border: 1px solid #3b82f6; margin-bottom: 20px;">
    <h4 style="color: #60a5fa; margin: 0; font-size: 0.9rem; text-align: center;">📊 全域決策控制系統</h4>
</div>
""", unsafe_allow_html=True)

    # Master Date Picker
    new_date = st.sidebar.date_input(
        "📅 決策基準日期 (Master Date)", 
        value=st.session_state['master_date'],
        max_value=datetime.date.today(),
        help="此日期將同步影響所有策略面板與回測引擎"
    )
    
    # Update session state if changed manually
    if new_date != st.session_state['master_date']:
        st.session_state['master_date'] = new_date
        st.rerun()

    col1, col2 = st.sidebar.columns(2)
    
    # Reset Button
    if col1.button("🔄 重置今天", use_container_width=True):
        st.session_state['master_date'] = datetime.date.today()
        st.rerun()
        
    # Refresh Button (Unified Dashboard Sync)
    if col2.button("⚡ 數據刷新", use_container_width=True):
        try:
            with st.status("正在同步最新市場數據...", expanded=True) as status:
                # Unified Backend Path (Robust Pathing)
                root_path = Path(__file__).parent.parent
                backend_sync = root_path / "backend" / "sync_engine.py"

                if backend_sync.exists():
                    st.write("🛰️ **Master Sync 引擎運行中 (Automated Mode)**")
                    
                    # Run the engine
                    # This will update data/Multi_indicator/ and data/Platinum_Results/ directly
                    subprocess.run([sys.executable, str(backend_sync)], check=True)
                    
                    st.info("💡 **自動化通知**: 此系統預設每天在 GitHub 雲端自動同步。手動刷新僅用於本地測試或緊急更新。")
                    status.update(label="✅ 系統數據已更新！", state="complete", expanded=False)
                    st.toast("數據同步完成！")
                else:
                    status.update(label="❌ 找不到同步引擎", state="error", expanded=True)
                    st.error("請確認 backend/sync_engine.py 是否存在。")
        except Exception as e:
            st.error(f"同步失敗: {e}")

    # Data Freshness Indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🕒 數據新鮮度監控 (Data Freshness)")
    freshness_data = get_data_freshness()
    
    # Check if Master Date is ahead of factual data
    sync_gap_detected = False
    for item in freshness_data:
        try:
            # Parse the "Last Update" string (format: %Y-%m-%d %H:%M)
            file_date = datetime.datetime.strptime(item["Last Update"], "%Y-%m-%d %H:%M").date()
            if st.session_state['master_date'] > file_date:
                sync_gap_detected = True
        except: pass
            
        color = "#22c55e" if "OK" in item["Status"] else "#ef4444"
        st.sidebar.markdown(f"""
        <div style="font-size: 0.8rem; margin-bottom: 5px;">
            <span style="color: #94a3b8;">{item['Source']}:</span><br/>
            <span style="color: {color}; font-weight: bold;">{item['Last Update']}</span>
        </div>
        """, unsafe_allow_html=True)

    if sync_gap_detected:
        st.sidebar.warning(f"⚠️ **數據延誤**: 目前基準日 ({st.session_state['master_date']}) 領先於數據庫。請點擊上方刷新按鈕。")

    st.sidebar.markdown("---")

def render_ecosystem_sidebar():
    """Unified sidebar navigation for the Makret-Bull-and-Bear-Trap ecosystem."""
    # CSS to hide the redundant default navigation and style the custom sidebar
    st.markdown("""
<style>
    [data-testid="stSidebarNav"] { display: none !important; }
    section[data-testid="stSidebar"] ul { display: none !important; }
    div[style*="cursor: pointer"]:hover {
        filter: brightness(1.2);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="background-color: #0f172a; padding: 10px; border-radius: 8px; margin-top: 10px; text-align: center; border: 1px solid #334155;">
<h3 style="color: white; margin-top: 0; font-size: 1.0rem;">🧠 量化決策與風險監控系統</h3>
</div>
<div style="margin-top: 15px;"></div>
""", unsafe_allow_html=True)
    
    st.page_link("pages/1_🔴_Market_Regime.py", label="市場體系：Market Regime", icon="🚦")
    st.page_link("pages/2_🐻_Bear_Trap.py", label="熊市陷阱：Bear Trap", icon="🐻")
    st.page_link("pages/3_🐂_Bull_Trap.py", label="牛市陷阱：Bull Trap", icon="🐂")
    st.page_link("pages/4_📊_ETF_Rotation.py", label="輪動監控：ETF Rotation", icon="🚀")
    st.page_link("pages/5_📈_200MA_Strategy.py", label="趨勢防禦：200MA Strategy", icon="🛡️")
    st.page_link("pages/6_🎯_Meta_Indicator.py", label="關鍵指標：Meta Indicator", icon="💎")

    st.markdown("""
<div style="background-color: #1e293b; padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center; border: 1px solid #475569;">
<p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 12px; font-weight: bold;">專業量化策略面板</p>
</div>
""", unsafe_allow_html=True)

    st.page_link("pages/7_🚀_FTD_Strategy.py", label="底部確認 : FTD 追蹤", icon="🚀")
    st.page_link("pages/8_🛡️_NTSX_Strategy.py", label="資產配置 : NTSX 策略", icon="🛡️")
    st.page_link("pages/9_💎_Platinum_Strategy.py", label="核心增益 : Platinum 策略", icon="💎")

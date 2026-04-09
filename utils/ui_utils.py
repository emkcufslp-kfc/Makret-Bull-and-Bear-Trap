import streamlit as st
import datetime
import os
import subprocess
import shutil

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
        
    # Refresh Button (Local Data Update)
    if col2.button("⚡ 數據刷新", use_container_width=True):
        try:
            with st.status("正在同步最新市場數據...", expanded=True) as status:
                # Determine paths (Self-correction for Cloud vs local Windows)
                is_windows = os.name == 'nt'
                
                # Use environment variables if present, else fallback
                sync_script = os.environ.get("SYNC_SCRIPT_PATH", r"d:\Backtest\Multi indicator\sync_all.py")
                repo_data_path = os.path.join(os.getcwd(), "data")
                source_multi = os.environ.get("MULTI_INDICATOR_PATH", r"d:\Backtest\Multi indicator")
                source_plat = os.environ.get("PLATINUM_RESULTS_PATH", r"d:\Backtest\Platinum_Results")

                if os.path.exists(sync_script):
                    st.write("🛰️ 正在執行 Master Sync 引擎...")
                    subprocess.run(["python", sync_script], check=True)
                    
                    st.write("📦 正在佈署數據至本地存儲庫...")
                    target_multi = os.path.join(repo_data_path, "Multi_indicator")
                    files_to_sync = ["dashboard_follow_through.html", "spy_data.js", "ntsx_dashboard.html", "ntsx_data.js"]
                    
                    os.makedirs(target_multi, exist_ok=True)
                    for f in files_to_sync:
                        src = os.path.join(source_multi, f)
                        if os.path.exists(src):
                            shutil.copy2(src, target_multi)
                    
                    target_plat = os.path.join(repo_data_path, "Platinum_Results")
                    if os.path.exists(source_plat):
                        shutil.copytree(source_plat, target_plat, dirs_exist_ok=True)
                    
                    status.update(label="✅ 系統數據已更新！", state="complete", expanded=False)
                    st.toast("數據同步完成！")
                else:
                    status.update(label="❌ 無法在雲端直接刷新數據", state="error", expanded=True)
                    if not is_windows:
                        st.markdown("""
                        ### ☁️ 雲端同步工作流 (Workflow)
                        目前處於雲端環境，無法直接執行本地腳本。請執行以下步驟更新數據：
                        1. **本地執行**: 在您的電腦上執行 `streamlit run Market_Regime.py`。
                        2. **點擊刷新**: 在本地端點擊 `⚡ 數據刷新`。
                        3. **推送 GitHub**: 執行 `git push` 將 `data/` 資料夾推送至倉庫。
                        4. **自動更新**: Streamlit Cloud 將自動同步最新數據。
                        """)
                    else:
                        st.error(f"找不到同步腳本，請確認路徑是否正確: {sync_script}")
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
    
    st.page_link("Market_Regime.py", label="市場體系：Market Regime", icon="🚦")
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

import streamlit as st
import datetime
import os
import subprocess
import shutil

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
                # 1. Run sync_all.py in the Multi indicator directory
                sync_script = r"d:\Backtest\Multi indicator\sync_all.py"
                if os.path.exists(sync_script):
                    st.write("🛰️ 正在執行 Master Sync 引擎...")
                    # We run this in the background to avoid blocking too much, 
                    # but here we wait for it to finish for "factual" parity.
                    subprocess.run(["python", sync_script], check=True)
                    
                    # 2. Copy the updated results to the repository's data folder
                    st.write("📦 正在佈署數據至本地存儲庫...")
                    repo_data_path = r"d:\Backtest\Makret-Bull-and-Bear-Trap\data"
                    
                    # Copy Multi indicator results
                    source_multi = r"d:\Backtest\Multi indicator"
                    target_multi = os.path.join(repo_data_path, "Multi_indicator")
                    files_to_sync = ["dashboard_follow_through.html", "spy_data.js", "ntsx_dashboard.html", "ntsx_data.js"]
                    for f in files_to_sync:
                        src = os.path.join(source_multi, f)
                        if os.path.exists(src):
                            shutil.copy2(src, target_multi)
                    
                    # Copy Platinum results
                    source_plat = r"d:\Backtest\Platinum_Results"
                    target_plat = os.path.join(repo_data_path, "Platinum_Results")
                    if os.path.exists(source_plat):
                        shutil.copytree(source_plat, target_plat, dirs_exist_ok=True)
                    
                    status.update(label="✅ 系統數據已更新！", state="complete", expanded=False)
                    st.toast("數據同步完成！")
                else:
                    st.error("找不到同步腳本，請確認路徑是否正確。")
        except Exception as e:
            st.error(f"同步失敗: {e}")

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
<div style="display: flex; flex-direction: column; gap: 10px;">
<a href="/FTD_Strategy" target="_self" style="text-decoration: none;">
<div style="background-color: #ef4444; color: black; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 0.9rem; transition: 0.3s; cursor: pointer;">🚀 底部確認 : FTD 追蹤</div>
</a>
<a href="/NTSX_Strategy" target="_self" style="text-decoration: none;">
<div style="background-color: #22c55e; color: black; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 0.9rem; transition: 0.3s; cursor: pointer;">🛡️ 資產配置 : NTSX 策略</div>
</a>
<a href="/Platinum_Strategy" target="_self" style="text-decoration: none;">
<div style="background-color: #eab308; color: black; padding: 10px; border-radius: 6px; font-weight: bold; font-size: 0.9rem; transition: 0.3s; cursor: pointer;">💎 核心增益 : Platinum 策略</div>
</a>
</div>
</div>
""", unsafe_allow_html=True)

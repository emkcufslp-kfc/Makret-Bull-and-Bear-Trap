import streamlit as st

def render_ecosystem_sidebar():
    """Unified sidebar navigation for the Makret-Bull-and-Bear-Trap ecosystem."""
    # CSS to hide the redundant default navigation and style the custom sidebar
    st.markdown("""
<style>
    /* Hide default Streamlit sidebar navigation */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    /* Hover effects for custom buttons */
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

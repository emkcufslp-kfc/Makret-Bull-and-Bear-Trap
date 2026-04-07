import streamlit as st

def render_ecosystem_sidebar():
    """Unified sidebar navigation for the Makret-Bull-and-Bear-Trap ecosystem."""
    st.markdown("""
    <div style="background-color: #0f172a; padding: 15px; border-radius: 8px; margin: 20px 0; text-align: center; border: 1px solid #334155;">
        <h3 style="color: white; margin-top: 0; font-size: 1.1rem;">🌐 量化決策生態系統</h3>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 5px;">快速切換監控面板</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Internal Streamlit Page Links (Private App Optimization)
    st.page_link("pages/1_🔴_Market_Regime.py", label="🔴 市場體系：Market Regime", icon="🚦")
    st.page_link("pages/2_🐻_Bear_Trap.py", label="🐻 熊市陷阱：Bear Trap", icon="🐻")
    st.page_link("pages/3_🐂_Bull_Trap.py", label="🐂 牛市陷阱：Bull Trap", icon="🐂")
    st.page_link("pages/4_📊_ETF_Rotation.py", label="📊 輪動監控：ETF Rotation", icon="🚀")
    st.page_link("pages/5_📈_200MA_Strategy.py", label="📈 趨勢防禦：200MA Strategy", icon="🛡️")
    st.page_link("pages/6_🎯_Meta_Indicator.py", label="🎯 核心增益：Meta Indicator", icon="💎")

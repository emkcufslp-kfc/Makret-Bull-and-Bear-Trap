import streamlit as st
import streamlit.components.v1 as components
import os

# Set Page Config
st.set_page_config(layout="wide", page_title="NTSX / KMLM / AVWS Strategy Dashboard", page_icon="🛡️")

def render_ntsx_dashboard():
    # Define relative paths from this file to the source dashboards
    current_dir = os.path.dirname(__file__)
    # Repository-root-relative path (from pages/ subdirectory)
    current_dir = os.path.dirname(__file__)
    # Backtest/Makret-Bull-and-Bear-Trap/pages -> Backtest/Makret-Bull-and-Bear-Trap/data/Multi_indicator/
    html_path = os.path.abspath(os.path.join(current_dir, "../data/Multi_indicator/ntsx_dashboard.html"))
    js_path = os.path.abspath(os.path.join(current_dir, "../data/Multi_indicator/ntsx_data.js"))

    if not os.path.exists(html_path):
        st.error(f"NTSX Dashboard HTML not found at {html_path}")
        return

    try:
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()
            
        # Optional: Inject JS data if not already embedded
        if os.path.exists(js_path):
            with open(js_path, "r", encoding="utf-8", errors="replace") as f:
                js_data = f.read()
            # Replace the JS source tag with the actual data
            html_content = html_content.replace(
                '<script src="ntsx_data.js"></script>',
                f'<script type="text/javascript">{js_data}</script>'
            )
        
        # Render the HTML component
        components.html(html_content, height=1500, scrolling=True)
        
    except Exception as e:
        st.error(f"Error rendering NTSX Dashboard: {e}")

if __name__ == "__main__":
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    with st.sidebar:
        render_master_controls()
        render_ecosystem_sidebar()
    render_ntsx_dashboard()

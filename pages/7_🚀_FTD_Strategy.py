import streamlit as st
import streamlit.components.v1 as components
import os

# Set Page Config
st.set_page_config(layout="wide", page_title="Market Pulse: FTD Tracker", page_icon="🚀")

def render_ftd_dashboard():
    # Define relative paths from this file to the source dashboards
    current_dir = os.path.dirname(__file__)
    # Backtest/Makret-Bull-and-Bear-Trap/pages -> Backtest/Multi indicator/
    html_path = os.path.abspath(os.path.join(current_dir, "../../Multi indicator/dashboard_follow_through.html"))
    js_path = os.path.abspath(os.path.join(current_dir, "../../Multi indicator/spy_data.js"))

    if not os.path.exists(html_path):
        st.error(f"FTD Dashboard HTML not found at {html_path}")
        return

    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        # Optional: Inject JS data if not already embedded
        if os.path.exists(js_path):
            with open(js_path, "r", encoding="utf-8") as f:
                js_data = f.read()
            # Replace the JS source tag with the actual data
            html_content = html_content.replace(
                '<script type="text/javascript" src="spy_data.js"></script>',
                f'<script type="text/javascript">{js_data}</script>'
            )
        
        # Render the HTML component
        components.html(html_content, height=1500, scrolling=True)
        
    except Exception as e:
        st.error(f"Error rendering FTD Dashboard: {e}")

if __name__ == "__main__":
    from utils.ui_utils import render_ecosystem_sidebar
    with st.sidebar:
        render_ecosystem_sidebar()
    render_ftd_dashboard()

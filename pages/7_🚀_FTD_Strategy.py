import streamlit as st
import streamlit.components.v1 as components
import os

# Set Page Config
st.set_page_config(layout="wide", page_title="Market Pulse: FTD Tracker", page_icon="🚀")

def render_ftd_dashboard():
    # Define relative paths from this file to the source dashboards
    current_dir = os.path.dirname(__file__)
    # Repository-root-relative path (from pages/ subdirectory)
    current_dir = os.path.dirname(__file__)
    # Backtest/Makret-Bull-and-Bear-Trap/pages -> Backtest/Makret-Bull-and-Bear-Trap/data/Multi_indicator/
    html_path = os.path.abspath(os.path.join(current_dir, "../data/Multi_indicator/dashboard_follow_through.html"))
    js_path = os.path.abspath(os.path.join(current_dir, "../data/Multi_indicator/spy_data.js"))

    if not os.path.exists(html_path):
        st.error(f"FTD Dashboard HTML not found at {html_path}")
        return

    try:
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()
            
        # Robust Regex-based Injection
        import re
        master_date_str = st.session_state['master_date'].strftime('%Y-%m-%d')
        js_data = ""
        
        if os.path.exists(js_path):
            with open(js_path, "r", encoding="utf-8", errors="replace") as f:
                js_data = f.read()
            
            # Prepare the filter script
            filter_script = f"""
// Master Date Sync (Enforced)
if (typeof spyHistoricalData !== 'undefined') {{
    spyHistoricalData = spyHistoricalData.filter(d => d.date <= '{master_date_str}');
}}
"""
            # Replace the JS source tag regardless of whitespace or type attribute
            # If the tag is not found (e.g. modified HTML), we append to head or body
            js_injection = f'<script type="text/javascript">{js_data}\n{filter_script}</script>'
            if re.search(r'<script\s+[^>]*src="spy_data\.js"[^>]*></script>', html_content):
                html_content = re.sub(r'<script\s+[^>]*src="spy_data\.js"[^>]*></script>', js_injection, html_content)
            else:
                # Prepend to closing body tag if script tag missing
                html_content = html_content.replace('</body>', f'{js_injection}</body>')
        else:
            # Fallback filter injection
            html_content = html_content.replace(
                '</body>',
                f"<script type='text/javascript'>if (typeof spyHistoricalData !== 'undefined') {{ spyHistoricalData = spyHistoricalData.filter(d => d.date <= '{master_date_str}'); }}</script></body>"
            )
        
        # Render the HTML component
        components.html(html_content, height=1500, scrolling=True)
        
    except Exception as e:
        st.error(f"Error rendering FTD Dashboard: {e}")

if __name__ == "__main__":
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    with st.sidebar:
        render_master_controls()
        render_ecosystem_sidebar()
    render_ftd_dashboard()

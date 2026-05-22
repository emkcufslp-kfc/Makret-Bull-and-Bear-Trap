import streamlit as st
import streamlit.components.v1 as components
import json
import re
from pathlib import Path

# Set Page Config
st.set_page_config(layout="wide", page_title="Market Pulse: FTD Tracker", page_icon="🚀")

def render_ftd_dashboard():
    root_path = Path(__file__).parent.parent
    html_path = root_path / "data" / "Multi_indicator" / "dashboard_follow_through.html"
    js_path = root_path / "data" / "Multi_indicator" / "spy_data.js"
    cache_path = root_path / "data" / "sentiment_cache.json"

    if not html_path.exists():
        st.error("FTD Dashboard HTML not found.")
        return

    # ✅ Load latest sentiment data
    sentiment_data = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                sentiment_data = json.load(f)
        except Exception as e:
            st.error(f"Error loading sentiment cache: {e}")
    
    try:
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()
    except Exception as e:
        st.error(f"Error reading HTML file: {e}")
        return

    # ✅ Load spy_data.js content
    js_data = ""
    if js_path.exists():
        try:
            with open(js_path, "r", encoding="utf-8", errors="replace") as f:
                js_data = f.read()
        except Exception as e:
            st.error(f"Error reading spy_data.js: {e}")

    # ✅ Inject spy_data.js directly into the HTML to fix relative path issues in streamlit iframe sandbox
    if js_data:
        js_injection = f'<script type="text/javascript">{js_data}</script>'
        if re.search(r'<script\s+[^>]*src="spy_data\.js"[^>]*></script>', html_content):
            html_content = re.sub(r'<script\s+[^>]*src="spy_data\.js"[^>]*></script>', lambda m: js_injection, html_content)
        else:
            html_content = html_content.replace('<head>', f'<head>{js_injection}')

    # ✅ Inject latest sentiment values into Indicator 2
    inject_script = f"""
    <script>
    window.LIVE_SENTIMENT = {json.dumps(sentiment_data)};
    document.addEventListener('DOMContentLoaded', function() {{
        document.querySelector('[id="aaii-bearish"]')?.textContent = 
            window.LIVE_SENTIMENT.aaii_bearish || '49.8%';
        document.querySelector('[id="cnn-fear"]')?.textContent = 
            window.LIVE_SENTIMENT.cnn_fear_index || 'Loading...';
    }});
    </script>
    """
    html_content = html_content.replace('</body>', inject_script + '</body>')
    
    components.html(html_content, height=1500, scrolling=True)

if __name__ == "__main__":
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    with st.sidebar:
        render_master_controls()
        render_ecosystem_sidebar()
    render_ftd_dashboard()



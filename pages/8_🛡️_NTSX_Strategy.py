import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import os
import re

# Set Page Config
st.set_page_config(layout="wide", page_title="NTSX / KMLM / AVWS Strategy Dashboard", page_icon="🛡️")

def render_ntsx_dashboard():
    # Repository-root-relative path (Solid Pathing for Public Deployment)
    root_path = Path(__file__).parent.parent
    html_path = root_path / "data" / "Multi_indicator" / "ntsx_dashboard.html"
    js_path = root_path / "data" / "Multi_indicator" / "ntsx_data.js"

    if not html_path.exists():
        st.error(f"NTSX Dashboard HTML not found")
        return

    try:
        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            html_content = f.read()
            
        # Robust Regex-based Injection
        master_date_str = st.session_state['master_date'].strftime('%Y-%m-%d')
        js_data = ""
        
        if js_path.exists():
            with open(js_path, "r", encoding="utf-8", errors="replace") as f:
                js_data = f.read()

            # Prepare the filter script
            filter_script = f"""
// Master Date Sync (Enforced)
if (typeof NTSX_EQUITY !== 'undefined') {{
    const mDate = '{master_date_str}';
    const mYear = parseInt(mDate.substring(0,4));
    NTSX_EQUITY = NTSX_EQUITY.filter(d => d.date <= mDate);
    NTSX_REBALANCES = NTSX_REBALANCES.filter(r => r.date <= mDate);
    NTSX_YEARLY = NTSX_YEARLY.filter(y => parseInt(y.year) <= mYear);
    NTSX_REB_DATES = NTSX_REB_DATES.filter(d => d <= mDate);
    
    // Explicitly update NTSX_CURRENT to reflect the filtered state
    if (NTSX_EQUITY.length > 0) {{
        const last = NTSX_EQUITY[NTSX_EQUITY.length - 1];
        if (typeof NTSX_CURRENT !== 'undefined') {{
            NTSX_CURRENT.as_of_date = last.date;
        }}
    }}
}}
"""
            # Replace the JS source tag with the actual data + filter
            # regex handles whitespace and any tag formatting
            js_injection = f'<script type="text/javascript">{js_data}\n{filter_script}</script>'
            html_content = re.sub(r'<script\s+src="ntsx_data\.js"\s*></script>', js_injection, html_content)
        else:
            # Fallback filter injection
            html_content = html_content.replace(
                '</body>',
                f"<script type='text/javascript'>const mDate = '{master_date_str}'; if (typeof NTSX_EQUITY !== 'undefined') {{ NTSX_EQUITY = NTSX_EQUITY.filter(d => d.date <= mDate); }}</script></body>"
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

import json
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(layout="wide", page_title="Market Pulse: FTD Tracker", page_icon="FTD")


def extract_latest_market_date(js_data: str):
    matches = re.findall(r'"date":\s*"(\d{4}-\d{2}-\d{2})"', js_data)
    if not matches:
        return None
    try:
        return datetime.strptime(matches[-1], "%Y-%m-%d")
    except ValueError:
        return None


def refresh_stale_date_labels(html_content: str, latest_date: datetime) -> str:
    full_date = latest_date.strftime("%B %d, %Y").replace(" 0", " ")
    month_year = latest_date.strftime("%B %Y")

    # Refresh the exported artifact's stale visible date labels from live market data.
    html_content = re.sub(r"\bMarch 30, 2026\b", full_date, html_content)
    html_content = re.sub(r"\bMarch 2026\b", month_year, html_content)
    html_content = re.sub(
        r"Comprehensive Finfluencer Consensus\s+[-—]?\s*[A-Za-z]+\s+\d{1,2},\s+\d{4}",
        f"Comprehensive Finfluencer Consensus - {full_date}",
        html_content,
        flags=re.IGNORECASE,
    )
    return html_content


def render_ftd_dashboard():
    root_path = Path(__file__).parent.parent
    html_path = root_path / "data" / "Multi_indicator" / "dashboard_follow_through.html"
    js_path = root_path / "data" / "Multi_indicator" / "spy_data.js"
    cache_path = root_path / "data" / "sentiment_cache.json"

    if not html_path.exists():
        st.error("FTD Dashboard HTML not found.")
        return

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

    js_data = ""
    if js_path.exists():
        try:
            with open(js_path, "r", encoding="utf-8", errors="replace") as f:
                js_data = f.read()
        except Exception as e:
            st.error(f"Error reading spy_data.js: {e}")

    latest_market_date = extract_latest_market_date(js_data) if js_data else None
    if latest_market_date is not None:
        html_content = refresh_stale_date_labels(html_content, latest_market_date)

    # Inline the market history so the iframe does not depend on a relative JS path.
    if js_data:
        js_injection = f'<script type="text/javascript">{js_data}</script>'
        if re.search(r'<script\s+[^>]*src="spy_data\.js"[^>]*></script>', html_content):
            html_content = re.sub(
                r'<script\s+[^>]*src="spy_data\.js"[^>]*></script>',
                lambda _: js_injection,
                html_content,
            )
        else:
            html_content = html_content.replace("<head>", f"<head>{js_injection}")

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
    html_content = html_content.replace("</body>", inject_script + "</body>")

    components.html(html_content, height=1500, scrolling=True)


if __name__ == "__main__":
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls

    with st.sidebar:
        render_master_controls()
        render_ecosystem_sidebar()
    render_ftd_dashboard()

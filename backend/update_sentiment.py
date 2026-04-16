import os
import re
import datetime

try:
    import fear_and_greed
except ImportError:
    print("Please install fear-and-greed package: pip install fear-and-greed")
    exit(1)

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.abspath(os.path.join(CURR_DIR, "../data/Multi_indicator/dashboard_follow_through.html"))

def map_score_to_label(score):
    if score < 25: return "Extreme Fear", "&#x1F480;"
    elif score < 45: return "Fear", "&#x1F628;"
    elif score <= 55: return "Neutral", "&#x1F610;"
    elif score <= 75: return "Greed", "&#x1F911;"
    else: return "Extreme Greed", "&#x1F680;"

def get_color(label):
    if "Fear" in label: return "var(--red-bear)"
    elif "Greed" in label: return "var(--green-bull)"
    return "#a0a0a0"

def update_sentiment():
    print("Fetching CNN Fear & Greed Index...")
    try:
        fng = fear_and_greed.get()
        score = float(fng.value)
        label, emoji = map_score_to_label(score)
        color = get_color(label)
    except: return False

    if not os.path.exists(HTML_PATH): return False

    with open(HTML_PATH, 'r', encoding='utf-8') as f: html = f.read()

    cnn_div_pattern = r'<div style="font-size:1\.5rem;font-weight:bold;color:[^>]+>.*?</div>'
    new_cnn_div = f'<div style="font-size:1.5rem;font-weight:bold;color:{color};">{emoji} {label} ({score:.1f})</div>'
    
    if re.search(cnn_div_pattern, html):
        html = re.sub(cnn_div_pattern, new_cnn_div, html)

    today_str = datetime.datetime.now().strftime('%b %d, %Y')
    html = re.sub(r'CNN Fear &amp; Greed Index</a>.*?Readings', f'CNN Fear &amp; Greed Index</a> (Live updated {today_str}). Readings', html)

    with open(HTML_PATH, 'w', encoding='utf-8') as f: f.write(html)
    return True

if __name__ == "__main__":
    update_sentiment()

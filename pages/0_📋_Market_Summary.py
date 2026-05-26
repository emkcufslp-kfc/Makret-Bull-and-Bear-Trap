import datetime
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None

from sklearn.ensemble import RandomForestClassifier

from utils.data_engine import (
    get_clean_master,
    get_data_freshness,
    get_gex,
    get_hy_spread,
    get_move,
    get_sp500_drawdown,
    get_t2108,
)


st.set_page_config(page_title="Market Summary Dashboard", page_icon="📋", layout="wide")

ROOT_DIR = Path(__file__).parent.parent
MARKET_PULSE_HTML = ROOT_DIR / "data" / "Multi_indicator" / "dashboard_follow_through.html"


def normalize(val, lower, upper, inverted=False):
    if inverted:
        if val >= lower:
            return 0.0
        if val <= upper:
            return 1.0
        return round((lower - val) / (lower - upper), 2)
    if val <= lower:
        return 0.0
    if val >= upper:
        return 1.0
    return round((val - lower) / (upper - lower), 2)


def safe_float(series, key, default=0.0):
    value = series.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


@st.cache_data(ttl=1800)
def load_master_slice(target_date):
    data = get_clean_master().ffill().dropna(how="all")
    if data.empty:
        return pd.DataFrame(), None
    ts = pd.Timestamp(target_date)
    valid_dates = data.index[data.index <= ts]
    actual_date = valid_dates[-1] if len(valid_dates) else data.index[-1]
    sliced = data.loc[:actual_date].copy()
    return sliced, actual_date


def calc_market_regime(d):
    latest = d.iloc[-1]
    sp_price = safe_float(latest, "^GSPC")
    dma200 = float(d["^GSPC"].rolling(200).mean().iloc[-1]) if "^GSPC" in d.columns else 0.0
    vix = safe_float(latest, "^VIX", 20.0)
    vix3m = safe_float(latest, "^VIX3M", 21.0)
    hy_spread_pct = get_hy_spread(d.index[-1].date())
    move = get_move(d.index[-1].date())
    t2108 = get_t2108(d.index[-1].date())
    dxy = safe_float(latest, "DX-Y.NYB", 100.0)
    liquidity = 7.5e12
    spy_gex = get_gex(d.index[-1].date())

    score = 0
    if sp_price < dma200:
        score += 15
    if hy_spread_pct > 5:
        score += 20
    if move > 100:
        score += 15
    if vix > 25:
        score += 10
    if vix > vix3m:
        score += 10
    if dxy > 105:
        score += 10
    if t2108 < 40:
        score += 10
    if spy_gex < 0:
        score += 5
    if liquidity < 7.0e12:
        score += 5

    probability = min(score, 100)
    if probability < 30:
        regime_status = "LOW RISK REGIME"
    elif probability < 55:
        regime_status = "EARLY WARNING"
    else:
        regime_status = "HIGH RISK"
    return {
        "probability": round(probability, 1),
        "color": "#22c55e" if probability < 30 else ("#f59e0b" if probability < 55 else "#ef4444"),
        "status": regime_status,
    }


def calc_bear_trap(d):
    latest = d.iloc[-1]
    yield_curve = safe_float(latest, "^TNX") - safe_float(latest, "^IRX")
    macro_score = normalize(yield_curve, 1.5, -0.5, inverted=True)

    irx_avg = d["^IRX"].rolling(120).mean().iloc[-1]
    liquidity_score = normalize(safe_float(latest, "^IRX"), irx_avg * 0.8, irx_avg * 1.2)

    hyg_ief = d["HYG"] / d["IEF"]
    hyg_ief_avg = hyg_ief.rolling(252).mean().iloc[-1]
    current_hyg_ief = hyg_ief.iloc[-1]
    credit_score = normalize(current_hyg_ief, hyg_ief_avg * 1.05, hyg_ief_avg * 0.9, inverted=True)

    spy_200ma = d["SPY"].rolling(200).mean().iloc[-1]
    breadth_score = normalize(safe_float(latest, "SPY"), spy_200ma * 1.05, spy_200ma * 0.95, inverted=True)
    vix_score = normalize(safe_float(latest, "^VIX"), 15, 35)

    total_score = (
        (macro_score * 0.25)
        + (liquidity_score * 0.20)
        + (credit_score * 0.20)
        + (breadth_score * 0.15)
        + (vix_score * 0.10)
        + (0.65 * 0.05)
        + (0.50 * 0.05)
    )

    if total_score < 0.4:
        risk_level, risk_color = "LOW RISK", "#22c55e"
    elif total_score < 0.55:
        risk_level, risk_color = "EARLY WARNING", "#f59e0b"
    else:
        risk_level, risk_color = "HIGH RISK", "#ef4444"

    return {
        "prob_3m": round(total_score * 0.60 * 100, 1),
        "prob_6m": round(total_score * 0.85 * 100, 1),
        "prob_12m": round(total_score * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
    }


def calc_bull_trap(d):
    latest = d.iloc[-1]
    prev_mo = d.iloc[max(0, len(d) - 23)]
    scores = {}

    curve = safe_float(latest, "^TNX") - safe_float(latest, "^IRX")
    prev_curve = safe_float(prev_mo, "^TNX") - safe_float(prev_mo, "^IRX")
    scores["Yield Curve"] = 1.0 if curve > 0 and prev_curve < 0 else (0.5 if curve > 0 else 0.0)

    vix_mavg = d["^VIX"].rolling(22).mean().iloc[-1]
    vix = safe_float(latest, "^VIX")
    scores["VIX Regime"] = 1.0 if vix < 15 else (0.5 if vix < vix_mavg else 0.0)

    hyg_ief = d["HYG"] / d["IEF"]
    scores["Credit Stress"] = 1.0 if hyg_ief.iloc[-1] > hyg_ief.rolling(22).mean().iloc[-1] else 0.0

    spy_200ma = d["SPY"].rolling(200).mean().iloc[-1]
    spy = safe_float(latest, "SPY")
    scores["Market Breadth"] = 1.0 if spy > spy_200ma * 1.05 else (0.5 if spy > spy_200ma else 0.0)

    spy_mom = (spy / safe_float(prev_mo, "SPY")) - 1
    scores["Accumulation"] = 1.0 if spy_mom > 0.02 else (0.5 if spy_mom > 0 else 0.0)

    scores["Liquidity"] = 1.0 if ("TIP" in d.columns and len(d) >= 22 and d["TIP"].iloc[-1] > d["TIP"].iloc[-22]) else 0.0
    scores["Valuation"] = 0.5
    scores["Insider Buying"] = 0.5

    total_score = min(10.0, sum(scores.values()) + 1.0)
    if total_score >= 10:
        regime, prob = "Structural Bull Market", 95.0
    elif total_score >= 8:
        regime, prob = "Strong Bull Market", 85.0
    elif total_score >= 6:
        regime, prob = "Early Bull Market", 65.0
    elif total_score >= 4:
        regime, prob = "Bull Trap Risk", 40.0
    else:
        regime, prob = "Bear Market", 20.0

    if total_score >= 6:
        market_status, color = "BULLISH / LOW RISK", "#22c55e"
    elif total_score >= 4:
        market_status, color = "CAUTION / BULL TRAP RISK", "#f59e0b"
    else:
        market_status, color = "BEARISH / HIGH RISK", "#ef4444"

    return {"probability": prob, "regime": regime, "market_status": market_status, "color": color}


def calc_etf_rotation(d):
    spy_price = safe_float(d.iloc[-1], "SPY")
    spy_200 = float(d["SPY"].rolling(200).mean().iloc[-1])
    vix = safe_float(d.iloc[-1], "^VIX", 20.0)
    if spy_price < spy_200 or vix > 20:
        return {"status": "EARLY WARNING", "color": "#f59e0b"}
    return {"status": "NORMAL", "color": "#22c55e"}


def calc_200ma_strategy(d):
    spy_hist = d[["^GSPC"]].copy()
    spy_hist.columns = ["Close"]
    spy_hist["200MA"] = spy_hist["Close"].rolling(window=200).mean()
    current_sp = float(spy_hist["Close"].iloc[-1])
    current_200ma = float(spy_hist["200MA"].iloc[-1])
    pct_diff = (current_sp - current_200ma) / current_200ma
    if pct_diff > 0.02:
        trend_status, color = "BULLISH (Wait for Exit)", "#22c55e"
    elif pct_diff >= -0.02:
        trend_status, color = "CAUTION (Trend Testing)", "#f59e0b"
    else:
        trend_status, color = "BEARISH (Keep Cash)", "#ef4444"
    return {"sp_price": current_sp, "sma_200": current_200ma, "trend_status": trend_status, "color": color}


class MLMetaIndicator:
    def __init__(self):
        self.meta_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        self.hmm_model = None

    def fit_predict(self, df):
        df = df.copy()
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Volatility"] = df["Return"].rolling(window=20).std()
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["Primary_Signal"] = 0
        df.loc[df["SMA_10"] > df["SMA_50"], "Primary_Signal"] = 1
        df.loc[df["SMA_10"] < df["SMA_50"], "Primary_Signal"] = -1
        df["Primary_Signal"] = df["Primary_Signal"].shift(1).fillna(0)
        df = df.dropna(subset=["SMA_10", "SMA_50", "Volatility"])
        if df.empty:
            return None
        if GaussianHMM is not None:
            x_hmm = df["Return"].fillna(0).values.reshape(-1, 1)
            self.hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
            self.hmm_model.fit(x_hmm)
            df["Regime"] = self.hmm_model.predict(x_hmm)
        else:
            df["Regime"] = 0

        meta_labels = pd.Series(index=df.index, data=np.nan, dtype=float)
        for idx in df[df["Primary_Signal"] != 0].index:
            idx_pos = df.index.get_loc(idx)
            if idx_pos + 15 >= len(df):
                continue
            entry_price = float(df["Close"].iloc[idx_pos])
            signal_side = float(df["Primary_Signal"].iloc[idx_pos])
            vol = float(df["Volatility"].iloc[idx_pos])
            pt_price = entry_price * (1 + signal_side * 1.5 * vol)
            sl_price = entry_price * (1 - signal_side * 1.0 * vol)
            path = df["Close"].iloc[idx_pos + 1: idx_pos + 16]
            label = 0
            for future_price in path:
                future_price = float(future_price)
                if (signal_side == 1 and future_price >= pt_price) or (signal_side == -1 and future_price <= pt_price):
                    label = 1
                    break
                if (signal_side == 1 and future_price <= sl_price) or (signal_side == -1 and future_price >= sl_price):
                    label = 0
                    break
            meta_labels.at[idx] = label
        df["Meta_Label"] = meta_labels
        ml_df = df.dropna(subset=["Meta_Label"]).copy()
        if ml_df.empty:
            df["Meta_Probability"] = 0.5
            return df[["Meta_Probability", "Primary_Signal"]]
        features = ["Return", "Volatility", "SMA_10", "SMA_50", "Regime"]
        self.meta_model.fit(ml_df[features].fillna(0), ml_df["Meta_Label"])
        x_all = df[features].fillna(0)
        df["Meta_Probability"] = self.meta_model.predict_proba(x_all)[:, 1]
        return df[["Meta_Probability", "Primary_Signal"]]


def calc_meta_indicator(d, analysis_date):
    spy = d[["SPY"]].rename(columns={"SPY": "Close"}).copy()
    start_date = pd.Timestamp(analysis_date) - pd.DateOffset(years=5)
    work = spy.loc[start_date:].copy()
    results = MLMetaIndicator().fit_predict(work)
    if results is None or results.empty:
        return {"trend_probability": 50.0, "meta_score": 50.0, "color": "#94a3b8"}
    current_prob = float(results["Meta_Probability"].iloc[-1] * 100)
    if current_prob > 60:
        color, status = "#22c55e", "HIGH CONFIDENCE (BUY/HOLD)"
    elif current_prob > 40:
        color, status = "#f59e0b", "NEUTRAL / CAUTION"
    else:
        color, status = "#ef4444", "LOW CONFIDENCE (REDUCE)"
    return {"trend_probability": round(current_prob, 1), "meta_score": round(current_prob, 1), "color": color, "status": status}


def get_latest_sync_timestamp(source_name_fragment):
    for item in get_data_freshness():
        if source_name_fragment.lower() in item["Source"].lower():
            return item["Last Update"]
    return "Unknown"


def extract_market_pulse():
    defaults = {"antiskilled_pct": "56%", "greed_index": "58.6", "contrarian_signal": "BULLISH", "updated": "Artifact"}
    if not MARKET_PULSE_HTML.exists():
        return defaults
    text = MARKET_PULSE_HTML.read_text(encoding="utf-8", errors="replace")

    antiskilled = re.search(r"Antiskilled\s*\((\d+)% majority\)", text, re.IGNORECASE)
    greed = re.search(r"Greed\s*\(([\d.]+)\)", text, re.IGNORECASE)
    verdict = re.search(r"SFI Contrarian Verdict:\s*([A-Z ]+)", text, re.IGNORECASE)
    contrarian = re.search(r"Contrarian Signal:\s*([A-Z ]+)", text, re.IGNORECASE)
    updated = re.search(r"CNN Fear &amp; Greed Index.*?Live updated ([A-Za-z]+\s+\d{1,2},\s+\d{4})", text, re.IGNORECASE | re.DOTALL)

    return {
        "antiskilled_pct": f"{antiskilled.group(1)}%" if antiskilled else defaults["antiskilled_pct"],
        "greed_index": greed.group(1) if greed else defaults["greed_index"],
        "contrarian_signal": (
            verdict.group(1).strip()
            if verdict
            else contrarian.group(1).strip() if contrarian else defaults["contrarian_signal"]
        ),
        "updated": updated.group(1) if updated else defaults["updated"],
    }


def badge(text, color):
    return f"<span class='status-pill' style='background:{color}1a;color:{color};border-color:{color}55;'>{text}</span>"


def render_page():
    with st.sidebar:
        from utils.ui_utils import render_ecosystem_sidebar, render_master_controls

        render_master_controls()
        render_ecosystem_sidebar()

    analysis_date = st.session_state["master_date"]
    data, actual_date = load_master_slice(analysis_date)
    if data.empty or actual_date is None or len(data) < 260:
        st.error("Insufficient data to build the Market Summary Dashboard.")
        return

    market_regime = calc_market_regime(data)
    bear_trap = calc_bear_trap(data)
    bull_trap = calc_bull_trap(data)
    etf_rotation = calc_etf_rotation(data)
    ma_200 = calc_200ma_strategy(data)
    meta_indicator = calc_meta_indicator(data, analysis_date)
    market_pulse = extract_market_pulse()
    master_sync = get_latest_sync_timestamp("Master DB")

    st.markdown(
        """
        <style>
        .summary-shell {padding-top: 0.25rem;}
        .summary-title {font-size: 2.1rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.2rem;}
        .summary-subtitle {font-size: 0.95rem; color: #94a3b8; margin-bottom: 1.25rem;}
        .summary-grid {display:grid; grid-template-columns: repeat(7, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px;}
        .summary-card {
            background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(15,23,42,0.88));
            border: 1px solid rgba(71,85,105,0.45);
            border-radius: 16px;
            padding: 16px 16px 14px 16px;
            min-height: 208px;
            box-shadow: 0 12px 32px rgba(2, 6, 23, 0.28);
        }
        .summary-kicker {font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.09em; color: #94a3b8; margin-bottom: 0.35rem;}
        .summary-head {font-size: 1rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.9rem;}
        .summary-metric {font-size: 2rem; font-weight: 800; line-height: 1.05; margin-bottom: 0.5rem;}
        .summary-label {font-size: 0.74rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em;}
        .summary-row {display:flex; justify-content:space-between; align-items:flex-end; gap: 10px; margin-bottom: 0.55rem;}
        .summary-value {font-size: 1.35rem; font-weight: 800; color: #f8fafc;}
        .summary-subvalue {font-size: 1.1rem; font-weight: 700; color: #e2e8f0;}
        .summary-divider {height: 1px; background: rgba(71,85,105,0.35); margin: 0.8rem 0;}
        .status-pill {
            display: inline-flex; align-items: center; justify-content: center;
            padding: 0.32rem 0.65rem; border-radius: 999px; border: 1px solid;
            font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
        }
        .snapshot-card {
            background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(15,23,42,0.88));
            border: 1px solid rgba(71,85,105,0.45);
            border-radius: 18px;
            padding: 18px;
            margin-top: 10px;
        }
        .snapshot-header {display:flex; justify-content:space-between; align-items:flex-start; gap: 12px; margin-bottom: 14px;}
        .snapshot-title {font-size: 1.05rem; font-weight: 800; color: #f8fafc;}
        .snapshot-subtitle {font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem;}
        .snapshot-grid {display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px;}
        .snapshot-item {background: rgba(2,6,23,0.3); border: 1px solid rgba(51,65,85,0.5); border-radius: 14px; padding: 14px;}
        .snapshot-item-title {font-size: 0.82rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.35rem;}
        .snapshot-item-text {font-size: 0.76rem; color: #94a3b8; line-height: 1.45;}
        @media (max-width: 1500px) {.summary-grid {grid-template-columns: repeat(4, minmax(0, 1fr));}}
        @media (max-width: 1100px) {.summary-grid, .snapshot-grid {grid-template-columns: repeat(2, minmax(0, 1fr));}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    market_pulse_signal = market_pulse["contrarian_signal"].upper()
    market_pulse_badge = "#22c55e" if "BULL" in market_pulse_signal else "#f59e0b"
    bull_badge = bull_trap["color"]

    cards_html = f"""
    <div class="summary-shell">
        <div class="summary-title">Market Summary Dashboard</div>
        <div class="summary-subtitle">Unified daily readout across all market modules. Analyzed through {actual_date.strftime('%Y-%m-%d')} with master sync {master_sync}.</div>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-kicker">1. Market Regime</div>
                <div class="summary-head">Crash Probability</div>
                <div class="summary-metric" style="color:{market_regime['color']};">{market_regime['probability']:.1f}%</div>
                <div style="margin-top:1.1rem;">{badge(market_regime['status'], market_regime['color'])}</div>
            </div>
            <div class="summary-card">
                <div class="summary-kicker">2. Bear Trap</div>
                <div class="summary-head">Bear Probability</div>
                <div class="summary-row"><span class="summary-label">3M</span><span class="summary-subvalue" style="color:#f8fafc;">{bear_trap['prob_3m']:.1f}%</span></div>
                <div class="summary-row"><span class="summary-label">6M</span><span class="summary-subvalue" style="color:#fbbf24;">{bear_trap['prob_6m']:.1f}%</span></div>
                <div class="summary-row"><span class="summary-label">12M</span><span class="summary-subvalue" style="color:#f87171;">{bear_trap['prob_12m']:.1f}%</span></div>
                <div class="summary-divider"></div>
                {badge(bear_trap['risk_level'], bear_trap['risk_color'])}
            </div>
            <div class="summary-card">
                <div class="summary-kicker">3. Bull Trap</div>
                <div class="summary-head">Bull Market Probability</div>
                <div class="summary-metric" style="color:{bull_trap['color']};">{bull_trap['probability']:.1f}%</div>
                <div class="summary-divider"></div>
                {badge(bull_trap['market_status'], bull_badge)}
            </div>
            <div class="summary-card">
                <div class="summary-kicker">4. ETF Rotation</div>
                <div class="summary-head">Current Status</div>
                <div style="margin-top:2.2rem;">{badge(etf_rotation['status'], etf_rotation['color'])}</div>
            </div>
            <div class="summary-card">
                <div class="summary-kicker">5. 200MA Strategy</div>
                <div class="summary-head">Trend Status</div>
                <div class="summary-row"><span class="summary-label">S&amp;P Current</span><span class="summary-value">${ma_200['sp_price']:,.0f}</span></div>
                <div class="summary-row"><span class="summary-label">200SMA</span><span class="summary-value">${ma_200['sma_200']:,.0f}</span></div>
                <div class="summary-divider"></div>
                {badge(ma_200['trend_status'], ma_200['color'])}
            </div>
            <div class="summary-card">
                <div class="summary-kicker">6. ML Meta-Indicator</div>
                <div class="summary-head">Trend-Following Success</div>
                <div class="summary-metric" style="color:{meta_indicator['color']};">{meta_indicator['trend_probability']:.1f}%</div>
                <div class="summary-divider"></div>
                {badge(meta_indicator['status'], meta_indicator['color'])}
            </div>
            <div class="summary-card">
                <div class="summary-kicker">7. Market Pulse</div>
                <div class="summary-head">Crowd / Contrarian Read</div>
                <div class="summary-row"><span class="summary-label">Antiskilled Finfluencers</span><span class="summary-subvalue" style="color:#f87171;">{market_pulse['antiskilled_pct']}</span></div>
                <div class="summary-row"><span class="summary-label">Greed Index</span><span class="summary-subvalue" style="color:#fbbf24;">{market_pulse['greed_index']}</span></div>
                <div class="summary-divider"></div>
                {badge(market_pulse['contrarian_signal'], market_pulse_badge)}
            </div>
        </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

    snapshot_html = f"""
    <div class="snapshot-card">
        <div class="snapshot-header">
            <div>
                <div class="snapshot-title">Cross-Market Snapshot</div>
                <div class="snapshot-subtitle">Updated Today · Signal Alignment · Confidence Overlay</div>
            </div>
            {badge('Live Summary', '#38bdf8')}
        </div>
        <div class="snapshot-grid">
            <div class="snapshot-item">
                <div class="snapshot-item-title">Risk Stack</div>
                <div class="snapshot-item-text">Crash probability is <strong style="color:{market_regime['color']};">{market_regime['probability']:.1f}%</strong>, while Bear Trap shows <strong>{bear_trap['prob_12m']:.1f}%</strong> 12M risk. Bull Trap still reads <strong style="color:{bull_trap['color']};">{bull_trap['market_status']}</strong>.</div>
            </div>
            <div class="snapshot-item">
                <div class="snapshot-item-title">Trend Filters</div>
                <div class="snapshot-item-text">ETF Rotation is <strong style="color:{etf_rotation['color']};">{etf_rotation['status']}</strong>. The 200MA system shows the S&amp;P at <strong>${ma_200['sp_price']:,.0f}</strong> versus a 200SMA of <strong>${ma_200['sma_200']:,.0f}</strong>.</div>
            </div>
            <div class="snapshot-item">
                <div class="snapshot-item-title">Machine Learning Overlay</div>
                <div class="snapshot-item-text">Trend-following success probability is <strong style="color:{meta_indicator['color']};">{meta_indicator['trend_probability']:.1f}%</strong>, which currently maps to <strong style="color:{meta_indicator['color']};">{meta_indicator['status']}</strong> on the ML verification page.</div>
            </div>
            <div class="snapshot-item">
                <div class="snapshot-item-title">Crowd Sentiment</div>
                <div class="snapshot-item-text">Market Pulse currently shows <strong>{market_pulse['antiskilled_pct']}</strong> antiskilled finfluencer dominance, Fear &amp; Greed at <strong>{market_pulse['greed_index']}</strong>, and a <strong style="color:{market_pulse_badge};">{market_pulse['contrarian_signal']}</strong> contrarian read.</div>
            </div>
            <div class="snapshot-item">
                <div class="snapshot-item-title">Implementation Note</div>
                <div class="snapshot-item-text">This page is read-only. It extracts the latest outputs and formulas from the existing modules without changing their internal models or execution flow.</div>
            </div>
            <div class="snapshot-item">
                <div class="snapshot-item-title">Data Lineage</div>
                <div class="snapshot-item-text">Master calculations are analyzed through <strong>{actual_date.strftime('%Y-%m-%d')}</strong> and the local master DB was last synced at <strong>{master_sync}</strong>. Market Pulse sentiment artifact last references <strong>{market_pulse['updated']}</strong>.</div>
            </div>
        </div>
    </div>
    """
    st.markdown(snapshot_html, unsafe_allow_html=True)


render_page()

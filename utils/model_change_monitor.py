import datetime as dt

import numpy as np
import pandas as pd

from backend.strategies.combined_macro_rwra import compute_combined_snapshot
from utils.data_engine import get_clean_master, get_gex, get_hy_spread, get_move, get_t2108

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None

from sklearn.ensemble import RandomForestClassifier


CORE_MODELS = [
    "Market Regime",
    "Bear Trap",
    "Bull Trap",
    "ETF Rotation",
    "200MA Strategy",
    "ML Meta-Indicator",
    "Combined Macro + RWRA",
]

ALL_MODELS = CORE_MODELS + ["Market Pulse"]


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
    return {"probability": round(probability, 1), "status": regime_status}


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
        risk_level = "LOW RISK"
    elif total_score < 0.55:
        risk_level = "EARLY WARNING"
    else:
        risk_level = "HIGH RISK"

    return {
        "prob_3m": round(total_score * 0.60 * 100, 1),
        "prob_6m": round(total_score * 0.85 * 100, 1),
        "prob_12m": round(total_score * 100, 1),
        "risk_level": risk_level,
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
        market_status = "BULLISH / LOW RISK"
    elif total_score >= 4:
        market_status = "CAUTION / BULL TRAP RISK"
    else:
        market_status = "BEARISH / HIGH RISK"

    return {"probability": prob, "regime": regime, "market_status": market_status}


def calc_etf_rotation(d):
    spy_price = safe_float(d.iloc[-1], "SPY")
    spy_200 = float(d["SPY"].rolling(200).mean().iloc[-1])
    vix = safe_float(d.iloc[-1], "^VIX", 20.0)
    if spy_price < spy_200 or vix > 20:
        return {"status": "EARLY WARNING"}
    return {"status": "NORMAL"}


def calc_200ma_strategy(d):
    spy_hist = d[["^GSPC"]].copy()
    spy_hist.columns = ["Close"]
    spy_hist["200MA"] = spy_hist["Close"].rolling(window=200).mean()
    current_sp = float(spy_hist["Close"].iloc[-1])
    current_200ma = float(spy_hist["200MA"].iloc[-1])
    pct_diff = (current_sp - current_200ma) / current_200ma
    if pct_diff > 0.02:
        trend_status = "BULLISH (Wait for Exit)"
    elif pct_diff >= -0.02:
        trend_status = "CAUTION (Trend Testing)"
    else:
        trend_status = "BEARISH (Keep Cash)"
    return {"sp_price": current_sp, "sma_200": current_200ma, "trend_status": trend_status}


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
        return {"trend_probability": 50.0, "status": "NEUTRAL / CAUTION"}
    current_prob = float(results["Meta_Probability"].iloc[-1] * 100)
    if current_prob > 60:
        status = "HIGH CONFIDENCE (BUY/HOLD)"
    elif current_prob > 40:
        status = "NEUTRAL / CAUTION"
    else:
        status = "LOW CONFIDENCE (REDUCE)"
    return {"trend_probability": round(current_prob, 1), "status": status}


def market_pulse_status(target_date):
    data, actual_date = load_master_slice(target_date)
    if data.empty or actual_date is None:
        return "UNKNOWN", {}
    latest = data.iloc[-1]
    greed_proxy = min(99.0, max(1.0, 50 + ((safe_float(latest, "SPY") / float(data["SPY"].rolling(22).mean().iloc[-1]) - 1.0) * 200)))
    antiskilled_pct = 56.0 if greed_proxy > 55 else 48.0
    contrarian_signal = "STRONGLY BULLISH" if greed_proxy >= 58 else "CAUTION / MIXED" if greed_proxy >= 45 else "DEFENSIVE"
    return contrarian_signal, {
        "greed_index": round(greed_proxy, 1),
        "antiskilled_pct": f"{antiskilled_pct:.0f}%",
    }


def get_model_status_row(target_date):
    data, actual_date = load_master_slice(target_date)
    if data.empty or actual_date is None or len(data) < 260:
        return None

    market_regime = calc_market_regime(data)
    bear_trap = calc_bear_trap(data)
    bull_trap = calc_bull_trap(data)
    etf_rotation = calc_etf_rotation(data)
    ma_200 = calc_200ma_strategy(data)
    meta_indicator = calc_meta_indicator(data, actual_date.date())
    combined = compute_combined_snapshot(actual_date.date())
    pulse_status, pulse_meta = market_pulse_status(actual_date.date())

    return {
        "Date": actual_date.date(),
        "SPY Price": round(float(data["SPY"].iloc[-1]), 2),
        "Market Regime": market_regime["status"],
        "Bear Trap": bear_trap["risk_level"],
        "Bull Trap": bull_trap["market_status"],
        "ETF Rotation": etf_rotation["status"],
        "200MA Strategy": ma_200["trend_status"],
        "ML Meta-Indicator": meta_indicator["status"],
        "Combined Macro + RWRA": f"{combined.macro_guardrail} | {combined.execution_status}",
        "Market Pulse": pulse_status,
        "Combined Guardrail": combined.macro_guardrail,
        "Combined Execution": combined.execution_status,
        "RWRA Bull Signal": combined.rwra_bull_signal,
        "Greed Index": pulse_meta.get("greed_index"),
    }


def status_shift_count(previous_row, current_row):
    changed = [name for name in CORE_MODELS if previous_row.get(name) != current_row.get(name)]
    return len(changed), changed


def build_warning_mode_events(end_date):
    end_ts = pd.Timestamp(end_date)
    start_ts = end_ts - pd.DateOffset(years=5)
    master = get_clean_master().ffill().dropna(how="all")
    if master.empty:
        return pd.DataFrame()

    valid = master.index[(master.index >= start_ts) & (master.index <= end_ts)]
    if len(valid) < 3:
        return pd.DataFrame()

    rows = []
    previous_snapshot = None
    previous_shift_count = 0
    previous_warning_mode = False

    for ts in valid:
        current_snapshot = get_model_status_row(ts.date())
        if current_snapshot is None:
            continue
        if previous_snapshot is None:
            previous_snapshot = current_snapshot
            continue

        shift_count, changed_models = status_shift_count(previous_snapshot, current_snapshot)
        current_warning_mode = shift_count >= 3
        warning_changed = current_warning_mode != previous_warning_mode

        if warning_changed:
            event = {
                "Date": current_snapshot["Date"].strftime("%Y-%m-%d"),
                "SPY Price": current_snapshot["SPY Price"],
                "Prev Shift Count": previous_shift_count,
                "New Shift Count": shift_count,
                "Warning Mode Change": "Triggered" if current_warning_mode else "Cleared",
                "Changed Core Models": ", ".join(changed_models) if changed_models else "None",
            }
            for model in ALL_MODELS:
                event[model] = current_snapshot[model]
            rows.append(event)

        previous_snapshot = current_snapshot
        previous_shift_count = shift_count
        previous_warning_mode = current_warning_mode

    return pd.DataFrame(rows)

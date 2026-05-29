from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.data_engine import get_clean_master, get_t2108


@dataclass(frozen=True)
class CombinedSnapshot:
    requested_date: dt.date
    resolved_date: dt.date
    previous_date: dt.date | None
    crash_probability: float
    macro_guardrail: str
    guardrail_color: str
    rwra_bull_signal: float
    rwra_probabilities: dict[str, float]
    execution_status: str
    execution_color: str
    turnover_signal: float
    risk_note: str


def _safe_float(series: pd.Series, key: str, default: float = 0.0) -> float:
    value = series.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _load_master_slice(target_date: dt.date) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    data = get_clean_master().ffill().dropna(how="all")
    if data.empty:
        return pd.DataFrame(), None
    ts = pd.Timestamp(target_date)
    valid_dates = data.index[data.index <= ts]
    if len(valid_dates) == 0:
        actual_date = data.index[-1]
    else:
        actual_date = valid_dates[-1]
    return data.loc[:actual_date].copy(), actual_date


def _market_regime_probability(d: pd.DataFrame) -> float:
    latest = d.iloc[-1]
    sp_price = _safe_float(latest, "^GSPC")
    dma200 = float(d["^GSPC"].rolling(200).mean().iloc[-1]) if "^GSPC" in d.columns else 0.0
    vix = _safe_float(latest, "^VIX", 20.0)
    vix3m = _safe_float(latest, "^VIX3M", 21.0)
    t2108 = get_t2108(d.index[-1].date())
    dxy = _safe_float(latest, "DX-Y.NYB", 100.0)
    hyg = _safe_float(latest, "HYG", 0.0)
    ief = _safe_float(latest, "IEF", 1.0)
    credit_proxy = ((ief / hyg) - 1.0) * 100 if hyg else 0.0

    score = 0
    if sp_price < dma200:
        score += 15
    if credit_proxy > 3:
        score += 20
    if vix > 25:
        score += 10
    if vix > vix3m:
        score += 10
    if dxy > 105:
        score += 10
    if t2108 < 40:
        score += 10
    if len(d) >= 22:
        recent_spy = d["SPY"].iloc[-1] / d["SPY"].iloc[-22] - 1 if "SPY" in d.columns else 0.0
        if recent_spy < -0.04:
            score += 15
    return round(min(score, 100), 1)


def _compute_rwra_probabilities(d: pd.DataFrame) -> dict[str, float]:
    latest = d.iloc[-1]
    bearish_signals = 0

    curve = _safe_float(latest, "^TNX") - _safe_float(latest, "^IRX")
    if curve < 0:
        bearish_signals += 1

    if {"HYG", "IEF"}.issubset(d.columns):
        hyg_ief = d["HYG"] / d["IEF"]
        hyg_ief_avg = hyg_ief.rolling(252).mean().iloc[-1]
        if hyg_ief.iloc[-1] < hyg_ief_avg:
            bearish_signals += 1

    vix = _safe_float(latest, "^VIX", 20.0)
    if vix > 20:
        bearish_signals += 1

    if "^GSPC" in d.columns:
        dma200 = d["^GSPC"].rolling(200).mean().iloc[-1]
        if _safe_float(latest, "^GSPC") < dma200:
            bearish_signals += 1

    if get_t2108(d.index[-1].date()) < 40:
        bearish_signals += 1

    mappings = {
        0: {"Bull": 70.0, "Neutral": 20.0, "Bear": 8.0, "Crisis": 2.0},
        1: {"Bull": 50.0, "Neutral": 30.0, "Bear": 15.0, "Crisis": 5.0},
        2: {"Bull": 30.0, "Neutral": 40.0, "Bear": 20.0, "Crisis": 10.0},
        3: {"Bull": 10.0, "Neutral": 35.0, "Bear": 40.0, "Crisis": 15.0},
        4: {"Bull": 5.0, "Neutral": 15.0, "Bear": 50.0, "Crisis": 30.0},
        5: {"Bull": 0.0, "Neutral": 5.0, "Bear": 35.0, "Crisis": 60.0},
    }
    probs = mappings[bearish_signals]
    if vix > 35:
        probs = {"Bull": 0.0, "Neutral": 0.0, "Bear": 0.0, "Crisis": 100.0}
    return probs


def _compute_guardrail(
    crash_probability: float,
    bear_12m: float,
    bull_probability: float,
    rwra_probs: dict[str, float],
) -> tuple[str, str]:
    if rwra_probs["Crisis"] >= 60 or crash_probability >= 85:
        return "CRISIS", "#ef4444"
    if rwra_probs["Bear"] >= 45 or bear_12m >= 75:
        return "RISK OFF", "#f97316"
    if rwra_probs["Bull"] >= 60 and bull_probability >= 20:
        return "STRONG BULL", "#22c55e"
    if rwra_probs["Bull"] >= 45:
        return "BULL", "#38bdf8"
    if rwra_probs["Neutral"] >= 35:
        return "NEUTRAL", "#fbbf24"
    return "RISK OFF", "#f97316"


def _normalize(val: float, lower: float, upper: float, inverted: bool = False) -> float:
    if inverted:
        if val >= lower:
            return 0.0
        if val <= upper:
            return 1.0
        return float((lower - val) / (lower - upper))
    if val <= lower:
        return 0.0
    if val >= upper:
        return 1.0
    return float((val - lower) / (upper - lower))


def _bear_trap_12m(d: pd.DataFrame) -> float:
    latest = d.iloc[-1]
    yield_curve = _safe_float(latest, "^TNX") - _safe_float(latest, "^IRX")
    macro_score = _normalize(yield_curve, 1.5, -0.5, inverted=True)

    irx_avg = d["^IRX"].rolling(120).mean().iloc[-1]
    liquidity_score = _normalize(_safe_float(latest, "^IRX"), irx_avg * 0.8, irx_avg * 1.2)

    hyg_ief = d["HYG"] / d["IEF"]
    hyg_ief_avg = hyg_ief.rolling(252).mean().iloc[-1]
    credit_score = _normalize(hyg_ief.iloc[-1], hyg_ief_avg * 1.05, hyg_ief_avg * 0.9, inverted=True)

    spy_200ma = d["SPY"].rolling(200).mean().iloc[-1]
    breadth_score = _normalize(_safe_float(latest, "SPY"), spy_200ma * 1.05, spy_200ma * 0.95, inverted=True)
    vix_score = _normalize(_safe_float(latest, "^VIX"), 15, 35)

    total_score = (
        (macro_score * 0.25)
        + (liquidity_score * 0.20)
        + (credit_score * 0.20)
        + (breadth_score * 0.15)
        + (vix_score * 0.10)
        + (0.65 * 0.05)
        + (0.50 * 0.05)
    )
    return round(total_score * 100, 1)


def _bull_trap_probability(d: pd.DataFrame) -> float:
    latest = d.iloc[-1]
    prev_mo = d.iloc[max(0, len(d) - 23)]
    scores = []

    curve = _safe_float(latest, "^TNX") - _safe_float(latest, "^IRX")
    prev_curve = _safe_float(prev_mo, "^TNX") - _safe_float(prev_mo, "^IRX")
    scores.append(1.0 if curve > 0 and prev_curve < 0 else (0.5 if curve > 0 else 0.0))

    vix_mavg = d["^VIX"].rolling(22).mean().iloc[-1]
    vix = _safe_float(latest, "^VIX")
    scores.append(1.0 if vix < 15 else (0.5 if vix < vix_mavg else 0.0))

    hyg_ief = d["HYG"] / d["IEF"]
    scores.append(1.0 if hyg_ief.iloc[-1] > hyg_ief.rolling(22).mean().iloc[-1] else 0.0)

    spy_200ma = d["SPY"].rolling(200).mean().iloc[-1]
    spy = _safe_float(latest, "SPY")
    scores.append(1.0 if spy > spy_200ma * 1.05 else (0.5 if spy > spy_200ma else 0.0))

    spy_mom = (spy / _safe_float(prev_mo, "SPY")) - 1 if _safe_float(prev_mo, "SPY") else 0.0
    scores.append(1.0 if spy_mom > 0.02 else (0.5 if spy_mom > 0 else 0.0))

    scores.extend([1.0 if ("TIP" in d.columns and len(d) >= 22 and d["TIP"].iloc[-1] > d["TIP"].iloc[-22]) else 0.0, 0.5, 0.5])

    total_score = min(10.0, sum(scores) + 1.0)
    if total_score >= 10:
        return 95.0
    if total_score >= 8:
        return 85.0
    if total_score >= 6:
        return 65.0
    if total_score >= 4:
        return 40.0
    return 20.0


def _compute_turnover_signal(
    prev_guardrail: str | None,
    prev_bull_signal: float | None,
    guardrail: str,
    bull_signal: float,
) -> float:
    if prev_guardrail is None or prev_bull_signal is None:
        return 0.0
    turnover = abs(bull_signal - prev_bull_signal)
    if prev_guardrail != guardrail:
        turnover += 12.5
    return round(turnover, 1)


def _risk_note(crash_probability: float, bear_12m: float, rwra_probs: dict[str, float]) -> str:
    if rwra_probs["Crisis"] >= 40:
        return "Crash hedges dominate. Maintain a defensive posture."
    if crash_probability >= 55 and bear_12m >= 50:
        return "Elevated downside risk is present even though the allocation engine still sees support."
    if rwra_probs["Bull"] >= 50:
        return "Risk environment remains supportive for higher-beta exposures."
    return "Signals are mixed; position sizing should stay measured."


def compute_combined_snapshot(target_date: dt.date) -> CombinedSnapshot:
    current, actual_ts = _load_master_slice(target_date)
    if current.empty or actual_ts is None or len(current) < 260:
        raise ValueError("Insufficient data to compute Combined Macro + RWRA snapshot.")

    current_dates = current.index
    prev_ts = current_dates[-2] if len(current_dates) > 1 else None
    previous = current.loc[:prev_ts].copy() if prev_ts is not None else pd.DataFrame()

    crash_probability = _market_regime_probability(current)
    bear_12m = _bear_trap_12m(current)
    bull_probability = _bull_trap_probability(current)
    rwra_probs = _compute_rwra_probabilities(current)
    guardrail, guardrail_color = _compute_guardrail(crash_probability, bear_12m, bull_probability, rwra_probs)

    prev_guardrail = None
    prev_bull_signal = None
    if not previous.empty and len(previous) >= 260:
        prev_probs = _compute_rwra_probabilities(previous)
        prev_guardrail, _ = _compute_guardrail(
            _market_regime_probability(previous),
            _bear_trap_12m(previous),
            _bull_trap_probability(previous),
            prev_probs,
        )
        prev_bull_signal = prev_probs["Bull"]

    turnover_signal = _compute_turnover_signal(prev_guardrail, prev_bull_signal, guardrail, rwra_probs["Bull"])
    if turnover_signal >= 12:
        execution_status, execution_color = "REBALANCE", "#f97316"
    else:
        execution_status, execution_color = "HOLD", "#22c55e"

    return CombinedSnapshot(
        requested_date=target_date,
        resolved_date=actual_ts.date(),
        previous_date=prev_ts.date() if prev_ts is not None else None,
        crash_probability=crash_probability,
        macro_guardrail=guardrail,
        guardrail_color=guardrail_color,
        rwra_bull_signal=rwra_probs["Bull"],
        rwra_probabilities=rwra_probs,
        execution_status=execution_status,
        execution_color=execution_color,
        turnover_signal=turnover_signal,
        risk_note=_risk_note(crash_probability, bear_12m, rwra_probs),
    )

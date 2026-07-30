"""
Shared Market Regime scorecard + calibrated crash-probability lookup.

The scorecard originally displayed on pages/1_Market_Regime.py had 9 rules,
two of which (GEX < 0, liquidity < $7.0T) were hardcoded stubs in
utils/data_engine.py that could never fire -- they are dropped here rather
than faked. The remaining 7 rules are unchanged from the original rubric.

Historically the raw 0-90 point total was displayed directly as a
"probability". A 20-year point-in-time backtest showed that's actually a
worse predictor of forward drawdowns than just guessing the unconditional
base rate (Brier 0.141 vs. 0.131) -- the raw score has real ranking
information but is not on a probability scale. `data/market_regime_calibration.json`
(built by backend/calibrate_market_regime.py, refreshed by daily_refresh.py)
holds a walk-forward isotonic mapping from raw score to an empirically
calibrated probability; `calibrated_probability()` below applies it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
CALIBRATION_FILE = ROOT_DIR / "data" / "market_regime_calibration.json"

# (name, weight, predicate over a components dict) -- single source of truth
# shared by the live page (scalar path) and the calibration trainer (vectorized path).
SCORE_RULES = [
    ("SPX below 200DMA",            15, lambda c: c["sp_price"] < c["dma200"]),
    ("HY credit spread > 5%",       20, lambda c: c["hy_spread_pct"] > 5),
    ("MOVE (bond vol) > 100",       15, lambda c: c["move"] > 100),
    ("VIX > 25",                    10, lambda c: c["vix"] > 25),
    ("VIX > VIX3M (backwardation)", 10, lambda c: c["vix"] > c["vix3m"]),
    ("DXY > 105",                   10, lambda c: c["dxy"] > 105),
    ("T2108 breadth < 40%",         10, lambda c: c["t2108"] < 40),
]
MAX_SCORE = sum(weight for _, weight, _ in SCORE_RULES)  # 90


def compute_score(sp_price, dma200, hy_spread_pct, move, vix, vix3m, dxy, t2108) -> dict:
    """Score a single point-in-time snapshot. Returns raw score + per-rule detail."""
    components = dict(sp_price=sp_price, dma200=dma200, hy_spread_pct=hy_spread_pct,
                       move=move, vix=vix, vix3m=vix3m, dxy=dxy, t2108=t2108)
    detail = []
    score = 0
    for name, weight, predicate in SCORE_RULES:
        try:
            fired = bool(predicate(components))
        except (TypeError, ValueError):
            fired = False  # NaN comparisons (e.g. VIX3M pre-2006-07-17) -> not fired
        detail.append({"rule": name, "weight": weight, "fired": fired})
        if fired:
            score += weight
    return {"raw_score": score, "max_score": MAX_SCORE, "detail": detail}


def compute_score_series(df: pd.DataFrame, hy_spread_series: pd.Series, t2108_series: pd.Series) -> pd.Series:
    """Vectorized raw score across full history.

    `df` must already be forward-filled. `hy_spread_series` and `t2108_series`
    are resolved by the caller (see backend/calibrate_market_regime.py) since
    they need data sources (FRED cache, the fixed T2108 ticker universe)
    outside this module's scope.
    """
    sp = df["^GSPC"]
    dma200 = sp.rolling(200).mean()
    vix = df["^VIX"]
    vix3m = df["^VIX3M"]
    dxy = df["DX-Y.NYB"]
    move = df["^MOVE"]
    hy = hy_spread_series.reindex(df.index).ffill()
    t2108 = t2108_series.reindex(df.index)

    score = pd.Series(0.0, index=df.index)
    score += (sp < dma200) * 15
    score += (hy > 5) * 20
    score += (move > 100) * 15
    score += (vix > 25) * 10
    score += (vix > vix3m) * 10
    score += (dxy > 105) * 10
    score += (t2108 < 40) * 10
    return score


def load_calibration() -> dict | None:
    if not CALIBRATION_FILE.exists():
        return None
    try:
        return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def calibrated_probability(raw_score: float, calibration: dict | None) -> float:
    """Map a raw score to the empirically calibrated probability (0-100%).

    Falls back to a naive raw_score/MAX_SCORE*100 reading if no calibration
    cache exists yet (e.g. before the first daily_refresh run) -- clearly
    inferior (see module docstring) but keeps the page functional.
    """
    if not calibration or not calibration.get("lookup"):
        return round(raw_score / MAX_SCORE * 100, 1)
    lookup = {float(k): v for k, v in calibration["lookup"].items()}
    keys = sorted(lookup)
    if raw_score <= keys[0]:
        return round(lookup[keys[0]], 1)
    if raw_score >= keys[-1]:
        return round(lookup[keys[-1]], 1)
    lo = max(k for k in keys if k <= raw_score)
    hi = min(k for k in keys if k >= raw_score)
    if lo == hi:
        return round(lookup[lo], 1)
    frac = (raw_score - lo) / (hi - lo)
    return round(lookup[lo] + frac * (lookup[hi] - lookup[lo]), 1)

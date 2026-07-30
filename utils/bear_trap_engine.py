"""
Shared Bear Trap scorecard + calibrated crash-probability lookup.

The composite on pages/2_Bear_Trap.py had 7 weighted components; two of them
(Valuation, Positioning) were hardcoded constants (0.65, 0.50) that never
varied with real data -- dropped here rather than faked, same treatment as
the GEX/liquidity legs in utils/market_regime_engine.py. The remaining 5
components' weights (0.25/0.20/0.20/0.15/0.10, summing to 0.90) are
renormalized to sum to 1.0 so the score can actually span 0-100%.

The page also displayed three "probabilities" (3M/6M/12M) computed as
raw_score * an arbitrary multiplier (0.60 / 0.85 / 1.0) with no historical
backing. A 20-year backtest showed the resulting "3-month probability",
taken at face value, is statistically no better than guessing the
unconditional base rate (Brier 0.132 vs 0.132). `data/bear_trap_calibration.json`
(built by backend/calibrate_bear_trap.py) holds three separate walk-forward
isotonic calibrations -- one per horizon -- mapping the raw composite score
to an empirically calibrated probability for that horizon.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
CALIBRATION_FILE = ROOT_DIR / "data" / "bear_trap_calibration.json"

# (name, weight, value_fn, normalize_fn) -- single source of truth shared by
# the live page (scalar path) and the calibration trainer (vectorized path).
# Weights are the original rubric's 0.25/0.20/0.20/0.15/0.10 renormalized to
# sum to 1.0 now that the two static legs are gone.
_RAW_WEIGHTS = {"macro": 0.25, "liquidity": 0.20, "credit": 0.20, "breadth": 0.15, "vix": 0.10}
_WEIGHT_SUM = sum(_RAW_WEIGHTS.values())  # 0.90
WEIGHTS = {k: v / _WEIGHT_SUM for k, v in _RAW_WEIGHTS.items()}
MAX_SCORE = 100.0


def _normalize(val, lower, upper, inverted=False):
    frac = (lower - val) / (lower - upper) if inverted else (val - lower) / (upper - lower)
    return np.clip(frac, 0.0, 1.0)


def compute_score(yield_curve, irx, irx_avg120, hyg_ief, hyg_ief_avg252, spy, spy_200ma, vix) -> dict:
    """Score a single point-in-time snapshot. Returns raw score (0-100) + per-component detail."""
    macro = _normalize(yield_curve, 1.5, -0.5, inverted=True)
    liquidity = _normalize(irx, irx_avg120 * 0.8, irx_avg120 * 1.2)
    credit = _normalize(hyg_ief, hyg_ief_avg252 * 1.05, hyg_ief_avg252 * 0.9, inverted=True)
    breadth = _normalize(spy, spy_200ma * 1.05, spy_200ma * 0.95, inverted=True)
    vix_s = _normalize(vix, 15, 35)

    components = {"macro": macro, "liquidity": liquidity, "credit": credit, "breadth": breadth, "vix": vix_s}
    detail = [{"component": name, "weight_pct": round(WEIGHTS[name] * 100, 1), "normalized_value": round(val, 3)}
              for name, val in components.items()]
    raw_score = sum(WEIGHTS[name] * val for name, val in components.items()) * 100
    return {"raw_score": round(raw_score, 2), "max_score": MAX_SCORE, "detail": detail}


def compute_score_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized raw score (0-100) across full history. `df` must already be forward-filled."""
    yield_curve = df["^TNX"] - df["^IRX"]
    macro = _normalize(yield_curve, 1.5, -0.5, inverted=True)

    irx_avg120 = df["^IRX"].rolling(120).mean()
    liquidity = _normalize(df["^IRX"], irx_avg120 * 0.8, irx_avg120 * 1.2)

    hyg_ief = df["HYG"] / df["IEF"]
    hyg_ief_avg252 = hyg_ief.rolling(252).mean()
    credit = _normalize(hyg_ief, hyg_ief_avg252 * 1.05, hyg_ief_avg252 * 0.9, inverted=True)

    spy_200ma = df["SPY"].rolling(200).mean()
    breadth = _normalize(df["SPY"], spy_200ma * 1.05, spy_200ma * 0.95, inverted=True)

    vix_s = _normalize(df["^VIX"], 15, 35)

    return (
        macro * WEIGHTS["macro"] + liquidity * WEIGHTS["liquidity"] + credit * WEIGHTS["credit"]
        + breadth * WEIGHTS["breadth"] + vix_s * WEIGHTS["vix"]
    ) * 100


def load_calibration() -> dict | None:
    if not CALIBRATION_FILE.exists():
        return None
    try:
        return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def calibrated_probability(raw_score: float, calibration: dict | None, horizon: str) -> float:
    """Map a raw score to the empirically calibrated probability (0-100%) for
    a given horizon ("3m", "6m", or "12m"). Falls back to a naive
    raw_score passthrough if no calibration cache exists yet."""
    horizons = (calibration or {}).get("horizons", {})
    lookup_raw = horizons.get(horizon, {}).get("lookup")
    if not lookup_raw:
        return round(max(0.0, min(100.0, raw_score)), 1)
    lookup = {float(k): v for k, v in lookup_raw.items()}
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

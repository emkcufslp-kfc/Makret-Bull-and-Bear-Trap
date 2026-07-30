"""
Shared Bull Trap scorecard + calibrated bull-continuation-probability lookup.

The composite on pages/3_Bull_Trap.py had 8 components; two of them
(Valuation, Insider Buying) were hardcoded constants (0.5 each) that never
varied with real data, and a further +1.0 "baseline" was added on top before
capping at 10 -- same treatment as utils/bear_trap_engine.py, dropped rather
than faked. The remaining 6 real components are equal-weighted (1/6 each,
matching their original 1-point-each design) and renormalized to sum to 100
so the score can actually span 0-100%, instead of maxing out at 8.0/10 (a
ceiling a 22-year backtest showed the old formula never once exceeded).

The page also displayed a single "Bull Market Probability" as one of five
hand-picked anchors (20/40/65/85/95%) keyed to the raw score band, with no
historical backing -- a 22-year reconstruction showed those anchors were
essentially uncorrelated with what the S&P 500 actually did next (corr with
forward 6-month return: -0.01). data/bull_trap_calibration.json (built by
backend/calibrate_bull_trap.py) holds three separate walk-forward isotonic
calibrations -- one per horizon (3/6/12 months) -- mapping the raw composite
score to the empirically observed probability that the S&P 500's total
return over that horizon was positive.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
CALIBRATION_FILE = ROOT_DIR / "data" / "bull_trap_calibration.json"

# Equal weight across the 6 real components -- same relative importance as
# the original 1-point-each design, just renormalized now that the two
# static 0.5 legs and the +1.0 baseline are gone.
WEIGHTS = {
    "yield_curve": 1 / 6, "vix": 1 / 6, "credit": 1 / 6,
    "breadth": 1 / 6, "momentum": 1 / 6, "liquidity": 1 / 6,
}
MAX_SCORE = 100.0


def _tier(cond_full, cond_half):
    """Vectorized-or-scalar 0.0/0.5/1.0 tier, matching the page's original discrete rubric."""
    if isinstance(cond_full, (bool, np.bool_)) or np.isscalar(cond_full):
        if cond_full:
            return 1.0
        if cond_half:
            return 0.5
        return 0.0
    return np.where(cond_full, 1.0, np.where(cond_half, 0.5, 0.0))


def compute_score(curve, prev_curve, vix, vix_mavg, hyg_ief, hyg_ief_mavg,
                   spy, spy_200ma, spy_mom, tip_now, tip_start) -> dict:
    """Score a single point-in-time snapshot. Returns raw score (0-100) + per-component detail."""
    yield_curve = _tier(curve > 0 and prev_curve < 0, curve > 0)
    vix_s = _tier(vix < 15, vix < vix_mavg)
    credit = 1.0 if hyg_ief > hyg_ief_mavg else 0.0
    breadth = _tier(spy > spy_200ma * 1.05, spy > spy_200ma)
    momentum = _tier(spy_mom > 0.02, spy_mom > 0)
    liquidity = 1.0 if tip_now > tip_start else 0.0

    components = {
        "yield_curve": yield_curve, "vix": vix_s, "credit": credit,
        "breadth": breadth, "momentum": momentum, "liquidity": liquidity,
    }
    detail = [{"component": name, "weight_pct": round(WEIGHTS[name] * 100, 1), "tier_value": val}
              for name, val in components.items()]
    raw_score = sum(WEIGHTS[name] * val for name, val in components.items()) * 100
    return {"raw_score": round(raw_score, 2), "max_score": MAX_SCORE, "detail": detail}


def compute_score_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized raw score (0-100) across full history. `df` must already be forward-filled."""
    curve = df["^TNX"] - df["^IRX"]
    prev_curve = curve.shift(22)

    vix_mavg = df["^VIX"].rolling(22).mean()

    hyg_ief = df["HYG"] / df["IEF"]
    hyg_ief_mavg = hyg_ief.rolling(22).mean()

    spy_200ma = df["SPY"].rolling(200).mean()
    spy_mom = df["SPY"] / df["SPY"].shift(22) - 1

    tip_shift = df["TIP"].shift(22)

    yield_curve = _tier((curve > 0) & (prev_curve < 0), curve > 0)
    vix_s = _tier(df["^VIX"] < 15, df["^VIX"] < vix_mavg)
    credit = np.where(hyg_ief > hyg_ief_mavg, 1.0, 0.0)
    breadth = _tier(df["SPY"] > spy_200ma * 1.05, df["SPY"] > spy_200ma)
    momentum = _tier(spy_mom > 0.02, spy_mom > 0)
    liquidity = np.where(df["TIP"] > tip_shift, 1.0, 0.0)

    raw = (
        yield_curve * WEIGHTS["yield_curve"] + vix_s * WEIGHTS["vix"] + credit * WEIGHTS["credit"]
        + breadth * WEIGHTS["breadth"] + momentum * WEIGHTS["momentum"] + liquidity * WEIGHTS["liquidity"]
    ) * 100
    return pd.Series(raw, index=df.index)


def load_calibration() -> dict | None:
    if not CALIBRATION_FILE.exists():
        return None
    try:
        return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def calibrated_probability(raw_score: float, calibration: dict | None, horizon: str) -> float:
    """Map a raw score to the empirically calibrated probability (0-100%) that the
    S&P 500's total return over that horizon ("3m", "6m", or "12m") was positive.
    Falls back to a naive raw_score passthrough if no calibration cache exists yet."""
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

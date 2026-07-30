"""
backend/calibrate_market_regime.py
===================================
Rebuild data/market_regime_calibration.json -- the walk-forward-validated
mapping from the Market Regime scorecard's raw point total (utils/market_regime_engine.py)
to an empirically calibrated probability of a >=10% S&P 500 drawdown within
the next 3 months.

Why: the scorecard's raw 0-90 point sum was previously displayed directly as
a "crash probability". Backtesting showed that's worse than useless as a
probability (Brier score worse than just guessing the historical base rate) --
it has real ranking information, just not on a probability scale, and it
could never reach 100% since two of its original nine legs were hardcoded
stubs. This script fits an isotonic mapping (raw score -> historical hit
rate) instead: real, bounded, and honest about the fact that nothing in 20
years of data ever justified more than roughly a 60-70% reading.

Run standalone:
    python backend/calibrate_market_regime.py

Run as part of the nightly pipeline: see daily_refresh.py Step 7.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_engine import (
    DATA_DIR,
    MASTER_FILE,
    T2108_TICKERS,
    hy_spread_proxy_series,
    refresh_hy_spread_cache,
    HY_SPREAD_CACHE_FILE,
)
from utils.market_regime_engine import CALIBRATION_FILE, MAX_SCORE, compute_score_series

FORWARD_DAYS = 63          # ~3 trading months
CRASH_THRESHOLD_PCT = -10  # >=10% peak-free drawdown within the forward window
MIN_HISTORY_DAYS = 500     # ~2 trading years burn-in before dma200/t2108 are stable
WALKFORWARD_MIN_TRAIN = 500
WALKFORWARD_REFIT_EVERY = 63
THRESHOLD_GRID = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
RAW_SCORE_THRESHOLDS = [0, 10, 20, 30, 40, 50, 60, 70, 80]
HORIZONS = [(21, "1 Month"), (63, "3 Months"), (126, "6 Months")]


def log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _build_hy_spread_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Real FRED history where available (cached by refresh_hy_spread_cache()),
    proxy formula filling any remaining gaps (older dates FRED doesn't cover,
    or if no FRED key is configured at all)."""
    proxy = hy_spread_proxy_series(df)
    if HY_SPREAD_CACHE_FILE.exists():
        try:
            real = pd.read_csv(HY_SPREAD_CACHE_FILE, index_col=0, parse_dates=True)["hy_oas_pct"]
            real = real.reindex(df.index)
            combined = real.combine_first(proxy)
            coverage = real.notna().mean()
            source = f"FRED real ({coverage:.0%} of days), proxy fallback for the rest"
            return combined, source
        except Exception:
            pass
    return proxy, "HYG/BND proxy only (no FRED cache available)"


def _forward_drawdown(sp: pd.Series, days: int) -> pd.Series:
    shifted = sp.shift(-1)
    fwd_min = shifted[::-1].rolling(days, min_periods=1).min()[::-1]
    return (fwd_min - sp) / sp * 100


def _t2108_series(df: pd.DataFrame) -> pd.Series:
    present = [t for t in T2108_TICKERS if t in df.columns]
    sub = df[present]
    ma40 = sub.rolling(40).mean()
    above = (sub > ma40).sum(axis=1)
    return above / len(T2108_TICKERS) * 100


def _brier(pred_pct: pd.Series, label: pd.Series) -> float:
    return float(np.mean((pred_pct / 100 - label) ** 2))


def _walkforward_diagnostics(raw_score: pd.Series, label: pd.Series) -> dict:
    """Out-of-sample-honest quality check: refit isotonic every ~quarter using
    only prior data, exactly like a live deployment would see it. Used only
    for the reported diagnostics -- the deployed lookup table itself is fit
    on the full sample (see main())."""
    from sklearn.isotonic import IsotonicRegression

    n = len(raw_score)
    calib = pd.Series(index=raw_score.index, dtype=float)
    for start in range(WALKFORWARD_MIN_TRAIN, n, WALKFORWARD_REFIT_EVERY):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(raw_score.iloc[:start], label.iloc[:start])
        end = min(start + WALKFORWARD_REFIT_EVERY, n)
        calib.iloc[start:end] = iso.predict(raw_score.iloc[start:end]) * 100

    valid = calib.notna()
    calib_v, label_v = calib[valid], label[valid]
    base_rate = float(label_v.mean())

    grid_rows = []
    for cutoff in THRESHOLD_GRID:
        warn = calib_v >= cutoff
        precision = float(label_v[warn].mean()) if warn.sum() else None
        recall = float(warn[label_v == 1].mean()) if (label_v == 1).sum() else None
        grid_rows.append({"cutoff": cutoff, "n_days_flagged": int(warn.sum()),
                           "precision": precision, "recall": recall})

    return {
        "n_evaluated_days": int(valid.sum()),
        "base_rate_pct": round(base_rate * 100, 2),
        "brier_raw_score_as_pct": round(_brier(raw_score[valid] / MAX_SCORE * 100, label_v), 4),
        "brier_always_base_rate": round(_brier(pd.Series(base_rate * 100, index=label_v.index), label_v), 4),
        "brier_walkforward_calibrated": round(_brier(calib_v, label_v), 4),
        "threshold_grid": grid_rows,
    }


def _threshold_horizon_table(raw_score: pd.Series, fwd_dd_by_horizon: dict[int, pd.Series]) -> list[dict]:
    """Direct answer to "if the score is above X, how likely is a crash within
    Y days" -- a simple historical conditional frequency (not walk-forward;
    this is a descriptive summary of 20 years of history, not a live signal)."""
    rows = []
    for threshold in RAW_SCORE_THRESHOLDS:
        mask = raw_score >= threshold
        for days, label in HORIZONS:
            fwd_dd = fwd_dd_by_horizon[days]
            valid = mask & fwd_dd.notna()
            crash = fwd_dd[valid] <= CRASH_THRESHOLD_PCT
            rows.append({
                "score_threshold": threshold,
                "horizon_days": days,
                "horizon_label": label,
                "n_days_at_or_above": int(valid.sum()),
                "probability_pct": round(float(crash.mean()) * 100, 1) if valid.sum() else None,
            })
    return rows


def _historical_episodes(raw_score: pd.Series, fwd_dd_3m: pd.Series, warning_raw_threshold: int) -> list[dict]:
    """Every real historical date the score crossed into Warning territory,
    grouped into contiguous episodes, with what actually happened next --
    the literal "map the score to a real historical date" view."""
    flagged = raw_score >= warning_raw_threshold
    if not flagged.any():
        return []
    group_id = (flagged != flagged.shift()).cumsum()
    episodes = []
    for gid, mask in flagged.groupby(group_id):
        if not mask.iloc[0]:
            continue
        idx = mask.index
        start, end = idx[0], idx[-1]
        peak_score = int(raw_score.loc[start:end].max())
        dd_at_start = fwd_dd_3m.loc[start]
        episodes.append({
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "trading_days": int(len(idx)),
            "peak_raw_score": peak_score,
            "fwd_3m_drawdown_from_start_pct": round(float(dd_at_start), 1) if pd.notna(dd_at_start) else None,
            "crash_followed": bool(pd.notna(dd_at_start) and dd_at_start <= CRASH_THRESHOLD_PCT),
        })
    return episodes


def _pick_thresholds(grid_rows: list[dict], base_rate_pct: float) -> dict:
    """Elevated = the point the alarm starts getting selective (recall drops
    below ~55%); Warning = the point it's genuinely high-conviction (recall
    drops below ~30%). Recall-based rather than precision-lift-based so the
    picked cutoffs describe "how sparing is this alarm" in a way that's
    stable even when precision is noisy at low sample counts."""
    elevated = warning = None
    for row in grid_rows:
        if row["recall"] is None:
            continue
        if elevated is None and row["recall"] <= 0.55:
            elevated = row["cutoff"]
        if warning is None and row["recall"] <= 0.30:
            warning = row["cutoff"]
    elevated = elevated or THRESHOLD_GRID[0]
    warning = warning or max(elevated, THRESHOLD_GRID[len(THRESHOLD_GRID) // 2])
    return {"elevated": elevated, "warning": warning}


def build_calibration() -> dict:
    log("Refreshing FRED HY spread cache...")
    got_fred = refresh_hy_spread_cache()
    log(f"  {'FRED cache refreshed' if got_fred else 'FRED unavailable, will rely on cache/proxy'}")

    df = pd.read_parquet(MASTER_FILE).ffill()
    hy_series, hy_source = _build_hy_spread_series(df)
    t2108_series = _t2108_series(df)

    raw_score_full = compute_score_series(df, hy_series, t2108_series)
    valid_from = df["^GSPC"].rolling(200).mean().first_valid_index()
    df, raw_score_full = df.loc[valid_from:], raw_score_full.loc[valid_from:]

    fwd_dd_by_horizon = {days: _forward_drawdown(df["^GSPC"], days) for days, _ in HORIZONS}
    fwd_dd_3m = fwd_dd_by_horizon[FORWARD_DAYS]
    label_full = (fwd_dd_3m <= CRASH_THRESHOLD_PCT).astype(int)
    valid = label_full.notna() & raw_score_full.notna()
    raw_score, label = raw_score_full[valid], label_full[valid]
    log(f"Scored {len(raw_score)} trading days ({raw_score.index.min().date()} to {raw_score.index.max().date()})")
    log(f"HY spread source: {hy_source}")

    log("Running walk-forward validation for honest diagnostics...")
    diagnostics = _walkforward_diagnostics(raw_score, label)
    thresholds = _pick_thresholds(diagnostics["threshold_grid"], diagnostics["base_rate_pct"])
    log(f"  Brier: raw-as-pct={diagnostics['brier_raw_score_as_pct']}  "
        f"base-rate-only={diagnostics['brier_always_base_rate']}  "
        f"walk-forward-calibrated={diagnostics['brier_walkforward_calibrated']}")

    log("Fitting production calibration on full history...")
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(raw_score, label)
    grid = np.arange(0, MAX_SCORE + 1, dtype=float)
    lookup = {str(int(g)): round(float(p) * 100, 2) for g, p in zip(grid, iso.predict(grid))}

    log("Building threshold/horizon table and historical episode log...")
    threshold_horizon_table = _threshold_horizon_table(raw_score_full, fwd_dd_by_horizon)
    warning_raw_threshold = next(
        (s for s in sorted(int(k) for k in lookup) if lookup[str(s)] >= thresholds["warning"]),
        MAX_SCORE,
    )
    historical_episodes = _historical_episodes(raw_score_full, fwd_dd_3m, warning_raw_threshold)

    # Per-score-level ground truth: every raw score actually observed in 20 years,
    # how many trading days landed there, and what fraction were actually followed
    # by a crash -- the real historical data the calibrated probability is derived
    # from (displayed as a table on the page, not just the smoothed output).
    grouped = pd.DataFrame({"raw_score": raw_score, "label": label}).groupby("raw_score")
    history_table = [
        {
            "raw_score": int(score_val),
            "n_days": int(g["label"].size),
            "actual_hit_rate_pct": round(float(g["label"].mean()) * 100, 1),
            "calibrated_probability_pct": lookup[str(int(score_val))],
        }
        for score_val, g in grouped
    ]

    calibration = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "method": "isotonic_regression_full_sample",
        "target": f">= {abs(CRASH_THRESHOLD_PCT)}% SPX drawdown within {FORWARD_DAYS} trading days",
        "history_start": str(raw_score.index.min().date()),
        "history_end": str(raw_score.index.max().date()),
        "n_observations": int(len(raw_score)),
        "max_raw_score": MAX_SCORE,
        "max_observed_calibrated_probability": round(max(lookup.values()), 1),
        "hy_spread_source": hy_source,
        "thresholds": thresholds,
        "warning_raw_threshold": warning_raw_threshold,
        "lookup": lookup,
        "history_table": history_table,
        "threshold_horizon_table": threshold_horizon_table,
        "historical_episodes": historical_episodes,
        "walkforward_diagnostics": diagnostics,
    }
    return calibration


def main() -> int:
    calibration = build_calibration()
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2))
    log(f"OK: wrote {CALIBRATION_FILE}  "
        f"(elevated>={calibration['thresholds']['elevated']}%, warning>={calibration['thresholds']['warning']}%, "
        f"ceiling={calibration['max_observed_calibrated_probability']}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

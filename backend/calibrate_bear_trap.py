"""
backend/calibrate_bear_trap.py
===============================
Rebuild data/bear_trap_calibration.json -- walk-forward-validated mappings
from the Bear Trap composite score (utils/bear_trap_engine.py) to empirically
calibrated probabilities of a >=10% S&P 500 drawdown, one calibration per
horizon (3/6/12 months, matching the page's own "Bear Probability (3M/6M/12M)"
cards).

Why: those three cards were previously computed as raw_score * an arbitrary
multiplier (0.60 / 0.85 / 1.0) with zero historical backing, and the raw
score itself had two hardcoded "static" legs (Valuation, Positioning) that
never varied with real data -- meaning the score could never read below
~5.8% or reach 100%, regardless of actual conditions. This script fits
isotonic mappings (raw score -> historical hit rate) per horizon instead:
real, bounded, and honest about what the last 18-20 years actually showed.

Run standalone:
    python backend/calibrate_bear_trap.py

Run as part of the nightly pipeline: see daily_refresh.py Step 8.
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

from utils.data_engine import MASTER_FILE
from utils.bear_trap_engine import CALIBRATION_FILE, MAX_SCORE, WEIGHTS, compute_score_series

CRASH_THRESHOLD_PCT = -10
HORIZONS = [(63, "3m", "3 Months"), (126, "6m", "6 Months"), (252, "12m", "12 Months")]
PRIMARY_HORIZON = "3m"  # drives the main gauge, thresholds, history_table, and episode log
RAW_SCORE_THRESHOLDS = [0, 10, 20, 30, 40, 50, 60, 70, 80]
THRESHOLD_GRID = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
WALKFORWARD_MIN_TRAIN = 500
WALKFORWARD_REFIT_EVERY = 63


def log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _forward_drawdown(sp: pd.Series, days: int) -> pd.Series:
    shifted = sp.shift(-1)
    fwd_min = shifted[::-1].rolling(days, min_periods=1).min()[::-1]
    return (fwd_min - sp) / sp * 100


def _brier(pred_pct: pd.Series, label: pd.Series) -> float:
    return float(np.mean((pred_pct / 100 - label) ** 2))


def _walkforward_diagnostics(raw_score: pd.Series, label: pd.Series) -> dict:
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
        "brier_raw_score_as_pct": round(_brier(raw_score[valid], label_v), 4),
        "brier_always_base_rate": round(_brier(pd.Series(base_rate * 100, index=label_v.index), label_v), 4),
        "brier_walkforward_calibrated": round(_brier(calib_v, label_v), 4),
        "threshold_grid": grid_rows,
    }


def _pick_thresholds(grid_rows: list[dict]) -> dict:
    """Same recall-based landmark picking as calibrate_market_regime.py:
    Elevated = alarm starts getting selective (recall <= 55%); Warning =
    genuinely high-conviction (recall <= 30%)."""
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


def _threshold_horizon_table(raw_score: pd.Series, fwd_dd_by_horizon: dict[str, pd.Series]) -> list[dict]:
    rows = []
    for threshold in RAW_SCORE_THRESHOLDS:
        mask = raw_score >= threshold
        for _, key, label in HORIZONS:
            fwd_dd = fwd_dd_by_horizon[key]
            valid = mask & fwd_dd.notna()
            crash = fwd_dd[valid] <= CRASH_THRESHOLD_PCT
            rows.append({
                "score_threshold": threshold,
                "horizon_key": key,
                "horizon_label": label,
                "n_days_at_or_above": int(valid.sum()),
                "probability_pct": round(float(crash.mean()) * 100, 1) if valid.sum() else None,
            })
    return rows


def _historical_episodes(raw_score: pd.Series, fwd_dd_primary: pd.Series, warning_raw_threshold: float) -> list[dict]:
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
        peak_score = round(float(raw_score.loc[start:end].max()), 1)
        dd_at_start = fwd_dd_primary.loc[start]
        episodes.append({
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "trading_days": int(len(idx)),
            "peak_raw_score": peak_score,
            "fwd_3m_drawdown_from_start_pct": round(float(dd_at_start), 1) if pd.notna(dd_at_start) else None,
            "crash_followed": bool(pd.notna(dd_at_start) and dd_at_start <= CRASH_THRESHOLD_PCT),
        })
    return episodes


def build_calibration() -> dict:
    df = pd.read_parquet(MASTER_FILE).ffill()
    raw_score_full = compute_score_series(df)

    valid_from = pd.concat([
        df["^IRX"].rolling(120).mean(), (df["HYG"] / df["IEF"]).rolling(252).mean(), df["SPY"].rolling(200).mean(),
    ], axis=1).dropna().index.min()
    df, raw_score_full = df.loc[valid_from:], raw_score_full.loc[valid_from:]

    fwd_dd_by_horizon = {key: _forward_drawdown(df["^GSPC"], days) for days, key, _ in HORIZONS}
    log(f"Scored {len(raw_score_full)} trading days ({raw_score_full.index.min().date()} to {raw_score_full.index.max().date()})")
    log(f"raw_score range observed: {raw_score_full.min():.1f} to {raw_score_full.max():.1f} "
        f"(theoretical range with static legs dropped: 0.0 to 100.0)")

    from sklearn.isotonic import IsotonicRegression

    horizons_out = {}
    thresholds = None
    for days, key, label in HORIZONS:
        fwd_dd = fwd_dd_by_horizon[key]
        label_series = (fwd_dd <= CRASH_THRESHOLD_PCT).astype(int)
        valid = label_series.notna() & raw_score_full.notna()
        rs, lbl = raw_score_full[valid], label_series[valid]

        log(f"[{label}] Running walk-forward validation...")
        diagnostics = _walkforward_diagnostics(rs, lbl)
        log(f"  [{label}] Brier: raw={diagnostics['brier_raw_score_as_pct']}  "
            f"base-rate={diagnostics['brier_always_base_rate']}  "
            f"walk-forward-calibrated={diagnostics['brier_walkforward_calibrated']}")

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(rs, lbl)
        grid = np.arange(0, int(MAX_SCORE) + 1, dtype=float)
        lookup = {str(int(g)): round(float(p) * 100, 2) for g, p in zip(grid, iso.predict(grid))}

        horizon_thresholds = _pick_thresholds(diagnostics["threshold_grid"])
        if key == PRIMARY_HORIZON:
            thresholds = horizon_thresholds

        horizons_out[key] = {
            "label": label,
            "forward_days": days,
            "n_observations": int(len(rs)),
            "max_observed_calibrated_probability": round(max(lookup.values()), 1),
            "thresholds": horizon_thresholds,
            "lookup": lookup,
            "walkforward_diagnostics": diagnostics,
        }

    primary_fwd_dd = fwd_dd_by_horizon[PRIMARY_HORIZON]
    primary_label = (primary_fwd_dd <= CRASH_THRESHOLD_PCT).astype(int)
    primary_valid = primary_label.notna() & raw_score_full.notna()
    rs_primary, lbl_primary = raw_score_full[primary_valid], primary_label[primary_valid]
    primary_lookup = horizons_out[PRIMARY_HORIZON]["lookup"]

    log("Building threshold/horizon table and historical episode log...")
    threshold_horizon_table = _threshold_horizon_table(raw_score_full, fwd_dd_by_horizon)
    # Cap the target at the actual observed ceiling -- the walk-forward-derived
    # "warning" cutoff can sit fractionally above the full-sample lookup's max
    # (different data slices), which would otherwise make it unreachable and
    # leave the episode log empty.
    warning_target = min(thresholds["warning"], max(primary_lookup.values()))
    warning_raw_threshold = next(
        s for s in sorted(int(k) for k in primary_lookup) if primary_lookup[str(s)] >= warning_target
    )
    historical_episodes = _historical_episodes(raw_score_full, primary_fwd_dd, warning_raw_threshold)

    # Per-score-bucket ground truth (nearest 5, since this composite is continuous
    # unlike Market Regime's discrete rule-sum) for the primary 3-month horizon.
    bucketed = (rs_primary / 5).round() * 5
    grouped = pd.DataFrame({"bucket": bucketed, "label": lbl_primary}).groupby("bucket")
    history_table = [
        {
            "raw_score_bucket": float(bucket),
            "n_days": int(g["label"].size),
            "actual_hit_rate_pct": round(float(g["label"].mean()) * 100, 1),
            "calibrated_probability_pct": primary_lookup.get(str(int(round(bucket))), None),
        }
        for bucket, g in grouped
    ]

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "method": "isotonic_regression_full_sample_per_horizon",
        "target": f">= {abs(CRASH_THRESHOLD_PCT)}% SPX drawdown",
        "primary_horizon": PRIMARY_HORIZON,
        "history_start": str(raw_score_full.index.min().date()),
        "history_end": str(raw_score_full.index.max().date()),
        "max_raw_score": MAX_SCORE,
        "weights_pct": {k: round(v * 100, 1) for k, v in WEIGHTS.items()},
        "thresholds": thresholds,
        "warning_raw_threshold": warning_raw_threshold,
        "history_table": history_table,
        "threshold_horizon_table": threshold_horizon_table,
        "historical_episodes": historical_episodes,
        "horizons": horizons_out,
    }


def main() -> int:
    calibration = build_calibration()
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2))
    primary = calibration["horizons"][PRIMARY_HORIZON]
    log(f"OK: wrote {CALIBRATION_FILE}  "
        f"(3M elevated>={calibration['thresholds']['elevated']}%, warning>={calibration['thresholds']['warning']}%, "
        f"ceiling={primary['max_observed_calibrated_probability']}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

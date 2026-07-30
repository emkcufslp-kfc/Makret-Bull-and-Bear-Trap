"""
backend/calibrate_bull_trap.py
===============================
Rebuild data/bull_trap_calibration.json -- walk-forward-validated mappings
from the Bull Trap composite score (utils/bull_trap_engine.py) to the
empirically observed probability that the S&P 500's total return over a
given horizon (3/6/12 months, matching the page's "Bull Continuation
Probability (3M/6M/12M)" cards) was actually positive.

Why: the live page previously read the raw score off a 5-band lookup table
(20/40/65/85/95%) picked by hand, with two of the raw score's own 8
components (Valuation, Insider Buying) hardcoded stubs that never varied
with real data -- meaning the score could never exceed 8.0/10 (80%),
regardless of actual conditions, and the "Structural Bull Market" (95%) tier
was mathematically unreachable. A 22-year reconstruction of that exact logic
found the stated probabilities essentially uncorrelated with what markets
actually did next (corr of score vs. forward 6-month return: -0.01) -- the
score's six real components are all momentum/coincident signals, so it
tended to read *most* bullish right at real bull-trap peaks (Jul 2011,
Sep 2018, Feb 2020, Jan 2022, Aug 2022) and *most* bearish right at real
capitulation bottoms (Mar 2009, Feb 2016, Dec 2018, Mar 2020, Oct 2022).

This script fits isotonic mappings (raw score -> historical continuation
rate) per horizon instead: real, bounded, and honest about what the last
20+ years actually showed -- plus an explicit "bull trap episode log":
every historical run where the score reached its highest-conviction bullish
reading, and whether a real >=10% drawdown followed within 3 months anyway.

Run standalone:
    python backend/calibrate_bull_trap.py

Run as part of the nightly pipeline: see daily_refresh.py Step 9.
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
from utils.bull_trap_engine import CALIBRATION_FILE, MAX_SCORE, WEIGHTS, compute_score_series

TRAP_DRAWDOWN_PCT = -10  # what counts as a real "bull trap" failure within 3 months
HORIZONS = [(63, "3m"), (126, "6m"), (252, "12m")]
HORIZON_LABELS = {"3m": "3 Months", "6m": "6 Months", "12m": "12 Months"}
PRIMARY_HORIZON = "6m"  # drives the main gauge, thresholds, history_table -- a "bull market"
                        # regime call is inherently a medium-term claim, not a 3-day one
RAW_SCORE_THRESHOLDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
THRESHOLD_GRID = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
WALKFORWARD_MIN_TRAIN = 500
WALKFORWARD_REFIT_EVERY = 63


def log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _forward_return(sp: pd.Series, days: int) -> pd.Series:
    return sp.shift(-days) / sp - 1


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
        flagged = calib_v >= cutoff
        precision = float(label_v[flagged].mean()) if flagged.sum() else None
        recall = float(flagged[label_v == 1].mean()) if (label_v == 1).sum() else None
        grid_rows.append({"cutoff": cutoff, "n_days_flagged": int(flagged.sum()),
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
    """Same recall-based landmark picking as calibrate_bear_trap.py, just relabeled
    for a bullish-conviction scale: 'confirmed' = the score starts being genuinely
    selective (recall <= 55%); 'strong' = high-conviction (recall <= 30%)."""
    confirmed = strong = None
    for row in grid_rows:
        if row["recall"] is None:
            continue
        if confirmed is None and row["recall"] <= 0.55:
            confirmed = row["cutoff"]
        if strong is None and row["recall"] <= 0.30:
            strong = row["cutoff"]
    confirmed = confirmed or THRESHOLD_GRID[0]
    strong = strong or max(confirmed, THRESHOLD_GRID[len(THRESHOLD_GRID) // 2])
    return {"confirmed": confirmed, "strong": strong}


def _threshold_horizon_table(raw_score: pd.Series, fwd_ret_by_horizon: dict[str, pd.Series]) -> list[dict]:
    rows = []
    for threshold in RAW_SCORE_THRESHOLDS:
        mask = raw_score >= threshold
        for _, key in HORIZONS:
            fwd_ret = fwd_ret_by_horizon[key]
            valid = mask & fwd_ret.notna()
            positive = fwd_ret[valid] > 0
            rows.append({
                "score_threshold": threshold,
                "horizon_key": key,
                "horizon_label": HORIZON_LABELS[key],
                "n_days_at_or_above": int(valid.sum()),
                "probability_pct": round(float(positive.mean()) * 100, 1) if valid.sum() else None,
            })
    return rows


def _bull_trap_episodes(raw_score: pd.Series, fwd_dd_3m: pd.Series, strong_threshold: float) -> list[dict]:
    """Every contiguous run where the score reached its highest-conviction bullish
    reading, and whether a real >=10% drawdown actually followed within 3 months --
    the direct historical answer to 'how often was a confident bullish call a trap.'"""
    flagged = raw_score >= strong_threshold
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
        dd_at_start = fwd_dd_3m.loc[start]
        episodes.append({
            "start_date": str(start.date()),
            "end_date": str(end.date()),
            "trading_days": int(len(idx)),
            "peak_raw_score": peak_score,
            "fwd_3m_drawdown_from_start_pct": round(float(dd_at_start), 1) if pd.notna(dd_at_start) else None,
            "trap_followed": bool(pd.notna(dd_at_start) and dd_at_start <= TRAP_DRAWDOWN_PCT),
        })
    return episodes


def build_calibration() -> dict:
    df = pd.read_parquet(MASTER_FILE).ffill()
    raw_score_full = compute_score_series(df)

    valid_from = pd.concat([
        df["^VIX"].rolling(22).mean(), (df["HYG"] / df["IEF"]).rolling(22).mean(),
        df["SPY"].rolling(200).mean(),
    ], axis=1).dropna().index.min()
    df, raw_score_full = df.loc[valid_from:], raw_score_full.loc[valid_from:]

    fwd_ret_by_horizon = {key: _forward_return(df["^GSPC"], days) for days, key in HORIZONS}
    fwd_dd_3m = _forward_drawdown(df["^GSPC"], 63)
    log(f"Scored {len(raw_score_full)} trading days ({raw_score_full.index.min().date()} to {raw_score_full.index.max().date()})")
    log(f"raw_score range observed: {raw_score_full.min():.1f} to {raw_score_full.max():.1f} "
        f"(theoretical range with static legs + baseline dropped: 0.0 to 100.0)")

    from sklearn.isotonic import IsotonicRegression

    horizons_out = {}
    thresholds = None
    for days, key in HORIZONS:
        label_name = HORIZON_LABELS[key]
        fwd_ret = fwd_ret_by_horizon[key]
        label_series = (fwd_ret > 0).astype(int)
        valid = label_series.notna() & raw_score_full.notna() & fwd_ret.notna()
        rs, lbl = raw_score_full[valid], label_series[valid]

        log(f"[{label_name}] Running walk-forward validation...")
        diagnostics = _walkforward_diagnostics(rs, lbl)
        log(f"  [{label_name}] Brier: raw={diagnostics['brier_raw_score_as_pct']}  "
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
            "label": label_name,
            "forward_days": days,
            "n_observations": int(len(rs)),
            "base_rate_pct": diagnostics["base_rate_pct"],
            "min_observed_calibrated_probability": round(min(lookup.values()), 1),
            "max_observed_calibrated_probability": round(max(lookup.values()), 1),
            "thresholds": horizon_thresholds,
            "lookup": lookup,
            "walkforward_diagnostics": diagnostics,
        }

    primary_fwd_ret = fwd_ret_by_horizon[PRIMARY_HORIZON]
    primary_label = (primary_fwd_ret > 0).astype(int)
    primary_valid = primary_label.notna() & raw_score_full.notna() & primary_fwd_ret.notna()
    rs_primary, lbl_primary = raw_score_full[primary_valid], primary_label[primary_valid]
    primary_lookup = horizons_out[PRIMARY_HORIZON]["lookup"]

    log("Building threshold/horizon table and bull-trap episode log...")
    threshold_horizon_table = _threshold_horizon_table(raw_score_full, fwd_ret_by_horizon)
    strong_target = min(thresholds["strong"], max(primary_lookup.values()))
    strong_raw_threshold = next(
        s for s in sorted(int(k) for k in primary_lookup) if primary_lookup[str(s)] >= strong_target
    )
    bull_trap_episodes = _bull_trap_episodes(raw_score_full, fwd_dd_3m, strong_raw_threshold)

    # Per-score-bucket ground truth (nearest 5) for the primary 6-month horizon.
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

    n_traps = sum(e["trap_followed"] for e in bull_trap_episodes)
    n_episodes = len(bull_trap_episodes)
    base_trap_rate_pct = round(float((fwd_dd_3m <= TRAP_DRAWDOWN_PCT).mean()) * 100, 1)

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "method": "isotonic_regression_full_sample_per_horizon",
        "target": "S&P 500 total return over horizon is positive",
        "trap_target": f"<= {TRAP_DRAWDOWN_PCT}% SPX drawdown within 3 months of a high-conviction reading",
        "primary_horizon": PRIMARY_HORIZON,
        "history_start": str(raw_score_full.index.min().date()),
        "history_end": str(raw_score_full.index.max().date()),
        "max_raw_score": MAX_SCORE,
        "weights_pct": {k: round(v * 100, 1) for k, v in WEIGHTS.items()},
        "thresholds": thresholds,
        "strong_raw_threshold": strong_raw_threshold,
        "history_table": history_table,
        "threshold_horizon_table": threshold_horizon_table,
        "bull_trap_episodes": bull_trap_episodes,
        "bull_trap_episode_summary": {
            "n_episodes": n_episodes,
            "n_traps": n_traps,
            "trap_rate_pct": round(n_traps / n_episodes * 100, 1) if n_episodes else None,
            "unconditional_base_trap_rate_pct": base_trap_rate_pct,
        },
        "horizons": horizons_out,
    }


def main() -> int:
    calibration = build_calibration()
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2))
    primary = calibration["horizons"][PRIMARY_HORIZON]
    trap_summary = calibration["bull_trap_episode_summary"]
    log(f"OK: wrote {CALIBRATION_FILE}  "
        f"(6M confirmed>={calibration['thresholds']['confirmed']}%, strong>={calibration['thresholds']['strong']}%, "
        f"ceiling={primary['max_observed_calibrated_probability']}%, floor={primary['min_observed_calibrated_probability']}%)")
    log(f"Bull-trap episode log: {trap_summary['n_traps']}/{trap_summary['n_episodes']} "
        f"high-conviction episodes were actually followed by a >=10% drawdown within 3 months "
        f"({trap_summary['trap_rate_pct']}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

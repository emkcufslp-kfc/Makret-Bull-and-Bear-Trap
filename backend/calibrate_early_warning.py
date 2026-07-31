"""
backend/calibrate_early_warning.py
====================================
Rebuild data/early_warning_calibration.json -- the Lead/Lift stats and
episode table shown on pages/14_Early_Warning.py.

Why: the three "3-Stage Alert Pipeline" chips displayed hardcoded strings
("15-20w" / "0.62x", "10-12w" / "2.23x", "6-8w" / "~34% drop prob") that were
never recomputed from data, using two different and undocumented masking
conventions between chips (Stage 1's number matched an "exactly Stage 1"
read, Stage 2's matched an "Stage 2 or higher" read). A 35-year audit found
Stage 2's figure still reproduced exactly, but Stage 3's had drifted from
~34% to 29.9% as new episodes (2022, 2023, 2025, 2026) were added. This
script recomputes all three under one documented convention (alert_stage >=
N) against one canonical outcome (>=10% S&P 500 drawdown within 13 weeks),
plus a corrected episode table.

The page's own "Historical Early Warning Episodes" table also measured
outcome as "S&P 500 change exactly 13 weeks after the episode started" --
which mislabels episodes that fired near a trough (2020-03-19, right at the
COVID bottom, reads "Rally" because SPY was +43% by week 13, even though the
alert correctly caught a crash already underway). This script replaces that
with the worst drawdown reached at any point during the episode plus the
following 13 weeks -- the actual question a warning system should answer.

Run standalone:
    python backend/calibrate_early_warning.py

Run as part of the nightly pipeline: see daily_refresh.py Step 10.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WEEKLY_PATH = ROOT / "exports" / "crash_predictor_study" / "weekly_feature_outcomes.csv"
CALIBRATION_FILE = ROOT / "data" / "early_warning_calibration.json"

CANONICAL_OUTCOME = "drop_10_13w"
CANONICAL_LABEL = ">=10% S&P 500 drawdown within 13 weeks"
EPISODE_FORWARD_WEEKS = 13  # how far past episode end to keep watching for the trough
DROP_LABEL_THRESHOLD = -10.0
CONTAINED_LABEL_THRESHOLD = -5.0
STAGE_LABELS = {1: "Monitor", 2: "Actionable", 3: "High Conviction"}


def log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_weekly() -> pd.DataFrame:
    df = pd.read_csv(WEEKLY_PATH, index_col=0, parse_dates=True).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["d1_norm"] = (df["d1_market_regime_score"] / 85 * 100).clip(0, 100).round(1)
    df["d2_norm"] = ((df["d2_score"] - 2) / 12 * 100).clip(0, 100).round(1)
    df["d3_norm"] = df["liquidity_score"].astype(float)
    df["credit_norm"] = ((df["credit_spread_for_model"] - 1.0) / 5.0 * 100).clip(0, 100).round(1)
    streak, cnt = [], 0
    for v in df["d1_market_regime_score"]:
        cnt = cnt + 1 if v > 25 else 0
        streak.append(cnt)
    df["d1_streak"] = streak

    def _stage(r):
        if r["d1_streak"] >= 8 and (r["d2_norm"] >= 50 or r["credit_norm"] >= 50):
            return 3
        if r["d1_streak"] >= 4:
            return 2
        if r["d3_norm"] >= 65 or r["credit_norm"] >= 50:
            return 1
        return 0

    df["alert_stage"] = df.apply(_stage, axis=1)
    return df


def _episodes(df: pd.DataFrame, min_stage: int) -> list[tuple]:
    """Contiguous runs where alert_stage >= min_stage. Returns (start, start_i, end, end_i, peak_stage)."""
    active = df["alert_stage"] >= min_stage
    rows, in_ep = [], False
    ep_start = ep_start_i = ep_max = None
    for i, (date, is_active) in enumerate(zip(df.index, active)):
        if is_active:
            if not in_ep:
                in_ep, ep_start, ep_start_i, ep_max = True, date, i, int(df["alert_stage"].iloc[i])
            else:
                ep_max = max(ep_max, int(df["alert_stage"].iloc[i]))
        else:
            if in_ep:
                rows.append((ep_start, ep_start_i, df.index[i - 1], i - 1, ep_max))
                in_ep = False
    if in_ep:
        rows.append((ep_start, ep_start_i, df.index[-1], len(df) - 1, ep_max))
    return rows


def _stage_lift(df: pd.DataFrame, min_stage: int) -> dict:
    mask = df["alert_stage"] >= min_stage
    valid = df.dropna(subset=[CANONICAL_OUTCOME])
    base_rate = float(valid[CANONICAL_OUTCOME].mean())
    sub = valid[mask.reindex(valid.index).fillna(False)]
    hit_rate = float(sub[CANONICAL_OUTCOME].mean()) if len(sub) else float("nan")
    lift = hit_rate / base_rate if base_rate else float("nan")
    return {
        "n_weeks_active": int(mask.sum()),
        "n_scored_weeks": int(len(sub)),
        "base_rate_pct": round(base_rate * 100, 1),
        "hit_rate_pct": round(hit_rate * 100, 1),
        "lift": round(lift, 2),
    }


def _stage_lead_and_episodes(df: pd.DataFrame, min_stage: int) -> dict:
    sp = df["sp500"].to_numpy()
    n = len(df)
    eps = _episodes(df, min_stage)
    leads, worst_dds = [], []
    for start, si, end, ei, peak in eps:
        win_end = min(n - 1, ei + EPISODE_FORWARD_WEEKS)
        window = sp[si : win_end + 1]
        if len(window) < 2:
            continue
        trough_rel = int(np.argmin(window))
        leads.append(trough_rel)
        worst_dds.append(float(window.min() / sp[si] - 1) * 100)
    pct_confirmed_drop = float(np.mean([1.0 if d <= DROP_LABEL_THRESHOLD else 0.0 for d in worst_dds])) * 100 if worst_dds else None
    return {
        "n_episodes": len(eps),
        "median_lead_weeks_to_trough": round(float(np.median(leads)), 1) if leads else None,
        "median_worst_drawdown_pct": round(float(np.median(worst_dds)), 1) if worst_dds else None,
        "pct_episodes_confirmed_drop": round(pct_confirmed_drop, 1) if pct_confirmed_drop is not None else None,
    }


def _episode_table(df: pd.DataFrame) -> list[dict]:
    """The page's displayed table: every Stage 2+ episode, with both the old
    (misleading) 13-weeks-after-start metric and the corrected worst-drawdown
    metric, so the fix is visible rather than silently swapped."""
    sp = df["sp500"].to_numpy()
    d1 = df["d1_norm"].to_numpy()
    d2 = df["d2_norm"].to_numpy()
    n = len(df)
    rows = []
    for start, si, end, ei, peak in _episodes(df, min_stage=2):
        entry = sp[si]
        d1_peak = float(np.nanmax(d1[si : ei + 1])) if ei >= si else float(d1[si])
        d2_window = d2[si : ei + 1]
        d2_peak = float(np.nanmax(d2_window)) if np.isfinite(d2_window).any() else 0.0
        fi = df.index.get_indexer([end + pd.Timedelta(weeks=EPISODE_FORWARD_WEEKS)], method="nearest")[0]
        spy_chg_13w_from_start = (sp[fi] / entry - 1) * 100 if 0 <= fi < n else None

        win_end = min(n - 1, ei + EPISODE_FORWARD_WEEKS)
        window = sp[si : win_end + 1]
        worst_dd = float(window.min() / entry - 1) * 100

        if worst_dd <= DROP_LABEL_THRESHOLD:
            corrected_outcome = "Confirmed Drop"
        elif worst_dd <= CONTAINED_LABEL_THRESHOLD:
            corrected_outcome = "Contained"
        else:
            corrected_outcome = "No Confirmed Drop"

        old_outcome = (
            "Drop" if (spy_chg_13w_from_start is not None and spy_chg_13w_from_start < -5)
            else "Flat" if (spy_chg_13w_from_start is not None and spy_chg_13w_from_start < 5)
            else "Rally"
        ) if ei < n - 1 or win_end < n - 1 else "Active"

        rows.append({
            "start_date": str(start.date()),
            "end_date": str(end.date()) if ei < n - 1 else "ONGOING",
            "peak_stage": peak,
            "d1_peak": round(d1_peak, 0),
            "d2_peak": round(d2_peak, 0),
            "spy_entry": round(float(entry), 0),
            "spy_chg_13w_from_start_pct": round(spy_chg_13w_from_start, 1) if spy_chg_13w_from_start is not None else None,
            "old_outcome_label": old_outcome if ei < n - 1 else "Active",
            "worst_drawdown_during_episode_pct": round(worst_dd, 1),
            "corrected_outcome_label": corrected_outcome if ei < n - 1 else "Active",
        })
    return rows


def build_calibration() -> dict:
    df = load_weekly()
    stages_out = {}
    for stage in (1, 2, 3):
        lift_stats = _stage_lift(df, stage)
        lead_stats = _stage_lead_and_episodes(df, stage)
        stages_out[str(stage)] = {"label": STAGE_LABELS[stage], **lift_stats, **lead_stats}

    episodes = _episode_table(df)
    n_confirmed = sum(1 for e in episodes if e["corrected_outcome_label"] == "Confirmed Drop")
    n_contained = sum(1 for e in episodes if e["corrected_outcome_label"] == "Contained")
    n_no_drop = sum(1 for e in episodes if e["corrected_outcome_label"] == "No Confirmed Drop")

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "mask_convention": "inclusive: alert_stage >= N (documented, applied consistently to all three chips)",
        "canonical_outcome": CANONICAL_OUTCOME,
        "canonical_outcome_label": CANONICAL_LABEL,
        "history_start": str(df.index.min().date()),
        "history_end": str(df.index.max().date()),
        "n_weeks": int(len(df)),
        "stages": stages_out,
        "episodes": episodes,
        "episode_summary": {
            "n_episodes": len(episodes),
            "n_confirmed_drop": n_confirmed,
            "n_contained": n_contained,
            "n_no_confirmed_drop": n_no_drop,
        },
    }


def main() -> int:
    calibration = build_calibration()
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(calibration, indent=2))
    s1, s2, s3 = calibration["stages"]["1"], calibration["stages"]["2"], calibration["stages"]["3"]
    log(f"OK: wrote {CALIBRATION_FILE}")
    log(f"  Stage 1 lift={s1['lift']}x  Stage 2 lift={s2['lift']}x  Stage 3 lift={s3['lift']}x "
        f"(canonical: {CANONICAL_LABEL})")
    es = calibration["episode_summary"]
    log(f"  Episodes: {es['n_episodes']} total, {es['n_confirmed_drop']} confirmed drop, "
        f"{es['n_contained']} contained, {es['n_no_confirmed_drop']} no confirmed drop")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Working notes — crash-probability calibration (Market Regime + Bear Trap)

Handoff doc for starting a fresh session. Delete or trim once everything below is resolved.

## Status

**Market Regime work: DONE and PUSHED.** Commit `3adbff77` ("Update dashboards and strategies")
is on `origin/main`. The deployed Streamlit Cloud app should reflect it within a couple minutes
of the push; if it still shows the old page (a `流動性縮減` liquidity row, static "35%" Polymarket
value, gauge titled just "Crash Probability Gauge" without "(calibrated)"), check the Streamlit
Cloud dashboard for a stuck/failed deploy rather than assuming the push didn't work —
`git log origin/main --oneline -1` confirms whether a given push landed.

**Bear Trap work: DONE, NOT YET COMMITTED/PUSHED.** Same treatment applied to
`pages/2_🐻_Bear_Trap.py` in a later session (see below) — still sitting as local uncommitted
changes as of this note. Run `safe_commit_push.bat` to ship it.

## What changed — Market Regime (pages/1_🔴_Market_Regime.py)

The Market Regime page (`pages/1_🔴_Market_Regime.py`) displayed a raw 0-100 point scorecard
as a "crash probability." A 20-year backtest showed that's worse than useless as a probability
(Brier score worse than just guessing the historical base rate), partly because two of the
original nine rules (GEX, liquidity) were hardcoded stubs that could never fire.

Built instead:
- **`utils/market_regime_engine.py`** — single source of truth for the 7 real scoring rules
  (GEX/liquidity dropped) + calibration lookup/interpolation.
- **`backend/calibrate_market_regime.py`** — walk-forward isotonic calibration trainer. Writes
  `data/market_regime_calibration.json`: raw-score → calibrated-probability lookup, data-derived
  Elevated/Warning thresholds, a per-score historical hit-rate table, a threshold × horizon
  ("if score ≥ X, odds of a crash within 1/3/6 months") table, and a log of every real historical
  date the score reached Warning level with the actual outcome. Runs in ~1-2s.
- **`utils/data_engine.py`** — `get_hy_spread()` now tries live FRED → cached FRED history
  (`data/hy_spread_fred_cache.csv`) → HYG/BND proxy, in that order. Benefits every caller, not
  just this page.
- **`daily_refresh.py`** — new Step 7 refreshes the FRED cache and rebuilds the calibration JSON
  daily. Also fixed a pre-existing Windows cp950-console crash (✓/✗ characters) hit while testing.
- **`pages/1_🔴_Market_Regime.py`** — shows the calibrated probability instead of the raw score;
  fixed Richmond Fed SOS / Polymarket rows to call the real (previously unused) data functions
  instead of hardcoded strings; dropped the fake liquidity row; added the rule-breakdown expander
  and the three new historical-mapping tables below the gauge.
- **`.streamlit/secrets.toml`** — `FRED_API_KEY` uncommented (gitignored, local only).
- **`.gitignore`** — added `!data/market_regime_calibration.json` (was being caught by the
  blanket `*.json` rule, same pattern as `r3_signal_cache.json`).
- **`backend/verify_daily_update.py`** — added an existence/sanity check for the calibration file.

Verified live in a local browser test (port 8510 — 8501 was occupied by an unrelated project,
"TQVS v1 Backtest", not touched).

**Known limitation**: this sandbox's network only reaches ~2.9 years of FRED history, so the
locally-generated calibration is trained on mostly-proxy HY spread data for older years. GitHub
Actions has `FRED_API_KEY` wired in with full internet access — confirm that secret is actually
set on the repo, then either wait for the next scheduled `update_data.yml` run or trigger it
manually (`workflow_dispatch`) to rebuild the calibration on the true 20-year FRED history.

## What changed — Bear Trap (pages/2_🐻_Bear_Trap.py)

Same underlying problem, arguably worse: the composite had 7 weighted components, two of which
(Valuation=0.65, Positioning=0.50, worth 10% combined) were hardcoded constants that never varied
with real data — meaning the raw score could never read below ~5.8% or reach 100%, regardless of
actual conditions. The three displayed "probabilities" (3M/6M/12M) were `raw_score * an arbitrary
multiplier` (0.60/0.85/1.0) with zero historical backing. A backtest showed the resulting 3-month
"probability", taken at face value, is statistically no better than the base rate (Brier 0.132 vs
0.132) — and even *after* walk-forward isotonic calibration, it's still slightly worse (0.1333 vs
0.1071 base rate). This composite has not demonstrated real predictive skill out-of-sample; the
page now says so explicitly via a warning banner, rather than overselling the fix.

Built:
- **`utils/bear_trap_engine.py`** — 5 real components only (static legs dropped), weights
  renormalized 0.25/0.20/0.20/0.15/0.10 → sum to 1.0 (was 0.90) + calibration lookup/interpolation
  parameterized by horizon (3m/6m/12m).
- **`backend/calibrate_bear_trap.py`** — mirrors `calibrate_market_regime.py`: THREE separate
  walk-forward isotonic calibrations (one per horizon, since the page shows 3 separate cards),
  history_table (score bucketed to nearest 5), threshold × horizon table, historical episode log.
  Writes `data/bear_trap_calibration.json`. One gotcha hit and fixed: the walk-forward-derived
  "warning" cutoff can sit fractionally above the full-sample lookup's actual ceiling (different
  data slices) — left unhandled, this makes the threshold unreachable and the episode log empty.
  Fixed by capping the inversion target at the observed max before searching for a matching score.
- **`daily_refresh.py`** — new Step 8.
- **`pages/2_🐻_Bear_Trap.py`** — calibrated probabilities per horizon instead of arbitrary
  multipliers; Indicator Breakdown table drops the two fake rows (with an explanatory caption,
  since they used to display literal "Elevated"/"Neutral" labels every single day); same three
  historical-mapping tables as Market Regime; explicit out-of-sample honesty warning on the gauge.
- `.gitignore` (`!data/bear_trap_calibration.json`) and `verify_daily_update.py` check, same
  pattern as Market Regime.

Verified live in a local browser test, same method as Market Regime.

## Not done — flagged as separate tasks, IN PROGRESS

**Market Regime centralization** (background session, branch `claude/great-tesla-db07f2`,
worktree at `.claude/worktrees/great-tesla-db07f2/`): the Market Regime scoring formula was found
duplicated (three different, inconsistent max-score normalizations — /100, /85, and now the
corrected /90) in `utils/model_change_monitor.py:63`, `pages/0_📋_Market_Summary.py:66`,
`backend/strategies/combined_macro_rwra.py:57`, `pages/14_📡_Early_Warning.py:40`. As of this
note, that worktree has made no commits yet (still at `7ad5738e`). `combined_macro_rwra.py` feeds
a guardrail state adjacent to `data/live_trading/` — re-deriving its thresholds needs its own
backtest, not a blind swap.

**R3's stale Bear Trap formula copy** (task_2473967a, not yet started): `daily_refresh.py`'s
`step5_r3_signal_cache()` has its own inline copy of the OLD Bear Trap formula (comment says
"exact logic from pages/2"), including the two static legs just removed from the real page. It
will now silently diverge from the corrected formula. R3's `bear_score` feeds
`net_confidence = bull_weight * (1 - bear_score)` in `data/r3_signal_cache.json`, adjacent to a
live paper-trading ledger — do not switch it to the new formula without backtesting R3's
deploy_pct/regime classification under both versions first.

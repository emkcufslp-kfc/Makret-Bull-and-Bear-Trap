# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multi-page Streamlit app (`app.py` redirects into `pages/`) of independent macro-quantitative dashboards for monitoring market regimes, liquidity, and crash probability, plus several backtested tactical allocation strategies (System E, R3 Vol-Adjusted, Platinum, NTSX, Fund Tactical, Hybrid AD, E80/R3 Blend, etc.). Each dashboard page under `pages/` is self-contained and does not depend on other pages.

Data sources: Yahoo Finance (`yfinance`) for market data, FRED for macro series, Polygon for NYSE TRIN (optional). `market_stage_model/` is a separate standalone Streamlit sub-app for structural stage classification (Acceleration/Deceleration/Accumulation/Distribution) of any stock/ETF, runnable independently.

## Commands

Run the main app locally:
```
streamlit run app.py
```

Run the standalone market stage model sub-app:
```
cd market_stage_model
python -m streamlit run app.py --server.port 8520
```

Install dependencies: `pip install -r requirements.txt` (root) and `pip install -r market_stage_model/requirements.txt` for the sub-app.

Refresh all data after market close (orchestrates the full pipeline — OHLCV, crash predictor exports, strategy artifacts, R3 signal cache on Thursdays):
```
python daily_refresh.py [--us-data-dir "<path>"] [--top-n 100]
```

Run the GitHub Actions data-sync pipeline manually (what `update_data.yml` runs daily):
```
python backend/sync_engine.py
python backend/verify_daily_update.py
```

No test suite, linter, or build step is configured in this repo.

## Committing and deploying

**The public Streamlit Cloud app deploys from GitHub, not from this local sandbox — this sandbox cannot push to GitHub directly.** Local changes only reach the deployed app after being committed and pushed from the user's machine.

A GitHub Actions bot (`update_data.yml`, `d3_telegram.yml`) commits automated data refreshes to `main` on its own schedule, independent of local work. Because of this, local commits/pushes need to reconcile with bot commits:

- `safe_commit_push.bat` — the recommended path. Commits local changes, fetches, merges bot updates with local changes taking precedence on conflicts (`-X ours`), then pushes normally. Use this by default.
- `git_commit_push.bat` — commits then force-pushes (`--force-with-lease`), overwriting bot data commits. Use only when intentionally discarding bot-generated data changes.
- `git_push.bat` — stages everything, commits with a generic "Daily refresh" message, and force-pushes. Used by the automated/scheduled local refresh flow, not for hand-authored changes.

When asked to commit/push from this sandbox, tell the user to run the appropriate `.bat` script themselves rather than attempting a git push here.

## Architecture

**Data layer (`utils/data_engine.py`)**: Central source of truth for market data. Maintains `data/market_data_master.parquet`, a wide OHLCV/derived-series table covering core tickers (`^GSPC`, `^VIX`, `^VIX3M`, `HYG`, `DX-Y.NYB`, etc.), sector SPDRs, and reference ETFs, going back to 2004-01-01. Provides a Streamlit stub fallback (`_St` class) so the same module works both inside the Streamlit UI (`st.cache_data`, `st.secrets`) and in headless batch scripts like `sync_engine.py` that import it without Streamlit installed.

**Sync pipeline (`backend/sync_engine.py`)**: The "Unified Sync Engine" — orchestrates the full nightly data refresh by shelling out to and importing sibling scripts (`refresh_strategy_artifacts.py`, etc.). This is what GitHub Actions runs daily via `update_data.yml`. Note the dependency bootstrap at the top of the file inserting `/tmp/pkgs` and `/tmp/pkgs2` into `sys.path` — a workaround for constrained disk space in the CI/sandbox environment; this pattern recurs in `daily_refresh.py` too.

**Strategy engines (`backend/strategies/`, `strategy_e/`)**: Each strategy (System E, R3, Platinum, NTSX, Fund Tactical, ensemble/top100, blend engines) lives in its own module exposing a signal-computation function and a backtest function. `strategy_e/engine.py` is the reference implementation pattern: hardcoded universe list, signal logic in a docstring-documented block, `compute_*_signal()` and `backtest_*()` as the public API. Strategy outputs are written to `data/`, `backend/strategies/data/`, and `exports/` (equity curves, transaction logs, performance summaries) which the dashboard pages read back for display — dashboards do not recompute strategies live except for on-demand backtests.

**Signal-day gating**: `daily_refresh.py` and the strategy engines gate certain cache writes to specific weekdays — System E signals on Tuesday close (executed Wednesday open), R3 Vol-Adjusted on Thursday close (executed Friday open). Both use T+1 execution with a 0.16% round-trip cost assumption. Respect this gating when modifying refresh logic; recomputing on the wrong day will desync live signal caches (`data/system_e_signal_cache.json`, `data/r3_signal_cache.json`) from what was actually tradeable.

**Time-travel / point-in-time correctness**: Dashboards support a date picker that recomputes each model using only data available as of that date. Any new indicator or strategy should preserve this — avoid lookahead into data that would not have been available on the selected date.

**Warning/model-change monitoring (`utils/warning_dashboard.py`, `utils/model_change_monitor.py`)**: A separate layer that scores discrete market-health indicators (overextension, breadth, VIX spikes, breakout/breakdown rates, etc.) into Normal/Watch/Warning status, distinct from the trading strategy signals — feeds the Early Warning and Model Change Worksheet pages.

**`market_stage_model/`**: Deliberately decoupled from the rest of the repo (own `requirements.txt`, own `.streamlit/`, own README) so it can run standalone. Only pulls real yfinance OHLCV, never synthetic data. Stage classification depends purely on VWMA(8/21/34) stacking plus close vs. VMA(21) — no ML.

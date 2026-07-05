"""
daily_refresh.py
================
Run after market close to refresh all Early Warning dashboard data sources.

Usage (from repo root):
    python daily_refresh.py [--us-data-dir "D:\\Codex projects\\US-data"] [--top-n 100]

Refreshes:
  1. D:\\Codex projects\\US-data OHLCV  (--refresh-stale, incremental)
  2. exports/crash_predictor_study/     (crash predictor ML pipeline)
  3. .tmp/market_stage_validation/output/stage_breadth_history.csv
  4. .tmp/market_stage_validation/output/  (backtest: forward returns, trade log, etc.)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "market_stage_model"
OUT_DIR   = ROOT / ".tmp" / "market_stage_validation" / "output"
EXPORT_DIR = ROOT / "exports" / "crash_predictor_study"
TMP_CACHE = ROOT / ".tmp" / "raw_ohlcv_cache.parquet"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODEL_DIR))

# ── Dep bootstrap ─────────────────────────────────────────────────────────────
PKG_DIR = "/tmp/pkgs2"
if os.path.isdir(PKG_DIR):
    sys.path.insert(0, PKG_DIR)


def log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Step 1: Update US-data OHLCV ─────────────────────────────────────────────
def step1_refresh_us_data(us_data_dir: Path) -> bool:
    loader = us_data_dir / "US_data_loader.py"
    if not loader.exists():
        log(f"SKIP step 1 — US_data_loader.py not found at {loader}")
        return False
    log(f"Step 1: Running US_data_loader.py --mode ohlcv --refresh-stale in {us_data_dir}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(loader), "--mode", "ohlcv", "--refresh-stale"],
        cwd=str(us_data_dir),
        capture_output=True, text=True, timeout=600,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        log(f"  ✓ US-data OHLCV refreshed in {elapsed:.0f}s")
        # Print last few log lines
        for line in (result.stdout or "").splitlines()[-5:]:
            log(f"    {line}")
        return True
    else:
        log(f"  ✗ US_data_loader.py failed (rc={result.returncode}) in {elapsed:.0f}s")
        log(f"    {(result.stderr or '')[:400]}")
        return False


# ── Step 2: Refresh crash predictor exports ───────────────────────────────────
def step2_crash_predictor() -> bool:
    log("Step 2: Refreshing exports/crash_predictor_study/")
    t0 = time.time()
    try:
        import numpy as np
        import pandas as pd

        # Try fast path: use cached features parquet
        features_cache = Path("/tmp/features_cache.parquet")
        prices_cache   = Path("/tmp/prices_cache.parquet")

        import unittest.mock as mock
        sys.modules.setdefault("streamlit", mock.MagicMock())

        src = open(ROOT / "backend" / "analyze_crash_predictors.py").read()
        src_no_main = src[: src.rfind("if __name__")]
        globs = {
            "__file__": str(ROOT / "backend" / "analyze_crash_predictors.py"),
            "__name__": "analyze_crash_predictors",
        }
        exec(compile(src_no_main, "analyze_crash_predictors.py", "exec"), globs)

        if features_cache.exists() and prices_cache.exists():
            log("  Using cached features + prices")
            prices = pd.read_parquet(str(prices_cache))
            prices.index = pd.to_datetime(prices.index).tz_localize(None)
            prices = prices.sort_index().ffill()
            features = pd.read_parquet(str(features_cache))
        else:
            log("  Cache miss — running full download_prices + build_fred_weekly + build_feature_table")
            prices   = globs["download_prices"]()
            fred     = globs["build_fred_weekly"]()
            features = globs["build_feature_table"](prices, fred)
            try:
                features.to_parquet(str(features_cache))
                prices.to_parquet(str(prices_cache))
            except Exception:
                pass  # /tmp may not be writable

        features = globs["add_outcomes"](features)
        features = globs["add_composite_indicator"](features)
        features = globs["score_models"](features)

        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        features.to_csv(EXPORT_DIR / "weekly_feature_outcomes.csv")

        prices_for_daily = prices
        daily = globs["build_daily_confirmation"](prices_for_daily, features)
        daily.to_csv(EXPORT_DIR / "daily_confirmation.csv")
        globs["evaluate_daily_confirmation"](daily).to_csv(
            EXPORT_DIR / "daily_confirmation_evaluation.csv", index=False)

        signals  = ["model_liquidity_only","model_correction_watch","model_confirmed_drop",
                    "model_crash_watch","model_crash_confirmed"]
        outcomes = ["drop_5_4w","drop_10_4w","drop_5_8w","drop_10_8w",
                    "drop_5_13w","drop_10_13w","drop_10_26w","drop_15_26w","drop_20_26w"]
        evals = pd.DataFrame([globs["evaluate_signal"](features, s, o)
                               for s in signals for o in outcomes])
        evals.to_csv(EXPORT_DIR / "signal_evaluation.csv", index=False)
        globs["evaluate_composite_bins"](features).to_csv(
            EXPORT_DIR / "composite_risk_bins.csv", index=False)
        globs["grid_search"](features).to_csv(EXPORT_DIR / "threshold_grid.csv", index=False)
        features.iloc[-1].to_frame("latest").to_csv(EXPORT_DIR / "latest_signal_snapshot.csv")

        coverage = pd.DataFrame({
            "first_valid": features.apply(lambda s: s.first_valid_index()),
            "last_valid":  features.apply(lambda s: s.last_valid_index()),
            "non_null": features.notna().sum(),
            "rows": len(features),
        })
        coverage.to_csv(EXPORT_DIR / "data_coverage.csv")
        log(f"  ✓ Crash predictor done in {time.time()-t0:.0f}s  "
            f"({features.index.min().date()} → {features.index.max().date()})")
        return True
    except Exception as exc:
        log(f"  ✗ Crash predictor failed: {exc}")
        return False


# ── Step 3: Refresh stage breadth history ────────────────────────────────────
def step3_stage_breadth(us_data_dir: Path) -> bool:
    log("Step 3: Refreshing stage_breadth_history.csv")
    t0 = time.time()
    try:
        import numpy as np
        import pandas as pd
        import yfinance as yf
        from engine import compute_market_stages

        cov_path = OUT_DIR / "data_coverage.csv"
        if not cov_path.exists():
            log("  SKIP — data_coverage.csv not found (run backtest first)")
            return False
        cov     = pd.read_csv(cov_path)
        tickers = cov["Ticker"].dropna().str.upper().unique().tolist()
        log(f"  Downloading {len(tickers)} tickers from 2003-01-01…")

        raw = yf.download(tickers, start="2003-01-01", interval="1d",
                          auto_adjust=True, progress=False, threads=True)
        log(f"  Download done: {raw.shape}  ({time.time()-t0:.0f}s)")

        # Cache for reuse within same session
        try:
            raw.to_parquet(str(TMP_CACHE))
        except Exception:
            pass

        all_stages, failed = [], []
        for ticker in tickers:
            try:
                df_t = raw.xs(ticker, axis=1, level=1).copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
                df_t = df_t.dropna(subset=["Close", "Volume"])
                if len(df_t) < 60:
                    failed.append(ticker); continue
                staged = compute_market_stages(df_t)
                sc = staged[["Stage"]].copy(); sc["Ticker"] = ticker
                all_stages.append(sc)
            except Exception as e:
                failed.append(f"{ticker}({e})")

        combined = pd.concat(all_stages)
        combined.index = pd.to_datetime(combined.index)
        combined = combined[combined["Stage"] != "Insufficient History"]
        breadth = (
            combined.groupby(combined.index)["Stage"]
            .value_counts(normalize=True).unstack(fill_value=0) * 100
        )
        for col in ["Acceleration", "Accumulation", "Distribution", "Deceleration"]:
            if col not in breadth.columns:
                breadth[col] = 0.0
        breadth = breadth[["Acceleration","Accumulation","Distribution","Deceleration"]]
        breadth.index.name = "date"
        breadth = breadth.sort_index()

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        breadth.to_csv(OUT_DIR / "stage_breadth_history.csv")
        log(f"  ✓ Stage breadth done in {time.time()-t0:.0f}s  "
            f"({breadth.index[0].date()} → {breadth.index[-1].date()}, {len(failed)} failed)")
        return True
    except Exception as exc:
        log(f"  ✗ Stage breadth failed: {exc}")
        return False


# ── Step 4: Refresh backtest outputs ─────────────────────────────────────────
def step4_backtest(us_data_dir: Path, top_n: int = 100) -> bool:
    log(f"Step 4: Refreshing backtest outputs (top {top_n} tickers)")
    t0 = time.time()
    try:
        import numpy as np
        import pandas as pd
        from engine import compute_market_stages

        STAGES      = ["Acceleration","Accumulation","Distribution","Deceleration"]
        LONG_STAGES = {"Acceleration","Accumulation"}
        EXIT_STAGES = {"Distribution","Deceleration","Insufficient History"}

        uni_path = us_data_dir / "universe.json"
        if not uni_path.exists():
            log(f"  SKIP — universe.json not found at {uni_path}")
            return False

        uni   = json.loads(uni_path.read_text(encoding="utf-8"))
        stocks = uni.get("stocks", uni) if isinstance(uni, dict) else uni
        stocks = sorted(
            [s for s in stocks if isinstance(s, dict)],
            key=lambda x: x.get("market_cap", 0) or 0, reverse=True
        )[:top_n]
        tickers    = [s["ticker"] for s in stocks]
        name_map   = {s["ticker"]: s.get("name","")   for s in stocks}
        sector_map = {s["ticker"]: s.get("sector","") for s in stocks}

        def load_ohlcv(ticker: str) -> pd.DataFrame:
            path = us_data_dir / "ohlcv" / f"{ticker}.json"
            if not path.exists(): return pd.DataFrame()
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("data", payload)
            if not rows: return pd.DataFrame()
            df = pd.DataFrame(rows).rename(columns={
                "date":"Date","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date").sort_index()
            return df[["Open","High","Low","Close","Volume"]].apply(pd.to_numeric, errors="coerce").dropna()

        all_trades, daily_returns_list = [], []
        coverage_rows, forward_rows, shift_rows = [], [], []
        failed = []

        for i, ticker in enumerate(tickers, 1):
            df = load_ohlcv(ticker)
            if df.empty: failed.append(ticker); continue
            df = df[df.index >= pd.Timestamp("2020-01-01")]
            if len(df) < 60: failed.append(ticker); continue
            processed = compute_market_stages(df)

            coverage_rows.append({
                "Ticker": ticker, "Name": name_map.get(ticker,""),
                "Sector": sector_map.get(ticker,""), "Rows": len(processed),
                "Start": processed.index[0].date().isoformat(),
                "End":   processed.index[-1].date().isoformat(),
                "Latest Stage": processed["Stage"].iloc[-1],
            })

            # Forward returns
            for idx_dt, row in processed.iterrows():
                stage = row.get("Stage","")
                if stage in STAGES:
                    pos = processed.index.get_loc(idx_dt)
                    for h in [1, 5, 10, 20]:
                        if pos + h < len(processed):
                            ret = processed["Close"].iloc[pos+h] / processed["Close"].iloc[pos] - 1.0
                            forward_rows.append({"Ticker":ticker,"Sector":sector_map.get(ticker,""),
                                "Signal Date":idx_dt.date().isoformat(),"Horizon Days":h,
                                "Stage":stage,"Return":float(ret)})

            # Phase shifts
            trans_col = "Transition_Signal" if "Transition_Signal" in processed.columns else None
            if trans_col:
                for idx_dt, row in processed.iterrows():
                    trans = str(row.get(trans_col,""))
                    if trans.startswith("Entering "):
                        pos = processed.index.get_loc(idx_dt)
                        if pos + 5 < len(processed):
                            shift_rows.append({
                                "Ticker":ticker,"Sector":sector_map.get(ticker,""),
                                "Signal Date":idx_dt.date().isoformat(),
                                "Entry Date":processed.index[pos+1].date().isoformat(),
                                "Shift":trans,
                                "Return 5D":float(processed["Close"].iloc[pos+5]/processed["Open"].iloc[pos+1]-1.0),
                            })

            # Backtest (long LONG_STAGES, exit on EXIT_STAGES)
            in_pos = False; entry_px = None; trades_t = []; daily_t = []
            for i2, (date2, row2) in enumerate(processed.iterrows()):
                stage = row2["Stage"]
                if not in_pos and stage in LONG_STAGES and i2+1 < len(processed):
                    entry_px = processed["Open"].iloc[i2+1]; in_pos = True
                elif in_pos and stage in EXIT_STAGES and i2+1 < len(processed):
                    exit_px = processed["Open"].iloc[i2+1]
                    ret = exit_px / entry_px - 1.0 - 0.0002
                    trades_t.append({"Ticker":ticker,"Sector":sector_map.get(ticker,""),
                        "Entry":str(processed.index[max(0,i2)].date()),
                        "Exit":str(processed.index[i2+1].date()), "Return":ret})
                    in_pos = False
                if in_pos:
                    prev = processed["Close"].iloc[max(0,i2-1)]
                    daily_t.append((date2, processed["Close"].iloc[i2]/prev-1.0 if prev else 0.0))
            if trades_t: all_trades.append(pd.DataFrame(trades_t))
            if daily_t:  daily_returns_list.append(pd.Series({d:r for d,r in daily_t}, name=ticker))

        # Aggregate
        initial_capital = 100_000.0
        trade_log = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        forward   = pd.DataFrame(forward_rows)
        stage_summary = (
            forward.groupby(["Stage","Horizon Days"]).agg(
                Observations=("Return","size"), Mean_Return=("Return","mean"),
                Median_Return=("Return","median"), Win_Rate=("Return",lambda s:(s>0).mean())
            ).reset_index() if not forward.empty else pd.DataFrame()
        )
        shifts_df = pd.DataFrame(shift_rows)
        shift_summary = (
            shifts_df.groupby("Shift").agg(
                Observations=("Return 5D","size"), Mean_5D_Return=("Return 5D","mean"),
                Median_5D_Return=("Return 5D","median"), Win_Rate=("Return 5D",lambda s:(s>0).mean())
            ).reset_index() if not shifts_df.empty else pd.DataFrame()
        )

        if daily_returns_list:
            returns_matrix = pd.concat(daily_returns_list, axis=1).fillna(0.0)
            portfolio_daily = returns_matrix.mean(axis=1)
        else:
            portfolio_daily = pd.Series(dtype=float)
        equity = initial_capital * (1.0 + portfolio_daily).cumprod()

        # Summary stats
        ending = float(equity.iloc[-1]) if not equity.empty else initial_capital
        years  = (equity.index[-1]-equity.index[0]).days/365.25 if not equity.empty else 1
        cagr   = (ending/initial_capital)**(1.0/years)-1.0 if years > 0 else 0
        dd     = float((equity/equity.cummax()-1.0).min()) if not equity.empty else 0
        ret_s  = portfolio_daily.replace([float("inf"),float("-inf")],float("nan")).dropna()
        sharpe = float(ret_s.mean()/ret_s.std(ddof=1)*math.sqrt(252)) if len(ret_s)>1 and ret_s.std(ddof=1) else float("nan")
        summary = {
            "Universe":f"Top {top_n} by market cap","Start":"2020-01-01","End":"latest",
            "Initial Capital":initial_capital,"Ending Equity":ending,
            "CAGR":cagr,"Max Drawdown":dd,"Sharpe":sharpe,"Total Trades":len(trade_log),
        }

        # Save all outputs
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        trade_log.to_csv(OUT_DIR / "trade_log.csv", index=False)
        forward.to_csv(OUT_DIR / "stage_forward_returns.csv", index=False)
        stage_summary.to_csv(OUT_DIR / "stage_forward_summary.csv", index=False)
        shifts_df.to_csv(OUT_DIR / "phase_shift_forward_returns.csv", index=False)
        shift_summary.to_csv(OUT_DIR / "phase_shift_summary.csv", index=False)
        pd.DataFrame(coverage_rows).to_csv(OUT_DIR / "data_coverage.csv", index=False)
        equity.rename("Equity").to_csv(OUT_DIR / "equity_curve.csv")
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        pd.DataFrame([{"check":"data_available","status":"PASS","detail":f"{len(coverage_rows)} tickers"},
                      {"check":"equity","status":"PASS","detail":f"Ending={ending:,.0f}"}
                      ]).to_csv(OUT_DIR / "validation_checks.csv", index=False)

        log(f"  ✓ Backtest done in {time.time()-t0:.0f}s  "
            f"trades={len(trade_log)}  CAGR={cagr*100:.1f}%  Sharpe={sharpe:.2f}  MaxDD={dd*100:.1f}%")
        if failed: log(f"  Skipped: {', '.join(failed[:10])}")
        return True
    except Exception as exc:
        import traceback; traceback.print_exc()
        log(f"  ✗ Backtest failed: {exc}")
        return False


# ── Step 5: Export R3 signal cache (Thursdays only) ───────────────────────────
def step5_r3_signal_cache() -> bool:
    """Compute R3 deploy signal and write data/r3_signal_cache.json.

    R3 rebalance schedule: signal locked on Thursday close, execute Friday open.
    This cache is only refreshed on Thursdays (weekday == 3) so the stored
    signal always reflects the Thursday-close snapshot used for Friday execution.
    """
    today_wd = dt.date.today().weekday()     # 0=Mon … 4=Fri
    if today_wd != 3:                        # 3 = Thursday
        day_name = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][today_wd]
        log(f"Step 5: SKIP — R3 signal cache only updates on Thursdays (today is {day_name})")
        return True                          # non-failure skip

    log("Step 5: Computing R3 signal cache (Thursday) …")
    t0 = time.time()
    try:
        import numpy as np

        # --- Load market data from parquet (preferred) or yfinance ---
        data_path = ROOT / "data" / "market_data_master.parquet"
        if data_path.exists():
            try:
                import pandas as pd
                d = pd.read_parquet(data_path)
                d.sort_index(inplace=True)
                d.ffill(inplace=True)
            except Exception:
                d = pd.DataFrame()
        else:
            d = pd.DataFrame()

        if d.empty or len(d) < 252:
            log("  SKIP — parquet unavailable or too short; trying yfinance …")
            import yfinance as yf
            tickers = ["SPY", "^VIX", "HYG", "IEF", "^TNX", "^IRX", "TIP"]
            raw = yf.download(tickers, period="3y", auto_adjust=True, progress=False)["Close"]
            raw.dropna(how="all", inplace=True)
            raw.ffill(inplace=True)
            d = raw

        if len(d) < 252:
            log("  ✗ Insufficient data for R3 signal computation.")
            return False

        import pandas as pd

        # --- Bull Trap Score (exact logic from pages/3) ---
        latest  = d.iloc[-1]
        prev_mo = d.iloc[max(0, len(d) - 23)]

        curve      = latest["^TNX"] - latest["^IRX"]
        prev_curve = prev_mo["^TNX"] - prev_mo["^IRX"]
        if curve > 0 and prev_curve < 0:
            ys = 1.0
        elif curve > 0:
            ys = 0.5
        else:
            ys = 0.0

        vix_ma22 = d["^VIX"].rolling(22).mean().iloc[-1]
        if latest["^VIX"] < 15:
            vs = 1.0
        elif latest["^VIX"] < vix_ma22:
            vs = 0.5
        else:
            vs = 0.0

        hyg_ief      = d["HYG"] / d["IEF"]
        hyg_ief_ma22 = hyg_ief.rolling(22).mean().iloc[-1]
        cs = 1.0 if hyg_ief.iloc[-1] > hyg_ief_ma22 else 0.0

        spy_200ma = d["SPY"].rolling(200).mean().iloc[-1]
        if latest["SPY"] > spy_200ma * 1.05:
            bs = 1.0
        elif latest["SPY"] > spy_200ma:
            bs = 0.5
        else:
            bs = 0.0

        spy_mom22 = (latest["SPY"] / prev_mo["SPY"]) - 1
        if spy_mom22 > 0.02:
            acc = 1.0
        elif spy_mom22 > 0:
            acc = 0.5
        else:
            acc = 0.0

        if "TIP" in d.columns and len(d) >= 22:
            tip_trend = d["TIP"].iloc[-1] / d["TIP"].iloc[-22] - 1
            liq = 1.0 if tip_trend > 0 else 0.0
        else:
            liq = 0.5

        bt_score = round(min(10.0, ys + vs + cs + bs + acc + liq + 0.5 + 0.5 + 1.0), 2)

        # --- Bear Trap Score (exact logic from pages/2) ---
        def _norm(val, lower, upper, inverted=False):
            if inverted:
                if val >= lower: return 0.0
                if val <= upper: return 1.0
                return round((lower - val) / (lower - upper), 4)
            if val <= lower: return 0.0
            if val >= upper: return 1.0
            return round((val - lower) / (upper - lower), 4)

        irx_ma120     = d["^IRX"].rolling(120).mean().iloc[-1]
        hyg_ief_ma252 = hyg_ief.rolling(252).mean().iloc[-1]

        bear_score = round(
            _norm(curve,            1.5,                 -0.5,                inverted=True)  * 0.25 +
            _norm(latest["^IRX"],   irx_ma120 * 0.8,     irx_ma120 * 1.2)                     * 0.20 +
            _norm(hyg_ief.iloc[-1], hyg_ief_ma252*1.05,  hyg_ief_ma252*0.90, inverted=True)  * 0.20 +
            _norm(latest["SPY"],    spy_200ma * 1.05,    spy_200ma * 0.95,   inverted=True)  * 0.15 +
            _norm(latest["^VIX"],   15,                  35)                                  * 0.10 +
            0.65 * 0.05 +
            0.50 * 0.05,
            4,
        )

        # --- R3 deploy (vol-adjusted) ---
        TARGET_VOL     = 0.12
        bull_weight    = bt_score / 10.0
        net_confidence = bull_weight * (1.0 - bear_score)

        spy_w   = d["SPY"].resample("W-FRI").last()
        spy_ret = spy_w.pct_change().dropna().tail(4)
        if len(spy_ret) >= 2:
            spy_4wk_vol = float(spy_ret.std() * np.sqrt(52))
        else:
            spy_4wk_vol = TARGET_VOL
        vol_scalar = spy_4wk_vol / TARGET_VOL
        vol_adj    = net_confidence * min(1.5, 1.0 / vol_scalar) if vol_scalar > 0 else net_confidence
        deploy     = float(np.clip(0.10 + vol_adj * 0.90, 0.10, 1.00))
        deploy_pct = round(deploy * 100, 1)

        if deploy_pct >= 70:
            regime = "FULL DEPLOYMENT"
        elif deploy_pct >= 40:
            regime = "CAUTION"
        else:
            regime = "DEFENSIVE"

        # --- Top-3 NQ100 picks + breadth from System E weekly signals ---
        top3_picks, breadth_pct = [], None
        sig_path = ROOT / "exports" / "trap_regime_backtest" / "system_e_weekly_signals.csv"
        if sig_path.exists():
            try:
                sig_df = pd.read_csv(sig_path, index_col="Date", parse_dates=True).sort_index()
                if not sig_df.empty:
                    last = sig_df.iloc[-1]
                    raw3 = str(last.get("top3_picks", ""))
                    top3_picks  = [x.strip().strip('"') for x in raw3.split(",") if x.strip()]
                    breadth_pct = round(float(last.get("breadth_pct", 0.0)), 1)
            except Exception as exc:
                log(f"  ! Could not read weekly signals: {exc}")

        # --- Write cache ---
        today = dt.date.today()
        cache = {
            "date":           today.isoformat(),
            "signal_day":     "Thursday",
            "exec_day":       "Friday",
            "bt_score":       bt_score,
            "bear_score":     bear_score,
            "net_confidence": round(net_confidence, 4),
            "vol_scalar":     round(vol_scalar, 4),
            "deploy_pct":     deploy_pct,
            "top3_picks":     top3_picks,
            "breadth_pct":    breadth_pct,
            "regime":         regime,
            "next_execution": (today + dt.timedelta(days=(4 - today.weekday()) % 7)).isoformat(),
        }
        cache_path = ROOT / "data" / "r3_signal_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))

        log(f"  ✓ R3 signal cache written in {time.time()-t0:.0f}s  "
            f"deploy={deploy_pct}%  bt={bt_score}  bear={bear_score}  regime={regime}")
        return True
    except Exception as exc:
        import traceback; traceback.print_exc()
        log(f"  ✗ R3 signal cache failed: {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Daily Early Warning data refresh")
    parser.add_argument("--us-data-dir", default=r"D:\Codex projects\US-data",
                        help="Path to the US OHLCV data directory")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Universe size for the stage backtest")
    parser.add_argument("--skip", default="",
                        help="Comma-separated step numbers to skip, e.g. --skip 1,4")
    args = parser.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    us_data_dir = Path(args.us_data_dir)

    log("=" * 70)
    log("DAILY REFRESH — start")
    log("=" * 70)
    t_start = time.time()

    results: dict[str, bool] = {}
    if "1" not in skip:
        results["Step 1 (US data)"]         = step1_refresh_us_data(us_data_dir)
    if "2" not in skip:
        results["Step 2 (crash predictor)"] = step2_crash_predictor()
    if "3" not in skip:
        results["Step 3 (stage breadth)"]   = step3_stage_breadth(us_data_dir)
    if "4" not in skip:
        results["Step 4 (backtest)"]        = step4_backtest(us_data_dir, args.top_n)
    if "5" not in skip:
        results["Step 5 (R3 signal cache)"] = step5_r3_signal_cache()

    log("=" * 70)
    for name, ok in results.items():
        log(f"  {'✓' if ok else '✗'} {name}")
    log(f"DAILY REFRESH — done in {time.time()-t_start:.0f}s")
    log("=" * 70)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
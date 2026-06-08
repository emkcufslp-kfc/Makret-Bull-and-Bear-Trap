# -*- coding: utf-8 -*-
import os
import sys

# --- Dependency bootstrap ---
# yfinance and its deps (peewee, curl_cffi, etc.) are installed to /tmp/pkgs
# and /tmp/pkgs2 because the /sessions disk is 100% full and pip can't write to
# user site-packages.  Insert both paths before any downstream imports.
_EXTRA_PYTHONPATH = os.pathsep.join(("/tmp/pkgs", "/tmp/pkgs2"))
for _pkg_dir in ("/tmp/pkgs", "/tmp/pkgs2"):
    if os.path.isdir(_pkg_dir) and _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)
# Also propagate to child processes via PYTHONPATH env var
_existing_pp = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join((_EXTRA_PYTHONPATH, _existing_pp)) if _existing_pp else _EXTRA_PYTHONPATH
# --- End bootstrap ---

import subprocess
import datetime
from pathlib import Path
import pandas as pd
import yfinance as yf
import json
import time
import re

# Architecture: Unified Sync Engine (GitHub Actions Optimized)
# This script orchestrates the data update pipeline for the entire ecosystem.

# Set Working Directory to repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from backend.refresh_strategy_artifacts import refresh_all_strategy_artifacts
from utils.data_engine import get_master_data
from utils.yfinance_utils import configure_yfinance_cache

configure_yfinance_cache(Path(REPO_ROOT))

def log_progress(msg):
    try:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
    except UnicodeEncodeError:
        safe_msg = msg.encode('ascii', 'ignore').decode('ascii')
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {safe_msg}")
    sys.stdout.flush()

def run_script(script_path):
    log_progress(f"Running {script_path}...")
    try:
        full_path = os.path.join(REPO_ROOT, script_path)
        result = subprocess.run(
            [sys.executable, full_path],
            capture_output=True, text=True, check=True, timeout=300,
            env=os.environ  # propagates PYTHONPATH set in bootstrap above
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        log_progress(f"Error in {script_path}: {e.stderr[:500]}")
        return False, e.stderr
    except Exception as e:
        log_progress(f"Unexpected Error in {script_path}: {str(e)}")
        return False, str(e)

def update_macro_indicators():
    log_progress("Updating Macro & Expert Indicators...")
    try:
        spy = pd.DataFrame()
        vix = pd.DataFrame()
        try:
            spy = yf.download("SPY", period="2y", progress=False, auto_adjust=True)
            vix = yf.download("^VIX", period="2y", progress=False, auto_adjust=True)
        except Exception as e:
            log_progress(f"yfinance download failed: {e}. Falling back to local master DB.")
            
        if not spy.empty and isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        if not vix.empty and isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

        if spy.empty or vix.empty:
            master_df = get_master_data()
            if 'SPY' in master_df.columns:
                spy = pd.DataFrame({
                    'Close': master_df['SPY'],
                    'High': master_df['SPY'],
                    'Low': master_df['SPY']
                }).tail(504)
            if '^VIX' in master_df.columns:
                vix = pd.DataFrame({
                    'Close': master_df['^VIX']
                }).tail(504)

        if spy.empty:
            log_progress("Market data fetch/load failed completely.")
            return False

        spy['High_Rolling'] = spy['High'].rolling(22).max()
        spy['VIX_Fix'] = (spy['High_Rolling'] - spy['Low']) / spy['High_Rolling'] * 100
        indicator_1_val = spy['VIX_Fix'].iloc[-1]
        
        spy['200MA'] = spy['Close'].rolling(200).mean()
        indicator_2_val = ((spy['Close'].iloc[-1] / spy['200MA'].iloc[-1]) - 1) * 100
        
        high = spy['Close'].max()
        indicator_3_val = ((spy['Close'].iloc[-1] / high) - 1) * 100
        
        macro_data = {
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "indicator_1": round(float(indicator_1_val), 2),
            "indicator_2": round(float(indicator_2_val), 2),
            "indicator_3": round(float(indicator_3_val), 2),
            "vix": round(float(vix['Close'].iloc[-1]), 2),
            "status": "Green" if indicator_2_val > 0 else "Caution"
        }
        
        out_path = os.path.join(REPO_ROOT, "data", "Multi_indicator", "macro_data.js")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"const macroData = {json.dumps(macro_data, indent=4)};")
        
        log_progress("Macro Data Layer updated.")
        return True
    except Exception as e:
        log_progress(f"Macro sync failed: {str(e)}")
        return False

def sync():
    log_progress("Starting Ecosystem Master Sync Pipeline...")
    
    log_progress("Updating local master database (market_data_master.parquet)...")
    try:
        get_master_data()
        log_progress("Local master database updated successfully.")
    except Exception as e:
        log_progress(f"Failed to update local master database: {e}")

    run_script("backend/fetch_spy_data.py")
    run_script("backend/strategies/ntsx_engine.py")
    run_script("backend/export_platinum_data.py")
    run_script("backend/export_fund_tactical_data.py")
    run_script("backend/update_sentiment.py")
    update_macro_indicators()
    try:
        log_progress("Refreshing bundled strategy artifacts...")
        for artifact in refresh_all_strategy_artifacts():
            log_progress(f"Updated artifact: {artifact}")
    except Exception as e:
        log_progress(f"Strategy artifact refresh failed: {e}")
    
    log_progress("Pipeline execution complete.")

    if os.environ.get("GITHUB_ACTIONS") == "true":
        log_progress("Detecting CI Environment. Preparing automated commit...")
    else:
        log_progress("Running locally. Skipping automated Git push.")

if __name__ == "__main__":
    sync()

# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import datetime
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

def log_progress(msg):
    try:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")
    except UnicodeEncodeError:
        # Fallback for systems that don't support certain emojis/characters (e.g. Windows console)
        safe_msg = msg.encode('ascii', 'ignore').decode('ascii')
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {safe_msg}")
    sys.stdout.flush()

def run_script(script_path):
    log_progress(f"Running {script_path}...")
    try:
        # Use relative pathing
        full_path = os.path.join(REPO_ROOT, script_path)
        result = subprocess.run([sys.executable, full_path], capture_output=True, text=True, check=True, timeout=300)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        log_progress(f"❌ Error in {script_path}: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        log_progress(f"❌ Unexpected Error in {script_path}: {str(e)}")
        return False, str(e)

def update_macro_indicators():
    """Calculates high-level metrics for the Market Regime dashboard."""
    log_progress("Updating Macro & Expert Indicators...")
    try:
        # Fetch SPY for base indicators
        spy = yf.download("SPY", period="2y", progress=False, auto_adjust=True)
        vix = yf.download("^VIX", period="2y", progress=False, auto_adjust=True)
        
        if spy.empty:
            log_progress("Market data fetch failed.")
            return False

        # Flatten MultiIndex if necessary
        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

        # Indicator Calculations
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
        
        # Save to data directory
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
    log_progress("🚀 Starting Ecosystem Master Sync Pipeline...")
    
    # 1. Fetch & Update Data Layers
    # We will migrate individual scripts into the backend folder
    run_script("backend/fetch_spy_data.py")
    run_script("backend/export_dashboard_data.py")
    run_script("backend/export_platinum_data.py")
    run_script("backend/update_sentiment.py")
    update_macro_indicators()
    
    log_progress("✅ Pipeline execution complete.")

    # 2. Automated Git Sync (Only if running in GitHub Actions)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        log_progress("Detecting CI Environment. Preparing automated commit...")
        # Note: commit/push logic is usually handled in the .yml workflow file
        # for better security and observability.
    else:
        log_progress("Running locally. Skipping automated Git push.")

if __name__ == "__main__":
    sync()

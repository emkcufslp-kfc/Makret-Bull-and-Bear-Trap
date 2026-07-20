"""
_refresh_data_step1.py
Incrementally updates data/market_data_master.parquet via utils.data_engine.get_master_data().
Prints: "Master data updated to: YYYY-MM-DD"
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Dep bootstrap (sandbox disk-space workaround; see CLAUDE.md) ──────────────
# /tmp/pkgs2 does not survive a cold sandbox restart, so fall back to an
# on-demand pip install targeting /tmp when the cache is missing.
for _p in ("/tmp/pkgs2", "/tmp/pkgs"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import yfinance  # noqa: F401
except ImportError:
    _FALLBACK_PKG_DIR = "/tmp/pkgs_fallback"
    if _FALLBACK_PKG_DIR not in sys.path:
        sys.path.insert(0, _FALLBACK_PKG_DIR)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "yfinance>=0.2.30", "pandas>=2.0.0",
         "pyarrow>=14.0.0",
         "--target", _FALLBACK_PKG_DIR, "--no-cache-dir", "--quiet"],
        env={**os.environ, "HOME": "/tmp", "TMPDIR": "/tmp"},
        capture_output=True,
    )

from utils.data_engine import get_master_data

df = get_master_data()
latest_date = df.index.max().date()
print(f"Master data updated to: {latest_date}")

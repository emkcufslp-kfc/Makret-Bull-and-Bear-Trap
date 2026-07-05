"""
_refresh_data_step1.py
Incrementally updates data/market_data_master.parquet via utils.data_engine.get_master_data().
Prints: "Master data updated to: YYYY-MM-DD"
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.data_engine import get_master_data

df = get_master_data()
latest_date = df.index.max().date()
print(f"Master data updated to: {latest_date}")

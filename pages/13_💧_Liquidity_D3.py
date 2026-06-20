from __future__ import annotations

import runpy
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT_DIR / "liquidity_dashboard_3.py"), run_name="__main__")

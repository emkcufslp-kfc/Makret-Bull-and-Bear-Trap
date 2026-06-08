from __future__ import annotations

from pathlib import Path

import yfinance as yf


def configure_yfinance_cache(repo_root: Path) -> Path:
    cache_dir = repo_root / ".cache" / "py-yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf.cache.set_cache_location(str(cache_dir))
    except Exception:
        pass
    return cache_dir

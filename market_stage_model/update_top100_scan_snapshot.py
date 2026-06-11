from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
US_DATA_LOADER = Path(r"D:\Codex projects\US-data\US_data_loader.py")
SNAPSHOT_FILE = APP_DIR / "top100_scan_snapshot.json"


def load_market_stage_app():
    sys.path.insert(0, str(APP_DIR))
    spec = importlib.util.spec_from_file_location("market_stage_app", APP_DIR / "app.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load market_stage_model/app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refresh_us_data() -> None:
    if not US_DATA_LOADER.exists():
        raise FileNotFoundError(f"US-data loader not found: {US_DATA_LOADER}")
    subprocess.run(
        [
            sys.executable,
            str(US_DATA_LOADER),
            "--mode",
            "ohlcv",
            "--top-n",
            "100",
            "--refresh-stale",
            "--delay",
            "0.1",
        ],
        cwd=str(ROOT_DIR),
        check=True,
    )


def build_snapshot() -> dict[str, object]:
    app = load_market_stage_app()
    us_data_dir = app.get_us_data_dir()
    if us_data_dir is None:
        raise FileNotFoundError("US-data directory with universe.json and ohlcv/ was not found.")

    universe = app.load_us_data_top100_universe()
    scan_data = {}
    missing = []
    for ticker in universe["Ticker"].tolist():
        data = app.load_us_data_price_history(ticker, 220, False)
        if data.empty:
            missing.append(ticker)
        else:
            scan_data[ticker] = data

    sector_lookup = dict(zip(universe["Ticker"], universe["Sector"]))
    results = app.attach_scan_sectors(app.scan_phase_shifts_for_modes(scan_data, "Both Shifts"), sector_lookup)
    as_of = None if results.empty else str(results["As Of"].max())
    return {
        "generated_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds"),
        "source_dir": str(us_data_dir),
        "source_universe": str(us_data_dir / "universe.json"),
        "source_ohlcv_dir": str(us_data_dir / "ohlcv"),
        "scan_mode": "Both Shifts",
        "universe_method": "Top 100 by market_cap from refreshed US-data universe.json with local OHLCV availability",
        "universe_count": int(len(universe)),
        "loaded_count": int(len(scan_data)),
        "missing_count": int(len(missing)),
        "missing_tickers": missing,
        "as_of": as_of,
        "results": results.to_dict("records"),
    }


def main() -> None:
    refresh_us_data()
    snapshot = build_snapshot()
    SNAPSHOT_FILE.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Wrote {SNAPSHOT_FILE}")
    print(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()

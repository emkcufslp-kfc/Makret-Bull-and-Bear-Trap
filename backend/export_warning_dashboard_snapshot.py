from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils.data_engine import get_clean_master
from utils.warning_dashboard import build_warning_dashboard

EXPORT_DIR = ROOT_DIR / "exports"


def main() -> int:
    master = get_clean_master().ffill().dropna(how="all")
    if master.empty:
        raise RuntimeError("Shared master dataset is empty.")
    target_date = master.index.max().date()
    payload = build_warning_dashboard(target_date)
    if payload is None:
        raise RuntimeError(f"Unable to build Warning dashboard payload for {target_date}")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "master_date": str(payload["master_date"]),
        "actual_date": str(payload["actual_date"]),
        "warning_level": payload["warning_level"],
        "trigger_score": payload["trigger_score"],
        "active_warnings": payload["active_warnings"],
        "escalation_status": payload["escalation_status"],
        "last_signal_change": payload["last_signal_change"],
        "summary_points": payload["summary_points"],
    }

    (EXPORT_DIR / "warning_dashboard_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    payload["indicator_matrix"].to_csv(EXPORT_DIR / "warning_indicator_matrix.csv", index=False, encoding="utf-8-sig")
    payload["timeline"].to_csv(EXPORT_DIR / "warning_timeline.csv", index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

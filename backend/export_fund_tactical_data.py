from pathlib import Path
import shutil
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
STRAT_DIR = REPO_ROOT / "backend" / "strategies"
if str(STRAT_DIR) not in sys.path:
    sys.path.append(str(STRAT_DIR))

import fund_tactical_engine as ftaa


TARGET_DIR = REPO_ROOT / "data" / "Fund_Tactical_Results"
OUTPUT_FILES = [
    "performance_summary.csv",
    "annual_returns.csv",
    "monthly_returns.csv",
    "equity_curve.csv",
    "weights.csv",
    "regimes.csv",
    "win_stats.csv",
    "equity_curve.png",
    "drawdown.png",
]


def export_data() -> None:
    temp_dir = REPO_ROOT / "backend" / "_tmp_fund_tactical"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("Running weekly F-TAA backtest from shared master dataset...")
    prices, _, _, _, _ = ftaa.save(ftaa.CONFIG, temp_dir)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for filename in OUTPUT_FILES:
        shutil.copy2(temp_dir / filename, TARGET_DIR / f"Fund_Tactical_{filename}")

    prices.to_csv(TARGET_DIR / "Fund_Tactical_Prices.csv")
    shutil.rmtree(temp_dir)
    print(f"Fund tactical outputs exported to {TARGET_DIR}")


if __name__ == "__main__":
    export_data()

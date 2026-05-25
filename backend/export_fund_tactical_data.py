import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT.parent / "fund_tactical_spy_beater_package"
PACKAGE_SCRIPT = PACKAGE_ROOT / "src" / "backtest_fund_tactical.py"
PACKAGE_OUTPUT = PACKAGE_ROOT / "output"
TARGET_DIR = REPO_ROOT / "data" / "Fund_Tactical_Results"
TICKERS = ["SPY", "QQQ", "GMOM", "RLY", "DBMF", "SGOV"]
OUTPUT_FILES = [
    "performance_summary.csv",
    "annual_returns.csv",
    "monthly_returns.csv",
    "equity_curve.csv",
    "weights.csv",
    "regimes.csv",
    "win_stats.csv",
]


def run_backtest() -> None:
    if not PACKAGE_SCRIPT.exists():
        print(f"Fund tactical source package not found at {PACKAGE_SCRIPT}. Using committed results if available.")
        return

    print(f"Running tactical strategy backtest from {PACKAGE_ROOT}...")
    subprocess.run(
        [sys.executable, str(PACKAGE_SCRIPT)],
        cwd=str(PACKAGE_ROOT),
        check=True,
        timeout=600,
    )


def copy_outputs() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for filename in OUTPUT_FILES:
        source = PACKAGE_OUTPUT / filename
        if not source.exists():
            existing = TARGET_DIR / f"Fund_Tactical_{filename}"
            if existing.exists():
                continue
            raise FileNotFoundError(f"Expected tactical output not found: {source}")
        shutil.copy2(source, TARGET_DIR / f"Fund_Tactical_{filename}")


def export_prices() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    data = yf.download(TICKERS, period="10d", auto_adjust=True, progress=False)
    close = data["Close"].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
    if close.empty:
        existing = TARGET_DIR / "Fund_Tactical_Prices.csv"
        if existing.exists():
            print("Price snapshot download unavailable. Keeping existing Fund_Tactical_Prices.csv.")
            return
        raise RuntimeError("Unable to download tactical strategy price snapshot.")

    close = close.ffill().dropna(how="all")
    close.to_csv(TARGET_DIR / "Fund_Tactical_Prices.csv")


def main() -> None:
    run_backtest()
    copy_outputs()
    export_prices()
    print(f"Fund tactical outputs exported to {TARGET_DIR}")


if __name__ == "__main__":
    main()

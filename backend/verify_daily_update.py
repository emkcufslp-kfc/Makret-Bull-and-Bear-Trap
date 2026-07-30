from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.strategies.allocator_engine import load_allocator_json
from backend.strategies.combined_macro_rwra import compute_combined_snapshot
from backend.strategies.ensemble_top100_engine import load_ensemble_top100_json
from backend.strategies.fund_tactical_engine import CONFIG as FUND_TACTICAL_CONFIG
from backend.strategies.fund_tactical_engine import load_prices as load_fund_tactical_prices
from backend.strategies.platinum_engine import fetch_data as load_platinum_prices
from utils.data_engine import get_clean_master
from utils.warning_dashboard import build_warning_dashboard
from utils.yfinance_utils import configure_yfinance_cache

configure_yfinance_cache(ROOT_DIR)


def latest_us_trading_date() -> pd.Timestamp:
    data = yf.download("SPY", period="10d", interval="1d", progress=False, auto_adjust=True)
    if data.empty:
        raise RuntimeError("Unable to fetch SPY from Yahoo Finance for freshness verification.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return pd.Timestamp(data.dropna(how="all").index.max()).normalize()


def read_latest_date_from_table(path: Path) -> pd.Timestamp:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        idx = pd.to_datetime(df.index, errors="coerce")
        latest = idx.max()
    else:
        df = pd.read_csv(path, index_col=0)
        idx = pd.to_datetime(df.index, errors="coerce")
        latest = idx.max()
        if pd.isna(latest) and "Date" in df.columns:
            latest = pd.to_datetime(df["Date"], errors="coerce").max()
    if pd.isna(latest):
        raise RuntimeError(f"Could not resolve a latest date from {path}")
    return pd.Timestamp(latest).normalize()


def read_latest_date_from_js(path: Path) -> pd.Timestamp:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"(20\d{2}-\d{2}-\d{2})", text)
    if not matches:
        raise RuntimeError(f"Could not find YYYY-MM-DD dates in {path}")
    return pd.Timestamp(max(matches)).normalize()


def read_ntsx_asof_date(path: Path) -> pd.Timestamp:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r'"as_of_date"\s*:\s*"(\d{4}-\d{2}-\d{2})"', text)
    if not match:
        raise RuntimeError(f'Could not find "as_of_date" in {path}')
    return pd.Timestamp(match.group(1)).normalize()


def check_exists_and_nonempty(label: str, path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"{label}: missing file {path}")
        return
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"{label}: failed to read {path} ({exc})")
        return
    if df.empty:
        errors.append(f"{label}: file is empty")
    else:
        print(f"[OK] {label}: {len(df)} rows")


def check_equal(label: str, actual: pd.Timestamp, expected: pd.Timestamp, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected.date()}, found {actual.date()}")
    else:
        print(f"[OK] {label}: {actual.date()}")


def resolve_ntsx_expected_date() -> pd.Timestamp:
    master = get_clean_master()
    tickers = [ticker for ticker in ["SPY", "TLT", "BIL", "KMLM", "DFSVX", "RYMFX"] if ticker in master.columns]
    valid = master.loc[:, tickers].dropna(how="any")
    if valid.empty:
        raise RuntimeError("Unable to resolve NTSX expected date from shared master data.")
    return pd.Timestamp(valid.index.max()).normalize()


def resolve_platinum_expected_date() -> pd.Timestamp:
    prices = load_platinum_prices()
    if prices.empty:
        raise RuntimeError("Unable to resolve Platinum expected date from strategy source data.")
    return pd.Timestamp(prices.index.max()).normalize()


def resolve_fund_tactical_expected_date() -> pd.Timestamp:
    prices = load_fund_tactical_prices(FUND_TACTICAL_CONFIG)
    if prices.empty:
        raise RuntimeError("Unable to resolve Fund Tactical expected date from strategy source data.")
    return pd.Timestamp(prices.index.max()).normalize()


def main() -> int:
    market_expected = latest_us_trading_date()
    ntsx_expected = resolve_ntsx_expected_date()
    platinum_expected = resolve_platinum_expected_date()
    fund_tactical_expected = resolve_fund_tactical_expected_date()

    print(f"Latest U.S. trading date from Yahoo: {market_expected.date()}")
    print(f"NTSX common-source date: {ntsx_expected.date()}")
    print(f"Platinum source date: {platinum_expected.date()}")
    print(f"Fund Tactical source date: {fund_tactical_expected.date()}")

    errors: list[str] = []

    tabular_checks = {
        "Master DB": (ROOT_DIR / "data" / "market_data_master.parquet", market_expected),
        "Platinum Equity": (ROOT_DIR / "data" / "Platinum_Results" / "Platinum_Equity.csv", platinum_expected),
        "Platinum Prices": (ROOT_DIR / "data" / "Platinum_Results" / "Platinum_Data_Used.csv", platinum_expected),
        "Fund Tactical Prices": (ROOT_DIR / "data" / "Fund_Tactical_Results" / "Fund_Tactical_Prices.csv", fund_tactical_expected),
        "Fund Tactical Equity": (ROOT_DIR / "data" / "Fund_Tactical_Results" / "Fund_Tactical_equity_curve.csv", fund_tactical_expected),
        "Top100 ETF Cache": (ROOT_DIR / "backend" / "strategies" / "data" / "top100_etf_prices.csv", market_expected),
    }

    for label, (path, expected) in tabular_checks.items():
        actual = read_latest_date_from_table(path)
        check_equal(label, actual, expected, errors)

    js_checks = {
        "SPY JS": (ROOT_DIR / "data" / "Multi_indicator" / "spy_data.js", market_expected),
        "Platinum JS": (ROOT_DIR / "data" / "Multi_indicator" / "platinum_data.js", platinum_expected),
    }

    for label, (path, expected) in js_checks.items():
        actual = read_latest_date_from_js(path)
        check_equal(label, actual, expected, errors)

    ntsx_actual = read_ntsx_asof_date(ROOT_DIR / "data" / "Multi_indicator" / "ntsx_data.js")
    check_equal("NTSX JS", ntsx_actual, ntsx_expected, errors)

    check_exists_and_nonempty(
        "Model Change Worksheet Export",
        ROOT_DIR / "exports" / "model_change_worksheet_full_history.csv",
        errors,
    )
    check_exists_and_nonempty(
        "Model Change Worksheet Bundled",
        ROOT_DIR / "backend" / "strategies" / "data" / "model_change_worksheet_full_history.csv",
        errors,
    )
    check_exists_and_nonempty(
        "Warning Indicator Matrix Export",
        ROOT_DIR / "exports" / "warning_indicator_matrix.csv",
        errors,
    )
    check_exists_and_nonempty(
        "Warning Timeline Export",
        ROOT_DIR / "exports" / "warning_timeline.csv",
        errors,
    )

    calibration_path = ROOT_DIR / "data" / "market_regime_calibration.json"
    if not calibration_path.exists():
        errors.append(f"Market Regime calibration: missing file {calibration_path}")
    else:
        try:
            import json as _json
            calibration = _json.loads(calibration_path.read_text(encoding="utf-8"))
            if not calibration.get("lookup"):
                errors.append("Market Regime calibration: file has no lookup table")
            else:
                print(
                    f"[OK] Market Regime calibration: elevated>={calibration['thresholds']['elevated']}% "
                    f"warning>={calibration['thresholds']['warning']}% "
                    f"ceiling={calibration['max_observed_calibrated_probability']}%"
                )
        except Exception as exc:
            errors.append(f"Market Regime calibration: failed to read {calibration_path} ({exc})")

    bear_trap_calibration_path = ROOT_DIR / "data" / "bear_trap_calibration.json"
    if not bear_trap_calibration_path.exists():
        errors.append(f"Bear Trap calibration: missing file {bear_trap_calibration_path}")
    else:
        try:
            import json as _json
            bear_calibration = _json.loads(bear_trap_calibration_path.read_text(encoding="utf-8"))
            primary = bear_calibration.get("horizons", {}).get(bear_calibration.get("primary_horizon", "3m"), {})
            if not primary.get("lookup"):
                errors.append("Bear Trap calibration: file has no lookup table")
            else:
                print(
                    f"[OK] Bear Trap calibration: 3M elevated>={bear_calibration['thresholds']['elevated']}% "
                    f"warning>={bear_calibration['thresholds']['warning']}% "
                    f"ceiling={primary['max_observed_calibrated_probability']}%"
                )
        except Exception as exc:
            errors.append(f"Bear Trap calibration: failed to read {bear_trap_calibration_path} ({exc})")

    combined = compute_combined_snapshot(market_expected.date())
    combined_resolved = pd.Timestamp(combined.resolved_date).normalize()
    if combined_resolved > market_expected:
        errors.append(
            f"Combined Macro + RWRA snapshot resolved to future date {combined.resolved_date}"
        )
    else:
        print(f"[OK] Combined Macro + RWRA snapshot resolved date: {combined.resolved_date}")

    allocator_payload = load_allocator_json(market_expected)
    if not allocator_payload:
        errors.append("Allocator strategy payload build returned no data.")
    else:
        print("[OK] Allocator strategy payload built successfully.")

    ensemble_payload = load_ensemble_top100_json(market_expected)
    if not ensemble_payload:
        errors.append("Ensemble Top-100 strategy payload build returned no data.")
    else:
        print("[OK] Ensemble Top-100 strategy payload built successfully.")

    warning_payload = build_warning_dashboard(market_expected.date())
    if not warning_payload:
        errors.append("Warning dashboard payload build returned no data.")
    else:
        warning_actual = pd.Timestamp(warning_payload["actual_date"]).normalize()
        if warning_actual > market_expected:
            errors.append(f"Warning dashboard resolved to future date {warning_actual.date()}")
        else:
            print(
                f"[OK] Warning dashboard payload built successfully "
                f"({warning_payload['warning_level']} | {warning_payload['active_warnings']} active)"
            )

    if errors:
        print("\nVerification failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("\nAll daily update checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

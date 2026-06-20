from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "exports" / "crash_predictor_study"


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def fmt(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def risk_dot(status: str) -> str:
    text = str(status)
    if text in {"Normal", "Stable", "Clear", "Bullish", "Inactive"}:
        return "🟢"
    if text in {"Watch", "Guarded", "Confirming", "Caution"}:
        return "🟡"
    if text in {"Warning", "Stress", "Elevated", "Critical", "High Confirmation", "Bearish", "Active"}:
        return "🔴"
    return "⚪"


def score_dot(value: float, watch: float, warning: float, inverted: bool = False) -> str:
    if pd.isna(value):
        return "⚪"
    if inverted:
        if value <= warning:
            return "🔴"
        if value <= watch:
            return "🟡"
        return "🟢"
    if value >= warning:
        return "🔴"
    if value >= watch:
        return "🟡"
    return "🟢"


def load_latest() -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    weekly = pd.read_csv(STUDY_DIR / "weekly_feature_outcomes.csv", parse_dates=["date"]).set_index("date").sort_index()
    daily = pd.read_csv(STUDY_DIR / "daily_confirmation.csv", parse_dates=["date", "weekly_date"]).set_index("date").sort_index()
    daily_eval = pd.read_csv(STUDY_DIR / "daily_confirmation_evaluation.csv")
    return weekly.iloc[-1], daily.iloc[-1], daily_eval


def matching_daily_eval(daily_row: pd.Series, daily_eval: pd.DataFrame) -> pd.Series | None:
    match = daily_eval[
        daily_eval["weekly_state"].eq(str(daily_row["composite_risk_state"]))
        & daily_eval["daily_confirmation_state"].eq(str(daily_row["daily_confirmation_state"]))
    ]
    if match.empty:
        return None
    return match.iloc[0]


def build_message() -> str:
    weekly, daily, daily_eval = load_latest()
    eval_row = matching_daily_eval(daily, daily_eval)

    if eval_row is None:
        probs = "4W >=5%: n/a | 8W >=10%: n/a | 13W >=10%: n/a | 26W >=20%: n/a"
    else:
        probs = (
            f"4W >=5%: {pct(eval_row['drop5_4w'])} | "
            f"8W >=10%: {pct(eval_row['drop10_8w'])}\n"
            f"13W >=10%: {pct(eval_row['drop10_13w'])} | "
            f"26W >=20%: {pct(eval_row['drop20_26w'])}"
        )

    d1_score = float(weekly.get("d1_market_regime_score", 0))
    bear_12m = float(weekly.get("d1_bear_prob12", float("nan")))
    bull_prob = float(weekly.get("d1_bull_prob", float("nan")))
    ma_state = str(weekly.get("d1_200ma_state", "n/a"))
    d2_score = float(weekly.get("d2_score", 0))
    d2_level = str(weekly.get("d2_level", "n/a"))
    liquidity_score = float(weekly.get("liquidity_score", 0))
    reserve_gdp = float(weekly.get("reserve_gdp", float("nan")))
    credit_state = "Active" if bool(weekly.get("credit_stress", False)) else "Inactive"

    indicators = "\n".join(
        [
            f"{score_dot(d1_score, 30, 55)} D1 Score: {fmt(d1_score, 0)}",
            f"{score_dot(bear_12m, 40, 55)} Bear 12M: {fmt(bear_12m)}%",
            f"{score_dot(bull_prob, 60, 40, inverted=True)} Bull Prob: {fmt(bull_prob)}%",
            f"{risk_dot(ma_state)} 200MA: {ma_state}",
            f"{score_dot(d2_score, 4, 7)} D2 Score/Level: {fmt(d2_score, 0)} / {d2_level}",
            f"{score_dot(liquidity_score, 30, 55)} D3 Liquidity: {fmt(liquidity_score, 0)}",
            f"{score_dot(reserve_gdp, 12, 10, inverted=True)} Reserve/GDP: {fmt(reserve_gdp)}%",
            f"{risk_dot(credit_state)} Credit Stress: {credit_state}",
        ]
    )

    read = "Risk present, but daily action is not confirming yet."
    if str(daily["daily_confirmation_state"]) == "Confirming":
        read = "Weekly risk is being confirmed by daily market internals."
    elif str(daily["daily_confirmation_state"]) == "High Confirmation":
        read = "High daily confirmation: treat drawdown risk as active now."

    return (
        f"💧 D3 Weekly Liquidity Risk\n"
        f"As of: {daily.name.date()}\n\n"
        f"{risk_dot(weekly['composite_risk_state'])} Weekly D3: {weekly['composite_risk_state']} / {fmt(weekly['composite_risk_score'])}/100\n"
        f"{risk_dot(daily['daily_confirmation_state'])} Daily Confirm: {daily['daily_confirmation_state']} / {fmt(daily['daily_confirmation_score'])}/100\n\n"
        f"🎯 Matched Drop Odds\n{probs}\n\n"
        f"📌 Indicator Colors\n{indicators}\n\n"
        f"Read: {read}"
    )


def send_telegram(message: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message, "disable_web_page_preview": True},
        timeout=30,
    )
    response.raise_for_status()


def main() -> None:
    message = build_message()
    print(message)
    send_telegram(message)


if __name__ == "__main__":
    main()

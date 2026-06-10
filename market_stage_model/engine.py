from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class StageSnapshot:
    ticker: str
    as_of: pd.Timestamp
    close: float
    stage: str
    transition_signal: str
    vwma_8: float
    vwma_21: float
    vwma_34: float
    vma_21: float
    volume_delta_20d: float


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance or user-supplied OHLCV data into one clean frame."""
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]

    lookup = {str(col).strip().lower(): col for col in out.columns}
    normalized: dict[str, pd.Series] = {}
    for required in REQUIRED_COLUMNS:
        source = lookup.get(required.lower())
        if source is None:
            raise ValueError(f"Missing required OHLCV column: {required}")
        normalized[required] = pd.to_numeric(out[source], errors="coerce")

    clean = pd.DataFrame(normalized, index=pd.to_datetime(out.index))
    clean = clean.sort_index()
    return clean.dropna(subset=["Close", "Volume"])


def calculate_vwma(df: pd.DataFrame, length: int) -> pd.Series:
    """Calculate volume weighted moving average using current/prior bars only."""
    clean = normalize_ohlcv_columns(df)
    price_volume = clean["Close"] * clean["Volume"]
    pv_sum = price_volume.rolling(length, min_periods=length).sum()
    volume_sum = clean["Volume"].rolling(length, min_periods=length).sum()
    return pv_sum / volume_sum.replace(0, np.nan)


def calculate_vma(df: pd.DataFrame, length: int = 21) -> pd.Series:
    """Calculate a volatility-adaptive moving average without future bars."""
    clean = normalize_ohlcv_columns(df)
    close = clean["Close"]
    vma = pd.Series(np.nan, index=clean.index, dtype="float64")
    if len(clean) < length:
        return vma

    price_change = close.diff().abs()
    volatility = close.rolling(length, min_periods=length).std()
    total_movement = price_change.rolling(length, min_periods=length).sum()
    volatility_factor = (volatility / total_movement.replace(0, np.nan)).fillna(0.1)
    volatility_factor = volatility_factor.clip(lower=0.05, upper=0.95)
    base_alpha = 2.0 / (length + 1.0)

    seed_idx = length - 1
    vma.iloc[seed_idx] = close.iloc[:length].mean()
    for i in range(seed_idx + 1, len(clean)):
        alpha = base_alpha * float(volatility_factor.iloc[i])
        vma.iloc[i] = close.iloc[i] * alpha + vma.iloc[i - 1] * (1.0 - alpha)

    return vma


def compute_market_stages(df: pd.DataFrame) -> pd.DataFrame:
    """Classify bars into Acceleration, Accumulation, Distribution, or Deceleration."""
    out = normalize_ohlcv_columns(df)
    if out.empty:
        return out

    out["VWMA_8"] = calculate_vwma(out, 8)
    out["VWMA_21"] = calculate_vwma(out, 21)
    out["VWMA_34"] = calculate_vwma(out, 34)
    out["VMA_21"] = calculate_vma(out, 21)

    ready = out[["VWMA_8", "VWMA_21", "VWMA_34", "VMA_21"]].notna().all(axis=1)
    out["is_bullish"] = ready & (out["VWMA_8"] > out["VWMA_21"]) & (out["VWMA_21"] > out["VWMA_34"])
    out["is_bearish"] = ready & (out["VWMA_8"] < out["VWMA_21"]) & (out["VWMA_21"] < out["VWMA_34"])
    out["is_neutral"] = ready & ~out["is_bullish"] & ~out["is_bearish"]

    conditions = [
        out["is_bullish"],
        out["is_bearish"],
        out["is_neutral"] & (out["Close"] >= out["VMA_21"]),
        out["is_neutral"] & (out["Close"] < out["VMA_21"]),
    ]
    choices = ["Acceleration", "Deceleration", "Accumulation", "Distribution"]
    out["Stage"] = np.select(conditions, choices, default="Insufficient History")

    out["Prev_Stage"] = out["Stage"].shift(1)
    valid_transition = ready & out["Prev_Stage"].notna() & (out["Prev_Stage"] != "Insufficient History")
    changed = valid_transition & (out["Stage"] != out["Prev_Stage"])
    out["Transition_Signal"] = np.where(changed, "Entering " + out["Stage"], "Stable")
    return out


def latest_stage_snapshot(ticker: str, df: pd.DataFrame) -> StageSnapshot:
    processed = compute_market_stages(df)
    if processed.empty:
        raise ValueError(f"No market data available for {ticker}")

    latest = processed.iloc[-1]
    volume_mean_20 = processed["Volume"].rolling(20, min_periods=20).mean().iloc[-1]
    volume_delta = np.nan
    if pd.notna(volume_mean_20) and volume_mean_20 != 0:
        volume_delta = (latest["Volume"] / volume_mean_20 - 1.0) * 100.0

    return StageSnapshot(
        ticker=ticker.upper(),
        as_of=pd.Timestamp(processed.index[-1]),
        close=float(latest["Close"]),
        stage=str(latest["Stage"]),
        transition_signal=str(latest["Transition_Signal"]),
        vwma_8=float(latest["VWMA_8"]) if pd.notna(latest["VWMA_8"]) else np.nan,
        vwma_21=float(latest["VWMA_21"]) if pd.notna(latest["VWMA_21"]) else np.nan,
        vwma_34=float(latest["VWMA_34"]) if pd.notna(latest["VWMA_34"]) else np.nan,
        vma_21=float(latest["VMA_21"]) if pd.notna(latest["VMA_21"]) else np.nan,
        volume_delta_20d=float(volume_delta) if pd.notna(volume_delta) else np.nan,
    )


def scan_phase_shifts(price_data: dict[str, pd.DataFrame], mode: str) -> pd.DataFrame:
    """Find tickers whose current bar just entered the selected structural shift."""
    rows: list[dict[str, object]] = []
    for ticker, df in price_data.items():
        processed = compute_market_stages(df)
        if len(processed) < 35:
            continue

        current = processed.iloc[-1]
        previous = processed.iloc[-2]
        volume_mean_20 = processed["Volume"].rolling(20, min_periods=20).mean().iloc[-1]
        volume_delta = np.nan
        if pd.notna(volume_mean_20) and volume_mean_20 != 0:
            volume_delta = (current["Volume"] / volume_mean_20 - 1.0) * 100.0

        if mode == "Acceleration Shift":
            matched = bool(current["is_bullish"] and current["Close"] >= current["VMA_21"] and not previous["is_bullish"])
            label = "Entry Into Acceleration"
        else:
            matched = bool(current["is_bearish"] and not previous["is_bearish"])
            label = "Entry Into Deceleration"

        if matched:
            rows.append(
                {
                    "Ticker": ticker.upper(),
                    "Phase Shift": label,
                    "Current Stage": current["Stage"],
                    "Close": float(current["Close"]),
                    "Volume Delta 20D": volume_delta,
                    "As Of": pd.Timestamp(processed.index[-1]).date().isoformat(),
                }
            )

    return pd.DataFrame(rows)

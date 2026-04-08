# 💎 The Platinum Strategy: Core & Satellite Integration

**Version**: 1.0 (Integration)
**Objective**: Maximize Total Return (CAGR) while preserving the Risk-Adjusted Stability (Sharpe) of the Golden Ratio.

---

## 1. Executive Summary

The **Platinum Strategy** is an advanced portfolio construction technique that integrates two distinct investment philosophies:
1.  **The Core (75%)**: The **Golden Ratio** strategy (Fusion + Fed + Titan). This provides a robust, "all-weather" foundation that adapts to inflation, deflation, and growth regimes.
2.  **The Satellite (25%)**: The **Ah_Pig (Proxy)** swing trading strategy. This provides "Alpha" by aggressively targeting TQQQ (3x Nasdaq) during confirmed market breakouts.

**Why Platinum?**
*   **Problem**: Pure "Golden Ratio" is excellent but can be "too defensive" during raging bull markets (like 2020 or 2023), leaving returns on the table.
*   **Solution**: Adding a 25% "Alpha Satellite" captures the "Right Tail" upside of bubble rallies without compromising the "Left Tail" protection of the Core.

---

## 2. Strategy Logic

### A. The Core: Golden Ratio (75% Weight)
*   **Logic**: "Sword & Shield"
    *   **Fusion (50%)**: Rotates into defensives (Low Vol) or Defensive Sectors (XLU/XLP/GLD) based on Volatility and Credit regimes.
    *   **Fed Pivot (35%)**: Monitors Liquidity. Shifts to Cash if Fed is tight + Trend is broken.
    *   **Titan (15%)**: Momentum Seeker. Allocates to the strongest asset (Equities, Gold, or Treasury).
*   **Role**: **Preservation**. It ensures we never blow up.

### B. The Satellite: Ah_Pig Swing (25% Weight)
*   **Logic**: "Sniper Entry"
    *   **Regime Filter**: SPY > SMA 200 (Market is Healthy).
    *   **Relative Strength**: QQQ / SPY Ratio > SMA 50 (Tech is Leading).
    *   **Instrument**: **TQQQ (ProShares UltraPro QQQ)** when conditions are met. **Cash (SHV)** otherwise.
*   **Role**: **Acceleration**. It leverages the strongest part of the curve.

---

## 3. Implementation Plan

### Portfolio Construction
We treat the portfolio as two separate sub-accounts (buckets) rebalanced monthly to maintain the 75/25 split.

| Bucket | Strategy | Target Weight | Rebalance Freq |
|:---|:---|:---|:---|
| **Core** | Golden Ratio | **75%** | Monthly |
| **Alpha** | Ah_Pig (Swing) | **25%** | Monthly (Reset weight) / Daily (Signals) |

### Execution Rules
1.  **Daily Monitoring**:
    *   **Ah_Pig Signal (Satellite)**:
        *   **Regime**: SPY Price > SMA 200?
        *   **Relative Strength**: (QQQ / SPY) > SMA 50?
        *   *Action*: If BOTH True → **100% TQQQ** (in Satellite bucket). Else → **100% SHV**.
    *   **Golden Ratio Signal (Core)**:
        *   **Component 1: Fusion (Trend + Macro)**:
            *   *Macro Score*: Based on SPY Trend, Credit Spreads (JNK/IEF), and Economic Cycle (XLI/XLP).
            *   *Attack Mode*: If Macro Score is healthy, buy Top 3 Momentum assets from [QQQ, SMH, XLY, XLC, XLK].
            *   *Defense Mode*: If Macro Score is weak, hold **50% GLD + 50% SHV**.
        *   **Component 2: Fed (Liquidity)**:
            *   *Bull Regime*: 26% QQQ + 26% GLD + 20% Satellite (TQQQ/SHV depending on RSI).
            *   *Bear Regime*: 16% QQQ + 16% GLD + 66% SHV.
        *   **Component 3: Titan (Momentum)**:
            *   Select Winner based on weighted momentum (1m, 3m, 6m).
            *   Candidates: SMH (→USD), QQQ (→QLD), SPY (→SSO), GLD.
2.  **Monthly Rebalancing**:
    *   At month-end, if the Alpha Bucket has grown to >30% of Total Portfolio (due to TQQQ gains), **Harvest Profits**: Sell excess TQQQ and buy Core assets to restore 75/25 split.
    *   If Alpha Bucket has shrunk to <20%, **Refill**: Sell Core assets to restore 25% Alpha weight.

---

## 4. Backtest Specifications

*   **Period**: Jan 2010 - Present.
*   **Execution Lag**: **1 Day** (Signals calculated at Close T, Executed at Close T+1).
*   **Transaction Costs**: **0.10% (10bps)** per turnover (Slippage + Commission).
*   **Data Source**: Adjusted Close (Dividends Reinvested).

## 5. Expected Performance Profile

*   **Bull Market (2017, 2020, 2023)**: Platinum > Golden Ratio (Due to TQQQ exposure).
*   **Bear Market (2022)**: Platinum < Golden Ratio (Slightly deeper drawdown due to TQQQ volatility before stop-out).
*   **Sideways Market (2015)**: Platinum ~ Golden Ratio.

---
**Status**: Ready for Live Tracking.

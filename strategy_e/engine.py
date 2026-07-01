"""
strategy_e/engine.py
====================
System E — NQ100 Momentum Strategy Engine

Logic (matches Alpha Trading Desk / system_e_trap_regime_backtest.py exactly):
  - Universe  : NQ100 tickers (91-stock list, same as backtest)
  - Momentum  : 44-week total-return minus 4-week total-return (skip-month)
  - Breadth   : ≥40% of NQ100 must have positive momentum to deploy
  - Regime    : BULL  → SPY above 200d MA  AND  NQ100 EW above 200d MA
                BEAR  → either index below 200d MA  (or breadth < 40%)
  - BULL alloc: 35% SPY + 21.7% each top-3 momentum picks  (= 100.1% ≈ 100%)
  - BEAR alloc: 10% SPY + 90% IEF (bonds)

Public API
----------
  NQ100_TICKERS                              list[str]
  compute_system_e_signal(df)  -> dict
  backtest_system_e(df)        -> pd.DataFrame   (weekly equity curve)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── NQ100 Universe ─────────────────────────────────────────────────────────────
NQ100_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA",
    "AVGO", "COST", "NFLX", "TMUS", "AMD",  "PEP",  "LIN",  "CSCO",
    "ADBE", "TXN",  "QCOM", "INTU", "ISRG", "AMGN", "CMCSA","HON",
    "AMAT", "MU",   "LRCX", "PANW", "KLAC", "CDNS", "SNPS", "MELI",
    "REGN", "GILD", "ADI",  "MDLZ", "VRTX", "ADP",  "SBUX", "CTAS",
    "MNST", "ORLY", "KDP",  "PAYX", "MCHP", "CPRT", "FTNT", "ODFL",
    "IDXX", "NXPI", "ROST", "BIIB", "DXCM", "EXC",  "FAST", "MRNA",
    "PCAR", "GEHC", "CEG",  "CTSH", "DLTR", "FANG", "TEAM", "VRSK",
    "TTWO", "ZS",   "CRWD", "WDAY", "DDOG", "ILMN", "WBD",  "LCID",
    "SIRI", "RIVN", "INTC", "PYPL", "EBAY", "DOCU", "OKTA", "ZM",
    "ABNB", "HOOD", "COIN", "RBLX", "ENPH", "SEDG", "ALGN", "LULU",
    "MTCH", "NTES", "JD",   "BIDU", "PDD",
]

# ── Constants ──────────────────────────────────────────────────────────────────
BREADTH_MIN  = 0.40          # minimum fraction with positive momentum to deploy
BULL_SPY_WT  = 0.350         # SPY weight in BULL regime
BULL_STK_WT  = 0.217         # per-stock weight (×3) in BULL regime
BEAR_SPY_WT  = 0.100         # SPY weight in BEAR regime
BEAR_IEF_WT  = 0.900         # IEF weight in BEAR regime
INIT_EQUITY  = 100_000.0


def _weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily DataFrame to weekly Friday close, forward-filling."""
    return df.resample("W-FRI").last().ffill()


def _available_nq(df: pd.DataFrame) -> list[str]:
    """Return NQ100 tickers present in df columns."""
    return [t for t in NQ100_TICKERS if t in df.columns]


# ── Live Signal ────────────────────────────────────────────────────────────────

def compute_system_e_signal(df: pd.DataFrame) -> dict:
    """
    Compute the current System E signal from daily market data.

    Parameters
    ----------
    df : pd.DataFrame
        Daily price DataFrame from get_clean_master().
        Must contain 'SPY' and as many NQ100 tickers as possible.

    Returns
    -------
    dict with keys:
        regime       : "BULL" | "BEAR"
        top3         : list[str]           (3 ticker symbols)
        breadth_pct  : float               (e.g. 0.62 → 62%)
        spy_200ma    : float
        nq_ew_200ma  : float
        spy_price    : float
        nq_ew_price  : float
        mom_scores   : dict[str, float]    (all tickers ranked)
        allocation   : dict[str, float]    (ticker → weight)
        bull_gate    : bool                (dual MA gate)
        breadth_gate : bool
    """
    if df is None or len(df) < 210:
        return _empty_signal()

    df = df.copy().ffill()
    nq_tickers = _available_nq(df)

    if not nq_tickers or "SPY" not in df.columns:
        return _empty_signal()

    # ── Resample to weekly ────────────────────────────────────────────
    w = _weekly(df)

    spy_w  = w["SPY"]
    nq_w   = w[[t for t in nq_tickers if t in w.columns]]

    # NQ100 equal-weight index (proxy)
    nq_ew = nq_w.mean(axis=1)

    # ── 200-day MA gate (use daily for accuracy) ──────────────────────
    spy_ma200    = df["SPY"].rolling(200).mean().iloc[-1]
    spy_price    = df["SPY"].iloc[-1]

    # NQ100 EW daily
    nq_ew_daily   = df[[t for t in nq_tickers if t in df.columns]].mean(axis=1)
    nq_ew_ma200   = nq_ew_daily.rolling(200).mean().iloc[-1]
    nq_ew_price   = nq_ew_daily.iloc[-1]

    bull_gate = (spy_price > spy_ma200) and (nq_ew_price > nq_ew_ma200)

    # ── Momentum scores (44w – 4w, skip-month) ───────────────────────
    if len(w) < 46:
        return _empty_signal()

    mom44     = nq_w.iloc[-1] / nq_w.iloc[-45] - 1   # 44-week return
    mom4      = nq_w.iloc[-1] / nq_w.iloc[-5]  - 1   # 4-week return  (skip)
    mom_score = (mom44 - mom4).dropna()

    breadth_pct = float((mom_score > 0).sum() / len(mom_score)) if len(mom_score) > 0 else 0.0
    breadth_gate = breadth_pct >= BREADTH_MIN

    # ── Top-3 picks ───────────────────────────────────────────────────
    top3 = list(mom_score.nlargest(3).index) if len(mom_score) >= 5 else []

    # ── Regime ───────────────────────────────────────────────────────
    regime = "BULL" if (bull_gate and breadth_gate and len(top3) == 3) else "BEAR"

    # ── Allocation ───────────────────────────────────────────────────
    if regime == "BULL":
        alloc: dict[str, float] = {"SPY": BULL_SPY_WT}
        for t in top3:
            alloc[t] = BULL_STK_WT
        cash = max(0.0, 1.0 - sum(alloc.values()))
        if cash > 0:
            alloc["CASH"] = round(cash, 4)
    else:
        alloc = {"SPY": BEAR_SPY_WT, "IEF": BEAR_IEF_WT}

    return {
        "regime":       regime,
        "top3":         top3,
        "breadth_pct":  round(breadth_pct * 100, 1),
        "spy_200ma":    round(float(spy_ma200), 2),
        "spy_price":    round(float(spy_price), 2),
        "nq_ew_200ma":  round(float(nq_ew_ma200), 4),
        "nq_ew_price":  round(float(nq_ew_price), 4),
        "bull_gate":    bull_gate,
        "breadth_gate": breadth_gate,
        "mom_scores":   {t: round(float(v), 4) for t, v in mom_score.sort_values(ascending=False).items()},
        "allocation":   alloc,
    }


def _empty_signal() -> dict:
    return {
        "regime": "UNKNOWN", "top3": [], "breadth_pct": 0.0,
        "spy_200ma": 0.0, "spy_price": 0.0,
        "nq_ew_200ma": 0.0, "nq_ew_price": 0.0,
        "bull_gate": False, "breadth_gate": False,
        "mom_scores": {}, "allocation": {},
    }


# ── 20-Year Backtest ───────────────────────────────────────────────────────────

def backtest_system_e(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full System E 20-year weekly backtest.

    Parameters
    ----------
    df : pd.DataFrame
        Daily price DataFrame from get_clean_master().

    Returns
    -------
    pd.DataFrame with columns:
        Date (index), System_E, Buy_Hold_SPY
    """
    if df is None or len(df) < 400:
        return pd.DataFrame()

    df = df.copy().ffill()
    nq_tickers = _available_nq(df)

    if not nq_tickers or "SPY" not in df.columns:
        return pd.DataFrame()

    # Ensure IEF present; if not, use 0 return for bear leg
    has_ief = "IEF" in df.columns

    w = _weekly(df)
    spy_w  = w["SPY"]
    nq_w   = w[[t for t in nq_tickers if t in w.columns]]
    ief_w  = w["IEF"] if has_ief else pd.Series(0.0, index=spy_w.index)

    # NQ100 EW price series
    nq_ew_daily  = df[[t for t in nq_tickers if t in df.columns]].mean(axis=1)
    spy_ma200_d  = df["SPY"].rolling(200).mean()
    nq_ma200_d   = nq_ew_daily.rolling(200).mean()
    spy_ma200_w  = spy_ma200_d.resample("W-FRI").last().reindex(spy_w.index).ffill()
    nq_ma200_w   = nq_ma200_d.resample("W-FRI").last().reindex(spy_w.index).ffill()
    nq_ew_w      = nq_ew_daily.resample("W-FRI").last().reindex(spy_w.index).ffill()

    # Weekly returns
    spy_ret  = spy_w.pct_change()
    ief_ret  = ief_w.pct_change()
    nq_ret   = nq_w.pct_change()

    # Momentum (skip-month: 44w minus 4w)
    mom44    = nq_w / nq_w.shift(44) - 1
    mom4     = nq_w / nq_w.shift(4)  - 1
    mom_sc   = mom44 - mom4

    equity   = [INIT_EQUITY]
    bh_start = spy_w.dropna().iloc[0]

    for i in range(1, len(spy_w)):
        # Signal computed from PREVIOUS week's close (no look-ahead)
        if i < 46:
            equity.append(equity[-1] * (1 + _safe(spy_ret, i) * 0.10 + _safe(ief_ret, i) * 0.90))
            continue

        # Regime gate
        spy_prev   = spy_w.iloc[i - 1]
        nq_ew_prev = nq_ew_w.iloc[i - 1]
        ma_spy     = spy_ma200_w.iloc[i - 1]
        ma_nq      = nq_ma200_w.iloc[i - 1]

        bull_gate = (
            pd.notna(spy_prev) and pd.notna(nq_ew_prev)
            and pd.notna(ma_spy) and pd.notna(ma_nq)
            and spy_prev > ma_spy
            and nq_ew_prev > ma_nq
        )

        # Breadth + top-3
        row = mom_sc.iloc[i - 1].dropna()
        breadth = float((row > 0).sum() / len(row)) if len(row) > 0 else 0.0
        breadth_ok = breadth >= BREADTH_MIN
        top3 = list(row.nlargest(3).index) if len(row) >= 5 else []

        rs = _safe(spy_ret, i)
        ri = _safe(ief_ret, i)

        if bull_gate and breadth_ok and len(top3) == 3:
            stk_r = sum(_safe_col(nq_ret, t, i) for t in top3)
            pret  = BULL_SPY_WT * rs + BULL_STK_WT * stk_r
        else:
            pret = BEAR_SPY_WT * rs + BEAR_IEF_WT * ri

        equity.append(equity[-1] * (1 + pret))

    eq_series = pd.Series(equity, index=spy_w.index, name="System_E")
    bh_series = (INIT_EQUITY * spy_w / spy_w.dropna().iloc[0]).rename("Buy_Hold_SPY")

    result = pd.concat([eq_series, bh_series], axis=1).dropna(how="all")
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(series: pd.Series, i: int) -> float:
    v = series.iloc[i]
    return 0.0 if pd.isna(v) else float(v)


def _safe_col(df: pd.DataFrame, col: str, i: int) -> float:
    if col not in df.columns:
        return 0.0
    v = df[col].iloc[i]
    return 0.0 if pd.isna(v) else float(v)

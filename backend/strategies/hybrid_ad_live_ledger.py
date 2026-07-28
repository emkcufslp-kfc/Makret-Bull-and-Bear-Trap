"""
backend/strategies/hybrid_ad_live_ledger.py
==============================================
Hybrid A/D — LIVE paper-trading ledger since 2026-07-01, $10,000 start.

Combines the E80/R3 blend target (80% Sys E + 20% R3, same as blend_live_ledger)
with a vol-target exposure overlay:
  - Strategy A (bull/low-vol): full 1.0x exposure to the 80/20 blend target
  - Strategy D (bear/high-vol): exposure scaled to min(1, 15%/realized_vol),
    with the unallocated remainder held in CASH (not moved to bonds — this is
    a pure de-leverage overlay, not a reallocation).

Regime switch (production signal, backtested Calmar 1.404): SPY 8-week
realized vol >= 15% for 2 CONSECUTIVE Friday closes to ENTER Strategy D;
exits immediately (no debounce) once vol drops back below 15%.

Deterministic reconstruction — same pattern as blend_live_ledger.py: every
call replays all signal-driven executions from START to the latest available
price bar, so the ledger is always self-consistent with market data.

Execution model:
  Sys E (80%)   : Tuesday close signal  -> execute next trading day (Wed)
  R3    (20%)   : Thursday close signal -> execute next trading day (Fri)
  Vol regime    : Friday close signal   -> execute next trading day (Mon)
  Each execution rebalances the WHOLE portfolio to the current
  expo * (0.8*w_E + 0.2*w_R3) target.

Fills approximated at execution day's CLOSE. Costs: 0.08%/side (0.16% RT).

Public API
----------
  START, START_CAPITAL
  build_hybrid_live_state(df) -> dict with keys:
      trades, ledger, positions, target, w_e, w_r3, expo,
      sys_e_sig, pending, summary, cash, shares, regime_history
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from backend.strategies.blend_live_ledger import (
    START, START_CAPITAL, COST_SIDE, STOP_ATR_MULT,
    _load_r3_history, _r3_signal_for, _sys_e_weights, _r3_weights, _combine,
)

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "data" / "live_trading_hybrid_ad"

VOL_THRESHOLD = 0.15    # 15% annualised SPY 8wk realized vol
VOL_WINDOW = 8
DEBOUNCE_WEEKS = 2       # consecutive weeks above threshold to ENTER Strategy D


# ── Vol regime (2-week entry confirmation, immediate exit) ────────────────────
def _spy_vol_regime(df: pd.DataFrame, asof: pd.Timestamp) -> tuple[float, float, bool]:
    spy_w = df["SPY"].loc[:asof].resample("W-FRI").last().dropna()
    ret = spy_w.pct_change().dropna()
    if len(ret) < VOL_WINDOW + 1:
        return np.nan, np.nan, False
    rvol = ret.rolling(VOL_WINDOW).std() * np.sqrt(52)
    vol_now, vol_prev = float(rvol.iloc[-1]), float(rvol.iloc[-2])
    in_defensive = (vol_now >= VOL_THRESHOLD) and (vol_prev >= VOL_THRESHOLD)
    return vol_now, vol_prev, in_defensive


def _exposure_scalar(vol_now: float) -> float:
    if np.isnan(vol_now) or vol_now <= 0:
        return 1.0
    return float(min(1.0, VOL_THRESHOLD / vol_now))


# ── Execution calendar: Tue SysE, Thu R3, Fri vol-regime check ────────────────
def _exec_events(df: pd.DataFrame) -> list[dict]:
    idx = df.index
    last_bar = idx[-1]
    today = pd.Timestamp(dt.date.today())
    events: list[dict] = []
    d = (START - pd.Timedelta(days=7)).normalize()
    while d <= today + pd.Timedelta(days=7):
        for sig_wd, sleeve in ((1, "SYS_E"), (3, "R3"), (4, "VOL_REGIME")):  # Tue,Thu,Fri
            sig_day = d + pd.Timedelta(days=(sig_wd - d.weekday()) % 7)
            sig_bars = idx[idx <= sig_day]
            if len(sig_bars) == 0:
                continue
            sig_asof = sig_bars[-1]
            exec_bars = idx[idx > sig_day]
            exec_date = exec_bars[0] if len(exec_bars) > 0 else None
            if exec_date is None or exec_date < START:
                exec_date_eff = idx[idx >= START]
                if len(exec_date_eff) == 0 or sig_day >= START:
                    if exec_date is None and sig_day < today:
                        events.append({"sleeve": sleeve, "signal_date": sig_asof,
                                       "exec_date": None, "pending": True})
                    continue
                exec_date = exec_date_eff[0]
            if exec_date > last_bar:
                if sig_day <= today:
                    events.append({"sleeve": sleeve, "signal_date": sig_asof,
                                   "exec_date": None, "pending": True})
                continue
            if sig_day > today:
                continue
            events.append({"sleeve": sleeve, "signal_date": sig_asof,
                           "exec_date": exec_date, "pending": False})
        d += pd.Timedelta(days=7)

    seen: dict = {}
    for e in events:
        key = (e["sleeve"], e["exec_date"])
        if key not in seen or (e["signal_date"] > seen[key]["signal_date"]):
            seen[key] = e
    out = sorted(seen.values(), key=lambda e: (e["exec_date"] or pd.Timestamp.max, e["sleeve"]))
    return [e for e in out if e["pending"] or e["exec_date"] >= START]


# ── Main reconstruction ───────────────────────────────────────────────────────
def build_hybrid_live_state(df: pd.DataFrame) -> dict:
    df = df.copy().ffill()
    idx = df.index
    hist = _load_r3_history()

    events = _exec_events(df)
    executed = [e for e in events if not e["pending"]]
    pending = [e for e in events if e["pending"]]

    cash = START_CAPITAL
    shares: dict[str, float] = {}
    basis: dict[str, float] = {}
    entry_date: dict[str, pd.Timestamp] = {}
    w_e_cur: dict = {"CASH": 1.0}
    w_r3_cur: dict = {"IEF": 1.0}
    expo_cur = 1.0
    vol_now_cur, vol_prev_cur, in_def_cur = np.nan, np.nan, False
    sys_e_last_sig: dict = {}
    trades: list[dict] = []
    realized_pnl = 0.0
    regime_hist: list[dict] = []

    def px(t: str, day: pd.Timestamp) -> float | None:
        if t in df.columns:
            v = df.at[day, t] if day in df.index else np.nan
            if pd.notna(v):
                return float(v)
        return None

    def port_value(day: pd.Timestamp) -> float:
        v = cash
        for t, sh in shares.items():
            p = px(t, day)
            if p:
                v += sh * p
        return v

    by_day: dict[pd.Timestamp, list[dict]] = {}
    for e in executed:
        by_day.setdefault(e["exec_date"], []).append(e)

    for day in sorted(by_day):
        reasons = []
        for e in by_day[day]:
            if e["sleeve"] == "SYS_E":
                w_e_cur, sys_e_last_sig = _sys_e_weights(df, e["signal_date"])
                reasons.append(f"Sys E signal {e['signal_date'].date()}")
            elif e["sleeve"] == "R3":
                r3sig = _r3_signal_for(e["signal_date"], df, hist)
                w_r3_cur = _r3_weights(r3sig, set(df.columns))
                reasons.append(f"R3 signal {e['signal_date'].date()} "
                               f"({r3sig.get('deploy_pct', '?')}% deploy)")
            else:  # VOL_REGIME
                vol_now_cur, vol_prev_cur, in_def_cur = _spy_vol_regime(df, e["signal_date"])
                expo_cur = _exposure_scalar(vol_now_cur) if in_def_cur else 1.0
                regime_hist.append({
                    "Date": e["signal_date"],
                    "Vol_%": round(vol_now_cur * 100, 2) if not np.isnan(vol_now_cur) else None,
                    "In_Defensive": in_def_cur, "Exposure": round(expo_cur, 3),
                })
                reasons.append(
                    f"Vol regime {e['signal_date'].date()}: "
                    f"{'DEFENSIVE' if in_def_cur else 'BULL'} "
                    f"(vol {vol_now_cur * 100:.1f}% expo {expo_cur:.2f})"
                    if not np.isnan(vol_now_cur) else "vol regime insufficient history"
                )

        blend_target = _combine(w_e_cur, w_r3_cur)
        target = {t: round(w * expo_cur, 4) for t, w in blend_target.items()}
        pv = port_value(day)
        reason = " + ".join(reasons) + f" -> rebalance (expo {expo_cur:.2f})"

        deltas = []
        tickers = set(shares) | {t for t in target if t != "CASH"}
        for t in sorted(tickers):
            p = px(t, day)
            if not p:
                continue
            tgt_val = target.get(t, 0.0) * pv
            cur_val = shares.get(t, 0.0) * p
            dv = tgt_val - cur_val
            if abs(dv) < max(1.0, 0.0005 * pv):
                continue
            deltas.append((t, p, dv))
        for t, p, dv in sorted(deltas, key=lambda x: x[2]):
            if dv < 0:  # SELL
                sh = -dv / p
                sold = min(sh, shares.get(t, 0.0))
                if sold <= 0:
                    continue
                cost = sold * p * COST_SIDE
                frac = sold / shares[t]
                cb = basis[t] * frac
                realized = sold * p - cb - cost
                realized_pnl += realized
                basis[t] -= cb
                shares[t] -= sold
                cash += sold * p - cost
                trades.append({"Date": day, "Ticker": t, "Action": "SELL",
                               "Shares": round(sold, 4), "Price": round(p, 2),
                               "Value": round(sold * p, 2), "Cost": round(cost, 2),
                               "Realized_PnL": round(realized, 2), "Reason": reason})
                if shares[t] < 1e-6:
                    shares.pop(t, None); basis.pop(t, None); entry_date.pop(t, None)
            else:  # BUY
                cost_est = dv * COST_SIDE
                spend = min(dv, max(0.0, cash - cost_est))
                if spend < 1.0:
                    continue
                sh_b = spend / p
                cost = spend * COST_SIDE
                cash -= spend + cost
                if t not in shares:
                    entry_date[t] = day
                shares[t] = shares.get(t, 0.0) + sh_b
                basis[t] = basis.get(t, 0.0) + spend + cost
                trades.append({"Date": day, "Ticker": t, "Action": "BUY",
                               "Shares": round(sh_b, 4), "Price": round(p, 2),
                               "Value": round(spend, 2), "Cost": round(cost, 2),
                               "Realized_PnL": 0.0, "Reason": reason})

    trades_df = pd.DataFrame(trades)

    # ── Daily ledger since START ──────────────────────────────────────────────
    live_days = idx[idx >= START]
    rows = []
    sh_t: dict[str, float] = {}
    cash_t = START_CAPITAL
    trades_by_day = trades_df.groupby("Date") if not trades_df.empty else None
    prev_eq = START_CAPITAL
    for day in live_days:
        desc = []
        flow = 0.0
        if day == live_days[0]:
            desc.append(f"Opening deposit ${START_CAPITAL:,.0f}")
        if trades_by_day is not None and day in trades_by_day.groups:
            for _, tr in trades_by_day.get_group(day).iterrows():
                sgn = 1 if tr["Action"] == "BUY" else -1
                sh_t[tr["Ticker"]] = sh_t.get(tr["Ticker"], 0.0) + sgn * tr["Shares"]
                if sh_t.get(tr["Ticker"], 0.0) < 1e-6:
                    sh_t.pop(tr["Ticker"], None)
                cash_t += (-1 if tr["Action"] == "BUY" else 1) * tr["Value"] - tr["Cost"]
                flow += (-1 if tr["Action"] == "BUY" else 1) * tr["Value"] - tr["Cost"]
            n = len(trades_by_day.get_group(day))
            desc.append(f"{n} rebalance trade{'s' if n > 1 else ''}")
        pos_val = sum(sh * float(df.at[day, t]) for t, sh in sh_t.items()
                      if t in df.columns and pd.notna(df.at[day, t]))
        eq = cash_t + pos_val
        rows.append({
            "Date": day, "Description": "; ".join(desc) or "Mark-to-market",
            "Cash_Flow": round(flow, 2), "Cash_Balance": round(cash_t, 2),
            "Positions_Value": round(pos_val, 2), "Total_Equity": round(eq, 2),
            "Daily_PnL": round(eq - prev_eq, 2),
            "Total_PnL": round(eq - START_CAPITAL, 2),
        })
        prev_eq = eq
    ledger_df = pd.DataFrame(rows)

    # ── Open positions w/ stops & floating P&L ────────────────────────────────
    last_day = live_days[-1] if len(live_days) else idx[-1]
    blend_target_now = _combine(w_e_cur, w_r3_cur)
    target_now = {t: round(w * expo_cur, 4) for t, w in blend_target_now.items()}
    cash_target_pct = round((1.0 - sum(target_now.values())) * 100, 1)

    pos_rows = []
    for t, sh in sorted(shares.items(), key=lambda x: -x[1] * (px(x[0], last_day) or 0)):
        p = px(t, last_day) or 0.0
        mv = sh * p
        cb = basis.get(t, 0.0)
        ed = entry_date.get(t, START)
        avg_entry = cb / sh if sh else 0.0
        ser = df[t].loc[:last_day].dropna()
        atr = float(ser.diff().abs().tail(14).mean()) if len(ser) > 15 else 0.0
        hi = float(ser.loc[ed:].max()) if len(ser.loc[ed:]) else p
        stop = max(0.0, hi - STOP_ATR_MULT * atr)
        total_val = cash + sum(shares[x] * (px(x, last_day) or 0) for x in shares)
        pos_rows.append({
            "Ticker": t, "Target_W_%": round(target_now.get(t, 0.0) * 100, 1),
            "Actual_W_%": round(mv / total_val * 100, 1) if total_val else 0.0,
            "Shares": round(sh, 4), "Avg_Entry": round(avg_entry, 2),
            "Last_Price": round(p, 2), "Market_Value": round(mv, 2),
            "Float_PnL_$": round(mv - cb, 2),
            "Float_PnL_%": round((mv / cb - 1) * 100, 2) if cb else 0.0,
            "Stop_(2xATR14)": round(stop, 2),
            "Stop_Dist_%": round((p / stop - 1) * 100, 1) if stop else np.nan,
            "Entry_Date": ed.date(),
        })
    positions_df = pd.DataFrame(pos_rows)

    # ── Pending executions ────────────────────────────────────────────────────
    pend = []
    for e in pending:
        if e["sleeve"] == "R3":
            r3sig = _r3_signal_for(e["signal_date"], df, hist)
            note = (f"R3 deploy {r3sig.get('deploy_pct', '?')}%, picks: "
                   f"{', '.join(r3sig.get('top3_picks') or []) or '—'}")
        elif e["sleeve"] == "SYS_E":
            _, s = _sys_e_weights(df, e["signal_date"])
            note = f"Sys E {s.get('regime')}, top3: {', '.join(s.get('top3') or [])}"
        else:
            vn, vp, ind = _spy_vol_regime(df, e["signal_date"])
            note = (f"Vol regime check: {'-> DEFENSIVE' if ind else '-> stays BULL'} "
                    f"(vol {vn*100:.1f}%)" if not np.isnan(vn) else "Insufficient vol history")
        pend.append({"sleeve": e["sleeve"], "signal_date": e["signal_date"].date(), "note": note})

    # ── Summary / risk ────────────────────────────────────────────────────────
    eq_ser = ledger_df.set_index("Date")["Total_Equity"] if not ledger_df.empty \
        else pd.Series(dtype=float)
    ret = eq_ser.pct_change().dropna()
    dd = (eq_ser / eq_ser.cummax() - 1).min() * 100 if len(eq_ser) else 0.0
    equity_expo = sum(w for t, w in target_now.items() if t not in ("IEF", "CASH"))
    summary = {
        "as_of": str(last_day.date()),
        "equity": float(eq_ser.iloc[-1]) if len(eq_ser) else START_CAPITAL,
        "cash": float(ledger_df["Cash_Balance"].iloc[-1]) if not ledger_df.empty else START_CAPITAL,
        "total_pnl": float(eq_ser.iloc[-1] - START_CAPITAL) if len(eq_ser) else 0.0,
        "total_pnl_pct": float((eq_ser.iloc[-1] / START_CAPITAL - 1) * 100) if len(eq_ser) else 0.0,
        "daily_pnl": float(ledger_df["Daily_PnL"].iloc[-1]) if not ledger_df.empty else 0.0,
        "realized_pnl": round(realized_pnl, 2),
        "max_dd_pct": round(float(dd), 2),
        "daily_vol_pct": round(float(ret.std() * 100), 2) if len(ret) > 1 else 0.0,
        "equity_exposure_pct": round(equity_expo * 100, 1),
        "cash_target_pct": cash_target_pct,
        "costs_paid": round(float(trades_df["Cost"].sum()), 2) if not trades_df.empty else 0.0,
        "n_trades": len(trades_df),
        "vol_now": vol_now_cur, "vol_prev": vol_prev_cur, "in_defensive": in_def_cur,
        "exposure_scalar": expo_cur,
    }

    # persist
    try:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        if not trades_df.empty:
            trades_df.to_csv(LIVE_DIR / "trades.csv", index=False)
        if not ledger_df.empty:
            ledger_df.to_csv(LIVE_DIR / "ledger.csv", index=False)
        if not positions_df.empty:
            positions_df.to_csv(LIVE_DIR / "positions.csv", index=False)
        if regime_hist:
            pd.DataFrame(regime_hist).to_csv(LIVE_DIR / "regime_history.csv", index=False)
    except Exception:
        pass

    return {
        "trades": trades_df, "ledger": ledger_df, "positions": positions_df,
        "target": target_now, "w_e": w_e_cur, "w_r3": w_r3_cur, "expo": expo_cur,
        "sys_e_sig": sys_e_last_sig, "pending": pend, "summary": summary,
        "cash": cash, "shares": shares, "regime_history": regime_hist,
    }

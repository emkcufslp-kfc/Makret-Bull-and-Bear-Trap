"""
backend/strategies/blend_live_ledger.py
========================================
E80/R3 Blend — LIVE paper-trading ledger since 2026-07-01, $10,000 start.

Deterministic reconstruction: every call replays all signal-driven executions
from START to the latest available price bar, so the ledger is always
self-consistent with the market data (no mutable state to corrupt).
Results are also persisted to data/live_trading/*.csv for record-keeping.

Execution model (mirrors the backtest schedule):
  Sys E (80%) : Tuesday close signal  → execute next trading day (Wed)
  R3    (20%) : Thursday close signal → execute next trading day (Fri)
  Each execution rebalances the WHOLE portfolio to the combined
  0.8·w_E + 0.2·w_R3 target (this doubles as the weekly drift correction).

Fills are approximated at the execution day's CLOSE (the master file stores
daily closes only). Costs: 0.08% per side on traded notional (=0.16% RT).
Fractional shares allowed (percent-based tracking, like the backtest).

Stops are INDICATIVE only (not part of the system): trailing
highest-close-since-entry − 2 × ATR-proxy(14) (mean abs daily close change).

Public API
----------
  START, START_CAPITAL
  build_live_state(df) -> dict with keys:
      trades      : pd.DataFrame   (trade log)
      ledger      : pd.DataFrame   (cash + equity ledger, daily)
      positions   : pd.DataFrame   (open positions w/ entry, stop, float P&L)
      target      : dict           (latest combined target weights)
      pending     : list[dict]     (signals awaiting execution)
      summary     : dict           (equity, cash, P&L, risk stats)
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "data" / "live_trading"
R3_CACHE = ROOT / "data" / "r3_signal_cache.json"
R3_HIST = LIVE_DIR / "r3_signal_history.json"
WEEKLY_SIG_CSV = ROOT / "exports" / "trap_regime_backtest" / "system_e_weekly_signals.csv"

START = pd.Timestamp("2026-07-01")          # first live trading day
START_CAPITAL = 10_000.0
COST_SIDE = 0.0008                          # 0.08% per side (0.16% round trip)
W_E, W_R3 = 0.80, 0.20
R3_BULL_SPY, R3_BULL_STK = 0.350, 0.217
TARGET_VOL = 0.12
STOP_ATR_MULT = 2.0


# ── R3 signal history ─────────────────────────────────────────────────────────
def _load_r3_history() -> dict:
    if R3_HIST.exists():
        try:
            return json.loads(R3_HIST.read_text())
        except Exception:
            pass
    return {}


def _sync_r3_cache_into_history() -> dict:
    """Append the current Thursday cache snapshot to the persistent history."""
    hist = _load_r3_history()
    if R3_CACHE.exists():
        try:
            snap = json.loads(R3_CACHE.read_text())
            d = snap.get("date")
            if d and d not in hist:
                hist[d] = {
                    "deploy_pct": float(snap.get("deploy_pct", 50.0)),
                    "top3_picks": snap.get("top3_picks") or [],
                    "source": "cache",
                }
                LIVE_DIR.mkdir(parents=True, exist_ok=True)
                R3_HIST.write_text(json.dumps(hist, indent=2))
        except Exception:
            pass
    return hist


def _reconstruct_r3(thu: pd.Timestamp, df: pd.DataFrame) -> dict:
    """Fallback: rebuild the Thursday R3 signal from the weekly signals CSV
    (BullTrap/BearTrap scores) + SPY weekly vol — mirrors pages/15 logic."""
    out = {"deploy_pct": 50.0, "top3_picks": [], "source": "fallback-default"}
    try:
        ws = pd.read_csv(WEEKLY_SIG_CSV, index_col="Date", parse_dates=True).sort_index()
        # weekly row for the week containing this Thursday (W-FRI stamp)
        wk = ws.loc[ws.index >= thu]
        if wk.empty:
            wk = ws.iloc[[-1]]
        row = wk.iloc[0]
        bt = float(row.get("BullTrap_Score", 2.5))
        bear = float(row.get("BearTrap_Score", 0.5) or 0.5)
        spy_w = df["SPY"].loc[:thu].resample("W-FRI").last().pct_change().dropna().tail(4)
        vol = spy_w.std() * np.sqrt(52) if len(spy_w) >= 2 else TARGET_VOL
        net_conf = (bt / 10.0) * (1.0 - bear)
        vs = vol / TARGET_VOL
        vol_adj = net_conf * min(1.5, 1.0 / vs) if vs > 0 else net_conf
        deploy = float(np.clip(0.10 + vol_adj * 0.90, 0.10, 1.00))
        picks = row.get("top3_picks")
        out = {
            "deploy_pct": round(deploy * 100, 1),
            "top3_picks": str(picks).split(",") if isinstance(picks, str) and picks else [],
            "source": "reconstructed",
        }
    except Exception:
        pass
    return out


def _r3_signal_for(thu: pd.Timestamp, df: pd.DataFrame, hist: dict) -> dict:
    key = thu.date().isoformat()
    if key in hist:
        return hist[key]
    return _reconstruct_r3(thu, df)


# ── Sleeve target weights ─────────────────────────────────────────────────────
def _sys_e_weights(df: pd.DataFrame, asof: pd.Timestamp) -> tuple[dict, dict]:
    from strategy_e.engine import compute_system_e_signal
    sig = compute_system_e_signal(df.loc[:asof])
    return dict(sig.get("allocation") or {"CASH": 1.0}), sig


def _r3_weights(r3sig: dict, available: set | None = None) -> dict:
    deploy = float(r3sig.get("deploy_pct", 50.0)) / 100.0
    picks = [p.strip() for p in (r3sig.get("top3_picks") or []) if p and p.strip()]
    w: dict[str, float] = {}
    if len(picks) >= 3:
        w["SPY"] = R3_BULL_SPY * deploy
        for t in picks[:3]:
            # untracked ticker → weight goes to SPY (sleeve stays deployed)
            key = t if (available is None or t in available) else "SPY"
            w[key] = w.get(key, 0.0) + R3_BULL_STK * deploy
    else:
        w["SPY"] = deploy
    w["IEF"] = w.get("IEF", 0.0) + (1.0 - deploy)
    return w


def _combine(w_e: dict, w_r3: dict) -> dict:
    combined: dict[str, float] = {}
    for t, wt in w_e.items():
        combined[t] = combined.get(t, 0.0) + W_E * wt
    for t, wt in w_r3.items():
        combined[t] = combined.get(t, 0.0) + W_R3 * wt
    return {t: round(w, 4) for t, w in combined.items() if w > 0.0005}


# ── Execution calendar ────────────────────────────────────────────────────────
def _exec_events(df: pd.DataFrame) -> list[dict]:
    """All signal→execution events from START to the last price bar.
    Returns executed events (exec date has a price bar) and marks the rest
    pending."""
    idx = df.index
    last_bar = idx[-1]
    hist = _sync_r3_cache_into_history()
    events: list[dict] = []

    # walk calendar weeks from the week before START
    d = (START - pd.Timedelta(days=7)).normalize()
    today = pd.Timestamp(dt.date.today())
    while d <= today + pd.Timedelta(days=7):
        for sig_wd, sleeve in ((1, "SYS_E"), (3, "R3")):      # Tue=1, Thu=3
            sig_day = d + pd.Timedelta(days=(sig_wd - d.weekday()) % 7)
            sig_bars = idx[idx <= sig_day]
            if len(sig_bars) == 0:
                continue
            sig_asof = sig_bars[-1]                            # last bar ≤ signal day
            exec_bars = idx[idx > sig_day]
            exec_date = exec_bars[0] if len(exec_bars) > 0 else None
            if exec_date is None or exec_date < START:
                # initial deployment: signals locked before START execute on
                # the first live day
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

    # de-duplicate (same sleeve + exec date → keep latest signal)
    seen: dict = {}
    for e in events:
        key = (e["sleeve"], e["exec_date"])
        if key not in seen or (e["signal_date"] > seen[key]["signal_date"]):
            seen[key] = e
    out = sorted(seen.values(), key=lambda e: (e["exec_date"] or pd.Timestamp.max,
                                               e["sleeve"]))
    # drop executed events whose exec_date < START (pre-live)
    return [e for e in out if e["pending"] or e["exec_date"] >= START]


# ── Main reconstruction ───────────────────────────────────────────────────────
def build_live_state(df: pd.DataFrame) -> dict:
    df = df.copy().ffill()
    idx = df.index
    hist = _load_r3_history()

    events = _exec_events(df)
    executed = [e for e in events if not e["pending"]]
    pending = [e for e in events if e["pending"]]

    # state
    cash = START_CAPITAL
    shares: dict[str, float] = {}
    basis: dict[str, float] = {}          # cost basis $ per ticker
    entry_date: dict[str, pd.Timestamp] = {}
    w_e_cur: dict = {"CASH": 1.0}
    w_r3_cur: dict = {"IEF": 1.0}
    sys_e_last_sig: dict = {}
    trades: list[dict] = []
    realized_pnl = 0.0

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

    # group events by exec_date → single rebalance per day
    by_day: dict[pd.Timestamp, list[dict]] = {}
    for e in executed:
        by_day.setdefault(e["exec_date"], []).append(e)

    for day in sorted(by_day):
        reasons = []
        for e in by_day[day]:
            if e["sleeve"] == "SYS_E":
                w_e_cur, sys_e_last_sig = _sys_e_weights(df, e["signal_date"])
                reasons.append(f"Sys E signal {e['signal_date'].date()}")
            else:
                r3sig = _r3_signal_for(e["signal_date"], df, hist)
                w_r3_cur = _r3_weights(r3sig, set(df.columns))
                reasons.append(f"R3 signal {e['signal_date'].date()} "
                               f"({r3sig.get('deploy_pct', '?')}% deploy)")
        target = _combine(w_e_cur, w_r3_cur)
        pv = port_value(day)
        reason = " + ".join(reasons) + " → rebalance to 80/20 target"

        # sells first, then buys
        deltas = []
        tickers = set(shares) | {t for t in target if t != "CASH"}
        for t in sorted(tickers):
            p = px(t, day)
            if not p:
                continue
            tgt_val = target.get(t, 0.0) * pv
            cur_val = shares.get(t, 0.0) * p
            dv = tgt_val - cur_val
            if abs(dv) < max(1.0, 0.0005 * pv):     # ignore dust < $1 / 5bps
                continue
            deltas.append((t, p, dv))
        for t, p, dv in sorted(deltas, key=lambda x: x[2]):   # sells first
            sh = dv / p
            cost = abs(dv) * COST_SIDE
            if dv < 0:  # SELL
                sold = min(-sh, shares.get(t, 0.0))
                if sold <= 0:
                    continue
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
            else:       # BUY
                spend = min(dv, max(0.0, cash - cost))
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
    # replay day by day for valuation (holdings only change on trade days,
    # which we already applied — so rebuild holdings timeline)
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
    target_now = _combine(w_e_cur, w_r3_cur)
    pos_rows = []
    for t, sh in sorted(shares.items(), key=lambda x: -x[1] * (px(x[0], last_day) or 0)):
        p = px(t, last_day) or 0.0
        mv = sh * p
        cb = basis.get(t, 0.0)
        ed = entry_date.get(t, START)
        avg_entry = cb / sh if sh else 0.0
        # ATR proxy: 14-day mean abs close-to-close change
        ser = df[t].loc[:last_day].dropna()
        atr = float(ser.diff().abs().tail(14).mean()) if len(ser) > 15 else 0.0
        hi = float(ser.loc[ed:].max()) if len(ser.loc[ed:]) else p
        stop = max(0.0, hi - STOP_ATR_MULT * atr)
        pos_rows.append({
            "Ticker": t, "Target_W_%": round(target_now.get(t, 0.0) * 100, 1),
            "Actual_W_%": round(mv / (cash + sum(shares[x] * (px(x, last_day) or 0) for x in shares)) * 100, 1),
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
            note = f"R3 deploy {r3sig.get('deploy_pct', '?')}%, picks: " \
                   f"{', '.join(r3sig.get('top3_picks') or []) or '—'}"
        else:
            _, s = _sys_e_weights(df, e["signal_date"])
            note = f"Sys E {s.get('regime')}, top3: {', '.join(s.get('top3') or [])}"
        pend.append({"sleeve": e["sleeve"], "signal_date": e["signal_date"].date(),
                     "note": note})

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
        "costs_paid": round(float(trades_df["Cost"].sum()), 2) if not trades_df.empty else 0.0,
        "n_trades": len(trades_df),
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
    except Exception:
        pass

    return {
        "trades": trades_df, "ledger": ledger_df, "positions": positions_df,
        "target": target_now, "w_e": w_e_cur, "w_r3": w_r3_cur,
        "sys_e_sig": sys_e_last_sig, "pending": pend, "summary": summary,
        "cash": cash, "shares": shares,
    }

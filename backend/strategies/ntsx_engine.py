import yfinance as yf
import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import date

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# OUTPUT PATH — adjust if needed
# ---------------------------------------------------------
OUTPUT_JS = Path(__file__).parent.parent / "data" / "Multi_indicator" / "ntsx_data.js"

# ---------------------------------------------------------
# 1. DOWNLOAD DATA
# ---------------------------------------------------------
TODAY = date.today().strftime('%Y-%m-%d')
print(f"Downloading data up to {TODAY}...")

tickers = ['SPY', 'TLT', 'BIL', 'DFSVX', 'RYMFX']
data = yf.download(tickers, start="2007-03-01", end=TODAY, auto_adjust=True)['Close']
returns = data.pct_change().dropna()

# ---------------------------------------------------------
# 2. CONSTRUCT PROXIES
# ---------------------------------------------------------
returns['NTSX_Proxy'] = (0.90 * returns['SPY']) + (0.60 * returns['TLT']) - (0.50 * returns['BIL'])
assets = returns[['NTSX_Proxy', 'DFSVX', 'RYMFX']]

ASSET_NAMES  = ['NTSX (Proxy)', 'AVWS (DFSVX)', 'KMLM (RYMFX)']
target_weights  = np.array([0.55, 0.12, 0.33])
lower_bounds    = np.array([0.50, 0.09, 0.28])
upper_bounds    = np.array([0.60, 0.15, 0.38])
current_weights = target_weights.copy()

# ---------------------------------------------------------
# 3. REBALANCING SIMULATION
# ---------------------------------------------------------
portfolio_daily_returns = []
rebalance_log = []   # full detail per rebalance
rebalance_dates = []

for date_idx, daily_return in assets.iterrows():
    asset_returns = daily_return.values
    updated_values = current_weights * (1 + asset_returns)
    daily_port_return = np.sum(updated_values) - 1
    portfolio_daily_returns.append(daily_port_return)
    current_weights = updated_values / np.sum(updated_values)

    if np.any(current_weights < lower_bounds) or np.any(current_weights > upper_bounds):
        date_str = date_idx.strftime('%Y-%m-%d')
        rebalance_dates.append(date_str)

        triggers = []
        for i, name in enumerate(ASSET_NAMES):
            if current_weights[i] < lower_bounds[i]:
                triggers.append({
                    "asset": name,
                    "direction": "Below Lower Band",
                    "weight_at_trigger": round(current_weights[i] * 100, 2),
                    "band": f"{int(lower_bounds[i]*100)}%\u2013{int(upper_bounds[i]*100)}%"
                })
            elif current_weights[i] > upper_bounds[i]:
                triggers.append({
                    "asset": name,
                    "direction": "Above Upper Band",
                    "weight_at_trigger": round(current_weights[i] * 100, 2),
                    "band": f"{int(lower_bounds[i]*100)}%\u2013{int(upper_bounds[i]*100)}%"
                })

        rebalance_log.append({
            "date": date_str,
            "month": date_idx.strftime('%b %Y'),
            "triggers": triggers,
            "weights_before": [round(w * 100, 2) for w in current_weights],
            "weights_after":  [round(w * 100, 2) for w in target_weights]
        })

        current_weights = target_weights.copy()

returns['Optimized_Portfolio'] = portfolio_daily_returns
returns.index = pd.to_datetime(returns.index)

print(f"Total rebalances: {len(rebalance_dates)}")

# ---------------------------------------------------------
# 4. EQUITY CURVE (indexed to 100)
# ---------------------------------------------------------
port_cum = (1 + returns['Optimized_Portfolio']).cumprod() * 100
spy_cum  = (1 + returns['SPY']).cumprod() * 100

equity_data = [
    {"date": d.strftime('%Y-%m-%d'),
     "port": round(float(p), 4),
     "spy":  round(float(s), 4)}
    for d, p, s in zip(returns.index, port_cum, spy_cum)
]

# ---------------------------------------------------------
# 5. YEAR-BY-YEAR METRICS
# ---------------------------------------------------------
yearly_rows = []
for year_end, grp in returns[['SPY', 'Optimized_Portfolio']].resample('YE'):
    if grp.empty:
        continue
    yr = year_end.year

    spy_ret  = (1 + grp['SPY']).prod() - 1
    port_ret = (1 + grp['Optimized_Portfolio']).prod() - 1

    spy_cum_y  = (1 + grp['SPY']).cumprod()
    port_cum_y = (1 + grp['Optimized_Portfolio']).cumprod()

    spy_dd  = ((spy_cum_y  / spy_cum_y.cummax())  - 1).min()
    port_dd = ((port_cum_y / port_cum_y.cummax()) - 1).min()

    winner = "Portfolio" if port_ret >= spy_ret else "SPY"

    yearly_rows.append({
        "year":       yr,
        "port_return": round(port_ret * 100, 2),
        "spy_return":  round(spy_ret  * 100, 2),
        "port_maxdd":  round(port_dd  * 100, 2),
        "spy_maxdd":   round(spy_dd   * 100, 2),
        "winner":      winner
    })

# ---------------------------------------------------------
# 6. OVERALL METRICS
# ---------------------------------------------------------
n_years = len(returns) / 252

spy_cagr  = ((1 + returns['SPY']).prod() ** (1 / n_years)) - 1
port_cagr = ((1 + returns['Optimized_Portfolio']).prod() ** (1 / n_years)) - 1

spy_dd_series  = (1 + returns['SPY']).cumprod()
port_dd_series = (1 + returns['Optimized_Portfolio']).cumprod()

spy_max_dd  = ((spy_dd_series  / spy_dd_series.cummax())  - 1).min()
port_max_dd = ((port_dd_series / port_dd_series.cummax()) - 1).min()

rf = 0.04  # approximate risk-free rate

spy_excess  = returns['SPY'] - rf / 252
port_excess = returns['Optimized_Portfolio'] - rf / 252

spy_sharpe  = round(float(spy_excess.mean()  / spy_excess.std()  * np.sqrt(252)), 2)
port_sharpe = round(float(port_excess.mean() / port_excess.std() * np.sqrt(252)), 2)

spy_down  = spy_excess[spy_excess < 0].std()  * np.sqrt(252)
port_down = port_excess[port_excess < 0].std() * np.sqrt(252)

spy_sortino  = round(float(spy_excess.mean()  * 252 / spy_down),  2)
port_sortino = round(float(port_excess.mean() * 252 / port_down), 2)

spy_calmar  = round(float(spy_cagr  / abs(spy_max_dd)),  2)
port_calmar = round(float(port_cagr / abs(port_max_dd)), 2)

metrics = {
    "portfolio": {
        "cagr":    round(port_cagr * 100, 2),
        "max_dd":  round(port_max_dd * 100, 2),
        "sharpe":  port_sharpe,
        "sortino": port_sortino,
        "calmar":  port_calmar
    },
    "spy": {
        "cagr":    round(spy_cagr * 100, 2),
        "max_dd":  round(spy_max_dd * 100, 2),
        "sharpe":  spy_sharpe,
        "sortino": spy_sortino,
        "calmar":  spy_calmar
    }
}

# ---------------------------------------------------------
# 7. CURRENT STATUS (live weights + action signal)
# ---------------------------------------------------------
last_date = returns.index[-1].strftime('%Y-%m-%d')
last_reb   = rebalance_log[-1] if rebalance_log else None
last_reb_date = last_reb["date"] if last_reb else "N/A"

days_since = (pd.Timestamp(last_date) - pd.Timestamp(last_reb_date)).days if last_reb else 0

curr_w = [round(float(w) * 100, 2) for w in current_weights]
dist_lower = [round(curr_w[i] - lower_bounds[i] * 100, 2) for i in range(3)]
dist_upper = [round(upper_bounds[i] * 100 - curr_w[i], 2) for i in range(3)]

# Determine signal
out_of_band  = any(current_weights[i] < lower_bounds[i] or current_weights[i] > upper_bounds[i] for i in range(3))
near_edge    = any(min(dist_lower[i], dist_upper[i]) < 1.0 for i in range(3))

if out_of_band:
    action_signal = "REBALANCE"
    action_color  = "#ff1744"
elif near_edge:
    action_signal = "WATCH"
    action_color  = "#ffea00"
else:
    action_signal = "HOLD"
    action_color  = "#00e676"

current_status = {
    "as_of_date":             last_date,
    "action_signal":          action_signal,
    "action_color":           action_color,
    "asset_names":            ASSET_NAMES,
    "current_weights":        curr_w,
    "target_weights":         [55.0, 12.0, 33.0],
    "lower_bounds":           [50.0,  9.0, 28.0],
    "upper_bounds":           [60.0, 15.0, 38.0],
    "distance_to_lower":      dist_lower,
    "distance_to_upper":      dist_upper,
    "days_since_last_rebalance": days_since,
    "last_rebalance_date":    last_reb_date
}

# ---------------------------------------------------------
# 8. WRITE ntsx_data.js
# ---------------------------------------------------------
OUTPUT_JS.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_JS, "w", encoding="utf-8") as f:
    f.write(f"// ntsx_data.js — Auto-generated on {date.today()}\n")
    f.write(f"const NTSX_CURRENT     = {json.dumps(current_status, indent=2)};\n")
    f.write(f"const NTSX_REBALANCES  = {json.dumps(rebalance_log,   indent=2)};\n")
    f.write(f"const NTSX_EQUITY      = {json.dumps(equity_data,     indent=2)};\n")
    f.write(f"const NTSX_YEARLY      = {json.dumps(yearly_rows,     indent=2)};\n")
    f.write(f"const NTSX_METRICS     = {json.dumps(metrics,         indent=2)};\n")
    f.write(f"const NTSX_REB_DATES   = {json.dumps(rebalance_dates, indent=2)};\n")

print(f"\n✅ ntsx_data.js written to: {OUTPUT_JS}")
print(f"   Equity points : {len(equity_data)}")
print(f"   Rebalances    : {len(rebalance_log)}")
print(f"   Yearly rows   : {len(yearly_rows)}")
print(f"   Action signal : {action_signal}")
print(f"   As of date    : {last_date}")

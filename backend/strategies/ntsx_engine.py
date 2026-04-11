import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. DEFINE PROXIES & DOWNLOAD FACTUAL DATA
# ---------------------------------------------------------
print("Downloading 19 years of factual daily market data (2007-2026)...")
tickers = ['SPY', 'TLT', 'BIL', 'DFSVX', 'RYMFX']
# RYMFX inception was March 2007. We run through the current date in 2026.
data = yf.download(tickers, start="2007-03-01", end="2026-03-28", auto_adjust=True)['Close']

# Calculate daily returns and drop missing data
returns = data.pct_change().dropna()

# ---------------------------------------------------------
# 2. CONSTRUCT THE SYNTHETIC PROXIES
# ---------------------------------------------------------
# NTSX Proxy: 90% S&P 500 + 60% Long Treasury - 50% Cash Borrowing Cost
returns['NTSX_Proxy'] = (0.90 * returns['SPY']) + (0.60 * returns['TLT']) - (0.50 * returns['BIL'])

# Isolate our three portfolio assets
# NTSX_Proxy (55%), DFSVX for AVWS (12%), RYMFX for KMLM (33%)
assets = returns[['NTSX_Proxy', 'DFSVX', 'RYMFX']]

# ---------------------------------------------------------
# 3. INITIALIZE THE TOLERANCE BAND REBALANCING ENGINE
# ---------------------------------------------------------
target_weights = np.array([0.55, 0.12, 0.33])
current_weights = np.array([0.55, 0.12, 0.33])

# Tolerance Bands
# NTSX: 50% to 60%
# AVWS: 9% to 15%
# KMLM: 28% to 38%
lower_bounds = np.array([0.50, 0.09, 0.28])
upper_bounds = np.array([0.60, 0.15, 0.38])

portfolio_daily_returns = []
rebalance_dates = []

print("Running Threshold Rebalancing Simulation...\n")

# Iterate through every single trading day
for date, daily_return in assets.iterrows():
    asset_returns = daily_return.values
    
    # Calculate how much each slice of the pie grew or shrank today
    updated_values = current_weights * (1 + asset_returns)
    
    # The total return of the portfolio for the day
    daily_port_return = np.sum(updated_values) - 1
    portfolio_daily_returns.append(daily_port_return)
    
    # Calculate the new drifted weights at the end of the day
    current_weights = updated_values / np.sum(updated_values)
    
    # CHECK TOLERANCE BANDS: If any asset is out of bounds, trigger a rebalance
    if np.any(current_weights < lower_bounds) or np.any(current_weights > upper_bounds):
        rebalance_dates.append(date.strftime('%Y-%m-%d'))
        # Execute Rebalance: Reset weights back to target
        current_weights = target_weights.copy()

# Add the simulated portfolio returns back to the dataframe
returns['Optimized_Portfolio'] = portfolio_daily_returns

# ---------------------------------------------------------
# 4. CALCULATE YEAR-BY-YEAR METRICS
# ---------------------------------------------------------
print("--- EXACT REBALANCE TRIGGERS LOG ---")
print(f"Total Rebalances Executed over 19 Years: {len(rebalance_dates)}")
# Show the first 10 and last 5 to avoid flooding the console
if len(rebalance_dates) > 15:
    print("First 10 Triggers: ", rebalance_dates[:10])
    print("Most Recent 5 Triggers: ", rebalance_dates[-5:])
else:
    print("Dates: ", rebalance_dates)

print("\n--- YEAR-BY-YEAR PERFORMANCE SUMMARY ---")
# Group data by year
returns.index = pd.to_datetime(returns.index)
yearly_data = returns[['SPY', 'Optimized_Portfolio']].resample('Y')

yearly_metrics = []

for year, data_group in yearly_data:
    if data_group.empty: continue
    
    # Annual Return
    spy_ret = (1 + data_group['SPY']).prod() - 1
    port_ret = (1 + data_group['Optimized_Portfolio']).prod() - 1
    
    # Annual Max Drawdown
    spy_cum = (1 + data_group['SPY']).cumprod()
    spy_dd = (spy_cum / spy_cum.cummax()) - 1
    spy_max_dd = spy_dd.min()
    
    port_cum = (1 + data_group['Optimized_Portfolio']).cumprod()
    port_dd = (port_cum / port_cum.cummax()) - 1
    port_max_dd = port_dd.min()
    
    yearly_metrics.append({
        'Year': year.year,
        'SPY Return': f"{spy_ret*100:.2f}%",
        'Port Return': f"{port_ret*100:.2f}%",
        'SPY MaxDD': f"{spy_max_dd*100:.2f}%",
        'Port MaxDD': f"{port_max_dd*100:.2f}%"
    })

df_yearly = pd.DataFrame(yearly_metrics)
print(df_yearly.to_string(index=False))

# ---------------------------------------------------------
# 5. OVERALL COMPOUND METRICS
# ---------------------------------------------------------
total_spy_cagr = ((1 + returns['SPY']).prod() ** (252/len(returns))) - 1
total_port_cagr = ((1 + returns['Optimized_Portfolio']).prod() ** (252/len(returns))) - 1

spy_total_dd = ((1 + returns['SPY']).cumprod() / (1 + returns['SPY']).cumprod().cummax() - 1).min()
port_total_dd = ((1 + returns['Optimized_Portfolio']).cumprod() / (1 + returns['Optimized_Portfolio']).cumprod().cummax() - 1).min()

print("\n--- 19-YEAR TOTAL COMPOUND METRICS ---")
print(f"SPY CAGR: {total_spy_cagr*100:.2f}% | SPY Overall Max DD: {spy_total_dd*100:.2f}%")
print(f"Portfolio CAGR: {total_port_cagr*100:.2f}% | Portfolio Overall Max DD: {port_total_dd*100:.2f}%")
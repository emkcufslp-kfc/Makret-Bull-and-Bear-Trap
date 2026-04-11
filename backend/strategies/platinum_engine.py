
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# ==========================================
# 0. CONFIGURATION
# ==========================================
START_DATE = '2010-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
OUTPUT_DIR = 'Platinum_Balanced_Results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

COST_BPS = 10  # 10 bps
TRADE_LAG = 1  # 1-Day Lag

# ==========================================
# 1. DATA ENGINE (Unified)
# ==========================================
def fetch_data():
    tickers = [
        'SPY', 'QQQ', 'GLD', 'SHV', 'TQQQ', 'BIL', 'USD', 'QLD', 'SSO', 
        'VGK', 'VNQ', 'GSG', 'TLT', 'HYG', 'LQD', 'IEF', 'EEM', 'EFA', 'AGG', 
        'XLY', 'XLC', 'XLP', 'XLU', 'XLV', 'USMV', 'JNK', 
        'VT', 'EWT', 'IWM', 'VSS', 'FEZ', 'EWJ', 'VIG', 'XLK'
    ]
    tickers = list(set(tickers))
    print(f"Fetching {len(tickers)} tickers...")
    
    try:
        df = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False, group_by='ticker', auto_adjust=True)
    except:
        df = yf.download(tickers, start=START_DATE, end=END_DATE, progress=False, group_by='ticker')
        
    prices = pd.DataFrame()
    for t in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if t in df.columns.levels[0]:
                    prices[t] = df[t]['Close']
                elif t in df.columns.levels[1]:
                    prices[t] = df.xs(t, axis=1, level=1)['Close']
            else:
                if t in df.columns:
                     prices[t] = df['Close']
        except KeyError:
            pass
            
    # Cleaning
    prices.ffill(inplace=True)
    prices.dropna(axis=1, how='all', inplace=True)
    prices.bfill(inplace=True) # Backfill for TQQQ inception gaps if any
    
    return prices

# ==========================================
# 2. STRATEGY LOGIC
# ==========================================
def run_backtest(weights, prices, cost_bps=COST_BPS, lag=TRADE_LAG):
    # Align
    common_idx = weights.index.intersection(prices.index)
    w = weights.reindex(common_idx).fillna(0.0)
    p = prices.reindex(common_idx)
    
    # Lag: Shift weights forward
    executed_w = w.shift(lag).fillna(0.0)
    
    # Returns
    asset_ret = p.pct_change().fillna(0.0)
    port_ret = (executed_w * asset_ret).sum(axis=1)
    
    # Costs
    turnover = executed_w.diff().abs().sum(axis=1)
    net_ret = port_ret - (turnover * (cost_bps/10000.0))
    
    equity = (1 + net_ret).cumprod()
    equity = equity / equity.iloc[0] * 10000 
    
    return equity, net_ret

def get_platinum_weights(prices):
    dates = prices.index
    
    # --- A. Golden Ratio (75%) ---
    # Simplified Adapter
    # Fusion (50%)
    spy_sma = prices['SPY'].rolling(200).mean()
    is_bull = prices['SPY'] > spy_sma
    
    # MRPS Proxy
    # Need JNK/IEF, XLI/XLP
    if all(x in prices.columns for x in ['JNK', 'IEF', 'XLI', 'XLP']):
        cr = (prices['JNK']/prices['IEF'])
        cr = (cr > cr.rolling(200).mean()).astype(int)*100
        eco = (prices['XLI']/prices['XLP'])
        eco = (eco > eco.rolling(200).mean()).astype(int)*100
        mrps = 0.25*(is_bull.astype(int)*100) + 0.25*cr + 0.25*eco + 12.5
    else:
        mrps = pd.Series(100, index=dates)

    # Fusion Loop (Monthly)
    m_mom = prices.pct_change(126).resample('ME').last()
    m_mrps = mrps.resample('ME').last()
    
    w_fusion = pd.DataFrame(0.0, index=m_mom.index, columns=prices.columns)
    
    bull_cands = ['QQQ', 'SMH', 'XLY', 'XLC', 'XLK']
    bear_cands = ['XLP', 'XLU', 'XLV', 'GLD', 'USMV']
    
    for d in m_mom.index:
        m = m_mrps.loc[d]
        mode = "DEFENSE"
        if m >= 40: mode = "ATTACK" # Allow Attack more easily
        
        if mode == "ATTACK":
            avail = [c for c in bull_cands if c in m_mom.columns]
            sc = m_mom.loc[d][avail].sort_values(ascending=False)
            picks = sc[sc>0].index[:3].tolist()
            wt = 1.0/3.0
            for p in picks: w_fusion.loc[d, p] += wt
            if len(picks)<3: w_fusion.loc[d, 'SHV'] += (3-len(picks))*wt
        else:
            w_fusion.loc[d, 'SHV'] += 0.5
            w_fusion.loc[d, 'GLD'] += 0.5
            
    daily_fusion = w_fusion.reindex(dates, method='ffill')
    
    # Win-Win (Fed+Titan)
    # Re-implement Fed+Titan Logic roughly (85/15)
    w_ww = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    
    # Fed Pivot logic
    delta = prices['QQQ'].diff()
    up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
    rsi = 100 - (100/(1 + up.rolling(5).mean()/down.rolling(5).mean()))
    rsi_sat = rsi < 40
    
    # Vectorized Fed Weights
    # If Bull: 0.85 * (0.266 QQQ, 0.266 GLD, 0.266 SHV or TQQQ)
    # If Bear: 0.85 * (0.166 QQQ, 0.166 GLD, 0.666 SHV)
    
    alloc_bull = is_bull
    alloc_bear = ~is_bull
    
    w_ww.loc[alloc_bull, 'QQQ'] += 0.266 * 0.85
    w_ww.loc[alloc_bull, 'GLD'] += 0.266 * 0.85
    w_ww.loc[alloc_bull & rsi_sat, 'TQQQ'] += 0.20 * 0.85
    w_ww.loc[alloc_bull & ~rsi_sat, 'SHV'] += 0.20 * 0.85
    
    w_ww.loc[alloc_bear, 'QQQ'] += 0.166 * 0.85
    w_ww.loc[alloc_bear, 'GLD'] += 0.166 * 0.85
    w_ww.loc[alloc_bear, 'SHV'] += 0.666 * 0.85
    
    # Titan (15%)
    # R12/6/3/1
    t_cands = ['SMH', 'QQQ', 'SPY', 'GLD']
    avail_t = [t for t in t_cands if t in prices.columns]
    score_t = (12*prices[avail_t].pct_change(21) + 4*prices[avail_t].pct_change(63)).resample('ME').last()
    
    titan_map = {'SMH':'USD', 'QQQ':'QLD', 'SPY':'SSO', 'GLD':'GLD'}
    w_titan = pd.DataFrame(0.0, index=score_t.index, columns=prices.columns)
    
    for d in score_t.index:
        sc = score_t.loc[d].dropna() # Fix: Drop NaNs
        if sc.empty:
            w_titan.loc[d, 'BIL'] += 0.15
            continue
            
        best = sc.idxmax()
        if sc[best] > 0:
            tgt = titan_map.get(best, 'SHV')
            w_titan.loc[d, tgt] += 0.15
        else:
            w_titan.loc[d, 'BIL'] += 0.15
            
    daily_titan = w_titan.reindex(dates, method='ffill')
    w_ww = w_ww + daily_titan # Add Titan to WinWin
    
    # Golden Ratio Mix: 50% Fusion + 50% WinWin
    gr_weights = 0.5 * daily_fusion + 0.5 * w_ww
    
    # --- B. Ah_Pig (25%) - Balanced QLD Edition ---
    w_ah = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    rs_line = prices['QQQ'] / prices['SPY']
    rs_bull = rs_line > rs_line.rolling(50).mean()
    sig_ah = is_bull & rs_bull
    
    targ_ah = 'QLD' if 'QLD' in prices else 'QQQ'
    w_ah.loc[sig_ah, targ_ah] = 1.0
    w_ah.loc[~sig_ah, 'SHV'] = 1.0
    
    # --- C. Platinum Balanced (75/25) ---
    plat_weights = 0.75 * gr_weights + 0.25 * w_ah
    
    return plat_weights

def generate_transaction_log(weights, prices, initial_capital=10000.0):
    print("Generating Advanced Transaction Log with P&L...")
    
    # We need to simulate the portfolio day-by-day to track "Average Cost" and "Realized P&L"
    # This transforms the vectorized weights back into a "Trade Ledger"
    
    # 1. Reindex weights and prices to ensure alignment
    common = weights.index.intersection(prices.index)
    w_df = weights.reindex(common).fillna(0.0)
    p_df = prices.reindex(common).ffill()
    
    log = []
    
    # State
    # positions = { 'Asset': { 'shares': float, 'avg_cost': float } }
    positions = {}
    current_equity = initial_capital
    
    # Detect rebalance days (where weights change significantly)
    # To save time, we only loop status changes, but we need daily equity for sizing
    # Let's iterate all days? simpler.
    
    prev_w = pd.Series(0.0, index=weights.columns)
    
    # Pre-calculate equity curve for sizing approximations (simpler than full sim)
    # We essentially "trust" the vectorized equity for the total portfolio value
    # But for accurate P&L we should track share counts.
    # Let's do a pure shadow portfolio.
    
    cash = initial_capital
    
    for d in common:
        # Check if weights changed from Yesterday
        # (This implies we trade AT CLOSE of d to match target weights for d+1?)
        # Our Backtest Logic: executed_w = w.shift(1). 
        # So weights[d] are the Target Weights for Close of d.
        
        target_w = w_df.loc[d]
        
        # If no change, skip (unless we want to track drifts, but let's stick to signals)
        # Threshold 1% change to reduce noise
        diff = (target_w - prev_w).abs().sum()
        if diff < 0.005 and d != common[0]:
            # Update equity value based on price moves for next step sizing
            portfolio_val = cash
            for t, pos in positions.items():
                if pos['shares'] > 0:
                    px = p_df.at[d, t]
                    portfolio_val += pos['shares'] * px
            current_equity = portfolio_val
            continue
            
        # Rebalancing Triggered
        
        # 1. Update Portfolio Value (Mark-to-Market)
        portfolio_val = cash
        for t, pos in positions.items():
            if pos['shares'] > 0:
                px = p_df.at[d, t]
                portfolio_val += pos['shares'] * px
        
        current_equity = portfolio_val
        
        # 2. Execute Sales First (to free up cash)
        # Target Value per asset
        target_vals = target_w * current_equity
        
        # Identify Sells
        for t in weights.columns:
            px = p_df.at[d, t]
            if pd.isna(px) or px <= 0: continue
            
            curr_pos = positions.get(t, {'shares': 0.0, 'avg_cost': 0.0})
            curr_val = curr_pos['shares'] * px
            tgt_val = target_vals.get(t, 0.0)
            
            if tgt_val < curr_val - 1.0: # Sell (tolerance $1)
                sell_val = curr_val - tgt_val
                shares_to_sell = sell_val / px
                
                # Update Cash
                cash += sell_val
                
                # P&L Calculation
                # Realized P&L = (Exit Price - Entry Price) * Shares
                pnl = (px - curr_pos['avg_cost']) * shares_to_sell
                pnl_pct = (px / curr_pos['avg_cost']) - 1 if curr_pos['avg_cost'] > 0 else 0
                
                # Log
                log.append({
                    'Date': d.date(),
                    'Asset': t,
                    'Action': 'SELL', # Or "CLOSE" / "TRIM"
                    'Price': px,
                    'Shares': -shares_to_sell,
                    'Value': -sell_val,
                    'Realized_PnL': pnl,
                    'PnL_Pct': pnl_pct,
                    'Type': 'CLOSE' if tgt_val < 1.0 else 'REDUCE'
                })
                
                # Update Position
                positions[t]['shares'] -= shares_to_sell
                if positions[t]['shares'] < 0.001:
                    del positions[t]

        # 3. Execute Buys
        for t in weights.columns:
            px = p_df.at[d, t]
            if pd.isna(px) or px <= 0: continue
            
            curr_pos = positions.get(t, {'shares': 0.0, 'avg_cost': 0.0})
            curr_val = curr_pos['shares'] * px
            tgt_val = target_vals.get(t, 0.0)
            
            if tgt_val > curr_val + 1.0: # Buy
                buy_val = tgt_val - curr_val
                
                # ── FIX: Never buy more than available cash ──
                buy_val = min(buy_val, max(cash, 0.0))
                if buy_val <= 0:
                    continue
                    
                shares_to_buy = buy_val / px
                
                # ── FIX: Subtract cash ──
                cash -= buy_val
                
                # Update Avg Cost
                total_shares = curr_pos['shares'] + shares_to_buy
                new_cost = ((curr_pos['shares'] * curr_pos['avg_cost']) + (shares_to_buy * px)) / total_shares
                
                positions[t] = {'shares': total_shares, 'avg_cost': new_cost}
                
                log.append({
                    'Date': d.date(),
                    'Asset': t,
                    'Action': 'BUY',
                    'Price': round(px, 4),
                    'Shares': round(shares_to_buy, 4),
                    'Value': round(buy_val, 2),
                    'Realized_PnL': 0.0,
                    'PnL_Pct': 0.0,
                    'Type': 'OPEN' if curr_pos['shares'] == 0 else 'ADD'
                })
                
        prev_w = target_w.copy()
        
    return pd.DataFrame(log)

def monte_carlo_simulation(returns, n_sims=1000, n_years=10):
    print(f"Running Monte Carlo ({n_sims} runs)...")
    sim_stats = []
    
    # Bootstrap chunks or simple daily returns?
    # Simple Bootstrap of daily returns breaks autocorrelation (Momentum needs autocorrelation).
    # We should use "Block Bootstrap" or just randomize "Start Date" (which we do in next section).
    # Actual Monte Carlo for Momentum is tricky. 
    # Let's do "Randomized Start Date + Duration" logic check instead?
    # Or just simple reshuffling to see if parameters are lucky? 
    # NO. Momentum relies on serial correlation. Shuffling kills it.
    
    # Better idea: "Block Bootstrap" (shuffle 3-month blocks).
    block_size = 63 # 3 months
    n_blocks = int(len(returns) / block_size)
    
    blocks = [returns.iloc[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    
    for i in range(n_sims):
        # Random sample blocks with replacement using indices
        indices = np.random.randint(0, n_blocks, size=n_blocks)
        random_blocks = [blocks[k] for k in indices]
        
        # Concat
        sim_ret = pd.concat((pd.Series(b) if isinstance(b, (list, np.ndarray)) else b for b in random_blocks))
        # Calc Stats
        cum = (1 + sim_ret).cumprod()
        # Handle empty or short sims
        if len(cum) < 252:
            sim_stats.append({'CAGR': 0.0, 'MaxDD': 0.0})
            continue
            
        cagr = cum.iloc[-1]**(252/len(cum)) - 1
        dd = (cum/cum.cummax())-1
        mdd = dd.min()
        
        sim_stats.append({'CAGR': cagr, 'MaxDD': mdd})
        
    df = pd.DataFrame(sim_stats)
    return df

def rolling_start_analysis(equity, months_step=1):
    print("Running Rolling Start Date Analysis...")
    # Resample equity to monthly to pick start dates
    monthly_eq = equity.resample('MS').first()
    
    stats = []
    
    for start_date in monthly_eq.index:
        # Slice equity from start_date to end
        sub_eq = equity.loc[start_date:]
        if len(sub_eq) < 252: continue # Skip if less than 1 year
        
        # Norm to 1.0
        sub_eq = sub_eq / sub_eq.iloc[0]
        
        # Calc stats
        final = sub_eq.iloc[-1]
        days = (sub_eq.index[-1] - sub_eq.index[0]).days
        years = days / 365.25
        cagr = final**(1/years) - 1
        
        dd = (sub_eq / sub_eq.cummax()) - 1
        mdd = dd.min()
        
        stats.append({
            'Start_Date': start_date.date(),
            'Duration_Years': years,
            'CAGR': cagr,
            'MaxDD': mdd,
            'Final_Equity_Factor': final
        })
        
    return pd.DataFrame(stats)

# ==========================================
# 3. MAIN
# ==========================================
def main():
    print(f"Running Platinum Strategy Backtest (1-Day Lag, {COST_BPS}bps Cost)...")
    prices = fetch_data()
    print(f"Data Loaded: {prices.index[0].date()} to {prices.index[-1].date()}")
    
    # Export Data Used
    prices.to_csv(f"{OUTPUT_DIR}/Platinum_Data_Used.csv")
    print(f"Data saved to {OUTPUT_DIR}/Platinum_Data_Used.csv")
    
    # Calculate Weights
    print("Calculating Strategy Weights...")
    plat_weights = get_platinum_weights(prices)
    
    # Run Backtest
    equity, net_ret = run_backtest(plat_weights, prices)
    
    # Benchmark
    bench_ret = prices['SPY'].pct_change().fillna(0.0)
    bench_eq = (1 + bench_ret).cumprod() * 10000
    
    # Metrics
    cagr = (equity.iloc[-1]/equity.iloc[0])**(365.25/(equity.index[-1]-equity.index[0]).days) - 1
    dd = (equity / equity.cummax()) - 1
    max_dd = dd.min()
    sharpe = net_ret.mean()/net_ret.std()*np.sqrt(252)
    
    print("\n" + "="*40)
    print("PLATINUM STRATEGY RESULTS")
    print("="*40)
    print(f"Final Equity: ${equity.iloc[-1]:,.2f}")
    print(f"CAGR:         {cagr:.2%}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print("="*40)

    # Export Daily Equity for Dashboard
    equity_df = pd.DataFrame({'Platinum_Equity': equity, 'Benchmark_Equity': bench_eq})
    equity_df.to_csv(f"{OUTPUT_DIR}/Platinum_Equity.csv")
    
    # Export Weights for Current Status
    plat_weights.to_csv(f"{OUTPUT_DIR}/Platinum_Weights.csv")
    print(f"Daily Equity & Weights saved to {OUTPUT_DIR}/")
    
    # --- DETAILED REPORTS ---
    
    # 1. Transaction Log
    tx_log = generate_transaction_log(plat_weights, prices)
    tx_log.to_csv(f"{OUTPUT_DIR}/Platinum_Transaction_Log.csv", index=False)
    print(f"Transaction Log saved ({len(tx_log)} records)")
    
    # 2. Monthly Returns
    monthly_ret = net_ret.resample('ME').apply(lambda x: (1+x).prod() - 1)
    monthly_table = monthly_ret.groupby([monthly_ret.index.year, monthly_ret.index.month]).sum().unstack()
    monthly_table.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_table['Year Total'] = monthly_ret.resample('YE').apply(lambda x: (1+x).prod() - 1).values
    
    print("\nMonthly Returns:")
    # print(monthly_table.applymap(lambda x: f"{x:.1%}" if pd.notnull(x) else "-"))
    monthly_table.to_csv(f"{OUTPUT_DIR}/Platinum_Monthly_Returns.csv")
    
    # 3. Monte Carlo
    mc_results = monte_carlo_simulation(net_ret)
    mc_results.to_csv(f"{OUTPUT_DIR}/Platinum_MonteCarlo.csv")
    print(f"Monte Carlo Results: CAGR [5%, 95%] = [{mc_results['CAGR'].quantile(0.05):.1%}, {mc_results['CAGR'].quantile(0.95):.1%}]")
    
    # 4. Rolling Start
    roll_stats = rolling_start_analysis(equity)
    roll_stats.to_csv(f"{OUTPUT_DIR}/Platinum_Rolling_Start.csv", index=False)
    worst_start = roll_stats.loc[roll_stats['CAGR'].idxmin()]
    best_start = roll_stats.loc[roll_stats['CAGR'].idxmax()]
    print(f"Worst Start Date: {worst_start['Start_Date']} (CAGR: {worst_start['CAGR']:.1%})")
    print(f"Best Start Date:  {best_start['Start_Date']} (CAGR: {best_start['CAGR']:.1%})")
    
    # CHARTING
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    
    # Equity
    axes[0].plot(equity, label='Platinum Mix', color='#2ecc71', linewidth=2)
    axes[0].plot(bench_eq, label='S&P 500', color='gray', linestyle='--', alpha=0.5)
    axes[0].set_yscale('log')
    axes[0].set_title('Growth of $10,000')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Drawdown
    axes[1].plot(dd, label='Platinum Drawdown', color='red', alpha=0.7)
    axes[1].fill_between(dd.index, dd, 0, color='red', alpha=0.1)
    axes[1].set_title('Drawdown Profile')
    axes[1].grid(True, alpha=0.3)
    
    # Rolling Sharpe (or Annual Returns bar chart?)
    # Let's do Annual Returns Bar
    annual_ret = monthly_table['Year Total']
    colors = ['green' if x > 0 else 'red' for x in annual_ret]
    axes[2].bar(annual_ret.index, annual_ret, color=colors, alpha=0.7)
    axes[2].set_title('Annual Returns')
    axes[2].axhline(0, color='black', linewidth=1)
    axes[2].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Platinum_Performance_Pack.png")
    print(f"\nChart Pack saved to {OUTPUT_DIR}/Platinum_Performance_Pack.png")

if __name__ == "__main__":
    main()

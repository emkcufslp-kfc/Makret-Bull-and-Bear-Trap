import sys
import os
import json
import pandas as pd
import numpy as np

# Set paths for relative imports
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
STRAT_DIR = os.path.join(CURR_DIR, "strategies")
sys.path.append(STRAT_DIR)

# Import the engine from the strategies folder
try:
    import platinum_engine as pbb
except ImportError:
    # Handle the case where rename happened
    import strategies.platinum_engine as pbb

def run_monte_carlo(net_ret, n_sims=1000):
    block_size = 63
    ret_arr = net_ret.values
    n = len(ret_arr)
    n_blocks = n // block_size
    if n_blocks == 0: return []
    blocks = [ret_arr[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    results = []
    for _ in range(n_sims):
        idxs = np.random.randint(0, n_blocks, size=n_blocks)
        sim = np.concatenate([blocks[k] for k in idxs])
        cum = np.cumprod(1 + sim)
        years = len(sim) / 252
        cagr = cum[-1]**(1/years) - 1 if years > 0 else 0
        dd = (cum / np.maximum.accumulate(cum)) - 1
        mdd = float(dd.min())
        sharpe = float(np.mean(sim) / np.std(sim) * np.sqrt(252)) if np.std(sim) > 0 else 0
        results.append({'cagr': round(cagr*100, 2), 'maxdd': round(mdd*100, 2), 'sharpe': round(sharpe, 2)})
    return results

def rolling_monthly(equity, net_ret):
    monthly_eq = equity.resample('ME').last()
    rolling = []
    if len(monthly_eq) < 13: return []
    for i in range(12, len(monthly_eq)):
        start_val = monthly_eq.iloc[i-12]
        end_val = monthly_eq.iloc[i]
        ret_12 = (end_val / start_val) - 1
        rolling.append({
            'date': monthly_eq.index[i].strftime('%Y-%m-%d'),
            'return_12m': round(ret_12 * 100, 2)
        })
    return rolling

def export_data():
    print("Fetching global factual market data...")
    prices = pbb.fetch_data()

    print("Calculating Strategy Weights...")
    plat_weights = pbb.get_platinum_weights(prices)

    print("Running Backtest Simulator...")
    equity, net_ret = pbb.run_backtest(plat_weights, prices)

    print("Generating Transaction Log...")
    transactions = pbb.generate_transaction_log(plat_weights, prices)

    # Output paths (Save to data/Platinum_Results/ for Streamlit Python reader)
    # AND to root/data/Multi_indicator/platinum_data.js for JS dashboards
    DATA_DIR = os.path.abspath(os.path.join(CURR_DIR, "../data/Platinum_Results"))
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save CSVs for Python Dashboard
    equity_df = pd.DataFrame({'Platinum_Equity': equity}) # Add benchmark if needed
    equity_df.to_csv(os.path.join(DATA_DIR, "Platinum_Equity.csv"))
    plat_weights.to_csv(os.path.join(DATA_DIR, "Platinum_Weights.csv"))
    transactions.to_csv(os.path.join(DATA_DIR, "Platinum_Transaction_Log.csv"), index=False)
    prices.to_csv(os.path.join(DATA_DIR, "Platinum_Data_Used.csv"))

    # Generate JS for HTML Dashboards
    dates = [d.strftime('%Y-%m-%d') for d in equity.index]
    js_tx = transactions.to_dict(orient='records')
    # Cleanup for JSON
    for tx in js_tx:
        for k, v in list(tx.items()):
            if isinstance(v, float) and np.isnan(v): tx[k] = 0
            if k == 'Date': tx[k] = str(v)

    stats_dict = {'last_updated': dates[-1]} # Simplified for brevity, add more as per old script

    out_file_js = os.path.join(CURR_DIR, "../data/Multi_indicator", "platinum_data.js")
    os.makedirs(os.path.dirname(out_file_js), exist_ok=True)
    with open(out_file_js, 'w', encoding='utf-8') as f:
        f.write("const PLAT_DATES = " + json.dumps(dates) + ";\n")
        f.write("const PLAT_EQUITY = " + json.dumps(equity.tolist()) + ";\n")
        f.write("const PLAT_STATS = " + json.dumps(stats_dict) + ";\n")
        # Add more PLAT_ constants as needed matching the original export_platinum_data.py

    print(f"[OK] Exported Platinum data to {DATA_DIR} and JS to {out_file_js}")

if __name__ == '__main__':
    export_data()

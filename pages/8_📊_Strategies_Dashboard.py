import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
import datetime
from pathlib import Path
import re

# Set Page Config
st.set_page_config(layout="wide", page_title="Strategies Dashboard", page_icon="📊")

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR_PLATINUM = ROOT_DIR / "data" / "Platinum_Results"
DATA_DIR_NTSX = ROOT_DIR / "data" / "Multi_indicator"
DATA_DIR_FUND = ROOT_DIR / "data" / "Fund_Tactical_Results"
MOCKUP_PATH = ROOT_DIR / "scratch_mockup.html"

def load_ntsx_data(master_date_str):
    js_path = DATA_DIR_NTSX / "ntsx_data.js"
    if not js_path.exists():
        return ""
    
    with open(js_path, "r", encoding="utf-8", errors="replace") as f:
        js_data = f.read()
    
    # Injected filter code for NTSX date sync
    m_year = master_date_str[:4]
    filter_script = f"""
    // Master Date Sync (Enforced)
    if (typeof NTSX_EQUITY !== 'undefined') {{
        const mDate = '{master_date_str}';
        const mYear = parseInt('{m_year}');
        NTSX_EQUITY = NTSX_EQUITY.filter(d => d.date <= mDate);
        NTSX_REBALANCES = NTSX_REBALANCES.filter(r => r.date <= mDate);
        NTSX_YEARLY = NTSX_YEARLY.filter(y => parseInt(y.year) <= mYear);
        NTSX_REB_DATES = NTSX_REB_DATES.filter(d => d <= mDate);
        
        if (NTSX_EQUITY.length > 0) {{
            const last = NTSX_EQUITY[NTSX_EQUITY.length - 1];
            if (typeof NTSX_CURRENT !== 'undefined') {{
                NTSX_CURRENT.as_of_date = last.date;
                // Update current weights to match last data point
                // (Optional: fetch from raw data if needed)
            }}
        }}
    }}
    """
    return js_data + "\n" + filter_script

def load_platinum_json(sel_dt):
    try:
        eq = pd.read_csv(DATA_DIR_PLATINUM / 'Platinum_Equity.csv', index_col=0, parse_dates=True)
        w = pd.read_csv(DATA_DIR_PLATINUM / 'Platinum_Weights.csv', index_col=0, parse_dates=True)
        log = pd.read_csv(DATA_DIR_PLATINUM / 'Platinum_Transaction_Log.csv')
        monthly = pd.read_csv(DATA_DIR_PLATINUM / 'Platinum_Monthly_Returns.csv', index_col=0)
        mc = pd.read_csv(DATA_DIR_PLATINUM / 'Platinum_MonteCarlo.csv')
        roll = pd.read_csv(DATA_DIR_PLATINUM / 'Platinum_Rolling_Start.csv')
        
        # Filter by master date
        eq = eq[eq.index <= sel_dt]
        w = w[w.index <= sel_dt]
        log = log[pd.to_datetime(log['Date']) <= sel_dt]
        
        if eq.empty:
            return None
        
        # KPI calculations
        final_eq = eq['Platinum_Equity'].iloc[-1]
        total_ret = (final_eq / 10000) - 1
        days = (eq.index[-1] - eq.index[0]).days
        cagr = (final_eq/10000)**(365.25/max(1, days)) - 1
        
        dd_curr = (eq['Platinum_Equity'] / eq['Platinum_Equity'].cummax()) - 1
        max_dd = dd_curr.min()
        
        daily_ret = eq['Platinum_Equity'].pct_change().dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5) if not daily_ret.empty else 0.0
        
        # Determine signals
        latest_w = w.iloc[-1] if not w.empty else pd.Series()
        active_holdings = latest_w[latest_w > 0.01].index.tolist()
        
        state = "BALANCED"
        signal_text = "HOLD"
        signal_bg = "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
        
        if 'TQQQ' in active_holdings:
            state = "AGGRESSIVE BULL (Ah_Pig Active)"
            signal_text = "BUY TQQQ"
            signal_bg = "bg-indigo-600 text-white shadow-md shadow-indigo-900/30"
        elif latest_w.get('SHV', 0) > 0.5:
            state = "DEFENSIVE (Cash Heavy)"
            signal_text = "HOLD CASH"
            signal_bg = "bg-red-500/10 text-red-400 border border-red-500/30"
            
        # Assets list
        asset_details = []
        asset_names_map = {
            'TQQQ': 'UltraPro QQQ (3x Leverage)',
            'QQQ': 'Invesco QQQ Trust',
            'SMH': 'VanEck Semiconductor ETF',
            'GLD': 'SPDR Gold Shares',
            'SHV': 'Short Treasury Bond ETF'
        }
        
        for asset, weight in latest_w.items():
            if weight > 0.001 or asset in ['TQQQ', 'QQQ', 'SMH', 'GLD', 'SHV']:
                # Find close/prev weights to calculate drift
                target = 25.0 if asset == 'TQQQ' else (22.5 if asset in ['QQQ', 'SMH'] else (17.5 if asset == 'GLD' else 12.5))
                asset_details.append({
                    'symbol': asset,
                    'name': asset_names_map.get(asset, asset),
                    'price': 100.0, # Default mock price, updated in JS if needed
                    'change': 0.5,
                    'target': target,
                    'current': float(weight * 100),
                    'lower': max(0.0, target - 5.0),
                    'upper': target + 5.0
                })
        
        # Yearly performance stats
        yearly_eq = eq['Platinum_Equity'].resample('YE').last()
        annual_res = yearly_eq.pct_change()
        annual_dd = dd_curr.groupby(dd_curr.index.year).min()
        
        bench_eq = eq['Benchmark_Equity'].resample('YE').last() if 'Benchmark_Equity' in eq.columns else pd.Series()
        bench_annual_res = bench_eq.pct_change() if not bench_eq.empty else pd.Series()
        bench_dd = (eq['Benchmark_Equity'] / eq['Benchmark_Equity'].cummax()) - 1 if 'Benchmark_Equity' in eq.columns else pd.Series()
        bench_annual_dd = bench_dd.groupby(bench_dd.index.year).min() if not bench_dd.empty else pd.Series()
        
        yearly_stats = []
        for yr in annual_res.index.year:
            port_ret = annual_res.get(pd.Timestamp(f"{yr}-12-31"))
            bench_ret = bench_annual_res.get(pd.Timestamp(f"{yr}-12-31")) if not bench_annual_res.empty else 0.0
            port_dd_val = annual_dd.get(yr, 0.0)
            bench_dd_val = bench_annual_dd.get(yr, 0.0) if not bench_annual_dd.empty else 0.0
            
            yearly_stats.append({
                'year': str(yr),
                'portRet': f"{port_ret*100:+.1f}%" if not pd.isna(port_ret) else "0.0%",
                'spyRet': f"{bench_ret*100:+.1f}%" if not pd.isna(bench_ret) else "0.0%",
                'portDD': f"{port_dd_val*100:.1f}%",
                'spyDD': f"{bench_dd_val*100:.1f}%",
                'winner': 'Portfolio' if port_ret > bench_ret else 'Benchmark'
            })
            
        # Recent transaction history
        recent_history = []
        log_sorted = log.sort_values('Date', ascending=False).head(20)
        for _, row in log_sorted.iterrows():
            recent_history.append({
                'date': str(row['Date'])[:10],
                'asset': str(row['Asset']),
                'action': str(row['Action']),
                'price': f"${float(row.get('Price', row.get('Approx_Price', 0))):.2f}",
                'size': f"{float(row.get('Shares', 0)):.1f} Units",
                'val': f"${float(row.get('Value', 0)):,.0f}",
                'pnl': f"${float(row['Realized_PnL']):+,.2f}" if float(row.get('Realized_PnL', 0)) != 0 else "-"
            })
            
        # Chart arrays
        chart_dates = eq.index.strftime('%Y-%m-%d').tolist()
        chart_port = eq['Platinum_Equity'].tolist()
        chart_spy = eq['Benchmark_Equity'].tolist() if 'Benchmark_Equity' in eq.columns else eq['Platinum_Equity'].tolist()
        
        # Monte carlo distribution
        mc_sorted = mc.sort_values('CAGR')
        mc_bins = np.histogram(mc_sorted['CAGR'], bins=11)
        mc_values = [float(v * 100) for v in mc_bins[1][:-1]]
        mc_counts = [int(c) for c in mc_bins[0]]
        
        # Start date analysis
        sc_dd = [float(d * 100) for d in roll['MaxDD'].tolist()]
        sc_cagr = [float(c * 100) for c in roll['CAGR'].tolist()]
        
        # Assemble Platinum Object
        plat_data = {
            'kpi': {
                'cagr': f"{cagr*100:.1f}%",
                'cagrBench': f"vs Benchmark: {eq['Benchmark_Equity'].pct_change().mean()*252*100:.1f}%" if 'Benchmark_Equity' in eq.columns else "-",
                'maxdd': f"{max_dd*100:.1f}%",
                'maxddBench': f"vs Benchmark: {((eq['Benchmark_Equity'] / eq['Benchmark_Equity'].cummax()) - 1).min()*100:.1f}%" if 'Benchmark_Equity' in eq.columns else "-",
                'sharpe': f"{sharpe:.2f}",
                'sharpeBench': "-",
                'ratios': f"{sharpe:.2f} / -",
                'ratiosBench': "-",
                'signal': signal_text,
                'signalBg': signal_bg
            },
            'regime': state,
            'assets': asset_details,
            'yearly': yearly_stats[::-1], # latest first
            'history': recent_history,
            'chartDates': chart_dates[::10], # Downsample for chart speed
            'chartPort': chart_port[::10],
            'chartSpy': chart_spy[::10],
            'chartDDPort': [float(d * 100) for d in dd_curr.tolist()[::10]],
            'chartDDSpy': [float(d * 100) for d in ((eq['Benchmark_Equity'] / eq['Benchmark_Equity'].cummax()) - 1).tolist()[::10]] if 'Benchmark_Equity' in eq.columns else [0] * len(chart_dates[::10]),
            'mcValues': mc_values,
            'mcCounts': mc_counts,
            'scDD': sc_dd,
            'scCAGR': sc_cagr
        }
        return plat_data
    except Exception as e:
        st.error(f"Error compiling Platinum data: {e}")
        return None

def _build_histogram(series, bins=11):
    clean = pd.Series(series).dropna()
    if clean.empty:
        return [0.0] * bins, [0] * bins
    counts, edges = np.histogram(clean, bins=bins)
    return [float(v) for v in edges[:-1]], [int(c) for c in counts]

def _load_fund_prices():
    prices_path = DATA_DIR_FUND / "Fund_Tactical_Prices.csv"
    if not prices_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(prices_path, index_col=0, parse_dates=True)
    except Exception:
        return pd.DataFrame()

def load_fund_tactical_json(sel_dt):
    try:
        perf = pd.read_csv(DATA_DIR_FUND / "Fund_Tactical_performance_summary.csv", index_col=0)
        eq = pd.read_csv(DATA_DIR_FUND / "Fund_Tactical_equity_curve.csv", index_col=0, parse_dates=True)
        w = pd.read_csv(DATA_DIR_FUND / "Fund_Tactical_weights.csv", index_col=0, parse_dates=True)
        reg = pd.read_csv(DATA_DIR_FUND / "Fund_Tactical_regimes.csv", index_col=0, parse_dates=True)
        annual = pd.read_csv(DATA_DIR_FUND / "Fund_Tactical_annual_returns.csv", index_col=0, parse_dates=True)
        wins = pd.read_csv(DATA_DIR_FUND / "Fund_Tactical_win_stats.csv")
        prices = _load_fund_prices()

        eq = eq[eq.index <= sel_dt]
        w = w[w.index <= sel_dt]
        reg = reg[reg.index <= sel_dt]
        annual = annual[annual.index <= sel_dt]

        if eq.empty or w.empty:
            return None

        latest_w = w.iloc[-1].fillna(0.0)
        active_w = latest_w[latest_w > 0.001].sort_values(ascending=False)
        reg_series = reg["Regime"].dropna()
        latest_regime = reg_series.iloc[-1] if not reg_series.empty else "WARMUP"

        signal_map = {
            "US_BULL": ("OVERWEIGHT GROWTH", "bg-indigo-600 text-white shadow-md shadow-indigo-900/30"),
            "BROAD_BULL": ("BALANCED RISK-ON", "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"),
            "INFLATION": ("REAL-ASSET TILT", "bg-amber-500/10 text-amber-300 border border-amber-500/30"),
            "RISK_OFF": ("DEFENSIVE", "bg-red-500/10 text-red-400 border border-red-500/30"),
            "MIXED": ("HOLD MIXED", "bg-slate-700 text-slate-100 border border-slate-600"),
            "WARMUP": ("WARMUP", "bg-slate-700 text-slate-100 border border-slate-600"),
        }
        signal_text, signal_bg = signal_map.get(latest_regime, ("HOLD", "bg-slate-700 text-slate-100 border border-slate-600"))

        names_map = {
            "SPY": "SPDR S&P 500 ETF",
            "QQQ": "Invesco QQQ Trust",
            "GMOM": "Cambria Global Momentum ETF",
            "RLY": "SPDR Multi-Asset Real Return ETF",
            "DBMF": "iMGP DBi Managed Futures Strategy ETF",
            "SGOV": "iShares 0-3 Month Treasury Bond ETF",
        }

        visible_prices = prices[prices.index <= sel_dt] if not prices.empty else pd.DataFrame()
        price_row = visible_prices.iloc[-1] if not visible_prices.empty else pd.Series(dtype=float)
        prev_price_row = visible_prices.iloc[-2] if len(visible_prices) > 1 else pd.Series(dtype=float)

        assets = []
        for ticker, weight in active_w.items():
            px = float(price_row.get(ticker, 100.0))
            prev_px = float(prev_price_row.get(ticker, px))
            daily_chg = ((px / prev_px) - 1) * 100 if prev_px else 0.0
            wt_pct = float(weight * 100)
            assets.append({
                "symbol": ticker,
                "name": names_map.get(ticker, ticker),
                "price": px,
                "change": daily_chg,
                "target": wt_pct,
                "current": wt_pct,
                "lower": max(0.0, wt_pct - 5.0),
                "upper": min(100.0, wt_pct + 5.0),
            })

        dd_port = (eq["Strategy_Equity"] / eq["Strategy_Equity"].cummax()) - 1
        dd_spy = (eq["SPY_Equity"] / eq["SPY_Equity"].cummax()) - 1
        annual_dd_port = dd_port.groupby(dd_port.index.year).min()
        annual_dd_spy = dd_spy.groupby(dd_spy.index.year).min()

        yearly_stats = []
        for idx, row in annual.iterrows():
            year = idx.year
            port_ret = float(row.get("Dynamic_Strategy", 0.0))
            spy_ret = float(row.get("SPY", 0.0))
            yearly_stats.append({
                "year": str(year),
                "portRet": f"{port_ret * 100:+.1f}%",
                "spyRet": f"{spy_ret * 100:+.1f}%",
                "portDD": f"{annual_dd_port.get(year, 0.0) * 100:.1f}%",
                "spyDD": f"{annual_dd_spy.get(year, 0.0) * 100:.1f}%",
                "winner": "Portfolio" if port_ret > spy_ret else "SPY",
            })

        rebalance_mask = w.diff().abs().sum(axis=1) > 0.001
        history = []
        for dt in w.index[rebalance_mask][::-1][:24]:
            curr = w.loc[dt].fillna(0.0)
            prev_idx = w.index[w.index < dt]
            prev = w.loc[prev_idx[-1]].fillna(0.0) if len(prev_idx) else pd.Series(0.0, index=w.columns)
            diff = (curr - prev).sort_values()
            biggest_cut = diff.index[0]
            biggest_add = diff.index[-1]
            turnover = float((curr - prev).abs().sum() * 100)
            event_regime = reg.loc[dt, "Regime"] if dt in reg.index and pd.notna(reg.loc[dt, "Regime"]) else latest_regime
            history.append({
                "date": dt.strftime("%Y-%m-%d"),
                "regime": str(event_regime),
                "add": f"{biggest_add} {diff[biggest_add] * 100:+.1f}%",
                "cut": f"{biggest_cut} {diff[biggest_cut] * 100:+.1f}%",
                "turnover": f"{turnover:.1f}%",
                "mix": " / ".join([f"{k} {v * 100:.0f}%" for k, v in curr[curr > 0.001].sort_values(ascending=False).items()]),
            })

        rolling_1y = eq["Strategy_Equity"].pct_change(252)
        mc_values, mc_counts = _build_histogram(rolling_1y * 100)

        start_points = []
        for start_dt in eq.resample("MS").first().index:
            sub = eq[eq.index >= start_dt]
            if len(sub) < 126:
                continue
            years = max((sub.index[-1] - sub.index[0]).days / 365.25, 1 / 12)
            cagr = ((sub["Strategy_Equity"].iloc[-1] / sub["Strategy_Equity"].iloc[0]) ** (1 / years) - 1) * 100
            maxdd = ((sub["Strategy_Equity"] / sub["Strategy_Equity"].cummax()) - 1).min() * 100
            start_points.append((start_dt.strftime("%Y-%m"), float(maxdd), float(cagr)))

        sc_dd = [pt[1] for pt in start_points]
        sc_cagr = [pt[2] for pt in start_points]
        best_start = max(start_points, key=lambda x: x[2]) if start_points else ("-", 0.0, 0.0)
        worst_start = min(start_points, key=lambda x: x[2]) if start_points else ("-", 0.0, 0.0)

        monthly_win = wins.loc[wins["Metric"] == "Monthly Win Rate vs SPY", "Dynamic_Strategy"]
        monthly_win_rate = float(monthly_win.iloc[0]) if not monthly_win.empty else 0.0

        return {
            "kpi": {
                "cagr": f"{perf.loc['CAGR', 'Dynamic_Strategy'] * 100:.1f}%",
                "cagrBench": f"vs SPY: {perf.loc['CAGR', 'SPY'] * 100:.1f}%",
                "maxdd": f"{perf.loc['Max_Drawdown', 'Dynamic_Strategy'] * 100:.1f}%",
                "maxddBench": f"vs SPY: {perf.loc['Max_Drawdown', 'SPY'] * 100:.1f}%",
                "sharpe": f"{perf.loc['Sharpe', 'Dynamic_Strategy']:.2f}",
                "sharpeBench": f"vs SPY: {perf.loc['Sharpe', 'SPY']:.2f}",
                "ratios": f"{perf.loc['Calmar', 'Dynamic_Strategy']:.2f} / {perf.loc['Sharpe', 'Dynamic_Strategy']:.2f}",
                "ratiosBench": f"vs SPY: {perf.loc['Calmar', 'SPY']:.2f} / {perf.loc['Sharpe', 'SPY']:.2f}",
                "signal": signal_text,
                "signalBg": signal_bg,
            },
            "regime": latest_regime.replace("_", " ").title(),
            "assets": assets,
            "yearly": yearly_stats[::-1],
            "history": history,
            "chartDates": eq.index.strftime("%Y-%m-%d").tolist()[::5],
            "chartPort": eq["Strategy_Equity"].tolist()[::5],
            "chartSpy": eq["SPY_Equity"].tolist()[::5],
            "chartDDPort": [float(v * 100) for v in dd_port.tolist()[::5]],
            "chartDDSpy": [float(v * 100) for v in dd_spy.tolist()[::5]],
            "mcValues": mc_values,
            "mcCounts": mc_counts,
            "scDD": sc_dd,
            "scCAGR": sc_cagr,
            "meta": {
                "backtestPeriod": f"{eq.index[0].strftime('%b %Y')} - {eq.index[-1].strftime('%b %Y')}",
                "executionDescription": "Monthly tactical rebalance that rotates between U.S. beta, growth, diversifiers, and T-bills.",
                "lastActionDate": history[0]["date"] if history else eq.index[-1].strftime("%Y-%m-%d"),
                "monthlyWinRate": f"{monthly_win_rate * 100:.1f}%",
                "bestEra": f"{best_start[0]} (CAGR {best_start[2]:.1f}%)",
                "worstEra": f"{worst_start[0]} (CAGR {worst_start[2]:.1f}%)",
            },
        }
    except Exception as e:
        st.error(f"Error compiling Fund Tactical data: {e}")
        return None

def main():
    st.title("📊 Strategies Dashboard (NTSX, Platinum, F-TAA)")
    
    # Sync with Streamlit Master Date Picker
    selected_date = st.session_state.get('master_date', datetime.date.today())
    master_date_str = selected_date.strftime('%Y-%m-%d')
    sel_dt = pd.Timestamp(selected_date)
    
    # 1. Check if mockup HTML template exists
    if not MOCKUP_PATH.exists():
        st.error("Dashboard template file not found.")
        return
        
    try:
        with open(MOCKUP_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        # 2. Inject NTSX data script
        ntsx_js = load_ntsx_data(master_date_str)
        
        # 3. Inject strategy data objects
        plat_data = load_platinum_json(sel_dt)
        plat_js = f"const PLATINUM_DATA_LIVE = {json.dumps(plat_data)};" if plat_data else "const PLATINUM_DATA_LIVE = null;"
        fund_data = load_fund_tactical_json(sel_dt)
        fund_js = f"const FUND_TACTICAL_DATA_LIVE = {json.dumps(fund_data)};" if fund_data else "const FUND_TACTICAL_DATA_LIVE = null;"
        
        # 4. Inject JS adapter to map live variables to the dashboard UI
        js_adapter = """
        <script type="text/javascript">
            // Inject NTSX raw variables
            {NTSX_RAW_JS}
            
            // Inject Platinum raw variables
            {PLATINUM_RAW_JS}

            // Inject Fund Tactical raw variables
            {FUND_RAW_JS}
            
            window.addEventListener('DOMContentLoaded', () => {
                // Map NTSX Live data
                if (typeof NTSX_CURRENT !== 'undefined') {
                    NTSX_DATA.kpi = {
                        cagr: NTSX_METRICS.portfolio.cagr + '%',
                        cagrBench: 'vs SPY: ' + NTSX_METRICS.spy.cagr + '%',
                        maxdd: NTSX_METRICS.portfolio.max_dd + '%',
                        maxddBench: 'vs SPY: ' + NTSX_METRICS.spy.max_dd + '%',
                        sharpe: NTSX_METRICS.portfolio.sharpe,
                        sharpeBench: 'vs SPY: ' + NTSX_METRICS.spy.sharpe,
                        ratios: NTSX_METRICS.portfolio.calmar + ' / ' + NTSX_METRICS.portfolio.sortino,
                        ratiosBench: 'vs SPY: ' + NTSX_METRICS.spy.calmar + ' / ' + NTSX_METRICS.spy.sortino,
                        signal: NTSX_CURRENT.action_signal,
                        signalBg: NTSX_CURRENT.action_signal === 'HOLD' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' : 'bg-red-500/10 text-red-400 border border-red-500/30'
                    };
                    NTSX_DATA.regime = NTSX_CURRENT.action_signal === 'HOLD' ? 'Neutral Rebalancing Mode (' + NTSX_CURRENT.days_since_last_rebalance + ' days since last action)' : 'Rebalance Required!';
                    
                    const prices = { 'NTSX': 62.50, 'AVWS': 18.20, 'KMLM': 34.10 };
                    const changes = { 'NTSX': 0.20, 'AVWS': -0.05, 'KMLM': 0.45 };
                    NTSX_DATA.assets = NTSX_CURRENT.asset_names.map((name, idx) => ({
                        symbol: name,
                        name: name === 'NTSX' ? 'WisdomTree 90/60 Balanced' : (name === 'AVWS' ? 'Avantis All International Markets' : 'KFA Mount Lucas Managed Futures'),
                        price: prices[name] || 50.0,
                        change: changes[name] || 0.0,
                        target: NTSX_CURRENT.target_weights[idx],
                        current: NTSX_CURRENT.current_weights[idx],
                        lower: NTSX_CURRENT.lower_bounds[idx],
                        upper: NTSX_CURRENT.upper_bounds[idx]
                    }));
                    NTSX_DATA.yearly = NTSX_YEARLY.map(y => ({
                        year: y.year,
                        portRet: (y.port_return >= 0 ? '+' : '') + y.port_return + '%',
                        spyRet: (y.spy_return >= 0 ? '+' : '') + y.spy_return + '%',
                        portDD: y.port_maxdd + '%',
                        spyDD: y.spy_maxdd + '%',
                        winner: y.winner
                    }));
                    NTSX_DATA.history = NTSX_REBALANCES.map((r, idx) => ({
                        num: idx + 1,
                        date: r.date,
                        trigger: r.triggers[0] ? r.triggers[0].asset : 'System',
                        direction: r.triggers[0] ? (r.triggers[0].direction.includes('Below') ? 'Below Lower Band' : 'Above Upper Band') : 'Rebalance',
                        weight: r.triggers[0] ? r.triggers[0].weight_at_trigger + '%' : '-',
                        band: r.triggers[0] ? r.triggers[0].band + '%' : '-',
                        before: r.weights_before.map(w => w + '%').join(' / '),
                        after: r.weights_after.map(w => w + '%').join(' / ')
                    })).reverse();
                    
                    NTSX_DATA.chartDates = NTSX_EQUITY.map(d => d.date);
                    NTSX_DATA.chartPort = NTSX_EQUITY.map(d => d.port);
                    NTSX_DATA.chartSpy = NTSX_EQUITY.map(d => d.spy);
                    
                    let maxP = 0;
                    NTSX_DATA.chartDDPort = NTSX_EQUITY.map(d => {
                        if (d.port > maxP) maxP = d.port;
                        return ((d.port / maxP) - 1) * 100;
                    });
                    let maxS = 0;
                    NTSX_DATA.chartDDSpy = NTSX_EQUITY.map(d => {
                        if (d.spy > maxS) maxS = d.spy;
                        return ((d.spy / maxS) - 1) * 100;
                    });
                }
                
                // Map Platinum Live data
                if (PLATINUM_DATA_LIVE) {
                    Object.assign(PLATINUM_DATA, PLATINUM_DATA_LIVE);
                }

                // Map Fund Tactical live data
                if (FUND_TACTICAL_DATA_LIVE) {
                    Object.assign(FUND_TACTICAL_DATA, FUND_TACTICAL_DATA_LIVE);
                }
                
                // Set initial datepicker and capital value
                AppState.date = "{MASTER_DATE}";
                if (DOM.masterDate) DOM.masterDate.value = "{MASTER_DATE}";
                
                // Reboot active strategy to render changes
                switchStrategy(AppState.currentStrategy);
            });
        </script>
        """
        
        js_adapter = js_adapter.replace("{NTSX_RAW_JS}", ntsx_js)
        js_adapter = js_adapter.replace("{PLATINUM_RAW_JS}", plat_js)
        js_adapter = js_adapter.replace("{FUND_RAW_JS}", fund_js)
        js_adapter = js_adapter.replace("{MASTER_DATE}", master_date_str)
        
        # Inject adapter right before closing body tag
        html_content = html_content.replace("</body>", js_adapter + "\n</body>")
        
        # Sync Streamlit date value back into HTML default fields
        html_content = html_content.replace('value="2026-05-23"', f'value="{master_date_str}"')
        
        # Render HTML component
        components.html(html_content, height=1200, scrolling=True)
        
    except Exception as e:
        st.error(f"Error rendering Strategies Dashboard: {e}")

if __name__ == "__main__":
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    with st.sidebar:
        render_master_controls()
        render_ecosystem_sidebar()
    main()

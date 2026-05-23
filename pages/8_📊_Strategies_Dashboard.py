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

def main():
    st.title("📊 Strategies Dashboard (NTSX & Platinum)")
    
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
        
        # 3. Inject Platinum data object
        plat_data = load_platinum_json(sel_dt)
        plat_js = f"const PLATINUM_DATA_LIVE = {json.dumps(plat_data)};" if plat_data else "const PLATINUM_DATA_LIVE = null;"
        
        # 4. Inject JS adapter to map live variables to the dashboard UI
        js_adapter = """
        <script type="text/javascript">
            // Inject NTSX raw variables
            {NTSX_RAW_JS}
            
            // Inject Platinum raw variables
            {PLATINUM_RAW_JS}
            
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
    main()

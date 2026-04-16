import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Set Page Config
st.set_page_config(layout="wide", page_title="Platinum Strategy Dashboard", page_icon="💎")

# Paths - Adjusted for multi-page structure (Strong Pathing for Public Deployment)
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "Platinum_Results"
DOCS_PATH = DATA_DIR / "Platinum_Strategy_Docs.md"

EQUITY_PATH = DATA_DIR / 'Platinum_Equity.csv'
WEIGHTS_PATH = DATA_DIR / 'Platinum_Weights.csv'
PRICES_PATH = DATA_DIR / 'Platinum_Data_Used.csv'
LOG_PATH = DATA_DIR / 'Platinum_Transaction_Log.csv'
MONTHLY_PATH = DATA_DIR / 'Platinum_Monthly_Returns.csv'
MC_PATH = DATA_DIR / 'Platinum_MonteCarlo.csv'
ROLL_PATH = DATA_DIR / 'Platinum_Rolling_Start.csv'
# DOCS_PATH already set correctly above

def load_data():
    try:
        eq = pd.read_csv(EQUITY_PATH, index_col=0, parse_dates=True)
        w = pd.read_csv(WEIGHTS_PATH, index_col=0, parse_dates=True)
        p = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
        log = pd.read_csv(LOG_PATH)
        monthly = pd.read_csv(MONTHLY_PATH, index_col=0)
        mc = pd.read_csv(MC_PATH)
        roll = pd.read_csv(ROLL_PATH)
        return eq, w, p, log, monthly, mc, roll
    except Exception as e:
        st.error(f"Error loading data from {DATA_DIR}: {e}")
        return None, None, None, None, None, None, None

def main():
    st.title("💎 Platinum Strategy Dashboard")
    
    eq_full, w_full, prices_full, log_full, monthly, mc, roll = load_data()
    
    if eq_full is None:
        st.warning("Data not found. Please ensure Platinum_Backtest.py has been run and results are in the Platinum_Results folder.")
        return

    # Master Date Synchronization (Time Travel)
    selected_date = st.session_state.get('master_date', pd.Timestamp.now().date())
    sel_dt = pd.Timestamp(selected_date)
    
    # Truncate all data to the selected master date
    eq = eq_full[eq_full.index <= sel_dt]
    w = w_full[w_full.index <= sel_dt]
    
    # Check for data gap (Stale Data)
    latest_factual_date = eq_full.index.max().date()
    if selected_date > latest_factual_date:
        st.warning(f"⚠️ **數據延遲 (Data Gap Detected)**: 您的基準日期為 {selected_date}，但本地數據僅更新至 {latest_factual_date}。")
        st.info("請在本地端執行 **⚡ 數據刷新** 以獲取最新市場事實資料。")
    
    prices = prices_full[prices_full.index <= sel_dt]
    log = log_full[pd.to_datetime(log_full['Date']) <= sel_dt].copy()
    
    if eq.empty:
        st.error(f"所選日期無效（早於回測開始日）: {selected_date}")
        return

    # Ensure Log P&L is numeric
    log['Realized_PnL'] = pd.to_numeric(log['Realized_PnL'], errors='coerce').fillna(0.0)

    # --- KPI CALCULATIONS ---
    final_eq = eq['Platinum_Equity'].iloc[-1]
    total_ret = (final_eq / 10000) - 1
    days = (eq.index[-1] - eq.index[0]).days
    if days <= 0: days = 1 # Avoid division by zero
    cagr = (final_eq/10000)**(365.25/days) - 1
    
    dd_curr = (eq['Platinum_Equity'] / eq['Platinum_Equity'].cummax()) - 1
    max_dd = dd_curr.min()
    
    daily_ret = eq['Platinum_Equity'].pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * (252**0.5) if not daily_ret.empty else 0.0

    # --- TABS ---
    tab_action, tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Action Card", "📈 Performance", "📅 Monthly Returns", "📝 Transaction Log", "🎲 Robustness", "🧠 Strategy Logic"])

    with tab_action:
        st.header("⚡ Action Required")
        
        # Synchronized Master Date
        selected_date = st.session_state['master_date']
        
        sel_dt = pd.Timestamp(selected_date)
        valid_dates = w.index
        past_dates = valid_dates[valid_dates <= sel_dt]
        
        if past_dates.empty:
            st.error("Date is before start of backtest.")
        else:
            eff_date = past_dates[-1]
            if eff_date.date() != selected_date:
                st.info(f"Showing data for closest trading day: {eff_date.date()}")
            
            current_w = w.loc[eff_date]
            
            # --- ACTION LOGIC ---
            prev_idx = w.index.get_loc(eff_date) - 1
            if prev_idx >= 0:
                prev_w = w.iloc[prev_idx]
            else:
                prev_w = pd.Series(0.0, index=current_w.index)
            
            actions = []
            delta = current_w - prev_w
            
            new_opens = current_w[(current_w > 0.01) & (prev_w <= 0.01)]
            for asset in new_opens.index:
                actions.append(f"🟢 **OPEN NEW TRADE**: Buy **{asset}** (Target: {current_w[asset]:.1%})")
                
            closes = prev_w[(prev_w > 0.01) & (current_w <= 0.01)]
            for asset in closes.index:
                actions.append(f"🔴 **CLOSE TRADE**: Sell All **{asset}**")
                
            rebalances = delta[(delta.abs() > 0.05) & (~delta.index.isin(new_opens.index)) & (~delta.index.isin(closes.index))]
            for asset, change in rebalances.items():
                direction = "Increase" if change > 0 else "Reduce"
                actions.append(f"⚖️ **REBALANCE**: {direction} **{asset}** by {abs(change):.1%} (Target: {current_w[asset]:.1%})")

            st.subheader("📋 Required Actions")
            if actions:
                for act in actions:
                    if "OPEN" in act: st.success(act)
                    elif "CLOSE" in act: st.error(act)
                    else: st.warning(act)
            else:
                st.info("✅ **NO ACTION REQUIRED**: Portfolio is stable. Keep Monitoring.")
            
            st.markdown("---")

            active_holdings = current_w[current_w > 0.01].sort_values(ascending=False)
            curr_eq_val = eq['Platinum_Equity'].asof(eff_date)
            curr_dd_val = dd_curr.asof(eff_date)
            curr_daily_ret = daily_ret.asof(eff_date)
            
            col_status, col_holdings = st.columns([1, 2])
            with col_status:
                st.markdown(f"### Status as of {eff_date.date()}")
                state = "BALANCED"; color = "blue"
                if 'TQQQ' in active_holdings:
                    state = "AGGRESSIVE BULL (Ah_Pig Active)"; color = "green"
                elif active_holdings.get('SHV', 0) > 0.5:
                    state = "DEFENSIVE (Cash Heavy)"; color = "red"
                
                st.markdown(f"**MARKET STATE**: :{color}[{state}]")
                st.markdown("---")
                st.metric("Total Equity", f"${curr_eq_val:,.0f}", f"{curr_daily_ret:.2%}")
                st.metric("Drawdown", f"{curr_dd_val:.2%}")
                
            with col_holdings:
                st.subheader("Portfolio Holdings")
                log_hist = log[pd.to_datetime(log['Date']) <= eff_date]
                holdings_data = []
                for asset, weight in active_holdings.items():
                    subset = log_hist[(log_hist['Asset'] == asset) & (log_hist['Action'] == 'BUY')]
                    entry_price = 0.0
                    if not subset.empty:
                        last_buy = subset.iloc[-1]
                        entry_price = float(last_buy['Price']) if 'Price' in last_buy else float(last_buy.get('Approx_Price', 0))
                    
                    try: curr_comp_price = prices[asset].asof(eff_date)
                    except: curr_comp_price = 0.0

                    alloc_val = curr_eq_val * weight
                    holdings_data.append({
                        'Asset': asset,
                        'Weight': f"{weight:.1%}",
                        'Value': f"${alloc_val:,.0f}",
                        'Entry Price': f"${entry_price:.2f}",
                        'Current Price': f"${curr_comp_price:.2f}",
                        'P&L (Unrealized)': f"{(curr_comp_price/entry_price - 1):.1%}" if entry_price > 0 else "-"
                    })
                
                if holdings_data: st.dataframe(pd.DataFrame(holdings_data), use_container_width=True)
                else: st.write("No active holdings (100% Cash/Idle).")
                st.info("💡 **Action**: Verify these weights match your broker. Rebalance if deviation > 5%.")

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Final Equity", f"${final_eq:,.0f}", f"{total_ret:.1%}")
        col2.metric("CAGR", f"{cagr:.1%}")
        col3.metric("Max Drawdown", f"{max_dd:.1%}")
        col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.markdown("---")
                st.subheader("Equity Curve")
        plot_cols = ['Platinum_Equity']
        if 'Benchmark_Equity' in eq.columns:
            plot_cols.append('Benchmark_Equity')
            
        fig = px.line(eq, x=eq.index, y=plot_cols, 
                      labels={'value': 'Equity ($)', 'index': 'Date'},
                      color_discrete_map={'Platinum_Equity': '#2ecc71', 'Benchmark_Equity': 'gray'})
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Drawdown Profile")
        dd_df = pd.DataFrame()
        dd_df['Platinum_DD'] = (eq['Platinum_Equity'] / eq['Platinum_Equity'].cummax()) - 1
        
        dd_plot_cols = ['Platinum_DD']
        if 'Benchmark_Equity' in eq.columns:
            dd_df['Benchmark_DD'] = (eq['Benchmark_Equity'] / eq['Benchmark_Equity'].cummax()) - 1
            dd_plot_cols.append('Benchmark_DD')
            
        fig_dd = px.area(dd_df, x=dd_df.index, y=dd_plot_cols, color_discrete_map={'Platinum_DD': 'red', 'Benchmark_DD': 'gray'})
        st.plotly_chart(fig_dd, use_container_width=True)


        st.subheader("Annual Performance")
        yearly_eq = eq['Platinum_Equity'].resample('YE').last()
        annual_res = yearly_eq.pct_change()
        annual_dd = dd_curr.groupby(dd_curr.index.year).min()
        ann_df = pd.DataFrame()
        ann_df['Return'] = annual_res.groupby(annual_res.index.year).last()
        ann_df['Max Drawdown'] = annual_dd
        fig_ann = px.bar(ann_df, x=ann_df.index, y='Return', color='Return', color_continuous_scale='RdYlGn', title="Annual Returns")
        st.plotly_chart(fig_ann, use_container_width=True)
        with st.expander("Detailed Annual Metrics"): st.dataframe(ann_df.style.format("{:.1%}"), use_container_width=True)

    with tab2:
        st.subheader("Monthly Returns Heatmap")
        heatmap_data = monthly.drop(columns=['Year Total'])
        fig_heat = px.imshow(heatmap_data, labels=dict(x="Month", y="Year", color="Return"),
                             x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                             y=heatmap_data.index, color_continuous_scale='RdYlGn', text_auto='.1%')
        fig_heat.update_layout(height=800)
        st.plotly_chart(fig_heat, use_container_width=True)
        with st.expander("View Data Table"): st.dataframe(monthly.style.format("{:.1%}", na_rep="-"))

    with tab3:
        st.subheader("Transaction Log")
        col_filters, col_stats = st.columns([1, 1])
        with col_filters:
            log_dates = pd.to_datetime(log['Date'])
            date_range = st.date_input("Filter Date Range", value=(log_dates.min().date(), log_dates.max().date()))
            assets_filter = st.multiselect("Filter Asset", options=log['Asset'].unique())
            actions_filter = st.multiselect("Filter Action", options=['BUY', 'SELL'])
        
        log_view = log.copy()
        log_view['Date'] = pd.to_datetime(log_view['Date']).dt.date
        if len(date_range) == 2:
            start_d, end_d = date_range
            log_view = log_view[(log_view['Date'] >= start_d) & (log_view['Date'] <= end_d)]
        if assets_filter: log_view = log_view[log_view['Asset'].isin(assets_filter)]
        if actions_filter: log_view = log_view[log_view['Action'].isin(actions_filter)]
            
        with col_stats:
            total_pnl = log_view['Realized_PnL'].sum()
            win_count = len(log_view[log_view['Realized_PnL'] > 0])
            total_trades = len(log_view[log_view['Action'] == 'SELL'])
            win_rate = win_count / total_trades if total_trades > 0 else 0
            st.metric("Realized P&L (Selection)", f"${total_pnl:,.2f}", f"Win Rate: {win_rate:.1%}")

        def color_pnl(val):
            if val > 0: return 'color: green'
            elif val < 0: return 'color: red'
            else: return ''

        st.dataframe(log_view.style.format({'Price': '${:.2f}', 'Value': '${:,.0f}', 'Realized_PnL': '${:,.2f}', 'PnL_Pct': '{:.1%}', 'Shares': '{:,.2f}'}).map(color_pnl, subset=['Realized_PnL', 'PnL_Pct']), use_container_width=True)

    with tab4:
        col_mc, col_roll = st.columns(2)
        with col_mc:
            st.subheader("Monte Carlo (1000 Runs)")
            fig_mc = px.histogram(mc, x="CAGR", nbins=50, title="CAGR Distribution", color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig_mc, use_container_width=True)
            st.write(f"**95% VaR (CAGR)**: {mc['CAGR'].quantile(0.05):.1%}")
            st.write(f"**Top 5% (CAGR)**: {mc['CAGR'].quantile(0.95):.1%}")
        with col_roll:
            st.subheader("Robustness (Start Date Analysis)")
            fig_roll = px.scatter(roll, x="Start_Date", y="CAGR", color="MaxDD", title="Start Date Sensitivity")
            st.plotly_chart(fig_roll, use_container_width=True)
            best = roll.loc[roll['CAGR'].idxmax()]; worst = roll.loc[roll['CAGR'].idxmin()]
            st.info(f"**Best Era**: {best['Start_Date']} (CAGR {best['CAGR']:.1%})")
            st.error(f"**Worst Era**: {worst['Start_Date']} (CAGR {worst['CAGR']:.1%})")

    with tab5:
        st.header("🧠 Platinum Strategy Logic")
        capital = st.number_input("Initial Capital ($)", min_value=1000, value=100000, step=1000, format="%d")
        core_pct = 0.75; sat_pct = 0.25; core_val = capital * core_pct; sat_val = capital * sat_pct
        fusion_alloc_pct = 0.50; fed_alloc_pct = 0.35; titan_alloc_pct = 0.15
        fusion_val = core_val * fusion_alloc_pct; fed_val = core_val * fed_alloc_pct; titan_val = core_val * titan_alloc_pct
        fusion_global_pct = core_pct * fusion_alloc_pct; fed_global_pct = core_pct * fed_alloc_pct; titan_global_pct = core_pct * titan_alloc_pct
        fusion_attack_each_val = fusion_val / 3; fusion_attack_each_pct = fusion_global_pct / 3
        fusion_def_gld_val = fusion_val * 0.5; fusion_def_shv_val = fusion_val * 0.5
        fusion_def_gld_pct = fusion_global_pct * 0.5; fusion_def_shv_pct = fusion_global_pct * 0.5
        
        st.graphviz_chart(f'''
        digraph Platinum {{
            rankdir=TB; node [shape=box, style=filled, fillcolor="white", fontname="Arial"];
            Portfolio [label="Platinum Portfolio\\n(100%)\\n${capital:,.0f}", shape=doubleoctagon, fillcolor="#2c3e50", fontcolor="white"];
            Core [label="CORE BUCKET\\n(75%)\\n${core_val:,.0f}", fillcolor="#95a5a6"];
            Sat [label="SATELLITE BUCKET\\n(25%)\\n${sat_val:,.0f}", fillcolor="#e74c3c", fontcolor="white"];
            Portfolio -> Core; Portfolio -> Sat;
            subgraph cluster_core {{ label = "Core Components"; style=dashed; color=gray;
                Fusion [label="1. FUSION (50% of Core)\\n${fusion_val:,.0f} ({fusion_global_pct:.1%})", fillcolor="#3498db", fontcolor="white"];
                Core -> Fusion; Fusion_Check [label="Macro Checking", shape=diamond]; Fusion -> Fusion_Check;
                Fusion_Attack [label="ATTACK\\n(Top 3 Momentum)\\n~${fusion_attack_each_val:,.0f} ({fusion_attack_each_pct:.1%}) each", fillcolor="#a9dfbf"];
                Fusion_Def [label="DEFENSE\\nGLD: ${fusion_def_gld_val:,.0f} ({fusion_def_gld_pct:.1%})\\nSHV: ${fusion_def_shv_val:,.0f} ({fusion_def_shv_pct:.1%})", fillcolor="#f9e79f"];
                Fusion_Check -> Fusion_Attack [label="Healthy"]; Fusion_Check -> Fusion_Def [label="Weak"];
                Fed [label="2. FED PIVOT (35% of Core)\\n${fed_val:,.0f} ({fed_global_pct:.1%})", fillcolor="#9b59b6", fontcolor="white"];
                Core -> Fed; Fed_Check [label="Liquidity Checking", shape=diamond]; Fed -> Fed_Check;
                Fed_Bull [label="BULL REGIME", fillcolor="#d2b4de"]; Fed_Bear [label="BEAR REGIME", fillcolor="#ebdef0"];
                Fed_Check -> Fed_Bull [label="Bull"]; Fed_Check -> Fed_Bear [label="Bear"];
                Titan [label="3. TITAN (15% of Core)\\n${titan_val:,.0f} ({titan_global_pct:.1%})", fillcolor="#f39c12", fontcolor="white"];
                Core -> Titan; Titan_Select [label="Winner Take All", fillcolor="#fcd4b4"]; Titan -> Titan_Select;
            }}
            subgraph cluster_sat {{ label = "Satellite Logic (Ah_Pig)"; style=dashed; color=red;
                Ah_Pig_Check [label="Regime Check", shape=diamond, fillcolor="#fadbd8"]; Sat -> Ah_Pig_Check;
                Sat_TQQQ [label="100% TQQQ", fillcolor="#2ecc71"]; Sat_Cash [label="100% SHV", fillcolor="#ec7063", fontcolor="white"];
                Ah_Pig_Check -> Sat_TQQQ [label="Pass"]; Ah_Pig_Check -> Sat_Cash [label="Fail"];
            }}
        }}
        ''')
        
        if DOCS_PATH.exists():
            with open(DOCS_PATH, "r", encoding="utf-8") as f: st.markdown("---"); st.markdown(f.read())

if __name__ == "__main__":
    from utils.ui_utils import render_ecosystem_sidebar, render_master_controls
    with st.sidebar:
        render_master_controls()
        render_ecosystem_sidebar()
    main()

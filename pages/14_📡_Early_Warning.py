"""14_📡_Early_Warning.py — Unified Early Warning System"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="📡 Early Warning System", page_icon="📡", layout="wide")

ROOT_DIR            = Path(__file__).resolve().parents[1]
WEEKLY_PATH         = ROOT_DIR / "exports" / "crash_predictor_study" / "weekly_feature_outcomes.csv"
STAGE_RETURNS_PATH  = ROOT_DIR / ".tmp" / "market_stage_validation" / "output" / "stage_forward_returns.csv"
STAGE_COVERAGE_PATH = ROOT_DIR / ".tmp" / "market_stage_validation" / "output" / "data_coverage.csv"
STAGE_BRD_HIST_PATH = ROOT_DIR / ".tmp" / "market_stage_validation" / "output" / "stage_breadth_history.csv"
EARLY_WARNING_CALIBRATION_PATH = ROOT_DIR / "data" / "early_warning_calibration.json"

# Fallback chip text if the calibration cache hasn't been built yet (see
# backend/calibrate_early_warning.py) -- the live-computed values replace
# these once data/early_warning_calibration.json exists.
DEFAULT_STAGE_TEXT = {
    1: {"lead": "n/a", "lift": "n/a"},
    2: {"lead": "n/a", "lift": "n/a"},
    3: {"lead": "n/a", "lift": "n/a"},
}

CRASH_ZONES = [
    ("1998-07-17","1998-10-08","LTCM","#f59e0b"),
    ("2000-03-24","2002-10-09","Dot-com Bust","#ef4444"),
    ("2007-10-09","2009-03-09","GFC","#dc2626"),
    ("2011-07-22","2011-10-03","Euro Crisis","#f97316"),
    ("2020-02-19","2020-03-23","COVID","#f97316"),
    ("2022-01-04","2022-10-12","Rate Hike Bear","#f97316"),
    ("2025-02-19","2025-04-08","Tariff Shock","#f59e0b"),
]
STAGE_COLORS = {"Acceleration":"#22c55e","Accumulation":"#f59e0b","Distribution":"#f97316","Deceleration":"#ef4444"}
ALERT_LABELS = {0:"ALL CLEAR",1:"MONITOR",2:"WATCH",3:"WARNING"}
ALERT_ICONS  = {0:"🟢",1:"🟡",2:"🟠",3:"🔴"}

@st.cache_data(show_spinner=False, ttl=3600)
def load_weekly():
    df = pd.read_csv(WEEKLY_PATH, index_col=0, parse_dates=True).sort_index()
    # Multiple scheduled refresh jobs (GitHub Actions + Cowork tasks) write this CSV
    # independently; a race between them can leave duplicate-date rows that break
    # index.get_indexer(method="nearest") below (requires a unique index). Keep the
    # most-recently-written row per date.
    df = df[~df.index.duplicated(keep="last")]
    df["d1_norm"]     = (df["d1_market_regime_score"] / 85 * 100).clip(0,100).round(1)
    df["d2_norm"]     = ((df["d2_score"] - 2) / 12 * 100).clip(0,100).round(1)
    df["d3_norm"]     = df["liquidity_score"].astype(float)
    df["credit_norm"] = ((df["credit_spread_for_model"] - 1.0) / 5.0 * 100).clip(0,100).round(1)
    streak, cnt = [], 0
    for v in df["d1_market_regime_score"]:
        cnt = cnt+1 if v > 25 else 0
        streak.append(cnt)
    df["d1_streak"] = streak
    def _stage(r):
        if r["d1_streak"]>=8 and (r["d2_norm"]>=50 or r["credit_norm"]>=50): return 3
        if r["d1_streak"]>=4: return 2
        if r["d3_norm"]>=65 or r["credit_norm"]>=50: return 1
        return 0
    df["alert_stage"] = df.apply(_stage, axis=1)
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def load_stage_breadth():
    if STAGE_BRD_HIST_PATH.exists():
        brd = pd.read_csv(STAGE_BRD_HIST_PATH, index_col=0, parse_dates=True)
        brd.index.name = "date"
        for c in ["Acceleration","Accumulation","Distribution","Deceleration"]:
            if c not in brd.columns: brd[c] = 0.0
        return brd.reset_index()
    if not STAGE_RETURNS_PATH.exists(): return pd.DataFrame()
    raw = pd.read_csv(STAGE_RETURNS_PATH)
    daily = raw[["Ticker","Signal Date","Stage"]].drop_duplicates(subset=["Ticker","Signal Date"]).copy()
    daily["Signal Date"] = pd.to_datetime(daily["Signal Date"])
    breadth = daily.groupby("Signal Date")["Stage"].value_counts(normalize=True).unstack(fill_value=0)*100
    for c in ["Acceleration","Accumulation","Distribution","Deceleration"]:
        if c not in breadth.columns: breadth[c]=0.0
    return breadth.reset_index().rename(columns={"Signal Date":"date"})

@st.cache_data(show_spinner=False, ttl=3600)
def load_stage_coverage():
    if not STAGE_COVERAGE_PATH.exists(): return pd.DataFrame()
    return pd.read_csv(STAGE_COVERAGE_PATH)

@st.cache_data(show_spinner=False, ttl=3600)
def load_early_warning_calibration():
    """Lead/Lift stats + corrected episode table for the 3-stage chips and the
    historical episode table below, rebuilt by backend/calibrate_early_warning.py
    (daily_refresh.py Step 10). Replaces what used to be hardcoded chip strings
    that were never recomputed from data."""
    if not EARLY_WARNING_CALIBRATION_PATH.exists():
        return None
    try:
        return json.loads(EARLY_WARNING_CALIBRATION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

def smooth(s, w): return s if w<=1 else s.rolling(w, min_periods=max(1,w//2)).mean()

# Load
df=load_weekly(); stage_brd=load_stage_breadth(); stage_cov=load_stage_coverage()
latest=df.iloc[-1]
d1_curr=float(latest["d1_norm"]); d2_curr=float(latest["d2_norm"])
d3_curr=float(latest["d3_norm"]); credit_curr=float(latest["credit_norm"])
streak_curr=int(latest["d1_streak"]); alert_curr=int(latest["alert_stage"])
s1_active=d3_curr>=65 or credit_curr>=50
s2_active=streak_curr>=4
s3_active=streak_curr>=8 and (d2_curr>=50 or credit_curr>=50)

# Sidebar
with st.sidebar:
    st.markdown("## 📡 Early Warning"); st.markdown("---")
    yr_from=st.slider("Chart start year",1991,2024,2005)
    smooth_weeks=st.select_slider("Smoothing (weeks)",options=[1,4,8,13,26],value=13)
    st.markdown("**Overlays:**")
    show_d1=st.toggle("D1 · Market Regime",value=True)
    show_d2=st.toggle("D2 · Technical",value=False)
    show_d3=st.toggle("D3 · Liquidity",value=True)
    show_credit=st.toggle("Credit Spread",value=False)
    show_crashes=st.toggle("Crash zones",value=True)
    show_stage=st.toggle("Stage breadth panel",value=True)
    st.markdown("---")
    if STAGE_BRD_HIST_PATH.exists(): st.success("Full stage history loaded.")
    else: st.warning("Stage breadth: Feb 2020 only.\nRun once:\n`python compute_stage_breadth_history.py`")
    st.markdown("---")
    st.caption("D1=raw/85. D2=(raw-2)/12. D3=step 0-100. Credit=(spread-1)/5. All x100.")

# Title & KPIs
st.title("📡 Early Warning System")
st.caption(f"D1 · D2 · D3 · Credit · Market Stage  ·  Data through **{df.index[-1].strftime('%b %d, %Y')}**")
k1,k2,k3,k4,k5,k6=st.columns(6)
k1.metric("D1 Regime",f"{d1_curr:.0f} / 100"); k2.metric("D2 Technical",f"{d2_curr:.0f} / 100")
k3.metric("D3 Liquidity",f"{d3_curr:.0f} / 100"); k4.metric("Credit",f"{latest['credit_spread_for_model']:.2f}%")
k5.metric("D1 Streak",f"{streak_curr}w >= 25"); k6.metric("Alert Stage",f"{ALERT_ICONS[alert_curr]} {ALERT_LABELS[alert_curr]}")

# 3-stage pipeline
st.markdown("### 🚨 3-Stage Alert Pipeline")
ew_calibration = load_early_warning_calibration()

def _chip(title,desc,active,lead,lift):
    fg="#ef4444" if active else "#6b7280"; bg="rgba(239,68,68,0.1)" if active else "rgba(31,41,55,0.6)"
    bdr="#ef4444" if active else "#374151"; ico="🔴 ACTIVE" if active else "inactive"
    return (f"<div style='background:{bg};border:1px solid {bdr};border-radius:8px;padding:14px;text-align:center;'>"
            f"<div style='font-size:0.8rem;color:#9ca3af;'>{title}</div>"
            f"<div style='font-size:1rem;font-weight:700;color:{fg};margin:4px 0;'>{ico}</div>"
            f"<div style='font-size:0.75rem;color:#6b7280;'>{desc}</div>"
            f"<div style='font-size:0.72rem;color:#9ca3af;margin-top:4px;'>Lead: {lead} | Lift: {lift}</div></div>")

def _stage_chip_text(stage_num):
    """Lead/Lift text computed live from data/early_warning_calibration.json
    (backend/calibrate_early_warning.py) instead of a hardcoded string that
    was never recomputed -- a 35-year audit found Stage 1's old figure (0.62x)
    used a different masking convention than Stage 2's, and Stage 3's (~34%)
    had drifted as new episodes accumulated. All three now use one documented
    convention (alert_stage >= N) against one canonical outcome."""
    if not ew_calibration:
        d = DEFAULT_STAGE_TEXT[stage_num]
        return d["lead"], d["lift"]
    s = ew_calibration["stages"].get(str(stage_num))
    if not s or s.get("median_lead_weeks_to_trough") is None:
        return "n/a", "n/a"
    return f"~{s['median_lead_weeks_to_trough']:.0f}w to trough", f"{s['lift']:.2f}x"

lead1, lift1 = _stage_chip_text(1)
lead2, lift2 = _stage_chip_text(2)
lead3, lift3 = _stage_chip_text(3)
c1,c2,c3=st.columns(3)
c1.markdown(_chip("Stage 1 · Monitor","D3 >= 65 or Credit >= 50",s1_active,lead1,lift1),unsafe_allow_html=True)
c2.markdown(_chip("Stage 2 · Actionable","D1 raw > 25 for >= 4 weeks",s2_active,lead2,lift2),unsafe_allow_html=True)
c3.markdown(_chip("Stage 3 · High Conviction","D1 streak >= 8w AND D2/Credit >= 50",s3_active,lead3,lift3),unsafe_allow_html=True)
if ew_calibration:
    st.caption(
        f"Lead = median weeks from an episode's start to the S&P 500 trough within it (episode-level, "
        f"n={ew_calibration['stages']['2']['n_episodes']} Stage 2+ episodes since {ew_calibration['history_start']}). "
        f"Lift = P({ew_calibration['canonical_outcome_label']} | alert_stage >= N) vs. the unconditional base rate "
        f"({ew_calibration['stages']['1']['base_rate_pct']}%), the same convention for all three chips. "
        "Recomputed each refresh -- see the Historical Episodes table below for why episode-level and week-level "
        "read differently."
    )
else:
    st.info("Lead/Lift not calibrated yet — run `python backend/calibrate_early_warning.py`.")
st.markdown("---")

# Inspector slider
st.markdown("### 🔍 Zone Inspector")
df_chart=df[df.index>=pd.Timestamp(f"{yr_from}-01-01")]
date_opts=df_chart.index.strftime("%Y-%m-%d").tolist()
sel_date_s=st.select_slider("Select week",options=date_opts,value=date_opts[-1],label_visibility="collapsed")
sel_date=pd.Timestamp(sel_date_s)

# Chart
has_stage=show_stage and not stage_brd.empty
if has_stage:
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],vertical_spacing=0.03,
                      specs=[[{"secondary_y":True}],[{"secondary_y":False}]],
                      subplot_titles=["SPY (log scale) vs Risk Indicators (0-100)",
                                      "Market Stage Breadth — % of ~100 large-caps"])
else:
    fig=make_subplots(rows=1,cols=1,specs=[[{"secondary_y":True}]],
                      subplot_titles=["SPY (log scale) vs Risk Indicators (0-100)"])

fig.add_trace(go.Scatter(x=df_chart.index,y=df_chart["sp500"],name="SPY",mode="lines",
                         line=dict(color="#e5e7eb",width=2.5),
                         hovertemplate="<b>SPY</b>: %{y:,.0f}<br>%{x|%Y-%m-%d}<extra></extra>"),
              row=1,col=1,secondary_y=False)

if show_crashes:
    for zs,ze,zlabel,zcolor in CRASH_ZONES:
        s,e=pd.Timestamp(zs),pd.Timestamp(ze)
        if e>=df_chart.index[0] and s<=df_chart.index[-1]:
            x0=max(s,df_chart.index[0]).strftime("%Y-%m-%d")
            x1=min(e,df_chart.index[-1]).strftime("%Y-%m-%d")
            fig.add_vrect(x0=x0,x1=x1,fillcolor=zcolor,opacity=0.07,line_width=0,row=1,col=1)
            fig.add_annotation(x=x0,y=0.98,xref="x",yref="paper",text=zlabel,showarrow=False,
                               font=dict(size=9,color=zcolor),xanchor="left",yanchor="top")

if show_d1:
    r=df_chart["d1_norm"].dropna()
    if smooth_weeks>1:
        fig.add_trace(go.Scatter(x=r.index,y=r.values,mode="lines",showlegend=False,
                                 line=dict(color="#f97316",width=0.8),opacity=0.2,hoverinfo="skip"),row=1,col=1,secondary_y=True)
    s_=smooth(r,smooth_weeks)
    fig.add_trace(go.Scatter(x=s_.index,y=s_.values,name=f"D1 ({smooth_weeks}w)" if smooth_weeks>1 else "D1",
                             mode="lines",line=dict(color="#f97316",width=2.5),
                             hovertemplate="<b>D1</b>: %{y:.1f}<extra></extra>"),row=1,col=1,secondary_y=True)

if show_d2:
    r=df_chart["d2_norm"].dropna()
    if smooth_weeks>1:
        fig.add_trace(go.Scatter(x=r.index,y=r.values,mode="lines",showlegend=False,
                                 line=dict(color="#a855f7",width=0.8),opacity=0.2,hoverinfo="skip"),row=1,col=1,secondary_y=True)
    s_=smooth(r,smooth_weeks)
    fig.add_trace(go.Scatter(x=s_.index,y=s_.values,name=f"D2 ({smooth_weeks}w)" if smooth_weeks>1 else "D2",
                             mode="lines",line=dict(color="#a855f7",width=2.5,dash="dash"),
                             hovertemplate="<b>D2</b>: %{y:.1f}<extra></extra>"),row=1,col=1,secondary_y=True)

if show_d3:
    d3=df_chart["d3_norm"].dropna()
    fig.add_trace(go.Scatter(x=d3.index,y=d3.values,name="D3 Liquidity",mode="lines",
                             line=dict(color="#38bdf8",width=2,shape="hv"),
                             fill="tozeroy",fillcolor="rgba(56,189,248,0.07)",
                             hovertemplate="<b>D3</b>: %{y:.0f}<extra></extra>"),row=1,col=1,secondary_y=True)

if show_credit:
    r=df_chart["credit_norm"].dropna()
    if smooth_weeks>1:
        fig.add_trace(go.Scatter(x=r.index,y=r.values,mode="lines",showlegend=False,
                                 line=dict(color="#f43f5e",width=0.8),opacity=0.2,hoverinfo="skip"),row=1,col=1,secondary_y=True)
    s_=smooth(r,smooth_weeks)
    fig.add_trace(go.Scatter(x=s_.index,y=s_.values,name=f"Credit ({smooth_weeks}w)" if smooth_weeks>1 else "Credit",
                             mode="lines",line=dict(color="#f43f5e",width=2.5,dash="dashdot"),
                             hovertemplate="<b>Credit</b>: %{y:.1f}<extra></extra>"),row=1,col=1,secondary_y=True)

fig.add_trace(go.Scatter(x=[df_chart.index[0].strftime("%Y-%m-%d"),df_chart.index[-1].strftime("%Y-%m-%d")],
                         y=[29.4,29.4],name="Stage 2 trigger",mode="lines",
                         line=dict(color="#f97316",width=1,dash="dot"),opacity=0.4,
                         hovertemplate="Stage 2 trigger<extra></extra>"),row=1,col=1,secondary_y=True)

fig.add_vline(x=sel_date.strftime("%Y-%m-%d"),line_dash="dash",line_color="#facc15",line_width=1.5,row=1,col=1)

if has_stage:
    sb=stage_brd[stage_brd["date"]>=df_chart.index[0]].copy().set_index("date")
    sb_s=sb.rolling(10,min_periods=3).mean().reset_index()
    fig.add_trace(go.Scatter(x=sb_s["date"],y=sb_s["Deceleration"],name="Decel %",mode="lines",
                             line=dict(color="#ef4444",width=2),fill="tozeroy",fillcolor="rgba(239,68,68,0.12)",
                             hovertemplate="<b>Decel</b>: %{y:.0f}%<extra></extra>"),row=2,col=1)
    fig.add_trace(go.Scatter(x=sb_s["date"],y=sb_s["Acceleration"],name="Accel %",mode="lines",
                             line=dict(color="#22c55e",width=2),
                             hovertemplate="<b>Accel</b>: %{y:.0f}%<extra></extra>"),row=2,col=1)
    fig.add_hline(y=50,line_dash="dot",line_color="#4b5563",line_width=1,row=2,col=1)
    fig.add_vline(x=sel_date.strftime("%Y-%m-%d"),line_dash="dash",line_color="#facc15",line_width=1.5,row=2,col=1)

fig.update_layout(height=740 if has_stage else 500,template="plotly_dark",
                  paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",hovermode="x unified",
                  legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1),
                  margin=dict(l=60,r=80,t=60,b=40))
fig.update_yaxes(type="log",title_text="SPY (log)",gridcolor="#1f2937",row=1,col=1,secondary_y=False)
fig.update_yaxes(title_text="Score (0-100)",range=[0,105],gridcolor="#1f2937",showgrid=False,row=1,col=1,secondary_y=True)
if has_stage: fig.update_yaxes(title_text="Stage %",range=[0,100],gridcolor="#1f2937",row=2,col=1)
fig.update_xaxes(gridcolor="#1f2937",showgrid=True)
st.plotly_chart(fig,use_container_width=True)

# Week Inspector
st.markdown(f"### 📊 Week Inspector: `{sel_date_s}`")
idx_pos=df.index.get_indexer([sel_date],method="nearest")[0]
sel_row=df.iloc[idx_pos]; sel_real=df.index[idx_pos]
p_left,p_right=st.columns(2)

with p_left:
    st.markdown("**Conditions that week**")
    d2_str=f"{sel_row['d2_norm']:.0f} / 100" if pd.notna(sel_row['d2_norm']) else "N/A (pre-1993)"
    cs_str=f"{sel_row['composite_risk_score']:.0f} / 100" if pd.notna(sel_row.get('composite_risk_score')) else "N/A"
    for k,v in [("Date",sel_real.strftime("%Y-%m-%d")),("SPY",f"{sel_row['sp500']:,.0f}"),
                ("D1 Regime",f"{sel_row['d1_norm']:.0f} / 100"),("D2 Technical",d2_str),
                ("D3 Liquidity",f"{sel_row['d3_norm']:.0f} / 100"),
                ("Credit Spread",f"{sel_row['credit_spread_for_model']:.2f}%"),
                ("VIX",f"{sel_row['vix']:.1f}"),("MOVE",f"{sel_row['move']:.1f}"),
                ("D1 Streak",f"{int(sel_row['d1_streak'])}w >= 25"),
                ("Alert Stage",f"{ALERT_ICONS[int(sel_row['alert_stage'])]} {ALERT_LABELS[int(sel_row['alert_stage'])]}"),
                ("Composite",cs_str)]:
        st.markdown(f"**{k}:** {v}")
    st.markdown("---"); st.markdown("**Actual forward outcomes**")
    any_out=False
    for ck,lbl in [("fwd_max_dd_4w","Max DD 4w"),("fwd_max_dd_8w","Max DD 8w"),
                   ("fwd_max_dd_13w","Max DD 13w"),("fwd_max_dd_26w","Max DD 26w")]:
        if ck in df.columns:
            val=sel_row.get(ck)
            if pd.notna(val):
                any_out=True
                icon="🔴" if val<-0.10 else "🟡" if val<-0.05 else "🟢"
                st.markdown(f"{icon} **{lbl}:** {val*100:.1f}%")
    if not any_out: st.caption("No forward data (too recent).")

with p_right:
    sel_stage=int(sel_row["alert_stage"]); stage_name=ALERT_LABELS[sel_stage]
    matching=df[df["alert_stage"]==sel_stage]
    matched_n=len(matching.dropna(subset=["drop_10_13w"]))
    st.markdown(f"**Drop rates: {ALERT_ICONS[sel_stage]} {stage_name}** (n={matched_n} weeks)")
    for col_key,label in [("drop_5_4w",">5% in 4w"),("drop_10_4w",">10% in 4w"),
                           ("drop_5_13w",">5% in 13w"),("drop_10_13w",">10% in 13w"),
                           ("drop_15_13w",">15% in 13w"),("drop_20_13w",">20% in 13w")]:
        if col_key not in df.columns: continue
        sub=matching.dropna(subset=[col_key]); base=df.dropna(subset=[col_key])
        if len(sub)<5: st.markdown(f"**{label}:** n/a"); continue
        rate=sub[col_key].mean()*100; base_rate=base[col_key].mean()*100
        lift=rate/base_rate if base_rate>0 else 1.0
        bar="X"*max(1,int(rate/5))
        clr="#ef4444" if rate>base_rate*1.5 else "#f59e0b" if rate>base_rate else "#22c55e"
        st.markdown(f"**{label}:** <span style='color:{clr};font-weight:700;'>{rate:.0f}%</span> "
                    f"<span style='color:#38bdf8;'>{bar}</span> "
                    f"<span style='color:#6b7280;'>(base {base_rate:.0f}% · {lift:.2f}x)</span>",
                    unsafe_allow_html=True)
    if not stage_brd.empty:
        st.markdown("---"); st.markdown("**Stage breadth at this date**")
        sb_match=stage_brd[(stage_brd["date"]-sel_date).abs()<=pd.Timedelta("5D")]
        if not sb_match.empty:
            row_sb=sb_match.iloc[-1]
            for s,ic in [("Acceleration","🟢"),("Accumulation","🟡"),("Distribution","🟠"),("Deceleration","🔴")]:
                st.markdown(f"{ic} **{s}:** {row_sb.get(s,0):.0f}%")
        else: st.caption("Stage breadth not available for this date (pre-2020 without full history).")

# Historical episode table
st.markdown("---"); st.markdown("### 📅 Historical Early Warning Episodes")
st.caption("All Stage 2+ activations since 1991. Shows when the pipeline fired and what actually followed.")
if ew_calibration and ew_calibration.get("episodes"):
    episodes = ew_calibration["episodes"]
    outcome_icon = {"Confirmed Drop": "🔴", "Contained": "🟡", "No Confirmed Drop": "🟢", "Active": "⏳"}
    old_icon = {"Drop": "🔴", "Flat": "🟡", "Rally": "🟢", "Active": "⏳"}
    hist = pd.DataFrame([
        {
            "Start": e["start_date"], "End": e["end_date"],
            "Stage": "🔴 Stage 3" if e["peak_stage"] == 3 else "🟠 Stage 2",
            "D1 peak": f"{e['d1_peak']:.0f}", "D2 peak": f"{e['d2_peak']:.0f}",
            "SPY entry": f"{e['spy_entry']:,.0f}",
            "SPY +13w from start (old metric)": (
                f"{old_icon.get(e['old_outcome_label'],'')} {e['spy_chg_13w_from_start_pct']:+.1f}%"
                if e["spy_chg_13w_from_start_pct"] is not None else f"{old_icon.get(e['old_outcome_label'],'')} —"
            ),
            "Worst drawdown, episode+13w (corrected)": f"{outcome_icon.get(e['corrected_outcome_label'],'')} {e['worst_drawdown_during_episode_pct']:+.1f}%",
        }
        for e in episodes
    ])
    st.dataframe(hist, use_container_width=True, hide_index=True)
    es = ew_calibration["episode_summary"]
    st.caption(
        f"{es['n_episodes']} episodes total  |  Confirmed drop (≥10% worst drawdown): {es['n_confirmed_drop']}  |  "
        f"Contained (5-10%): {es['n_contained']}  |  No confirmed drop: {es['n_no_confirmed_drop']}  |  "
        f"Current: {ALERT_ICONS[alert_curr]} {ALERT_LABELS[alert_curr]}"
    )
    st.caption(
        "Two outcome columns on purpose. **Old metric** (SPY change exactly 13 weeks after the episode "
        "*started*) is what this table used to show alone -- it can mislabel an episode that fired near a "
        "trough as a 'Rally' even when the alert correctly caught a crash already underway (2020-03-19 is "
        "the clean example: +44.6% by the old metric, because the episode began right at the COVID bottom). "
        "**Corrected metric** (worst drawdown reached at any point from the episode's start through 13 weeks "
        "after it ends) answers the more useful question and matches the methodology the Lift figures above "
        "already use. Even under the corrected metric, most Stage 2+ activations are still false alarms in "
        "absolute terms -- but that's expected and doesn't contradict the Lift numbers, which measure "
        "improvement *over the base rate*, not a >50% hit-rate guarantee."
    )
else:
    st.info("Episode table not available yet — run `python backend/calibrate_early_warning.py`.")

# Current stage snapshot
if not stage_cov.empty:
    st.markdown("---"); st.markdown("### 🎯 Current Market Stage Snapshot")
    snap_date=stage_brd["date"].max().strftime("%Y-%m-%d") if not stage_brd.empty else "latest scan"
    st.caption(f"As of **{snap_date}** · {len(stage_cov)} tickers")
    stage_counts=stage_cov["Latest Stage"].value_counts(); total=stage_counts.sum()
    sn1,sn2,sn3,sn4=st.columns(4)
    for col_w,sname in zip([sn1,sn2,sn3,sn4],["Acceleration","Accumulation","Distribution","Deceleration"]):
        cnt=int(stage_counts.get(sname,0)); pct=cnt/total*100 if total>0 else 0
        clr=STAGE_COLORS[sname]
        col_w.markdown(f"<div style='text-align:center;background:#1f2937;border-radius:8px;padding:16px;"
                       f"border-left:4px solid {clr};'><div style='color:{clr};font-size:2rem;font-weight:700;'>{pct:.0f}%</div>"
                       f"<div style='color:#d1d5db;font-size:0.9rem;'>{sname}</div>"
                       f"<div style='color:#6b7280;font-size:0.75rem;'>{cnt} tickers</div></div>",unsafe_allow_html=True)
    if "Sector" in stage_cov.columns:
        with st.expander("Sector breakdown",expanded=False):
            sec=stage_cov.groupby(["Sector","Latest Stage"]).size().unstack(fill_value=0)
            for s in ["Acceleration","Accumulation","Distribution","Deceleration"]:
                if s not in sec.columns: sec[s]=0
            sec=sec[[c for c in ["Acceleration","Accumulation","Distribution","Deceleration"] if c in sec.columns]]
            st.dataframe(sec,use_container_width=True)

st.markdown("---")
with st.expander("📖 Signal lead-time reference",expanded=False):
    if ew_calibration:
        s1d,s2d,s3d = ew_calibration["stages"]["1"],ew_calibration["stages"]["2"],ew_calibration["stages"]["3"]
        st.markdown(f"""
| Signal | Lead time (median, to trough) | Lift ({ew_calibration['canonical_outcome_label']}) |
|---|---|---|
| D3 >= 65 or Credit >= 50 (Stage 1) | ~{s1d['median_lead_weeks_to_trough']:.0f} weeks | {s1d['lift']:.2f}x |
| D1 raw > 25 for 4+ weeks (Stage 2) | ~{s2d['median_lead_weeks_to_trough']:.0f} weeks | **{s2d['lift']:.2f}x actionable** |
| D1 streak >= 8w + D2/Credit (Stage 3) | ~{s3d['median_lead_weeks_to_trough']:.0f} weeks | **{s3d['lift']:.2f}x, {s3d['pct_episodes_confirmed_drop']:.0f}% of episodes confirmed drop** |
| Market Stage Decel >= 50% | Same week | Confirming signal (not separately calibrated) |

Recomputed from {ew_calibration['n_weeks']:,} weekly observations, {ew_calibration['history_start']} to {ew_calibration['history_end']} — see `backend/calibrate_early_warning.py`.

**D3 limitation:** does not detect exogenous shocks (2025 tariff: D3 stayed at 20 throughout — confirmed directly against the data; the pipeline still escalated to Stage 3 via D1 and credit instead).

**To fill stage breadth history:** run `python compute_stage_breadth_history.py` once from the project folder.
""")
    else:
        st.info("Lead/Lift table not calibrated yet — run `python backend/calibrate_early_warning.py`.")
        st.markdown("**To fill stage breadth history:** run `python compute_stage_breadth_history.py` once from the project folder.")
st.caption("weekly_feature_outcomes.csv (1991-2026) | .tmp/market_stage_validation/ (Feb 2020-2026) | credit_spread_for_model = BAA10Y/HY blend")

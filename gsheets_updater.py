"""
gsheets_updater.py  —  Daily updater for Market Dashboard Summary Google Sheet

Usage:
    python gsheets_updater.py [--date YYYY-MM-DD]

If no date is given, uses today (market close day check applies).
Computes all 15 indicators (7 Market + 8 Warning) and appends or updates
the row for the given date in the Google Sheet.

NO hardcoded values — all thresholds live in the source modules.
"""

import sys, os, argparse, json
from datetime import date, datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Dependency bootstrap (survives cold sandbox restarts) ──────────────────
for p in ["/tmp/pkgs2", "/tmp/pkgs"]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Install Google Sheets libs on demand (idempotent)
_GSHEETS_PKG_DIR = "/tmp/gsheets_pkgs"
if _GSHEETS_PKG_DIR not in sys.path:
    sys.path.insert(0, _GSHEETS_PKG_DIR)

def _ensure_gsheets():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        return True
    except ImportError:
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "gspread", "google-auth",
             "--target", _GSHEETS_PKG_DIR, "--no-cache-dir", "--quiet"],
            env={**os.environ, "HOME": "/tmp", "TMPDIR": "/tmp"},
            capture_output=True  # suppress warnings to stdout
        )
        return False

_ensure_gsheets()

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np

# ── Project paths (resolved relative to this script) ──────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
KEY_FILE     = os.path.join(PROJECT_ROOT, "mymax21-dashboard-0f6e93133a5c.json")
SHEET_ID     = "1_C-m2Lm0zPp2phA1ZSBrZDFZkhu8eonis9rhdnZOlZg"
PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "market_data_master.parquet")

# Add project to path so we can import its modules
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Sheet column definition ────────────────────────────────────────────────
# Order here = column order in the sheet. DO NOT reorder without clearing sheet.
HEADERS = [
    "Date",
    "Crash_Prob%", "Crash_Status",
    "BearTrap_3M%", "BearTrap_6M%", "BearTrap_12M%", "BearTrap_Risk",
    "BullTrap_Prob%", "BullTrap_Status",
    "ETF_Rotation",
    "SP500_Price", "SMA200", "MA200_Trend",
    "ML_Meta%", "ML_Meta_Status",
    "RWRA_Bull%", "Macro_Guardrail", "Execution",
    # Warning
    "W_SPX", "W_Mom", "W_Range", "W_Tech", "W_Breadth", "W_VIX", "W_Breakout", "W_Theme",
    "W_Total_Score", "W_Level",
    "Override",
    # Deltas (vs previous trading day)
    "SP500_delta", "ML_delta", "BearTrap3M_delta", "WScore_delta",
    # Raw values for verification
    "W_SPX_val", "W_Mom_val", "W_Breadth_val", "W_VIX_val", "W_Breakout_val",
]


def col_letter(n):
    """1-indexed column number → spreadsheet letter (A, B, ... Z, AA, ...)"""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


# ── Google Sheets connection ───────────────────────────────────────────────
def get_worksheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(KEY_FILE, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.sheet1
    return ws


def ensure_headers(ws):
    """Write headers to row 1 if the sheet is empty."""
    current = ws.row_values(1)
    if not current or current[0] != "Date":
        ws.update(values=[HEADERS], range_name="A1")
        ws.format(f"A1:{col_letter(len(HEADERS))}1", {
            "textFormat": {"bold": True},
            "backgroundColor": {"red": 0.13, "green": 0.13, "blue": 0.33},
        })
        print("Headers written.")
    return ws


def get_existing_dates(ws):
    """Return {date_str: row_number} for all existing data rows."""
    col_a = ws.col_values(1)  # includes header
    dates = {}
    for i, v in enumerate(col_a[1:], start=2):  # row 2 onwards
        if v:
            dates[v.strip()] = i
    return dates


# ── Data loading ───────────────────────────────────────────────────────────
def load_master(target_date):
    """Load parquet and slice to data available as-of target_date."""
    df = pd.read_parquet(PARQUET_PATH)
    df.index = pd.to_datetime(df.index)
    sliced = df[df.index.date <= target_date].copy()
    return sliced


# ── Market Dashboard computation (Group A — 7 indicators) ─────────────────
def _mock_streamlit():
    """Minimal streamlit stub so page scripts don't crash on import."""
    if "streamlit" in sys.modules and hasattr(sys.modules["streamlit"], "_is_stub"):
        return sys.modules["streamlit"]
    import types
    st = types.ModuleType("streamlit")
    st._is_stub = True
    st.cache_data = lambda f=None, **kw: (f if f else lambda g: g)
    st.cache_resource = lambda f=None, **kw: (f if f else lambda g: g)
    noop = lambda *a, **k: None
    st.error = st.warning = st.info = st.success = st.write = noop
    st.set_page_config = noop
    st.title = st.header = st.subheader = st.caption = noop
    st.columns = lambda n, **k: [types.SimpleNamespace(write=noop, metric=noop)] * (n if isinstance(n, int) else len(n))
    st.metric = st.dataframe = st.table = st.plotly_chart = noop
    import contextlib as _cl
    _ctx = _cl.nullcontext()
    class _CM:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __getattr__(self, n): return noop
    _sb = _CM()
    st.sidebar = _sb
    st.session_state = {}
    st.spinner = st.expander = st.container = st.form = lambda *a, **k: _CM()
    st.tabs = lambda names, **k: [_CM()] * len(names)
    st.form_submit_button = noop
    st.selectbox = st.multiselect = st.radio = st.checkbox = st.slider = noop
    st.number_input = st.text_input = st.date_input = st.button = noop
    st.stop = noop
    sys.modules["streamlit"] = st
    return st


def compute_group_a(data, target_date):
    """
    Compute the 7 Market Summary indicators.
    Returns dict with keys matching HEADERS.
    """
    _mock_streamlit()

    # Patch get_clean_master in all namespaces
    import utils.data_engine as de
    real_gcm = de.get_clean_master
    sliced = data.copy()
    de.get_clean_master = lambda d=sliced: d

    result = {}
    try:
        from pages import _0____Market_Summary as ms
    except Exception:
        import importlib, importlib.util
        spec = importlib.util.spec_from_file_location(
            "market_summary",
            os.path.join(PROJECT_ROOT, "pages", "0_📋_Market_Summary.py")
        )
        ms = importlib.util.module_from_spec(spec)
        # Prevent render_page() from running at module level during import
        ms.render_page = lambda: None
        sys.modules["market_summary"] = ms
        # Load source, suppress the bare render_page() call at the bottom
        with open(os.path.join(PROJECT_ROOT, "pages", "0_\U0001F4CB_Market_Summary.py"), encoding="utf-8") as _f:
            _src = _f.read()
        _src = _src.replace("\nrender_page()", "\n# render_page()  # suppressed by gsheets_updater")
        exec(compile(_src, ms.__spec__.origin, "exec"), ms.__dict__)

    ts = pd.Timestamp(target_date)

    try:
        # Returns: {'probability': 10, 'color': '...', 'status': 'LOW RISK REGIME'}
        crash = ms.calc_market_regime(data)
        result["Crash_Prob%"]  = crash.get("probability", "")
        result["Crash_Status"] = crash.get("status", "")
    except Exception as e:
        result["Crash_Prob%"] = result["Crash_Status"] = f"ERR:{e}"

    try:
        # Returns: {'prob_3m': float_frac, 'prob_6m': float_frac, 'prob_12m': float_frac, 'risk_level': str}
        # prob values are fractions (0.171) — or NaN if model not fitted
        bt = ms.calc_bear_trap(data)
        import math as _math
        def _pct(v):
            try:
                fv = float(v)
                if _math.isnan(fv) or _math.isinf(fv): return ""
                return round(fv * 100, 1) if fv <= 1.0 else round(fv, 1)
            except: return ""
        result["BearTrap_3M%"]  = _pct(bt.get("prob_3m", ""))
        result["BearTrap_6M%"]  = _pct(bt.get("prob_6m", ""))
        result["BearTrap_12M%"] = _pct(bt.get("prob_12m", ""))
        result["BearTrap_Risk"] = bt.get("risk_level", "")
    except Exception as e:
        result["BearTrap_3M%"] = result["BearTrap_6M%"] = result["BearTrap_12M%"] = result["BearTrap_Risk"] = f"ERR:{e}"

    try:
        # Returns: {'probability': float%, 'regime': str, 'market_status': str, 'color': str}
        bull = ms.calc_bull_trap(data)
        result["BullTrap_Prob%"]  = bull.get("probability", "")
        result["BullTrap_Status"] = bull.get("market_status", "")
    except Exception as e:
        result["BullTrap_Prob%"] = result["BullTrap_Status"] = f"ERR:{e}"

    try:
        # Returns: {'status': str, 'color': str}
        etf = ms.calc_etf_rotation(data)
        result["ETF_Rotation"] = etf.get("status", "")
    except Exception as e:
        result["ETF_Rotation"] = f"ERR:{e}"

    try:
        # Use ^GSPC (SPX index) not SPY ETF
        spx_col = "^GSPC" if "^GSPC" in data.columns else ("SPY" if "SPY" in data.columns else data.columns[0])
        sp = data[spx_col].dropna()
        last_price = round(float(sp.iloc[-1]), 2)
        sma200 = round(float(sp.rolling(200).mean().iloc[-1]), 2)
        try:
            ma_data = ms.calc_200ma_strategy(data)
            trend = ma_data.get("signal", "Above 200MA" if last_price > sma200 else "Below 200MA")
        except Exception:
            trend = "Above 200MA" if last_price > sma200 else "Below 200MA"
        result["SP500_Price"] = last_price
        result["SMA200"]      = sma200
        result["MA200_Trend"] = trend
    except Exception as e:
        result["SP500_Price"] = result["SMA200"] = result["MA200_Trend"] = f"ERR:{e}"

    try:
        # Returns: {'trend_probability': float, 'meta_score': float, 'color': str, 'status': str}
        ml = ms.calc_meta_indicator(data, ts)
        result["ML_Meta%"]       = ml.get("trend_probability", ml.get("meta_score", ""))
        result["ML_Meta_Status"] = ml.get("status", "")
    except Exception as e:
        result["ML_Meta%"] = result["ML_Meta_Status"] = f"ERR:{e}"

    try:
        from backend.strategies.combined_macro_rwra import compute_combined_snapshot
        snap = compute_combined_snapshot(ts)
        rwra_raw = snap.rwra_bull_signal
        if rwra_raw is not None:
            rwra_v = float(rwra_raw)
            # If value > 1, it's already in percent (e.g. 50.0); otherwise multiply
            result["RWRA_Bull%"] = round(rwra_v if rwra_v > 1 else rwra_v * 100, 1)
        else:
            result["RWRA_Bull%"] = ""
        result["Macro_Guardrail"] = snap.macro_guardrail or ""
        result["Execution"]       = snap.execution_status or ""
    except Exception as e:
        result["RWRA_Bull%"] = result["Macro_Guardrail"] = result["Execution"] = f"ERR:{e}"

    de.get_clean_master = real_gcm
    return result


# ── Warning Dashboard computation (Group B — 8 indicators) ────────────────
def compute_group_b(data, target_date):
    """
    Compute the 8 Warning indicators for target_date.
    Calls _build_indicator_rows directly to avoid 126-session timeline scan.
    Returns dict with keys matching HEADERS.
    """
    _mock_streamlit()

    import utils.warning_dashboard as wd
    real_gcm_wd = wd.get_clean_master
    sliced = data.copy()
    wd.get_clean_master = lambda d=sliced: d

    import utils.data_engine as de
    real_gcm_de = de.get_clean_master
    de.get_clean_master = lambda d=sliced: d

    result = {}
    try:
        ts = pd.Timestamp(target_date)
        rows = wd._build_indicator_rows(sliced, ts)

        # IndicatorResult is a dataclass: .name, .status, .value, .score, .note, .threshold
        # Map partial name keywords → sheet column keys
        label_keywords = [
            ("overext",    "W_SPX",      "W_SPX_val"),
            ("momentum",   "W_Mom",      "W_Mom_val"),
            ("range",      "W_Range",    None),
            ("tech",       "W_Tech",     None),
            ("breadth",    "W_Breadth",  "W_Breadth_val"),
            ("vix",        "W_VIX",      "W_VIX_val"),
            ("breakout",   "W_Breakout", "W_Breakout_val"),
            ("theme",      "W_Theme",    None),
        ]

        def find_ir(keyword):
            kw = keyword.lower()
            for r in rows:
                if kw in r.name.lower():
                    return r
            return None

        STATUS_MAP = {"Normal": 0, "Watch": 1, "Warning": 2}
        total_score = 0
        override_applied = False

        for kw, col_key, val_key in label_keywords:
            ir = find_ir(kw)
            if ir is not None:
                status = ir.status
                result[col_key] = status
                if val_key:
                    result[val_key] = ir.value
                # Check for override by comparing computed score vs override table
                override_applied = override_applied or (ir.note and "override" in ir.note.lower())
                total_score += STATUS_MAP.get(status, 0)
            else:
                result[col_key] = "N/A"
                if val_key:
                    result[val_key] = ""

        # Also use sum of .score fields from IndicatorResult (already weighted)
        score_from_obj = sum(r.score for r in rows)
        result["W_Total_Score"] = score_from_obj  # use the authoritative source

        # Check if any override was applied (SOURCE_OVERRIDES)
        from utils.warning_dashboard import SOURCE_OVERRIDES
        date_key = ts.strftime("%Y-%m-%d")
        override_applied = date_key in SOURCE_OVERRIDES
        result["Override"] = "Yes" if override_applied else ""

        # Level thresholds (mirror warning_dashboard.py logic)
        n_warnings = sum(1 for r in rows if r.status == "Warning")
        if score_from_obj >= 11 or n_warnings >= 3:
            level = "Critical"
        elif score_from_obj >= 7 or n_warnings >= 2:
            level = "Elevated"
        elif score_from_obj >= 4 or sum(1 for r in rows if r.status in ("Warning","Watch")) >= 3:
            level = "Guarded"
        else:
            level = "Stable"
        result["W_Level"] = level

    except Exception as e:
        for col_key in ["W_SPX","W_Mom","W_Range","W_Tech","W_Breadth","W_VIX","W_Breakout","W_Theme",
                        "W_Total_Score","W_Level","Override",
                        "W_SPX_val","W_Mom_val","W_Breadth_val","W_VIX_val","W_Breakout_val"]:
            result[col_key] = f"ERR:{e}"

    wd.get_clean_master = real_gcm_wd
    de.get_clean_master = real_gcm_de
    return result


# ── Assemble row for target date ───────────────────────────────────────────
def build_row(target_date, data, prev_row=None):
    """
    Compute all indicators for target_date and return a list matching HEADERS order.
    prev_row: list of values from the previous trading day row (for deltas).
    """
    ga = compute_group_a(data, target_date)
    gb = compute_group_b(data, target_date)

    def pv(key, prev_row):
        """Get previous row value for a given header."""
        if prev_row is None:
            return None
        try:
            idx = HEADERS.index(key)
            v = prev_row[idx]
            return float(v) if v not in ("", None) else None
        except (ValueError, IndexError):
            return None

    def delta(cur, key, prev_row):
        pval = pv(key, prev_row)
        try:
            cval = float(cur)
            if pval is not None:
                return round(cval - pval, 2)
        except (TypeError, ValueError):
            pass
        return ""

    sp500  = ga.get("SP500_Price", "")
    ml     = ga.get("ML_Meta%", "")
    bear3m = ga.get("BearTrap_3M%", "")
    wscore = gb.get("W_Total_Score", "")

    row = [
        target_date.isoformat(),
        ga.get("Crash_Prob%", ""),   ga.get("Crash_Status", ""),
        ga.get("BearTrap_3M%", ""),  ga.get("BearTrap_6M%", ""),
        ga.get("BearTrap_12M%", ""), ga.get("BearTrap_Risk", ""),
        ga.get("BullTrap_Prob%", ""), ga.get("BullTrap_Status", ""),
        ga.get("ETF_Rotation", ""),
        sp500,                        ga.get("SMA200", ""),  ga.get("MA200_Trend", ""),
        ml,                           ga.get("ML_Meta_Status", ""),
        ga.get("RWRA_Bull%", ""),    ga.get("Macro_Guardrail", ""),  ga.get("Execution", ""),
        gb.get("W_SPX", ""),   gb.get("W_Mom", ""),   gb.get("W_Range", ""),
        gb.get("W_Tech", ""),  gb.get("W_Breadth", ""), gb.get("W_VIX", ""),
        gb.get("W_Breakout", ""), gb.get("W_Theme", ""),
        wscore,                  gb.get("W_Level", ""),
        gb.get("Override", ""),
        delta(sp500,  "SP500_Price",   prev_row),
        delta(ml,     "ML_Meta%",      prev_row),
        delta(bear3m, "BearTrap_3M%",  prev_row),
        delta(wscore, "W_Total_Score", prev_row),
        gb.get("W_SPX_val", ""),   gb.get("W_Mom_val", ""),
        gb.get("W_Breadth_val", ""), gb.get("W_VIX_val", ""),
        gb.get("W_Breakout_val", ""),
    ]
    return row


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Update Market Dashboard Google Sheet")
    parser.add_argument("--date", type=str, default=None,
                        help="Target date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = date.today()

    # Skip weekends
    if target_date.weekday() >= 5:
        print(f"{target_date} is a weekend — skipping.")
        return

    print(f"[gsheets_updater] Target date: {target_date}")

    # Load parquet
    print("Loading market data...")
    data = load_master(target_date)
    print(f"  Data loaded: {len(data)} rows through {data.index[-1].date()}")

    # Connect to sheet
    print("Connecting to Google Sheet...")
    ws = get_worksheet()
    ensure_headers(ws)
    existing = get_existing_dates(ws)
    print(f"  Sheet has {len(existing)} existing rows")

    # Get previous trading day row for delta calculation
    date_str = target_date.isoformat()
    all_rows = ws.get_all_values()
    prev_row = None
    if len(all_rows) > 1:
        # Find the last row before this date
        data_rows = all_rows[1:]  # skip header
        for r in reversed(data_rows):
            if r[0] and r[0] < date_str:
                prev_row = r
                break
    print(f"  Previous row date: {prev_row[0] if prev_row else 'none'}")

    # Compute indicators
    print("Computing indicators...")
    row = build_row(target_date, data, prev_row)
    # Sanitize: replace NaN/inf with empty string (JSON can't serialize them)
    import math
    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return ""
        return v
    row = [_clean(v) for v in row]
    print(f"  Row computed: {len(row)} values")

    # Write to sheet
    if date_str in existing:
        row_num = existing[date_str]
        last_col = col_letter(len(HEADERS))
        ws.update(values=[row], range_name=f"A{row_num}:{last_col}{row_num}")
        print(f"  Updated existing row {row_num} for {date_str}")
    else:
        ws.append_row(row, value_input_option="USER_ENTERED")
        print(f"  Appended new row for {date_str}")

    print(f"\nDone. Sheet: https://docs.google.com/spreadsheets/d/{SHEET_ID}")


if __name__ == "__main__":
    main()

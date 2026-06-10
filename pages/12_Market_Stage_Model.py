from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import streamlit as st

from utils.ui_utils import render_ecosystem_sidebar, render_master_controls


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "market_stage_model"
MODEL_APP_PATH = MODEL_DIR / "app.py"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

spec = importlib.util.spec_from_file_location("market_stage_model_streamlit_app", MODEL_APP_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load Market Stage model from {MODEL_APP_PATH}")

market_stage_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(market_stage_app)


def main() -> None:
    market_stage_app.inject_css()

    with st.sidebar:
        render_master_controls()
        render_ecosystem_sidebar()
        st.markdown("---")
        st.header("Market Stage Controls")
        primary_ticker = st.text_input("Primary ticker", value="SPY").strip().upper() or "SPY"
        lookback_days = st.slider("Lookback days", min_value=90, max_value=730, value=365, step=30)
        include_partial = st.toggle("Include in-progress daily bar", value=False)
        if market_stage_app.get_polygon_api_key():
            st.caption("Polygon API key loaded from local secrets.")
        else:
            st.warning("Polygon API key not found. NYSE TRIN will be unavailable.")
        st.info("Real historical OHLCV only. Close-confirmed signals should use next-session execution.")

    st.title("Market Stage Model")
    st.caption("Strategy Suite model: independent VWMA/VMA market-stage classifier.")
    market_stage_app.render_dashboard(primary_ticker, lookback_days, include_partial)


main()

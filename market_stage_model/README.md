# Market Stage Model

Standalone Streamlit dashboard for identifying the current structural market stage of a stock or ETF.

All dashboard data is pulled from real yfinance historical OHLCV. The app does not generate synthetic/demo market bars.

## Run

```powershell
cd D:\Backtest\Makret-Bull-and-Bear-Trap\market_stage_model
python -m streamlit run app.py --server.port 8520
```

If the root project uses the local `.deps` dependency folder, run from the repo root instead:

```powershell
$env:PYTHONPATH=".deps"
python -m streamlit run market_stage_model\app.py --server.port 8520
```

## Polygon TRIN

NYSE TRIN is calculated from Polygon grouped daily bars and NYSE ticker reference data when `POLYGON_API_KEY` is available.

PowerShell:

```powershell
$env:POLYGON_API_KEY="your_polygon_key"
```

Streamlit secrets:

```toml
POLYGON_API_KEY = "your_polygon_key"
```

Formula:

```text
TRIN = (advancing issues / declining issues) / (advancing volume / declining volume)
```

If the key is missing, the dashboard marks TRIN unavailable rather than substituting yfinance data. Plain `TRIN` on Yahoo is Trinity Capital Inc., not the Arms Index.

## Model

Stages are classified from current and historical bars only:

- `Acceleration`: VWMA 8 > VWMA 21 > VWMA 34
- `Deceleration`: VWMA 8 < VWMA 21 < VWMA 34
- `Accumulation`: neutral VWMA stack and close >= VMA 21
- `Distribution`: neutral VWMA stack and close < VMA 21

Signals are close-confirmed. A trading backtest should execute no earlier than the next session open.

The dashboard also shows a `Last 1 Month Stage History` table below the main chart. Stages in that table are computed from the full loaded history first, then filtered to the latest one-month display window.

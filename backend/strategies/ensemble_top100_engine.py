from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from utils.model_change_monitor import CORE_MODELS, get_model_status_row, status_shift_count


ROOT_DIR = Path(__file__).resolve().parents[2]
BUNDLED_DATA_DIR = Path(__file__).resolve().parent / "data"
ETF_PRICE_CSV = BUNDLED_DATA_DIR / "top100_etf_prices.csv"

START_DATE = pd.Timestamp("2008-05-30")
INITIAL_CAPITAL = 100000.0
TRADE_COST_BPS = 5.0
RECENT_WINDOW = 36

TOP100_ETFS = [
    "VOO", "IVV", "SPY", "VTI", "QQQ", "VEA", "VUG", "GLD", "IEFA", "VTV",
    "BND", "IEMG", "AGG", "VXUS", "VWO", "IWF", "VGT", "IJH", "SPYM", "VIG", "VO",
    "IJR", "XLK", "RSP", "SCHD", "IAU", "ITOT", "EFA", "BNDX", "VYM", "SGOV",
    "IWM", "VB", "QQQM", "IWD", "VT", "VCIT", "IVW", "SCHX", "VEU", "SCHF",
    "IXUS", "IBIT", "SCHG", "XLF", "QUAL", "IVE", "IWR", "SMH", "IEF", "VV",
    "SLV", "IWB", "TLT", "SPYG", "JEPI", "DIA", "BSV", "BIL", "MUB", "DFAC",
    "XLV", "VTEB", "VCSH", "VGIT", "MBB", "VONG", "SCHB", "SPDW", "DGRO", "JPST",
    "XLE", "VNQ", "IUSB", "GOVT", "GDX", "JEPQ", "VBR", "GLDM", "DYNF", "SPYV",
    "VGK", "XLI", "EFV", "LQD", "CGDV", "EEM", "MGK", "IDEV", "ACWI", "OEF",
    "BIV", "TQQQ", "VGSH", "IUSG", "JAAA", "VXF", "XLC", "USHY", "MDY",
]

DEFENSIVE_ETFS = {
    "GLD", "IAU", "SLV", "GLDM", "GDX",
    "SGOV", "BIL", "VGSH", "JPST", "JAAA",
    "BND", "AGG", "BNDX", "VCIT", "IEF", "TLT", "BSV", "MUB", "VTEB", "VCSH",
    "VGIT", "MBB", "IUSB", "GOVT", "LQD", "BIV", "USHY",
    "JEPI", "JEPQ",
}
SPECIAL_ETFS = {"IBIT", "TQQQ"}
KEY_WARNING_MODELS = {"Market Regime", "Bear Trap", "200MA Strategy", "Combined Macro + RWRA"}

NO_SPECIAL_CONFIG = {
    "score_mode": "defensive_quality",
    "min_history_days": 252,
    "floor": 0.45,
    "power": 0.9,
    "penalty_scale": 0.5,
    "mix_style": "tilted",
}

MODEL_SCORE_MAP = {
    "Market Regime": {
        "LOW RISK REGIME": 85.0,
        "EARLY WARNING": 50.0,
        "HIGH RISK": 15.0,
    },
    "Bear Trap": {
        "LOW RISK": 80.0,
        "EARLY WARNING": 50.0,
        "HIGH RISK": 20.0,
    },
    "Bull Trap": {
        "BULLISH / LOW RISK": 85.0,
        "CAUTION / BULL TRAP RISK": 50.0,
        "BEARISH / HIGH RISK": 15.0,
    },
    "ETF Rotation": {
        "NORMAL": 75.0,
        "EARLY WARNING": 30.0,
    },
}


@dataclass(frozen=True)
class StrategyConfig:
    score_mode: str
    min_history_days: int
    floor: float
    power: float
    penalty_scale: float
    mix_style: str


@dataclass(frozen=True)
class EnsembleTop100Result:
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trade_log: pd.DataFrame
    score_log: pd.DataFrame
    holdings_history: pd.DataFrame
    target_history: pd.DataFrame
    price_frame: pd.DataFrame
    monthly_signal_table: pd.DataFrame


def _score_200ma(status: str) -> float:
    upper = str(status).upper()
    if "BULLISH" in upper:
        return 80.0
    if "CAUTION" in upper:
        return 50.0
    return 20.0


def _score_ml(status: str) -> float:
    upper = str(status).upper()
    if "HIGH CONFIDENCE" in upper:
        return 80.0
    if "NEUTRAL" in upper:
        return 50.0
    return 25.0


def _score_combined(status: str) -> float:
    guardrail = str(status).split("|")[0].strip().upper()
    return {
        "STRONG BULL": 95.0,
        "BULL": 80.0,
        "NEUTRAL": 50.0,
        "RISK OFF": 20.0,
        "CRISIS": 5.0,
    }.get(guardrail, 50.0)


def normalized_model_score(model: str, status: str) -> float:
    if model == "200MA Strategy":
        return _score_200ma(status)
    if model == "ML Meta-Indicator":
        return _score_ml(status)
    if model == "Combined Macro + RWRA":
        return _score_combined(status)
    return MODEL_SCORE_MAP.get(model, {}).get(str(status), 50.0)


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 6 or x.nunique() < 2 or y.nunique() < 2:
        return 0.5
    corr = x.corr(y, method="spearman")
    if pd.isna(corr):
        return 0.5
    return float((corr + 1) / 2)


def _percentile_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(ascending=ascending, pct=True, method="average").fillna(0.0)


def _compute_metrics(equity: pd.Series) -> dict[str, float]:
    ret = equity.pct_change().fillna(0.0)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    vol = ret.std() * np.sqrt(252)
    sharpe = ((ret.mean() - 0.02 / 252) / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0.0
    downside = ret[ret < 0].std() * np.sqrt(252)
    sortino = ((ret.mean() - 0.02 / 252) * 252 / downside) if downside and not np.isnan(downside) else 0.0
    drawdown = (equity / equity.cummax()) - 1
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "final_equity": float(equity.iloc[-1]),
    }


def _slice_metrics(equity: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    strategy = _compute_metrics(equity)
    spy = _compute_metrics(benchmark)
    return {
        "cagr_pct": round(strategy["cagr"] * 100, 2),
        "vol_pct": round(strategy["vol"] * 100, 2),
        "sharpe": round(strategy["sharpe"], 3),
        "sortino": round(strategy["sortino"], 3),
        "max_dd_pct": round(strategy["max_dd"] * 100, 2),
        "calmar": round(strategy["calmar"], 3),
        "final_equity": round(strategy["final_equity"], 2),
        "benchmark_cagr_pct": round(spy["cagr"] * 100, 2),
        "benchmark_vol_pct": round(spy["vol"] * 100, 2),
        "benchmark_sharpe": round(spy["sharpe"], 3),
        "benchmark_sortino": round(spy["sortino"], 3),
        "benchmark_max_dd_pct": round(spy["max_dd"] * 100, 2),
        "benchmark_calmar": round(spy["calmar"], 3),
        "benchmark_final_equity": round(spy["final_equity"], 2),
    }


def _monthly_returns(equity: pd.Series) -> pd.Series:
    return (1 + equity.pct_change().fillna(0.0)).resample("ME").prod() - 1


def _rolling_histogram(series: pd.Series, bins: int = 11) -> tuple[list[float], list[int]]:
    clean = pd.Series(series).dropna()
    if clean.empty:
        return [0.0] * bins, [0] * bins
    counts, edges = np.histogram(clean, bins=bins)
    return [float(v) for v in edges[:-1]], [int(v) for v in counts]


def _start_sensitivity(equity: pd.Series) -> tuple[list[float], list[float], str, str]:
    points: list[tuple[str, float, float]] = []
    for start_dt in equity.resample("MS").first().index:
        sub = equity[equity.index >= start_dt]
        if len(sub) < 126:
            continue
        years = max((sub.index[-1] - sub.index[0]).days / 365.25, 1 / 12)
        cagr = ((sub.iloc[-1] / sub.iloc[0]) ** (1 / years) - 1) * 100
        max_dd = ((sub / sub.cummax()) - 1).min() * 100
        points.append((start_dt.strftime("%Y-%m"), float(max_dd), float(cagr)))
    if not points:
        return [], [], "-", "-"
    best = max(points, key=lambda item: item[2])
    worst = min(points, key=lambda item: item[2])
    return [p[1] for p in points], [p[2] for p in points], f"{best[0]} (CAGR {best[2]:.1f}%)", f"{worst[0]} (CAGR {worst[2]:.1f}%)"


def _display_name(ticker: str) -> str:
    special = {
        "VBR": "Vanguard Small-Cap Value ETF",
        "VXF": "Vanguard Extended Market ETF",
        "EFV": "iShares MSCI EAFE Value ETF",
        "IJH": "iShares Core S&P Mid-Cap ETF",
        "MUB": "iShares National Muni Bond ETF",
        "VGK": "Vanguard FTSE Europe ETF",
        "SLV": "iShares Silver Trust",
        "SMH": "VanEck Semiconductor ETF",
        "VGT": "Vanguard Information Technology ETF",
        "XLV": "Health Care Select Sector SPDR Fund",
        "XLF": "Financial Select Sector SPDR Fund",
        "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "QQQ": "Invesco QQQ Trust",
        "SPY": "SPDR S&P 500 ETF Trust",
        "GLD": "SPDR Gold Shares",
    }
    return special.get(ticker, ticker)


def _load_prices() -> pd.DataFrame:
    if not ETF_PRICE_CSV.exists():
        raise FileNotFoundError(f"Missing bundled top-100 ETF price cache: {ETF_PRICE_CSV}")
    prices = pd.read_csv(ETF_PRICE_CSV, index_col=0, parse_dates=True).sort_index().ffill()
    keep = list(dict.fromkeys([c for c in TOP100_ETFS + ["SPY"] if c in prices.columns]))
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices[keep]
    return prices.loc[prices.index >= START_DATE].copy()


def _build_monthly_signal_table(price_index: pd.DatetimeIndex) -> pd.DataFrame:
    rebal_dates = pd.Series(price_index, index=price_index).groupby(price_index.to_period("M")).first().tolist()
    rows: list[dict[str, object]] = []
    previous_snapshot = None
    previous_shift_count = 0
    for rebalance_date in rebal_dates:
        prev_dates = price_index[price_index < rebalance_date]
        if len(prev_dates) == 0:
            continue
        signal_date = prev_dates[-1]
        snapshot = get_model_status_row(signal_date.date())
        if snapshot is None:
            continue
        shift_count = 0
        changed_models: list[str] = []
        if previous_snapshot is not None:
            shift_count, changed_models = status_shift_count(previous_snapshot, snapshot)
        row = dict(snapshot)
        row["Rebalance Date"] = pd.Timestamp(rebalance_date)
        row["Signal Date"] = pd.Timestamp(signal_date)
        row["Shift Count"] = shift_count
        row["Changed Models"] = ", ".join(changed_models) if changed_models else "None"
        row["Prev Shift Count"] = previous_shift_count
        rows.append(row)
        previous_snapshot = snapshot
        previous_shift_count = shift_count
    return pd.DataFrame(rows)


def _forward_metrics(signal_table: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in signal_table.iterrows():
        signal_date = pd.Timestamp(row["Signal Date"])
        future = prices.loc[prices.index > signal_date]
        if len(future) < 63:
            continue
        px0 = float(prices.loc[signal_date, "SPY"])
        px1m = float(future.iloc[min(20, len(future) - 1)]["SPY"])
        px3m = float(future.iloc[min(62, len(future) - 1)]["SPY"])
        window = future["SPY"].iloc[:63]
        rec = dict(row)
        rec["fwd_1m_ret"] = px1m / px0 - 1
        rec["fwd_3m_ret"] = px3m / px0 - 1
        rec["fwd_3m_maxdd"] = float(((window / window.cummax()) - 1).min())
        rows.append(rec)
    df = pd.DataFrame(rows)
    for model in CORE_MODELS:
        df[f"{model}__score"] = df[model].apply(lambda v, m=model: normalized_model_score(m, v))
    return df


def _model_confidence(history: pd.DataFrame, model: str) -> tuple[float, float, float]:
    score = history[f"{model}__score"] / 100.0
    y1 = (history["fwd_1m_ret"] > 0).astype(float)
    y3 = (history["fwd_3m_ret"] > 0).astype(float)
    safe = (history["fwd_3m_maxdd"] > -0.10).astype(float)
    acc1 = 1.0 - float(((score - y1) ** 2).mean())
    acc3 = 1.0 - float(((score - y3) ** 2).mean())
    dd_acc = 1.0 - float(((score - safe) ** 2).mean())
    ic3 = _safe_spearman(score, history["fwd_3m_ret"])
    combo = 0.35 * acc1 + 0.35 * acc3 + 0.20 * dd_acc + 0.10 * ic3
    return acc1, acc3, combo


def _confidence_weights(feature_df: pd.DataFrame, as_of_signal_date: pd.Timestamp) -> dict[str, float]:
    prior = feature_df[feature_df["Signal Date"] < as_of_signal_date].copy()
    if len(prior) < 24:
        return {model: 1 / len(CORE_MODELS) for model in CORE_MODELS}
    recent = prior.tail(RECENT_WINDOW)
    weights = {}
    for model in CORE_MODELS:
        _, _, long_combo = _model_confidence(prior, model)
        _, _, recent_combo = _model_confidence(recent if len(recent) >= 12 else prior, model)
        raw = 0.6 * long_combo + 0.4 * recent_combo
        weights[model] = min(0.24, max(0.08, raw))
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def _warning_penalty(row: pd.Series, scale: float) -> float:
    count = int(row["Shift Count"])
    base = 0.0
    if count >= 5:
        base = 12.0
    elif count == 4:
        base = 8.0
    elif count == 3:
        base = 5.0
    changed = {x.strip() for x in str(row["Changed Models"]).split(",") if x.strip() and x.strip() != "None"}
    if changed & KEY_WARNING_MODELS:
        base += 3.0
    return base * scale


def _risk_fraction(score: float, floor: float, power: float) -> float:
    x = max(0.0, min(1.0, score / 100.0))
    return min(1.0, max(floor, floor + (1.0 - floor) * (x**power)))


def _sleeve_weights(score: float, cfg: StrategyConfig, shift_count: int) -> tuple[float, float]:
    risk = _risk_fraction(score, cfg.floor, cfg.power)
    x = max(0.0, min(1.0, score / 100.0))
    qqq_share = min(0.50, max(0.18, 0.18 + 0.32 * max(0.0, (score - 45.0) / 55.0)))
    gold_def_share = min(0.85, max(0.25, 0.70 - 0.45 * x))
    if shift_count >= 5:
        qqq_share *= 0.55
        gold_def_share = min(0.95, gold_def_share + 0.12)
    elif shift_count >= 3:
        qqq_share *= 0.75
        gold_def_share = min(0.92, gold_def_share + 0.07)
    offensive = risk
    defensive = 1.0 - risk
    return offensive, defensive * gold_def_share + defensive * (1.0 - gold_def_share)


def _eligible_tickers(prices: pd.DataFrame, as_of: pd.Timestamp, min_history_days: int) -> list[str]:
    eligible = []
    for ticker in TOP100_ETFS:
        if ticker in SPECIAL_ETFS:
            continue
        if ticker not in prices.columns:
            continue
        if len(prices.loc[:as_of, ticker].dropna()) >= min_history_days:
            eligible.append(ticker)
    return eligible


def _split_pools(eligible: list[str]) -> tuple[list[str], list[str]]:
    offensive = []
    defensive = []
    for ticker in eligible:
        if ticker in DEFENSIVE_ETFS:
            defensive.append(ticker)
        else:
            offensive.append(ticker)
    return offensive, defensive


def _compute_scores(window_prices: pd.DataFrame, tickers: list[str], mode: str) -> pd.Series:
    if not tickers:
        return pd.Series(dtype=float)
    sub = window_prices[tickers]
    ret_63 = sub.iloc[-1] / sub.iloc[-63] - 1 if len(sub) >= 63 else pd.Series(index=tickers, dtype=float)
    ret_126 = sub.iloc[-1] / sub.iloc[-126] - 1 if len(sub) >= 126 else pd.Series(index=tickers, dtype=float)
    ret_252 = sub.iloc[-1] / sub.iloc[-252] - 1 if len(sub) >= 252 else pd.Series(index=tickers, dtype=float)
    vol_63 = sub.pct_change().tail(63).std()
    dd_126 = (sub.tail(126) / sub.tail(126).cummax() - 1).min()
    r3 = _percentile_rank(ret_63, ascending=True)
    r6 = _percentile_rank(ret_126, ascending=True)
    r12 = _percentile_rank(ret_252, ascending=True)
    inv_vol = _percentile_rank(vol_63, ascending=False)
    shallow_dd = _percentile_rank(dd_126, ascending=False)
    if mode == "defensive_quality":
        score = 0.10 * r3 + 0.25 * r6 + 0.25 * r12 + 0.15 * inv_vol + 0.25 * shallow_dd
    elif mode == "risk_adjusted":
        score = 0.15 * r3 + 0.25 * r6 + 0.25 * r12 + 0.20 * inv_vol + 0.15 * shallow_dd
    else:
        score = 0.20 * r3 + 0.35 * r6 + 0.35 * r12 + 0.10 * inv_vol
    return score.sort_values(ascending=False)


def _offensive_mix(score: float, style: str) -> tuple[int, int]:
    if style == "tilted":
        if score >= 80:
            return 5, 0
        if score >= 65:
            return 4, 1
        if score >= 50:
            return 3, 2
        if score >= 35:
            return 2, 3
        return 1, 4
    if style == "balanced":
        if score >= 80:
            return 4, 1
        if score >= 65:
            return 3, 2
        if score >= 50:
            return 3, 2
        if score >= 35:
            return 2, 3
        return 1, 4
    if score >= 80:
        return 4, 1
    if score >= 65:
        return 3, 2
    if score >= 50:
        return 2, 3
    if score >= 35:
        return 1, 4
    return 0, 5


@lru_cache(maxsize=1)
def run_ensemble_top100_backtest() -> EnsembleTop100Result:
    cfg = StrategyConfig(**NO_SPECIAL_CONFIG)
    prices = _load_prices()
    signals = _build_monthly_signal_table(prices.index)
    feature_df = _forward_metrics(signals, prices[["SPY"]]).copy()

    holdings: dict[str, float] = {}
    cash = INITIAL_CAPITAL
    equity_rows = []
    benchmark_rows = []
    trade_rows = []
    score_rows = []
    holding_weight_rows = []
    target_weight_rows = []

    spy_shares = (INITIAL_CAPITAL * (1 - TRADE_COST_BPS / 10000.0)) / float(prices.loc[START_DATE:, "SPY"].iloc[0])
    signal_lookup = {pd.Timestamp(r["Rebalance Date"]): r for _, r in feature_df.iterrows()}
    rebalance_dates = set(signal_lookup)

    for current_date in prices.index:
        day_prices = prices.loc[current_date]
        portfolio_value = cash + sum(shares * day_prices.get(ticker, np.nan) for ticker, shares in holdings.items() if pd.notna(day_prices.get(ticker, np.nan)))

        if current_date in rebalance_dates:
            row = pd.Series(signal_lookup[current_date])
            weights = _confidence_weights(feature_df, pd.Timestamp(row["Signal Date"]))
            raw_score = sum(normalized_model_score(model, row[model]) * weights[model] for model in CORE_MODELS)
            penalty = _warning_penalty(row, cfg.penalty_scale)
            final_score = min(100.0, max(0.0, raw_score - penalty))

            eligible = _eligible_tickers(prices, current_date, cfg.min_history_days)
            offensive_pool, defensive_pool = _split_pools(eligible)
            lookback = prices.loc[:current_date].tail(max(cfg.min_history_days, 252) + 5)
            offensive_scores = _compute_scores(lookback, offensive_pool, cfg.score_mode)
            defensive_scores = _compute_scores(lookback, defensive_pool, cfg.score_mode)
            off_n, def_n = _offensive_mix(final_score, cfg.mix_style)
            selected_off = list(offensive_scores.index[:off_n])
            selected_def = list(defensive_scores.index[:def_n])
            selected = selected_off + selected_def
            if len(selected) < 5:
                combined = pd.concat([offensive_scores, defensive_scores]).sort_values(ascending=False)
                for ticker in combined.index:
                    if ticker not in selected:
                        selected.append(ticker)
                    if len(selected) == 5:
                        break
            selected = selected[:5]

            offensive_weight, defensive_weight = _sleeve_weights(final_score, cfg, int(row["Shift Count"]))
            target_weights: dict[str, float] = {}
            if selected_off:
                for ticker in selected_off:
                    target_weights[ticker] = offensive_weight / len(selected_off)
            if selected_def:
                for ticker in selected_def:
                    target_weights[ticker] = defensive_weight / len(selected_def)
            if len(target_weights) < len(selected):
                leftovers = [ticker for ticker in selected if ticker not in target_weights]
                residual = 1.0 - sum(target_weights.values())
                if leftovers and residual > 0:
                    for ticker in leftovers:
                        target_weights[ticker] = residual / len(leftovers)
            total_weight = sum(target_weights.values())
            if total_weight > 0:
                target_weights = {ticker: weight / total_weight for ticker, weight in target_weights.items()}

            current_weights = {}
            if portfolio_value > 0:
                for ticker, shares in holdings.items():
                    px = day_prices.get(ticker, np.nan)
                    if pd.notna(px):
                        current_weights[ticker] = float((shares * px) / portfolio_value)
            turnover = sum(abs(target_weights.get(ticker, 0.0) - current_weights.get(ticker, 0.0)) for ticker in set(current_weights) | set(target_weights))
            trade_cost = portfolio_value * turnover * TRADE_COST_BPS / 10000.0
            portfolio_value = max(portfolio_value - trade_cost, 0.0)

            new_holdings = {}
            deployed_value = 0.0
            for ticker, target_weight in target_weights.items():
                px = day_prices.get(ticker, np.nan)
                if pd.isna(px) or px <= 0:
                    continue
                dollar = portfolio_value * target_weight
                new_holdings[ticker] = dollar / px
                deployed_value += dollar
            cash = max(portfolio_value - deployed_value, 0.0)
            holdings = new_holdings

            diff = {ticker: target_weights.get(ticker, 0.0) - current_weights.get(ticker, 0.0) for ticker in set(current_weights) | set(target_weights)}
            biggest_add = max(diff.items(), key=lambda item: item[1]) if diff else ("-", 0.0)
            biggest_cut = min(diff.items(), key=lambda item: item[1]) if diff else ("-", 0.0)
            trade_rows.append(
                {
                    "rebalance_date": current_date.strftime("%Y-%m-%d"),
                    "signal_date": pd.Timestamp(row["Signal Date"]).strftime("%Y-%m-%d"),
                    "final_score": round(final_score, 2),
                    "raw_score": round(raw_score, 2),
                    "warning_penalty": round(penalty, 2),
                    "selected": ", ".join(selected),
                    "weights": ", ".join(f"{ticker}:{weight:.1%}" for ticker, weight in sorted(target_weights.items())),
                    "buys": ", ".join([ticker for ticker in target_weights if target_weights.get(ticker, 0.0) > current_weights.get(ticker, 0.0)]) or "-",
                    "sells": ", ".join([ticker for ticker in current_weights if current_weights.get(ticker, 0.0) > target_weights.get(ticker, 0.0)]) or "-",
                    "turnover_pct": round(turnover * 100, 2),
                    "trade_cost": round(trade_cost, 2),
                    "shift_count": int(row["Shift Count"]),
                    "Changed Models": row["Changed Models"],
                    "Market Regime": row["Market Regime"],
                    "Bear Trap": row["Bear Trap"],
                    "Bull Trap": row["Bull Trap"],
                    "ETF Rotation": row["ETF Rotation"],
                    "200MA Strategy": row["200MA Strategy"],
                    "ML Meta-Indicator": row["ML Meta-Indicator"],
                    "Combined Macro + RWRA": row["Combined Macro + RWRA"],
                    "bucket": "risk_on" if final_score >= 80 else "tilted" if final_score >= 65 else "balanced" if final_score >= 50 else "cautious" if final_score >= 35 else "defensive",
                    "largest_add": f"{biggest_add[0]} {biggest_add[1] * 100:+.1f}%",
                    "largest_cut": f"{biggest_cut[0]} {biggest_cut[1] * 100:+.1f}%",
                }
            )
            score_rows.append(
                {
                    "rebalance_date": current_date,
                    "signal_date": pd.Timestamp(row["Signal Date"]),
                    "raw_score": round(raw_score, 2),
                    "warning_penalty": round(penalty, 2),
                    "final_score": round(final_score, 2),
                    "shift_count": int(row["Shift Count"]),
                    "changed_models": row["Changed Models"],
                    "bucket": "risk_on" if final_score >= 80 else "tilted" if final_score >= 65 else "balanced" if final_score >= 50 else "cautious" if final_score >= 35 else "defensive",
                    **{f"score_{model}": round(normalized_model_score(model, row[model]), 2) for model in CORE_MODELS},
                    **{f"weight_{model}": round(weights[model], 4) for model in CORE_MODELS},
                }
            )

        portfolio_value = cash + sum(shares * day_prices.get(ticker, np.nan) for ticker, shares in holdings.items() if pd.notna(day_prices.get(ticker, np.nan)))
        benchmark_value = spy_shares * float(day_prices["SPY"])
        current_weights = {}
        if portfolio_value > 0:
            for ticker, shares in holdings.items():
                px = day_prices.get(ticker, np.nan)
                if pd.notna(px):
                    current_weights[ticker] = float((shares * px) / portfolio_value)
        latest_target = {}
        if trade_rows:
            latest_selected = trade_rows[-1]["selected"].split(", ") if trade_rows[-1]["selected"] else []
            latest_weights = trade_rows[-1]["weights"].split(", ") if trade_rows[-1]["weights"] else []
            for part in latest_weights:
                if ":" in part:
                    ticker, pct = part.split(":")
                    latest_target[ticker] = float(pct.rstrip("%")) / 100.0

        equity_rows.append((current_date, portfolio_value))
        benchmark_rows.append((current_date, benchmark_value))
        holding_weight_rows.append((current_date, current_weights))
        target_weight_rows.append((current_date, latest_target))

    equity = pd.Series({dt_idx: value for dt_idx, value in equity_rows}, name="EnsembleTop100")
    benchmark = pd.Series({dt_idx: value for dt_idx, value in benchmark_rows}, name="SPY")
    holdings_history = pd.DataFrame({dt_idx: weights for dt_idx, weights in holding_weight_rows}).T.fillna(0.0)
    target_history = pd.DataFrame({dt_idx: weights for dt_idx, weights in target_weight_rows}).T.fillna(0.0)
    trade_log = pd.DataFrame(trade_rows)
    score_log = pd.DataFrame(score_rows).set_index("rebalance_date") if score_rows else pd.DataFrame()
    return EnsembleTop100Result(
        equity_curve=equity,
        benchmark_curve=benchmark,
        trade_log=trade_log,
        score_log=score_log,
        holdings_history=holdings_history,
        target_history=target_history,
        price_frame=prices,
        monthly_signal_table=feature_df,
    )


def _build_assets(selected_date: pd.Timestamp, result: EnsembleTop100Result) -> list[dict[str, float | str]]:
    latest_hold = result.holdings_history.loc[selected_date].fillna(0.0) if selected_date in result.holdings_history.index else pd.Series(dtype=float)
    latest_target = result.target_history.loc[selected_date].fillna(0.0) if selected_date in result.target_history.index else pd.Series(dtype=float)
    tickers = sorted({ticker for ticker in latest_hold.index if latest_hold.get(ticker, 0.0) > 0.001} | {ticker for ticker in latest_target.index if latest_target.get(ticker, 0.0) > 0.001})
    if not tickers and not result.trade_log.empty:
        tickers = result.trade_log.iloc[-1]["selected"].split(", ")
    price_slice = result.price_frame.loc[:selected_date]
    latest_prices = price_slice.iloc[-1] if not price_slice.empty else pd.Series(dtype=float)
    prev_prices = price_slice.iloc[-2] if len(price_slice) > 1 else latest_prices
    assets = []
    for ticker in tickers:
        px = float(latest_prices.get(ticker, np.nan))
        prev_px = float(prev_prices.get(ticker, px))
        change = ((px / prev_px) - 1) * 100 if prev_px and not np.isnan(px) else 0.0
        current = float(latest_hold.get(ticker, 0.0) * 100)
        target = float(latest_target.get(ticker, 0.0) * 100)
        assets.append(
            {
                "symbol": ticker,
                "name": _display_name(ticker),
                "price": round(px, 2) if not np.isnan(px) else 0.0,
                "change": round(change, 2),
                "target": round(target, 1),
                "current": round(current, 1),
                "lower": max(0.0, round(target - 7.5, 1)),
                "upper": min(100.0, round(target + 7.5, 1)),
            }
        )
    return sorted(assets, key=lambda item: (-item["target"], item["symbol"]))


def _risk_stack(snapshot: pd.Series) -> list[dict[str, str]]:
    return [
        {"label": "Market Regime", "value": str(snapshot["Market Regime"])},
        {"label": "Bear Trap", "value": str(snapshot["Bear Trap"])},
        {"label": "Bull Trap", "value": str(snapshot["Bull Trap"])},
        {"label": "ETF Rotation", "value": str(snapshot["ETF Rotation"])},
        {"label": "200MA Strategy", "value": str(snapshot["200MA Strategy"])},
        {"label": "ML Meta-Indicator", "value": str(snapshot["ML Meta-Indicator"])},
        {"label": "Combined Macro + RWRA", "value": str(snapshot["Combined Macro + RWRA"])},
    ]


def load_ensemble_top100_json(sel_dt: pd.Timestamp) -> dict[str, object] | None:
    result = run_ensemble_top100_backtest()
    eq = result.equity_curve[result.equity_curve.index <= sel_dt]
    bench = result.benchmark_curve[result.benchmark_curve.index <= sel_dt]
    if eq.empty or bench.empty:
        return None

    selected_date = eq.index[-1]
    metrics = _slice_metrics(eq, bench)
    assets = _build_assets(selected_date, result)

    trade_log = result.trade_log[pd.to_datetime(result.trade_log["rebalance_date"]) <= selected_date].copy()
    current_trade = trade_log.iloc[-1] if not trade_log.empty else None
    current_score = result.score_log.loc[:selected_date].iloc[-1] if not result.score_log.empty else pd.Series(dtype=object)
    signal_snapshot = result.monthly_signal_table.loc[result.monthly_signal_table["Rebalance Date"] <= selected_date].iloc[-1]

    monthly = _monthly_returns(eq)
    monthly_spy = _monthly_returns(bench)
    monthly_win = ((monthly > monthly_spy).mean() * 100) if not monthly.empty else 0.0
    rolling_1y = eq.pct_change(252)
    mc_values, mc_counts = _rolling_histogram(rolling_1y * 100)
    sc_dd, sc_cagr, best_era, worst_era = _start_sensitivity(eq)

    action_items = []
    for asset in assets:
        drift = asset["current"] - asset["target"]
        if abs(drift) < 1.0:
            continue
        dollar_drift = INITIAL_CAPITAL * abs(drift) / 100
        tone = "sell" if drift > 0 else "buy"
        action_items.append(
            {
                "tone": tone,
                "label": f"{'Trim' if tone == 'sell' else 'Add'} {asset['symbol']}",
                "detail": f"As of {selected_date.strftime('%Y-%m-%d')}: current {asset['current']:.1f}% vs target {asset['target']:.1f}% ({dollar_drift:,.0f} USD {'overweight' if tone == 'sell' else 'underweight'}).",
            }
        )
    if not action_items:
        action_items.append(
            {
                "tone": "hold",
                "label": "Hold targets",
                "detail": f"As of {selected_date.strftime('%Y-%m-%d')}, the current ETF basket is already close to the target weights for this PIT snapshot.",
            }
        )

    history_rows = []
    for _, row in trade_log.sort_values("rebalance_date", ascending=False).head(24).iterrows():
        history_rows.append(
            {
                "date": row["rebalance_date"],
                "regime": str(row["bucket"]).upper(),
                "add": row["largest_add"],
                "cut": row["largest_cut"],
                "turnover": f"{float(row['turnover_pct']):.1f}%",
                "mix": row["weights"].replace(":", " ").replace(", ", " / "),
                "event": "Monthly Rebalance",
                "weights": row["weights"].replace(":", " ").replace(", ", " / "),
                "source": str(row["largest_cut"]).split()[0],
                "destination": str(row["largest_add"]).split()[0],
                "change": f"{float(row['turnover_pct']):.1f}%",
                "rationale": f"Score {float(row['final_score']):.1f} | {row['Changed Models']}",
            }
        )

    yearly = []
    yearly_strategy = (1 + eq.pct_change().fillna(0.0)).resample("YE").prod() - 1
    yearly_spy = (1 + bench.pct_change().fillna(0.0)).resample("YE").prod() - 1
    dd_strategy = ((eq / eq.cummax()) - 1).groupby(eq.index.year).min()
    dd_spy = ((bench / bench.cummax()) - 1).groupby(bench.index.year).min()
    for dt_idx, value in yearly_strategy.dropna().items():
        year = dt_idx.year
        spy_ret = float(yearly_spy.get(dt_idx, 0.0))
        yearly.append(
            {
                "year": str(year),
                "portRet": f"{value * 100:+.1f}%",
                "spyRet": f"{spy_ret * 100:+.1f}%",
                "portDD": f"{dd_strategy.get(year, 0.0) * 100:.1f}%",
                "spyDD": f"{dd_spy.get(year, 0.0) * 100:.1f}%",
                "winner": "Portfolio" if value > spy_ret else "SPY",
            }
        )

    current_bucket = str(current_score.get("bucket", "tilted")).upper().replace("_", " ")
    final_score = float(current_score.get("final_score", 50.0))
    raw_score = float(current_score.get("raw_score", final_score))
    warning_penalty = float(current_score.get("warning_penalty", 0.0))
    shift_count = int(current_score.get("shift_count", 0))
    changed_models = str(current_score.get("changed_models", "None"))
    risk_note = (
        f"As of {selected_date.strftime('%Y-%m-%d')}, the frozen no-special rule keeps the strategy in {current_bucket.lower()} mode. "
        f"The 7-model ensemble score is {final_score:.1f} after a {warning_penalty:.1f}-point alert penalty, so the selector favors offensive ETFs but still carries a defensive sleeve."
    )

    state_weights = [
        {"state": "Risk-On", "mix": "5 offensive ETFs when score >= 80"},
        {"state": "Tilted", "mix": "4 offensive / 1 defensive when score 65-79"},
        {"state": "Balanced", "mix": "3 offensive / 2 defensive when score 50-64"},
        {"state": "Cautious", "mix": "2 offensive / 3 defensive when score 35-49"},
        {"state": "Defensive", "mix": "1 offensive / 4 defensive when score < 35"},
    ]

    return {
        "kpi": {
            "cagr": f"{metrics['cagr_pct']:.2f}%",
            "cagrBench": f"vs SPY: {metrics['benchmark_cagr_pct']:.2f}%",
            "maxdd": f"{metrics['max_dd_pct']:.2f}%",
            "maxddBench": f"vs SPY: {metrics['benchmark_max_dd_pct']:.2f}%",
            "sharpe": f"{metrics['sharpe']:.3f}",
            "sharpeBench": f"vs SPY: {metrics['benchmark_sharpe']:.3f}",
            "ratios": f"{metrics['calmar']:.3f} / {metrics['sortino']:.3f}",
            "ratiosBench": f"vs SPY: {metrics['benchmark_calmar']:.3f} / {metrics['benchmark_sortino']:.3f}",
            "signal": "REBALANCE" if action_items[0]["tone"] != "hold" else "HOLD",
            "signalBg": "bg-orange-500/15 text-orange-300 border border-orange-500/40" if action_items[0]["tone"] != "hold" else "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30",
        },
        "regime": f"{current_bucket} | Score {final_score:.1f} | {'ALERT ACTIVE' if shift_count >= 3 else 'NO ALERT'}",
        "assets": assets,
        "yearly": yearly[::-1],
        "history": history_rows,
        "chartDates": eq.index.strftime("%Y-%m-%d").tolist()[::5],
        "chartPort": eq.tolist()[::5],
        "chartSpy": bench.tolist()[::5],
        "chartDDPort": [float(v * 100) for v in ((eq / eq.cummax()) - 1).tolist()[::5]],
        "chartDDSpy": [float(v * 100) for v in ((bench / bench.cummax()) - 1).tolist()[::5]],
        "mcValues": mc_values,
        "mcCounts": mc_counts,
        "scDD": sc_dd,
        "scCAGR": sc_cagr,
        "meta": {
            "backtestPeriod": f"{eq.index[0].strftime('%b %Y')} - {eq.index[-1].strftime('%b %Y')}",
            "executionDescription": f"As-of {selected_date.strftime('%Y-%m-%d')} PIT instruction set for the frozen no-special ensemble rule. Rebalance monthly into max 5 ETFs using the 7-model score as a risk governor, with 5 bps turnover cost.",
            "lastActionDate": history_rows[0]["date"] if history_rows else selected_date.strftime("%Y-%m-%d"),
            "monthlyWinRate": f"{monthly_win:.1f}%",
            "bestEra": best_era,
            "worstEra": worst_era,
            "sleeves": [
                {"symbol": asset["symbol"], "name": asset["name"], "weight": asset["target"]} for asset in assets[:5]
            ],
            "actionItems": action_items,
            "stateWeights": state_weights,
            "currentState": current_bucket.title(),
            "warningMode": "ON" if shift_count >= 3 else "OFF",
            "guardrail": str(signal_snapshot["Combined Macro + RWRA"]).split("|")[0].strip(),
            "riskNote": risk_note,
            "combinedSignal": str(signal_snapshot["Combined Macro + RWRA"]),
            "tradeCount": int(len(trade_log)),
            "turnoverX": round(trade_log["turnover_pct"].astype(float).sum() / 100.0, 2) if not trade_log.empty else 0.0,
            "benchmarkFinalEquity": metrics["benchmark_final_equity"],
            "allocatorFinalEquity": metrics["final_equity"],
            "allocatorVsSpy": round(metrics["final_equity"] - metrics["benchmark_final_equity"], 2),
            "ensembleScore": round(final_score, 2),
            "rawScore": round(raw_score, 2),
            "warningPenalty": round(warning_penalty, 2),
            "shiftCount": shift_count,
            "changedModels": changed_models,
            "riskStack": _risk_stack(signal_snapshot),
            "logicSummary": {
                "title": "Frozen No-Special Ensemble Top-100 Rule",
                "subtitle": "This strategy is optimized, but constrained. It uses one fixed no-special configuration rather than a live optimizer, so the dashboard shows a documented rule, not a moving target.",
                "formula": [
                    "Each of the 7 core models maps to a 0-100 directional score.",
                    "Each model gets a confidence weight from prior historical usefulness only.",
                    "Raw ensemble score = sum(model score × model confidence weight).",
                    "Alert penalty = 3-of-7 warning modifier scaled by which models changed.",
                    "Final score = clamp(raw score - alert penalty, 0, 100).",
                    "The ETF selector ranks eligible top-100 ETFs by defensive-quality momentum: 10% 3M momentum + 25% 6M momentum + 25% 12M momentum + 15% low 3M vol + 25% shallow 6M drawdown.",
                    "The final score controls offensive/defensive sleeve counts; holdings are capped at 5 and rebalanced monthly.",
                ],
                "guardrails": [
                    "No IBIT and no TQQQ in the deployed dashboard version.",
                    "Minimum ETF history requirement: 252 trading days.",
                    "Monthly rebalance only, with 5 bps turnover cost.",
                    "Frozen no-special spec selected because it beat SPY in the robust out-of-sample checks while keeping drawdown materially lower.",
                ],
                "caveat": "This is not overfit-proof. There is still survivorship bias in the ETF universe and prior rule selection. The dashboard uses the frozen robust candidate because it is more trustworthy than the flashier full-sample winner.",
            },
        },
    }

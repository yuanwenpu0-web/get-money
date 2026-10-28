"""Fixed 15m strong-pullback trailing-stop grid, validation then holdout.

Read-only. Reproduce: .venv/Scripts/python.exe backend/scripts/tune_pullback_15m.py
The universe, dates and candidates are fixed before inspecting the holdout.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.backtest.execution import simulate
from app.backtest.metrics import metrics_from_equity
from app.backtest.significance import bootstrap_expectancy_ci
from app.db import make_engine
from app.features.pullback import add_pullback_context
from app.ml.dataset import build_dataset
from app.strategy.rules import evaluate_signals, resolve_params

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT", "BCHUSDT", "DOTUSDT"]
END = pd.Timestamp("2026-10-27T06:00:00Z")
VAL_START = pd.Timestamp("2026-04-01T00:00:00Z")
TEST_START = pd.Timestamp("2026-09-01T00:00:00Z")
FEE, SLIPPAGE, INITIAL = 0.0004, 0.0002, 10000.0
# Frozen values observed from the database before this experiment, to make reruns
# independent of subsequent user setting changes. See resolved_parameters in output.
FROZEN_STATE = {
    "min_score": 0.55, "heavy_score": 0.75, "base_position_pct": 0.08,
    "max_position_pct": 0.35, "max_leverage": 1, "sl_atr_mult": 1.5,
    "tp_atr_mult": 2.5, "trailing_atr_mult": 1.2, "min_sl_pct": 0.0025,
    "min_tp_pct": 0.005, "min_hold_bars": 0, "max_hold_bars": 192,
    "fee_rate": FEE, "slippage": SLIPPAGE, "regime_filter": True,
    "cost_filter": True, "require_model": False, "funding_filter": False,
    "risk_per_trade_pct": 0.005, "cooldown_bars": 10,
    "min_atr_cost_ratio": 0.5, "min_net_reward_risk": 1.0, "max_atr_pct": 0.03,
    "min_model_edge": 0.05, "trailing_enabled": True, "trail_activation_atr": 1.5,
    "max_signal_age_bars": 2, "ema_exit": True, "partial_tp_r": 0.0,
    "partial_tp_frac": 0.5,
}
FROZEN_OVERRIDES = {
    **{k: v for k, v in FROZEN_STATE.items() if k not in ("fee_rate", "slippage")},
    "htf_adx_filter": 22.0, "pullback_min_bars": 8, "pullback_confirm_bars": 8,
    "target_r": 10.0, "stop_buffer_pct": 0.0015, "max_hold_hours": 48.0,
}
CANDIDATES = {
    "current": {"ema_exit": True, "trailing_enabled": True, "trail_activation_atr": 1.5, "trailing_atr_mult": 1.2},
    "trail_1_08": {"ema_exit": False, "trailing_enabled": True, "trail_activation_atr": 1.0, "trailing_atr_mult": 0.8},
    "trail_1_12": {"ema_exit": False, "trailing_enabled": True, "trail_activation_atr": 1.0, "trailing_atr_mult": 1.2},
    "trail_15_12": {"ema_exit": False, "trailing_enabled": True, "trail_activation_atr": 1.5, "trailing_atr_mult": 1.2},
    "trail_15_20": {"ema_exit": False, "trailing_enabled": True, "trail_activation_atr": 1.5, "trailing_atr_mult": 2.0},
    "trail_2_20": {"ema_exit": False, "trailing_enabled": True, "trail_activation_atr": 2.0, "trailing_atr_mult": 2.0},
    "trail_1_12_adx18": {"ema_exit": False, "trailing_enabled": True, "trail_activation_atr": 1.0, "trailing_atr_mult": 1.2, "htf_adx_filter": 18.0},
    "trail_1_12_adx27": {"ema_exit": False, "trailing_enabled": True, "trail_activation_atr": 1.0, "trailing_atr_mult": 1.2, "htf_adx_filter": 27.0},
}


def evaluate(data, params, start, end):
    # Calculate state over historical warmup before the slice; no future values are used.
    enriched = add_pullback_context(data, params)
    signals = evaluate_signals(enriched, "strong_pullback", params)
    mask = (data.index >= start) & (data.index < end)
    frame, signals = enriched.loc[mask], signals.loc[mask]
    if frame.empty:
        raise ValueError("empty period")
    equity, trades = simulate(frame, signals, params, INITIAL, FEE, SLIPPAGE, "15m")
    metrics = metrics_from_equity(equity, trades, INITIAL, "15m")
    return {"metrics": metrics, "trades": trades, "exits": dict(Counter(t["status"] for t in trades))}


def summary(records):
    runs = [x for x in records if "error" not in x]
    if not runs:
        return {"runs": 0}
    m = [x["metrics"] for x in runs]
    trades = [t for x in runs for t in x["trades"]]
    ret = statistics.mean(x["total_return_pct"] for x in m)
    dd = statistics.mean(x["max_drawdown_pct"] for x in m)
    return {"runs": len(runs), "trades": len(trades), "mean_return_pct": round(ret, 4), "mean_drawdown_pct": round(dd, 4), "objective": round(ret + 0.5 * dd, 4), "positive_runs": sum(x["total_return_pct"] > 0 for x in m), "trade_ci": bootstrap_expectancy_ci(trades), "exits": dict(sum((Counter(x["exits"]) for x in runs), Counter()))}


def main():
    engine = make_engine(read_only=True)
    state, overrides = FROZEN_STATE, FROZEN_OVERRIDES
    resolved = {name: resolve_params("strong_pullback", state, {**overrides, **changes}, interval="15m") for name, changes in CANDIDATES.items()}
    frames, quality, errors = {}, {}, {}
    for symbol in SYMBOLS:
        try:
            with Session(engine) as db:
                frame = build_dataset(db, symbol, "15m", days=250, end_ms=int(END.timestamp() * 1000))
            quality[symbol] = frame.attrs.get("data_quality", {})
            if frame.index[0] > VAL_START or frame.index[-1] < END - pd.Timedelta(hours=2):
                raise ValueError(f"insufficient coverage: {frame.index[0]} to {frame.index[-1]}")
            frames[symbol] = frame
        except Exception as exc:
            errors[symbol] = str(exc)
        print(f"load {symbol}: {'ok' if symbol in frames else errors[symbol]}", flush=True)
    if not frames:
        raise RuntimeError("no eligible symbols")
    validation = {}
    for name, changes in CANDIDATES.items():
        rows = []
        for symbol, data in frames.items():
            try:
                rows.append({"symbol": symbol, **evaluate(data, resolved[name], VAL_START, TEST_START)})
            except Exception as exc:
                rows.append({"symbol": symbol, "error": str(exc)})
        validation[name] = rows
        print(f"validation {name}: {summary(rows)}", flush=True)
    scores = {name: summary(rows) for name, rows in validation.items()}
    eligible = [(v["objective"], k) for k, v in scores.items() if v.get("runs") == len(frames) and v.get("trades", 0) >= 30 and k != "current"]
    if not eligible:
        raise RuntimeError("No trailing candidate met the validation minimum of 30 trades")
    chosen = max(eligible)[1]
    holdout = {}
    for name in ("current", chosen):
        rows = []
        for symbol, data in frames.items():
            try:
                rows.append({"symbol": symbol, **evaluate(data, resolved[name], TEST_START, END)})
            except Exception as exc:
                rows.append({"symbol": symbol, "error": str(exc)})
        holdout[name] = rows
        print(f"holdout {name}: {summary(rows)}", flush=True)
    payload = {"selection": {"selected": chosen, "parameters": CANDIDATES[chosen], "objective": "mean symbol return + 0.5 * mean symbol max drawdown", "minimum_trades": 30}, "periods": {"validation_start": str(VAL_START), "test_start": str(TEST_START), "end_exclusive": str(END)}, "fee": FEE, "slippage": SLIPPAGE, "initial_per_symbol": INITIAL, "symbols": SYMBOLS, "strategy_overrides": overrides, "resolved_parameters": resolved, "load_errors": errors, "data_quality": quality, "validation_summary": scores, "holdout_summary": {k: summary(v) for k, v in holdout.items()}, "validation": validation, "holdout": holdout}
    output = ROOT / "docs" / "strong-pullback-15m-trailing-tuning.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()

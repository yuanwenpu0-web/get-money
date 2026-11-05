"""Train a gate from Supertrend signals' REALIZED strategy outcomes.

Research-only script. Labels are isolated net returns of each signal under the
strategy's actual exits: ATR stop, 1R/2R/3R scaled take-profits, opposite-signal
exit, data gaps, fees and slippage. Features use only the 10 completed 15m bars
before the signal bar plus the strategy's already-issued side.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Keep BLAS single-threaded inside spawned workers.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.backtest.execution import simulate  # noqa: E402
from app.backtest.isolated import isolated_trade  # noqa: E402
from app.backtest.significance import bootstrap_expectancy_ci  # noqa: E402
from app.backtest.trades import aggregate_positions  # noqa: E402
from app.config import DEFAULT_STATE  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.strategy.rules import evaluate_signals, prepare_frame, resolve_params  # noqa: E402
from app.utils import interval_ms  # noqa: E402

CODE = "supertrend_current"
INTERVAL = "15m"
STEP = pd.Timedelta(minutes=15)
STEP_MS = interval_ms(INTERVAL)
DAY_MS = 86_400_000
WINDOW_BARS = 10
MIN_SEGMENT_BARS = 800
CASH = 10_000.0
# Side-linear features can be negated for the counterfactual direction;
# max/min and the positive-bar fraction need their own remapping below.
_LINEAR_ALIGNED = (
    ["side"]
    + [f"ret_{j}" for j in range(WINDOW_BARS)]
    + [f"body_{j}" for j in range(WINDOW_BARS)]
    + ["window_return", "body_sum", "last3_momentum", "high_distance", "low_distance"]
)
DIRECTION_COLS = tuple(_LINEAR_ALIGNED + ["best_aligned_return", "worst_aligned_return", "up_bars_aligned"])


def opposite_view(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Features the model would see for the counterfactual opposite direction."""
    out = frame[feature_cols].copy()
    out[_LINEAR_ALIGNED] *= -1
    best, worst = out["best_aligned_return"].copy(), out["worst_aligned_return"].copy()
    out["best_aligned_return"] = -worst
    out["worst_aligned_return"] = -best
    out["up_bars_aligned"] = 1.0 - out["up_bars_aligned"]
    return out


def make_features(frame: pd.DataFrame, i: int, side: int) -> dict:
    w = frame.iloc[i - WINDOW_BARS:i]
    o = w.open.to_numpy(float)
    h = w.high.to_numpy(float)
    low = w.low.to_numpy(float)
    c = w.close.to_numpy(float)
    v = w.volume.to_numpy(float)
    base = c[-1]
    bar_range = np.maximum(h - low, 1e-12)
    rets = c / np.r_[c[0], c[:-1]] - 1.0
    bodies = c / o - 1.0
    upper_wick = (h - np.maximum(o, c)) / bar_range
    lower_wick = (np.minimum(o, c) - low) / bar_range
    aligned_rets = rets * side
    out = {"side": float(side)}
    for j in range(WINDOW_BARS):
        out.update({
            f"ret_{j}": float(aligned_rets[j]),
            f"body_{j}": float(bodies[j] * side),
            f"range_{j}": float(bar_range[j] / c[j]),
            f"upper_wick_{j}": float(upper_wick[j]),
            f"lower_wick_{j}": float(lower_wick[j]),
            f"volume_{j}": float(v[j] / max(v.mean(), 1e-12)),
        })
    cumulative = base / c[0] - 1.0
    out.update({
        "window_return": float(cumulative * side),
        "body_sum": float(bodies.sum() * side),
        "return_mean_abs": float(np.abs(rets).mean()),
        "return_std": float(rets.std(ddof=0)),
        "best_aligned_return": float(aligned_rets.max()),
        "worst_aligned_return": float(aligned_rets.min()),
        "up_bars_aligned": float((aligned_rets > 0).mean()),
        "close_position": float((base - low.min()) / max(h.max() - low.min(), 1e-12)),
        "volume_last3_first3": float(v[-3:].mean() / max(v[:3].mean(), 1e-12)),
        "range_last3_first3": float(bar_range[-3:].mean() / max(bar_range[:3].mean(), 1e-12)),
        "last3_momentum": float((c[-1] / c[-4] - 1.0) * side),
        "high_distance": float((h.max() / base - 1.0) * side),
        "low_distance": float((base / low.min() - 1.0) * side),
    })
    return out


def load_symbol_data(symbol: str, start_ms: int, end_ms: int, params: dict):
    engine = make_engine(read_only=True, null_pool=True)
    db = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)()
    try:
        frame = MarketDataService(db).load_frame(symbol, INTERVAL, start_ms=start_ms, end_ms=end_ms)
    finally:
        db.close()
        engine.dispose()
    if frame.empty:
        return []
    frame = frame.loc[(frame.index.as_unit("ms").asi8 + STEP_MS) <= end_ms].copy()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    mask = numeric.notna().all(axis=1)
    mask &= (numeric[["open", "high", "low", "close"]] > 0).all(axis=1)
    mask &= numeric.volume.ge(0)
    mask &= numeric.high.ge(numeric[["open", "close", "low"]].max(axis=1))
    mask &= numeric.low.le(numeric[["open", "close", "high"]].min(axis=1))
    frame = frame.loc[mask]
    if frame.empty:
        return []
    diffs = frame.index.to_series().diff().dt.total_seconds() * 1000
    parts = []
    for segment in [g for _, g in frame.groupby(diffs.ne(STEP_MS).cumsum())]:
        if len(segment) >= MIN_SEGMENT_BARS:
            prepared = prepare_frame(segment, CODE, params, symbol=symbol)
            signals = evaluate_signals(prepared, CODE, params, symbol=symbol)
            parts.append((prepared, signals))
    return parts


def generate_symbol(task: dict) -> dict:
    params = task["params"]
    try:
        parts = load_symbol_data(task["symbol"], task["start_ms"], task["end_ms"], params)
    except Exception as exc:
        return {"symbol": task["symbol"], "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    rows = []
    coverage = []
    for frame, signals in parts:
        coverage.append({
            "start": str(frame.index[0]), "end": str(frame.index[-1] + STEP), "bars": len(frame),
        })
        indices = np.flatnonzero(signals.direction.to_numpy() != 0)
        for i in indices:
            if i < WINDOW_BARS:
                continue
            outcome = isolated_trade(frame, signals, int(i), params, task["fee"], task["slippage"], STEP_MS)
            if outcome is None:
                continue
            rows.append({
                "symbol": task["symbol"],
                "time": frame.index[i],
                "entry_time": pd.Timestamp(outcome["entry_time"]),
                "exit_time": pd.Timestamp(outcome["exit_time"]),
                "side": int(outcome["side"]),
                "status": outcome["status"],
                "hold_bars": int(outcome["hold_bars"]),
                "net_return_pct": float(outcome["return_pct"]),
                "label": int(outcome["return_pct"] > 0),
                "censored": bool(outcome["censored"]),
                **make_features(frame, int(i), int(outcome["side"])),
            })
    return {"symbol": task["symbol"], "status": "ok", "rows": rows, "coverage": coverage}


def gate_frame(signals: pd.DataFrame, probability: pd.Series, opposite: pd.Series, threshold: float | None):
    gated = signals.copy()
    agree = probability.gt(opposite).reindex(signals.index)
    if threshold is None:
        allow = pd.Series(True, index=signals.index)
    else:
        allow = agree & probability.ge(threshold).reindex(signals.index)
    gated["allow_long_entry"] = allow
    gated["allow_short_entry"] = allow
    return gated, int(agree.fillna(False).reindex(signals[signals.direction != 0].index).sum())


def path_for_symbol(task: dict) -> dict:
    params = task["params"]
    parts = load_symbol_data(task["symbol"], task["start_ms"], task["end_ms"], params)
    thresholds = [None] + task["thresholds"]
    result = {"symbol": task["symbol"], "periods": {}}
    for period_name, start, end in (
        ("validation", task["validation_start"], task["validation_end"]),
        ("test", task["test_start"], task["test_signal_end"]),
    ):
        by_threshold = {}
        for threshold in thresholds:
            factors = []
            equity_parts = []
            trades = []
            raw_agreements = 0
            raw_signals = 0
            for frame, signals in parts:
                mask = (frame.index >= start) & (frame.index < end)
                period_frame = frame.loc[mask]
                if len(period_frame) < 2:
                    continue
                period_signals = signals.loc[mask].copy()
                probs = task["probability"].reindex(period_frame.index)
                opps = task["opposite"].reindex(period_frame.index)
                gated, agreements = gate_frame(period_signals, probs, opps, threshold)
                raw_agreements += agreements
                raw_signals += int(period_signals.direction.ne(0).sum())
                equity, raw_trades = simulate(period_frame, gated, params, CASH, task["fee"], task["slippage"], INTERVAL)
                factor = float(equity.iloc[-1]) / CASH
                factors.append(factor)
                equity_parts.append(equity * float(np.prod(factors[:-1])) if len(factors) > 1 else equity)
                trades.extend(aggregate_positions(raw_trades))
            if not factors:
                by_threshold[str(threshold)] = None
                continue
            combined_equity = pd.concat(equity_parts)
            compound = float(np.prod(factors) - 1.0) * 100
            drawdown = float((combined_equity / combined_equity.cummax().clip(lower=CASH) - 1).min() * 100)
            by_threshold[str(threshold)] = {
                "return_pct": compound,
                "max_drawdown_pct": drawdown,
                "trades": len(trades),
                "agreement_signals": raw_agreements,
                "raw_signals": raw_signals,
                "trade_returns": [[float(t["return_pct"]), "long" if str(t["side"]).lower() in ("long", "1") else "short"] for t in trades],
                "status_counts": {str(k): int(v) for k, v in pd.Series([t["status"] for t in trades]).value_counts().items()},
            }
        result["periods"][period_name] = by_threshold
    return result


def summarize_paths(results: list[dict], threshold: float | None, period: str) -> dict:
    key = str(threshold)
    accounts, all_returns, longs, shorts, status = [], [], [], [], {}
    signals = agreements = trades = 0
    drawdowns = []
    for item in results:
        value = item["periods"][period].get(key)
        if not value:
            continue
        accounts.append(value["return_pct"])
        drawdowns.append(value["max_drawdown_pct"])
        signals += value["raw_signals"]
        agreements += value["agreement_signals"]
        trades += value["trades"]
        for ret, side in value["trade_returns"]:
            all_returns.append(ret)
            (longs if side == "long" else shorts).append(ret)
        for name, count in value["status_counts"].items():
            status[name] = status.get(name, 0) + count
    def ci(values):
        return bootstrap_expectancy_ci([{"return_pct": x} for x in values], min_trades=30)
    return {
        "symbols": len(accounts),
        "account_mean_return_pct": round(float(np.mean(accounts)), 4) if accounts else None,
        "account_median_return_pct": round(float(np.median(accounts)), 4) if accounts else None,
        "positive_accounts": int(sum(x > 0 for x in accounts)),
        "mean_max_drawdown_pct": round(float(np.mean(drawdowns)), 4) if drawdowns else None,
        "raw_signals": signals,
        "model_agreement_signals": agreements,
        "trades": trades,
        "pooled": ci(all_returns),
        "long": ci(longs),
        "short": ci(shorts),
        "status_counts": dict(sorted(status.items(), key=lambda x: -x[1])),
    }


def classification_stats(part: pd.DataFrame) -> dict:
    mature = part.loc[~part.censored]
    result = {
        "samples": int(len(part)),
        "mature_samples": int(len(mature)),
        "positive_rate": round(float(mature.label.mean()), 4),
        "mean_net_return_pct": round(float(mature.net_return_pct.mean()), 4),
    }
    if mature.label.nunique() == 2:
        result["auc"] = round(float(roc_auc_score(mature.label, mature.probability)), 4)
        ranks = mature.probability.rank()
        y = mature.net_return_pct.rank()
        ic = float(np.corrcoef(ranks, y)[0, 1])
        result["return_rank_ic"] = round(ic, 4)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--min-bars", type=int, default=15_000)
    ap.add_argument("--processes", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--out", default="output/supertrend_ml_realized_20261104")
    ap.add_argument("--train-end", default="2026-08-01T00:00:00Z")
    ap.add_argument("--validation-end", default="2026-09-15T00:00:00Z")
    ap.add_argument("--test-start", default="2026-09-15T00:00:00Z")
    ap.add_argument("--test-signal-end", default="2026-10-25T00:00:00Z")
    ap.add_argument("--min-agreement-samples", type=int, default=200)
    args = ap.parse_args()

    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"输出目录已存在: {out}")
    out.mkdir(parents=True)

    fee = float(DEFAULT_STATE["fee_rate"])
    slippage = float(DEFAULT_STATE["slippage"])
    params = resolve_params(CODE, {"fee_rate": fee, "slippage": slippage}, interval=INTERVAL)
    now = pd.Timestamp.now(tz="UTC").floor("15min")
    end_ms = int(now.timestamp() * 1000)
    start_ms = int(pd.Timestamp("2025-12-01T00:00:00Z").timestamp() * 1000)

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        engine = make_engine(read_only=True)
        with engine.connect() as conn:
            from sqlalchemy import text
            rows = conn.execute(text(
                "select symbol from klines where interval='15m' group by symbol "
                "having count(*) >= :n order by symbol"
            ), {"n": args.min_bars})
            symbols = [r[0] for r in rows]
        engine.dispose()

    tasks = [{"symbol": s, "start_ms": start_ms, "end_ms": end_ms, "params": params,
              "fee": fee, "slippage": slippage} for s in symbols]
    print(f"生成真实成交标签：{len(symbols)} 币，{args.processes} 进程", flush=True)
    generated = []
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = {pool.submit(generate_symbol, task): task["symbol"] for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            value = future.result()
            generated.append(value)
            if done % 10 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}", flush=True)
    errors = [x for x in generated if x["status"] != "ok"]
    ok = [x for x in generated if x["status"] == "ok" and x["rows"]]
    rows = pd.DataFrame([row for item in ok for row in item["rows"]]).sort_values(["time", "symbol"])
    if len(rows) < 1000:
        raise SystemExit(f"有效样本不足: {len(rows)}")

    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)
    test_start = pd.Timestamp(args.test_start)
    test_signal_end = pd.Timestamp(args.test_signal_end)
    train = rows[(rows.time < train_end) & (rows.exit_time < train_end) & ~rows.censored]
    validation = rows[(rows.time >= train_end) & (rows.time < validation_end) & (rows.exit_time < validation_end) & ~rows.censored]
    test = rows[(rows.time >= test_start) & (rows.time < test_signal_end) & ~rows.censored]
    feature_cols = [c for c in rows.columns if c not in (
        "symbol", "time", "entry_time", "exit_time", "side", "status", "hold_bars",
        "net_return_pct", "label", "censored", "probability", "opposite_probability",
    )]
    # side must remain available to the model; it is known when the strategy issues a signal.
    feature_cols = ["side"] + feature_cols

    model = LGBMClassifier(
        n_estimators=240, num_leaves=15, learning_rate=0.03, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
        random_state=20261104, n_jobs=1, verbose=-1,
    )
    model.fit(train[feature_cols], train.label)
    logistic = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=1000))
    logistic.fit(train[feature_cols], train.label)

    def score(frame: pd.DataFrame, estimator) -> pd.Series:
        return pd.Series(estimator.predict_proba(frame[feature_cols])[:, 1], index=frame.index)
    rows["probability"] = score(rows, model)
    rows["logistic_probability"] = score(rows, logistic)
    counterfactual = opposite_view(rows, feature_cols)
    rows["opposite_probability"] = model.predict_proba(counterfactual)[:, 1]

    # Rebuild split views after scoring so they carry probability columns.
    train = rows[(rows.time < train_end) & (rows.exit_time < train_end) & ~rows.censored]
    validation = rows[(rows.time >= train_end) & (rows.time < validation_end) & (rows.exit_time < validation_end) & ~rows.censored]
    test = rows[(rows.time >= test_start) & (rows.time < test_signal_end) & ~rows.censored]
    split_stats = {
        "train": classification_stats(train),
        "validation": classification_stats(validation),
        "test": classification_stats(test),
        "test_logistic": classification_stats(test.assign(probability=test.logistic_probability)),
    }

    valid_positive = validation.loc[validation.probability > validation.opposite_probability, "probability"]
    if len(valid_positive) < args.min_agreement_samples:
        raise SystemExit(
            f"验证期方向一致样本不足: {len(valid_positive)} < {args.min_agreement_samples}"
        )
    thresholds = [float(valid_positive.quantile(q)) for q in (0.0, 0.25, 0.50, 0.75)]
    probability = {s: g.set_index("time").probability for s, g in rows.groupby("symbol")}
    opposite = {s: g.set_index("time").opposite_probability for s, g in rows.groupby("symbol")}
    empty = pd.Series(dtype=float)

    def series_for(mapping: dict, symbol: str) -> pd.Series:
        return mapping.get(symbol, empty)

    path_tasks = [{
        "symbol": s, "start_ms": start_ms, "end_ms": end_ms, "params": params, "fee": fee,
        "slippage": slippage, "thresholds": thresholds,
        "probability": series_for(probability, s),
        "opposite": series_for(opposite, s), "validation_start": train_end,
        "validation_end": validation_end, "test_start": test_start,
        "test_signal_end": test_signal_end,
    } for s in symbols]
    print(f"回放真实账户路径：{len(symbols)} 币", flush=True)
    path_results = []
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = {pool.submit(path_for_symbol, task): task["symbol"] for task in path_tasks}
        for done, future in enumerate(as_completed(futures), 1):
            path_results.append(future.result())
            if done % 10 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}", flush=True)

    validation_paths = [summarize_paths(path_results, t, "validation") for t in thresholds]
    baseline_validation = summarize_paths(path_results, None, "validation")
    chosen_index = max(range(len(thresholds)), key=lambda i: validation_paths[i]["account_mean_return_pct"])
    chosen_threshold = thresholds[chosen_index]
    test_baseline = summarize_paths(path_results, None, "test")
    test_selected = summarize_paths(path_results, chosen_threshold, "test")

    rows.to_csv(out / "samples.csv", index=False, float_format="%.17g")
    import pickle
    with (out / "model.pkl").open("wb") as f:
        pickle.dump({"model": model, "features": feature_cols, "threshold": chosen_threshold,
                     "direction_features": DIRECTION_COLS, "params": params}, f)
    feature_importance = sorted(
        [{"feature": k, "importance": int(v)} for k, v in zip(feature_cols, model.feature_importances_)],
        key=lambda x: -x["importance"],
    )[:20]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_end_exclusive": str(now),
        "label": "每个信号独立按策略实际止损/1R2R3R分批止盈/反向信号平仓后的净收益率",
        "symbols_requested": len(symbols),
        "symbols_with_samples": len(ok),
        "errors": errors,
        "costs": {"fee_rate": fee, "slippage": slippage, "cash_per_symbol": CASH},
        "splits": {"train_end": str(train_end), "validation_end": str(validation_end),
                   "test_start": str(test_start), "test_signal_end": str(test_signal_end)},
        "model": {"type": "LightGBM binary classifier", "features": len(feature_cols),
                  "feature_window_bars": WINDOW_BARS},
        "classification": split_stats,
        "thresholds": thresholds,
        "validation_baseline": baseline_validation,
        "validation_by_threshold": validation_paths,
        "chosen_threshold": chosen_threshold,
        "test_baseline": test_baseline,
        "test_selected": test_selected,
        "top_feature_importance": feature_importance,
        "coverage": {x["symbol"]: x["coverage"] for x in ok},
        "leakage_controls": [
            "特征只取信号K线之前10根已完成K线；信号方向在信号根收盘后才可得",
            "训练标签必须在训练截止前完成退出；测试阈值只由验证期账户结果选择",
            "次根开盘成交，止损/止盈从信号收盘价推导，计双边手续费和滑点",
            "最终路径重新模拟账户，筛掉信号会改变后续持仓与可交易信号",
        ],
        "limitations": [
            "逐币独立账户，不是共享资金组合；未模拟资金费率、盘口深度和下单延迟",
            "历史约300天，且不同币种上市时间不同；单次样本外切分不能替代长期滚动验证",
            "隔离标签用于训练以消除持仓占用删失，最终部署效果以真实账户路径回测为准",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "samples": {k: v for k, v in split_stats.items()},
        "threshold": chosen_threshold,
        "validation_baseline": baseline_validation,
        "validation_selected": validation_paths[chosen_index],
        "test_baseline": test_baseline,
        "test_selected": test_selected,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

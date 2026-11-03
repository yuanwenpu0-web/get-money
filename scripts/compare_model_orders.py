"""Minute-by-minute offline order simulation on the common held-out window."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import joblib
import httpx
import numpy as np
import pandas as pd
import torch
from sqlalchemy import text
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.backtest.execution import simulate  # noqa: E402
from app.backtest.metrics import metrics_from_equity  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.ml.minute_research import classify, data_fingerprint, make_features, make_labels, time_partitions  # noqa: E402
from app.ml.research_data import load_candles, validate_candles  # noqa: E402
from app.ml.transformer_research import PriceTransformer, Sequences, sequence_endpoints  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from train_transformer_research import logits_input, predict  # noqa: E402
from backfill_all_history import AccessBlocked, RateLimitedClient  # noqa: E402

RUN = ROOT / "data" / "model_research" / "verified_20261103"
BINARY = ROOT / "data" / "model_research" / "binary_15m_20261103T003231207547Z"
EIGHT_HOURS = 8 * 60 * 60000
INITIAL, FEE, SLIPPAGE = 10000., .0004, .0002
POSITION_FRACTION, STOP, TARGET, RISK_BUDGET = .08, .005, .01, .005
THRESHOLD, HOLD, COOLDOWN = .45, 15, 5


def common_window(engine, symbols, run=RUN, binary_root=BINARY):
    starts, ends = [], []
    with engine.connect() as conn:
        for symbol in symbols:
            config = json.loads((run / symbol / "model_config.json").read_text(encoding="utf-8"))
            tconfig = json.loads((run / symbol / "transformer" / "model_config.json").read_text(encoding="utf-8"))
            starts.append(config["split"]["test"]["first_open_ms"] + (tconfig["sequence_length"] - 1) * 60000)
            ends.append(config["split"]["test"]["last_open_ms"])
            binary_report = binary_root / symbol / "report.json"
            if binary_report.exists():
                binary_config = json.loads(binary_report.read_text(encoding="utf-8"))
                starts.append(binary_config["split"]["test"]["first_open_ms"])
                ends.append(binary_config["split"]["test"]["last_open_ms"])
            first, last = conn.execute(text("SELECT min(funding_time),max(funding_time) FROM funding_rates WHERE symbol=:symbol"), {"symbol": symbol}).one()
            if first is None or last is None:
                raise ValueError(f"{symbol} 缺少资金费率数据")
            starts.append(((int(first) + EIGHT_HOURS - 1) // EIGHT_HOURS) * EIGHT_HOURS)
            ends.append((int(last) // EIGHT_HOURS) * EIGHT_HOURS + EIGHT_HOURS - 60000)
    start, end = max(starts), min(ends)
    if start >= end:
        raise ValueError("无法找到共同样本外窗口")
    return start, end


def load_funding(engine, symbol, start, end, client):
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT funding_time,funding_rate,mark_price FROM funding_rates WHERE symbol=:s AND funding_time>=:start AND funding_time<:end ORDER BY funding_time"),
                            {"s": symbol, "start": start, "end": end + 60000}).all()
    grouped = {}
    approximate = 0
    for timestamp, rate, mark in rows:
        if timestamp % EIGHT_HOURS >= 60000 or rate is None or not np.isfinite(rate):
            raise ValueError(f"{symbol} 资金费率时间或数值异常")
        event = timestamp // EIGHT_HOURS * EIGHT_HOURS
        grouped.setdefault(event, []).append((float(rate), float(mark) if mark is not None and np.isfinite(mark) and mark > 0 else None))
    events = {}
    resolved = 0
    for event, candidates in grouped.items():
        if len(candidates) == 1:
            events[event] = candidates[0]
        else:
            response = client.get("https://fapi.binance.com/fapi/v1/fundingRate", params={"symbol": symbol,
                "startTime": event, "endTime": event + 59999, "limit": 10})
            response.raise_for_status()
            authoritative = response.json()
            if len(authoritative) != 1 or authoritative[0]["symbol"] != symbol or int(authoritative[0]["fundingTime"]) // EIGHT_HOURS * EIGHT_HOURS != event:
                raise ValueError(f"{symbol} 资金费率重复且无法确认")
            rate = float(authoritative[0]["fundingRate"])
            mark = float(authoritative[0].get("markPrice") or 0)
            if not np.isfinite(rate) or not np.isfinite(mark):
                raise ValueError(f"{symbol} 资金费率来源异常")
            events[event] = (rate, mark if mark > 0 else None)
            resolved += 1
        approximate += int(events[event][1] is None)
    expected = set(range(((start + EIGHT_HOURS - 1) // EIGHT_HOURS) * EIGHT_HOURS, end + 1, EIGHT_HOURS))
    if set(events) != expected:
        raise ValueError(f"{symbol} 测试期间资金费率不完整")
    return {pd.Timestamp(k, unit="ms", tz="UTC"): v for k, v in events.items()}, approximate, resolved


def decisions(probabilities, threshold=THRESHOLD):
    direction = np.zeros(len(probabilities), dtype=np.int8)
    direction[(probabilities[:, 2] >= threshold) & (probabilities[:, 2] > probabilities[:, 0]) & (probabilities[:, 2] > probabilities[:, 1])] = 1
    direction[(probabilities[:, 0] >= threshold) & (probabilities[:, 0] > probabilities[:, 2]) & (probabilities[:, 0] > probabilities[:, 1])] = -1
    return direction


def simulate_orders(frame, endpoints, directions, funding):
    direction_values = np.zeros(len(frame), dtype=np.int8)
    direction_values[endpoints] = np.asarray(directions, dtype=np.int8)
    signals = pd.DataFrame({"direction": direction_values, "position_pct": POSITION_FRACTION,
                            "leverage": 1, "sl_pct": STOP, "tp_pct": TARGET}, index=frame.index)
    params = {"min_hold_bars": HOLD, "max_hold_bars": HOLD, "cooldown_bars": COOLDOWN,
              "risk_per_trade_pct": RISK_BUDGET, "trailing_enabled": False}
    equity, trades = simulate(frame, signals, params, INITIAL, FEE, SLIPPAGE, "1m", funding_rates=funding)
    metrics = metrics_from_equity(equity, trades, INITIAL, "1m")
    metrics["funding_pnl"] = round(sum(t["funding_pnl"] for t in trades), 4)
    metrics["long_pnl"] = round(sum(t["pnl"] for t in trades if t["side"] == "long"), 4)
    metrics["short_pnl"] = round(sum(t["pnl"] for t in trades if t["side"] == "short"), 4)
    metrics["mean_holding_minutes"] = round(float(np.mean([(pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds() / 60 for t in trades])), 2) if trades else None
    monthly = equity.resample("ME").last()
    previous = monthly.shift(1)
    if len(previous):
        previous.iloc[0] = INITIAL
    metrics["monthly_return_pct"] = {str(k.date()): float(v) for k, v in ((monthly / previous - 1) * 100).round(4).items()}
    return metrics, trades, equity


def one_symbol(engine, symbol, start, end, threads, output, client, run=RUN, binary_root=BINARY):
    model_dir = run / symbol
    lconfig = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    tdir = model_dir / "transformer"
    tconfig = json.loads((tdir / "model_config.json").read_text(encoding="utf-8"))
    if tconfig["data_fingerprint"] != lconfig["data_fingerprint"]:
        raise ValueError("两个模型数据指纹不一致")
    funding, proxies, resolved = load_funding(engine, symbol, start, end, client)
    (output / f"{symbol}_funding.json").write_text(json.dumps([
        {"time": stamp.isoformat(), "rate": rate, "mark_price": mark}
        for stamp, (rate, mark) in sorted(funding.items())], ensure_ascii=False, indent=2), encoding="utf-8")
    raw = load_candles(engine, symbol, lconfig["as_of_ms"])
    frame, quality = validate_candles(raw, symbol, lconfig["as_of_ms"])
    if not quality["passed"] or data_fingerprint(frame) != lconfig["data_fingerprint"]:
        raise ValueError("历史数据质量或快照指纹不一致")
    del raw
    features, labels = make_features(frame), make_labels(frame)
    valid = features.notna().all(axis=1) & labels.notna().all(axis=1)
    original = np.flatnonzero(valid.to_numpy())
    features = features.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].reset_index(drop=True)
    tr, va, te = time_partitions(len(features))
    times = frame.open_time.iloc[original].to_numpy()
    for name, part in (("train", tr), ("validation", va), ("test", te)):
        expected = lconfig["split"][name]
        if len(times[part]) != expected["rows"] or int(times[part][0]) != expected["first_open_ms"] or int(times[part][-1]) != expected["last_open_ms"]:
            raise ValueError("样本外切分与训练配置不一致")
    endpoints = sequence_endpoints(original, te, tconfig["sequence_length"], 1)
    endpoints = endpoints[(times[endpoints] >= start) & (times[endpoints] <= end - (HOLD + 1) * 60000)]
    if not len(endpoints):
        raise ValueError("没有共同样本外预测时点")
    test_features = features.iloc[endpoints].astype(np.float32)
    lightgbm = joblib.load(model_dir / "model.pkl")
    lp = lightgbm["direction"].predict_proba(test_features)
    if lightgbm["calibrator"] is not None:
        lp = lightgbm["calibrator"].predict_proba(logits_input(lp))
    scaled = (features.to_numpy(dtype=np.float32) - np.asarray(tconfig["feature_mean"], dtype=np.float32)) / np.asarray(tconfig["feature_scale"], dtype=np.float32)
    dummy_class = np.zeros(len(features), dtype=np.int64)
    dummy_reg = np.zeros((len(features), 2), dtype=np.float32)
    loader = DataLoader(Sequences(scaled, dummy_class, dummy_reg, endpoints, tconfig["sequence_length"]), batch_size=128, shuffle=False, num_workers=0)
    model = PriceTransformer(len(features.columns), tconfig["sequence_length"], tconfig["d_model"], tconfig["heads"], tconfig["layers"])
    model.load_state_dict(torch.load(tdir / "model.pt", map_location="cpu", weights_only=True))
    tp, _ = predict(model, loader, np.asarray(tconfig["target_mean"]), np.asarray(tconfig["target_scale"]))
    calibrator = tdir / "calibrator.pkl"
    if calibrator.exists():
        tp = joblib.load(calibrator).predict_proba(logits_input(tp))
    weight = tconfig["validation_weight"]
    ensemble = weight * lp + (1 - weight) * tp
    directions = {"lightgbm": decisions(lp), "transformer": decisions(tp), "ensemble_research": decisions(ensemble),
                  "momentum": np.where(test_features.return_15 > lconfig["direction_threshold"], 1,
                                       np.where(test_features.return_15 < -lconfig["direction_threshold"], -1, 0))}
    binary_path = binary_root / symbol / "candidate.joblib"
    if binary_path.exists():
        binary_report = json.loads((binary_path.parent / "report.json").read_text(encoding="utf-8"))
        if binary_report["data_fingerprint"] != lconfig["data_fingerprint"]:
            raise ValueError("二分类模型数据指纹不一致")
        binary = joblib.load(binary_path)
        if binary["features"] != list(test_features.columns):
            raise ValueError("二分类模型特征不一致")
        prob = binary["binary"].predict_proba(test_features)[:, 1]
        directions["binary"] = np.where(prob >= .5, 1, -1)
        confidence = float(binary["confidence_threshold"])
        directions["binary_filtered"] = np.where(prob >= confidence, 1, np.where(prob <= 1-confidence, -1, 0))
    lo = int(np.searchsorted(frame.open_time.to_numpy(), start))
    hi = int(np.searchsorted(frame.open_time.to_numpy(), end, side="right"))
    bars = frame.iloc[lo:hi].copy()
    bars.index = pd.to_datetime(bars.open_time, unit="ms", utc=True)
    bars["atr_pct"] = 0.
    positions = original[endpoints] - lo
    results = {}
    for name, direction in directions.items():
        metrics, trades, equity = simulate_orders(bars, positions, direction, funding)
        results[name] = metrics
        pd.DataFrame(trades).to_csv(output / f"{symbol}_{name}_trades.csv", index=False)
        daily = equity.resample("1D").last()
        daily.to_csv(output / f"{symbol}_{name}_equity_daily.csv", header=["equity"])
    return {"symbol": symbol, "signals_evaluated": len(endpoints), "funding_events": len(funding),
            "funding_mark_price_fallbacks": proxies, "duplicate_funding_events_resolved_via_api": resolved,
            "ensemble_holdout_gate_prior_test_metric_only": tconfig["ensemble_enabled"], "strategies": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=list(MAJOR_SYMBOLS), choices=MAJOR_SYMBOLS)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads 必须为正")
    torch.set_num_threads(args.threads)
    output = RUN / ("order_backtest_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True, exist_ok=False)
    engine = make_engine(read_only=True, null_pool=True)
    results = {}
    try:
        start, end = common_window(engine, args.symbols)
        config = {"symbols": args.symbols, "start_ms": start, "end_ms": end, "initial_usdt_per_symbol_per_strategy": INITIAL,
                  "leverage": 1, "margin_fraction": POSITION_FRACTION, "risk_per_trade": RISK_BUDGET,
                  "stop_pct": STOP, "take_profit_pct": TARGET, "hold_minutes": HOLD, "cooldown_minutes": COOLDOWN,
                  "stop_target_reference": "上一根产生信号的已收盘K线收盘价，非实际成交价",
                  "fee_per_side": FEE, "slippage_per_side": SLIPPAGE, "probability_threshold": THRESHOLD,
                  "signal_frequency": "1m", "trade_on": "next_open", "funding": "8h, mark price or candle-open proxy",
                  "duplicate_funding_resolution": "同一结算窗口重复时查询币安官方历史资金费率，不修改数据库",
                  "ensemble": "权重仅由验证集确定；本测试区间内只作假设性对比，不能依据同一测试集上的融合门槛择优或宣称已部署",
                  "limitations": ["各币种与策略使用独立账户，不是共享资金组合", "不模拟深度、排队、延迟、强平与交易所最小步长", "止盈止损同根触发时先止损", "资金费率按分钟开盘近似结算", "二分类模型原以15分钟抽样训练，本次逐分钟推断存在分布差异"]}
        (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"共同测试范围: {pd.Timestamp(start,unit='ms',tz='UTC')} 至 {pd.Timestamp(end,unit='ms',tz='UTC')}；结果目录 {output}", flush=True)
        with httpx.Client(timeout=60, follow_redirects=True) as raw_client:
            client = RateLimitedClient(raw_client)
            for symbol in args.symbols:
                started = time.monotonic()
                print(f"{symbol}: 检查数据、逐分钟推断与模拟", flush=True)
                try:
                    result = one_symbol(engine, symbol, start, end, args.threads, output, client)
                    results[symbol] = {"status": "complete", "elapsed_seconds": round(time.monotonic()-started, 1), **result}
                except AccessBlocked:
                    raise
                except Exception as exc:
                    results[symbol] = {"status": "failed", "error": type(exc).__name__, "detail": str(exc) if isinstance(exc, ValueError) else "请查看日志"}
                (output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                print(f"{symbol}: {results[symbol]['status']} ({time.monotonic()-started:.1f}s)", flush=True)
    finally:
        engine.dispose()
    return int(any(result["status"] != "complete" for result in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())

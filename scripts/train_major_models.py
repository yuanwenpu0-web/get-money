"""Train independent, offline 15-minute LightGBM research models per major coin."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import tempfile
import uuid
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine  # noqa: E402
from app.ml.minute_research import FEATURE_VERSION, HORIZON, LABEL_VERSION, classify, data_fingerprint, make_features, make_labels, time_partitions, validation_blocks  # noqa: E402
from app.ml.research_data import load_candles, validate_candles  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402

# Excludes the unresolved BTC candle at 2026-11-02 13:42 UTC and later rows.
SAFE_END_MS = 1793626920000


def register_session(config, session_id):
    if not session_id or session_id in config.get("used_session_ids", []):
        raise ValueError("每次训练或续跑必须使用新的唯一 session-id")
    config.setdefault("used_session_ids", []).append(session_id)


def classification_metrics(y, p):
    labels = np.argmax(p, axis=1)
    return {"log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
            "macro_f1": float(f1_score(y, labels, labels=[0, 1, 2], average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, labels)),
            "brier_score": float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))}


def regression_metrics(y, prediction):
    correlation = float(np.corrcoef(y, prediction)[0, 1]) if np.std(y) and np.std(prediction) else None
    if correlation is not None and not np.isfinite(correlation):
        correlation = None
    return {"mae": float(mean_absolute_error(y, prediction)),
            "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
            "correlation": correlation, "bias": float(np.mean(prediction - y))}


def fit_models(frame, output, trees=120, threads=3, metadata=None):
    features = make_features(frame)
    labels = make_labels(frame)
    valid = features.notna().all(axis=1) & labels.notna().all(axis=1)
    features = features.loc[valid].astype(np.float32)
    labels = labels.loc[valid]
    times = frame.loc[valid, "open_time"].reset_index(drop=True)
    features = features.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    tr, va, te = time_partitions(len(features))
    if len(features.iloc[tr]) < 500 or len(features.iloc[va]) < 100 or len(features.iloc[te]) < 100:
        raise ValueError("有效样本不足以训练和评估")
    threshold = float(labels.iloc[tr].future_return.abs().quantile(1 / 3))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("训练集方向阈值无效")
    direction = classify(labels.future_return.to_numpy(), threshold)
    if len(np.unique(direction[tr])) != 3:
        raise ValueError("训练集未覆盖三个方向类别")
    params = dict(n_estimators=trees, learning_rate=.05, num_leaves=31, min_child_samples=100,
                  n_jobs=threads, random_state=42, verbosity=-1, deterministic=True, force_col_wise=True)
    xtr, xval, xtest = features.iloc[tr], features.iloc[va], features.iloc[te]
    classifier = lgb.LGBMClassifier(objective="multiclass", **params)
    classifier.fit(xtr, direction[tr])
    if list(classifier.classes_) != [0, 1, 2]:
        raise ValueError("分类模型类别不完整")
    validation_probs = classifier.predict_proba(xval)
    test_probs = classifier.predict_proba(xtest)
    _, calibration_fit, calibration_select, _ = validation_blocks(times.iloc[va].to_numpy())
    calibration = None
    chosen = False
    if len(np.unique(direction[va][calibration_fit])) == 3:
        calibration = LogisticRegression(max_iter=300, random_state=42)
        calibration.fit(np.log(np.clip(validation_probs[calibration_fit], 1e-12, 1)), direction[va][calibration_fit])
        raw_score = log_loss(direction[va][calibration_select], validation_probs[calibration_select], labels=[0, 1, 2])
        cal_score = log_loss(direction[va][calibration_select], calibration.predict_proba(np.log(np.clip(validation_probs[calibration_select], 1e-12, 1))), labels=[0, 1, 2])
        chosen = bool(cal_score < raw_score)
    models = {"direction": classifier, "calibrator": calibration if chosen else None}
    report = {"direction_raw": classification_metrics(direction[te], test_probs),
              "calibration_selected_on_validation": chosen}
    if chosen:
        report["direction_calibrated"] = classification_metrics(direction[te], calibration.predict_proba(np.log(np.clip(test_probs, 1e-12, 1))))
    counts = np.bincount(direction[tr], minlength=3) / len(direction[tr])
    report["majority_baseline"] = classification_metrics(direction[te], np.tile(counts, (len(xtest), 1)))
    momentum = classify(xtest["return_15"].to_numpy(), threshold)
    report["momentum_baseline"] = {"macro_f1": float(f1_score(direction[te], momentum, labels=[0, 1, 2], average="macro", zero_division=0)),
                                   "balanced_accuracy": float(balanced_accuracy_score(direction[te], momentum))}
    for target in ("future_return", "future_volatility"):
        regressor = lgb.LGBMRegressor(objective="regression", **params)
        regressor.fit(xtr, labels.iloc[tr][target])
        models[target] = regressor
        report[target] = regression_metrics(labels.iloc[te][target].to_numpy(), regressor.predict(xtest))
    config = {"model_type": "lightgbm", "status": "candidate", "horizon_minutes": HORIZON,
              "feature_version": FEATURE_VERSION, "label_version": LABEL_VERSION,
              "feature_names": list(features.columns), "direction_threshold": threshold,
              "split": {name: {"rows": len(times.iloc[part]), "first_open_ms": int(times.iloc[part].iloc[0]),
                               "last_open_ms": int(times.iloc[part].iloc[-1])} for name, part in (("train", tr), ("validation", va), ("test", te))},
              "model_parameters": params, "calibration": "validation_block_2_fit_block_3_select_purged",
              "data_fingerprint": data_fingerprint(frame), "fingerprint_method": "pandas_hash_rows_sha256_v1"}
    config.update(metadata or {})
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, output / "model.pkl", compress=3)
    (output / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame({"feature": features.columns, "importance": classifier.feature_importances_}).sort_values("importance", ascending=False).to_csv(output / "feature_importance.csv", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=list(MAJOR_SYMBOLS), choices=MAJOR_SYMBOLS)
    parser.add_argument("--end-ms", type=int, default=SAFE_END_MS)
    parser.add_argument("--trees", type=int, default=120)
    parser.add_argument("--threads", type=int, choices=range(1, 65), default=3)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()
    session_id = args.session_id or uuid.uuid4().hex
    if args.end_ms > SAFE_END_MS or args.trees < 1:
        parser.error("截止时间不能超过已核查范围，trees 必须为正")
    run_dir = args.run_dir or ROOT / "data" / "model_research" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_config = {"symbols": args.symbols, "as_of_ms": args.end_ms, "model": "lightgbm",
                  "horizon_minutes": HORIZON, "trees": args.trees, "threads": args.threads}
    if args.resume:
        previous = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in run_config.items() if key != "threads"):
            raise ValueError("续跑配置与原任务不一致")
        register_session(previous, session_id)
        if previous.get("threads") != args.threads:
            history = previous.setdefault("thread_history", [])
            history.append({"previous": previous.get("threads"), "current": args.threads,
                            "changed_at": datetime.now(timezone.utc).isoformat()})
            previous["threads"] = args.threads
        (run_dir / "run_config.json").write_text(json.dumps(previous, indent=2), encoding="utf-8")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        register_session(run_config, session_id)
        (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    engine = make_engine(read_only=True, null_pool=True)
    status_path = run_dir / "run_status.json"
    results = json.loads(status_path.read_text(encoding="utf-8")) if args.resume and status_path.exists() else {}
    try:
        for symbol in args.symbols:
            started = time.monotonic()
            target_dir = run_dir / symbol
            if target_dir.exists():
                if all((target_dir / name).is_file() for name in ("model.pkl", "model_config.json", "evaluation_report.json", "data_quality_report.json")):
                    prior = json.loads((target_dir / "model_config.json").read_text(encoding="utf-8"))
                    quality = json.loads((target_dir / "data_quality_report.json").read_text(encoding="utf-8"))
                    if prior.get("symbol") != symbol or prior.get("as_of_ms") != args.end_ms or not quality.get("passed") or not prior.get("data_fingerprint"):
                        raise ValueError(f"{symbol} 已有结果的配置不一致，拒绝覆盖")
                    if results.get(symbol, {}).get("status") != "candidate":
                        results[symbol] = {"status": "candidate", "recovered": True}
                    temp_status = status_path.with_suffix(".tmp")
                    temp_status.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                    temp_status.replace(status_path)
                    print(f"{symbol}: 已完成，跳过", flush=True)
                    continue
                raise ValueError(f"{symbol} 已有未确认结果，拒绝覆盖")
            directory = Path(tempfile.mkdtemp(prefix=f".{symbol}-attempt-", dir=run_dir))
            results[symbol] = {"status": "running", "session_id": session_id}
            temp_status = status_path.with_suffix(".tmp")
            temp_status.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_status.replace(status_path)
            print(f"{symbol}: 读取并检查历史数据", flush=True)
            try:
                frame = load_candles(engine, symbol, args.end_ms)
                closed, quality = validate_candles(frame, symbol, args.end_ms)
                (directory / "data_quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
                del frame
                if not quality["passed"]:
                    raise ValueError("数据质量检查未通过，停止该币种训练")
                print(f"{symbol}: 检查通过，训练 {len(closed)} 根 K 线", flush=True)
                report = fit_models(closed, directory, args.trees, args.threads,
                                    {"symbol": symbol, "as_of_ms": args.end_ms, "model_version": run_dir.name})
                directory.rename(target_dir)
                results[symbol] = {"status": "candidate", "session_id": session_id, "elapsed_seconds": round(time.monotonic()-started, 1),
                                   "test_log_loss": report["direction_raw"]["log_loss"]}
                del closed, report
            except Exception as exc:
                results[symbol] = {"status": "failed", "session_id": session_id, "error": type(exc).__name__, "detail": str(exc) if isinstance(exc, ValueError) else "请查看运行环境和数据"}
            finally:
                gc.collect()
                temp_status = status_path.with_suffix(".tmp")
                temp_status.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                temp_status.replace(status_path)
                print(f"{symbol}: {results[symbol]}", flush=True)
    finally:
        engine.dispose()
    return int(any(r["status"] == "failed" for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())

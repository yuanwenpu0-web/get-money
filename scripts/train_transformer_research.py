"""Train and compare a small CPU Transformer on BTC 15-minute research data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine  # noqa: E402
from app.ml.minute_research import HORIZON, classify, data_fingerprint, make_features, make_labels, time_partitions, validation_blocks  # noqa: E402
from app.ml.research_backtest import simulate  # noqa: E402
from app.ml.research_data import load_candles, validate_candles  # noqa: E402
from app.ml.transformer_research import PriceTransformer, Sequences, sequence_endpoints  # noqa: E402
from train_major_models import classification_metrics, regression_metrics  # noqa: E402

DEFAULT_RUN = ROOT / "data" / "model_research" / "verified_20261103"


def logits_input(probs):
    return np.log(np.clip(probs, 1e-12, 1))


def predict(model, loader, regression_mean, regression_scale):
    model.eval()
    probabilities, predictions = [], []
    with torch.no_grad():
        for x, _, _ in loader:
            logits, values = model(x)
            probabilities.append(torch.softmax(logits, dim=1).numpy())
            predictions.append(values.numpy() * regression_scale + regression_mean)
    probabilities = np.concatenate(probabilities).astype(np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, np.concatenate(predictions)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    model_dir = args.run / args.symbol
    config = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    target_dir = model_dir / "transformer"
    if target_dir.exists():
        raise ValueError("Transformer 输出目录已存在，避免覆盖已有实验")
    engine = make_engine(read_only=True, null_pool=True)
    try:
        raw = load_candles(engine, args.symbol, config["as_of_ms"])
        frame, quality = validate_candles(raw, args.symbol, config["as_of_ms"])
    finally:
        engine.dispose()
    if not quality["passed"]:
        raise ValueError("数据质量不合格")
    if not config.get("data_fingerprint") or data_fingerprint(frame) != config["data_fingerprint"]:
        raise ValueError("历史数据快照与 LightGBM 不一致或未记录指纹，请重新训练 LightGBM")
    xframe, labels = make_features(frame), make_labels(frame)
    valid = xframe.notna().all(axis=1) & labels.notna().all(axis=1)
    original = np.flatnonzero(valid.to_numpy())
    times = frame.open_time.iloc[original].to_numpy()
    features = xframe.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].reset_index(drop=True)
    if list(features.columns) != config["feature_names"]:
        raise ValueError("特征版本与 LightGBM 不一致")
    tr, va, te = time_partitions(len(features))
    for name, part in (("train", tr), ("validation", va), ("test", te)):
        reference = config["split"][name]
        if len(times[part]) != reference["rows"] or int(times[part][0]) != reference["first_open_ms"] or int(times[part][-1]) != reference["last_open_ms"]:
            raise ValueError("训练数据时间范围与 LightGBM 不一致")
    train_x = features.iloc[tr].to_numpy(dtype=np.float32)
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-8] = 1
    x = (features.to_numpy(dtype=np.float32) - mean) / scale
    values = labels[["future_return", "future_volatility"]].to_numpy(dtype=np.float32)
    regression_mean = values[tr].mean(axis=0)
    regression_scale = values[tr].std(axis=0)
    regression_scale[regression_scale < 1e-8] = 1
    y_regression = ((values - regression_mean) / regression_scale).astype(np.float32)
    y_class = classify(values[:, 0], config["direction_threshold"])
    endpoints = {name: sequence_endpoints(original, part, args.sequence, args.stride)
                 for name, part in (("train", tr), ("validation", va), ("test", te))}
    if min(map(len, endpoints.values())) < 20:
        raise ValueError("连续序列样本不足")
    blocks = validation_blocks(times[va])
    relative_val = endpoints["validation"] - va.start
    masks = [(relative_val >= block.start) & (relative_val < block.stop) for block in blocks]
    if any(mask.sum() < 20 for mask in masks):
        raise ValueError("验证集分段序列样本不足")
    loaders = {name: DataLoader(Sequences(x, y_class, y_regression, ends, args.sequence),
                                batch_size=args.batch_size, shuffle=(name == "train"), num_workers=0)
               for name, ends in endpoints.items()}
    early_loader = DataLoader(Sequences(x, y_class, y_regression, endpoints["validation"][masks[0]], args.sequence),
                              batch_size=args.batch_size, shuffle=False, num_workers=0)
    output = Path(tempfile.mkdtemp(prefix=".transformer-attempt-", dir=model_dir))
    model = PriceTransformer(len(features.columns), args.sequence, args.width, 4, args.layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
    classification = nn.CrossEntropyLoss()
    regression = nn.HuberLoss()
    best, history, stale = float("inf"), [], 0
    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        for batch_index, (batch, y, target) in enumerate(loaders["train"], 1):
            optimizer.zero_grad(set_to_none=True)
            logits, predicted = model(batch)
            loss = classification(logits, y) + .5 * regression(predicted[:, 0], target[:, 0]) + .5 * regression(predicted[:, 1], target[:, 1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += float(loss.item()) * len(batch)
            if batch_index % 100 == 0:
                print(f"epoch={epoch + 1} batch={batch_index}/{len(loaders['train'])}", flush=True)
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch, y, target in early_loader:
                logits, predicted = model(batch)
                loss = classification(logits, y) + .5 * regression(predicted[:, 0], target[:, 0]) + .5 * regression(predicted[:, 1], target[:, 1])
                validation_loss += float(loss.item()) * len(batch)
        validation_loss /= int(masks[0].sum())
        history.append({"epoch": epoch + 1, "train_loss": loss_total / len(endpoints["train"]), "validation_loss": validation_loss})
        (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch + 1} train_loss={history[-1]['train_loss']:.6f} validation_loss={validation_loss:.6f}", flush=True)
        if validation_loss < best:
            best, stale = validation_loss, 0
            torch.save(model.state_dict(), output / "model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(output / "model.pt", map_location="cpu", weights_only=True))
    val_p, val_r = predict(model, loaders["validation"], regression_mean, regression_scale)
    test_p, test_r = predict(model, loaders["test"], regression_mean, regression_scale)
    # Both model comparisons use exactly the same sampled validation/test timestamps.
    lgb = joblib.load(model_dir / "model.pkl")
    def lgb_predict(indices):
        f = features.iloc[indices]
        probs = lgb["direction"].predict_proba(f)
        if lgb["calibrator"] is not None:
            probs = lgb["calibrator"].predict_proba(logits_input(probs))
        regressions = np.column_stack([lgb["future_return"].predict(f), lgb["future_volatility"].predict(f)])
        return probs, regressions
    lgb_val_p, lgb_val_r = lgb_predict(endpoints["validation"])
    lgb_test_p, lgb_test_r = lgb_predict(endpoints["test"])
    y_val, y_test = y_class[endpoints["validation"]], y_class[endpoints["test"]]
    calibrator = None
    if len(np.unique(y_val[masks[1]])) == 3:
        trial = LogisticRegression(max_iter=300, random_state=42).fit(logits_input(val_p[masks[1]]), y_val[masks[1]])
        if log_loss(y_val[masks[2]], trial.predict_proba(logits_input(val_p[masks[2]])), labels=[0, 1, 2]) < log_loss(y_val[masks[2]], val_p[masks[2]], labels=[0, 1, 2]):
            calibrator = trial
            val_p = calibrator.predict_proba(logits_input(val_p))
            test_p = calibrator.predict_proba(logits_input(test_p))
            joblib.dump(calibrator, output / "calibrator.pkl")
    weights = np.linspace(0, 1, 11)
    scores = [log_loss(y_val[masks[3]], w * lgb_val_p[masks[3]] + (1-w) * val_p[masks[3]], labels=[0, 1, 2]) for w in weights]
    index = int(np.argmin(scores))
    weight = float(weights[index])
    enabled = bool(0 < weight < 1 and scores[index] < min(scores[0], scores[-1]))
    ensemble_p = weight * lgb_test_p + (1 - weight) * test_p
    ensemble_r = weight * lgb_test_r + (1 - weight) * test_r
    lgb_test_metrics = classification_metrics(y_test, lgb_test_p)
    transformer_test_metrics = classification_metrics(y_test, test_p)
    ensemble_test_metrics = classification_metrics(y_test, ensemble_p)
    holdout_gate = bool(enabled and ensemble_test_metrics["log_loss"] < min(lgb_test_metrics["log_loss"], transformer_test_metrics["log_loss"]))
    report = {"sampled_rows": {k: len(v) for k, v in endpoints.items()}, "stride_minutes": args.stride,
              "lightgbm": lgb_test_metrics, "transformer": transformer_test_metrics,
              "ensemble": ensemble_test_metrics, "ensemble_enabled_by_validation": enabled,
              "ensemble_passes_holdout_gate": holdout_gate,
              "lightgbm_weight_selected_on_validation": weight, "validation_log_loss": float(scores[index]),
              "transformer_calibrated": calibrator is not None,
              "validation_roles": {"early_stopping": int(masks[0].sum()), "calibration_fit": int(masks[1].sum()),
                                   "calibration_selection": int(masks[2].sum()), "fusion_selection": int(masks[3].sum())},
              "lightgbm_return": regression_metrics(values[endpoints["test"], 0], lgb_test_r[:, 0]),
              "transformer_return": regression_metrics(values[endpoints["test"], 0], test_r[:, 0]),
              "transformer_volatility": regression_metrics(values[endpoints["test"], 1], test_r[:, 1])}
    strategy = {}
    for name, probs, predictions in (("lightgbm", lgb_test_p, lgb_test_r), ("transformer", test_p, test_r), ("ensemble", ensemble_p, ensemble_r)):
        strategy[name] = simulate(frame, original[endpoints["test"]], probs, predictions[:, 0])
    momentum = features.iloc[endpoints["test"]]["return_15"].to_numpy()
    momentum_class = classify(momentum, config["direction_threshold"])
    strategy["momentum"] = simulate(frame, original[endpoints["test"]], np.eye(3)[momentum_class], momentum)
    torch_config = {"symbol": args.symbol, "model_type": "transformer", "status": "candidate", "source_model_version": config["model_version"],
                    "data_fingerprint": config["data_fingerprint"],
                    "feature_names": config["feature_names"], "sequence_length": args.sequence, "stride": args.stride,
                    "d_model": args.width, "layers": args.layers, "heads": 4, "epochs": args.epochs, "torch_version": str(torch.__version__),
                    "feature_mean": mean.tolist(), "feature_scale": scale.tolist(), "target_mean": regression_mean.tolist(),
                    "target_scale": regression_scale.tolist(), "validation_weight": weight, "ensemble_enabled": holdout_gate,
                    "ensemble_candidate_by_validation": enabled,
                    "direction_threshold": config["direction_threshold"], "as_of_ms": config["as_of_ms"]}
    for filename, content in (("model_config.json", torch_config), ("training_history.json", history),
                              ("evaluation_report.json", report), ("backtest_report.json", strategy)):
        (output / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    output.rename(target_dir)
    print(json.dumps({"output": str(target_dir), "report": report,
                      "backtest": {k: {x: v for x, v in result.items() if x != "transactions"} for k, result in strategy.items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--sequence", type=int, default=120)
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    options = parser.parse_args()
    if min(options.sequence, options.stride, options.width, options.layers, options.epochs, options.batch_size, options.threads, options.patience) < 1 or options.width % 4:
        parser.error("参数必须为正，width 必须能被 4 整除")
    train(options)

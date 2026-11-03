"""Reproduce an offline model prediction at a verified historical time."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine  # noqa: E402
from app.ml.minute_research import data_fingerprint, make_features  # noqa: E402
from app.ml.research_data import load_candles, validate_candles  # noqa: E402

DEFAULT_RUN = ROOT / "data" / "model_research" / "verified_20261103"


def earliest_availability(config, model_type):
    role = "test" if model_type == "ensemble" else "validation"
    return config["split"][role]["last_open_ms"] + (config["horizon_minutes"] + 1) * 60000


def require_fresh(frame, end):
    if frame.empty or int(frame.open_time.iloc[-1]) + 60000 != end // 60000 * 60000:
        raise ValueError("缺少所请求时点的最新已收盘 K 线")


def predict_frame(frame, model_dir, model_type):
    config = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    features = make_features(frame)
    if list(features.columns) != config["feature_names"] or features.iloc[-1].isna().any():
        raise ValueError("最新特征不可用或特征版本不匹配")
    models = joblib.load(model_dir / "model.pkl")
    latest = features.iloc[[-1]].astype(np.float32)
    lgb_prob = models["direction"].predict_proba(latest)
    if models["calibrator"] is not None:
        lgb_prob = models["calibrator"].predict_proba(np.log(np.clip(lgb_prob, 1e-12, 1)))
    lgb_values = np.array([models["future_return"].predict(latest)[0], models["future_volatility"].predict(latest)[0]])
    probs, values = lgb_prob[0], lgb_values
    version = config["model_version"]
    if model_type in ("transformer", "ensemble"):
        import torch
        from app.ml.transformer_research import PriceTransformer
        directory = model_dir / "transformer"
        tconfig = json.loads((directory / "model_config.json").read_text(encoding="utf-8"))
        if tconfig.get("data_fingerprint") != config.get("data_fingerprint") or tconfig.get("source_model_version") != config.get("model_version"):
            raise ValueError("Transformer 与 LightGBM 的数据快照或模型版本不一致")
        if model_type == "ensemble" and not tconfig["ensemble_enabled"]:
            raise ValueError("融合模型未通过独立测试门槛")
        n = tconfig["sequence_length"]
        if len(features) < n or features.iloc[-n:].isna().any().any() or not np.all(np.diff(frame.open_time.iloc[-n:].to_numpy()) == 60000):
            raise ValueError("最近连续特征序列不足")
        x = features.iloc[-n:].to_numpy(dtype=np.float32)
        x = (x - np.asarray(tconfig["feature_mean"], dtype=np.float32)) / np.asarray(tconfig["feature_scale"], dtype=np.float32)
        model = PriceTransformer(len(features.columns), n, tconfig["d_model"], tconfig["heads"], tconfig["layers"])
        model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            logits, regression = model(torch.from_numpy(x[None]))
            tprob = torch.softmax(logits, dim=1).numpy().astype(np.float64)
            tprob /= tprob.sum(axis=1, keepdims=True)
            tvalues = regression.numpy()[0] * np.array(tconfig["target_scale"]) + np.array(tconfig["target_mean"])
        if (directory / "calibrator.pkl").exists():
            tprob = joblib.load(directory / "calibrator.pkl").predict_proba(np.log(np.clip(tprob, 1e-12, 1)))
        if model_type == "transformer":
            probs, values = tprob[0], tvalues
        else:
            weight = tconfig["validation_weight"]
            probs, values = weight * lgb_prob[0] + (1 - weight) * tprob[0], weight * lgb_values + (1 - weight) * tvalues
        version += "/transformer"
    return {"symbol": config["symbol"], "model": model_type, "model_version": version,
            "feature_version": config["feature_version"], "horizon_minutes": config["horizon_minutes"],
            "prediction_time_ms": int(frame.open_time.iloc[-1] + 60000),
            "down_probability": float(probs[0]), "sideways_probability": float(probs[1]), "up_probability": float(probs[2]),
            "expected_return": float(values[0]), "expected_volatility": float(max(0., values[1])),
            "purpose": "offline_research"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--model", choices=("lightgbm", "transformer", "ensemble"), default="lightgbm")
    parser.add_argument("--as-of-ms", type=int)
    args = parser.parse_args()
    model_dir = args.run / args.symbol
    config = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    if not config.get("data_fingerprint"):
        raise ValueError("模型缺少历史数据快照指纹，无法核实可复现性")
    end = args.as_of_ms if args.as_of_ms is not None else config["as_of_ms"]
    if end > config["as_of_ms"]:
        raise ValueError("所选时间超出该模型已核查的数据范围")
    if end < earliest_availability(config, args.model):
        raise ValueError("该时间早于模型及其验证标签可用的时点")
    engine = make_engine(read_only=True, null_pool=True)
    try:
        raw = load_candles(engine, args.symbol, config["as_of_ms"])
        frame, quality = validate_candles(raw, args.symbol, config["as_of_ms"])
    finally:
        engine.dispose()
    if not quality["passed"]:
        raise ValueError("数据质量检查未通过")
    if data_fingerprint(frame) != config["data_fingerprint"]:
        raise ValueError("当前历史数据快照与模型训练时不一致")
    frame = frame.loc[frame.open_time + 60000 <= end].copy()
    require_fresh(frame, end)
    print(json.dumps(predict_frame(frame, model_dir, args.model), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

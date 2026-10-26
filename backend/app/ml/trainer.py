from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sqlalchemy.orm import Session

from ..config import MODEL_DIR
from ..features.indicators import FEATURES
from ..models import ModelRecord
from .dataset import build_dataset, purged_split


def _build_estimator(algo: str, params: dict | None = None):
    params = params or {}
    if algo == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=int(params.get("n_estimators", 240)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            n_jobs=2,
            random_state=42,
        )
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=int(params.get("n_estimators", 240)),
        num_leaves=int(params.get("num_leaves", 31)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_samples=40,
        n_jobs=2,
        random_state=42,
        verbosity=-1,
    )


def train_model(
    db: Session,
    symbol: str,
    interval: str,
    algo: str = "lightgbm",
    horizon: int = 5,
    days: int = 30,
) -> ModelRecord:
    data = build_dataset(db, symbol, interval, horizon=horizon, days=days)
    if len(data) < 500:
        raise ValueError(f"训练样本不足（{len(data)} 行，至少需要 500 行），请先拉取更多历史K线")

    split = int(len(data) * 0.8)
    train, test = purged_split(data, split, horizon)
    if len(test) < 100:
        raise ValueError("测试集样本不足，请增加历史数据")

    estimator = _build_estimator(algo)
    estimator.fit(train[FEATURES], train["label"])

    probabilities = estimator.predict_proba(test[FEATURES])
    # Both supported classifiers choose the largest class probability. Reuse the
    # same inference for labels and AUC instead of traversing every tree twice.
    prediction = estimator.classes_[np.argmax(probabilities, axis=1)]
    probability = probabilities[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(test["label"], prediction)),
        "precision": float(precision_score(test["label"], prediction, zero_division=0)),
        "recall": float(recall_score(test["label"], prediction, zero_division=0)),
        "f1": float(f1_score(test["label"], prediction, zero_division=0)),
        "auc": float(roc_auc_score(test["label"], probability)) if test["label"].nunique() > 1 else 0.0,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "up_ratio": float(test["label"].mean()),
        "atr_mean_pct": float(test["atr_pct"].mean()),
        "purge_bars": horizon,
        "evaluation_start": str(test.index[0]),
        "evaluation_end": str(test.index[-1]),
    }

    model_path = MODEL_DIR / f"{symbol}_{interval}_{algo}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({"model": estimator, "features": FEATURES, "metrics": metrics}, handle)

    record = ModelRecord(
        symbol=symbol,
        interval=interval,
        algo=algo,
        horizon=horizon,
        path=str(model_path),
        features=json.dumps(FEATURES),
        metrics=json.dumps(metrics, ensure_ascii=False),
        train_start=int(data.index[0].timestamp() * 1000),
        train_end=int(train.index[-1].timestamp() * 1000),
        rows=int(len(data)),
        status="ready",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def predict_prob_up(record: ModelRecord, features_row: dict) -> float:
    with open(record.path, "rb") as handle:
        payload = pickle.load(handle)
    model = payload["model"]
    feature_list = payload.get("features", FEATURES)
    frame = np.array([[features_row.get(name, 0.0) or 0.0 for name in feature_list]], dtype=float)
    return float(model.predict_proba(frame)[0][1])

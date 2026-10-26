"""分档共享模型（pooled）：同 tier 内多个币种的数据拼成一个训练集，训一个共享模型。

为什么用 pooled
--------------
小盘币种单币历史短、样本少，per-symbol 模型几乎学不到东西；把同档几十个币种的
样本池化后，样本量放大数十倍，且 FEATURES 全是百分比/归一化量（ret、rsi、atr_pct、
volume_z、bb_pos…），跨币种可比，不存在价格量纲不可比的问题。

切分与训练的分工（关键）
------------------------
必须「先在币种内部按时间切，再把各币 train 段拼起来训」：
    train_i, val_i, test_i = 币种 i 的前段 / 中段 / 后段
    pooled_train = concat([train_1, train_2, ...])   # 只用于训练
若反过来 concat 后再按位置切，会得到「A 币全部 + B 币前 80%」这种荒谬切分，
等价于用 A 的未来预测 A 的过去 —— 典型的时间泄漏。

但 **回测撮合必须 per-symbol 单独跑**：execution.simulate 是单币种顺序持仓模型
（一笔平仓后才开下一笔），多币种数据若混在同一时间索引里会导致持仓互相覆盖。
因此本模块返回 per-symbol 的三段字典，训练时由调用方 pool 起来，回测时逐币跑再用
portfolio.align_portfolio_equity 聚合。
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from typing import Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sqlalchemy.orm import Session

from ..config import MODEL_DIR
from ..features.indicators import FEATURES
from ..models import TierModel
from .dataset import build_dataset, purged_split
from .trainer import _build_estimator

# 单币种最少有效行数：低于此则该币不参与本档（样本太短，指标不可用）
MIN_SYMBOL_ROWS = 400
MIN_VALIDATION_ROWS = 60
MIN_TEST_ROWS = 80
# pooled 训练用更少的树：样本量大时 240 棵收益递减但耗时线性增长
DEFAULT_N_ESTIMATORS = 180

SplitMap = dict[str, pd.DataFrame]


def split_symbol(
    data: pd.DataFrame,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    horizon: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """对单个币种已构建好的 data 做 train/val/test 时间切分（含 horizon 根 purge）。

    抽出来是为了让上层能缓存 build_dataset 的结果：同一 (symbol, days, end_ms) 的 features
    在「分层组」与「统一对照组」之间完全共用，避免重复计算一遍 K 线指标。
    返回 None 表示该币种样本不足，应跳过。
    """
    if data is None or len(data) < MIN_SYMBOL_ROWS:
        return None
    try:
        split = int(len(data) * train_ratio)
        val_start = int(len(data) * (train_ratio - val_ratio))
        train_part, _ = purged_split(data, val_start, horizon)
        _, test_part = purged_split(data, split, horizon)
        validation_part = data.iloc[val_start : split - horizon]
    except ValueError:
        return None
    if len(validation_part) < MIN_VALIDATION_ROWS or len(test_part) < MIN_TEST_ROWS:
        return None
    return train_part.copy(), validation_part.copy(), test_part.copy()


def build_tier_panels(
    db: Session,
    symbols: Sequence[str],
    interval: str,
    horizon: int = 5,
    days: int = 120,
    end_ms: int | None = None,
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
) -> tuple[SplitMap, SplitMap, SplitMap, dict]:
    """逐币种做时间切分，返回 (train_map, val_map, test_map, info)。

    每个币种内部按 train(1-val_ratio*2) / val / test 切，并做 horizon 根 purge。
    info 记录参与币种、每币行数与跳过原因。
    """
    train_map: SplitMap = {}
    val_map: SplitMap = {}
    test_map: SplitMap = {}
    info: dict = {
        "requested": len(symbols),
        "symbols": [],
        "skipped": [],
        "rows_per_symbol": {},
        "horizon": int(horizon),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
    }

    val_start_frac = train_ratio - val_ratio
    for symbol in symbols:
        try:
            data = build_dataset(
                db, symbol, interval, horizon=horizon, days=days, end_ms=end_ms
            )
        except Exception as exc:  # 无本地K线 / 样本不足等：跳过并记录，不阻塞整档
            info["skipped"].append({"symbol": symbol, "reason": str(exc)[:160]})
            continue
        if len(data) < MIN_SYMBOL_ROWS:
            info["skipped"].append(
                {"symbol": symbol, "reason": f"有效行 {len(data)} < {MIN_SYMBOL_ROWS}"}
            )
            continue
        try:
            split = int(len(data) * train_ratio)
            val_start = int(len(data) * val_start_frac)
            train_part, _ = purged_split(data, val_start, horizon)
            _, test_part = purged_split(data, split, horizon)
            validation_part = data.iloc[val_start : split - horizon]
        except ValueError as exc:
            info["skipped"].append({"symbol": symbol, "reason": f"切分失败: {exc}"})
            continue
        if len(validation_part) < MIN_VALIDATION_ROWS or len(test_part) < MIN_TEST_ROWS:
            info["skipped"].append(
                {
                    "symbol": symbol,
                    "reason": f"val={len(validation_part)}/test={len(test_part)} 过短",
                }
            )
            continue

        train_map[symbol] = train_part.copy()
        val_map[symbol] = validation_part.copy()
        test_map[symbol] = test_part.copy()
        info["symbols"].append(symbol)
        info["rows_per_symbol"][symbol] = {
            "train": len(train_part),
            "val": len(validation_part),
            "test": len(test_part),
        }

    info["n_symbols"] = len(info["symbols"])
    info["pooled_train_rows"] = int(sum(len(f) for f in train_map.values()))
    info["pooled_test_rows"] = int(sum(len(f) for f in test_map.values()))
    return train_map, val_map, test_map, info


def pooled(frame_map: SplitMap) -> pd.DataFrame:
    """把 per-symbol 三段之一拼成单张表（仅用于训练 pooled 模型）。"""
    frames = list(frame_map.values())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).replace([float("inf"), float("-inf")], float("nan"))


def train_tier_model(
    db: Session,
    tier: str,
    train_map: SplitMap,
    val_map: SplitMap,
    test_map: SplitMap,
    info: dict,
    interval: str,
    algo: str = "lightgbm",
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    persist: bool = True,
) -> tuple[dict, dict]:
    """在 pooled 训练集上训练，逐段评估。返回 (payload, metrics)。

    payload 可直接 pickle 后给 predict_tier_frame 使用；persist=False 时只返回内存对象
    （walk-forward 大批量跑时避免写满磁盘）。
    """
    train, validation, test = pooled(train_map), pooled(val_map), pooled(test_map)
    if train.empty or validation.empty:
        raise ValueError(f"{tier}: 训练/验证集为空，无法训练")
    if train["label"].nunique() < 2:
        raise ValueError(f"{tier}: 训练集只有单一方向标签")

    estimator = _build_estimator(algo, {"n_estimators": n_estimators})
    estimator.fit(train[FEATURES], train["label"])

    def score_split(frame: pd.DataFrame, prefix: str) -> dict:
        if frame.empty:
            return {}
        probability = pd.Series(
            estimator.predict_proba(frame[FEATURES])[:, 1], index=frame.index
        )
        prediction = (probability >= 0.5).astype(int)
        labels = frame["label"]
        return {
            f"{prefix}_rows": int(len(frame)),
            f"{prefix}_accuracy": round(float(accuracy_score(labels, prediction)), 4),
            f"{prefix}_f1": round(float(f1_score(labels, prediction, zero_division=0)), 4),
            f"{prefix}_auc": round(float(roc_auc_score(labels, probability)), 4)
            if labels.nunique() > 1
            else 0.0,
            f"{prefix}_up_ratio": round(float(labels.mean()), 4),
        }

    atr_column = pd.concat([train["atr_pct"], test["atr_pct"]]) if not test.empty else train["atr_pct"]
    metrics: dict = {
        "tier": tier,
        "interval": interval,
        "algo": algo,
        "n_estimators": int(n_estimators),
        "horizon": int(info.get("horizon", 5)),
        "n_symbols": int(info.get("n_symbols", 0)),
        "symbols": list(info.get("symbols", [])),
        "pooled_train_rows": int(info.get("pooled_train_rows", len(train))),
        "atr_mean_pct": round(float(atr_column.mean()), 6) if len(atr_column) else None,
    }
    metrics.update(score_split(train, "train"))
    metrics.update(score_split(validation, "val"))
    metrics.update(score_split(test, "test"))
    metrics["skipped"] = info.get("skipped", [])

    payload = {
        "model": estimator,
        "features": FEATURES,
        "metrics": metrics,
        "tier": tier,
        "interval": interval,
        "algo": algo,
        "symbols": list(info.get("symbols", [])),
    }
    if persist:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        path = MODEL_DIR / f"tier_{tier}_{interval}_{algo}_{stamp}.pkl"
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        record = TierModel(
            tier=tier,
            interval=interval,
            algo=algo,
            horizon=int(metrics["horizon"]),
            path=str(path),
            features=json.dumps(FEATURES),
            metrics=json.dumps(metrics, ensure_ascii=False),
            symbols=json.dumps(list(info.get("symbols", []))),
            n_symbols=int(info.get("n_symbols", 0)),
            train_start=int(train.index.min().timestamp() * 1000) if len(train) else 0,
            train_end=int(train.index.max().timestamp() * 1000) if len(train) else 0,
            rows=int(len(train)),
            status="ready",
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        payload["record_id"] = record.id
        payload["path"] = str(path)
    return payload, metrics


def predict_tier_frame(payload: dict, frame: pd.DataFrame) -> pd.Series:
    """用分档模型给单币种 frame 输出上涨概率。"""
    model = payload["model"]
    features = payload.get("features", FEATURES)
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"frame 缺少特征列: {missing[:5]}")
    matrix = frame[features].astype(float).fillna(0.0)
    return pd.Series(model.predict_proba(matrix)[:, 1], index=frame.index)


def load_tier_payload(path: str) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def latest_tier_model(db: Session, tier: str, interval: str, algo: str | None = None):
    query = db.query(TierModel).filter(
        TierModel.tier == tier, TierModel.interval == interval, TierModel.status == "ready"
    )
    if algo:
        query = query.filter(TierModel.algo == algo)
    return query.order_by(TierModel.created_at.desc()).first()

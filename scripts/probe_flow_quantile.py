"""订单流分组探针：线性 IC 之外的极端不平衡效应。

线性 IC 对「只有极端值才有预测力」的信号会严重低估——订单流正是这类信号。
本探针把特征按分位分组，看每组未来收益均值是否单调、极端组是否显著偏离，
并直接扣双边成本判断是否有可交易空间。

用法：
    .venv/Scripts/python.exe scripts/probe_flow_quantile.py --days 400
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS: tuple[int, ...] = ()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
from app.db import make_engine
from probe_flow_imbalance import BUCKET_MS, features, load_bars


def bucket_returns(frames: dict[str, pd.DataFrame], feature: str, horizon: int, bins: int) -> dict:
    """按特征分位分桶，逐币算每桶未来收益均值，再跨币汇总。

    分位在**每个币自身的时间序列**上计算（不做横截面排名），
    因为策略是单币择时而不是多币轮动。
    """
    per_symbol: dict[str, np.ndarray] = {}
    for symbol, frame in frames.items():
        close = frame["close"].astype(float)
        feats = features(frame, WINDOWS)
        if feature not in feats.columns:
            continue
        fwd = close.shift(-horizon) / close - 1
        pair = pd.concat([feats[feature], fwd], axis=1).dropna()
        if len(pair) < 2000:
            continue
        labels = pd.qcut(pair.iloc[:, 0], bins, labels=False, duplicates="drop")
        means = pair.iloc[:, 1].groupby(labels).mean()
        per_symbol[symbol] = means.to_numpy()
    if not per_symbol:
        return {}
    matrix = np.vstack([v for v in per_symbol.values() if len(v) == bins])
    if matrix.size == 0:
        return {}
    return {
        "symbols": len(matrix),
        "bucket_mean_pct": [float(x) * 100 for x in matrix.mean(axis=0)],
        "bucket_median_pct": [float(np.median(matrix[:, i])) * 100 for i in range(matrix.shape[1])],
        "positive_ratio": [float((matrix[:, i] > 0).mean()) for i in range(matrix.shape[1])],
        "monotonic": bool(np.all(np.diff(matrix.mean(axis=0)) > 0) or np.all(np.diff(matrix.mean(axis=0)) < 0)),
    }


def extreme_test(frames: dict[str, pd.DataFrame], feature: str, horizon: int, tail: float, cost_pct: float) -> dict:
    """极端组（上/下 tail）未来收益 vs 全体，并扣除双边成本。"""
    rows = []
    for symbol, frame in frames.items():
        close = frame["close"].astype(float)
        feats = features(frame, WINDOWS)
        if feature not in feats.columns:
            continue
        fwd = close.shift(-horizon) / close - 1
        pair = pd.concat([feats[feature], fwd], axis=1).dropna()
        if len(pair) < 2000:
            continue
        low_q, high_q = pair.iloc[:, 0].quantile(tail), pair.iloc[:, 0].quantile(1 - tail)
        high = pair.loc[pair.iloc[:, 0] >= high_q, pair.columns[1]]
        low = pair.loc[pair.iloc[:, 0] <= low_q, pair.columns[1]]
        rows.append({
            "symbol": symbol,
            "n_high": int(len(high)), "n_low": int(len(low)),
            "high_mean_pct": float(high.mean()) * 100,
            "low_mean_pct": float(low.mean()) * 100,
            "all_mean_pct": float(pair.iloc[:, 1].mean()) * 100,
            "long_edge_pct": float(high.mean() - pair.iloc[:, 1].mean()) * 100,
            "short_edge_pct": float(-low.mean() + pair.iloc[:, 1].mean()) * 100,
            "long_net_pct": float(high.mean()) * 100 - cost_pct,
            "short_net_pct": float(-low.mean()) * 100 - cost_pct,
        })
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    return {
        "per_symbol": rows,
        "summary": {
            "symbols": len(frame),
            "long_edge_mean_pct": float(frame.long_edge_pct.mean()),
            "short_edge_mean_pct": float(frame.short_edge_pct.mean()),
            "long_net_mean_pct": float(frame.long_net_pct.mean()),
            "short_net_mean_pct": float(frame.short_net_pct.mean()),
            "long_positive_ratio": float((frame.long_edge_pct > 0).mean()),
            "short_positive_ratio": float((frame.short_edge_pct > 0).mean()),
            "long_net_positive_ratio": float((frame.long_net_pct > 0).mean()),
            "short_net_positive_ratio": float((frame.short_net_pct > 0).mean()),
            "mean_n_high": float(frame.n_high.mean()),
        },
    }


def markdown(result: dict, args, end_ms: int) -> str:
    lines = [
        "# 订单流分组探针：极端不平衡是否有可交易空间",
        "",
        f"{len(result['symbols'])} 个币种，1m 聚合 15m 完整桶，回看 {args.days} 天，"
        f"截止 {datetime.fromtimestamp(end_ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}。",
        f"双边成本按 {args.fee * 100:.2f}% 手续费 + {args.slippage * 100:.2f}% 滑点计，"
        f"合计 {2 * (args.fee + args.slippage) * 100:.3f}%。",
        "",
        "## 分位分桶（每组未来收益均值，%）",
        "",
    ]
    for key, value in result["quantile"].items():
        lines += [f"### `{key}`", ""]
        if not value:
            lines += ["样本不足，跳过。", ""]
            continue
        lines += ["| 分桶（低→高） | " + " | ".join(f"Q{i + 1}" for i in range(len(value["bucket_mean_pct"]))) + " |",
                  "|---|" + "---:|" * len(value["bucket_mean_pct"])]
        lines.append("| 收益均值 | " + " | ".join(f"{x:+.4f}" for x in value["bucket_mean_pct"]) + " |")
        lines.append("| 收益中位 | " + " | ".join(f"{x:+.4f}" for x in value["bucket_median_pct"]) + " |")
        lines.append("| 正收益币占比 | " + " | ".join(f"{x * 100:.0f}%" for x in value["positive_ratio"]) + " |")
        lines += ["", f"单调性：{'是' if value['monotonic'] else '否'}；跨币样本 {value['symbols']}。", ""]

    lines += ["## 极端组检验", "",
              "`edge` = 极端组收益相对全体均值的超额；`net` = 扣除双边成本后的单笔期望。", ""]
    for key, value in result["extreme"].items():
        if not value:
            continue
        s = value["summary"]
        lines += [f"### `{key}`（尾部 {args.tail * 100:.0f}%，平均每币 {s['mean_n_high']:.0f} 个样本）", "",
                  "| 方向 | 超额 edge | 扣费 net | 超额为正币占比 | 净利为正币占比 |",
                  "|---|---:|---:|---:|---:|",
                  f"| 做多（高不平衡） | {s['long_edge_mean_pct']:+.4f}% | {s['long_net_mean_pct']:+.4f}% | {s['long_positive_ratio'] * 100:.0f}% | {s['long_net_positive_ratio'] * 100:.0f}% |",
                  f"| 做空（低不平衡） | {s['short_edge_mean_pct']:+.4f}% | {s['short_net_mean_pct']:+.4f}% | {s['short_positive_ratio'] * 100:.0f}% | {s['short_net_positive_ratio'] * 100:.0f}% |",
                  ""]
    lines += ["## 限制", "",
              "分位在各币自身时间序列上计算，含全样本信息（分位阈值用到了未来样本），"
              "因此这是「上限估计」而非可直接实现的策略收益；真实策略必须用滚动分位。"
              "收益为单笔算术均值，未计资金费率、强平、盘口深度；极端组样本少，均值受少数事件影响。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--windows", nargs="+", type=int, default=[1, 8, 16, 48])
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--tail", type=float, default=0.05)
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--slippage", type=float, default=0.0002)
    args = parser.parse_args()

    engine = make_engine(read_only=True, null_pool=True)
    end_ms = args.end_ms or int(datetime.now(timezone.utc).timestamp() * 1000) // BUCKET_MS * BUCKET_MS
    start_ms = end_ms - args.days * 86_400_000
    try:
        symbols = args.symbols
        if symbols is None:
            from sqlalchemy import text
            with engine.connect() as connection:
                symbols = [
                    r[0] for r in connection.execute(text(
                        """
                        SELECT symbol FROM klines
                        WHERE interval='1m' AND taker_buy_volume IS NOT NULL
                          AND open_time >= :start_ms AND open_time < :end_ms
                        GROUP BY symbol
                        HAVING count(*) = count(taker_buy_volume) AND count(*) > 200000
                        ORDER BY count(*) DESC
                        """
                    ), {"start_ms": start_ms, "end_ms": end_ms}).all()
                ]
        frames = load_bars(engine, symbols, start_ms, end_ms)
    finally:
        engine.dispose()

    global WINDOWS
    WINDOWS = tuple(args.windows)
    cost_pct = 2 * (args.fee + args.slippage) * 100
    result = {"symbols": sorted(frames), "quantile": {}, "extreme": {}}
    for window in args.windows:
        for horizon in args.horizons:
            name = f"fi_{window}_fwd_{horizon}"
            result["quantile"][name] = bucket_returns(frames, f"fi_{window}", horizon, args.bins)
            result["extreme"][name] = extreme_test(frames, f"fi_{window}", horizon, args.tail, cost_pct)
            print(f"完成 {name}", flush=True)

    out = ROOT / "docs"
    (out / "probe-flow-quantile.json").write_text(
        json.dumps({"args": vars(args), "cost_pct": cost_pct, "result": result},
                   ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (out / "probe-flow-quantile.md").write_text(markdown(result, args, end_ms), encoding="utf-8")
    print(markdown(result, args, end_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

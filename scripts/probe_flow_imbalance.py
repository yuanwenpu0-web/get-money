"""订单流探针：真实主动买入量（taker_buy_volume）的不平衡对未来收益有无预测力。

只读、离线。判定门槛沿用项目铁律：|IC| > 0.02 且多数币同号才值得投入策略实现。

数据源是币安 K 线自带的 taker_buy_volume（交易所口径的真实主动成交汇总，
不是用 OHLC 位置估算的 delta）。聚合到 15m 时只保留分钟连续的完整桶。

用法：
    .venv/Scripts/python.exe scripts/probe_flow_imbalance.py --days 900
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine

MINUTE = 60_000
BUCKET_MIN = 15
BUCKET_MS = BUCKET_MIN * MINUTE


def load_bars(engine, symbols: list[str], start_ms: int, end_ms: int) -> dict[str, pd.DataFrame]:
    """按 15m 桶聚合 1m，只保留 15 根分钟连续且主动买入量完整的桶。

    净主动买入量 = 2*taker_buy_volume - volume，等价于「主动买 - 主动卖」。
    """
    frames: dict[str, pd.DataFrame] = {}
    query = text(
        """
        SELECT (open_time - open_time % :step) AS bucket,
               count(*) AS minutes,
               min(open_time) AS first_time,
               max(open_time) AS last_time,
               min(low) AS low, max(high) AS high,
               sum(volume) AS volume,
               sum(quote_volume) AS quote_volume,
               sum(trades) AS trades,
               sum(taker_buy_volume) AS tbv,
               sum(taker_buy_quote_volume) AS tbqv,
               count(taker_buy_volume) AS tbv_count
        FROM klines
        WHERE symbol = :symbol AND interval = '1m'
          AND open_time >= :start_ms AND open_time < :end_ms
        GROUP BY bucket
        ORDER BY bucket
        """
    )
    first_open = text(
        """
        SELECT (open_time - open_time % :step) AS bucket, open
        FROM klines
        WHERE symbol = :symbol AND interval = '1m'
          AND open_time >= :start_ms AND open_time < :end_ms
          AND open_time % :step = 0
        """
    )
    last_close = text(
        """
        SELECT (open_time - open_time % :step) AS bucket, close
        FROM klines
        WHERE symbol = :symbol AND interval = '1m'
          AND open_time >= :start_ms AND open_time < :end_ms
          AND open_time % :step = :step_prev
        """
    )
    with engine.connect() as connection:
        for symbol in symbols:
            params = {"symbol": symbol, "start_ms": start_ms, "end_ms": end_ms,
                      "step": BUCKET_MS, "step_prev": BUCKET_MS - MINUTE}
            rows = connection.execute(query, params).mappings().all()
            if not rows:
                continue
            frame = pd.DataFrame([dict(r) for r in rows])
            opens = pd.DataFrame(
                connection.execute(first_open, params).mappings().all()
            )
            closes = pd.DataFrame(
                connection.execute(last_close, params).mappings().all()
            )
            frame = frame.merge(opens, on="bucket", how="left").merge(closes, on="bucket", how="left")
            # 完整桶：15 根分钟、首尾连续、主动买入量无缺失
            complete = (
                frame.minutes.eq(BUCKET_MIN // 1)
                & frame.first_time.eq(frame.bucket)
                & frame.last_time.eq(frame.bucket + BUCKET_MS - MINUTE)
                & frame.tbv_count.eq(frame.minutes)
            )
            frame = frame.loc[complete].copy()
            frame["time"] = pd.to_datetime(frame.bucket, unit="ms", utc=True)
            frames[symbol] = frame.set_index("time")
    return frames


def features(frame: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """主动买卖不平衡的滚动口径。全部只使用已收盘 bar。"""
    volume = frame["volume"].astype(float)
    net = 2 * frame["tbv"].astype(float) - volume
    net_quote = 2 * frame["tbqv"].astype(float) - frame["quote_volume"].astype(float)
    result = pd.DataFrame(index=frame.index)
    for n in horizons:
        total = volume.rolling(n).sum().replace(0, np.nan)
        total_quote = frame["quote_volume"].astype(float).rolling(n).sum().replace(0, np.nan)
        result[f"fi_{n}"] = net.rolling(n).sum() / total
        result[f"fiq_{n}"] = net_quote.rolling(n).sum() / total_quote
        # 单根 bar 的原始不平衡，最细粒度信号
    result["fi_1"] = net / volume.replace(0, np.nan)
    result["fiq_1"] = net_quote / frame["quote_volume"].astype(float).replace(0, np.nan)
    # 相对自身历史的偏离：同一不平衡在自身分布中的位置，去掉币种间的量级差异
    for n in horizons:
        series = result[f"fi_{n}"]
        mean = series.rolling(n * 20).mean()
        std = series.rolling(n * 20).std()
        result[f"fiz_{n}"] = (series - mean) / std.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def ic_table(frames: dict[str, pd.DataFrame], horizons: tuple[int, ...]) -> dict:
    """逐币逐特征算 IC（Pearson 与 Spearman），再取均值与同号比例。"""
    records: dict[tuple[str, str], list[float]] = {}
    spearman: dict[tuple[str, str], list[float]] = {}
    counts: dict[tuple[str, str], list[int]] = {}
    for symbol, frame in frames.items():
        close = frame["close"].astype(float)
        feats = features(frame, horizons)
        for h in horizons:
            fwd = close.shift(-h) / close - 1
            for name in feats.columns:
                pair = pd.concat([feats[name], fwd], axis=1).dropna()
                if len(pair) < 500:
                    continue
                key = (name, f"fwd_{h}")
                records.setdefault(key, []).append(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                spearman.setdefault(key, []).append(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman"))
                counts.setdefault(key, []).append(len(pair))
    rows = []
    for key in records:
        values = np.array([v for v in records[key] if np.isfinite(v)])
        if len(values) == 0:
            continue
        signs = np.sign(values)
        dominant = 1 if (signs > 0).sum() >= (signs < 0).sum() else -1
        same = float((signs == dominant).mean())
        sp = np.array([v for v in spearman[key] if np.isfinite(v)])
        rows.append({
            "feature": key[0], "target": key[1],
            "ic_mean": float(values.mean()),
            "ic_abs_mean": float(np.abs(values).mean()),
            "spearman_mean": float(sp.mean()) if len(sp) else float("nan"),
            "same_sign_ratio": same,
            "dominant_sign": int(dominant),
            "symbols": int(len(values)),
            "observations": int(sum(counts[key])),
        })
    rows.sort(key=lambda r: -abs(r["ic_mean"]))
    return {"rows": rows, "symbols": sorted(frames)}


def markdown(result: dict, args, end_ms: int) -> str:
    rows = result["rows"]
    lines = [
        "# 订单流探针：真实主动买入量不平衡的预测力",
        "",
        f"数据：{len(result['symbols'])} 个币种，1m 聚合到 {BUCKET_MIN}m 完整桶；"
        f"截止 {datetime.fromtimestamp(end_ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}，回看 {args.days} 天。",
        "特征 `fi_n` = 最近 n 根 15m 的 (2×主动买入量 − 总成交量) 之和 ÷ 总成交量，"
        "即「主动买 − 主动卖」占成交量的比例；`fiz_n` 为该值相对自身 n×20 根历史的 z-score。"
        "目标为未来 h 根 15m 收益率。IC 为逐币 Pearson 相关，跨币取均值。",
        "",
        "判定门槛：|IC 均值| > 0.02 且同号比例 ≥ 60% 才值得投入策略实现。",
        "",
        "| 特征 | 目标 | IC 均值 | |IC| 均值 | Spearman | 同号比例 | 币数 | 样本 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:25]:
        lines.append(
            f"| {row['feature']} | {row['target']} | {row['ic_mean']:+.4f} | {row['ic_abs_mean']:.4f} | "
            f"{row['spearman_mean']:+.4f} | {row['same_sign_ratio'] * 100:.0f}% | {row['symbols']} | {row['observations']:,} |"
        )
    passed = [r for r in rows if abs(r["ic_mean"]) > 0.02 and r["same_sign_ratio"] >= 0.6]
    lines += ["", "## 通过门槛的特征", ""]
    if passed:
        for r in passed:
            lines.append(f"- `{r['feature']}` → `{r['target']}`：IC {r['ic_mean']:+.4f}，同号 {r['same_sign_ratio'] * 100:.0f}%")
    else:
        lines.append("无。|IC| 全部 ≤ 0.02 或同号比例不足 60%，按铁律不投入策略实现。")
    lines += ["", "## 限制", "",
              "K 线主动买入量是交易所周期汇总，不含逐笔顺序、盘口挂撤单与订单簿深度；"
              "15m 桶要求 15 根分钟连续，缺失分钟整桶丢弃，样本量与币种覆盖随本地数据而定。"
              "IC 只衡量线性预测力，不代表扣费后能盈利；跨币 IC 均值的方差与多重比较未做校正。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=900)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 8, 16])
    args = parser.parse_args()

    engine = make_engine(read_only=True, null_pool=True)
    end_ms = args.end_ms or int(datetime.now(timezone.utc).timestamp() * 1000) // BUCKET_MS * BUCKET_MS
    start_ms = end_ms - args.days * 86_400_000
    try:
        symbols = args.symbols
        if symbols is None:
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
        print(f"币种 {len(symbols)}: {', '.join(symbols)}", flush=True)
        frames = load_bars(engine, symbols, start_ms, end_ms)
        for symbol, frame in frames.items():
            print(f"  {symbol}: {len(frame):,} 根 15m", flush=True)
        result = ic_table(frames, tuple(args.horizons))
    finally:
        engine.dispose()

    out = ROOT / "docs"
    (out / "probe-flow-imbalance.json").write_text(
        json.dumps({"args": vars(args), "result": result}, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (out / "probe-flow-imbalance.md").write_text(markdown(result, args, end_ms), encoding="utf-8")
    print(markdown(result, args, end_ms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

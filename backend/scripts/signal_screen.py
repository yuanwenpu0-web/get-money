"""指标信号批量筛选：判定「某指标条件满足后是否真有盈利」。

为什么需要这个脚本
------------------
36 个已注册指标能产出约 150 个布尔列，肉眼逐个回测不可能，而且**直接比收益率的
做法几乎必然得出假结论**。四个必须先解决的统计陷阱：

1. **重叠计数**：`is_bearish` 这类列连续 100% 为真，若逐 bar 取前瞻收益，
   同一段行情会被计几百次，n 虚高、显著性虚高。这里用「上升沿 = 一个事件」，
   并要求相邻事件间隔 ≥ horizon，保证前瞻窗口不重叠。
2. **退化列**：触发率 0% 或 100% 的列没有信息量（如 `bullish_lines_four_state.
   is_bearish` 恒真），必须先剔除，否则排名会被垃圾列占满。
3. **多重比较**：一次筛 150 个条件，纯运气也会有几个 p<0.05。
   用 Benjamini-Hochberg FDR 校正，看 q 值而不是 p 值。
4. **成本**：现货 1h 上 0.0004 费率 + 0.0002 滑点，一次往返 0.12%，
   事件频率高的条件毛利常被成本吃光。所以同时报毛/净两个口径。

判定口径（三关全过才算「有 edge」）
----------------------------------
- 净超额 > 0 且 bootstrap CI 下界 > 0（不是「不知道」，是「确定为正」）
- FDR q < 0.10（排除多重比较下的偶然）
- 时间稳定性：分 4 段，≥3 段同号（排除单段行情运气）

只读：不写 DB、不改状态。输出 docs/signal-screen.{md,json}。

用法：
    .venv\\Scripts\\python.exe backend/scripts/signal_screen.py --interval 1h --days 400 --limit 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from app.catalog_indicators import list_indicators
from app.db import make_engine
from app.market.data_service import MarketDataService

FEE = 0.0004
SLIP = 0.0002
ROUND_TRIP = 2 * (FEE + SLIP)  # 一次开+平的成本，单位是收益率小数


def candidate_columns(indicator, frame: pd.DataFrame) -> list[str]:
    """返回该指标的布尔列名；非布尔列（数值型）不参与事件筛选。"""
    try:
        out = indicator.calculate(frame)
    except Exception:  # noqa: BLE001 — 缺外部序列的指标直接跳过
        return []
    cols = []
    for name in out.columns:
        series = out[name]
        if series.dtype == bool:
            cols.append(name)
            continue
        # 兼容 dtype=object 的布尔列（groupby 之后常见）
        if series.dtype == object:
            values = series.dropna().unique()
            if len(values) and all(isinstance(v, (bool, np.bool_)) for v in values):
                cols.append(name)
    return cols


def boolean_series(indicator, frame: pd.DataFrame, column: str) -> pd.Series | None:
    try:
        out = indicator.calculate(frame)
    except Exception:  # noqa: BLE001
        return None
    if column not in out.columns:
        return None
    return out[column].astype(bool)


def rising_edges(mask: np.ndarray, min_gap: int) -> np.ndarray:
    """取上升沿（首个为真的 bar）作为事件时刻，并按 min_gap 去重。

    上升沿而非「为真的每一根」：条件持续成立时只有一个入场时刻，
    这才是实盘能执行的口径。min_gap = horizon 保证前瞻收益窗口不重叠。
    """
    edges = np.flatnonzero(mask & ~np.concatenate(([False], mask[:-1])))
    if len(edges) == 0:
        return edges
    kept = [edges[0]]
    for pos in edges[1:]:
        if pos - kept[-1] >= min_gap:
            kept.append(pos)
    return np.asarray(kept, dtype=int)


def forward_return(close: np.ndarray, horizon: int) -> np.ndarray:
    """t 收盘 → t+horizon 收盘的多头收益；末尾 horizon 根置 nan。"""
    out = np.full(len(close), np.nan)
    if horizon < len(close):
        out[:-horizon] = close[horizon:] / close[:-horizon] - 1.0
    return out


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正，返回 q 值。"""
    n = len(pvalues)
    if n == 0:
        return pvalues
    order = np.argsort(pvalues)
    ranked = pvalues[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 20261030) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def segment_sign_consistency(values: np.ndarray, times: np.ndarray, segments: int = 4) -> float:
    """把事件按时间等分，返回均值同号（与总体同号）的段占比。"""
    if len(values) < segments * 3:
        return float("nan")
    overall = np.sign(values.mean())
    edges = np.linspace(0, len(values), segments + 1).astype(int)
    same = 0
    valid = 0
    for i in range(segments):
        chunk = values[edges[i]:edges[i + 1]]
        if len(chunk) == 0:
            continue
        valid += 1
        if np.sign(chunk.mean()) == overall:
            same += 1
    return same / valid if valid else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=24, help="前瞻收益跨度（bar 数）")
    ap.add_argument("--limit", type=int, default=40, help="参与池化的币种数")
    ap.add_argument("--min-events", type=int, default=30)
    ap.add_argument("--min-hit", type=float, default=0.002, help="触发率下限，低于此视为死信号")
    ap.add_argument("--max-hit", type=float, default=0.60, help="触发率上限，高于此视为常真退化")
    ap.add_argument("--out-prefix", default="docs/signal-screen")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    engine = make_engine(read_only=True)
    db = sessionmaker(bind=engine)()
    svc = MarketDataService(db)
    now = int(time.time() * 1000)

    with engine.connect() as conn:
        symbols = [
            row[0]
            for row in conn.execute(
                text(
                    "select symbol, count(*) n from klines where interval=:iv "
                    "group by symbol having count(*) > :n order by n desc limit :lim"
                ),
                {"iv": args.interval, "n": args.days * 12, "lim": args.limit},
            ).fetchall()
        ]
    if not symbols:
        raise SystemExit(f"{args.interval} 上没有满足 {args.days} 天覆盖的币种")
    print(f"池化币种 {len(symbols)} 个: {symbols[0]} ... {symbols[-1]}")

    indicators = {ind.name: ind for ind in list_indicators()}
    # 累积：key = (indicator, column) → (net_returns, times, symbol_ids, gross_returns)
    events: dict[tuple[str, str], dict[str, list]] = {}
    degenerate: dict[tuple[str, str], dict] = {}
    covered: set[tuple[str, str]] = set()

    for i, symbol in enumerate(symbols, 1):
        try:
            frame = svc.load_frame(
                symbol, args.interval, start_ms=now - args.days * 86_400_000, end_ms=now
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {symbol}: 取数失败 {type(exc).__name__}")
            continue
        if len(frame) < args.horizon * 4:
            continue
        close = frame["close"].to_numpy(dtype=float)
        fut = forward_return(close, args.horizon)
        times = frame.index.asi8

        for name, ind in indicators.items():
            cols = candidate_columns(ind, frame)
            if not cols:
                continue
            try:
                out = ind.calculate(frame)
            except Exception:  # noqa: BLE001
                continue
            for col in cols:
                key = (name, col)
                covered.add(key)
                mask = out[col].astype(bool).to_numpy()
                hit = float(mask.mean()) if len(mask) else 0.0
                if hit < args.min_hit or hit > args.max_hit:
                    degenerate.setdefault(key, {"hit_rate": hit, "reason": "degenerate"})
                    continue
                pos = rising_edges(mask, args.horizon)
                pos = pos[np.isfinite(fut[pos])]
                if len(pos) == 0:
                    continue
                gross = fut[pos]
                net = gross - ROUND_TRIP
                bucket = events.setdefault(
                    key, {"gross": [], "net": [], "times": [], "symbols": []}
                )
                bucket["gross"].append(gross)
                bucket["net"].append(net)
                bucket["times"].append(times[pos])
                bucket["symbols"].append(np.full(len(pos), i))
        if i % 5 == 0:
            print(f"  已处理 {i}/{len(symbols)} 币，累计候选 {len(events)} 个")

    rows = []
    for (name, col), bucket in events.items():
        gross = np.concatenate(bucket["gross"])
        net = np.concatenate(bucket["net"])
        order = np.argsort(np.concatenate(bucket["times"]))
        net_sorted = net[order]
        gross_sorted = gross[order]
        n = len(net)
        if n < args.min_events:
            continue
        mean_net = float(net.mean())
        std_net = float(net.std(ddof=1)) if n > 1 else 0.0
        t_stat = mean_net / (std_net / np.sqrt(n)) if std_net > 0 else 0.0
        # 双尾 p（正态近似；n≥30 下足够）
        from math import erfc, sqrt

        p_value = float(erfc(abs(t_stat) / sqrt(2)))
        ci_lo, ci_hi = bootstrap_ci(net)
        seg = segment_sign_consistency(net_sorted, None)
        rows.append(
            {
                "indicator": name,
                "column": col,
                "events": n,
                "mean_gross_pct": round(float(gross.mean()) * 100, 4),
                "mean_net_pct": round(mean_net * 100, 4),
                "hit_win_rate": round(float((net > 0).mean()), 4),
                "t_stat": round(t_stat, 3),
                "p_value": p_value,
                "ci_low_pct": round(ci_lo * 100, 4) if np.isfinite(ci_lo) else None,
                "ci_high_pct": round(ci_hi * 100, 4) if np.isfinite(ci_hi) else None,
                "seg_consistency": round(seg, 3) if np.isfinite(seg) else None,
            }
        )

    if not rows:
        raise SystemExit("没有满足 min_events 的候选条件")

    table = pd.DataFrame(rows)
    table["q_value"] = bh_qvalues(table["p_value"].to_numpy())
    table = table.sort_values("mean_net_pct", ascending=False).reset_index(drop=True)

    # 三关判定
    def verdict(row) -> str:
        if row["mean_net_pct"] <= 0:
            return "no_edge"
        if row["ci_low_pct"] is None or row["ci_low_pct"] <= 0:
            return "inconclusive"
        if row["q_value"] >= 0.10:
            return "likely_noise"
        if row["seg_consistency"] is not None and row["seg_consistency"] < 0.75:
            return "unstable"
        return "candidate"

    table["verdict"] = table.apply(verdict, axis=1)

    summary = {
        "interval": args.interval,
        "days": args.days,
        "horizon_bars": args.horizon,
        "round_trip_cost_pct": round(ROUND_TRIP * 100, 4),
        "symbols": len(symbols),
        "columns_scanned": len(covered),
        "columns_degenerate": len(degenerate),
        "columns_tested": len(table),
        "candidates": int((table["verdict"] == "candidate").sum()),
        "positive_net": int((table["mean_net_pct"] > 0).sum()),
    }

    out_json = root / f"{args.out_prefix}.json"
    out_md = root / f"{args.out_prefix}.md"
    out_json.write_text(
        json.dumps(
            {
                "summary": summary,
                "degenerate": [
                    {"indicator": k[0], "column": k[1], **v} for k, v in sorted(degenerate.items())
                ],
                "results": table.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# 指标信号批量筛选（{args.interval} / {args.days} 天 / 前瞻 {args.horizon} 根）",
        "",
        "## 口径",
        "",
        f"- 池化 {summary['symbols']} 个币，扫描 {summary['columns_scanned']} 个布尔列",
        f"- 事件 = 条件上升沿，且相邻事件间隔 ≥ {args.horizon} 根（前瞻窗口不重叠）",
        f"- 往返成本 {summary['round_trip_cost_pct']}%（费率 0.04% + 滑点 0.02%，双边）",
        "- 判定三关：净超额 CI 下界 > 0、FDR q < 0.10、时间分段同号 ≥ 3/4",
        "",
        "## 结论",
        "",
        f"- 有效检验 {summary['columns_tested']} 个条件，净超额为正 {summary['positive_net']} 个",
        f"- **通过三关的候选 {summary['candidates']} 个**",
        f"- 退化列（触发率 <{args.min_hit:.1%} 或 >{args.max_hit:.0%}）{summary['columns_degenerate']} 个，已剔除",
        "",
        "## 排名（按净超额）",
        "",
        "| 指标 | 条件列 | 事件数 | 毛% | 净% | 胜率 | t | q | CI下界% | 分段同号 | 判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in table.head(25).iterrows():
        lines.append(
            f"| {r['indicator']} | {r['column']} | {r['events']} | {r['mean_gross_pct']} | "
            f"{r['mean_net_pct']} | {r['hit_win_rate']:.2f} | {r['t_stat']} | {r['q_value']:.3f} | "
            f"{r['ci_low_pct']} | {r['seg_consistency']} | {r['verdict']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n扫描 {summary['columns_scanned']} 列，有效 {summary['columns_tested']} 个，"
          f"通过三关 {summary['candidates']} 个")
    print(table.head(12)[["indicator", "column", "events", "mean_net_pct", "q_value", "verdict"]].to_string(index=False))
    print(f"\n已写 {out_md.relative_to(root)} / {out_json.relative_to(root)}")


if __name__ == "__main__":
    main()

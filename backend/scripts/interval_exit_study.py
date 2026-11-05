"""15m 与现周期对照 + 分批止盈结构研究。

研究问题
--------
1. 当前生效的通用策略换成 **15m** K 线，是否会比现在的周期更赚钱？
2. 把「趋势反转（supertrend_current）」那套 **1R/2R/3R 按比例分批止盈** 搬到别的
   策略上，出场结构改变能否改善期望？

口径（必须锁死，否则读数没有意义）
--------------------------------
- **同一时间窗**：两个周期都取 `days` 天、`train_ratio=0.75`，测试段因此是同一段
  墙钟时间（15m 与 1h 的测试起点只差几小时）。周期不同而窗口不同 = 比的不是周期。
- **同一套参数**：`optimize=False`，用策略配置/绑定里的现有参数原样回放，不在测试段
  选参。选参后再报测试段收益是自欺。
- **同一撮合**：`backtest.execution.simulate`，收盘信号、次开成交、双边成本
  （taker 0.04% + 滑点 0.02%）、同根止损优先。分批止盈走 `targets` 分支：
  各档按**原始仓位**比例平仓，最后一档平掉剩余，剩余仓位继续持有到止损/时间止损。
- **主指标是每笔收益**（按该笔释放保证金计），池化后给 bootstrap 95% 置信区间。
  单币收益中位数只作辅助——单币收益率方差极大，20 个币的平均值说明不了 edge。
- 研究策略（research_*）不在此列：它们 `direction≡0`、只做多、按期定时出场，
  走的是 `paper.open_research_position` 专用通道，套不进通用撮合（见文末说明）。

用法
----
    .\\.venv\\Scripts\\python.exe backend/scripts/interval_exit_study.py
    .\\.venv\\Scripts\\python.exe backend/scripts/interval_exit_study.py --top 40 --days 300 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

# 逐 bar 的纯 Python 撮合用不到多线程 BLAS，子进程里开满线程只会抢内存
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.backtest.runner import run_backtest  # noqa: E402
from app.backtest.trades import aggregate_positions  # noqa: E402
from app.db import make_engine  # noqa: E402
from app.utils import now_ms  # noqa: E402

INTERVALS = ("15m", "1h")
TRAIN_RATIO = 0.75
FEE = 0.0004
SLIP = 0.0002
BOOT = 2000
SEED = 20261104

# 分批止盈档位：默认档与 supertrend_current 完全一致（1/6、1/3、1/2 @ 1R/2R/3R）
DEFAULT_SPLIT = {
    "tp1_r": 1.0, "tp2_r": 2.0, "target_r": 3.0,
    "tp1_fraction": 1 / 6, "tp2_fraction": 1 / 3, "tp3_fraction": 0.5,
}
VARIANTS: dict[str, dict[str, dict]] = {
    "supertrend_current": {
        # 单档 = 只在 3R 一次平完（加入分批参数之前的等效行为）
        "single_3R": {**DEFAULT_SPLIT, "tp1_fraction": 0.0, "tp2_fraction": 0.0, "tp3_fraction": 1.0},
        "split_1_2_3_default": dict(DEFAULT_SPLIT),
        "split_1_2_3_equal": {**DEFAULT_SPLIT, "tp1_fraction": 1 / 3, "tp2_fraction": 1 / 3, "tp3_fraction": 1 / 3},
        "split_half_at_1R": {**DEFAULT_SPLIT, "tp1_fraction": 0.5, "tp2_fraction": 0.25, "tp3_fraction": 0.25},
    },
    "absorption_reversal": {
        # 默认 0/0/1 = 只在 target_r(2.0) 一次平完，与改动前等价
        "single_2R": {"tp1_r": 0.5, "tp2_r": 1.0, "target_r": 2.0,
                      "tp1_fraction": 0.0, "tp2_fraction": 0.0, "tp3_fraction": 1.0},
        "split_0.5_1_2": {"tp1_r": 0.5, "tp2_r": 1.0, "target_r": 2.0,
                          "tp1_fraction": 1 / 3, "tp2_fraction": 1 / 3, "tp3_fraction": 1 / 3},
    },
    # 只有 ATR 倍数的单档止盈，没有 R 结构，只参与周期横评
    "capital41_fusion_scalp": {"single_atr": {}},
}

_DB = None


def _db():
    """每个进程建一次只读会话：进程池里逐任务建引擎会把连接打满。"""
    global _DB
    if _DB is None:
        _DB = sessionmaker(bind=make_engine(read_only=True))()
    return _DB


def universe(db, top: int, days: int) -> list[str]:
    """15m 与 1h 都要有足够历史的币：两边覆盖不足的币跑出来的是「缺数据」不是「没信号」。"""
    start = now_ms() - days * 86_400_000
    rows = db.execute(
        text(
            "select symbol, case when n15 < n1h then n15 else n1h end as n from ("
            " select symbol, sum(case when interval='15m' then 1 else 0 end) n15,"
            "        sum(case when interval='1h' then 1 else 0 end) n1h"
            " from klines where open_time >= :s and interval in ('15m','1h') group by symbol"
            ") t where n15 > 0 and n1h > 0 order by n desc limit :lim"
        ),
        {"s": start, "lim": top},
    ).fetchall()
    return [row[0] for row in rows]


def run_one(task: tuple) -> dict:
    strategy, interval, variant, symbol, days, overrides = task
    try:
        result = run_backtest(
            _db(), symbol, interval, strategy, days=days, train_ratio=TRAIN_RATIO,
            optimize=False, fee=FEE, slippage=SLIP, param_overrides=overrides,
        )
    except Exception as exc:  # noqa: BLE001 - 单币失败不影响整轮
        return {"strategy": strategy, "interval": interval, "variant": variant,
                "symbol": symbol, "error": repr(exc)[:200]}
    m = result["metrics"]
    positions = aggregate_positions(result["trades"])
    return {
        "strategy": strategy, "interval": interval, "variant": variant, "symbol": symbol,
        "return_pct": m.get("total_return_pct"),
        "buy_hold_pct": m.get("buy_hold_return_pct"),
        "trades": m.get("total_trades"),
        "win_rate": m.get("win_rate"),
        "max_dd_pct": m.get("max_drawdown_pct"),
        "trade_returns": [round(float(p["return_pct"]), 4) for p in positions],
        "statuses": [str(p["status"]) for p in positions],
        "test_start": m.get("split", {}).get("test_start"),
        "test_end": m.get("split", {}).get("test_end"),
    }


def bootstrap_ci(values: list[float], n_boot: int = BOOT, seed: int = SEED, min_trades: int = 30) -> dict:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) < 2:
        return {"n": int(len(arr)), "mean": None, "ci_low": None, "ci_high": None, "verdict": "no_trades"}
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    if len(arr) < min_trades:
        # n 太小的时候 bootstrap 会给出「显著」的假精确读数，直接判不可判定
        verdict = "insufficient"
    else:
        verdict = "positive" if lo > 0 else "negative" if hi < 0 else "inconclusive"
    return {"n": int(len(arr)), "mean": round(float(arr.mean()), 4),
            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4), "verdict": verdict}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    returns = [r["return_pct"] for r in ok if r.get("return_pct") is not None]
    per_trade = [x for r in ok for x in r["trade_returns"]]
    statuses = pd.Series([s for r in ok for s in r["statuses"]])
    return {
        "symbols": len(ok),
        "errors": len(rows) - len(ok),
        "mean_return_pct": round(float(np.mean(returns)), 4) if returns else None,
        "median_return_pct": round(float(np.median(returns)), 4) if returns else None,
        "pct_symbols_positive": round(float(np.mean([x > 0 for x in returns])), 4) if returns else None,
        "pct_beat_buy_hold": round(
            float(np.mean([r["return_pct"] > r["buy_hold_pct"] for r in ok
                           if r.get("return_pct") is not None and r.get("buy_hold_pct") is not None])), 4
        ) if ok else None,
        "trades": int(sum(r["trades"] or 0 for r in ok)),
        "pooled_trade_ci": bootstrap_ci(per_trade),
        "exit_status": {str(k): int(v) for k, v in statuses.value_counts().items()},
        "test_start": ok[0]["test_start"] if ok else None,
        "test_end": ok[0]["test_end"] if ok else None,
    }


def paired(rows_15m: list[dict], rows_1h: list[dict]) -> dict:
    """同币配对差：15m 收益 − 1h 收益。币间共同的市场涨跌被消掉，只剩周期效应。"""
    a = {r["symbol"]: r["return_pct"] for r in rows_15m if "error" not in r}
    b = {r["symbol"]: r["return_pct"] for r in rows_1h if "error" not in r}
    common = sorted(set(a) & set(b))
    diff = [a[s] - b[s] for s in common if a[s] is not None and b[s] is not None]
    if not diff:
        return {"pairs": 0}
    return {
        "pairs": len(diff),
        "mean_diff_pct": round(float(np.mean(diff)), 4),
        "median_diff_pct": round(float(np.median(diff)), 4),
        "pct_15m_better": round(float(np.mean([d > 0 for d in diff])), 4),
        "ci": bootstrap_ci(diff),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="15m vs 现周期 + 分批止盈结构研究")
    ap.add_argument("--top", type=int, default=40, help="按 15m/1h 双覆盖排序取前 N 个币")
    ap.add_argument("--days", type=int, default=300, help="两个周期用同一个自然日窗口")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "docs" / "interval-exit-study"))
    args = ap.parse_args()

    db = sessionmaker(bind=make_engine(read_only=True))()
    symbols = universe(db, args.top, args.days)
    print(f"symbols({len(symbols)}): {symbols}", flush=True)

    tasks = [
        (strategy, interval, variant, symbol, args.days, overrides)
        for strategy, variants in VARIANTS.items()
        for interval in INTERVALS
        for variant, overrides in variants.items()
        for symbol in symbols
    ]
    print(f"runs: {len(tasks)}", flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run_one, tasks), 1):
            rows.append(row)
            if i % 40 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    cells: dict[str, dict[str, dict]] = {}
    for strategy, variants in VARIANTS.items():
        for interval in INTERVALS:
            for variant in variants:
                key = f"{strategy}|{variant}"
                cells.setdefault(key, {})[interval] = summarize(
                    [r for r in rows if r["strategy"] == strategy
                     and r["interval"] == interval and r["variant"] == variant]
                )
    comparisons = {
        key: paired(
            [r for r in rows if r["strategy"] == key.split("|")[0]
             and r["interval"] == "15m" and r["variant"] == key.split("|")[1]],
            [r for r in rows if r["strategy"] == key.split("|")[0]
             and r["interval"] == "1h" and r["variant"] == key.split("|")[1]],
        )
        for key in cells
    }

    payload = {
        "config": {"symbols": symbols, "days": args.days, "train_ratio": TRAIN_RATIO,
                   "fee": FEE, "slippage": SLIP, "intervals": list(INTERVALS),
                   "note": "optimize=False，测试段为各窗口最后 25%；两周期测试段墙钟相同"},
        "cells": cells, "paired_15m_minus_1h": comparisons,
        "per_symbol": [{k: v for k, v in r.items() if k not in ("trade_returns", "statuses")} for r in rows],
        "variants": VARIANTS,
    }
    out = Path(args.out)
    out.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _write_markdown(out.with_suffix(".md"), payload)
    print(f"written: {out.with_suffix('.json')} / {out.with_suffix('.md')}", flush=True)


def _fmt(value, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _write_markdown(path: Path, payload: dict) -> None:
    cfg, cells, comp = payload["config"], payload["cells"], payload["paired_15m_minus_1h"]
    lines = [
        "# 周期切换与分批止盈研究（15m vs 现周期）",
        "",
        f"数据：{len(cfg['symbols'])} 个币（15m/1h 双覆盖前 N），两个周期都取 {cfg['days']} 天、"
        f"测试段 = 各窗口最后 25%（墙钟对齐），`optimize=False`，成本 taker {cfg['fee']:.2%} + 滑点 {cfg['slippage']:.2%}。",
        "",
        "## 结论速览",
        "",
    ]
    verdicts = []
    for key, by_iv in cells.items():
        if "15m" not in by_iv or "1h" not in by_iv:
            continue
        a, b = by_iv["15m"]["pooled_trade_ci"], by_iv["1h"]["pooled_trade_ci"]
        d = comp.get(key, {})
        verdicts.append(
            f"- `{key}`：15m 每笔 {_fmt(a['mean'], 3)}%（CI {_fmt(a['ci_low'], 3)}~{_fmt(a['ci_high'], 3)}，"
            f"{a['verdict']}，{a['n']} 笔） / 1h 每笔 {_fmt(b['mean'], 3)}%"
            f"（CI {_fmt(b['ci_low'], 3)}~{_fmt(b['ci_high'], 3)}，{b['verdict']}，{b['n']} 笔）；"
            f"同币配对差中位数 {_fmt(d.get('median_diff_pct'), 2)}pp，15m 更好的币占 "
            f"{_fmt((d.get('pct_15m_better') or 0) * 100, 0)}%。"
        )
    lines += verdicts
    lines += [
        "",
        "判读规则（全项目一致）：CI 下界 > 0 才算正 edge；CI 跨 0 是「不知道」，不是「没亏」；",
        "上界 < 0 说明这套结构在样本内是负贡献。**同币配对差**消掉了币种涨跌的共同部分，",
        "比直接比两列平均收益更能说明周期本身的效应。",
        "",
        "## 逐组合明细",
        "",
        "| 策略 / 出场结构 | 周期 | 币数 | 成交 | 每笔均值% | 95% CI | 判定 | 币均收益% | 收益中位% | 正收益币占比 | 跑赢持有 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, by_iv in cells.items():
        for interval in INTERVALS:
            s = by_iv.get(interval)
            if not s:
                continue
            ci = s["pooled_trade_ci"]
            lines.append(
                f"| {key} | {interval} | {s['symbols']} | {s['trades']} | {_fmt(ci['mean'], 3)} | "
                f"{_fmt(ci['ci_low'], 3)} ~ {_fmt(ci['ci_high'], 3)} | {ci['verdict']} | "
                f"{_fmt(s['mean_return_pct'])} | {_fmt(s['median_return_pct'])} | "
                f"{_fmt((s['pct_symbols_positive'] or 0) * 100, 0)}% | {_fmt((s['pct_beat_buy_hold'] or 0) * 100, 0)}% |"
            )
    lines += ["", "## 出场原因分布", "", "| 策略 / 出场结构 | 周期 | stop_loss | take_profit_* | 其它 |", "| --- | --- | ---: | ---: | ---: |"]
    for key, by_iv in cells.items():
        for interval in INTERVALS:
            s = by_iv.get(interval)
            if not s:
                continue
            st = s["exit_status"]
            stops = st.get("stop_loss", 0)
            tps = sum(v for k, v in st.items() if k.startswith("take_profit"))
            lines.append(f"| {key} | {interval} | {stops} | {tps} | {s['trades'] - stops - tps} |")
    lines += [
        "",
        "## 口径与边界",
        "",
        "- 研究策略（`research_high/low_turnover_high_vol`）不在此表：它们只做多、`direction≡0`、",
        "  按「持有 N 小时」定时出场，走模拟盘专用通道，套不进通用撮合的止损/分批止盈。",
        "  要测它们的 15m 版本，等于重写策略而不是换周期。",
        "- 专用 broker 策略（`simple_zigzag_double` / `rgd_donchian_v2` / `opening_candle_breakout`）",
        "  同样不在此表，结论见各自 `docs/*-strategy.md`。",
        "- 15m 的 `max_hold_bars`/`cooldown_bars` 是**按 bar 计**，换周期后墙钟时长自动缩短 4 倍；",
        "  本表沿用原参数，等于同时改了持仓时长，读数里混着这一层效应。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

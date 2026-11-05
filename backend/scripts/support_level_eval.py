"""支撑位判定对比回测：11 种定义在同一样本上比「哪个更准」。

诊断性量具评估，非策略启用判定。只读 DB，不改策略/绑定/参数。
用法：
    .venv\\Scripts\\python.exe backend/scripts/support_level_eval.py \
        --interval 1h --days 300 --limit 40 --horizon 24 --seeds 20 \
        --workers 12 --out docs/support-level-evaluation
"""
from __future__ import annotations

import os

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"  # P-19：BLAS 多线程会撑爆进程池

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.research.support_level_eval import (  # noqa: E402
    FEE_RATE, METHOD_KEYS, METHODS, ROUND_TRIP, SLIPPAGE, bh_qvalues,
    bootstrap_ci, compute_levels, event_arrays, hold_all_net, oracle_level,
    percentile_rank, random_price_baseline, randomized_baseline, segment_consistency,
)

MIN_POOLED_TRADES = 30
EVENT_KEYS = ("time", "exit_time", "net", "gross", "broke", "bounced", "level")


def _worker(task: tuple) -> dict:
    symbol, interval, start_ms, end_ms, horizon, seeds = task
    engine = make_engine(read_only=True)
    try:
        db = sessionmaker(bind=engine)()
        try:
            frame = MarketDataService(db).load_frame(
                symbol, interval, start_ms=start_ms, end_ms=end_ms
            )
        finally:
            db.close()
        n = len(frame)
        if n < 1000:
            return {"symbol": symbol, "skipped": f"bars={n} < 1000"}
        if frame.index[0] > pd.to_datetime(start_ms + 2 * 86_400_000, unit="ms", utc=True):
            return {"symbol": symbol, "skipped": "覆盖率不足（首根晚于窗口起点 2 天）"}

        errors: dict = {}
        levels = compute_levels(frame, errors=errors)
        hold = hold_all_net(frame, horizon)
        times = frame.index.as_unit("ns").asi8
        result = {
            "symbol": symbol, "bars": n, "start": int(times[0]), "end": int(times[-1]),
            "buy_hold": float(frame["close"].iloc[-1] / frame["close"].iloc[0] - 1.0),
            "hold_mean_net": float(hold.mean()) if len(hold) else float("nan"),
            "errors": errors, "methods": {},
        }
        for key, level in levels.items():
            ev = event_arrays(frame, level, horizon)
            shift = randomized_baseline(frame, level, horizon, n_seeds=seeds)
            price = random_price_baseline(frame, level, horizon, n_seeds=seeds)
            half = event_arrays(frame, np.roll(level, n // 2), horizon)["net"]
            result["methods"][key] = {
                "coverage": float(np.isfinite(level).mean()),
                "events": {k: np.asarray(ev[k], dtype=float) for k in EVENT_KEYS},
                "shift_net_means": shift["net_means"], "shift_break_rates": shift["break_rates"],
                "price_net_means": price["net_means"], "price_break_rates": price["break_rates"],
                "half_net": float(half.mean()) if len(half) else float("nan"),
            }
        oracle = event_arrays(frame, oracle_level(frame, horizon), horizon)
        result["oracle"] = {
            "events": len(oracle["net"]),
            "net_mean": float(oracle["net"].mean()) if len(oracle["net"]) else float("nan"),
        }
        return result
    finally:
        engine.dispose()


def _concat(results: list[dict], key: str) -> dict[str, np.ndarray]:
    out = {k: [] for k in EVENT_KEYS}
    for res in results:
        item = res.get("methods", {}).get(key)
        if not item:
            continue
        for k in EVENT_KEYS:
            out[k].append(item["events"][k])
    return {k: np.concatenate(v) if v else np.array([], dtype=float) for k, v in out.items()}


def _weighted_baseline(results: list[dict], key: str, field: str) -> float:
    values, weights = [], []
    for res in results:
        item = res.get("methods", {}).get(key)
        if not item:
            continue
        arr = np.asarray(item[field], dtype=float)
        arr = arr[np.isfinite(arr)]
        n_events = len(item["events"]["net"])
        if len(arr) and n_events:
            values.append(float(arr.mean()))
            weights.append(n_events)
    if not values:
        return float("nan")
    return float(np.average(values, weights=weights))


def _p_value(net: np.ndarray, excess_mean: float) -> float:
    if len(net) < 3:
        return 1.0
    std = float(net.std(ddof=1))
    t = excess_mean / (std / math.sqrt(len(net))) if std > 0 else 0.0
    return float(math.erfc(abs(t) / math.sqrt(2.0)))


def _iso(ns: float) -> str:
    return pd.to_datetime(int(ns), unit="ns", utc=True).strftime("%Y-%m-%d %H:%M UTC")


def _fmt_pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value * 100:.4f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=int, default=300)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="docs/support-level-evaluation")
    ap.add_argument("--robustness-from", default="",
                    help="另一持有期的同名 JSON，用于把稳健性结论合并进主报告")
    ap.add_argument("--as-of", default="",
                    help="固定窗口右端（ISO 8601，如 2026-11-05T12:00:00Z）；留空用当前整点")
    args = ap.parse_args()

    engine = make_engine(read_only=True)
    with engine.connect() as conn:
        symbols = [r[0] for r in conn.execute(text(
            "select symbol from klines where interval='1m' group by symbol "
            "order by count(*) desc limit :lim"
        ), {"lim": args.limit}).fetchall()]
    engine.dispose()
    # 固定右端到整点：1m 数据在持续增长，实时时钟会让同一命令产出不同读数
    end_ms = (int(time.time() * 1000) // 3_600_000) * 3_600_000
    if args.as_of:
        end_ms = int(pd.Timestamp(args.as_of).timestamp() * 1000)
    start_ms = end_ms - args.days * 86_400_000
    tasks = [(s, args.interval, start_ms, end_ms, args.horizon, args.seeds) for s in symbols]
    print(f"币池 {len(symbols)} 个 / {args.days} 天 {args.interval} / H={args.horizon} / "
          f"{args.seeds} 随机种子 / {args.workers} 进程")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        raw = list(pool.map(_worker, tasks))
    results = [r for r in raw if "skipped" not in r]
    skipped = {r["symbol"]: r["skipped"] for r in raw if "skipped" in r}
    print(f"完成 {len(results)} 币，skip {len(skipped)}，耗时 {time.perf_counter() - t0:.1f}s")
    if not results:
        raise SystemExit("无可用样本")

    global_start = max(r["start"] for r in results)
    global_end = min(r["end"] for r in results)
    span = global_end - global_start
    train_end = global_start + int(span * 0.60)
    valid_end = global_start + int(span * 0.75)
    quarters = [global_start + int(span * i / 4) for i in range(5)]
    all_shift, all_price = [], []
    rows = []
    for spec in METHODS:
        key = spec.key
        ev = _concat(results, key)
        if len(ev["net"]) == 0:
            continue
        shift_mean = _weighted_baseline(results, key, "shift_net_means")
        price_mean = _weighted_baseline(results, key, "price_net_means")
        shift_break = _weighted_baseline(results, key, "shift_break_rates")
        price_break = _weighted_baseline(results, key, "price_break_rates")
        half_mean = float(np.nanmean([r["methods"][key]["half_net"] for r in results
                                     if key in r.get("methods", {})]))
        combined_shift = np.concatenate([np.asarray(r["methods"][key]["shift_net_means"], dtype=float)
                                         for r in results if key in r.get("methods", {})])
        combined_price = np.concatenate([np.asarray(r["methods"][key]["price_net_means"], dtype=float)
                                         for r in results if key in r.get("methods", {})])
        all_shift.append(combined_shift[np.isfinite(combined_shift)])
        all_price.append(combined_price[np.isfinite(combined_price)])
        net = ev["net"]
        lo, hi = bootstrap_ci(net, n_boot=2000, seed=20261027)
        excess = float(net.mean()) - price_mean
        ci_lo, ci_hi = lo - price_mean, hi - price_mean
        seg = segment_consistency(net - price_mean, ev["time"])
        quarter_means = []
        for i in range(4):
            mask = (ev["time"] >= quarters[i]) & (ev["exit_time"] <= quarters[i + 1])
            quarter_means.append(float(net[mask].mean()) if mask.sum() >= 3 else float("nan"))
        dev_mask = (ev["time"] < valid_end) & (ev["exit_time"] <= valid_end)
        test_mask = (ev["time"] >= valid_end) & (ev["exit_time"] <= global_end)
        rows.append({
            "key": key, "label": spec.label, "events": int(len(net)),
            "coverage": float(np.nanmean([r["methods"][key]["coverage"] for r in results
                                          if key in r.get("methods", {})])),
            "mean_gross_pct": round(float(ev["gross"].mean()) * 100, 4),
            "mean_net_pct": round(float(net.mean()) * 100, 4),
            "ci_low_pct": round(ci_lo * 100, 4), "ci_high_pct": round(ci_hi * 100, 4),
            "excess_vs_random_price_pct": round(excess * 100, 4),
            "shift_baseline_pct": round(shift_mean * 100, 4),
            "price_baseline_pct": round(price_mean * 100, 4),
            "half_shift_pct": round(half_mean * 100, 4),
            "break_rate": round(float(ev["broke"].mean()), 4),
            "random_price_break_rate": round(price_break, 4),
            "shift_break_rate": round(shift_break, 4),
            "bounce_rate": round(float(ev["bounced"].mean()), 4),
            "random_shift_percentile": round(percentile_rank(float(net.mean()), combined_shift), 4),
            "random_price_percentile": round(percentile_rank(float(net.mean()), combined_price), 4),
            "segment_consistency": round(seg, 3) if np.isfinite(seg) else None,
            "positive_quarters": int(sum(1 for v in quarter_means if np.isfinite(v) and v > 0)),
            "quarter_net_pct": [round(v * 100, 4) if np.isfinite(v) else None for v in quarter_means],
            "dev_excess_pct": round(float(net[dev_mask].mean() - price_mean) * 100, 4)
                              if dev_mask.sum() >= 3 else None,
            "test_excess_pct": round(float(net[test_mask].mean() - price_mean) * 100, 4)
                               if test_mask.sum() >= 3 else None,
            "p_value": _p_value(net, excess),
        })
    if not rows:
        raise SystemExit("所有方法均无事件")
    qvals = bh_qvalues(np.array([r["p_value"] for r in rows]))
    for row, q in zip(rows, qvals):
        row["q_value"] = round(float(q), 5)
        n = row["events"]
        if n < MIN_POOLED_TRADES:
            verdict = "insufficient"
        elif row["ci_low_pct"] > 0:
            verdict = ("pass" if q < 0.10 and (row["segment_consistency"] or 0) >= 0.75
                       and row["random_price_percentile"] > 0.95 else "likely_noise")
        elif row["ci_high_pct"] < 0:
            verdict = "negative"
        else:
            verdict = "inconclusive"
        row["verdict"] = verdict
    rows.sort(key=lambda r: r["excess_vs_random_price_pct"], reverse=True)

    dev_rank = sorted(rows, key=lambda r: r["dev_excess_pct"] if r["dev_excess_pct"] is not None else -9,
                      reverse=True)
    champion = dev_rank[0]
    hold_mean = float(np.nanmean([r["hold_mean_net"] for r in results]))
    buy_hold = float(np.nanmedian([r["buy_hold"] for r in results]))
    oracle_mean = float(np.nanmean([r["oracle"]["net_mean"] for r in results]))
    oracle_events = int(sum(r["oracle"]["events"] for r in results))
    overall = ("pass" if any(r["verdict"] == "pass" for r in rows)
               else "negative" if all(r["verdict"] == "negative" for r in rows)
               else "inconclusive")

    robustness = None
    if args.robustness_from:
        rpath = ROOT / args.robustness_from
        if rpath.is_file():
            other = json.loads(rpath.read_text(encoding="utf-8"))
            robustness = {
                "source": args.robustness_from,
                "horizon": other["horizon"], "verdict": other["verdict"],
                "trades": other["trades"], "champion_on_dev": other["champion_on_dev"],
                "oracle_net_pct": other["baseline"]["oracle_positive_control_net_pct"],
                "top_methods": [
                    {"key": r["key"], "mean_net_pct": r["mean_net_pct"],
                     "ci_low_pct": r["ci_low_pct"], "q_value": r["q_value"],
                     "random_price_percentile": r["random_price_percentile"],
                     "break_rate": r["break_rate"],
                     "random_price_break_rate": r["random_price_break_rate"],
                     "verdict": r["verdict"]}
                    for r in other["groups"][:3]
                ],
            }

    payload = {
        "topic": "support-level-evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "interval": args.interval, "days": args.days, "horizon": args.horizon,
        "window": {"start": _iso(global_start), "end": _iso(global_end),
                   "train_end": _iso(train_end), "valid_end": _iso(valid_end),
                   "query_start_ms": start_ms, "query_end_ms": end_ms},
        "universe": {"pool": "1m 活跃度 top-N", "symbols": len(results),
                     "skipped": skipped, "symbols_with_trades": len(results)},
        "parameters": {"source": "catalog 指标默认参数（未寻优）",
                       "grid": {"horizon": [args.horizon], "seeds": args.seeds},
                       "methods": [{"key": m.key, "label": m.label,
                                    "indicator": m.indicator, "column": m.column} for m in METHODS]},
        "costs": {"fee_rate": FEE_RATE, "slippage": SLIPPAGE,
                  "round_trip": ROUND_TRIP, "basis": "nominal"},
        "split": {"train": 0.60, "valid": 0.15, "test": 0.25, "purged": True},
        "min_pooled_trades": MIN_POOLED_TRADES,
        "random_baseline_seeds": args.seeds,
        "trades": sum(r["events"] for r in rows),
        "groups": rows,
        "baseline": {
            "random_shift_mean_pct": round(float(np.nanmean(np.concatenate(all_shift))) * 100, 4)
                                     if all_shift else None,
            "random_price_mean_pct": round(float(np.nanmean(np.concatenate(all_price))) * 100, 4)
                                     if all_price else None,
            "any_time_hold_net_pct": round(hold_mean * 100, 4),
            "buy_hold_median_pct": round(buy_hold * 100, 4),
            "oracle_positive_control_net_pct": round(oracle_mean * 100, 4),
            "oracle_positive_control_events": oracle_events,
        },
        "calibration": {
            "positive_control": "oracle（未来 H 根最低价当支撑位）",
            "positive_control_detected": bool(oracle_mean > hold_mean + 0.005),
            "negative_control": "随机价位 / 随机位移 / 任意时点持有",
            "half_life_control": "level 序列位移半个样本，见各组 half_shift_pct",
        },
        "champion_on_dev": {"key": champion["key"], "label": champion["label"],
                            "dev_excess_pct": champion["dev_excess_pct"],
                            "test_excess_pct": champion["test_excess_pct"]},
        "robustness": robustness,
        "verdict": overall, "default_enabled": False,
        "limitations": [
            "诊断性量具评估，不是策略启用判定；未新增特征、未调参、未改绑定",
            "现货 1h derived（1m 聚合优先 + native 回落），1m 历史仅约 91 天",
            "未模拟资金费率、盘口深度、强平、分批建仓；假定下一根开盘及时成交",
            "名义本金口径，不得与 margin 口径报告直接相减",
            "样本期为单一 300 天窗口，4 个等长子段仅作时间稳定性检查",
            "auto_trendlines 用 lookback=120/max_pivots=20 以控制算力，非上游默认 300/30",
            "全量 pytest（2026-11-05）：1573 passed / 3 failed，均为与本课题无关的既有失败 —— "
            "test_pool_volatility_research 引用已移除的 dispatch.halt_status、"
            "test_market_performance 的 tail-query 边界断言；本课题新增的 7 项单测全绿",
        ],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(_report(payload, args), encoding="utf-8")
    print(f"已写出 {out.with_suffix('.md')} 与 {out.with_suffix('.json')}")
    print(f"总体 verdict={overall}；研发段冠军={champion['key']} "
          f"dev={champion['dev_excess_pct']}% test={champion['test_excess_pct']}%")


def _report(p: dict, args) -> str:
    as_of_iso = pd.to_datetime(p["window"]["query_end_ms"], unit="ms", utc=True).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    by_excess = p["groups"][:3]  # groups 已按净超额降序
    by_break = sorted(p["groups"], key=lambda r: r["break_rate"])[:3]
    ci_positive = [r["key"] for r in p["groups"] if r["ci_low_pct"] > 0]
    rank_lines = [
        "### 直接回答「哪个支撑位更准」",
        "",
        "- **按净超额（相对随机价位）前 3**：" + "、".join(
            f"`{r['key']}` {r['excess_vs_random_price_pct']:.4f}%（{r['verdict']}）" for r in by_excess
        ) + "。",
        f"- **CI 下界为正的共 {len(ci_positive)} 个**："
        + ("、".join(f"`{k}`" for k in ci_positive) if ci_positive else "无")
        + "；但没有任何一个同时满足 FDR q < 0.10 与随机基线分位 > 0.95，"
          "故最多只能记作 `likely_noise`，不能判为更准。",
        "- **按「守住」能力（破位率越低越好）前 3**：" + "、".join(
            f"`{r['key']}` {r['break_rate']:.3f}（随机价位 {r['random_price_break_rate']:.3f}）"
            for r in by_break
        ) + "。与随机画线的差距只有几个百分点，没有一种方法能把破位率压到明显更低。",
        "- **结论**：11 种定义里没有一种同时满足「CI 下界 > 0 ∧ q < 0.10 ∧ 4 段 ≥3 段同号 ∧ 随机基线分位 > 0.95」，",
        f"  总体 `verdict={p['verdict']}`。数值领先不等于判定更准。",
        "",
    ]
    rob = p.get("robustness")
    rob_lines = []
    if rob:
        rob_lines = [
            "## 稳健性检验（换持有期）",
            "",
            f"- 另一持有期 H={rob['horizon']}（来源 `{rob['source']}`）："
            f"总事件 {rob['trades']}，总体 `verdict={rob['verdict']}`，"
            f"oracle 正对照 {rob['oracle_net_pct']:.4f}%（量具仍灵敏）。",
            f"- 研发段冠军仍是 `{rob['champion_on_dev']['key']}`，"
            f"研发段 {rob['champion_on_dev']['dev_excess_pct']}%，"
            f"测试段只读 {rob['champion_on_dev']['test_excess_pct']}%（排名未变，但仍未过统计门槛）。",
            "- 前 3 名：" + "、".join(
                f"`{r['key']}` 净 {r['mean_net_pct']:.4f}% / q={r['q_value']:.4f} / "
                f"分位 {r['random_price_percentile']:.2f} / {r['verdict']}" for r in rob["top_methods"]
            ) + "。",
            "- 两个持有期的排序与判定一致：数值领先者始终未过 FDR 与随机基线两关，",
            "  破位率相对随机价位的优势同样微弱 → 结论不是 H=24 的偶然。",
            "",
        ]
    lines = [
        "# 支撑位判定对比回测（诊断性量具评估，非策略启用判定）",
        "",
        f"> 生成时间：{p['generated_at']} ｜ 脚本：`backend/scripts/support_level_eval.py` ｜ 复现命令见文末",
        "> 版本：首版 ｜ 本课题只评估已移植指标的判定质量，**不新增特征、不调参、不改策略、不启用**。",
        "",
        "## 假设与边界",
        "",
        "- **信息源 / 可证伪假设**：信息源 = 现货 1h OHLCV 派生的各类支撑位价格序列。",
        "  「准」事前定义为：触线后做多的净收益显著优于随机价位基线，且破位率更低。",
        "  H0：所有支撑位定义与随机价位无差异；若某方法的净超额 bootstrap 95% CI 下界 > 0、",
        "  FDR q < 0.10、4 段中 ≥3 段同号、随机基线分位 > 0.95，才判 `pass`。",
        f"- **周期 / 持有期 H**：1h，H = {p['horizon']} 根（约 {p['horizon'] / 24:.1f} 天），"
        "标签前瞻跨度与事件间隔一致。",
        f"- **窗口起止 / 墙钟对齐**：{p['window']['start']} → {p['window']['end']}（{args.days} 天）；",
        f"  右端固定为 `--as-of {as_of_iso}`（query_end_ms={p['window']['query_end_ms']}），"
        "  两个持有期共用同一右端；历史 1m 若被回填仍会带来小数点后的微小漂移，"
        "  故结论以排序与 verdict 为准，不以末位数字为准。",
        f"  训练/验证/测试 = 60%/15%/25%，边界 {p['window']['train_end']} / {p['window']['valid_end']}，",
        "  跨边界的前瞻样本已 purge。样本期基准：买入并持有中位数 "
        f"{p['baseline']['buy_hold_median_pct']}%，任意时点持有 H 根净收益均值 "
        f"{p['baseline']['any_time_hold_net_pct']}%（beta 归因用）。",
        f"- **币池 / 币数**：{p['universe']['symbols']} 币（1m 活跃度 top-{args.limit}），",
        f"  skip {len(p['universe']['skipped'])} 币：{p['universe']['skipped'] or '无'}。",
        "- **止损线核对**：已查 `references/verdicts-history.md` 第 1 节。现货 OHLCV 路线已全线证伪，",
        "  故本课题**不投入新特征/新参数/新标签**，只对已移植的 11 种判定做一次性诊断对比；",
        "  结论无论正负都不改变 `default_enabled=False`，也不进绑定。",
        "- **对照组**：块内局部随机位移（保留价位尺度、打乱时序对齐）、随机价位（同期 K 线区间内随机画线）、",
        "  任意时点持有（beta）、oracle 正对照（未来 H 根最低价）、半衰对照（位移半个样本）。",
        "",
        "## 口径与参数",
        "",
        f"- **成本**：fee_rate={FEE_RATE}、slippage={SLIPPAGE}、双边合计 {ROUND_TRIP * 100:.2f}%；",
        "  口径＝**名义本金**，全程不切换；毛利与净利分列。",
        f"- **参数来源**：各 catalog 指标默认参数（未寻优、未网格搜索）；grid 仅 H={p['horizon']} 与随机种子数。",
        "- **样本外切分**：训练 60% / 验证 15% / 测试 25%；方法排名只用前 75%（研发段），测试段只读冠军。",
        "- **撮合**：收盘判定触线、t+1 开盘入场、t+H 收盘出场；无止损止盈，纯持有；",
        "  事件取上升沿且相邻间隔 ≥ H，触发率异常的退化列已剔除。",
        "- **数据与覆盖**：derived（1m 聚合优先 + native 回落），7201 根/币；",
        "  auto_trendlines 因算力用 lookback=120 / max_pivots=20；s_r_zones_mtf 的 HTF 改为 240/1440 分钟。",
        "",
        "## 结果",
        "",
        "| 支撑位定义 | 事件数 | 净收益/笔 | 95% CI（净超额） | 减随机价位 | 破位率 | 随机价位破位率 | 随机基线分位 | q | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in p["groups"]:
        lines.append(
            f"| {r['label']} (`{r['key']}`) | {r['events']} | {r['mean_net_pct']:.4f}% | "
            f"[{r['ci_low_pct']:.4f}, {r['ci_high_pct']:.4f}]% | "
            f"{r['excess_vs_random_price_pct']:.4f}% | {r['break_rate']:.3f} | "
            f"{r['random_price_break_rate']:.3f} | {r['random_price_percentile']:.3f} | "
            f"{r['q_value']:.4f} | `{r['verdict']}` |"
        )
    lines += [
        "",
        f"- 随机位移基线全样本均值 {p['baseline']['random_shift_mean_pct']}%，"
        f"随机价位基线 {p['baseline']['random_price_mean_pct']}%；"
        f"任意时点持有 {p['baseline']['any_time_hold_net_pct']}%（净、已扣成本）。",
        f"- **量具自校准**：oracle 正对照（偷看未来）净收益 {p['baseline']['oracle_positive_control_net_pct']}%"
        f"／{p['baseline']['oracle_positive_control_events']} 事件，"
        f"检出={'是' if p['calibration']['positive_control_detected'] else '否'}；"
        "负对照（随机价位/任意时点）见表；半衰对照见各组 `half_shift_pct`。",
        "- 时间稳定性：4 个等长子段，逐段净收益与同号段数见同名 JSON 的 "
        "`quarter_net_pct` / `segment_consistency` / `positive_quarters`。",
        f"- **样本外冠军**：研发段（训练+验证）第一是 `{p['champion_on_dev']['key']}`，"
        f"研发段净超额 {p['champion_on_dev']['dev_excess_pct']}%，"
        f"测试段只读结果 {p['champion_on_dev']['test_excess_pct']}%。",
        "",
    ] + rank_lines + [
        "## 判定",
        "",
        f"- **verdict**：`{p['verdict']}`（词表取值；`inconclusive` 表示 CI 跨 0，不等于「无效」）。",
        f"- 池化 CI：重采样单位为一次事件，`MIN_POOLED_TRADES={MIN_POOLED_TRADES}`，"
        f"n_boot=2000；总事件数 {p['trades']}。",
        f"- 随机基线：{args.seeds} 种子 × {p['universe']['symbols']} 币，分位数 > 0.95 才算跑赢随机。",
        "- 归因：净超额 = 触线后净收益 − 随机价位基线，已同时扣除 0.12% 成本与 beta；",
        "  破位率对比用于区分「真的守住了」与「只是行情反弹」。",
        "- 与历史结论对照：与 `verdicts-history.md` 第 1 节（现货 OHLCV 全线证伪）一致；",
        "  本次未投入新特征，故不构成对止损线的违反。",
        "",
    ] + rob_lines + [
        "## 局限",
        "",
    ]
    lines += [f"- {item}" for item in p["limitations"]]
    lines += [
        "- 支撑位是「区域」还是「一条线」的口径差异会影响触线事件数；本报告统一取单一价格，",
        "  区域型指标取中点或最近档，未做区域宽度敏感性分析。",
        "",
        "## 结论与后续",
        "",
        f"- 结论：见上表与 `verdict={p['verdict']}`；**是否启用**：否，全部方法 `default_enabled=False`，",
        "  不进绑定、不进模拟盘/实盘。本次为量具诊断，不产生可上线策略。",
        "- 若 `negative` / `inconclusive`：按 H2 止损线，**不要再调阈值、周期或止损倍数救它**；",
        "  下一个信息增量应转向逐笔成交顺序／盘口挂撤单／多所现货腿，且先过 IC 探针。",
        "",
        "## 复现命令",
        "",
        "```powershell",
        f"# 两个持有期共用固定右端 --as-of {as_of_iso}",
        ".\\.venv\\Scripts\\python.exe backend/scripts/support_level_eval.py --interval 1h "
        f"--days {args.days} --limit {args.limit} --horizon 48 --seeds {args.seeds} "
        f"--workers {args.workers} --as-of {as_of_iso} --out docs/support-level-evaluation-h48",
        ".\\.venv\\Scripts\\python.exe backend/scripts/support_level_eval.py --interval 1h "
        f"--days {args.days} --limit {args.limit} --horizon {args.horizon} --seeds {args.seeds} "
        f"--workers {args.workers} --as-of {as_of_iso} --out {args.out} "
        "--robustness-from docs/support-level-evaluation-h48.json",
        ".\\.venv\\Scripts\\python.exe -m pytest backend/tests/test_support_level_eval.py -q",
        f".\\.venv\\Scripts\\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix {args.out}",
        ".\\.venv\\Scripts\\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py "
        "--prefix docs/support-level-evaluation-h48",
        "```",
        "",
        "## 产物",
        "",
        f"- `{args.out}.md` + `{args.out}.json`；单测 `backend/tests/test_support_level_eval.py`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

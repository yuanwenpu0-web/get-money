"""指标信号检测与优化实验入口。

用法：
    .venv\\Scripts\\python.exe backend/scripts/indicator_lab.py --interval 1h --days 400 --limit 40
    .venv\\Scripts\\python.exe backend/scripts/indicator_lab.py --limit 6 --days 120 --out-prefix docs/indicator-lab-smoke

做四件事：遍历单指标参数空间 → 从训练集前 K 名做两两组合 → 合并多重比较校正 →
样本外与滚动验证 + 随机基线。所有结果写 `docs/indicator-lab.{md,json}`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy.orm import sessionmaker

from app.db import make_engine
from app.research import lab_combos, lab_experiment, lab_grid, lab_report
from app.research.lab_grid import Candidate
from app.research.lab_signals import Signals
from app.research.lab_sim import SimConfig, Trade, simulate, to_ms
from app.research.lab_stats import summarize

WARMUP = 200


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--limit", type=int, default=40, help="参与评估的币种数（按 bar 数取前 N）")
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--max-hold", type=int, default=240)
    ap.add_argument("--side", choices=["both", "long", "short"], default="both", help="只评估某一方向的信号")
    ap.add_argument("--top", type=int, default=8, help="进入组合搜索的候选数")
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--draws", type=int, default=60)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--families", default="", help="逗号分隔，只跑指定指标")
    ap.add_argument("--out-prefix", default="docs/indicator-lab")
    ap.add_argument("--skip-baseline", action="store_true")
    return ap.parse_args()


def always_side(ds, cfg: SimConfig, side: int) -> list[Trade]:
    trades: list[Trade] = []
    for symbol, frame in ds.frames.items():
        n = len(frame)
        mask = np.zeros(n, dtype=bool)
        mask[min(WARMUP, n - 1):] = True
        signals = Signals(mask, np.zeros(n, dtype=bool)) if side == 1 else Signals(np.zeros(n, dtype=bool), mask)
        trades.extend(simulate(symbol, frame, signals, ds.exits[symbol], cfg))
    return trades


def _leg(symbol: str, entry_open: float, exit_close: float, entry_ts: int, exit_ts: int, bars: int, cfg: SimConfig) -> Trade:
    entry = entry_open * (1 + cfg.slippage)
    exit_ = exit_close * (1 - cfg.slippage)
    qty = cfg.stake / entry
    pnl = qty * (exit_ - entry) - cfg.fee * cfg.stake - cfg.fee * qty * exit_
    return Trade(symbol, 1, int(entry_ts), int(exit_ts), entry, exit_, int(bars),
                 float(pnl), float(pnl / cfg.stake), 0.0, "buy_hold")


def buy_hold(ds, cfg: SimConfig) -> list[Trade]:
    """买入持有，按切分点拆成训练腿与样本外腿，保证与其它候选同一套切分口径。"""
    trades: list[Trade] = []
    for symbol, frame in ds.frames.items():
        if len(frame) <= WARMUP + 2:
            continue
        stamps = to_ms(frame.index)
        open_ = frame["open"].to_numpy(dtype=float)
        close = frame["close"].to_numpy(dtype=float)
        cut = int(np.searchsorted(stamps, ds.split_ts))
        if cut - 1 > WARMUP:
            trades.append(_leg(symbol, open_[WARMUP], close[cut - 1], stamps[WARMUP], stamps[cut - 1], cut - 1 - WARMUP, cfg))
        if 0 <= cut < len(frame) - 1:
            trades.append(_leg(symbol, open_[cut], close[-1], stamps[cut], stamps[-1], len(frame) - 1 - cut, cfg))
    return trades


def buy_hold_single(ds, cfg: SimConfig) -> list[Trade]:
    """单笔持有到期末：作为「全期」读数的口径，不能与两段腿相加。"""
    trades: list[Trade] = []
    for symbol, frame in ds.frames.items():
        if len(frame) <= WARMUP + 2:
            continue
        stamps = to_ms(frame.index)
        trades.append(_leg(symbol, frame["open"].to_numpy(float)[WARMUP], frame["close"].to_numpy(float)[-1],
                           stamps[WARMUP], stamps[-1], len(frame) - 1 - WARMUP, cfg))
    return trades


def split_summary(trades: list[Trade], ds, cfg: SimConfig, full: list[Trade] | None = None) -> dict:
    train, test, _ = lab_experiment.split_trades(trades, ds.split_ts)
    return {
        "train": summarize(train, cfg.stake, len(ds.frames)),
        "test": summarize(test, cfg.stake, len(ds.frames)),
        "full": summarize(full if full is not None else trades, cfg.stake, len(ds.frames)),
    }


def main() -> None:
    args = parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()] or None
    cands = lab_grid.candidates(families)
    side_code = {"both": 0, "long": 1, "short": -1}[args.side]
    cfg = SimConfig(args.stake, args.fee, args.slippage, args.max_hold, side_code)
    bench_cfg = SimConfig(args.stake, args.fee, args.slippage, args.max_hold, 0)
    started = time.time()

    engine = make_engine(read_only=True)
    db = sessionmaker(bind=engine)()
    print(f"加载 {args.interval} / {args.days} 天 / 前 {args.limit} 币 ...", flush=True)
    ds = lab_experiment.load_dataset(db, args.interval, args.days, args.limit, train_ratio=args.train_ratio)
    print(f"  {len(ds.frames)} 币，{ds.meta()['bars_total']} 根 bar，切点 {ds.meta()['split'][:16]}", flush=True)
    bad = [q for q in ds.quality if (q["gap_pct"] or 0) > 2]
    if bad:
        print(f"  注意：{len(bad)} 个币缺口率 >2%：{bad[0]['symbol']} {bad[0]['gap_pct']}%", flush=True)

    print(f"单指标遍历：{len(cands)} 组参数 ...", flush=True)
    singles = lab_experiment.run_singles(ds, cands, cfg, args.min_trades)
    ranked = lab_experiment.rank(singles)
    print(f"  完成，用时 {time.time() - started:.0f}s", flush=True)

    sample_ok = [r for r in ranked if r["verdict"] != "insufficient"]
    by_key = {c.key: c for c in cands}
    tops = [by_key[r["key"]] for r in sample_ok[: args.top] if r["key"] in by_key]
    print(f"组合搜索：{len(tops)} 个候选（训练集盈亏前 {args.top}，且样本量达标）两两组合 ...", flush=True)
    combos = lab_combos.run_combos(ds, tops, cfg) if len(tops) >= 2 else []
    lab_combos.apply_global_fdr(singles, combos, args.min_trades)
    all_rows = lab_experiment.rank(singles + combos)
    print(f"  完成，组合 {len(combos)} 组，用时 {time.time() - started:.0f}s", flush=True)

    wf_targets = tops[: min(10, len(tops))] or []
    walk = lab_combos.walk_forward(ds, wf_targets, cfg, args.folds, max(10, args.min_trades // 3)) if wf_targets else {"folds": []}

    baseline = {"target": "—", "draws": 0, "seed": args.seed, "actual_total_pnl": 0.0,
                "baseline_mean": 0.0, "baseline_p95": 0.0, "percentile": None, "p_value": None}
    if all_rows and not args.skip_baseline:
        best_single_row = next((r for r in all_rows if "family" in r), None)
        target = by_key.get(best_single_row["key"]) if best_single_row else None
        if target is not None:
            print("随机基线 ...", flush=True)
            baseline = lab_combos.random_baseline(ds, target, cfg, args.draws, args.seed)
            baseline["target"] = target.label

    benchmarks = {
        "always_long": split_summary(always_side(ds, bench_cfg, 1), ds, bench_cfg),
        "always_short": split_summary(always_side(ds, bench_cfg, -1), ds, bench_cfg),
        "buy_hold": split_summary(buy_hold(ds, bench_cfg), ds, bench_cfg, full=buy_hold_single(ds, bench_cfg)),
    }

    config = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "interval": args.interval, "days": args.days, "limit": args.limit,
        "stake": args.stake, "fee": args.fee, "slippage": args.slippage,
        "round_trip_pct": round(cfg.round_trip_cost_pct, 4),
        "side": args.side,
        "max_hold_bars": args.max_hold, "min_trades": args.min_trades,
        "top": args.top, "folds": args.folds, "draws": args.draws,
        "train_ratio": args.train_ratio, "seed": args.seed,
        "families": families, "exit_rule": "rsi50_cross(14)",
    }
    config["config_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    best_single = next((r for r in all_rows if "family" in r), None)
    best_combo = next((r for r in all_rows if "rule" in r), None)
    summary = {
        "single_tests": len(singles),
        "combo_tests": len(combos),
        "total_tests": len(singles) + len(combos),
        "passed": int(sum(1 for r in singles + combos if r.get("verdict") == "candidate")),
        "train_positive": int(sum(1 for r in singles if r["train"]["total_pnl"] > 0)),
        "test_positive": int(sum(1 for r in singles if r["test"]["total_pnl"] > 0)),
        "consistent": int(sum(1 for r in singles if r["train"]["total_pnl"] > 0 and r["test"]["total_pnl"] > 0)),
        "test_top": [r["label"] for r in sorted(singles, key=lambda r: r["test"]["total_pnl"], reverse=True)[:5]],
        "best_single": best_single,
        "best_combo": best_combo,
        "elapsed_seconds": round(time.time() - started, 1),
    }

    payload = {
        "config": config,
        "dataset": ds.meta(),
        "grid": lab_grid.manifest(),
        "quality": ds.quality[:60],
        "singles_all": ranked,
        "singles_top": ranked[:15],
        "combos_all": lab_experiment.rank(combos) if combos else [],
        "combos_top": lab_experiment.rank(combos)[:10] if combos else [],
        "walk_forward": walk,
        "random_baseline": baseline,
        "benchmarks": benchmarks,
        "summary": summary,
        "controls": [
            "搜索空间预登记：参数区间与步长只在 lab_grid.GRID 定义，运行时写入配置快照并计算 sha256",
            f"时间切分 {args.train_ratio:.0%}/{1 - args.train_ratio:.0%}，跨越切点的持仓两边剔除（purge）",
            "候选排序只看训练集；样本外仅作判定，不参与任何选择",
            "单指标与组合合并做 BH-FDR（组合来自训练集排名，已消耗自由度，必须计入校正）",
            f"锚定式滚动验证 {args.folds} 折：每折用历史段选参，在紧随的验证段判定",
            f"随机基线：同频率随机入场 {args.draws} 次，固定种子 {args.seed}",
            "全部随机性来自 numpy default_rng(seed)，无系统时间依赖；配置哈希可复核一次运行",
        ],
        "limitations": [
            "单一 1h 周期、约 13 个月窗口，跨不过完整牛熊，结论只在样本期内成立",
            "native 1h 直查与 1m 聚合口径的差异未在本轮复核，两者不可混用",
            "固定 10 名义本金、无杠杆、未计永续资金费（现货口径），容量与冲击成本未估计",
            "单币同时一仓，未处理组合层面的资金竞争与相关性",
            "多空规则对称，未做方向择向，因此不含「只做多」这类有效子集",
            "rsi50 出场与逆势入场存在结构性冲突：RSI 低于 50 的多头会被立刻平掉",
            "FDR 假设检验独立，指标高度相关时 q 值偏保守，只能当筛子不能当定论",
        ],
    }

    root = Path(__file__).resolve().parents[2]
    md_path, json_path = lab_report.write_reports(root / args.out_prefix, payload)
    print(f"\n已写 {md_path.relative_to(root)} / {json_path.relative_to(root)}")
    if best_single:
        print(f"训练集最优单指标 {best_single['label']} 训练 {best_single['train']['total_pnl']:.2f} "
              f"样本外 {best_single['test']['total_pnl']:.2f} 判定 {best_single['verdict']}")
    print(f"通过全部关卡 {summary['passed']}/{summary['total_tests']}，总用时 {summary['elapsed_seconds']}s")


if __name__ == "__main__":
    main()

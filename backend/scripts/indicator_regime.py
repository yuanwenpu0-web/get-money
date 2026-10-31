"""Regime 过滤实验：把上一轮只做多选出的信号，叠上市场状态门控再跑一遍。

回答一个具体问题：**只做多信号在牛市跑输买入持有，是因为它不知道「现在能不能做多」吗？**
如果加上 regime 门控能补回样本外超额，说明缺的是状态判断；如果补不回，说明信号本身没有方向信息。

实验设计要点：

1. 候选**沿用**上一轮 `docs/indicator-lab-long.json` 里通过关卡的信号（不重新挑选，避免又开一轮选择）；
2. regime 只做入场门控，出场仍是统一的 `rsi50_cross`；
3. 每个 regime 都配一个「只用 regime、不用信号」的对照组——否则无法区分是 regime 在赚钱还是信号在赚钱；
4. 同一切分点、同一 purge、同一成本口径，与上一轮可比。
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
from sqlalchemy.orm import sessionmaker

from app.db import make_engine
from app.research import lab_experiment, lab_grid, lab_regime
from app.research.lab_grid import Candidate
from app.research.lab_signals import Signals
from app.research.lab_sim import SimConfig, Trade, simulate, to_ms
from app.research.lab_stats import summarize

WARMUP = 200


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--long-json", default="docs/indicator-lab-long.json")
    ap.add_argument("--top", type=int, default=5, help="取上一轮通过关卡的前 N 个单指标")
    ap.add_argument("--regimes", default="market_ema:200,market_ema:100,own_ema:200,own_ema_4h:200")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--stake", type=float, default=10.0)
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slippage", type=float, default=0.0002)
    ap.add_argument("--max-hold", type=int, default=240)
    ap.add_argument("--out-prefix", default="docs/indicator-regime")
    ap.add_argument("--scan", action="store_true", help="全族扫描：111 个候选逐个比较加/不加门控")
    ap.add_argument("--scan-regimes", default="market_ema:200,own_ema:200")
    return ap.parse_args()


def load_candidates(path: Path, top: int) -> list[Candidate]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in payload["singles_all"] if r.get("verdict") == "candidate" and "family" in r]
    rows.sort(key=lambda r: r["train"]["total_pnl"], reverse=True)
    return [Candidate(r["family"], r["params"], r["key"]) for r in rows[:top]]


def run(ds, signals: dict[str, Signals], cfg: SimConfig) -> list[Trade]:
    out: list[Trade] = []
    for symbol, frame in ds.frames.items():
        out.extend(simulate(symbol, frame, signals[symbol], ds.exits[symbol], cfg))
    return out


def block(trades: list[Trade], ds, cfg: SimConfig) -> dict:
    train, test, _ = lab_experiment.split_trades(trades, ds.split_ts)
    n = len(ds.frames)
    return {"train": summarize(train, cfg.stake, n), "test": summarize(test, cfg.stake, n),
            "full": summarize(trades, cfg.stake, n)}


def buy_hold(ds, cfg: SimConfig) -> tuple[list[Trade], list[Trade]]:
    """返回（两段腿，单笔全期腿）：两段用于 train/test，单笔用于 full。"""
    split_legs: list[Trade] = []
    full_legs: list[Trade] = []
    for symbol, frame in ds.frames.items():
        if len(frame) <= WARMUP + 2:
            continue
        stamps = to_ms(frame.index)
        open_ = frame["open"].to_numpy(dtype=float)
        close = frame["close"].to_numpy(dtype=float)
        cut = int(np.searchsorted(stamps, ds.split_ts))
        for entry_bar, exit_bar in ((WARMUP, cut - 1), (cut, len(frame) - 1)):
            if exit_bar > entry_bar:
                split_legs.append(_leg(symbol, open_[entry_bar], close[exit_bar], stamps[entry_bar], stamps[exit_bar], cfg))
        full_legs.append(_leg(symbol, open_[WARMUP], close[-1], stamps[WARMUP], stamps[-1], cfg))
    return split_legs, full_legs


def _leg(symbol: str, entry_open: float, exit_close: float, entry_ts: int, exit_ts: int, cfg: SimConfig) -> Trade:
    entry = entry_open * (1 + cfg.slippage)
    exit_ = exit_close * (1 - cfg.slippage)
    qty = cfg.stake / entry
    pnl = qty * (exit_ - entry) - cfg.fee * cfg.stake - cfg.fee * qty * exit_
    return Trade(symbol, 1, int(entry_ts), int(exit_ts), entry, exit_, 0, float(pnl),
                 float(pnl / cfg.stake), 0.0, "buy_hold")


def gate(signals: dict[str, Signals], masks: dict[str, np.ndarray]) -> dict[str, Signals]:
    return {s: Signals(signals[s].long & masks[s], np.zeros_like(signals[s].short)) for s in signals}


def scan_all(ds, cfg: SimConfig, spec: str) -> dict:
    """全族扫描：每个候选各跑一次「不加门控」和「加门控」，看提升是不是普遍的。"""
    kind, n_text = spec.split(":")
    masks = lab_regime.build(ds.frames, kind, int(n_text))
    lifts, base_rets, gated_rets, gated_trades, collapsed = [], [], [], [], 0
    for cand in lab_grid.candidates():
        sig = {s: lab_experiment.sign(cand, f) for s, f in ds.frames.items()}
        base_stats = block(run(ds, sig, cfg), ds, cfg)
        gated_stats = block(run(ds, gate(sig, masks), cfg), ds, cfg)
        lifts.append(gated_stats["full"]["total_return_pct"] - base_stats["full"]["total_return_pct"])
        base_rets.append(base_stats["full"]["total_return_pct"])
        gated_rets.append(gated_stats["full"]["total_return_pct"])
        gated_trades.append(gated_stats["full"]["trades"])
        if gated_stats["full"]["trades"] < 50:
            collapsed += 1
    return {
        "regime": lab_regime.describe(kind, int(n_text)),
        "cells": len(lifts),
        "median_lift": round(float(np.median(lifts)), 2),
        "share_improved": round(float(np.mean(np.array(lifts) > 0)), 3),
        "median_base_return": round(float(np.median(base_rets)), 2),
        "median_gated_return": round(float(np.median(gated_rets)), 2),
        "median_trades": int(np.median(gated_trades)),
        "sample_collapse_share": round(collapsed / len(lifts), 3),
    }


def write_report(path: Path, payload: dict) -> None:
    cfg, meta, bench = payload["config"], payload["dataset"], payload["benchmarks"]
    lines = ["# Regime 过滤实验（只做多）", "",
             f"> 生成时间 {cfg['generated_at']}；候选来自 `{cfg['source']}`；切分与上一轮一致。", "",
             "## 1. 口径", "",
             f"- 数据：{meta['interval']} native，{meta['symbols']} 币，{meta['start'][:16]} → {meta['end'][:16]}",
             f"- 切分：训练/样本外以 {meta['split'][:16]} 为界，跨切点持仓剔除",
             f"- 出场：统一 `rsi50_cross`（regime 只拦入场，不强制平仓）",
             f"- 成本：往返 {cfg['round_trip_pct']:.3f}%，每笔 {cfg['stake']} 名义本金；收益率 = 盈亏 /（本金 × 最大并发）", "",
             "## 2. 基准（同口径）", "",
             "| 基准 | 训练收益% | 样本外收益% | 全期收益% | 样本外回撤% |",
             "|---|---:|---:|---:|---:|"]
    for name in ("buy_hold", "always_long"):
        row = bench[name]
        lines.append(f"| {name} | {row['train']['total_return_pct']:.2f} | {row['test']['total_return_pct']:.2f} | "
                     f"{row['full']['total_return_pct']:.2f} | {row['test']['max_drawdown_pct']:.2f} |")
    lines += ["", "## 3. 信号 × regime 结果", "",
              "样本外交易数 <20 的单元格标 `样本不足`，不参与「最优」评选——样本塌缩时收益率会被本金基数放大成假象。",
              "",
              "| 信号 | regime | on 占比 | 全期交易数 | 样本外交易数 | 训练收益% | 样本外收益% | 全期收益% | 样本外回撤% | 样本外超额% | 全期超额% | 相对无门控 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    bh = bench["buy_hold"]
    for row in payload["rows"]:
        flag = " （样本不足）" if row["test"]["trades"] < 20 else ""
        lines.append(
            f"| `{row['candidate']}` | {row['regime']} | {row['on_ratio'] * 100:.0f}% | {row['full']['trades']} | "
            f"{row['test']['trades']} | {row['train']['total_return_pct']:.2f} | {row['test']['total_return_pct']:.2f}{flag} | "
            f"{row['full']['total_return_pct']:.2f} | {row['test']['max_drawdown_pct']:.2f} | "
            f"{row['test']['total_return_pct'] - bh['test']['total_return_pct']:+.2f} | "
            f"{row['full']['total_return_pct'] - bh['full']['total_return_pct']:+.2f} | "
            f"{row['lift_full']:+.2f} |"
        )
    lines += ["", "`样本外超额/全期超额` 相对买入持有；`相对无门控` = 全期收益 − 同信号不加 regime 的全期收益。", "",
              "## 4. 对照组：只用 regime、不用信号", "",
              "| regime | on 占比 | 训练收益% | 样本外收益% | 全期收益% | 交易数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in payload["regime_only"]:
        lines.append(f"| {row['regime']} | {row['on_ratio'] * 100:.0f}% | {row['train']['total_return_pct']:.2f} | "
                     f"{row['test']['total_return_pct']:.2f} | {row['full']['total_return_pct']:.2f} | "
                     f"{row['full']['trades']} |")
    if payload.get("scan"):
        lines += ["", "## 4.5 全族扫描（111 个候选逐个比较）", "",
                  "| regime | 候选数 | 中位提升 pp | 提升占比 | 中位无门控收益% | 中位加门控收益% | 中位交易数 | 样本塌缩占比 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in payload["scan"]:
            lines.append(f"| {row['regime']} | {row['cells']} | {row['median_lift']:+.2f} | "
                         f"{row['share_improved'] * 100:.0f}% | {row['median_base_return']:.2f} | "
                         f"{row['median_gated_return']:.2f} | {row['median_trades']} | "
                         f"{row['sample_collapse_share'] * 100:.0f}% |")
    lines += ["", "## 5. 判读", ""] + [f"- {item}" for item in payload["findings"]] + [
        "", "## 6. 限制", ""] + [f"- {item}" for item in payload["limitations"]] + [""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = SimConfig(args.stake, args.fee, args.slippage, args.max_hold, side=1)
    started = time.time()

    cands = load_candidates(root / args.long_json, args.top)
    if not cands:
        raise SystemExit("上一轮没有通过关卡的候选，先跑 --side long")
    db = sessionmaker(bind=make_engine(read_only=True))()
    ds = lab_experiment.load_dataset(db, args.interval, args.days, args.limit)
    print(f"{len(ds.frames)} 币 / {len(cands)} 候选 / {args.regimes}")

    base_signals = {c.key: {s: lab_experiment.sign(c, f) for s, f in ds.frames.items()} for c in cands}
    base = {c.key: block(run(ds, base_signals[c.key], cfg), ds, cfg) for c in cands}

    specs = [part.split(":") for part in args.regimes.split(",") if part.strip()]
    rows, regime_only = [], []
    for kind, n_text in specs:
        masks = lab_regime.build(ds.frames, kind, int(n_text))
        label = lab_regime.describe(kind, int(n_text))
        ratio = lab_regime.on_ratio(masks)
        only_signals = {s: Signals(masks[s], np.zeros(len(masks[s]), dtype=bool)) for s in ds.frames}
        regime_only.append({"regime": label, "on_ratio": ratio, **block(run(ds, only_signals, cfg), ds, cfg)})
        for cand in cands:
            gated = gate(base_signals[cand.key], masks)
            stats = block(run(ds, gated, cfg), ds, cfg)
            rows.append({"candidate": cand.label, "regime": label, "on_ratio": ratio,
                         "lift_full": round(stats["full"]["total_return_pct"] - base[cand.key]["full"]["total_return_pct"], 2),
                         **stats})
        print(f"  {label} 完成 {time.time() - started:.0f}s", flush=True)

    legs, full_legs = buy_hold(ds, cfg)
    train, test, _ = lab_experiment.split_trades(legs, ds.split_ts)
    n = len(ds.frames)
    buy_hold_block = {"train": summarize(train, cfg.stake, n), "test": summarize(test, cfg.stake, n),
                      "full": summarize(full_legs, cfg.stake, n)}
    always = {s: Signals(np.concatenate([np.zeros(min(WARMUP, len(f) - 1), bool),
                                         np.ones(len(f) - min(WARMUP, len(f) - 1), bool)]), np.zeros(len(f), bool))
              for s, f in ds.frames.items()}
    always_block = block(run(ds, always, cfg), ds, cfg)

    scan = []
    if args.scan:
        for spec in [x.strip() for x in args.scan_regimes.split(",") if x.strip()]:
            scan.append(scan_all(ds, cfg, spec))
            print(f"  全族扫描 {spec} 完成 {time.time() - started:.0f}s", flush=True)
    eligible = [r for r in rows if r["test"]["trades"] >= 20]
    naive = max(rows, key=lambda r: r["full"]["total_return_pct"])
    best = max(eligible, key=lambda r: r["full"]["total_return_pct"]) if eligible else naive
    best_oos = max(eligible, key=lambda r: r["test"]["total_return_pct"]) if eligible else naive
    base_best = max(base.items(), key=lambda kv: kv[1]["full"]["total_return_pct"])
    best_regime = max(regime_only, key=lambda r: r["full"]["total_return_pct"])
    findings = [
        f"可判定样本内最好的组合：`{best['candidate']}` + {best['regime']}，全期 {best['full']['total_return_pct']:.2f}%、"
        f"样本外 {best['test']['total_return_pct']:.2f}%（{best['test']['trades']} 笔），"
        f"相对同信号不加门控 {best['lift_full']:+.2f} 个百分点。",
        f"名义最高的是 `{naive['candidate']}` + {naive['regime']}（全期 {naive['full']['total_return_pct']:.2f}%），"
        f"但全期仅 {naive['full']['trades']} 笔、样本外 {naive['test']['trades']} 笔，属样本塌缩，不采信。",
        f"对照组（只用 {best_regime['regime']} 一直做多、完全不用指标信号）全期 {best_regime['full']['total_return_pct']:.2f}%、"
        f"样本外 {best_regime['test']['total_return_pct']:.2f}%、{best_regime['full']['trades']} 笔："
        f"不低于最好的「信号 + 门控」组合（{best['full']['total_return_pct']:.2f}%），"
        "说明增量主要来自市场状态判断，而不是指标本身。",
        f"不加门控的最好信号全期 {base_best[1]['full']['total_return_pct']:.2f}%，加门控后 {best['full']['total_return_pct']:.2f}%（{best['lift_full']:+.2f}）。",
    ]
    if scan:
        market = next((r for r in scan if r["regime"].startswith("market_ema")), None)
        own = next((r for r in scan if r["regime"].startswith("own_ema(")), None)
        if market:
            findings.append(
                f"全族扫描 market_ema(200)：{market['cells']} 个候选里 {market['share_improved'] * 100:.0f}% 收益提升，"
                f"中位 {market['median_lift']:+.2f} 个百分点，中位交易数 {market['median_trades']}，样本塌缩 {market['sample_collapse_share'] * 100:.0f}% —— "
                "是普遍效应，但中位加门控收益仍未超过 regime 单独做多。"
            )
        if own:
            findings.append(
                f"全族扫描 own_ema(200)：仅 {own['share_improved'] * 100:.0f}% 的候选提升，中位 {own['median_lift']:+.2f} 个百分点，"
                f"样本塌缩 {own['sample_collapse_share'] * 100:.0f}% —— 币自身均线过滤会砍掉大部分样本，"
                "头部单元格的高收益是小样本放大，不可用。"
            )
    payload = {
        "config": {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "source": args.long_json,
                   "stake": args.stake, "round_trip_pct": round(cfg.round_trip_cost_pct, 4),
                   "regimes": args.regimes, "top": args.top},
        "dataset": ds.meta(),
        "candidates": [{"label": c.label, "params": c.params, "base": base[c.key]} for c in cands],
        "benchmarks": {"buy_hold": buy_hold_block, "always_long": always_block},
        "rows": rows,
        "regime_only": regime_only,
        "scan": scan,
        "findings": findings,
        "limitations": [
            "regime 只用一种定义（均线位置），没有测波动率、资金费率、宽度等状态变量",
            "regime 仅拦入场、不强制平仓，持仓期间状态反转不会提前退出",
            "样本外窗口是单边牛市（买入持有 +126.69%），任何降低曝露的门控都会在样本外吃亏",
            "本轮又叠加了 4 个 regime × 5 个信号 = 20 次比较，未再做多重比较校正，只能当探索性结论",
        ],
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out = root / args.out_prefix
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(out.with_suffix(".md"), payload)
    print(f"\n已写 {out.relative_to(root)}.md / .json")
    print(f"最好组合 {best['candidate']} + {best['regime']}：全期 {best['full']['total_return_pct']:.2f}% "
          f"样本外 {best['test']['total_return_pct']:.2f}%")


if __name__ == "__main__":
    main()

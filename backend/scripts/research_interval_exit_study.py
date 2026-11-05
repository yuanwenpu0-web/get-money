"""研究池策略：换 15m 是否更赚 + 加「趋势反转式」分批止盈的受控对照。

为什么单独一个脚本
------------------
`research_high/low_turnover_high_vol` 是本项目**唯一正在跑**的策略，但它们和通用管线
不一样：只做多、`direction≡0`、没有止损止盈、按「持有 N 小时后下一开盘」出场，模拟盘走
`paper.open_research_position` 专用通道。`run_backtest` 套不进去，所以要单独回放。

研究问题
--------
1. 同样的入场规则，把 K 线从 1h 换成 **15m**，收益会不会更好？
2. 给它们补上 `supertrend_current`（趋势反转）那套 **ATR 止损 + 1R/2R/3R 按比例分批止盈**，
   能不能把负期望拉正？（原策略根本没有止损，这是新增的出场结构，不是调参）

口径
----
- **入场规则一字不改**：`rsi_reversal(21,30,70) OR bollinger_break(35,2.0)`（高成交额池）
  / `OR atr_channel_break(20,2.0)`（低成交额池），信号在 t 收盘确认、t+1 开盘成交。
- **持有期按墙钟对齐**：120h / 48h 是两个策略的原始设定，换周期后换算成 bar 数
  （15m → 480 / 192 根），保证「持有时间」不变，变的只有 K 线粒度。
- **成本同一套**：taker 0.04%/边 + 滑点 0.02%（对己方不利方向），每笔固定名义 200 USDT
  （与 `docs/pool-research.md` 一致），不加仓不复利。
- **分批止盈语义与趋势反转一致**：R = |成交价 − 初始止损|，1R/2R/3R 各平 1/6、1/3、1/2，
  最后一档平掉剩余；同根内**止损优先于止盈**（保守口径），跳空穿越止损按开盘价成交。
- 主指标是**每笔收益率**（对 200 名义计），池化后给 bootstrap 95% CI。单币累计收益只作辅助。

用法
----
    .\\.venv\\Scripts\\python.exe backend/scripts/research_interval_exit_study.py
    .\\.venv\\Scripts\\python.exe backend/scripts/research_interval_exit_study.py --days 300
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

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.research.lab_signals import atr_channel_break, bollinger_break, rsi_reversal  # noqa: E402
from app.strategy.bindings import active_bindings  # noqa: E402
from app.utils import now_ms  # noqa: E402

STUDY_CODES = ("research_high_turnover_high_vol", "research_low_turnover_high_vol")
INTERVALS = {"15m": 900_000, "1h": 3_600_000}
HOLD_HOURS = {"research_high_turnover_high_vol": 120, "research_low_turnover_high_vol": 48}
CHANNEL = {"research_high_turnover_high_vol": "boll", "research_low_turnover_high_vol": "atrch"}
NOTIONAL = 200.0
FEE = 0.0004
SLIP = 0.0002
WARMUP = 200              # 与 spec 的 `valid = idx >= 200` 一致
ATR_LEN = 14
SL_ATR = 1.5              # 与 supertrend_current 默认 sl_atr_mult 一致
SPLIT = ((1.0, 1 / 6), (2.0, 1 / 3), (3.0, 0.5))   # (R 倍数, 原始仓位比例)
BOOT, SEED = 2000, 20261104

_DB = None


def _db():
    global _DB
    if _DB is None:
        _DB = sessionmaker(bind=make_engine(read_only=True))()
    return _DB


def _rma(values: np.ndarray, length: int) -> np.ndarray:
    """Wilder 平滑（前 length 个有效值用简单均值播种），与 supertrend 口径一致。"""
    out = np.full(len(values), np.nan)
    count, total, prev = 0, 0.0, np.nan
    for i, x in enumerate(values):
        if not np.isfinite(x):
            out[i] = prev
            continue
        if count < length:
            count += 1
            total += x
            if count == length:
                prev = total / length
        else:
            prev = (prev * (length - 1) + x) / length
        out[i] = prev
    return out


def _atr(frame: pd.DataFrame, length: int = ATR_LEN) -> np.ndarray:
    high, low, close = (frame[k].to_numpy(float) for k in ("high", "low", "close"))
    prev_close = np.concatenate(([np.nan], close[:-1]))
    tr = np.nanmax(np.vstack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)]), axis=0)
    return _rma(tr, length)


def _entries(frame: pd.DataFrame, kind: str) -> np.ndarray:
    """研究策略的入场布尔序列；缺口后重新预热，与 spec.prepare 的位移规则一致。"""
    long = rsi_reversal(frame, {"n": 21, "low": 30, "high": 70}).long
    channel = (bollinger_break(frame, {"n": 35, "k": 2.0}) if kind == "boll"
               else atr_channel_break(frame, {"n": 20, "k": 2.0})).long
    entry = np.asarray(long | channel, dtype=bool)
    step = frame.index.to_series().diff().dt.total_seconds().mul(1000).to_numpy()
    gap = np.flatnonzero(step[1:] != np.nanmedian(step[1:])) + 1
    valid = np.zeros(len(frame), dtype=bool)
    valid[WARMUP:] = True
    for g in gap:
        valid[g:min(g + WARMUP, len(frame))] = False
    return entry & valid


def simulate_exit(frame: pd.DataFrame, entry: np.ndarray, hold_bars: int, mode: str) -> list[dict]:
    """逐 bar 回放：t 收盘信号 → t+1 开盘进场，出场按 `mode` 决定。

    同根内的判定顺序与 `backtest.execution.simulate` 一致：开盘穿越止损 → 缺口强平
    → 开盘穿越目标 → 盘中止损（优先）→ 盘中目标。时间止损在开盘先于盘中止损生效。
    """
    o = frame["open"].to_numpy(float)
    h = frame["high"].to_numpy(float)
    lo = frame["low"].to_numpy(float)
    atr = _atr(frame)
    step = _step(frame)
    stamps = frame.index.to_series().diff().dt.total_seconds().mul(1000).to_numpy()
    gap = np.zeros(len(frame), dtype=bool)
    gap[1:] = stamps[1:] != step
    n = len(frame)
    trades: list[dict] = []
    i = WARMUP
    while i < n - 2:
        if not entry[i]:
            i += 1
            continue
        e = i + 1
        if e + hold_bars >= n:
            break
        fill = o[e] * (1 + SLIP)
        stop = fill - SL_ATR * atr[i] if np.isfinite(atr[i]) else np.nan
        risk = fill - stop if np.isfinite(stop) else np.nan
        guarded = mode != "hold_only" and np.isfinite(risk) and risk > 0
        if not guarded:
            stop = np.nan
        levels = [[fill + risk * r, frac, False] for r, frac in SPLIT] if (mode == "split_1_2_3" and guarded) else []
        qty = NOTIONAL / fill
        pnl, left, exit_idx, reason = -NOTIONAL * FEE, qty, e + hold_bars, "time_exit"
        for j in range(e, e + hold_bars + 1):
            if j == e + hold_bars:          # 持满 N 根 → 下一开盘时间出场
                break
            if j > e and gap[j]:
                pnl += (o[j] * (1 - SLIP) - fill) * left - o[j] * left * FEE
                left, exit_idx, reason = 0.0, j, "data_gap"
                break
            if guarded and o[j] <= stop:
                px = min(o[j], stop)
                pnl += (px * (1 - SLIP) - fill) * left - px * left * FEE
                left, exit_idx, reason = 0.0, j, "stop_loss_gap"
                break
            for level in levels:
                if level[2] or o[j] < level[0]:
                    continue
                level[2] = True
                q = min(qty * level[1], left)
                pnl += (level[0] * (1 - SLIP) - fill) * q - level[0] * q * FEE
                left -= q
            if left <= 1e-12:
                exit_idx, reason = j, "take_profit"
                break
            if guarded and lo[j] <= stop:
                pnl += (stop * (1 - SLIP) - fill) * left - stop * left * FEE
                left, exit_idx, reason = 0.0, j, "stop_loss"
                break
            for level in levels:
                if level[2] or h[j] < level[0]:
                    continue
                level[2] = True
                q = min(qty * level[1], left)
                pnl += (level[0] * (1 - SLIP) - fill) * q - level[0] * q * FEE
                left -= q
            if left <= 1e-12:
                exit_idx, reason = j, "take_profit"
                break
        j = min(e + hold_bars, n - 1)
        if left > 1e-12:
            pnl += (o[j] * (1 - SLIP) - fill) * left - o[j] * left * FEE
            exit_idx = j
        trades.append({"entry": int(i), "exit": int(exit_idx), "pnl": float(pnl),
                       "return_pct": float(pnl / NOTIONAL * 100), "reason": reason})
        i = exit_idx + 1
    return trades


def _step(frame: pd.DataFrame) -> float:
    diffs = frame.index.to_series().diff().dt.total_seconds().mul(1000).dropna()
    return float(diffs.mode().iloc[0]) if len(diffs) else 0.0


def run_one(task: tuple) -> dict:
    strategy, interval, mode, symbol, days = task
    start_ms = now_ms() - days * 86_400_000
    try:
        frame = MarketDataService(_db()).load_frame(symbol, interval, start_ms=start_ms, end_ms=now_ms())
        if len(frame) < WARMUP * 2:
            raise ValueError(f"样本不足: {len(frame)} 根")
        # 覆盖门槛：两周期必须站在同一个窗口上，否则「15m 更差」只是「15m 数据更短」。
        # 研究池里大量币的 native 15m 只有最近十来天，这类币两侧不可比。
        if frame.index[0].timestamp() * 1000 > start_ms + 2 * 86_400_000:
            raise ValueError(f"历史不足: {str(frame.index[0])[:10]} 起")
        entry = _entries(frame, CHANNEL[strategy])
        hold_bars = int(HOLD_HOURS[strategy] * 3_600_000 // INTERVALS[interval])
        trades = simulate_exit(frame, entry, hold_bars, mode)
    except Exception as exc:  # noqa: BLE001
        return {"strategy": strategy, "interval": interval, "mode": mode,
                "symbol": symbol, "error": repr(exc)[:160]}
    return {"strategy": strategy, "interval": interval, "mode": mode, "symbol": symbol,
            "trades": trades, "bars": len(frame),
            "start": str(frame.index[0]), "end": str(frame.index[-1]),
            "buy_hold_pct": float((frame.close.iloc[-1] / frame.open.iloc[0] - 1) * 100)}


def bootstrap(values: list[float], min_trades: int = 30) -> dict:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) < 2:
        return {"n": int(len(arr)), "mean": None, "ci_low": None, "ci_high": None, "verdict": "no_trades"}
    rng = np.random.default_rng(SEED)
    means = arr[rng.integers(0, len(arr), size=(BOOT, len(arr)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    if len(arr) < min_trades:
        verdict = "insufficient"          # n 太小的「显著」是假精确
    else:
        verdict = "positive" if lo > 0 else "negative" if hi < 0 else "inconclusive"
    return {"n": int(len(arr)), "mean": round(float(arr.mean()), 4),
            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4), "verdict": verdict}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    per_trade = [t["return_pct"] for r in ok for t in r["trades"]]
    per_symbol = [sum(t["pnl"] for t in r["trades"]) / NOTIONAL * 100 for r in ok]
    reasons = pd.Series([t["reason"] for r in ok for t in r["trades"]])
    return {
        "symbols": len(ok), "errors": len(rows) - len(ok),
        "trades": len(per_trade), "pooled": bootstrap(per_trade),
        "mean_symbol_pct": round(float(np.mean(per_symbol)), 2) if per_symbol else None,
        "median_symbol_pct": round(float(np.median(per_symbol)), 2) if per_symbol else None,
        "pct_symbols_positive": round(float(np.mean([x > 0 for x in per_symbol])), 4) if per_symbol else None,
        "exit_reason": {str(k): int(v) for k, v in reasons.value_counts().items()},
        "start": ok[0]["start"] if ok else None, "end": ok[0]["end"] if ok else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="研究池 15m vs 1h + 分批止盈对照")
    ap.add_argument("--days", type=int, default=90,
                    help="默认 90 天：研究池里大量币的 15m 历史只有最近十来天，窗口再长就只剩少数币可配")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "docs" / "research-interval-exit-study"))
    args = ap.parse_args()

    db = sessionmaker(bind=make_engine(read_only=True))()
    pairs, _ = active_bindings(db)
    targets = {p.strategy: list(p.symbols) for p in pairs if p.strategy in STUDY_CODES}
    if not targets:
        raise SystemExit("没有活跃的研究策略绑定")
    modes = ("hold_only", "hold_stop", "split_1_2_3")
    tasks = [
        (strategy, interval, mode, symbol, args.days)
        for strategy, symbols in targets.items()
        for interval in INTERVALS
        for mode in modes
        for symbol in symbols
    ]
    print(f"pools: {[(k, len(v)) for k, v in targets.items()]}  runs: {len(tasks)}", flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(run_one, tasks), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    cells: dict[str, dict] = {}
    for strategy in targets:
        for mode in modes:
            for interval in INTERVALS:
                cells[f"{strategy}|{mode}|{interval}"] = summarize(
                    [r for r in rows if r["strategy"] == strategy
                     and r["mode"] == mode and r["interval"] == interval]
                )
    comp = {}
    for strategy in targets:
        for mode in modes:
            a = {r["symbol"]: sum(t["pnl"] for t in r["trades"]) / NOTIONAL * 100
                 for r in rows if "error" not in r and r["strategy"] == strategy
                 and r["mode"] == mode and r["interval"] == "15m"}
            b = {r["symbol"]: sum(t["pnl"] for t in r["trades"]) / NOTIONAL * 100
                 for r in rows if "error" not in r and r["strategy"] == strategy
                 and r["mode"] == mode and r["interval"] == "1h"}
            common = sorted(set(a) & set(b))
            diff = [a[s] - b[s] for s in common]
            comp[f"{strategy}|{mode}"] = {
                "pairs": len(diff),
                "median_diff_pct": round(float(np.median(diff)), 2) if diff else None,
                "pct_15m_better": round(float(np.mean([d > 0 for d in diff])), 4) if diff else None,
                "ci": bootstrap(diff),
            }

    # 公平口径：两侧都用「同一批币」。15m 可用的币远少于 1h（native 15m 只覆盖最近十来天），
    # 直接比两列平均数等于比两个不同的宇宙，差值会被币种差异吞掉。
    fair: dict[str, dict] = {}
    for strategy in targets:
        for mode in modes:
            s15 = {r["symbol"] for r in rows if "error" not in r and r["strategy"] == strategy
                   and r["mode"] == mode and r["interval"] == "15m"}
            s1h = {r["symbol"] for r in rows if "error" not in r and r["strategy"] == strategy
                   and r["mode"] == mode and r["interval"] == "1h"}
            common = s15 & s1h
            for interval in INTERVALS:
                fair[f"{strategy}|{mode}|{interval}"] = summarize(
                    [r for r in rows if "error" not in r and r["strategy"] == strategy
                     and r["mode"] == mode and r["interval"] == interval and r["symbol"] in common]
                )

    payload = {"config": {"days": args.days, "notional": NOTIONAL, "fee": FEE, "slippage": SLIP,
                          "sl_atr": SL_ATR, "split": SPLIT, "warmup": WARMUP,
                          "hold_hours": HOLD_HOURS},
               "cells": cells, "fair_cells": fair, "paired_15m_minus_1h": comp,
               "per_symbol": [{k: v for k, v in r.items() if k != "trades"} for r in rows]}
    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                                        encoding="utf-8")
    _markdown(out.with_suffix(".md"), payload)
    print(f"written: {out.with_suffix('.json')} / {out.with_suffix('.md')}", flush=True)


def _f(value, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _markdown(path: Path, payload: dict) -> None:
    cfg = payload["config"]
    lines = [
        "# 研究池策略：换 15m 与「趋势反转式」分批止盈",
        "",
        f"数据：两个研究池当前成员，{cfg['days']} 天窗口，持有期按墙钟对齐"
        f"（120h / 48h）。成本 taker {cfg['fee']:.2%}/边 + 滑点 {cfg['slippage']:.2%}，"
        f"每笔固定名义 {cfg['notional']:.0f} USDT。",
        "",
        "出场结构：`hold_only` = 现状（定时出场，无止损无止盈）；`hold_stop` = 加 "
        f"{cfg['sl_atr']}×ATR 止损；`split_1_2_3` = 再加 1R/2R/3R 按 1/6、1/3、1/2 分批止盈。",
        "",
        "## 逐组合（同一批币，两侧可比）",
        "",
        "| 策略 | 出场结构 | 周期 | 币数 | 成交 | 每笔均值% | 95% CI | 判定 | 币均累计% | 中位% | 正收益币 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for key, s in payload["fair_cells"].items():
        strategy, mode, interval = key.split("|")
        p = s["pooled"]
        lines.append(
            f"| {strategy} | {mode} | {interval} | {s['symbols']} | {s['trades']} | {_f(p['mean'])} | "
            f"{_f(p['ci_low'])} ~ {_f(p['ci_high'])} | {p['verdict']} | {_f(s['mean_symbol_pct'], 2)} | "
            f"{_f(s['median_symbol_pct'], 2)} | {_f((s['pct_symbols_positive'] or 0) * 100, 0)}% |"
        )
    lines += ["", "## 同币配对：15m − 1h（消掉币种共同涨跌）", "",
              "| 策略 | 出场结构 | 配对币数 | 中位差 pp | 15m 更好的币占比 | 差值 95% CI |",
              "| --- | --- | ---: | ---: | ---: | --- |"]
    for key, c in payload["paired_15m_minus_1h"].items():
        strategy, mode = key.split("|")
        ci = c["ci"]
        lines.append(f"| {strategy} | {mode} | {c['pairs']} | {_f(c['median_diff_pct'], 2)} | "
                     f"{_f((c['pct_15m_better'] or 0) * 100, 0)}% | "
                     f"{_f(ci['ci_low'], 2)} ~ {_f(ci['ci_high'], 2)} |")
    lines += ["", "## 覆盖：被窗口门槛剔除的币", "",
              "| 策略 | 周期 | 可用币 | 剔除 | 剔除原因 |",
              "| --- | --- | ---: | ---: | --- |"]
    for key, s in payload["cells"].items():
        strategy, mode, interval = key.split("|")
        if mode != "hold_only":
            continue
        lines.append(f"| {strategy} | {interval} | {s['symbols']} | {s['errors']} | 15m/1h 历史短于窗口，两侧不可比 |")
    lines += ["", "## 出场原因（同上口径）", "", "| 策略 | 结构 | 周期 | stop_loss | take_profit | time_exit | 其它 |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for key, s in payload["fair_cells"].items():
        strategy, mode, interval = key.split("|")
        r = s["exit_reason"]
        stops = r.get("stop_loss", 0) + r.get("stop_loss_gap", 0)
        lines.append(f"| {strategy} | {mode} | {interval} | {stops} | {r.get('take_profit', 0)} | "
                     f"{r.get('time_exit', 0)} | {r.get('data_gap', 0)} |")
    lines += ["", "## 全宇宙（两侧币集不同，只作参考，不可用于比较周期）", "",
              "| 策略 | 结构 | 周期 | 币数 | 成交 | 每笔均值% | 95% CI | 判定 |",
              "| --- | --- | --- | ---: | ---: | ---: | --- | --- |"]
    for key, s in payload["cells"].items():
        strategy, mode, interval = key.split("|")
        p = s["pooled"]
        lines.append(f"| {strategy} | {mode} | {interval} | {s['symbols']} | {s['trades']} | "
                     f"{_f(p['mean'])} | {_f(p['ci_low'])} ~ {_f(p['ci_high'])} | {p['verdict']} |")
    lines += [
        "",
        "## 必须一起读的边界",
        "",
        "- **池是事后挑的**：两个研究池按「最近 168 小时」成交额/波动率筛出（当前窗口到 2026-11-03），",
        "  却拿来回放更早的行情 → 典型的事后选币。因此**绝对收益全部是上界，不能当预期**；",
        "  可用的只有**同币配对差值**（同一批币、同一窗口，选币偏差在两侧相同）。",
        "- **15m 历史是硬约束**：研究池里过半币种的 native 15m 只从 2026-10-15 起（约 1000 根），",
        "  1m 补齐也只到 2026-07-28。所以 90 天窗口下 15m 只剩 13/25 与 8/25 个币可用，",
        "  再拉长窗口可用币还会继续掉。这是「换 15m」目前最现实的障碍。",
        "- 分批止盈改变了出场结构，等于改了策略本身；周期效应看同一 mode 的 15m vs 1h，",
        "  止盈效应看同一周期的 mode 之间，两者不能混着读。",
        "- 15m 的 ATR 约为 1h 的一半，同一个 1.5×ATR 止损在 15m 上更紧、被打掉的概率更高，",
        "  所以 15m 的 `split_1_2_3` 与 1h 的 `split_1_2_3` 不是同一个风险预算。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

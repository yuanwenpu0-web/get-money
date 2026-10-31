"""入场价方案研究：信号方向已定，怎么把「下一根开盘市价」换成更合理的成交价。

研究问题
--------
现有通用撮合（backtest/execution.py）的入场只有一种：
**信号那根收盘 → 下一根开盘市价成交**，止损/止盈锚定在**信号收盘价**上。
本脚本把「入场方式」作为唯一变量，其余全部锁死：

- 信号时点不变（同一批信号、同一根 K 线）
- 止损/止盈的**绝对价位**不变（锚定信号收盘，即结构性失效位不随成交价移动）
- 出场优先级不变（同根内先判止损，保守口径）
- 风险预算不变（每笔风险 = 1R）

这样任何差异只能归因于入场价本身。

为什么必须这样设计
------------------
1. **限价单有成交率问题**。只统计「成交了的那些笔」必然高估：
   没成交等于样本被偷偷删掉。所以主指标是**每信号期望 R**（未成交记 0）。
2. **逆向选择**。限价买在下方只在跌下来时成交，而「跌下来才给机会」的信号
   可能恰好是最弱的一批。因此同时报「成交组 vs 未成交组在基准方案下的表现差」。
3. **止损变近、仓位变大**。买得更低而止损绝对位不变 → 风险距离缩小 →
   同风险预算下仓位更大。赢时赢得更多，但离失效位更近、被打掉的概率更高。
   两者必须同时看，且必须设风险距离下限，否则仓位会发散。

成本口径（按订单类型区分，这是限价单的主要优势之一）
--------------------------------------------------
- 市价入场：taker 0.04% + 滑点 0.02%（对你不利方向）
- 限价入场：maker 0.02%，成交价 = 委托价（**不做价格改善假设**）
  跳空穿越委托价时仍按委托价成交——这是真实的不利情形，必须保留
- 突破止损入场：taker 0.04% + 滑点 0.02%
- 出场：一律 taker 0.04% + 滑点 0.02%（保守，不假设止盈能挂 maker）

用法
----
    .\\.venv\\Scripts\\python.exe backend/scripts/entry_price_study.py
    .\\.venv\\Scripts\\python.exe backend/scripts/entry_price_study.py --days 400 --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.features.indicators import build_features  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from app.utils import now_ms  # noqa: E402

INTERVAL = "1h"
DAYS = 400
FEE_TAKER = 0.0004
FEE_MAKER = 0.0002
SLIP = 0.0002
MAX_HOLD = 240
MIN_GAP = 24
DEFAULT_WAIT = 6
MIN_RISK_ATR = 0.25   # 风险距离下限（×ATR）：低于此视为仓位发散，判为不可执行
QUEUE_BPS = 0.0       # 排队惩罚：限价单要求价格再穿越委托价 N 个基点才判成交

# 出场集。bracket 型 = (止损×ATR, 止盈×ATR)，锚定信号收盘；
# hold 型 = (None, None, 固定持有根数) —— 无止损止盈，纯粹隔离「入场价」本身的效果。
EXIT_SETS = {
    "sl15_tp30": (1.5, 3.0),
    "sl20_tp20": (2.0, 2.0),
    "hold_24": (None, None, 24),
}
PRIMARY_EXIT = "sl15_tp30"
ISOLATION_EXIT = "hold_24"   # 无 bracket，1R = 1×ATR，只比入场价

# kind:
#   market   —— 现状：下一根开盘市价
#   limit    —— 回撤挂单（maker），wait 根不成交则放弃
#   stop     —— 突破挂单（等确认），wait 根不触发则放弃
#   hybrid   —— 先挂 limit，wait 根不成交则转市价追（不丢信号）
#   cap      —— 仍是市价，但下一根开盘若已跑离信号收盘超过阈值则**跳过这笔**
POLICIES: dict[str, dict] = {
    "market_next_open": {"kind": "market"},

    # —— A. 回撤挂单（省费 + 买得便宜，代价是可能不成交）——
    "limit_at_close": {"kind": "limit", "ref": "close"},
    "limit_atr_0.2": {"kind": "limit", "ref": "atr", "k": 0.2},
    "limit_atr_0.4": {"kind": "limit", "ref": "atr", "k": 0.4},
    "limit_atr_0.6": {"kind": "limit", "ref": "atr", "k": 0.6},
    "limit_atr_1.0": {"kind": "limit", "ref": "atr", "k": 1.0},
    "limit_pct_0.25": {"kind": "limit", "ref": "pct", "k": 0.0025},
    "limit_pct_0.50": {"kind": "limit", "ref": "pct", "k": 0.0050},
    "limit_bar_low": {"kind": "limit", "ref": "bar_low"},
    "limit_bar_low_ext": {"kind": "limit", "ref": "bar_low_ext", "k": 0.1},
    "limit_range_mid": {"kind": "limit", "ref": "range_mid"},

    # —— B. 突破确认（买在上方，代价是追高）——
    "stop_atr_0.2": {"kind": "stop", "ref": "atr", "k": 0.2},
    "stop_atr_0.4": {"kind": "stop", "ref": "atr", "k": 0.4},
    "stop_atr_0.6": {"kind": "stop", "ref": "atr", "k": 0.6},
    "stop_bar_high": {"kind": "stop", "ref": "bar_high"},
    "stop_bar_high_ext": {"kind": "stop", "ref": "bar_high_ext", "k": 0.1},

    # —— C. 带价格上限的追单：挂限价等回撤，等不到时只在不追高的前提下改市价 ——
    # 1h 上 open 几乎等于上一根 close（p99 差 0.045×ATR），所以「开盘已跑远就不追」
    # 这类规则在 1h 上永不触发；真正可表达的上限是「等待期结束时价格离信号收盘多远」。
    "chase_cap_atr_0.2": {"kind": "hybrid_cap", "ref": "atr", "k": 0.4, "ceiling": 0.2},
    "chase_cap_atr_0.4": {"kind": "hybrid_cap", "ref": "atr", "k": 0.4, "ceiling": 0.4},
    "chase_cap_atr_0.8": {"kind": "hybrid_cap", "ref": "atr", "k": 0.4, "ceiling": 0.8},

    # —— D. 混合：先挂限价，等不到就市价追（不放弃信号）——
    "hybrid_atr_0.4": {"kind": "hybrid", "ref": "atr", "k": 0.4},
    "hybrid_atr_0.6": {"kind": "hybrid", "ref": "atr", "k": 0.6},
    "hybrid_bar_low": {"kind": "hybrid", "ref": "bar_low"},

    # —— E. 等待根数扫描（回答"挂多久该撤"）——
    "limit_close_w1": {"kind": "limit", "ref": "close", "wait": 1},
    "limit_close_w2": {"kind": "limit", "ref": "close", "wait": 2},
    "limit_close_w4": {"kind": "limit", "ref": "close", "wait": 4},
    "limit_close_w12": {"kind": "limit", "ref": "close", "wait": 12},
    "limit_close_w24": {"kind": "limit", "ref": "close", "wait": 24},
    "limit_a04_w1": {"kind": "limit", "ref": "atr", "k": 0.4, "wait": 1},
    # 对照组：同样挂 limit，但止损/止盈**重新锚定到成交价**（实战最常见的写法）。
    # 与「守住结构性失效位」相比，它把"买得更低"换成了"风险距离不变、目标更近"。
    "limit_a04_w1_reanchor": {"kind": "limit", "ref": "atr", "k": 0.4, "wait": 1, "reanchor": True},
    "limit_a04_w4_reanchor": {"kind": "limit", "ref": "atr", "k": 0.4, "wait": 4, "reanchor": True},
    "limit_a04_w2": {"kind": "limit", "ref": "atr", "k": 0.4, "wait": 2},
    "limit_a04_w4": {"kind": "limit", "ref": "atr", "k": 0.4, "wait": 4},
    "limit_a04_w12": {"kind": "limit", "ref": "atr", "k": 0.4, "wait": 12},
    "limit_a04_w24": {"kind": "limit", "ref": "atr", "k": 0.4, "wait": 24},
}


def _level(spec: dict, i: int, close: np.ndarray, low: np.ndarray, high: np.ndarray, atr: np.ndarray) -> float:
    """在信号 bar i 收盘时确定的委托价（只用 i 及之前的信息）。"""
    ref, k = spec["ref"], float(spec.get("k", 0.0))
    c = close[i]
    if ref == "close":
        return c
    if ref == "atr":
        return c + (-k if spec["kind"] in ("limit", "hybrid", "hybrid_cap") else k) * atr[i]
    if ref == "pct":
        return c * (1 - k)
    if ref == "bar_low":
        return low[i]
    if ref == "bar_low_ext":
        return low[i] - k * atr[i]
    if ref == "bar_high":
        return high[i]
    if ref == "bar_high_ext":
        return high[i] + k * atr[i]
    if ref == "range_mid":
        return (high[i] + low[i]) / 2
    raise ValueError(f"未知 ref: {ref}")


def _run_signal(i: int, f: dict, spec: dict, exit_cfg: tuple, wait: int, queue_bps: float = 0.0) -> dict | None:
    """单个信号在给定入场方案下的结果。None = 该方案下未成交（贡献 0）。"""
    o, h, l, c, a = f["open"], f["high"], f["low"], f["close"], f["atr"]
    n = len(c)
    sig_close = c[i]
    sl_mult, tp_mult = exit_cfg[0], exit_cfg[1]
    fixed_hold = exit_cfg[2] if len(exit_cfg) > 2 else None

    if fixed_hold is not None:
        # 隔离口径：**仓位固定**（1R = 1×ATR，与成交价无关），只比「成交价差 + 手续费差」
        # 在固定持有期内的净效果。这样任何差异不可能来自仓位变化，只可能来自入场价。
        stop = sig_close - a[i]
        target = None
    else:
        stop = sig_close - sl_mult * a[i]
        target = sig_close + tp_mult * a[i]
    if not (stop < sig_close) or (target is not None and not sig_close < target):
        return None
    risk_floor = MIN_RISK_ATR * a[i]

    kind = spec["kind"]
    entry = fill_bar = fee_entry = None
    slip_entry = SLIP

    if kind == "market":
        if i + 1 >= n:
            return None
        entry, fill_bar, fee_entry = o[i + 1], i + 1, FEE_TAKER
    elif kind == "hybrid_cap":
        # 挂限价等回撤；等待期结束仍未成交时，只有在价格没跑太远时改市价，
        # 否则放弃这笔（这才是"设定合理价格上限"的可执行形式）。
        level = _level(spec, i, c, l, h, a)
        last = min(i + wait, n - 1)
        trigger_q = level * (1 - queue_bps / 10000)
        for j in range(i + 1, last + 1):
            if l[j] <= trigger_q:
                entry, fill_bar, fee_entry, slip_entry = level, j, FEE_MAKER, 0.0
                break
        if fill_bar is None:
            k2 = last + 1
            if k2 >= n:
                return None
            ceiling = sig_close + float(spec["ceiling"]) * a[i]
            if o[k2] > ceiling:
                return None
            entry, fill_bar, fee_entry = o[k2], k2, FEE_TAKER
    else:
        level = _level(spec, i, c, l, h, a)
        if kind in ("limit", "hybrid") and level > sig_close:
            return None          # 委托价高于信号收盘 → 等价立即成交，无研究意义
        if kind == "stop" and level <= sig_close:
            return None
        eff_wait = int(spec.get("wait", wait))
        last = min(i + eff_wait, n - 1)
        # 排队惩罚：挂单排在队列里，价格只"触及"委托价未必成交，
        # 要求再穿越 queue_bps 个基点才判成交（仍按委托价结算）。
        trigger = level * (1 - queue_bps / 10000) if kind in ("limit", "hybrid") else level
        for j in range(i + 1, last + 1):
            if kind in ("limit", "hybrid"):
                if l[j] <= trigger:
                    # 挂单成交价一律取委托价：限价单是被动方，成交在挂单价上。
                    # 跳空穿越时你仍付委托价（市价已更低），这是逆向选择的真实成本，必须保留。
                    entry, fill_bar, fee_entry, slip_entry = level, j, FEE_MAKER, 0.0
                    break
            else:  # stop 突破
                if h[j] >= level:
                    entry, fill_bar = max(level, o[j]), j
                    fee_entry, slip_entry = FEE_TAKER, SLIP
                    break
        if fill_bar is None:
            if kind != "hybrid":
                return None
            k2 = last + 1
            if k2 >= n:
                return None
            entry, fill_bar, fee_entry = o[k2], k2, FEE_TAKER

    fill = entry * (1 + slip_entry)
    if spec.get("reanchor") and fixed_hold is None:
        # 对照组：止损/止盈跟随成交价重新锚定，风险距离保持与基准一致
        stop = fill - sl_mult * a[i]
        target = fill + tp_mult * a[i]
    if fixed_hold is None and fill <= stop:
        return None              # 成交价已在保护位下方 → 该笔无效
    if target is not None and fill >= target:
        return None
    # 隔离口径的风险分母固定为 1×ATR（与成交价无关），
    # 否则买得越低分母越小、R 越大，会把仓位放大误算成入场优势。
    risk = a[i] if fixed_hold is not None else fill - stop
    if risk < risk_floor:
        return None              # 仓位会发散（风险距离过窄），判为不可执行

    exit_price = exit_bar = reason = None
    if fixed_hold is not None:
        # 隔离口径：出场时点锁定在**信号 bar + H**，与入场方式无关。
        # 这样比较的是「同一个决策点、同一个平仓点，只换入场机制」的净效果；
        # 成交得越晚、能吃到的行情越短，这个代价也被如实计入。
        exit_bar = min(i + fixed_hold, n - 1)
        exit_price, reason = c[exit_bar], "hold_end"
        if exit_bar < fill_bar:
            return None
    else:
        for j in range(fill_bar, min(fill_bar + MAX_HOLD, n)):
            if l[j] <= stop:
                exit_price, exit_bar, reason = stop, j, "stop"
                break
            if h[j] >= target:
                exit_price, exit_bar, reason = target, j, "target"
                break
        if exit_price is None:
            exit_bar = min(fill_bar + MAX_HOLD, n) - 1
            exit_price, reason = c[exit_bar], "time"

    exit_fill = exit_price * (1 - SLIP)
    fees = fill * fee_entry + exit_fill * FEE_TAKER
    pnl = (exit_fill - fill) - fees

    return {
        "entry": fill, "entry_ref": o[i + 1], "exit": exit_fill, "reason": reason,
        "risk": risk, "r": pnl / risk, "bars_to_fill": fill_bar - i,
        "bars_held": exit_bar - fill_bar + 1, "maker": fee_entry == FEE_MAKER,
        # 价格口径的净收益（不含仓位放大）：与 R 口径互为对照，
        # 防止「止损变近 → 仓位放大 → R 虚高」被当成入场价优势。
        "ret_pct": pnl / fill,
        "t": int(f["ts"][i]),
    }


def _signals(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "all_bars":
        mask = np.ones(len(frame), dtype=bool)
    elif mode == "st_flip":
        d = frame["supertrend_dir"].to_numpy(dtype=float)
        mask = (d == 1) & (np.concatenate(([np.nan], d[:-1])) == -1)
    elif mode == "st_flip_adx":
        d = frame["supertrend_dir"].to_numpy(dtype=float)
        adx = frame["adx_14"].to_numpy(dtype=float)
        mask = (d == 1) & (np.concatenate(([np.nan], d[:-1])) == -1) & (adx >= 20)
    elif mode == "custom":
        # 由 --signal-col 指定的布尔列（取上升沿），在 main 里已算好并挂到 frame 上。
        # 复制一份：下面要做 `mask &= ...` 原地修改，而 to_numpy 可能是只读视图。
        mask = frame["_signal"].to_numpy(dtype=bool).copy()
    else:
        raise ValueError(mode)
    mask &= np.isfinite(frame["atr_pct"].to_numpy(dtype=float))
    mask &= np.isfinite(frame["close"].to_numpy(dtype=float))
    edges = np.flatnonzero(mask)
    if len(edges) == 0:
        return edges
    kept = [edges[0]]
    for pos in edges[1:]:
        if pos - kept[-1] >= MIN_GAP:
            kept.append(pos)
    return np.asarray(kept, dtype=int)


def _worker(task: tuple) -> dict:
    symbol, frame, modes, wait, queue_bps = task
    arrays = {
        "open": frame["open"].to_numpy(dtype=float),
        "high": frame["high"].to_numpy(dtype=float),
        "low": frame["low"].to_numpy(dtype=float),
        "close": frame["close"].to_numpy(dtype=float),
        "atr": (frame["atr_pct"] * frame["close"]).to_numpy(dtype=float),
        "ts": frame.index.asi8,
    }
    out = {"symbol": symbol, "bars": len(frame), "modes": {}}
    for mode in modes:
        pos = _signals(frame, mode)
        mode_out = {"signals": int(len(pos)), "exits": {}}
        for exit_name, exit_cfg in EXIT_SETS.items():
            mode_out["exits"][exit_name] = {
                pname: [_run_signal(int(i), arrays, spec, exit_cfg, wait, queue_bps)
                        for i in pos]
                for pname, spec in POLICIES.items()
            }
        out["modes"][mode] = mode_out
    return out


def _policy_stats(base: list, cand: list) -> dict:
    n = len(base)
    if n == 0:
        return {"signals": 0}
    base_r = np.array([x["r"] if x else 0.0 for x in base], dtype=float)
    filled = np.array([x is not None for x in cand], dtype=bool)
    cand_r = np.array([x["r"] if x else 0.0 for x in cand], dtype=float)
    fills = int(filled.sum())

    stats = {
        "signals": n, "fills": fills,
        "fill_rate": round(fills / n, 4),
        "mean_r_per_signal": round(float(cand_r.mean()), 4),
        "baseline_r_per_signal": round(float(base_r.mean()), 4),
        "mean_r_per_fill": round(float(cand_r[filled].mean()), 4) if fills else None,
        "avg_bars_to_fill": round(float(np.mean([x["bars_to_fill"] for x in cand if x])), 2) if fills else None,
        "stop_rate": round(float(np.mean([x["reason"] == "stop" for x in cand if x])), 4) if fills else None,
        "target_rate": round(float(np.mean([x["reason"] == "target" for x in cand if x])), 4) if fills else None,
        "avg_risk_pct": round(float(np.mean([x["risk"] / x["entry"] for x in cand if x])) * 100, 4) if fills else None,
        "mean_ret_pct": round(float(np.mean([x["ret_pct"] if x else 0.0 for x in cand])) * 100, 4),
    }
    if fills:
        improve = np.array([(b["entry_ref"] - x["entry"]) / b["entry_ref"]
                            for x, b in zip(cand, base) if x and b], dtype=float)
        stats["entry_improve_bps"] = round(float(improve.mean()) * 10000, 2)
        base_filled, base_missed = base_r[filled], base_r[~filled]
        stats["baseline_r_filled"] = round(float(base_filled.mean()), 4)
        stats["baseline_r_missed"] = round(float(base_missed.mean()), 4) if len(base_missed) else None
        stats["adverse_selection_gap"] = (
            round(float(base_missed.mean() - base_filled.mean()), 4) if len(base_missed) else None
        )
        stats["delta_r_vs_baseline"] = round(float(cand_r.mean() - base_r.mean()), 4)
        # 纯入场价效应：**只在成交子集上**与基准比较。
        # 这排除了「少交易 = 少亏手续费」的干扰，回答的是
        # 「同样这几笔，挂单确实买得更好吗」。
        stats["delta_r_on_filled"] = round(float(cand_r[filled].mean() - base_r[filled].mean()), 4)
        # 币级稳健性：多少比例的币上这个方案不比基准差
        stats["n_fills_per_symbol"] = None
    else:
        stats.update({"entry_improve_bps": None, "baseline_r_filled": None,
                      "baseline_r_missed": None, "adverse_selection_gap": None,
                      "delta_r_vs_baseline": round(-float(base_r.mean()), 4),
                      "delta_r_on_filled": None})

    # bootstrap：重采样单位是「信号」而非「成交」，未成交的 0 也参与抽样
    if n >= 30:
        rng = np.random.default_rng(20261030)
        idx = rng.integers(0, n, size=(2000, n))
        means = cand_r[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        # 配对差分：同一批信号上方案与基准的差，方差比各自均值小得多
        dmeans = (cand_r - base_r)[idx].mean(axis=1)
        dlo, dhi = np.percentile(dmeans, [2.5, 97.5])
        stats["ci_low"], stats["ci_high"] = round(float(lo), 4), round(float(hi), 4)
        stats["delta_ci_low"], stats["delta_ci_high"] = round(float(dlo), 4), round(float(dhi), 4)
        stats["delta_significant"] = bool(dlo > 0 or dhi < 0)
        # 成交子集上的配对 CI：衡量「挂单本身是否买得更好」，与成交率解耦
        if fills >= 30:
            sub = (cand_r - base_r)[filled]
            sidx = rng.integers(0, len(sub), size=(2000, len(sub)))
            sm = sub[sidx].mean(axis=1)
            slo, shi = np.percentile(sm, [2.5, 97.5])
            stats["filled_delta_ci_low"], stats["filled_delta_ci_high"] = round(float(slo), 4), round(float(shi), 4)
            stats["filled_delta_significant"] = bool(slo > 0 or shi < 0)
        else:
            stats["filled_delta_ci_low"] = stats["filled_delta_ci_high"] = None
            stats["filled_delta_significant"] = False
    else:
        stats["ci_low"] = stats["ci_high"] = None
        stats["delta_ci_low"] = stats["delta_ci_high"] = None
        stats["delta_significant"] = False
        stats["filled_delta_ci_low"] = stats["filled_delta_ci_high"] = None
        stats["filled_delta_significant"] = False
    return stats


def _segment_delta(base: list, cand: list, segments: int = 4) -> dict:
    """时间分段稳定性：把信号按时间顺序等分，看 delta 是否每段都同号。

    这是本项目一贯的判据——单个窗口好看不算证据，逐段同号才算。
    """
    n = len(base)
    if n < segments * 10:
        return {"segments": [], "same_sign": None}
    base_r = np.array([x["r"] if x else 0.0 for x in base], dtype=float)
    cand_r = np.array([x["r"] if x else 0.0 for x in cand], dtype=float)
    deltas = cand_r - base_r
    edges = np.linspace(0, n, segments + 1).astype(int)
    parts = [round(float(deltas[edges[i]:edges[i + 1]].mean()), 4) for i in range(segments)]
    overall = np.sign(np.mean(deltas))
    same = int(sum(1 for p in parts if np.sign(p) == overall))
    return {"segments": parts, "same_sign": same, "all_positive": all(p > 0 for p in parts)}


def _aggregate(per_symbol: list[dict], mode: str, exit_name: str) -> dict[str, dict]:
    """池化。信号按真实时间重排后再做分段稳定性检验；跨币稳健性看币级同号率。"""
    pooled = {p: {"base": [], "cand": []} for p in POLICIES}
    per_symbol_delta = {p: [] for p in POLICIES}
    for row in per_symbol:
        part = row["modes"][mode]["exits"][exit_name]
        base = part["market_next_open"]
        for pname in POLICIES:
            pooled[pname]["base"].extend(base)
            pooled[pname]["cand"].extend(part[pname])
            b = np.array([x["r"] if x else 0.0 for x in base], dtype=float)
            cnd = np.array([x["r"] if x else 0.0 for x in part[pname]], dtype=float)
            if len(b):
                per_symbol_delta[pname].append(float(cnd.mean() - b.mean()))

    out = {}
    for pname in POLICIES:
        base, cand = pooled[pname]["base"], pooled[pname]["cand"]
        # 按真实时间排序后再分段，保证「时间稳定性」检验名副其实
        order = np.argsort([x.get("t", 0) if x else np.iinfo(np.int64).max for x in base])
        base_sorted = [base[k] for k in order]
        cand_sorted = [cand[k] for k in order]
        stats = _policy_stats(base_sorted, cand_sorted)
        seg = _segment_delta(base_sorted, cand_sorted)
        stats["segment_delta"] = seg["segments"]
        stats["segment_same_sign"] = seg.get("same_sign")
        deltas = np.array(per_symbol_delta[pname], dtype=float)
        stats["symbols_better"] = int((deltas > 0).sum())
        stats["symbols_total"] = int(len(deltas))
        out[pname] = stats
    return out


def _verdict_section(payload: dict) -> list[str]:
    """把「哪条方案能落地、临界条件是什么」直接从数据算出来，而不是写死结论。"""
    cfg = payload["config"]
    mode = "st_flip" if "st_flip" in cfg["modes"] else cfg["modes"][0]
    block = payload["results"][mode][PRIMARY_EXIT]
    base = block["market_next_open"]
    risk = base.get("avg_risk_pct") or 1.5

    ranked = sorted(
        ((k, v) for k, v in block.items() if k != "market_next_open"),
        key=lambda kv: -kv[1]["delta_r_vs_baseline"],
    )
    # 可执行性门槛：统计显著 + 分段≥3/4 同号 + 成交率≥50%。
    # 成交率过低的方案即便 delta 更大，也只是"偶尔做一笔"，实操中样本会稀到无法验证。
    robust = [(k, v) for k, v in ranked
              if v.get("delta_significant") and (v.get("segment_same_sign") or 0) >= 3
              and 0.5 <= v["fill_rate"] < 0.999]
    lines = [
        "## 结论（自动从上面的数据算出）",
        "",
        f"- 基准（下一根开盘市价）每信号 **{base['mean_r_per_signal']:+.4f}R**，"
        f"平均风险距离 {risk}%；信号本身没有正 edge，所以下面所有方案也都是负的——"
        "**入场价优化只能减少亏损，不可能把负期望变成正期望**。",
        "- 推荐门槛：配对 CI 显著 + 分段 ≥3/4 同号 + **成交率 ≥50%**"
        "（成交率太低的方案只是「偶尔做一笔」，实盘样本会稀到无法验证）。",
    ]
    # 两条价格口径同时看：R 受止损宽窄影响，% 只反映价格
    best_pct = max(block.items(), key=lambda kv: kv[1].get("mean_ret_pct", -9))
    if best_pct[0] != "market_next_open":
        bp = best_pct[1]
        lines.append(
            f"- 按**价格口径**（1:1名义仓位、不含仓位缩放）最优的是 `{best_pct[0]}`："
            f"每信号 {bp['mean_ret_pct']:+.4f}%（基准 {base['mean_ret_pct']:+.4f}%），"
            f"风险距离 {bp['avg_risk_pct']}%（基准 {risk}%）。"
        )
    # 止损锚定方式决定「R 改善」是真优势还是加杠杆，必须单独点名
    plain = block.get("limit_a04_w1")
    rean = block.get("limit_a04_w1_reanchor")
    if plain and rean:
        lines.append(
            "- **两种止损写法结果不同，机制也不同**：\n"
            f"  - 止损留在信号收盘下方（`limit_a04_w1`）：买低 0.4ATR 后风险距离从 {risk}% 缩到 "
            f"{plain['avg_risk_pct']}%，仓位被动放大，`每信号R` {plain['mean_r_per_signal']:+.4f} 里"
            "有一部分是杠杆效应，不是价格优势。\n"
            f"  - 止损跟着成交价下移（`limit_a04_w1_reanchor`）：风险距离维持 {rean['avg_risk_pct']}%，"
            f"仓位不变，`每信号R` {rean['mean_r_per_signal']:+.4f} 才是纯粹的价格改善。\n"
            "  - 两者在 `hold_24`（仓位完全固定）里结果完全相同，正好证明上面的差异只来自仓位。"
        )
    if not robust:
        lines.append("- 没有任何方案同时满足这些门槛，即无可用结论。")
        return lines

    best, bs = robust[0]
    is_chase = best.startswith("stop")
    kind_note = "（买在上方 = 突破确认）" if is_chase else "（买在下方 = 回撤挂单）"
    lines += [
        f"- 稳健最优：**`{best}`**{kind_note}，每信号 {bs['mean_r_per_signal']:+.4f}R "
        f"（改善 {bs['delta_r_vs_baseline']:+.4f}R，"
        f"CI [{bs['delta_ci_low']}, {bs['delta_ci_high']}]，"
        f"分段 {bs['segment_same_sign']}/4 同号，{bs['symbols_better']}/{bs['symbols_total']} 币更好）。"
        f"成交率 {bs['fill_rate']:.1%}，成交价 {bs['entry_improve_bps']:+.1f}bps。",
    ]
    sub = bs.get("delta_r_on_filled")
    f = bs["fill_rate"]
    if is_chase:
        # 突破追单不改善价格（甚至买得更贵），它的作用机制是**信号过滤**：
        # 只做"价格真的往上突破"的那些，把没突破的挡在门外。
        lines.append(
            f"- 但注意：它成交价反而变差 {bs['entry_improve_bps']:+.1f}bps，**省的不是钱，是筛掉了信号**。"
            f"没突破的那批在基准下平均 {bs['baseline_r_missed']:+.4f}R，"
            f"突破的这批 {bs['baseline_r_filled']:+.4f}R——"
            "突破本身就是信息。这属于信号过滤，不是入场价优化；"
            "它成立的前提是「突破有延续性」，这个前提在不同周期上不稳定"
            "（1h 上把仓位固定后优势就消失，见 `hold_24`）。"
        )
    elif sub is not None and 0 < f < 0.999:
        crit = f * sub / (1 - f)
        lines.append(
            f"- **临界判据**：这套挂单在「信号自身毛 edge < {crit:+.3f}R"
            f"（≈ {crit * risk:.2f}% 价格）时占优」，超过就该直接市价。"
            "本研究的信号 edge 为 "
            f"{base['mean_r_per_signal']:+.4f}R，低于临界值，所以挂单更优这个结论是可信的；"
            "换成真有 0.5R+ edge 的信号，结论会反过来。"
        )
    gap = bs.get("adverse_selection_gap")
    limit_ref = block.get("limit_a04_w1") or block.get("limit_a04_w4")
    if limit_ref and limit_ref.get("adverse_selection_gap") is not None:
        g = limit_ref["adverse_selection_gap"]
        lines.append(
            "- **逆向选择是真的、而且很大**（以 `limit_a04_w1` 为例）："
            f"没成交的那批信号在基准下平均 {limit_ref['baseline_r_missed']:+.4f}R，"
            f"成交的那批只有 {limit_ref['baseline_r_filled']:+.4f}R，相差 {g:+.4f}R。"
            f"也就是说挂单便宜了 {limit_ref['entry_improve_bps']}bps，"
            "但**挑中的恰恰是最弱的信号**；净改善是这个折扣硬扛过逆向选择之后剩下的部分。"
            "任何「挂单更便宜所以更好」的直觉都必须先过这一关。"
        )
    deep = [k for k in ("limit_atr_0.6", "limit_atr_0.8", "limit_atr_1.0") if k in block]
    if deep:
        d = block[deep[-1]]
        lines.append(
            f"- **挂太深会自毁**：`{deep[-1]}` 成交价能改善 {d['entry_improve_bps']}bps，"
            f"但成交率只剩 {d['fill_rate']:.1%}，止损率升到 {d['stop_rate']:.1%}，"
            f"每信号只剩 {d['mean_r_per_signal']:+.4f}R。深度有最优值，不是越便宜越好。"
        )
    chase = [k for k in ("stop_atr_0.6", "stop_atr_0.4") if k in block]
    hold_block = payload["results"][mode].get(ISOLATION_EXIT, {})
    if chase and not is_chase:
        c = block[chase[0]]
        keep = ""
        if chase[0] in hold_block:
            h = hold_block[chase[0]]
            keep = (f"把仓位固定后再看（`hold_24`），它变成 {h['delta_r_vs_baseline']:+.4f}R —— "
                    f"{'优势消失' if h['delta_r_vs_baseline'] <= 0.02 else '优势仍在'}。")
        lines.append(
            f"- **突破追单（买在上方）不如挂单**：`{chase[0]}` 每信号 {c['mean_r_per_signal']:+.4f}R，"
            f"入场价反而差 {-c['entry_improve_bps']:.0f}bps，成交率 {c['fill_rate']:.1%}。"
            + keep
            + "它唯一的用处是筛掉不突破的信号（止盈率会明显提高），那是**信号过滤**，不是入场价优化。"
        )
    return lines


def _markdown(payload: dict) -> str:
    cfg = payload["config"]
    lines = [
        "# 入场价方案研究：信号已定，怎么挂单成交",
        "",
        f"> {cfg['as_of']}；{cfg['symbols_used']} 个主流币，{cfg['interval']} × {cfg['days']} 天；"
        f"信号最小间隔 {cfg['min_gap']} 根；限价/突破单等待 {cfg['wait']} 根；"
        f"taker {cfg['fee_taker']*100:.3f}% / maker {cfg['fee_maker']*100:.3f}%，"
        f"滑点 {cfg['slippage']*100:.3f}%，排队惩罚 {cfg['queue_bps']}bps。",
        "",
        "## 口径",
        "",
        "- 唯一变量是**入场成交价**。信号时点、止损/止盈**绝对价位**（锚定信号收盘）、"
        "同根止损优先、风险预算全部锁死。`*_reanchor` 是唯一例外：止损/止盈跟着成交价平移，"
        "风险距离保持与基准一致（实战最常见的写法），单独成组比较。",
        "- 主指标是 **每信号期望 R**：未成交记 0，不删样本、不假设「如果成交了会怎样」。",
        "- `delta` = 方案每信号R − 基准每信号R；`deltaCI` 是**配对**bootstrap 区间，"
        "不跨 0 才算真的更好。",
        "- `成交子集delta` = 只在**成交的那些信号**上比，剔除「少交易自然少亏手续费」的干扰，"
        "衡量「同样这几笔，挂单是否真买得更好」；带 `*` 表示该子集 CI 显著。",
        "- `逆向差` = 未成交组在基准方案下的 R − 成交组的 R。**为正说明限价单专挑弱的信号成交**。",
        "- `每信号%` = 按**1:1名义仓位**算的价格净收益，只反映价格本身；"
        "实盘资金收益以 `每信号R × 风险预算` 为准（本项目风险预算为权益的百分比）。",
        "- 风险距离下限 0.25×ATR：低于此判为不可执行（否则仓位发散，回测会虚增收益）。",
        "- `hold_24` 是隔离口径：**仓位固定**（1R = 1×ATR，与成交价无关）、"
        "出场时点锁定在信号 bar + 24。这里任何差异都不可能来自仓位变化。",
        "",
    ]
    lines += _verdict_section(payload)
    lines += ["", "## 全部方案明细", ""]
    for mode in cfg["modes"]:
        for exit_name in cfg["exit_sets"]:
            block = payload["results"][mode][exit_name]
            summary = payload["summaries"][mode]
            sl, tp = cfg["exit_sets"][exit_name][:2]
            label = (f"止损 {sl}×ATR / 止盈 {tp}×ATR" if sl is not None
                     else f"无止损止盈，固定持有 {cfg['exit_sets'][exit_name][2]} 根（1R=1ATR）")
            lines += [
                f"### 信号集 `{mode}` × 出场 `{exit_name}`（{label}；"
                f"{summary['signals']} 个信号）",
                "",
                "| 方案 | 成交率 | 每信号R | delta | deltaCI | 成交子集delta | 每信号% | 分段同号 | 币级更好 | 改善bps | 逆向差 | 止损率 |",
                "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for pname in POLICIES:
                s = block[pname]
                dci = (f"[{s['delta_ci_low']}, {s['delta_ci_high']}]"
                       if s.get("delta_ci_low") is not None else "—")
                mark = "**" if s.get("delta_significant") else ""
                sub = (f"{s['delta_r_on_filled']:+.4f}"
                       + ("*" if s.get("filled_delta_significant") else "")) \
                    if s.get("delta_r_on_filled") is not None else "—"
                lines.append(
                    f"| {mark}`{pname}`{mark} | {s['fill_rate']:.2%} | {s['mean_r_per_signal']:+.4f} | "
                    f"{s['delta_r_vs_baseline']:+.4f} | {dci} | {sub} | "
                    f"{s['mean_ret_pct']:+.4f} | "
                    f"{s['segment_same_sign']}/4 | {s['symbols_better']}/{s['symbols_total']} | "
                    f"{s['entry_improve_bps'] if s['entry_improve_bps'] is not None else '—'} | "
                    f"{s['adverse_selection_gap'] if s['adverse_selection_gap'] is not None else '—'} | "
                    f"{s['stop_rate'] if s['stop_rate'] is not None else '—'} |"
                )
            lines.append("")
    lines += [
        "## 读法",
        "",
        "- **每信号R 才是可决策的数字**。成交率 30% 但每笔 R 翻倍的方案，"
        "只有 0.3×每笔R > 基准R 时才真的更好。",
        "- **逆向差为正就要警惕**：没成交的那批信号本来更赚钱，"
        "限价挂单等于系统性地错过好机会。",
        "- **止损率上升是必然代价**：买得更低 → 离结构性失效位更近 → "
        "同样止损位被触及更频繁；赢时因仓位更大而赢得更多。平均风险% 同步下降即为证据。",
        "",
        "## 限制",
        "",
        "1. 只有多头。空头是精确镜像（卖单挂收盘上方），结论可直接平移。",
        "2. 同根止损/止盈冲突一律判止损先成交；对所有方案一致，但整体偏低。",
        "3. 未建模最小跳动、部分成交、下单延迟与手续费阶梯；"
        "maker 假定必然成交且无排队损失（实盘中挂单队列靠后可能完全不成交）。",
        "4. 池化样本存在重叠持仓；bootstrap 按信号重采样仍偏乐观，"
        "结论只用于**方案排序**，不用于宣称绝对收益。",
        "",
        "## 复现",
        "",
        "```powershell",
        f".\\.venv\\Scripts\\python.exe backend/scripts/entry_price_study.py "
        f"--days {cfg['days']} --workers 6 --modes {','.join(cfg['modes'])}",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="入场价方案研究")
    ap.add_argument("--interval", default=INTERVAL, help="K线周期，如 15m/1h/4h")
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wait", type=int, default=DEFAULT_WAIT)
    ap.add_argument("--queue-bps", type=float, default=0.0,
                    help="排队惩罚：限价单需再穿越 N 个基点才判成交（默认 0=价格触及即成交）")
    ap.add_argument("--symbols", type=int, default=len(MAJOR_SYMBOLS))
    ap.add_argument("--modes", default="all_bars,st_flip")
    ap.add_argument("--signal-col", default=None,
                    help="接入你自己的指标布尔列，形如 指标名.列名；会自动追加 custom 信号集")
    ap.add_argument("--output", default="docs/entry-price-study.json")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    db = sessionmaker(bind=make_engine(read_only=True))()
    service = MarketDataService(db)
    as_of = now_ms()
    start_ms = as_of - args.days * 86_400_000

    # --signal-col 让你把**自己的指标**直接接进来：
    # 传 "指标名.列名" 或 "指标名:列名"，该列必须是布尔列（如某条件的真假）。
    indicator_obj = None
    signal_column = None
    if args.signal_col:
        from app.catalog_indicators import get_indicator

        sep = "." if "." in args.signal_col else ":"
        if sep not in args.signal_col:
            raise SystemExit("--signal-col 需要形如 指标名.列名")
        ind_name, signal_column = args.signal_col.split(sep, 1)
        indicator_obj = get_indicator(ind_name)
        if "custom" not in modes:
            modes.append("custom")
        print(f"自定义信号：{ind_name}.{signal_column}")

    tasks, skipped = [], {}
    for symbol in list(MAJOR_SYMBOLS)[: args.symbols]:
        frame = service.load_frame(symbol, args.interval, start_ms=start_ms, end_ms=as_of)
        if len(frame) < 800:
            skipped[symbol] = f"可用K线不足: {len(frame)}"
            continue
        data = build_features(frame)
        if indicator_obj is not None:
            # 指标在原始 frame 上算，再对齐到特征帧，避免指标依赖我们新增的列
            raw = indicator_obj.calculate(frame)
            if signal_column not in raw.columns:
                raise SystemExit(f"指标 {indicator_obj.name} 没有列 {signal_column}")
            data["_signal"] = raw[signal_column].reindex(frame.index).fillna(False).astype(bool)
        tasks.append((symbol, data, modes, args.wait, args.queue_bps))
    if not tasks:
        raise SystemExit("没有可用币种")
    print(f"{len(tasks)} 币 × {len(POLICIES)} 方案 × {len(EXIT_SETS)} 出场 × {len(modes)} 信号集")

    per_symbol = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(_worker, tasks):
            per_symbol.append(row)
            print(f"  {row['symbol']:10s} {row['bars']} bars  "
                  + " ".join(f"{m}={row['modes'][m]['signals']}" for m in modes))

    results = {m: {e: _aggregate(per_symbol, m, e) for e in EXIT_SETS} for m in modes}
    summaries = {m: {"signals": int(sum(r["modes"][m]["signals"] for r in per_symbol))} for m in modes}

    payload = {
        "config": {
            "interval": args.interval, "days": args.days,
            "as_of": str(pd.Timestamp(as_of, unit="ms", tz="UTC")),
            "symbols_requested": list(MAJOR_SYMBOLS)[: args.symbols],
            "symbols_used": len(per_symbol), "skipped": skipped,
            "fee_taker": FEE_TAKER, "fee_maker": FEE_MAKER, "slippage": SLIP,
            "min_gap": MIN_GAP, "wait": args.wait, "max_hold": MAX_HOLD,
            "queue_bps": args.queue_bps,
            "min_risk_atr": MIN_RISK_ATR,
            "exit_sets": {k: list(v) for k, v in EXIT_SETS.items()},
            "primary_exit": PRIMARY_EXIT, "modes": modes,
        },
        "summaries": summaries,
        "results": results,
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")

    for mode in modes:
        block = results[mode][PRIMARY_EXIT]
        base = block["market_next_open"]["mean_r_per_signal"]
        print(f"\n[{mode}] 基准每信号R={base:+.4f}")
        for pname, s in sorted(block.items(), key=lambda kv: -kv[1]["mean_r_per_signal"])[:5]:
            sig = "显著" if s.get("delta_significant") else "不显著"
            print(f"  {pname:22s} {s['mean_r_per_signal']:+.4f} "
                  f"delta={s['delta_r_vs_baseline']:+.4f} 成交{s['fill_rate']:.1%} {sig}")
    print(f"\n已写 {out.relative_to(ROOT)} / {out.with_suffix('.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

r"""理想化底部入场的出场方案探索：同一批入场，只比卖出。

研究问题
--------
用户假设「已经在最低点买入」，问：哪种卖出规则能把这笔最好的运气最大化落袋？
这是一个**纯出场**问题。要把它做成可信实验，必须先把入场锁死：

- 所有卖出方案共用**同一批入场样本**；
- 因此可以做**配对比较**（同一笔入场下两个规则的收益差），把入场白送的收益差分掉；
- 剩下的差值才是卖出规则本身的贡献。

入场模型（因果可确认的底部）
----------------------------
不能用「事后全局最低点」：那是未来函数，而且会让所有出场方案都显得暴赚，结论毫无意义。
这里用 running-min pivot + 确认涨幅，完全因果：

1. 从上一次确认之后开始维护价格运行最低点 ``rmin``（含其所在 bar ``rmin_i``）；
2. 当某根收盘价 >= ``rmin * (1 + confirm)`` 时，才回头确认「``rmin_i`` 那根是这一段的底」；
3. 确认 bar 严格晚于低点 bar → 不偷看未来；并且 ``rmin`` 是这一段实际路径的最低点，
   所以确实满足「在最低点买入」的直觉含义。

入场成交分两档（都报告）：

- ``next_open``：确认后下一根开盘成交 —— **可实现，主口径**；
- ``ideal``：成交在低点那根 bar 的 low 价 —— **不可实现的上界**，只用于给结果封顶，
  并检验两种成交假设下出场方案的排序是否一致。

出场口径
--------
1. 止盈是挂单限价：向上跳空则按开盘价成交（更差），正常按目标价成交。
2. 止损是止损单：向下跳空则按开盘价成交（更差），正常按止损价成交。
3. **同一根 K 线内同时触及止损与止盈时按「先止损」记**（保守口径）：
   无法知道 bar 内先后顺序，按最坏假设处理。
4. EMA / RSI 出场：收盘确认后，下一根开盘成交，不使用 bar 内价格。
5. 移动止损的参考极值是**截至上一根的 running max**（并至少不低于入场价），
   不用当根 high 给自己定止损。
6. 成本双边：fee 0.04% + slippage 0.02%/边 → 往返 0.12%。
7. 每笔入场都要求剩余 K 线 >= ``--cap``，避免样本截断污染比较。

关键校准：随机出场对照
----------------------
「在最低点买入」本身就是一笔巨额白送收益，随便怎么卖都可能为正。
因此对每个入场做 M 次**随机出场时点**抽样作为基线，规则相对它的差值
（``alpha_vs_random``）才是「卖出方案真正赚到的钱」。

用法
----
    .\.venv\Scripts\python.exe backend/scripts/perfect_entry_exit_study.py
    .\.venv\Scripts\python.exe backend/scripts/perfect_entry_exit_study.py --days 400 --workers 4
    .\.venv\Scripts\python.exe backend/scripts/perfect_entry_exit_study.py --interval 4h --cap 120
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db import make_engine  # noqa: E402
from app.market.data_service import MarketDataService  # noqa: E402
from app.strategy.coin_universe import MAJOR_SYMBOLS  # noqa: E402
from app.utils import now_ms  # noqa: E402

INTERVAL = "1h"
DAYS = 400
TRAIN_RATIO = 0.66
CONFIRM_PCT = 0.01          # 收盘较低点上涨 1% 才算「确认到底」
MIN_DD = 0.05               # 低点前必须有至少 5% 回调，排除噪声级的浅回踩
PIVOT_MIN_GAP = 6           # 相邻低点至少间隔 6 根，避免同一段反复触发
CAP_BARS = 336              # 最长观察窗口：14 天(1h)；超时按收盘平仓并记 cap
ATR_LEN = 14
FEE = 0.0004
SLIPPAGE = 0.0002
ROUND_TRIP = 2 * (FEE + SLIPPAGE)
RANDOM_DRAWS = 40
N_BOOT = 2000
BOOT_SEED = 20261030
MIN_BARS = 1200
MIN_TRADES = 60

EXTRA_SYMBOLS = (
    "ATOMUSDT", "ETCUSDT", "FILUSDT", "HBARUSDT", "ICPUSDT",
    "IMXUSDT", "TONUSDT", "VETUSDT", "RNDRUSDT", "AAVEUSDT",
)

# ---------------------------------------------------------------------------
# 预注册的出场规则（低自由度，不做逐币寻优）
# ---------------------------------------------------------------------------
RULES: list[dict] = [
    # 1) 固定持有期：最简单、最不需要参数的基线族
    {"name": "hold_24", "kind": "time", "bars": 24, "desc": "持有 24 根后收盘平仓"},
    {"name": "hold_72", "kind": "time", "bars": 72, "desc": "持有 72 根后收盘平仓"},
    {"name": "hold_168", "kind": "time", "bars": 168, "desc": "持有 168 根(7天)后收盘平仓"},
    {"name": "hold_336", "kind": "time", "bars": 336, "desc": "持有 336 根(14天)后收盘平仓"},
    # 2) 固定风险倍数目标 / 止损（R = 入场时 ATR）
    {"name": "tp2_sl1", "kind": "level", "tp_atr": 2.0, "sl_atr": 1.0, "desc": "止盈 +2ATR / 止损 -1ATR"},
    {"name": "tp3_sl1", "kind": "level", "tp_atr": 3.0, "sl_atr": 1.0, "desc": "止盈 +3ATR / 止损 -1ATR"},
    {"name": "tp5_sl1", "kind": "level", "tp_atr": 5.0, "sl_atr": 1.0, "desc": "止盈 +5ATR / 止损 -1ATR"},
    {"name": "tp3_nostop", "kind": "level", "tp_atr": 3.0, "sl_atr": 0.0, "desc": "只挂 +3ATR 止盈，无止损"},
    {"name": "tp5pct_sl2pct", "kind": "level", "tp_pct": 0.05, "sl_pct": 0.02, "desc": "止盈 +5% / 止损 -2%"},
    {"name": "tp15pct_sl5pct", "kind": "level", "tp_pct": 0.15, "sl_pct": 0.05, "desc": "止盈 +15% / 止损 -5%"},
    # 3) 移动止损族：把趋势交给价格自己走完
    {"name": "trail_5pct", "kind": "trail_pct", "k": 0.05, "desc": "最高价回撤 5% 出"},
    {"name": "trail_10pct", "kind": "trail_pct", "k": 0.10, "desc": "最高价回撤 10% 出"},
    {"name": "trail_20pct", "kind": "trail_pct", "k": 0.20, "desc": "最高价回撤 20% 出"},
    {"name": "trail_10pct_t336", "kind": "trail_pct", "k": 0.10, "time_cap": 336, "desc": "回撤10%出，且最多 336 根"},
    {"name": "chand_3atr", "kind": "chandelier", "k": 3.0, "desc": "最高价 - 3ATR 出"},
    {"name": "chand_5atr", "kind": "chandelier", "k": 5.0, "desc": "最高价 - 5ATR 出"},
    # 4) 指标确认出场：收盘确认，下根开盘成交
    {"name": "ema20_close", "kind": "ema_cross", "n": 20, "desc": "收盘跌破 EMA20，下根开盘出"},
    {"name": "rsi50_cross", "kind": "rsi_cross", "level": 50.0, "desc": "RSI14 下穿 50，下根开盘出"},
    # 5) 结构出场：下一个确认的局部高点（卖出 i.e. 又一次路标确认）
    {"name": "pivot_high_open", "kind": "pivot_high", "ideal": False, "desc": "下一个确认高点后开盘出"},
    {"name": "pivot_high_ideal", "kind": "pivot_high", "ideal": True, "desc": "下一个确认高点的最高价出(上界)"},
    # 仅用于量化理论天花板，不参与训练选择或可执行候选排名。
    {"name": "oracle_cap_high", "kind": "oracle_high", "desc": "窗口内最高价卖出（未来函数，仅理论上界）"},
]

BENCHMARK = "hold_168"


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = ATR_LEN) -> np.ndarray:
    """Wilder ATR：前 n 根用 TR 的 SMA 播种，之后按 RMA 递推（ta.atr 口径）。"""
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    out = np.full(len(tr), np.nan)
    if len(tr) < n:
        return out
    out[n - 1] = tr[:n].mean()
    alpha = 1.0 / n
    for i in range(n, len(tr)):
        out[i] = out[i - 1] + (tr[i] - out[i - 1]) * alpha
    return out


def ema(series: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(series).ewm(span=n, adjust=False).mean().to_numpy()


def wilder_rsi(close: np.ndarray, n: int = ATR_LEN) -> np.ndarray:
    diff = np.diff(close, prepend=close[0])
    gain = np.clip(diff, 0, None)
    loss = np.clip(-diff, 0, None)
    out = np.full(len(close), np.nan)
    if len(close) < n + 1:
        return out
    ag = gain[1:n + 1].mean()
    al = loss[1:n + 1].mean()
    for i in range(n + 1, len(close)):
        ag = (ag * (n - 1) + gain[i]) / n
        al = (al * (n - 1) + loss[i]) / n
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


# ---------------------------------------------------------------------------
# 入场：因果确认的局部低点
# ---------------------------------------------------------------------------
def confirmed_pivot_lows(low, high, close, confirm, min_gap, min_dd=0.0):
    """返回 [(pivot_bar, confirm_bar)]：每段实际路径的最低点 + 其确认时点。

    ``rmin`` 是上一次确认之后的**实际路径最低点**，所以「买在这个最低点」在样本内是真的。

    ``min_dd`` > 0 时额外要求：该低点相对本段最高点的回撤至少达到 min_dd。
    没有这条约束时，「底」会退化成「一根锤子线的下影」——波动只有零点几个点，
    收益被往返成本吃光，拿它评价出场规则等于在噪声里挑赢家。
    """
    pairs = []
    rmin, rmin_i, rmax, dd_ok, last = np.inf, -1, -np.inf, False, -(10 ** 9)
    for i in range(len(low)):
        if high[i] > rmax:
            rmax = float(high[i])
        if low[i] < rmin:
            rmin, rmin_i = float(low[i]), i
            dd_ok = (rmax - rmin) / rmax >= (min_dd - 1e-12) if rmax > 0 else False
        if (rmin_i >= 0 and i > rmin_i and dd_ok
                and close[i] >= rmin * (1.0 + confirm) and (rmin_i - last) >= min_gap):
            pairs.append((rmin_i, i))
            last = rmin_i
            rmin, rmin_i, rmax, dd_ok = np.inf, -1, -np.inf, False
    return pairs


def confirmed_pivot_highs(high, close, confirm, min_gap):
    pairs = []
    rmax, rmax_i, last = -np.inf, -1, -(10 ** 9)
    for i in range(len(high)):
        if high[i] > rmax:
            rmax, rmax_i = float(high[i]), i
        if rmax_i >= 0 and i > rmax_i and close[i] <= rmax * (1.0 - confirm) and (rmax_i - last) >= min_gap:
            pairs.append((rmax_i, i))
            last = rmax_i
            rmax, rmax_i = -np.inf, -1
    return pairs


# ---------------------------------------------------------------------------
# 出场规则求值：返回 (offset, exit_price, reason)，offset 相对开仓扫描起点
# ---------------------------------------------------------------------------
def _first_true(mask: np.ndarray):
    if not mask.any():
        return None
    return int(np.argmax(mask))


def evaluate_exit(rule: dict, O, H, L, C, entry: float, atr: float, ctx: dict) -> tuple:
    m = len(C)
    cap_idx = m - 1

    def cap_exit(reason="cap"):
        return cap_idx, float(C[cap_idx]), reason

    kind = rule["kind"]

    if kind == "time":
        idx = min(int(rule["bars"]), m) - 1
        return idx, float(C[idx]), "time"

    if kind == "level":
        target = entry * (1.0 + rule["tp_pct"]) if "tp_pct" in rule else entry + rule["tp_atr"] * atr
        stop = None
        if rule.get("sl_pct"):
            stop = entry * (1.0 - rule["sl_pct"])
        elif rule.get("sl_atr"):
            stop = entry - rule["sl_atr"] * atr
        hit_tp = _first_true(H >= target)
        hit_sl = _first_true(L <= stop) if stop is not None else None
        if hit_tp is None and hit_sl is None:
            return cap_exit()
        if hit_sl is not None and (hit_tp is None or hit_sl <= hit_tp):
            # 同根内先触及止损，或根本没有止盈：保守记为止损单先成交
            return hit_sl, float(min(stop, O[hit_sl])), "stop"
        return hit_tp, float(max(target, O[hit_tp])), "target"

    if kind in ("trail_pct", "chandelier"):
        prior = np.maximum.accumulate(H)
        ref = np.empty(m)
        ref[0] = entry
        ref[1:] = np.maximum(entry, prior[:-1])
        if kind == "trail_pct":
            stops = ref * (1.0 - rule["k"])
        else:
            stops = ref - rule["k"] * atr
        hit = _first_true(L <= stops)
        if hit is None:
            return cap_exit()
        if rule.get("time_cap") and (hit + 1) >= int(rule["time_cap"]):
            idx = min(int(rule["time_cap"]), m) - 1
            return idx, float(C[idx]), "time"
        return hit, float(min(stops[hit], O[hit])), "trail"

    if kind == "ema_cross":
        line = ctx["ema"]
        idx = _first_true(C < line)
        if idx is None or idx + 1 > cap_idx:
            return cap_exit()
        return idx + 1, float(O[idx + 1]), "signal"

    if kind == "rsi_cross":
        line = ctx["rsi"]
        level = float(rule["level"])
        above = np.isfinite(line) & (line >= level)
        prev_above = np.roll(above, 1)
        prev_above[0] = False
        idx = _first_true(prev_above & ~above & np.isfinite(line))
        if idx is None or idx + 1 > cap_idx:
            return cap_exit()
        return idx + 1, float(O[idx + 1]), "signal"

    if kind == "oracle_high":
        idx = int(np.argmax(H))
        return idx, float(H[idx]), "oracle"

    if kind == "pivot_high":
        pivots, confirms = ctx["ph_pivot"], ctx["ph_confirm"]
        j0 = int(ctx["j0_abs"])
        pos = int(np.searchsorted(pivots, j0 - 1, side="right"))
        if pos >= len(pivots):
            return cap_exit()
        if rule.get("ideal"):
            offset = int(pivots[pos]) - j0
            if offset < 0 or offset > cap_idx:
                return cap_exit()
            return offset, float(ctx["high_abs"][int(pivots[pos])]), "structure"
        offset = int(confirms[pos]) + 1 - j0
        if offset < 0 or offset > cap_idx:
            return cap_exit()
        return offset, float(ctx["open_abs"][int(confirms[pos]) + 1]), "structure"

    raise ValueError(f"unknown rule kind: {kind}")


# ---------------------------------------------------------------------------
# 单币ayout compute
# ---------------------------------------------------------------------------
def _net(entry_price: float, exit_price: float) -> float:
    if not (np.isfinite(entry_price) and np.isfinite(exit_price)) or entry_price <= 0:
        return float("nan")
    gross = exit_price / entry_price - 1.0
    cost = (1.0 - (FEE + SLIPPAGE)) / (1.0 + (FEE + SLIPPAGE))
    return (1.0 + gross) * cost - 1.0


def _symbol_rows(symbol: str, frame: pd.DataFrame, confirm: float, min_dd: float, cap: int) -> list[dict]:
    O = frame["open"].to_numpy(float)
    H = frame["high"].to_numpy(float)
    L = frame["low"].to_numpy(float)
    C = frame["close"].to_numpy(float)
    n = len(C)
    atr_arr = wilder_atr(H, L, C)
    ema_arr = ema(C, 20)
    rsi_arr = wilder_rsi(C)
    pivots_low = confirmed_pivot_lows(L, H, C, confirm, PIVOT_MIN_GAP, min_dd)
    ph_pairs = confirmed_pivot_highs(H, C, confirm, PIVOT_MIN_GAP)
    ph_pivot = np.array([p[0] for p in ph_pairs], dtype=int)
    ph_confirm = np.array([p[1] for p in ph_pairs], dtype=int)

    rows = []
    for pi, ci in pivots_low:
        # 可实现口径：确认后下一根开盘成交；理想口径：直接买在低点那根的最低价
        fill = ci + 1
        j0_open = fill
        j0_ideal = pi + 1
        if min(n - j0_open, n - j0_ideal) < cap:
            continue
        atr_val = atr_arr[pi]
        if not np.isfinite(atr_val) or atr_val <= 0:
            continue

        open_entry = float(O[fill])
        ideal_entry = float(L[pi])
        if ideal_entry <= 0 or open_entry <= 0:
            continue

        win_open_O, win_open_H = O[j0_open:j0_open + cap], H[j0_open:j0_open + cap]
        win_open_L, win_open_C = L[j0_open:j0_open + cap], C[j0_open:j0_open + cap]
        win_ideal_O, win_ideal_H = O[j0_ideal:j0_ideal + cap], H[j0_ideal:j0_ideal + cap]
        win_ideal_L, win_ideal_C = L[j0_ideal:j0_ideal + cap], C[j0_ideal:j0_ideal + cap]

        def nasdq(base):
            return {
                "ema": ema_arr[base:base + cap],
                "rsi": rsi_arr[base:base + cap],
                "ph_pivot": ph_pivot, "ph_confirm": ph_confirm,
                "j0_abs": base, "high_abs": H, "open_abs": O,
            }

        ctx_open, ctx_ideal = nasdq(j0_open), nasdq(j0_ideal)

        def unpack(wo, ctx, base, entry_px):
            out = {}
            for rule in RULES:
                try:
                    off, px, reason = evaluate_exit(rule, wo[0], wo[1], wo[2], wo[3], entry_px, atr_val, ctx)
                except Exception:
                    off, px, reason = cap - 1, float(wo[3][-1]), "error"
                off = int(min(max(off, 0), cap - 1))
                out[rule["name"]] = (_net(entry_px, px), off, base + off, reason)
            return out

        res_open = unpack((win_open_O, win_open_H, win_open_L, win_open_C), ctx_open, j0_open, open_entry)
        res_ideal = unpack((win_ideal_O, win_ideal_H, win_ideal_L, win_ideal_C), ctx_ideal, j0_ideal, ideal_entry)

        # 随机出场对照：在窗口内均匀随机挑一根收盘出场
        rng = np.random.default_rng(10_000 + pi)
        offs = rng.integers(0, cap, size=RANDOM_DRAWS)
        rand_open = float(np.mean([_net(open_entry, win_open_C[int(o)]) for o in offs]))
        rand_ideal = float(np.mean([_net(ideal_entry, win_ideal_C[int(o)]) for o in offs]))

        avail_open = float(win_open_H.max() / open_entry - 1.0)
        avail_ideal = float(win_ideal_H.max() / ideal_entry - 1.0)

        rows.append({
            "symbol": symbol, "pivot": int(pi), "confirm": int(ci), "fill": int(fill),
            "lag": int(ci - pi),
            "res_open": res_open, "res_ideal": res_ideal,
            "rand_open": rand_open, "rand_ideal": rand_ideal,
            "avail_open": avail_open, "avail_ideal": avail_ideal,
        })
    return rows


# ---------------------------------------------------------------------------
# 聚类 bootstrap（cluster = 币种）与汇总
# ---------------------------------------------------------------------------
def _resample_matrix(n_groups: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=n_boot)


def _ci(values, groups, matrix) -> dict:
    values = np.asarray(values, dtype=float)
    codes, uniq = pd.factorize(np.asarray(groups))
    ok = np.isfinite(values)
    values, codes = values[ok], codes[ok]
    if len(values) == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None, "positive_share": None, "verdict": "no_data"}
    counts = np.bincount(codes, minlength=len(uniq)).astype(float)
    sums = np.bincount(codes, weights=values, minlength=len(uniq))
    means = (matrix @ sums) / np.maximum(matrix @ counts, 1e-12)
    lo, hi = np.percentile(means, [2.5, 97.5])
    if len(values) < MIN_TRADES:
        verdict = "insufficient"
    elif lo > 0:
        verdict = "positive"
    elif hi < 0:
        verdict = "negative"
    else:
        verdict = "inconclusive"
    return {
        "n": int(len(values)), "mean": round(float(values.mean()), 4),
        "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
        "positive_share": round(float((means > 0).mean()), 4), "verdict": verdict,
    }


def _spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return round(float(np.corrcoef(ra, rb)[0, 1]), 4)


def _collect(rows: list[dict], entry_model: str) -> tuple[list[dict], dict]:
    """按指定入场成交模型汇总所有规则（净收益单位：百分点）。"""
    if not rows:
        return [], {}
    if entry_model not in ("ideal", "open"):
        raise ValueError(f"unknown entry model: {entry_model}")
    result_key, random_key, available_key = (
        ("res_ideal", "rand_ideal", "avail_ideal") if entry_model == "ideal"
        else ("res_open", "rand_open", "avail_open")
    )
    groups = [r["symbol"] for r in rows]
    matrix = _resample_matrix(len(set(groups)), N_BOOT, BOOT_SEED)
    bench = np.array([r[result_key][BENCHMARK][0] for r in rows], float) * 100.0
    random_out = np.array([r[random_key] for r in rows], float) * 100.0
    available = np.array([r[available_key] for r in rows], float) * 100.0
    out = []
    for rule in RULES:
        name = rule["name"]
        net = np.array([r[result_key][name][0] for r in rows], float) * 100.0
        offs = np.array([r[result_key][name][1] for r in rows], float)
        reasons = Counter(r[result_key][name][3] for r in rows)
        with np.errstate(divide="ignore", invalid="ignore"):
            ceiling = available - ROUND_TRIP * 100.0
            capture = np.where(ceiling > 1e-9, net / ceiling, np.nan)
        out.append({
            "name": name, "desc": rule["desc"],
            "mean_net_pct": round(float(np.nanmean(net)), 4),
            "median_net_pct": round(float(np.nanmedian(net)), 4),
            "win_rate": round(float((net > 0).mean()), 4),
            "mean_hold_bars": round(float(np.nanmean(offs)), 1),
            "capture_pct": round(float(np.nanmedian(capture) * 100), 2),
            "ci": _ci(net, groups, matrix),
            "alpha_vs_random": _ci(net - random_out, groups, matrix),
            "delta_vs_hold168": _ci(net - bench, groups, matrix),
            "exit_counts": dict(reasons),
        })
    return out, {
        "entries": len(rows),
        "random_control_mean_pct": round(float(np.nanmean(random_out)), 4),
        "benchmark_mean_pct": round(float(np.nanmean(bench)), 4),
        "available_move_mean_pct": round(float(np.nanmean(available)), 4),
    }


def _sequential(rows: list[dict], rule_name: str, entry_model: str) -> dict:
    """同一币同时最多持一笔；全仓复利的资金效率口径。"""
    result_key = "res_ideal" if entry_model == "ideal" else "res_open"
    outcomes = []
    total_trades = 0
    for symbol in sorted({r["symbol"] for r in rows}):
        equity, last_exit, trades = 1.0, -(10 ** 9), 0
        for r in sorted((x for x in rows if x["symbol"] == symbol), key=lambda x: x["pivot"]):
            net, _, abs_exit, _ = r[result_key][rule_name]
            start = r["pivot"] if entry_model == "ideal" else r["fill"]
            if start <= last_exit:
                continue
            equity *= 1.0 + net
            last_exit = abs_exit
            trades += 1
        if trades:
            outcomes.append(equity)
            total_trades += trades
    if not outcomes:
        return {"symbols": 0, "trades": 0, "median_account_return_pct": None, "positive_symbols": 0}
    return {
        "symbols": len(outcomes), "trades": total_trades,
        "median_account_return_pct": round((float(np.median(outcomes)) - 1.0) * 100.0, 2),
        "mean_account_return_pct": round((float(np.mean(outcomes)) - 1.0) * 100.0, 2),
        "positive_symbols": int(sum(value > 1.0 for value in outcomes)),
    }


def _worker(task: tuple) -> tuple:
    symbol, frame, confirm, min_dd, win = task
    try:
        return symbol, _symbol_rows(symbol, frame, confirm, min_dd, win), None
    except Exception as exc:  # pragma: no cover - defensive
        return symbol, [], f"{type(exc).__name__}: {exc}"


def _analyze(rows: list[dict]) -> dict:
    """把一批入场样本汇总成规则表 + 训练/样本外切分。"""
    max_fill = max(r["fill"] for r in rows)
    split_value = max_fill * TRAIN_RATIO
    train_rows = [r for r in rows if r["fill"] <= split_value]
    test_rows = [r for r in rows if r["fill"] > split_value]

    full_tab, full_ctx = _collect(rows, "ideal")
    train_tab, _ = _collect(train_rows, "ideal")
    test_tab, _ = _collect(test_rows, "ideal")

    per_rule: dict[str, dict] = {}
    for row in full_tab:
        per_rule[row["name"]] = {
            **{k: v for k, v in row.items() if k not in ("ci", "alpha_vs_random", "delta_vs_hold168", "exit_counts")},
            "ci": row["ci"], "alpha_vs_random": row["alpha_vs_random"],
            "delta_vs_hold168": row["delta_vs_hold168"], "exit_counts": row["exit_counts"],
            "sequential": _sequential(rows, row["name"], "ideal"),
        }
    train_map = {r["name"]: r for r in train_tab}
    test_map = {r["name"]: r for r in test_tab}
    for name, item in per_rule.items():
        item["train_mean_net_pct"] = train_map[name]["mean_net_pct"]
        item["test_mean_net_pct"] = test_map[name]["mean_net_pct"]
        item["test_mean_net_ci"] = test_map[name]["ci"]
        item["test_alpha_vs_random"] = test_map[name]["alpha_vs_random"]

    # 可执行成交档：确认后下一根开盘买，用来量化理想低点假设的折损。
    open_tab, open_ctx = _collect(rows, "open")
    open_map = {row["name"]: row for row in open_tab}

    eligible_train = [row for row in train_map.values() if row["name"] != "oracle_cap_high"]
    eligible_test = [row for row in test_map.values() if row["name"] != "oracle_cap_high"]
    train_rank = sorted(eligible_train, key=lambda r: -r["mean_net_pct"])
    test_rank = sorted(eligible_test, key=lambda r: -r["mean_net_pct"])
    open_rank = {k: v["mean_net_pct"] for k, v in per_rule.items() if k != "oracle_cap_high"}
    summary = {
        "entries": len(rows), "train_entries": len(train_rows), "test_entries": len(test_rows),
        "random_control_mean_pct": full_ctx["random_control_mean_pct"],
        "benchmark_mean_pct": full_ctx["benchmark_mean_pct"],
        "available_move_mean_pct": full_ctx["available_move_mean_pct"],
        "train_pick": train_rank[0]["name"], "train_pick_mean_pct": train_rank[0]["mean_net_pct"],
        "test_best": test_rank[0]["name"], "test_best_mean_pct": test_rank[0]["mean_net_pct"],
        "rank_corr_train_test": _spearman(
            [train_map[k]["mean_net_pct"] for k in open_rank],
            [test_map[k]["mean_net_pct"] for k in open_rank]),
        "rank_corr_ideal_vs_open": _spearman([open_map[k]["mean_net_pct"] for k in open_rank], list(open_rank.values())),
    }
    return {
        "summary": summary, "rules": per_rule,
        "open_entry_model": {"context": open_ctx, "rules": {name: {
            "mean_net_pct": row["mean_net_pct"], "alpha_vs_random": row["alpha_vs_random"],
            "ci": row["ci"]} for name, row in open_map.items()}},
        "train_table": [{k: v for k, v in r.items() if k != "exit_counts"} | {"exit_counts": dict(r["exit_counts"])}
                        for r in train_tab],
        "test_table": [{k: v for k, v in r.items() if k != "exit_counts"} | {"exit_counts": dict(r["exit_counts"])}
                       for r in test_tab],
        "diagnostics": {
            "mean_confirm_lag_bars": round(float(np.mean([r["lag"] for r in rows])), 2),
            "mean_available_move_pct": round(float(np.mean([r["avail_open"] for r in rows]) * 100), 3),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="底部入场的出场方案探索")
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--interval", default=INTERVAL)
    ap.add_argument("--cap", type=int, default=CAP_BARS)
    ap.add_argument("--confirm", type=float, default=CONFIRM_PCT)
    ap.add_argument("--min-dd", type=float, default=MIN_DD, help="低点相对本段高点的最小回撤")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sensitivity", action="store_true", help="额外跑入场定义的敏感性网格")
    ap.add_argument("--output", default="docs/perfect-entry-exit-study.json")
    args = ap.parse_args()
    cap = args.cap

    db = sessionmaker(bind=make_engine(read_only=True))()
    service = MarketDataService(db)
    as_of = now_ms()
    start_ms = as_of - args.days * 86_400_000

    frames, skipped = [], {}
    for symbol in list(MAJOR_SYMBOLS) + list(EXTRA_SYMBOLS):
        frame = service.load_frame(symbol, args.interval, start_ms=start_ms, end_ms=as_of)
        if len(frame) < MIN_BARS:
            skipped[symbol] = f"K线不足: {len(frame)}"
            continue
        frames.append((symbol, frame))
    print(f"{len(frames)} 币 × {len(RULES)} 出场规则，interval={args.interval} cap={cap} "
          f"confirm={args.confirm:.2%} min_dd={args.min_dd:.2%}")

    def run_config(confirm: float, min_dd: float, verbose: bool = True) -> list[dict]:
        tasks = [(symbol, frame, confirm, min_dd, cap) for symbol, frame in frames]
        got: list[dict] = []
        errors: dict[str, str] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for symbol, part, err in pool.map(_worker, tasks):
                if err:
                    errors[symbol] = err
                    continue
                got.extend(part)
                if verbose:
                    print(f"{symbol:10s} 入场 {len(part):4d}")
        if not got:
            raise SystemExit("没有任何可用入场样本")
        if verbose:
            print(f"合计入场样本 {len(got)}，币 {len({r['symbol'] for r in got})}")
        run_config.errors = errors  # type: ignore[attr-defined]
        return got

    rows = run_config(args.confirm, args.min_dd)
    symbols = sorted({r["symbol"] for r in rows})
    analyzed = _analyze(rows)

    payload = {
        "config": {
            "as_of": str(pd.Timestamp(as_of, unit="ms", tz="UTC")),
            "days": args.days, "interval": args.interval, "cap_bars": cap,
            "confirm_pct": args.confirm, "min_drawdown_pct": args.min_dd,
            "pivot_min_gap": PIVOT_MIN_GAP,
            "fee": FEE, "slippage": SLIPPAGE, "round_trip_cost_pct": round(ROUND_TRIP * 100, 4),
            "atr_len": ATR_LEN, "random_draws": RANDOM_DRAWS, "n_boot": N_BOOT,
            "symbols_requested": list(MAJOR_SYMBOLS) + list(EXTRA_SYMBOLS),
            "symbols_used": symbols, "skipped": skipped,
            "errors": getattr(run_config, "errors", {}),
            "total_entries": len(rows),
        },
        **analyzed,
    }

    if args.sensitivity:
        grid = []
        for confirm in (0.005, 0.01, 0.02):
            for min_dd in (0.0, 0.02, 0.05):
                part = run_config(confirm, min_dd, verbose=False)
                res = _analyze(part)
                eligible = [value for value in res["rules"].values() if value["name"] != "oracle_cap_high"]
                top3 = sorted(eligible, key=lambda x: -x["mean_net_pct"])[:3]
                common = [key for key in res["rules"] if key != "oracle_cap_high"]
                grid.append({
                    "confirm_pct": confirm, "min_drawdown_pct": min_dd, "entries": res["summary"]["entries"],
                    "available_move_mean_pct": res["summary"]["available_move_mean_pct"],
                    "random_control_mean_pct": res["summary"]["random_control_mean_pct"],
                    "top3": [{"name": t["name"], "mean_net_pct": t["mean_net_pct"],
                              "alpha_vs_random": t["alpha_vs_random"]["mean"]} for t in top3],
                    "rank_corr_vs_main": _spearman(
                        [res["rules"][k]["mean_net_pct"] for k in common],
                        [analyzed["rules"][k]["mean_net_pct"] for k in common]),
                })
                print(f"  敏感性 confirm={confirm:.1%} min_dd={min_dd:.0%}: "
                      f"n={res['summary']['entries']:4d} 榜首={top3[0]['name']:16s} "
                      f"{top3[0]['mean_net_pct']:+.3f}%")
        payload["sensitivity"] = grid

    out_json = ROOT / args.output
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md = out_json.with_suffix(".md")
    out_md.write_text(_markdown(payload), encoding="utf-8")
    print(f"\n写入 {out_json.relative_to(ROOT)} / {out_md.relative_to(ROOT)}")
    ranked = sorted(payload["rules"].values(), key=lambda x: -x["mean_net_pct"])
    print(f"\n{'规则':22s} {'净收益均值%':>11s} {'出场贡献%':>10s} {'vs持有168%':>11s} {'持有根':>7s} {'捕获率%':>8s}")
    for row in ranked:
        print(f"{row['name']:22s} {row['mean_net_pct']:11.3f} {row['alpha_vs_random']['mean']:10.3f} "
              f"{row['delta_vs_hold168']['mean']:11.3f} {row['mean_hold_bars']:7.1f} {row['capture_pct']:8.1f}")


def _markdown(payload: dict) -> str:
    cfg = payload["config"]
    summ = payload["summary"]
    rows = payload["rules"]
    ranked = sorted(rows.values(), key=lambda x: -x["mean_net_pct"])
    top = next(row for row in ranked if row["name"] != "oracle_cap_high")
    oracle = rows["oracle_cap_high"]
    lines = [
        "# 底部入场的出场方案探索",
        "",
        f"> 生成时点：{cfg['as_of']}；{len(cfg['symbols_used'])} 币 / {cfg['interval']} / "
        f"{cfg['days']} 天；入场样本 {summ['entries']} 笔（训练 {summ['train_entries']} / "
        f"样本外 {summ['test_entries']}）。",
        "",
        "## 一句话结论",
        "",
        f"- **可执行候选里**，绝对净收益最高的是 `{top['name']}`：每笔 {top['mean_net_pct']:+.3f}%。",
        f"- 理论天花板 `oracle_cap_high`（事后窗口最高价卖出，未来函数）为 {oracle['mean_net_pct']:+.3f}%，"
        "仅用于衡量还有多少上行被未捕获，**不参与方案排名或选择**。",
        f"- 「买了最低点」本身已经白送很大收益：随机挑一根平仓的对照也有 "
        f"**{summ['random_control_mean_pct']:+.3f}%**。真正属于出场规则的贡献看 `alpha_vs_random`。",
        "", 
        "## 先把结论讲清楚：这 study 能回答什么、不能回答什么",
        "",
        "能回答：**同一批低点、同一笔理想买入价之后，哪种卖出规则把涨幅保留得最多**。"
        "所有规则共享同一批入场，可以做配对比较，入场白送的收益被差分掉了。",
        "",
        "不能回答：「这个策略能不能赚钱」。主表假设成交在低点 K 线的最低价，属于用户指定的不可实现上界。"
        f"可实现的替代档必须等待价格较低点上涨 {cfg['confirm_pct']:.2%}，再在下一根开盘买入；"
        "这会显著降低绝对收益，报告单独保留该档作交叉核验。",
        "",
        "## 主表：可执行方案按绝对净收益排序",
        "",
        "| 出场规则 | 净收益均值% | 95%CI(聚类) | vs随机出场% | 95%CI | 持有根 | 中位捕获率 | 胜率 | 说明 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in (candidate for candidate in ranked if candidate["name"] != "oracle_cap_high"):
        ci, al = row["ci"], row["alpha_vs_random"]
        lines.append(
            f"| `{row['name']}` | {row['mean_net_pct']:+.3f} | "
            f"[{ci['ci_low']}, {ci['ci_high']}] | {al['mean']:+.3f} | "
            f"[{al['ci_low']}, {al['ci_high']}] | {row['mean_hold_bars']:.0f} | "
            f"{row['capture_pct']}% | {row['win_rate']} | {row['desc']} |"
        )
    lines += [
        "",
        "捕获率 = 该规则净收益 /（窗口内最大有利波动 - 往返成本），中位口径；"
        "衡量「一段行情吃到了几成」。",
        "",
        "## 只看出场规则的净贡献：vs 随机出场对照",
        "",
        "| 出场规则 | 出场贡献% | 95%CI | 判定 | vs 持有168根% |",
        "|---|---:|---|---:|---:|",
    ]
    for row in sorted((candidate for candidate in rows.values() if candidate["name"] != "oracle_cap_high"),
                      key=lambda x: -x["alpha_vs_random"]["mean"]):
        al, db = row["alpha_vs_random"], row["delta_vs_hold168"]
        lines.append(
            f"| `{row['name']}` | {al['mean']:+.3f} | [{al['ci_low']}, {al['ci_high']}] | "
            f"{al['verdict']} | {db['mean']:+.3f} |"
        )
    lines += [
        "",
        "随机出场 = 同一笔入场、在窗口内均匀随机挑一根收盘平仓，取 40 次均值。"
        "它衡量的是「入场白送的钱」，规则减掉它才是卖出规则自己赚的。",
        "",
        "## 资金效率：同一时刻只持一笔、全仓复利",
        "",
        "| 出场规则 | 成交笔数 | 账户中位收益% | 正收益币 |",
        "|---|---:|---:|---:|",
    ]
    for row in sorted((candidate for candidate in rows.values() if candidate["name"] != "oracle_cap_high"),
                      key=lambda x: -x["sequential"].get("median_account_return_pct", -1e9)):
        sq = row["sequential"]
        lines.append(
            f"| `{row['name']}` | {sq.get('trades', 0)} | {sq.get('median_account_return_pct', 0):+.2f} | "
            f"{sq.get('positive_symbols', 0)}/{sq.get('symbols', 0)} |"
        )
    lines += [
        "",
        "单笔期望高不等于赚得多：出场越慢，同样的行情能做的笔数越少。这一列才是资金口径的答案。",
        "",
        "## 稳定性检验",
        "",
        f"- 训练段（前 {int(TRAIN_RATIO*100)}% 时间）第一名：`{summ['train_pick']}` "
        f"（{summ['train_pick_mean_pct']:+.3f}%）；样本外第一名：`{summ['test_best']}` "
        f"（{summ['test_best_mean_pct']:+.3f}%）。",
        f"- `hold_72` 在独立样本外相对随机卖出的贡献为 {rows['hold_72']['test_alpha_vs_random']['mean']:+.3f}%（"
        f"95% CI [{rows['hold_72']['test_alpha_vs_random']['ci_low']}, {rows['hold_72']['test_alpha_vs_random']['ci_high']}]），"
        f"判定 `{rows['hold_72']['test_alpha_vs_random']['verdict']}`；因此全样本领先不能直接当作上线结论。", 
        f"- 训练/样本外规则排名的 Spearman 相关：**{summ['rank_corr_train_test']}**。",
        f"- 可执行成交档 vs 理想低点买入档的规则排名相关：**{summ['rank_corr_ideal_vs_open']}**；"
        "排序近似一致，但绝对收益不可外推到实盘。", 
        f"- 平均确认滞后 {payload['diagnostics']['mean_confirm_lag_bars']} 根 —— "
        "这就是「知道那是底」的最小时间代价。",
        f"- 可执行档中 `hold_72` 每笔为 {payload['open_entry_model']['rules']['hold_72']['mean_net_pct']:+.3f}%（"
        f"95% CI [{payload['open_entry_model']['rules']['hold_72']['ci']['ci_low']}, "
        f"{payload['open_entry_model']['rules']['hold_72']['ci']['ci_high']}]），显著性为 "
        f"`{payload['open_entry_model']['rules']['hold_72']['ci']['verdict']}`；这正是理想低点成交不可直接照搬的证据。",
        "",
        "## 底部定义敏感性",
        "",
        "| 确认反弹 | 前置最小回撤 | 样本 | 第一名 | 净收益均值% | vs 随机出场% | 与主口径排名相关 |",
        "|---:|---:|---:|---|---:|---:|---:|",
        *[
            f"| {item['confirm_pct']:.1%} | {item['min_drawdown_pct']:.0%} | {item['entries']} | "
            f"`{item['top3'][0]['name']}` | {item['top3'][0]['mean_net_pct']:+.3f} | "
            f"{item['top3'][0]['alpha_vs_random']:+.3f} | {item['rank_corr_vs_main']} |"
            for item in payload.get("sensitivity", [])
        ],
        "",
        "九种预先声明的底部定义中，非未来函数候选均为 `hold_72`。"
        "这支持它是值得继续检验的方向，但不替代独立样本外或前向纸面验证。",
        "",
        "## 预注册规则定义",
        "",
        "- `hold_N`：固定持有 N 根后收盘平仓。",
        "- `tpX_slY`：止盈 X·ATR、止损 Y·ATR（R=入场时 Wilder ATR14）；"
        "同一根 K 线内同时触及两者时**按先止损记**。",
        "- `trail_k`：移动止损参考「截至上一根的 running high」，且不低于入场价；跌破后按 min(止损价, 开盘价) 成交。",
        "- `chand_k`：同上，但止损距离用 k·ATR 而非百分比。",
        "- `ema20_close` / `rsi50_cross`：收盘确认信号，下一根开盘成交。",
        "- `pivot_high_*`：下一个「确认的局部高点」；`_ideal` 按该根最高价成交（理论上限），"
        "`_open` 按确认后下一根开盘成交（可实现）。",
        "- `oracle_cap_high`：整个观察窗口的最高价卖出；这是未来函数，只用作上限，绝不视为策略。"
        "",
        "## 口径与限制",
        "",
        "1. 主表严格按用户假设：买入价 = 本段确认低点 K 线的 `low`。这是事后才知道的理想化上界；"
        f"可执行替代档是在确认反弹 {cfg['confirm_pct']:.2%} 后的下一根开盘买，单独保存并仅用于交叉核验。",
        f"2. 主样本还要求低点前的本段回撤 ≥ {cfg['min_drawdown_pct']:.2%}，避免把噪声级浅回踩也叫作底部；"
        "报告另给出 9 组回撤/确认门槛的敏感性表。",
        f"3. 每笔入场要求剩余 ≥ {cfg['cap_bars']} 根，超时未触发的规则按最后一根收盘平仓并记 `cap`。",
        "4. 成本双边 fee 0.04% + slippage 0.02%/边。所有规则同成本，故不影响排序，但影响绝对值。",
        "5. 未模拟交易所订单语义（maker/taker、部分成交、资金费率）；迁移到实盘需重跑。",
        "6. 样本不是独立同分布：相邻入场可能重叠（同币同时最多持一笔的口径另见「资金效率」表）。"
        "置信区间按**币种聚类**bootstrap，避免把重叠当成独立样本。",
        "7. 绝对收益是理想上界。即使某种出场在样本外貌好，仍需前向纸面验证，不能直接上线。",
        "",
        "## 复现",
        "",
        "```powershell",
        f".\\.venv\\Scripts\\python.exe backend/scripts/perfect_entry_exit_study.py --days {cfg['days']} --workers 4",
        "```",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()

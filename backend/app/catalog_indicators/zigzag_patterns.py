"""ZigZag Patterns [WavesUnchained] — ZigZag 枢轴引擎 + ABC/三角/Wolfe/1-2-3 形态。

**枢轴引擎**：`ta.pivothigh/pivotlow(high/low, len, len)` 确认后按「同类型只保留
更极端者」的规则写入历史（`max_pivots` 上限），每根确认枢轴都会补一条 ZigZag 段。

**智能标签**：对最新枢轴给出 `HH/LH/HL/LL` 方向标签（与前一个同类型枢轴比较），
叠加腿强度（`Imp`：本腿 ≥ min_leg_atr×ATR 且 > 上腿 1.15 倍；`Corr`：< 0.85 倍）
与影线拒斥（高点枢轴看上影 ≥ wick_atr×ATR 且大于下影、实体不超过上影；低点镜像），
信号数 ≥ `label_min_signals` 才输出文本。RSI 背离标记不参与信号计数。

**形态**（只在新枢轴确认时扫描）：
- ABC：最近 3 枢轴交替（低-高-低 / 高-低-高），回撤比 ∈ [min_retr, max_retr]
  且 C 高于 A（看跌镜像为 C 低于 A）。
- 三角：最近 5 枢轴交替，两侧取前两个同类型点，按 `flat_tol×ATR` 判定走平；
  Sym（高点下移+低点上移且两侧都不走平）、Rising（高点走平+低点上移且低点不走平）、
  Falling（低点走平+高点下移且高点不走平）。
- Wolfe：最近 5 枢轴交替，`|P3-P4| < |P1-P2|`（收敛）且 P5 越过 P3 同向（超调），
  时间跨度 ≥ `len×3`。
- 1-2-3（Sperandeo）：最近 3 枢轴交替且首尾同类型，要求枢轴 1 之前有真实趋势
  （看涨需枢轴1标签 LL 且前一同类型枢轴 LH），枢轴 3 未破枢轴 1，回撤比在区间内；
  随后由**任意一根** K 线收盘穿越枢轴 2 触发（不是枢轴事件），质量分
  = 回撤居中 35 + 失败幅度 35 + 突破 ATR 幅度 30。

丢弃：全部 line/label 绘图、ZigZag 线段数组、告警配置。

Source: pine-collection/indicators/market_structure/zigzag_patterns_framework/zigzag_patterns_framework.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    boolean,
    flags,
    integer,
    number,
    ohlcv,
    pivothigh,
    pivotlow,
    rsi,
)

__all__ = ["zigzag_patterns"]


class _PivotStore:
    """Bounded pivot history with Pine's same-type replacement rule."""

    def __init__(self, cap: int) -> None:
        self.price: list[float] = []
        self.index: list[int] = []
        self.kind: list[int] = []
        self.cap = cap

    def __len__(self) -> int:
        return len(self.kind)

    def push(self, bar: int, price: float, kind: int) -> bool:
        if self.kind and self.kind[-1] == kind:
            previous = self.price[-1]
            replace = price > previous if kind == 1 else price < previous
            if replace:
                self.price[-1] = price
                self.index[-1] = bar
            return False
        self.kind.append(kind)
        self.price.append(price)
        self.index.append(bar)
        while len(self.kind) > self.cap:
            self.kind.pop(0)
            self.price.pop(0)
            self.index.pop(0)
        return True

    def tag(self, position: int) -> str:
        kind = self.kind[position]
        value = self.price[position]
        previous = math.nan
        for back in range(position - 1, -1, -1):
            if self.kind[back] == kind:
                previous = self.price[back]
                break
        if math.isnan(previous):
            return "NA"
        if kind == 1:
            return "HH" if value > previous else "LH"
        return "HL" if value > previous else "LL"


def _leg_strength(store: _PivotStore, atr_now: float, min_leg_atr: float) -> str:
    if len(store) < 3:
        return ""
    c, b, a = store.price[-1], store.price[-2], store.price[-3]
    leg1 = abs(b - a)
    leg2 = abs(c - b)
    impulse = leg2 >= atr_now * min_leg_atr and leg2 > leg1 * 1.15
    if impulse:
        return "Imp"
    if leg2 < leg1 * 0.85:
        return "Corr"
    return ""


def _wick_reject(highs: np.ndarray, lows: np.ndarray, opens: np.ndarray,
                 closes: np.ndarray, pivot_bar: int, kind: int,
                 atr_now: float, wick_atr: float) -> str:
    if pivot_bar < 0 or pivot_bar >= len(highs):
        return ""
    up_wick = highs[pivot_bar] - max(opens[pivot_bar], closes[pivot_bar])
    down_wick = min(opens[pivot_bar], closes[pivot_bar]) - lows[pivot_bar]
    body = abs(closes[pivot_bar] - opens[pivot_bar])
    if kind == 1:
        rejected = up_wick >= atr_now * wick_atr and up_wick > down_wick and body <= up_wick
        return "Rej↑" if rejected else ""
    rejected = down_wick >= atr_now * wick_atr and down_wick > up_wick and body <= down_wick
    return "Rej↓" if rejected else ""


def _rsi_divergence(store: _PivotStore, rsi_values: np.ndarray, bar: int) -> tuple[bool, str]:
    kind = store.kind[-1]
    previous = None
    for back in range(len(store) - 2, -1, -1):
        if store.kind[back] == kind:
            previous = back
            break
    if previous is None:
        return False, ""
    cur_bar = store.index[-1]
    prev_bar = store.index[previous]
    if not (0 <= cur_bar < len(rsi_values) and 0 <= prev_bar < len(rsi_values)):
        return False, ""
    price_cur, price_prev = store.price[-1], store.price[previous]
    rsi_cur, rsi_prev = rsi_values[cur_bar], rsi_values[prev_bar]
    if not (np.isfinite(rsi_cur) and np.isfinite(rsi_prev)):
        return False, ""
    if kind == 1:
        diverged = price_cur > price_prev and rsi_cur < rsi_prev
    else:
        diverged = price_cur < price_prev and rsi_cur > rsi_prev
    return (True, "RSI div") if diverged else (False, "")


def _abc(store: _PivotStore, min_retr: float, max_retr: float) -> tuple[bool, bool, float]:
    if len(store) < 3:
        return False, False, math.nan
    a, b, c = store.price[-3], store.price[-2], store.price[-1]
    ta, tb, tc = store.kind[-3], store.kind[-2], store.kind[-1]
    bull = ta == -1 and tb == 1 and tc == -1
    bear = ta == 1 and tb == -1 and tc == 1
    ab = abs(b - a)
    retr = abs(b - c) / ab if ab > 0 else math.nan
    if math.isnan(retr) or not (min_retr <= retr <= max_retr):
        return False, False, retr
    return bull and c > a, bear and c < a, retr


def _triangle(store: _PivotStore, atr_now: float, flat_atr: float) -> tuple[bool, bool, bool]:
    if len(store) < 5:
        return False, False, False
    start = len(store) - 5
    kinds = store.kind[start:]
    if any(kinds[i] == kinds[i + 1] for i in range(len(kinds) - 1)):
        return False, False, False
    highs = [store.price[i] for i in range(start, len(store)) if store.kind[i] == 1]
    lows = [store.price[i] for i in range(start, len(store)) if store.kind[i] == -1]
    if len(highs) < 2 or len(lows) < 2:
        return False, False, False
    h1, h2 = highs[0], highs[1]
    l1, l2 = lows[0], lows[1]
    tol = atr_now * flat_atr
    if not np.isfinite(tol):
        return False, False, False
    highs_down = h2 < h1
    lows_up = l2 > l1
    highs_flat = abs(h2 - h1) <= tol
    lows_flat = abs(l2 - l1) <= tol
    sym = highs_down and lows_up and not highs_flat and not lows_flat
    rising = highs_flat and lows_up and not lows_flat
    falling = lows_flat and highs_down and not highs_flat
    return sym, rising, falling


def _wolfe(store: _PivotStore, pivot_len: int) -> int:
    """Return +1/-1 for a Wolfe wedge whose last pivot overshoots, else 0."""
    if len(store) < 5:
        return 0
    start = len(store) - 5
    kinds = store.kind[start:]
    if any(kinds[i] == kinds[i + 1] for i in range(len(kinds) - 1)):
        return 0
    p1, p2, p3, p4, p5 = (store.price[i] for i in range(start, len(store)))
    x1, x5 = store.index[start], store.index[-1]
    contracting = abs(p3 - p4) < abs(p1 - p2)
    overshoot = (kinds[-1] == -1 and p5 < p3) or (kinds[-1] == 1 and p5 > p3)
    spaced = (x5 - x1) >= pivot_len * 3
    if contracting and overshoot and spaced:
        return kinds[-1]
    return 0


def _setup_123(store: _PivotStore, min_retr: float, max_retr: float) -> dict | None:
    if len(store) < 4:
        return None
    p1, p2, p3 = store.price[-3], store.price[-2], store.price[-1]
    x1, x2, x3 = store.index[-3], store.index[-2], store.index[-1]
    t1, t2, t3 = store.kind[-3], store.kind[-2], store.kind[-1]
    if not (t1 != t2 and t2 != t3 and t1 == t3):
        return None
    bull = t1 == -1
    tag_p1 = store.tag(len(store) - 3)
    tag_prev = store.tag(len(store) - 4)
    prior_trend = (tag_p1 == "LL" and tag_prev == "LH") if bull else (tag_p1 == "HH" and tag_prev == "HL")
    fails = p3 > p1 if bull else p3 < p1
    leg_ab = abs(p2 - p1)
    retr = abs(p2 - p3) / leg_ab if leg_ab > 0 else math.nan
    if not prior_trend or not fails or math.isnan(retr):
        return None
    if not (min_retr <= retr <= max_retr):
        return None
    return {"bull": bull, "p1": p1, "p2": p2, "p3": p3, "retr": retr,
            "x1": x1, "x2": x2, "x3": x3}


def _score_123(p1: float, p2: float, p3: float, retr: float, break_atr: float,
               min_retr: float, max_retr: float) -> int:
    mid = (min_retr + max_retr) / 2
    span = (max_retr - min_retr) / 2
    retr_score = (1 - min(abs(retr - mid) / span, 1)) * 35 if span > 0 else 0.0
    leg_ab = abs(p2 - p1)
    fail_margin = abs(p1 - p3) / leg_ab if leg_ab > 0 else 0.0
    fail_score = min(fail_margin, 1) * 35
    break_score = min(max(break_atr, 0), 1) * 30
    return int(round(retr_score + fail_score + break_score))


def zigzag_patterns(
    frame: pd.DataFrame,
    *,
    pivot_len: int = 5,
    max_pivots: int = 80,
    show_smart_labels: bool = True,
    label_min_signals: int = 1,
    show_hh_ll: bool = True,
    show_momentum_div: bool = False,
    rsi_len: int = 14,
    use_wick_reject: bool = True,
    use_leg_strength: bool = True,
    enable_abc: bool = True,
    enable_triangle: bool = True,
    enable_wolfe: bool = True,
    enable_123: bool = True,
    atr_len: int = 14,
    flat_atr: float = 0.25,
    min_retr: float = 0.382,
    max_retr: float = 0.886,
    min_leg_atr: float = 0.8,
    wick_atr: float = 0.6,
) -> pd.DataFrame:
    """返回 ZigZag 枢轴流与四类形态信号。"""
    pivot_len = integer("pivot_len", pivot_len, 2, 30)
    max_pivots = integer("max_pivots", max_pivots, 10, 300)
    show_smart_labels = boolean("show_smart_labels", show_smart_labels)
    label_min_signals = integer("label_min_signals", label_min_signals, 1, 4)
    show_hh_ll = boolean("show_hh_ll", show_hh_ll)
    show_momentum_div = boolean("show_momentum_div", show_momentum_div)
    rsi_len = integer("rsi_len", rsi_len, 2)
    use_wick_reject = boolean("use_wick_reject", use_wick_reject)
    use_leg_strength = boolean("use_leg_strength", use_leg_strength)
    enable_abc = boolean("enable_abc", enable_abc)
    enable_triangle = boolean("enable_triangle", enable_triangle)
    enable_wolfe = boolean("enable_wolfe", enable_wolfe)
    enable_123 = boolean("enable_123", enable_123)
    atr_len = integer("atr_len", atr_len, 1)
    flat_atr = number("flat_atr", flat_atr, 0.0)
    min_retr = number("min_retr", min_retr, 0.0, 1.0)
    max_retr = number("max_retr", max_retr, 0.0, 1.0)
    min_leg_atr = number("min_leg_atr", min_leg_atr, 0.0)
    wick_atr = number("wick_atr", wick_atr, 0.0)

    high, low, open_, close = ohlcv(frame, "high", "low", "open", "close")
    atr_values = atr(frame, atr_len).to_numpy()
    rsi_values = rsi(close, rsi_len).to_numpy()
    highs = high.to_numpy()
    lows = low.to_numpy()
    opens = open_.to_numpy()
    closes = close.to_numpy()
    ph = pivothigh(high, pivot_len, pivot_len).to_numpy()
    pl = pivotlow(low, pivot_len, pivot_len).to_numpy()

    size = len(frame)
    store = _PivotStore(max_pivots)
    pending = False
    pending_dir = 0
    pending_p1 = pending_p2 = pending_p3 = pending_retr = math.nan

    out = {name: np.zeros(size, dtype=bool) for name in
           ("pivot_new", "bear_div", "bull_div", "abc_bull", "abc_bear",
            "triangle_sym", "triangle_rising", "triangle_falling", "wolfe",
            "setup_123", "long_123", "short_123", "pending_123")}
    out["pivot_price"] = np.full(size, np.nan)
    out["pivot_type"] = np.full(size, np.nan)
    out["pivot_bar"] = np.full(size, np.nan)
    out["pivot_signals"] = np.full(size, np.nan)
    out["pivot_label"] = np.full(size, "", dtype=object)
    out["abc_retr"] = np.full(size, np.nan)
    out["wolfe_dir"] = np.full(size, np.nan)
    out["setup_123_dir"] = np.full(size, np.nan)
    out["setup_123_retr"] = np.full(size, np.nan)
    out["break_123_score"] = np.full(size, np.nan)

    for bar in range(size):
        pivot_bar = bar - pivot_len
        new_pivot = False
        if not np.isnan(ph[bar]) or not np.isnan(pl[bar]):
            both = not np.isnan(ph[bar]) and not np.isnan(pl[bar])
            if both:
                last_kind = store.kind[-1] if len(store) else 0
                order = ((pl[bar], -1), (ph[bar], 1)) if last_kind == 1 else ((ph[bar], 1), (pl[bar], -1))
            elif not np.isnan(ph[bar]):
                order = ((ph[bar], 1),)
            else:
                order = ((pl[bar], -1),)
            for price, kind in order:
                new_pivot = store.push(pivot_bar, float(price), kind) or new_pivot

        if new_pivot:
            out["pivot_new"][bar] = True
            position = len(store) - 1
            out["pivot_price"][bar] = store.price[position]
            out["pivot_type"][bar] = store.kind[position]
            out["pivot_bar"][bar] = store.index[position]

            if show_smart_labels and len(store) >= 2:
                atr_now = atr_values[bar]
                text = ""
                signals = 0
                if show_hh_ll:
                    tag = store.tag(position)
                    if tag != "NA":
                        text += tag
                        signals += 1
                if use_leg_strength and np.isfinite(atr_now):
                    leg = _leg_strength(store, atr_now, min_leg_atr)
                    if leg:
                        text += ("" if text == "" else " | ") + leg
                        signals += 1
                if use_wick_reject and np.isfinite(atr_now):
                    wick = _wick_reject(highs, lows, opens, closes, store.index[position],
                                        store.kind[position], atr_now, wick_atr)
                    if wick:
                        text += ("" if text == "" else " | ") + wick
                        signals += 1
                if show_momentum_div:
                    diverged, _ = _rsi_divergence(store, rsi_values, bar)
                    if diverged:
                        if store.kind[position] == 1:
                            out["bear_div"][bar] = True
                        else:
                            out["bull_div"][bar] = True
                out["pivot_signals"][bar] = signals
                if signals >= label_min_signals and text != "":
                    out["pivot_label"][bar] = text

            if enable_abc:
                bull, bear, retr = _abc(store, min_retr, max_retr)
                out["abc_retr"][bar] = retr
                out["abc_bull"][bar] = bull
                out["abc_bear"][bar] = bear
            if enable_triangle:
                sym, rising, falling = _triangle(store, atr_values[bar], flat_atr)
                out["triangle_sym"][bar] = sym
                out["triangle_rising"][bar] = rising
                out["triangle_falling"][bar] = falling
            if enable_wolfe:
                direction = _wolfe(store, pivot_len)
                out["wolfe"][bar] = direction != 0
                out["wolfe_dir"][bar] = direction if direction else np.nan
            if enable_123:
                setup = _setup_123(store, min_retr, max_retr)
                if setup is not None:
                    late = (close.iloc[bar] > setup["p2"]) if setup["bull"] else (close.iloc[bar] < setup["p2"])
                    if not late:
                        pending = True
                        pending_dir = 1 if setup["bull"] else -1
                        pending_p1, pending_p2, pending_p3 = setup["p1"], setup["p2"], setup["p3"]
                        pending_retr = setup["retr"]
                        out["setup_123"][bar] = True
                        out["setup_123_dir"][bar] = pending_dir
                        out["setup_123_retr"][bar] = pending_retr

        if pending:
            price = closes[bar]
            broke_up = pending_dir == 1 and price > pending_p2
            broke_down = pending_dir == -1 and price < pending_p2
            failed_up = pending_dir == 1 and price < pending_p1
            failed_down = pending_dir == -1 and price > pending_p1
            if broke_up or broke_down:
                atr_now = atr_values[bar]
                break_atr = ((price - pending_p2) if broke_up else (pending_p2 - price)) / atr_now \
                    if np.isfinite(atr_now) and atr_now > 0 else math.nan
                score = _score_123(pending_p1, pending_p2, pending_p3, pending_retr,
                                   break_atr if np.isfinite(break_atr) else 0.0,
                                   min_retr, max_retr)
                out["break_123_score"][bar] = score
                out["long_123"][bar] = pending_dir == 1
                out["short_123"][bar] = pending_dir == -1
                pending = False
            elif failed_up or failed_down:
                pending = False
        out["pending_123"][bar] = pending

    index = frame.index
    data: dict[str, pd.Series] = {
        "pivot_new": flags(pd.Series(out["pivot_new"], index=index)),
        "pivot_price": pd.Series(out["pivot_price"], index=index),
        "pivot_type": pd.Series(out["pivot_type"], index=index).astype("Int64"),
        "pivot_bar": pd.Series(out["pivot_bar"], index=index).astype("Int64"),
        "pivot_signals": pd.Series(out["pivot_signals"], index=index).astype("Int64"),
        "pivot_label": pd.Series(out["pivot_label"], index=index, dtype=object),
        "bear_div": flags(pd.Series(out["bear_div"], index=index)),
        "bull_div": flags(pd.Series(out["bull_div"], index=index)),
        "abc_bull": flags(pd.Series(out["abc_bull"], index=index)),
        "abc_bear": flags(pd.Series(out["abc_bear"], index=index)),
        "abc_retr": pd.Series(out["abc_retr"], index=index),
        "triangle_sym": flags(pd.Series(out["triangle_sym"], index=index)),
        "triangle_rising": flags(pd.Series(out["triangle_rising"], index=index)),
        "triangle_falling": flags(pd.Series(out["triangle_falling"], index=index)),
        "wolfe": flags(pd.Series(out["wolfe"], index=index)),
        "wolfe_dir": pd.Series(out["wolfe_dir"], index=index).astype("Int64"),
        "setup_123": flags(pd.Series(out["setup_123"], index=index)),
        "setup_123_dir": pd.Series(out["setup_123_dir"], index=index).astype("Int64"),
        "setup_123_retr": pd.Series(out["setup_123_retr"], index=index),
        "long_123": flags(pd.Series(out["long_123"], index=index)),
        "short_123": flags(pd.Series(out["short_123"], index=index)),
        "break_123_score": pd.Series(out["break_123_score"], index=index).astype("Int64"),
        "pending_123": flags(pd.Series(out["pending_123"], index=index)),
    }
    return pd.DataFrame(data, index=index)

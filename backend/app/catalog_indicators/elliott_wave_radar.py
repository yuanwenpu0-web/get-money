"""Elliott Wave Radar [WavesUnchained] — 规则校验的艾略特波浪计数。

**枢轴引擎（两个级别）**：主级 `pivot_len`、子级 `sub_len`。每次枢轴确认都先在该
级别枢轴**之前**的枢轴之后扫描「真实极值」（Pine 的 `f_extreme`，最多回看 900 根），
避免把波浪点落在确认值而非真实极值上。同类型枢轴只保留更极端者；交替枢轴必须
`|新价 − 上一个枢轴价| >= min_swing_atr×ATR` 才入库（子级不过滤）。库上限
`max_pivots`，超出从最旧端淘汰。

**硬规则**（完整推动 6 点 P0..P5，方向 = P5 类型）：
- W2 不得越过 W1 起点；W3 不得同时短于 W1 与 W5；W4 不得进入 W1 区间
  （`allow_diagonal` 可放行）；进度要求 W3 越过 W1 终点且 W5 越过 W3 终点。

**评分（0..100）**：W2/W1 比 0.585（宽 0.40）、W3/W1 取 1.618 与 0.85×2.618 的较大、
W4/W3 比 0.382（宽 0.35）、W5/W1 取 0.618 与 1.0 的较大；再加「交替分」
（|rr2−rr4| ≥ 0.10 满分，否则线性）；权重 0.20/0.30/0.15/0.20/0.15。

**锯齿修正（4 点）**：B/A ∈ [0.236, 0.886]、C 越过 A 端、C/A ∈ [0.5, 2.0]；
评分 0.40×B（0.55/0.35）+ 0.60×C（1.0 或 0.80×1.618）。

**平台修正（4 点）**：B/A ∈ [0.90, 1.38]（regular≈1.0、expanded≈1.05–1.38）、
C 越过 A 端、C/A ∈ [0.8, 1.8]；评分 0.45×B + 0.55×C。

**发展中的计数**：
- C 波：上一次完整推动结束点后紧跟 A、B 且 B 未越过推动极值、B/A ∈ [0.236, 0.886]。
- W3：P1 为 W1 极值、W2 未破 W1 起点且 W2/W1 ∈ [0.236, 1.0)。
- W5：W1–W4 已确认（同硬规则，另要求 W3/W1 ≥ 0.8）。

**选择**：只有原始分 ≥ `min_score` 的候选参与，按「原始分 + 完备度加成」
（IMP 15 / ABC 10 / FLAT 10 / DEVC 8 / DEV4 5 / DEV2 0）取最大者，次优记为备选计数。
发展中的计数记录失效价，`low < inv`（多）或 `high > inv`（空）即失效并清空。

**目标投影**（仅输出，不绘图）：W3 用 ×1.0/1.618/2.618、W5 用 ×0.618/1.0/1.618、
C 波用 ×1.0/1.618、完整推动后修正位 38.2/50/61.8%。

丢弃：全部 line/label/table 绘图、dashboard、告警配置、子浪标签绘制（只保留腿数统计）。

Source: pine-collection/indicators/market_structure/elliott_wave_radar/elliott_wave_radar.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, boolean, integer, number, ohlcv, pivothigh, pivotlow

__all__ = ["elliott_wave_radar"]

_BONUS = {"IMP": 15.0, "ABC": 10.0, "FLAT": 10.0, "DEVC": 8.0, "DEV4": 5.0, "DEV2": 0.0}
_POINTS = {"IMP": 6, "ABC": 4, "FLAT": 4, "DEVC": 3, "DEV4": 5, "DEV2": 3}


def _score(value: float, ideal: float, width: float) -> float:
    """Linear closeness: 1.0 at the ideal ratio, 0 beyond ideal ± width."""
    return max(0.0, 1.0 - abs(value - ideal) / width)


class _WaveStore:
    """Pivot history at one degree, storing the true extreme since the prior pivot."""

    __slots__ = ("price", "bar", "kind", "cap", "highs", "lows")

    def __init__(self, cap: int, highs: np.ndarray, lows: np.ndarray) -> None:
        self.price: list[float] = []
        self.bar: list[int] = []
        self.kind: list[int] = []
        self.cap = cap
        self.highs = highs
        self.lows = lows

    def __len__(self) -> int:
        return len(self.kind)

    def _extreme(self, ref_bar: int, price: float, bar: int, kind: int) -> tuple[float, int]:
        start = max(ref_bar + 1, bar - 899)
        if bar < start:
            return price, bar
        values = (self.highs if kind == 1 else self.lows)[start: bar + 1]
        if not len(values):
            return price, bar
        target = values.max() if kind == 1 else values.min()
        # Pine scans recent → old and only replaces on a strict improvement, so the
        # *most recent* bar holding the extreme wins.
        offset = len(values) - 1 - (np.argmax(values[::-1]) if kind == 1
                                    else np.argmin(values[::-1]))
        return float(target), start + offset

    def push(self, bar: int, price: float, kind: int, min_swing: float) -> bool:
        changed = False
        if self.kind and self.kind[-1] == kind:
            ref = self.bar[-2] if len(self.kind) > 1 else self.bar[-1]
            best, best_bar = self._extreme(ref, price, bar, kind)
            last = self.price[-1]
            more = best > last if kind == 1 else best < last
            if more:
                self.price[-1] = best
                self.bar[-1] = best_bar
                changed = True
        else:
            best, best_bar = price, bar
            if self.kind:
                best, best_bar = self._extreme(self.bar[-1], price, bar, kind)
            size_ok = True
            if self.kind and min_swing > 0:
                size_ok = abs(best - self.price[-1]) >= min_swing
            if size_ok:
                self.price.append(best)
                self.bar.append(best_bar)
                self.kind.append(kind)
                changed = True
        while len(self.kind) > self.cap:
            self.kind.pop(0)
            self.bar.pop(0)
            self.price.pop(0)
        return changed

    def price_at(self, back: int) -> float:
        return self.price[-1 - back]

    def bar_at(self, back: int) -> int:
        return self.bar[-1 - back]

    def kind_at(self, back: int) -> int:
        return self.kind[-1 - back]


def _eval_impulse(store: _WaveStore, allow_diagonal: bool) -> tuple[bool, float]:
    if len(store) < 6:
        return False, 0.0
    direction = store.kind_at(0)
    p = [store.price_at(back) for back in range(5, -1, -1)]
    w1, w2, w3, w4, w5 = (abs(p[i + 1] - p[i]) for i in range(5))
    up = direction == 1
    r1 = p[2] > p[0] if up else p[2] < p[0]
    r2 = not (w3 < w1 and w3 < w5)
    r3 = allow_diagonal or (p[4] > p[1] if up else p[4] < p[1])
    progress = (p[3] > p[1] and p[5] > p[3]) if up else (p[3] < p[1] and p[5] < p[3])
    if not (r1 and r2 and r3 and progress and w1 > 0 and w3 > 0):
        return False, 0.0
    rr2, e3, rr4, e5 = w2 / w1, w3 / w1, w4 / w3, w5 / w1
    s2 = _score(rr2, 0.585, 0.40)
    s3 = max(_score(e3, 1.618, 1.20), 0.85 * _score(e3, 2.618, 1.20))
    s4 = _score(rr4, 0.382, 0.35)
    s5 = max(_score(e5, 0.618, 0.50), _score(e5, 1.0, 0.50))
    gap = abs(rr2 - rr4)
    s_alt = 1.0 if gap >= 0.10 else gap / 0.10
    return True, 100.0 * (0.20 * s2 + 0.30 * s3 + 0.15 * s4 + 0.20 * s5 + 0.15 * s_alt)


def _eval_abc(store: _WaveStore) -> tuple[bool, float]:
    if len(store) < 4:
        return False, 0.0
    direction = store.kind_at(0)
    p0, pa, pb, pc = (store.price_at(back) for back in (3, 2, 1, 0))
    wa, wb, wc = abs(pa - p0), abs(pb - pa), abs(pc - pb)
    if wa <= 0:
        return False, 0.0
    rb, rc = wb / wa, wc / wa
    b_ok = 0.236 <= rb <= 0.886
    c_ok = pc > pa if direction == 1 else pc < pa
    c_len = 0.5 <= rc <= 2.0
    if not (b_ok and c_ok and c_len):
        return False, 0.0
    s_b = _score(rb, 0.550, 0.35)
    s_c = max(_score(rc, 1.0, 0.60), 0.80 * _score(rc, 1.618, 0.60))
    return True, 100.0 * (0.40 * s_b + 0.60 * s_c)


def _eval_flat(store: _WaveStore) -> tuple[bool, float]:
    if len(store) < 4:
        return False, 0.0
    direction = store.kind_at(0)
    p0, pa, pb, pc = (store.price_at(back) for back in (3, 2, 1, 0))
    wa, wb, wc = abs(pa - p0), abs(pb - pa), abs(pc - pb)
    if wa <= 0:
        return False, 0.0
    rb, rc = wb / wa, wc / wa
    b_ok = 0.90 <= rb <= 1.38
    c_ok = pc > pa if direction == 1 else pc < pa
    c_len = 0.8 <= rc <= 1.8
    if not (b_ok and c_ok and c_len):
        return False, 0.0
    s_b = max(_score(rb, 1.0, 0.20), 0.90 * _score(rb, 1.27, 0.25))
    s_c = max(_score(rc, 1.0, 0.50), 0.85 * _score(rc, 1.618, 0.50))
    return True, 100.0 * (0.45 * s_b + 0.55 * s_c)


def _eval_dev_c(store: _WaveStore, impulse_dir: int, end_bar: float,
                end_price: float) -> tuple[bool, float]:
    if impulse_dir == 0 or not np.isfinite(end_bar) or len(store) < 3:
        return False, 0.0
    if store.bar_at(2) != int(end_bar) or store.kind_at(2) != impulse_dir:
        return False, 0.0
    pa, pb = store.price_at(1), store.price_at(0)
    wa, wb = abs(pa - store.price_at(2)), abs(pb - pa)
    if wa <= 0:
        return False, 0.0
    rb = wb / wa
    inside = pb < end_price if impulse_dir == 1 else pb > end_price
    if inside and 0.236 <= rb <= 0.886:
        return True, 100.0 * _score(rb, 0.550, 0.35)
    return False, 0.0


def _eval_dev2(store: _WaveStore) -> tuple[bool, float]:
    if len(store) < 3:
        return False, 0.0
    direction = store.kind_at(1)
    p0, p1, p2 = (store.price_at(back) for back in (2, 1, 0))
    w1, w2 = abs(p1 - p0), abs(p2 - p1)
    if w1 <= 0:
        return False, 0.0
    rr2 = w2 / w1
    r1 = p2 > p0 if direction == 1 else p2 < p0
    if r1 and 0.236 <= rr2 < 1.0:
        return True, 100.0 * _score(rr2, 0.585, 0.45)
    return False, 0.0


def _eval_dev4(store: _WaveStore, allow_diagonal: bool) -> tuple[bool, float]:
    if len(store) < 5:
        return False, 0.0
    direction = store.kind_at(1)
    p0, p1, p2, p3, p4 = (store.price_at(back) for back in (4, 3, 2, 1, 0))
    w1, w2, w3, w4 = abs(p1 - p0), abs(p2 - p1), abs(p3 - p2), abs(p4 - p3)
    if w1 <= 0 or w3 <= 0:
        return False, 0.0
    up = direction == 1
    r1 = p2 > p0 if up else p2 < p0
    r3 = allow_diagonal or (p4 > p1 if up else p4 < p1)
    progress = p3 > p1 if up else p3 < p1
    w3_ok = w3 / w1 >= 0.8
    if not (r1 and r3 and progress and w3_ok):
        return False, 0.0
    rr2, e3, rr4 = w2 / w1, w3 / w1, w4 / w3
    s2 = _score(rr2, 0.585, 0.40)
    s3 = max(_score(e3, 1.618, 1.20), 0.85 * _score(e3, 2.618, 1.20))
    s4 = _score(rr4, 0.382, 0.35)
    return True, 100.0 * (0.35 * s2 + 0.35 * s3 + 0.30 * s4)


def _sub_count(store: _WaveStore, first_bar: int, second_bar: int) -> int:
    return sum(1 for bar in store.bar if first_bar < bar < second_bar)


def elliott_wave_radar(
    frame: pd.DataFrame,
    *,
    pivot_len: int = 8,
    sub_len: int = 3,
    min_swing_atr: float = 0.5,
    max_pivots: int = 60,
    allow_diagonal: bool = False,
    min_score: int = 55,
    show_projection: bool = True,
    show_sub_waves: bool = True,
) -> pd.DataFrame:
    """返回主级/子级枢轴、波浪计数、评分、失效价、目标投影与事件。"""
    pivot_len = integer("pivot_len", pivot_len, 3)
    sub_len = integer("sub_len", sub_len, 2)
    min_swing_atr = number("min_swing_atr", min_swing_atr, 0.0)
    max_pivots = integer("max_pivots", max_pivots, 12, 200)
    allow_diagonal = boolean("allow_diagonal", allow_diagonal)
    min_score = integer("min_score", min_score, 0, 100)
    show_projection = boolean("show_projection", show_projection)
    show_sub_waves = boolean("show_sub_waves", show_sub_waves)

    high, low, close = ohlcv(frame, "high", "low", "close")
    highs = high.to_numpy()
    lows = low.to_numpy()
    size = len(frame)

    atr_values = atr(frame, 14).to_numpy()
    ph = pivothigh(high, pivot_len, pivot_len).to_numpy()
    pl = pivotlow(low, pivot_len, pivot_len).to_numpy()
    sph = pivothigh(high, sub_len, sub_len).to_numpy()
    spl = pivotlow(low, sub_len, sub_len).to_numpy()

    primary = _WaveStore(max_pivots, highs, lows)
    sub = _WaveStore(max_pivots, highs, lows)

    count_type = "NONE"
    count_score = 0.0
    count_dir = 0
    invalidation = math.nan
    alt_type = "—"
    alt_score = 0.0
    points_price: list[float] = []
    points_bar: list[int] = []
    impulse_dir = 0
    impulse_end_bar = math.nan
    impulse_end_price = math.nan

    out = {name: np.zeros(size, dtype=bool) for name in
           ("new_pivot", "ev_imp", "ev_abc", "ev_flat", "ev_c", "ev_w3", "ev_w5",
            "invalidation_hit")}
    out["pivot_count"] = np.zeros(size, dtype=int)
    out["pivot_price"] = np.full(size, np.nan)
    out["pivot_bar"] = np.full(size, np.nan)
    out["pivot_type"] = np.full(size, np.nan)
    out["count_dir"] = np.zeros(size, dtype=int)
    out["count_score"] = np.full(size, np.nan)
    out["count_type"] = np.full(size, "", dtype=object)
    out["alt_type"] = np.full(size, "", dtype=object)
    out["alt_score"] = np.full(size, np.nan)
    out["invalidation"] = np.full(size, np.nan)
    out["sub_pivots"] = np.zeros(size, dtype=int)
    out["sub_w1_legs"] = np.full(size, np.nan)
    out["sub_w3_legs"] = np.full(size, np.nan)
    out["sub_structure"] = np.full(size, "", dtype=object)
    for level in range(3):
        out[f"proj_{level + 1}"] = np.full(size, np.nan)
        out[f"proj_{level + 1}_label"] = np.full(size, "", dtype=object)
    for point in range(6):
        out[f"p{point}_price"] = np.full(size, np.nan)
        out[f"p{point}_bar"] = np.full(size, np.nan)

    for bar in range(size):
        new_pivot = False
        if not np.isnan(ph[bar]) and primary.push(bar - pivot_len, float(ph[bar]), 1,
                                                 min_swing_atr * atr_values[bar]
                                                 if np.isfinite(atr_values[bar]) else 0.0):
            new_pivot = True
        if not np.isnan(pl[bar]) and primary.push(bar - pivot_len, float(pl[bar]), -1,
                                                 min_swing_atr * atr_values[bar]
                                                 if np.isfinite(atr_values[bar]) else 0.0):
            new_pivot = True
        if not np.isnan(sph[bar]):
            sub.push(bar - sub_len, float(sph[bar]), 1, 0.0)
        if not np.isnan(spl[bar]):
            sub.push(bar - sub_len, float(spl[bar]), -1, 0.0)
        out["sub_pivots"][bar] = len(sub)

        if new_pivot:
            out["new_pivot"][bar] = True
            out["pivot_count"][bar] = len(primary)
            out["pivot_price"][bar] = primary.price_at(0)
            out["pivot_bar"][bar] = primary.bar_at(0)
            out["pivot_type"][bar] = primary.kind_at(0)

            candidates: list[tuple[str, float]] = []
            for name, result in (
                ("IMP", _eval_impulse(primary, allow_diagonal)),
                ("ABC", _eval_abc(primary)),
                ("FLAT", _eval_flat(primary)),
                ("DEVC", _eval_dev_c(primary, impulse_dir, impulse_end_bar, impulse_end_price)),
                ("DEV4", _eval_dev4(primary, allow_diagonal)),
                ("DEV2", _eval_dev2(primary)),
            ):
                if result[0] and result[1] >= min_score:
                    candidates.append((name, result[1]))

            best_type, best_eff, best_raw = "NONE", -1.0, 0.0
            second_type, second_eff, second_raw = "NONE", -1.0, 0.0
            for name, score in candidates:
                effective = score + _BONUS[name]
                if effective > best_eff:
                    second_type, second_eff, second_raw = best_type, best_eff, best_raw
                    best_type, best_eff, best_raw = name, effective, score
                elif effective > second_eff:
                    second_type, second_eff, second_raw = name, effective, score

            previous_type = count_type
            count_type = best_type
            count_score = best_raw
            alt_type = "—" if second_type == "NONE" else second_type
            alt_score = second_raw if second_type != "NONE" else math.nan

            if count_type != "NONE":
                count_dir = primary.kind_at(0) if count_type in ("IMP", "ABC", "FLAT") else (
                    -impulse_dir if count_type == "DEVC" else primary.kind_at(1))
                points_price = [primary.price_at(_POINTS[count_type] - 1 - index)
                                for index in range(_POINTS[count_type])]
                points_bar = [primary.bar_at(_POINTS[count_type] - 1 - index)
                              for index in range(_POINTS[count_type])]
                if count_type == "IMP":
                    invalidation = math.nan
                    impulse_dir = count_dir
                    impulse_end_bar = float(primary.bar_at(0))
                    impulse_end_price = primary.price_at(0)
                elif count_type in ("ABC", "FLAT"):
                    invalidation = math.nan
                elif count_type == "DEVC":
                    invalidation = impulse_end_price
                elif count_type == "DEV4":
                    invalidation = points_price[1]
                else:
                    invalidation = points_price[0]
            else:
                count_dir = 0
                invalidation = math.nan
                points_price = []
                points_bar = []

            for event, name in (("ev_imp", "IMP"), ("ev_abc", "ABC"), ("ev_flat", "FLAT"),
                                ("ev_c", "DEVC"), ("ev_w3", "DEV2"), ("ev_w5", "DEV4")):
                out[event][bar] = count_type == name and previous_type != name

        # --- invalidation watch (developing counts only) --------------------
        if count_type in ("DEV2", "DEV4", "DEVC") and np.isfinite(invalidation):
            hit = (lows[bar] < invalidation) if count_dir == 1 else (highs[bar] > invalidation)
            if hit:
                out["invalidation_hit"][bar] = True
                count_type, count_score, count_dir = "NONE", 0.0, 0
                invalidation = math.nan
                alt_type, alt_score = "—", math.nan
                points_price, points_bar = [], []

        out["count_type"][bar] = count_type
        out["count_score"][bar] = count_score if count_type != "NONE" else math.nan
        out["count_dir"][bar] = count_dir
        out["alt_type"][bar] = alt_type
        out["alt_score"][bar] = alt_score if alt_type != "—" else math.nan
        out["invalidation"][bar] = invalidation if np.isfinite(invalidation) else math.nan
        for point in range(6):
            if point < len(points_price):
                out[f"p{point}_price"][bar] = points_price[point]
                out[f"p{point}_bar"][bar] = points_bar[point]

        # --- sub-wave structure and projections -----------------------------
        if count_type != "NONE" and len(points_price) >= 3:
            if count_type in ("IMP", "DEV4") and len(points_bar) >= 4:
                w1_legs = _sub_count(sub, points_bar[0], points_bar[1])
                w3_legs = _sub_count(sub, points_bar[2], points_bar[3])
                out["sub_w1_legs"][bar] = w1_legs
                out["sub_w3_legs"][bar] = w3_legs
                if show_sub_waves:
                    out["sub_structure"][bar] = (
                        f"W1: {'5✓' if w1_legs == 4 else w1_legs + 1} legs · "
                        f"W3: {'5✓' if w3_legs == 4 else w3_legs + 1} legs")
            if show_projection:
                direction_factor = 1.0 if count_dir == 1 else -1.0
                targets: list[tuple[float, str]] = []
                if count_type == "DEV2":
                    w1 = abs(points_price[1] - points_price[0])
                    base = points_price[2]
                    targets = [(base + direction_factor * w1 * mult, f"W3 ×{label}")
                               for mult, label in ((1.0, "1.0"), (1.618, "1.618"), (2.618, "2.618"))]
                elif count_type == "DEV4":
                    w1 = abs(points_price[1] - points_price[0])
                    base = points_price[4]
                    targets = [(base + direction_factor * w1 * mult, f"W5 ×{label}")
                               for mult, label in ((0.618, "0.618"), (1.0, "1.0"), (1.618, "1.618"))]
                elif count_type == "DEVC":
                    wa = abs(points_price[1] - points_price[0])
                    base = points_price[2]
                    targets = [(base + direction_factor * wa * mult, f"C ×{label}")
                               for mult, label in ((1.0, "1.0"), (1.618, "1.618"))]
                elif count_type == "IMP":
                    rng = abs(points_price[5] - points_price[0])
                    top = points_price[5]
                    targets = [(top - direction_factor * rng * frac, f"Corr {label}")
                               for frac, label in ((0.382, "38.2%"), (0.5, "50%"), (0.618, "61.8%"))]
                for level, (price, label) in enumerate(targets[:3]):
                    out[f"proj_{level + 1}"][bar] = price
                    out[f"proj_{level + 1}_label"][bar] = label

    index = frame.index
    data: dict[str, pd.Series] = {
        "new_pivot": pd.Series(out["new_pivot"], index=index, dtype=bool),
        "pivot_count": pd.Series(out["pivot_count"], index=index).astype("int64"),
        "pivot_price": pd.Series(out["pivot_price"], index=index),
        "pivot_bar": pd.Series(out["pivot_bar"], index=index).astype("Int64"),
        "pivot_type": pd.Series(out["pivot_type"], index=index).astype("Int64"),
        "count_type": pd.Series(out["count_type"], index=index, dtype=object),
        "count_score": pd.Series(out["count_score"], index=index),
        "count_dir": pd.Series(out["count_dir"], index=index).astype("int64"),
        "alt_type": pd.Series(out["alt_type"], index=index, dtype=object),
        "alt_score": pd.Series(out["alt_score"], index=index),
        "invalidation": pd.Series(out["invalidation"], index=index),
        "sub_pivots": pd.Series(out["sub_pivots"], index=index).astype("int64"),
        "sub_w1_legs": pd.Series(out["sub_w1_legs"], index=index).astype("Int64"),
        "sub_w3_legs": pd.Series(out["sub_w3_legs"], index=index).astype("Int64"),
        "sub_structure": pd.Series(out["sub_structure"], index=index, dtype=object),
        "ev_imp": pd.Series(out["ev_imp"], index=index, dtype=bool),
        "ev_abc": pd.Series(out["ev_abc"], index=index, dtype=bool),
        "ev_flat": pd.Series(out["ev_flat"], index=index, dtype=bool),
        "ev_c": pd.Series(out["ev_c"], index=index, dtype=bool),
        "ev_w3": pd.Series(out["ev_w3"], index=index, dtype=bool),
        "ev_w5": pd.Series(out["ev_w5"], index=index, dtype=bool),
        "invalidation_hit": pd.Series(out["invalidation_hit"], index=index, dtype=bool),
    }
    for level in range(3):
        data[f"proj_{level + 1}"] = pd.Series(out[f"proj_{level + 1}"], index=index)
        data[f"proj_{level + 1}_label"] = pd.Series(out[f"proj_{level + 1}_label"],
                                                    index=index, dtype=object)
    for point in range(6):
        data[f"p{point}_price"] = pd.Series(out[f"p{point}_price"], index=index)
        data[f"p{point}_bar"] = pd.Series(out[f"p{point}_bar"], index=index).astype("Int64")
    return pd.DataFrame(data, index=index)

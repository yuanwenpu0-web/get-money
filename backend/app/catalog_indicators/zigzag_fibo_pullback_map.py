"""ZigZag Fibo Pullback Map [WavesUnchained] — ZigZag 回撤到斐波那契的标注与活动扇形。

**枢轴引擎**：`ta.pivothigh/pivotlow(high/low, pivot_len)` 对称确认；同类型只保留
更极端者，交替才入库，上限 `max_pivots`。

**回撤锁定**（需要枢轴 i 的下一个反向枢轴 i+1 存在，避免把还会被延长的临时回撤
当终点）：取 i-2 / i-1 / i 三点为 a / b / c，
- `impulse_up = b > a`，参考腿长 `|b-a|`，回撤 `|c-b|`，`ratio = 回撤/腿长`。
- 腿过滤：`|b-a| >= min_leg_atr×ATR` 且 `|b-a|/|b|×100 >= min_leg_pct`。
- 只统计 `0.236 <= ratio <= max_ratio`。

**斐波那契匹配**：标准阶梯 0.236/0.382/0.5/0.618/0.786。
`matched` = 距离最近且 ≤ `match_tolerance` 的标准位；`reached` = ≤ ratio 的最深标准位；
`dominant = matched or reached`。连续多个回撤落在同一 dominant 上累加 `cluster`。

**深度/效率**：`Shallow(<0.382)`、`Normal(≤0.618)`、`Deep(≤0.786)`、`Extreme`；
效率 = 冲量速度（腿长/冲量根数）÷ 回撤速度（回撤/回撤根数）。

**反应**（需 i+2 存在）：`d = piv[i+1]`、`e = piv[i+2]`；
`impulse_up ? (e<c → Break, d>b → HH, else Weak) : (e>c → Break, d<c → LL, else Weak)`。

**展望评分**：深度（≤0.382 +2 / ≤0.618 +1 / ≤0.786 −0.5 / 其他 −2）+ 时间
（≤5 根 +1 / ≤10 0 / 其他 −1）+ 效率（≥1.2 +1.5 / ≥0.9 +0.5 / ≥0.6 −0.5 / 其他 −1.5）
+ 反应（HH/LL +1.5 / Weak −0.5 / Break −2）→ ≥3 Continuation Likely、
≥1 Balanced、>−1 Fragile、否则 Failure Risk。

**活动扇形**（未完成腿）：`base0 = piv[n-2]`、`base1 = piv[n-1]`，在 base1 之后回看
`draw_lookback_bars` 根找极值（Pine 从最近向旧扫描，仅在更极端时替换 → 取最旧极值），
`live_ratio = |极值-base1| / |base0-base1|`，各标准位价格 = `base1 + (base0-base1)*ratio`。

**Confluence 区**：满足 `ratio ∈ [0.236,0.786]`、回撤 ≤16 根、效率 ≥0.55（或缺失）、
反应非 Break、展望分 ≥ −0.5 的回撤进入最近 `confluence_lookback` 个候选；
候选数 ≥2 且落在均值 ± `confluence_tol_atr×ATR` 内的 ≥2 时输出均值区。

丢弃：全部 line/label/box/table 绘图、统计表、颜色/透明度样式参数、tooltip。

Source: pine-collection/indicators/market_structure/zigzag_fibo_pullback_map/zigzag_fibo_pullback_map.pine
License: WavesUnchained 公开脚本（随清单收录）。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, boolean, choice, flags, integer, number, ohlcv, pivothigh, pivotlow

__all__ = ["zigzag_fibo_pullback_map"]

_STANDARD = (0.236, 0.382, 0.500, 0.618, 0.786)
_EXTENSIONS = (1.000, 1.272, 1.414, 1.618, 2.000, 2.272, 2.618)


class _Pivots:
    """Alternating pivot history with same-type replacement and a hard cap."""

    __slots__ = ("price", "index", "kind", "cap")

    def __init__(self, cap: int) -> None:
        self.price: list[float] = []
        self.index: list[int] = []
        self.kind: list[int] = []
        self.cap = cap

    def __len__(self) -> int:
        return len(self.kind)

    def push(self, bar: int, price: float, kind: int) -> bool:
        if self.kind and self.kind[-1] == kind:
            last = self.price[-1]
            replace = price > last if kind == 1 else price < last
            if replace:
                self.price[-1] = price
                self.index[-1] = bar
            return False
        self.kind.append(kind)
        self.price.append(price)
        self.index.append(bar)
        while len(self.kind) > self.cap:
            self.kind.pop(0)
            self.index.pop(0)
            self.price.pop(0)
        return True


def _nearest_standard(ratio: float, tolerance: float) -> float:
    best, best_dist = math.nan, 10.0
    for candidate in _STANDARD:
        distance = abs(candidate - ratio)
        if distance < best_dist:
            best, best_dist = candidate, distance
    return best if best_dist <= tolerance else math.nan


def _reached_standard(ratio: float) -> float:
    reached = math.nan
    for candidate in _STANDARD:
        if ratio >= candidate:
            reached = candidate
    return reached


def _depth_regime(ratio: float) -> str:
    if ratio < 0.382:
        return "Shallow"
    if ratio <= 0.618:
        return "Normal"
    if ratio <= 0.786:
        return "Deep"
    return "Extreme"


def _zone_bounds(ratio: float) -> tuple[float, float]:
    lower = 0.0
    for edge in (0.236, 0.382, 0.500, 0.618, 0.786, 1.000):
        if ratio >= edge:
            lower = edge
    upper = 0.236
    for edge in (0.236, 0.382, 0.500, 0.618, 0.786, 1.000, 1.272):
        upper = edge
        if ratio < edge:
            break
    return lower, upper


def _zone_text(ratio: float) -> str:
    edges = ((0.236, "Below 0.236"), (0.382, "0.236-0.382"), (0.500, "0.382-0.500"),
             (0.618, "0.500-0.618"), (0.786, "0.618-0.786"), (1.000, "0.786-1.000"))
    for edge, text in edges:
        if ratio < edge:
            return text
    return "> 1.000" if ratio > 1.0 else "0.786-1.000"


def _outlook_score(ratio: float, bars: int, efficiency: float, reaction: str) -> float:
    score = 2.0 if ratio <= 0.382 else 1.0 if ratio <= 0.618 else -0.5 if ratio <= 0.786 else -2.0
    score += 1.0 if bars <= 5 else 0.0 if bars <= 10 else -1.0
    if np.isfinite(efficiency):
        score += 1.5 if efficiency >= 1.2 else 0.5 if efficiency >= 0.9 else -0.5 if efficiency >= 0.6 else -1.5
    score += 1.5 if reaction in ("HH", "LL") else -0.5 if reaction == "Weak" else -2.0 if reaction == "Break" else 0.0
    return score


def _outlook_text(score: float) -> str:
    if score >= 3.0:
        return "Continuation Likely"
    if score >= 1.0:
        return "Balanced Continuation"
    if score > -1.0:
        return "Fragile Continuation"
    return "Failure Risk"


def _zone_bias(outlook: str) -> str:
    if outlook == "Failure Risk":
        return "risk"
    return "fragile" if outlook == "Fragile Continuation" else "supportive"


def _outcome_text(reaction: str) -> str:
    return {"HH": "HH formed", "LL": "LL formed", "Break": "Structure Break",
            "Weak": "Weak follow-through"}.get(reaction, "Pending")


def _efficiency(distance_a: float, bars_a: int, distance_b: float, bars_b: int) -> float:
    rate_a = distance_a / bars_a if bars_a > 0 else math.nan
    rate_b = distance_b / bars_b if bars_b > 0 else math.nan
    if not np.isfinite(rate_a) or not np.isfinite(rate_b) or rate_b <= 0:
        return math.nan
    return rate_a / rate_b


def _reaction_text(impulse_up: bool, b: float, c: float, d: float, e: float) -> str:
    if impulse_up:
        return "Break" if e < c else "HH" if d > b else "Weak"
    return "Break" if e > c else "LL" if d < c else "Weak"


def zigzag_fibo_pullback_map(
    frame: pd.DataFrame,
    *,
    pivot_len: int = 8,
    max_pivots: int = 80,
    draw_lookback_bars: int = 1500,
    show_pullback_labels: bool = True,
    pullback_max_ratio: float = 1.0,
    match_tolerance: float = 0.03,
    show_extensions: bool = True,
    show_confluence: bool = True,
    confluence_lookback: int = 5,
    confluence_tol_atr: float = 0.35,
    atr_len: int = 14,
    min_leg_atr: float = 0.30,
    min_leg_pct: float = 0.10,
    label_mode: str = "Standard + Reached",
    active_fib_display: str = "Key Levels",
) -> pd.DataFrame:
    """返回回撤标注（斐波那契匹配/深度/效率/反应/展望）与活动扇形水平。"""
    pivot_len = integer("pivot_len", pivot_len, 2)
    max_pivots = integer("max_pivots", max_pivots, 20, 200)
    draw_lookback_bars = integer("draw_lookback_bars", draw_lookback_bars, 200, 5000)
    show_pullback_labels = boolean("show_pullback_labels", show_pullback_labels)
    pullback_max_ratio = number("pullback_max_ratio", pullback_max_ratio, 0.236, 1.272)
    match_tolerance = number("match_tolerance", match_tolerance, 0.001, 0.100)
    show_extensions = boolean("show_extensions", show_extensions)
    show_confluence = boolean("show_confluence", show_confluence)
    confluence_lookback = integer("confluence_lookback", confluence_lookback, 2, 12)
    confluence_tol_atr = number("confluence_tol_atr", confluence_tol_atr, 0.05, 2.0)
    atr_len = integer("atr_len", atr_len, 2)
    min_leg_atr = number("min_leg_atr", min_leg_atr, 0.0, 10.0)
    min_leg_pct = number("min_leg_pct", min_leg_pct, 0.0, 10.0)
    label_mode = choice("label_mode", label_mode, ("All", "Standard + Reached", "Standard Only"))
    active_fib_display = choice("active_fib_display", active_fib_display,
                                ("Auto", "Current Zone", "Key Levels", "Full Ladder"))

    high, low, close = ohlcv(frame, "high", "low", "close")
    highs, lows = high.to_numpy(), low.to_numpy()
    size = len(frame)
    atr_values = atr(frame, atr_len).to_numpy()
    ph = pivothigh(high, pivot_len, pivot_len).to_numpy()
    pl = pivotlow(low, pivot_len, pivot_len).to_numpy()

    pivots = _Pivots(max_pivots)
    records: list[dict] = []
    prior_cluster_fib = math.nan
    cluster_count = 0
    confluence: list[dict] = []

    ladder = list(_STANDARD) + ([1.000] + list(_EXTENSIONS[1:]) if show_extensions else [])

    out = {name: np.zeros(size, dtype=bool) for name in
           ("pivot_new", "pullback", "reaction_update", "confluence_zone")}
    out["pivot_count"] = np.zeros(size, dtype=int)
    out["pivot_price"] = np.full(size, np.nan)
    out["pivot_type"] = np.full(size, np.nan)
    for name in ("pb_ratio", "pb_matched", "pb_reached", "pb_dominant", "pb_level",
                 "pb_efficiency", "pb_bars", "pb_price", "pb_bar", "pb_score",
                 "pb_cluster", "pb_impulse_bars", "pb_leg_size"):
        out[name] = np.full(size, np.nan)
    for name in ("pb_depth", "pb_reaction", "pb_outcome", "pb_outlook", "pb_bias"):
        out[name] = np.full(size, "", dtype=object)
    out["pb_impulse_up"] = np.zeros(size, dtype=bool)
    for name in ("live_ratio", "live_extreme", "live_next_fib", "live_zone_lo", "live_zone_hi"):
        out[name] = np.full(size, np.nan)
    out["live_zone"] = np.full(size, "", dtype=object)
    out["live_move_up"] = np.zeros(size, dtype=bool)
    for ratio in ladder:
        out[f"live_level_{int(round(ratio * 1000))}"] = np.full(size, np.nan)
    out["conf_level"] = np.full(size, np.nan)
    out["conf_hits"] = np.zeros(size, dtype=int)
    out["conf_outlook"] = np.full(size, "", dtype=object)

    def _relevant(matched: float, reached: float) -> bool:
        if label_mode == "All":
            return True
        if label_mode == "Standard Only":
            return np.isfinite(matched)
        return np.isfinite(matched) or np.isfinite(reached)

    for bar in range(size):
        new_pivot = False
        pivot_bar = bar - pivot_len
        if not np.isnan(ph[bar]) or not np.isnan(pl[bar]):
            both = not np.isnan(ph[bar]) and not np.isnan(pl[bar])
            if both:
                last_kind = pivots.kind[-1] if len(pivots) else 0
                order = ((pl[bar], -1), (ph[bar], 1)) if last_kind == 1 else ((ph[bar], 1), (pl[bar], -1))
            elif not np.isnan(ph[bar]):
                order = ((ph[bar], 1),)
            else:
                order = ((pl[bar], -1),)
            for price, kind in order:
                new_pivot = pivots.push(pivot_bar, float(price), kind) or new_pivot

        if new_pivot:
            out["pivot_new"][bar] = True
            out["pivot_count"][bar] = len(pivots)
            out["pivot_price"][bar] = pivots.price[-1]
            out["pivot_type"][bar] = pivots.kind[-1]
            if len(pivots) >= 4:
                position = len(pivots) - 2
                a, b, c = (pivots.price[position - 2], pivots.price[position - 1],
                           pivots.price[position])
                a_idx, b_idx, c_idx = (pivots.index[position - 2], pivots.index[position - 1],
                                       pivots.index[position])
                impulse_up = b > a
                leg_size = abs(b - a)
                pullback_size = abs(c - b)
                impulse_bars = max(1, b_idx - a_idx)
                pullback_bars = max(1, c_idx - b_idx)
                ratio = pullback_size / leg_size if leg_size > 0 else math.nan
                leg_percent = leg_size / abs(b) * 100.0 if b != 0 else 0.0
                atr_now = atr_values[bar]
                qualified = (leg_size >= (atr_now * min_leg_atr if np.isfinite(atr_now) else math.inf)
                             and leg_percent >= min_leg_pct)
                if np.isfinite(ratio) and 0.236 <= ratio <= pullback_max_ratio and qualified:
                    matched = _nearest_standard(ratio, match_tolerance)
                    reached = _reached_standard(ratio)
                    dominant = matched if np.isfinite(matched) else reached
                    efficiency = _efficiency(leg_size, impulse_bars, pullback_size, pullback_bars)
                    record = {"position": position, "a": a, "b": b, "c": c, "index": c_idx,
                              "up": impulse_up, "ratio": ratio, "matched": matched,
                              "reached": reached, "dominant": dominant,
                              "bars": pullback_bars, "impulse_bars": impulse_bars,
                              "efficiency": efficiency, "leg_size": leg_size,
                              "reaction": "Pending", "resolved": False}
                    if np.isfinite(dominant) and np.isfinite(prior_cluster_fib) \
                            and abs(dominant - prior_cluster_fib) < 0.0005:
                        cluster_count += 1
                    else:
                        cluster_count = 1
                    prior_cluster_fib = dominant
                    record["cluster"] = cluster_count
                    records.append(record)

                    if show_pullback_labels and _relevant(matched, reached):
                        out["pullback"][bar] = True
                        out["pb_ratio"][bar] = ratio
                        out["pb_matched"][bar] = matched
                        out["pb_reached"][bar] = reached
                        out["pb_dominant"][bar] = dominant
                        out["pb_level"][bar] = (b + (a - b) * dominant
                                                if np.isfinite(dominant) else math.nan)
                        out["pb_efficiency"][bar] = efficiency
                        out["pb_bars"][bar] = pullback_bars
                        out["pb_impulse_bars"][bar] = impulse_bars
                        out["pb_leg_size"][bar] = leg_size
                        out["pb_price"][bar] = c
                        out["pb_bar"][bar] = c_idx
                        out["pb_cluster"][bar] = cluster_count
                        out["pb_impulse_up"][bar] = impulse_up
                        out["pb_depth"][bar] = _depth_regime(ratio)

        # --- reaction resolution (needs the pivot two steps after c) --------
        for record in records:
            if record["resolved"]:
                continue
            position = record["position"]
            if position + 2 > len(pivots) - 1:
                continue
            d = pivots.price[position + 1]
            e = pivots.price[position + 2]
            record["reaction"] = _reaction_text(record["up"], record["b"], record["c"], d, e)
            record["resolved"] = True
            out["reaction_update"][bar] = True

        # --- snapshot of the newest record ---------------------------------
        if records:
            record = records[-1]
            score = _outlook_score(record["ratio"], record["bars"], record["efficiency"],
                                   record["reaction"])
            outlook = _outlook_text(score)
            out["pb_score"][bar] = score
            out["pb_reaction"][bar] = record["reaction"]
            out["pb_outcome"][bar] = _outcome_text(record["reaction"])
            out["pb_outlook"][bar] = outlook
            out["pb_bias"][bar] = _zone_bias(outlook)
            dominant = record["dominant"]
            relevant = _relevant(record["matched"], record["reached"])
            candidate = (show_confluence and relevant and np.isfinite(dominant)
                         and 0.236 <= record["ratio"] <= 0.786 and record["bars"] <= 16
                         and (not np.isfinite(record["efficiency"])
                              or record["efficiency"] >= 0.55)
                         and record["reaction"] != "Break" and score >= -0.5)
            if candidate:
                confluence.append({"level": record["b"] + (record["a"] - record["b"]) * dominant,
                                   "index": record["index"], "bias": _zone_bias(outlook),
                                   "outlook": outlook})
                while len(confluence) > confluence_lookback:
                    confluence.pop(0)
        while len(records) > max_pivots:
            records.pop(0)

        # --- active (unfinished) fan ---------------------------------------
        if len(pivots) >= 3:
            base0 = pivots.price[-2]
            base1 = pivots.price[-1]
            base_index = pivots.index[-1]
            lookback = min(bar - base_index + 1, draw_lookback_bars)
            if base_index >= 0 and lookback > 0:
                last_kind = pivots.kind[-1]
                extreme = base1
                for offset in range(lookback):
                    index = bar - offset
                    if index < 0:
                        break
                    if last_kind == -1 and highs[index] > extreme:
                        extreme = highs[index]
                    elif last_kind == 1 and lows[index] < extreme:
                        extreme = lows[index]
                reference = abs(base0 - base1)
                live_ratio = abs(extreme - base1) / reference if reference > 0 else math.nan
                out["live_extreme"][bar] = extreme
                out["live_move_up"][bar] = extreme > base1
                out["live_ratio"][bar] = live_ratio
                if np.isfinite(live_ratio):
                    out["live_zone"][bar] = _zone_text(live_ratio)
                    lo_ratio, hi_ratio = _zone_bounds(live_ratio)
                    out["live_zone_lo"][bar] = lo_ratio
                    out["live_zone_hi"][bar] = hi_ratio
                    next_fib = math.nan
                    for candidate_ratio in _STANDARD:
                        if candidate_ratio > live_ratio:
                            next_fib = candidate_ratio
                            break
                    out["live_next_fib"][bar] = next_fib
                for ratio in ladder:
                    out[f"live_level_{int(round(ratio * 1000))}"][bar] = base1 + (base0 - base1) * ratio

        # --- confluence zone ---------------------------------------------
        if confluence and len(confluence) >= 2:
            atr_now = atr_values[bar]
            tol = atr_now * confluence_tol_atr if np.isfinite(atr_now) else math.nan
            average = sum(item["level"] for item in confluence) / len(confluence)
            hits = 0
            if np.isfinite(tol):
                hits = sum(1 for item in confluence if abs(item["level"] - average) <= tol)
            out["conf_hits"][bar] = hits
            if hits >= 2:
                out["confluence_zone"][bar] = True
                out["conf_level"][bar] = average
                out["conf_outlook"][bar] = confluence[-1]["outlook"]

    index = frame.index
    data: dict[str, pd.Series] = {
        "pivot_new": flags(pd.Series(out["pivot_new"], index=index)),
        "pivot_count": pd.Series(out["pivot_count"], index=index).astype("int64"),
        "pivot_price": pd.Series(out["pivot_price"], index=index),
        "pivot_type": pd.Series(out["pivot_type"], index=index).astype("Int64"),
        "pullback": flags(pd.Series(out["pullback"], index=index)),
        "pb_ratio": pd.Series(out["pb_ratio"], index=index),
        "pb_matched": pd.Series(out["pb_matched"], index=index),
        "pb_reached": pd.Series(out["pb_reached"], index=index),
        "pb_dominant": pd.Series(out["pb_dominant"], index=index),
        "pb_level": pd.Series(out["pb_level"], index=index),
        "pb_price": pd.Series(out["pb_price"], index=index),
        "pb_bar": pd.Series(out["pb_bar"], index=index).astype("Int64"),
        "pb_bars": pd.Series(out["pb_bars"], index=index).astype("Int64"),
        "pb_impulse_bars": pd.Series(out["pb_impulse_bars"], index=index).astype("Int64"),
        "pb_leg_size": pd.Series(out["pb_leg_size"], index=index),
        "pb_efficiency": pd.Series(out["pb_efficiency"], index=index),
        "pb_cluster": pd.Series(out["pb_cluster"], index=index).astype("Int64"),
        "pb_depth": pd.Series(out["pb_depth"], index=index, dtype=object),
        "pb_impulse_up": pd.Series(out["pb_impulse_up"], index=index, dtype=bool),
        "pb_reaction": pd.Series(out["pb_reaction"], index=index, dtype=object),
        "pb_outcome": pd.Series(out["pb_outcome"], index=index, dtype=object),
        "pb_score": pd.Series(out["pb_score"], index=index),
        "pb_outlook": pd.Series(out["pb_outlook"], index=index, dtype=object),
        "pb_bias": pd.Series(out["pb_bias"], index=index, dtype=object),
        "reaction_update": pd.Series(out["reaction_update"], index=index, dtype=bool),
        "live_ratio": pd.Series(out["live_ratio"], index=index),
        "live_extreme": pd.Series(out["live_extreme"], index=index),
        "live_next_fib": pd.Series(out["live_next_fib"], index=index),
        "live_zone": pd.Series(out["live_zone"], index=index, dtype=object),
        "live_zone_lo": pd.Series(out["live_zone_lo"], index=index),
        "live_zone_hi": pd.Series(out["live_zone_hi"], index=index),
        "live_move_up": pd.Series(out["live_move_up"], index=index, dtype=bool),
        "confluence_zone": pd.Series(out["confluence_zone"], index=index, dtype=bool),
        "conf_level": pd.Series(out["conf_level"], index=index),
        "conf_hits": pd.Series(out["conf_hits"], index=index).astype("int64"),
        "conf_outlook": pd.Series(out["conf_outlook"], index=index, dtype=object),
    }
    for ratio in ladder:
        data[f"live_level_{int(round(ratio * 1000))}"] = pd.Series(
            out[f"live_level_{int(round(ratio * 1000))}"], index=index)
    return pd.DataFrame(data, index=index)

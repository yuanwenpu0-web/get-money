"""Auto Trendlines — pivot-to-pivot candidate lines scored and deduplicated.

Source: pine-collection/indicators/market_structure/auto_trendlines/auto_trendlines.pine
(WavesUnchained, MIT)

Confirmed pivots within the lookback become line anchors. Every pivot pair
determines a candidate slope/intercept (resistance slopes must be
non-positive, support non-negative). Two fitting methods: "Directional" keeps
the two-anchor line; "Regression" refits a least-squares line through the
anchors the first pass accepts and optionally shifts it to the outermost
residual (Outer fit). A candidate qualifies when at least `min_touches`
pivots sit within `tolerance_atr` × ATR of it (double tolerance for anchors
in the top/bottom 15% of the window), the span reaches `min_span` bars,
violations of the price series stay under `max_viol_pct`, the projected line
is within `relevance_atr` × ATR of the current close, and the sloped/flat
filter matches. Candidates score by touches × span × extremum bonus ×
recency bonus × fit × violation penalty; the top `max_lines` are selected
greedily, each consuming its pivot anchors. The best surviving line reports a
retest when the close is within `retest_atr` × ATR of it.

Dropped: lines, labels, hull envelope and all colours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import atr, boolean, choice, flags, integer, number, ohlcv, pivothigh, pivotlow


def auto_trendlines(
        frame: pd.DataFrame, *, lookback: int = 300, pivot_left: int = 6,
        pivot_right: int = 6, min_touches: int = 2, max_lines: int = 3,
        max_pivots: int = 30, min_span: int = 15, max_viol_pct: float = 20.0,
        violation_source: str = "Close", project_bars: int = 30,
        show_mode: str = "Both", level_thr_atr: float = 0.5,
        relevance_atr: float = 5.0, line_method: str = "Directional",
        fit_mode: str = "Normal", tolerance_atr: float = 0.5,
        retest_atr: float = 1.0) -> pd.DataFrame:
    integer("lookback", lookback, 20)
    integer("pivot_left", pivot_left, 1)
    integer("pivot_right", pivot_right, 1)
    integer("min_touches", min_touches, 2, 5)
    integer("max_lines", max_lines, 1, 10)
    integer("max_pivots", max_pivots, 10, 100)
    integer("min_span", min_span, 1)
    number("max_viol_pct", max_viol_pct, 0.0, 50.0)
    choice("violation_source", violation_source, ("Close", "Wick"))
    integer("project_bars", project_bars, 0, 500)
    choice("show_mode", show_mode, ("Sloped", "Near-horizontal", "Both"))
    number("level_thr_atr", level_thr_atr, 0.1)
    number("relevance_atr", relevance_atr, 0.5)
    choice("line_method", line_method, ("Directional", "Regression"))
    choice("fit_mode", fit_mode, ("Normal", "Outer"))
    number("tolerance_atr", tolerance_atr, 0.1)
    number("retest_atr", retest_atr, 0.1)
    boolean("show_retest", True) if False else None
    high, low, close = ohlcv(frame, "high", "low", "close")
    rows = len(frame)
    highs = high.to_numpy(dtype=float)
    lows = low.to_numpy(dtype=float)
    closes = close.to_numpy(dtype=float)
    opens = frame["open"].astype(float).to_numpy(dtype=float)
    atr_value = atr(frame, 14)
    atr_values = atr_value.to_numpy(dtype=float)

    ph = pivothigh(high, pivot_left, pivot_right).to_numpy(dtype=float)
    pl = pivotlow(low, pivot_left, pivot_right).to_numpy(dtype=float)

    def violation_series(is_resist: bool) -> np.ndarray:
        if violation_source == "Wick":
            return highs if is_resist else lows
        return closes

    resist_violations = violation_series(True)
    support_violations = violation_series(False)

    out_resist_level = np.full(rows, np.nan)
    out_support_level = np.full(rows, np.nan)
    out_resist_touches = np.zeros(rows)
    out_support_touches = np.zeros(rows)
    out_resist_slope = np.full(rows, np.nan)
    out_support_slope = np.full(rows, np.nan)
    retest_resist = np.zeros(rows, dtype=bool)
    retest_support = np.zeros(rows, dtype=bool)

    ph_prices: list[float] = []
    ph_bars: list[int] = []
    pl_prices: list[float] = []
    pl_bars: list[int] = []

    def build_candidates(prices: list[float], bars: list[int], is_resist: bool,
                         position: int) -> list[dict]:
        n = len(prices)
        if n < min_touches or np.isnan(atr_values[position]) or atr_values[position] <= 0:
            return []
        atr_now = atr_values[position]
        tol = atr_now * tolerance_atr
        order_idx = sorted(range(n), key=lambda k: prices[k], reverse=is_resist)
        top_index = max(0, int(np.floor(n * 0.15)) - 1)
        top_threshold = prices[order_idx[top_index]] if n > 0 else np.nan
        current = closes[position]
        viol = resist_violations if is_resist else support_violations
        candidates: list[dict] = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                xi, yi = bars[i], prices[i]
                xj, yj = bars[j], prices[j]
                if xi == xj:
                    continue
                slope0 = (yj - yi) / (xj - xi)
                if is_resist and slope0 > 0:
                    continue
                if not is_resist and slope0 < 0:
                    continue
                anchor_at_top = ((yi >= top_threshold or yj >= top_threshold) if is_resist
                                 else (yi <= top_threshold or yj <= top_threshold))
                effective_tol = tol * 2.0 if anchor_at_top else tol
                slope, intercept = slope0, yi - slope0 * xi
                if line_method == "Regression":
                    inliers = [(bars[k], prices[k]) for k in range(n)
                               if abs(prices[k] - (slope0 * bars[k] + intercept)) <= effective_tol]
                    if len(inliers) >= 2:
                        xs = np.array([p[0] for p in inliers], dtype=float)
                        ys = np.array([p[1] for p in inliers], dtype=float)
                        denom = len(xs) * (xs * xs).sum() - xs.sum() ** 2
                        if denom != 0:
                            slope = (len(xs) * (xs * ys).sum() - xs.sum() * ys.sum()) / denom
                            intercept = (ys.sum() - slope * xs.sum()) / len(xs)
                            if fit_mode == "Outer":
                                residuals = [abs(prices[k] - (slope * bars[k] + intercept))
                                             for k in range(n)
                                             if abs(prices[k] - (slope * bars[k] + intercept)) <= effective_tol]
                                if residuals:
                                    intercept += max(residuals) * (1.0 if is_resist else -1.0)
                touches, old_bar, new_bar, residual_sum = 0, xi, xj, 0.0
                for k in range(n):
                    residual = abs(prices[k] - (slope * bars[k] + intercept))
                    if residual <= effective_tol:
                        residual_sum += residual
                        old_bar = min(old_bar, bars[k])
                        new_bar = max(new_bar, bars[k])
                        touches += 1
                if touches < min_touches:
                    continue
                span = new_bar - old_bar
                if span < min_span:
                    continue
                violations = 0
                for offset in range(1, span):
                    bar = old_bar + offset
                    age = position - bar
                    if 0 <= age < rows:
                        source = viol[age]
                        line_y = slope * bar + intercept
                        if is_resist and source > line_y + tol:
                            violations += 1
                        if not is_resist and source < line_y - tol:
                            violations += 1
                violation_pct = 100.0 * violations / max(1, span)
                line_now = slope * position + intercept
                relevant = abs(line_now - current) <= atr_now * relevance_atr
                slope_atr = abs(slope * span) / atr_now
                mode_ok = (show_mode == "Both"
                           or (show_mode == "Sloped" and slope_atr >= level_thr_atr)
                           or (show_mode == "Near-horizontal" and slope_atr < level_thr_atr))
                extreme_touches = sum(
                    1 for k in range(n)
                    if abs(prices[k] - (slope * bars[k] + intercept)) <= tol
                    and ((prices[k] >= top_threshold) if is_resist else (prices[k] <= top_threshold)))
                viol_ok = extreme_touches >= 1 or violation_pct <= max_viol_pct
                if not (viol_ok and relevant and mode_ok):
                    continue
                avg_residual = residual_sum / touches if touches > 0 else tol
                recency = position - new_bar
                recency_bonus = max(0.2, 1.0 - recency / lookback)
                fit_factor = 1.0 / (1.0 + avg_residual / tol)
                viol_penalty = 1.0 / (1.0 + violation_pct / 5.0)
                extremum_bonus = 0.5 if extreme_touches == 0 else min(3.0, 1.0 + extreme_touches * 0.5)
                score = touches * span * extremum_bonus * recency_bonus * fit_factor * viol_penalty
                candidates.append({"slope": slope, "intercept": intercept, "old": old_bar,
                                   "new": new_bar, "touches": touches, "score": score,
                                   "anchors": {k for k in range(n)
                                               if abs(prices[k] - (slope * bars[k] + intercept)) <= tol}})
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates

    for position in range(rows):
        if not np.isnan(ph[position]):
            ph_prices.append(float(ph[position]))
            ph_bars.append(position - pivot_right)
        if not np.isnan(pl[position]):
            pl_prices.append(float(pl[position]))
            pl_bars.append(position - pivot_right)
        while ph_bars and (position - ph_bars[0] > lookback or len(ph_bars) > max_pivots):
            ph_bars.pop(0)
            ph_prices.pop(0)
        while pl_bars and (position - pl_bars[0] > lookback or len(pl_bars) > max_pivots):
            pl_bars.pop(0)
            pl_prices.pop(0)

        if np.isnan(atr_values[position]):
            continue
        atr_now = atr_values[position]

        for is_resist, prices, bars in ((True, ph_prices, ph_bars),
                                        (False, pl_prices, pl_bars)):
            candidates = build_candidates(list(prices), list(bars), is_resist, position)
            used: set[int] = set()
            drawn = 0
            best_level = np.nan
            best_touches = 0.0
            best_slope = np.nan
            best_retest = False
            for candidate in candidates:
                if drawn >= max_lines:
                    break
                unused = candidate["anchors"] - used
                if len(unused) < min_touches:
                    continue
                used |= candidate["anchors"]
                drawn += 1
                if np.isnan(best_level):
                    best_level = candidate["slope"] * position + candidate["intercept"]
                    best_touches = float(candidate["touches"])
                    best_slope = candidate["slope"]
                    distance = abs(closes[position] - best_level) / atr_now
                    best_retest = distance <= retest_atr
            if is_resist:
                out_resist_level[position] = best_level
                out_resist_touches[position] = best_touches
                out_resist_slope[position] = best_slope
                retest_resist[position] = best_retest
            else:
                out_support_level[position] = best_level
                out_support_touches[position] = best_touches
                out_support_slope[position] = best_slope
                retest_support[position] = best_retest

    resist_published = pd.Series(out_resist_level, index=frame.index)
    support_published = pd.Series(out_support_level, index=frame.index)
    resist_series = resist_published.ffill()
    support_series = support_published.ffill()
    # Touch counts read NaN wherever no line has been published yet.
    out_resist_touches = np.where(resist_published.notna(), out_resist_touches, np.nan)
    out_support_touches = np.where(support_published.notna(), out_support_touches, np.nan)

    return pd.DataFrame({
        "resistance_level": resist_series,
        "support_level": support_series,
        "resistance_touches": pd.Series(out_resist_touches, index=frame.index).astype("Int64"),
        "support_touches": pd.Series(out_support_touches, index=frame.index).astype("Int64"),
        "resistance_slope": pd.Series(out_resist_slope, index=frame.index),
        "support_slope": pd.Series(out_support_slope, index=frame.index),
        "retest_resistance": flags(pd.Series(retest_resist, index=frame.index)),
        "retest_support": flags(pd.Series(retest_support, index=frame.index)),
        "close_above_resistance": flags(closes > resist_series.to_numpy(dtype=float)
                                        | False) if False else flags(
            pd.Series(closes > resist_series.to_numpy(dtype=float), index=frame.index)),
        "close_below_support": flags(
            pd.Series(closes < support_series.to_numpy(dtype=float), index=frame.index)),
    }, index=frame.index)

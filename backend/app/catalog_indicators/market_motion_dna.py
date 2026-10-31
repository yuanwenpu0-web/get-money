"""Market Motion DNA [WavesUnchained]  v1.5.2  —  Python port.

Objective movement-character engine. It does NOT count Elliott waves. It
classifies each confirmed/developing pivot-to-pivot leg as primarily impulsive,
corrective, or neutral using measurable price-action properties:

  * Directional efficiency (net distance / path distance)
  * Candle-range overlap density
  * Internal retracement pressure (max adverse excursion / net distance)
  * Directional close commitment
  * Wick cleanliness
  * Momentum expansion (2nd-half vs 1st-half directional body, ATR-floored)
  * True-range expansion (2nd-half vs 1st-half TR)
  * Moving-average support (price + slope)
  * Internal direction-change cleanliness
  * Directional acceptance
  * Impulse / Correction scores (weighted blend) and classification confidence

The Impulse/Correction scores always sum to 100. A green/red leg means
"impulsive / corrective", NOT "bullish / bearish" — direction is independent
of character.

Computational core ported:
  * Alternating confirmed-pivot engine with same-type extreme replacement
  * Live pivot-to-price leg DNA (the indicator's exposed live readout)
  * Impulse/Correction scores, classification, live-class hysteresis

Dropped (display / logging / non-computational):
  * All box / label / line / barcolor / alertcondition / plot drawing
  * Segment-level envelope colouring and "coherent-wave" merge: this only
    re-paints completed-leg *boxes*; the live readout is computed straight
    from the canonical alternating pivot list and is unaffected by merging.
  * HTF request.security overlays (none present in this source)
  * Detail tooltip text (label only)

All bars are treated as *confirmed* (we only process historical/closed bars).

Note on the live readout: a young live leg is pulled toward neutral until it
has at least ``minimum_mature_bars`` evaluated bars and its impulse/correction
separation reaches ``minimum_confidence`` -- this matches the source's gated
``f_classify`` (which requires ``maturity >= 0.999``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import (
    atr,
    choice,
    ema,
    hma,
    integer,
    number,
    pivothigh_strict,
    pivotlow_strict,
    rma,
    sma,
    wma,
)


def _ma(series: pd.Series, length: int, kind: str) -> pd.Series:
    """Mirror Pine ``f_ma``: SMA / RMA / WMA / HMA / (default) EMA."""
    if kind == "SMA":
        return sma(series, length)
    if kind == "RMA":
        return rma(series, length)
    if kind == "WMA":
        return wma(series, length)
    if kind == "HMA":
        return hma(series, length)
    return ema(series, length)


def _safe_div(num: float, den: float, fallback: float) -> float:
    return num / den if den != 0.0 else fallback


def _clamp(v: float, low: float, high: float) -> float:
    return max(low, min(high, v))


def _map01(v: float, low: float, high: float) -> float:
    if high <= low:
        return 0.5
    return _clamp((v - low) / (high - low), 0.0, 1.0)


def _classify(impulse: float, correction: float, maturity: float,
              impulse_threshold: float, correction_threshold: float,
              minimum_confidence: float) -> int:
    """Mirror Pine ``f_classify``. Returns -1 / 0 / 1."""
    separation = abs(impulse - correction)
    result = 0
    if maturity >= 0.999 and separation >= minimum_confidence:
        if impulse >= impulse_threshold and impulse > correction:
            result = 1
        elif correction >= correction_threshold and correction > impulse:
            result = -1
    return result


def market_motion_dna(
    frame: pd.DataFrame,
    *,
    pivot_left: int = 5,
    pivot_right: int = 5,
    max_leg_bars: int = 350,
    retained_legs: int = 40,
    min_leg_atr: float = 0.50,
    atr_length: int = 14,
    minimum_mature_bars: int = 8,
    retracement_ceiling: float = 0.80,
    momentum_low_ratio: float = 0.60,
    momentum_high_ratio: float = 1.40,
    tr_low_ratio: float = 0.70,
    tr_high_ratio: float = 1.30,
    ma_type: str = "EMA",
    ma_length: int = 34,
    sub_pivot_width: int = 2,
    weight_efficiency: float = 20.0,
    weight_overlap: float = 15.0,
    weight_retracement: float = 12.5,
    weight_commitment: float = 10.0,
    weight_wicks: float = 7.5,
    weight_momentum: float = 10.0,
    weight_true_range: float = 5.0,
    weight_ma_support: float = 7.5,
    weight_cleanliness: float = 7.5,
    weight_acceptance: float = 5.0,
    impulse_threshold: float = 62.0,
    correction_threshold: float = 62.0,
    minimum_confidence: float = 18.0,
    character_change_buffer: float = 5.0,
    mintick: float = 1e-8,
):
    # ---- validate (must run on the *actual* values, not defaults) ----
    pivot_left = integer("pivot_left", pivot_left, 1, 50)
    pivot_right = integer("pivot_right", pivot_right, 1, 50)
    max_leg_bars = integer("max_leg_bars", max_leg_bars, 20, 1500)
    retained_legs = integer("retained_legs", retained_legs, 3, 120)
    min_leg_atr = number("min_leg_atr", min_leg_atr, 0.0, 10.0)
    atr_length = integer("atr_length", atr_length, 2, 200)
    minimum_mature_bars = integer("minimum_mature_bars", minimum_mature_bars, 2, 100)
    retracement_ceiling = number("retracement_ceiling", retracement_ceiling, 0.20, 2.0)
    momentum_low_ratio = number("momentum_low_ratio", momentum_low_ratio, 0.10, 1.0)
    momentum_high_ratio = number("momentum_high_ratio", momentum_high_ratio, 1.0, 4.0)
    tr_low_ratio = number("tr_low_ratio", tr_low_ratio, 0.10, 1.0)
    tr_high_ratio = number("tr_high_ratio", tr_high_ratio, 1.0, 4.0)
    ma_type = choice("ma_type", ma_type, ("EMA", "SMA", "RMA", "WMA", "HMA"))
    ma_length = integer("ma_length", ma_length, 2, 500)
    sub_pivot_width = integer("sub_pivot_width", sub_pivot_width, 1, 20)
    weight_efficiency = number("weight_efficiency", weight_efficiency, 0.0, 100.0)
    weight_overlap = number("weight_overlap", weight_overlap, 0.0, 100.0)
    weight_retracement = number("weight_retracement", weight_retracement, 0.0, 100.0)
    weight_commitment = number("weight_commitment", weight_commitment, 0.0, 100.0)
    weight_wicks = number("weight_wicks", weight_wicks, 0.0, 100.0)
    weight_momentum = number("weight_momentum", weight_momentum, 0.0, 100.0)
    weight_true_range = number("weight_true_range", weight_true_range, 0.0, 100.0)
    weight_ma_support = number("weight_ma_support", weight_ma_support, 0.0, 100.0)
    weight_cleanliness = number("weight_cleanliness", weight_cleanliness, 0.0, 100.0)
    weight_acceptance = number("weight_acceptance", weight_acceptance, 0.0, 100.0)
    impulse_threshold = number("impulse_threshold", impulse_threshold, 50.0, 95.0)
    correction_threshold = number("correction_threshold", correction_threshold, 50.0, 95.0)
    minimum_confidence = number("minimum_confidence", minimum_confidence, 0.0, 80.0)
    character_change_buffer = number("character_change_buffer", character_change_buffer, 0.0, 25.0)

    idx = frame.index
    close = frame["close"].to_numpy(dtype=float)
    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    n = len(close)
    R = pivot_right

    atr_arr = atr(frame, atr_length).to_numpy(dtype=float)
    ma_arr = _ma(frame["close"], ma_length, ma_type).to_numpy(dtype=float)
    sub_high = pivothigh_strict(frame["high"], sub_pivot_width, sub_pivot_width).to_numpy(dtype=float)
    sub_low = pivotlow_strict(frame["low"], sub_pivot_width, sub_pivot_width).to_numpy(dtype=float)
    cand_high = pivothigh_strict(frame["high"], pivot_left, pivot_right).to_numpy(dtype=float)
    cand_low = pivotlow_strict(frame["low"], pivot_left, pivot_right).to_numpy(dtype=float)

    weight_sum = (
        weight_efficiency + weight_overlap + weight_retracement + weight_commitment
        + weight_wicks + weight_momentum + weight_true_range + weight_ma_support
        + weight_cleanliness + weight_acceptance
    )
    safe_weight_sum = max(weight_sum, 0.0001)

    def calc_leg(start_pos: int, start_price: float, end_pos: int, end_price: float,
                 direction: int, is_complete: bool):
        """Port of Pine ``f_calculateLeg``. Returns the 21-tuple of DNA values."""
        raw_bars = max(end_pos - start_pos, 1)
        evaluated = min(raw_bars, max_leg_bars)
        eff_start = end_pos - evaluated
        eff_start_price = start_price if eff_start == start_pos else close[eff_start]
        net_distance = max(abs(end_price - eff_start_price), mintick)

        path_distance = 0.0
        overlap_sum = 0.0
        overlap_count = 0
        commitment_sum = 0.0
        wick_clean_sum = 0.0
        acceptance_sum = 0.0
        ma_support_sum = 0.0
        feature_count = 0
        direction_changes = 0
        previous_delta_sign = 0

        favorable_extreme = eff_start_price
        max_adverse = 0.0
        highest_high = max(eff_start_price, end_price)
        lowest_low = min(eff_start_price, end_price)

        first_half_body = 0.0
        second_half_body = 0.0
        first_half_tr = 0.0
        second_half_tr = 0.0
        first_half_count = 0
        second_half_count = 0

        last_sub_type = 0
        last_sub_price = math.nan
        internal_swing_count = 0
        internal_swing_sum = 0.0
        swing_sizes: list[float] = []

        half_boundary = int(math.ceil(evaluated * 0.5))

        for step in range(1, evaluated + 1):
            abs_pos = eff_start + step
            prev_pos = abs_pos - 1
            cur_close = close[abs_pos]
            prev_close = close[prev_pos]
            cur_open = open_[abs_pos]
            cur_high = high[abs_pos]
            cur_low = low[abs_pos]
            prev_high = high[prev_pos]
            prev_low = low[prev_pos]
            cur_ma = ma_arr[abs_pos]
            prev_ma = ma_arr[prev_pos]

            cur_range = max(cur_high - cur_low, mintick)
            prev_range = max(prev_high - prev_low, mintick)
            delta = cur_close - prev_close
            delta_sign = 1 if delta > 0.0 else (-1 if delta < 0.0 else 0)
            directional_delta = direction * delta
            directional_body = max(direction * (cur_close - cur_open), 0.0)
            cur_tr = max(cur_high - cur_low,
                         max(abs(cur_high - prev_close), abs(cur_low - prev_close)))

            path_distance += cur_tr

            overlap_amount = max(0.0, min(cur_high, prev_high) - max(cur_low, prev_low))
            overlap_sum += _safe_div(overlap_amount, min(cur_range, prev_range), 0.0)
            overlap_count += 1

            close_location = (
                (cur_close - cur_low) / cur_range if direction == 1
                else (cur_high - cur_close) / cur_range
            )
            commitment_sum += _clamp(close_location, 0.0, 1.0)

            total_wick = cur_range - abs(cur_close - cur_open)
            wick_clean_sum += 1.0 - _clamp(total_wick / cur_range, 0.0, 1.0)

            acceptance_sum += (
                1.0 if directional_delta > 0.0
                else (0.0 if directional_delta < 0.0 else 0.5)
            )

            price_supported = (
                cur_close >= cur_ma if direction == 1 else cur_close <= cur_ma
            )
            slope_supported = (
                cur_ma >= prev_ma if direction == 1 else cur_ma <= prev_ma
            )
            ma_support_sum += (0.6 if price_supported else 0.0) + (0.4 if slope_supported else 0.0)

            if delta_sign != 0:
                if previous_delta_sign != 0 and delta_sign != previous_delta_sign:
                    direction_changes += 1
                previous_delta_sign = delta_sign

            if direction == 1:
                favorable_extreme = max(favorable_extreme, cur_high)
                max_adverse = max(max_adverse, favorable_extreme - cur_low)
            else:
                favorable_extreme = min(favorable_extreme, cur_low)
                max_adverse = max(max_adverse, cur_high - favorable_extreme)

            highest_high = max(highest_high, cur_high)
            lowest_low = min(lowest_low, cur_low)

            if step <= half_boundary:
                first_half_body += directional_body
                first_half_tr += cur_tr
                first_half_count += 1
            else:
                second_half_body += directional_body
                second_half_tr += cur_tr
                second_half_count += 1

            # Internal sub-pivot at this bar (confirmation lands sub_pivot_width
            # bars later, i.e. at abs_pos + sub_pivot_width).
            confirm_idx = abs_pos + sub_pivot_width
            sub_chosen_type = 0
            sub_chosen_price = math.nan
            if confirm_idx <= end_pos:
                svh = sub_high[confirm_idx]
                svl = sub_low[confirm_idx]
                is_sub_high = not math.isnan(svh)
                is_sub_low = not math.isnan(svl)
                if is_sub_high and is_sub_low:
                    sub_chosen_type = 1 if last_sub_type <= 0 else -1
                    sub_chosen_price = svh if sub_chosen_type == 1 else svl
                elif is_sub_high:
                    sub_chosen_type = 1
                    sub_chosen_price = svh
                elif is_sub_low:
                    sub_chosen_type = -1
                    sub_chosen_price = svl

                if sub_chosen_type != 0:
                    if last_sub_type == 0:
                        last_sub_type = sub_chosen_type
                        last_sub_price = sub_chosen_price
                        internal_swing_count += 1
                    elif sub_chosen_type == last_sub_type:
                        sub_more_extreme = (
                            sub_chosen_price > last_sub_price
                            if sub_chosen_type == 1
                            else sub_chosen_price < last_sub_price
                        )
                        if sub_more_extreme:
                            last_sub_price = sub_chosen_price
                    else:
                        completed_size = abs(sub_chosen_price - last_sub_price)
                        internal_swing_sum += completed_size
                        swing_sizes.append(completed_size)
                        internal_swing_count += 1
                        last_sub_type = sub_chosen_type
                        last_sub_price = sub_chosen_price

            feature_count += 1

        efficiency = _clamp(_safe_div(net_distance, max(path_distance, net_distance), 0.0), 0.0, 1.0)
        overlap = _clamp(overlap_sum / overlap_count, 0.0, 1.0) if overlap_count > 0 else 0.5
        retr_raw = _safe_div(max_adverse, net_distance, 0.0)
        retracement = _clamp(retr_raw / retracement_ceiling, 0.0, 1.0)
        commitment = _clamp(commitment_sum / feature_count, 0.0, 1.0) if feature_count > 0 else 0.5
        wick_clean = _clamp(wick_clean_sum / feature_count, 0.0, 1.0) if feature_count > 0 else 0.5
        acceptance = _clamp(acceptance_sum / feature_count, 0.0, 1.0) if feature_count > 0 else 0.5
        ma_support = _clamp(ma_support_sum / feature_count, 0.0, 1.0) if feature_count > 0 else 0.5
        cleanliness = (
            1.0 - _clamp(direction_changes / max(evaluated - 1.0, 1.0), 0.0, 1.0)
            if evaluated > 1 else 0.5
        )

        atr_at_end = atr_arr[end_pos] if not math.isnan(atr_arr[end_pos]) else 0.0

        avg_first_body = first_half_body / first_half_count if first_half_count > 0 else 0.0
        avg_second_body = (
            second_half_body / second_half_count if second_half_count > 0 else avg_first_body
        )
        momentum_first_base = max(avg_first_body, atr_at_end * 0.25)
        momentum_ratio = avg_second_body / momentum_first_base if momentum_first_base > mintick else 1.0
        momentum_expansion = _map01(momentum_ratio, momentum_low_ratio, momentum_high_ratio)

        avg_first_tr = first_half_tr / first_half_count if first_half_count > 0 else 0.0
        avg_second_tr = second_half_tr / second_half_count if second_half_count > 0 else avg_first_tr
        tr_ratio = avg_second_tr / avg_first_tr if avg_first_tr > mintick else 1.0
        tr_expansion = _map01(tr_ratio, tr_low_ratio, tr_high_ratio)

        raw_impulse = (
            efficiency * weight_efficiency
            + (1.0 - overlap) * weight_overlap
            + (1.0 - retracement) * weight_retracement
            + commitment * weight_commitment
            + wick_clean * weight_wicks
            + momentum_expansion * weight_momentum
            + tr_expansion * weight_true_range
            + ma_support * weight_ma_support
            + cleanliness * weight_cleanliness
            + acceptance * weight_acceptance
        ) / safe_weight_sum * 100.0

        raw_correction = (
            (1.0 - efficiency) * weight_efficiency
            + overlap * weight_overlap
            + retracement * weight_retracement
            + (1.0 - commitment) * weight_commitment
            + (1.0 - wick_clean) * weight_wicks
            + (1.0 - momentum_expansion) * weight_momentum
            + (1.0 - tr_expansion) * weight_true_range
            + (1.0 - ma_support) * weight_ma_support
            + (1.0 - cleanliness) * weight_cleanliness
            + (1.0 - acceptance) * weight_acceptance
        ) / safe_weight_sum * 100.0

        maturity = 1.0 if is_complete else _clamp(evaluated / float(minimum_mature_bars), 0.0, 1.0)
        impulse_score = 50.0 + (raw_impulse - 50.0) * maturity
        correction_score = 50.0 + (raw_correction - 50.0) * maturity
        confidence = abs(impulse_score - correction_score) * maturity

        completed_swings = len(swing_sizes)
        avg_internal_swing_atr = (
            _safe_div(internal_swing_sum / completed_swings, atr_at_end, 0.0)
            if completed_swings > 0 else 0.0
        )

        progression_ratio = _clamp(_safe_div(net_distance, internal_swing_sum, 1.0), 0.0, 1.0)

        first_swing_sum = 0.0
        second_swing_sum = 0.0
        first_swing_count = 0
        second_swing_count = 0
        swing_half = int(math.ceil(completed_swings * 0.5))
        for k in range(completed_swings):
            sz = swing_sizes[k]
            if k < swing_half:
                first_swing_sum += sz
                first_swing_count += 1
            else:
                second_swing_sum += sz
                second_swing_count += 1
        avg_first_swing = first_swing_sum / first_swing_count if first_swing_count > 0 else 0.0
        avg_second_swing = (
            second_swing_sum / second_swing_count if second_swing_count > 0 else avg_first_swing
        )
        sub_leg_growth_ratio = avg_second_swing / avg_first_swing if avg_first_swing > mintick else 1.0

        return (
            impulse_score, correction_score, confidence, maturity, evaluated,
            efficiency, overlap, retracement, commitment, wick_clean,
            momentum_expansion, tr_expansion, ma_support, cleanliness, acceptance,
            highest_high, lowest_low, internal_swing_count, avg_internal_swing_atr,
            progression_ratio, sub_leg_growth_ratio,
        )

    # ---------------------------------------------------------------- outputs
    impulse_o = np.full(n, 50.0)
    correction_o = np.full(n, 50.0)
    confidence_o = np.full(n, 0.0)
    maturity_o = np.full(n, 0.0)
    bars_o = np.zeros(n, dtype=np.int64)
    efficiency_o = np.full(n, 0.5)
    overlap_o = np.full(n, 0.5)
    retracement_o = np.full(n, 0.5)
    commitment_o = np.full(n, 0.5)
    wick_o = np.full(n, 0.5)
    momentum_o = np.full(n, 0.5)
    tr_o = np.full(n, 0.5)
    ma_support_o = np.full(n, 0.5)
    cleanliness_o = np.full(n, 0.5)
    acceptance_o = np.full(n, 0.5)
    class_o = np.zeros(n, dtype=np.int64)
    internal_swings_o = np.zeros(n, dtype=np.int64)
    avg_swing_atr_o = np.zeros(n)
    progression_o = np.full(n, 1.0)
    growth_o = np.full(n, 1.0)

    pivot_bars: list[int] = []
    pivot_prices: list[float] = []
    pivot_types: list[int] = []
    live_class_state = 0
    prev_live_class_state = 0

    for i in range(n):
        has_high = not math.isnan(cand_high[i])
        has_low = not math.isnan(cand_low[i])
        chosen_type = 0
        chosen_price = math.nan

        if has_high and has_low:
            if len(pivot_types) == 0:
                dl = abs(cand_high[i] - close[i - R])
                dr = abs(close[i - R] - cand_low[i])
                chosen_type = 1 if dl >= dr else -1
            else:
                chosen_type = -pivot_types[-1]
            chosen_price = cand_high[i] if chosen_type == 1 else cand_low[i]
        elif has_high:
            chosen_type = 1
            chosen_price = cand_high[i]
        elif has_low:
            chosen_type = -1
            chosen_price = cand_low[i]

        if chosen_type != 0 and not math.isnan(chosen_price):
            cb = i - R
            pc = len(pivot_bars)
            if pc == 0:
                pivot_bars.append(cb)
                pivot_prices.append(chosen_price)
                pivot_types.append(chosen_type)
            else:
                last_type = pivot_types[-1]
                last_price = pivot_prices[-1]
                if chosen_type == last_type:
                    is_more = (
                        chosen_price > last_price
                        if chosen_type == 1
                        else chosen_price < last_price
                    )
                    if is_more:
                        pivot_bars[-1] = cb
                        pivot_prices[-1] = chosen_price
                        pivot_types[-1] = chosen_type
                else:
                    atr_at_pivot = (
                        atr_arr[i - R] if not math.isnan(atr_arr[i - R]) else 0.0
                    )
                    large_enough = (
                        min_leg_atr <= 0.0
                        or abs(chosen_price - last_price) >= atr_at_pivot * min_leg_atr
                    )
                    if large_enough and cb > pivot_bars[-1]:
                        pivot_bars.append(cb)
                        pivot_prices.append(chosen_price)
                        pivot_types.append(chosen_type)
                        while len(pivot_bars) > retained_legs + 2:
                            pivot_bars.pop(0)
                            pivot_prices.pop(0)
                            pivot_types.pop(0)

        if len(pivot_bars) > 0:
            s_b = pivot_bars[-1]
            s_p = pivot_prices[-1]
            s_t = pivot_types[-1]
            live_dir = 1 if s_t == -1 else -1
            live_end = high[i] if live_dir == 1 else low[i]

            res = calc_leg(s_b, s_p, i, live_end, live_dir, False)
            (
                impulse_o[i], correction_o[i], confidence_o[i], maturity_o[i], bars_o[i],
                efficiency_o[i], overlap_o[i], retracement_o[i], commitment_o[i], wick_o[i],
                momentum_o[i], tr_o[i], ma_support_o[i], cleanliness_o[i], acceptance_o[i],
                _hh, _ll, internal_swings_o[i], avg_swing_atr_o[i],
                progression_o[i], growth_o[i],
            ) = res

            raw_class = _classify(
                impulse_o[i], correction_o[i], maturity_o[i],
                impulse_threshold, correction_threshold, minimum_confidence,
            )
            prev_live_class_state = live_class_state
            if live_class_state == 1:
                keep = (
                    impulse_o[i] >= impulse_threshold - character_change_buffer
                    and impulse_o[i] >= correction_o[i]
                )
                live_class_state = 1 if keep else raw_class
            elif live_class_state == -1:
                keep = (
                    correction_o[i] >= correction_threshold - character_change_buffer
                    and correction_o[i] >= impulse_o[i]
                )
                live_class_state = -1 if keep else raw_class
            else:
                live_class_state = raw_class
            class_o[i] = live_class_state
            _ = prev_live_class_state

    out = pd.DataFrame(index=idx)
    out["impulse_score"] = impulse_o
    out["correction_score"] = correction_o
    out["confidence"] = confidence_o
    out["maturity"] = maturity_o
    out["leg_bars"] = bars_o
    out["efficiency"] = efficiency_o
    out["overlap"] = overlap_o
    out["retracement_pressure"] = retracement_o
    out["close_commitment"] = commitment_o
    out["wick_clean"] = wick_o
    out["momentum_expansion"] = momentum_o
    out["tr_expansion"] = tr_o
    out["ma_support"] = ma_support_o
    out["direction_cleanliness"] = cleanliness_o
    out["acceptance"] = acceptance_o
    out["live_class"] = pd.Series(class_o, index=idx).astype("Int64")
    out["internal_swing_count"] = pd.Series(internal_swings_o, index=idx).astype("Int64")
    out["avg_internal_swing_atr"] = avg_swing_atr_o
    out["progression_ratio"] = progression_o
    out["sub_leg_growth_ratio"] = growth_o
    return out

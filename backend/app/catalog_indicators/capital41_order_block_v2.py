"""Capital41 Order Block Detector v2 — 冲动 K 线后回溯寻找订单块，并跟踪其缓解与存活。

Source: pine-crypto-collection/capital41_btc_general/Capital41_Order_Block/
Capital41_Order_Block_v2.pine

MIT License — Copyright (c) 2026 Capital41

逻辑：ATR 过滤的冲动 K 线（实体 > ATR × 倍数）出现时，从当前根往回扫最多 `ob_lookback` 根，取第一根
反向 K 线作为订单块——看涨冲动取最后一根阴线（默认全幅 high/low，或 `use_body` 时取 open/close），
看跌冲动取最后一根阳线；与同方向既有区重叠则跳过。已登记的区按「年龄 ≥ 最小缓解年龄」判断缓解：
看涨区 low ≤ 上沿、看跌区 high ≥ 下沿即缓解；可开启缓解即删除，超过 `freshness_length × 3` 根的区
直接淘汰，数量超过 `max_active` 时淘汰最旧的。每根输出冲动/新建/缓解事件、计数，以及同方向最新一个
仍在跟踪的区的上下沿与年龄（根）。

上游的 `barstate.isconfirmed` 在历史回放里按「每根都已收盘」处理，最后一根同样参与判定。

Dropped: 订单块方框与其透明度/配色（含按新鲜度淡化的展示逻辑）、OB 标签、右下角计数徽章、
`alertcondition`（`showLabels` / `showBadge` 等纯展示开关未进签名）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._pine import atr, boolean, flags, integer, nullable_int, number, ohlcv

_DIRECTIONS = (1, -1)
_BOOLEAN_COLUMNS = ("bull_ob_created", "bear_ob_created", "bull_mitigated_now",
                    "bear_mitigated_now")
_INTEGER_COLUMNS = ("new_bull_ob_offset", "new_bear_ob_offset", "latest_bull_ob_age",
                    "latest_bear_ob_age", "bull_active_zones", "bear_active_zones",
                    "bull_mitigated_zones", "bear_mitigated_zones", "total_zones")


def _has_nearby(zones: list[list], top: float, bottom: float, direction: int) -> bool:
    """Pine's `f_hasNearby`: an existing zone of the same direction overlaps the candidate."""
    return any(top >= zone[1] and bottom <= zone[0]
               for zone in zones if zone[2] == direction)


def _latest(zones: list[list], direction: int):
    for zone in reversed(zones):
        if zone[2] == direction:
            return zone
    return None


def capital41_order_block_v2(frame: pd.DataFrame, *, atr_length: int = 14,
                             impulse_mult: float = 1.5, ob_lookback: int = 10,
                             max_active: int = 25, freshness_length: int = 80,
                             min_mitigation_age: int = 2, use_body: bool = False,
                             remove_mitigated: bool = False) -> pd.DataFrame:
    integer("atr_length", atr_length, 1)
    number("impulse_mult", impulse_mult, 0.5, 5.0)
    integer("ob_lookback", ob_lookback, 1, 30)
    integer("max_active", max_active, 5, 100)
    integer("freshness_length", freshness_length, 10, 500)
    integer("min_mitigation_age", min_mitigation_age, 1, 20)
    boolean("use_body", use_body)
    boolean("remove_mitigated", remove_mitigated)

    opening, high, low, close = ohlcv(frame, "open", "high", "low", "close")
    average_range = atr(frame, atr_length)
    highs, lows = high.to_numpy(dtype=float), low.to_numpy(dtype=float)
    opens, closes = opening.to_numpy(dtype=float), close.to_numpy(dtype=float)

    bullish_impulse = flags((close > opening) & ((close - opening) > average_range * impulse_mult))
    bearish_impulse = flags((close < opening) & ((opening - close) > average_range * impulse_mult))
    # [top, bottom, direction, born, mitigated]
    zones: list[list] = []
    columns: dict[str, list] = {name: [] for name in (
        "bull_ob_created", "bear_ob_created", "new_bull_ob_top", "new_bull_ob_bottom",
        "new_bull_ob_offset", "new_bear_ob_top", "new_bear_ob_bottom", "new_bear_ob_offset",
        "bull_mitigated_now", "bear_mitigated_now", "latest_bull_ob_top", "latest_bull_ob_bottom",
        "latest_bull_ob_age", "latest_bear_ob_top", "latest_bear_ob_bottom", "latest_bear_ob_age",
        "bull_active_zones", "bear_active_zones", "bull_mitigated_zones", "bear_mitigated_zones",
        "total_zones")}

    for bar in range(len(frame)):
        mitigated_bull = mitigated_bear = False
        for index in range(len(zones) - 1, -1, -1):
            zone = zones[index]
            age = bar - zone[3]
            if not zone[4] and age >= min_mitigation_age:
                if zone[2] == 1 and lows[bar] <= zone[0]:
                    zone[4] = True
                    mitigated_bull = True
                if zone[2] == -1 and highs[bar] >= zone[1]:
                    zone[4] = True
                    mitigated_bear = True
            if zone[4] and remove_mitigated:
                del zones[index]
                continue
            if age > freshness_length * 3:
                del zones[index]

        created = {1: None, -1: None}
        for direction in _DIRECTIONS:
            if direction == 1 and not bullish_impulse.iloc[bar]:
                continue
            if direction == -1 and not bearish_impulse.iloc[bar]:
                continue
            for offset in range(1, ob_lookback + 1):
                position = bar - offset
                if position < 0:
                    break  # Pine reads na here, so no candle can qualify
                rising = closes[position] > opens[position]
                if (direction == 1 and rising) or (direction == -1 and not rising):
                    continue
                if use_body:
                    top = opens[position] if direction == 1 else closes[position]
                    bottom = closes[position] if direction == 1 else opens[position]
                else:
                    top, bottom = highs[position], lows[position]
                if not _has_nearby(zones, top, bottom, direction):
                    zones.append([top, bottom, direction, bar, False])
                    created[direction] = (top, bottom, offset)
                break  # Pine stops at the first opposite candle, created or not

        while len(zones) > max_active:
            del zones[0]

        columns["bull_ob_created"].append(created[1] is not None)
        columns["bear_ob_created"].append(created[-1] is not None)
        for direction, prefix in ((1, "bull"), (-1, "bear")):
            zone = created[direction]
            columns[f"new_{prefix}_ob_top"].append(zone[0] if zone else np.nan)
            columns[f"new_{prefix}_ob_bottom"].append(zone[1] if zone else np.nan)
            columns[f"new_{prefix}_ob_offset"].append(zone[2] if zone else np.nan)
        columns["bull_mitigated_now"].append(mitigated_bull)
        columns["bear_mitigated_now"].append(mitigated_bear)
        for direction, prefix in ((1, "bull"), (-1, "bear")):
            zone = _latest(zones, direction)
            columns[f"latest_{prefix}_ob_top"].append(zone[0] if zone else np.nan)
            columns[f"latest_{prefix}_ob_bottom"].append(zone[1] if zone else np.nan)
            columns[f"latest_{prefix}_ob_age"].append(bar - zone[3] if zone else np.nan)
        columns["bull_active_zones"].append(
            sum(1 for zone in zones if zone[2] == 1 and not zone[4]))
        columns["bear_active_zones"].append(
            sum(1 for zone in zones if zone[2] == -1 and not zone[4]))
        columns["bull_mitigated_zones"].append(
            sum(1 for zone in zones if zone[2] == 1 and zone[4]))
        columns["bear_mitigated_zones"].append(
            sum(1 for zone in zones if zone[2] == -1 and zone[4]))
        columns["total_zones"].append(len(zones))

    result = pd.DataFrame({
        "bull_impulse": bullish_impulse,
        "bear_impulse": bearish_impulse,
        **{name: pd.Series(values, index=frame.index) for name, values in columns.items()},
    }, index=frame.index)
    for name in _BOOLEAN_COLUMNS:
        result[name] = result[name].astype(bool)
    for name in _INTEGER_COLUMNS:
        result[name] = nullable_int(result[name])
    return result

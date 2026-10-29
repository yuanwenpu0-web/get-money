"""交易时段判定：TradingView `time(timeframe.period, session, tz)` 的本地等价物。

判定规则（与 TradingView 一致）
------------------------------
- 一根 K 线属于某个会话 ⟺ **它的开盘时间**（换算到会话时区后的墙上时间）落在
  `[start, end)` 区间内；右端是开区间。
- 脚本 `sessFirst="0930-0935"` 的自检说明里写着「必须恰好命中一根K线」——正是因为
  右端开区间：5 分钟图上只有 09:30 那根命中，若当成闭区间，09:30 与 09:35 两根都会命中。
- 跨零点的会话（`1700-0900`）按环绕处理，没有「先跳过周末/节假日」的过滤：脚本的
  session 字符串没写星期前缀，加密货币 24 小时交易，天天都是交易日。

策略参数只接受数值/布尔，所以时区用编号选择（`session.tz_choices()` 暴露候选），
并保留 `fallback_offset_minutes` 作为 tzdata 缺失时的固定偏移兜底。
"""

from __future__ import annotations

from datetime import timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

MINUTES_IN_DAY = 1440

# 编号 → IANA 时区名。业务相关的就这几个，需要更多时在这里加一行即可。
TZ_CHOICES: dict[int, str] = {
    0: "UTC",
    1: "America/New_York",
    2: "Europe/London",
    3: "Asia/Shanghai",
    4: "Asia/Tokyo",
}

DEFAULT_TZ_ID = 1
# America/New_York 的标准时偏移（EST，UTC-5）。tzdata 不可用时用它，夏令时会差 1 小时。
DEFAULT_TZ_FALLBACK_MINUTES = -300


def tz_choices() -> dict[int, str]:
    return dict(TZ_CHOICES)


def timezone_of(tz_id: int, fallback_offset_minutes: int = DEFAULT_TZ_FALLBACK_MINUTES):
    """把时区分辨率编号解析成 tzinfo；tzdata 缺失时退化为固定偏移。"""
    name = TZ_CHOICES.get(int(tz_id))
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError, KeyError):  # pragma: no cover - 环境差异
            pass
    offset = int(fallback_offset_minutes)
    offset = max(-720, min(840, offset))
    return timezone(timedelta(minutes=offset))


def _as_utc(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize("UTC") if idx.tz is None else idx


def local_index(
    index,
    tz_id: int = DEFAULT_TZ_ID,
    fallback_offset_minutes: int = DEFAULT_TZ_FALLBACK_MINUTES,
) -> pd.DatetimeIndex:
    """把（通常是 UTC 的）K线索引换算成会话时区的墙上时间索引。"""
    idx = _as_utc(index)
    return idx.tz_convert(timezone_of(tz_id, fallback_offset_minutes))


def local_minutes(
    index,
    tz_id: int = DEFAULT_TZ_ID,
    fallback_offset_minutes: int = DEFAULT_TZ_FALLBACK_MINUTES,
) -> pd.Series:
    """当地「日內分钟数」0..1439。"""
    idx = _as_utc(index)
    local = local_index(idx, tz_id, fallback_offset_minutes)
    minutes = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    return pd.Series(minutes.astype(int), index=index)


def local_day(
    index,
    tz_id: int = DEFAULT_TZ_ID,
    fallback_offset_minutes: int = DEFAULT_TZ_FALLBACK_MINUTES,
) -> pd.Series:
    """当地「日」（dayofmonth）。脚本用 `dayofmonth(time, tz)` 判定换日。"""
    idx = _as_utc(index)
    local = local_index(idx, tz_id, fallback_offset_minutes)
    return pd.Series(local.day.to_numpy().astype(int), index=index)


def session_flags(
    index,
    start_min: int,
    end_min: int,
    tz_id: int = DEFAULT_TZ_ID,
    fallback_offset_minutes: int = DEFAULT_TZ_FALLBACK_MINUTES,
) -> pd.Series:
    """会话命中掩码：`start <= 开盘时间 < end`（跨零点时环绕）。

    `start == end` 表示空会话（任一 K 线都不命中），与 TradingView 里空 session 串行为一致。
    """
    minutes = local_minutes(index, tz_id, fallback_offset_minutes)
    start = int(start_min) % MINUTES_IN_DAY
    end = int(end_min) % MINUTES_IN_DAY
    if start < end:
        mask = minutes.ge(start) & minutes.lt(end)
    elif start > end:
        mask = minutes.ge(start) | minutes.lt(end)
    else:
        mask = pd.Series(False, index=minutes.index)
    return mask


__all__ = [
    "DEFAULT_TZ_FALLBACK_MINUTES",
    "DEFAULT_TZ_ID",
    "MINUTES_IN_DAY",
    "TZ_CHOICES",
    "local_day",
    "local_index",
    "local_minutes",
    "session_flags",
    "timezone_of",
    "tz_choices",
]

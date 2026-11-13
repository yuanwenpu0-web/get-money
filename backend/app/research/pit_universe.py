"""PIT（point-in-time）币种排序与选币 —— 纯函数，不碰 DB、不联网。

为什么独立成模块
----------------
「缩小币种池」能不能赚钱，取决于成员是**怎么来的**。用今天知道会涨的币去赌
昨天，读数必然虚高：`docs/binding-combo-optimization.json` 的 `pit` 段实测同一
策略同一窗口，事后选币 +7.1003%/笔（positive）vs 时点重建 −1.0136%/笔
（inconclusive），**8.1pp 落差纯由泄漏造成**。

所以本模块的唯一硬不变量是因果性：`rank_at` 只使用 `[end_ms - window, end_ms)`
内的 bar，任何 end_ms 及之后的 bar 一律丢弃。调用方即使多取了未来数据，这里也
不会让它影响成员构成。

与既有实现的分工
----------------
- `scripts/probe_p16_fixed_pool.py::eligible_ranking` 只做成交额排序，且沿用
  `pool_maintenance.EXCLUDED`（含 BTC/ETH/稳定币对）。
- 本模块额外给出波动率排序，且**排除集可配置**：supertrend 的 `all` 基线臂与
  线上 `pool_code='all'` 都含 BTC/ETH，若强制排除就同时改了两个变量。
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

HOUR = 3_600_000
WINDOW_HOURS = 168          # 与 pool_maintenance.WINDOW_HOURS 一致
MIN_BARS = 160              # 与 pit_pool_check.rebuild 一致
STALE_HOURS = 2             # 末根早于 end-2h 视为停更
LIQUIDITY_SHORTLIST = 100   # 波动率排序只在成交额 Top100 内做
LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")


def symbol_allowed(symbol: str, excluded: Iterable[str]) -> bool:
    """USDT 后缀、非杠杆代币、不在排除集内。排除集由调用方决定。"""
    text = str(symbol)
    return (text.endswith("USDT") and text not in set(excluded)
            and not text.endswith(LEVERAGED_SUFFIXES))


def rank_at(
    rows: Iterable[tuple],
    end_ms: int,
    *,
    excluded: Iterable[str] = frozenset(),
    min_bars: int = MIN_BARS,
    window_hours: int = WINDOW_HOURS,
) -> list[dict]:
    """按事前窗口给出合格币的成交额与波动率，成交额降序、并列按字母序。

    `rows` 是 `(symbol, open_time, close, quote_volume)` 序列（1h bar）。
    零成交额单根仍计为有效 bar，合格条件用窗口**总和** > 0。
    """
    start_ms = end_ms - window_hours * HOUR
    blocked = set(excluded)
    grouped: dict[str, list[tuple]] = {}
    for symbol, open_time, close, volume in rows:
        # 因果闸门：窗口外的 bar 一律不参与，未来 bar 尤其不能进
        if not (start_ms <= open_time < end_ms):
            continue
        if not symbol_allowed(symbol, blocked):
            continue
        grouped.setdefault(str(symbol), []).append((open_time, close, volume))

    out: list[dict] = []
    for symbol, bars in grouped.items():
        if len(bars) < min_bars:
            continue
        bars.sort(key=lambda item: item[0])
        if bars[-1][0] + HOUR < end_ms - STALE_HOURS * HOUR:
            continue
        closes = np.asarray([item[1] for item in bars], dtype=float)
        volumes = np.asarray([item[2] for item in bars], dtype=float)
        if not np.isfinite(closes).all() or (closes <= 0).any():
            continue
        if not np.isfinite(volumes).all():
            continue
        total = float(volumes.sum())
        if total <= 0:
            continue
        volatility = (float(np.std(np.diff(np.log(closes)), ddof=1))
                      if len(closes) > 1 else 0.0)
        out.append({"symbol": symbol, "quote_volume": total, "volatility": volatility})
    out.sort(key=lambda item: (-item["quote_volume"], item["symbol"]))
    return out


def select(ranked: Sequence[dict], kind: str, n: int) -> list[str]:
    """从排序结果取 TopN。不足 N 个时返回全部可用成员（缺口由调用方记录）。"""
    if kind == "liquidity":
        pool = list(ranked)
    elif kind == "volatility":
        pool = sorted(ranked[:LIQUIDITY_SHORTLIST],
                      key=lambda item: (-item["volatility"], item["symbol"]))
    else:
        raise ValueError(f"未知选币口径: {kind}")
    return [item["symbol"] for item in pool[:max(0, int(n))]]
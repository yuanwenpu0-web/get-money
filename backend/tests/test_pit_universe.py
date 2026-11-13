"""P17 / H-E PIT 选币量具单测（不联网、不连 DB）。

核心不变量是**因果性**：排序只能用 end_ms 之前的 bar。任何让未来 bar 影响
成员构成的实现都必须在这里挂掉，否则整个选币实验退化成事后选币。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.research.pit_universe import (  # noqa: E402
    HOUR, WINDOW_HOURS, rank_at, select, symbol_allowed,
)

END = 1_800_000_000_000  # 对齐到小时的固定端点


def _bars(symbol: str, count: int, *, end: int = END, volume: float = 10.0,
          closes=None, start: int | None = None) -> list[tuple]:
    """生成 count 根小时 bar，末根 open_time = end - HOUR（满足新鲜度）。"""
    last = end - HOUR
    first = last - (count - 1) * HOUR
    if start is not None:
        first = start
    out = []
    for i in range(count):
        close = 100.0 if closes is None else float(closes[i % len(closes)])
        out.append((symbol, first + i * HOUR, close, volume))
    return out


def test_symbol_allowed_keeps_majors_when_excluded_set_is_blacklist_only():
    # all 基线臂与线上 pool_code='all' 都含 BTC/ETH，故排除集必须可配置
    assert symbol_allowed("BTCUSDT", frozenset({"BANKUSDT"})) is True
    assert symbol_allowed("SOLUSDT", frozenset({"BANKUSDT"})) is True
    assert symbol_allowed("BANKUSDT", frozenset({"BANKUSDT"})) is False
    assert symbol_allowed("BTCBUSD", frozenset()) is False
    for suffix in ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT"):
        assert symbol_allowed("BTC" + suffix, frozenset()) is False


def test_rank_ignores_bars_at_or_after_end_ms():
    # 未来 bar 即使成交额巨大也不能进排序：这是事后选币的唯一入口
    rows = _bars("AAAUSDT", 168, volume=1.0)
    rows += [("AAAUSDT", END + i * HOUR, 100.0, 1e12) for i in range(168)]
    rows += _bars("BBBUSDT", 168, volume=2.0)
    out = rank_at(rows, END, excluded=frozenset())
    assert [item["symbol"] for item in out] == ["BBBUSDT", "AAAUSDT"]
    assert out[1]["quote_volume"] == 168.0


def test_rank_ignores_bars_before_window_start():
    # 窗口是 [end-168h, end)，更早的 bar 不计入成交额也不计入根数
    rows = _bars("AAAUSDT", 168, volume=1.0)
    rows += [("AAAUSDT", END - (WINDOW_HOURS + i) * HOUR, 100.0, 1e9) for i in range(1, 40)]
    out = rank_at(rows, END, excluded=frozenset())
    assert out[0]["quote_volume"] == 168.0


def test_rank_requires_min_bars():
    rows = _bars("FULLUSDT", 168) + _bars("THINUSDT", 159)
    out = rank_at(rows, END, excluded=frozenset())
    assert [item["symbol"] for item in out] == ["FULLUSDT"]


def test_rank_requires_fresh_last_bar():
    # 末根早于 end-2h 视为停更，不可交易
    rows = _bars("FRESHUSDT", 168)
    stale = _bars("STALEUSDT", 168)
    stale = [(s, t - 5 * HOUR, c, v) for s, t, c, v in stale]
    out = rank_at(rows + stale, END, excluded=frozenset())
    assert [item["symbol"] for item in out] == ["FRESHUSDT"]


def test_rank_requires_positive_total_volume():
    rows = _bars("ZEROUSDT", 168, volume=0.0) + _bars("OKUSDT", 168, volume=1.0)
    out = rank_at(rows, END, excluded=frozenset())
    assert [item["symbol"] for item in out] == ["OKUSDT"]


def test_rank_rejects_non_finite_and_non_positive_close():
    rows = _bars("BADUSDT", 168, closes=[float("nan")]) + _bars("NEGUSDT", 168, closes=[-1.0])
    rows += _bars("GOODUSDT", 168)
    out = rank_at(rows, END, excluded=frozenset())
    assert [item["symbol"] for item in out] == ["GOODUSDT"]


def test_rank_is_deterministic_on_volume_ties():
    rows = _bars("ZZZUSDT", 168, volume=5.0) + _bars("AAAUSDT", 168, volume=5.0)
    first = rank_at(rows, END, excluded=frozenset())
    second = rank_at(list(reversed(rows)), END, excluded=frozenset())
    assert [i["symbol"] for i in first] == ["AAAUSDT", "ZZZUSDT"]
    assert first == second


def test_volatility_is_std_of_log_returns():
    # 等成交额下，对数收益恒定的币波动率为 0，交替涨跌的币 > 0
    flat = [100.0 * (2 ** i) for i in range(168)]
    choppy = [100.0 * (1.5 if i % 2 else 1.0) for i in range(168)]
    rows = _bars("FLATUSDT", 168, volume=5.0, closes=flat)
    rows += _bars("CHOPUSDT", 168, volume=5.0, closes=choppy)
    out = {i["symbol"]: i for i in rank_at(rows, END, excluded=frozenset())}
    expected = float(np.std(np.diff(np.log(np.array(choppy))), ddof=1))
    assert abs(out["CHOPUSDT"]["volatility"] - expected) < 1e-12
    # 恒定对数收益（纯趋势）的波动率应趋近 0；用容差而非 ==0 避免浮点脆性
    assert out["FLATUSDT"]["volatility"] < 1e-9 < out["CHOPUSDT"]["volatility"]


def test_select_liquidity_takes_top_n_by_volume():
    rows = []
    for name, vol in (("LOWUSDT", 1.0), ("HIGHUSDT", 9.0), ("MIDUSDT", 5.0)):
        rows += _bars(name, 168, volume=vol)
    ranked = rank_at(rows, END, excluded=frozenset())
    assert select(ranked, "liquidity", 2) == ["HIGHUSDT", "MIDUSDT"]
    assert select(ranked, "liquidity", 99) == ["HIGHUSDT", "MIDUSDT", "LOWUSDT"]


def test_select_volatility_only_within_top100_liquidity():
    # 成交额第 101 名即使波动率最高也不能入选：与线上研究池规则一致
    rows = []
    for i in range(100):
        rows += _bars(f"L{i:03d}USDT", 168, volume=1000.0 - i, closes=[100.0, 100.0])
    rows += _bars("WILDUSDT", 168, volume=1.0, closes=[100.0, 300.0])
    ranked = rank_at(rows, END, excluded=frozenset())
    picked = select(ranked, "volatility", 5)
    assert "WILDUSDT" not in picked
    assert len(picked) == 5
    assert all(s.startswith("L") for s in picked)


def test_select_rejects_unknown_kind():
    ranked = rank_at(_bars("AAAUSDT", 168), END, excluded=frozenset())
    try:
        select(ranked, "momentum", 5)
    except ValueError:
        return
    raise AssertionError("未知选币口径必须报错，不能静默回退到成交额排序")
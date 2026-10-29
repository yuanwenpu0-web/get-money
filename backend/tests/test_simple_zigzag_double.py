import pandas as pd
import pytest

from app.backtest.zigzag_execution import simulate_zigzag
from app.strategy.registry import PrepareContext
from app.strategy.rules import evaluate_signals, resolve_params
from app.strategy.specs import simple_zigzag_double as zz


def _bars(closes):
    index = pd.date_range("2024-01-01", periods=len(closes), freq="15min", tz="UTC")
    return pd.DataFrame({"close": closes, "open": closes,
                         "high": [v + 0.5 for v in closes],
                         "low": [v - 0.5 for v in closes]}, index=index)


def test_triple_ema_pivots_and_persistent_levels():
    bars = _bars([10, 9, 8, 9, 10, 11, 10, 9, 8, 7, 8, 9, 10, 11, 10])
    long, short = zz._levels(bars, 2, 2)
    assert long.iloc[:6].isna().all()
    assert long.iloc[6:].tolist() == [11.5] * 9
    assert short.iloc[:10].isna().all()
    assert short.iloc[10:].tolist() == [6.5] * 5


def test_strict_virtual_state_is_updated_before_order_conditions(monkeypatch):
    bars = _bars([100, 110, 90, 100, 120])
    monkeypatch.setattr(zz, "_levels", lambda frame, length, detection: (
        pd.Series(105.0, index=frame.index), pd.Series(95.0, index=frame.index)))
    params = resolve_params(zz.CODE, {}, {"strict_position": True})
    frame = zz.prepare(bars, params, PrepareContext())
    assert frame.zz_virtual_pos.tolist() == [0, 1, -1, -1, 1]
    assert frame.zz_long_entry.tolist() == [True, False, True, True, False]
    assert frame.zz_short_entry.tolist() == [True, True, False, False, True]


def test_window_boundaries_and_stop_order_columns(monkeypatch):
    bars = _bars([100, 100, 100, 100])
    monkeypatch.setattr(zz, "_levels", lambda frame, length, detection: (
        pd.Series(105.0, index=frame.index), pd.Series(95.0, index=frame.index)))
    start, end = [int(x.timestamp() * 1000) for x in (bars.index[0], bars.index[2])]
    params = resolve_params(zz.CODE, {}, {"timestamp_start": start, "timestamp_end": end})
    result = evaluate_signals(bars, zz.CODE, params)
    assert result.zz_long_entry.tolist() == [False, True, False, False]
    assert result.zz_short_entry.tolist() == [False, True, False, False]
    # Ordinary market-order dispatch cannot execute the strategy's stop orders.
    assert result.direction.eq(0).all()


def test_defaults_and_parameter_validation():
    params = resolve_params(zz.CODE, {})
    assert params["lot_long"] == params["lot_short"] == 100.0
    assert params["use_long"] and params["use_short"]
    assert not params["strict_position"] and not params["fixed_lot"]
    with pytest.raises(ValueError):
        resolve_params(zz.CODE, {}, {"long_length": 1})


def test_virtual_position_tracks_breakouts_without_strict_position(monkeypatch):
    # In the Pine source the two direction flips are top-level statements, so
    # virtual_pos tracks breaches even with strict_position=false (the default).
    bars = _bars([100, 110, 100, 90, 100])
    monkeypatch.setattr(zz, "_levels", lambda frame, length, detection: (
        pd.Series(105.0, index=frame.index), pd.Series(95.0, index=frame.index)))
    params = resolve_params(zz.CODE, {}, {})
    frame = zz.prepare(bars, params, PrepareContext())
    assert frame.zz_virtual_pos.tolist() == [0, 1, 1, -1, -1]
    # Once virtually long, the long entry condition is suppressed until price
    # breaches the short level and virtual_pos flips to -1.
    assert frame.zz_long_entry.tolist() == [True, False, False, True, True]
    assert frame.zz_short_entry.tolist() == [True, True, True, False, False]


def test_strict_position_does_not_change_final_virtual_state(monkeypatch):
    # Pine resets virtual_pos inside the strict_position guard, but the two
    # unconditional flips immediately after overwrite that reset. The switch
    # has no effect on the final state in this script; keep it as a 1:1 mapping.
    bars = _bars([100, 110, 90, 100, 120])
    monkeypatch.setattr(zz, "_levels", lambda frame, length, detection: (
        pd.Series(105.0, index=frame.index), pd.Series(95.0, index=frame.index)))
    loose = zz.prepare(bars, resolve_params(zz.CODE, {}, {}), PrepareContext())
    strict = zz.prepare(bars, resolve_params(zz.CODE, {}, {"strict_position": True}), PrepareContext())
    for col in ("zz_virtual_pos", "zz_long_entry", "zz_short_entry"):
        assert loose[col].tolist() == strict[col].tolist()
    assert loose.zz_virtual_pos.tolist() == [0, 1, -1, -1, 1]


def test_breached_level_suppresses_long_stop_order_end_to_end(monkeypatch):
    # The first bar already closes above LEVEL_LONG: Pine flips virtual_pos to
    # 1, so long_entry_condition is false and no Long stop order is ever placed.
    # A virtual_pos stuck at 0 would submit a marketable Long that fills at the
    # next open, which is not what the script does.
    bars = _bars([110, 100, 100, 115])
    monkeypatch.setattr(zz, "_levels", lambda frame, length, detection: (
        pd.Series([105.0, 105.0, 105.0, 120.0], index=frame.index),
        pd.Series(95.0, index=frame.index)))
    params = resolve_params(zz.CODE, {}, {})
    data = zz.prepare(bars, params, PrepareContext())
    assert data.zz_virtual_pos.tolist() == [1, 1, 1, 1]
    assert data.zz_long_entry.tolist() == [False, False, False, False]
    equity, trades = simulate_zigzag(bars, zz.signals(data, params), params, 10000, 0.001, 0, "15m")
    assert trades == []
    assert equity.attrs["open_position"] is None


def _ohlc(rows, freq="15min"):
    index = pd.date_range("2024-01-01", periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)


def test_stop_touch_on_next_bar_becomes_directional_signal(monkeypatch):
    # 挂单时序与回测撮合一致：T-1 收盘按当时通道价提交，T 根内触价成交。
    index_bars = _ohlc([[100.0, 100.5, 99.5, 100.0],
                        [100.0, 100.5, 99.5, 100.0],
                        [100.0, 110.5, 99.5, 110.0]])
    monkeypatch.setattr(zz, "_levels", lambda frame, length, detection: (
        pd.Series(105.0, index=index_bars.index), pd.Series(95.0, index=index_bars.index)))
    params = resolve_params(zz.CODE, {}, {})
    result = evaluate_signals(index_bars, zz.CODE, params)
    # bar0/bar1 双向挂单（virtual_pos=0）；bar2 触及 105 → 多头成交信号
    assert result.direction.tolist() == [0, 0, 1]
    assert result.block_reason.iloc[-1] == ""
    assert result.sl_pct.iloc[-1] == pytest.approx(1 - 95 / 110)
    assert result.tp_pct.eq(0).all()
    assert result.leverage.iloc[-1] == 1
    assert result.position_pct.iloc[-1] == pytest.approx(1.0)
    assert result.score.iloc[-1] == 0.6


def test_double_touch_same_bar_resolves_by_open_proximity(monkeypatch):
    index_bars = _ohlc([[100.0, 100.5, 99.5, 100.0],
                        [101.0, 120.0, 80.0, 100.0]])
    monkeypatch.setattr(zz, "_levels", lambda frame, length, detection: (
        pd.Series(105.0, index=index_bars.index), pd.Series(95.0, index=index_bars.index)))
    result = evaluate_signals(index_bars, zz.CODE, resolve_params(zz.CODE, {}, {}))
    # 同一根K线双向都触价：开盘离 105 更近 → 先触多头档（与回测路径假设同源）
    assert result.direction.tolist() == [0, 1]


def test_invalid_reverse_level_blocks_direction(monkeypatch):
    index_bars = _ohlc([[100.0, 100.5, 99.5, 100.0],
                        [100.0, 100.5, 99.5, 100.0],
                        [100.0, 110.5, 99.5, 110.0]])
    monkeypatch.setattr(zz, "_levels", lambda frame, length, detection: (
        pd.Series(105.0, index=index_bars.index), pd.Series(112.0, index=index_bars.index)))
    result = evaluate_signals(index_bars, zz.CODE, resolve_params(zz.CODE, {}, {}))
    # 反向通道在收盘价上方 → 止损落在价格错误一侧，方向清零并给出原因
    assert result.direction.tolist() == [0, 0, 0]
    assert result.block_reason.iloc[-1] == "no_reverse_level"


def test_flat_frames_still_carry_levels_for_stop_following(monkeypatch):
    bars = _bars([100.0])
    monkeypatch.setattr(zz, "_levels", lambda frame, length, detection: (
        pd.Series(105.0, index=bars.index), pd.Series(95.0, index=bars.index)))
    params = resolve_params(zz.CODE, {}, {})
    result = evaluate_signals(bars, zz.CODE, params)
    assert result.direction.tolist() == [0]
    assert result.block_reason.iloc[0] == "awaiting_stop_fill"
    # FLAT 信号也要带通道价位：持仓止损跟随依赖 reason.zz_levels
    data = zz.prepare(bars, params, PrepareContext())
    info = zz.describe(data.iloc[-1])
    assert info["zz_levels"] == {"long": 105.0, "short": 95.0}
    assert info["zz_virtual_pos"] == 0

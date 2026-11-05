"""订单流净买入策略专项测试：全部离线，不访问网络。

覆盖四层：
1. 特征层：净主动买入量口径（2×taker_buy−volume）、缺失主动买入量不得当作 0、
   滚动分位阈值严格因果（shift(1)，不含本根）。
2. 参数层：分位窗口必须大于累计窗口、入场分位必须高于出场分位、默认不启用。
3. 信号层：只做多、分位下方全部拦截、止损止盈按 ATR 倍数、评分分层。
4. 数据层：1m→15m 聚合的主动买入量按分钟求和，任一分钟缺失则该桶为 NULL。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.strategy.registry import get_spec
from app.strategy.rules import evaluate_signals, resolve_params

CODE = "flow_imbalance_long"
MINUTE = 60_000
START = int(pd.Timestamp("2026-01-01T00:00:00Z").timestamp() * 1000)


def _params(**overrides) -> dict:
    return resolve_params(CODE, {"fee_rate": 0.0004, "slippage": 0.0002}, overrides or None)


def _frame(bars: int = 1200, buy_ratio: float = 0.6, missing_from: int | None = None,
          vary: bool = True) -> pd.DataFrame:
    """构造确定性行情。vary=True 时主动买入占比周期波动，才能形成有意义的分位阈值。"""
    rng = np.random.default_rng(7)
    close = 100 * np.cumprod(1 + rng.normal(0.0002, 0.004, bars))
    high = close * 1.003
    low = close * 0.997
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = np.full(bars, 1000.0)
    ratio = np.full(bars, buy_ratio)
    if vary:
        ratio = np.clip(buy_ratio + 0.25 * np.sin(np.arange(bars) / 7.0), 0.02, 0.98)
    taker = volume * ratio
    if missing_from is not None:
        taker[missing_from:] = np.nan
    index = pd.DatetimeIndex(
        pd.to_datetime(START + np.arange(bars) * 15 * MINUTE, unit="ms", utc=True), name="time"
    )
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "quote_volume": volume * close, "trades": 100,
        "taker_buy_volume": taker, "taker_buy_quote_volume": taker * close,
    }, index=index)


def test_net_flow_ratio_matches_definition():
    """fi_ratio 必须等于 sum(2×主动买 − 总量) ÷ sum(总量)。"""
    frame = _frame(800, buy_ratio=0.6, vary=False)
    params = _params(flow_window=48, quantile_window=240)
    data = get_spec(CODE).prepare(frame.copy(), params, None)
    tail = data.iloc[-1]
    manual = (2 * 0.6 - 1.0) / 1.0
    assert tail["fi_ratio"] == pytest.approx(manual, abs=1e-9)
    # 主动买占比 0.6 → 净买入占比 0.2
    assert tail["fi_ratio"] == pytest.approx(0.2, abs=1e-9)


def test_missing_taker_volume_is_not_treated_as_zero():
    """主动买入量缺失的 bar 必须让 fi_ratio 变 NaN，而不是把它当成 0 买入。"""
    frame = _frame(800, buy_ratio=0.6, missing_from=700)
    params = _params(flow_window=8, quantile_window=240)
    data = get_spec(CODE).prepare(frame.copy(), params, None)
    assert data["fi_ratio"].iloc[700:].isna().all()
    # 缺失段之前的正常数据不受影响
    assert np.isfinite(data["fi_ratio"].iloc[600])


def test_quantile_threshold_is_causal():
    """分位阈值只能由 shift(1) 之后的历史决定：改动最后一根不得改变其阈值。"""
    frame = _frame(1000, buy_ratio=0.62)
    params = _params(flow_window=16, quantile_window=240)
    full = get_spec(CODE).prepare(frame.copy(), params, None)
    # 把最后一根改成极端主动买入，倒数第二根的阈值不应变化
    altered = frame.copy()
    altered.loc[altered.index[-1], "taker_buy_volume"] = altered["volume"].iloc[-1] * 0.999
    changed = get_spec(CODE).prepare(altered, params, None)
    assert full["fi_threshold"].iloc[-2] == pytest.approx(changed["fi_threshold"].iloc[-2])


def test_signal_is_long_only():
    """只做多：direction 不得出现 -1。"""
    frame = _frame(1200, buy_ratio=0.75)
    params = _params()
    signals = evaluate_signals(frame, CODE, params)
    assert set(signals.direction.unique()) <= {0, 1}


def test_below_threshold_is_blocked():
    """主动买占比 0.5（净流入为 0）时不应产出方向。"""
    frame = _frame(1200, buy_ratio=0.5, vary=False)
    params = _params()
    signals = evaluate_signals(frame, CODE, params)
    assert signals.direction.eq(0).all()
    assert (signals.block_reason != "").all()


def test_stop_and_target_follow_atr_multiples():
    """止损/止盈距离必须等于 ATR × 配置倍数。"""
    frame = _frame(1200, buy_ratio=0.8)
    params = _params(flow_window=16, quantile_window=240, sl_atr_mult=2.0, target_r=3.0)
    data = get_spec(CODE).prepare(frame.copy(), params, None)
    signals = evaluate_signals(frame, CODE, params)
    expected_sl = data["atr_pct"] * 2.0
    expected_tp = expected_sl * 3.0
    mask = signals.direction.ne(0)
    assert mask.any()
    assert np.allclose(signals.loc[mask, "sl_pct"], expected_sl[mask], equal_nan=False)
    assert np.allclose(signals.loc[mask, "tp_pct"], expected_tp[mask], equal_nan=False)


def test_validate_rejects_bad_windows():
    """分位窗口必须大于累计窗口，否则分位由同一批样本决定。"""
    with pytest.raises(ValueError, match="分位窗口必须大于"):
        _params(flow_window=480, quantile_window=240)


def test_validate_rejects_inverted_quantiles():
    with pytest.raises(ValueError, match="入场分位必须高于出场分位"):
        _params(entry_quantile=0.5, exit_quantile=0.6)


def test_spec_defaults_disabled_and_grid_declared():
    spec = get_spec(CODE)
    assert spec.default_enabled is False
    assert "flow_window" in spec.grid and "max_hold_bars" in spec.grid
    assert spec.tags == ("order_flow", "long_only")


def test_aggregation_sums_taker_volume_and_nulls_incomplete_buckets(db):
    """1m→15m 聚合：主动买入量按分钟求和；缺一分钟则整桶为 NULL。"""
    from app.market.data_service import MarketDataService

    service = MarketDataService(db)
    rows = [
        {"open_time": START + i * MINUTE, "close_time": START + (i + 1) * MINUTE - 1,
         "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
         "volume": 10.0, "quote_volume": 1000.0, "trades": 5,
         "taker_buy_volume": 6.0, "taker_buy_quote_volume": 600.0}
        for i in range(15)
    ]
    service.store("BTCUSDT", "1m", rows)
    frame = service.load_frame("BTCUSDT", "15m")
    assert len(frame) == 1
    assert frame["taker_buy_volume"].iloc[0] == pytest.approx(90.0)
    assert frame["volume"].iloc[0] == pytest.approx(150.0)

    # 再写入第二桶，但故意让其中一分钟缺主动买入量
    rows2 = [
        {"open_time": START + (15 + i) * MINUTE, "close_time": START + (16 + i) * MINUTE - 1,
         "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
         "volume": 10.0, "quote_volume": 1000.0, "trades": 5,
         "taker_buy_volume": None if i == 3 else 6.0,
         "taker_buy_quote_volume": None if i == 3 else 600.0}
        for i in range(15)
    ]
    service.store("BTCUSDT", "1m", rows2)
    frame = service.load_frame("BTCUSDT", "15m")
    assert len(frame) == 2
    assert frame["taker_buy_volume"].isna().iloc[1]

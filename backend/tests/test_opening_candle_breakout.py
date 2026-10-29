"""Opening Candle Breakout [OCB] 本地适配专项测试：全部离线，不访问网络。

分五层校验，重点是「入场/出场严格按脚本」而不是收益：

1. **会话层** `features.session`：时区换算（含夏令时）、右端开区间、跨零点会话、空会话。
2. **判定层** `specs.opening_candle_breakout`：Wilder ATR 播种、实体门槛与方向、
   风险距离的双向夹逼、止损/目标价位锁定在信号收盘、`use_target` 关闭。
3. **信号层**：统一信号帧方向恒为 0（不许偷偷走通用撮合），拦截原因按顺序先到先得。
4. **撮合层** `backtest.ocb_execution`：次根开盘成交、同根即可被打止、10R 止盈、
   **会话强平成交在强平窗之后的那一根开盘**、每天一笔、风险预算定手数、现金护栏。
5. **因果性**：截断到最后一根重算，重叠区间必须与全帧完全一致。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.backtest import runner
from app.backtest.ocb_execution import simulate_ocb
from app.features.session import session_flags, tz_choices
from app.strategy.defaults import DEFAULT_STRATEGIES
from app.strategy.registry import get_spec
from app.strategy.rules import evaluate_signals, known_codes, prepare_frame, resolve_params
from app.strategy.specs import opening_candle_breakout as ocb

CODE = "opening_candle_breakout"
ZERO_COST = {"fee_rate": 0.0, "slippage": 0.0}
FIRST_START, FIRST_END = 570, 575   # 09:30-09:35 America/New_York
FLAT_START, FLAT_END = 955, 960     # 15:55-16:00 America/New_York


def _params(state: dict | None = None, **overrides) -> dict:
    return resolve_params(CODE, dict(state or {"fee_rate": 0.0004, "slippage": 0.0}), overrides or None)


def _index(days: int = 5, freq: str = "5min") -> pd.DatetimeIndex:
    per_day = {"5min": 288, "15min": 96, "1h": 24}[freq]
    return pd.date_range("2026-01-01", periods=days * per_day, freq=freq, tz="UTC")


def _walk(index: pd.DatetimeIndex, base: float = 200.0, vol: float = 0.0015, seed: int = 7):
    rng = np.random.default_rng(seed)
    return base * (1 + rng.normal(0.0, vol, len(index)).cumsum())


def _frame(index: pd.DatetimeIndex, closes, wick: float = 0.0006) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    open_ = np.concatenate([[closes[0]], closes[:-1]])
    high = np.maximum(open_, closes) * (1 + wick)
    low = np.minimum(open_, closes) * (1 - wick)
    return pd.DataFrame(
        dict(open=open_, high=high, low=low, close=closes, volume=1000.0), index=index
    )


def _patch(frame: pd.DataFrame, stamp, **levels) -> pd.DataFrame:
    """覆盖单根 OHLC；未给的高/低自动夹逼到合法范围。"""
    data = frame.copy()
    open_ = float(levels.get("open", data.at[stamp, "open"]))
    close = float(levels.get("close", data.at[stamp, "close"]))
    high = max(float(levels["high"]), open_, close) if "high" in levels else max(open_, close)
    low = min(float(levels["low"]), open_, close) if "low" in levels else min(open_, close)
    data.loc[stamp, ["open", "high", "low", "close"]] = [open_, high, low, close]
    return data


def _first_candles(index: pd.DatetimeIndex, params: dict | None = None) -> list:
    params = params or _params()
    flags = session_flags(
        index,
        int(params["first_session_start_min"]),
        int(params["first_session_end_min"]),
        int(params["session_tz"]),
        int(params["tz_offset_minutes"]),
    )
    return list(index[flags.to_numpy()])


def _scenario(*, days: int = 1, day: int = 0, direction: int = 1, body_atr: float = 1.0,
              risk_atr: float = 0.3, atr_len: int = 14, **overrides) -> SimpleNamespace:
    """造一个「只有一根 bar 有 setup」的受控帧。

    其它开盘首根被糊成十字星（`close == open` → 脚本天然不发信号）；目标那根按 ATR 为单位
    指定实体与影线，使原始风险距离落在 `[min_stop_atr, max_stop_atr] × ATR` 之内不被夹逼。
    """
    index = _index(days)
    base = _frame(index, _walk(index))
    frame = base
    for stamp in _first_candles(index):
        frame = _patch(frame, stamp, close=float(base.at[stamp, "open"]))
    stamp = _first_candles(index)[day]
    k = int(index.get_loc(stamp))
    atr0 = float(ocb.wilder_atr(base, atr_len).iloc[k])
    open_price = float(base.at[stamp, "open"])
    if direction >= 0:
        frame = _patch(frame, stamp, close=open_price + body_atr * atr0, low=open_price - risk_atr * atr0)
    else:
        frame = _patch(frame, stamp, close=open_price - body_atr * atr0, high=open_price + risk_atr * atr0)
    params = _params(**overrides)
    data = prepare_frame(frame, CODE, params)
    return SimpleNamespace(
        index=index, frame=frame, data=data, signals=get_spec(CODE).signals(data, params),
        params=params, k=k, stamp=stamp, atr0=atr0,
    )


def _bars(frame: pd.DataFrame, index: pd.DatetimeIndex, k: int, rows):
    """依次覆盖第 k+1、k+2 …… 根的 open/close（完全受控：不给 Chain 逻辑留随机开口）。"""
    for offset, levels in enumerate(rows, start=1):
        frame = _patch(frame, index[k + offset], **levels)
    return frame


def _flat(frame: pd.DataFrame, index: pd.DatetimeIndex, k: int, level: float):
    """把第 k 根之后到末尾全部压平成同一价位（既不打止盈也不打止损）。"""
    return _bars(frame, index, k, [dict(open=level, close=level)] * (len(index) - k - 1))


def _touch(frame: pd.DataFrame, index: pd.DatetimeIndex, k: int, level: float, signal_close: float):
    """入场那一根原地不动（`open=close=signal_close`），下一根才走向 `level`。"""
    return _bars(frame, index, k, [
        dict(open=signal_close, close=signal_close),
        dict(open=signal_close, close=level),
    ])


def _run(frame: pd.DataFrame, params: dict, initial: float = 10_000.0, fee: float = 0.0, slippage: float = 0.0):
    signals = evaluate_signals(frame, CODE, params)
    equity, trades = simulate_ocb(frame, signals, params, initial, fee, slippage, "5m")
    return signals, equity, trades


# ------------------------------------------------------------------ 会话层

def test_session_flags_follow_dst():
    # 1 月是 EST(UTC-5)：09:30 对应 14:30 UTC；7 月切到 EDT(UTC-4)：09:30 对应 13:30 UTC
    winter = pd.DatetimeIndex(["2026-01-05 14:30", "2026-01-05 13:30"], tz="UTC")
    summer = pd.DatetimeIndex(["2026-07-06 13:30", "2026-07-06 14:30"], tz="UTC")
    assert session_flags(winter, FIRST_START, FIRST_END, 1, -300).tolist() == [True, False]
    assert session_flags(summer, FIRST_START, FIRST_END, 1, -300).tolist() == [True, False]


def test_session_end_is_exclusive():
    index = pd.DatetimeIndex(["2026-01-05 14:30", "2026-01-05 14:35"], tz="UTC")
    assert session_flags(index, FIRST_START, FIRST_END, 1, -300).tolist() == [True, False]


def test_session_wraps_around_midnight_and_empty_window_matches_nothing():
    index = pd.DatetimeIndex(["2026-01-05 23:00", "2026-01-05 02:00", "2026-01-05 12:00"], tz="UTC")
    assert session_flags(index, 22 * 60, 3 * 60, 0, 0).tolist() == [True, True, False]
    assert session_flags(index, 9 * 60, 9 * 60, 0, 0).tolist() == [False, False, False]


def test_tz_choices_exposes_new_york():
    assert tz_choices()[1] == "America/New_York"


# ------------------------------------------------------------------ 判定层

def test_defaults_match_the_pine_script():
    params = _params()
    assert (params["first_session_start_min"], params["first_session_end_min"]) == (570, 575)
    assert (params["flat_session_start_min"], params["flat_session_end_min"]) == (955, 960)
    assert params["session_tz"] == 1 and params["min_body_atr"] == 0.5
    assert params["atr_len"] == 14 and params["min_stop_atr"] == 0.1
    assert params["max_stop_atr"] == 2.0 and params["target_r"] == 10.0
    assert params["use_target"] is True and params["trade_longs"] is True


def test_wilder_atr_seeds_with_sma_like_pine():
    index = _index(days=1)
    frame = _frame(index, _walk(index))
    atr = ocb.wilder_atr(frame, 14)
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [frame["high"] - frame["low"],
         (frame["high"] - prev_close).abs(),
         (frame["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    assert np.isnan(atr.iloc[:13]).all()
    assert atr.iloc[13] == pytest.approx(true_range.iloc[:14].mean())
    manual = (1 / 14) * true_range.iloc[14] + (13 / 14) * atr.iloc[13]
    assert atr.iloc[14] == pytest.approx(manual)


def test_body_and_direction_gating():
    bull = _scenario(direction=1)
    assert int(bull.data["ocb_dir"].iloc[bull.k]) == 1
    assert bull.signals["ocb_dir"].ne(0).sum() == 1

    bear = _scenario(direction=-1)
    assert int(bear.data["ocb_dir"].iloc[bear.k]) == -1

    index = _index()
    base = _frame(index, _walk(index))
    doji = _patch(base, bull.stamp, close=float(base.at[bull.stamp, "open"]))
    prepared = prepare_frame(doji, CODE, _params(min_body_atr=0.0))
    assert int(prepared["ocb_dir"].iloc[bull.k]) == 0  # 严格 close > open：十字星没有方向

    tall = prepare_frame(_frame(index, _walk(index)), CODE, _params(min_body_atr=5.0))
    assert int(tall["ocb_dir"].sum()) == 0  # 实体门槛拦住所有自然 bar


def test_risk_is_clamped_from_both_sides():
    inside = _scenario(risk_atr=0.3)
    atr = float(inside.data["ocb_atr"].iloc[inside.k])
    assert 0.1 * atr < inside.data["ocb_risk"].iloc[inside.k] < 2.0 * atr

    wide = _scenario(risk_atr=50.0, min_body_atr=0.0)  # 长下影 → 顶到上夹逼
    assert wide.data["ocb_risk"].iloc[wide.k] == pytest.approx(float(wide.data["ocb_atr"].iloc[wide.k]) * 2.0)

    tight = _scenario(body_atr=0.01, risk_atr=0.0, min_body_atr=0.0)  # 极小的实体与影线 → 顶到下夹逼
    assert tight.data["ocb_risk"].iloc[tight.k] == pytest.approx(float(tight.data["ocb_atr"].iloc[tight.k]) * 0.1)


def test_levels_are_locked_on_the_signal_close():
    long_case = _scenario()
    close = float(long_case.data["close"].iloc[long_case.k])
    risk = float(long_case.data["ocb_risk"].iloc[long_case.k])
    assert long_case.data["ocb_stop"].iloc[long_case.k] == pytest.approx(close - risk)
    assert long_case.data["ocb_target"].iloc[long_case.k] == pytest.approx(close + risk * long_case.params["target_r"])

    short_case = _scenario(direction=-1)
    close = float(short_case.data["close"].iloc[short_case.k])
    risk = float(short_case.data["ocb_risk"].iloc[short_case.k])
    assert short_case.data["ocb_stop"].iloc[short_case.k] == pytest.approx(close + risk)
    assert short_case.data["ocb_target"].iloc[short_case.k] == pytest.approx(close - risk * short_case.params["target_r"])


def test_use_target_off_leaves_no_limit_price():
    index = _index()
    off = prepare_frame(_frame(index, _walk(index)), CODE, _params(use_target=False, min_body_atr=0.0))
    assert np.isnan(off["ocb_target"]).all()


def test_direction_switches_and_reason_order():
    no_long = prepare_frame(_frame(_index(), _walk(_index())), CODE, _params(min_body_atr=0.0, trade_longs=False))
    no_short = prepare_frame(_frame(_index(), _walk(_index())), CODE, _params(min_body_atr=0.0, trade_shorts=False))
    assert int(no_long["ocb_dir"].gt(0).sum()) == 0
    assert int(no_short["ocb_dir"].lt(0).sum()) == 0

    case = _scenario(direction=1, min_body_atr=0.0, trade_longs=False)
    row = case.signals.iloc[case.k]
    assert row["block_reason"] == "long_disabled" and row["direction"] == 0


# ------------------------------------------------------------------ 信号层

def test_generic_frame_never_directions_so_the_shared_pipeline_cannot_trade_it():
    index = _index()
    signals = evaluate_signals(_frame(index, _walk(index)), CODE, _params())
    assert int(signals["direction"].abs().sum()) == 0
    assert int(signals["position_pct"].sum()) == 0
    assert set(signals["block_reason"].unique()) <= {
        "invalid_indicators", "not_opening_candle", "body_too_small", "requires_session_broker"
    }
    setups = signals["ocb_dir"].ne(0) & np.isfinite(signals["ocb_atr"])
    assert (signals.loc[setups, "block_reason"] == "requires_session_broker").all()


def test_reason_priority_follows_the_script():
    signals = evaluate_signals(_frame(_index(), _walk(_index())), CODE, _params())
    counts = signals["block_reason"].value_counts()
    assert counts["not_opening_candle"] > counts.get("body_too_small", 0)


def test_unknown_param_and_cost_override_are_rejected():
    with pytest.raises(ValueError, match="未知策略参数"):
        resolve_params(CODE, {}, {"not_a_knob": 1})
    with pytest.raises(ValueError, match="撮合成本"):
        resolve_params(CODE, {}, {"fee_rate": 0.0})


def test_registered_but_disabled_by_default():
    assert CODE in known_codes()
    row = next(item for item in DEFAULT_STRATEGIES if item["code"] == CODE)
    assert row["enabled"] == 0
    assert get_spec(CODE).default_enabled is False


# ------------------------------------------------------------------ 撮合层

def test_entry_fills_at_the_next_bar_open():
    case = _scenario()
    _, _, trades = _run(case.frame, case.params)
    assert len(trades) == 1
    trade = trades[0]
    assert trade["entry_time"] == str(case.index[case.k + 1])
    assert trade["entry_price"] == pytest.approx(float(case.frame["open"].iloc[case.k + 1]))
    assert trade["side"] == "long"


def test_stop_exit_is_filled_on_the_same_bar_it_is_crossed():
    case = _scenario()
    stop = float(case.data["ocb_stop"].iloc[case.k])
    signal_close = float(case.data["close"].iloc[case.k])
    frame = _touch(case.frame, case.index, case.k, stop * 0.99, signal_close)
    _, _, trades = _run(frame, case.params)
    assert [t["status"] for t in trades] == ["stop_loss"]
    assert trades[0]["exit_time"] == str(case.index[case.k + 2])
    assert trades[0]["exit_price"] == pytest.approx(stop)


def test_target_exit_uses_the_ten_r_limit():
    case = _scenario()
    target = float(case.data["ocb_target"].iloc[case.k])
    signal_close = float(case.data["close"].iloc[case.k])
    frame = _touch(case.frame, case.index, case.k, target * 1.01, signal_close)
    _, _, trades = _run(frame, case.params)
    assert [t["status"] for t in trades] == ["take_profit"]
    assert trades[0]["exit_price"] == pytest.approx(target)


def test_stop_wins_when_the_entry_bar_gaps_through_it():
    """跳空穿过止损时按开盘价成交（Pine 的市价单在次根开盘成交，同一根立刻被判定打止）。"""
    case = _scenario()
    stop = float(case.data["ocb_stop"].iloc[case.k])
    frame = _bars(case.frame, case.index, case.k, [dict(open=stop * 0.99, close=stop * 0.99)])
    _, _, trades = _run(frame, case.params)
    assert [t["status"] for t in trades] == ["stop_loss"]
    assert trades[0]["exit_time"] == str(case.index[case.k + 1])
    assert trades[0]["exit_price"] == pytest.approx(stop * 0.99)


def test_session_close_fills_on_the_bar_after_the_flat_window():
    case = _scenario()
    entry = float(case.data["close"].iloc[case.k])  # 入场那一根会被设成原地不动 → 成交价就是它
    frame = _flat(case.frame, case.index, case.k, entry)
    flat_stamp = next(
        stamp for stamp in case.index
        if bool(session_flags(pd.DatetimeIndex([stamp]), FLAT_START, FLAT_END, 1, -300).iloc[0])
        and stamp > case.stamp
    )
    _, equity, trades = _run(frame, case.params)
    assert [t["status"] for t in trades] == ["session_close"]
    # 强平窗那根收盘提交 close_all → 成交在它之后那一根的开盘
    expected = case.index[int(case.index.get_loc(flat_stamp)) + 1]
    assert trades[0]["exit_time"] == str(expected)
    assert trades[0]["exit_price"] == pytest.approx(entry)
    assert equity.attrs["open_position"] is None


def test_short_side_is_mirrored():
    case = _scenario(direction=-1)
    stop = float(case.data["ocb_stop"].iloc[case.k])
    signal_close = float(case.data["close"].iloc[case.k])
    frame = _touch(case.frame, case.index, case.k, stop * 1.01, signal_close)
    _, _, trades = _run(frame, case.params)
    assert [t["status"] for t in trades] == ["stop_loss"]
    assert trades[0]["side"] == "short"
    assert trades[0]["exit_price"] == pytest.approx(stop)


def test_only_one_trade_per_day_even_with_many_opening_setups():
    index = _index(days=4)
    frame = _frame(index, _walk(index))
    params = _params(min_body_atr=0.0, first_session_end_min=660)  # 09:30-11:00：一天 18 根「开盘首根」
    signals = evaluate_signals(frame, CODE, params)
    setups = int(signals["ocb_dir"].ne(0).sum())
    _, trades = simulate_ocb(frame, signals, params, 10_000.0, 0.0, 0.0, "5m")
    days_covered = len(_first_candles(index, params)) // 18
    assert setups > days_covered  # 每个交易日确实不止一个 setup
    assert len(trades) <= days_covered


def test_position_size_is_the_risk_budget_over_the_risk_distance():
    case = _scenario(risk_atr=0.3)
    params = _params(ZERO_COST, risk_mode=0, risk_usd=100.0, max_position_pct=1.0)
    signal_close = float(case.data["close"].iloc[case.k])
    stop = float(case.data["ocb_stop"].iloc[case.k])
    # 让成交价 = 信号收盘价（脚本 R 的分母），下一根直接打穿止损
    frame = _touch(case.frame, case.index, case.k, stop * 0.99, signal_close)
    signals = evaluate_signals(frame, CODE, params)
    _, trades = simulate_ocb(frame, signals, params, 1_000_000.0, 0.0, 0.0, "5m")
    assert len(trades) == 1
    assert trades[0]["pnl"] == pytest.approx(-100.0, abs=1e-6)


def test_max_position_pct_caps_the_notional():
    case = _scenario()
    params = _params(ZERO_COST, risk_mode=0, risk_usd=100_000.0, max_position_pct=0.05,
                     base_position_pct=0.01)
    signal_close = float(case.data["close"].iloc[case.k])
    frame = _flat(case.frame, case.index, case.k, signal_close)
    signals = evaluate_signals(frame, CODE, params)
    _, trades = simulate_ocb(frame, signals, params, 10_000.0, 0.0, 0.0, "5m")
    assert trades[0]["margin"] == pytest.approx(500.0, abs=1.0)


def test_allow_min_position_is_the_local_analogue_of_allow_min_lot():
    case = _scenario()
    tiny = _params(ZERO_COST, risk_mode=0, risk_usd=0.02, max_position_pct=1.0)
    _, _, trades = _run(case.frame, tiny, initial=10_000.0)
    assert trades == []  # 不足最小名义 → 不下单（脚本里 q<1 且 allowMin=false）
    allowed = _params(ZERO_COST, risk_mode=0, risk_usd=0.02, max_position_pct=1.0, allow_min_position=True)
    _, _, trades = _run(case.frame, allowed, initial=10_000.0)
    assert len(trades) == 1 and trades[0]["margin"] == pytest.approx(10.0, abs=0.5)


def test_equity_stays_positive_and_fully_covers_the_window():
    case = _scenario()
    frame = _flat(case.frame, case.index, case.k, float(case.frame["open"].iloc[case.k + 1]))
    _, equity, trades = _run(frame, case.params, initial=1_000.0)
    assert len(equity) == len(frame)
    assert equity.min() > 0
    assert len(trades) == 1


# ------------------------------------------------------------------ 因果性

def test_prepare_is_causal():
    index = _index(days=6)
    frame = _frame(index, _walk(index, seed=21))
    params = _params()
    full = prepare_frame(frame, CODE, params)
    truncated = prepare_frame(frame.iloc[:-37], CODE, params)
    for column in ocb.OCB_COLUMNS:
        left, right = full[column].iloc[:-37], truncated[column]
        if pd.api.types.is_float_dtype(left):
            assert np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True)
        else:
            assert left.reset_index(drop=True).equals(right.reset_index(drop=True))


# ------------------------------------------------------------------ 回测路由

@pytest.fixture
def ocb_market(monkeypatch):
    index = pd.date_range("2026-01-01", periods=101, freq="5min", tz="UTC")
    close = 100 + 5 * np.sin(np.arange(len(index)) / 3)
    frame = pd.DataFrame(dict(open=close, high=close + 1, low=close - 1,
                              close=close, volume=100.0), index=index)
    calls = []
    as_of = int(index[-1].timestamp() * 1000)

    class StubMarket:
        def __init__(self, db):
            pass

        def load_frame(self, symbol, interval, start_ms, end_ms):
            calls.append((symbol, interval, start_ms, end_ms))
            return frame.copy()

    monkeypatch.setattr(runner, "MarketDataService", StubMarket)
    monkeypatch.setattr(runner, "now_ms", lambda: as_of)
    monkeypatch.setattr(runner, "build_dataset", lambda *a, **kw: pytest.fail("generic dataset path used"))
    return index, calls, as_of


def test_runner_routes_to_the_session_broker(db, ocb_market):
    index, calls, as_of = ocb_market
    result = runner.run_backtest(db, "BTCUSDT", "5m", CODE, days=2)
    metrics = result["metrics"]
    assert calls == [("BTCUSDT", "5m", as_of - 2 * 86_400_000, as_of)]
    assert metrics["execution"] == "pine_session_orders"
    assert metrics["engine"] == "pine_session"
    assert metrics["initial_cash"] == 10_000.0
    assert metrics["fee_rate"] == 0.0004 and metrics["slippage"] == 0.0002
    assert metrics["strategy"] == CODE and metrics["validation"]["enabled"] is False
    # 该窗口里没有任何 09:30 bar → 明确告诉用户零成交是这个原因，而不是没有信号逻辑
    assert metrics["opening_bars"] == 0 and metrics["signal_bars"] == 0
    assert any("开盘首窗" in text for text in metrics["warnings"])
    assert len(result["equity_curve"]) == 100
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="不支持通用验证集选参"):
        runner.run_backtest(db, "BTCUSDT", "5m", CODE, optimize=True)

"""Martin Luk 强势回调（本地适配）专项测试：全部离线，不访问网络。

覆盖四层：
1. 特征层 `features.pullback`：4h 强势段锚定 AVWAP、首次回调状态机、转强确认、EMA9 失守，
   以及**反前视**（截断重算一致）。
2. 参数层：只做多 1 倍杠杆的默认值、48 小时按周期换算、范围校验、策略播种与接口暴露。
3. 信号层：入场门控（4h 趋势 / 首次回调 / 转强确认）、10R 目标、名义仓位 35% 上限。
4. 撮合层：共享回测撮合的 EMA9 软出场，以及持仓管理里的同类回放。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from app.backtest.execution import simulate
from app.features.indicators import build_features
from app.features.pullback import PULLBACK_COLUMNS, add_pullback_context
from app.models import Order, StrategyConfig
from app.schemas import BacktestRequest
from app.strategy.defaults import DEFAULT_STRATEGIES, seed_strategies
from app.strategy.rules import evaluate_signals, resolve_params
from app.trading import order_manager
from pydantic import ValidationError

CODE = "strong_pullback"


def _params(interval: str = "15m", **overrides) -> dict:
    return resolve_params(CODE, {}, overrides or None, interval=interval)


def _pullback_frame(
    periods: int = 1200, dip_at: int = 1050, dip_bars: int = 4, depth: float = 0.005, rec: int = 6
) -> pd.DataFrame:
    """构造 15m 帧：单边上行 + 一次 V 形回调（先跌 dip_bars 根，再 rec 根收回）。"""
    base = np.linspace(100.0, 140.0, periods)
    close = base.copy()
    for k in range(dip_bars):
        close[dip_at + k] = base[dip_at] * (1 - depth * (k + 1) / dip_bars)
    start = dip_at + dip_bars
    for k in range(rec):
        close[start + k] = close[start - 1] + (base[start + k] - close[start - 1]) * (k + 1) / rec
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    idx = pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC")
    return pd.DataFrame(
        dict(open=open_, high=high, low=low, close=close, volume=np.full(periods, 100.0)),
        index=idx,
    )


@pytest.fixture(scope="module")
def enriched() -> pd.DataFrame:
    frame = build_features(_pullback_frame(), with_label=False)
    return add_pullback_context(frame, _params())


@pytest.fixture(scope="module")
def confirm_row(enriched) -> pd.DataFrame:
    """已富化的「转强确认」单根帧：方向/仓位/10R 断言都基于它。"""
    return enriched.loc[[enriched.index[enriched.pb_confirm.eq(1)][0]]]


def _row(confirm_row: pd.DataFrame, **overrides) -> pd.DataFrame:
    row = confirm_row.copy()
    for key, value in overrides.items():
        row[key] = value
    return row


# ---------------------------------------------------------------- 参数与注册


def test_defaults_are_long_only_one_x_with_risk_cap():
    params = _params()
    assert params["max_leverage"] == 1
    assert params["max_position_pct"] == 0.35
    assert params["risk_per_trade_pct"] == 0.005
    assert params["target_r"] == 10.0
    assert params["max_hold_hours"] == 48.0
    assert params["ema_exit"] is True


def test_partial_tp_defaults_and_validation():
    """partial_tp_r 默认 0.0（禁用分段落袋）；A/B 验证 2R 减半无 edge，回退 fixed。
    partial_tp_frac 保留默认值。其他策略亦默认禁用（0），范围校验生效。"""
    params = _params()
    assert params["partial_tp_r"] == 0.0
    assert params["partial_tp_frac"] == 0.5
    assert resolve_params("trend_follow", {}, interval="15m")["partial_tp_r"] == 0.0
    with pytest.raises(ValueError):
        _params(partial_tp_r=60.0)
    with pytest.raises(ValueError):
        _params(partial_tp_frac=1.0)


@pytest.mark.parametrize(
    "interval, bars", [("15m", 192), ("1h", 48), ("4h", 12), ("1d", 2)]
)
def test_hold_hours_converted_to_bars_per_interval(interval, bars):
    """48 小时硬约束在不同周期上都表示同样的墙钟时长。"""
    assert _params(interval)["max_hold_bars"] == bars


def test_resolve_rejects_out_of_range_and_unknown():
    with pytest.raises(ValueError):
        resolve_params(CODE, {}, {"target_r": 0.5})
    with pytest.raises(ValueError):
        resolve_params(CODE, {}, {"nonexistent": 1})
    with pytest.raises(ValueError):
        resolve_params(CODE, {}, {"ema_exit": 1.0})  # 布尔键不接受数字


def test_seed_strategies_is_idempotent(db):
    seed_strategies(db)
    seed_strategies(db)
    rows = db.query(StrategyConfig).filter(StrategyConfig.code == CODE).all()
    assert len(rows) == 1
    assert "Martin Luk" in rows[0].name


def test_strategy_listing_exposes_effective_params(db):
    """策略管理接口返回的 effective_params 必须能算出「只做多 / 1 倍 / 35% 名义上限」。"""
    from app.api.strategies import list_strategies

    payload = list_strategies(db)
    entry = next(row for row in payload["strategies"] if row["code"] == CODE)
    assert entry["name"] == "Martin Luk 强势回调（本地适配）"
    assert entry["effective_params"]["max_leverage"] == 1
    assert entry["effective_params"]["max_position_pct"] == 0.35
    assert any(item["code"] == CODE for item in DEFAULT_STRATEGIES)


def test_backtest_request_accepts_strategy():
    request = BacktestRequest(symbol="BTCUSDT", interval="15m", strategy=CODE)
    assert request.strategy == CODE
    with pytest.raises(ValidationError):
        BacktestRequest(symbol="BTCUSDT", interval="15m", strategy="not_a_strategy")


# ------------------------------------------------------------------ 特征层


def test_pullback_columns_present(enriched):
    missing = set(PULLBACK_COLUMNS) - set(enriched.columns)
    assert not missing


def test_avwap_is_vwap_anchored_at_trend_start(enriched):
    """AVWAP 必须等于「强势段起点 → 当前 bar」的累计成交额加权均价。"""
    segment = enriched.pb_trend_ok.ne(enriched.pb_trend_ok.shift(1).fillna(0)).cumsum()
    typical = (enriched["high"] + enriched["low"] + enriched["close"]) / 3.0
    target = enriched.index[enriched.pb_confirm.eq(1)][0]
    segment_id = segment.loc[target]
    seg = enriched.loc[segment.eq(segment_id) & (enriched.index <= target)]
    expected = float((typical * enriched["volume"]).loc[seg.index].sum() / enriched["volume"].loc[seg.index].sum())
    assert enriched.at[target, "avwap"] == pytest.approx(expected, rel=1e-9)


def test_first_touch_requires_runup():
    """回调若发生在强势段刚开始（不足 pullback_min_bars 根），不算首次回调。"""
    frame = build_features(_pullback_frame(), with_label=False)
    params = _params(pullback_min_bars=12)
    out = add_pullback_context(frame, params)
    segment = out.pb_trend_ok.ne(out.pb_trend_ok.shift(1).fillna(0)).cumsum()
    run_position = out.groupby(segment).cumcount()
    first = out.index[out.pb_first_touch.eq(1)]
    assert len(first) > 0
    # 段内已走满 min_bars 根（runup_ok 用 cumcount >= min_bars）才允许记首次回调
    assert (run_position.loc[first] >= params["pullback_min_bars"]).all()


def test_single_dip_yields_one_first_touch():
    """一次 V 形回调只记一次首次触及（连续擦 EMA9 不重复计数）。"""
    frame = build_features(_pullback_frame(), with_label=False)
    out = add_pullback_context(frame, _params())
    dip = out.loc[
        (out.index >= frame.index[1048]) & (out.index <= frame.index[1062])
    ]
    assert int(dip.pb_first_touch.sum()) == 1


def test_separate_dips_each_get_own_first_touch():
    """相隔足够远的两次回调各自产生一次首次触及。"""
    periods = 1500
    base = np.linspace(100.0, 150.0, periods)
    close = base.copy()
    for dip_at in (1050, 1300):
        for k in range(4):
            close[dip_at + k] = base[dip_at] * (1 - 0.005 * (k + 1) / 4)
        for k in range(6):
            i = dip_at + 4 + k
            close[i] = close[dip_at + 3] + (base[i] - close[dip_at + 3]) * (k + 1) / 6
    open_ = np.concatenate([[close[0]], close[:-1]])
    frame = pd.DataFrame(
        dict(
            open=open_,
            high=np.maximum(open_, close) + 0.05,
            low=np.minimum(open_, close) - 0.05,
            close=close,
            volume=np.full(periods, 100.0),
        ),
        index=pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC"),
    )
    out = add_pullback_context(build_features(frame, with_label=False), _params())
    assert int(out.pb_first_touch.sum()) >= 2
    assert int(out.pb_confirm.sum()) >= 2


def test_confirm_requires_reclaim_bar():
    """跌破 EMA9 后若持续走弱（无收阳收复），不给确认。"""
    periods = 1200
    base = np.linspace(100.0, 140.0, periods)
    close = base.copy()
    close[1050:] = base[1050] * 0.97  # 直接跳水后横盘走弱，不收回
    open_ = np.concatenate([[close[0]], close[:-1]])
    frame = pd.DataFrame(
        dict(
            open=open_,
            high=np.maximum(open_, close) + 0.02,
            low=np.minimum(open_, close) - 0.02,
            close=close,
            volume=np.full(periods, 100.0),
        ),
        index=pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC"),
    )
    out = add_pullback_context(build_features(frame, with_label=False), _params())
    assert int(out.pb_confirm.sum()) == 0


def test_confirm_window_is_respected(enriched):
    """确认必须落在 pullback_confirm_bars 窗口内：窗口缩到 1 根时，原信号消失。"""
    assert int(enriched.pb_confirm.sum()) == 1
    narrow = add_pullback_context(_pullback_frame(), _params(pullback_confirm_bars=1))
    assert int(narrow.pb_confirm.sum()) == 0


def test_confirm_only_once_per_pullback(enriched):
    assert int(enriched.pb_first_touch.sum()) == 1
    assert int(enriched.pb_confirm.sum()) == 1


def test_causality_truncation_matches(enriched):
    """反前视：把输入截断到确认 bar 后重算，该 bar 的所有派生列必须一致。"""
    target = enriched.index[enriched.pb_confirm.eq(1)][0]
    truncated = add_pullback_context(
        build_features(_pullback_frame().loc[:target], with_label=False), _params()
    )
    for column in ("avwap", "ema9", "pb_stop_ref", "pb_depth", "pb_bars_since_touch", "pb_confirm"):
        left, right = enriched.at[target, column], truncated.at[target, column]
        if pd.isna(left):
            assert pd.isna(right), f"{column} 出现泄漏"
        else:
            assert left == pytest.approx(float(right), rel=1e-9), f"{column} 出现泄漏"


def test_exit_flag_marks_close_below_ema9(enriched):
    below = enriched.loc[enriched["close"].lt(enriched["ema9"])]
    assert below["pb_exit_long"].eq(1).all()
    above = enriched.loc[enriched["close"].ge(enriched["ema9"])]
    assert above["pb_exit_long"].eq(0).all()


# ------------------------------------------------------------------ 信号层


def test_signal_opens_long_on_confirmation(confirm_row):
    signal = evaluate_signals(confirm_row, CODE, _params()).iloc[0]
    assert signal.direction == 1
    assert signal.block_reason == ""
    assert 0.5 <= signal.score <= 1.0


def test_signal_blocked_without_htf_trend(confirm_row):
    signal = evaluate_signals(
        _row(confirm_row, pb_trend_ok=0, pb_dir_4h=-1.0), CODE, _params()
    ).iloc[0]
    assert signal.direction == 0
    assert signal.block_reason == "htf_trend_weak"


def test_signal_blocked_without_pullback(confirm_row):
    signal = evaluate_signals(
        _row(confirm_row, pb_confirm=0, pb_bars_since_touch=np.nan), CODE, _params()
    ).iloc[0]
    assert signal.direction == 0
    assert signal.block_reason == "no_pullback"


def test_signal_blocked_without_confirmation(confirm_row):
    signal = evaluate_signals(_row(confirm_row, pb_confirm=0), CODE, _params()).iloc[0]
    assert signal.direction == 0
    assert signal.block_reason == "no_confirmation"


def test_signal_never_shorts(enriched):
    """只做多：任意一根（含暴跌段）都不允许出现 direction=-1。"""
    signals = evaluate_signals(enriched, CODE, _params())
    assert int(signals.direction.min()) >= 0
    assert int(signals.direction.eq(-1).sum()) == 0


def test_position_cap_leverage_and_risk_budget(confirm_row):
    signal = evaluate_signals(confirm_row, CODE, _params()).iloc[0]
    assert signal.leverage == 1
    assert 0 < signal.position_pct <= 0.35 + 1e-12
    assert signal.risk_pct <= 0.005 + 1e-12


def test_target_is_ten_r(confirm_row):
    signal = evaluate_signals(confirm_row, CODE, _params()).iloc[0]
    assert signal.tp_pct == pytest.approx(10 * signal.sl_pct, rel=1e-9)
    assert signal.net_reward_risk > 5


def test_stop_sits_below_pullback_low(confirm_row):
    """结构硬止损必须在回调低点下方（含缓冲），而不是随意一个 ATR 距离。"""
    row = confirm_row.iloc[0]
    signal = evaluate_signals(confirm_row, CODE, _params()).iloc[0]
    stop_price = float(row["close"]) * (1 - signal.sl_pct)
    assert stop_price < float(row["pb_stop_ref"])


def test_missing_context_does_not_crash():
    """未富化的单根帧（例如遗漏上下文）必须安全降级为不开仓。"""
    bare = pd.DataFrame(
        dict(
            open=[100.0],
            high=[100.2],
            low=[99.8],
            close=[100.0],
            volume=[1.0],
            atr_pct=[0.002],
            rsi_14=[55.0],
            adx_14=[20.0],
            supertrend_dir=[1.0],
            ema_ratio_9_21=[0.0],
            ema_ratio_21_50=[0.0],
            macd_hist_norm=[0.0],
            bb_pos=[0.5],
        ),
        index=pd.date_range("2026-01-01", periods=1, freq="15min", tz="UTC"),
    )
    signal = evaluate_signals(bare, CODE, _params()).iloc[0]
    assert signal.direction == 0
    assert signal.block_reason


# ------------------------------------------------------------------ 撮合层


def _sim_frame(dip_to: float = 99.5) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=40, freq="15min", tz="UTC")
    close = np.concatenate([np.full(10, 100.0), np.full(30, dip_to)])
    return pd.DataFrame(
        dict(open=close, high=close + 0.05, low=close - 0.05, close=close, atr_pct=0.002),
        index=idx,
    )


def _sim_signals(idx, exit_from: int | None) -> pd.DataFrame:
    signals = pd.DataFrame(
        dict(
            direction=0,
            sl_pct=0.02,
            tp_pct=0.2,
            position_pct=0.35,
            leverage=1,
            exit_long=0,
        ),
        index=idx,
    )
    signals.loc[idx[5], "direction"] = 1
    if exit_from is not None:
        signals.loc[idx[exit_from]:, "exit_long"] = 1
    return signals


def test_shared_engine_closes_on_ema_exit():
    data = _sim_frame()
    equity, trades = simulate(
        data, _sim_signals(data.index, 10), _params(), 10_000.0, 0.0004, 0.0002, "15m"
    )
    assert trades and trades[0]["status"] == "ema_exit"
    assert trades[0]["side"] == "long"
    assert len(equity) == len(data)


def test_other_strategies_ignore_exit_long_flag():
    """ema_exit 关闭时（其他策略默认），exit_long 列不得触发平仓。"""
    data = _sim_frame()
    other = resolve_params("trend_follow", {}, interval="15m")
    _, trades = simulate(
        data, _sim_signals(data.index, 10), other, 10_000.0, 0.0004, 0.0002, "15m"
    )
    assert trades[0]["status"] == "close_at_end"


def test_manage_order_replays_ema_exit(db, monkeypatch):
    """持仓管理按「上一根收盘失守 EMA9 → 本根开盘平仓」回放，与回测同口径。"""
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    order = Order(
        symbol="BTCUSDT",
        interval="15m",
        side="LONG",
        qty=1.0,
        entry_price=100.0,
        mark_price=100.0,
        margin=100.0,
        leverage=1,
        sl=90.0,
        tp=200.0,
        strategy=CODE,
        status="OPEN",
        created_at=start.to_pydatetime(),
    )
    db.add(order)
    db.commit()
    idx = pd.date_range(start - pd.Timedelta(minutes=600), periods=60, freq="15min")
    close = np.concatenate([np.full(50, 100.0), np.full(10, 99.0)])
    frame = pd.DataFrame(
        dict(open=close, high=close + 0.05, low=close - 0.05, close=close, volume=1.0),
        index=idx,
    )
    monkeypatch.setattr(
        order_manager.MarketDataService, "load_frame", lambda *a, **k: frame
    )
    monkeypatch.setattr(
        order_manager,
        "now_ms",
        lambda: int((idx[-1] + pd.Timedelta(minutes=15)).timestamp() * 1000),
    )
    order_manager.manage_order(db, order)
    assert order.status == "CLOSED"
    assert order.close_reason == "ema_exit"


# ---------------------------------------------------------- 撮合层：2R 减半出场


def _rally_frame(second_leg: float) -> pd.DataFrame:
    """10 根横盘 100（bar5 给信号、bar6 开盘 ≈100.02 入场，止损 98、2R≈104.06）
    → 5 根 105（越过 2R 触发减半）→ 第二段走势（回落或延伸）。"""
    idx = pd.date_range("2026-01-01", periods=40, freq="15min", tz="UTC")
    close = np.concatenate(
        [np.full(10, 100.0), np.full(5, 105.0), np.full(25, second_leg)]
    )
    return pd.DataFrame(
        dict(open=close, high=close + 0.05, low=close - 0.05, close=close, atr_pct=0.002),
        index=idx,
    )


def test_partial_tp_locks_half_then_breakeven():
    """2R 触发减半：第一笔 partial_tp 落袋，余仓止损移到保本价，回落以 stop_loss 收场。
    注：partial_tp_r 功能代码仍可用（默认 0.0 禁用），此处显式开启验证逻辑。"""
    data = _rally_frame(99.0)
    _, trades = simulate(
        data, _sim_signals(data.index, None), _params(partial_tp_r=2.0), 10_000.0, 0.0004, 0.0002, "15m"
    )
    assert [t["status"] for t in trades] == ["partial_tp", "stop_loss"]
    assert trades[0]["pnl"] > 0
    assert trades[0]["size"] == pytest.approx(trades[1]["size"], rel=1e-9)
    assert trades[0]["margin"] == pytest.approx(trades[1]["margin"], rel=1e-9)
    # 合计仍为正：2R 落袋的一半盖过余仓保本附近的小亏
    assert sum(t["pnl"] for t in trades) > 0


def test_partial_disabled_keeps_fixed_take_profit():
    """partial_tp_r=0 时行为不变：同一走势直接打 10R 固定目标。"""
    data = _rally_frame(130.0)
    _, trades = simulate(
        data,
        _sim_signals(data.index, None),
        _params(partial_tp_r=0.0),
        10_000.0,
        0.0004,
        0.0002,
        "15m",
    )
    assert [t["status"] for t in trades] == ["take_profit"]


def test_partial_cancels_fixed_tp_and_rides_on():
    """减半后取消固定目标：价格越过原 10R 不再 take_profit，余仓靠 EMA9 软出场收尾。
    注：partial_tp_r 功能代码仍可用（默认 0.0 禁用），此处显式开启验证逻辑。"""
    data = _rally_frame(130.0)
    _, trades = simulate(
        data, _sim_signals(data.index, 20), _params(partial_tp_r=2.0), 10_000.0, 0.0004, 0.0002, "15m"
    )
    assert [t["status"] for t in trades] == ["partial_tp", "ema_exit"]
    assert trades[1]["pnl"] > trades[0]["pnl"]  # 余仓吃到了 105→130 的延伸段

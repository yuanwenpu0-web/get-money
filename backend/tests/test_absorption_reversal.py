"""吸收反转（Absorption Strat 本地适配）专项测试：全部离线，不访问网络。

覆盖五层：
1. 特征层 `features.absorption`：
   - delta 估计与 intrabar 聚合（脚本 f_ibDelta 规则）；
   - 滚动最近秩百分位（NIST nearest-rank，与暴力排序对照）；
   - c1..c5 五个吸收条件、吸收方向语义（买方被吸收→空、卖方被吸收→多）；
   - 确认状态机（窗口内收盘穿越实体 / 打穿极值作废 / 超时作废）；
   - 信号间隔节流的两种语义（修正 vs 脚本 and/or 优先级原式）。
2. 参数层：脚本 1:1 默认值、范围校验、策略播种（默认不启用）。
3. 信号层：止损宽度门控、方向开关、趋势/时段过滤、2R 目标、仓位与杠杆。
4. 撮合层：1R 保本止损（回测与持仓管理同一实现）。
5. 因果性：截断到最后一根重算，派生列必须与全帧一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from app.backtest.execution import simulate
from app.features.absorption import (
    ABSORPTION_COLUMNS,
    SENSITIVITY_PRESETS,
    add_absorption_context,
    bucket_delta,
    intrabar_delta,
    percentile_nearest_rank,
)
from app.features.indicators import build_features
from app.models import Order, StrategyConfig
from app.schemas import BacktestRequest
from app.strategy.coin_universe import MAJOR_SYMBOLS, symbol_allowed
from app.strategy.defaults import DEFAULT_STRATEGIES, seed_strategies
from app.strategy.rules import evaluate_signals, resolve_params
from app.trading import order_manager
from app.trading.risk import breakeven_stop
from pydantic import ValidationError

CODE = "absorption_reversal"


def _params(interval: str = "15m", **overrides) -> dict:
    return resolve_params(CODE, {"fee_rate": 0.0004, "slippage": 0.0002}, overrides or None,
                          interval=interval)


def _frame(periods: int = 1200, seed: int = 7, base: float = 100.0) -> pd.DataFrame:
    """随机游走 OHLCV：用来让 c1..c5 自然会命中若干 bar。"""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.004, periods).cumsum()
    close = base * (1 + steps)
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(rng.normal(0.0, 0.003, periods))
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    volume = np.abs(rng.normal(1000.0, 300.0, periods)) + 100.0
    return pd.DataFrame(
        dict(open=open_, high=high, low=low, close=close, volume=volume),
        index=pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC"),
    )


@pytest.fixture(scope="module")
def enriched() -> pd.DataFrame:
    return add_absorption_context(
        build_features(_frame(), with_label=False), _params()
    )


# ---------------------------------------------------------------- 参数与注册


def test_script_defaults_are_faithful():
    params = _params()
    assert params["delta_window"] == 500
    assert params["delta_pct"] == 95.0
    assert params["body_mult"] == 0.25
    assert params["extreme_lookback"] == 20
    assert params["wick_mult"] == 0.20
    assert params["dominance_pct"] == 20.0
    assert params["abs_gap_bars"] == 10
    assert params["conf_mode"] == 1
    assert params["conf_window_bars"] == 3
    assert params["sl_buffer_atr"] == 0.25
    assert params["target_r"] == 2.0
    assert params["be_trigger_r"] == 1.0
    assert params["min_sl_atr"] == 0.30
    assert params["max_sl_atr"] == 3.00
    assert params["max_hold_bars"] == 30
    # 脚本 default_qty_type = percent_of_equity 30 → 名义 30%、1 倍杠杆
    assert params["max_leverage"] == 1
    assert params["base_position_pct"] == 0.30
    assert params["max_position_pct"] == 0.30
    # 脚本没有追踪止损、没有平仓后冷却
    assert params["trailing_enabled"] is False
    assert params["cooldown_bars"] == 0
    assert params["legacy_gap_logic"] is False


def test_sensitivity_presets_match_script():
    assert SENSITIVITY_PRESETS["low"] == {
        "delta_pct": 97.5, "body_mult": 0.20, "extreme_lookback": 40, "wick_mult": 0.30,
    }
    assert SENSITIVITY_PRESETS["medium"] == {
        "delta_pct": 95.0, "body_mult": 0.25, "extreme_lookback": 20, "wick_mult": 0.20,
    }
    assert SENSITIVITY_PRESETS["high"] == {
        "delta_pct": 90.0, "body_mult": 0.35, "extreme_lookback": 10, "wick_mult": 0.15,
    }


def test_resolve_rejects_out_of_range_and_unknown():
    with pytest.raises(ValueError):
        _params(delta_pct=120.0)
    with pytest.raises(ValueError):
        _params(max_sl_atr=0.0)
    with pytest.raises(ValueError):
        _params(target_r=0.5)
    with pytest.raises(ValueError):
        _params(nonexistent=1)
    with pytest.raises(ValueError):
        _params(legacy_gap_logic=1.0)  # 布尔参数不接受数字


def test_seed_registers_disabled_by_default(db):
    seed_strategies(db)
    row = db.query(StrategyConfig).filter(StrategyConfig.code == CODE).one()
    # 样本不足以证明 edge → 注册但不在自动巡检里参与资金竞争
    assert row.enabled == 0
    assert any(item["code"] == CODE for item in DEFAULT_STRATEGIES)


def test_backtest_request_accepts_strategy():
    assert BacktestRequest(symbol="BTCUSDT", interval="15m", strategy=CODE).strategy == CODE
    with pytest.raises(ValidationError):
        BacktestRequest(symbol="BTCUSDT", interval="15m", strategy="not_a_strategy")


def test_universe_binding_limits_strategy_to_majors():
    """策略级宇宙：只在主流币白名单上运行，其余策略不受限制。"""
    assert "BTCUSDT" in MAJOR_SYMBOLS
    assert symbol_allowed(CODE, "BTCUSDT")
    assert symbol_allowed(CODE, "btcusdt")  # 大小写不敏感
    assert not symbol_allowed(CODE, "WHATEVERUSDT")
    assert symbol_allowed("trend_follow", "WHATEVERUSDT")  # 未登记宇宙的策略不约束


# ---------------------------------------------------------------- 特征层


def test_columns_present(enriched):
    assert not set(ABSORPTION_COLUMNS) - set(enriched.columns)


def test_estimated_delta_matches_script_formula(enriched):
    """脚本降级分支：delta = volume × (2*close − high − low) / (high − low)。"""
    row = enriched.iloc[-1]
    expected = (
        row["volume"] * (2 * row["close"] - row["high"] - row["low"]) / (row["high"] - row["low"])
    )
    assert row["abs_delta"] == pytest.approx(expected, rel=1e-9)
    assert row["abs_dominance"] == pytest.approx(abs(expected) / row["volume"] * 100, rel=1e-9)


def test_intrabar_delta_follows_ib_rule():
    """f_ibDelta：实体定方向，十字星退化到「与前一根收盘比较」，再平为 0。"""
    idx = pd.date_range("2026-01-01", periods=5, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        dict(
            #          阳          阴         十字(高于前收)  十字(低于前收)  十字(等于前收)
            open=[10.0, 10.0, 10.2, 10.1, 10.1],
            high=[10.7] * 5,
            low=[9.4] * 5,
            close=[10.5, 9.5, 10.2, 10.1, 10.1],
            volume=[1.0, 2.0, 3.0, 4.0, 5.0],
        ),
        index=idx,
    )
    values = intrabar_delta(frame).tolist()
    assert values[0] == pytest.approx(1.0)    # 阳 → +v
    assert values[1] == pytest.approx(-2.0)   # 阴 → −v
    assert values[2] == pytest.approx(3.0)    # 十字 + 高于前收(9.5) → +v
    assert values[3] == pytest.approx(-4.0)   # 十字 + 低于前收(10.2) → −v
    assert values[4] == pytest.approx(0.0)    # 十字 + 与前收(10.1)相等 → 0


def test_bucket_delta_sums_lower_bars_into_parent():
    parent_idx = pd.date_range("2026-01-01T00:00", periods=2, freq="1h", tz="UTC")
    lower_idx = pd.date_range("2026-01-01T00:00", periods=8, freq="15min", tz="UTC")
    lower = pd.Series([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], index=lower_idx)
    buckets = bucket_delta(lower, parent_idx)
    assert buckets.iloc[0] == pytest.approx(10.0)
    assert buckets.iloc[1] == pytest.approx(-10.0)


def test_percentile_nearest_rank_matches_bruteforce():
    values = np.arange(1.0, 61.0)  # 1..60
    got = percentile_nearest_rank(values, window=20, pct=95.0)
    assert np.isnan(got[:19]).all()  # 窗口不足 → NaN（Pine 预热期返回 na）
    for i in (19, 25, 40, 59):
        window = np.sort(values[i - 19 : i + 1])
        expected = window[int(np.ceil(95 / 100 * 20)) - 1]  # NIST nearest-rank
        assert got[i] == pytest.approx(expected)


def test_absorption_requires_all_five_conditions():
    """五个吸收条件必须同时成立：|delta| 进百分位、小实体或与 K 线方向背离、
    处于摆动极值、有拒绝影线、单边支配度达标。"""
    enriched, spot, params = _scenario(1)
    row = enriched.iloc[spot]
    assert row["abs_raw"] > 0
    assert row["abs_abs_delta"] >= row["abs_percentile"]
    tiny_body = abs(row["close"] - row["open"]) <= params["body_mult"] * row["abs_atr"]
    disagree = not (
        (row["abs_delta"] > 0 and row["close"] > row["open"])
        or (row["abs_delta"] < 0 and row["close"] < row["open"])
    )
    assert tiny_body or disagree
    assert row["abs_wick_ratio"] >= params["wick_mult"]
    assert row["abs_dominance"] >= params["dominance_pct"]
    assert row["high"] >= row["abs_extreme_high"]


def _scenario(delta_sign: int, follow: str = "flat", periods: int = 300) -> tuple:
    """构造**确定性**的吸收场景（用 lower-tf delta，避免 estDelta 的量纲耦合）。

    前 250 根小幅漫游提供百分位/ATR 基线；第 250 根是手工放大的吸收 bar：
    - delta 符号由 delta_sign 指定（+1 买方被吸收 / -1 卖方被吸收），绝对值远大于其他 bar；
    - 该 bar 是近 20 根最高（做空侧）或最低（做多侧）、小实体、大影线 → 五个条件齐备；
    - follow 控制吸收 bar 之后的走势：`rally`（收盘逐级上行，用于做多确认）、
      `break`（先打穿吸收极值）、默认 `flat`（横着不动 → 超时作废）。
    返回 (enriched, 吸收 bar 位置, params)。测试专用参数在内部覆盖（窗口缩到 60/5 根）。
    """
    rng = np.random.default_rng(3)
    drift = rng.normal(0.0, 0.0005, periods).cumsum()
    close = 100.0 * (1 + drift)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0008)
    low = np.minimum(open_, close) * (0.9992)
    volume = np.abs(rng.normal(1000.0, 80.0, periods)) + 500.0
    index = pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC")

    spot = 250
    if delta_sign > 0:  # 买方被吸收：极值在上方（high 是近 20 根最高）
        high[spot] = max(high[spot - 20 : spot]) * 1.004
        low[spot] = min(low[spot - 20 : spot]) * 0.996
        close[spot] = close[spot] * 0.9995   # 小实体：收盘几乎不动
        open_[spot] = close[spot] * 1.0003
    else:               # 卖方被吸收：极值在下方（low 是近 20 根最低）
        low[spot] = min(low[spot - 20 : spot]) * 0.996
        high[spot] = max(high[spot - 20 : spot]) * 1.004
        close[spot] = close[spot] * 1.0005
        open_[spot] = close[spot] * 0.9997
    volume[spot] = float(volume.mean())  # 量中性，|delta| 的优势完全来自手工注入
    body_top = max(open_[spot], close[spot])
    body_bottom = min(open_[spot], close[spot])
    if follow == "rally":  # 后续逐级上行 → 收盘穿越实体上沿（做多确认）
        for offset in range(1, 8):
            i = spot + offset
            close[i] = close[i - 1] * 1.004
            open_[i] = close[i - 1]
            high[i] = close[i] * 1.0008
            low[i] = max(open_[i], close[i]) * 0.9992
    elif follow == "break":  # 后续先跌穿吸收极值（做多信号作废）
        for offset in range(1, 8):
            i = spot + offset
            close[i] = close[i - 1] * 0.996
            open_[i] = close[i - 1]
            high[i] = close[i] * 1.0008
            low[i] = low[spot] * 0.995
    else:  # flat：整根静止在「实体上沿下方、吸收极值上方」→ 既不确认也不打穿 → 超时作废
        still = min(body_top * 0.9995, max(body_bottom, low[spot] * 1.0005))
        for offset in range(1, 8):
            i = spot + offset
            open_[i] = high[i] = low[i] = close[i] = still

    frame = pd.DataFrame(
        dict(open=open_, high=high, low=low, close=close, volume=volume), index=index
    )
    lower = pd.Series(rng.normal(0.0, 1.0, periods) * 10.0, index=index)
    lower.iloc[spot] = delta_sign * 40_000.0  # 远超窗口内的其他 bar → 必然进入百分位
    params = _params(delta_window=60, extreme_lookback=5)
    enriched = add_absorption_context(
        build_features(frame, with_label=False), params, lower_tf_delta=lower
    )
    enriched.attrs["body_top"] = body_top
    enriched.attrs["body_bottom"] = body_bottom
    return enriched, spot, params


def test_direction_semantics_buyer_absorbed_means_short():
    """买方被吸收（delta>0 却推不动价格）→ 待做空；卖方被吸收 → 待做多。"""
    for sign, expected_dir in ((1, -1.0), (-1, 1.0)):
        enriched, spot, _ = _scenario(sign)
        row = enriched.iloc[spot]
        assert row["abs_bar"] > 0, "手工场景必须命中吸收 bar"
        assert row["abs_dir"] == expected_dir
        assert row["abs_extreme"] == (row["high"] if sign > 0 else row["low"])
        expected_level = row["abs_body_top"] if expected_dir == 1 else row["abs_body_bottom"]
        assert row["abs_level"] == pytest.approx(expected_level)


def _rerun(frame: pd.DataFrame, **overrides) -> pd.DataFrame:
    """把已富化帧的 OHLCV 抽出，换参数重算（用于窗口/灵敏度对照）。"""
    base = dict(delta_window=60, extreme_lookback=5)
    base.update(overrides)
    return add_absorption_context(
        build_features(frame[["open", "high", "low", "close", "volume"]], with_label=False),
        _params(**base),
        lower_tf_delta=frame["abs_delta"],
    )


def test_confirmation_needs_close_through_body():
    """做多信号只在「收盘站回实体上沿」的那根出现，止损落在吸收极值下方。"""
    enriched, spot, _ = _scenario(-1, follow="rally")  # 卖方被吸收 → 待做多
    hits = np.flatnonzero(enriched["abs_signal"].to_numpy())
    assert len(hits) == 1, "一次吸收最多确认一次"
    idx = int(hits[0])
    assert idx > spot and idx - spot <= 3
    assert enriched["close"].iloc[idx] > enriched.at[enriched.index[spot], "abs_body_top"]
    assert enriched["abs_signal"].iloc[idx] == 1
    assert enriched["abs_sl"].iloc[idx] < enriched.at[enriched.index[spot], "abs_extreme"]


def test_confirmation_cancelled_if_extreme_breached_first():
    """价格先打穿吸收极值 → 信号作废，即便后面又穿越实体也不补发。"""
    enriched, _, _ = _scenario(-1, follow="break")
    assert int(enriched["abs_signal"].ne(0).sum()) == 0


def test_confirmation_expires_after_window():
    """横着不动超过 conf_window_bars → 超时作废；窗口放宽后同帧可能出现信号。"""
    enriched, _, _ = _scenario(-1)
    assert int(enriched["abs_signal"].ne(0).sum()) == 0
    assert int(_rerun(enriched, conf_window_bars=1)["abs_signal"].ne(0).sum()) == 0
    immediate = _rerun(enriched, conf_mode=0)  # conf_mode=0：吸收 bar 收盘即入场
    assert int(immediate["abs_signal"].ne(0).sum()) == 1


def test_legacy_gap_logic_reproduces_script_precedence_bug():
    """脚本原式 `(absRaw and na(lastAbs)) or (bar_index - lastAbs > absGap)` 因 and 优先级高于 or，
    等价于「距上次信号超过 absGap 根就判定为吸收 bar」，与是否真有吸收无关。
    这里只验证开关确实改变行为（且修正语义下的信号子集关系），不评价脚本本身。"""
    frame = build_features(_frame(periods=2000, seed=11), with_label=False)
    params = {**_params(), "delta_window": 40, "extreme_lookback": 3, "delta_pct": 60.0}
    fixed = add_absorption_context(frame, params)
    legacy = add_absorption_context(frame, {**params, "legacy_gap_logic": True})
    assert int(legacy["abs_bar"].sum()) > int(fixed["abs_bar"].sum())
    # 修正语义下，每个吸收 bar 都必须是原始吸收 bar
    assert fixed.loc[fixed["abs_bar"] > 0, "abs_raw"].gt(0).all()


def test_stop_sits_beyond_absorption_extreme():
    """止损 = 吸收极值 ∓ sl_buffer_atr×ATR（极值是 cena 参照，不是随手写个 ATR 距离）。"""
    for sign, direction in ((-1, 1), (1, -1)):
        enriched, spot, params = _scenario(sign, follow="rally" if sign < 0 else "flat")
        hits = np.flatnonzero(enriched["abs_signal"].to_numpy())
        if len(hits) == 0:
            continue
        # 极值与 ATR：极值取自吸收 bar，ATR 取自确认 bar（脚本在确认时用当前 bar 的 ATR）
        extreme = float(enriched.iloc[spot]["abs_extreme"])
        row = enriched.iloc[int(hits[0])]
        buffered = row["abs_atr"] * params["sl_buffer_atr"]
        if direction == 1:  # 做多：止损在吸收最低点下方
            assert row["abs_sl"] == pytest.approx(extreme - buffered)
        else:
            assert row["abs_sl"] == pytest.approx(extreme + buffered)
    assert row["abs_signal"] in (1.0, -1.0)


def test_causality_truncation_matches(enriched):
    """反前视：把输入截断到中间某根后重算，该根的派生列必须一致。"""
    target = enriched.index[900]
    truncated = add_absorption_context(
        build_features(_frame().loc[:target], with_label=False), _params()
    )
    for column in ("abs_delta", "abs_percentile", "abs_bar", "abs_signal", "abs_sl"):
        left, right = enriched.at[target, column], truncated.at[target, column]
        if pd.isna(left):
            assert pd.isna(right), f"{column} 出现泄漏"
        else:
            assert float(left) == pytest.approx(float(right), rel=1e-9), f"{column} 出现泄漏"


def test_empty_frame_is_safe():
    out = add_absorption_context(pd.DataFrame(), _params())
    assert set(ABSORPTION_COLUMNS).issubset(out.columns)


# ---------------------------------------------------------------- 信号层


def _signal_row(enriched_frame: pd.DataFrame, offset: int = 0, **overrides):
    """取一根已确认的信号行；随机帧可能没有信号时调用方自行构造。"""
    hits = enriched_frame.index[enriched_frame["abs_signal"] != 0]
    if len(hits) == 0:
        return None
    row = enriched_frame.loc[[hits[min(offset, len(hits) - 1)]]].copy()
    for key, value in overrides.items():
        row[key] = value
    return row


def _synthetic_signal_row(direction: int = 1) -> pd.DataFrame:
    """手搓一根「确认 bar」：close=100，止损 99（1% 风险），ATR=1（≈1%）。"""
    row = {name: np.nan for name in ABSORPTION_COLUMNS}
    row.update(
        open=100.0, high=100.4, low=99.6, close=100.0,
        atr_pct=0.01, rsi_14=50.0, adx_14=25.0, supertrend_dir=1.0,
        ema_ratio_9_21=0.0, ema_ratio_21_50=0.0, macd_hist_norm=0.0, bb_pos=0.5,
        abs_signal=direction, abs_sl=99.0 if direction == 1 else 101.0,
        abs_risk=1.0, abs_atr=1.0, abs_delta=100.0, abs_abs_delta=100.0,
        abs_dominance=60.0, abs_percentile=90.0, abs_trend_ema=100.0,
        volume=1000.0,
    )
    return pd.DataFrame([row], index=pd.date_range("2026-01-01T12:00", periods=1, freq="15min", tz="UTC"))


def test_signal_opens_and_target_is_two_r():
    signals = evaluate_signals(_synthetic_signal_row(1), CODE, _params()).iloc[0]
    assert signals.direction == 1
    assert signals.block_reason == ""
    assert signals.sl_pct == pytest.approx(0.01, rel=1e-6)   # 100 → 99
    assert signals.tp_pct == pytest.approx(0.02, rel=1e-6)   # 2R
    assert signals.leverage == 1
    assert 0 < signals.position_pct <= 0.30 + 1e-12
    assert 0.75 <= signals.score <= 1.0


def test_short_signal_is_allowed_by_default():
    assert evaluate_signals(_synthetic_signal_row(-1), CODE, _params()).iloc[0].direction == -1


def test_direction_mode_blocks_one_side():
    long_only = evaluate_signals(
        _synthetic_signal_row(-1), CODE, _params(direction_mode=1)
    ).iloc[0]
    assert long_only.direction == 0
    assert long_only.block_reason == "short_disabled"
    short_only = evaluate_signals(
        _synthetic_signal_row(1), CODE, _params(direction_mode=2)
    ).iloc[0]
    assert short_only.direction == 0
    assert short_only.block_reason == "long_disabled"


def test_stop_width_gate_rejects_too_wide_and_too_tight():
    # 风险 1.0 = 1 ATR → 通过；改成 4 ATR → 过宽；0.2 ATR → 过窄
    wide = _synthetic_signal_row(1).copy()
    wide["abs_sl"] = 96.0
    wide["abs_risk"] = 4.0
    assert evaluate_signals(wide, CODE, _params()).iloc[0].block_reason == "stop_too_wide"
    tight = _synthetic_signal_row(1).copy()
    tight["abs_sl"] = 99.8
    tight["abs_risk"] = 0.2
    assert evaluate_signals(tight, CODE, _params()).iloc[0].block_reason == "stop_too_tight"


def test_stop_wrong_side_is_rejected():
    row = _synthetic_signal_row(1).copy()
    row["abs_sl"] = 101.0  # 做多止损却在上方
    row["abs_risk"] = 1.0
    assert evaluate_signals(row, CODE, _params()).iloc[0].block_reason == "stop_wrong_side"


def test_trend_filter_with_and_against_ema():
    above = _synthetic_signal_row(1)  # close=100 == ema 100（不 > ema）→ with 模式应拦
    assert evaluate_signals(above, CODE, _params(trend_mode=1)).iloc[0].block_reason == "trend_filter"
    against = evaluate_signals(above, CODE, _params(trend_mode=2)).iloc[0]
    assert against.block_reason == "trend_filter"  # close 不 < ema → 逆势条件也不满足


def test_session_filter_by_utc_hour():
    row = _synthetic_signal_row(1)  # 12:00 UTC
    inside = evaluate_signals(row, CODE, _params(session_start_hour=9.5, session_end_hour=16.0))
    assert inside.iloc[0].direction == 1
    outside = evaluate_signals(row, CODE, _params(session_start_hour=20.0, session_end_hour=4.0))
    assert outside.iloc[0].block_reason == "session_filter"


def test_no_signal_rows_are_blocked():
    row = _synthetic_signal_row(1).copy()
    row["abs_signal"] = 0.0
    assert evaluate_signals(row, CODE, _params()).iloc[0].block_reason == "no_absorption_signal"


def test_real_frame_never_shorts_when_long_only(enriched):
    signals = evaluate_signals(enriched, CODE, _params(direction_mode=1))
    assert int(signals.direction.min()) >= 0


# ---------------------------------------------------------------- 撮合层：1R 保本


def test_breakeven_stop_is_monotonic_and_triggered_once():
    # 做多：entry 100、risk 2 → 1R 触发价 102；触及过就把止损抬到 100 且不再回落
    stop = breakeven_stop(1, 100.0, 98.0, high=101.9, low=99.0, trigger_r=1.0, risk=2.0)
    assert stop == 98.0
    stop = breakeven_stop(1, 100.0, 98.0, high=102.5, low=99.0, trigger_r=1.0, risk=2.0)
    assert stop == 100.0
    stop = breakeven_stop(1, 100.0, 100.0, high=101.0, low=99.0, trigger_r=1.0, risk=2.0)
    assert stop == 100.0  # 之后未触及也不回落
    # 做空对称
    stop = breakeven_stop(-1, 100.0, 102.0, high=101.0, low=97.5, trigger_r=1.0, risk=2.0)
    assert stop == 100.0


def test_be_disabled_keeps_initial_stop():
    assert breakeven_stop(1, 100.0, 98.0, high=105.0, low=99.0, trigger_r=0.0, risk=2.0) == 98.0


def _flat_frame(peak: float) -> pd.DataFrame:
    """10 根横盘 100（bar 5 出信号、bar 6 开盘 ≈100.02 入场，止损 99、目标 102）
    → bars 10~19 抬到 peak（越过 1R≈101.04 但不触及 2R=102）→ bar 20 起跌到 99 下方。"""
    idx = pd.date_range("2026-01-01", periods=40, freq="15min", tz="UTC")
    close = np.concatenate([np.full(10, 100.0), np.full(30, peak)])
    frame = pd.DataFrame(
        dict(open=close, high=close + 0.02, low=close - 0.02, close=close, atr_pct=0.01),
        index=idx,
    )
    frame.iloc[20:, frame.columns.get_loc("low")] = 98.9
    frame.iloc[20:, frame.columns.get_loc("close")] = 99.0
    return frame


def _flat_signals_from(index) -> pd.DataFrame:
    frame = pd.DataFrame(
        dict(direction=0, sl_pct=0.01, tp_pct=0.02, position_pct=0.30, leverage=1, exit_long=0),
        index=index,
    )
    frame.loc[index[5], "direction"] = 1
    return frame


def test_shared_engine_moves_stop_to_breakeven_after_one_r():
    """先看到 1R（≈101.04）后回落：止损已被抬到保本价 100 → 以 stop_loss 在 100 附近收场。"""
    _, trades = simulate(
        _flat_frame(101.5), _flat_signals_from(_flat_frame(101.5).index), _params(be_trigger_r=1.0),
        10_000.0, 0.0004, 0.0002, "15m",
    )
    assert trades and trades[0]["status"] == "stop_loss"
    assert trades[0]["exit_price"] == pytest.approx(100.0, abs=0.05)  # 保本价成交


def test_without_be_the_same_path_exits_at_initial_stop():
    """be_trigger_r=0 时行为不变：同一走势回到初始止损 99 才平。"""
    frame = _flat_frame(101.5)
    _, trades = simulate(
        frame, _flat_signals_from(frame.index), _params(be_trigger_r=0.0),
        10_000.0, 0.0004, 0.0002, "15m",
    )
    assert trades and trades[0]["exit_price"] == pytest.approx(99.0, abs=0.05)


def test_manage_order_replays_breakeven(db, monkeypatch):
    """实盘持仓管理与回测同一口径：触及 1R 后止损移到入场价。"""
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    order = Order(
        symbol="BTCUSDT", interval="15m", side="LONG", qty=1.0, entry_price=100.0,
        mark_price=100.0, margin=100.0, leverage=1, sl=98.0, tp=104.0, strategy=CODE,
        status="OPEN", created_at=start.to_pydatetime(),
    )
    db.add(order)
    db.commit()
    idx = pd.date_range(start - pd.Timedelta(minutes=750), periods=60, freq="15min")
    close = np.concatenate([np.full(52, 100.0), np.full(8, 99.0)])
    frame = pd.DataFrame(
        dict(open=close, high=np.where(close >= 100.0, 102.5, close + 0.02),
             low=close - 0.02, close=close, volume=1.0),
        index=idx,
    )
    monkeypatch.setattr(order_manager.MarketDataService, "load_frame", lambda *a, **k: frame)
    monkeypatch.setattr(
        order_manager, "now_ms",
        lambda: int((idx[-1] + pd.Timedelta(minutes=15)).timestamp() * 1000),
    )
    # 初始风险 = |100 − 98| = 2 → 1R 触发价 102；前 52 根 high=102.5 已触及
    order_manager.manage_order(db, order)
    assert float(order.sl or 0.0) == pytest.approx(100.0, abs=0.01)

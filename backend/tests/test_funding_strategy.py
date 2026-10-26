"""funding_extreme 方向逻辑与 funding_filter 拦截测试。全部本地帧构造，不发起真实请求。"""

from app.features.funding import attach_funding
from app.features.indicators import build_features
from app.strategy.rules import evaluate_signals, resolve_params

from funding_helpers import bar_frame, funding_frame


def _extreme_frame(zscore_value, adx_value):
    frame = bar_frame(periods=200)
    funding = funding_frame([1_767_225_600_000, 1_767_254_400_000], [0.0001, 0.0005])
    attach_funding(frame, funding)
    data = build_features(frame, funding=funding)
    data["funding_zscore"] = zscore_value  # 直接设定极端值验证方向逻辑
    data["adx_14"] = adx_value
    return data


def test_funding_extreme_long_on_negative_extreme():
    data = _extreme_frame(-3.0, adx_value=15.0)
    result = evaluate_signals(data, "funding_extreme", resolve_params("funding_extreme", {}))
    assert result.direction.iloc[-1] == 1  # 负费率极端 + 震荡市 → 多
    assert "trending_market" not in set(result.block_reason)


def test_funding_extreme_short_on_positive_extreme():
    data = _extreme_frame(3.0, adx_value=15.0)
    result = evaluate_signals(data, "funding_extreme", resolve_params("funding_extreme", {}))
    assert result.direction.iloc[-1] == -1  # 正费率极端 + 震荡市 → 空
    assert "trending_market" not in set(result.block_reason)


def test_funding_extreme_blocked_by_trending_market():
    data = _extreme_frame(-3.0, adx_value=40.0)
    result = evaluate_signals(data, "funding_extreme", resolve_params("funding_extreme", {}))
    assert (result.direction == 0).all()
    assert result.block_reason.iloc[-1] == "trending_market"


def test_funding_extreme_flat_below_threshold():
    data = _extreme_frame(0.5, adx_value=15.0)
    result = evaluate_signals(data, "funding_extreme", resolve_params("funding_extreme", {}))
    assert (result.direction == 0).all()


def _trend_frame(zscore_value):
    frame = bar_frame(periods=200)
    data = build_features(frame, funding=funding_frame([1_767_225_600_000], [0.0001]))
    data["funding_zscore"] = zscore_value
    # 构造趋势跟随的多头环境
    data["supertrend_dir"] = 1.0
    data["ema_ratio_9_21"] = 0.01
    data["ema_ratio_21_50"] = 0.01
    data["macd_hist_norm"] = 0.01
    data["adx_14"] = 40.0
    return data


def test_funding_filter_blocks_long_on_positive_extreme():
    data = _trend_frame(3.0)
    params = resolve_params("trend_follow", {})
    params["funding_filter"] = True
    result = evaluate_signals(data, "trend_follow", params)
    assert (result.direction == 0).all()
    assert result.block_reason.iloc[-1] == "funding_filter_long"


def test_funding_filter_blocks_short_on_negative_extreme():
    data = _trend_frame(-3.0)
    data["supertrend_dir"] = -1.0
    data["ema_ratio_9_21"] = -0.01
    data["ema_ratio_21_50"] = -0.01
    data["macd_hist_norm"] = -0.01
    params = resolve_params("trend_follow", {})
    params["funding_filter"] = True
    result = evaluate_signals(data, "trend_follow", params)
    assert (result.direction == 0).all()
    assert result.block_reason.iloc[-1] == "funding_filter_short"


def test_funding_filter_off_by_default():
    data = _trend_frame(3.0)
    result = evaluate_signals(data, "trend_follow", resolve_params("trend_follow", {}))
    assert result.direction.iloc[-1] == 1  # 默认关闭，不拦截

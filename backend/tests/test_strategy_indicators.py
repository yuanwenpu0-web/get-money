import pandas as pd
import pytest

from app.strategy import indicators
from app.strategy.registry import PrepareContext, get_spec
from app.strategy.rules import resolve_params


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(indicators, "_INDICATORS", {})


def test_registration_and_parameter_declarations(registry):
    indicator = indicators.register_strategy_indicator("supertrend_current")
    spec = get_spec("supertrend_current")
    assert indicators.get_indicator(spec.code) is indicator
    assert indicators.all_indicators() == (indicator,)
    assert indicator.defaults == spec.defaults
    assert indicator.ranges == spec.ranges
    assert indicator.bool_keys == spec.bool_keys
    assert indicator.int_keys == spec.int_keys
    assert indicator.prepare_keys == spec.prepare_keys
    indicator.defaults["atr_period"] = 999
    assert spec.defaults["atr_period"] != 999
    with pytest.raises(ValueError, match="指标重复注册"):
        indicators.register_strategy_indicator(spec.code)
    with pytest.raises(ValueError, match="未知指标"):
        indicators.get_indicator("missing")
    with pytest.raises(ValueError, match="未知策略"):
        indicators.register_strategy_indicator("missing")


def test_parameters_reuse_strategy_validation(registry):
    indicator = indicators.register_strategy_indicator("supertrend_current")
    overrides = {"atr_period": 20, "use_adx_filter": True}
    params = indicator.validate(overrides=overrides)
    assert params == resolve_params(indicator.code, {}, overrides)
    assert indicator.lookback(params) == indicator.spec.lookback(params)
    for invalid in ({"unknown": 1}, {"atr_period": 0}, {"atr_period": 1.5},
                    {"use_adx_filter": 1}, {"fee_rate": 0.001},
                    {"tp1_r": 5}):
        with pytest.raises(ValueError):
            indicator.validate(overrides=invalid)


def test_prepare_and_signals_delegate_with_context(registry):
    indicator = indicators.register_strategy_indicator("supertrend_current")
    params = indicator.resolve_params()
    frame = pd.DataFrame({"high": [11.0] * 30, "low": [9.0] * 30,
                          "close": [10.0] * 30})
    ctx = PrepareContext(symbol="TEST")
    prepared = indicator.prepare(frame, params, ctx)
    expected = indicator.spec.prepare(frame, params, ctx)
    pd.testing.assert_frame_equal(prepared, expected)
    pd.testing.assert_frame_equal(indicator.signals(prepared, params),
                                  indicator.spec.signals(expected, params))
    pd.testing.assert_frame_equal(indicator.prepare(frame, params),
                                  indicator.spec.prepare(frame, params, PrepareContext()))


@pytest.mark.parametrize("exit_config", [None, {"method": "atr_r"}])
def test_engine_routes_through_indicator_and_preserves_rule_results(monkeypatch, registry, exit_config):
    from app.strategy import engine, rules

    code = "supertrend_current"
    params = rules.resolve_params(code, {}, {"exit": exit_config} if exit_config else None)
    index = pd.date_range("2026-01-01", periods=150, freq="1h", tz="UTC")
    frame = pd.DataFrame({"open": [10.0] * 150, "high": [11.0] * 150,
                          "low": [9.0] * 150, "close": [10.0] * 150,
                          "atr_pct": [0.1] * 150}, index=index)
    instance = engine.StrategyEngine(None)
    monkeypatch.setattr(instance, "_get_config", lambda code: None)
    monkeypatch.setattr(instance, "_prepare_features", lambda *args: frame.copy())
    monkeypatch.setattr(engine.metrics_mod, "load_latest_metrics", lambda *args: {})
    monkeypatch.setattr(engine, "now_ms", lambda: int(index[-1].timestamp() * 1000) + 3600000)

    prepared = rules.prepare_frame(frame.copy(), code, params, symbol="TEST", service=instance.service)
    window = prepared if exit_config else prepared.iloc[-1:]
    expected = rules.evaluate_signals(window, code, params, symbol="TEST")
    calls = []
    original_prepare = indicators.StrategyIndicator.prepare
    original_signals = indicators.StrategyIndicator.signals

    def prepare(self, data, values, ctx=None):
        calls.append(("prepare", len(data), ctx.symbol, ctx.service))
        return original_prepare(self, data, values, ctx)

    def signals(self, data, values):
        calls.append(("signals", len(data)))
        result = original_signals(self, data, values)
        return result

    monkeypatch.setattr(indicators.StrategyIndicator, "prepare", prepare)
    monkeypatch.setattr(indicators.StrategyIndicator, "signals", signals)
    original_overlay = engine.apply_exit_overlay

    def overlay(data, result, values):
        actual = original_overlay(data, result, values)
        pd.testing.assert_frame_equal(actual, expected)
        return actual

    monkeypatch.setattr(engine, "apply_exit_overlay", overlay)
    signal = instance.evaluate("TEST", "1h", code, params=params)
    assert signal is not None
    assert calls == [("prepare", 150, "TEST", instance.service),
                     ("prepare", len(window), "TEST", None), ("signals", len(window))]
    assert indicators.all_indicators() == ()

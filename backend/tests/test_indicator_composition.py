import pandas as pd
import pytest

from app.strategy.indicator_composition import IndicatorComposition
from app.strategy.registry import PrepareContext
from app.strategy.rules import evaluate_signals, resolve_params


def frame():
    index = pd.date_range("2026-01-01", periods=150, freq="1h", tz="UTC")
    return pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0,
                         "volume": 1.0, "atr_pct": 0.1}, index=index)


def config(*entries):
    return {"components": list(entries)}


def test_single_component_matches_existing_path():
    entry = {"name": "primary", "code": "supertrend_current", "params": {"atr_period": 20}}
    composition = IndicatorComposition(config(entry))
    data = frame()
    before = data.copy(deep=True)
    result, details = composition.evaluate(data, ctx=PrepareContext(symbol="TEST"))
    expected = evaluate_signals(data.copy(), entry["code"], resolve_params(entry["code"], {}, entry["params"]), symbol="TEST")
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(details["primary"], expected)
    pd.testing.assert_frame_equal(data, before)
    assert composition.lookback() == composition.components[0].indicator.lookback(composition.components[0].params)


@pytest.mark.parametrize("bad", [
    {}, {"components": []}, {"components": "x"},
    config({"name": "x", "code": "missing"}),
    config({"name": "x", "code": "supertrend_current", "params": None}),
    config({"name": "x", "code": "supertrend_current", "params": {"fee_rate": 0.001}}),
    config({"name": "x", "code": "supertrend_current", "params": {"atr_period": 0}}),
    config({"name": "x", "code": "supertrend_current"}, {"name": "x", "code": "supertrend_current"}),
])
def test_invalid_configuration(bad):
    with pytest.raises(ValueError):
        IndicatorComposition(bad)


def test_confirmation_only_blocks_existing_primary_signals(monkeypatch):
    composition = IndicatorComposition(config({"name": "a", "code": "supertrend_current"},
                                              {"name": "b", "code": "supertrend_current"}))
    index = pd.RangeIndex(4)
    primary = pd.DataFrame({"direction": [1, -1, 1, 0], "position_pct": [0.2] * 4,
                            "leverage": [2] * 4, "risk_pct": [0.01] * 4,
                            "block_reason": ["", "", "", "existing"], "score": [0.7] * 4}, index=index)
    confirmation = pd.DataFrame({"direction": [1, 1, 0, 1]}, index=index)
    monkeypatch.setattr(composition, "calculate", lambda *a, **k: {"a": primary, "b": confirmation})
    result, _ = composition.evaluate(pd.DataFrame(index=index))
    assert result.direction.tolist() == [1, 0, 0, 0]
    assert result.block_reason.tolist() == ["", "composition_confirmation", "composition_confirmation", "existing"]
    assert result.position_pct.tolist() == [0.2, 0, 0, 0.2]
    assert primary.direction.tolist() == [1, -1, 1, 0]


def test_components_get_isolated_input(monkeypatch):
    composition = IndicatorComposition(config({"name": "a", "code": "supertrend_current"},
                                              {"name": "b", "code": "supertrend_current"}))
    first, second = composition.components
    original = type(first.indicator).prepare
    calls = []

    def prepare(self, data, params, ctx=None):
        calls.append("polluted" in data.columns)
        data["polluted"] = 1
        return original(self, data, params, ctx)

    monkeypatch.setattr(type(first.indicator), "prepare", prepare)
    composition.calculate(frame())
    assert calls == [False, False]

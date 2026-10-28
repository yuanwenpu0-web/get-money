"""Cycle-local feature reuse must preserve the evaluation result."""

import numpy as np
import pandas as pd
import pytest

from app.models import Signal
from app.strategy import engine


STRATEGY = "absorption_reversal"


@pytest.fixture
def candle_frame(monkeypatch):
    index = pd.date_range("2026-01-01", periods=400, freq="1h", tz="UTC")
    close = 100 + np.arange(400) * .04 + np.sin(np.arange(400) / 8)
    frame = pd.DataFrame(
        dict(open=close - .02, high=close + .3, low=close - .3,
             close=close, volume=100 + np.arange(400) % 10), index=index,
    )
    monkeypatch.setattr(engine, "now_ms", lambda: int(index[-1].timestamp() * 1000) + 3_600_000)
    monkeypatch.setattr(engine.MarketDataService, "load_frame", lambda *a, **k: frame)
    return frame


def test_cycle_reuses_features_without_changing_saved_signals(db, monkeypatch, candle_frame):
    expected = engine.StrategyEngine(db).evaluate("BTCUSDT", "1h", STRATEGY).to_dict()
    calls = dict(features=0, candles=0, state=0)

    def counted(name, original):
        def run(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)
        return run

    monkeypatch.setattr(engine, "build_features", counted("features", engine.build_features))
    monkeypatch.setattr(engine, "all_state", counted("state", engine.all_state))
    monkeypatch.setattr(
        engine.MarketDataService, "load_frame",
        counted("candles", engine.MarketDataService.load_frame),
    )

    evaluator = engine.StrategyEngine(db, cycle_snapshot=True)
    first = evaluator.evaluate("BTCUSDT", "1h", STRATEGY, save=True)
    second = evaluator.evaluate("BTCUSDT", "1h", STRATEGY, save=True)
    assert first.to_dict() == second.to_dict() == expected
    assert db.query(Signal).count() == 2
    # 同币种同周期只计算一次特征（K 线会因 lower-tf delta 探测再取一次，故不比对）
    assert calls["features"] == 1
    assert calls["state"] == 1

    # 换币种必须重新加载，缓存按当前序列单槽保留
    evaluator.evaluate("ETHUSDT", "1h", STRATEGY)
    assert calls["features"] == 2
    assert len(evaluator._frames) == 1

    # 新一轮巡检创建新实例，必须重新取数
    engine.StrategyEngine(db, cycle_snapshot=True).evaluate("BTCUSDT", "1h", STRATEGY)
    assert calls["features"] == 3


def test_default_engine_refreshes_data_every_call(db, monkeypatch, candle_frame):
    calls = []
    original = engine.build_features

    def build(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "build_features", build)
    evaluator = engine.StrategyEngine(db)
    first = evaluator.evaluate("BTCUSDT", "1h", STRATEGY)
    candle_frame.iloc[-1, candle_frame.columns.get_loc("close")] += .01
    second = evaluator.evaluate("BTCUSDT", "1h", STRATEGY)
    assert len(calls) == 2
    assert first.price != second.price


def test_engine_requires_explicit_strategy(db, candle_frame):
    """收敛后不再支持 auto 路由：未显式给出策略代码一律不评估。"""
    evaluator = engine.StrategyEngine(db)
    assert evaluator.evaluate("BTCUSDT", "1h", "auto") is None
    assert evaluator.evaluate("BTCUSDT", "1h", None) is None

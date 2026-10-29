"""The ZigZag request must reach its own order broker with Pine defaults."""
import json

import numpy as np
import pandas as pd
import pytest

from app.backtest import runner


@pytest.fixture
def zigzag_market(monkeypatch):
    idx = pd.date_range('2026-01-01', periods=101, freq='15min', tz='UTC')
    close = 100 + 5 * np.sin(np.arange(len(idx)) / 3)
    frame = pd.DataFrame(dict(open=close, high=close + 1, low=close - 1,
                              close=close, volume=100.0), index=idx)
    calls = []
    as_of = int(idx[-1].timestamp() * 1000)  # last bar is not finished

    class StubMarket:
        def __init__(self, db):
            pass

        def load_frame(self, symbol, interval, start_ms, end_ms):
            calls.append((symbol, interval, start_ms, end_ms))
            return frame.copy()

    monkeypatch.setattr(runner, 'MarketDataService', StubMarket)
    monkeypatch.setattr(runner, 'now_ms', lambda: as_of)
    monkeypatch.setattr(runner, 'build_dataset', lambda *a, **kw: pytest.fail('generic dataset path used'))
    return idx, calls, as_of


def test_zigzag_runner_uses_full_closed_window_and_pine_defaults(db, zigzag_market):
    idx, calls, as_of = zigzag_market
    result = runner.run_backtest(db, 'BTCUSDT', '15m', 'simple_zigzag_double', days=2)
    metrics = result['metrics']
    assert calls == [('BTCUSDT', '15m', as_of - 2 * 86_400_000, as_of)]
    assert metrics['split']['test_rows'] == 100
    assert metrics['split']['test_start'] == str(idx[0])
    assert metrics['split']['test_end'] == str(idx[-2])
    assert metrics['initial_cash'] == 1_000_000
    assert metrics['fee_rate'] == 0.001
    assert metrics['slippage'] == 0
    assert metrics['execution'] == 'pine_stop_orders'
    assert metrics['baseline'] is None and metrics['random_baseline'] is None
    assert metrics['validation']['enabled'] is False
    assert len(result['equity_curve']) == 100
    json.dumps(result, allow_nan=False)


def test_zigzag_runner_honors_overrides_and_rejects_generic_optimization(db, zigzag_market):
    result = runner.run_backtest(
        db, 'BTCUSDT', '15m', 'simple_zigzag_double', initial_cash=20000,
        fee=0.002, slippage=0.001, param_overrides={'fixed_lot': True, 'lot_long': 2},
    )
    metrics = result['metrics']
    assert metrics['initial_cash'] == 20000
    assert metrics['fee_rate'] == 0.002 and metrics['slippage'] == 0.001
    assert metrics['parameters']['fixed_lot'] is True
    assert metrics['parameters']['lot_long'] == 2
    assert any('滑点' in text for text in metrics['warnings'])
    with pytest.raises(ValueError, match='不支持通用验证集选参'):
        runner.run_backtest(db, 'BTCUSDT', '15m', 'simple_zigzag_double', optimize=True)

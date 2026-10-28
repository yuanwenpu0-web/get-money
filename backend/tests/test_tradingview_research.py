import pandas as pd
import pytest
from fastapi import HTTPException

from app.api import tradingview_strategies as api
from app.backtest import tradingview_research as research
from app.backtest import tradingview_native as native


def candles():
    index = pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC")
    return pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100.}, index=index)


def test_research_uses_stored_closed_window_and_reports_assumptions(monkeypatch):
    frame = candles()
    monkeypatch.setattr(research.MarketDataService, "load_frame", lambda *a, **k: frame)
    monkeypatch.setattr(research, "now_ms", lambda: int((frame.index[-1] + pd.Timedelta(hours=1)).timestamp() * 1000))
    result = research.run_research(None, "sma_crossover", "BTCUSDT", "1h", 1)
    assert result["metrics"]["bars"] == 24
    assert result["metrics"]["initial_cash"] == 10000
    assert result["metrics"]["total_trades"] == 0
    assert result["metrics"]["end"].startswith("2026-01-01 23:00")
    assert result["assumptions"] and len(result["equity_curve"]) == 24


def test_research_rejects_missing_and_incomplete_candles(monkeypatch):
    frame = candles()
    monkeypatch.setattr(research, "now_ms", lambda: int((frame.index[-1] + pd.Timedelta(hours=1)).timestamp() * 1000))
    monkeypatch.setattr(research.MarketDataService, "load_frame", lambda *a, **k: frame.drop(frame.index[2]))
    with pytest.raises(ValueError, match="不足"):
        research.run_research(None, "sma_crossover", "BTCUSDT", "1h", 1)
    broken = frame.copy()
    broken.index = broken.index.where(broken.index != frame.index[2], frame.index[2] + pd.Timedelta(minutes=1))
    monkeypatch.setattr(research.MarketDataService, "load_frame", lambda *a, **k: broken)
    with pytest.raises(ValueError, match="缺口"):
        research.run_research(None, "sma_crossover", "BTCUSDT", "1h", 1)


def test_api_refuses_unported_source(monkeypatch):
    monkeypatch.setattr(api, "get_tradingview_strategy", lambda _: {"execution_status": "not_migrated"})
    with pytest.raises(HTTPException) as exc:
        api.backtest_tradingview_strategy("id", api.ResearchRequest(symbol="BTCUSDT"), None)
    assert exc.value.status_code == 400


def test_flat_only_strategy_closes_without_reversing(monkeypatch):
    bars = [{"time": i, "open": 100., "high": 100., "low": 100., "close": 100.} for i in range(5)]
    signals = [
        {"long": True, "short": False, "close_long": False, "close_short": False},
        {"long": False, "short": True, "close_long": True, "close_short": False},
        {"long": False, "short": False, "close_long": False, "close_short": False},
        {"long": False, "short": True, "close_long": False, "close_short": False},
        {"long": False, "short": False, "close_long": False, "close_short": False},
    ]
    monkeypatch.setattr(native, "tradingview_signals", lambda *args: signals)
    result = native.run_tradingview_backtest(bars, "cci_strategy")
    assert result["initial_capital"] == 1000 and result["position_pct"] == 1
    assert result["commission_rate"] == 0
    assert len(result["trades"]) == 1
    assert result["trades"][0]["entry_time"] == 1 and result["trades"][0]["exit_time"] == 2
    assert result["open_position"]["entry_time"] == 4
    assert result["open_position"]["direction"] == -1

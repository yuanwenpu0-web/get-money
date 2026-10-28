import pandas as pd
import pytest

from app.pine.backtest import run_backtest


def candles():
    return pd.DataFrame({
        "open_time": [0, 60000, 120000, 180000],
        "open": [100., 110., 120., 115.],
        "high": [102., 112., 122., 117.],
        "low": [98., 108., 118., 113.],
        "close": [101., 111., 121., 116.],
        "volume": [1., 1., 1., 1.],
    })


def decisions(**overrides):
    data = {key: [False] * 4 for key in ("long_entry", "short_entry", "long_close", "short_close")}
    data.update(overrides)
    return pd.DataFrame(data)


def test_next_open_reversal_costs_and_pending_decision():
    signals = decisions(long_entry=[True, False, False, False],
                        short_entry=[False, True, False, False],
                        short_close=[False, False, False, True])
    result = run_backtest(candles(), signals, quantity=2, fee_rate=0.001, slippage=0.01)
    trade = result["trades"][0]
    assert trade["entry_time"] == 60000
    assert trade["exit_time"] == 120000
    assert trade["entry_price"] == pytest.approx(111.1)
    assert trade["exit_price"] == pytest.approx(118.8)
    assert trade["pnl"] == pytest.approx((118.8 - 111.1) * 2 - (111.1 + 118.8) * 2 * 0.001)
    assert result["open_position"]["side"] == -1
    assert result["pending_signals"]["short_close"]
    assert result["final_equity"] == pytest.approx(result["balance"] + (116 - 118.8) * 2 * -1)


def test_close_decision_waits_for_next_open_and_position_is_not_forced_closed():
    signals = decisions(long_entry=[True, False, False, False], long_close=[False, True, False, False])
    result = run_backtest(candles(), signals, fee_rate=0)
    assert result["trades"][0]["exit_time"] == 120000
    assert result["final_equity"] == 10010
    assert result["open_position"] is None


def test_gap_and_ambiguous_or_misaligned_signals_fail():
    data = candles()
    data.loc[2, "open_time"] += 1
    with pytest.raises(ValueError, match="continuous"):
        run_backtest(data, decisions())
    with pytest.raises(ValueError, match="simultaneous"):
        run_backtest(candles(), decisions(long_entry=[True, False, False, False], short_entry=[True, False, False, False]))
    with pytest.raises(ValueError, match="same non-empty index"):
        run_backtest(candles(), decisions().set_axis([1, 2, 3, 4]))


def test_multiple_order_decisions_on_one_bar_are_rejected():
    with pytest.raises(ValueError, match="simultaneous"):
        run_backtest(candles(), decisions(long_entry=[True, False, False, False], long_close=[True, False, False, False]))

import pandas as pd
import pytest

from app.backtest.zigzag_execution import simulate_zigzag


def replay(ohlc, rows, **overrides):
    idx = pd.date_range('2026-01-01', periods=len(ohlc), freq='15min', tz='UTC')
    data = pd.DataFrame(ohlc, columns=['open', 'high', 'low', 'close'], index=idx)
    signals = pd.DataFrame(rows, index=idx)
    params = dict(use_long=True, use_short=True, fixed_lot=True, lot_long=1.0,
                  lot_short=1.0, timestamp_end=4102444800000)
    params.update(overrides)
    equity, trades = simulate_zigzag(data, signals, params, 10000, 0.001, 0, '15m')
    return equity, trades, idx


def row(long_level=105., short_level=95., long=True, short=True):
    return dict(zz_long_level=long_level, zz_short_level=short_level,
                zz_long_entry=long, zz_short_entry=short)


def test_stops_follow_ohlc_path_and_reverse_same_bar():
    equity, trades, idx = replay(
        [(100, 101, 99, 100), (100, 108, 94, 106)], [row(), row()])
    assert len(trades) == 1
    assert trades[0]['side'] == 'short'
    assert trades[0]['entry_price'] == 95
    assert trades[0]['exit_price'] == 105
    assert trades[0]['status'] == 'signal_reverse'
    assert trades[0]['fees'] == pytest.approx(0.2)
    assert equity.iloc[-1] == pytest.approx(10000 - 0.095 - 10 - 0.105 - 0.105 + 1)
    assert trades[0]['entry_time'] == trades[0]['exit_time'] == str(idx[1])


def test_pending_order_survives_false_condition_and_fills_gap_at_open():
    equity, trades, _ = replay(
        [(100, 101, 99, 100), (100, 103, 99, 101), (110, 111, 109, 110)],
        [row(short=False), row(long=False, short=False), row(long=False, short=False)],
        use_short=False)
    assert trades == []  # still open: no synthetic close at data end
    assert equity.iloc[-1] == pytest.approx(10000 - 0.110)
    assert equity.attrs['open_position'] == {
        'side': 'long', 'qty': 1.0, 'entry_price': 110.0,
        'entry_time': str(equity.index[-1]), 'entry_fee': 0.110, 'unrealized_pnl': 0.0,
    }


def test_single_direction_exit_obeys_price_path_across_bars():
    equity, trades, idx = replay(
        [(100, 101, 99, 100), (100, 106, 94, 101), (100, 101, 93, 95)],
        [row(short=False), row(short=False), row(short=False)], use_short=False)
    assert len(trades) == 1
    assert trades[0]['entry_price'] == 105
    assert trades[0]['exit_price'] == 95
    assert trades[0]['exit_time'] == str(idx[2])
    assert trades[0]['status'] == 'stop_loss'
    assert equity.iloc[-1] == pytest.approx(10000 - 10 - .2)
    assert equity.attrs['open_position'] is None


def test_stop_attached_to_pending_entry_can_fill_on_same_bar():
    equity, trades, idx = replay(
        [(100, 101, 99, 100), (100, 106, 90, 101)],
        [row(short=False), row(short=False)], use_short=False)
    assert len(trades) == 1
    assert trades[0]['entry_price'] == 105
    assert trades[0]['exit_price'] == 95
    assert trades[0]['entry_time'] == trades[0]['exit_time'] == str(idx[1])
    assert equity.attrs['open_position'] is None


def test_pending_entry_and_attached_exit_update_before_fill():
    _, trades, idx = replay(
        [(100, 101, 99, 100), (100, 106, 99, 105), (105, 111, 97, 100)],
        [row(long_level=110, short_level=95, short=False),
         row(long_level=110, short_level=98, short=False), row(short=False)], use_short=False)
    assert len(trades) == 1
    assert trades[0]['entry_price'] == 110
    assert trades[0]['exit_price'] == 98
    assert trades[0]['entry_time'] == trades[0]['exit_time'] == str(idx[2])


def test_marketable_stop_is_filled_at_next_open():
    _, trades, idx = replay(
        [(100, 101, 99, 100), (104, 110, 103, 105), (90, 91, 89, 90)],
        [row(long_level=99, short=False), row(short=False), row(short=False)],
        use_short=False)
    assert trades[0]['entry_price'] == 104
    assert trades[0]['exit_price'] == 90
    assert trades[0]['exit_time'] == str(idx[2])


def test_end_time_close_all_occurs_at_next_open():
    idx = pd.date_range('2026-01-01', periods=4, freq='15min', tz='UTC')
    _, trades, _ = replay(
        [(100, 101, 99, 100), (100, 106, 99, 105), (106, 108, 105, 107), (108, 110, 107, 109)],
        [row(short=False)] * 4, use_short=False,
        timestamp_end=int(idx[1].timestamp() * 1000))
    assert trades[0]['status'] == 'time_end'
    assert trades[0]['exit_price'] == 108
    assert trades[0]['exit_time'] == str(idx[3])


def test_na_lot_initially_uses_equity_then_explicit_lot_recomputed_on_position_change():
    equity, trades, _ = replay(
        [(100, 101, 99, 100), (100, 106, 99, 105), (105, 106, 94, 95)],
        [row(), row(), row()], fixed_lot=False, lot_long=50, lot_short=50)
    assert trades[0]['size'] == pytest.approx(10000 / 105)
    assert trades[0]['exit_price'] == 95
    assert equity.iloc[-1] < 10000

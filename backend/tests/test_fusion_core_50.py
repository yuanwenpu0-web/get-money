import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('fusion_core_50', Path(__file__).parents[1] / 'scripts/fusion_core_50.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def fixture_frame():
    frame = pd.DataFrame({'open': [100, 100, 100, 100], 'high': [101, 110, 110, 101],
                          'low': [99, 90, 90, 99], 'close': [100, 100, 100, 100]},
                         index=pd.date_range('2026-01-01', periods=4, freq='h', tz='UTC'))
    sig = pd.DataFrame({'long': [True, False, False, False], 'short': False,
                        'atr': 2., 'ema': 90., 'osc': 60.}, index=frame.index)
    return frame, sig


def test_bracket_activates_after_first_position_bar():
    frame, sig = fixture_frame()
    result = mod.simulate(frame, sig, start_index=0)
    trade = result['trades'][0]
    assert trade['entry_ms'] == mod.to_ms(frame.index)[1]
    assert trade['exit_ms'] == mod.to_ms(frame.index)[2]
    assert trade['reason'] == 'both_stop'
    assert trade['exit_price'] == pytest.approx(96.4)
    assert trade['pnl'] == pytest.approx(-3.6 - (100 + 96.4) * mod.FEE)


def test_gap_stop_fills_at_open_and_no_future_signal():
    frame, sig = fixture_frame()
    frame.iloc[2] = [95, 100, 94, 99]
    result = mod.simulate(frame, sig, start_index=0)
    assert result['trades'][0]['reason'] == 'stop_gap'
    assert result['trades'][0]['exit_price'] == 95
    sig.loc[sig.index[0], 'long'] = False
    sig.loc[sig.index[-1], 'long'] = True
    result = mod.simulate(frame, sig, start_index=0)
    assert not result['trades'] and result['open_position'] is None


def test_indicators_are_prefix_causal():
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(size=400))
    frame = pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1,
                          'close': close, 'volume': rng.uniform(1, 100, len(close))})
    full = mod.signals(frame)
    prefix = mod.signals(frame.iloc[:250])
    pd.testing.assert_frame_equal(full.iloc[:250], prefix)


def test_normalized_groups_and_truncated_coverage():
    trade = {'side': -1, 'entry_ms': mod.SPLIT - mod.BAR, 'exit_ms': mod.SPLIT,
             'entry_price': 100., 'exit_price': 90., 'pnl': 10 - 190 * mod.FEE}
    payload = {'results': [{'symbol': 'TEST', 'trades': [trade],
                            'last_open_utc': mod.iso(mod.SPLIT + 9 * mod.BAR),
                            'first_missing_utc': mod.iso(mod.SPLIT + 10 * mod.BAR)}]}
    analysis = mod.add_analysis(payload)['normalized_analysis']
    assert trade['net_return_on_entry_notional'] == pytest.approx(0.1 - 1.9 * mod.FEE)
    assert analysis['post_cross_boundary']['trades'] == 1
    assert analysis['post_entry_and_exit']['trades'] == 0
    assert analysis['post_by_exit_and_side']['short']['mean_net_return'] == pytest.approx(trade['pnl'] / 100)
    assert analysis['coverage']['symbols'][0]['post_bars'] == 10

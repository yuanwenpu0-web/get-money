import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from test_minute_research import candles

PATH = Path(__file__).resolve().parents[2] / 'scripts' / 'research_binary_15m_rolling.py'
spec = importlib.util.spec_from_file_location('research_binary_15m_rolling', PATH)
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def test_features_are_causal_and_segments_do_not_borrow_history():
    frame = candles(2000)
    changed = frame.copy()
    changed.loc[1001:, ['volume', 'taker_buy_volume', 'close', 'high', 'low']] *= 2
    pd.testing.assert_frame_equal(research.extra_features(frame).iloc[:1001], research.extra_features(changed).iloc[:1001])
    segment = frame.iloc[1000:].reset_index(drop=True)
    x, _, times, _, _ = research.prepare_segment(segment)
    assert times[0] >= segment.open_time.iloc[0] + 240 * 60000
    assert len(x) == len(times)


def test_rolling_split_labels_and_test_mutation_do_not_change_first_fold_selection():
    frame = candles(24000)
    original = research.experiment(frame, trees=3, threads=1)
    changed = frame.copy()
    test_start = len(frame) * 55 // 100
    changed.loc[test_start:, ['open', 'high', 'low', 'close']] *= 1.07
    updated = research.experiment(changed, trees=3, threads=1)
    f1, f2 = original['folds'][0], updated['folds'][0]
    for left, right in [('train', 'validation'), ('validation', 'test')]:
        assert f1['segments'][left]['last_label_close_ms'] < f1['segments'][right]['first_open_ms']
    for name in research.VARIANTS:
        a, b = f1['results'][name], f2['results'][name]
        assert a['threshold'] == b['threshold']
        assert a['validation_selected'] == b['validation_selected']
        assert a['test_all']['total'] == f1['results']['baseline']['test_all']['total']
    for fold in original['folds']:
        assert len({r['test_all']['total'] for r in fold['results'].values()}) == 1
    for name in research.VARIANTS:
        result = original['aggregate'][name]['test_filtered']
        assert result['correct'] == sum(f['results'][name]['test_filtered']['correct'] for f in original['folds'])
        assert np.isclose(result['coverage'], result['selected'] / result['total'])

import importlib.util
import json
from pathlib import Path

import numpy as np

from test_minute_research import candles

PATH = Path(__file__).resolve().parents[2] / 'scripts' / 'research_binary_15m.py'
spec = importlib.util.spec_from_file_location('research_binary_15m', PATH)
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


def test_threshold_uses_validation_and_respects_minimum_coverage():
    y = np.array([1, 0, 1, 0])
    p = np.array([.9, .1, .6, .6])
    threshold = research.select_threshold(y, p)
    assert threshold == .8
    assert np.mean(np.maximum(p, 1-p) >= threshold) == .5


def test_pipeline_purges_time_boundaries_and_saves_candidate(tmp_path):
    report = research.experiment(candles(15000), tmp_path / 'candidate', trees=3, threads=1)
    split = report['split']
    assert split['train']['last_open_ms'] + 15 * 60000 < split['validation']['first_open_ms']
    assert split['validation']['last_open_ms'] + 15 * 60000 < split['test']['first_open_ms']
    assert report['test']['binary']['selected'] == split['test']['rows'] - split['test']['flat_rows']
    assert report['validation_selected']['coverage'] >= .5
    assert (tmp_path / 'candidate' / 'candidate.joblib').exists()
    assert json.loads((tmp_path / 'candidate' / 'report.json').read_text(encoding='utf-8')) == report


def test_test_period_changes_do_not_change_validation_threshold(tmp_path):
    frame = candles(15000)
    original = research.experiment(frame, tmp_path / 'original', trees=3, threads=1)
    changed = frame.copy()
    changed.loc[14000:, ['open', 'high', 'low', 'close']] *= 1.1
    updated = research.experiment(changed, tmp_path / 'updated', trees=3, threads=1)
    assert original['confidence_threshold'] == updated['confidence_threshold']
    assert original['validation_selected'] == updated['validation_selected']

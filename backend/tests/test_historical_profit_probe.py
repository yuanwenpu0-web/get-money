import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('historical_profit_probe', SCRIPTS / 'probe_historical_profit_exhaustion.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_decision_uses_next_open_and_excludes_exit_bar(monkeypatch):
    idx = pd.date_range('2026-01-01', periods=30, freq='15min', tz='UTC')
    frame = pd.DataFrame({'open':100. + np.arange(30), 'close':101. + np.arange(30)}, index=idx)
    feat = pd.DataFrame({'rsi':50., 'change':-2., 'top':40., 'bot':20.}, index=idx)
    monkeypatch.setattr(probe, 'START', idx[0])
    monkeypatch.setattr(probe, 'TRAIN_END', idx[20])
    monkeypatch.setattr(probe, 'VALID_END', idx[29])
    monkeypatch.setattr(probe, 'LOOKBACK', 0)
    positions = [{'entry_time':str(idx[1]),'exit_time':str(idx[3]),'side':'long','entry_price':100.}]
    result = probe.label_and_candidates(frame,feat,positions,'BTCUSDT')
    assert len(result)==1
    assert result[0]['time']==idx[1].isoformat()
    assert result[0]['y']==frame.open.iloc[10]/frame.open.iloc[2]-1
    assert result[0]['rsi_pullback']==2.
    positions[0]['exit_time']=str(idx[1])
    assert probe.label_and_candidates(frame,feat,positions,'BTCUSDT')==[]


def test_train_label_cannot_cross_split(monkeypatch):
    idx = pd.date_range('2026-01-01', periods=40, freq='15min', tz='UTC')
    frame = pd.DataFrame({'open':100. + np.arange(40), 'close':101. + np.arange(40)}, index=idx)
    feat = pd.DataFrame({'rsi':50., 'change':1., 'top':40., 'bot':20.}, index=idx)
    monkeypatch.setattr(probe, 'START', idx[0])
    monkeypatch.setattr(probe, 'TRAIN_END', idx[10])
    monkeypatch.setattr(probe, 'VALID_END', idx[35])
    monkeypatch.setattr(probe, 'LOOKBACK', 0)
    position = {'entry_time':str(idx[2]),'exit_time':str(idx[20]),'side':'long','entry_price':100.}
    assert probe.label_and_candidates(frame,feat,[position],'BTCUSDT')==[]

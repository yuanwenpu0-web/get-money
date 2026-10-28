import pandas as pd
import pytest

from app.pine.paper import append_candles, create_account, read_account


SOURCE = '''//@version=6
strategy("paper")
if close > 105
    strategy.entry("L", strategy.long)
if close < 100
    strategy.close("L")
'''


def bars():
    return pd.DataFrame({
        "open_time": [0, 60000, 120000, 180000],
        "open": [100., 110., 120., 115.],
        "high": [102., 112., 122., 117.],
        "low": [98., 108., 118., 95.],
        "close": [101., 111., 121., 96.],
        "volume": [1., 1., 1., 1.],
    })


def test_paper_account_recovers_and_repeated_bars_are_idempotent(tmp_path):
    path = tmp_path / "account.json"
    create_account(path, SOURCE, {"fee_rate": 0.0})
    initial = append_candles(path, bars().iloc[:3])
    assert initial["result"]["open_position"]["entry_time"] == 120000
    assert read_account(path) == initial
    assert append_candles(path, bars().iloc[:3]) == initial
    updated = append_candles(path, bars().iloc[2:])
    assert len(updated["candles"]) == 4
    assert updated["result"]["pending_signals"]["long_close"] is True
    assert updated["result"]["open_position"] is not None


def test_bad_updates_leave_saved_account_untouched(tmp_path):
    path = tmp_path / "account.json"
    create_account(path, SOURCE)
    state = append_candles(path, bars().iloc[:2])
    changed = bars().iloc[[1]].copy()
    changed.loc[1, "close"] = 110.
    with pytest.raises(ValueError, match="changed"):
        append_candles(path, changed)
    with pytest.raises(ValueError, match="continuous"):
        append_candles(path, bars().iloc[[3]])
    assert read_account(path) == state
    with pytest.raises(FileExistsError):
        create_account(path, SOURCE)


def test_open_candle_is_rejected_without_changing_account(tmp_path, monkeypatch):
    path = tmp_path / "account.json"
    state = create_account(path, SOURCE)
    monkeypatch.setattr("app.pine.paper.time.time", lambda: 30.0)
    with pytest.raises(ValueError, match="unconfirmed"):
        append_candles(path, bars().iloc[:1])
    assert read_account(path) == state
    monkeypatch.setattr("app.pine.paper.time.time", lambda: 60.0)
    assert len(append_candles(path, bars().iloc[:1])["candles"]) == 1

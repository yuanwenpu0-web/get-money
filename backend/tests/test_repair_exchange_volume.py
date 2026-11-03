import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("repair_exchange_volume", scripts / "repair_exchange_volume.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def test_rebuild_requires_other_fields_and_trade_ids_to_match():
    stamp = repair.TIMESTAMP
    candle = dict(open_time=stamp, close_time=stamp + 59999, open=10., high=11., low=10., close=11.,
                  volume=1., quote_volume=31., trades=3, taker_buy_volume=2., taker_buy_quote_volume=20.)
    trades = [["0", "10", "2", "10", "11", str(stamp + 1), "false"],
              ["1", "11", "1", "12", "12", str(stamp + 2), "true"]]
    assert repair.rebuild(candle, trades)["volume"] == 3.
    with pytest.raises(ValueError):
        repair.rebuild({**candle, "quote_volume": 33.}, trades)
    with pytest.raises(ValueError):
        repair.rebuild(candle, [trades[0], [*trades[1][:3], "13", "13", *trades[1][5:]]])

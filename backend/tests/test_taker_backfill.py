import importlib.util
from pathlib import Path

from sqlalchemy import create_engine, select

from app.db import Base
from app.market.binance import parse_klines
from app.models import Kline

spec = importlib.util.spec_from_file_location("backfill_taker_data", Path(__file__).resolve().parents[2] / "scripts" / "backfill_taker_data.py")
backfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill)


def test_archive_backfill_preserves_existing_and_refuses_mismatch():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[Kline.__table__])
    raw = [[i * 60000, "10", "12", "9", "11", "2", i * 60000 + 59999, "20", 3, "1", "10", "0"] for i in range(3)]
    rows = parse_klines(raw)
    with engine.begin() as conn:
        conn.execute(Kline.__table__.insert(), [
            {"symbol": "BTCUSDT", "interval": "1m", **{**rows[0], "taker_buy_volume": None, "taker_buy_quote_volume": None}},
            {"symbol": "BTCUSDT", "interval": "1m", **{**rows[1], "close": 10.5, "taker_buy_volume": None, "taker_buy_quote_volume": None}},
        ])
    assert backfill.apply_rows(engine, "BTCUSDT", rows, 0, 120000) == {"inserted": 1, "updated": 1, "mismatched": 1, "invalid_times": []}
    with engine.connect() as conn:
        stored = conn.execute(select(Kline).order_by(Kline.open_time)).all()
    assert stored[0].taker_buy_volume == 1
    assert stored[1].close == 10.5 and stored[1].taker_buy_volume is None
    assert stored[2].taker_buy_quote_volume == 10
    assert backfill.apply_rows(engine, "BTCUSDT", rows, 0, 120000) == {"inserted": 0, "updated": 0, "mismatched": 1, "invalid_times": []}


def test_invalid_candle_is_quarantined_without_losing_valid_rows():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine, tables=[Kline.__table__])
    raw = [[i * 60000, "10", "12", "9", "11", "2", i * 60000 + 59999, "20", 3, "1", "10", "0"] for i in range(2)]
    rows = parse_klines(raw)
    rows[0]["taker_buy_volume"] = 3
    result = backfill.apply_rows(engine, "BTCUSDT", rows, 0, 60000)
    assert result["invalid_times"] == [0] and result["inserted"] == 1
    with engine.connect() as conn:
        assert [r.open_time for r in conn.execute(select(Kline)).all()] == [60000]

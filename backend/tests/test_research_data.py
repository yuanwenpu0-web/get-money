import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from app.ml.research_data import MINUTE, aggregate_candles, load_candles, validate_candles


def candles(count=10):
    return pd.DataFrame([dict(open_time=i * MINUTE, close_time=(i + 1) * MINUTE - 1,
                              open=10., high=12., low=9., close=11., volume=2.,
                              quote_volume=20., trades=3, taker_buy_volume=1.,
                              taker_buy_quote_volume=10.) for i in range(count)])


def test_quality_excludes_unfinished_and_detects_bad_rows_and_missing_fields():
    data = candles()
    _, report = validate_candles(data, "BTCUSDT", 9 * MINUTE + 1)
    assert report["passed"] and report["unfinished_rows"] == 1
    data.loc[2, "taker_buy_volume"] = 3
    data.loc[3, "high"] = 1
    data = pd.concat([data.drop(index=4), data.iloc[[2]]])
    _, report = validate_candles(data.drop(columns="taker_buy_quote_volume"), "BTCUSDT", 11 * MINUTE)
    assert not report["passed"]
    assert (report["invalid_rows"], report["duplicate_count"], report["missing_minutes"]) == (3, 1, 1)
    assert report["missing_fields"] == ["taker_buy_quote_volume"]


def test_bad_or_future_timestamps_are_not_silently_discarded():
    data = candles()
    data.loc[0, "open_time"] = np.nan
    data.loc[1, "open_time"] = 100 * MINUTE
    _, report = validate_candles(data, "BTCUSDT", 20 * MINUTE)
    assert not report["passed"] and report["invalid_rows"] == 2


def test_aggregation_requires_entire_closed_bucket():
    data = candles()
    result = aggregate_candles(data, 5, 7 * MINUTE)
    assert result.open_time.tolist() == [0]
    assert result.iloc[0]["volume"] == 10
    assert aggregate_candles(data.drop(index=2), 5, 10 * MINUTE).open_time.tolist() == [5 * MINUTE]
    assert aggregate_candles(pd.concat([data, data.iloc[[2]]]), 5, 10 * MINUTE).open_time.tolist() == [5 * MINUTE]


def test_loader_uses_optional_database_columns():
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE klines (symbol TEXT, interval TEXT, open_time BIGINT, close_time BIGINT, open FLOAT, high FLOAT, low FLOAT, close FLOAT, volume FLOAT, quote_volume FLOAT, trades INTEGER)"))
        conn.execute(text("INSERT INTO klines VALUES ('BTCUSDT','1m',0,59999,10,12,9,11,2,20,3)"))
    frame = load_candles(engine, "BTCUSDT")
    _, report = validate_candles(frame, "BTCUSDT", MINUTE)
    assert len(frame) == 1 and not report["passed"]
    assert report["missing_fields"] == ["taker_buy_quote_volume", "taker_buy_volume"]

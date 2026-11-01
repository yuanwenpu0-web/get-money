import json
import math
import statistics
import time

import pytest

from app.models import FundingRate, Kline, Sentiment, SymbolPool
from app.strategy.pools import _pool_metrics, pool_symbols, resolve_members, validate_pool


def test_latest_multidimensional_filters_exclude_missing_and_stale(db):
    now = int(time.time() * 1000)
    db.add_all([
        FundingRate(symbol='AAAUSDT', funding_time=now - 1000, funding_rate=0),
        FundingRate(symbol='AAAUSDT', funding_time=now - 2000, funding_rate=0.2),
        FundingRate(symbol='BBBUSDT', funding_time=now - 1000, funding_rate=0),
        FundingRate(symbol='CCCUSDT', funding_time=now - 90000000, funding_rate=0),
        Sentiment(symbol='AAAUSDT', ts=now - 1000, oi_value=100, long_short_ratio=1.2),
        Sentiment(symbol='BBBUSDT', ts=now - 1000, oi_value=0, long_short_ratio=1.2),
        Sentiment(symbol='CCCUSDT', ts=now - 1000, oi_value=100, long_short_ratio=1.2),
    ])
    db.commit()
    pool = SymbolPool(kind='rule', symbols=json.dumps(['AAAUSDT', 'BBBUSDT', 'CCCUSDT', 'DDDUSDT']), rule=json.dumps({
        'base': 'static', 'filters': {'funding_rate': {'min': 0, 'max': 0}, 'oi_value': {'min': 50}}, 'max_age_hours': 24,
    }))
    assert pool_symbols(db, pool) == ['AAAUSDT']
    assert pool_symbols(db, pool, allowed=['BBBUSDT']) == []


@pytest.mark.parametrize('rule', [
    {'filters': {'unknown': {'min': 1}}},
    {'filters': {'funding_rate': {'min': 2, 'max': 1}}},
    {'filters': {'funding_rate': {'min': float('nan')}}},
    {'filters': {'funding_rate': {}}},
    {'max_age_hours': 0},
    {'volatility_window_hours': 1},
    {'volatility_window_hours': 2.5},
    {'volatility_max_age_hours': 0},
    {'filters': {'volatility_1h': {'min': -1}}},
    {'filters': {'quote_volume_24h': {'max': -1}}},
])
def test_invalid_filters_rejected(rule):
    with pytest.raises(ValueError):
        validate_pool('rule', [], rule)


def test_volatility_uses_latest_closed_contiguous_candles(db):
    hour = 3600000
    current = int(time.time() * 1000) // hour * hour
    prices = [100, 110, 99, 108]
    for i, price in enumerate(prices):
        start = current - (4 - i) * hour
        db.add(Kline(symbol='AAAUSDT', interval='1h', open_time=start, close_time=start + hour - 1,
                     open=price, high=price, low=price, close=price))
    db.add(Kline(symbol='AAAUSDT', interval='1h', open_time=current, close_time=current + hour - 1,
                 open=1, high=1, low=1, close=1))
    db.commit()
    rule = {'base': 'static', 'volatility_window_hours': 3, 'volatility_max_age_hours': 2,
            'filters': {'volatility_1h': {'min': 0}}}
    actual = _pool_metrics(db, ['AAAUSDT'], rule)['AAAUSDT']['volatility_1h']
    expected = statistics.pstdev([math.log(b / a) for a, b in zip(prices, prices[1:])])
    assert actual == pytest.approx(expected)
    pool = SymbolPool(kind='rule', symbols='["AAAUSDT"]', rule=json.dumps(rule))
    assert pool_symbols(db, pool) == ['AAAUSDT']
    db.query(Kline).filter(Kline.symbol == 'AAAUSDT', Kline.open_time == current - 2 * hour).delete()
    db.commit()
    assert pool_symbols(db, pool) == []


def test_volatility_excludes_stale_invalid_and_short_history(db):
    hour = 3600000
    current = int(time.time() * 1000) // hour * hour
    for symbol, offsets, price in [('STALE', [6, 5, 4], 10), ('SHORT', [2, 1], 10), ('BAD', [3, 2, 1], 0)]:
        for offset in offsets:
            start = current - offset * hour
            db.add(Kline(symbol=symbol, interval='1h', open_time=start, close_time=start + hour - 1,
                         open=price, high=price, low=price, close=price))
    db.commit()
    rule = {'filters': {'volatility_1h': {'min': 0}}, 'volatility_window_hours': 2, 'volatility_max_age_hours': 2}
    assert _pool_metrics(db, ['STALE', 'SHORT', 'BAD'], rule) == {}


def test_strict_volume_rejects_missing_and_preserves_legacy():
    common = dict(kind='rule', symbols=[], universe=lambda: ['AAA', 'BBB'], volumes=lambda: {'AAA': 20})
    assert resolve_members(rule={'min_quote_volume_usd': 10}, **common) == ['AAA', 'BBB']
    assert resolve_members(rule={'filters': {'quote_volume_24h': {'min': 10}}}, **common) == ['AAA']
    common['volumes'] = lambda: {}
    assert resolve_members(rule={'min_quote_volume_usd': 10}, **common) == ['AAA', 'BBB']
    assert resolve_members(rule={'filters': {'quote_volume_24h': {'min': 10}}}, **common) == []
    common['volumes'] = lambda: {'AAA': 0, 'BBB': float('nan')}
    assert resolve_members(rule={'filters': {'quote_volume_24h': {'max': 100}}}, **common) == []

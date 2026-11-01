import json

from app.models import Kline, SymbolPool, StrategyConfig, StrategyBinding
from app.strategy.defaults import seed_research_objects
from app.strategy.specs.pool_volatility_research import HIGH_CODE, LOW_CODE
from app.strategy.pool_maintenance import HOUR, POOL_CODES, WINDOW_HOURS, refresh_research_pools
from app.trading.scheduler import TradingScheduler


def test_minute_native_gaps_and_symbol_failure_are_isolated(db, monkeypatch):
    from app.strategy import pool_maintenance as maintenance

    monkeypatch.setattr(maintenance, 'WINDOW_HOURS', 3)
    end = 200 * HOUR
    for n in range(102):
        symbol = f'COIN{n}USDT'
        for i in range(4):
            t = end - (3 - i) * HOUR
            price = 100 + ((n * 37) % 102 + 1) * (i % 2) / 100
            db.add(Kline(symbol=symbol, interval='1h', open_time=t, close_time=t + HOUR - 1,
                         open=100, high=102, low=100, close=price,
                         quote_volume=1000 - n))
    db.flush()
    # COIN0 的旧 native 末根不能使用；完整分钟线覆盖该小时。
    db.query(Kline).filter(Kline.symbol == 'COIN0USDT', Kline.open_time == end).delete()
    for minute in range(60):
        t = end - HOUR + minute * 60000
        db.add(Kline(symbol='COIN0USDT', interval='1m', open_time=t, close_time=t + 59999,
                     open=110, high=110, low=110, close=110, quote_volume=10))
    # COIN1 有缺口，且对应 native 是最后一根快照，应被排除。
    db.query(Kline).filter(Kline.symbol == 'COIN1USDT', Kline.open_time == end).delete()
    for minute in range(59):
        t = end - HOUR + minute * 60000
        db.add(Kline(symbol='COIN1USDT', interval='1m', open_time=t, close_time=t + 59999,
                     open=100, high=100, low=100, close=100, quote_volume=10))
    db.commit()
    original = maintenance.MarketDataService._merged_candle_rows

    def fail_one(self, symbol, *args, **kwargs):
        if symbol == 'COIN2USDT':
            raise ValueError('坏币')
        return original(self, symbol, *args, **kwargs)

    monkeypatch.setattr(maintenance.MarketDataService, '_merged_candle_rows', fail_one)
    result = maintenance.research_pool_candidates(db, end + 1000)
    assert result['updated']
    assert result['eligible'] == 100
    members = sum(result['members'].values(), [])
    assert 'COIN1USDT' not in members and 'COIN2USDT' not in members
    assert not maintenance.research_pool_candidates(db, end + 3 * HOUR)['updated']


def test_refresh_preserves_existing_on_insufficient_data(db):
    pool = SymbolPool(code=POOL_CODES[0], name='原池', kind='static', symbols='["OLDUSDT"]', rule='{}')
    db.add(pool)
    db.commit()
    assert not refresh_research_pools(db, 200 * HOUR)['updated']
    assert pool.symbols == '["OLDUSDT"]'
    assert db.query(SymbolPool).count() == 1


def test_99_eligible_symbols_do_not_replace_existing_pools(db, monkeypatch):
    from app.strategy import pool_maintenance as maintenance

    monkeypatch.setattr(maintenance, 'WINDOW_HOURS', 3)
    end = 200 * HOUR
    for code in POOL_CODES:
        db.add(SymbolPool(code=code, name=code, kind='static', symbols='["OLDUSDT"]',
                          rule=json.dumps({'maintenance_owner': maintenance.OWNER})))
    for n in range(99):
        for i in range(4):
            t = end - (3 - i) * HOUR
            db.add(Kline(symbol=f'COIN{n:03}USDT', interval='1h', open_time=t,
                         close_time=t + HOUR - 1, open=100, high=102, low=100,
                         close=100 + i % 2, quote_volume=1000 - n))
    db.commit()
    result = refresh_research_pools(db, end + 1000)
    assert result == {'updated': False, 'reason': '合格行情不足100个', 'eligible': 99}
    assert all(p.symbols == '["OLDUSDT"]' for p in db.query(SymbolPool).all())


def test_refresh_creates_and_updates_without_future_candles(db):
    end = 200 * HOUR
    for n in range(101):
        symbol = f'COIN{n:03}USDT'
        for i in range(WINDOW_HOURS):
            price = 100 + ((n * 37) % 100 + 1) * (i % 2) / 100
            t = end - (WINDOW_HOURS - i) * HOUR
            db.add(Kline(symbol=symbol, interval='1h', open_time=t, close_time=t + HOUR - 1,
                         open=price, high=price, low=price, close=price, quote_volume=1000 - n))
        db.add(Kline(symbol=symbol, interval='1h', open_time=end, close_time=end + HOUR - 1,
                     open=1, high=1, low=1, close=1, quote_volume=1e12))
    db.commit()
    result = refresh_research_pools(db, end + 1000)
    assert result['updated']
    pools = db.query(SymbolPool).order_by(SymbolPool.code).all()
    assert len(pools) == 2
    assert all(p.enabled == 0 for p in pools)
    before = [json.loads(p.symbols) for p in pools]
    assert sum(map(len, before)) == 50
    expected = {f'COIN{n:03}USDT' for n in range(100) if (n * 37) % 100 >= 50}
    assert set(before[0]) == {s for s in expected if int(s[4:7]) < 50}
    assert set(before[1]) == {s for s in expected if int(s[4:7]) >= 50}
    assert all('至少100个' in p.description for p in pools)
    assert refresh_research_pools(db, end + 1000)['updated']
    assert db.query(SymbolPool).count() == 2
    assert [json.loads(p.symbols) for p in pools] == before
    assert not refresh_research_pools(db, end + 3 * HOUR)['updated']
    assert [json.loads(p.symbols) for p in pools] == before


def test_seed_research_objects_is_atomic_idempotent_and_owned(db):
    result = seed_research_objects(db)
    assert result['candidates']['updated'] is False
    db.commit()
    pools = db.query(SymbolPool).filter(SymbolPool.code.in_(POOL_CODES)).all()
    assert len(pools) == 2
    assert all(p.enabled == 0 and json.loads(p.symbols) == [] for p in pools)
    assert all(json.loads(p.rule)['maintenance_status'] == '待维护' for p in pools)
    assert all(json.loads(p.rule)['maintenance_owner'] == 'rolling_high_volatility_v1' for p in pools)
    assert {s.code for s in db.query(StrategyConfig).all() if s.enabled == 0} >= {HIGH_CODE, LOW_CODE}
    assert {(b.strategy_code, b.pool_code, b.interval, b.enabled) for b in db.query(StrategyBinding).all()} >= {
        (HIGH_CODE, POOL_CODES[0], '1h', 0), (LOW_CODE, POOL_CODES[1], '1h', 0)}
    snapshot = [(p.id, p.symbols, p.rule, p.enabled) for p in pools]
    seed_research_objects(db)
    db.commit()
    assert [(p.id, p.symbols, p.rule, p.enabled) for p in pools] == snapshot
    assert db.query(StrategyBinding).filter(StrategyBinding.strategy_code.in_((HIGH_CODE, LOW_CODE))).count() == 2


def test_seed_rejects_foreign_pool_without_writing(db):
    import pytest
    db.add(SymbolPool(code=POOL_CODES[0], name='占用', kind='static', symbols='[]', rule='{}'))
    db.commit()
    with pytest.raises(ValueError):
        seed_research_objects(db)
    db.rollback()
    assert db.query(StrategyConfig).filter(StrategyConfig.code.in_((HIGH_CODE, LOW_CODE))).count() == 0
    assert db.query(StrategyBinding).filter(StrategyBinding.strategy_code.in_((HIGH_CODE, LOW_CODE))).count() == 0


def test_maintenance_independent_of_trading_pause():
    scheduler = TradingScheduler()
    scheduler._maintain_pools = lambda: None
    try:
        scheduler.start_pool_maintenance()
        scheduler.start_pool_maintenance()
        scheduler.apply(False, 60)
        assert scheduler.scheduler.get_job('research_pool_maintenance') is not None
        assert len(scheduler.scheduler.get_jobs()) == 1
    finally:
        scheduler.scheduler.shutdown(wait=True)


def test_maintenance_api_reports_members_and_failed_attempt(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from sqlalchemy.pool import StaticPool
    from app.api.pools import router
    from app.db import Base, get_db
    from app.strategy import pool_maintenance

    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        for code in POOL_CODES:
            session.add(SymbolPool(code=code, name=code, kind='static', symbols='["OLDUSDT"]',
                                   rule=json.dumps({'maintenance_owner': pool_maintenance.OWNER, 'window_end_ms': 123}), enabled=0))
        session.commit()
        app = FastAPI()
        app.include_router(router, prefix='/api')
        app.dependency_overrides[get_db] = lambda: session
        monkeypatch.setattr(pool_maintenance, 'research_pool_candidates',
                            lambda db, now_ms=None: {'updated': False, 'reason': '行情不足', 'eligible': 1})
        client = TestClient(app)
        assert client.post('/api/pools/research/maintenance').json()['reason'] == '行情不足'
        status = client.get('/api/pools/research/maintenance').json()
        assert status['last_attempt']['updated'] is False
        assert status['last_attempt']['attempted_at']
        assert [p['members'] for p in status['pools']] == [['OLDUSDT'], ['OLDUSDT']]
        assert all(p['window_end_ms'] == 123 for p in status['pools'])
        monkeypatch.setattr(pool_maintenance, 'research_pool_candidates', lambda db, now_ms=None: {
            'updated': True, 'eligible': 100, 'window_end_ms': 456,
            'members': {POOL_CODES[0]: ['AAAUSDT'], POOL_CODES[1]: ['BBBUSDT']},
        })
        assert client.post('/api/pools/research/maintenance').json()['updated'] is True
        status = client.get('/api/pools/research/maintenance').json()
        assert [p['members'] for p in status['pools']] == [['AAAUSDT'], ['BBBUSDT']]
        assert all(p['window_end_ms'] == 456 for p in status['pools'])
    engine.dispose()


def test_maintenance_exception_rolls_back_and_records_failure(db, monkeypatch):
    from app.strategy import pool_maintenance
    pool = SymbolPool(code=POOL_CODES[0], name='原池', kind='static', symbols='["OLDUSDT"]', rule='{}')
    db.add(pool)
    db.commit()
    def fail(session):
        pool.symbols = '[]'
        session.flush()
        raise RuntimeError('failure')
    monkeypatch.setattr(pool_maintenance, 'refresh_research_pools', fail)
    result = pool_maintenance.run_research_pool_maintenance(db)
    assert not result['updated']
    assert db.get(SymbolPool, pool.id).symbols == '["OLDUSDT"]'
    assert pool_maintenance.research_pool_status(db)['last_attempt'] == result


def test_maintenance_rejects_overlapping_runs(db):
    from app.strategy import pool_maintenance
    with pool_maintenance._maintenance_lock:
        result = pool_maintenance.run_research_pool_maintenance(db)
    assert not result['updated']
    assert '进行中' in result['reason']
    assert pool_maintenance.research_pool_status(db)['last_attempt'] is None

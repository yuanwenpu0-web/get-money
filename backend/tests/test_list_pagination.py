"""Server pagination uses bounded queries, stable ordering, and full-filter totals."""

import json
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import event

from app.api import backtests, models_api, trading
from app.db import get_db
from app.market.data_service import MarketDataService
from app.models import BacktestRun, JobLog, ModelRecord, Order


CREATED = datetime(2026, 1, 1, tzinfo=timezone.utc)


def seed_models(db, count):
    for index in range(count):
        db.add(ModelRecord(symbol=f"MODEL{index}", interval="1h", algo="lightgbm",
                           path="unused.pkl", features='["unused"]', created_at=CREATED))
    db.commit()


def seed_backtests(db, count):
    for index in range(count):
        db.add(BacktestRun(symbol=f"BACKTEST{index}", interval="1h", strategy="trend_follow",
                           params='{"days": 30}', equity_curve='[{"equity": 10000}]',
                           trades=json.dumps([{"trade": trade} for trade in range(45)]),
                           created_at=CREATED))
    db.commit()


def order(db, symbol="BTCUSDT", status="OPEN", side="LONG", pnl=0, mark=105):
    row = Order(symbol=symbol, interval="1h", side=side, status=status, qty=2,
                entry_price=100, mark_price=mark, realized_pnl=pnl, created_at=CREATED)
    db.add(row)
    return row


def test_model_pages_are_bounded_stable_and_cover_more_than_old_limit(db):
    seed_models(db, 105)
    first = models_api.list_models(db)
    assert (first["total"], first["page"], first["page_size"], first["pages"]) == (105, 1, 20, 6)
    assert [row["id"] for row in first["models"]] == list(range(105, 85, -1))
    pages = [models_api.list_models(db, page=page)["models"] for page in range(1, 7)]
    assert [row["id"] for rows in pages for row in rows] == list(range(105, 0, -1))
    assert len(models_api.list_models(db, page=6)["models"]) == 5


def test_deleted_last_model_page_remains_explicitly_empty(db):
    seed_models(db, 21)
    db.delete(db.get(ModelRecord, 1))
    db.commit()
    result = models_api.list_models(db, page=2)
    assert result["models"] == []
    assert (result["total"], result["page"], result["pages"]) == (20, 2, 1)


def test_order_summary_covers_entire_filtered_set_and_mark_fallback(db):
    order(db, status="CLOSED", pnl=15)
    order(db, status="CLOSED", pnl=-3)
    order(db, side="LONG", mark=105)  # +10
    order(db, side="SHORT", mark=95)  # +10
    order(db, side="LONG", mark=0)  # mark=0 falls back to entry, as serializer does
    order(db, symbol="ETHUSDT", status="CLOSED", pnl=1000)
    order(db, status="CANCELED", pnl=999)
    db.commit()
    result = trading.list_orders(symbol="btcusdt", page=2, page_size=2, db=db)
    assert len(result["orders"]) == 2
    assert (result["total"], result["page"], result["page_size"], result["pages"]) == (6, 2, 2, 3)
    assert result["summary"] == {"open": 3, "closed": 2, "realized_pnl": 12, "unrealized_pnl": 20}
    closed = trading.list_orders(status="closed", symbol="BTCUSDT", page_size=1, db=db)
    assert closed["total"] == 2
    assert closed["summary"] == {"open": 0, "closed": 2, "realized_pnl": 12, "unrealized_pnl": 0}
    assert [row["id"] for row in closed["orders"]] == [2]
    assert [row["id"] for row in trading.list_orders(status="closed", symbol="BTCUSDT", page=2, page_size=1, db=db)["orders"]] == [1]


def test_order_and_log_legacy_limits_are_bounded_and_explicit_page_size_wins(db):
    for _ in range(5):
        order(db)
        db.add(JobLog(job="test", message="test", created_at=CREATED))
    db.commit()
    assert len(trading.list_orders(limit=3, db=db)["orders"]) == 3
    assert len(trading.list_orders(limit=3, page_size=2, db=db)["orders"]) == 2
    assert [row["id"] for row in trading.logs(limit=3, page=2, db=db)["logs"]] == [2, 1]
    assert len(trading.logs(limit=3, page_size=2, db=db)["logs"]) == 2


def test_empty_lists_have_zero_pages_and_summary(db):
    for result, key in [(models_api.list_models(db), "models"),
                        (backtests.list_backtests(db), "backtests"),
                        (trading.list_orders(db=db), "orders"),
                        (trading.logs(db=db), "logs")]:
        assert result[key] == []
        assert result["total"] == result["pages"] == 0
        assert result["page"] == 1
        assert result["page_size"] == 20
    assert trading.list_orders(db=db)["summary"] == {
        "open": 0, "closed": 0, "realized_pnl": 0, "unrealized_pnl": 0,
    }


def test_backtest_history_pages_do_not_load_detail_json(db):
    seed_backtests(db, 45)
    sql = []
    event.listen(db.get_bind(), "before_cursor_execute", lambda conn, cursor, statement, *args: sql.append(statement))
    result = backtests.list_backtests(db, page=2)
    assert result["total"] == 45
    assert [row["id"] for row in result["backtests"]] == list(range(25, 5, -1))
    selects = [statement for statement in sql if not statement.lstrip().startswith("SELECT count(")]
    assert len(selects) == 1
    assert all(column not in selects[0] for column in ("backtest_runs.trades", "backtest_runs.equity_curve", "backtest_runs.params"))
    assert all("trades" not in row and "equity_curve" not in row for row in result["backtests"])


def test_model_list_does_not_fetch_paths_or_unused_feature_json(db):
    seed_models(db, 2)
    sql = []
    event.listen(db.get_bind(), "before_cursor_execute", lambda conn, cursor, statement, *args: sql.append(statement))
    models_api.list_models(db)
    selects = [statement for statement in sql if not statement.lstrip().startswith("SELECT count(")]
    assert len(selects) == 1
    assert "models.path" not in selects[0]
    assert "models.features" not in selects[0]


def test_backtest_details_can_skip_trades_and_fetch_them_in_pages(db):
    seed_backtests(db, 1)
    db.expunge_all()
    sql = []
    event.listen(db.get_bind(), "before_cursor_execute", lambda conn, cursor, statement, *args: sql.append(statement))
    detail = backtests.get_backtest(1, db, include_trades=False)
    assert "trades" not in detail
    assert detail["equity_curve"] == [{"equity": 10000}]
    assert detail["params"] == {"days": 30}
    assert all("backtest_runs.trades" not in statement for statement in sql)
    pages = [backtests.get_backtest_trades(1, db, page=page) for page in (1, 2, 3)]
    assert [len(result["trades"]) for result in pages] == [20, 20, 5]
    assert [row["trade"] for result in pages for row in result["trades"]] == list(range(45))
    assert pages[0]["total"] == 45 and pages[0]["pages"] == 3
    assert backtests.get_backtest_trades(1, db, page=4)["trades"] == []
    assert len(backtests.get_backtest(1, db)["trades"]) == 45  # preserve legacy detail callers


@pytest.mark.parametrize("raw", ["[]", "{}", "null", "broken JSON"])
def test_empty_or_corrupt_backtest_trade_arrays_are_bounded(db, raw):
    db.add(BacktestRun(symbol="BTCUSDT", interval="1h", strategy="trend_follow", trades=raw))
    db.commit()
    result = backtests.get_backtest_trades(1, db)
    assert result["trades"] == []
    assert result["total"] == result["pages"] == 0


def test_missing_backtest_trade_list_preserves_404(db):
    with pytest.raises(HTTPException) as exc:
        backtests.get_backtest_trades(999, db)
    assert exc.value.status_code == 404


def test_dashboard_can_skip_unused_full_market_statistics(db, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Dashboard requested unused full market statistics")

    monkeypatch.setattr(MarketDataService, "stats", unexpected)
    result = trading.dashboard(db, include_market=False)
    assert result["market"] == []
    assert result["portfolio"]["equity"] == 10000


def test_dashboard_keeps_legacy_market_statistics_by_default(db, monkeypatch):
    monkeypatch.setattr(MarketDataService, "stats", lambda *a, **k: [{"symbol": "BTCUSDT", "count": 1}])
    assert trading.dashboard(db)["market"] == [{"symbol": "BTCUSDT", "count": 1}]


@pytest.mark.parametrize("path", ["/trading/orders", "/trading/logs", "/models", "/backtests", "/backtests/1/trades"])
@pytest.mark.parametrize("params", [{"page": 0}, {"page": -1}, {"page_size": 0}, {"page_size": 101}])
def test_pagination_rejects_invalid_query_parameters_before_accessing_database(db, path, params):
    app = FastAPI()
    for router in (trading.router, models_api.router, backtests.router):
        app.include_router(router)
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as client:
        response = client.get(path, params=params)
    assert response.status_code == 422

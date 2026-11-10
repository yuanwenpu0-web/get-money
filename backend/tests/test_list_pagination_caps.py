"""全站列表分页契约：每页上限 15、筛选在服务端、总数用全量口径。

这些用例守的是「列表接口不会因为数据变多而返回全量」，以及分页带来的两个副作用：
1. 前端只能看到当前页，所以「启用条数 / 清空覆盖」这类全量语义必须由后端提供；
2. 图表标记要的是「最近 N 笔」，不该和分页列表共用同一个接口。
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import event

from app.api import bindings as bindings_api
from app.api import indicators as indicators_api
from app.api import market, symbol_params, trading
from app.db import get_db
from app.models import JobLog, Order, Signal, StrategyBinding, SymbolPool
from app.pagination import MAX_PAGE_SIZE, page_metadata


def make_order(db, index=0, status="OPEN", symbol="BTCUSDT"):
    row = Order(
        symbol=symbol, interval="1h", side="LONG", status=status,
        qty=1, entry_price=100, mark_price=100, realized_pnl=0,
        strategy="absorption_reversal",
    )
    db.add(row)
    db.commit()
    return row


def test_orders_markers_are_lightweight_and_bounded(db):
    for index in range(20):
        make_order(db, index, status="CLOSED" if index % 2 else "OPEN")
    result = trading.list_order_markers(db=db, limit=5)
    assert len(result["markers"]) == 5
    # 只回传画标记要用的字段：整条订单序列化会带上参数快照等大字段
    assert set(result["markers"][0]) == {
        "id", "symbol", "side", "status", "created_at", "closed_at", "entry_price", "close_price",
    }
    only_eth = trading.list_order_markers(db=db, symbol="ethusdt")
    assert only_eth["markers"] == []


def test_dashboard_paginates_each_card_with_full_totals(db):
    for index in range(20):
        make_order(db, index)
    for index in range(5):
        db.add(JobLog(job="test", level="info", message=f"log {index}"))
        db.add(Signal(symbol="BTCUSDT", interval="1h", strategy="absorption_reversal",
                      ts=1_700_000_000_000 + index, price=100, direction="LONG"))
    db.commit()

    first = trading.dashboard(db, include_market=False)
    for key, rows in (("positions", "positions"), ("orders", "recent_orders"),
                      ("signals", "recent_signals"), ("logs", "logs")):
        assert first[f"{key}_pagination"]["page_size"] == MAX_PAGE_SIZE
        assert len(first[rows]) <= MAX_PAGE_SIZE
    assert first["positions_pagination"] == page_metadata(20, 1, MAX_PAGE_SIZE)
    assert first["logs_pagination"]["total"] == 5

    second = trading.dashboard(db, include_market=False, positions_page=2)
    assert second["positions_pagination"]["page"] == 2
    assert len(second["positions"]) == 20 - MAX_PAGE_SIZE
    # 翻持仓页不该影响别的卡片
    assert second["logs_pagination"]["page"] == 1


def test_bindings_page_keeps_full_enabled_total(db):
    db.add(SymbolPool(code="p1", name="池一", kind="static", symbols="[]"))
    db.commit()
    # (策略, 池, 周期) 唯一：用不同周期凑出 18 条绑定
    intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
    for index in range(18):
        db.add(StrategyBinding(
            strategy_code="absorption_reversal", pool_code=f"p{index}", interval="1h",
            enabled=1 if index < 17 else 0, priority=index, params="{}",
        ))
    db.commit()

    page2 = bindings_api.list_bindings(db=db, page=2)
    assert page2["total"] == 18
    assert page2["page_size"] == MAX_PAGE_SIZE
    assert len(page2["bindings"]) == 18 - MAX_PAGE_SIZE
    # 第 2 页只有 3 条，但「已启用 17 条」必须是全量口径
    assert page2["enabled_total"] == 17


def test_symbol_params_clear_all_is_not_page_scoped(db, monkeypatch):
    from app.models import SymbolParam
    for index in range(20):
        db.add(SymbolParam(strategy_code="absorption_reversal", symbol=f"S{index:02d}USDT",
                           interval="1h", params="{}"))
    db.commit()
    described = symbol_params.list_symbol_params(db=db, page=2)
    assert described["total"] == 20 and len(described["items"]) == 5

    removed = symbol_params.delete_all_symbol_params(db=db)
    assert removed["removed"] == 20
    assert symbol_params.list_symbol_params(db=db)["total"] == 0


def test_indicator_catalog_filters_and_pages_on_the_server(monkeypatch):
    catalog = [
        {"id": f"id{i}", "title": title, "original_title": "", "description": "", "category": category,
         "collection": "pine-collection", "indicator_type": "趋势方向", "version": None,
         "source": "hidden", "license_text": "hidden"}
        for i, (title, category) in enumerate([("趋势线", "trend"), ("RSI", "momentum"), ("MACD", "momentum")])
    ]
    monkeypatch.setattr(indicators_api, "_catalog", lambda _dir: catalog)

    page = indicators_api.list_indicators(page=1, page_size=2)
    assert page["total"] == 3 and page["pages"] == 2 and page["catalog_total"] == 3
    assert len(page["items"]) == 2
    assert all("source" not in item and "license_text" not in item for item in page["items"])
    assert page["facets"]["categories"] == ["momentum", "trend"]

    filtered = indicators_api.list_indicators(q="macd")
    assert filtered["total"] == 1 and filtered["items"][0]["title"] == "MACD"

    by_type = indicators_api.list_indicators(category="momentum")
    assert by_type["total"] == 2


@pytest.mark.parametrize(
    "path",
    ["/market/sources", "/trading/orders", "/trading/logs", "/backtests",
     "/optimizations", "/indicators", "/bindings", "/symbol-params"],
)
def test_every_list_endpoint_rejects_oversized_pages(path):
    """页大小校验发生在参数解析阶段，因此不碰数据库（内存 SQLite 不能跨 TestClient 线程）。"""
    from app.api import backtests, optimizations

    app = FastAPI()
    for router in (market.router, trading.router, backtests.router, optimizations.router,
                   indicators_api.router, bindings_api.router, symbol_params.router):
        app.include_router(router)
    with TestClient(app) as client:
        assert client.get(path, params={"page_size": MAX_PAGE_SIZE + 1}).status_code == 422
        assert client.get(path, params={"page_size": 0}).status_code == 422


def test_list_queries_stay_bounded(db):
    for index in range(30):
        make_order(db, index)
    statements = []
    event.listen(db.get_bind(), "before_cursor_execute",
                 lambda conn, cursor, statement, *args: statements.append(statement))
    trading.list_orders(db=db, page_size=MAX_PAGE_SIZE)
    # 取行的那条 select 必须带 LIMIT；计数聚合可以没有
    row_selects = [
        s for s in statements
        if s.lstrip().upper().startswith("SELECT") and "FROM orders" in s and "ORDER BY" in s
    ]
    assert row_selects and all("LIMIT" in s.upper() for s in row_selects)

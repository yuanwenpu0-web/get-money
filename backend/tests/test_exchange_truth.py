"""「以币安为准」回归测试：账户覆盖、下单留痕、保护单同步、真实成交回填。

全部网络层为 Fake，不发起真实请求，也不会碰到真实资金。
"""
import json

import pytest

from app.market.futures import BinanceApiError
from app.models import ExchangeEvent, Order
from app.state import all_state, set_state
from app.trading import exchange_gateway as gw
from app.trading import exchange_truth, live, live_sync, portfolio
from app.utils import now_ms

from fakes import FakeFuturesClient, live_order, market_orders, patch_live, signal


class TruthClient(FakeFuturesClient):
    """在订单 Fake 上补齐账户/流水/成交/保护单状态四个只读面。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.protection = {}
        self.fills = []

    def account(self):
        return {
            "totalWalletBalance": "1050.5", "availableBalance": "900.0",
            "totalMarginBalance": "1060.5", "totalUnrealizedProfit": "10.0",
            "totalPositionInitialMargin": "150.0",
            "assets": [{"asset": "USDT", "walletBalance": "1050.5"}],
        }

    def balance(self):
        return [{"asset": "USDT", "balance": "1050.5", "availableBalance": "900.0"}]

    def income(self, start_ms=None, limit=1000, income_type=None):
        return [
            {"incomeType": "COMMISSION", "income": "-1.2", "time": now_ms()},
            {"incomeType": "FUNDING_FEE", "income": "-0.3", "time": now_ms()},
        ]

    def user_trades(self, symbol, start_ms=None, end_ms=None, limit=1000):
        return self.fills

    def get_order(self, symbol, order_id):
        info = self.protection.get(int(order_id))
        if info is not None:
            return {"orderId": order_id, **info}
        return super().get_order(symbol, order_id)

    def get_algo_order(self, symbol, algo_id):
        info = self.protection.get(int(algo_id)) or {}
        return {
            "algoId": algo_id,
            "algoStatus": info.get("algoStatus", info.get("status", "NEW")),
            "triggerPrice": info.get("triggerPrice", info.get("stopPrice", "0")),
            "actualPrice": info.get("actualPrice", info.get("avgPrice", "0")),
        }


def _enable_live(db, monkeypatch, fake):
    patch_live(db, monkeypatch, fake)
    set_state(db, "execution_mode", "live")
    set_state(db, "testnet", True)
    set_state(db, "initial_capital", 1000.0)
    set_state(db, "cash", 1000.0)


def test_submit_records_success_and_exchange_rejection(db, monkeypatch):
    fake = TruthClient()
    _enable_live(db, monkeypatch, fake)
    gw.submit(db, fake, "open", {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET", "quantity": "1"})
    rows = db.query(ExchangeEvent).all()
    assert len(rows) == 1 and rows[0].status in ("filled", "accepted")
    assert rows[0].env == "demo" and rows[0].exchange_order_id

    def rejected(**params):
        raise BinanceApiError("insufficient balance", -2019)

    monkeypatch.setattr(fake, "new_order", rejected)
    with pytest.raises(BinanceApiError):
        gw.submit(db, fake, "open", {"symbol": "BTCUSDT", "side": "BUY", "type": "MARKET"})
    rejection = db.query(ExchangeEvent).order_by(ExchangeEvent.id.desc()).first()
    assert rejection.status == "rejected" and rejection.code == -2019


def test_sync_account_overlays_portfolio_with_binance_balances(db, monkeypatch):
    fake = TruthClient()
    _enable_live(db, monkeypatch, fake)
    acc = exchange_truth.sync_account(db, force=True)
    assert acc["wallet_balance"] == 1050.5 and acc["account_source"] == "account"
    snap = portfolio.portfolio_snapshot(db, update_marks=False)
    assert snap["pnl_source"] == "binance"
    assert snap["equity"] == 1060.5 and snap["cash"] == 900.0
    assert snap["unrealized_pnl"] == 10.0
    assert snap["realized_pnl"] == 50.5  # 钱包余额 − 收益基准，含手续费与资金费
    assert snap["total_return_pct"] == 6.05
    # live 下本地 cash 不随订单变动，本地权益不可比；对账改看逐笔订单盈亏口径差
    assert snap["ledger_equity"] is None
    assert snap["reconcile_diff"] == -60.5
    assert snap["exchange_commission"] == -1.2


def test_sync_account_falls_back_to_balance_endpoint(db, monkeypatch):
    fake = TruthClient()
    monkeypatch.setattr(fake, "account", lambda: (_ for _ in ()).throw(RuntimeError("unsupported")))
    _enable_live(db, monkeypatch, fake)
    acc = gw.fetch_account(fake)
    assert acc["source"] == "balance" and acc["wallet_balance"] == 1050.5


def test_backfill_fills_uses_real_prices_and_fees(db, monkeypatch):
    fake = TruthClient()
    fake.fills = [
        {"orderId": 100, "symbol": "BTCUSDT", "price": "100.0", "qty": "1",
         "commission": "0.04", "commissionAsset": "USDT", "realizedPnl": "0", "time": now_ms()},
        {"orderId": 102, "symbol": "BTCUSDT", "price": "103.0", "qty": "1",
         "commission": "0.0412", "commissionAsset": "USDT", "realizedPnl": "3.0", "time": now_ms()},
    ]
    _enable_live(db, monkeypatch, fake)
    order = live_order(status="CLOSED", close_price=101.5, realized_pnl=1.0)
    db.add(order)
    db.commit()
    assert exchange_truth.backfill_fills(db, order, fake) is True
    assert order.entry_price == 100.0 and order.close_price == 103.0
    assert order.entry_fee == 0.04 and order.exit_fee == 0.0412
    assert order.realized_pnl == round(3.0 - 0.04 - 0.0412, 6)
    assert json.loads(order.note)["pnl_source"] == "binance"


def test_backfill_fills_without_trades_keeps_estimate(db, monkeypatch):
    fake = TruthClient()
    _enable_live(db, monkeypatch, fake)
    order = live_order(status="CLOSED", realized_pnl=1.0)
    db.add(order)
    db.commit()
    assert exchange_truth.backfill_fills(db, order, fake) is False
    assert order.realized_pnl == 1.0


def test_sync_live_writes_exchange_stop_price_and_restores_missing_protection(db, monkeypatch):
    fake = TruthClient(position_amt="1", mark_price="100.5")
    fake.protection = {
        101: {"status": "NEW", "stopPrice": "98.5", "avgPrice": "0"},
        102: {"status": "CANCELED", "stopPrice": "103", "avgPrice": "0"},
    }
    _enable_live(db, monkeypatch, fake)
    order = live_order()
    db.add(order)
    db.commit()
    assert live_sync.sync_live(db, order) is None
    assert order.sl == 98.5  # 交易所触发价回写本地
    note = json.loads(order.note)
    assert note["sl_status"] == "NEW" and note["tp_status"] == "NEW"
    assert note["tp_order_id"] != 102  # 被撤销的止盈单已自动重挂


def test_sync_live_settles_at_protection_fill_price_not_mark(db, monkeypatch):
    fake = TruthClient(position_amt="0", mark_price="101.5")
    fake.protection = {
        101: {"status": "FILLED", "stopPrice": "99", "avgPrice": "99.2"},
        102: {"status": "NEW", "stopPrice": "103", "avgPrice": "0"},
    }
    _enable_live(db, monkeypatch, fake)
    order = live_order()
    db.add(order)
    db.commit()
    result = live_sync.sync_live(db, order)
    assert result.close_reason == "exchange_sl"
    assert result.close_price == 99.2  # 真实成交均价，而非标记价 101.5


def test_rebase_and_archive_paper_positions(db, monkeypatch):
    fake = TruthClient()
    _enable_live(db, monkeypatch, fake)
    paper = Order(symbol="ETHUSDT", interval="1m", side="LONG", qty=1,
                  entry_price=100, mark_price=101, status="OPEN", source="paper")
    db.add(paper)
    db.commit()
    assert exchange_truth.archive_paper_positions(db) == 1
    assert paper.status == "CLOSED" and paper.realized_pnl == 0.0
    assert paper.close_reason == "env_switch_archived"
    result = exchange_truth.rebase(db, force=True)
    assert result["changed"] is True and result["initial_capital"] == 1050.5
    state = all_state(db)
    assert state["exchange_baseline_env"] == "demo"
    assert state["cash"] == 900.0


def test_refresh_available_updates_cached_balance(db, monkeypatch):
    """同周期连开多仓后可用余额必须刷新，否则下一单会重复用旧余额。"""
    fake = TruthClient()
    _enable_live(db, monkeypatch, fake)
    exchange_truth.sync_account(db, force=True)
    assert exchange_truth.cached(db)["available_balance"] == 900.0
    original = fake.account
    fake.account = lambda: {**original(), "availableBalance": "321.5"}
    assert exchange_truth.refresh_available(db, fake) == 321.5
    assert exchange_truth.cached(db)["available_balance"] == 321.5


def test_unsupported_symbols_skip_and_reset_on_rebase(db, monkeypatch):
    fake = TruthClient()
    _enable_live(db, monkeypatch, fake)
    exchange_truth.mark_unsupported(db, "FAKEUSDT")
    assert exchange_truth.is_unsupported(db, "FAKEUSDT") is True
    exchange_truth.rebase(db, force=True)
    assert exchange_truth.is_unsupported(db, "FAKEUSDT") is False


def _account_with(available: float, position_margin: float):
    """构造「满仓」账户：钱包余额不变，可用余额被已有持仓保证金吃掉。"""
    return lambda: {
        "totalWalletBalance": "1050.5",
        "totalMarginBalance": "1060.5",
        "totalUnrealizedProfit": "10.0",
        "totalPositionInitialMargin": str(position_margin),
        "assets": [{"asset": "USDT", "walletBalance": "1050.5",
                    "availableBalance": str(available)}],
    }


def test_open_live_caps_margin_by_binance_available_balance(db, monkeypatch):
    """满仓时按币安可用余额裁剪仓位，不能按总权益算到吃满保证金被 -2019 拒单。"""
    fake = TruthClient()
    fake.account = _account_with(30.0, 1020.0)
    _enable_live(db, monkeypatch, fake)
    exchange_truth.sync_account(db, force=True)
    assert exchange_truth.cached(db)["available_balance"] == 30.0

    order = live.open_live(db, signal())
    assert order is not None
    # 可用 30 × 0.90 缓冲 = 27；qty 必须同步按 27 × 10 / 100 缩放
    assert order.margin == 27.0
    assert order.qty == pytest.approx(2.7, abs=0.002)


def test_open_live_skips_when_binance_margin_exhausted(db, monkeypatch):
    """可用保证金低于最小开仓额时安静跳过，不发任何注定被交易所拒绝的请求。"""
    fake = TruthClient()
    fake.account = _account_with(5.0, 1045.0)
    _enable_live(db, monkeypatch, fake)
    exchange_truth.sync_account(db, force=True)

    assert live.open_live(db, signal()) is None
    assert market_orders(fake) == []
    assert db.query(Order).count() == 0
    row = db.query(ExchangeEvent).order_by(ExchangeEvent.id.desc()).first()
    assert row.status == "blocked" and "可用保证金不足" in row.message


def test_open_live_respects_relative_min_margin(db, monkeypatch):
    """币安口径下最低保证金按收益基准取 2%，避免满仓后连开碎仓吃光可用余额。"""
    fake = TruthClient()
    # initial_capital=1000 → 下限 20；可用 20 时 cap = 18 < 20，必须跳过
    fake.account = _account_with(20.0, 1030.0)
    _enable_live(db, monkeypatch, fake)
    exchange_truth.sync_account(db, force=True)

    assert live.open_live(db, signal()) is None
    assert market_orders(fake) == []
    row = db.query(ExchangeEvent).order_by(ExchangeEvent.id.desc()).first()
    assert row.status == "blocked" and "最低开仓保证金 20.0000" in row.message

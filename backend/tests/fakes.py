"""测试共享 Fake：合约客户端与订单/信号构造helper。全部 mock，不发起真实请求。

被 test_live_trading.py / test_maker_orders.py 共用；本文件不是测试用例（pytest 不收集）。
"""

import json
from types import SimpleNamespace

from app.market.futures import BinanceApiError
from app.models import Order, StrategyConfig
from app.trading import live

# 本仓库只剩一套策略，它的默认参数是 scripts 移植口径（1 倍杠杆 / 30% 名义）。
# 期货链路要测的是交易所取整、保护单与回滚，不应被策略默认杠杆牵连，
# 因此统一把 unrealistic 权限放宽回测试期望的通用档位。
EXECUTION_TEST_PARAMS = {
    "max_leverage": 10,
    "base_position_pct": 0.25,
    "max_position_pct": 0.25,
    "cooldown_bars": 10,
    "risk_per_trade_pct": 0.005,   # 策略默认 0.02 会放宽风险预算，这里退回全局默认
}


def signal(**overrides):
    return SimpleNamespace(
        **(
            dict(
                symbol="BTCUSDT",
                interval="1m",
                direction="LONG",
                price=100.0,
                sl=99.0,
                tp=103.0,
                leverage=10,
                position_pct=0.25,
                strategy="absorption_reversal",
                score=1.0,
                prob_up=0.8,
            )
            | overrides
        )
    )


def live_order(**overrides):
    defaults = dict(
        symbol="BTCUSDT",
        interval="1m",
        side="LONG",
        qty=1.0,
        entry_price=100.0,
        mark_price=100.0,
        margin=10.0,
        leverage=5,
        sl=99.0,
        tp=103.0,
        status="OPEN",
        source="live",
        strategy="absorption_reversal",
        note=json.dumps({"exchange_order_id": 100, "sl_order_id": 101, "tp_order_id": 102}),
    )
    return Order(**(defaults | overrides))


def seed_strategy_params(db, code="absorption_reversal", **overrides):
    """给指定策略写一份「测试档」参数，避免策略默认（1 倍杠杆/零冷却）掩盖被测行为。"""
    row = db.query(StrategyConfig).filter(StrategyConfig.code == code).first()
    params = json.dumps({**EXECUTION_TEST_PARAMS, **overrides}, ensure_ascii=False)
    if row is None:
        db.add(StrategyConfig(code=code, name=code, description="", enabled=1, params=params))
    else:
        row.params = params
    db.commit()


def seed_binding(
    db,
    symbols,
    *,
    strategy="absorption_reversal",
    interval="1m",
    params=None,
    pool_code="test",
    priority=10,
    enabled=1,
    strategy_enabled=1,
):
    """装一条「策略 × 静态池」绑定。

    巡检现在由绑定驱动：没有绑定就没有任何币种会被同步/评估。
    测试里用静态池精确圈定要被跑到的币种，避免依赖 universe 网络拉取。
    """
    from app.models import StrategyBinding, SymbolPool

    list_symbols = [symbol.upper() for symbol in symbols]
    row = db.query(StrategyConfig).filter(StrategyConfig.code == strategy).first()
    if row is None:
        db.add(
            StrategyConfig(
                code=strategy, name=strategy, description="",
                enabled=strategy_enabled, params=json.dumps(params or {}),
            )
        )
    else:
        row.enabled = strategy_enabled
    pool = db.query(SymbolPool).filter(SymbolPool.code == pool_code).first()
    if pool is None:
        pool = SymbolPool(
            code=pool_code, name=pool_code, kind="static",
            symbols=json.dumps(list_symbols), rule="{}", enabled=1,
        )
        db.add(pool)
    else:
        pool.kind = "static"
        pool.symbols = json.dumps(list_symbols)
        pool.rule = "{}"
        pool.enabled = 1
    db.commit()

    binding = (
        db.query(StrategyBinding)
        .filter(
            StrategyBinding.strategy_code == strategy,
            StrategyBinding.pool_code == pool_code,
            StrategyBinding.interval == interval,
        )
        .first()
    )
    if binding is None:
        binding = StrategyBinding(
            strategy_code=strategy, pool_code=pool_code, interval=interval,
            enabled=enabled, priority=priority,
            params=json.dumps(params or {}, ensure_ascii=False),
        )
        db.add(binding)
    else:
        binding.enabled = enabled
        binding.priority = priority
        binding.params = json.dumps(params or {}, ensure_ascii=False)
    db.commit()
    return binding


def market_orders(fake):
    return [params for call in fake.calls if call[0] == "order" for params in [call[1]]]


def patch_live(db, monkeypatch, fake):
    seed_strategy_params(db)
    monkeypatch.setattr(live, "load_credentials", lambda session: ("key", "secret"))
    monkeypatch.setattr(live, "BinanceFuturesClient", lambda *args, **kwargs: fake)


class FakeFuturesClient:
    """合约客户端 Fake：记录全部下单/撤单调用，可按参数注入拒单与成交状态。"""

    def __init__(
        self,
        step="0.001",
        notional="5",
        fail_protection=False,
        fail_tp=False,
        fail_leverage=False,
        cancel_error=None,
        reject_reduce_only=False,
        position_amt="0",
        sl_status="NEW",
        tp_status="NEW",
        mark_price="101.0",
        avg_price="100.5",
        limit_reject=None,
        maker_status="NEW",
        maker_avg_price="100.4",
    ):
        self.calls = []
        self.cancel_calls = []
        self.step = step
        self.notional = notional
        self.fail_protection = fail_protection
        self.fail_tp = fail_tp
        self.fail_leverage = fail_leverage
        self.cancel_error = cancel_error
        self.reject_reduce_only = reject_reduce_only
        self.position_amt = position_amt
        self.sl_status = sl_status
        self.tp_status = tp_status
        self.mark_price = mark_price
        self.avg_price = avg_price
        self.limit_reject = limit_reject
        self.maker_status = maker_status
        self.maker_avg_price = maker_avg_price
        self.limit_order_id = None

    def exchange_info(self, symbol):
        return {
            "symbols": [
                {
                    "symbol": symbol,
                    "filters": [
                        {"filterType": "LOT_SIZE", "stepSize": self.step, "minQty": "0.001"},
                        {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
                        {"filterType": "MIN_NOTIONAL", "notional": self.notional},
                    ],
                }
            ]
        }

    def set_position_side(self, dual):
        self.calls.append(("dual", dual))
        return {}

    def set_margin_type(self, symbol, margin_type):
        self.calls.append(("margin_type", symbol, margin_type))
        return {}

    def set_leverage(self, symbol, leverage):
        if self.fail_leverage:
            raise BinanceApiError("leverage rejected", -4028)
        self.calls.append(("leverage", symbol, leverage))
        return {}

    def new_order(self, **params):
        self.calls.append(("order", params))
        if self.fail_protection and params.get("type") in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
            raise BinanceApiError("protection rejected", -2021)
        if self.fail_tp and params.get("type") == "TAKE_PROFIT_MARKET":
            raise BinanceApiError("tp rejected", -2021)
        if self.reject_reduce_only and params.get("reduceOnly") == "true":
            raise BinanceApiError("ReduceOnly Order is rejected", -2022)
        if params.get("type") == "LIMIT":
            if self.limit_reject is not None:
                raise self.limit_reject
            self.limit_order_id = 100 + len(self.calls)
            return {"orderId": self.limit_order_id, "status": "NEW", "avgPrice": "0"}
        return {"orderId": 100 + len(self.calls), "avgPrice": self.avg_price, "status": "FILLED"}

    def cancel_order(self, symbol, order_id):
        self.cancel_calls.append((symbol, order_id))
        if self.cancel_error is not None:
            raise self.cancel_error
        return {}

    def get_order(self, symbol, order_id):
        if order_id == self.limit_order_id:
            return {"orderId": order_id, "status": self.maker_status,
                    "avgPrice": self.maker_avg_price, "executedQty": "0"}
        status = {101: self.sl_status, 102: self.tp_status}.get(order_id, "NEW")
        return {"orderId": order_id, "status": status}

    def premium_index(self, symbol):
        return {"markPrice": self.mark_price}

    def position_risk(self, symbol):
        return [{"symbol": symbol, "positionAmt": self.position_amt}]

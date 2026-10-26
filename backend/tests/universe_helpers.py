"""币种宇宙测试共享helper：exchangeInfo 行构造、Fake 客户端、monkeypatch 装配、缓存清理。
被 test_universe.py / test_blacklist.py 共用；非测试用例（pytest 不收集）。
各测试文件需自行定义 autouse fixture 调用 clear_universe_cache()（fixture 不跨模块继承）。"""

from app.market import universe


def row(symbol, contract="PERPETUAL", quote="USDT", status="TRADING"):
    return {"symbol": symbol, "contractType": contract, "quoteAsset": quote, "status": status}


class FakeUniverseClient:
    """exchangeInfo + ticker_24hr 双 Fake：可注入拒单/异常，记录调用次数。"""

    def __init__(self, symbols=None, error=None, volumes=None, ticker_error=None):
        self.symbols = symbols if symbols is not None else []
        self.error = error
        self.ticker_error = ticker_error
        self.calls = 0
        self.ticker_calls = 0
        rows = self.symbols if self.symbols is not None else []
        # 默认全部给足 24h 成交额，保持既有过滤断言不受流动性门槛影响
        self.volumes = volumes if volumes is not None else {
            str(item.get("symbol")): 50_000_000.0 for item in rows
        }

    def exchange_info(self, symbol=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {"symbols": self.symbols}

    def ticker_24hr(self, symbol=None):
        self.ticker_calls += 1
        if self.ticker_error is not None:
            raise self.ticker_error
        return dict(self.volumes)


def patch_universe(monkeypatch, fake):
    monkeypatch.setattr(universe, "BinanceFuturesClient", lambda **kwargs: fake)


def clear_universe_cache():
    universe._cache.clear()
    universe._cache_at.clear()
    universe._ticker_cache.clear()
    universe._ticker_cache_at.clear()

"""全局代理配置测试。全部 mock httpx.Client，不发起真实网络请求。"""

import httpx
import pytest
from fastapi import HTTPException

from app.api.settings import _validate
from app.market.binance import BinanceClient
from app.market.data_service import default_market_client
from app.market.futures import BinanceFuturesClient
from app.state import set_state
from app.trading import live


class _FakeResponse:
    status_code = 200

    def json(self):
        return {"serverTime": 1700000000000}


class _FakeClient:
    def __init__(self, kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def request(self, method, path, params=None, headers=None):
        return _FakeResponse()


class _FakeClientFactory:
    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeClient(kwargs)


@pytest.fixture
def fake_httpx(monkeypatch):
    factory = _FakeClientFactory()
    monkeypatch.setattr(httpx, "Client", factory)
    return factory


def test_spot_client_passes_proxy_to_httpx(fake_httpx):
    BinanceClient(testnet=False, proxy="http://127.0.0.1:7890").server_time()
    assert fake_httpx.calls[-1]["proxy"] == "http://127.0.0.1:7890"


def test_futures_client_passes_proxy_to_httpx(fake_httpx):
    BinanceFuturesClient(testnet=True, proxy="socks5://127.0.0.1:1080").server_time()
    assert fake_httpx.calls[-1]["proxy"] == "socks5://127.0.0.1:1080"


def test_empty_proxy_passes_none_to_httpx(fake_httpx):
    BinanceClient(testnet=False, proxy="").server_time()
    assert fake_httpx.calls[-1]["proxy"] is None


def test_market_client_reads_proxy_from_state(db):
    set_state(db, "proxy_url", "http://127.0.0.1:7890")
    assert default_market_client(db).proxy == "http://127.0.0.1:7890"
    set_state(db, "proxy_url", "")
    assert default_market_client(db).proxy is None  # 留空 = 直连
    assert default_market_client(None).proxy is None  # db=None 兼容分支同样直连


def test_market_client_proxy_reaches_futures_in_live_mode(db):
    set_state(db, "proxy_url", "socks5://127.0.0.1:1080")
    set_state(db, "execution_mode", "live")
    client = default_market_client(db)
    assert isinstance(client, BinanceFuturesClient)
    assert client.proxy == "socks5://127.0.0.1:1080"


def test_live_order_client_reads_proxy_from_state(db):
    set_state(db, "proxy_url", "http://127.0.0.1:7890")
    assert live.client_for(db).proxy == "http://127.0.0.1:7890"


@pytest.mark.parametrize(
    "value",
    ["", "http://127.0.0.1:7890", "https://proxy.example.com:3128",
     "socks5://127.0.0.1:1080", "socks5h://127.0.0.1:1080"],
)
def test_proxy_url_accepts_valid(value):
    _validate({"proxy_url": value})


@pytest.mark.parametrize(
    "value",
    ["127.0.0.1:7890", "http://", "ftp://127.0.0.1:7890", "http://host", "http://host:99999", "http://host:0"],
)
def test_proxy_url_rejects_invalid(value):
    with pytest.raises(HTTPException) as exc:
        _validate({"proxy_url": value})
    assert exc.value.status_code == 400

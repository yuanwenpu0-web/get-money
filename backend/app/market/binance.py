"""币安 REST 客户端：公开行情 + 签名私有接口（模拟阶段不直接调用下单）。"""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from contextlib import ExitStack, contextmanager, nullcontext
from urllib.parse import urlencode

import httpx

from ..config import BINANCE_BASE_URL, BINANCE_TESTNET_URL

PUBLIC_HOSTS = [
    BINANCE_BASE_URL,
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
]


class BinanceError(RuntimeError):
    pass


class PooledHttpClient:
    """Reuse connections within an explicit scope, including across worker threads."""

    def _init_http_pool(self) -> None:
        self._pool_lock = threading.RLock()
        self._pool_users = 0
        self._pool_clients = {}
        self._pool_stack = None

    @contextmanager
    def connection_pool(self):
        with self._pool_lock:
            if not self._pool_users:
                self._pool_stack = ExitStack()
            self._pool_users += 1
        try:
            yield self
        finally:
            stack = None
            with self._pool_lock:
                self._pool_users -= 1
                if not self._pool_users:
                    stack, self._pool_stack = self._pool_stack, None
                    self._pool_clients = {}
            if stack is not None:
                stack.close()

    def _http_client(self, host: str):
        with self._pool_lock:
            if self._pool_users:
                if host not in self._pool_clients:
                    self._pool_clients[host] = self._pool_stack.enter_context(
                        httpx.Client(base_url=host, timeout=self.timeout, proxy=self.proxy)
                    )
                return nullcontext(self._pool_clients[host])
        return httpx.Client(base_url=host, timeout=self.timeout, proxy=self.proxy)


class BinanceClient(PooledHttpClient):
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True, timeout: float = 20.0,
                 proxy: str | None = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = BINANCE_TESTNET_URL if testnet else BINANCE_BASE_URL
        self.hosts = [BINANCE_TESTNET_URL] if testnet else PUBLIC_HOSTS
        self.timeout = timeout
        self.proxy = proxy or None
        self._init_http_pool()

    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False) -> dict | list:
        params = dict(params or {})
        headers = {}
        if signed:
            if not self.api_key or not self.api_secret:
                raise BinanceError("未配置币安 API key/secret")
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 5000
            query = urlencode(params)
            signature = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            params["signature"] = signature
            headers["X-MBX-APIKEY"] = self.api_key

        last_error: Exception | None = None
        for attempt in range(3):
            for host in self.hosts:
                try:
                    with self._http_client(host) as client:
                        response = client.request(method, path, params=params, headers=headers)
                    if response.status_code >= 400:
                        raise BinanceError(f"币安接口错误 {response.status_code}: {response.text[:300]}")
                    return response.json()
                except (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError) as exc:
                    last_error = exc
                    continue
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        raise BinanceError(f"币安网络请求失败（已重试多域名）: {last_error}")

    def klines(self, symbol: str, interval: str, start_ms: int | None = None, limit: int = 1000) -> list[list]:
        params: dict = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        return self._request("GET", "/api/v3/klines", params)

    def server_time(self) -> int:
        data = self._request("GET", "/api/v3/time")
        return int(data["serverTime"])

    def account(self) -> dict:
        return self._request("GET", "/api/v3/account", signed=True)

    def new_order(self, **params) -> dict:
        return self._request("POST", "/api/v3/order", params, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._request("DELETE", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    def open_orders(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/api/v3/openOrders", params, signed=True)


def parse_klines(rows: list[list]) -> list[dict]:
    parsed = []
    for row in rows:
        parsed.append(
            {
                "open_time": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time": int(row[6]),
                "quote_volume": float(row[7]),
                "trades": int(row[8]),
            }
        )
    return parsed

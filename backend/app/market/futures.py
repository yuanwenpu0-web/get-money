"""币安 U 本位合约 REST 客户端（主网 / 测试网）。

签名逻辑与现货客户端一致：HMAC SHA256 + X-MBX-APIKEY 头。
签名前用 /fapi/v1/time 校准服务器时间偏移量（缓存 10 分钟复用），
避免本机时钟偏差导致 -1021 时间戳越窗错误。
"""

from __future__ import annotations

import hashlib
import hmac
import math
import threading
import time
from urllib.parse import urlencode

import httpx

from ..config import FUTURES_MAIN_HOSTS, FUTURES_TESTNET_HOSTS
from .binance import BinanceError, PooledHttpClient

# 撤单时订单不存在 / 仓位方向无需修改等可忽略的业务错误码
IGNORED_CANCEL_CODES = {-2011}
TIME_OFFSET_TTL_SECONDS = 600


class BinanceApiError(BinanceError):
    """带交易所错误码的业务异常（如 -2011 订单不存在、-2022 ReduceOnly 拒绝）。"""

    def __init__(self, message: str, code: int = 0):
        super().__init__(message)
        self.code = code


def _api_error(response: httpx.Response) -> BinanceApiError:
    try:
        payload = response.json()
        code = int(payload.get("code", 0))
        message = str(payload.get("msg", response.text[:300]))
    except ValueError:
        code, message = 0, response.text[:300]
    return BinanceApiError(f"币安合约接口错误 {response.status_code}: {code} {message}", code)


class BinanceFuturesClient(PooledHttpClient):
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True, timeout: float = 20.0,
                 proxy: str | None = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.hosts = FUTURES_TESTNET_HOSTS if testnet else FUTURES_MAIN_HOSTS
        self.timeout = timeout
        self.proxy = proxy or None
        self._time_offset_ms = 0
        self._offset_checked_at = 0.0
        self._time_lock = threading.Lock()
        self._init_http_pool()

    def _calibrate_time(self) -> None:
        if time.time() - self._offset_checked_at < TIME_OFFSET_TTL_SECONDS:
            return
        with self._time_lock:
            if time.time() - self._offset_checked_at < TIME_OFFSET_TTL_SECONDS:
                return
            data = self._request("GET", "/fapi/v1/time")
            self._time_offset_ms = int(data["serverTime"]) - int(time.time() * 1000)
            self._offset_checked_at = time.time()

    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False) -> dict | list:
        params = dict(params or {})
        headers = {}
        if signed:
            if not self.api_key or not self.api_secret:
                raise BinanceError("未配置币安合约 API key/secret")
            self._calibrate_time()
            params["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
            params["recvWindow"] = 5000
            query = urlencode(params)
            params["signature"] = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
            headers["X-MBX-APIKEY"] = self.api_key

        last_error: Exception | None = None
        for attempt in range(3):
            for host in self.hosts:
                try:
                    with self._http_client(host) as client:
                        response = client.request(method, path, params=params, headers=headers)
                    if response.status_code >= 400:
                        raise _api_error(response)
                    return response.json()
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    last_error = exc
                    continue
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        raise BinanceError(f"币安合约接口网络请求失败（已重试多域名）: {last_error}")

    # 公开接口
    def klines(self, symbol: str, interval: str, start_ms: int | None = None, limit: int = 1000) -> list[list]:
        params: dict = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        return self._request("GET", "/fapi/v1/klines", params)

    def server_time(self) -> int:
        return int(self._request("GET", "/fapi/v1/time")["serverTime"])

    def exchange_info(self, symbol: str | None = None) -> dict:
        params = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/fapi/v1/exchangeInfo", params)

    def premium_index(self, symbol: str) -> dict:
        return self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})

    def ticker_24hr(self, symbol: str | None = None) -> dict[str, float]:
        """24h 滚动行情（公开免 Key）。无 symbol 返回全量数组并解析为
        {symbol: quoteVolume} 映射；quoteVolume 缺失/不可解析记 0.0。"""
        params = {"symbol": symbol} if symbol else {}
        rows = self._request("GET", "/fapi/v1/ticker/24hr", params)
        if isinstance(rows, dict):
            rows = [rows]
        volumes: dict[str, float] = {}
        for row in rows:
            name = str(row.get("symbol") or "")
            if not name:
                continue
            try:
                volumes[name] = float(row.get("quoteVolume") or 0.0)
            except (TypeError, ValueError):
                volumes[name] = 0.0
        return volumes

    def funding_rate(self, symbol: str, start_ms: int | None = None, end_ms: int | None = None,
                     limit: int = 1000) -> list[dict]:
        """资金费历史（公开）。startTime 起算 INCLUSIVE、升序返回；解析为 float。"""
        params: dict = {"symbol": symbol, "limit": min(max(int(limit), 1), 1000)}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        rows = self._request("GET", "/fapi/v1/fundingRate", params)
        return [
            {
                "funding_time": int(row["fundingTime"]),
                "funding_rate": float(row["fundingRate"]),
                "mark_price": float(row.get("markPrice") or 0),
                "rate_type": str(row.get("rateType") or ""),
            }
            for row in rows
        ]

    def long_short_ratio(self, symbol: str, period: str = "5m", limit: int = 1) -> list[dict]:
        """多空账户数比（散户情绪，公开免 Key）。period 支持 5m/15m/30m/1h/2h/4h/6h/12h/1d。
        返回升序 [{timestamp, long_account, short_account, long_short_ratio}]（float）。"""
        params = {"symbol": symbol, "period": period, "limit": int(limit)}
        rows = self._request("GET", "/futures/data/globalLongShortAccountRatio", params)
        return [
            {
                "timestamp": int(row["timestamp"]),
                "long_account": float(row["longAccount"]),
                "short_account": float(row["shortAccount"]),
                "long_short_ratio": float(row["longShortRatio"]),
            }
            for row in rows
        ]

    def top_trader_long_short(self, symbol: str, period: str = "5m", limit: int = 1) -> list[dict]:
        """大户/巨鲸持仓多空比（聪明钱情绪，公开免 Key）。结构与散户比一致。"""
        params = {"symbol": symbol, "period": period, "limit": int(limit)}
        rows = self._request("GET", "/futures/data/topLongShortPositionRatio", params)
        return [
            {
                "timestamp": int(row["timestamp"]),
                "long_account": float(row["longAccount"]),
                "short_account": float(row["shortAccount"]),
                "long_short_ratio": float(row["longShortRatio"]),
            }
            for row in rows
        ]

    def taker_buy_sell_ratio(self, symbol: str, period: str = "5m", limit: int = 1) -> list[dict]:
        """主动买卖量比（taker 主动吃单方向，公开免 Key）。
        返回升序 [{timestamp, buy_sell_ratio, buy_vol, sell_vol}]（float）。"""
        params = {"symbol": symbol, "period": period, "limit": int(limit)}
        rows = self._request("GET", "/futures/data/takerlongshortRatio", params)
        return [
            {
                "timestamp": int(row["timestamp"]),
                "buy_sell_ratio": float(row["buySellRatio"]),
                "buy_vol": float(row["buyVol"]),
                "sell_vol": float(row["sellVol"]),
            }
            for row in rows
        ]

    def open_interest(self, symbol: str) -> dict:
        """当前未平仓合约量（张数，公开免 Key）。返回 {open_interest(float), time(int)}。"""
        row = self._request("GET", "/fapi/v1/openInterest", {"symbol": symbol})
        return {"open_interest": float(row["openInterest"]), "time": int(row["time"])}

    def open_interest_history(self, symbol: str, period: str = "5m", limit: int = 1) -> list[dict]:
        """未平仓合约量历史（张数 + 名义价值，公开免 Key）。
        返回升序 [{timestamp, sum_open_interest, sum_open_interest_value}]（float）。"""
        params = {"symbol": symbol, "period": period, "limit": int(limit)}
        rows = self._request("GET", "/futures/data/openInterestHist", params)
        return [
            {
                "timestamp": int(row["timestamp"]),
                "sum_open_interest": float(row["sumOpenInterest"]),
                "sum_open_interest_value": float(row["sumOpenInterestValue"]),
            }
            for row in rows
        ]

    # 签名接口
    def account(self) -> dict:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def balance(self) -> list[dict]:
        return self._request("GET", "/fapi/v2/balance", signed=True)

    def new_order(self, **params) -> dict:
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._request("DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    def get_order(self, symbol: str, order_id: int) -> dict:
        return self._request("GET", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    def open_orders(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/fapi/v1/openOrders", params, signed=True)

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        return self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": int(leverage)}, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> dict:
        return self._request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type}, signed=True)

    def position_risk(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else {}
        return self._request("GET", "/fapi/v2/positionRisk", params, signed=True)

    def set_position_side(self, dual: bool) -> dict:
        return self._request(
            "POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "true" if dual else "false"}, signed=True
        )


def symbol_filters(exchange_info: dict, symbol: str) -> dict:
    """从 exchangeInfo 提取指定合约的 filters，按 filterType 建索引。"""
    for row in exchange_info.get("symbols", []):
        if row.get("symbol") == symbol:
            return {item["filterType"]: item for item in row.get("filters", [])}
    raise BinanceError(f"exchangeInfo 中缺少合约 {symbol}")


def floor_to_step(value: float, step: float) -> float:
    """按下单步长向下取整（永远不向上放大仓位）。"""
    if step <= 0:
        return value
    steps = math.floor(value / step + 1e-9)
    return round(steps * step, 10)


def round_to_tick(value: float, tick: float) -> float:
    if tick <= 0:
        return value
    return round(round(value / tick) * tick, 10)


def format_amount(value: float) -> str:
    """合约下单数量/价格字符串：固定 8 位小数再去尾零，避免科学计数法。"""
    text = f"{value:.8f}".rstrip("0").rstrip(".")
    return text or "0"

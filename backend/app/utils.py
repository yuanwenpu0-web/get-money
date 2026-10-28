from datetime import datetime, timezone

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def interval_ms(interval: str) -> int:
    if interval not in INTERVAL_MS:
        raise ValueError(f'不支持的K线周期: {interval}')
    return INTERVAL_MS[interval]


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def iso_utc(dt: datetime | None = None) -> str:
    value = dt or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def dt_ms(dt: datetime) -> int:
    value = dt
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


def round_price(value: float) -> float:
    # pandas/NumPy 运算会保留 numpy 标量；数据库驱动需要原生数值。
    value = float(value)
    if value >= 1000:
        return round(value, 2)
    if value >= 1:
        return round(value, 4)
    return round(value, 6)

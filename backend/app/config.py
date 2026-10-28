import os
from pathlib import Path
from urllib.parse import urlparse

try:  # python-dotenv 是可选依赖：缺失时静默退化为纯环境变量
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if load_dotenv is not None:
    # override=False：已存在的环境变量优先，方便运行时覆盖 .env
    load_dotenv(BASE_DIR / ".env", override=False)

DB_PATH = DATA_DIR / "trading.db"

# 支持的方言白名单（驱动前缀统一由 SQLAlchemy 处理）
SUPPORTED_DB_SCHEMES = {
    "sqlite",
    "postgresql",
    "postgresql+psycopg2",
    "postgresql+psycopg",
}

DEFAULT_DB_URL = f"sqlite:///{DB_PATH}"


def _normalize_db_url(raw: str) -> str:
    """校验并归一化数据库 URL。

    快速失败：URL 非法或方言不受支持时立刻抛错，避免带着错误配置启动到一半才发现。
    """
    url = (raw or "").strip()
    if not url:
        return DEFAULT_DB_URL
    parsed = urlparse(url)
    scheme = parsed.scheme
    if scheme == "postgres":  # 常见别名
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
        scheme = "postgresql+psycopg2"
    if scheme not in SUPPORTED_DB_SCHEMES:
        raise ValueError(
            f"不支持的数据库方言：{scheme!r}（DATABASE_URL={url!r}）。"
            f"可选：{sorted(SUPPORTED_DB_SCHEMES)}"
        )
    return url


DATABASE_URL = _normalize_db_url(os.getenv("DATABASE_URL", ""))
DB_URL = DATABASE_URL  # 向后兼容：旧代码引用 DB_URL

# 连接池参数（SQLite 不使用，留空即 SQLAlchemy 默认）
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "30"))
# 网络请求并发上限；数据库写入仍由调用线程顺序提交。
MARKET_SYNC_WORKERS = max(1, min(32, int(os.getenv("MARKET_SYNC_WORKERS", "8"))))
# 非 1m 周期在 1m 缺失时，是否用已单独存储的同周期行补齐。
MARKET_NATIVE_FALLBACK = os.getenv("MARKET_NATIVE_FALLBACK", "1").lower() not in ("0", "false", "no")

BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_TESTNET_URL = "https://testnet.binance.vision"

# 币安 U 本位合约（fapi）：主网带备用域名，测试网单域名
FUTURES_MAIN_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
]
FUTURES_TESTNET_HOSTS = ["https://testnet.binancefuture.com"]

# 历史数据库中的代理设置仍可用于公共行情。
DEFAULT_STATE = {
    "proxy_url": "",
}

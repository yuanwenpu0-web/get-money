import os
from pathlib import Path
from urllib.parse import urlparse

try:  # python-dotenv 是可选依赖：缺失时静默退化为纯环境变量
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
# 前提：docs/native-consistency.md 的重叠区间比对必须完全一致，否则关闭。
MARKET_NATIVE_FALLBACK = os.getenv("MARKET_NATIVE_FALLBACK", "1").lower() not in ("0", "false", "no")
FERNET_KEY_FILE = DATA_DIR / "fernet.key"

BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_TESTNET_URL = "https://testnet.binance.vision"

# 币安 U 本位合约（fapi）：主网带备用域名，测试网单域名
FUTURES_MAIN_HOSTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
]
FUTURES_TESTNET_HOSTS = ["https://testnet.binancefuture.com"]

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_INTERVAL = "15m"
SUPPORTED_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]

# 模拟撮合默认参数
DEFAULT_FEE_RATE = 0.0004
DEFAULT_SLIPPAGE = 0.0002
DEFAULT_MAX_LEVERAGE = 10

# 系统设置默认值（存于 app_state 表）
DEFAULT_STATE = {
    "data_source": "binance",
    "symbols": DEFAULT_SYMBOLS,
    "interval": DEFAULT_INTERVAL,
    "symbol_scope": "all",
    "execution_mode": "paper",
    "testnet": True,
    "proxy_url": "",
    "total_capital": 10000.0,
    "initial_capital": 10000.0,
    "cash": 10000.0,
    "fee_rate": DEFAULT_FEE_RATE,
    "slippage": DEFAULT_SLIPPAGE,
    "max_leverage": DEFAULT_MAX_LEVERAGE,
    "max_position_pct": 0.25,
    "base_position_pct": 0.08,
    "heavy_score": 0.90,
    "min_score": 0.75,
    "min_hold_bars": 30,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 2.5,
    "min_sl_pct": 0.0025,
    "min_tp_pct": 0.005,
    "backtest_trailing": False,
    "trailing_atr_mult": 1.2,
    "max_hold_bars": 120,
    "risk_per_trade_pct": 0.005,
    "cooldown_bars": 10,
    "trailing_enabled": True,
    "trail_activation_atr": 1.5,
    "max_signal_age_bars": 2,
    "min_quote_volume_usd": 30_000_000,  # 流动性过滤：24h 成交额下限（0 = 不过滤）
    "symbol_blacklist": [],              # 币种黑名单（大写符号数组，从宇宙排除）
    "risk_halt": False,                  # 账户级熔断开关（触发后需人工置 false 恢复）
    "risk_halt_reason": "",              # 最近一次熔断原因
    "max_drawdown_halt_pct": 0.10,       # 总回撤熔断线（0 = 关闭该检查）
    "consecutive_loss_halt": 5,          # 连续亏损笔数熔断线（0 = 关闭该检查）
    "order_type": "market",              # 实盘开仓主单类型：market / limit_maker
    "schedule_enabled": True,
    "schedule_seconds": 60,
    "train_horizon": 5,
    # 训练窗口 60 天：40 天在 1h 上只产出 ~903 样本、测试段 181 行，
    # 低于晋升门禁 MIN_TEST_ROWS=200，会使门禁结构性无法通过（迭代9 实测发现）。
    # 修复方向是加严数据侧（扩窗口），而非下调样本下限或 AUC 阈值。
    "auto_train_days": 60,
    "metrics_assist_enabled": False,  # 多源指标辅助决策开关（默认关：仅观测，不改现有决策）
    # 迭代9：持续训练（训练本身无害→默认开）与晋升门禁（启用 ml_ensemble 的闸门）
    "auto_train_enabled": True,        # 周期性自动训练开关（只耗 CPU，不改决策）
    "auto_train_interval_hours": 24,   # 同币种最小重训间隔（模型新鲜则不重训）
    "auto_train_top_n": 10,            # 训练目标：成交额 Top N
    "auto_train_workers": 2,           # 独立币种并行训练数（还会按 CPU 预算封顶）
    "ml_min_auc": 0.55,                # 晋升门禁：样本外 AUC 下限
    "ml_promote_rounds": 2,            # 晋升门禁：需连续达标的轮数
    "ml_auto_enable": False,           # 门禁通过后是否允许自动启用 ml_ensemble（默认否）
}

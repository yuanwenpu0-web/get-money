from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "trading.db"
DB_URL = f"sqlite:///{DB_PATH}"
FERNET_KEY_FILE = DATA_DIR / "fernet.key"

BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_TESTNET_URL = "https://testnet.binance.vision"

# 已有K线来源：nofx-new 项目的 PostgreSQL（只读导入）
NOFX_DB_URL = os.getenv("NOFX_DATABASE_URL", "postgresql://postgres:admin@127.0.0.1:5432/nofx_lite")
NOFX_SYMBOL_PREFIX = os.getenv("NOFX_SYMBOL_PREFIX", "BINANCE_")

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_INTERVAL = "1m"
SUPPORTED_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]

# 模拟撮合默认参数
DEFAULT_FEE_RATE = 0.0004
DEFAULT_SLIPPAGE = 0.0002
DEFAULT_MAX_LEVERAGE = 10

# 系统设置默认值（存于 app_state 表）
DEFAULT_STATE = {
    "data_source": "nofx",
    "symbols": DEFAULT_SYMBOLS,
    "interval": DEFAULT_INTERVAL,
    "execution_mode": "paper",
    "testnet": True,
    "total_capital": 10000.0,
    "initial_capital": 10000.0,
    "cash": 10000.0,
    "fee_rate": DEFAULT_FEE_RATE,
    "slippage": DEFAULT_SLIPPAGE,
    "max_leverage": DEFAULT_MAX_LEVERAGE,
    "max_position_pct": 0.25,
    "base_position_pct": 0.08,
    "heavy_score": 0.75,
    "min_score": 0.55,
    "min_hold_bars": 30,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 2.5,
    "min_sl_pct": 0.0025,
    "min_tp_pct": 0.005,
    "backtest_trailing": False,
    "trailing_atr_mult": 1.2,
    "max_hold_bars": 120,
    "schedule_enabled": True,
    "schedule_seconds": 60,
    "train_horizon": 5,
    "auto_train_days": 40,
}

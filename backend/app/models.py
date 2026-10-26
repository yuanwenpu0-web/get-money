from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, Float, Integer, String, Text, UniqueConstraint

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AppState(Base):
    __tablename__ = "app_state"

    key = Column(String(64), primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Credential(Base):
    __tablename__ = "credentials"

    id = Column(Integer, primary_key=True)
    api_key_enc = Column(Text, nullable=False)
    api_secret_enc = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Kline(Base):
    __tablename__ = "klines"
    __table_args__ = (UniqueConstraint("symbol", "interval", "open_time", name="uq_kline"),)

    id = Column(Integer, primary_key=True)
    symbol = Column(String(24), nullable=False, index=True)
    interval = Column(String(8), nullable=False, index=True)
    open_time = Column(BigInteger, nullable=False, index=True)
    close_time = Column(BigInteger, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False, default=0.0)
    quote_volume = Column(Float, nullable=False, default=0.0)
    trades = Column(Integer, nullable=False, default=0)


class FundingRate(Base):
    __tablename__ = "funding_rates"
    __table_args__ = (UniqueConstraint("symbol", "funding_time", name="uq_funding"),)

    id = Column(Integer, primary_key=True)
    symbol = Column(String(24), nullable=False, index=True)
    funding_time = Column(BigInteger, nullable=False, index=True)
    funding_rate = Column(Float, nullable=False)
    mark_price = Column(Float, default=0.0)
    rate_type = Column(String(16), default="")


class StrategyConfig(Base):
    __tablename__ = "strategy_configs"

    id = Column(Integer, primary_key=True)
    code = Column(String(48), unique=True, nullable=False)
    name = Column(String(96), nullable=False)
    description = Column(Text, default="")
    enabled = Column(Integer, default=1)
    params = Column(Text, default="{}")
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class ModelRecord(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(24), nullable=False, index=True)
    interval = Column(String(8), nullable=False)
    algo = Column(String(24), nullable=False)
    horizon = Column(Integer, default=5)
    path = Column(Text, nullable=False)
    features = Column(Text, default="[]")
    metrics = Column(Text, default="{}")
    train_start = Column(BigInteger, default=0)
    train_end = Column(BigInteger, default=0)
    rows = Column(Integer, default=0)
    status = Column(String(16), default="ready")
    created_at = Column(DateTime(timezone=True), default=utcnow)


class TierModel(Base):
    """按「规模×波动」分档的共享模型（pooled）：一个模型服务同档多个币种。

    与 models 表（per-symbol 模型）刻意分开：
    - 语义不同：一条记录对应 N 个币种，ModelRecord.symbol 是 NOT NULL 单值，装不下；
    - 隔离风险：独立新表不触碰现有训练/晋升/实盘链路，create_all 直接建，无需迁移。
    """

    __tablename__ = "tier_models"

    id = Column(Integer, primary_key=True)
    tier = Column(String(32), nullable=False, index=True)
    interval = Column(String(8), nullable=False)
    algo = Column(String(24), nullable=False)
    horizon = Column(Integer, default=5)
    path = Column(Text, nullable=False)
    features = Column(Text, default="[]")
    metrics = Column(Text, default="{}")
    symbols = Column(Text, default="[]")          # JSON 数组：参与训练的币种
    n_symbols = Column(Integer, default=0)
    train_start = Column(BigInteger, default=0)
    train_end = Column(BigInteger, default=0)
    rows = Column(Integer, default=0)
    status = Column(String(16), default="ready")
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(24), nullable=False, index=True)
    interval = Column(String(8), nullable=False)
    strategy = Column(String(48), nullable=False)
    ts = Column(BigInteger, nullable=False, index=True)
    price = Column(Float, nullable=False)
    direction = Column(String(8), nullable=False)
    score = Column(Float, default=0.0)
    prob_up = Column(Float, default=0.0)
    sl = Column(Float, default=0.0)
    tp = Column(Float, default=0.0)
    leverage = Column(Integer, default=1)
    position_pct = Column(Float, default=0.0)
    reason = Column(Text, default="{}")
    created_at = Column(DateTime(timezone=True), default=utcnow)


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(24), nullable=False, index=True)
    interval = Column(String(8), nullable=False)
    side = Column(String(8), nullable=False)  # LONG / SHORT
    qty = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    mark_price = Column(Float, default=0.0)
    margin = Column(Float, default=0.0)
    leverage = Column(Integer, default=1)
    sl = Column(Float, default=0.0)
    tp = Column(Float, default=0.0)
    status = Column(String(16), default="OPEN", index=True)  # OPEN / CLOSED / CANCELED
    source = Column(String(16), default="paper")
    strategy = Column(String(48), default="")
    signal_id = Column(Integer, default=0)
    entry_fee = Column(Float, default=0.0)
    exit_fee = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    close_price = Column(Float, default=0.0)
    close_reason = Column(String(32), default="")
    note = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    closed_at = Column(DateTime(timezone=True), nullable=True)


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"

    id = Column(Integer, primary_key=True)
    ts = Column(BigInteger, nullable=False, index=True)
    cash = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0.0)
    realized_pnl_cum = Column(Float, default=0.0)
    open_positions = Column(Integer, default=0)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(24), nullable=False)
    interval = Column(String(8), nullable=False)
    strategy = Column(String(48), nullable=False)
    algo = Column(String(24), default="rule")
    params = Column(Text, default="{}")
    metrics = Column(Text, default="{}")
    equity_curve = Column(Text, default="[]")
    trades = Column(Text, default="[]")
    status = Column(String(16), default="running")
    error = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=utcnow)


class JobLog(Base):
    __tablename__ = "job_logs"

    id = Column(Integer, primary_key=True)
    job = Column(String(48), nullable=False)
    level = Column(String(16), default="info")
    message = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=utcnow)


class CoinMetric(Base):
    __tablename__ = "coin_metrics"

    id = Column(Integer, primary_key=True)
    symbol = Column(String(24), nullable=False, index=True)
    ts = Column(BigInteger, nullable=False, index=True)
    metrics = Column(Text, default="{}")

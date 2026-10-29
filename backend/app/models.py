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


class Sentiment(Base):
    """正交衍生品情绪时序（P9）：持仓量 + 多空比 + 主动买卖比，全部免 Key 公开接口。

    数据源 /futures/data/* 端点仅保留最近 ~30 天 1h 历史（500 条上限），
    因此定位是横截面 IC 探针与近期特征，不做长历史 walk-forward。
    """

    __tablename__ = "sentiment"
    __table_args__ = (UniqueConstraint("symbol", "ts", name="uq_sentiment"),)

    id = Column(Integer, primary_key=True)
    symbol = Column(String(24), nullable=False, index=True)
    ts = Column(BigInteger, nullable=False, index=True)
    open_interest = Column(Float, default=0.0)      # 未平仓量（张数）
    oi_value = Column(Float, default=0.0)            # 未平仓名义价值 USD
    long_short_ratio = Column(Float, default=0.0)    # 全市场多空账户比（散户情绪）
    top_long_short = Column(Float, default=0.0)      # 大户持仓多空比（聪明钱）
    taker_buy_sell = Column(Float, default=0.0)      # taker 主动买卖量比（订单流方向）


class StrategyConfig(Base):
    __tablename__ = "strategy_configs"

    id = Column(Integer, primary_key=True)
    code = Column(String(48), unique=True, nullable=False)
    name = Column(String(96), nullable=False)
    description = Column(Text, default="")
    enabled = Column(Integer, default=1)
    params = Column(Text, default="{}")
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class SymbolPool(Base):
    """币种池：策略的运行范围单位。

    kind=static → 成员就是 `symbols` 里手写的列表；
    kind=rule   → 成员由 `rule` 动态解析（base / 成交额门槛 / 排除主流 / TopN / 排除项）。
    两种都再减去全局黑名单，见 strategy/pools.py。
    """

    __tablename__ = "symbol_pools"

    id = Column(Integer, primary_key=True)
    code = Column(String(48), unique=True, nullable=False)
    name = Column(String(96), nullable=False)
    description = Column(Text, default="")
    kind = Column(String(16), default="static")   # static / rule
    symbols = Column(Text, default="[]")          # JSON 数组，static 池的成员
    rule = Column(Text, default="{}")             # JSON 对象，rule 池的筛选条件
    enabled = Column(Integer, default=1)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class SymbolParam(Base):
    """币种级参数覆盖：同一策略、同一周期，某个币种可以有自己的参数。

    参数优先级：全局设置 < 策略默认 < 绑定 params < 本表。
    只有「真的按币种寻出不同最优参数」时才写这一行——绝大多数币种应当沿用
    绑定参数，覆盖表保持小而可审计。`source=optimization` 的记录带 `run_id`，
    可以追溯到是哪一次参数寻优写进来的。
    """

    __tablename__ = "symbol_params"
    __table_args__ = (
        UniqueConstraint("strategy_code", "symbol", "interval", name="uq_symbol_param"),
    )

    id = Column(Integer, primary_key=True)
    strategy_code = Column(String(48), nullable=False, index=True)
    symbol = Column(String(24), nullable=False, index=True)
    interval = Column(String(8), nullable=False)
    params = Column(Text, default="{}")           # JSON 对象，覆盖绑定/策略默认
    source = Column(String(32), default="manual")  # manual / optimization
    run_id = Column(Integer, default=0)            # 产生该覆盖的寻优任务
    note = Column(String(200), default="")
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class StrategyBinding(Base):
    """策略 × 币种池 的运行绑定：一条绑定 = 一套完整参数 + 一个K线周期。

    同一策略可以有多条绑定（主流 15m / 山寨 5m / 山寨 1m），各自参数互不影响。
    参数优先级：全局设置 < 策略默认 < 本条绑定的 params。
    """

    __tablename__ = "strategy_bindings"
    __table_args__ = (
        UniqueConstraint("strategy_code", "pool_code", "interval", name="uq_binding_scope"),
    )

    id = Column(Integer, primary_key=True)
    strategy_code = Column(String(48), nullable=False, index=True)
    pool_code = Column(String(48), nullable=False, index=True)
    interval = Column(String(8), nullable=False)
    enabled = Column(Integer, default=1)
    priority = Column(Integer, default=100)       # 仅用于展示排序；同币多绑定时按评分择优
    params = Column(Text, default="{}")           # JSON 对象，覆盖策略默认
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
    pool = Column(String(48), default="")  # 产生该信号的币种池（参数归属追溯）
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
    pool = Column(String(48), default="")  # 该仓位所用参数所属的币种池
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


class OptimizationRun(Base):
    """一次全币种参数寻优任务。

    结果落在 `optimization_results`（每币种一行），本表只存任务级配置与进度，
    避免前端轮询时把 500 多行结果一起拉回来。
    """

    __tablename__ = "optimization_runs"

    id = Column(Integer, primary_key=True)
    strategy = Column(String(48), nullable=False)
    interval = Column(String(8), nullable=False)
    pool = Column(String(48), default="")        # 空 = 全部有足够历史的币种
    days = Column(Integer, default=90)
    windows = Column(Integer, default=3)
    workers = Column(Integer, default=0)         # 0 = 自动（CPU-4）
    status = Column(String(16), default="pending")  # pending/running/done/failed/cancelled
    config = Column(Text, default="{}")          # 网格、基准参数、成本、门槛
    progress = Column(Text, default="{}")        # total/done/failed/elapsed/eta/current
    summary = Column(Text, default="{}")         # 计数与最优分布统计
    error = Column(Text, default="")
    created_at = Column(DateTime(timezone=True), default=utcnow)
    finished_at = Column(DateTime(timezone=True), default=None)


class OptimizationResult(Base):
    """寻优结果：单币种的最优参数、逐窗口指标与判定。"""

    __tablename__ = "optimization_results"
    __table_args__ = (
        UniqueConstraint("run_id", "symbol", name="uq_optimization_result"),
    )

    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, nullable=False, index=True)
    symbol = Column(String(24), nullable=False, index=True)
    params = Column(Text, default="{}")          # 最优组合（只存与基准不同的键）
    params_before = Column(Text, default="{}")   # 寻优前该币种实际生效的参数
    metrics = Column(Text, default="{}")         # 训练/验证逐窗指标与合计
    score = Column(Float, default=0.0)
    verdict = Column(String(24), default="")     # pass/inconclusive/negative/no_trades/insufficient
    applied_mode = Column(String(24), default="")  # 应用方式（留痕）
    applied_at = Column(DateTime(timezone=True), default=None)


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

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CapitalAdjust(BaseModel):
    amount: float = Field(gt=0)
    mode: str = "add"  # add / withdraw / set
    note: str = ""


class CredentialsPayload(BaseModel):
    api_key: str
    api_secret: str
    testnet: bool = True


class SettingsUpdate(BaseModel):
    data_source: str | None = None
    symbols: list[str] | None = None
    symbol_scope: str | None = None
    interval: str | None = None
    execution_mode: str | None = None
    proxy_url: str | None = None
    testnet: bool | None = None
    fee_rate: float | None = None
    slippage: float | None = None
    max_leverage: int | None = None
    max_position_pct: float | None = None
    base_position_pct: float | None = None
    heavy_score: float | None = None
    min_score: float | None = None
    min_hold_bars: int | None = None
    sl_atr_mult: float | None = None
    tp_atr_mult: float | None = None
    min_sl_pct: float | None = None
    min_tp_pct: float | None = None
    backtest_trailing: bool | None = None
    trailing_atr_mult: float | None = None
    max_hold_bars: int | None = None
    risk_per_trade_pct: float | None = Field(default=None, ge=.0001, le=.05, allow_inf_nan=False)
    cooldown_bars: int | None = Field(default=None, ge=0, le=10000)
    trailing_enabled: bool | None = None
    trail_activation_atr: float | None = Field(default=None, ge=.1, le=20, allow_inf_nan=False)
    max_signal_age_bars: int | None = Field(default=None, ge=1, le=10)
    min_quote_volume_usd: int | None = Field(default=None, ge=0)
    symbol_blacklist: list[str] | None = None
    risk_halt: bool | None = None
    risk_halt_reason: str | None = Field(default=None, max_length=500)
    max_drawdown_halt_pct: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    consecutive_loss_halt: int | None = Field(default=None, ge=0, le=1000)
    order_type: str | None = None
    schedule_enabled: bool | None = None
    schedule_seconds: int | None = None


class StrategyUpdate(BaseModel):
    enabled: bool | None = None
    params: dict | None = None


class PoolPayload(BaseModel):
    """币种池定义。kind=static 用 symbols；kind=rule 用 rule（base/成交额/TopN/排除）。"""

    code: str = Field(min_length=1, max_length=48, pattern=r'^[a-z0-9_]+$')
    name: str = Field(min_length=1, max_length=96)
    description: str = Field(default="", max_length=2000)
    kind: Literal['static', 'rule'] = 'rule'
    symbols: list[str] = Field(default_factory=list)
    rule: dict = Field(default_factory=dict)
    enabled: bool = True


class PoolUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=96)
    description: str | None = Field(default=None, max_length=2000)
    kind: Literal['static', 'rule'] | None = None
    symbols: list[str] | None = None
    rule: dict | None = None
    enabled: bool | None = None


class BindingPayload(BaseModel):
    """一条「策略 × 币种池」绑定：周期与参数都在这条记录上。"""

    strategy: str = Field(min_length=1, max_length=48)
    pool: str = Field(min_length=1, max_length=48)
    interval: Literal['1m', '5m', '15m', '1h', '4h', '1d']
    enabled: bool = True
    priority: int = Field(default=100, ge=0, le=10_000)
    params: dict = Field(default_factory=dict)


class BindingUpdate(BaseModel):
    enabled: bool | None = None
    priority: int | None = Field(default=None, ge=0, le=10_000)
    interval: Literal['1m', '5m', '15m', '1h', '4h', '1d'] | None = None
    params: dict | None = None


class SyncRequest(BaseModel):
    symbol: str = ""
    symbols: list[str] | None = None  # 非空=这些币种；空数组=全量 U 本位永续
    funding: bool = False  # True 时对 symbols 逐个同步资金费历史（天数用 days）
    interval: str = "1m"
    days: int = 30
    source: str = "binance"


class PreviewRequest(BaseModel):
    symbol: str
    interval: str = "1m"
    strategy: str = "absorption_reversal"
    # 可选：按某条「策略 × 池」绑定的参数预演，留空则用策略配置 + 全局设置
    pool: str | None = None


class MetricsSyncRequest(BaseModel):
    symbols: list[str] | None = None  # 非空=这些币种；空=全量宇宙
    source: str = "binance"
    period: str = "5m"  # 多空比/主动买卖量/未平仓量历史聚合周期


class BacktestRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    symbol: str = Field(min_length=2, max_length=24, pattern=r'^[A-Za-z0-9]+$')
    interval: Literal['1m', '5m', '15m', '1h', '4h', '1d'] = '1m'
    strategy: Literal['absorption_reversal'] = 'absorption_reversal'
    # 可选：用某条「策略 × 池」绑定的参数回测；留空则用策略配置
    pool: str | None = None
    days: int = Field(default=30, ge=1, le=180)
    initial_cash: float | None = Field(default=None, gt=0)
    fee: float | None = Field(default=None, ge=0, le=.01)
    slippage: float | None = Field(default=None, ge=0, le=.01)
    threshold: float | None = Field(default=None, ge=.5, le=1)
    optimize: bool = False


class OrderPatch(BaseModel):
    sl: float | None = None
    tp: float | None = None

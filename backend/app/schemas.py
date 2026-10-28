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
    train_horizon: int | None = None
    auto_train_days: int | None = None
    metrics_assist_enabled: bool | None = None
    auto_train_enabled: bool | None = None
    auto_train_interval_hours: int | None = Field(default=None, ge=1, le=720)
    auto_train_top_n: int | None = Field(default=None, ge=1, le=50)
    auto_train_workers: int | None = Field(default=None, ge=1, le=8, strict=True)
    ml_min_auc: float | None = Field(default=None, ge=.5, le=.9, allow_inf_nan=False)
    ml_promote_rounds: int | None = Field(default=None, ge=1, le=10)
    ml_auto_enable: bool | None = None


class StrategyUpdate(BaseModel):
    enabled: bool | None = None
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
    strategy: str = "ml_ensemble"


class MetricsSyncRequest(BaseModel):
    symbols: list[str] | None = None  # 非空=这些币种；空=全量宇宙
    source: str = "binance"
    period: str = "5m"  # 多空比/主动买卖量/未平仓量历史聚合周期


class TrainRequest(BaseModel):
    symbol: str
    interval: str = "1m"
    algo: str = "lightgbm"
    horizon: int = 5
    days: int = 30


class BacktestRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    symbol: str = Field(min_length=2, max_length=24, pattern=r'^[A-Za-z0-9]+$')
    interval: Literal['1m', '5m', '15m', '1h', '4h', '1d'] = '1m'
    strategy: Literal['ml_ensemble', 'trend_follow', 'mean_reversion', 'funding_extreme', 'mtf_resonance', 'strong_pullback', 'absorption_reversal'] = 'ml_ensemble'
    algo: Literal['rule', 'lightgbm', 'xgboost'] = 'lightgbm'
    days: int = Field(default=30, ge=1, le=180)
    initial_cash: float | None = Field(default=None, gt=0)
    fee: float | None = Field(default=None, ge=0, le=.01)
    slippage: float | None = Field(default=None, ge=0, le=.01)
    threshold: float | None = Field(default=None, ge=.5, le=1)
    optimize: bool = False


class OrderPatch(BaseModel):
    sl: float | None = None
    tp: float | None = None

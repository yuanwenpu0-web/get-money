from __future__ import annotations

from pydantic import BaseModel, Field


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
    interval: str | None = None
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
    schedule_enabled: bool | None = None
    schedule_seconds: int | None = None
    train_horizon: int | None = None
    auto_train_days: int | None = None


class StrategyUpdate(BaseModel):
    enabled: bool | None = None
    params: dict | None = None


class SyncRequest(BaseModel):
    symbol: str
    interval: str = "1m"
    days: int = 30
    source: str = "nofx"  # nofx / binance


class PreviewRequest(BaseModel):
    symbol: str
    interval: str = "1m"
    strategy: str = "ml_ensemble"


class TrainRequest(BaseModel):
    symbol: str
    interval: str = "1m"
    algo: str = "lightgbm"
    horizon: int = 5
    days: int = 30


class BacktestRequest(BaseModel):
    symbol: str
    interval: str = "1m"
    strategy: str = "ml_ensemble"
    algo: str = "lightgbm"
    days: int = 30
    initial_cash: float | None = None
    fee: float | None = None
    slippage: float | None = None
    threshold: float | None = None


class OrderPatch(BaseModel):
    sl: float | None = None
    tp: float | None = None

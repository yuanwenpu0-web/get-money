from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import pandas as pd
from sqlalchemy.orm import Session

from ..features.indicators import build_features
from ..market import metrics as metrics_mod
from ..market.data_service import MarketDataService
from ..models import Signal, StrategyConfig
from ..state import all_state
from ..utils import interval_ms, now_ms, round_price
from .registry import StrategySpec, get_spec
from .rules import evaluate_signals, prepare_frame, resolve_params


def _fingerprint(params: dict) -> str:
    """参数指纹：不同池参数会产生不同特征，缓存键必须带上它。"""
    return json.dumps(params, sort_keys=True, default=str)


@dataclass
class TradeSignal:
    symbol: str
    interval: str
    strategy: str
    pool: str
    ts: int
    price: float
    direction: str
    score: float
    prob_up: float
    sl: float
    tp: float
    leverage: int
    position_pct: float
    atr_pct: float
    reason: dict = field(default_factory=dict)
    # 产生该信号的绑定参数（进程内透传给撮合，避免再按策略配置解析一次而用错池参数）
    params: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("params", None)
        return data


class StrategyEngine:
    def __init__(self, db: Session, *, cycle_snapshot: bool = False):
        """cycle_snapshot 仅供单轮巡检复用输入；下一轮必须创建新实例。"""
        self.db = db
        self.service = MarketDataService(db)
        self.cycle_snapshot = cycle_snapshot
        # 缓存键 (symbol, interval, limit, 参数指纹)：同一币种在不同池下参数不同、
        # 特征也不同，不能只按 (symbol, interval) 缓存。
        self._frames: dict[tuple[str, str, int, str], pd.DataFrame | None] = {}
        self._metrics: dict[str, dict] = {}
        self._configs: dict[str, StrategyConfig | None] = {}
        self._state: dict | None = None

    def _get_config(self, code: str) -> StrategyConfig | None:
        if not self.cycle_snapshot or code not in self._configs:
            config = self.db.query(StrategyConfig).filter(StrategyConfig.code == code).first()
            if not self.cycle_snapshot:
                return config
            self._configs[code] = config
        return self._configs[code]

    def _get_state(self) -> dict:
        if not self.cycle_snapshot:
            return all_state(self.db)
        if self._state is None:
            self._state = all_state(self.db)
        return self._state

    def _prepare_features(
        self, symbol: str, interval: str, limit: int = 400
    ) -> pd.DataFrame | None:
        frame = self.service.load_frame(symbol, interval, limit=limit)
        step = interval_ms(interval)
        frame = (
            frame.loc[(frame.index.as_unit("ms").asi8 + step) <= now_ms()]
            if not frame.empty
            else frame
        )
        if len(frame) < 120:
            return None
        # Indicators restart after missing candles, as they do in the research data.
        discontinuity = (
            frame.index.to_series().diff().dt.total_seconds().mul(1000).ne(step)
        )
        frame = frame.loc[discontinuity.cumsum().eq(discontinuity.cumsum().iloc[-1])]
        if len(frame) < 120:
            return None
        # 资金费特征：加载窗口前 90 天的历史，保证 z-score/24h 累计从窗口起点就有数
        frame_start_ms = int(frame.index[0].timestamp() * 1000)
        funding = self.service.load_funding_frame(symbol, frame_start_ms - 90 * 86_400_000, now_ms())
        return build_features(frame, with_label=False, funding=funding)

    def _frame(
        self,
        symbol: str,
        interval: str,
        spec: StrategySpec,
        params: dict,
    ) -> pd.DataFrame | None:
        """取指标帧并按策略补上下文；同轮同参数的重复请求走缓存。"""
        limit = spec.lookback(params)
        key = (symbol, interval, limit, _fingerprint(params))
        if self.cycle_snapshot and key in self._frames:
            return self._frames[key]
        data = self._prepare_features(symbol, interval, limit)
        if data is not None:
            data = prepare_frame(data, spec.code, params, symbol=symbol, service=self.service)
        if self.cycle_snapshot:
            # 巡检按「绑定 × 币种」遍历，只保留当前序列，避免宇宙级缓存堆积
            self._frames.clear()
            self._frames[key] = data
        return data

    def evaluate(
        self,
        symbol: str,
        interval: str,
        strategy_code: str,
        save: bool = False,
        *,
        params: dict | None = None,
        pool: str = "",
        respect_enabled: bool = True,
    ) -> TradeSignal | None:
        """评估单个 (策略, 池, 币种, 周期) 的信号。

        `params` 一般由 `bindings.active_bindings()` 解析好传入（已含池参数覆盖）；
        不传时按策略配置 + 全局设置自行解析，供预演等临时调用使用。
        `respect_enabled=False` 用于「预演」：策略总开关关闭时也允许算一次看看。
        """
        if strategy_code in (None, "auto"):
            return None
        spec = get_spec(strategy_code)
        config = self._get_config(strategy_code)
        if respect_enabled and config is not None and not config.enabled:
            return None
        if params is None:
            merged = json.loads(config.params or "{}") if config else {}
            params = resolve_params(
                strategy_code, self._get_state(), merged or None, interval=interval
            )
        data = self._frame(symbol, interval, spec, params)
        if data is None:
            return None
        step = interval_ms(interval)
        row = data.iloc[-1]
        if not all(
            math.isfinite(float(row[key])) and row[key] > 0
            for key in ("open", "high", "low", "close")
        ):
            return None
        if row["high"] < max(row["open"], row["close"], row["low"]) or row["low"] > min(
            row["open"], row["close"], row["high"]
        ):
            return None
        # 观测接入：把最新多源指标快照附在 reason 上（采集用），不影响任何决策。
        if self.cycle_snapshot:
            if symbol not in self._metrics:
                self._metrics[symbol] = metrics_mod.load_latest_metrics(self.db, symbol) or {}
            metrics_features = self._metrics[symbol]
        else:
            metrics_features = metrics_mod.load_latest_metrics(self.db, symbol) or {}
        decision = evaluate_signals(data.iloc[-1:], strategy_code, params, symbol=symbol).iloc[0]
        ts = int(data.index[-1].timestamp() * 1000)
        stale = now_ms() - (ts + step) > params["max_signal_age_bars"] * step
        direction = 0 if stale else int(decision.direction)
        price = float(row["close"])
        atr_pct = float(row["atr_pct"]) if math.isfinite(float(row["atr_pct"])) else 0.0
        observation = spec.describe(row) if spec.describe else {}
        signal = TradeSignal(
            symbol=symbol,
            interval=interval,
            strategy=strategy_code,
            pool=pool,
            ts=ts,
            price=round_price(price),
            direction={1: "LONG", -1: "SHORT", 0: "FLAT"}[direction],
            score=round(float(decision.score), 4),
            prob_up=round(float(decision.prob_up), 4),
            sl=round_price(price * (1 - direction * decision.sl_pct))
            if direction
            else 0.0,
            tp=round_price(price * (1 + direction * decision.tp_pct))
            if direction
            else 0.0,
            leverage=int(decision.leverage) if direction else 0,
            position_pct=float(decision.position_pct) if direction else 0.0,
            atr_pct=round(atr_pct, 6),
            reason={
                "signed": round(float(decision.signed), 4),
                "atr": round(atr_pct * price, 4),
                "block_reason": "stale_data" if stale else str(decision.block_reason),
                "risk_pct": float(decision.risk_pct) if direction else 0.0,
                "net_reward_risk": float(decision.net_reward_risk)
                if math.isfinite(float(decision.net_reward_risk))
                else 0.0,
                "pool": pool,
                "metrics": metrics_features,  # 观测用多源指标快照（附件，不影响决策）
                **observation,
            },
            params=dict(params),
        )
        if save:
            payload = signal.to_dict()
            payload.pop("atr_pct")
            payload["reason"] = json.dumps(signal.reason, ensure_ascii=False)
            self.db.add(Signal(**payload))
            self.db.commit()
        return signal

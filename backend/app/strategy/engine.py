from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import pandas as pd
from sqlalchemy.orm import Session

from ..features.indicators import build_features
from ..market.data_service import MarketDataService
from ..ml.predictor import latest_model, predict_row
from ..models import Signal, StrategyConfig
from ..state import all_state
from ..utils import interval_ms, now_ms, round_price
from .rules import evaluate_signals, resolve_params


@dataclass
class TradeSignal:
    symbol: str
    interval: str
    strategy: str
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

    def to_dict(self) -> dict:
        return asdict(self)


class StrategyEngine:
    def __init__(self, db: Session):
        self.db = db
        self.service = MarketDataService(db)

    def _get_config(self, code: str) -> StrategyConfig | None:
        return self.db.query(StrategyConfig).filter(StrategyConfig.code == code).first()

    def evaluate(
        self, symbol: str, interval: str, strategy_code: str, save: bool = False
    ) -> TradeSignal | None:
        config = self._get_config(strategy_code)
        if config is not None and not config.enabled:
            return None
        params = resolve_params(
            strategy_code,
            all_state(self.db),
            json.loads(config.params or "{}") if config else None,
        )
        frame = self.service.load_frame(symbol, interval, limit=400)
        step = interval_ms(interval)
        frame = (
            frame.loc[(frame.index.asi8 // 1_000_000 + step) <= now_ms()]
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
        data = build_features(frame, with_label=False)
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
        probability, model = None, None
        if strategy_code == "ml_ensemble":
            model = latest_model(self.db, symbol, interval)
            if model is not None:
                probability = pd.Series(
                    [predict_row(model, row)], index=data.index[-1:]
                )
        decision = evaluate_signals(
            data.iloc[-1:], strategy_code, params, probability
        ).iloc[0]
        ts = int(data.index[-1].timestamp() * 1000)
        stale = now_ms() - (ts + step) > params["max_signal_age_bars"] * step
        direction = 0 if stale else int(decision.direction)
        price = float(row["close"])
        atr_pct = float(row["atr_pct"]) if math.isfinite(float(row["atr_pct"])) else 0.0
        signal = TradeSignal(
            symbol=symbol,
            interval=interval,
            strategy=strategy_code,
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
                "model_id": model.id if model is not None else 0,
                "algo": model.algo if model is not None else "rule",
            },
        )
        if save:
            payload = signal.to_dict()
            payload.pop("atr_pct")
            payload["reason"] = json.dumps(signal.reason, ensure_ascii=False)
            self.db.add(Signal(**payload))
            self.db.commit()
        return signal

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


def _sign(value: float) -> float:
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class StrategyEngine:
    def __init__(self, db: Session):
        self.db = db
        self.service = MarketDataService(db)

    def _get_config(self, code: str) -> StrategyConfig | None:
        return self.db.query(StrategyConfig).filter(StrategyConfig.code == code).first()

    def _merge_params(self, config: StrategyConfig | None, state: dict) -> dict:
        params = {}
        if config:
            try:
                params = json.loads(config.params or "{}")
            except ValueError:
                params = {}
        defaults = {
            "min_score": state.get("min_score", 0.55),
            "heavy_score": state.get("heavy_score", 0.75),
            "base_position_pct": state.get("base_position_pct", 0.08),
            "max_position_pct": state.get("max_position_pct", 0.25),
            "max_leverage": state.get("max_leverage", 10),
            "sl_atr_mult": state.get("sl_atr_mult", 1.5),
            "tp_atr_mult": state.get("tp_atr_mult", 2.5),
            "trailing_atr_mult": state.get("trailing_atr_mult", 1.2),
            "max_hold_bars": state.get("max_hold_bars", 120),
            "min_sl_pct": state.get("min_sl_pct", 0.0025),
            "min_tp_pct": state.get("min_tp_pct", 0.005),
        }
        defaults.update(params)
        return defaults

    def _direction_score(self, code: str, row: pd.Series, config: StrategyConfig | None, params: dict) -> tuple[float, float, dict]:
        """返回 (有向强度 [-1,1], 上涨概率, 说明)。"""
        st_dir = float(row.get("supertrend_dir", 0) or 0)
        ema_dir = _sign(float(row.get("ema_ratio_9_21", 0) or 0))
        macd_dir = _sign(float(row.get("macd_hist_norm", 0) or 0))
        rsi = float(row.get("rsi_14", 50) or 50)

        if code == "mean_reversion":
            rsi_low = float(params.get("rsi_low", 30))
            rsi_high = float(params.get("rsi_high", 70))
            signed = 0.0
            if rsi < rsi_low:
                signed = (rsi_low - rsi) / max(rsi_low, 1) * 0.9
            elif rsi > rsi_high:
                signed = -(rsi - rsi_high) / max(100 - rsi_high, 1) * 0.9
            prob_up = 0.5 + signed / 2
            return _clamp(signed, -1, 1), prob_up, {"rsi": rsi, "mode": "mean_reversion"}

        if code == "trend_follow":
            adx = float(row.get("adx_14", 0) or 0)
            adx_filter = float(params.get("adx_filter", 18))
            strength = 1.0 if (math.isnan(adx) or adx >= adx_filter) else 0.5
            signed = (0.5 * st_dir + 0.3 * ema_dir + 0.2 * macd_dir) * strength
            prob_up = 0.5 + signed / 2
            return _clamp(signed, -1, 1), prob_up, {"supertrend": st_dir, "ema": ema_dir, "macd": macd_dir, "adx": adx}

        # ml_ensemble：模型概率为主，规则做确认
        model = latest_model(self.db, str(row.get("_symbol")), str(row.get("_interval")))
        if model is not None:
            prob_up = predict_row(model, row)
            ml_signal = (prob_up - 0.5) * 2
            reason = {"model_id": model.id, "algo": model.algo, "prob_up": round(prob_up, 4)}
        else:
            ml_signal = 0.5 * st_dir + 0.5 * macd_dir
            prob_up = 0.5 + ml_signal / 2
            reason = {"model_id": 0, "algo": "rule_fallback", "prob_up": round(prob_up, 4)}

        weight = float(params.get("ml_weight", 0.6))
        signed = weight * ml_signal + (1 - weight) * (0.5 * st_dir + 0.3 * ema_dir + 0.2 * macd_dir)
        reason.update({"supertrend": st_dir, "ema": ema_dir, "macd": macd_dir, "rsi": rsi})
        return _clamp(signed, -1, 1), prob_up, reason

    def _position_pct(self, score: float, params: dict) -> float:
        base = float(params["base_position_pct"])
        peak = float(params["max_position_pct"])
        heavy = float(params["heavy_score"])
        min_score = float(params["min_score"])
        if score >= heavy:
            return peak
        if score <= min_score:
            return 0.0
        ratio = (score - min_score) / max(heavy - min_score, 1e-6)
        return round(base + ratio * (peak * 0.7 - base), 4)

    def _leverage(self, atr_pct: float, score: float, params: dict) -> int:
        max_lev = int(params["max_leverage"])
        if atr_pct <= 0:
            return 1
        raw = 0.015 / max(atr_pct, 1e-4)
        confidence = _clamp((score - 0.5) / 0.5, 0.0, 1.0)
        leverage = round(raw * (0.6 + 0.4 * confidence))
        return int(_clamp(leverage, 1, max_lev))

    def evaluate(self, symbol: str, interval: str, strategy_code: str, save: bool = False) -> TradeSignal | None:
        config = self._get_config(strategy_code)
        if config is not None and not config.enabled:
            return None
        state = all_state(self.db)
        params = self._merge_params(config, state)

        frame = self.service.load_frame(symbol, interval, limit=400)
        if len(frame) < 120:
            return None
        # 丢弃尚未收盘的K线
        last_open = int(frame.index[-1].timestamp() * 1000)
        if last_open + interval_ms(interval) > now_ms() + 1000:
            frame = frame.iloc[:-1]
        if len(frame) < 120:
            return None

        data = build_features(frame, with_label=False).dropna(subset=["atr_pct", "rsi_14"])
        if data.empty:
            return None
        row = data.iloc[-1].copy()
        row["_symbol"] = symbol
        row["_interval"] = interval

        signed, prob_up, reason = self._direction_score(strategy_code, row, config, params)
        score = round(0.5 + 0.5 * abs(signed), 4)
        atr_pct = float(row["atr_pct"])
        atr = atr_pct * float(row["close"])
        price = float(row["close"])

        min_score = float(params["min_score"])
        below_threshold = score < min_score or abs(signed) < 1e-9
        position_pct = self._position_pct(score, params) if not below_threshold else 0.0
        if below_threshold or position_pct <= 0:
            direction = "FLAT"
            position_pct = 0.0
            leverage = 0
            sl = tp = 0.0
        else:
            direction = "LONG" if signed > 0 else "SHORT"
            leverage = self._leverage(atr_pct, score, params)
            sl_dist = max(atr * float(params["sl_atr_mult"]), price * float(params.get("min_sl_pct", 0.0025)))
            tp_dist = max(atr * float(params["tp_atr_mult"]), price * float(params.get("min_tp_pct", 0.005)))
            if direction == "LONG":
                sl = round_price(price - sl_dist)
                tp = round_price(price + tp_dist)
            else:
                sl = round_price(price + sl_dist)
                tp = round_price(price - tp_dist)

        ts = int(data.index[-1].timestamp() * 1000)
        signal = TradeSignal(
            symbol=symbol,
            interval=interval,
            strategy=strategy_code,
            ts=ts,
            price=round_price(price),
            direction=direction,
            score=score,
            prob_up=round(prob_up, 4),
            sl=sl,
            tp=tp,
            leverage=leverage,
            position_pct=position_pct,
            atr_pct=round(atr_pct, 6),
            reason={**reason, "signed": round(signed, 4), "atr": round(atr, 4)},
        )

        if save:
            self.db.add(
                Signal(
                    symbol=symbol,
                    interval=interval,
                    strategy=strategy_code,
                    ts=ts,
                    price=signal.price,
                    direction=signal.direction,
                    score=signal.score,
                    prob_up=signal.prob_up,
                    sl=signal.sl,
                    tp=signal.tp,
                    leverage=signal.leverage,
                    position_pct=signal.position_pct,
                    reason=json.dumps(signal.reason, ensure_ascii=False),
                )
            )
            self.db.commit()
        return signal

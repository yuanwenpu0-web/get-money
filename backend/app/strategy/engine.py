from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import pandas as pd
from sqlalchemy.orm import Session

from ..features.absorption import add_absorption_context, load_lower_tf_delta
from ..features.indicators import build_features
from ..market.data_service import MarketDataService
from ..market import metrics as metrics_mod
from ..models import Signal, StrategyConfig
from ..state import all_state
from ..utils import interval_ms, now_ms, round_price
from .rules import evaluate_signals, resolve_params


def _absorption_lookback(params: dict) -> int:
    """吸收反转的回看长度：|delta| 百分位需要 delta_window 根窗口，EMA(trend_len) 需约 3×跨度预热。
    取两者的较大者并留 1200 根底线，避免交易日不足时百分位/EMA 在末根仍失真。"""
    return max(1200, int(params.get("delta_window", 500)) + 3 * int(params.get("trend_len", 200)))


def _absorption_reason(row) -> dict:
    """吸收信号的观测快照：delta/支配度/百分位阈值、吸收极值与待穿越关卡（不影响决策）。"""
    def value(name):
        try:
            number = float(row[name])
        except (KeyError, TypeError, ValueError):
            return 0.0
        return number if math.isfinite(number) else 0.0

    return {
        "abs_delta": round(value("abs_delta"), 4),
        "abs_dominance": round(value("abs_dominance"), 4),
        "abs_percentile": round(value("abs_percentile"), 4),
        "abs_wick_ratio": round(value("abs_wick_ratio"), 6),
        "abs_atr": round_price(value("abs_atr")),
        "structure_extreme": round_price(value("abs_extreme")),
        "confirm_level": round_price(value("abs_level")),
        "signal_dir": value("abs_signal"),
        "stop_price": round_price(value("abs_sl")),
    }


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
    def __init__(self, db: Session, *, cycle_snapshot: bool = False):
        """cycle_snapshot 仅供单轮巡检复用输入；下一轮必须创建新实例。"""
        self.db = db
        self.service = MarketDataService(db)
        self.cycle_snapshot = cycle_snapshot
        self._features: dict[tuple[str, str, int], pd.DataFrame | None] = {}
        self._mtf_features: dict[tuple[str, str, str], pd.DataFrame] = {}
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

    def evaluate(
        self, symbol: str, interval: str, strategy_code: str, save: bool = False
    ) -> TradeSignal | None:
        """评估单个 symbol 上指定策略的信号（策略必须显式给出）。

        历史版本支持 strategy_code="auto"/None 走币级路由在多策略间挑选；仓库收敛到
        单一策略后该层已删除，调用方一律显式传入策略代码。
        """
        if strategy_code in (None, "auto"):
            return None
        config = self._get_config(strategy_code)
        if config is not None and not config.enabled:
            return None
        merged_params = json.loads(config.params or "{}") if config else {}
        state = self._get_state()
        params = resolve_params(
            strategy_code, state, merged_params or None,
            interval=interval,
        )
        # 吸收上下文：|delta| 百分位要 delta_window 根窗口、EMA(trend_len) 要 3 倍跨度预热
        limit = _absorption_lookback(params)
        key = (symbol, interval, limit)
        if self.cycle_snapshot:
            if key not in self._features:
                # 巡检按币种遍历，只保留当前序列，避免全量宇宙积累数百份指标帧。
                self._features.clear()
                self._mtf_features.clear()
                self._metrics.clear()
                self._features[key] = self._prepare_features(symbol, interval, limit)
            data = self._features[key]
        else:
            data = self._prepare_features(symbol, interval, limit)
        if data is None:
            return None
        # 吸收上下文整帧富化：delta / 滚动百分位 / c1..c5 / 确认状态机（纯因果）
        context_key = (symbol, interval)
        if self.cycle_snapshot and context_key in self._mtf_features:
            data = self._mtf_features[context_key]
        else:
            # 优先用脚本的 lower-tf intrabar delta；拿不到数据自动降级 bar shape 估计
            data = add_absorption_context(
                data, params, load_lower_tf_delta(self.service, symbol, data.index)
            )
            if self.cycle_snapshot:
                self._mtf_features[context_key] = data
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
        # 观测接入：把最新多源指标快照附在 reason 上（迭代 7 采集），不影响任何决策。
        if self.cycle_snapshot:
            if symbol not in self._metrics:
                self._metrics[symbol] = metrics_mod.load_latest_metrics(self.db, symbol) or {}
            metrics_features = self._metrics[symbol]
        else:
            metrics_features = metrics_mod.load_latest_metrics(self.db, symbol) or {}
        decision = evaluate_signals(data.iloc[-1:], strategy_code, params).iloc[0]
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
                "metrics": metrics_features,  # 观测用多源指标快照（附件，不影响决策）
                **_absorption_reason(row),
            },
        )
        if save:
            payload = signal.to_dict()
            payload.pop("atr_pct")
            payload["reason"] = json.dumps(signal.reason, ensure_ascii=False)
            self.db.add(Signal(**payload))
            self.db.commit()
        return signal

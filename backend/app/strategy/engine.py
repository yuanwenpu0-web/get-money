from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import pandas as pd
from sqlalchemy.orm import Session

from ..features.absorption import add_absorption_context, load_lower_tf_delta
from ..features.indicators import build_features
from ..features.mtf import add_mtf_trend
from ..features.pullback import add_pullback_context
from ..market.data_service import MarketDataService
from ..ml.predictor import latest_model, predict_row
from ..market import metrics as metrics_mod
from ..models import Signal, StrategyConfig
from ..state import all_state
from ..utils import interval_ms, now_ms, round_price
from .coin_router import get_route
from .rules import evaluate_signals, resolve_params

# 需要额外多周期上下文的策略（引擎负责整帧富化，规则层只读末根）
CONTEXT_CODES = ("mtf_resonance", "strong_pullback", "absorption_reversal")
# 强回调的 15m 回看长度：1200 根 = 300 小时 ≈ 75 个 4h 桶，足够 4h ADX(14) 与 ema50 预热
PULLBACK_LOOKBACK = 1200


def _absorption_lookback(params: dict) -> int:
    """吸收反转的回看长度：|delta| 百分位需要 delta_window 根窗口，EMA(trend_len) 需约 3×跨度预热。
    取两者的较大者并留 1200 根底线，避免交易日不足时百分位/EMA 在末根仍失真。"""
    return max(1200, int(params.get("delta_window", 500)) + 3 * int(params.get("trend_len", 200)))


def _pullback_reason(row) -> dict:
    """强回调信号的观测快照：4h 趋势强度、EMA9/AVWAP 位置与回调进度（不影响决策）。"""
    def value(name):
        try:
            number = float(row[name])
        except (KeyError, TypeError, ValueError):
            return 0.0
        return number if math.isfinite(number) else 0.0

    return {
        "pb_htf_dir": value("pb_dir_4h"),
        "pb_htf_adx": value("pb_adx_4h"),
        "ema9": round_price(value("ema9")),
        "avwap": round_price(value("avwap")),
        "avwap_dist": round(value("avwap_dist"), 6),
        "bars_since_touch": value("pb_bars_since_touch"),
        "pullback_depth": round(value("pb_depth"), 6),
        "structure_stop": round_price(value("pb_stop_ref")),
    }


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
        self, symbol: str, interval: str, strategy_code: str | None, save: bool = False
    ) -> TradeSignal | None:
        """评估单个 symbol 的策略信号。

        strategy_code=None 或 "auto" 时启用币级路由 — 按 symbol 查表选最优策略+参数。
        显式指定 strategy_code 时（UI 手动选 / 已在执行的订单）路由不介入，行为与历史一致。
        """
        # --- 币级路由（仅 auto/None 触发） ---
        use_route = strategy_code in (None, "auto")
        route_params: dict = {}
        if use_route:
            route = get_route(symbol)
            if route is None:
                return None  # 黑名单 / 宇宙外 → 不产生信号
            strategy_code = route["strategy"]
            route_params = route.get("params", {})

        config = self._get_config(strategy_code)
        if config is not None and not config.enabled:
            return None
        # 合并：DB 里 StrategyConfig.params 为主，路由 params 作为补充
        merged_params = json.loads(config.params or "{}") if config else {}
        for k, v in route_params.items():
            merged_params.setdefault(k, v)  # DB 参数优先级更高
        state = self._get_state()
        params = resolve_params(
            strategy_code, state, merged_params or None,
            interval=interval,
        )
        # 强回调要 4h 上下文：15m 下 400 根只有 100 小时（≈25 个 4h 桶），
        # 不足以让 4h ADX(14) 稳定预热，故按策略加长回看窗口。
        limit = 400
        if strategy_code == "strong_pullback":
            limit = PULLBACK_LOOKBACK
        elif strategy_code == "absorption_reversal":
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
        # 只有多周期策略消费这些列，其他策略直接复用基础指标。
        if strategy_code in CONTEXT_CODES:
            context_key = (symbol, interval, strategy_code)
            if self.cycle_snapshot and context_key in self._mtf_features:
                data = self._mtf_features[context_key]
            else:
                if strategy_code == "mtf_resonance":
                    data = add_mtf_trend(data)
                elif strategy_code == "absorption_reversal":
                    # 优先用脚本的 lower-tf intrabar delta；拿不到数据自动降级 bar shape 估计
                    data = add_absorption_context(
                        data, params, load_lower_tf_delta(self.service, symbol, data.index)
                    )
                else:
                    data = add_pullback_context(data, params)
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
        probability, model = None, None
        if strategy_code == "ml_ensemble":
            model = latest_model(self.db, symbol, interval)
            if model is not None:
                probability = pd.Series(
                    [predict_row(model, row)], index=data.index[-1:]
                )
        # 观测接入：附加最新多源指标特征（迭代 7）。开关关闭时仅附带、不改变决策；
        # 开关开启时把特征作为 market_context 透传给规则引擎（当前规则不消费，行为不变）。
        if self.cycle_snapshot:
            if symbol not in self._metrics:
                self._metrics[symbol] = metrics_mod.load_latest_metrics(self.db, symbol) or {}
            metrics_features = self._metrics[symbol]
        else:
            metrics_features = metrics_mod.load_latest_metrics(self.db, symbol) or {}
        decision = evaluate_signals(
            data.iloc[-1:], strategy_code, params, probability,
            market_context=metrics_features if state.get("metrics_assist_enabled") else None,
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
                "metrics": metrics_features,  # 观测用多源指标快照（附件，不影响决策）
                **(_pullback_reason(row) if strategy_code == "strong_pullback" else {}),
                **(
                    _absorption_reason(row)
                    if strategy_code == "absorption_reversal"
                    else {}
                ),
            },
        )
        if save:
            payload = signal.to_dict()
            payload.pop("atr_pct")
            payload["reason"] = json.dumps(signal.reason, ensure_ascii=False)
            self.db.add(Signal(**payload))
            self.db.commit()
        return signal

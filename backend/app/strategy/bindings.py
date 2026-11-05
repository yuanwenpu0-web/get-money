"""策略 × 币种池绑定：把「哪套参数、跑哪些币、用什么周期、怎么出场」变成一等公民。

一条绑定 `(strategy_code, pool_code, interval)` 携带一整套参数覆盖 + 一套出场口径，
于是同一个策略可以同时有：

- `absorption_reversal × major × 15m`（主流，脚本默认灵敏度）
- `absorption_reversal × alt   × 5m`（山寨，更宽的 delta 门槛 + 更紧的止损）
- `absorption_reversal × alt   × 1m`（短线，只做多）

参数三层合并：全局设置 < 策略默认 < 本条绑定的 params（见 `rules.resolve_params`）。
再往上还有一层**币种级覆盖**（`symbol_params` 表）：同一绑定内，某些币种可以带
自己寻优得到的参数，见 `params_for()`。

出场口径随绑定走（`exit_method` / `support_indicator` / `exit_params`），
以保留键 `exit` 注入有效参数，回测、模拟盘、巡检、订单管理读到的是同一份配置。

同一币种可能同时落进多个池（如 BTC 既在 major 又在 all）：各绑定独立评估，
巡检按评分择优只开一仓，不做互斥校验，避免池组合被卡死。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ..config import SUPPORTED_INTERVALS
from ..models import AppState, StrategyBinding, StrategyConfig, SymbolPool
from ..state import all_state
from .exits import EXIT_KEY, label as exit_label
from .pools import load_pool_spec, pool_symbols
from .registry import get_spec, has_spec
from .rules import resolve_params
from .symbol_params import get_override, load_map


def ensure_execution_supported(strategy: str, params: dict) -> None:
    """巡检和模拟盘不支持需要专用撮合的策略模式。"""
    if strategy == "opening_candle_breakout":
        raise ValueError("此策略仅支持专用撮合回测，巡检和模拟盘尚不支持挂单/会话强平")
    if strategy == "supertrend_current" and params.get("cl_enabled"):
        raise ValueError("订单闭环模式仅支持专用撮合回测，巡检和模拟盘尚不支持挂单状态机")


@dataclass(frozen=True)
class ResolvedBinding:
    """已解析好的绑定：参数合并完毕、成员已展开，可直接拿去评估。"""

    id: int
    strategy: str
    pool: str
    interval: str
    priority: int
    params: dict
    symbols: tuple[str, ...]
    # 该「策略 × 周期」下的币种覆盖（symbol → 覆盖层）。巡检一次展开几百个
    # (币种, 绑定) 对，逐条查库会把 SQL 往返变成瓶颈，所以在解析绑定时一次性载入。
    symbol_overrides: dict = field(default_factory=dict)
    # 本条绑定的出场口径（已归一化）；`params[EXIT_KEY]` 是同一份内容的解析结果，
    # 这里单独留一份给「巡检汇总 / 预演」直接展示，不用再翻参数。
    exit: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.strategy, self.pool, self.interval)

    @property
    def exit_label(self) -> str:
        return exit_label(self.exit)

    def params_for(self, symbol: str) -> dict:
        """单个币种的最终参数：绑定参数 + 该币种的覆盖。"""
        override = self.symbol_overrides.get(symbol)
        if not override:
            ensure_execution_supported(self.strategy, self.params)
            return self.params
        if not isinstance(override, dict) or {"fee_rate", "slippage"} & override.keys():
            raise ValueError(f"币种 {symbol} 的参数覆盖无效或包含成本")
        layer = {
            key: value
            for key, value in {**self.params, **override}.items()
            if key not in ("fee_rate", "slippage")
        }
        resolved = resolve_params(self.strategy, {}, layer, interval=self.interval)
        resolved["fee_rate"] = self.params["fee_rate"]
        resolved["slippage"] = self.params["slippage"]
        ensure_execution_supported(self.strategy, resolved)
        return resolved


def exit_config_of(row: StrategyBinding) -> dict | None:
    """绑定行上的出场配置 → `{"method","support","params"}`；全默认时返回 None。

    返回 None 表示「不注入」，让策略配置里可能存在的 `exit` 键仍然生效；
    只要绑定的三列里有一项非默认，就以绑定列为准（列覆盖 JSON 参数）。
    """
    method = (row.exit_method or "strategy").strip() or "strategy"
    support = (row.support_indicator or "").strip()
    try:
        parsed = json.loads(row.exit_params or "{}")
    except ValueError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    if method == "strategy" and not support and not parsed:
        return None
    return {"method": method, "support": support, "params": parsed}


def binding_params(db: Session, binding: StrategyBinding, state: dict | None = None) -> dict:
    """单条绑定的有效参数（三层合并 + 校验），不含任何币种级覆盖。

    币种级覆盖在本层之上，见 `ResolvedBinding.params_for()` / `resolve_order_params()`。
    出场口径以保留键 `exit` 注入，并通过 `resolve_params` 一并校验。
    """
    overrides = json.loads(binding.params or "{}")
    if not isinstance(overrides, dict):
        raise ValueError("绑定参数必须是 JSON 对象")
    config = db.query(StrategyConfig).filter(StrategyConfig.code == binding.strategy_code).first()
    strategy_params = json.loads(config.params or "{}") if config else {}
    if not isinstance(strategy_params, dict):
        raise ValueError("策略参数必须是 JSON 对象")
    overrides = {**strategy_params, **overrides}
    exit_config = exit_config_of(binding)
    if exit_config is not None:
        overrides[EXIT_KEY] = exit_config
    return resolve_params(
        binding.strategy_code,
        state if state is not None else all_state(db),
        overrides or None,
        interval=binding.interval,
    )


def validate_binding(
    db: Session,
    strategy_code: str,
    pool_code: str,
    interval: str,
    params: object,
    exit_config: dict | None = None,
) -> dict:
    """校验一条绑定定义并返回合并后的有效参数；非法配置抛 ValueError。"""
    if not has_spec(strategy_code):
        raise ValueError(f"未知策略: {strategy_code}")
    if db.query(SymbolPool).filter(SymbolPool.code == pool_code).first() is None:
        raise ValueError(f"未知币种池: {pool_code}")
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"不支持的周期: {interval}")
    if not isinstance(params, dict):
        raise ValueError("绑定参数必须是 JSON 对象")
    config = db.query(StrategyConfig).filter(StrategyConfig.code == strategy_code).first()
    strategy_params = json.loads(config.params or "{}") if config else {}
    if not isinstance(strategy_params, dict):
        raise ValueError("策略参数必须是 JSON 对象")
    layer = {**strategy_params, **params}
    if exit_config is not None:
        layer[EXIT_KEY] = exit_config
    return resolve_params(strategy_code, all_state(db), layer, interval=interval)


def validate_execution(db: Session, strategy: str, interval: str, params: dict) -> None:
    """校验绑定底座及所有可能适用的币种覆盖。"""
    ensure_execution_supported(strategy, params)
    overrides = {symbol: value for (code, symbol, iv), value in load_map(db, strategy, interval, strict=True).items()
                 if code == strategy and iv == interval}
    binding = ResolvedBinding(0, strategy, "", interval, 0, params, (), overrides)
    for symbol in overrides:
        binding.params_for(symbol)


def active_bindings(
    db: Session,
    *,
    state: dict | None = None,
    allowed_symbols: list[str] | None = None,
) -> tuple[list[ResolvedBinding], list[str]]:
    """读取所有生效绑定，返回 `(bindings, errors)`。

    生效 = 绑定启用 **且** 策略已注册 **且** 池启用。
    单条绑定配置坏掉（手改库导致参数非法、池被删）只记错误、不影响其它绑定。
    """
    if db.get(AppState, "binding_enablement_rollback") is not None:
        return [], ["绑定启停迁移已回滚，新版巡检已停用"]
    state = state if state is not None else all_state(db)
    pools = {row.code: row for row in db.query(SymbolPool).filter(SymbolPool.enabled == 1).all()}
    rows = (
        db.query(StrategyBinding)
        .filter(StrategyBinding.enabled == 1)
        .order_by(StrategyBinding.priority.asc(), StrategyBinding.id.asc())
        .all()
    )

    bindings: list[ResolvedBinding] = []
    errors: list[str] = []
    # 覆盖表整体很小（正常只有几十行），一次取回再切片，避免逐币查库
    overrides_all = load_map(db, strict=True) if rows else {}
    for row in rows:
        if not has_spec(row.strategy_code):
            errors.append(f"{row.strategy_code}×{row.pool_code}×{row.interval}: 策略不存在")
            continue
        pool = pools.get(row.pool_code)
        if pool is None:
            errors.append(f"{row.strategy_code}×{row.pool_code}×{row.interval}: 币种池不存在或已停用")
            continue
        try:
            ensure_execution_supported(row.strategy_code, {})
            params = binding_params(db, row, state)
            ensure_execution_supported(row.strategy_code, params)
            members = pool_symbols(db, pool, allowed=allowed_symbols)
        except Exception as exc:  # 单条坏配置不拖垮整轮巡检
            errors.append(f"{row.strategy_code}×{row.pool_code}×{row.interval}: {exc}")
            continue
        if not members:
            errors.append(f"{row.strategy_code}×{row.pool_code}×{row.interval}: 币种池为空或被全局交易范围过滤")
            continue
        symbol_overrides = {
            symbol: overrides_all[(row.strategy_code, symbol, row.interval)]
            for symbol in members
            if (row.strategy_code, symbol, row.interval) in overrides_all
        }
        bindings.append(
            ResolvedBinding(
                id=row.id,
                strategy=row.strategy_code,
                pool=row.pool_code,
                interval=row.interval,
                priority=int(row.priority or 0),
                params=params,
                symbols=tuple(members),
                symbol_overrides=symbol_overrides,
                exit=params.get(EXIT_KEY, {}),
            )
        )
    return bindings, errors


def expand(bindings: list[ResolvedBinding]) -> list[tuple[str, ResolvedBinding]]:
    """展开成 `(symbol, binding)` 扁平列表，顺序按绑定优先级。"""
    return [(symbol, binding) for binding in bindings for symbol in binding.symbols]


def binding_overrides(
    db: Session, strategy: str, pool: str, interval: str, symbol: str = ""
) -> dict | None:
    """绑定的原始参数覆盖层（未与全局合并），传 symbol 时叠上该币种的覆盖。

    回测走这条：和实盘完全同一层参数，避免「回测用策略配置、实盘用绑定」的口径分裂。
    没有对应绑定也没有币种覆盖时返回 None，调用方回落到策略配置。
    """
    layer: dict | None = None
    if pool:
        row = (
            db.query(StrategyBinding)
            .filter(
                StrategyBinding.strategy_code == strategy,
                StrategyBinding.pool_code == pool,
                StrategyBinding.interval == interval,
            )
            .first()
        )
        if row is not None:
            try:
                parsed = json.loads(row.params or "{}")
            except ValueError:
                parsed = None
            if isinstance(parsed, dict) and parsed:
                layer = parsed
            exit_config = exit_config_of(row)
            if exit_config is not None:
                layer = {**(layer or {}), EXIT_KEY: exit_config}
    if symbol:
        override = get_override(db, strategy, symbol, interval)
        if override:
            layer = {**(layer or {}), **override}
    return layer


# 无版本快照早于这些功能；补入的值固定为旧行为，不能跟随当前默认值变化。
_LEGACY_SNAPSHOT_ADDITIONS = {
    "exit": {"method": "strategy", "support": "", "params": {}},
}
_LEGACY_SUPERTREND_ADDITIONS = {
    "cl_enabled": False, "cl_entry_mode": 1, "cl_tp_mode": 0,
    "cl_support_buffer_atr": 0.1, "cl_max_entry_distance_atr": 3.0,
    "cl_pending_expire_bars": 8, "cl_dynamic_activate_r": 1.0,
    "cl_trailing_atr_mult": 1.5, "cl_cooldown_bars": 0,
}


def encode_order_params(strategy: str, interval: str, params: dict) -> str:
    """校验并序列化完整有效参数，必须在任何撮合或外部下单前调用。"""
    if not isinstance(params, dict) or "fee_rate" not in params or "slippage" not in params:
        raise ValueError("订单参数快照不完整")
    state = {key: params[key] for key in ("fee_rate", "slippage")}
    layer = {key: value for key, value in params.items() if key not in state}
    resolved = resolve_params(strategy, state, layer, interval=interval)
    if resolved != params:
        raise ValueError("订单参数快照不完整或未归一化")
    return json.dumps({"version": 1, "params": params}, ensure_ascii=False, allow_nan=False)


def decode_order_params(strategy: str, interval: str, raw: str) -> dict:
    """读取版本化或无版本快照，仅补充明确登记的历史兼容字段。"""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"重复字段: {key}")
            result[key] = value
        return result

    try:
        snapshot = json.loads(raw, object_pairs_hook=unique_object,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        if not isinstance(snapshot, dict):
            raise ValueError("快照必须是 JSON 对象")
        if "version" in snapshot or "params" in snapshot:
            if set(snapshot) != {"version", "params"} or type(snapshot["version"]) is not int or snapshot["version"] != 1:
                raise ValueError("快照版本或格式不受支持")
            params = snapshot["params"]
        else:
            params = dict(snapshot)
            additions = dict(_LEGACY_SNAPSHOT_ADDITIONS)
            if strategy == "supertrend_current" and not params.get("cl_enabled", False):
                additions.update(_LEGACY_SUPERTREND_ADDITIONS)
            for key, value in additions.items():
                if key not in params:
                    params[key] = value.copy() if isinstance(value, dict) else value
        encode_order_params(strategy, interval, params)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"订单参数快照无效: {exc}") from exc
    return params


def resolve_order_params(
    db: Session,
    strategy: str,
    interval: str,
    pool: str = "",
    state: dict | None = None,
    symbol: str = "",
) -> dict:
    """订单/信号所属绑定的有效参数（含该币种的覆盖层）。

    优先级：币种覆盖 > 绑定（strategy × pool × interval）> 策略配置行 > 策略默认。
    老订单没有 pool，或手工构造的信号没有对应绑定，都会回落到策略配置，行为与改造前一致。
    """
    layer: dict = {}
    config = db.query(StrategyConfig).filter(StrategyConfig.code == strategy).first()
    if config is not None:
        try:
            parsed = json.loads(config.params or "{}")
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            layer.update(parsed)
    if pool:
        row = (
            db.query(StrategyBinding)
            .filter(
                StrategyBinding.strategy_code == strategy,
                StrategyBinding.pool_code == pool,
                StrategyBinding.interval == interval,
            )
            .first()
        )
        if row is not None:
            try:
                parsed = json.loads(row.params or "{}")
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                layer.update(parsed)
            exit_config = exit_config_of(row)
            if exit_config is not None:
                layer[EXIT_KEY] = exit_config
    if symbol:
        layer.update(get_override(db, strategy, symbol, interval))
    return resolve_params(
        strategy,
        state if state is not None else all_state(db),
        layer or None,
        interval=interval,
    )


def describe_binding(
    db: Session,
    row: StrategyBinding,
    state: dict | None = None,
    *,
    with_members: bool = False,
) -> dict:
    """给 API/前端用的绑定描述：附有效参数、成员数、可读名称。"""
    payload = {
        "id": row.id,
        "strategy": row.strategy_code,
        "pool": row.pool_code,
        "interval": row.interval,
        "enabled": bool(row.enabled),
        "priority": int(row.priority or 0),
        "params": json.loads(row.params or "{}") or {},
        "exit_method": row.exit_method or "strategy",
        "support_indicator": row.support_indicator or "",
        "exit_params": json.loads(row.exit_params or "{}") or {},
        "updated_at": row.updated_at.isoformat() if row.updated_at else "",
    }
    try:
        payload["effective_params"] = binding_params(db, row, state)
        effective_exit = payload["effective_params"].get(EXIT_KEY, {})
    except Exception as exc:
        payload["effective_params"] = {}
        effective_exit = {}
        payload["error"] = str(exc)
    payload["exit"] = {
        **effective_exit,
        "label": exit_label(effective_exit),
        # 支撑位出场未过统计验证，前端必须显式标注，避免被当成已验证配置
        "verified": bool(effective_exit) and effective_exit.get("method") != "support_r",
    }

    pool = db.query(SymbolPool).filter(SymbolPool.code == row.pool_code).first()
    if pool is not None:
        payload["pool_name"] = pool.name
        try:
            members = pool_symbols(db, pool)
            payload["member_count"] = len(members)
            if with_members:
                payload["members"] = members
        except Exception:
            payload["member_count"] = 0
    else:
        payload["pool_name"] = row.pool_code
        payload["member_count"] = 0

    if has_spec(row.strategy_code):
        spec = get_spec(row.strategy_code)
        payload["strategy_name"] = spec.name
        payload["lookback"] = spec.lookback(payload["effective_params"] or spec.defaults)
    else:
        payload["strategy_name"] = row.strategy_code
    return payload


def pool_summary(db: Session) -> dict[str, dict]:
    """池代码 → 规范化的池定义与成员数，供前端下拉与校验提示。"""
    result: dict[str, dict] = {}
    for pool in db.query(SymbolPool).order_by(SymbolPool.id.asc()).all():
        spec = load_pool_spec(pool)
        try:
            count = len(pool_symbols(db, pool))
        except Exception:
            count = 0
        result[pool.code] = {
            "code": pool.code,
            "name": pool.name,
            "kind": spec["kind"],
            "rule": spec["rule"],
            "member_count": count,
            "enabled": bool(pool.enabled),
        }
    return result

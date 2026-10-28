"""策略 × 币种池绑定：把「哪套参数、跑哪些币、用什么周期」变成一等公民。

一条绑定 `(strategy_code, pool_code, interval)` 携带一整套参数覆盖，
于是同一个策略可以同时有：

- `absorption_reversal × major × 15m`（主流，脚本默认灵敏度）
- `absorption_reversal × alt   × 5m`（山寨，更宽的 delta 门槛 + 更紧的止损）
- `absorption_reversal × alt   × 1m`（短线，只做多）

参数三层合并：全局设置 < 策略默认 < 本条绑定的 params（见 `rules.resolve_params`）。

同一币种可能同时落进多个池（如 BTC 既在 major 又在 all）：各绑定独立评估，
巡检按评分择优只开一仓，不做互斥校验，避免池组合被卡死。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy.orm import Session

from ..config import SUPPORTED_INTERVALS
from ..models import StrategyBinding, StrategyConfig, SymbolPool
from ..state import all_state
from .pools import load_pool_spec, pool_symbols
from .registry import get_spec, has_spec
from .rules import resolve_params


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

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.strategy, self.pool, self.interval)


def binding_params(db: Session, binding: StrategyBinding, state: dict | None = None) -> dict:
    """单条绑定的有效参数（三层合并 + 校验）。"""
    overrides = json.loads(binding.params or "{}")
    if not isinstance(overrides, dict):
        raise ValueError("绑定参数必须是 JSON 对象")
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
    return resolve_params(strategy_code, all_state(db), params or None, interval=interval)


def active_bindings(
    db: Session,
    *,
    state: dict | None = None,
    allowed_symbols: list[str] | None = None,
) -> tuple[list[ResolvedBinding], list[str]]:
    """读取所有生效绑定，返回 `(bindings, errors)`。

    生效 = 绑定启用 **且** 策略总开关启用 **且** 池启用。
    单条绑定配置坏掉（手改库导致参数非法、池被删）只记错误、不影响其它绑定。
    """
    state = state if state is not None else all_state(db)
    enabled_strategies = {
        row.code
        for row in db.query(StrategyConfig).filter(StrategyConfig.enabled == 1).all()
        if has_spec(row.code)
    }
    pools = {row.code: row for row in db.query(SymbolPool).filter(SymbolPool.enabled == 1).all()}
    rows = (
        db.query(StrategyBinding)
        .filter(StrategyBinding.enabled == 1)
        .order_by(StrategyBinding.priority.asc(), StrategyBinding.id.asc())
        .all()
    )

    bindings: list[ResolvedBinding] = []
    errors: list[str] = []
    for row in rows:
        if row.strategy_code not in enabled_strategies:
            continue
        pool = pools.get(row.pool_code)
        if pool is None:
            errors.append(f"{row.strategy_code}×{row.pool_code}×{row.interval}: 币种池不存在或已停用")
            continue
        try:
            params = binding_params(db, row, state)
            members = pool_symbols(db, pool, allowed=allowed_symbols)
        except Exception as exc:  # 单条坏配置不拖垮整轮巡检
            errors.append(f"{row.strategy_code}×{row.pool_code}×{row.interval}: {exc}")
            continue
        if not members:
            continue
        bindings.append(
            ResolvedBinding(
                id=row.id,
                strategy=row.strategy_code,
                pool=row.pool_code,
                interval=row.interval,
                priority=int(row.priority or 0),
                params=params,
                symbols=tuple(members),
            )
        )
    return bindings, errors


def expand(bindings: list[ResolvedBinding]) -> list[tuple[str, ResolvedBinding]]:
    """展开成 `(symbol, binding)` 扁平列表，顺序按绑定优先级。"""
    return [(symbol, binding) for binding in bindings for symbol in binding.symbols]


def binding_overrides(
    db: Session, strategy: str, pool: str, interval: str
) -> dict | None:
    """绑定的原始参数覆盖层（未与全局合并）。

    回测走这条：和实盘完全同一层参数，避免「回测用策略配置、实盘用绑定」的口径分裂。
    没有对应绑定时返回 None，调用方回落到策略配置。
    """
    if not pool:
        return None
    row = (
        db.query(StrategyBinding)
        .filter(
            StrategyBinding.strategy_code == strategy,
            StrategyBinding.pool_code == pool,
            StrategyBinding.interval == interval,
        )
        .first()
    )
    if row is None:
        return None
    try:
        overrides = json.loads(row.params or "{}")
    except ValueError:
        return None
    return overrides if isinstance(overrides, dict) else None


def resolve_order_params(
    db: Session, strategy: str, interval: str, pool: str = "", state: dict | None = None
) -> dict:
    """订单/信号所属绑定的有效参数。

    优先级：绑定（strategy × pool × interval）> 策略配置行 > 策略默认。
    老订单没有 pool，或手工构造的信号没有对应绑定，都会回落到策略配置，行为与改造前一致。
    """
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
            return binding_params(db, row, state)
    config = db.query(StrategyConfig).filter(StrategyConfig.code == strategy).first()
    overrides = json.loads(config.params or "{}") if config else None
    return resolve_params(
        strategy,
        state if state is not None else all_state(db),
        overrides or None,
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
        "updated_at": row.updated_at.isoformat() if row.updated_at else "",
    }
    try:
        payload["effective_params"] = binding_params(db, row, state)
    except Exception as exc:
        payload["effective_params"] = {}
        payload["error"] = str(exc)

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

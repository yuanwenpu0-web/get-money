"""策略 × 币种池绑定：把「哪套参数、跑哪些币、用什么周期」变成一等公民。

一条绑定 `(strategy_code, pool_code, interval)` 携带一整套参数覆盖，
于是同一个策略可以同时有：

- `absorption_reversal × major × 15m`（主流，脚本默认灵敏度）
- `absorption_reversal × alt   × 5m`（山寨，更宽的 delta 门槛 + 更紧的止损）
- `absorption_reversal × alt   × 1m`（短线，只做多）

参数三层合并：全局设置 < 策略默认 < 本条绑定的 params（见 `rules.resolve_params`）。
再往上还有一层**币种级覆盖**（`symbol_params` 表）：同一绑定内，某些币种可以带
自己寻优得到的参数，见 `params_for()`。

同一币种可能同时落进多个池（如 BTC 既在 major 又在 all）：各绑定独立评估，
巡检按评分择优只开一仓，不做互斥校验，避免池组合被卡死。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ..config import SUPPORTED_INTERVALS
from ..models import StrategyBinding, StrategyConfig, SymbolPool
from ..state import all_state
from .pools import load_pool_spec, pool_symbols
from .registry import get_spec, has_spec
from .rules import resolve_params
from .symbol_params import get_override, load_map


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

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.strategy, self.pool, self.interval)

    def params_for(self, symbol: str) -> dict:
        """单个币种的最终参数：绑定参数 + 该币种的覆盖。"""
        override = self.symbol_overrides.get(symbol)
        if not override:
            return self.params
        try:
            # self.params 是完整解析结果（含 fee_rate / slippage），而
            # resolve_params 拒绝把成本当覆盖传进来，必须先把这两键剔掉
            layer = {
                key: value
                for key, value in {**self.params, **override}.items()
                if key not in ("fee_rate", "slippage")
            }
            resolved = resolve_params(self.strategy, {}, layer, interval=self.interval)
            resolved["fee_rate"] = self.params["fee_rate"]
            resolved["slippage"] = self.params["slippage"]
            return resolved
        except Exception:
            # 覆盖行被手改坏时宁可退回绑定参数，也不让整个币种停止评估
            return self.params


def binding_params(db: Session, binding: StrategyBinding, state: dict | None = None) -> dict:
    """单条绑定的有效参数（三层合并 + 校验），不含任何币种级覆盖。

    币种级覆盖在本层之上，见 `ResolvedBinding.params_for()` / `resolve_order_params()`。
    """
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
    # 覆盖表整体很小（正常只有几十行），一次取回再切片，避免逐币查库
    overrides_all = load_map(db) if rows else {}
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
    if symbol:
        override = get_override(db, strategy, symbol, interval)
        if override:
            layer = {**(layer or {}), **override}
    return layer


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
    matched = False
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
                matched = True
    if not matched:
        config = db.query(StrategyConfig).filter(StrategyConfig.code == strategy).first()
        if config is not None:
            try:
                parsed = json.loads(config.params or "{}")
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                layer.update(parsed)
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

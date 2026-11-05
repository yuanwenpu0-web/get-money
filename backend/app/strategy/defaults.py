"""默认数据种子：策略行、币种池、策略×池绑定。

- **策略清单来自注册表**（`strategy/specs/`）：新增一个 TradingView 迁移过来的策略，
  只要实现 spec 并注册，这里自动获得默认行，无需改本文件。
- **池与绑定**是「开箱即用的跑法」：主流 15m、山寨 5m 各一条示例，**默认都关闭**。
  只在对应表为空时写入——用户删除后不会被自动复活。
"""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from ..models import StrategyBinding, StrategyConfig, SymbolPool
from . import specs  # noqa: F401  导入即注册内置策略
from .registry import all_specs, available_codes
from .pool_maintenance import OWNER, POOL_CODES, WINDOW_HOURS, research_pool_candidates
from .specs.pool_volatility_research import HIGH_CODE, LOW_CODE

DEFAULT_STRATEGIES = [
    {
        "code": spec.code,
        "name": spec.name,
        "description": spec.description,
        "params": dict(spec.defaults),
        "enabled": 1 if spec.default_enabled else 0,
    }
    for spec in all_specs()
]

ALLOWED_CODES = tuple(item["code"] for item in DEFAULT_STRATEGIES)

DEFAULT_POOLS = [
    {
        "code": "major",
        "name": "主流币",
        "description": "长期成交额靠前、分钟数据连续的高流动性永续（16 个）",
        "kind": "rule",
        "symbols": [],
        "rule": {"base": "major"},
    },
    {
        "code": "alt",
        "name": "山寨币",
        "description": "全量永续剔除主流币；可在 rule 里加成交额门槛或 TopN 限制来收缩范围",
        "kind": "rule",
        "symbols": [],
        "rule": {"base": "all", "exclude_major": True},
    },
    {
        "code": "all",
        "name": "全量永续",
        "description": "币安 U 本位永续全量（含主流与山寨）",
        "kind": "rule",
        "symbols": [],
        "rule": {"base": "all"},
    },
]

# 默认绑定：同一策略的不同池可以有完全不同的参数与周期。
# 两条都默认关闭（enabled=0）：绑定只是把「跑法」摆好，通电要人工确认。
# 原因：absorption_reversal 在池化样本上 CI 跨 0（见 docs/absorption-reversal-strategy.md），
# spec.default_enabled 也是 False；若种子绑定直接 enabled=1，而库里策略行又被手工开过，
# 部署后巡检会立刻真跑一个没有验证过 edge 的策略。
# 开启方式：在绑定上勾选「启用」；旧策略总开关不再控制绑定。
DEFAULT_BINDINGS = [
    {
        "strategy_code": "absorption_reversal",
        "pool_code": "major",
        "interval": "15m",
        "priority": 10,
        "params": {},
        "enabled": 0,
    },
    {
        "strategy_code": "absorption_reversal",
        "pool_code": "alt",
        "interval": "5m",
        "priority": 20,
        "params": {},
        "enabled": 0,
    },
]


def seed_strategies(db: Session, *, include_research: bool = True) -> None:
    """按注册表补齐缺失的策略行，并把已下线的策略强制停用。

    老库里可能残留其它策略的 enabled=1 记录；那些代码本仓库已删除，
    若继续参与巡检会在参数解析阶段直接抛错，所以这里统一置 0。
    """
    rows = db.query(StrategyConfig).all()
    existing = {row.code: row for row in rows}
    for row in rows:
        if row.code not in ALLOWED_CODES and int(row.enabled or 0) != 0:
            row.enabled = 0
    for item in DEFAULT_STRATEGIES:
        if item["code"] not in existing and (include_research or item["code"] not in (HIGH_CODE, LOW_CODE)):
            db.add(
                StrategyConfig(
                    code=item["code"],
                    name=item["name"],
                    description=item["description"],
                    enabled=item["enabled"],
                    params=json.dumps(item["params"], ensure_ascii=False),
                )
            )
    db.commit()


def seed_pools(db: Session) -> None:
    """首次建库时写入默认池；已有池（哪怕被删剩一个）就不再补，尊重用户增删。"""
    if db.query(SymbolPool).count():
        return
    for item in DEFAULT_POOLS:
        db.add(
            SymbolPool(
                code=item["code"],
                name=item["name"],
                description=item["description"],
                kind=item["kind"],
                symbols=json.dumps(item["symbols"], ensure_ascii=False),
                rule=json.dumps(item["rule"], ensure_ascii=False),
                enabled=1,
            )
        )
    db.commit()


def seed_bindings(db: Session) -> None:
    if db.query(StrategyBinding).count():
        return
    pools = {row.code for row in db.query(SymbolPool).all()}
    for item in DEFAULT_BINDINGS:
        if item["strategy_code"] not in ALLOWED_CODES or item["pool_code"] not in pools:
            continue
        db.add(
            StrategyBinding(
                strategy_code=item["strategy_code"],
                pool_code=item["pool_code"],
                interval=item["interval"],
                priority=item["priority"],
                params=json.dumps(item["params"], ensure_ascii=False),
                enabled=item["enabled"],
            )
        )
    db.commit()


def seed_research_objects(db: Session) -> dict:
    """仅补齐缺失的研究对象；由调用方提交或回滚整个事务。"""
    pairs = ((HIGH_CODE, POOL_CODES[0]), (LOW_CODE, POOL_CODES[1]))
    strategies = {row.code for row in db.query(StrategyConfig).filter(StrategyConfig.code.in_((HIGH_CODE, LOW_CODE))).all()}
    pools = {row.code: row for row in db.query(SymbolPool).filter(SymbolPool.code.in_(POOL_CODES)).all()}
    for pool in pools.values():
        try:
            rule = json.loads(pool.rule or "{}")
        except (TypeError, ValueError):
            rule = {}
        if pool.kind != "static" or not isinstance(rule, dict) or rule.get("maintenance_owner") != OWNER:
            raise ValueError(f"研究池代码已被其他对象占用: {pool.code}")
    missing_pools = [code for code in POOL_CODES if code not in pools]
    candidates = research_pool_candidates(db) if missing_pools else None
    for item in DEFAULT_STRATEGIES:
        if item["code"] in (HIGH_CODE, LOW_CODE) and item["code"] not in strategies:
            db.add(StrategyConfig(code=item["code"], name=item["name"], description=item["description"],
                                  enabled=0, params=json.dumps(item["params"], ensure_ascii=False)))
    for code, name in zip(POOL_CODES, ("高成交额高波动", "低成交额高波动")):
        if code not in missing_pools:
            continue
        ready = candidates["updated"]
        rule = {"base": "static", "maintenance_owner": OWNER, "window_hours": WINDOW_HOURS}
        if ready:
            rule["window_end_ms"] = candidates["window_end_ms"]
        else:
            rule["maintenance_status"] = "待维护"
            rule["maintenance_reason"] = candidates["reason"]
        db.add(SymbolPool(code=code, name=name, description="最近168小时合格币种至少100个，成交额前100与波动率前50交叉分类",
                          kind="static", enabled=0,
                          symbols=json.dumps(candidates["members"][code] if ready else [], ensure_ascii=False),
                          rule=json.dumps(rule, ensure_ascii=False)))
    for strategy, pool in pairs:
        exists = db.query(StrategyBinding).filter_by(strategy_code=strategy, pool_code=pool, interval="1h").first()
        if exists is None:
            db.add(StrategyBinding(strategy_code=strategy, pool_code=pool, interval="1h",
                                   enabled=0, priority=100, params="{}"))
    db.flush()
    return {"candidates": candidates, "created_strategies": [s for s, _ in pairs if s not in strategies],
            "created_pools": missing_pools}


def seed_defaults(db: Session) -> None:
    """启动时一次性补齐策略、池与绑定。"""
    seed_strategies(db, include_research=False)
    seed_pools(db)
    seed_bindings(db)
    try:
        seed_research_objects(db)
        db.commit()
    except Exception:
        db.rollback()
        raise


__all__ = [
    "ALLOWED_CODES",
    "DEFAULT_BINDINGS",
    "DEFAULT_POOLS",
    "DEFAULT_STRATEGIES",
    "available_codes",
    "seed_bindings",
    "seed_defaults",
    "seed_pools",
    "seed_strategies",
]

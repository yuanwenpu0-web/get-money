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
# 开启方式：策略页绑定向导里勾「启用」，并把策略总开关打开。
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


def seed_strategies(db: Session) -> None:
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
        if item["code"] not in existing:
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


def seed_defaults(db: Session) -> None:
    """启动时一次性补齐策略、池与绑定。"""
    seed_strategies(db)
    seed_pools(db)
    seed_bindings(db)


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

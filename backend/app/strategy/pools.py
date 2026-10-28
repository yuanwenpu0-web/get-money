"""币种池：策略的运行范围单位，也是参数归属的第二维（策略 × 池）。

两种池：

| kind | 含义 |
| --- | --- |
| `static` | 成员就是 `symbols` 里手写的列表，完全可控 |
| `rule` | 成员按规则动态解析，见下表 |

`rule` 支持的字段（全部可选）：

| 字段 | 含义 |
| --- | --- |
| `base` | 起点集合：`all`（U 本位永续全量）/ `major`（主流白名单）/ `static`（用 symbols） |
| `min_quote_volume_usd` | 24h 成交额门槛；0 或缺失表示不过滤。拿不到成交额时**不过滤**（宁贵不漏） |
| `exclude_major` | 剔除主流白名单（做「山寨池」的标准做法） |
| `max_symbols` | 只保留成交额最大的前 N 个；缺成交额时按字母序取前 N |
| `exclude` | 显式排除的币种 |

所有池的最终成员都会再减去全局黑名单（`coin_universe.BLACKLIST`）。

解析是纯读操作，可被回测/巡检反复调用；成交额走 `market.universe` 的 1 小时 TTL 缓存。
"""

from __future__ import annotations

import json
import math
from typing import Callable, Iterable

from sqlalchemy.orm import Session

from ..market.universe import universe_symbols, universe_volumes
from ..models import SymbolPool
from .coin_universe import BLACKLIST, MAJOR_SET, MAJOR_SYMBOLS

POOL_KINDS = ("static", "rule")
RULE_KEYS = ("base", "min_quote_volume_usd", "exclude_major", "max_symbols", "exclude")
BASE_MODES = ("all", "major", "static")


def _as_symbols(raw: object) -> list[str]:
    """列表归一：大写、去空白、去重、保序；非数组当空列表。"""
    if not isinstance(raw, (list, tuple, set)):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        symbol = str(item).strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    return result


def load_pool_spec(row: SymbolPool) -> dict:
    """把库里的 JSON 文本解析成规范化结构（坏数据按空处理，不抛错）。"""
    try:
        symbols = _as_symbols(json.loads(row.symbols or "[]"))
    except ValueError:
        symbols = []
    try:
        rule = json.loads(row.rule or "{}")
        rule = rule if isinstance(rule, dict) else {}
    except ValueError:
        rule = {}
    kind = row.kind if row.kind in POOL_KINDS else "static"
    return {"kind": kind, "symbols": symbols, "rule": rule}


def validate_pool(kind: str, symbols: object, rule: object) -> tuple[str, list[str], dict]:
    """校验并归一池定义；非法配置直接抛 ValueError，由 API 转 400。"""
    if kind not in POOL_KINDS:
        raise ValueError(f"池类型必须是 {' / '.join(POOL_KINDS)}")
    normalized_symbols = _as_symbols(symbols)
    if not isinstance(rule, dict):
        raise ValueError("池规则必须是 JSON 对象")
    unknown = set(rule) - set(RULE_KEYS)
    if unknown:
        raise ValueError(f"未知池规则字段: {', '.join(sorted(unknown))}")

    normalized_rule: dict = {}
    base = rule.get("base", "static" if kind == "static" else "all")
    if base not in BASE_MODES:
        raise ValueError(f"base 必须是 {' / '.join(BASE_MODES)}")
    if kind == "static" and base != "static":
        raise ValueError("static 池的 base 只能是 static（动态范围请用 rule 池）")
    normalized_rule["base"] = base

    if "min_quote_volume_usd" in rule and rule["min_quote_volume_usd"] is not None:
        try:
            threshold = float(rule["min_quote_volume_usd"])
        except (TypeError, ValueError) as exc:
            raise ValueError("min_quote_volume_usd 必须是数字") from exc
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("min_quote_volume_usd 必须是非负有限数")
        normalized_rule["min_quote_volume_usd"] = threshold

    if "exclude_major" in rule:
        if not isinstance(rule["exclude_major"], bool):
            raise ValueError("exclude_major 必须是布尔值")
        normalized_rule["exclude_major"] = rule["exclude_major"]

    if rule.get("max_symbols") is not None:
        try:
            limit = int(rule["max_symbols"])
        except (TypeError, ValueError) as exc:
            raise ValueError("max_symbols 必须是整数") from exc
        if limit < 1:
            raise ValueError("max_symbols 必须 ≥ 1")
        normalized_rule["max_symbols"] = limit

    if "exclude" in rule:
        normalized_rule["exclude"] = _as_symbols(rule["exclude"])

    if kind == "static" and not normalized_symbols:
        raise ValueError("static 池必须给出 symbols 列表")
    return kind, normalized_symbols, normalized_rule


def resolve_members(
    kind: str,
    symbols: Iterable[str],
    rule: dict,
    *,
    universe: Callable[[], list[str]],
    volumes: Callable[[], dict[str, float]],
    allowed: Iterable[str] | None = None,
) -> list[str]:
    """纯函数版成员解析（不碰数据库，便于测试与回测复用）。

    `allowed` 是全局限制（如 `symbol_scope=custom` 的自定义列表），池在其内部再筛选。
    """
    base = rule.get("base", "static" if kind == "static" else "all")
    if base == "major":
        members = list(MAJOR_SYMBOLS)
    elif base == "all":
        members = list(universe())
    else:
        members = _as_symbols(list(symbols))

    if rule.get("exclude_major"):
        members = [symbol for symbol in members if symbol not in MAJOR_SET]

    threshold = rule.get("min_quote_volume_usd")
    limit = rule.get("max_symbols")
    # 门槛过滤与 TopN 排序都要成交额：任一存在就取一次（走 1h TTL 缓存）
    volume_map = volumes() if (threshold or limit) else {}
    if threshold and volume_map:
        # 宁贵不漏：成交额缺失的币种保留，只剔除明确低于门槛的
        members = [
            symbol for symbol in members
            if float(volume_map.get(symbol, threshold)) >= threshold
        ]

    excluded = set(rule.get("exclude") or []) | BLACKLIST
    members = [symbol for symbol in members if symbol not in excluded]

    if allowed is not None:
        allow_set = set(allowed)
        members = [symbol for symbol in members if symbol in allow_set]

    if limit:
        if volume_map:
            members.sort(key=lambda symbol: volume_map.get(symbol, 0.0), reverse=True)
        members = members[: int(limit)]
    return members


def pool_symbols(db: Session, pool: SymbolPool, allowed: Iterable[str] | None = None) -> list[str]:
    """从数据库池定义解析当前成员。"""
    spec = load_pool_spec(pool)
    return resolve_members(
        spec["kind"],
        spec["symbols"],
        spec["rule"],
        universe=lambda: universe_symbols(db),
        volumes=lambda: universe_volumes(db),
        allowed=allowed,
    )


def pool_symbols_by_code(
    db: Session, code: str, allowed: Iterable[str] | None = None
) -> list[str]:
    pool = db.query(SymbolPool).filter(SymbolPool.code == code).first()
    if pool is None:
        return []
    return pool_symbols(db, pool, allowed=allowed)


def describe_pool(pool: SymbolPool, members: list[str] | None = None) -> dict:
    """给 API/前端用的池描述（含成员数与解析后的规则）。"""
    spec = load_pool_spec(pool)
    payload = {
        "id": pool.id,
        "code": pool.code,
        "name": pool.name,
        "description": pool.description,
        "kind": spec["kind"],
        "symbols": spec["symbols"],
        "rule": spec["rule"],
        "enabled": bool(pool.enabled),
        "updated_at": pool.updated_at.isoformat() if pool.updated_at else "",
    }
    if members is not None:
        payload["member_count"] = len(members)
        payload["members_sample"] = members[:20]
    return payload

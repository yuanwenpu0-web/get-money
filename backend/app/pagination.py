"""Shared metadata and limits for bounded list responses.

全站统一口径：任何列表接口单页最多 15 条（`MAX_PAGE_SIZE`），默认一页也是 15 条。
这样前端每个列表都能直接渲染分页器，不用先拉全量再本地切；后端也始终有 LIMIT，
不会因为某张表涨到百万行就把响应体撑爆。
"""

from threading import Lock
from time import monotonic
from typing import Annotated, Any, Callable
from weakref import WeakKeyDictionary

from fastapi import Query

MAX_PAGE_SIZE = 15
DEFAULT_PAGE_SIZE = 15

# 直接塞进路由签名里用，保证「默认值 / 上限 / 校验」三处不会各写一套漂移。
PageQuery = Annotated[int, Query(ge=1, description="页码，从 1 开始")]
PageSizeQuery = Annotated[
    int, Query(ge=1, le=MAX_PAGE_SIZE, description=f"每页条数，最多 {MAX_PAGE_SIZE}")
]


def page_metadata(total: int, page: int, page_size: int) -> dict:
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
    }


def offset_of(page: int, page_size: int) -> int:
    """SQL OFFSET；page 至少为 1，避免负数偏移。"""
    return (max(1, page) - 1) * page_size


_COUNTS: "WeakKeyDictionary[Any, dict[str, tuple[float, int]]]" = WeakKeyDictionary()
_COUNTS_LOCK = Lock()


def cached_count(scope: Any, key: str, loader: Callable[[], int], ttl: float = 120.0) -> int:
    """给「不变但很贵」的 total 计数做进程内缓存。

    `count(distinct symbol)` 这类聚合在千万行表上没法靠索引走捷径（PostgreSQL 没有
    index skip scan），实测 26s；而币种覆盖集合只在新币同步时才变，按 TTL 缓存足以，
    缓存过期后重新计算一次也不会阻塞已有分页结果。

    `scope` 传 Engine（`db.get_bind()`）：缓存以连接池为界，测试里每个用例一个独立
    in-memory 库时不会串味——这是字符串 key 做全局缓存踩过的坑。
    """
    now = monotonic()
    with _COUNTS_LOCK:
        per_scope = _COUNTS.get(scope)
        hit = (per_scope or {}).get(key)
        if hit is not None and now - hit[0] < ttl:
            return hit[1]
    value = int(loader())
    with _COUNTS_LOCK:
        _COUNTS.setdefault(scope, {})[key] = (now, value)
    return value

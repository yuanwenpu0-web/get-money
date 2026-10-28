"""按币种并行的回测与数据提取执行器。

为什么是线程而不是进程
------------------------
1. 每个币种的任务里 `build_dataset` 有大量 DB 往返，线程能重叠 IO 等待；
2. 需要共享只读 engine 与数据集缓存，进程池会强制序列化大 DataFrame；
3. Windows 下 spawn 开销大，且项目既有并发（auto_train、行情同步）已是线程模型，保持一致。

代价：特征计算与 pandas 运算受 GIL 制约，纯 CPU 段加速有限；
DB IO 占比越高收益越大。默认并发取 min(cpu//2, 8) 而非 cpu，
避免把机器打满影响线上服务与巡检。

用法：
    from app.backtest.parallel import parallel_symbol_map
    results = parallel_symbol_map(db, symbols, lambda worker_db, s: build_dataset(...))
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

from sqlalchemy.orm import Session, sessionmaker

DEFAULT_MAX_WORKERS = 8


def resolve_workers(requested: int | None, targets: int) -> int:
    """并发度：显式指定优先，否则按 CPU 预算与任务数取小。

    取 cpu//2 是因为每个模型/特征计算内部还可能有自己的线程预算，
    打满逻辑核反而会因上下文切换变慢。
    """
    count = max(1, int(targets or 0))
    if requested is not None:
        return max(1, min(int(requested), count))
    cpu = os.cpu_count() or 2
    return max(1, min(cpu // 2 or 1, DEFAULT_MAX_WORKERS, count))


def parallel_symbol_map(
    db: Session,
    symbols: Sequence[str],
    fn: Callable[[Session, str], object],
    workers: int | None = None,
    thread_name: str = "backtest",
    reraise: bool = False,
) -> list:
    """按币种并行执行 `fn(worker_db, symbol)`，**保持输入顺序**返回结果。

    - 每个 worker 用独立 Session：连接不是线程安全的，且 PG 只读会话可并发；
      内存 SQLite 与共享 StaticPool 由 SQLAlchemy 自动串行，行为不变。
    - 单个币种抛异常时默认转成 `{"symbol": ..., "error": ...}` 返回，
      不中断其余任务；`reraise=True` 用于需要快速失败的调用方。
    - workers 解析为 1 时退化为串行调用，保证可测试与结果确定性。
    """
    items = list(symbols)
    if not items:
        return []
    count = resolve_workers(workers, len(items))
    if count == 1:
        return [_invoke(db, symbol, fn, reraise) for symbol in items]

    factory = sessionmaker(bind=db.get_bind(), autoflush=False, expire_on_commit=False)

    def run(symbol: str):
        with factory() as worker_db:
            return _invoke(worker_db, symbol, fn, reraise)

    with ThreadPoolExecutor(max_workers=count, thread_name_prefix=thread_name) as executor:
        # map 保序：即便后面的币种先算完，结果仍按输入顺序排列。
        return list(executor.map(run, items))


def _invoke(db: Session, symbol: str, fn: Callable[[Session, str], object], reraise: bool):
    try:
        return fn(db, symbol)
    except Exception as exc:  # noqa: BLE001 - 单币种失败不能拖垮整轮回测
        if reraise:
            raise
        return {"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"}

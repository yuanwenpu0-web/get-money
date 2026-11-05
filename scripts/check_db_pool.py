"""数据库连接池判断：连接是否真的被复用、容量是否够、探活是否通。

用法::

    python scripts/check_db_pool.py                  # 判断后端服务用的全局池
    python scripts/check_db_pool.py --probe         # 额外探活 + 读 PG 服务端连接数
    python scripts/check_db_pool.py --probe --json  # 机器可读
    python scripts/check_db_pool.py --null-pool     # 对照组：应当判定为「未复用」

退出码：0 = 已池化且复用自检通过；1 = 未池化 / 复用异常 / 探活失败。
输出不含连接串、用户名或密码。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import db  # noqa: E402


def _fmt(value, suffix: str = "") -> str:
    return "-" if value is None else f"{value}{suffix}"


def _human(status: dict, reuse: dict) -> str:
    counters = status.get("counters") or {}
    ratio = status.get("reuse_ratio")
    utilization = status.get("utilization")
    recycle = status.get("pool_recycle")
    recycle_txt = f"{recycle}s" if recycle and recycle > 0 else "-"
    lines = [
        f"数据库      : {status['dialect']} ({status['driver']})  "
        f"{status['database'] or '-'} @ {status['host'] or 'local'}",
        f"连接池      : {status['pool_class']}  模式={status['reuse_mode']}  "
        f"容量={status['capacity']} "
        f"({_fmt(status['pool_size'])} 常驻 + {_fmt(status['max_overflow'])} 溢出)",
        f"当前占用    : 借出 {status['checkedout']} / 空闲 {_fmt(status['checkedin'])} / "
        f"利用率 {f'{utilization:.1%}' if utilization is not None else '-'}",
        f"超时/回收   : pool_timeout={_fmt(status['pool_timeout'], 's')}  "
        f"recycle={recycle_txt}  pre_ping={status['pre_ping']}",
        f"累计连接    : 物理新建 {counters.get('connections_created', '-')} / "
        f"借出 {counters.get('checkouts', '-')} / 归还 {counters.get('checkins', '-')} / "
        f"失效 {counters.get('invalidations', '-')} / 复用率 "
        f"{f'{ratio:.1%}' if ratio is not None else '-'}",
        f"复用自检    : {reuse['queries']} 次查询 (并发 {reuse['workers']}) -> "
        f"新建 {reuse['connections_created']} 条物理连接，{reuse['verdict']} "
        f"({reuse['elapsed_ms']} ms)",
    ]
    probe = status.get("probe")
    if probe is not None:
        server = status.get("server") or {}
        extra = ""
        if server:
            extra = (
                f"  max_connections={server['max_connections']} "
                f"本库占用={server['database_connections']} 余量={server['headroom']}"
            )
        lines.append(f"探活        : {'ok' if probe['ok'] else 'failed'} "
                     f"({probe.get('ms')} ms, {probe.get('error') or '-'}){extra}")
    warnings = status.get("warnings") or []
    if warnings:
        lines.append("告警        :")
        lines.extend(f"  ! {item}" for item in warnings)
    else:
        lines.append("告警        : 无")
    ok = status["pooled"] is True and reuse["reuse_ok"] and status["healthy"] is not False
    lines.append(f"判断        : {'PASS' if ok else 'FAIL'} —— "
                 f"{'连接已复用，未频繁新建' if ok else '存在未池化或频繁新建连接的问题'}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="判断数据库连接池是否复用连接")
    parser.add_argument("--probe", action="store_true", help="探活并读取 PostgreSQL 服务端连接数")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--url", default=None, help="判断另一个数据库 URL（不会打印它）")
    parser.add_argument("--null-pool", action="store_true", help="对照组：强制 NullPool")
    parser.add_argument("--times", type=int, default=20, help="复用自检查询次数（默认 20）")
    parser.add_argument("--workers", type=int, default=1, help="复用自检并发数（默认 1）")
    args = parser.parse_args()

    eng = db.engine
    own_engine = bool(args.url or args.null_pool)
    if own_engine:
        eng = db.make_engine(url=args.url, null_pool=args.null_pool)
    try:
        # 先跑复用自检，让 pool_status 里的累计计数与告警包含这轮查询。
        reuse = db.probe_pool_reuse(eng, times=args.times, workers=args.workers)
        status = db.pool_status(eng, probe=args.probe)
        status["reuse_probe"] = reuse
        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print(_human(status, reuse))
        ok = status["pooled"] is True and reuse["reuse_ok"] and status["healthy"] is not False
        return 0 if ok else 1
    finally:
        if own_engine:
            eng.dispose()


if __name__ == "__main__":
    raise SystemExit(main())

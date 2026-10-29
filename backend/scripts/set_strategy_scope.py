"""把某个策略绑定的运行范围（币种池）改成指定池。

用法：
    python scripts/set_strategy_scope.py --strategy absorption_reversal --pool all
    python scripts/set_strategy_scope.py --strategy absorption_reversal --pool all --dry-run
    python scripts/set_strategy_scope.py --strategy absorption_reversal --pool all --enable

为什么需要脚本而不是直接改库：
- 绑定唯一键是 `(strategy_code, pool_code, interval)`，改池可能撞上已有绑定，
  必须先检查再决定是合并还是跳过，裸 UPDATE 会直接抛唯一约束错误。
- 改范围会改变回测/寻优的「基准参数」来源（`_base_params` 取优先级最小的绑定），
  所以要打印改动前后的池成员数与参数，便于复核。

幂等：重复执行只在池不一致时才写库。
默认**不改 `enabled`**：只换范围，不改变线上是否参与巡检。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.orm import Session  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.models import StrategyBinding, SymbolPool  # noqa: E402
from app.strategy.pools import pool_symbols  # noqa: E402


def rebind_scope(
    db: Session,
    strategy: str,
    pool_code: str,
    *,
    interval: str = "",
    enable: bool | None = None,
    dry_run: bool = False,
) -> dict:
    """把 `strategy` 的绑定池改成 `pool_code`，返回改动摘要（不打印）。

    `enable=None` 表示不动 `enabled`；True/False 强制启用/停用。
    目标 `(策略, 池, 周期)` 已被另一条绑定占用时**跳过该条**并记入 `skipped`，
    不删任何绑定：删绑定是不可逆的线上配置变更，应该由人决定。
    """
    target = db.query(SymbolPool).filter(SymbolPool.code == pool_code).first()
    if target is None:
        raise ValueError(f"未知币种池: {pool_code}")

    query = db.query(StrategyBinding).filter(StrategyBinding.strategy_code == strategy)
    if interval:
        query = query.filter(StrategyBinding.interval == interval)
    bindings = query.order_by(StrategyBinding.priority.asc()).all()
    if not bindings:
        raise ValueError(f"策略 {strategy} 没有绑定")

    members = pool_symbols(db, target)
    changes: list[dict] = []
    skipped: list[dict] = []

    for row in bindings:
        clash = (
            db.query(StrategyBinding)
            .filter(
                StrategyBinding.strategy_code == strategy,
                StrategyBinding.pool_code == pool_code,
                StrategyBinding.interval == row.interval,
                StrategyBinding.id != row.id,
            )
            .first()
        )
        if clash is not None:
            skipped.append(
                {"id": row.id, "interval": row.interval, "reason": f"已被 #{clash.id} 占用"}
            )
            continue

        before_pool = (
            db.query(SymbolPool).filter(SymbolPool.code == row.pool_code).first()
        )
        before_count = len(pool_symbols(db, before_pool)) if before_pool else 0
        touched = row.pool_code != pool_code
        if enable is not None and bool(row.enabled) != enable:
            touched = True
        if touched:
            changes.append(
                {
                    "id": row.id,
                    "interval": row.interval,
                    "from": row.pool_code,
                    "to": pool_code,
                    "from_count": before_count,
                    "to_count": len(members),
                    "enabled_from": int(row.enabled or 0),
                    "enabled_to": int(row.enabled or 0) if enable is None else int(enable),
                }
            )

        if dry_run:
            continue
        row.pool_code = pool_code
        if enable is not None:
            row.enabled = int(enable)

    if dry_run:
        db.rollback()
    else:
        db.commit()

    return {
        "pool": pool_code,
        "pool_count": len(members),
        "changes": changes,
        "skipped": skipped,
        "binding_count": len(bindings),
        "dry_run": dry_run,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="设置策略的运行范围（币种池）")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--pool", required=True, help="目标池代码，如 all / major / alt")
    parser.add_argument("--interval", default="", help="只改这个周期的绑定；留空改全部")
    parser.add_argument("--enable", action="store_true", help="同时把绑定设为启用（默认不动）")
    parser.add_argument("--disable", action="store_true", help="同时把绑定设为停用")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要做的改动")
    args = parser.parse_args()

    if args.enable and args.disable:
        raise SystemExit("--enable 与 --disable 不能同时使用")
    enable = True if args.enable else False if args.disable else None

    db = SessionLocal()
    try:
        try:
            summary = rebind_scope(
                db,
                args.strategy,
                args.pool,
                interval=args.interval,
                enable=enable,
                dry_run=args.dry_run,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        print(f"目标池 {summary['pool']}：{summary['pool_count']} 个币种")
        print()
        for item in summary["changes"]:
            print(
                f"  改 #{item['id']} {item['from']}×{item['interval']} → {item['to']}×{item['interval']}"
                f"（{item['from_count']} 币 → {item['to_count']} 币"
                f"，enabled {item['enabled_from']} → {item['enabled_to']}）"
            )
        for item in summary["skipped"]:
            print(f"  跳过 #{item['id']} {item['interval']}：{item['reason']}")
        if args.dry_run:
            print(f"\n[dry-run] 预计改动 {len(summary['changes'])} 处，未写库")
            return 0
        print(f"\n已改动 {len(summary['changes'])} 处")

        print("\n当前绑定：")
        for row in (
            db.query(StrategyBinding)
            .filter(StrategyBinding.strategy_code == args.strategy)
            .order_by(StrategyBinding.priority.asc())
            .all()
        ):
            params = json.loads(row.params or "{}")
            print(
                f"  #{row.id} {row.pool_code}×{row.interval} "
                f"enabled={row.enabled} priority={row.priority} "
                f"params={json.dumps(params, ensure_ascii=False)[:100]}"
            )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

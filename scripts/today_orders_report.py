"""今日订单与策略归属速查。只读，不写库。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from sqlalchemy import text  # noqa: E402

from app.db import make_engine  # noqa: E402

TS = sys.argv[1] if len(sys.argv) > 1 else "2026-11-02T00:00:00+08:00"
FILTER = "created_at >= :ts OR closed_at >= :ts"

ORDER_SQL = f"""
SELECT id, symbol, side, entry_price, close_price, status, strategy, source,
       realized_pnl, close_reason,
       created_at AT TIME ZONE 'Asia/Shanghai' AS created_local,
       closed_at AT TIME ZONE 'Asia/Shanghai' AS closed_local
FROM orders
WHERE {FILTER}
ORDER BY created_at DESC
"""

SUM_SQL = f"""
SELECT COALESCE(strategy, '(未填)') AS st,
       COUNT(*) AS n,
       SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) AS closed_n,
       SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) AS open_n,
       SUM(realized_pnl) AS pnl,
       SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS win,
       SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) AS lose
FROM orders
WHERE {FILTER}
GROUP BY strategy
ORDER BY pnl DESC
"""


def main() -> None:
    eng = make_engine(read_only=True)
    with eng.connect() as conn:
        rows = conn.execute(text(ORDER_SQL), {"ts": TS}).fetchall()
        print(f"=== 今日订单 {len(rows)} 条 ===")
        for r in rows:
            print(
                f"id={r[0]:>3} {r[1]:<14} {r[2]:<5} {r[3]:<12} {str(r[4]):<12} "
                f"{r[5]:<6} {r[6]:<22} pnl={r[8]:>10.2f} {r[9]:<16} "
                f"open={str(r[10])[:19]} close={str(r[11])[:19]}"
            )
        print()
        print("=== 按策略汇总 ===")
        for r in conn.execute(text(SUM_SQL), {"ts": TS}).fetchall():
            print(
                f"{r[0]:<24} 笔数={r[1]:>3} 平={r[2]:>2} 开={r[3]:>2} "
                f"盈亏={r[4]:>10.2f} 盈={r[5]} 亏={r[6]}"
            )


if __name__ == "__main__":
    main()

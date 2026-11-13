"""存量 OPEN live 仓位：把保护单从信号价平移到实际成交价（工程修复，非 edge 判定）。

开仓路径已改为锚定 fill_price；本脚本把同一口径补到存量仓位：绝对价格距离不变，
因此 sl_atr_mult×ATR 与 target_r 在成交后仍成立，qty×距离 恰等于开仓前的风险预算。

信号价取自 `signals` 表里「下单前最后一条同向信号」，而不是按 created_at 重放 kline：
引擎按本地已同步的 bar 发信号，重放会多取到下单时还没入库的那一根。

默认只读 dry-run；--apply 才撤旧挂新。现价已越过新止损/新止盈的仓位一律跳过，
不主动把浮亏变成已实现亏损。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.db import make_engine  # noqa: E402
from app.market.futures import round_to_tick  # noqa: E402
from app.models import Order, Signal  # noqa: E402
from app.trading.live import _tick_size, client_for, modify_live, reanchor_exits  # noqa: E402


def to_ms(dt) -> int:
    return int(dt.timestamp() * 1000)


def select_signal(rows: list[dict], *, side: str, created_ms: int) -> dict | None:
    """下单前最后一条同向信号；rows 需含 created_ms/direction/price/sl/tp。"""
    eligible = [r for r in rows if r["direction"] == side and r["created_ms"] <= created_ms]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r["created_ms"])


def plan_reanchor(
    *, side: str, entry: float, sl: float, tp: float, qty: float,
    signal_price: float, signal_sl: float, signal_tp: float, mark: float, tick: float,
) -> tuple[dict | None, str]:
    """纯函数：给出重挂后的止损/止盈，或跳过原因。不写库、不下单。"""
    if tp <= 0:
        return None, "no_tp"
    direction = 1.0 if side == "LONG" else -1.0
    if (entry - sl) * direction <= 0:
        return None, "bad_sl_side"
    # 落库水平必须仍是信号水平按 tick 取整的结果；被手工/保本移过就不能盲目平移
    if round_to_tick(signal_sl, tick) != sl or round_to_tick(signal_tp, tick) != tp:
        return None, "not_signal_anchored"
    new_sl, new_tp = reanchor_exits(entry, signal_price, sl, tp)
    new_sl, new_tp = round_to_tick(new_sl, tick), round_to_tick(new_tp, tick)
    if (entry - new_sl) * direction <= 0 or (new_tp - entry) * direction <= 0:
        return None, "tick_invalid"
    if new_sl == sl and new_tp == tp:
        return None, "already_anchored"
    if (mark - new_sl) * direction <= 0:
        return None, "would_stop_out"
    if (new_tp - mark) * direction <= 0:
        return None, "would_take_profit"
    distance = abs(entry - new_sl)
    old_distance = abs(entry - sl)
    return {
        "sl": new_sl, "tp": new_tp, "old_sl": sl, "old_tp": tp,
        "entry": entry, "signal_price": signal_price, "mark": mark, "tick": tick,
        "stop_distance": distance,
        "risk_usdt_before": qty * old_distance,
        "risk_usdt_after": qty * distance,
        "rr_before": abs(tp - entry) / old_distance,
        "rr_after": abs(new_tp - entry) / distance,
        "mark_room_to_sl_pct": direction * (mark - new_sl) / mark * 100.0,
    }, ""


def build_plans(db: Session, ids: list[int]) -> tuple[list[dict], list[dict]]:
    client = client_for(db)
    query = db.query(Order).filter(Order.source == "live", Order.status == "OPEN")
    if ids:
        query = query.filter(Order.id.in_(ids))
    plans, skipped = [], []
    for order in query.order_by(Order.id).all():
        base = {"id": order.id, "symbol": order.symbol, "side": order.side,
                "interval": order.interval, "strategy": order.strategy}
        created_ms = to_ms(order.created_at)
        rows = db.query(Signal).filter(
            Signal.symbol == order.symbol, Signal.interval == order.interval,
            Signal.strategy == order.strategy, Signal.created_at <= order.created_at,
        ).order_by(Signal.created_at.desc()).limit(50).all()
        signal = select_signal(
            [{"created_ms": to_ms(s.created_at), "direction": s.direction,
              "price": float(s.price), "sl": float(s.sl), "tp": float(s.tp)} for s in rows],
            side=order.side, created_ms=created_ms,
        )
        if signal is None:
            skipped.append({**base, "why": "no_signal_row"})
            continue
        try:
            tick = _tick_size(client, order.symbol)
            mark = float(client.premium_index(order.symbol)["markPrice"])
        except Exception as exc:  # 交易所取数失败：跳过，不用猜测的价格改单
            skipped.append({**base, "why": f"exchange_read_failed: {exc}"})
            continue
        plan, why = plan_reanchor(
            side=order.side, entry=float(order.entry_price), sl=float(order.sl),
            tp=float(order.tp), qty=float(order.qty), signal_price=signal["price"],
            signal_sl=signal["sl"], signal_tp=signal["tp"], mark=mark, tick=tick,
        )
        detail = {**base, "signal_price": signal["price"], "signal_sl": signal["sl"],
                  "signal_tp": signal["tp"], "signal_lag_min": (created_ms - signal["created_ms"]) / 60000.0,
                  "mark": mark, "tick": tick}
        if plan is None:
            skipped.append({**detail, "why": why})
            continue
        plans.append({**detail, "qty": float(order.qty), "margin": float(order.margin or 0), **plan})
    return plans, skipped


def apply_plans(db: Session, plans: list[dict]) -> list[dict]:
    results = []
    for plan in plans:
        order = db.get(Order, plan["id"])
        if order is None or order.status != "OPEN":
            results.append({"id": plan["id"], "symbol": plan["symbol"], "ok": False,
                            "why": "no_longer_open"})
            continue
        # dry-run 与 apply 之间价格可能已穿过新水平，挂出去就是立刻触发
        client = client_for(db)
        mark = float(client.premium_index(order.symbol)["markPrice"])
        direction = 1.0 if order.side == "LONG" else -1.0
        if (mark - plan["sl"]) * direction <= 0 or (plan["tp"] - mark) * direction <= 0:
            results.append({"id": plan["id"], "symbol": plan["symbol"], "ok": False,
                            "why": "mark_moved_past_new_level", "mark": mark})
            continue
        try:
            modify_live(db, order, sl=plan["sl"], tp=plan["tp"])
            results.append({"id": plan["id"], "symbol": plan["symbol"], "ok": True,
                            "sl": order.sl, "tp": order.tp})
        except Exception as exc:
            # modify_live 先撤后挂：挂新单失败会让仓位裸奔，尽力挂回旧水平
            db.rollback()
            order = db.get(Order, plan["id"])
            restored = False
            try:
                modify_live(db, order, sl=plan["old_sl"], tp=plan["old_tp"])
                restored = True
            except Exception:
                db.rollback()
            results.append({"id": plan["id"], "symbol": plan["symbol"], "ok": False,
                            "why": f"modify_failed: {exc}", "restored_old_levels": restored})
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真的撤旧保护单并挂新单")
    ap.add_argument("--ids", default="", help="只处理这些订单 id，逗号分隔")
    ap.add_argument("--out", default="output/reanchor_open_protections.json")
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    engine = make_engine(read_only=not args.apply)
    with Session(engine) as db:
        plans, skipped = build_plans(db, ids)
        results = apply_plans(db, plans) if args.apply else []
    payload = {"applied": args.apply, "n_plans": len(plans), "n_skipped": len(skipped),
               "plans": plans, "skipped": skipped, "results": results}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    for item in plans:
        print(f"#{item['id']} {item['symbol']} {item['side']} entry={item['entry']} "
              f"sig={item['signal_price']} mark={item['mark']} "
              f"sl {item['old_sl']}->{item['sl']} tp {item['old_tp']}->{item['tp']} "
              f"risk {item['risk_usdt_before']:.2f}->{item['risk_usdt_after']:.2f} "
              f"rr {item['rr_before']:.3f}->{item['rr_after']:.3f}")
    for item in skipped:
        print(f"SKIP #{item['id']} {item['symbol']}: {item['why']} "
              f"sig={item.get('signal_price')} signal_sl_tp=({item.get('signal_sl')},{item.get('signal_tp')})")
    for item in results:
        detail = item.get("why") or "sl={} tp={}".format(item.get("sl"), item.get("tp"))
        print(f"{'OK' if item['ok'] else 'FAIL'} #{item['id']} {item['symbol']}: {detail}")
    print(f"wrote {out}  plans={len(plans)} skipped={len(skipped)} applied={args.apply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
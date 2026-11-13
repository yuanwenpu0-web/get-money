"""存量 OPEN 仓位保护单重挂：纯规划函数单测（不联网、不连 DB、不下单）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
from reanchor_open_protections import plan_reanchor, select_signal  # noqa: E402


def plan(**over):
    base = dict(
        side="LONG", entry=100.5, sl=99.0, tp=103.0, qty=2.0,
        signal_price=100.0, signal_sl=99.0, signal_tp=103.0,
        mark=100.6, tick=0.1,
    )
    base.update(over)
    return plan_reanchor(**base)


def test_plan_translates_levels_to_fill_and_keeps_absolute_distance():
    got, why = plan()
    assert why == ""
    assert (got["sl"], got["tp"]) == (99.5, 103.5)
    # 绝对距离不变 ⇒ 1.5×ATR 与 3R 在成交后仍成立
    assert got["entry"] - got["sl"] == pytest.approx(got["signal_price"] - 99.0)
    assert got["tp"] - got["entry"] == pytest.approx(103.0 - got["signal_price"])
    assert got["rr_after"] == pytest.approx(3.0)


def test_plan_mirrors_for_short():
    got, why = plan(side="SHORT", entry=99.5, sl=101.0, tp=97.0,
                    signal_price=100.0, signal_sl=101.0, signal_tp=97.0, mark=99.4)
    assert why == ""
    assert (got["sl"], got["tp"]) == (100.5, 96.5)
    assert got["rr_after"] == pytest.approx(3.0)


def test_plan_accepts_levels_that_are_tick_rounded_signal_levels():
    # 落库 sl/tp 是信号水平按 tick 取整后的值，不能因为取整差就判成「被改过」
    got, why = plan(signal_sl=99.04, signal_tp=103.02)
    assert why == ""
    assert (got["sl"], got["tp"]) == (99.5, 103.5)


def test_plan_restores_budgeted_risk_regardless_of_fill():
    # 预算风险 = qty × 1.5×ATR 绝对距离；追价成交不应把它放大
    risks = []
    for entry in (100.5, 101.0, 102.0, 100.2):
        got, why = plan(entry=entry, mark=entry + 0.1)
        assert why == "", (entry, why)
        risks.append(got["risk_usdt_after"])
    assert max(risks) - min(risks) == pytest.approx(0.0, abs=1e-9)
    assert risks[0] == pytest.approx(2.0 * 1.0)


def test_plan_reports_inflated_risk_before_reanchor():
    got, why = plan(entry=102.0, mark=102.1)
    assert why == ""
    assert got["risk_usdt_before"] == pytest.approx(2.0 * 3.0)
    assert got["risk_usdt_after"] == pytest.approx(2.0 * 1.0)
    assert got["rr_before"] == pytest.approx(1.0 / 3.0)


def test_plan_skips_when_mark_already_past_new_stop():
    # 重挂会把止损拉近到预算距离；若现价已越过新止损，应用即立刻 realized 亏损
    got, why = plan(mark=99.4)
    assert got is None
    assert why == "would_stop_out"


def test_plan_skips_when_mark_already_past_new_target():
    got, why = plan(mark=103.6)
    assert got is None
    assert why == "would_take_profit"


def test_plan_skips_when_levels_already_anchor_fill():
    got, why = plan(entry=100.0, mark=100.1)
    assert got is None
    assert why == "already_anchored"


def test_plan_skips_when_stored_levels_differ_from_signal_levels():
    # 止损被手工/保本移到别处后不能盲目平移
    got, why = plan(sl=100.2)
    assert got is None
    assert why == "not_signal_anchored"


def test_plan_skips_when_tick_rounding_collapses_stop():
    # 粗 tick：信号止损 99.6 取整成 100，追价到 100.6 后平移再取整变成 101，越过成交价
    got, why = plan(entry=100.6, sl=100.0, tp=101.0, signal_price=100.0,
                    signal_sl=99.6, signal_tp=101.2, mark=100.7, tick=1.0)
    assert got is None
    assert why == "tick_invalid"


def test_plan_skips_when_order_has_no_take_profit():
    got, why = plan(tp=0.0, signal_tp=0.0)
    assert got is None
    assert why == "no_tp"


def test_plan_skips_when_current_stop_is_on_wrong_side_of_entry():
    got, why = plan(sl=101.0)
    assert got is None
    assert why == "bad_sl_side"


def row(created_ms, direction="SHORT", price=0.03572, sl=0.036962, tp=0.031994):
    return {"created_ms": created_ms, "direction": direction, "price": price, "sl": sl, "tp": tp}


def test_select_signal_takes_latest_row_not_newer_than_order():
    # 引擎按本地已同步 kline 发信号，可能早于下单时刻最后一根收盘 bar；
    # 必须取「下单前最后一条同向信号」，不能按 created_at 重放 kline
    rows = [row(1000), row(2000), row(3000, direction="FLAT"), row(5000)]
    got = select_signal(rows, side="SHORT", created_ms=4000)
    assert got["created_ms"] == 2000


def test_select_signal_ignores_opposite_direction():
    rows = [row(1000), row(2000, direction="LONG")]
    got = select_signal(rows, side="SHORT", created_ms=3000)
    assert got["created_ms"] == 1000


def test_select_signal_returns_none_without_eligible_row():
    assert select_signal([row(5000)], side="SHORT", created_ms=4000) is None
    assert select_signal([], side="SHORT", created_ms=4000) is None


# --- apply 路径：真的会撤单/挂单，本轮线上未触发，必须单测覆盖 ---

from app.market.futures import BinanceApiError  # noqa: E402
from app.models import Order  # noqa: E402
from fakes import FakeFuturesClient, live_order, market_orders, patch_live  # noqa: E402
from reanchor_open_protections import apply_plans  # noqa: E402

PLAN = {"id": 1, "symbol": "BTCUSDT", "sl": 99.5, "tp": 103.5, "old_sl": 99.0, "old_tp": 103.0}


def seed_open(db, **over):
    order = live_order(**over)
    db.add(order)
    db.commit()
    return order


def test_apply_cancels_old_protections_and_places_new_levels(db, monkeypatch):
    order = seed_open(db, entry_price=100.5)
    fake = FakeFuturesClient(mark_price="100.6")
    patch_live(db, monkeypatch, fake)
    plan = {**PLAN, "id": order.id}
    results = apply_plans(db, [plan])
    assert results[0]["ok"] is True
    assert (order.sl, order.tp) == (99.5, 103.5)
    assert fake.cancel_calls == [("BTCUSDT", 101), ("BTCUSDT", 102)]
    placed = market_orders(fake)
    assert [p["type"] for p in placed] == ["STOP_MARKET", "TAKE_PROFIT_MARKET"]
    assert [p["triggerPrice"] for p in placed] == ["99.5", "103.5"]


def test_apply_skips_order_that_is_no_longer_open(db, monkeypatch):
    order = seed_open(db, status="CLOSED")
    fake = FakeFuturesClient(mark_price="100.6")
    patch_live(db, monkeypatch, fake)
    results = apply_plans(db, [{**PLAN, "id": order.id}])
    assert results[0]["ok"] is False
    assert results[0]["why"] == "no_longer_open"
    assert fake.calls == [] and fake.cancel_calls == []


def test_apply_skips_when_mark_moved_past_new_stop(db, monkeypatch):
    # dry-run 与 apply 之间价格穿过新止损：不得挂出一张立刻触发的止损单
    order = seed_open(db, entry_price=100.5)
    fake = FakeFuturesClient(mark_price="99.4")
    patch_live(db, monkeypatch, fake)
    results = apply_plans(db, [{**PLAN, "id": order.id}])
    assert results[0]["ok"] is False
    assert results[0]["why"] == "mark_moved_past_new_level"
    assert fake.calls == [] and fake.cancel_calls == []
    assert (order.sl, order.tp) == (99.0, 103.0)


class FailOnceTpClient(FakeFuturesClient):
    """只让第一次挂止盈失败，用于覆盖「挂新单失败后挂回旧水平」的兜底。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tp_attempts = 0

    def new_algo_order(self, **params):
        if params.get("type") == "TAKE_PROFIT_MARKET":
            self.tp_attempts += 1
            if self.tp_attempts == 1:
                raise BinanceApiError("tp rejected", -2021)
        return super().new_algo_order(**params)


def test_apply_restores_old_levels_when_new_placement_fails(db, monkeypatch):
    # modify_live 先撤后挂，失败会让仓位裸奔；兜底必须把旧水平挂回去
    order = seed_open(db, entry_price=100.5)
    fake = FailOnceTpClient(mark_price="100.6")
    patch_live(db, monkeypatch, fake)
    results = apply_plans(db, [{**PLAN, "id": order.id}])
    assert results[0]["ok"] is False
    assert results[0]["restored_old_levels"] is True
    assert fake.tp_attempts == 2
    assert (order.sl, order.tp) == (99.0, 103.0)
    assert [p["triggerPrice"] for p in market_orders(fake)][-2:] == ["99", "103"]


def test_apply_reports_unrestored_when_recovery_also_fails(db, monkeypatch):
    order = seed_open(db, entry_price=100.5)
    fake = FakeFuturesClient(mark_price="100.6", fail_tp=True)
    patch_live(db, monkeypatch, fake)
    results = apply_plans(db, [{**PLAN, "id": order.id}])
    assert results[0]["ok"] is False
    assert results[0]["restored_old_levels"] is False
    assert "modify_failed" in results[0]["why"]
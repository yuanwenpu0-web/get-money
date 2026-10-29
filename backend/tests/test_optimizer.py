"""参数寻优与币种级覆盖的单元测试。

进程池不在单测里跑真实数据（慢且吃核），这里覆盖「选参规则」与「参数层级」——
那两处出错的代价最高：网格抽样偏了会漏掉好参数，参数层级接错会让回测和实盘
跑两套参数而没人发现。
"""

import json

import pytest

from app.backtest.optimizer import (
    OptimizationRequest,
    VERDICT_RANK,
    _verdict,
    build_grid,
    build_tasks,
    default_workers,
    diff_params,
    grid_size,
    pick_best_per_symbol,
)
from app.models import StrategyBinding, SymbolParam, SymbolPool, StrategyConfig
from app.state import all_state
from app.strategy.bindings import active_bindings, binding_overrides, resolve_order_params
from app.strategy.registry import get_spec
from app.strategy.rules import resolve_params
from app.strategy.symbol_params import (
    delete_overrides,
    get_override,
    load_map,
    merged_params,
    set_overrides,
)
from app.config import DEFAULT_STATE
from app.state import set_state

CODE = "absorption_reversal"


def _spec():
    return get_spec(CODE)


def _base(overrides=None):
    state = dict(DEFAULT_STATE)
    return resolve_params(CODE, state, overrides, interval="15m")


# ---------------------------------------------------------------- 网格


def test_grid_size_is_cartesian_product():
    spec = _spec()
    expected = 1
    for values in spec.grid.values():
        expected *= len(values)
    assert grid_size(spec) == expected
    assert expected > 100


def test_build_grid_respects_requested_count_and_includes_base():
    spec = _spec()
    base = _base()
    grid = build_grid(spec, base, n=8)
    assert len(grid) == 8
    base_combo = {key: base[key] for key in spec.grid}
    assert base_combo in grid, "基准参数必须在候选集里，否则无法判断「不改」是否更好"


def test_build_grid_is_reproducible_but_seed_sensitive():
    spec = _spec()
    base = _base()
    first = build_grid(spec, base, n=12, seed=11)
    again = build_grid(spec, base, n=12, seed=11)
    other = build_grid(spec, base, n=12, seed=99)
    assert first == again, "同种子必须完全一致，否则结果无法复现"
    assert first != other


def test_build_grid_covers_every_value_of_every_key():
    """均匀下标抽样的意义：每个参数档位都要被用到，否则等于把某个档位排除在外。"""
    spec = _spec()
    base = _base()
    grid = build_grid(spec, base, n=64, seed=3)
    base_combo = {key: base[key] for key in spec.grid}
    # 基准组合是被强制塞进来的对照（它的 min_score 可能不在候选档位里），
    # 覆盖度只看采样出来的那部分
    sampled = [combo for combo in grid if combo != base_combo]
    assert len(sampled) >= 60
    for key in spec.grid:
        seen = {combo[key] for combo in sampled}
        assert seen, key
        assert min(sum(1 for c in sampled if c[key] == v) for v in seen) >= 3, key


def test_build_grid_uses_whole_space_when_small_enough():
    spec = _spec()
    base = _base()
    small = {"delta_pct": (90.0, 95.0), "target_r": (2.0, 3.0)}
    grid = build_grid(spec, base, n=100, custom=small)
    assert len(grid) == 4


def test_build_grid_rejects_unknown_shape():
    spec = _spec()
    base = _base()
    with pytest.raises(ValueError):
        build_grid(spec, base, custom={"delta_pct": ()})


def test_diff_params_keeps_only_changes():
    base = {"a": 1, "b": 2}
    assert diff_params({"a": 1, "b": 3}, base) == {"b": 3}
    assert diff_params({"a": 1, "b": 2}, base) == {}


def test_default_workers_leaves_room_for_trading():
    assert 1 <= default_workers() <= 24


# ---------------------------------------------------------------- 判定


def test_verdict_prefers_explicit_states():
    assert _verdict({}, {}, 5) == "no_trades"
    assert _verdict({"total_trades": 2}, {"verdict": "positive"}, 10) == "insufficient"
    assert _verdict({"total_trades": 30}, {"verdict": "positive"}, 5) == "pass"
    assert _verdict({"total_trades": 30}, {"verdict": "negative"}, 5) == "negative"
    assert _verdict({"total_trades": 30}, {"verdict": "inconclusive"}, 5) == "inconclusive"


# ---------------------------------------------------------------- 请求校验


def test_request_validation_bounds():
    ok = OptimizationRequest(strategy=CODE, interval="15m")
    ok.validate()
    for kwargs in (
        {"days": 10},
        {"days": 1000},
        {"train_ratio": 0.4},
        {"combos": 2},
        {"combos": 999},
        {"min_trades": 0},
        {"objective": "profit"},
        {"delta_source": "borrowed"},
    ):
        with pytest.raises(ValueError):
            OptimizationRequest(strategy=CODE, interval="15m", **kwargs).validate()
    with pytest.raises(ValueError):
        OptimizationRequest(strategy=CODE, interval="7m").validate()


# ---------------------------------------------------------------- 币种级覆盖


def test_symbol_override_wins_over_binding(db):
    set_state(db, "min_score", 0.75)
    pool = SymbolPool(code="p1", name="p1", kind="static", symbols=json.dumps(["BTCUSDT"]))
    db.add(pool)
    db.commit()
    db.add(
        StrategyBinding(
            strategy_code=CODE, pool_code="p1", interval="15m", enabled=1,
            priority=10, params=json.dumps({"target_r": 2.5}),
        )
    )
    db.commit()

    db.add(
        StrategyConfig(code=CODE, name=CODE, enabled=1, params="{}")
    )
    db.commit()

    state = all_state(db)
    baseline = merged_params(db, CODE, "BTCUSDT", "15m", overrides={"target_r": 2.5}, state=state)
    assert baseline["target_r"] == 2.5

    set_overrides(db, CODE, "15m", {"BTCUSDT": {"target_r": 4.0}}, source="optimization", run_id=7)
    after = merged_params(db, CODE, "BTCUSDT", "15m", overrides={"target_r": 2.5}, state=state)
    assert after["target_r"] == 4.0
    # 其它币种不受影响
    other = merged_params(db, CODE, "ETHUSDT", "15m", overrides={"target_r": 2.5}, state=state)
    assert other["target_r"] == 2.5


def test_active_bindings_expose_per_symbol_params(db):
    db.add(
        SymbolPool(
            code="p2", name="p2", kind="static",
            symbols=json.dumps(["BTCUSDT", "ETHUSDT"]),
        )
    )
    db.add(StrategyConfig(code=CODE, name=CODE, enabled=1, params="{}"))
    db.commit()
    db.add(
        StrategyBinding(
            strategy_code=CODE, pool_code="p2", interval="15m", enabled=1, priority=5, params="{}"
        )
    )
    db.commit()
    set_overrides(db, CODE, "15m", {"BTCUSDT": {"max_hold_bars": 60}}, source="manual")

    bindings, errors = active_bindings(db, state=all_state(db))
    assert not errors
    assert len(bindings) == 1
    binding = bindings[0]
    assert binding.params["max_hold_bars"] == 30
    assert binding.params_for("BTCUSDT")["max_hold_bars"] == 60
    assert binding.params_for("ETHUSDT")["max_hold_bars"] == 30
    assert binding.params_for("UNKNOWNUSDT")["max_hold_bars"] == 30


def test_binding_overrides_merges_symbol_layer_for_backtest(db):
    db.add(SymbolPool(code="p3", name="p3", kind="static", symbols=json.dumps(["BTCUSDT"])))
    db.commit()
    db.add(
        StrategyBinding(
            strategy_code=CODE, pool_code="p3", interval="15m", enabled=1,
            priority=1, params=json.dumps({"target_r": 3.0}),
        )
    )
    db.commit()
    set_overrides(db, CODE, "15m", {"BTCUSDT": {"target_r": 1.5}})
    merged = binding_overrides(db, CODE, "p3", "15m", symbol="BTCUSDT")
    assert merged["target_r"] == 1.5
    # 不带 symbol 时只看绑定层
    assert binding_overrides(db, CODE, "p3", "15m")["target_r"] == 3.0
    # 无绑定但有覆盖时也要返回覆盖层
    assert binding_overrides(db, CODE, "nope", "15m", symbol="BTCUSDT")["target_r"] == 1.5


def test_resolve_order_params_falls_back_to_symbol_override(db):
    db.add(StrategyConfig(code=CODE, name=CODE, enabled=1, params=json.dumps({"target_r": 2.0})))
    db.commit()
    set_overrides(db, CODE, "15m", {"BTCUSDT": {"target_r": 5.0}})
    params = resolve_order_params(db, CODE, "15m", pool="", symbol="BTCUSDT")
    assert params["target_r"] == 5.0
    other = resolve_order_params(db, CODE, "15m", pool="", symbol="SOLUSDT")
    assert other["target_r"] == 2.0


def test_set_overrides_rejects_invalid_and_costs(db):
    with pytest.raises(ValueError):
        set_overrides(db, CODE, "15m", {"BTCUSDT": {"bogus_key": 1}})
    with pytest.raises(ValueError):
        set_overrides(db, CODE, "15m", {"BTCUSDT": {"fee_rate": 0.0}})
    with pytest.raises(ValueError):
        set_overrides(db, CODE, "15m", {"BTCUSDT": {"target_r": 999}})
    with pytest.raises(ValueError):
        set_overrides(db, CODE, "15m", {"BTCUSDT": {}})
    # 全部校验失败，库里不留半条数据
    assert get_override(db, CODE, "BTCUSDT", "15m") == {}


def test_set_overrides_is_idempotent_and_deletable(db):
    assert set_overrides(db, CODE, "15m", {"BTCUSDT": {"target_r": 3}}) == 1
    assert set_overrides(db, CODE, "15m", {"BTCUSDT": {"target_r": 4}}) == 1
    assert db.query(SymbolParam).count() == 1
    assert get_override(db, CODE, "BTCUSDT", "15m")["target_r"] == 4
    assert delete_overrides(db, CODE, "15m", ["BTCUSDT"]) == 1
    assert db.query(SymbolParam).count() == 0


def test_load_map_slices_by_strategy_and_interval(db):
    set_overrides(db, CODE, "15m", {"BTCUSDT": {"target_r": 3}})
    set_overrides(db, CODE, "1h", {"ETHUSDT": {"target_r": 4}})
    everything = load_map(db)
    assert len(everything) == 2
    assert len(load_map(db, CODE, "15m")) == 1
    assert (CODE, "BTCUSDT", "15m") in everything


def test_broken_override_row_does_not_break_binding(db):
    db.add(SymbolPool(code="p4", name="p4", kind="static", symbols=json.dumps(["BTCUSDT"])))
    db.add(StrategyConfig(code=CODE, name=CODE, enabled=1, params="{}"))
    db.commit()
    db.add(
        StrategyBinding(
            strategy_code=CODE, pool_code="p4", interval="15m", enabled=1, priority=1, params="{}"
        )
    )
    db.commit()
    # 手改库写进一个非法值：应回落到绑定参数，而不是让这个币种消失
    db.add(
        SymbolParam(
            strategy_code=CODE, symbol="BTCUSDT", interval="15m",
            params=json.dumps({"target_r": 9999}), source="manual",
        )
    )
    db.commit()
    bindings, _ = active_bindings(db, state=all_state(db))
    # 非法覆盖被丢弃 → 回落到绑定/策略默认，币种不能因此从巡检里消失
    assert bindings[0].params_for("BTCUSDT")["target_r"] == 2.0


# ---------------------------------------------------------------- 多周期寻优


def test_interval_list_dedupes_and_defaults():
    req = OptimizationRequest(strategy=CODE, interval="15m")
    assert req.interval_list == ["15m"]
    req = OptimizationRequest(strategy=CODE, interval="15m", intervals=["4h", "1h", "4h", ""])
    assert req.interval_list == ["4h", "1h"]


def test_validate_rejects_unknown_interval_in_list():
    OptimizationRequest(strategy=CODE, intervals=["1h", "4h"]).validate()
    with pytest.raises(ValueError):
        OptimizationRequest(strategy=CODE, intervals=["1h", "7m"]).validate()


def test_pick_best_per_symbol_prefers_stronger_verdict():
    items = [
        {"symbol": "BTCUSDT", "interval": "15m", "verdict": "no_trades", "score": 9.0, "valid": {}},
        {"symbol": "BTCUSDT", "interval": "4h", "verdict": "negative", "score": -0.4, "valid": {"total_trades": 20}},
        {"symbol": "BTCUSDT", "interval": "1h", "verdict": "inconclusive", "score": 0.1, "valid": {"total_trades": 31}},
    ]
    best = pick_best_per_symbol(items)
    # no_trades 的分数是 0 分制里的最高值也不能赢：那是「没交易」不是「赚最多」
    assert best["BTCUSDT"]["interval"] == "1h"


def test_pick_best_per_symbol_breaks_ties_by_score_then_trades():
    items = [
        {"symbol": "ETHUSDT", "interval": "1h", "verdict": "pass", "score": 0.5, "valid": {"total_trades": 40}},
        {"symbol": "ETHUSDT", "interval": "4h", "verdict": "pass", "score": 0.9, "valid": {"total_trades": 12}},
    ]
    assert pick_best_per_symbol(items)["ETHUSDT"]["interval"] == "4h"
    items[1]["score"] = 0.5
    # 同分看样本量
    assert pick_best_per_symbol(items)["ETHUSDT"]["interval"] == "1h"


def test_verdict_rank_covers_every_verdict():
    for key in ("pass", "inconclusive", "negative", "insufficient", "no_trades"):
        assert key in VERDICT_RANK


def test_build_tasks_expands_symbol_interval_pairs():
    request = OptimizationRequest(
        strategy=CODE, intervals=["1h", "4h"], days=365,
        initial=10000.0, fee=0.0005, slippage=0.0002,
    )
    meta = {
        "base_params": {"target_r": 2.0},
        "per_interval": {
            "1h": {"base_params": {"target_r": 2.0}},
            "4h": {"base_params": {"target_r": 3.0}},
        },
    }
    grids = {"1h": [{"a": 1}, {"a": 2}], "4h": [{"a": 3}]}
    symbols = {"1h": ["BTCUSDT", "ETHUSDT"], "4h": ["BTCUSDT"]}
    tasks = build_tasks(request, meta, grids, symbols)
    assert len(tasks) == 3
    keys = {task["key"] for task in tasks}
    assert keys == {"BTCUSDT@1h", "ETHUSDT@1h", "BTCUSDT@4h"}
    by_key = {task["key"]: task for task in tasks}
    # 每个周期用自己的基准参数与网格，不能串
    assert by_key["BTCUSDT@4h"]["base_params"]["target_r"] == 3.0
    assert by_key["BTCUSDT@1h"]["grid"] == [{"a": 1}, {"a": 2}]
    # min_bars 随周期缩放：4h 的 365 天只有 1h 的四分之一根数
    assert by_key["BTCUSDT@4h"]["min_bars"] < by_key["BTCUSDT@1h"]["min_bars"]


def test_build_tasks_skips_empty_interval():
    request = OptimizationRequest(strategy=CODE, intervals=["1h", "4h"], initial=1.0, fee=0.0, slippage=0.0)
    meta = {"base_params": {}, "per_interval": {}}
    tasks = build_tasks(
        request, meta, {"1h": [{}], "4h": [{}]}, {"1h": ["BTCUSDT"], "4h": []}
    )
    assert [task["key"] for task in tasks] == ["BTCUSDT@1h"]


def test_pooled_ci_requires_minimum_trades():
    """池化 CI 在样本过少时必须判 insufficient，不能给出「显著为正」。

    实测 4h × 365 天全市场只有 4 笔成交，bootstrap 在 n=4 上会给出
    CI [0.86%, 10.48%] 这种假精确读数。
    """
    from app.backtest.optimizer import MIN_POOLED_TRADES
    from app.backtest.significance import bootstrap_expectancy_ci

    few = [{"return_pct": 5.0} for _ in range(4)]
    loose = bootstrap_expectancy_ci(few)
    assert loose["verdict"] == "positive"  # 不设门槛时确实会误报

    guarded = bootstrap_expectancy_ci(few, min_trades=MIN_POOLED_TRADES)
    assert guarded["verdict"] == "insufficient"
    # 数值照给：判不住不等于不能看
    assert guarded["trades"] == 4
    assert guarded["ci_low_pct"] > 0


def test_pooled_ci_with_enough_trades_still_judges():
    from app.backtest.significance import bootstrap_expectancy_ci

    many = [{"return_pct": 0.5} for _ in range(40)]
    out = bootstrap_expectancy_ci(many, min_trades=30)
    assert out["verdict"] == "positive"


def test_reap_stale_runs_marks_orphans_failed(db):
    """进程重启后遗留的 pending/running 任务必须被标失败。

    否则前端永远显示「运行中」，用户以为还在跑，而线程其实早没了。
    """
    from app.api.optimizations import reap_stale_runs
    from app.models import OptimizationRun

    db.add_all(
        [
            OptimizationRun(strategy=CODE, interval="15m", status="running", config="{}"),
            OptimizationRun(strategy=CODE, interval="15m", status="pending", config="{}"),
            OptimizationRun(strategy=CODE, interval="15m", status="done", config="{}"),
        ]
    )
    db.commit()

    assert reap_stale_runs(db) == 2
    statuses = sorted(row.status for row in db.query(OptimizationRun).all())
    assert statuses == ["done", "failed", "failed"]
    interrupted = db.query(OptimizationRun).filter(OptimizationRun.status == "failed").first()
    assert "进程重启" in interrupted.error
    assert interrupted.finished_at is not None
    # 幂等：再跑一次没有可清理的
    assert reap_stale_runs(db) == 0

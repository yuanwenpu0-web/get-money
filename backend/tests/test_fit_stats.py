"""`fit_stats` 的判定测试：方向对齐、双基准、覆盖缺失、四道关与收缩。

这些用例是**故意造出来的**：合成张量里每条边的符号、分段、币间一致性都是人为设定的，
用来验证判定链条会不会把「没有 edge」误判成「有」，或把「有」误判成「没有」。
"""
from __future__ import annotations

import numpy as np
import pytest

from app.research.fit_events import FIELDS, FI
from app.research.fit_stats import (
    ROUND_TRIP,
    Evaluator,
    apply_fdr,
    bh_qvalues,
    boot_p,
    shrink,
    verdict,
)
from app.research.fit_stats import GateConfig

NH = 2
HORIZONS = [6, 24]


def cell(n=20, x_mean=0.005, r_mean=0.008, cs_mean=0.003, hit=100, segments=(5, 5, 5, 5),
         sd=0.0):
    """造一个币一行聚合量：均值、分段、触发数全部显式给定。

    `sd` 是给 `sum_sq` 用的**每笔标准差**：Σx² = n(μ²+σ²)。默认 0 会让 iid 标准误退化，
    只有要比 CI 宽度时才需要给非 0（见 test_bootstrap_is_symbol_block_not_event_iid）。
    """
    row = np.zeros((NH, len(FIELDS)))
    for hi in range(NH):
        x = x_mean * n
        row[hi, FI["n"]] = n
        row[hi, FI["sum_r"]] = r_mean * n
        row[hi, FI["sum_x"]] = x
        row[hi, FI["sum_sq"]] = n * (x_mean * x_mean + sd * sd)
        row[hi, FI["n_above"]] = n
        row[hi, FI["sum_above"]] = x
        row[hi, FI["sum_cs"]] = cs_mean * n
        row[hi, FI["n_cs"]] = n
        row[hi, FI["sum_r_cs"]] = r_mean * n
        row[hi, FI["hit_bars"]] = hit
        for s, sn in enumerate(segments):
            row[hi, FI[f"seg_n{s}"]] = sn
            row[hi, FI[f"seg_x{s}"]] = x_mean * sn
    return row


def build(rows: dict[str, dict[str, np.ndarray]], symbols: list[str]):
    keys = sorted(rows)
    matrix = np.zeros((len(keys), len(symbols), NH, len(FIELDS)))
    covered = np.zeros((len(keys), len(symbols)), dtype=bool)
    for ki, key in enumerate(keys):
        for sym, arr in rows[key].items():
            covered[ki, symbols.index(sym)] = True
            matrix[ki, symbols.index(sym)] = arr
    return keys, symbols, matrix, covered


def evaluator(rows, symbols, meta, bars=None, cfg=None):
    keys, syms, matrix, covered = build(rows, symbols)
    bars = bars or {s: 10_000 for s in syms}
    return Evaluator(keys, syms, matrix, covered, meta, bars, cfg=cfg)


ALL8 = [f"S{i}USDT" for i in range(8)]
# 生产里最小的币种池也有 20 个币；要考察「能拿到 candidate」的路径，夹具至少要有
# blocks_for_candidate 个币块，否则会被「币块太少 → p 打到离散下界」这道门槛降级。
ALL16 = [f"S{i}USDT" for i in range(16)]


def make_scopes(symbols):
    return {"all": list(symbols)}


def test_positive_uniform_edge_becomes_candidate():
    ev = evaluator(
        {"ind_a|up": {s: cell() for s in ALL16}},
        ALL16,
        {"ind_a|up": {"direction": 1, "direction_source": "declared", "matched": ["up"]}},
    )
    rows = apply_fdr(ev.evaluate(HORIZONS, make_scopes(ALL16)))
    uncond = [r for r in rows if r["benchmark"] == "unconditional" and r["horizon_bars"] == 6]
    assert len(uncond) == 1
    r = uncond[0]
    assert r["events"] == 320
    assert r["mean_gross_pct"] == pytest.approx(0.5)
    assert r["mean_net_pct"] == pytest.approx(0.5 - ROUND_TRIP * 100)
    assert r["ci_low_pct"] > 0
    assert r["seg_consistency"] == 1.0
    assert r["symbol_sign_rate"] == 1.0
    assert r["verdict"] == "candidate"


def test_bearish_column_is_tested_short_not_forced_long():
    """老脚本只看多头：看跌列的原始超额为负，永远不可能「有 edge」。方向对齐后应能判正。"""
    meta = {"ind_b|down": {"direction": -1, "direction_source": "declared", "matched": ["down"]}}
    ev = evaluator({"ind_b|down": {s: cell(x_mean=-0.005) for s in ALL16}}, ALL16, meta)
    rows = apply_fdr(ev.evaluate(HORIZONS, make_scopes(ALL16)))
    r = [x for x in rows if x["benchmark"] == "unconditional"][0]
    assert r["direction"] == -1
    assert r["mean_net_pct"] > 0
    assert r["verdict"] == "candidate"


def test_only_declared_direction_is_tested_once_but_ambiguous_twice():
    rows = apply_fdr(
        evaluator(
            {
                "ind_a|up": {s: cell() for s in ALL16},
                "ind_c|flag": {s: cell() for s in ALL16},
            },
            ALL16,
            {
                "ind_a|up": {"direction": 1, "direction_source": "declared", "matched": []},
                "ind_c|flag": {"direction": 0, "direction_source": "ambiguous", "matched": []},
            },
        ).evaluate(HORIZONS, make_scopes(ALL16))
    )
    by_key = {}
    for r in rows:
        by_key.setdefault(r["key"], set()).add(r["direction"])
    assert by_key["ind_a|up"] == {1}
    assert by_key["ind_c|flag"] == {1, -1}
    amb = [r for r in rows if r["key"] == "ind_c|flag"]
    # 方向不明的列两个方向都试：反向那行当然还是 no_edge，但任何一行都不许爬到 candidate
    assert {r["verdict"] for r in amb} <= {"exploratory", "no_edge"}
    assert any(r["verdict"] == "exploratory" for r in amb)
    dec = [r for r in rows if r["key"] == "ind_a|up" and r["direction"] == 1]
    assert all(r["verdict"] == "candidate" for r in dec), "同分布的已声明列该拿到 candidate"


def test_missing_symbol_column_is_not_counted_as_zero_edge():
    """某指标只在部分币上产出布尔列：缺失的币必须整列剔除，不能按 0 参与池化。"""
    present = ALL8[:3]
    ev = evaluator(
        {"ind_a|up": {s: cell(x_mean=0.02) for s in present}},
        ALL8,
        {"ind_a|up": {"direction": 1, "direction_source": "declared", "matched": []}},
    )
    r = apply_fdr(ev.evaluate(HORIZONS, make_scopes(ALL8)))[0]
    assert r["events"] == 60 and r["symbols_covered"] == 3
    assert r["mean_gross_pct"] == pytest.approx(2.0), "按 8 个币池化会变成 0.75，那是假的"
    # 触发率分母同样只算覆盖币的 bar：3×10000 而不是 8×10000
    assert r["hit_rate"] == pytest.approx(300 / 30_000, abs=1e-6)


def test_cross_section_uses_only_events_with_aligned_basket():
    """横截面基准只对「该时刻能对齐 ≥5 个币」的事件求均值：分母必须是 n_cs。"""
    rows = {s: cell(n=20, r_mean=0.01, cs_mean=0.004) for s in ALL8}
    for s, arr in rows.items():
        arr[:, FI["n_cs"]] = 5  # 只有 5 笔能对齐篮子
        arr[:, FI["sum_r_cs"]] = 0.01 * 5
        arr[:, FI["sum_cs"]] = 0.004 * 5
    ev = evaluator({"ind_a|up": rows}, ALL8,
                   {"ind_a|up": {"direction": 1, "direction_source": "declared", "matched": []}})
    out = apply_fdr(ev.evaluate(HORIZONS, make_scopes(ALL8)))
    cs = [r for r in out if r["benchmark"] == "cross_section"][0]
    un = [r for r in out if r["benchmark"] == "unconditional"][0]
    assert cs["events"] == 40 and un["events"] == 160
    assert cs["mean_gross_pct"] == pytest.approx(0.6)  # (0.01-0.004)*100
    assert un["mean_gross_pct"] == pytest.approx(0.5)
    assert cs["cs_coverage"] == pytest.approx(0.25)
    assert cs["win_rate"] is None, "胜率没有按 cs 可用性切分，cross_section 行不许报"


def test_negative_edge_is_no_edge_not_silently_dropped():
    ev = evaluator({"ind_a|up": {s: cell(x_mean=-0.01) for s in ALL8}}, ALL8,
                   {"ind_a|up": {"direction": 1, "direction_source": "declared", "matched": []}})
    r = apply_fdr(ev.evaluate(HORIZONS, make_scopes(ALL8)))[0]
    assert r["mean_net_pct"] < 0 and r["verdict"] == "no_edge"


def test_degenerate_hit_rate_columns_are_skipped():
    """恒真列（触发率 100%）与几乎不触发的列都不进排名。"""
    rows = {
        "ind_always|flag": {s: cell(hit=9_000) for s in ALL8},
        "ind_never|flag": {s: cell(hit=1) for s in ALL8},
        "ind_ok|flag": {s: cell(hit=100) for s in ALL8},
    }
    meta = {k: {"direction": 1, "direction_source": "declared", "matched": []} for k in rows}
    out = apply_fdr(evaluator(rows, ALL8, meta).evaluate(HORIZONS, make_scopes(ALL8)))
    assert {r["key"] for r in out} == {"ind_ok|flag"}


def test_unstable_and_not_universal_are_blocked():
    # 事件全挤在第 1 段：有效段只剩 1 个，时间稳定性无从证明 → unstable
    bad_seg = {s: cell(segments=(20, 0, 0, 0)) for s in ALL8}
    for arr in bad_seg.values():
        arr[:, FI["seg_x0"]] = -0.01
    # 一半币反号：这里故意把正号币的单笔均值做得远大于反号币（0.10% vs −0.01% 量级），
    # 让池化 CI 撑过第 1 关，从而真正考察「币间同号率」这道关而不是被 CI 提前拦掉。
    mixed = {s: cell(x_mean=0.10 if i < 4 else -0.01) for i, s in enumerate(ALL8)}
    meta = {
        "ind_a|seg": {"direction": 1, "direction_source": "declared", "matched": []},
        "ind_b|mixed": {"direction": 1, "direction_source": "declared", "matched": []},
    }
    ev = evaluator({"ind_a|seg": bad_seg, "ind_b|mixed": mixed}, ALL8, meta)
    out = apply_fdr(ev.evaluate(HORIZONS, make_scopes(ALL8)))
    verdicts = {r["key"]: r["verdict"] for r in out if r["benchmark"] == "unconditional"}
    by_key = {}
    for r in out:
        by_key.setdefault(r["key"], []).append(r)
    assert verdicts["ind_a|seg"] == "unstable"
    assert by_key["ind_a|seg"][0]["seg_valid"] == 1
    assert verdicts["ind_b|mixed"] in {"not_universal", "inconclusive"}, verdicts["ind_b|mixed"]
    if verdicts["ind_b|mixed"] == "not_universal":
        assert by_key["ind_b|mixed"][0]["symbol_sign_rate"] < GateConfig().sym_sign_min


def test_cost_fragile_when_double_cost_kills_it():
    """毛利只有 0.15%：两倍成本下转负 → 实盘不可信。"""
    ev = evaluator({"ind_a|up": {s: cell(x_mean=0.0015) for s in ALL8}}, ALL8,
                   {"ind_a|up": {"direction": 1, "direction_source": "declared", "matched": []}})
    r = apply_fdr(ev.evaluate(HORIZONS, make_scopes(ALL8)))[0]
    assert r["mean_net_pct"] > 0
    assert r["cost_stress_pct"]["x2"] < 0
    assert r["verdict"] == "cost_fragile"


def test_candidate_needs_enough_symbol_blocks():
    """币块自助法的 p 有离散下界：k 个币全同号时 p≈2^(1-k)/(B+1)，币少时 q<0.10 是免费的。

    6 币探针实测出 116 个 candidate 就是这个机制，不是 edge。所以「四道关全过但币块不
    够」必须降级为 underpowered_blocks，数字照留、标签不上候选。
    """
    cfg = GateConfig()

    def run(n):
        syms = [f"S{i}USDT" for i in range(n)]
        ev = evaluator({"ind_a|up": {s: cell() for s in syms}}, syms,
                       {"ind_a|up": {"direction": 1, "direction_source": "declared",
                                     "matched": []}})
        rows = apply_fdr(ev.evaluate(HORIZONS, make_scopes(syms)))
        return [r for r in rows if r["benchmark"] == "unconditional" and r["horizon_bars"] == 6][0]

    low, high = run(cfg.blocks_for_candidate - 1), run(cfg.blocks_for_candidate)
    assert low["symbols_in_stat"] == cfg.blocks_for_candidate - 1
    # 四道关本身全过——降级只是因为币块数，不是因为证据不足
    assert low["ci_low_pct"] > 0 and low["q_value"] < cfg.q_alpha
    assert low["seg_consistency"] == 1.0 and low["cost_stress_pct"]["x2"] > 0
    assert low["verdict"] == "underpowered_blocks"
    assert high["verdict"] == "candidate"
    assert high["mean_net_pct"] == low["mean_net_pct"], "门槛只改标签，不许动数字"


def test_interaction_test_marks_pool_effect_as_no_heterogeneity():
    """格子结论必须证明「这个池和其余样本确实不同」，否则只是把整体结论复读一遍。"""
    # 6 个币进格子：120 笔 > cell_events_min，剩下 2 个币 40 笔 > pooled_events_min，
    # 否则先被 underpowered_cell 拦掉，考察不到交互检验本身。
    scopes = {"all": ALL8, "liq_q1": ALL8[:6]}
    ev = evaluator({"ind_a|up": {s: cell() for s in ALL8}}, ALL8,
                   {"ind_a|up": {"direction": 1, "direction_source": "declared", "matched": []}})
    out = apply_fdr(ev.evaluate(HORIZONS, scopes))
    cell_rows = [r for r in out if r["scope"] == "liq_q1"]
    assert cell_rows, "scope 只有一部分币种时也要出结果"
    for r in cell_rows:
        assert r["events"] == 120
        assert r["interaction"] is not None and r["interaction"]["rest_events"] == 40
        assert r["interaction"]["p"] > GateConfig().interaction_alpha
        assert verdict(r, cell=True) == "no_heterogeneity"


def test_scope_with_single_symbol_is_skipped():
    ev = evaluator({"ind_a|up": {s: cell() for s in ALL8}}, ALL8,
                   {"ind_a|up": {"direction": 1, "direction_source": "declared", "matched": []}})
    assert ev.evaluate(HORIZONS, {"solo": ALL8[:1]}) == []


def test_bh_qvalues_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.03, 0.5])
    q = bh_qvalues(p)
    assert q[0] <= q[1] <= q[2] <= q[3]
    assert (q <= 1.0).all() and (q > 0).all()
    assert q[1] == pytest.approx(0.02 * 4 / 2)


def test_boot_p_never_reports_zero():
    rng = np.random.default_rng(0)
    values = np.abs(rng.normal(1, 0.1, 2000))  # 全部为正
    assert boot_p(values) == pytest.approx(1 / 2001)


def test_shrink_pulls_small_cells_to_pooled():
    cfg = GateConfig()
    pooled = {"mean_net_pct": 0.10}
    cell_row = {"mean_net_pct": 0.90, "events": 30, "ci_low_pct": 0.2, "q_value": 0.01,
                "seg_consistency": 1.0, "symbol_sign_rate": 1.0,
                "cost_stress_pct": {"x2": 0.5}, "interaction": {"p": 0.02},
                "symbols_with_events": 6, "direction_source": "declared"}
    out = shrink(cell_row, pooled, k=cfg.shrink_k)
    w = 30 / (30 + cfg.shrink_k)
    assert out["shrink_weight"] == pytest.approx(round(w, 3))
    assert out["shrunk_net_pct"] == pytest.approx(round(w * 0.90 + (1 - w) * 0.10, 4))
    assert out["delta_vs_pooled_pct"] == pytest.approx(0.8)


def test_bootstrap_is_symbol_block_not_event_iid():
    """一个币贡献 160 笔、其余币各 1 笔时，事件级 iid 会把 CI 做得过窄。

    三个小币的单笔均值与大币不同号：按事件重抽时它们只占 3/163，几乎不影响分布；
    按币重抽时「大币没被抽到」是 10% 量级的事件，整个均值跟着翻负——这才是真实不确定性。
    """
    rows = {ALL8[0]: cell(n=160, x_mean=0.005, sd=0.02)}
    for sym, x in zip(ALL8[1:4], (-0.03, -0.02, -0.01)):
        rows[sym] = cell(n=1, x_mean=x, sd=0.02)
    ev = evaluator({"ind_a|up": rows}, ALL8,
                   {"ind_a|up": {"direction": 1, "direction_source": "declared", "matched": []}},
                   cfg=GateConfig(n_boot=400))
    r = apply_fdr(ev.evaluate(HORIZONS, make_scopes(ALL8)))[0]
    assert r["events"] == 163
    assert r["symbols_in_stat"] == 1, "样本量不足的币不进入币间一致性统计"
    iid_half = 1.96 * r["se_iid_pct"]  # 两个量都是百分点，直接可比
    boot_half = r["mean_net_pct"] - r["ci_low_pct"]
    assert boot_half > 3 * iid_half, (boot_half, iid_half)

"""P17 / H-E 臂构造与汇总单测（不联网、不连 DB）。

最关键的一条：**所有臂必须共用同一份逐币成交**。若某个臂重新撮合了一遍，
`posthoc − pit` 的差就不再是纯粹的选币泄漏，第 5 节的诊断也就失去意义。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.research.pit_pool_arms import build_arms, summarise  # noqa: E402


def _ranked():
    return [
        {"symbol": "CCCUSDT", "quote_volume": 300.0, "volatility": 0.5},
        {"symbol": "AAAUSDT", "quote_volume": 200.0, "volatility": 0.9},
        {"symbol": "BBBUSDT", "quote_volume": 100.0, "volatility": 0.1},
    ]


def _meta(index=0):
    return {"index": index, "start_iso": "2025-11-12", "end_iso": "2026-01-24",
            "ranked": _ranked(), "eligible": ["AAAUSDT", "BBBUSDT", "CCCUSDT"]}


def _result(symbol, returns, acct=1.0, rnd=(0.5, -0.5)):
    return {"symbol": symbol, "status": "ok",
            "folds": [{"index": 0, "trades": [[r, "long"] for r in returns], "bars": 900,
                       "account_return_pct": acct, "random_samples": list(rnd)}]}


def _results():
    return [_result("AAAUSDT", [5.0, 5.0], acct=4.0, rnd=(2.0, 0.0)),
            _result("BBBUSDT", [-1.0], acct=-2.0, rnd=(0.0, -2.0)),
            _result("CCCUSDT", [2.0, 2.0, 2.0], acct=1.0, rnd=(1.0, 1.0))]


def test_every_arm_reuses_the_same_per_symbol_trades():
    arms = build_arms([_meta()], _results(), (2,), 2)
    # CCC 的三笔 +2.0 在 all 与 pit_liq_2 里必须是同一批数字，不是重新撮合的结果
    assert arms["all"]["trades"].count([2.0, "long"]) == 3
    assert arms["pit_liq_2"]["trades"].count([2.0, "long"]) == 3
    assert len(arms["all"]["trades"]) == 6
    assert len(arms["pit_liq_2"]["trades"]) == 5


def test_posthoc_arm_ranks_by_realized_return_and_is_the_cheating_arm():
    arms = build_arms([_meta()], _results(), (2,), 2)
    assert arms["posthoc_2"]["folds"][0]["members"] == ["AAAUSDT", "CCCUSDT"]


def test_liquidity_arm_follows_the_precomputed_pit_ranking():
    arms = build_arms([_meta()], _results(), (2,), 2)
    assert arms["pit_liq_2"]["folds"][0]["members"] == ["CCCUSDT", "AAAUSDT"]


def test_volatility_arm_ranks_within_the_liquidity_shortlist():
    arms = build_arms([_meta()], _results(), (2,), 2)
    assert arms["pit_vol_2"]["folds"][0]["members"] == ["AAAUSDT", "CCCUSDT"]


def test_ranked_symbol_without_fold_data_is_dropped_not_fabricated():
    meta = _meta()
    meta["ranked"].insert(0, {"symbol": "ZZZUSDT", "quote_volume": 999.0, "volatility": 9.9})
    arms = build_arms([meta], _results(), (2,), 2)
    assert "ZZZUSDT" not in arms["pit_liq_2"]["folds"][0]["members"]
    assert arms["pit_liq_2"]["folds"][0]["members"] == ["CCCUSDT", "AAAUSDT"]


def test_skipped_symbols_never_enter_any_arm():
    results = _results() + [{"symbol": "DDDUSDT", "status": "skipped", "reason": "无数据"}]
    arms = build_arms([_meta()], results, (2,), 2)
    assert "DDDUSDT" not in arms["all"]["folds"][0]["members"]


def test_summarise_counts_positive_folds_from_fold_means():
    arms = build_arms([_meta()], _results(), (2,), 2)
    out = summarise(arms)
    assert out["all"]["positive_folds"] == 1
    assert out["all"]["fold_means"] == [2.5]


def test_summarise_random_percentile_uses_equal_weight_mean_per_seed():
    arms = build_arms([_meta()], _results(), (2,), 2)
    out = summarise(arms)
    # all 成员账户均值 = (4 - 2 + 1)/3 = 1.0；种子0 随机均值 = (2+0+1)/3 = 1.0，
    # 种子1 = (0-2+1)/3 = -0.333 → 严格小于 1.0 的样本占 1/2
    assert out["all"]["account_mean_pct"] == 1.0
    assert out["all"]["percentile_rank"] == 0.5


def test_summarise_splits_long_and_short():
    results = [_result("AAAUSDT", [5.0], acct=4.0)]
    results[0]["folds"][0]["trades"] = [[5.0, "long"], [-3.0, "short"]]
    out = summarise(build_arms([_meta()], results, (2,), 2))
    assert out["all"]["side_split"]["long"]["trades"] == 1
    assert out["all"]["side_split"]["short"]["trades"] == 1
    assert out["all"]["side_split"]["short"]["mean_return_pct"] == -3.0


def test_empty_arm_yields_no_trades_not_a_fake_zero_return():
    meta = _meta()
    meta["ranked"] = []
    meta["eligible"] = ["AAAUSDT"]
    out = summarise(build_arms([meta], _results(), (2,), 2))
    assert out["pit_liq_2"]["pooled"]["verdict"] == "no_trades"
    assert out["pit_liq_2"]["percentile_rank"] is None


def test_same_member_set_in_different_order_gives_identical_pooled_ci():
    # 15m 只有 30 币时 pit_liq_30 与 pit_vol_30 成员集合相同、只是排序不同。
    # bootstrap 是有放回重采样，输入数组顺序不该改变结论；固定种子下必须逐位一致，
    # 否则同一批成交会因排序不同报出两个 CI，读者会误以为有差异。
    meta = _meta()
    meta["ranked"] = [
        {"symbol": "AAAUSDT", "quote_volume": 300.0, "volatility": 0.1},
        {"symbol": "BBBUSDT", "quote_volume": 200.0, "volatility": 0.9},
        {"symbol": "CCCUSDT", "quote_volume": 100.0, "volatility": 0.5},
    ]
    out = summarise(build_arms([meta], _results(), (3,), 2))
    liq, vol = out["pit_liq_3"], out["pit_vol_3"]
    assert liq["folds"][0]["members"] != vol["folds"][0]["members"]  # 排序确实不同
    assert set(liq["folds"][0]["members"]) == set(vol["folds"][0]["members"])
    assert liq["pooled"] == vol["pooled"]
    assert liq["account_mean_pct"] == vol["account_mean_pct"]
    assert liq["percentile_rank"] == vol["percentile_rank"]
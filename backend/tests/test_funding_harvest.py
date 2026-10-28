"""funding_harvest 纯函数单测（不触 DB/网络）。"""

import pandas as pd
import pytest

from app.strategy.funding_harvest import (
    COST_PER_NAME,
    match_spot_base,
    rebalance_cost,
    select_from_grid,
    turnover,
)


class TestMatchSpotBase:
    BASES = {"PEPE", "BONK", "BTC", "1000SATS"}

    def test_direct(self):
        assert match_spot_base("BTCUSDT", TestMatchSpotBase.BASES) == "BTC"

    def test_1000_prefix_swap(self):
        assert match_spot_base("1000PEPEUSDT", TestMatchSpotBase.BASES) == "PEPE"

    def test_1000_prefix_direct(self):
        assert match_spot_base("1000SATSUSDT", TestMatchSpotBase.BASES) == "1000SATS"

    def test_no_spot(self):
        assert match_spot_base("BROCCOLIF3BUSDT", TestMatchSpotBase.BASES) is None

    def test_non_usdt_suffix(self):
        assert match_spot_base("BTC", TestMatchSpotBase.BASES) == "BTC"


def _grid(cols: dict[str, list[float]], buckets: int = 30) -> pd.DataFrame:
    """构造规则 8h 网格：每列常数费率（最后一个桶起就有完整 trailing）。"""
    idx = pd.date_range("2026-10-01", periods=buckets, freq="8h", tz="UTC")
    return pd.DataFrame({s: [r] * buckets for s, r in cols.items()}, index=idx)


class TestSelectFromGrid:
    def test_ranks_descending_and_caps_k(self):
        grid = _grid({"AAAUSDT": 0.001, "BBBUSDT": 0.0005, "CCCUSDT": 0.0001})
        out = select_from_grid(grid, k=2)
        assert [t["symbol"] for t in out] == ["AAAUSDT", "BBBUSDT"]
        assert out[0]["trail_rate_8h"] == pytest.approx(0.001)

    def test_min_rate_filters_negative(self):
        grid = _grid({"AAAUSDT": 0.001, "NEGBUSDT": -0.0005, "ZEROUSDT": 0.0})
        out = select_from_grid(grid, k=5, min_rate=0.0)
        assert {t["symbol"] for t in out} == {"AAAUSDT"}

    def test_eligible_whitelist(self):
        grid = _grid({"AAAUSDT": 0.002, "BBBUSDT": 0.001})
        out = select_from_grid(grid, k=5, eligible={"BBBUSDT"})
        assert [t["symbol"] for t in out] == ["BBBUSDT"]

    def test_empty_grid(self):
        assert select_from_grid(pd.DataFrame(), k=3) == []

    def test_annualized_conversion(self):
        # 0.01%/8h → 年化 0.0001 * 3 * 365 = 10.95%
        grid = _grid({"AAAUSDT": 0.0001})
        out = select_from_grid(grid, k=1)
        assert out[0]["annualized_pct"] == pytest.approx(10.95)


class TestTurnover:
    def test_first_time_full(self):
        assert turnover(None, ["A", "B"]) == 1.0

    def test_identical_zero(self):
        assert turnover(["A", "B"], ["A", "B"]) == 0.0

    def test_partial(self):
        # 交集 1（B）/ 并集 4（A,B,C,D）→ 1 − 1/4 = 0.75
        assert turnover(["A", "B"], ["B", "C", "D"]) == pytest.approx(0.75)

    def test_empty_cur(self):
        assert turnover(["A"], []) == 1.0

    def test_both_empty(self):
        assert turnover([], []) == 0.0


class TestCost:
    def test_rebalance_cost_scales_with_turnover(self):
        assert rebalance_cost(None, ["A"]) == pytest.approx(COST_PER_NAME)
        assert rebalance_cost(["A", "B"], ["A", "B"]) == 0.0
        assert rebalance_cost(["A", "B"], ["B", "C"]) == pytest.approx(COST_PER_NAME * 2 / 3)
        assert rebalance_cost(["A", "B"], ["B", "C", "D"]) == pytest.approx(COST_PER_NAME * 0.75)

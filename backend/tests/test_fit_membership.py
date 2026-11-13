"""P18 币种池成员口径单测（不联网、不连 DB）。

两件事必须被钉住：
1. 自然月应有柱数不能溢出。pandas 3.x 的 `.dt.days_in_month` 是 int32，乘 86_400_000
   对任何 ≥25 天的月份都会翻负，于是 `partial` 恒为 False、残月从不被剔除——
   A 轮 Q 档中位数因此混进了窗口首尾的残月。
2. PIT 分档只能用「事件月之前已完成的月」，且必须显式报告哪些月根本无从定档。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.research import fit_data as fd  # noqa: E402


# --------------------------------------------------------------------- 溢出回归


@pytest.mark.parametrize("ym,days", [("2026-02", 28), ("2024-02", 29), ("2026-09", 30),
                                     ("2026-10", 31), ("2026-12", 31)])
@pytest.mark.parametrize("interval,per_day", [("5m", 288), ("15m", 96), ("30m", 48), ("1h", 24)])
def test_month_expected_bars_never_overflows(ym, days, interval, per_day):
    got = fd.month_expected_bars(pd.Series([ym]), interval)
    assert int(got.iloc[0]) == days * per_day
    assert int(got.iloc[0]) > 0, "负数会让 partial 判定恒为 False"


def test_month_expected_bars_keeps_input_index():
    ym = pd.Series(["2026-09", "2026-10"], index=[7, 3])
    got = fd.month_expected_bars(ym, "1h")
    assert list(got.index) == [7, 3]
    assert got.tolist() == [720, 744]


def test_truncated_month_is_flagged_partial():
    """窗口只盖到月中：bars 远小于应有根数，必须标 partial。"""
    frame = pd.DataFrame({"symbol": ["AUSDT", "AUSDT"], "ym": ["2026-09", "2026-11"],
                          "quote_volume": [1e9, 2e8], "bars": [720, 288]})
    out = fd.mark_partial_months(frame, "1h")
    assert out["partial"].tolist() == [False, True]
    assert out["expected_bars"].tolist() == [720, 720]  # 9 月与 11 月都是 30 天
    assert frame.get("partial") is None, "不许原地改调用方的帧"


def test_full_month_just_above_threshold_is_not_partial():
    frame = pd.DataFrame({"symbol": ["AUSDT"], "ym": ["2026-09"],
                          "quote_volume": [1e9], "bars": [504]})  # 0.7 × 720
    assert fd.mark_partial_months(frame, "1h")["partial"].tolist() == [False]


def test_fixed_tiers_drop_truncated_months():
    """残月必须真的被剔除：窗口内才上市的币只有残月，不该拿到档位。

    这正是 `assign_fixed_tiers` docstring 里写的危害（「月初那几天才上市的币会因为
    样本短而排到错误的档位」）。四个老币有完整月，NEW 只在残月出现：
    修好后 NEW 无完整月 → 不定档；溢出未修时 partial 恒 False → NEW 被塞进低档。
    """
    rows = []
    for sym, full, stub in [("AUSDT", 1e9, 3e8), ("BUSDT", 9e8, 2.7e8),
                            ("CUSDT", 8e8, 2.4e8), ("DUSDT", 1e7, 3e6)]:
        rows.append({"symbol": sym, "ym": "2026-09", "quote_volume": full, "bars": 720})
        rows.append({"symbol": sym, "ym": "2026-11", "quote_volume": stub, "bars": 240})
    rows.append({"symbol": "NEWUSDT", "ym": "2026-11", "quote_volume": 5e6, "bars": 240})
    frame = fd.mark_partial_months(pd.DataFrame(rows), "1h")
    assert frame.loc[frame["symbol"] == "NEWUSDT", "partial"].tolist() == [True]
    tiers = fd.assign_fixed_tiers(frame, groups=4)
    assert "NEWUSDT" not in tiers, "只有残月的币不该拿到档位"
    assert tiers["AUSDT"] == "Q1" and tiers["DUSDT"] == "Q4"
    contaminated = fd.assign_fixed_tiers(frame.assign(partial=False), groups=4)
    assert "NEWUSDT" in contaminated, "残月未被剔除时新币会被塞进档位——这就是溢出的实际危害"


# --------------------------------------------------------------------- PIT 分档


def _months(symbols, spec):
    """spec: {ym: {symbol: quote_volume}}，bars 一律给满（完整月）。"""
    rows = []
    for ym, per in spec.items():
        for sym, vol in per.items():
            rows.append({"symbol": sym, "ym": ym, "quote_volume": vol, "bars": 744})
    frame = pd.DataFrame(rows)
    return fd.mark_partial_months(frame, "1h")


def test_pit_tier_uses_only_strictly_prior_complete_months():
    from app.research.fit_membership import pit_tiers_by_month

    frame = _months(["A", "B", "C", "D"], {
        "2026-08": {"A": 10.0, "B": 9.0, "C": 8.0, "D": 1.0},
        "2026-09": {"A": 1.0, "B": 9.0, "C": 8.0, "D": 10.0},
        "2026-10": {"A": 1.0, "B": 9.0, "C": 8.0, "D": 10.0},
    })
    out = pit_tiers_by_month(frame, groups=2)
    # 8 月没有更早的完整月 → 无从定档，不许拿自己或未来月补
    assert out["2026-08"]["status"] == "NO_PRIOR_COMPLETE_MONTH"
    assert out["2026-08"]["tiers"] == {}
    # 9 月只能用 8 月：A 高、D 低
    assert out["2026-09"]["tiers"] == {"A": "Q1", "B": "Q1", "C": "Q2", "D": "Q2"}
    assert out["2026-09"]["prior_months"] == ["2026-08"]
    # 10 月用 8+9 月中位数：A=(10+1)/2=5.5, B=9, C=8, D=(1+10)/2=5.5 → B/C 高
    assert out["2026-10"]["tiers"] == {"B": "Q1", "C": "Q1", "A": "Q2", "D": "Q2"}
    assert out["2026-10"]["prior_months"] == ["2026-08", "2026-09"]


def test_pit_tier_never_reads_the_event_month_itself():
    from app.research.fit_membership import pit_tiers_by_month

    frame = _months(["A", "B", "C", "D"], {
        "2026-08": {"A": 10.0, "B": 9.0, "C": 8.0, "D": 1.0},
        "2026-09": {"A": 1000.0, "B": 1.0, "C": 1.0, "D": 1.0},
    })
    out = pit_tiers_by_month(frame, groups=2)
    # 9 月自己把 A抬到 1000、把 B/C 压到 1：若偷看当月，B/C 会掉档
    assert out["2026-09"]["tiers"] == {"A": "Q1", "B": "Q1", "C": "Q2", "D": "Q2"}


def test_pit_tier_skips_symbols_without_prior_volume():
    from app.research.fit_membership import pit_tiers_by_month

    frame = _months(["A", "B", "C", "D", "NEW"], {
        "2026-08": {"A": 10.0, "B": 9.0, "C": 8.0, "D": 1.0},
        "2026-09": {"A": 10.0, "B": 9.0, "C": 8.0, "D": 1.0, "NEW": 50.0},
    })
    out = pit_tiers_by_month(frame, groups=2)
    assert "NEW" not in out["2026-09"]["tiers"], "新上市币在事件月之前无量，不能定档"
    assert out["2026-09"]["unavailable"] == ["NEW"]


def test_pit_tier_reports_partial_prior_month_as_unusable():
    from app.research.fit_membership import pit_tiers_by_month

    rows = []
    for ym, bars in [("2026-08", 100), ("2026-09", 720)]:  # 8 月是残月
        for sym, vol in [("A", 10.0), ("B", 9.0), ("C", 8.0), ("D", 1.0)]:
            rows.append({"symbol": sym, "ym": ym, "quote_volume": vol, "bars": bars})
    frame = fd.mark_partial_months(pd.DataFrame(rows), "1h")
    out = pit_tiers_by_month(frame, groups=2)
    assert out["2026-09"]["status"] == "NO_PRIOR_COMPLETE_MONTH"
    assert out["2026-09"]["prior_months"] == []
    assert out["2026-09"]["tiers"] == {}


def test_membership_drift_compares_exact_sets_not_counts():
    from app.research.fit_membership import membership_drift

    same = membership_drift({"liq_q1": ["A", "B"]}, {"liq_q1": ["B", "A"]})
    assert same["liq_q1"]["status"] == "MATCH"
    # 数量相同但成员换了一个：必须判 DRIFT，不能只比 len
    swapped = membership_drift({"liq_q1": ["A", "B"]}, {"liq_q1": ["A", "C"]})
    assert swapped["liq_q1"]["status"] == "DRIFT"
    assert swapped["liq_q1"]["added"] == ["C"] and swapped["liq_q1"]["removed"] == ["B"]
    missing = membership_drift({"major": ["A"]}, {})
    assert missing["major"]["status"] == "MISSING_NOW"


class _StubDB:
    """只记录 SQL 参数的假会话：验证窗口末端口径，不连真库。"""

    def __init__(self, rows):
        self.rows = rows
        self.params = []
        self.statements = []
        self.max_open_time_calls = 0

    def execute(self, sql, params=None):
        text_sql = str(sql)
        if "max(open_time)" in text_sql:
            self.max_open_time_calls += 1
            return _Scalar(1_800_000_000_000)
        self.params.append(dict(params or {}))
        self.statements.append(text_sql)
        return _Mappings(self.rows)


class _Scalar:
    def __init__(self, value):
        self.value = value

    def scalar(self):
        return self.value


class _Mappings:
    def __init__(self, rows):
        self.rows = rows

    def mappings(self):
        return self

    def all(self):
        return self.rows


def test_monthly_quote_volume_honours_explicit_window_end():
    """审计复现冻结窗口时必须用显式末端，不能随线上表增长而漂移。"""
    rows = [{"symbol": "AUSDT", "ym": "2026-09", "quote_volume": 1e9, "bars": 720}]
    db = _StubDB(rows)
    fd.monthly_quote_volume(db, "1h", ["AUSDT"], days=90, end_ms=1_794_495_600_000)
    assert db.max_open_time_calls == 0, "给了 end_ms 就不该再查当前末端"
    assert db.params[0]["lo"] == 1_794_495_600_000 - 90 * fd.DAY_MS
    assert db.params[0]["hi"] == 1_794_495_600_000
    assert "open_time<=:hi" in db.statements[0]


def test_monthly_quote_volume_defaults_to_live_window_end():
    rows = [{"symbol": "AUSDT", "ym": "2026-09", "quote_volume": 1e9, "bars": 720}]
    db = _StubDB(rows)
    fd.monthly_quote_volume(db, "1h", ["AUSDT"], days=90)
    assert db.max_open_time_calls == 1
    assert db.params[0]["lo"] == 1_800_000_000_000 - 90 * fd.DAY_MS
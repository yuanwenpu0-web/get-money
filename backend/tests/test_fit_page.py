"""矩阵页的取数测试：页面只许读 stats 产物，不许自己算判定，也不许把周期排成字典序。

两个坑都在此钉住：
- 列次序：`sorted()` 给的是 1h / 30m 这种字典序，而报告用的是「从快到慢」。同一份数据在
  两处给出两套次序，读者就没法把页面的列对上报告的行。
- 指标族计数：「6 列说得出口」不等于 6 份证据，页面顶部必须先给分母（多少指标进过统计）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend" / "scripts"))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tests"))
import fit_matrix_page as fmp  # noqa: E402
import test_fit_report as tf  # noqa: E402  复用报告夹具：两个渲染层必须读同一条款


def two_interval_doc() -> dict:
    """故意把 1h 写在前面：产物里的键次序不代表显示次序。"""
    doc = tf.stats_doc([tf.row("adx|x", "all", "candidate"),
                        tf.row("noise|z", "all", "no_edge", mean_net_pct=-0.4,
                               ci_low_pct=-1.0, verdict="no_edge",
                               verdict_cell="no_edge")])
    one_h = {**doc["intervals"][tf.IV], "symbols": 119,
             "scope_sizes": {"all": 119}}
    thirty = {**one_h, "symbols": 99, "scope_sizes": {"all": 99},
              # 同一批行对象会被盖 interval 标签，两个周期不能共用
              "rows": [tf.row("adx|x", "all", "candidate"),
                       tf.row("noise|z", "all", "no_edge", mean_net_pct=-0.4,
                              ci_low_pct=-1.0, verdict="no_edge",
                              verdict_cell="no_edge")]}
    doc["intervals"] = {tf.IV: one_h, "30m": thirty}
    return doc


def build(tmp: Path, doc: dict) -> dict:
    stats_path = tmp / "stats.json"
    dirs_path = tmp / "dirs.json"
    stats_path.write_text(json.dumps(doc), encoding="utf-8")
    dirs_path.write_text(json.dumps(tf.DIRS), encoding="utf-8")
    return fmp.build_payload("T", stats_path, dirs_path)


def test_interval_columns_use_bar_order_not_lexicographic(tmp_path):
    p = build(tmp_path, two_interval_doc())
    assert p["intervals"] == ["30m", "1h"]           # 与报告的次序一致
    assert set(p["cells"]["adx|x"]) == {"30m", "1h"}  # 同一列在两个周期各有一格
    assert set(p["cells"]["adx|x"]["30m"]) == {"all"}


def test_indicator_family_counts_are_column_plus_verdict_based(tmp_path):
    p = build(tmp_path, two_interval_doc())
    assert p["columns"] == 2                          # adx|x、noise|z 都进过统计
    assert p["indicators_total"] == 2
    assert p["indicators_promotable"] == 1            # no_edge 的 noise 不算说得出口
    assert p["detail_keys"] == 1                      # 明细只给说得出口的列
    assert p["scope_sizes"]["30m"]["all"] == 99
    assert p["scope_sizes"]["1h"]["all"] == 119       # 池规模按周期分开，不是一句「99 币」


def test_page_discloses_exploratory_status_before_filters():
    assert "审计修正：仅供探索性复核" in fmp.HTML
    assert fmp.HTML.index("审计修正：仅供探索性复核") < fmp.HTML.index('id="summary"')


def test_page_data_source_is_the_file_actually_read(tmp_path):
    p = build(tmp_path, two_interval_doc())
    assert p["stats_source"] == "stats.json"          # 顶部「数据源」必须跟着 --stats 走


def test_verdict_counts_payload_is_flat_per_interval(tmp_path):
    """产物里 `verdict_counts` 是按池嵌套的；页面卡片要的是扁平分布。

    直接把嵌套结构喂给 `Object.values(...).reduce(...)` 会得到 0，条带图印成
    `[object Object]`——顶部卡片就是这么坏掉的，所以钉成用例。
    """
    p = build(tmp_path, two_interval_doc())
    assert p["verdict_counts"]["1h"] == {"candidate": 1, "no_edge": 1}
    assert all(isinstance(v, int) for v in p["verdict_counts"]["1h"].values())


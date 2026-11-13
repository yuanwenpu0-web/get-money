"""`fit_events` 的口径测试：事件定义、前瞻收益、聚合量往返。

这些是「测评结论能不能信」的地基，所以每条都对着注释里的口径写死，改口径必先改测试。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.research import fit_events as fe
from app.research.fit_events import (
    FIELDS,
    FI,
    CacheLayout,
    accumulate,
    bool_column_positions,
    forward_returns,
    load_cached,
    rising_edges,
    save_symbol,
)


def times(n, step_ms=3_600_000, start=1_700_000_000_000):
    return pd.DatetimeIndex(pd.to_datetime(np.arange(n) * step_ms + start, unit="ms", utc=True))


# --------------------------------------------------------------------- 事件定义


def test_rising_edges_only_takes_first_true_bar():
    mask = np.array([False, True, True, True, False, True, True])
    assert list(rising_edges(mask, min_gap=1)) == [1, 5]


def test_rising_edges_drops_events_inside_min_gap():
    """min_gap=horizon 保证前瞻窗口不重叠：靠得太近的第二个事件必须丢掉。"""
    mask = np.zeros(12, dtype=bool)
    mask[[2, 3, 8, 11]] = True  # 上升沿 = 2, 8, 11
    assert list(rising_edges(mask, min_gap=6)) == [2, 8]  # 11 距 8 只有 3 根，窗口会重叠


def test_rising_edges_empty_and_short():
    assert rising_edges(np.zeros(5, dtype=bool), 3).size == 0
    assert rising_edges(np.array([True]), 3).tolist() == [0]


# --------------------------------------------------------------------- 前瞻收益


def test_forward_return_enters_next_open_exits_h_close():
    open_ = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    close = np.array([100.5, 101.5, 102.5, 103.5, 104.5])
    r = forward_returns(open_, close, horizon=2)
    # t=0 决策 → t=1 开盘 101 进 → t=2 收盘 102.5 出
    assert r[0] == pytest.approx(102.5 / 101.0 - 1.0)
    # 尾部两根凑不齐窗口，必须是 nan 而不是 0（0 会被当成「无超额」参与池化）
    assert np.isnan(r[-1]) and np.isnan(r[-2])


def test_forward_return_all_nan_when_series_too_short():
    r = forward_returns(np.array([1.0, 2.0]), np.array([1.0, 2.0]), horizon=5)
    assert np.all(np.isnan(r))


def test_forward_return_rejects_bad_entry_price():
    open_ = np.array([100.0, 0.0, 102.0, 103.0])
    close = np.array([100.0, 101.0, 102.0, 103.0])
    r = forward_returns(open_, close, horizon=1)
    assert np.isnan(r[0]), "进场价 <=0 不能算出一笔「0 收益」"
    assert r[1] == pytest.approx(102.0 / 102.0 - 1.0)
    assert np.isnan(r[-1])


# --------------------------------------------------------------------- 布尔列识别


def test_bool_column_positions_accepts_object_bool_and_skips_numeric():
    frame = pd.DataFrame(
        {
            "a": np.array([True, False]),
            "b": np.array([1.0, 2.0]),
            "c": pd.Series([True, None], dtype=object),
            "d": pd.Series(["yes", "no"], dtype=object),
        }
    )
    got = bool_column_positions(frame)
    assert [(j, name) for j, name in got] == [(0, "a"), (2, "c")]


def test_bool_column_positions_survives_duplicate_names():
    frame = pd.DataFrame([[True, 1.0], [False, 2.0]], columns=["x", "x"])
    assert [name for _, name in bool_column_positions(frame)] == ["x"]


# --------------------------------------------------------------------- 聚合量


def test_accumulate_matches_bruteforce_sums():
    r = np.array([0.01, -0.02, 0.03, 0.04])
    bench = 0.005
    x = r - bench
    cs = np.array([0.002, np.nan, 0.001, 0.0])
    seg = np.array([0, 1, 3, 3])
    row = accumulate(np.arange(4), r, x, cs, seg, hit_bars=10)
    assert row[FI["n"]] == 4
    assert row[FI["hit_bars"]] == 10
    assert row[FI["sum_r"]] == pytest.approx(r.sum())
    assert row[FI["sum_x"]] == pytest.approx(x.sum())
    assert row[FI["sum_sq"]] == pytest.approx((x * x).sum())
    assert row[FI["n_above"]] == 3 and row[FI["n_below"]] == 1
    assert row[FI["sum_above"]] == pytest.approx(x[x > 0].sum())
    assert row[FI["sum_below"]] == pytest.approx(x[x < 0].sum())
    # cross_section 只在 cs 可用的事件上求和，分子分母集合必须一致
    ok = np.isfinite(cs)
    assert row[FI["n_cs"]] == int(ok.sum())
    assert row[FI["sum_cs"]] == pytest.approx(cs[ok].sum())
    assert row[FI["sum_r_cs"]] == pytest.approx(r[ok].sum())
    assert row[FI["seg_n3"]] == 2 and row[FI["seg_x3"]] == pytest.approx(x[2:].sum())


def test_accumulate_on_no_events_is_all_zero_not_nan():
    row = accumulate(np.asarray([], dtype=int), np.array([]), np.array([]), np.array([]),
                     np.asarray([], dtype=int), hit_bars=3)
    assert row[FI["n"]] == 0 and row[FI["sum_x"]] == 0
    assert np.isfinite(row).all(), "空事件必须是 0，不能留 NaN 污染池化"


# --------------------------------------------------------------------- 缓存往返


def _fake_res(symbol, nh=len(fe.HORIZONS)):
    rng = np.random.default_rng(abs(hash(symbol)) % 2**32)
    cols = {}
    for key in ("ind_a|up_cross", "ind_b|down_cross"):
        agg = np.zeros((nh, len(FIELDS)))
        for h in range(nh):
            agg[h, FI["n"]] = 20 + h
            agg[h, FI["sum_x"]] = float(rng.normal(0.001, 0.002)) * (20 + h)
            agg[h, FI["sum_r"]] = 0.002 * (20 + h)
            agg[h, FI["hit_bars"]] = 100
            agg[h, FI["n_cs"]] = 15
            agg[h, FI["sum_r_cs"]] = 0.03
            agg[h, FI["sum_cs"]] = 0.02
            agg[h, FI["seg_n0"]] = 5
            agg[h, FI["seg_x0"]] = 0.01
        cols[key] = {"direction": 1, "direction_source": "declared", "matched": ["cross"], "agg": agg}
    return {
        "symbol": symbol, "status": "ok", "bars": 1000, "first_ms": 1, "last_ms": 2,
        "bench": {"6": 0.001}, "elapsed": 1.0, "timings": {"ind_a": 0.2}, "cols": cols,
    }


def test_save_and_load_roundtrip_rebuilds_matrix(tmp_path):
    layout = CacheLayout(root=tmp_path, scope_sha="abc123", interval="1h")
    for sym in ("ETHUSDT", "BTCUSDT"):
        save_symbol(layout.symbol_file(sym), _fake_res(sym))
    keys, syms, matrix, covered, meta, sym_meta = load_cached(layout, ["BTCUSDT", "ETHUSDT", "NOUSDT"])
    assert syms == ["BTCUSDT", "ETHUSDT"], "缺缓存的币要跳过而不是占一列 0"
    assert keys == sorted(keys), "列顺序必须稳定，否则两次运行的下标不可比"
    assert matrix.shape == (2, 2, len(fe.HORIZONS), len(FIELDS))
    assert covered.all()
    assert meta["ind_a|up_cross"]["direction_source"] == "declared"
    assert sym_meta["BTCUSDT"]["bars"] == 1000


def test_load_cached_ignores_stale_horizon_layout(tmp_path):
    layout = CacheLayout(root=tmp_path, scope_sha="abc123", interval="1h")
    save_symbol(layout.symbol_file("BTCUSDT"), _fake_res("BTCUSDT"))
    # 手工把 meta 里的 horizons 改成别的口径，模拟换 horizon 后的旧缓存
    with np.load(layout.symbol_file("BTCUSDT"), allow_pickle=False) as data:
        meta = json.loads(str(data["__meta__"][0]))
        arrays = {k: data[k] for k in data.files if k != "__meta__"}
    meta["horizons"] = [1, 2, 3]
    np.savez_compressed(layout.symbol_file("BTCUSDT"),
                        __meta__=np.array([json.dumps(meta)]), **arrays)
    _, syms, matrix, _, _, _ = load_cached(layout, ["BTCUSDT"])
    assert syms == [] and matrix.shape[0] == 0


def test_cache_layout_paths_are_scoped_by_hash(tmp_path):
    a = CacheLayout(root=tmp_path, scope_sha="sha1", interval="5m")
    b = CacheLayout(root=tmp_path, scope_sha="sha2", interval="5m")
    assert a.symbol_file("BTCUSDT") != b.symbol_file("BTCUSDT")
    assert "sha1" in str(a.dir) and a.csbench.parent == a.dir


# --------------------------------------------------------------------- 横截面基准


def test_build_cs_bench_needs_min_aligned_symbols():
    idx = times(6)
    frames = {}
    for i, sym in enumerate(["A", "B", "C"]):  # 只有 3 个币 < MIN_CS_SYMBOLS
        frames[sym] = pd.DataFrame(
            {"open": np.arange(6) + 100.0, "close": np.arange(6) + 101.0}, index=idx
        )
    payload = fe.build_cs_bench(frames, horizons=[2])
    assert np.all(np.isnan(payload["h2"])), "可对齐币数不足时必须全是 nan，不能给假基准"

    for i in range(fe.MIN_CS_SYMBOLS):
        frames[f"S{i}"] = frames["A"]
    payload = fe.build_cs_bench(frames, horizons=[2])
    assert np.isfinite(payload["h2"]).sum() >= 1


def test_build_cs_bench_scatters_each_symbol_on_its_own_grid():
    """篮子均值按时刻对齐：某币缺某个时刻的行，不该把它的收益挪到别的时刻上。"""
    idx = times(8)
    frames = {}
    for i in range(fe.MIN_CS_SYMBOLS):
        frames[f"S{i}"] = pd.DataFrame(
            {"open": np.full(8, 100.0), "close": np.full(8, 100.0 + i)}, index=idx
        )
    payload = fe.build_cs_bench(frames, horizons=[2])
    # 每币收益恒为 i/100，等权篮子 = mean(0, .01, .02, .03, .04) = 0.02
    assert payload["h2"][0] == pytest.approx(0.02)
    assert payload["time_ms"][0] == idx[0].value // 1_000_000

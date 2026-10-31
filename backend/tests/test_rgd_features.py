"""Contract-first tests for Regime-Gated Donchian v2 (RGD).

These cases deliberately use independently calculated values and deterministic
OHLCV. They cover the pure feature/spec side of the Pine-to-local port; broker
timing is covered by ``test_rgd_execution.py``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.backtest.execution import simulate
from app.backtest.rgd_execution import simulate_rgd
from app.features import rgd
from app.features.indicators import _atr
from app.features.rgd import auto_tf, htf_interval_for, state_machine
from app.features.rgd_htf import HTF_COLUMNS, attach_htf_levels
from app.features.rgd_indicators import (
    rolling_pearson,
    sign as _sign,
    tension_series,
    tension_step,
    wilder_atr,
)
from app.models import StrategyConfig
from app.strategy.defaults import seed_strategies
from app.strategy.rules import prepare_frame, resolve_params
from app.strategy.specs import rgd_donchian_v2 as spec
from app.strategy.specs.supertrend_current import _rma


CODE = spec.CODE
ROOT = Path(__file__).resolve().parents[2]


def _params(**overrides) -> dict:
    return resolve_params(
        CODE,
        {"fee_rate": 0.0004, "slippage": 0.0002},
        overrides or None,
        interval="15m",
    )


def _base_frame(periods: int = 320) -> pd.DataFrame:
    """Deterministic 15-minute OHLCV with non-flat ATR/ER/correlation inputs."""
    index = pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC")
    move = np.linspace(0.0, 24.0, periods) + np.sin(np.arange(periods) / 5) * 0.8
    close = 100.0 + move
    open_ = np.r_[close[0] - 0.2, close[:-1]]
    high = np.maximum(open_, close) + 0.8 + (np.arange(periods) % 3) * 0.03
    low = np.minimum(open_, close) - 0.7 - (np.arange(periods) % 5) * 0.02
    volume = 1000.0 + (np.arange(periods) % 19) * 41.0
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _htf_frame(base: pd.DataFrame) -> pd.DataFrame:
    """Native 4-hour bars covering the base frame plus prior completed history."""
    start = base.index[0].floor("4h") - pd.Timedelta(hours=8)
    index = pd.date_range(
        start, base.index[-1].ceil("4h") + pd.Timedelta(hours=4), freq="4h", tz="UTC"
    )
    sequence = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {"open": 90 + sequence, "high": 105 + sequence * 2, "low": 95 + sequence}, index=index
    )


def _attached(periods: int = 320) -> pd.DataFrame:
    base = _base_frame(periods)
    as_of = int((base.index[-1] + pd.Timedelta(minutes=15)).timestamp() * 1000)
    return attach_htf_levels(base, _htf_frame(base), "4h", as_of)


def _prepared(periods: int = 320) -> pd.DataFrame:
    return prepare_frame(_attached(periods), CODE, _params())


def _assert_same(left, right) -> None:
    if pd.isna(left):
        assert pd.isna(right)
    else:
        assert float(right) == pytest.approx(float(left), rel=1e-10, abs=1e-10)


def test_a1_auto_tf_complete_mapping_and_unsupported_weekly_route():
    assert {key: auto_tf(key) for key in ("1m", "5m", "15m", "1h", "4h", "1d")} == {
        "1m": "60", "5m": "60", "15m": "240", "1h": "1D", "4h": "1D", "1d": "1W",
    }
    assert htf_interval_for("15m", True, 1) == "4h"
    with pytest.raises(ValueError, match=r"autoTF\(\)=1W"):
        htf_interval_for("1d", True, 1)


def test_a2_tension_state_machine_build_release_clamp_and_nan_recovery():
    assert _sign(-2.0) == -1
    assert _sign(0.0) == 0
    assert _sign(2.0) == 1

    assert tension_step(0.0, 7.0, 6, 20, 8.25) == pytest.approx(2.0)
    assert tension_step(2.0, 1.0, 6, 20, 8.25) == pytest.approx(2.0 - 2.0 / 21.0)
    assert tension_step(2.0, -7.0, 6, 20, 8.25) == pytest.approx(2.0 - 9.0 * 2.0 / 21.0)
    assert tension_step(-2.0, 0.0, 6, 20, 8.25) == pytest.approx(-2.0 + 2.0 * 2.0 / 21.0)
    assert tension_step(8.2, 100.0, 6, 20, 8.25) == 8.25
    assert tension_step(-8.2, -100.0, 6, 20, 8.25) == -8.25

    values = tension_series(np.array([4.0, np.nan, 4.0]), 6, 20, 8.25)
    assert values[0] == pytest.approx(8.0 / 7.0)
    assert np.isnan(values[1])
    assert values[2] == pytest.approx(8.0 / 7.0)


def test_a3_percentrank_uses_only_the_preceding_window():
    previous = np.array([98.0, 100.0, 102.0, 104.0, 108.0])
    got = [
        rgd.percentrank_preceding(np.r_[previous, current], 5)[5]
        for current in (105.0, 97.0, 102.0, 108.0)
    ]
    assert got == pytest.approx([80.0, 0.0, 60.0, 100.0])

    all_equal = rgd.percentrank_preceding(np.array([7.0] * 6), 5)
    assert np.isnan(all_equal[:5]).all()
    assert all_equal[5] == 100.0
    lower_current = rgd.percentrank_preceding(np.array([100., 101., 102., 103., 104., 99.]), 5)
    assert lower_current[5] == 0.0
    nan_window = rgd.percentrank_preceding(np.array([1., 2., np.nan, 4., 5., 6.]), 5)
    assert np.isnan(nan_window[5])


def test_a4_wilder_atr_matches_project_rma_and_not_ewm_seed():
    index = pd.date_range("2026-01-01", periods=7, freq="15min", tz="UTC")
    high = pd.Series([11., 13., 14., 17., 16., 19., 21.], index=index)
    low = pd.Series([9., 10., 11., 12., 13., 15., 17.], index=index)
    close = pd.Series([10., 12., 12., 16., 14., 18., 20.], index=index)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    actual = wilder_atr(high, low, close, 3)
    expected = _rma(tr, 3)
    pd.testing.assert_series_equal(actual, expected)

    ewm_atr = _atr(pd.DataFrame({"high": high, "low": low, "close": close}), 3)
    assert np.isnan(actual.iloc[0]) and np.isnan(actual.iloc[1])
    assert actual.iloc[2] == pytest.approx(float(tr.iloc[:3].mean()))
    assert not np.isclose(actual.iloc[2], float(ewm_atr.iloc[2]))


def test_a5_efficiency_noise_shift_and_threshold_are_hand_calculated():
    index = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    close = pd.Series([10., 11., 13., 12., 14.], index=index)
    values = rgd._efficiency(close, 3)
    assert values["rgd_noise"][3] == pytest.approx(4.0)
    assert values["rgd_er"][3] == pytest.approx(0.25)
    assert values["rgd_noise"][4] == pytest.approx(5.0)
    assert values["rgd_er"][4] == pytest.approx(0.425)
    assert values["rgd_er"][3] < 0.30 <= values["rgd_er"][4]


def test_a6_acceptance_weighted_defense_uses_full_mixed_formula():
    index = pd.date_range("2026-01-01", periods=5, freq="15min", tz="UTC")
    high = pd.Series([100., 90., 80., 70., 60.], index=index)
    low = pd.Series([10., 20., 30., 40., 50.], index=index)
    close = pd.Series([0., 0., 0., 0., 100.], index=index)
    values = rgd._defense(
        high, low, close, np.zeros(5), np.full(5, 100.0),
        {"def_fast": 2, "def_slow": 3, "acc_len": 1, "max_depth": 0.70},
    )
    assert values["rgd_pos"].tolist() == pytest.approx([0., 0., 0., 0., 1.])
    assert values["rgd_acceptance"].tolist() == pytest.approx([0., 0., 0., 0., 1.])
    assert values["rgd_bull_q"][3] == 0.0 and values["rgd_bear_q"][3] == 1.0
    assert values["rgd_bull_q"][4] == 1.0 and values["rgd_bear_q"][4] == 0.0
    assert values["rgd_def_long"][3] == pytest.approx(20.0)
    assert values["rgd_def_short"][3] == pytest.approx(0.3 * 90.0 + 0.7 * 100.0)
    assert values["rgd_def_long"][4] == pytest.approx(0.3 * 30.0 + 0.7 * 20.0)
    assert values["rgd_def_long"][4] != 20.0
    assert values["rgd_def_short"][4] == pytest.approx(80.0)


def test_a7_state_machine_priority_gates_and_prepare_causality():
    states, previous, longs, shorts = state_machine(
        np.array([True, True, False, False, False]),
        np.array([True, False, True, False, False]),
        np.array([100., 100., 100., 90., 110.]),
        np.array([0., 95., 0., 95., 0.]),
        np.array([200., 200., 200., 95., 105.]),
    )
    assert previous.tolist() == [0, 1, 1, -1, -1]
    # The fabricated raw predicates deliberately collide on bar zero: long has priority.
    assert longs.tolist() == [True, False, False, False, False]
    assert shorts.tolist() == [True, False, True, False, False]
    assert states.tolist() == [1, 1, -1, -1, 0]

    enriched = _prepared()
    assert not (enriched["rgd_long_break"] & enriched["rgd_short_break"]).any()
    assert not (enriched["rgd_long_pull_core"] & enriched["rgd_short_pull_core"]).any()


def test_a8_ready_is_exact_six_finite_inputs_and_blocks_all_entries():
    enriched = _prepared()
    needed = ("rgd_ent_up", "rgd_ent_lo", "rgd_def_long", "rgd_def_short", "rgd_er", "rgd_atr")
    expected = np.isfinite(enriched.loc[:, needed].to_numpy(dtype=float)).all(axis=1)
    assert enriched["rgd_ready"].to_numpy(dtype=bool).tolist() == expected.tolist()
    not_ready = ~enriched["rgd_ready"].to_numpy(dtype=bool)
    assert not enriched.loc[not_ready, "rgd_long_break"].any()
    assert not enriched.loc[not_ready, "rgd_short_break"].any()
    assert not enriched.loc[not_ready, "rgd_long_pull_core"].any()
    assert not enriched.loc[not_ready, "rgd_short_pull_core"].any()


def test_a9_htf_join_is_causal_constant_per_bucket_and_drops_open_bucket():
    base = _base_frame(48)
    htf = _htf_frame(base)
    as_of = int((base.index[-1] + pd.Timedelta(minutes=15)).timestamp() * 1000)
    out = attach_htf_levels(base, htf, "4h", as_of)
    # No HTF history means the first base bar has pos<0 and cannot be marked valid.
    no_history = attach_htf_levels(base, htf.iloc[2:], "4h", as_of)
    assert not bool(no_history["rgd_htf_ok"].iloc[0])
    for _, bucket in out.groupby(out.index.floor("4h")):
        valid = bucket.loc[bucket["rgd_htf_ok"]]
        if not valid.empty:
            assert valid["rgd_htf_hh"].nunique() == 1
            assert valid["rgd_htf_ll"].nunique() == 1

    htf_with_unfinished_outlier = htf.copy()
    htf_with_unfinished_outlier.loc[htf_with_unfinished_outlier.index[-1], ["high", "low"]] = [9999., 1.]
    before_close = int((htf_with_unfinished_outlier.index[-1] + pd.Timedelta(hours=3)).timestamp() * 1000)
    out_without_last = attach_htf_levels(base, htf_with_unfinished_outlier, "4h", before_close)
    assert float(out_without_last["rgd_htf_hh"].max()) < 9999.0

    cut = 33
    truncated = attach_htf_levels(base.iloc[:cut], htf, "4h", as_of)
    pd.testing.assert_frame_equal(out.iloc[:cut][list(HTF_COLUMNS)], truncated[list(HTF_COLUMNS)])


def test_a10_htf_np_uses_base_close_and_invalid_htf_forces_neutral_target():
    attached = _attached()
    changed = attached.copy()
    changed.loc[changed.index[-1], "close"] += 10.0
    original = rgd.build_features(attached, _params())
    altered = rgd.build_features(changed, _params())
    last = attached.index[-1]
    assert original.at[last, "rgd_htf_hh"] == altered.at[last, "rgd_htf_hh"]
    assert original.at[last, "rgd_htf_ll"] == altered.at[last, "rgd_htf_ll"]
    assert original.at[last, "rgd_np"] != altered.at[last, "rgd_np"]

    degraded = rgd.build_features(_base_frame(), _params())
    assert degraded["rgd_htf_ok"].eq(False).all()
    assert degraded["rgd_np"].isna().all()
    assert degraded["rgd_tg"].eq(0.0).all()


def test_a11_rolling_correlation_is_affine_invariant_in_bar_index():
    index = pd.date_range("2026-01-01", periods=12, freq="15min", tz="UTC")
    close = pd.Series([100, 101, 99, 103, 102, 106, 105, 107, 111, 110, 114, 116], index=index, dtype=float)
    local = rolling_pearson(close, pd.Series(np.arange(len(close)), index=index), 5)
    absolute = rolling_pearson(close, pd.Series(np.arange(len(close)) + 12345, index=index), 5)
    np.testing.assert_allclose(local.to_numpy(), absolute.to_numpy(), equal_nan=True, rtol=0, atol=1e-12)


def test_a20_prepare_is_causal_for_every_rgd_column():
    attached = _attached(320)
    full = prepare_frame(attached, CODE, _params())
    rgd_columns = [column for column in full if column.startswith("rgd_")]
    assert len(rgd_columns) >= 40
    for cut in (120, 209, 320):
        partial = prepare_frame(attached.iloc[:cut], CODE, _params())
        for column in rgd_columns:
            _assert_same(full[column].iloc[cut - 1], partial[column].iloc[-1])


def test_a21_params_reject_cost_overrides_wrong_types_and_invalid_cross_constraints():
    defaults = _params()
    contract_keys = (
        "dir_mode", "use_pull", "cooldown", "brk_fast", "brk_slow", "atr_len", "rank_len",
        "rank_cut", "er_len", "er_min", "brk_buf", "use_ten", "htf_in", "auto_htf",
        "build_len", "rel_len", "max_ten", "ten_th", "use_corr", "corr_len", "corr_min",
        "use_vol", "vol_min", "def_fast", "def_slow", "acc_len", "max_depth", "risk_pct",
        "max_stop_atr", "min_stop_atr", "tp1_r", "tp1_pct",
    )
    assert {key: defaults[key] for key in contract_keys} == {key: spec.DEFAULTS[key] for key in contract_keys}
    for override in (
        {"fee_rate": 0.0005}, {"slippage": 0.0001}, {"brk_fast": 14.5}, {"use_ten": 1},
        {"min_stop_atr": 3.0, "max_stop_atr": 2.5}, {"brk_fast": 34, "brk_slow": 34},
        {"def_fast": 21, "def_slow": 21},
    ):
        with pytest.raises(ValueError):
            _params(**override)


def test_a23_neutral_signals_block_the_generic_broker():
    prepared = _prepared()
    signals = spec.signals(prepared, _params())
    assert signals["direction"].eq(0).all()
    assert signals["block_reason"].eq("stop_order_requires_dedicated_broker").all()
    equity, trades = simulate(prepared, signals, _params(), 10_000.0, 0.0004, 0.0002, "15m")
    assert trades == []
    assert equity.iloc[-1] == pytest.approx(10_000.0)


def test_a25_registry_import_and_seed_keep_rgd_disabled(db):
    program = "from app.strategy import specs; from app.strategy.registry import has_spec; assert has_spec('rgd_donchian_v2')"
    completed = subprocess.run(
        [sys.executable, "-c", program], cwd=ROOT / "backend", capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert spec.SPEC.default_enabled is False
    assert "dedicated_broker" in spec.SPEC.tags
    seed_strategies(db)
    row = db.query(StrategyConfig).filter(StrategyConfig.code == CODE).one()
    assert row.enabled == 0


def test_a27_every_new_rgd_source_file_is_present_and_under_300_lines():
    paths = (
        "backend/app/features/rgd.py",
        "backend/app/features/rgd_indicators.py",
        "backend/app/features/rgd_htf.py",
        "backend/app/strategy/specs/rgd_donchian_v2.py",
        "backend/app/backtest/rgd_execution.py",
        "backend/app/backtest/rgd_orders.py",
        "backend/scripts/rgd_backtest.py",
        "backend/scripts/rgd_backtest_worker.py",
        "backend/scripts/rgd_backtest_report.py",
        "backend/scripts/rgd_baselines.py",
    )
    for relative in paths:
        path = ROOT / relative
        assert path.is_file(), f"missing required RGD source: {relative}"
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 300, f"{relative} exceeds 300 lines"


def test_a28_missing_htf_is_observable_and_attached_frame_keeps_real_breakout_path():
    bare = _base_frame()
    degraded = prepare_frame(bare, CODE, _params(), service=None)
    assert degraded.attrs.get("rgd_htf_degraded") is True
    assert degraded["rgd_htf_ok"].eq(False).all()
    assert not degraded["rgd_long_break"].any() and not degraded["rgd_short_break"].any()

    attached = _attached()
    assert set(HTF_COLUMNS).issubset(attached.columns)
    prepared = prepare_frame(attached, CODE, _params())
    assert prepared.attrs.get("rgd_htf_degraded") is not True
    repeat = prepare_frame(prepared, CODE, _params())
    pd.testing.assert_frame_equal(prepared, repeat)

    # The accepted pipeline is not merely feature-complete: attached data can reach the
    # dedicated broker and create a transaction. This injection happens after prepare,
    # so it tests order wiring rather than reimplementing the feature predicates.
    broker_signals = spec.signals(prepared, _params()).copy()
    target = broker_signals.index[-2]
    broker_signals.loc[target, "rgd_long_break"] = True
    broker_signals.loc[target, "rgd_allow_long"] = True
    broker_signals.loc[target, "rgd_stp_long"] = 1.0
    broker_signals.loc[target, "rgd_rd_long"] = 1.0
    broker_signals.loc[target, "rgd_tp1_long"] = 1_000.0
    equity, trades = simulate_rgd(
        prepared, broker_signals, _params(), 10_000.0, 0.0, 0.0, "15m"
    )
    assert trades == []
    assert equity.attrs["open_position"] is not None
    assert equity.attrs["open_position"]["side"] == "long"

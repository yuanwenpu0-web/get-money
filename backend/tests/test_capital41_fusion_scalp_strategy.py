import numpy as np
import pandas as pd
import pytest

from app.catalog_indicators.capital41_fusion_scalp import capital41_fusion_scalp
from app.models import StrategyConfig
from app.strategy.defaults import seed_strategies
from app.strategy.registry import get_spec
from app.strategy.rules import evaluate_signals, known_codes, prepare_frame, resolve_params

CODE = "capital41_fusion_scalp"


def params(**overrides):
    return resolve_params(CODE, {}, {"cost_filter": False, "max_atr_pct": 0.2, **overrides})


def frame():
    rng = np.random.default_rng(37)
    close = 100 + np.cumsum(rng.normal(0, 0.8, 1000))
    return pd.DataFrame({"open": close, "high": close + rng.uniform(0.1, 2, len(close)),
                         "low": close - rng.uniform(0.1, 2, len(close)), "close": close,
                         "volume": rng.uniform(10, 100, len(close))},
                        index=pd.date_range("2026-01-01", periods=len(close), freq="30min", tz="UTC"))


def test_registration_disabled_and_parameters(db):
    assert CODE in known_codes()
    seed_strategies(db)
    assert db.query(StrategyConfig).filter_by(code=CODE).one().enabled == 0
    assert params()["require_turn"] is True
    with pytest.raises(ValueError):
        params(rsi_length=2.5)
    with pytest.raises(ValueError):
        params(volume_multiplier=4)
    with pytest.raises(ValueError):
        params(require_turn=1)


@pytest.mark.parametrize("require_turn", [True, False])
def test_catalog_signals_preserved_and_causal(require_turn):
    raw = frame()
    p = params(require_turn=require_turn, ema_length=50, rsi_length=3, wr_length=3,
               rsi_overbought=50, rsi_oversold=50, wr_overbought=-50, wr_oversold=-50,
               atr_length=3, atr_average_length=5, atr_expansion_multiplier=1.0,
               volume_length=5, volume_multiplier=0.5)
    prepared = prepare_frame(raw, CODE, p)
    expected = capital41_fusion_scalp(raw, **{k: p[k] for k in get_spec(CODE).prepare_keys})
    pd.testing.assert_series_equal(prepared.fusion_scalp_long, expected.long_signal, check_names=False)
    pd.testing.assert_series_equal(prepared.fusion_scalp_short, expected.short_signal, check_names=False)
    assert expected.long_signal.any() and expected.short_signal.any()
    decisions = evaluate_signals(prepared, CODE, p)
    np.testing.assert_array_equal(decisions.signed, expected.long_signal.astype(int) - expected.short_signal.astype(int))
    prefix = prepare_frame(raw.iloc[:700], CODE, p)
    pd.testing.assert_series_equal(prepared.fusion_scalp_long.iloc[:700], prefix.fusion_scalp_long)
    pd.testing.assert_series_equal(prepared.fusion_scalp_short.iloc[:700], prefix.fusion_scalp_short)
    for idx in decisions.index[decisions.direction.ne(0)][:4]:
        assert evaluate_signals(prepared.loc[[idx]], CODE, p).iloc[0].direction == decisions.loc[idx, "direction"]


def test_generic_atr_risk_and_invalid_atr():
    data = pd.DataFrame({"close": [100., 100., 100., 100.],
                         "fusion_scalp_atr": [2., 2., np.nan, -1.],
                         "fusion_scalp_long": [True, False, True, True],
                         "fusion_scalp_short": [False, True, False, False],
                         "atr_pct": [0.02, 0.02, np.nan, -0.01]})
    p = params(sl_atr_mult=2, tp_atr_mult=4, min_sl_pct=0.001, min_tp_pct=0.002)
    result = evaluate_signals(data, CODE, p)
    assert result.direction.tolist() == [1, -1, 0, 0]
    assert result.sl_pct.iloc[:2].tolist() == pytest.approx([0.04, 0.04])
    assert result.tp_pct.iloc[:2].tolist() == pytest.approx([0.08, 0.08])
    assert result.block_reason.iloc[2:].tolist() == ["invalid_indicators", "invalid_indicators"]

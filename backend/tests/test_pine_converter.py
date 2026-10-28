import pandas as pd
import pytest

from app.pine import PineUnsupportedError, compare_csv, compile_pine


SOURCE = '''//@version=6
strategy("Cross")
fast = ta.ema(close, 2)
slow = ta.sma(close, 3)
rsi = ta.rsi(close, 2)
atr = ta.atr(2)
if ta.crossover(fast, slow)
    strategy.entry("L", strategy.long)
if ta.crossunder(fast, slow)
    strategy.close("L")
'''


def bars():
    close = [1., 2., 3., 2., 1., 3.]
    return pd.DataFrame({"open_time": [1700000000000 + 60000*i for i in range(6)],
                         "open": close, "high": [x + 1 for x in close], "low": [x - 1 for x in close],
                         "close": close, "volume": [1.] * 6})


def test_closed_bar_indicators_and_signals():
    result = compile_pine(SOURCE).evaluate(bars())
    assert result.fast.tolist() == pytest.approx([1, 5/3, 23/9, 59/27, 113/81, 599/243])
    assert pd.isna(result.rsi.iloc[1])
    assert result.rsi.iloc[2] == pytest.approx(100)
    assert result.atr.iloc[1] == pytest.approx(2)
    assert result.long_close.tolist() == [False, False, False, True, False, False]
    assert result.long_entry.tolist() == [False, False, False, False, False, True]
    assert not result.short_entry.any()


@pytest.mark.parametrize("extra", [
    'x = request.security(syminfo.tickerid, "60", close)',
    'x = close[-1]',
    'x = ta.ema(close, 2, 3)',
    'strategy.exit("stop", "L", stop=close)',
    'if true\n        strategy.entry("x", strategy.long)',
    'var x = 0',
])
def test_unsupported_fails_closed(extra):
    with pytest.raises(PineUnsupportedError):
        compile_pine(SOURCE + extra + "\n")


def test_comparison_detects_signal_data_and_missing_bars():
    actual = compile_pine(SOURCE).evaluate(bars())
    reference = actual.copy()
    assert compare_csv(reference, actual, ["fast", "atr"])["passed"]
    reference.loc[3, "long_close"] = False
    reference.loc[4, "close"] += 2
    reference = reference.iloc[:-1]
    report = compare_csv(reference, actual, ["fast", "atr"])
    assert not report["passed"]
    assert report["mismatch_count"] == 3
    assert {item["field"] for item in report["mismatches"]} == {"long_close", "close", "bar"}


def test_comparison_rejects_duplicate_times_and_missing_columns():
    actual = compile_pine(SOURCE).evaluate(bars())
    bad = actual.copy()
    bad.loc[1, "open_time"] = bad.loc[0, "open_time"]
    with pytest.raises(ValueError):
        compare_csv(bad, actual)
    with pytest.raises(ValueError):
        compare_csv(actual.drop(columns=["long_entry"]), actual)


def test_generated_python_uses_local_runtime():
    program = compile_pine(SOURCE)
    namespace = {}
    exec(program.to_python(), namespace)
    pd.testing.assert_frame_equal(namespace["evaluate"](bars()), program.evaluate(bars()))


def test_order_ids_and_repeated_actions_are_not_silently_merged():
    with pytest.raises(PineUnsupportedError):
        compile_pine(SOURCE + 'if close > 0\n    strategy.entry("Other", strategy.long)\n')
    with pytest.raises(PineUnsupportedError):
        compile_pine(SOURCE + 'if close > 0\n    strategy.entry("L", strategy.long)\n')


def test_conflicting_commands_on_one_bar_are_rejected():
    source = '''//@version=6
strategy("conflict")
if close > 0
    strategy.entry("L", strategy.long)
if close > 0
    strategy.entry("S", strategy.short)
'''
    with pytest.raises(PineUnsupportedError, match="同一根"):
        compile_pine(source).evaluate(bars())


def test_price_comparison_is_exact_even_when_indicators_have_tolerance():
    actual = compile_pine(SOURCE).evaluate(bars())
    reference = actual.copy()
    reference.loc[0, "close"] += 1e-10
    report = compare_csv(reference, actual, atol=1)
    assert not report["passed"]


@pytest.mark.parametrize("expression", ["ta", "input"])
def test_module_names_are_not_standalone_values(expression):
    with pytest.raises(PineUnsupportedError):
        compile_pine(SOURCE + f"x = {expression}\n")


@pytest.mark.parametrize("expression", ["close / 0", "1 / 0", "ta.sma(close / (close - 2), 2)"])
def test_invalid_arithmetic_is_reported_as_unsupported(expression):
    with pytest.raises(PineUnsupportedError):
        compile_pine(SOURCE + f"x = {expression}\n").evaluate(bars())


def test_indicator_rejects_intermediate_missing_input():
    source = SOURCE + "x = ta.sma(close % (close - 2), 2)\n"
    with pytest.raises(PineUnsupportedError, match="缺失"):
        compile_pine(source).evaluate(bars())

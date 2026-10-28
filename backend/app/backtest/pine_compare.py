"""Nine-decimal rounding for Pine float comparisons, without rounding series.

TradingView documents nine fractional digits for comparisons but does not
specify tie-breaking. Python's binary-float round uses ties-to-even; exact ties
and decimal-to-binary conversion therefore still require TradingView parity tests.
"""


def _pair(a, b):
    if a is None or b is None:
        return None
    return round(a, 9), round(b, 9)


def gt(a, b):
    values = _pair(a, b)
    return values is not None and values[0] > values[1]


def ge(a, b):
    values = _pair(a, b)
    return values is not None and values[0] >= values[1]


def lt(a, b):
    values = _pair(a, b)
    return values is not None and values[0] < values[1]


def le(a, b):
    values = _pair(a, b)
    return values is not None and values[0] <= values[1]

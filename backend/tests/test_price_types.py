"""NumPy prices must reach PostgreSQL as native numeric values."""

import numpy as np
import pytest

from app.utils import round_price


@pytest.mark.parametrize("scalar", [float, np.float32, np.float64])
@pytest.mark.parametrize("value,digits", [(0.716601, 6), (17.123456, 4), (1234.56789, 2)])
def test_rounded_prices_are_native_floats(scalar, value, digits):
    number = scalar(value)
    result = round_price(number)
    assert type(result) is float
    assert result == round(float(number), digits)


def test_postgres_adapts_numpy_price_as_a_numeric_literal():
    from psycopg2.extensions import adapt

    result = round_price(np.float64(0.716601))
    assert adapt(result).getquoted() == b"0.716601"

"""迭代9：持续训练/晋升门禁相关设置的校验与默认值单测（与管线测试分文件，遵守单文件上限）。

离线：只做参数边界校验与 DEFAULT_STATE 断言，不触网、不训练。
"""

import pytest
from fastapi import HTTPException

from app.ml import auto_trainer


def _validate(payload):
    from app.api.settings import _validate
    _validate(payload)


@pytest.mark.parametrize("key,value", [
    ("auto_train_interval_hours", 1), ("auto_train_interval_hours", 720),
    ("auto_train_top_n", 1), ("auto_train_top_n", 50),
    ("ml_promote_rounds", 1), ("ml_promote_rounds", 10),
    ("ml_min_auc", 0.5), ("ml_min_auc", 0.9),
])
def test_numeric_settings_accept_boundaries(key, value):
    _validate({key: value})  # 不应抛


@pytest.mark.parametrize("key,value", [
    ("auto_train_interval_hours", 0), ("auto_train_interval_hours", 721),
    ("auto_train_top_n", 0), ("auto_train_top_n", 51),
    ("ml_promote_rounds", 0), ("ml_promote_rounds", 11),
    ("ml_min_auc", 0.49), ("ml_min_auc", 0.91),
])
def test_numeric_settings_reject_out_of_range(key, value):
    with pytest.raises(HTTPException) as exc:
        _validate({key: value})
    assert exc.value.status_code == 400


@pytest.mark.parametrize("key", ["auto_train_interval_hours", "auto_train_top_n",
                                 "ml_promote_rounds"])
def test_integer_settings_reject_non_integer(key):
    with pytest.raises(HTTPException):
        _validate({key: 1.5})
    with pytest.raises(HTTPException):
        _validate({key: True})


@pytest.mark.parametrize("key", ["auto_train_enabled", "ml_auto_enable"])
def test_bool_settings_reject_non_bool(key):
    _validate({key: True})
    _validate({key: False})
    with pytest.raises(HTTPException):
        _validate({key: "yes"})


def test_new_keys_are_editable():
    from app.api.settings import EDITABLE_KEYS
    for key in ("auto_train_enabled", "auto_train_interval_hours", "auto_train_top_n",
                "ml_min_auc", "ml_promote_rounds", "ml_auto_enable"):
        assert key in EDITABLE_KEYS


def test_defaults_training_on_but_auto_enable_off():
    from app.config import DEFAULT_STATE
    assert DEFAULT_STATE["auto_train_enabled"] is True     # 训练无害 → 默认开
    assert DEFAULT_STATE["ml_auto_enable"] is False         # 启用要过门禁 → 默认关


def test_default_train_window_yields_enough_test_rows():
    """回归：auto_train_days 过小会让测试段低于 MIN_TEST_ROWS，门禁结构性不可能通过。"""
    from app.config import DEFAULT_STATE
    # 实测标定：1h 上每天约 24 行，特征预热约耗 57 行，测试段约占总样本 20%。
    usable = DEFAULT_STATE["auto_train_days"] * 24 - 57
    assert int(usable * 0.2) >= auto_trainer.MIN_TEST_ROWS

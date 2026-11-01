import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from app import models  # noqa: F401
from app.db import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        yield session
    engine.dispose()


@pytest.fixture(autouse=True)
def _clear_universe_cache():
    """模块级宇宙缓存跨测试隔离：每个用例前清空，避免前一个用例的缓存键污染后续用例。"""
    from universe_helpers import clear_universe_cache

    clear_universe_cache()
    yield
    clear_universe_cache()


@pytest.fixture(autouse=True)
def _catalog_runtime_warnings_as_errors(request):
    """Pine 移植模块的测试里，RuntimeWarning 一律升级为错误。

    这一类告警基本都是数值缺陷的信号：

    * ``np.nanmax/nanmin`` 遇到全 NaN 窗口（暖机期）—— 返回 NaN 并被当成假；
    * ``np.where`` 两个分支都求值导致的除零 / 无效值 —— 结果被静默替换；
    * 空切片上的 ``nanmean`` 等。

    它们不会让测试变红，只会让数值悄悄错。本轮已把 catalog 测试的告警清到零，
    这里升级为错误，防止再次被悄悄引入。
    """
    path = str(getattr(request, "path", getattr(request.node, "fspath", "")))
    if "test_catalog_port_" not in path:
        yield
        return
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield

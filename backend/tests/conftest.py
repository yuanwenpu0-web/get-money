import sys
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

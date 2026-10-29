"""运维脚本 set_strategy_scope：把策略的运行范围（币种池）改成指定池。

这个脚本动的是线上配置，测的是「不会把绑定改坏」：撞唯一键要跳过而不是抛错、
dry-run 不落库、默认不动 enabled。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from set_strategy_scope import rebind_scope  # noqa: E402

from app.models import StrategyBinding, SymbolPool  # noqa: E402

STRATEGY = "absorption_reversal"


def _pool(db, code, symbols):
    row = SymbolPool(
        code=code,
        name=code,
        kind="static",
        symbols=json.dumps(list(symbols)),
        rule="{}",
        enabled=1,
    )
    db.add(row)
    db.commit()
    return row


def _binding(db, pool, interval="15m", enabled=0, priority=10):
    row = StrategyBinding(
        strategy_code=STRATEGY,
        pool_code=pool,
        interval=interval,
        enabled=enabled,
        priority=priority,
        params="{}",
    )
    db.add(row)
    db.commit()
    return row


def test_rebind_moves_every_binding_to_target_pool(db):
    _pool(db, "major", ["AAAUSDT"])
    _pool(db, "all", ["AAAUSDT", "BBBUSDT", "CCCUSDT"])
    first = _binding(db, "major", "15m")
    second = _binding(db, "major", "5m", priority=20)

    out = rebind_scope(db, STRATEGY, "all")

    assert out["pool_count"] == 3
    assert len(out["changes"]) == 2
    assert out["skipped"] == []
    for row in (first, second):
        db.refresh(row)
        assert row.pool_code == "all"
        # 默认不动 enabled：换范围不等于上线
        assert row.enabled == 0


def test_rebind_skips_when_target_scope_already_taken(db):
    _pool(db, "major", ["AAAUSDT"])
    _pool(db, "all", ["AAAUSDT", "BBBUSDT"])
    _binding(db, "major", "15m")
    existing = _binding(db, "all", "15m", priority=99)

    out = rebind_scope(db, STRATEGY, "all")

    assert out["changes"] == []
    assert len(out["skipped"]) == 1
    assert "占用" in out["skipped"][0]["reason"]
    # 既不能被删，也不能被改
    assert db.query(StrategyBinding).count() == 2
    assert db.get(StrategyBinding, existing.id).pool_code == "all"


def test_rebind_dry_run_does_not_write(db):
    _pool(db, "major", ["AAAUSDT"])
    _pool(db, "all", ["AAAUSDT", "BBBUSDT"])
    row = _binding(db, "major", "15m")

    out = rebind_scope(db, STRATEGY, "all", dry_run=True)

    assert len(out["changes"]) == 1
    assert out["dry_run"] is True
    db.refresh(row)
    assert row.pool_code == "major"


def test_rebind_can_enable_or_disable(db):
    _pool(db, "major", ["AAAUSDT"])
    _pool(db, "all", ["AAAUSDT", "BBBUSDT"])
    row = _binding(db, "major", "15m", enabled=0)

    rebind_scope(db, STRATEGY, "all", enable=True)
    db.refresh(row)
    assert row.enabled == 1

    rebind_scope(db, STRATEGY, "all", enable=False)
    db.refresh(row)
    assert row.enabled == 0


def test_rebind_is_idempotent(db):
    _pool(db, "all", ["AAAUSDT", "BBBUSDT"])
    _binding(db, "all", "15m")

    first = rebind_scope(db, STRATEGY, "all")
    second = rebind_scope(db, STRATEGY, "all")

    assert first["changes"] == []
    assert second["changes"] == []


def test_rebind_rejects_unknown_pool_and_missing_binding(db):
    _pool(db, "major", ["AAAUSDT"])
    with pytest.raises(ValueError, match="未知币种池"):
        rebind_scope(db, STRATEGY, "nope")
    with pytest.raises(ValueError, match="没有绑定"):
        rebind_scope(db, STRATEGY, "major")

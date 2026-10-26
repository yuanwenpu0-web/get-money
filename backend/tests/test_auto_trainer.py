"""迭代9：持续训练管线 + 训练历史 + 晋升门禁的离线单测。

不访问网络、不真跑 LightGBM：train_model 与币种宇宙全部 monkeypatch 替换，
存储用 conftest 的 :memory: sqlite。
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.ml import auto_trainer
from app.models import ModelRecord, StrategyConfig
from app.state import set_state

SYMBOLS = ["AAAUSDT", "BBBUSDT"]
CALLS: list[str] = []
FAILURES: set[str] = set()


def make_model(db, symbol, interval="1h", auc=0.6, test_rows=300, status="ready",
               hours_ago=0.0):
    record = ModelRecord(
        symbol=symbol, interval=interval, algo="lightgbm", horizon=5,
        path="/tmp/not-a-real-model.pkl", features="[]",
        metrics=json.dumps({"auc": auc, "accuracy": 0.55, "test_rows": test_rows}),
        train_start=0, train_end=0, rows=1000, status=status,
        created_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
    )
    db.add(record)
    db.commit()
    return record


def fake_train_model(db, symbol, interval, algo="lightgbm", horizon=5, days=40):
    CALLS.append(symbol)
    if symbol in FAILURES:
        raise ValueError(f"mock training failure {symbol}")
    return make_model(db, symbol, interval)


@pytest.fixture(autouse=True)
def _reset():
    CALLS.clear()
    FAILURES.clear()
    yield
    CALLS.clear()
    FAILURES.clear()


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(auto_trainer, "universe_symbols", lambda db: list(SYMBOLS))
    monkeypatch.setattr(auto_trainer, "universe_volumes", lambda db: {})
    monkeypatch.setattr(auto_trainer, "train_model", fake_train_model)


@pytest.fixture
def base_state(db):
    set_state(db, "interval", "1h")
    set_state(db, "auto_train_enabled", True)
    set_state(db, "auto_train_top_n", 2)
    set_state(db, "auto_train_interval_hours", 24)
    set_state(db, "ml_min_auc", 0.55)
    set_state(db, "ml_promote_rounds", 2)
    set_state(db, "ml_auto_enable", False)
    return db


# ---------- A. 持续训练 ----------

def test_trains_when_no_model_exists(patched, base_state):
    cycle = auto_trainer.run_auto_train_cycle(base_state)
    assert [item["symbol"] for item in cycle["trained"]] == SYMBOLS
    assert cycle["skipped"] == [] and cycle["failed"] == []
    assert CALLS == SYMBOLS


def test_skips_when_existing_model_is_fresh(patched, base_state):
    make_model(base_state, "AAAUSDT", hours_ago=1.0)
    cycle = auto_trainer.run_auto_train_cycle(base_state)
    assert [item["symbol"] for item in cycle["skipped"]] == ["AAAUSDT"]
    assert [item["symbol"] for item in cycle["trained"]] == ["BBBUSDT"]
    assert CALLS == ["BBBUSDT"]  # 新鲜的不重训


def test_trains_when_existing_model_is_stale(patched, base_state):
    make_model(base_state, "AAAUSDT", hours_ago=48.0)
    cycle = auto_trainer.run_auto_train_cycle(base_state)
    assert [item["symbol"] for item in cycle["trained"]] == SYMBOLS
    assert cycle["skipped"] == []


def test_interval_zero_forces_retrain_despite_fresh(patched, base_state):
    # 回归：fresh_hours=0 应表示“无冷却、每次都重训”，不能用 `or 24` 把 0 吞成 24
    set_state(base_state, "auto_train_interval_hours", 0)
    make_model(base_state, "AAAUSDT", hours_ago=0.01)  # 极新模型
    cycle = auto_trainer.run_auto_train_cycle(base_state)
    assert [item["symbol"] for item in cycle["trained"]] == SYMBOLS
    assert cycle["skipped"] == []  # 即便最新也强制重训



def test_single_failure_does_not_break_the_round(patched, base_state):
    FAILURES.add("AAAUSDT")
    cycle = auto_trainer.run_auto_train_cycle(base_state)
    assert [item["symbol"] for item in cycle["failed"]] == ["AAAUSDT"]
    assert [item["symbol"] for item in cycle["trained"]] == ["BBBUSDT"]  # 整轮未被中断


def test_disabled_returns_empty(patched, base_state):
    set_state(base_state, "auto_train_enabled", False)
    cycle = auto_trainer.run_auto_train_cycle(base_state)
    assert cycle["trained"] == [] and cycle["skipped"] == [] and cycle["failed"] == []
    assert cycle["enabled"] is False
    assert CALLS == []  # 开关关闭时不训练


def test_top_n_limits_targets(patched, base_state):
    set_state(base_state, "auto_train_top_n", 1)
    cycle = auto_trainer.run_auto_train_cycle(base_state)
    assert len(cycle["trained"]) == 1


# ---------- B. 训练历史 ----------

def test_list_training_history_is_ascending_and_parses_auc(db):
    make_model(db, "BTCUSDT", auc=0.51, hours_ago=3.0)
    make_model(db, "BTCUSDT", auc=0.58, hours_ago=2.0)
    make_model(db, "BTCUSDT", auc=0.62, hours_ago=1.0)
    history = auto_trainer.list_training_history(db, "BTCUSDT", "1h")
    assert [item["auc"] for item in history] == [0.51, 0.58, 0.62]
    times = [item["created_at"] for item in history]
    assert times == sorted(times)  # 升序（旧→新）
    assert history[-1]["test_rows"] == 300 and history[-1]["status"] == "ready"


def test_list_training_history_empty_for_unknown(db):
    assert auto_trainer.list_training_history(db, "NOPEUSDT", "1h") == []


# ---------- C. 晋升门禁 ----------

def _seed_two_good_rounds(db, symbol="BTCUSDT", auc=0.7, test_rows=300):
    make_model(db, symbol, auc=auc, test_rows=test_rows, hours_ago=2.0)
    make_model(db, symbol, auc=auc, test_rows=test_rows, hours_ago=1.0)


def test_promotion_passes_when_all_rounds_qualify(base_state):
    _seed_two_good_rounds(base_state)
    verdict = auto_trainer.evaluate_promotion(base_state, "BTCUSDT", "1h")
    assert verdict["promotable"] is True
    assert verdict["rounds_checked"] == 2 and verdict["rounds_required"] == 2


def test_promotion_fails_when_auc_below_threshold(base_state):
    _seed_two_good_rounds(base_state, auc=0.51)
    verdict = auto_trainer.evaluate_promotion(base_state, "BTCUSDT", "1h")
    assert verdict["promotable"] is False
    assert "AUC" in verdict["reason"]


def test_promotion_fails_when_rounds_insufficient(base_state):
    make_model(base_state, "BTCUSDT", auc=0.9, hours_ago=1.0)  # 仅 1 轮，需 2 轮
    verdict = auto_trainer.evaluate_promotion(base_state, "BTCUSDT", "1h")
    assert verdict["promotable"] is False
    assert "轮数不足" in verdict["reason"]


def test_promotion_fails_when_test_rows_insufficient(base_state):
    _seed_two_good_rounds(base_state, test_rows=50)
    verdict = auto_trainer.evaluate_promotion(base_state, "BTCUSDT", "1h")
    assert verdict["promotable"] is False
    assert "测试样本" in verdict["reason"]


def test_promotion_fails_when_any_round_not_ready(base_state):
    make_model(base_state, "BTCUSDT", auc=0.9, hours_ago=2.0)
    make_model(base_state, "BTCUSDT", auc=0.9, status="failed", hours_ago=1.0)
    verdict = auto_trainer.evaluate_promotion(base_state, "BTCUSDT", "1h")
    assert verdict["promotable"] is False
    assert "非 ready" in verdict["reason"]


def _ml_config(db, enabled=0):
    config = db.query(StrategyConfig).filter(StrategyConfig.code == "ml_ensemble").first()
    if config is None:
        config = StrategyConfig(code="ml_ensemble", name="ML 趋势共振", description="",
                                enabled=enabled, params="{}")
        db.add(config)
        db.commit()
    else:
        config.enabled = enabled
        db.commit()
    return config


def test_apply_promotion_never_enables_when_ml_auto_enable_false(patched, base_state):
    """红线用例：ml_auto_enable=False 时，即使门禁全部达标也绝不启用 ml_ensemble。"""
    _seed_two_good_rounds(base_state, auc=0.8)
    config = _ml_config(base_state, enabled=0)
    result = auto_trainer.apply_promotion(base_state, symbols=["BTCUSDT"], interval="1h")
    assert result["auto_enable"] is False
    assert result["promoted"] == []
    assert result["candidates"] == ["BTCUSDT"]   # 只记录候选
    assert result["ml_ensemble_enabled"] is False
    assert config.enabled == 0                   # 数据库侧确认未启用


def test_apply_promotion_enables_only_when_flag_on(patched, base_state):
    _seed_two_good_rounds(base_state, auc=0.8)
    config = _ml_config(base_state, enabled=0)
    set_state(base_state, "ml_auto_enable", True)
    result = auto_trainer.apply_promotion(base_state, symbols=["BTCUSDT"], interval="1h")
    assert result["promoted"] == ["BTCUSDT"]
    assert result["ml_ensemble_enabled"] is True
    assert config.enabled == 1


def test_apply_promotion_does_not_enable_when_gate_fails(patched, base_state):
    _seed_two_good_rounds(base_state, auc=0.40)  # 门禁不达标
    config = _ml_config(base_state, enabled=0)
    set_state(base_state, "ml_auto_enable", True)  # 即便开关打开
    result = auto_trainer.apply_promotion(base_state, symbols=["BTCUSDT"], interval="1h")
    assert result["promoted"] == [] and result["candidates"] == []
    assert result["ml_ensemble_enabled"] is False
    assert config.enabled == 0


def test_cycle_top_n_override_widens_target(patched, base_state):
    """--symbols-limit 必须真正作用于训练轮次，不能只影响汇总（曾静默失效）。"""
    result = auto_trainer.run_auto_train_cycle(base_state, top_n=1)
    assert [item["symbol"] for item in result["trained"]] == ["AAAUSDT"]
    assert CALLS == ["AAAUSDT"]


def test_cycle_symbols_override_bypasses_universe(patched, base_state):
    """显式给 symbols 时不走宇宙解析，按给定顺序训练。"""
    result = auto_trainer.run_auto_train_cycle(base_state, symbols=["ZZZUSDT"])
    assert [item["symbol"] for item in result["trained"]] == ["ZZZUSDT"]
    assert CALLS == ["ZZZUSDT"]


def test_promotion_aucs_are_chronological(patched, base_state):
    """门禁 aucs 必须与训练历史同口径（旧→新），避免与趋势表方向相反。"""
    make_model(base_state, "BTCUSDT", auc=0.70, hours_ago=48)
    make_model(base_state, "BTCUSDT", auc=0.90, hours_ago=1)
    verdict = auto_trainer.evaluate_promotion(base_state, "BTCUSDT", "1h")
    assert verdict["aucs"] == [0.70, 0.90]
    assert verdict["promotable"] is True

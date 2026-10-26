"""Offline checks for bounded training and one-pass evaluation."""

from threading import Event, Lock, get_ident

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.orm import Session

from app.db import Base, make_engine
from app.features.indicators import FEATURES
from app.ml import auto_trainer, trainer
from app.models import JobLog, ModelRecord
from app.state import set_state


@pytest.fixture
def training_db(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'training.db').as_posix()}")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        set_state(session, "auto_train_workers", 2)
        yield session
    engine.dispose()


def _record(db, symbol, interval):
    record = ModelRecord(symbol=symbol, interval=interval, algo="lightgbm",
                         path="unused.pkl", rows=1000, metrics='{"auc": 0.6}')
    db.add(record)
    db.commit()
    return record


def test_parallel_training_has_independent_sessions_and_stable_order(training_db, monkeypatch):
    monkeypatch.setattr(auto_trainer.os, "cpu_count", lambda: 8)
    second_finished = Event()
    owners = []
    completion = []
    guard = Lock()

    def train(worker_db, symbol, interval, **kwargs):
        with guard:
            owners.append((id(worker_db), get_ident()))
        if symbol == "AAAUSDT":
            assert second_finished.wait(5), "second worker did not start concurrently"
        record = _record(worker_db, symbol, interval)
        with guard:
            completion.append(symbol)
        if symbol == "BBBUSDT":
            second_finished.set()
        return record

    monkeypatch.setattr(auto_trainer, "train_model", train)
    cycle = auto_trainer.run_auto_train_cycle(training_db, symbols=["AAAUSDT", "BBBUSDT"])
    assert cycle["workers"] == 2
    assert [row["symbol"] for row in cycle["trained"]] == ["AAAUSDT", "BBBUSDT"]
    assert completion == ["BBBUSDT", "AAAUSDT"]
    assert len({owner for owner, _ in owners}) == 2
    assert all(owner != id(training_db) and thread != get_ident() for owner, thread in owners)


def test_failed_transaction_is_rolled_back_and_other_workers_finish(training_db, monkeypatch):
    monkeypatch.setattr(auto_trainer.os, "cpu_count", lambda: 8)

    def train(worker_db, symbol, interval, **kwargs):
        if symbol == "FAILUSDT":
            # A real DB error leaves the Session unusable until rollback.
            worker_db.add(ModelRecord(symbol=symbol, interval=interval, algo="lightgbm", path=None))
            worker_db.flush()
        return _record(worker_db, symbol, interval)

    monkeypatch.setattr(auto_trainer, "train_model", train)
    cycle = auto_trainer.run_auto_train_cycle(
        training_db, symbols=["AAAUSDT", "FAILUSDT", "BBBUSDT", "CCCUSDT"]
    )
    assert [row["symbol"] for row in cycle["trained"]] == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    assert [row["symbol"] for row in cycle["failed"]] == ["FAILUSDT"]
    assert training_db.query(ModelRecord).count() == 3
    assert training_db.query(JobLog).filter(JobLog.level == "error").count() == 1


def test_worker_count_respects_cpu_target_and_requested_bounds(training_db, monkeypatch):
    monkeypatch.setattr(auto_trainer.os, "cpu_count", lambda: 6)
    assert auto_trainer._training_workers(training_db, {"auto_train_workers": 8}, 20) == 3
    assert auto_trainer._training_workers(training_db, {"auto_train_workers": 8}, 1) == 1
    assert auto_trainer._training_workers(training_db, {"auto_train_workers": 1}, 20) == 1
    monkeypatch.setattr(auto_trainer.os, "cpu_count", lambda: 64)
    assert auto_trainer._training_workers(training_db, {"auto_train_workers": 100}, 20) == 8


def test_memory_sqlite_remains_serial_and_duplicate_targets_train_once(db, monkeypatch):
    calls = []

    def train(worker_db, symbol, interval, **kwargs):
        assert worker_db is db
        calls.append(symbol)
        return _record(worker_db, symbol, interval)

    monkeypatch.setattr(auto_trainer, "train_model", train)
    cycle = auto_trainer.run_auto_train_cycle(db, symbols=["AAAUSDT", "AAAUSDT", "BBBUSDT"])
    assert cycle["workers"] == 1
    assert calls == ["AAAUSDT", "BBBUSDT"]


@pytest.mark.parametrize("algo", ["lightgbm", "xgboost"])
def test_probability_argmax_matches_classifier_labels_including_ties(algo):
    pytest.importorskip(algo)
    matrix = np.random.default_rng(42).normal(size=(500, 3))
    labels = (matrix[:, 0] > 0).astype(int)
    estimator = trainer._build_estimator(algo, {"n_estimators": 5})
    estimator.fit(matrix, labels)
    probabilities = estimator.predict_proba(matrix)
    np.testing.assert_array_equal(estimator.predict(matrix), estimator.classes_[np.argmax(probabilities, axis=1)])
    # sklearn's argmax breaks an exact 0.5/0.5 tie toward the first class.
    tied = trainer._build_estimator(algo, {"n_estimators": 1})
    if algo == "xgboost":
        tied.set_params(n_estimators=0, base_score=0.5)
    constant_matrix = np.zeros_like(matrix)
    tied.fit(constant_matrix, np.tile([0, 1], 250))
    tie_probabilities = tied.predict_proba(constant_matrix)
    np.testing.assert_allclose(tie_probabilities, 0.5)
    np.testing.assert_array_equal(tied.predict(constant_matrix), tied.classes_[np.argmax(tie_probabilities, axis=1)])


def test_training_evaluation_runs_inference_once(db, tmp_path, monkeypatch):
    data = pd.DataFrame(0.1, index=pd.date_range("2026-01-01", periods=600, freq="h", tz="UTC"),
                        columns=FEATURES)
    data["label"] = np.arange(len(data)) % 2
    calls = []

    class Estimator:
        classes_ = np.array([0, 1])

        def fit(self, matrix, labels):
            return self

        def predict_proba(self, matrix):
            calls.append(len(matrix))
            # Include a tie, a short, and a long prediction.
            probability = np.resize([0.5, 0.4, 0.7], len(matrix))
            return np.column_stack([1 - probability, probability])

    monkeypatch.setattr(trainer, "build_dataset", lambda *args, **kwargs: data)
    monkeypatch.setattr(trainer, "_build_estimator", lambda algo: Estimator())
    monkeypatch.setattr(trainer, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(trainer.pickle, "dump", lambda *args: None)
    record = trainer.train_model(db, "AAAUSDT", "1h")
    assert record.rows == 600
    assert calls == [120]

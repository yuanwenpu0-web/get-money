import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.minute_research import data_fingerprint, make_features, make_labels, time_partitions, validation_blocks


def candles(n=2500):
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .001, n)))
    return pd.DataFrame({"open_time": np.arange(n) * 60000, "close_time": np.arange(n) * 60000 + 59999, "open": close * .9999,
                         "high": close * 1.001, "low": close * .999, "close": close,
                         "volume": rng.uniform(1, 5, n), "quote_volume": rng.uniform(100, 500, n),
                         "trades": rng.integers(1, 20, n), "taker_buy_volume": rng.uniform(.1, .5, n),
                         "taker_buy_quote_volume": rng.uniform(10, 50, n)})


def test_features_ignore_future_and_labels_use_only_future_window():
    frame = candles()
    changed = frame.copy()
    changed.loc[501:, "close"] *= 2
    pd.testing.assert_frame_equal(make_features(frame).iloc[:501], make_features(changed).iloc[:501])
    labels = make_labels(frame)
    assert labels.future_return.iloc[500] == frame.close.iloc[515] / frame.close.iloc[500] - 1
    assert labels.iloc[-15:].isna().all().all()


def test_time_partitions_purge_label_horizon():
    train, validation, test = time_partitions(1000)
    assert train.stop - 1 + 15 < validation.start
    assert validation.stop - 1 + 15 < test.start


def test_validation_roles_are_purged_and_snapshot_detects_changes():
    times = np.arange(400) * 60000
    blocks = validation_blocks(times)
    for left, right in zip(blocks, blocks[1:]):
        assert times[left.stop - 1] + 15 * 60000 < times[right.start]
    frame = candles()
    changed = frame.copy()
    changed.loc[400, "volume"] += 1
    assert data_fingerprint(frame) != data_fingerprint(changed)


def test_lightgbm_pipeline_emits_versioned_models_and_holdout_metrics(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_major_models.py"
    spec = importlib.util.spec_from_file_location("train_major_models", path)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    report = training.fit_models(candles(), tmp_path, trees=4, threads=1, metadata={"symbol": "BTCUSDT"})
    config = json.loads((tmp_path / "model_config.json").read_text())
    assert config["symbol"] == "BTCUSDT" and config["status"] == "candidate"
    assert config["split"]["train"]["last_open_ms"] + 15 * 60000 < config["split"]["validation"]["first_open_ms"]
    assert (tmp_path / "model.pkl").exists() and "direction_raw" in report and "future_volatility" in report

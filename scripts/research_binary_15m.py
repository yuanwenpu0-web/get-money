"""独立离线研究：15 分钟方向、三分类弃判与验证集置信度筛选。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'scripts'))
from app.db import make_engine
from app.ml.minute_research import HORIZON, classify, data_fingerprint, make_features, make_labels, time_partitions
from app.ml.research_data import load_candles, validate_candles
from train_major_models import SAFE_END_MS


def select_threshold(y, probabilities, minimum_coverage=0.5):
    """预设概率阈值网格，仅在验证集最大化准确率；平分时优先覆盖率。"""
    confidence = np.maximum(probabilities, 1 - probabilities)
    predicted = (probabilities >= .5).astype(int)
    candidates = []
    for threshold in np.arange(.5, .801, .025):
        mask = confidence >= threshold
        if mask.mean() >= minimum_coverage:
            candidates.append((float(np.mean(predicted[mask] == y[mask])), float(mask.mean()), float(round(threshold, 3))))
    return max(candidates)[2]


def metrics(y, prediction, mask):
    count = int(mask.sum())
    return {'total': int(len(y)), 'selected': count, 'coverage': count / len(y),
            'accuracy': float(np.mean(prediction[mask] == y[mask])) if count else None,
            'balanced_accuracy': float(balanced_accuracy_score(y[mask], prediction[mask])) if count and len(np.unique(y[mask])) == 2 else None,
            'up_fraction': float(np.mean(y[mask])) if count else None}


def experiment(frame, output, trees=120, threads=3):
    features = make_features(frame)
    labels = make_labels(frame).future_return
    valid = features.notna().all(axis=1) & labels.notna() & (frame.open_time % (15 * 60000) == 0)
    x = features.loc[valid].astype(np.float32).reset_index(drop=True)
    returns = labels.loc[valid].reset_index(drop=True).to_numpy()
    times = frame.loc[valid, 'open_time'].reset_index(drop=True).to_numpy()
    tr, va, te = time_partitions(len(x))
    if min(len(x.iloc[s]) for s in (tr, va, te)) < 100:
        raise ValueError('有效样本不足')
    # 保守复用原函数的 15 样本隔离，并对真实时间再次断言。
    if not (times[tr.stop-1] + HORIZON*60000 < times[va.start] and times[va.stop-1] + HORIZON*60000 < times[te.start]):
        raise ValueError('标签跨越时间分界')
    flat = returns == 0
    direction = (returns > 0).astype(int)
    magnitude = float(np.quantile(np.abs(returns[tr]), 1/3))
    if magnitude <= 0:
        raise ValueError('三分类训练阈值无效')
    three_y = classify(returns, magnitude)
    params = dict(n_estimators=trees, learning_rate=.05, num_leaves=31, min_child_samples=100,
                  n_jobs=threads, random_state=42, verbosity=-1, deterministic=True, force_col_wise=True)
    train_nonflat = ~flat[tr]
    if len(np.unique(direction[tr][train_nonflat])) != 2 or len(np.unique(three_y[tr])) != 3:
        raise ValueError('训练类别不足')
    binary = lgb.LGBMClassifier(objective='binary', **params)
    binary.fit(x.iloc[tr].iloc[train_nonflat], direction[tr][train_nonflat])
    three = lgb.LGBMClassifier(objective='multiclass', **params)
    three.fit(x.iloc[tr], three_y[tr])
    val_mask = ~flat[va]
    val_probs = binary.predict_proba(x.iloc[va])[val_mask, 1]
    threshold = select_threshold(direction[va][val_mask], val_probs)
    test_mask = ~flat[te]
    y = direction[te][test_mask]
    if not len(y):
        raise ValueError('测试集无非零涨跌样本')
    p = binary.predict_proba(x.iloc[te])[test_mask, 1]
    p3 = three.predict_proba(x.iloc[te])[test_mask]
    pred3 = np.argmax(p3, axis=1)
    pred = (p >= .5).astype(int)
    baseline = int(np.mean(direction[tr][train_nonflat]) >= .5)
    report = {'status': 'research_candidate', 'horizon_minutes': HORIZON, 'sampling_minutes': 15,
              'feature_time': '1m candle close; decision after candle close',
              'zero_return_policy': 'excluded from binary training, threshold selection and direction metrics; retained in three-class training',
              'three_class_rule': 'training absolute return 1/3 quantile; neutral predictions abstain',
              'three_class_threshold': magnitude, 'confidence_threshold': threshold,
              'threshold_rule': 'validation grid 0.500..0.800 step 0.025; max accuracy with coverage >= 0.5, ties prefer coverage',
              'split': {}, 'test': {}, 'model_parameters': params, 'feature_names': list(x.columns),
              'data_fingerprint': data_fingerprint(frame)}
    for name, s in [('train', tr), ('validation', va), ('test', te)]:
        report['split'][name] = {'rows': len(times[s]), 'flat_rows': int(flat[s].sum()),
                                 'first_open_ms': int(times[s][0]), 'last_open_ms': int(times[s][-1])}
    report['validation_selected'] = metrics(direction[va][val_mask], (val_probs >= .5).astype(int), np.maximum(val_probs, 1-val_probs) >= threshold)
    report['test']['majority_train'] = metrics(y, np.full(len(y), baseline), np.ones(len(y), dtype=bool))
    report['test']['three_class_direction'] = metrics(y, (pred3 == 2).astype(int), pred3 != 1)
    report['test']['binary'] = metrics(y, pred, np.ones(len(y), dtype=bool))
    report['test']['binary']['log_loss'] = float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1]))
    report['test']['binary_filtered'] = metrics(y, pred, np.maximum(p, 1-p) >= threshold)
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump({'binary': binary, 'three_class': three, 'features': list(x.columns), 'confidence_threshold': threshold,
                 'three_class_threshold': magnitude}, output / 'candidate.joblib', compress=3)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'])
    parser.add_argument('--end-ms', type=int, default=SAFE_END_MS)
    parser.add_argument('--trees', type=int, default=120)
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    if args.end_ms > SAFE_END_MS or args.trees < 1 or args.threads < 1:
        parser.error('截止时间超出核查范围或参数无效')
    run = ROOT / 'data' / 'model_research' / ('binary_15m_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(parents=True, exist_ok=False)
    engine = make_engine(read_only=True, null_pool=True)
    results = {}
    try:
        for symbol in args.symbols:
            started = time.monotonic()
            print(f'{symbol}: 读取数据', flush=True)
            try:
                frame = load_candles(engine, symbol, args.end_ms)
                frame, quality = validate_candles(frame, symbol, args.end_ms)
                (run / f'{symbol}_quality.json').write_text(json.dumps(quality, indent=2), encoding='utf-8')
                if not quality['passed']:
                    raise ValueError('数据质量检查未通过')
                print(f'{symbol}: 训练，K线 {len(frame)}', flush=True)
                report = experiment(frame, run / symbol, args.trees, args.threads)
                results[symbol] = {'status': 'candidate', 'elapsed_seconds': round(time.monotonic()-started, 2), 'test': report['test']}
            except Exception as exc:
                results[symbol] = {'status': 'failed', 'error': type(exc).__name__, 'detail': str(exc)}
            (run / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            print(f'{symbol}: {results[symbol]}', flush=True)
    finally:
        engine.dispose()
    print(f'研究目录: {run}', flush=True)
    return int(any(r['status'] != 'candidate' for r in results.values()))


if __name__ == '__main__':
    raise SystemExit(main())

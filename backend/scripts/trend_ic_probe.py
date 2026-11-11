"""只读趋势 IC 探针：所有数据库查询截止于既有测试段起点。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, binomtest
from scipy.signal import lfilter
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.db import make_engine

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT', 'LTCUSDT', 'BCHUSDT']
START = pd.Timestamp('2026-01-01T00:00:00Z')
TRAIN_END = pd.Timestamp('2026-08-01T00:00:00Z')
VALID_END = pd.Timestamp('2026-10-02T12:00:00Z')
END = pd.Timestamp('2026-10-25T00:00:00Z')
INTERVALS = {'15m': 15, '1h': 60, '4h': 240}
FEATURES = ['momentum_24h', 'ema_gap_12_48h', 'efficiency_24h']
SQL = text('SELECT open_time, open, close FROM klines WHERE symbol=:symbol AND interval=:iv AND open_time>=:start AND open_time<:end ORDER BY open_time')
SEED = 20261110


def ic(x, y):
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return None
    a, b = rankdata(x[ok]), rankdata(y[ok])
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def features(c, h):
    contiguous = c.notna().rolling(h + 1, min_periods=h + 1).sum().eq(h + 1)
    mom = (c / c.shift(h) - 1).where(contiguous)
    groups = c.isna().cumsum()
    fast = c.groupby(groups).transform(lambda x: x.ewm(span=h // 2, adjust=False, min_periods=h // 2).mean())
    slow = c.groupby(groups).transform(lambda x: x.ewm(span=h * 2, adjust=False, min_periods=h * 2).mean())
    path = c.diff().abs().rolling(h, min_periods=h).sum()
    return [mom, fast / slow - 1, ((c - c.shift(h)) / path.replace(0, np.nan)).where(contiguous)]


def circular_p(x, y, h, rng, n=199):
    # 保留每条序列的自相关；以整段循环位移形成近似平稳零分布。
    if len(x) < max(100, 8 * h) or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    observed = float(np.mean(rx * ry))
    offsets = rng.integers(2 * h, len(x) - 2 * h, n)
    null = np.array([np.mean(rx * np.roll(ry, k)) for k in offsets])
    return float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (n + 1))


def calibration():
    # 独立生成的 AR(1) 因子、噪声与标签，避免用真实未来收益自身作正对照。
    rng = np.random.default_rng(SEED)
    n, h = 2400, 24
    x, noise = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        x[i] = .85 * x[i-1] + rng.normal()
        noise[i] = .85 * noise[i-1] + rng.normal()
    x /= x.std(); noise /= noise.std()
    base = rng.normal(size=n)
    rows = []
    for name, strength in [('positive', .8), ('weak', .2), ('negative', 0.)]:
        y = strength * x + np.sqrt(1 - strength ** 2) * noise
        rows.append({'control': name, 'injected_strength': strength, 'ic': ic(x, y), 'circular_p': circular_p(x, y, h, rng, 999)})
    # 在多次独立零假设模拟上测经验假阳性，保留自相关结构。
    pvalues = []
    for _ in range(100):
        a, b = np.zeros(1200), np.zeros(1200)
        for i in range(1, len(a)):
            a[i] = .9 * a[i-1] + rng.normal()
            b[i] = .9 * b[i-1] + rng.normal()
        pvalues.append(circular_p(a, b, h, rng, 199))
    false_rate = float(np.mean(np.array(pvalues) <= .05))
    return {'controls': rows, 'null_simulations': 100, 'null_rejection_rate_05': false_rate,
            'passed': bool(rows[0]['circular_p'] <= .05 and 0 < rows[1]['ic'] < rows[0]['ic'] and rows[2]['circular_p'] > .05 and false_rate <= .10),
            'method': '独立 AR(1) 合成序列，循环位移检验；100 次零假设模拟，允许至多 10% 拒绝率'}


def causal_check(c, h):
    # 截断以及截断后扰动独立复算；逐特征比较此前全部结果。
    cut = len(c) // 2
    original = features(c, h)
    truncated = features(c.iloc[:cut], h)
    changed = c.copy()
    changed.iloc[cut:] = changed.iloc[cut:] * 3 + 17
    modified = features(changed, h)
    return all(np.allclose(a.iloc[:cut], b, equal_nan=True) and np.allclose(a.iloc[:cut], d.iloc[:cut], equal_nan=True)
               for a, b, d in zip(original, truncated, modified))


def main():
    rng = np.random.default_rng(SEED)
    result = {'config': {'symbols_requested': SYMBOLS, 'start': START.isoformat(), 'train_end': TRAIN_END.isoformat(), 'validation_end': VALID_END.isoformat(), 'reserved_test_start': VALID_END.isoformat(), 'original_window_end': END.isoformat(), 'intervals': INTERVALS, 'horizon_hours': 24, 'features': FEATURES, 'min_split_coverage': .90, 'seed': SEED}, 'coverage': [], 'skipped': [], 'results': [], 'summary': [], 'calibration': calibration(), 'causality': [], 'baseline': []}
    with make_engine(read_only=True).connect() as conn:
        for interval, minutes in INTERVALS.items():
            step, h = pd.Timedelta(minutes=minutes), 1440 // minutes
            index = pd.date_range(START, VALID_END, freq=step, inclusive='left')
            for symbol in SYMBOLS:
                rows = conn.execute(SQL, {'symbol': symbol, 'iv': interval, 'start': int(START.timestamp()*1000), 'end': int(VALID_END.timestamp()*1000)}).fetchall()
                frame = pd.DataFrame(rows, columns=['time', 'open', 'close'])
                frame['time'] = pd.to_datetime(frame['time'], unit='ms', utc=True)
                frame = frame.set_index('time').reindex(index).astype(float)
                c, o = frame['close'].where(frame['close'] > 0), frame['open'].where(frame['open'] > 0)
                fs = features(c, h)
                causal = causal_check(c, h) if c.notna().sum() >= 4 * h else None
                result['causality'].append({'symbol': symbol, 'interval': interval, 'passed': causal})
                future = o.shift(-(h + 1)) / o.shift(-1) - 1
                future_ok = c.notna().iloc[::-1].rolling(h + 2, min_periods=h + 2).sum().iloc[::-1].eq(h + 2)
                future = future.where(future_ok)
                for split, lo, hi in [('train', START, TRAIN_END), ('validation', TRAIN_END, VALID_END)]:
                    split_mask = (index >= lo) & (index < hi)
                    coverage = float((c.notna() & o.notna())[split_mask].mean())
                    result['coverage'].append({'symbol': symbol, 'interval': interval, 'split': split, 'fraction': coverage, 'rows': int((c.notna() & o.notna())[split_mask].sum()), 'expected': int(split_mask.sum())})
                    if coverage < .9:
                        result['skipped'].append({'symbol': symbol, 'interval': interval, 'split': split, 'reason': 'coverage_below_90_percent', 'coverage': coverage})
                        continue
                    available = c[split_mask].dropna()
                    if len(available) > 1:
                        result['baseline'].append({'symbol': symbol, 'interval': interval, 'split': split, 'close_change': float(available.iloc[-1] / available.iloc[0] - 1)})
                    mask = (index >= lo) & ((index + (h + 1) * step) < hi)
                    for name, feature in zip(FEATURES, fs):
                        pair = pd.concat([feature.rename('x'), future.where(mask).rename('y')], axis=1).dropna()
                        # 连续且不重叠的固定相位样本，避免把重叠标签当独立观测。
                        pair = pair.iloc[np.flatnonzero((index.get_indexer(pair.index) % h) == 0)]
                        value = ic(pair.x, pair.y)
                        result['results'].append({'symbol': symbol, 'interval': interval, 'split': split, 'feature': name, 'pairs_nonoverlap': len(pair), 'ic': value})
    for interval in INTERVALS:
        for feature in FEATURES:
            for split in ['train', 'validation']:
                rows = [r for r in result['results'] if r['interval'] == interval and r['split'] == split and r['feature'] == feature and r['ic'] is not None]
                vals = np.array([r['ic'] for r in rows])
                if len(vals):
                    mean = float(vals.mean()); same = float(np.mean(np.sign(vals) == np.sign(mean)))
                    result['summary'].append({'interval': interval, 'feature': feature, 'split': split, 'symbols': len(vals), 'mean_ic': mean, 'same_sign_fraction': same, 'threshold_met': bool(abs(mean) > .02 and same >= .60), 'pairs_nonoverlap': sum(r['pairs_nonoverlap'] for r in rows)})
    result.update({'gate1': {'passed': False, 'reason': 'calibration_failed' if not result['calibration']['passed'] else 'independent_gate_review_required'}, 'costs': {'fee_rate': .0004, 'slippage': .0002, 'round_trip_approx': .0012, 'basis': 'notional', 'deducted_from_ic': False}, 'window': result['config'], 'universe': SYMBOLS, 'trades': None, 'ci': None, 'verdict': 'insufficient', 'limitations': ['IC is not net trade performance', 'Synthetic controls do not establish stationarity of market data', 'Cross-symbol dependence and short validation window', 'No FDR or trade-level random baseline']})
    (ROOT / 'docs' / 'probe-trend-causal-ic.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'summary': result['summary'], 'calibration': result['calibration'], 'causal_failures': sum(x['passed'] is False for x in result['causality']), 'causal_unchecked': sum(x['passed'] is None for x in result['causality']), 'skipped': result['skipped']}, ensure_ascii=False, indent=2))


def review():
    # 固定方案：独立于先前模拟的种子，2000 次、AR(1)=0.9、长度1200、199 位移。
    # 预设容忍条件为误报率单侧95%精确上限 < 10%，另报告对5%的双侧检验。
    path = ROOT / 'docs' / 'probe-trend-causal-ic.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    rng = np.random.default_rng(20261111)
    n = 2000
    pvalues = []
    for _ in range(n):
        innovations = rng.normal(size=(2, 1200))
        innovations[:, 0] = 0
        a, b = lfilter([1.], [1., -.9], innovations, axis=1)
        pvalues.append(circular_p(a, b, 24, rng, 199))
    count = sum(p <= .05 for p in pvalues)
    btest = binomtest(count, n, .05)
    bounds = btest.proportion_ci(.95, method='exact')
    upper = binomtest(count, n, .1, alternative='less').proportion_ci(.95, method='exact').high
    old = binomtest(7, 100, .05)
    old_ci = old.proportion_ci(.95, method='exact')
    pairs = []
    for interval in INTERVALS:
        for feature in FEATURES:
            rows = [r for r in result['summary'] if r['interval'] == interval and r['feature'] == feature]
            train = next(r for r in rows if r['split'] == 'train')
            valid = next(r for r in rows if r['split'] == 'validation')
            same = train['mean_ic'] * valid['mean_ic'] > 0
            pairs.append({'interval': interval, 'feature': feature, 'train_threshold': train['threshold_met'], 'validation_threshold': valid['threshold_met'], 'same_direction': same, 'eligible': bool(same and train['threshold_met'] and valid['threshold_met'])})
    result['statistical_review'] = {'protocol': {'seed': 20261111, 'simulations': n, 'ar_coefficient': .9, 'length': 1200, 'circular_shifts': 199, 'h': 24, 'alpha': .05, 'tolerance_one_sided_95_upper_below': .10}, 'previous_100': {'rejections': 7, 'ci_95': [old_ci.low, old_ci.high], 'p_two_sided_vs_05': old.pvalue}, 'independent': {'rejections': count, 'pvalues': pvalues, 'rate': count / n, 'ci_95': [bounds.low, bounds.high], 'upper_one_sided_95': upper, 'p_two_sided_vs_05': btest.pvalue, 'tolerance_passed': bool(upper < .1)}, 'signals': pairs, 'note': '单次负对照显著不代表校准失败；原 calibration 字段保留历史结论，不作为复核判定。'}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result['statistical_review'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    if sys.argv[1:] == ['--review']:
        review()
    elif not sys.argv[1:]:
        raise SystemExit('为保护已有结果，原数据库探针不再覆盖报告；请另行明确安排输出。')
    else:
        raise SystemExit('支持参数：--review')

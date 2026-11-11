"""离线 4h EMA 差交易评估；禁止读取预留测试段。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend' / 'scripts'))
from trend_ic_probe import START, TRAIN_END, VALID_END, SYMBOLS, features
from app.db import make_engine

STEP = pd.Timedelta(hours=4)
H = 6
FEE, SLIP = .0004, .0002
SQL = text("SELECT open_time, open, high, low, close FROM klines WHERE symbol=:symbol AND interval='4h' AND open_time>=:start AND open_time<:end ORDER BY open_time")
RULES = [(direction, size, stop) for direction in ('reverse', 'trend', 'long', 'short')
         for size in ('fixed', 'vol_target') for stop in ('hold', 'atr_1_2', 'atr_2_4')]


def replay(o, hi, lo, side, atr_ratio, rule):
    entry = o[0]
    if rule == 'hold':
        raw_exit = o[H]
    else:
        sl, tp = (1., 2.) if rule == 'atr_1_2' else (2., 4.)
        loss = entry * (1 - side * sl * atr_ratio)
        profit = entry * (1 + side * tp * atr_ratio)
        raw_exit = o[H]
        for j in range(H):
            if side * (o[j] - loss) <= 0:
                raw_exit = o[j]; break
            if side * (o[j] - profit) >= 0:
                raw_exit = o[j]; break
            stop_hit = lo[j] <= loss if side == 1 else hi[j] >= loss
            tp_hit = hi[j] >= profit if side == 1 else lo[j] <= profit
            if stop_hit:
                raw_exit = loss; break
            if tp_hit:
                raw_exit = profit; break
    entry_fill = entry * (1 + side * SLIP)
    exit_fill = raw_exit * (1 - side * SLIP)
    gross = side * (exit_fill / entry_fill - 1)
    return gross - FEE * (1 + exit_fill / entry_fill)


def bh(p):
    order = np.argsort(p)
    q = np.minimum.accumulate((np.asarray(p)[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    out = np.empty(len(p)); out[order] = np.minimum(q, 1)
    return out


def main():
    index = pd.date_range(START, VALID_END, freq=STEP, inclusive='left')
    entries, coverage, causality = [], [], []
    with make_engine(read_only=True).connect() as conn:
        for symbol in SYMBOLS:
            rows = conn.execute(SQL, {'symbol': symbol, 'start': int(START.timestamp()*1000), 'end': int(VALID_END.timestamp()*1000)}).fetchall()
            df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close'])
            df['time'] = pd.to_datetime(df.time, unit='ms', utc=True)
            df = df.set_index('time').reindex(index).astype(float)
            valid = (df > 0).all(axis=1) & (df.high >= df[['open', 'close', 'low']].max(axis=1)) & (df.low <= df[['open', 'close', 'high']].min(axis=1))
            c = df.close.where(valid)
            gap = features(c, H)[1]
            cut = len(c)//2
            altered = c.copy(); altered.iloc[cut:] = altered.iloc[cut:] * 3 + 17
            truncated = features(c.iloc[:cut], H)[1]
            changed = features(altered, H)[1]
            finite_before = int(gap.iloc[:cut].notna().sum())
            causal = bool(finite_before > 0 and np.allclose(gap.iloc[:cut], truncated, equal_nan=True) and np.allclose(gap.iloc[:cut], changed.iloc[:cut], equal_nan=True))
            causality.append({'symbol': symbol, 'finite_before_cut': finite_before, 'passed': causal})
            prev = c.shift(1)
            tr = pd.concat([(df.high-df.low), (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1).where(valid & prev.notna())
            atr = tr.rolling(14, min_periods=14).mean() / c
            for split, start, end in [('train', START, TRAIN_END), ('validation', TRAIN_END, VALID_END)]:
                mask = (index >= start) & (index < end)
                fraction = float(valid[mask].mean())
                coverage.append({'symbol': symbol, 'split': split, 'fraction': fraction, 'eligible': fraction >= .9})
                if fraction < .9: continue
                for i in np.flatnonzero(mask & (np.arange(len(index)) % H == 0)):
                    if i + H + 1 >= len(index) or index[i + H + 1] >= end or not np.isfinite(gap.iloc[i]) or not np.isfinite(atr.iloc[i]) or atr.iloc[i] <= 0:
                        continue
                    window = df.iloc[i+1:i+H+2]
                    if not valid.iloc[i+1:i+H+2].all(): continue
                    o, hi, lo = (window[k].to_numpy() for k in ('open', 'high', 'low'))
                    rr = float(atr.iloc[i]); weight = min(1., .02 / rr)
                    entry = {'symbol': symbol, 'split': split, 'time': index[i].isoformat(), 'signal_side': -1 if gap.iloc[i] > 0 else 1, 'vol_weight': weight, 'returns': {}}
                    for stop in ('hold', 'atr_1_2', 'atr_2_4'):
                        entry['returns'][stop] = [replay(o, hi, lo, side, rr, stop) for side in (-1, 1)]
                    entries.append(entry)
    rng = np.random.default_rng(20261112)
    output = {'window': {'start': START.isoformat(), 'train_end': TRAIN_END.isoformat(), 'validation_end': VALID_END.isoformat(), 'test_read': False}, 'universe': SYMBOLS, 'coverage': coverage, 'causality': causality, 'costs': {'fee_rate': FEE, 'slippage': SLIP, 'basis': 'notional'}, 'rules': [list(r) for r in RULES], 'parameters': {'holding_hours': 24, 'entry': 'next bar open', 'tp_sl': 'ATR14 1:2 or 2:4, stop first if both', 'size': 'fixed 1 or min(1,0.02/ATR_fraction)', 'random_seeds': 100, 'bootstraps': 2000, 'seed': 20261112, 'fdr_family': 'all 24 validation rules, exploratory'}, 'split_results': {}}
    days = {}
    for split in ('train', 'validation'):
        subset = [e for e in entries if e['split'] == split]
        ts = sorted(set(e['time'] for e in subset)); lookup = {t: i for i,t in enumerate(ts)}
        sums = np.zeros((len(RULES), len(ts))); counts = np.zeros(len(ts)); direction_returns = np.zeros((len(RULES), len(subset), 2))
        for j,e in enumerate(subset):
            d = lookup[e['time']]; counts[d] += 1
            for k,(direction,size,stop) in enumerate(RULES):
                side = {'reverse': e['signal_side'], 'trend': -e['signal_side'], 'long': 1, 'short': -1}[direction]
                w = 1 if size == 'fixed' else e['vol_weight']
                direction_returns[k,j] = np.array(e['returns'][stop]) * w
                sums[k,d] += direction_returns[k,j,int(side == 1)]
        days[split] = (subset, ts, counts, sums, direction_returns)
        output['split_results'][split] = [{'rule': list(rule), 'trades': len(subset), 'mean_net': float(sums[k].sum()/counts.sum())} for k,rule in enumerate(RULES)]
    train = output['split_results']['train']
    selected = max(range(len(RULES)), key=lambda k: train[k]['mean_net']) if len(days['train'][0]) >= 30 else None
    subset, ts, counts, sums, dr = days['validation']
    if selected is None or len(subset) < 30 or len(ts) < 4:
        output['verdict'] = 'insufficient'
    else:
        n = len(ts)
        boot = rng.integers(0, n, (2000, n))
        observed = sums.sum(axis=1) / counts.sum()
        ci = np.quantile(sums[:,boot].sum(axis=2) / counts[boot].sum(axis=1), [.025,.975], axis=1).T
        signs = rng.choice([-1.,1.], size=(5000,n))
        p = (1 + (np.abs(signs @ sums.T) >= np.abs(sums.sum(axis=1))).sum(axis=0)) / 5001
        q = bh(p)
        random_means = []
        for _ in range(100):
            sides = rng.integers(0,2,len(subset))
            random_means.append(float(dr[selected,np.arange(len(subset)),sides].mean()))
        seg = [float(sums[selected,ids].sum()/counts[ids].sum()) for ids in np.array_split(np.arange(n),4)]
        low, high = map(float,ci[selected])
        percentile = float(np.mean(np.asarray(random_means) < observed[selected]))
        raw_verdict = 'pass' if low > 0 else 'negative' if high < 0 else 'inconclusive'
        output['selection'] = {'rule': list(RULES[selected]), 'selection': 'training maximum net mean among 24 rules; no validation selection', 'training_trades': len(days['train'][0])}
        output['validation'] = {'trades': len(subset), 'days': n, 'mean_net': float(observed[selected]), 'ci_95': [low,high], 'ci_unit': 'UTC day blocks, all coins jointly', 'raw_ci_verdict': raw_verdict, 'random_means': random_means, 'random_percentile': percentile, 'p_signflip_day': float(p[selected]), 'fdr_q_24': float(q[selected]), 'four_time_segments': seg, 'same_sign_segments': max(sum(v > 0 for v in seg),sum(v < 0 for v in seg))}
        for i, row in enumerate(output['split_results']['validation']):
            row.update({'ci_95': ci[i].tolist(), 'p_signflip_day': float(p[i]), 'fdr_q_24': float(q[i])})
        gates = low > 0 and percentile > .95 and q[selected] < .1 and sum(v > 0 for v in seg) >= 3
        output['verdict'] = 'pass' if gates else ('negative' if high < 0 else 'inconclusive')
        output['validation']['all_gates_pass'] = bool(gates)
        # 单独在信号驱动族内用训练段选择，避免恒定方向夺冠后无法评估信号。
        signal_index = max(range(12), key=lambda k: train[k]['mean_net'])
        signal_random = []
        for _ in range(100):
            sides = rng.integers(0, 2, len(subset))
            signal_random.append(float(dr[signal_index, np.arange(len(subset)), sides].mean()))
        signal_segments = [float(sums[signal_index,ids].sum()/counts[ids].sum()) for ids in np.array_split(np.arange(n),4)]
        output['signal_family_analysis'] = {'selection': 'training maximum net mean among 12 signal-driven rules', 'rule': list(RULES[signal_index]), 'train_mean_net': train[signal_index]['mean_net'], 'trades': len(subset), 'mean_net': float(observed[signal_index]), 'ci_95': ci[signal_index].tolist(), 'p_signflip_day': float(p[signal_index]), 'fdr_q_24': float(q[signal_index]), 'random_means': signal_random, 'random_percentile': float(np.mean(np.asarray(signal_random) < observed[signal_index])), 'four_time_segments': signal_segments}
    if output.get('signal_family_analysis'):
        signal = output['signal_family_analysis']
        output['ci'] = {'of': 'training-selected signal family, day-block bootstrap', 'ci_95': signal['ci_95']}
        output['trades'] = signal['trades']
        output['signal_ci_verdict'] = 'pass' if signal['ci_95'][0] > 0 else ('negative' if signal['ci_95'][1] < 0 else 'inconclusive')
    output['limitations'] = ['Validation period was already inspected during IC probe: exploratory, not fresh confirmation', 'H2 stop line applies; user requested research only', 'No funding, order book depth, liquidation or partial fills', 'Daily bootstrap and sign flip assume independent/exchangeable day blocks; cross-day dependence remains', 'Fixed notional contribution, no joint portfolio margin or capital constraints', 'Training and validation coverage cohorts differ']
    out = Path(sys.argv[1]) if len(sys.argv) == 2 else ROOT / 'docs' / 'trend-ema-offline-eval.json'
    if len(sys.argv) > 2:
        raise SystemExit('用法：trend_ema_offline_eval.py [新输出文件路径]')
    if out.exists():
        existing = json.loads(out.read_text(encoding='utf-8'))
        if 'signal_family_analysis' in existing:
            raise FileExistsError('信号族结果已存在，拒绝覆盖')
        if existing['split_results'] != output['split_results']:
            raise ValueError('原始评估与重算不一致，拒绝混合结果')
        existing['signal_family_analysis'] = output['signal_family_analysis']
        output = existing
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'selection': output.get('selection'), 'validation': {k:v for k,v in output.get('validation',{}).items() if k != 'random_means'}, 'signal_family_analysis': {k:v for k,v in output.get('signal_family_analysis',{}).items() if k != 'random_means'}, 'verdict': output['verdict'], 'causality': causality}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

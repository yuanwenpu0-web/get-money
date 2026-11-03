"""独立离线研究：逐币滚动比较基线、订单流及波动状态特征。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'scripts'))
from app.db import make_engine
from app.ml.minute_research import HORIZON, data_fingerprint, make_features, make_labels
from app.ml.research_data import load_candles, validate_candles
from research_binary_15m import metrics, select_threshold
from train_major_models import SAFE_END_MS

MINUTE = 60000
VARIANTS = ('baseline', 'order_flow', 'volatility', 'combined')


def extra_features(frame):
    result = pd.DataFrame(index=frame.index)
    ret = frame.close.pct_change(fill_method=None)
    imbalance = 2 * frame.taker_buy_volume - frame.volume
    quote_imbalance = 2 * frame.taker_buy_quote_volume - frame.quote_volume
    for n in (15, 60, 240):
        result[f'flow_imbalance_{n}'] = imbalance.rolling(n).sum() / frame.volume.rolling(n).sum().replace(0, np.nan)
        result[f'flow_quote_imbalance_{n}'] = quote_imbalance.rolling(n).sum() / frame.quote_volume.rolling(n).sum().replace(0, np.nan)
        result[f'flow_activity_{n}'] = frame.trades.rolling(n).mean() / frame.trades.rolling(240).mean().replace(0, np.nan)
        result[f'vol_realized_{n}'] = np.sqrt(ret.pow(2).rolling(n).sum())
        result[f'vol_range_{n}'] = ((frame.high - frame.low) / frame.close).rolling(n).mean()
    result['vol_short_long_ratio'] = result['vol_realized_15'] / result['vol_realized_240'].replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def prepare_segment(frame):
    # 独立计算每段全部滚动及 EMA，禁止任何特征窗口跨分界。
    base = make_features(frame)
    extra = extra_features(frame)
    features = pd.concat([base, extra], axis=1)
    returns = make_labels(frame).future_return
    valid = features.notna().all(axis=1) & returns.notna() & (frame.open_time % (15 * MINUTE) == 0)
    x = features.loc[valid].astype(np.float32).reset_index(drop=True)
    r = returns.loc[valid].to_numpy()
    times = frame.loc[valid, 'open_time'].to_numpy(dtype=np.int64)
    return x, r, times, list(base.columns), list(extra.columns)


def split_frames(frame, fold):
    if fold not in (0, 1, 2):
        raise ValueError('折数必须为 0、1、2')
    # 40% 初始训练，随后的四个 15% 区块轮流作为验证和测试。
    n = len(frame)
    boundaries = [0, n * 40 // 100, n * 55 // 100, n * 70 // 100, n * 85 // 100, n]
    return tuple(frame.iloc[a:b].reset_index(drop=True) for a, b in
                 ((0, boundaries[fold + 1]), (boundaries[fold + 1], boundaries[fold + 2]),
                  (boundaries[fold + 2], boundaries[fold + 3])))


def segment_info(times, returns):
    return {'rows': int(len(times)), 'nonflat_rows': int(np.count_nonzero(returns)),
            'first_open_ms': int(times[0]), 'last_open_ms': int(times[-1]),
            'last_label_close_ms': int(times[-1] + (HORIZON + 1) * MINUTE - 1)}


def experiment(frame, trees=120, threads=3):
    params = dict(n_estimators=trees, learning_rate=.05, num_leaves=31, min_child_samples=100,
                  n_jobs=threads, random_state=42, verbosity=-1, deterministic=True, force_col_wise=True)
    folds = []
    for fold in range(3):
        segments = split_frames(frame, fold)
        prepared = [prepare_segment(s) for s in segments]
        if min(len(p[1]) for p in prepared) < 100:
            raise ValueError('隔离和特征预热后有效样本不足')
        for left, right in zip(prepared, prepared[1:]):
            if left[2][-1] + (HORIZON + 1) * MINUTE > right[2][0]:
                raise ValueError('标签跨越分区')
        xtr, rtr, _, base, extra = prepared[0]
        xva, rva, _, _, _ = prepared[1]
        xte, rte, _, _, _ = prepared[2]
        train_mask, val_mask, test_mask = rtr != 0, rva != 0, rte != 0
        ytr, yva, yte = (rtr[train_mask] > 0).astype(int), (rva[val_mask] > 0).astype(int), (rte[test_mask] > 0).astype(int)
        if min(len(ytr), len(yva), len(yte)) < 100 or len(np.unique(ytr)) != 2:
            raise ValueError('非零样本或训练类别不足')
        flow = [c for c in extra if c.startswith('flow_')]
        volatility = [c for c in extra if c.startswith('vol_')]
        groups = {'baseline': base, 'order_flow': base + flow, 'volatility': base + volatility, 'combined': base + extra}
        results, predictions = {}, {}
        for name in VARIANTS:
            cols = groups[name]
            model = lgb.LGBMClassifier(objective='binary', **params)
            model.fit(xtr.loc[train_mask, cols], ytr)
            pv = model.predict_proba(xva.loc[val_mask, cols])[:, 1]
            threshold = select_threshold(yva, pv)
            pt = model.predict_proba(xte.loc[test_mask, cols])[:, 1]
            pred = (pt >= .5).astype(int)
            chosen = np.maximum(pt, 1 - pt) >= threshold
            full = metrics(yte, pred, np.ones(len(yte), dtype=bool))
            filtered = metrics(yte, pred, chosen)
            full['correct'] = int(np.sum(pred == yte))
            filtered['correct'] = int(np.sum((pred == yte) & chosen))
            results[name] = {'threshold': threshold,
                             'validation_selected': metrics(yva, (pv >= .5).astype(int), np.maximum(pv, 1-pv) >= threshold),
                             'test_all': full, 'test_filtered': filtered}
            predictions[name] = (pred, chosen)
        paired = {}
        for name in VARIANTS[1:]:
            p0, m0 = predictions['baseline']
            p1, m1 = predictions[name]
            shared = m0 & m1
            paired[name] = {'shared_selected': int(shared.sum()), 'shared_coverage': float(shared.mean()),
                            'baseline_accuracy': float(np.mean(p0[shared] == yte[shared])) if shared.any() else None,
                            'variant_accuracy': float(np.mean(p1[shared] == yte[shared])) if shared.any() else None}
        folds.append({'fold': fold + 1, 'segments': {name: segment_info(p[2], p[1]) for name, p in zip(('train', 'validation', 'test'), prepared)},
                      'results': results, 'paired_filtered_intersection': paired})
        print(f'  第 {fold + 1} 折完成', flush=True)
    aggregate = {}
    for name in VARIANTS:
        aggregate[name] = {}
        for kind in ('test_all', 'test_filtered'):
            items = [f['results'][name][kind] for f in folds]
            total, selected, correct = (sum(i[k] for i in items) for k in ('total', 'selected', 'correct'))
            aggregate[name][kind] = {'total': total, 'selected': selected, 'correct': correct,
                                     'coverage': selected / total, 'accuracy': correct / selected if selected else None}
    return {'fingerprint_sha256': data_fingerprint(frame), 'model_parameters': params,
            'feature_names': groups, 'folds': folds, 'aggregate': aggregate}


def utc(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M UTC')


def pct(value):
    return '无' if value is None else f'{value * 100:.2f}%'


def markdown(results, end_ms):
    lines = ['# 15 分钟方向：滚动特征对照实验', '',
             f'截止时点：{utc(end_ms)}（不含该分钟）；数据源：本地 1 分钟 K 线，只读。',
             '标签为当前分钟收盘到未来第 15 根分钟 K 线收盘的收益方向；零收益从二分类训练、验证和测试排除。每 15 分钟按 UTC 开盘时间取样，特征在该分钟收盘后可用。',
             '每币独立三折：40% 初始训练，后续四个 15% 连续区块依次组成验证与测试；训练窗口扩展。各段独立计算滚动窗口及 EMA，首 240 分钟及缺失特征样本自然丢弃；各段尾部无完整未来标签的样本丢弃。',
             '所有特征组以全部特征共同有效的相同时点训练、验证和测试。基线已有单分钟主动买入比例；订单流为 15/60/240 分钟主动买卖量和成交额不平衡、成交笔数状态；波动状态为对应窗口实现波动、平均振幅及短长波动比。',
             '四种模型参数相同；每折各模型仅用此前验证集选择置信度阈值：0.500 至 0.800，步长 0.025，覆盖至少 50%，准确率最大，相同时覆盖率优先。测试期间不选择模型或阈值。',
             '覆盖率分母是该测试折的非零收益共同有效样本；过滤结果的模型各自选中时点可不同，同样本全部预测和共同选中交集另列。汇总为各折样本数加权，而非折平均。', '']
    for symbol, result in results.items():
        lines += [f'## {symbol}', '']
        if result['status'] != 'ok':
            lines += [f"运行失败：{result['error']}", '']
            continue
        q, exp = result['quality'], result['experiment']
        lines += [f"原始数据：{q['closed_rows']:,} 根，{utc(q['first_open_time_ms'])} 至 {utc(q['last_open_time_ms'])}；缺失分钟 {q['missing_minutes']}，无效行 {q['invalid_rows']}，重复 {q['duplicate_count']}。",
                  f"数据 SHA-256：`{exp['fingerprint_sha256']}`。", '']
        for fold in exp['folds']:
            lines += [f"### 第 {fold['fold']} 折", '']
            for part, info in fold['segments'].items():
                lines.append(f"- {part}：{utc(info['first_open_ms'])} 至 {utc(info['last_open_ms'])}，有效 {info['rows']:,}，非零 {info['nonflat_rows']:,}；末标签收盘 {utc(info['last_label_close_ms'])}。")
            lines += ['', '| 特征 | 阈值 | 验证筛选准确率/覆盖率 | 测试全部准确率 | 测试筛选准确率 | 测试覆盖率 | 选中/总数 |', '|---|---:|---:|---:|---:|---:|---:|']
            for name, r in fold['results'].items():
                a, b, v = r['test_all'], r['test_filtered'], r['validation_selected']
                lines.append(f"| {name} | {r['threshold']:.3f} | {pct(v['accuracy'])} / {pct(v['coverage'])} | {pct(a['accuracy'])} | {pct(b['accuracy'])} | {pct(b['coverage'])} | {b['selected']}/{b['total']} |")
            lines += ['', '筛选交集上的基线 / 扩展准确率：']
            for name, p in fold['paired_filtered_intersection'].items():
                lines.append(f"- {name}：{p['shared_selected']} 条，覆盖 {pct(p['shared_coverage'])}，{pct(p['baseline_accuracy'])} / {pct(p['variant_accuracy'])}。")
            lines.append('')
        lines += ['### 三折汇总', '', '| 特征 | 全部准确率 | 筛选准确率 | 筛选覆盖率 | 筛选数/总数 |', '|---|---:|---:|---:|---:|']
        for name, item in exp['aggregate'].items():
            a, b = item['test_all'], item['test_filtered']
            lines.append(f"| {name} | {pct(a['accuracy'])} | {pct(b['accuracy'])} | {pct(b['coverage'])} | {b['selected']}/{b['total']} |")
        lines.append('')
    lines += ['## 数据与解释限制', '', '数据质量检查只覆盖本地已有时间范围，无法证明上市以来或范围外数据完整；本实验依赖 K 线主动买入字段，不能反映逐笔盘口、挂单、撤单或交易方向以外的完整订单流。',
              '各折测试期不重叠，但后续折训练可使用先前测试期已到期数据；不能把各折视为相互独立的市场实验。模型筛选覆盖率在测试期可能偏离验证期目标。',
              '准确率不包含费用、滑点、资金费率或成交约束，不等同于交易收益；多币种、多特征和多折比较存在随机波动和多重比较问题。结果仅为离线研究，不接交易，不保存或覆盖模型。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT', 'XRPUSDT'])
    parser.add_argument('--end-ms', type=int, default=SAFE_END_MS)
    parser.add_argument('--trees', type=int, default=120)
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    if args.end_ms > SAFE_END_MS or args.trees < 1 or args.threads < 1:
        parser.error('截止时间超出核查范围或参数无效')
    run = ROOT / 'data' / 'model_research' / ('rolling_features_15m_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(parents=True, exist_ok=False)
    engine = make_engine(read_only=True, null_pool=True)
    results = {}
    try:
        for symbol in args.symbols:
            print(f'{symbol}: 读取并核验数据', flush=True)
            try:
                frame = load_candles(engine, symbol, args.end_ms)
                frame, quality = validate_candles(frame, symbol, args.end_ms)
                if not quality['passed']:
                    results[symbol] = {'status': 'failed', 'error': '数据质量检查未通过', 'quality': quality}
                    continue
                print(f'{symbol}: {len(frame)} 根，开始滚动实验', flush=True)
                result = experiment(frame, args.trees, args.threads)
                results[symbol] = {'status': 'ok', 'quality': quality, 'experiment': result}
            except Exception as exc:
                traceback.print_exc()
                results[symbol] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            finally:
                (run / 'report.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
                (run / 'report.md').write_text(markdown(results, args.end_ms), encoding='utf-8')
    finally:
        engine.dispose()
    print(f'报告目录: {run}', flush=True)
    return int(any(r['status'] != 'ok' for r in results.values()))


if __name__ == '__main__':
    raise SystemExit(main())

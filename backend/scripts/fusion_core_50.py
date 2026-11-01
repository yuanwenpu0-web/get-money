"""Fusion Core v3：固定币池的本地历史回测。"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.catalog_indicators.capital41_fusion_oscillator import capital41_fusion_oscillator
from app.db import make_engine
from app.research.lab_sim import to_ms

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/pine-crypto-collection/capital41_btc_general/Capital41_Fusion Strategy/Capital41_Fusion_Core_Strategy_v3.pine'
OUT = ROOT / 'docs/fusion-core-50'
BAR = 3_600_000
START = int(pd.Timestamp('2025-09-27T00:00:00Z').timestamp() * 1000)
SPLIT = int(pd.Timestamp('2026-07-01T00:00:00Z').timestamp() * 1000)
END = int(pd.Timestamp('2026-10-26T13:00:00Z').timestamp() * 1000)
FEE = 0.00075
EXCLUDED = {'BTCUSDT', 'ETHUSDT', 'USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'USDPUSDT', 'DAIUSDT', 'EURUSDT', 'GBPUSDT', 'PAXGUSDT'}


def signals(frame):
    ind = capital41_fusion_oscillator(frame)
    long = ind.trend_long & (ind.fusion_smoothed >= 75) & ind.expansion_ok
    short = ind.trend_short & (ind.fusion_smoothed <= 25) & ind.expansion_ok
    return pd.DataFrame({'long': long & ~long.shift(1, fill_value=False),
                         'short': short & ~short.shift(1, fill_value=False),
                         'atr': ind.atr, 'ema': ind.ema, 'osc': ind.fusion_smoothed}, index=frame.index)


def simulate(frame, sig, start_index=200):
    """市价下一开盘，止损单于持仓首根收盘后激活；每笔默认一单位。"""
    trades, pos, pending, bracket = [], None, None, None
    ignored, resets, ambiguous = 0, 0, 0
    last_atr = None
    arr = frame[['open', 'high', 'low', 'close']].to_numpy(float)
    stamps = to_ms(frame.index)

    def close_position(i, price, reason):
        nonlocal pos, bracket
        pnl = pos['side'] * (price - pos['price']) - FEE * (pos['price'] + price)
        trades.append({'side': pos['side'], 'entry_ms': int(stamps[pos['index']]), 'exit_ms': int(stamps[i]),
                       'entry_price': pos['price'], 'exit_price': float(price), 'pnl': float(pnl),
                       'fees': float(FEE * (pos['price'] + price)), 'reason': reason})
        pos, bracket = None, None

    for i, (op, hi, lo, cl) in enumerate(arr):
        if pending:
            kind, side = pending
            if kind == 'exit':
                if pos and pos['side'] == side:
                    close_position(i, op, 'core_exit')
            elif pos is None or pos['side'] != side:
                if pos:
                    close_position(i, op, 'reversal')
                pos = {'side': side, 'price': float(op), 'index': i}
            else:
                ignored += 1
            pending = None
        if pos and bracket:
            stop, target = bracket
            side = pos['side']
            if (op <= stop if side == 1 else op >= stop):
                close_position(i, op, 'stop_gap')
            elif (op >= target if side == 1 else op <= target):
                close_position(i, op, 'target_gap')
            else:
                hit_stop = lo <= stop if side == 1 else hi >= stop
                hit_target = hi >= target if side == 1 else lo <= target
                if hit_stop and hit_target:
                    ambiguous += 1
                    # TradingView 默认 OHLC 路径：先到更近的极值；相等时保守取止损。
                    high_first = hi - op < op - lo
                    hit_stop_first = (not high_first) if side == 1 else high_first
                    close_position(i, stop if hit_stop_first else target, 'both_stop' if hit_stop_first else 'both_target')
                elif hit_stop:
                    close_position(i, stop, 'stop')
                elif hit_target:
                    close_position(i, target, 'target')
        if i < start_index:
            continue
        row = sig.iloc[i]
        direction = 1 if row['long'] else -1 if row['short'] else 0
        if direction:
            if pos and pos['side'] == direction:
                resets += 1
            last_atr = float(row['atr'])
            pending = ('entry', direction)
        if pos:
            if (pos['side'] == 1 and (row['osc'] < 50 or cl < row['ema'])) or (pos['side'] == -1 and (row['osc'] > 50 or cl > row['ema'])):
                # Pine 中同一收盘同时发反向 entry 和 close 的订单顺序有平台依赖。
                if not pending:
                    pending = ('exit', pos['side'])
            if last_atr is not None and np.isfinite(last_atr):
                dist = last_atr * 1.8
                bracket = (pos['price'] - pos['side'] * dist, pos['price'] + pos['side'] * dist * 1.5)
    mark = None
    if pos:
        mark = {'side': pos['side'], 'entry_ms': int(stamps[pos['index']]), 'entry_price': pos['price'],
                'mark_price': float(arr[-1, 3]), 'unrealized_after_entry_fee': float(pos['side'] * (arr[-1, 3] - pos['price']) - FEE * pos['price'])}
    return {'trades': trades, 'open_position': mark, 'pending_at_end': pending,
            'ignored_same_side_orders': ignored, 'same_side_atr_resets': resets, 'both_touched': ambiguous}


def iso(ms):
    return pd.to_datetime(ms, unit='ms', utc=True).isoformat()


def add_analysis(payload):
    results = payload['results']
    trades = [t for r in results for t in r['trades']]
    for t in trades:
        t['net_return_on_entry_notional'] = t['pnl'] / t['entry_price']

    def stats(ts):
        returns = [t['net_return_on_entry_notional'] for t in ts]
        return {'trades': len(ts), 'wins': sum(x > 0 for x in returns),
                'win_rate': sum(x > 0 for x in returns) / len(ts) if ts else None,
                'sum_net_returns': sum(returns),
                'mean_net_return': sum(returns) / len(ts) if ts else None}

    post = [t for t in trades if t['exit_ms'] >= SPLIT]
    post_entered = [t for t in post if t['entry_ms'] >= SPLIT]
    duration = (END - SPLIT) // BAR
    coverage = []
    for r in results:
        last = int(pd.Timestamp(r['last_open_utc']).timestamp() * 1000)
        bars = max(0, (min(END, last + BAR) - SPLIT) // BAR)
        coverage.append({'symbol': r['symbol'], 'post_bars': bars,
                         'post_coverage_fraction': bars / duration,
                         'first_missing_utc': r['first_missing_utc'],
                         'last_open_utc': r['last_open_utc'],
                         'full_post_coverage': bars == duration})
    payload['normalized_analysis'] = {
        'definition': '逐笔净收益率 = pnl / entry_price = side × (exit_price - entry_price) / entry_price - 0.00075 × (entry_price + exit_price) / entry_price；side 多=1、空=-1，数量=1。',
        'aggregation': '收益率求和和逐笔算术均值为交易样本统计，非复利、年化或组合收益；不模拟资金占用、重叠持仓及仓位约束。',
        'overall': stats(trades),
        'by_side': {name: stats([t for t in trades if t['side'] == side]) for name, side in [('long', 1), ('short', -1)]},
        'post_by_exit': stats(post),
        'post_by_exit_and_side': {name: stats([t for t in post if t['side'] == side]) for name, side in [('long', 1), ('short', -1)]},
        'post_entry_and_exit': stats(post_entered),
        'post_cross_boundary': stats([t for t in post if t['entry_ms'] < SPLIT]),
        'by_symbol': {r['symbol']: {'overall': stats(r['trades']),
                                    'post_by_exit': stats([t for t in r['trades'] if t['exit_ms'] >= SPLIT])} for r in results},
        'coverage': {'expected_post_bars_per_symbol': duration,
                     'full_coverage_symbols': sum(x['full_post_coverage'] for x in coverage),
                     'observed_post_symbol_bars': sum(x['post_bars'] for x in coverage),
                     'expected_post_symbol_bars': duration * len(results),
                     'symbols': coverage}}
    return payload


def main():
    db = sessionmaker(bind=make_engine(read_only=True))()
    try:
        rows = db.execute(text("select symbol, count(*) n, sum(quote_volume) q from klines where interval='1h' and open_time>=:s and open_time<:e group by symbol having count(*)>=:n order by q desc, symbol"),
                          {'s': START, 'e': SPLIT, 'n': (SPLIT-START)//BAR}).all()
        chosen = [r[0] for r in rows if r[0].endswith('USDT') and r[0] not in EXCLUDED and not r[0].endswith(('UPUSDT', 'DOWNUSDT', 'BULLUSDT', 'BEARUSDT'))][:50]
        if len(chosen) != 50:
            raise RuntimeError(f'合格币不足50: {len(chosen)}')
        results = []
        for symbol in chosen:
            rows = db.execute(text("select open_time, open, high, low, close, volume from klines where symbol=:symbol and interval='1h' and open_time>=:s and open_time<:e order by open_time"), {'symbol': symbol, 's': START, 'e': END}).mappings().all()
            frame = pd.DataFrame([dict(row) for row in rows])
            stamps = frame.open_time.to_numpy(dtype=np.int64)
            if len(stamps) < 201 or stamps[0] != START:
                raise RuntimeError(f'{symbol}: 起始数据不足')
            breaks = np.flatnonzero(np.diff(stamps) != BAR)
            if len(breaks) and stamps[breaks[0]] < SPLIT - BAR:
                raise RuntimeError(f'{symbol}: 币池选择期内存在缺口')
            gap_at = int(stamps[breaks[0]] + BAR) if len(breaks) else None
            if len(breaks):
                frame = frame.iloc[:breaks[0] + 1].copy()
            frame.index = pd.to_datetime(frame.pop('open_time'), unit='ms', utc=True)
            frame = frame.astype(float)
            values = frame.to_numpy()
            if not np.isfinite(values).all() or (frame[['open', 'high', 'low', 'close']] <= 0).any().any() or (frame.volume < 0).any() or (frame.high < frame[['open', 'close', 'low']].max(axis=1)).any() or (frame.low > frame[['open', 'close', 'high']].min(axis=1)).any():
                raise RuntimeError(f'{symbol}: 无效 OHLCV')
            result = simulate(frame, signals(frame))
            result.update({'symbol': symbol, 'bars': len(frame), 'last_open_utc': frame.index[-1].isoformat(),
                           'first_missing_utc': iso(gap_at) if gap_at else None,
                           'missing_tail_bars': int((END - BAR - to_ms(frame.index)[-1]) // BAR),
                           'zero_volume_bars': int((frame.volume == 0).sum())})
            results.append(result)
            print(f'{len(results)}/50 {symbol}: {len(result["trades"])} 笔', flush=True)
    finally:
        db.close()
    trades = [t for r in results for t in r['trades']]
    def summary(ts):
        return {'trades': len(ts), 'pnl_units_quote': sum(t['pnl'] for t in ts), 'fees_quote': sum(t['fees'] for t in ts),
                'wins': sum(t['pnl'] > 0 for t in ts), 'reasons': dict(Counter(t['reason'] for t in ts))}
    payload = {'config': {'pine_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'source': str(SOURCE.relative_to(ROOT)),
        'data_source': '项目配置数据库 klines 原生 1h；上游交易所未逐条核验', 'start_utc': iso(START), 'selection_end_utc': iso(SPLIT), 'end_exclusive_utc': iso(END),
        'universe': '选择期完整1h、成交额降序前50；排除BTC、ETH、指定稳定币及杠杆代币，完整名单见 results',
        'warmup_bars': 200, 'commission_per_side': FEE, 'slippage': 0, 'quantity': '每个独立标的每单1单位；不模拟组合资金限制',
        'execution': '收盘信号下根开盘市价；持仓首根收盘后挂止盈止损，次根生效；缺口开盘成交；同K触及按开盘近端极值先到，相等取止损；末尾持仓不强平',
        'limitations': ['1h替代Pine建议30m；无TradingView逐笔对照', '不模拟最小价格步长、最小数量、保证金、强平、资金费率、盘口、冲击和滑点', '同收盘反向开仓与核心平仓同时发出时，采用反手优先，平台订单冲突语义未经验证', '同向新信号即使入场被金字塔限制也更新ATR，按源码执行；可能改变原持仓止盈止损', '原脚本默认数量为1单位，各币面额不一，总损益不代表等权组合收益', '按后续选择期成交额筛选币池，存在选择及幸存者偏差；本次期间不是独立盲测', '缺口后截断，不对缺口后的成交做假设；末尾挂单和持仓不虚构成交']},
        'summary': summary(trades), 'pre_selection_end_by_exit': summary([t for t in trades if t['exit_ms'] < SPLIT]),
        'post_selection_end_by_exit': summary([t for t in trades if t['exit_ms'] >= SPLIT]), 'results': results}
    write_report(payload)


def write_report(payload):
    add_analysis(payload)
    results = payload['results']
    OUT.with_suffix('.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    s = payload['summary']
    lines = ['# Fusion Core v3 50 山寨币回测', '', f"数据：{payload['config']['data_source']}，UTC [{iso(START)}, {iso(END)})；前200根预热。原脚本建议30分钟，此处使用覆盖更长的原生1小时数据，未验证30分钟表现。", '',
        f"币池以 [{iso(START)}, {iso(SPLIT)}) 完整K线成交额排序选择50币；这是事后固定币池，选择期结果具有前视选择偏差，后段也非全项目独立盲测。完整币名单与逐笔交易见JSON。", '',
        f"共 {s['trades']} 笔已平仓，盈利笔数 {s['wins']}，一单位口径已实现损益 {s['pnl_units_quote']:.6f} 报价货币，手续费 {s['fees_quote']:.6f}。不同币单价差异很大，汇总损益不是等权组合收益。",
        f"选择期后按平仓时间统计：{payload['post_selection_end_by_exit']['trades']} 笔，损益 {payload['post_selection_end_by_exit']['pnl_units_quote']:.6f}；跨界交易未剔除。", '',
        f"缺口或提前结束币数：{sum(r['missing_tail_bars'] > 0 for r in results)}；未平仓数：{sum(r['open_position'] is not None for r in results)}；同K止盈止损双触及：{sum(r['both_touched'] for r in results)}。逐币缺口、持仓及明细见JSON。", '',
        'Pine默认参数：EMA50、RSI14（35/65）、W%R14（-80/-20）、ATR14/均值20 ×1.1、量均50 ×1、震荡器平滑3、阈值75/25、出场中线50、止损1.8 ATR、止盈1.5R；入场信号锁定ATR但持仓期间同向新信号也会刷新。',
        '市价在信号收盘后下一根开盘成交；止盈止损在持仓首根收盘提交、下一根开始有效。每单1单位、单边手续费0.075%、滑点0；跳空按开盘成交，同K双触及按开盘距离较近的高低点先到（等距取止损）。不将末尾持仓强平。']
    a = payload['normalized_analysis']
    c = a['coverage']
    def fmt(x):
        return f"{x['trades']} | {x['win_rate']:.2%} | {x['mean_net_return']:.4%} | {x['sum_net_returns']:.4%}" if x['trades'] else '0 | — | — | 0.0000%'
    lines += ['', '## 入场名义金额标准化的已平仓交易', '', a['definition'], '', a['aggregation'],
              '胜率为净收益率大于零的笔数除以总笔数。后段以平仓K线开盘时间 ≥ 2026-07-01 00:00 UTC 为准；跨界交易的入场发生在该时点之前。', '',
              '| 分组 | 笔数 | 胜率 | 平均每笔净收益率 | 净收益率之和 |', '|---|---:|---:|---:|---:|']
    for name, group in [('全部', a['overall']), ('多', a['by_side']['long']), ('空', a['by_side']['short']),
                        ('后段按平仓', a['post_by_exit']), ('后段多', a['post_by_exit_and_side']['long']),
                        ('后段空', a['post_by_exit_and_side']['short']), ('后段入场且后段平仓', a['post_entry_and_exit']),
                        ('跨界入场、后段平仓', a['post_cross_boundary'])]:
        lines.append(f'| {name} | {fmt(group)} |')
    lines += ['', '## 按币统计', '', '| 币种 | 全期笔数 | 全期平均每笔净收益率 | 后段平仓笔数 | 后段平均每笔净收益率 | 后段覆盖 |', '|---|---:|---:|---:|---:|---:|']
    by_coverage = {x['symbol']: x for x in c['symbols']}
    for symbol, groups in a['by_symbol'].items():
        overall, post = groups['overall'], groups['post_by_exit']
        avg = lambda x: f"{x['mean_net_return']:.4%}" if x['trades'] else '—'
        lines.append(f"| {symbol} | {overall['trades']} | {avg(overall)} | {post['trades']} | {avg(post)} | {by_coverage[symbol]['post_coverage_fraction']:.2%} |")
    lines += ['', '## 后段数据覆盖', '',
              f"后段每币应有 {c['expected_post_bars_per_symbol']} 根；完整覆盖 {c['full_coverage_symbols']}/{len(results)} 币，实际覆盖 {c['observed_post_symbol_bars']}/{c['expected_post_symbol_bars']} 币小时（{c['observed_post_symbol_bars']/c['expected_post_symbol_bars']:.2%}）。覆盖只计首个缺口之前的连续K线；提前结束同样减少覆盖。截断后不生成交易，因此后段统计并非全币、全时段均衡样本，不能推断缺失期间表现。", '',
              '| 未完整覆盖币种 | 首次缺口 UTC（无则提前结束） | 最后K线开盘 UTC | 后段覆盖根数 | 覆盖率 |', '|---|---|---|---:|---:|']
    for x in c['symbols']:
        if not x['full_post_coverage']:
            lines.append(f"| {x['symbol']} | {x['first_missing_utc'] or '—'} | {x['last_open_utc']} | {x['post_bars']} | {x['post_coverage_fraction']:.2%} |")
    lines += ['', '## 时序核查', '', '指标使用当前及历史K线；信号在收盘生成、下一根开盘成交，止盈止损在提交后下一根生效。逐笔标准化只用于平仓后的事后统计，不参与信号或选币。原有前缀因果测试及下一根成交测试可检查实现；这些检查不消除选择期事后筛选币池的前视选择偏差。', '', '## 限制', '']
    lines += ['- ' + x for x in payload['config']['limitations']]
    lines += ['', '运行：`./.venv/Scripts/python.exe backend/scripts/fusion_core_50.py`。', '详细参数、原始脚本 SHA256、逐币质量和交易记录见同名 JSON。']
    OUT.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

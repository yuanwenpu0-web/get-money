"""固定币池、统一参数网格的 50 山寨币研究；仅使用项目已有信号与撮合器。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db import make_engine
from app.research import lab_experiment, lab_grid, lab_combos
from app.research.lab_signals import Signals
from app.research.lab_sim import SimConfig, exit_signals, simulate, to_ms

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'altcoin-50-independent'
BAR = 3_600_000
END = int(pd.Timestamp('2026-10-26T12:00:00Z').timestamp() * 1000) + BAR
START = int(pd.Timestamp('2025-09-27T00:00:00Z').timestamp() * 1000)
SPLIT = int(pd.Timestamp('2026-07-01T00:00:00Z').timestamp() * 1000)
WARMUP = 200
EXCLUDED = {'BTCUSDT', 'ETHUSDT', 'USDCUSDT', 'FDUSDUSDT', 'TUSDUSDT', 'USDPUSDT', 'DAIUSDT', 'EURUSDT', 'GBPUSDT', 'PAXGUSDT'}
CFG = SimConfig(stake=200, fee=0.0004, slippage=0.0002, max_hold_bars=240, side=1)


def iso(ms):
    return pd.to_datetime(ms, unit='ms', utc=True).isoformat()


def metrics(trades):
    ordered = sorted(trades, key=lambda t: t.exit_ts)
    pnl = float(sum(t.pnl for t in ordered))
    eq = np.r_[10000., 10000. + np.cumsum([t.pnl for t in ordered])]
    dd = float(np.min(eq / np.maximum.accumulate(eq) - 1) * 100)
    return {'trades': len(trades), 'pnl': round(pnl, 4), 'return_pct': round(pnl / 100, 4),
            'realized_drawdown_pct': round(-dd, 4),
            'win_rate_pct': round(100 * sum(t.pnl > 0 for t in trades) / len(trades), 2) if trades else None,
            'fees': round(sum(t.fees for t in trades), 4)}


def segment(frame, sig, start, end, symbol):
    stamps = to_ms(frame.index)
    stop = int(np.searchsorted(stamps, end))
    # 数据只保留该段结束之前，指标在完整历史上生成；只允许段内成交。
    f = frame.iloc[:stop]
    allowed = (stamps[:stop] >= start) & (stamps[:stop] < end - BAR)
    allowed[:min(WARMUP, stop)] = False
    masked = Signals(sig.long[:stop] & allowed, np.zeros(stop, dtype=bool))
    trades = simulate(symbol, f, masked, exit_signals(f['close'].to_numpy(float)), CFG)
    assert all(start <= t.entry_ts < end and start <= t.exit_ts < end for t in trades)
    return trades


def run(signals, frames):
    train, test = [], []
    by_symbol = []
    for symbol, frame in frames.items():
        a = segment(frame, signals[symbol], START, SPLIT, symbol)
        b = segment(frame, signals[symbol], SPLIT, END, symbol)
        train.extend(a); test.extend(b)
        by_symbol.append({'symbol': symbol, 'train_pnl': round(sum(t.pnl for t in a), 4),
                          'test_pnl': round(sum(t.pnl for t in b), 4), 'test_trades': len(b)})
    return {'train': metrics(train), 'test': metrics(test), 'by_symbol': by_symbol}


def main():
    t0 = time.time()
    db = sessionmaker(bind=make_engine(read_only=True))()
    rows = db.execute(text("select symbol, count(*) n, sum(quote_volume) q from klines where interval='1h' and open_time>=:s and open_time<:e group by symbol having count(*)>=:n order by q desc, symbol"),
                      {'s': START, 'e': SPLIT, 'n': (SPLIT-START)//BAR}).all()
    chosen = [r[0] for r in rows if r[0].endswith('USDT') and r[0] not in EXCLUDED and not r[0].endswith(('UPUSDT','DOWNUSDT','BULLUSDT','BEARUSDT'))][:50]
    if len(chosen) != 50:
        raise RuntimeError(f'训练期合格币不足50: {len(chosen)}')
    frames = {}
    quality = []
    for symbol in chosen:
        frame = lab_experiment._frame(db, symbol, '1h', START)
        frame = frame[(to_ms(frame.index) >= START) & (to_ms(frame.index) < END)]
        stamps = to_ms(frame.index)
        if len(stamps) == 0 or stamps[0] != START:
            raise RuntimeError(f'{symbol} 缺少起始行情')
        breaks = np.flatnonzero(np.diff(stamps) != BAR)
        if len(breaks):
            cut = int(breaks[0] + 1)
            if stamps[cut - 1] < SPLIT:
                raise RuntimeError(f'{symbol} 训练期行情存在缺口')
            frame = frame.iloc[:cut]
            stamps = stamps[:cut]
        if stamps[-1] < SPLIT:
            raise RuntimeError(f'{symbol} 样本外无数据')
        if not np.isfinite(frame[['open','high','low','close','volume']].to_numpy()).all() or (frame[['open','high','low','close']] <= 0).any().any():
            raise RuntimeError(f'{symbol} 行情数据无效')
        frames[symbol] = frame
        quality.append({'symbol': symbol, 'bars': len(frame), 'last_open_utc': iso(int(stamps[-1])), 'missing_tail_bars': int((END - BAR - stamps[-1]) // BAR), 'zero_volume_bars': int((frame.volume <= 0).sum())})
    print(f'已加载 {len(frames)} 币 × {(END-START)//BAR} 根，开始测试', flush=True)
    candidates = lab_grid.candidates()
    rows_out = []
    for i, cand in enumerate(candidates, 1):
        sigs = {s: lab_experiment.sign(cand, f) for s, f in frames.items()}
        row = {'family': cand.family, 'params': cand.params, 'key': cand.key, 'label': cand.label, **run(sigs, frames)}
        rows_out.append(row)
        if i % 10 == 0: print(f'指标参数 {i}/{len(candidates)}', flush=True)
    ranked = sorted(rows_out, key=lambda x: (-x['train']['pnl'], x['key']))
    selected = []
    for row in ranked:
        if row['train']['trades'] >= 30 and row['family'] not in [x['family'] for x in selected]:
            selected.append(row)
        if len(selected) == 4: break
    lookup = {c.key: c for c in candidates}
    cache = {r['key']: {s: lab_experiment.sign(lookup[r['key']], f) for s, f in frames.items()} for r in selected}
    combos = []
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            a, b = selected[i], selected[j]
            for rule in ('and', 'or'):
                sigs = {s: lab_combos.combine(cache[a['key']][s], cache[b['key']][s], rule) for s in frames}
                combos.append({'components': [a['label'], b['label']], 'rule': rule, **run(sigs, frames)})
    combos.sort(key=lambda x: -x['train']['pnl'])
    hold = []
    for symbol, frame in frames.items():
        f = frame[(to_ms(frame.index) >= SPLIT) & (to_ms(frame.index) < END)]
        entry = float(f.open.iloc[0]) * (1 + CFG.slippage)
        exit_ = float(f.close.iloc[-1]) * (1 - CFG.slippage)
        qty = CFG.stake / entry
        hold.append(qty * (exit_ - entry) - CFG.fee * (CFG.stake + qty * exit_))
    payload = {'config': {'source': '项目配置数据库 klines 表，native 1h；上游来源需另行核验', 'start_utc': iso(START),
        'split_utc': iso(SPLIT), 'end_exclusive_utc': iso(END), 'symbol_selection': '仅训练期完整K线，按训练期 quote_volume 降序；排除BTC、ETH、指定稳定币及杠杆代币后前50',
        'symbols': chosen, 'warmup_bars': WARMUP, 'fee_per_side': CFG.fee, 'slippage_per_side': CFG.slippage,
        'capital': 10000, 'stake_per_symbol': CFG.stake, 'side': 'long', 'max_hold_bars': CFG.max_hold_bars,
        'execution': '收盘信号、下一根开盘执行；各区间独立重置，区间最后收盘强平；最后一根不发起入场',
        'exit': 'RSI14下穿50或最多持有240根，区间结束收盘强平', 'candidate_count': len(candidates)},
        'quality': quality, 'grid': lab_grid.manifest(), 'singles': rows_out,
        'selected_training_top_distinct_families': [r['label'] for r in selected], 'combinations': combos,
        'buy_hold_oos': {'pnl': round(sum(hold),4), 'return_pct': round(sum(hold)/100,4)},
        'elapsed_seconds': round(time.time()-t0, 2)}
    OUT.with_suffix('.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    def line(r):
        a,b = r['train'],r['test']
        return f"{a['return_pct']:.2f}% | {b['return_pct']:.2f}% | {b['trades']} | {b['realized_drawdown_pct']:.2f}%"
    s = [f"# 50 山寨币统一指标参数与组合回测", '',
         f"数据：项目数据库 klines native 1h。UTC 训练 [{iso(START)}, {iso(SPLIT)})；样本外 [{iso(SPLIT)}, {iso(END)})。",
         f"币池：训练期完整数据、训练期成交额排序，排除 BTC、ETH、指定稳定币和杠杆币；共 {len(chosen)} 币，名单见 JSON。首次样本外缺口截断后逐小时连续；{sum(q['missing_tail_bars'] > 0 for q in quality)} 币提前结束，其中 {sum(q['missing_tail_bars'] >= 24*90-3 for q in quality)} 币缺少约90天，详见质量记录。",
         f"统一网格 {len(candidates)} 组、{len(lab_grid.GRID)} 个指标家族；仅做多，所有币使用同一套候选。训练期盈利排序选择四个不同家族，逐对 AND/OR 形成 {len(combos)} 组；组合本身按训练期排序。",
         '初始资金 10000，每币固定名义 200，不复利，空闲资金不使用；收盘信号下根开盘成交，RSI14 下穿50或240根超时退出，分段末收盘平仓。单边手续费0.04%、滑点0.02%，往返近似0.12%。',
         '', '## 训练期排名前五单指标', '', '| 指标和参数 | 训练收益 | 样本外收益 | 样本外交易数 | 已实现回撤 |', '|---|---:|---:|---:|---:|']
    for r in ranked[:5]: s.append(f"| {r['label']} | {line(r)} |")
    s += ['', '## 训练期排名前五组合', '', '| 规则 | 训练收益 | 样本外收益 | 样本外交易数 | 已实现回撤 |', '|---|---:|---:|---:|---:|']
    for r in combos[:5]: s.append(f"| {r['components'][0]} {r['rule'].upper()} {r['components'][1]} | {line(r)} |")
    s += ['', f"样本外同期等权买入持有收益 {sum(hold)/100:.2f}%（相同成本、每币200）。", '',
          '## 限制与解释', '',
          '- 训练期选出的候选在该次运行中使用后续时段验证；已有历史研究可能已经观察同一段行情，因此并非全项目未触碰的独立盲测。',
          '- 币池按训练期成交额筛选，并非逐时点历史成分；没有上市、退市和存续偏差完整数据库，存在样本选择偏差。样本外首次缺口后截断该币并按缺口前最后收盘平仓，不替换；可能影响比较。',
          '- 单边做多；未测试做空、资金费率、杠杆、强平、借币、交易深度和冲击，不能代表永续或可交易收益。',
          '- 最大回撤依据已平仓盈亏，不含持仓浮亏，可能低估真实回撤；固定本金而非复利。',
          '- 多次参数与组合搜索会产生选择偏差；样本外一次切分不足以证明统计显著性，未进行多重检验或独立性修正。',
          '- 只覆盖 JSON grid 所列15个指标家族，不代表项目所有指标或策略已测试。行情来源的上游交易所需进一步核实。',
          '', '运行命令：`.venv/Scripts/python.exe backend/scripts/altcoin_50_research.py`。',
          '完整结果、参数、逐币损益和数据质量见同名前缀 JSON。']
    OUT.with_suffix('.md').write_text('\n'.join(s)+'\n', encoding='utf-8')
    print(f'完成，输出 {OUT}.json / .md', flush=True)

if __name__ == '__main__':
    main()

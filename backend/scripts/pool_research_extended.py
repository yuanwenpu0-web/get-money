"""扩展币池独立研究；仅使用训练期筛选和分组。"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

import pool_research as base
from app.research.lab_combos import combine
from app.research.lab_experiment import _frame, sign
from app.research.lab_grid import candidates
from app.research.lab_sim import exit_signals, to_ms

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'pool-research-extended'
N = 100


def excluded(s):
    return not s.endswith('USDT') or s in base.EXCLUDED or s.endswith(('UPUSDT', 'DOWNUSDT', 'BULLUSDT', 'BEARUSDT'))


def main():
    db = sessionmaker(bind=base.make_engine(read_only=True))()
    expected = (base.SPLIT - base.START) // base.BAR
    rows = db.execute(text("select symbol,count(*) n,count(distinct open_time) d,min(open_time) first,max(open_time) last,sum(quote_volume) q from klines where interval='1h' and open_time>=:s and open_time<:e group by symbol order by symbol"), {'s': base.START, 'e': base.SPLIT}).mappings().all()
    audit = []
    for r in rows:
        reason = '非目标交易对或排除名单' if excluded(r['symbol']) else ('训练期小时数量或边界不完整' if r['n'] != expected or r['d'] != expected or r['first'] != base.START or r['last'] != base.SPLIT-base.BAR else None)
        audit.append({'symbol': r['symbol'], 'bars': r['n'], 'distinct_bars': r['d'], 'first': int(r['first']), 'last': int(r['last']), 'train_quote_volume': float(r['q']) if r['q'] is not None else None, 'excluded_reason': reason})
    ranked = sorted((r for r in audit if r['excluded_reason'] is None and r['train_quote_volume'] is not None), key=lambda r: (-r['train_quote_volume'], r['symbol']))
    if len(ranked) < N:
        raise RuntimeError(f'合格币不足 {N}: {len(ranked)}')
    chosen = [r['symbol'] for r in ranked[:N]]
    lookup = {c.label: c for c in candidates()}
    frames, features, quality, signals, exits = {}, {}, {}, {}, {}
    digest = hashlib.sha256()
    for idx, s in enumerate(chosen):
        f = _frame(db, s, '1h', base.START)
        f = f[(to_ms(f.index) >= base.START) & (to_ms(f.index) < base.END)]
        ts = to_ms(f.index)
        train = f[ts < base.SPLIT]
        if not np.array_equal(to_ms(train.index), np.arange(base.START, base.SPLIT, base.BAR)):
            raise RuntimeError(s + '训练期不连续')
        vals = f[['open', 'high', 'low', 'close', 'volume', 'quote_volume']].to_numpy()
        if not np.isfinite(vals).all() or (f[['open','high','low','close']] <= 0).any().any() or (f[['volume','quote_volume']] < 0).any().any() or (f.high < f[['open','low','close']].max(axis=1)).any() or (f.low > f[['open','high','close']].min(axis=1)).any():
            raise RuntimeError(s + '行情无效')
        digest.update(s.encode()); digest.update(ts.astype('<i8').tobytes()); digest.update(vals.astype('<f8').tobytes())
        frames[s] = f
        features[s] = {'train_quote_volume': float(train.quote_volume.sum()), 'train_hourly_log_vol': float(np.log(train.close).diff().std())}
        oos = ts[(ts >= base.SPLIT) & (ts < base.END)]
        missing = np.setdiff1d(np.arange(base.SPLIT, base.END, base.BAR), oos)
        gaps = np.flatnonzero(np.diff(ts) != base.BAR)
        quality[s] = {'observed_oos_bars': len(oos), 'expected_oos_bars': (base.END-base.SPLIT)//base.BAR, 'missing_oos_bars': len(missing), 'first_missing_utc': base.iso(missing[0]) if len(missing) else None, 'last_open_utc': base.iso(ts[-1]), 'missing_intervals_utc': []}
        if len(missing):
            starts = np.r_[0, np.flatnonzero(np.diff(missing) != base.BAR)+1]
            stops = np.r_[starts[1:]-1, len(missing)-1]
            quality[s]['missing_intervals_utc'] = [[base.iso(missing[a]), base.iso(missing[b]+base.BAR)] for a,b in zip(starts,stops)]
        valid = np.ones(len(f), dtype=bool); valid[:200] = False
        for g in gaps: valid[g+1:min(g+201,len(f))] = False
        signals[s] = {label: sign(lookup[label], f) for label in base.LABELS}
        signals[s]['valid'] = valid
        exits[s] = exit_signals(f.close.to_numpy())[0]
        if (idx+1) % 20 == 0: print('读取',idx+1,flush=True)
    db.close()
    vol_ranked = sorted(chosen, key=lambda s: (-features[s]['train_hourly_log_vol'], s))
    liquid = set(chosen[:N//2]); volatile = set(vol_ranked[:N//2])
    pools = {name: [s for s in chosen if (s in liquid) == liq and (s in volatile) == vol] for name,liq,vol in [('高成交额高波动',True,True),('高成交额低波动',True,False),('低成交额高波动',False,True),('低成交额低波动',False,False)]}
    configs = [{'a':a,'b':b,'combine':how,'exit':ex} for a,b in itertools.combinations(base.LABELS,2) for how in ('and','or') for ex in base.EXITS]
    def run(s, cfg, start, end, hold=False):
        sig = signals[s]
        ent = combine(sig[cfg['a']],sig[cfg['b']],cfg['combine']).long & sig['valid'] if cfg else np.zeros(len(frames[s]), dtype=bool)
        return base.execute(frames[s], ent, exits[s], start, end, cfg['exit'] if cfg else '', hold)
    training = {name: [] for name in pools}
    unified_training = []
    for i,cfg in enumerate(configs):
        by_symbol = {s:run(s,cfg,base.START,base.SPLIT) for s in chosen}
        for name,syms in pools.items(): training[name].append({'config':cfg,'metrics':base.aggregate({s:by_symbol[s] for s in syms},base.STAKE*len(syms))})
        unified_training.append({'config':cfg,'metrics':base.aggregate(by_symbol,base.STAKE*N)})
        print('训练候选',i+1,'/',len(configs),flush=True)
    def select(rows, minimum):
        valid = [r for r in rows if r['metrics']['closed_trades'] >= minimum]
        if not valid: raise RuntimeError('无足够交易次数的候选')
        return sorted(valid,key=lambda r:(-r['metrics']['pnl'],json.dumps(r['config'],sort_keys=True)))[0]
    selected = {name:select(training[name], max(1, int(np.ceil(1.2*len(syms))))) for name,syms in pools.items()}
    overall = select(unified_training,120)
    # 所有训练选择固定后才读取样本外收益用于计算。
    oos = {}
    for name,syms in pools.items():
        strat = {s:run(s,selected[name]['config'],base.SPLIT,base.END) for s in syms}
        hold = {s:run(s,None,base.SPLIT,base.END,True) for s in syms}
        oos[name] = {'strategy':base.aggregate(strat,base.STAKE*len(syms)), 'buy_hold':base.aggregate(hold,base.STAKE*len(syms)), 'strategy_by_symbol':strat, 'buy_hold_by_symbol':hold}
    uni = {s:run(s,overall['config'],base.SPLIT,base.END) for s in chosen}
    payload = {'config': {'source':'项目配置数据库 klines native 1h；上游交易所未逐条核实','start_utc':base.iso(base.START),'split_utc':base.iso(base.SPLIT),'end_exclusive_utc':base.iso(base.END),'symbols':N,'fee_per_side':base.FEE,'slippage_per_side':base.SLIP,'stake_per_symbol':base.STAKE,'capital':base.STAKE*N,'pool_definition':'训练期连续小时数据中成交额前100；前50高成交额；100币训练期小时对数收益标准差前50高波动，分别按symbol打破平手，交叉四池。','selection':'每池36项，训练期平仓至少ceil(1.2*币数)，按净损益排序，平手按配置JSON字典序；统一池至少120笔。','execution':'收盘信号下一实际K线开盘成交；缺口后200根禁止入场；缺口中挂单等待；最后预定小时收盘平仓，缺末尾按陈旧末价估值且不扣假想卖出成本。','scope':'仅做多，固定名义，不复利；无资金费率和深度。','candidate_labels':base.LABELS,'candidate_exits':base.EXITS,'input_sha256':digest.hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'base_script_sha256':hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()}, 'training_universe_audit':audit,'eligible_ranked':ranked,'pools':pools,'features':features,'quality':quality,'training_candidates':training,'unified_training_candidates':unified_training,'selected':selected,'oos':oos,'unified':{'selected_training':overall,'oos':base.aggregate(uni,base.STAKE*N),'by_symbol':uni}}
    OUT.with_suffix('.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines = ['# 扩展币池与组合卖出研究','',f"数据：项目配置数据库 `klines` native 1h，上游交易所未逐条核实；UTC 训练 [{base.iso(base.START)}, {base.iso(base.SPLIT)})，样本外 [{base.iso(base.SPLIT)}, {base.iso(base.END)})。样本外不是未曾观察过的盲测。",'',f'训练期数据库有 {len(rows)} 个标的，按交易对和完整小时过滤后 {len(ranked)} 个候选；训练期成交额前100为研究币池。完整标的、剔除原因、全部排名、特征和逐币缺失时段见 JSON。','', '按训练期成交额前50与小时对数收益标准差前50各分高低，交叉为四池；平手按 symbol 排序。波动率不表示市值或行业分类。所有分池依据均在样本外开始前已知，但完整训练期选样会抬高训练期可得性，且实际数据库本身可能带存续偏差。','', '复用上一轮四个入场信号，两两 AND/OR，配合 RSI50或240小时、48小时、120小时三种卖出，共36候选；每池训练至少每币1.2笔平仓（向上取整），按训练净损益选定，统一池至少120笔。每币200，单边手续费0.04%、滑点0.02%，闲置资金不计收益、不复利。信号收盘确认，下一实际小时开盘成交；缺口后200根禁止新入场，挂单等待，末小时收盘退出；缺末尾按最后价格估值且未计假想卖出成本。买入持有使用相同成本及时间。','', '| 币池 | 数量 | 训练选定入场与卖出 | 训练收益 | 样本外收益 | 样本外平仓/未平 | 买入持有 |','|---|---:|---|---:|---:|---:|---:|']
    for name,syms in pools.items():
        sel=selected[name]; c=sel['config']; a=oos[name]['strategy']; b=oos[name]['buy_hold']
        lines.append(f"| {name} | {len(syms)} | {c['a']} {c['combine'].upper()} {c['b']} / {c['exit']} | {sel['metrics']['return_pct']:.2f}% | {a['return_pct']:.2f}% | {a['closed_trades']}/{a['open_positions']} | {b['return_pct']:.2f}% |")
    total=sum(oos[n]['strategy']['pnl'] for n in pools)/(base.STAKE*N)*100
    bh=sum(oos[n]['buy_hold']['pnl'] for n in pools)/(base.STAKE*N)*100
    lines += ['',f"四池合计样本外 **{total:.2f}%**；同币等权买入持有 **{bh:.2f}%**；全池训练选一个统一策略样本外 **{payload['unified']['oos']['return_pct']:.2f}%**，配置 `{json.dumps(overall['config'],ensure_ascii=False)}`。所有收益除以每币200的固定资本，而非实际持仓平均资金。",'',f"样本外应有 {(base.END-base.SPLIT)//base.BAR} 根/币；完整 {sum(q['missing_oos_bars']==0 for q in quality.values())}/{N}，缺失 {sum(q['missing_oos_bars'] for q in quality.values())} 币小时。未平仓以陈旧末价估值，收益不代表可实现平仓价格。",'', '该研究是有限候选的训练期选择，并非全局最优。样本外时段及候选曾用于以往研究，存在研究者选择和多重检验偏差；单次切分不能证明稳健性。未核验退市、当时可交易性或交易对类型，稳定币及杠杆币名称过滤不保证穷尽；未计冲击、盘口、资金费率，不代表实盘或永续合约收益。未计算逐小时权益回撤和显著性。','', '复现：`./.venv/Scripts/python.exe backend/scripts/pool_research_extended.py`。需要相同依赖、数据库配置和历史数据快照；JSON 记录输入与脚本哈希。']
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('完成',OUT,flush=True)


if __name__ == '__main__':
    main()

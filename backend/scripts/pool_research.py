"""训练期成交额分池的独立回测；不修改项目策略系统。"""
from __future__ import annotations
import hashlib
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db import make_engine
from app.research.lab_experiment import _frame, sign
from app.research.lab_grid import candidates
from app.research.lab_sim import to_ms, exit_signals
from app.research.lab_combos import combine

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'pool-research'
BAR = 3600000
START = int(pd.Timestamp('2025-09-27', tz='UTC').timestamp()*1000)
SPLIT = int(pd.Timestamp('2026-07-01', tz='UTC').timestamp()*1000)
END = int(pd.Timestamp('2026-10-26T13:00:00Z').timestamp()*1000)
FEE, SLIP, STAKE, CAPITAL = .0004, .0002, 200., 10000.
EXCLUDED = {'BTCUSDT','ETHUSDT','USDCUSDT','FDUSDUSDT','TUSDUSDT','USDPUSDT','DAIUSDT','EURUSDT','GBPUSDT','PAXGUSDT'}
# 在此次运行前明确固定的候选，不依据样本外结果更改。
LABELS = ['volume_break(k=3.0,n=20)', 'rsi_reversal(high=70,low=30,n=21)',
          'bollinger_break(k=2.0,n=35)', 'atr_channel_break(k=2.0,n=20)']
EXITS = ('rsi50_240', 'hold48', 'hold120')


def iso(t):
    return pd.to_datetime(int(t), unit='ms', utc=True).isoformat()


def execute(f, entry, exit_rsi, begin, end, rule, hold=False):
    """缺失小时不成交；挂单保留到下一根实际K线开盘；末日无K线则按末价估值。"""
    ts = to_ms(f.index)
    start, stop = np.searchsorted(ts, [begin, end])
    if start == stop:
        return {'pnl': 0., 'trades': 0, 'fees': 0., 'open': False, 'events': [], 'unpriced': True}
    op, close = f.open.to_numpy(), f.close.to_numpy()
    qty, pending, entry_i = 0., None, -1
    cash, fees, trades, events = 0., 0., 0, []
    if hold:
        pending = 'buy'
    for i in range(start, stop):
        if pending == 'buy' and qty == 0:
            price = op[i] * (1 + SLIP)
            qty = STAKE / price
            fee = STAKE * FEE
            cash -= STAKE + fee; fees += fee; entry_i = i
            events.append((int(ts[i]), float(cash)))
        elif pending == 'sell' and qty:
            proceeds = qty * op[i] * (1 - SLIP)
            fee = proceeds * FEE
            cash += proceeds - fee; fees += fee; qty = 0.; trades += 1
            events.append((int(ts[i]), float(cash)))
        pending = None
        # 最后一小时为预定边界，统一在该小时收盘退出，不预知数据缺口。
        if ts[i] == end - BAR:
            if qty:
                proceeds = qty * close[i] * (1 - SLIP)
                fee = proceeds * FEE
                cash += proceeds - fee; fees += fee; qty = 0.; trades += 1
                events.append((int(end), float(cash)))
            break
        if not hold:
            if qty:
                elapsed = (ts[i] - ts[entry_i]) // BAR + 1
                if (rule == 'rsi50_240' and (exit_rsi[i] or elapsed >= 240)) or (rule == 'hold48' and elapsed >= 48) or (rule == 'hold120' and elapsed >= 120):
                    pending = 'sell'
            elif entry[i]:
                pending = 'buy'
    marked = cash + qty * close[stop-1]
    return {'pnl': float(marked), 'trades': trades, 'fees': float(fees), 'open': bool(qty),
            'events': events, 'unpriced': False, 'last_time': int(ts[stop-1]), 'mark_value': float(qty*close[stop-1])}


def aggregate(results, capital):
    pnl = sum(x['pnl'] for x in results.values())
    return {'pnl': pnl, 'return_pct': 100*pnl/capital, 'closed_trades': sum(x['trades'] for x in results.values()),
            'fees': sum(x['fees'] for x in results.values()), 'open_positions': sum(x['open'] for x in results.values()),
            'unpriced_symbols': sum(x['unpriced'] for x in results.values())}


def main():
    db = sessionmaker(bind=make_engine(read_only=True))()
    rows = db.execute(text("select symbol, count(*) n, sum(quote_volume) q from klines where interval='1h' and open_time>=:s and open_time<:e group by symbol having count(*)>=:n order by q desc, symbol"), {'s':START,'e':SPLIT,'n':(SPLIT-START)//BAR}).all()
    chosen = [r[0] for r in rows if r[0].endswith('USDT') and r[0] not in EXCLUDED and not r[0].endswith(('UPUSDT','DOWNUSDT','BULLUSDT','BEARUSDT'))][:50]
    if len(chosen) != 50: raise RuntimeError('训练期合格币不足50')
    frames, features, quality, signals, exits = {}, {}, {}, {}, {}
    lookup = {c.label:c for c in candidates()}
    for s in chosen:
        f = _frame(db,s,'1h',START)
        f = f[(to_ms(f.index)>=START)&(to_ms(f.index)<END)]
        ts = to_ms(f.index); train = f[ts<SPLIT]
        if len(train)!=(SPLIT-START)//BAR or not np.array_equal(to_ms(train.index),np.arange(START,SPLIT,BAR)): raise RuntimeError(s+'训练期不连续')
        vals = f[['open','high','low','close','volume','quote_volume']].to_numpy()
        if not np.isfinite(vals).all() or (f[['open','high','low','close']]<=0).any().any() or (f.volume<0).any() or (f.quote_volume<0).any() or (f.high<f[['open','low','close']].max(axis=1)).any() or (f.low>f[['open','high','close']].min(axis=1)).any(): raise RuntimeError(s+'无效行情')
        frames[s]=f
        features[s]={'train_quote_volume':float(train.quote_volume.sum()), 'train_hourly_log_vol':float(np.log(train.close).diff().std())}
        gaps = np.flatnonzero(np.diff(ts)!=BAR)
        quality[s]={'observed_oos_bars':int(((ts>=SPLIT)&(ts<END)).sum()), 'expected_oos_bars':int((END-SPLIT)//BAR), 'first_gap_utc':iso(ts[gaps[0]]+BAR) if len(gaps) else None, 'last_open_utc':iso(ts[-1])}
        # 缺口后200根禁入场，避免用非连续K线直接产生新信号；已有持仓和挂单按下一实际开盘处理。
        valid=np.ones(len(f),dtype=bool); valid[:200]=False
        for g in gaps: valid[g+1:min(g+201,len(f))]=False
        signals[s]={label:sign(lookup[label],f) for label in LABELS}
        signals[s]['valid']=valid
        exits[s]=exit_signals(f.close.to_numpy())[0]
    db.close()
    ranked=sorted(chosen,key=lambda s:(-features[s]['train_quote_volume'],s))
    pools={'高成交额':ranked[:25], '低成交额':ranked[25:]}
    configs=[{'a':a,'b':b,'combine':how,'exit':ex} for a,b in itertools.combinations(LABELS,2) for how in ('and','or') for ex in EXITS]
    def run(symbols, cfg, start, end, hold=False):
        result={}
        for s in symbols:
            sig=signals[s]
            ent=(combine(sig[cfg['a']],sig[cfg['b']],cfg['combine']).long & sig['valid']) if cfg else np.zeros(len(frames[s]),dtype=bool)
            result[s]=execute(frames[s],ent,exits[s],start,end,cfg['exit'] if cfg else '',hold)
        return result
    selected={}; all_training={}
    for name, syms in pools.items():
        rows=[]
        for cfg in configs:
            r=run(syms,cfg,START,SPLIT)
            m=aggregate(r,STAKE*len(syms))
            rows.append({'config':cfg,'metrics':m})
        eligible=[r for r in rows if r['metrics']['closed_trades']>=30]
        if not eligible: raise RuntimeError(name+'没有满足30笔的候选')
        eligible.sort(key=lambda r:(-r['metrics']['pnl'],json.dumps(r['config'],sort_keys=True)))
        selected[name]=eligible[0]; all_training[name]=rows
        print(name,'训练选择',selected[name],flush=True)
    # 所有选择固定后才计算样本外。
    results={}
    for name,syms in pools.items():
        cfg=selected[name]['config']
        strategy=run(syms,cfg,SPLIT,END)
        benchmark=run(syms,None,SPLIT,END,True)
        results[name]={'strategy':aggregate(strategy,STAKE*len(syms)), 'buy_hold':aggregate(benchmark,STAKE*len(syms)), 'strategy_by_symbol':strategy, 'buy_hold_by_symbol':benchmark}
    # 未分池对照使用同一36候选，按全50币训练期损益独立选取。
    merged=[]
    for i,cfg in enumerate(configs):
        a,b=(all_training[n][i]['metrics'] for n in pools)
        merged.append({'config':cfg,'pnl':a['pnl']+b['pnl'],'trades':a['closed_trades']+b['closed_trades']})
    eligible=sorted((x for x in merged if x['trades']>=60),key=lambda x:(-x['pnl'],json.dumps(x['config'],sort_keys=True)))
    overall=eligible[0]
    unified=run(chosen,overall['config'],SPLIT,END)
    payload={'config':{'data_source':'项目配置数据库 klines native 1h，上游交易所未逐条核实', 'start_utc':iso(START),'split_utc':iso(SPLIT),'end_exclusive_utc':iso(END),'fee_per_side':FEE,'slippage_per_side':SLIP,'stake_per_symbol':STAKE,'capital_total':CAPITAL,'pool_definition':'训练期成交额降序前50均分，高25/低25；相同成交额按symbol排序','candidate_labels':LABELS,'candidate_exits':EXITS,'selection':'每池36个候选，训练平仓至少30笔，按训练净损益降序，平手按配置JSON字典序；全池至少60笔','execution':'收盘信号下一实际K线开盘成交，缺口内不成交，挂单等待；缺口后200根不新开仓；已知区间末最后一小时收盘平仓，缺末尾则按最后收盘估值，未扣假想平仓成本','scope':'仅做多，现货式固定名义，不复利，不计资金费率','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'pools':pools,'features':features,'quality':quality,'training_candidates':all_training,'selected':selected,'oos':results,'unified':{'selected_training':overall,'oos':aggregate(unified,CAPITAL),'by_symbol':unified}}
    OUT.with_suffix('.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# 训练期成交额分池回测','',f"数据为项目配置数据库 native 1h（上游交易所未逐条核实）；UTC 训练 [{iso(START)}, {iso(SPLIT)})，样本外 [{iso(SPLIT)}, {iso(END)})。",'', '训练期完整连续K线的USDT币按成交额选前50，排除BTC、ETH、指定稳定币和杠杆币，再按训练期成交额排名均分高、低各25；所有成员和训练期成交额、波动率见JSON。训练期用于选币会对训练收益引入选样偏差。', '', '入场候选固定为四个现有指标，两两 AND/OR，共12种；与RSI14下穿50或240小时、固定持有48小时、固定持有120小时三种卖出规则组合，共36个。每池训练期至少30笔平仓，按净损益选一组，平手按配置字典序；未分池对照要求60笔。两个池的选择完成后才计算样本外。', '', '每币固定名义200，单边手续费0.04%、滑点0.02%；每池基准资本5000，总资本10000，闲置资金不计收益、不复利；做多、同币单仓。信号在1小时收盘确认，下一个实际有K线的开盘成交。缺失小时不成交，挂单等待下一实际开盘，缺口后200根禁止新入场；超时按实际时钟小时。预定区间最后一小时收盘平仓；若末尾缺失，则持仓按最后已知收盘估值、未计假想卖出成本。买入持有同一标的、同成本、同缺口口径。', '', '| 币池 | 训练选定组合 / 卖出 | 训练收益 | 样本外收益 | 样本外平仓 | 未平仓 | 同池买入持有 |', '|---|---|---:|---:|---:|---:|---:|']
    for name in pools:
        c=selected[name]['config']; tr=selected[name]['metrics']; o=results[name]['strategy']; bh=results[name]['buy_hold']
        lines.append(f"| {name} | {c['a']} {c['combine'].upper()} {c['b']} / {c['exit']} | {tr['return_pct']:.2f}% | {o['return_pct']:.2f}% | {o['closed_trades']} | {o['open_positions']} | {bh['return_pct']:.2f}% |")
    total=sum(results[n]['strategy']['pnl'] for n in pools)/CAPITAL*100
    bh=sum(results[n]['buy_hold']['pnl'] for n in pools)/CAPITAL*100
    u=aggregate(unified,CAPITAL)
    lines += ['',f"分池合计样本外收益 **{total:.2f}%**，等权买入持有 **{bh:.2f}%**；未分池统一规则样本外 **{u['return_pct']:.2f}%**，训练选中 `{json.dumps(overall['config'],ensure_ascii=False)}`。",'', '## 数据质量与限制','', f"样本外每币应有 {(END-SPLIT)//BAR} 根；完整覆盖 {sum(q['observed_oos_bars']==q['expected_oos_bars'] for q in quality.values())}/50，实际 {sum(q['observed_oos_bars'] for q in quality.values())}/{50*((END-SPLIT)//BAR)} 币小时。逐币首次缺口和最后时间见 JSON。",'- 数据缺口、末尾持仓估值与缺口后挂单可能显著影响结果；最后价格陈旧，缺失期损益无法观测。实际不能保证下一可见开盘价成交。','- 研究复用了此前已观察过的样本外时段和四项历史实验候选，虽在脚本中严格按训练选择，本次并非预注册的全项目独立盲测；多重搜索与选择偏差仍在。','- 无盘口深度、冲击、最小下单量、资金费率、借贷、退市价格及可交易性核验；不是永续实盘收益。分池依据训练期成交额并非历史逐时成分，存在存续偏差。','- 未计算逐小时盯市回撤、统计显著性及多重检验校正；固定分割仅一次，不能证明稳健性。','', '运行：`./.venv/Scripts/python.exe backend/scripts/pool_research.py`。需要相同项目依赖、数据库配置及相同历史数据快照。完整候选、成员、逐币收益和质量记录见 JSON。']
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('完成', OUT,flush=True)

if __name__=='__main__': main()

"""Capital41 Fusion Scalp：训练期全可用币分池、参数选择及固定样本外验证。"""
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
from app.catalog_indicators.capital41_fusion_scalp import capital41_fusion_scalp
from app.research.lab_experiment import _frame
from app.research.lab_sim import to_ms

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/capital41-scalp-pool-oos'
BAR = 3600000
START = int(pd.Timestamp('2025-09-27', tz='UTC').timestamp()*1000)
SPLIT = int(pd.Timestamp('2026-07-01', tz='UTC').timestamp()*1000)
END = int(pd.Timestamp('2026-10-26T13:00:00Z').timestamp()*1000)
FEE, SLIP, STAKE = .0004, .0002, 200.
EXCLUDED = {'USDCUSDT','FDUSDUSDT','TUSDUSDT','USDPUSDT','DAIUSDT','EURUSDT','GBPUSDT','PAXGUSDT'}
CONFIGS = [{'require_turn': turn, 'atr_expansion_multiplier': atr, 'hold': hold}
           for turn, atr, hold in itertools.product((True, False), (1.0, 1.1), (24, 72))]


def iso(t):
    return pd.to_datetime(int(t), unit='ms', utc=True).isoformat()


def simulate(f, sig, begin, end, hold):
    ts = to_ms(f.index)
    lo, hi = np.searchsorted(ts, [begin, end])
    if lo == hi:
        return {'pnl': 0., 'trades': 0, 'fees': 0., 'open': False, 'unpriced': True}
    opens, closes = f.open.to_numpy(), f.close.to_numpy()
    position, pending, entered, cash, fees, trades = 0, 0, 0, 0., 0., 0
    quantity = 0.
    for i in range(lo, hi):
        if pending and not position:
            position = pending
            price = opens[i] * (1 + SLIP * position)
            quantity = STAKE / price
            fee = STAKE * FEE
            cash -= fee
            fees += fee
            entered = ts[i]
        elif pending == 2 and position:
            price = opens[i] * (1 - SLIP * position)
            fee = quantity * price * FEE
            cash += position * quantity * (price - entry_price) - fee
            fees += fee
            position = 0
            trades += 1
        if position and pending in (-1, 1):
            entry_price = price
        pending = 0
        if ts[i] == end - BAR:
            if position:
                price = closes[i] * (1 - SLIP * position)
                fee = quantity * price * FEE
                cash += position * quantity * (price-entry_price) - fee
                fees += fee
                position = 0
                trades += 1
            break
        if position:
            if ts[i] - entered >= (hold-1)*BAR:
                pending = 2
        elif sig[i]:
            pending = int(sig[i])
    if position:
        cash += position * quantity * (closes[hi-1] - entry_price)
    return {'pnl': float(cash), 'trades': trades, 'fees': float(fees), 'open': bool(position),
            'unpriced': False, 'last_open_utc': iso(ts[hi-1])}


def aggregate(rows):
    return {'pnl': sum(r['pnl'] for r in rows), 'return_pct': 100*sum(r['pnl'] for r in rows)/(STAKE*len(rows)),
            'trades': sum(r['trades'] for r in rows), 'fees': sum(r['fees'] for r in rows),
            'open': sum(r['open'] for r in rows), 'unpriced': sum(r['unpriced'] for r in rows)}


def main():
    db = sessionmaker(bind=make_engine(read_only=True))()
    expected = (SPLIT-START)//BAR
    rows = db.execute(text("select symbol, count(*) n, sum(quote_volume) q from klines where interval='1h' and open_time>=:s and open_time<:e group by symbol having count(*)>=:n order by q desc, symbol"), {'s': START, 'e': SPLIT, 'n': expected}).all()
    universe = [r[0] for r in rows if r[0].endswith('USDT') and r[0] not in EXCLUDED and not r[0].endswith(('UPUSDT','DOWNUSDT','BULLUSDT','BEARUSDT'))]
    print('训练期候选', len(universe), flush=True)
    records, rejected = {}, {}
    for j, symbol in enumerate(universe):
        f = _frame(db, symbol, '1h', START)
        f = f[(to_ms(f.index)>=START)&(to_ms(f.index)<END)]
        ts = to_ms(f.index)
        train = f[ts<SPLIT]
        if not np.array_equal(to_ms(train.index), np.arange(START,SPLIT,BAR)):
            rejected[symbol] = '训练期不连续或重复数据'; continue
        vals = f[['open','high','low','close','volume','quote_volume']].to_numpy()
        if not np.isfinite(vals).all() or (f[['open','high','low','close']]<=0).any().any() or (f[['volume','quote_volume']]<0).any().any() or (f.high<f[['open','low','close']].max(axis=1)).any() or (f.low>f[['open','high','close']].min(axis=1)).any():
            # 仅训练期质量可决定是否入池；测试期无效数据中止，不能事后删除币。
            raise ValueError(symbol+' 数据无效，需检查训练期或测试期行情')
        gaps = np.flatnonzero(np.diff(ts)!=BAR)
        valid = np.ones(len(f), dtype=bool); valid[:200] = False
        for g in gaps:
            valid[g+1:min(g+201,len(f))] = False
        signal_map = {}
        for turn, atr in itertools.product((True, False), (1.0, 1.1)):
            ind = capital41_fusion_scalp(f, require_turn=turn, atr_expansion_multiplier=atr)
            signal_map[(turn,atr)] = (ind.long_signal.to_numpy().astype(int)-ind.short_signal.to_numpy().astype(int))*valid
        train_results, test_results = [], []
        for cfg in CONFIGS:
            sig=signal_map[(cfg['require_turn'],cfg['atr_expansion_multiplier'])]
            train_results.append(simulate(f,sig,START,SPLIT,cfg['hold']))
            test_results.append(simulate(f,sig,SPLIT,END,cfg['hold']))
        records[symbol]={'train_quote_volume':float(train.quote_volume.sum()),
                         'train':train_results, 'test':test_results,
                         'quality':{'train_bars':len(train),'test_bars':int(((ts>=SPLIT)&(ts<END)).sum()),
                                    'expected_test_bars':int((END-SPLIT)//BAR),
                                    'first_gap_utc':iso(ts[gaps[0]]+BAR) if len(gaps) else None,
                                    'last_open_utc':iso(ts[-1])}}
        if (j+1)%25==0: print('完成',j+1,'/',len(universe),flush=True)
    db.close()
    ranked=sorted(records,key=lambda s:(-records[s]['train_quote_volume'],s))
    if len(ranked)<6: raise RuntimeError('合格币少于6')
    pools={f'成交额第{i+1}组':list(x) for i,x in enumerate(np.array_split(ranked,3))}
    def choose(syms, minimum):
        candidates=[]
        for k,c in enumerate(CONFIGS):
            m=aggregate([records[s]['train'][k] for s in syms])
            candidates.append({'config':c,'metrics':m})
        eligible=[(i,x) for i,x in enumerate(candidates) if x['metrics']['trades']>=minimum]
        if not eligible: return None,candidates
        eligible.sort(key=lambda x:(-x[1]['metrics']['pnl'],x[0]))
        return eligible[0][0],candidates
    selected, results = {}, {}
    for name,syms in pools.items():
        idx,candidates=choose(syms,30)
        selected[name]={'index':idx,'candidates':candidates}
        if idx is not None:
            results[name]=aggregate([records[s]['test'][idx] for s in syms])
    u, unified_candidates=choose(ranked,90)
    unified=aggregate([records[s]['test'][u] for s in ranked]) if u is not None else None
    all_results=[records[s]['test'][selected[name]['index']] for name,syms in pools.items() if selected[name]['index'] is not None for s in syms]
    payload={'config':{'source':'项目配置数据库 klines 原生1h；上游未逐条核验', 'start_utc':iso(START),'split_utc':iso(SPLIT),'end_exclusive_utc':iso(END),'fee_per_side':FEE,'slippage_per_side':SLIP,'stake_per_symbol':STAKE,'candidates':CONFIGS,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'candidate_count':len(universe),'rejected':rejected,'pools':pools,'records':records,'selection':selected,'oos':results,'pooled_total':aggregate(all_results) if len(all_results)==len(ranked) else None,'unified':{'index':u,'candidates':unified_candidates,'oos':unified}}
    OUT.with_suffix('.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    quality=[records[s]['quality'] for s in ranked]
    lines=['# Capital41 Fusion Scalp 全可用币分池样本外实验','',f"数据：项目配置数据库 `klines` 原生 1h，上游交易所未逐条核实。UTC 训练 [{iso(START)}, {iso(SPLIT)})；测试 [{iso(SPLIT)}, {iso(END)})。数据库候选 {len(universe)} 个，训练期连续且合格 {len(ranked)} 个；完整币种、逐币结果、参数和质量记录见配套 JSON。",'', '币池仅按训练期全周期完整的 USDT 币选取，排除 USDC、FDUSD、TUSD、USDP、DAI、EUR、GBP、PAXG 对 USDT 的交易对，以及名称后缀 UP、DOWN、BULL、BEAR 的杠杆币；BTC、ETH 包含在内。按训练期总成交额降序均分三池，平手按名称排序。该范围是数据库中满足训练覆盖的全部币，并非交易所全部历史上市币；固定全训练期连续会造成选样和存续偏差。', '', '复用项目原指标，多空均做。其 EMA、RSI、Williams %R、ATR 与成交量条件均须同时满足；原指标无退出规则。候选为 require_turn 真/假、ATR 扩张倍数 1.0/1.1、固定持有 24/72 小时共 8 组，其他指标参数用原默认值。每池训练期至少 30 笔平仓，按训练净损益最大选择，平手按候选次序；未分池统一参数至少 90 笔。候选未满足交易数时不选择。', '', 'K线收盘信号在下一根实际K线开盘成交；每币单仓、双向、每笔固定名义200，池资本=币数×200，闲置资金不计收益，不复利。单边手续费0.04%、滑点0.02%，均已计入损益；费用另列，仅含手续费，滑点体现在成交价。持有期按实际时钟，缺口中挂单等待下一根实际开盘，缺口后200根不新开仓。预定区间末小时收盘强平；末尾缺失则按最后收盘盯市且未计假想退出成本。未计资金费、借币费、冲击、保证金限制。', '', '| 池 | 币数 | 训练选择（turn / ATR / 持有小时） | 训练收益 | 训练笔数 | 测试收益 | 测试笔数 | 手续费 | 测试未平仓 |', '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for name,syms in pools.items():
        idx=selected[name]['index']
        if idx is None:
            lines.append(f'| {name} | {len(syms)} | 训练交易数不足 | — | — | — | — | — | — |'); continue
        c=CONFIGS[idx]; tr=selected[name]['candidates'][idx]['metrics']; o=results[name]
        lines.append(f"| {name} | {len(syms)} | {c['require_turn']} / {c['atr_expansion_multiplier']} / {c['hold']} | {tr['return_pct']:.3f}% | {tr['trades']} | {o['return_pct']:.3f}% | {o['trades']} | {o['fees']:.2f} | {o['open']} |")
    if payload['pooled_total'] and unified:
        lines += ['',f"分池合计测试收益 {payload['pooled_total']['return_pct']:.3f}%，平仓 {payload['pooled_total']['trades']} 笔；统一参数 {CONFIGS[u]} 的测试收益 {unified['return_pct']:.3f}%，平仓 {unified['trades']} 笔。两者使用相同币池和固定资本。"]
    lines += ['',f"测试期每币应有 {(END-SPLIT)//BAR} 根，完整覆盖 {sum(q['test_bars']==q['expected_test_bars'] for q in quality)}/{len(quality)} 币，实际 {sum(q['test_bars'] for q in quality)}/{len(quality)*((END-SPLIT)//BAR)} 币小时。",'', '局限：样本外区间此前在项目其他研究中已被观察，不是项目层面的盲测；仅一次切分和8组候选，参数搜索仍有过拟合风险。缺口会影响信号、挂单和持仓估值，滞后末价不代表可成交；指标在缺口后重热身200根但不重置历史。未核对历史上架、退市及可交易性，未计资金费和市场冲击；结果不是可直接复现的永续账户收益。未计算逐时权益回撤、统计显著性或多重检验校正。训练期选币会使训练收益带选样偏差。研究退出机制与正在开发的 ATR 止损止盈 spec 不同，不能等同其绩效。', '', '运行：`.venv/Scripts/python.exe backend/scripts/capital41_scalp_pool_oos.py`，结果依赖本地数据库快照。']
    OUT.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('完成',OUT,flush=True)

if __name__ == '__main__': main()

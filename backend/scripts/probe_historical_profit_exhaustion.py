"""离线历史策略持仓盈利决策点探针；只读 PostgreSQL。"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import rankdata
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'backend'), str(ROOT/'backend/scripts')]
from app.db import make_engine
from app.backtest.execution import simulate
from app.backtest.trades import aggregate_positions
from app.strategy.rules import prepare_frame, evaluate_signals
from probe_profit_exhaustion import features, correlation
from supertrend_exit_diagnosis import _db, resolve_for

START = pd.Timestamp('2026-01-29T14:30:00Z')
END = pd.Timestamp('2026-07-28T14:30:00Z')
TRAIN_END = START + pd.Timedelta(days=108)
VALID_END = START + pd.Timedelta(days=144)
STEP = 900_000
H = 8
LOOKBACK = 600
SQL = text("""SELECT open_time,open,high,low,close,volume FROM klines
 WHERE symbol=:symbol AND interval='15m' AND open_time>=:start AND open_time<:end ORDER BY open_time""")


def to_ms(value):
    return int(value.timestamp()*1000)


def label_and_candidates(frame, feat, positions, symbol):
    times = frame.index
    out = []
    last = -H-1
    for p in sorted(positions, key=lambda x:x['entry_time']):
        entry = times.searchsorted(pd.Timestamp(p['entry_time']))
        exit_i = times.searchsorted(pd.Timestamp(p['exit_time']))
        side = 1 if p['side']=='long' else -1
        for i in range(entry, min(exit_i, len(frame)-H-1)):
            t = times[i]
            if t < START or t >= VALID_END or i < LOOKBACK or i-last < H:
                continue
            if side*(float(frame.close.iloc[i])/float(p['entry_price'])-1) <= .0012:
                continue
            f = feat.iloc[i]
            if not np.isfinite(f.to_numpy(dtype=float)).all():
                continue
            end = times[i+H+1]
            split = 'train' if end < TRAIN_END else ('validation' if t >= TRAIN_END and end < VALID_END else None)
            if split is None:
                break
            y = side*(float(frame.open.iloc[i+H+1])/float(frame.open.iloc[i+1])-1)
            out.append({'symbol':symbol,'time':t.isoformat(),'side':p['side'], 'split':split,
                        'rsi_pullback':float(-side*f['change']),
                        'exhaustion':float(f['top'] if side==1 else f['bot']), 'y':y,
                        'entry_time':p['entry_time'],'exit_time':p['exit_time']})
            last = i
            break
    return out


def worker(task):
    symbol, params = task
    try:
        with make_engine(read_only=True, null_pool=True).connect() as conn:
            rows = conn.execute(SQL, {'symbol':symbol, 'start':to_ms(START)-LOOKBACK*STEP,
                                      'end':to_ms(VALID_END)}).fetchall()
        frame = pd.DataFrame(rows,columns=['time','open','high','low','close','volume'])
        frame['time'] = pd.to_datetime(frame.time,unit='ms',utc=True)
        frame = frame.set_index('time').sort_index().astype(float)
        valid = (frame[['open','high','low','close']]>0).all(axis=1) & (frame.volume>=0)
        valid &= frame.high >= frame[['open','low','close']].max(axis=1)
        valid &= frame.low <= frame[['open','high','close']].min(axis=1)
        frame = frame[valid]
        parts = frame.index.to_series().diff().ne(pd.Timedelta(minutes=15)).cumsum()
        positions, records, checks, signals_count = [], [], [], Counter()
        for _, segment in frame.groupby(parts):
            if len(segment)<LOOKBACK+H+2: continue
            prepared = prepare_frame(segment, 'supertrend_current', params, symbol=symbol)
            signals = evaluate_signals(prepared, 'supertrend_current', params)
            signals_count.update({str(k):int(v) for k,v in signals.direction.value_counts().items()})
            _, legs = simulate(prepared,signals,params,10000.,.0004,.0002,'15m')
            trades = aggregate_positions(legs)
            positions.extend(trades)
            feat = features(segment)
            if symbol=='BTCUSDT':
                cut = len(segment)//2
                prefix = features(segment.iloc[:cut])
                n = int(prefix.notna().all(axis=1).sum())
                checks.append({'finite':n,'passed':bool(n and np.allclose(feat.iloc[:cut],prefix,equal_nan=True))})
            records.extend(label_and_candidates(segment,feat,trades,symbol))
        return {'symbol':symbol,'records':records,'position_count':len(positions),
                'signal_counts':dict(signals_count),'causality':checks,'error':None}
    except Exception as exc:
        return {'symbol':symbol,'records':[],'error':repr(exc)}


def calibration():
    rng = np.random.default_rng(20261111)
    n, sims = 1200, 500
    def ic(a,b):
        x,y = rankdata(a),rankdata(b)
        return float(np.corrcoef(x,y)[0,1])
    def pvalue(a,b):
        x,y = rankdata(a),rankdata(b)
        x=(x-x.mean())/x.std(); y=(y-y.mean())/y.std()
        obs=abs(np.mean(x*y))
        offsets=rng.integers(32,n-32,size=199)
        return (1+sum(abs(np.mean(x*np.roll(y,int(k))))>=obs for k in offsets))/200
    def ar():
        return lfilter([1.],[1.,-.9],rng.normal(size=n))
    x,z=ar(),ar(); x=x/x.std(); z=z/z.std()
    strong=.8*x+.6*z; weak=.2*x+np.sqrt(.96)*z
    pos={'ic':ic(x,strong),'p':pvalue(x,strong)}
    half={'ic':ic(x,weak),'p':pvalue(x,weak)}
    rejects=0
    for _ in range(sims):
        rejects += pvalue(ar(),ar())<=.05
    rate=rejects/sims
    return {'positive':pos,'weak':half,'null_simulations':sims,'null_rejections':rejects,
            'null_rate':rate,'passed':bool(pos['p']<=.05 and 0<half['ic']<pos['ic'] and rate<=.10),
            'note':'合成 AR(1) 独立序列的循环位移校准，不保证市场平稳'}


def main():
    params=resolve_for(_db(),'trend_reversal_fit','15m')
    if params.get('cl_enabled'): raise ValueError('闭环策略须使用专用撮合器')
    if params['fee_rate']!=.0004 or params['slippage']!=.0002: raise ValueError('成本与研究口径不符')
    with make_engine(read_only=True).connect() as conn:
        rows=conn.execute(text("""SELECT symbol,count(*) n,min(open_time) lo,max(open_time) hi
          FROM klines WHERE interval='15m' AND open_time>=:s AND open_time<:e GROUP BY symbol ORDER BY symbol"""),
          {'s':to_ms(START),'e':to_ms(END)}).mappings().all()
    coverage=[dict(r) for r in rows]
    selected=[r['symbol'] for r in coverage if r['n']>=int(180*96*.99) and r['lo']==to_ms(START) and r['hi']==to_ms(END)-STEP]
    print('selected',len(selected),'window',START,END,flush=True)
    cal=calibration()
    outputs=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        tasks=[pool.submit(worker,(s,params)) for s in selected]
        for task in as_completed(tasks):
            item=task.result(); outputs.append(item)
            print('done',len(outputs),'/',len(selected),item['symbol'],'error',item.get('error'),flush=True)
    records=[r for o in outputs for r in o['records']]
    summaries=[]
    for feature in ('rsi_pullback','exhaustion'):
        for split in ('train','validation'):
            rs=[r for r in records if r['split']==split]
            per=[]
            for sym in selected:
                sub=[r for r in rs if r['symbol']==sym]
                v=correlation([r[feature] for r in sub],[r['y'] for r in sub])
                per.append({'symbol':sym,'pairs':len(sub),'ic':v})
            vals=[r['ic'] for r in per if r['ic'] is not None]
            avg=float(np.mean(vals)) if vals else None
            share=float(np.mean(np.sign(vals)==np.sign(avg))) if vals and avg else None
            summaries.append({'feature':feature,'split':split,'pairs':len(rs),'eligible_symbols':len(vals),
                              'mean_ic':avg,'same_sign_fraction':share,'per_symbol':per})
    pairs=[]
    for feature in ('rsi_pullback','exhaustion'):
        a,b=[r for r in summaries if r['feature']==feature]
        enough=a['eligible_symbols']>=10 and b['eligible_symbols']>=10
        numerical=bool(enough and a['mean_ic']<-.02 and b['mean_ic']<-.02 and a['same_sign_fraction']>=.6 and b['same_sign_fraction']>=.6)
        pairs.append({'feature':feature,'eligible_symbols_min_10':enough,'numerical_gate':numerical,
                      'gate_pass':bool(numerical and cal['passed'] and not any(o['error'] for o in outputs))})
    result={'window':{'start':START.isoformat(),'train_end':TRAIN_END.isoformat(),
                      'validation_end':VALID_END.isoformat(),'coverage_end':END.isoformat(),'reserved_test_read':False},
            'coverage':coverage,'selected':selected,'skipped':[r for r in coverage if r['symbol'] not in selected],
            'parameters':params,'costs':{'fee_rate':.0004,'slippage':.0002,'basis':'notional','ic_is_gross':True},
            'calibration':cal,'summaries':summaries,'gates':pairs,'workers':[{k:v for k,v in o.items() if k!='records'} for o in outputs],
            'record_counts':dict(Counter((r['split']+'/'+r['side']) for r in records)), 'records':records,
            'limitations':['择币按整个180天覆盖率会产生历史存续偏差','当前参数不代表历史每时刻配置',
                           '训练与验证前瞻收益可越过模拟平仓，是反事实；非实际退出收益',
                           '模拟器以 bar 为单位处理止损及分批止盈，同 bar 先止损',
                           '未读取预留测试段价格，覆盖筛选读取测试段数量和边界','未计资金费率、深度、部分成交',
                           '相关性非净收益交易回测；多币相关和多个特征的比较仍需独立验证']}
    out=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'docs/probe-historical-profit-exhaustion.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False,
                              default=lambda x: x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
    print(json.dumps({'counts':result['record_counts'],'summary':[{k:v for k,v in r.items() if k!='per_symbol'} for r in summaries],
                      'calibration':cal,'gates':pairs,'errors':[o for o in outputs if o['error']]},ensure_ascii=False,indent=2,
                     default=lambda x: x.item() if isinstance(x,np.generic) else str(x)))


if __name__=='__main__': main()

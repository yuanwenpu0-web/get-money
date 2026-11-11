"""真实订单盈利时点的离线 IC 探针；只读数据库，不修改交易配置。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.db import make_engine
from app.catalog_indicators._pine import rsi
from app.catalog_indicators.exhaustion_scanner import exhaustion_scanner
from exit_metalabel_probe import CANDLE_SQL

STEP = 900_000
H = 8
MIN_PAIRS = 30
SQL_ORDERS = text("""SELECT id, symbol, side, source, entry_price, created_at, closed_at
 FROM orders WHERE strategy='supertrend_current' AND interval='15m'
 AND status='CLOSED' AND source IN ('paper','live') AND entry_price>0
 AND closed_at IS NOT NULL ORDER BY created_at,id""")
SQL_CANDLES = text("""SELECT open_time,open,high,low,close,volume FROM klines
 WHERE symbol=:symbol AND interval='15m' AND open_time>=:start AND open_time<:end
 ORDER BY open_time""")


def ms(x):
    return int(x.timestamp() * 1000)


def correlation(x, y):
    if len(x) < MIN_PAIRS or np.std(x) == 0 or np.std(y) == 0:
        return None
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def features(frame):
    rs = rsi(frame['close'], 14)
    scan = exhaustion_scanner(frame, preset='Crypto')
    return pd.DataFrame({'rsi': rs, 'change': rs.diff(),
                         'top': scan['top_score'], 'bot': scan['bot_score']})


def main():
    with make_engine(read_only=True).connect() as conn:
        orders = [dict(x) for x in conn.execute(SQL_ORDERS).mappings()]
        frames = {}
        for symbol in sorted({x['symbol'] for x in orders}):
            subset = [o for o in orders if o['symbol'] == symbol]
            start = min(ms(o['created_at']) for o in subset) - 500 * STEP
            end = max(ms(o['closed_at']) for o in subset) + (H + 3) * STEP
            rows = conn.execute(text(CANDLE_SQL), {'sym': symbol, 'start_ms': start, 'end_ms': end, 'step': STEP, 'minutes': 15}).fetchall()
            frame = pd.DataFrame(rows, columns=['time','open','high','low','close','volume','quote_volume'])
            if frame.empty:
                continue
            frame = frame.set_index('time').astype(float).sort_index()
            frames[symbol] = frame
    cuts = {}
    for source in ('paper', 'live'):
        times = sorted(ms(o['created_at']) for o in orders if o['source'] == source)
        if times:
            cuts[source] = times[int(len(times)*.7)]
    records, skips, causality = [], {}, []
    for symbol, frame in frames.items():
        valid = (frame[['open','high','low','close']] > 0).all(axis=1) & (frame.volume >= 0)
        valid &= frame.high >= frame[['open','low','close']].max(axis=1)
        valid &= frame.low <= frame[['open','high','close']].min(axis=1)
        frame = frame[valid]
        groups = frame.index.to_series().diff().ne(STEP).cumsum()
        for _, seg in frame.groupby(groups):
            if len(seg) < 120:
                continue
            feat = features(seg)
            cut = len(seg)//2
            if cut >= 100:
                truncated = features(seg.iloc[:cut])
                finite = int(feat.iloc[:cut].notna().all(axis=1).sum())
                ok = bool(finite and np.allclose(feat.iloc[:cut], truncated, equal_nan=True))
                causality.append({'symbol':symbol,'finite':finite,'passed':ok})
            times = seg.index.to_numpy(dtype=np.int64)
            for order in (o for o in orders if o['symbol'] == symbol):
                entry, end = ms(order['created_at']), ms(order['closed_at'])
                indices = np.flatnonzero((times + STEP > entry) & (times + STEP < end))
                side = 1 if order['side'] == 'LONG' else -1
                for i in indices:
                    # Only a candle opened after the actual fill can drive an exit decision.
                    if times[i] < entry or i < 100 or i + H + 1 >= len(seg):
                        continue
                    x = feat.iloc[i]
                    if not np.isfinite(x.to_numpy(dtype=float)).all():
                        continue
                    if side * (float(seg.close.iloc[i]) / float(order['entry_price']) - 1) <= .0012:
                        continue
                    decision = int(times[i] + STEP)
                    target_end = int(times[i + H + 1])
                    split = 'train' if target_end < cuts[order['source']] else ('validation' if decision >= cuts[order['source']] else None)
                    if split is None:
                        skips['split_purge'] = skips.get('split_purge', 0) + 1
                        break
                    y = side * (float(seg.open.iloc[i + H + 1]) / float(seg.open.iloc[i + 1]) - 1)
                    records.append({'id':order['id'],'source':order['source'],'symbol':symbol,
                                    'side':order['side'],'decision_ms':decision,'split':split,
                                    'rsi_pullback':float(-side*x['change']),
                                    'exhaustion':float(x['top'] if side == 1 else x['bot']),
                                    'future_hold_return':y})
                    break
    # A given order is counted once, even if fragmented candles offered multiple segments.
    unique = {}
    for row in sorted(records, key=lambda r:r['decision_ms']):
        unique.setdefault(row['id'], row)
    records = list(unique.values())
    summaries = []
    for source in ('paper','live'):
        for split in ('train','validation'):
            rows = [r for r in records if r['source']==source and r['split']==split]
            for feature in ('rsi_pullback','exhaustion'):
                per = []
                for symbol in sorted({r['symbol'] for r in rows}):
                    subset = [r for r in rows if r['symbol']==symbol]
                    value = correlation([r[feature] for r in subset],[r['future_hold_return'] for r in subset])
                    per.append({'symbol':symbol,'pairs':len(subset),'ic':value})
                vals = [p['ic'] for p in per if p['ic'] is not None]
                mean = float(np.mean(vals)) if vals else None
                share = float(np.mean(np.sign(vals)==np.sign(mean))) if vals and mean else None
                pooled = correlation([r[feature] for r in rows],[r['future_hold_return'] for r in rows])
                summaries.append({'source':source,'split':split,'feature':feature,'pairs':len(rows),
                                  'pooled_ic_descriptive':pooled,'eligible_symbols':len(vals),
                                  'mean_symbol_ic':mean,'same_sign_fraction':share,'per_symbol':per})
    rng = np.random.default_rng(20261111)
    ys = np.array([r['future_hold_return'] for r in records])
    controls = {'oracle_ic':correlation(ys,ys), 'shuffled_ic':correlation(rng.permutation(ys),ys)}
    result = {'window':{'train_validation_cuts_utc':{k:pd.Timestamp(v,unit='ms',tz='UTC').isoformat() for k,v in cuts.items()}},
              'config':{'source':'local PostgreSQL orders and complete 1m-derived 15m klines','holding_bars':H,
                        'decision':'first fully post-entry completed bar with mark profit > 0.12%; next open',
                        'label':'directional open-to-open return from next open to 8 bars later; counterfactual after actual exit allowed',
                        'fee_rate':.0004,'slippage':.0002,'ic_is_gross':True,'min_pairs_per_symbol':MIN_PAIRS},
              'orders':len(orders),'orders_per_source':{s:sum(o['source']==s for o in orders) for s in cuts},
              'records':records,'skipped_count':len(orders)-len(records),'skip_reasons_partial':skips,
              'summaries':summaries,'causality':causality,'controls':controls,
              'gate':'insufficient' if any(s['eligible_symbols']==0 for s in summaries) else 'requires review',
              'limitations':['真实订单每币样本稀少；池化 IC 不能替代同号币门槛',
                             '真实订单触发与成交受既有配置和服务存续期影响，不代表完整历史策略信号',
                             '持仓盈利筛选为收盘估算且依赖实际订单平仓时间，未来标签允许越过实际平仓',
                             '正对照刻意使用未来收益，只用于校验；随机打乱未保留时序相关性',
                             '无交易回测、净收益置信区间、随机交易基线或多重检验']}
    out = ROOT/'docs/probe-profit-exhaustion-derived.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'orders':len(orders),'records':len(records),'summaries':[{k:v for k,v in s.items() if k!='per_symbol'} for s in summaries], 'controls':controls,'causal_failures':sum(not c['passed'] for c in causality),'causal_checks':len(causality),'cuts':result['window']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

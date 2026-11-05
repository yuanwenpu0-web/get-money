"""独立研究：反转前10根15m K线、成熟标签、时间隔离和实际仓位路径回放。"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.backtest.execution import simulate
from app.backtest.trades import aggregate_positions
from app.db import make_engine
from app.models import StrategyBinding, StrategyConfig, SymbolParam
from app.state import all_state
from app.strategy.bindings import binding_params
from app.strategy.rules import evaluate_signals, prepare_frame, resolve_params

CODE = 'supertrend_current'
STEP = pd.Timedelta(minutes=15)
HORIZON = 16
CASH = 10000.0


def features(frame, i, side):
    # 信号K线 i 已收盘，但特征严格只取 i-10 ... i-1。
    w = frame.iloc[i-10:i]
    c = w.close.to_numpy(dtype=float)
    o = w.open.to_numpy(dtype=float)
    h = w.high.to_numpy(dtype=float)
    l = w.low.to_numpy(dtype=float)
    v = w.volume.to_numpy(dtype=float)
    base = c[-1]
    out = {'side': float(side)}
    for j in range(10):
        out.update({f'c{j}': (c[j] / base - 1) * side,
                    f'body{j}': (c[j] / o[j] - 1) * side,
                    f'range{j}': (h[j] - l[j]) / c[j],
                    f'volume{j}': v[j] / max(v.mean(), 1e-12)})
    return out


def load(db, symbol, end_ms):
    sql = text("select open_time, open, high, low, close, volume from klines "
               "where symbol=:symbol and interval='15m' and open_time < :end order by open_time")
    df = pd.read_sql_query(sql, db.connection(), params={'symbol': symbol, 'end': end_ms})
    if df.empty:
        return []
    df.index = pd.to_datetime(df.pop('open_time'), unit='ms', utc=True)
    df = df[(df.index + STEP <= pd.Timestamp(end_ms, unit='ms', tz='UTC'))]
    valid = np.isfinite(df[['open','high','low','close','volume']]).all(axis=1)
    valid &= (df[['open','high','low','close']] > 0).all(axis=1) & (df.volume >= 0)
    valid &= df.high >= df[['open','low','close']].max(axis=1)
    valid &= df.low <= df[['open','high','close']].min(axis=1)
    df = df.loc[valid]
    group = df.index.to_series().diff().ne(STEP).cumsum()
    return [g.copy() for _, g in df.groupby(group) if len(g) >= 800]


def path_score(parts, start, end, threshold, probs, params, fee, slip):
    details = []
    for symbol, frame, sig in parts:
        mask = (frame.index >= start) & (frame.index < end)
        d = frame.loc[mask]
        if len(d) < 2:
            continue
        s = sig.loc[mask].copy()
        # direction 原样保留，使被筛掉的反向信号仍能平仓。
        p = probs.get(symbol, pd.Series(dtype=float)).reindex(s.index)
        allow = p.ge(threshold) if threshold is not None else pd.Series(True, index=s.index)
        s['allow_long_entry'] = allow
        s['allow_short_entry'] = allow
        eq, raw = simulate(d, s, params[symbol], CASH, fee, slip, '15m')
        trades = aggregate_positions(raw)
        dd = (eq / eq.cummax() - 1).min() * 100
        details.append({'symbol': symbol, 'return_pct': float((eq.iloc[-1]/CASH-1)*100),
                        'max_drawdown_pct': float(dd), 'trades': len(trades),
                        'pnl': float(sum(t['pnl'] for t in trades)),
                        'fees': float(sum(t.get('fees', 0) for t in trades))})
    return {'accounts': details, 'mean_return_pct': float(np.mean([x['return_pct'] for x in details])) if details else None,
            'trades': sum(x['trades'] for x in details)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='BTCUSDT,ETHUSDT,SOLUSDT')
    ap.add_argument('--end', default='2026-10-25T00:00:00Z')
    ap.add_argument('--out', default='output/supertrend_reversal_ml_20261025')
    ap.add_argument('--replay', default='', help='先前研究输出目录；使用其行情快照和配置重跑')
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f'输出目录已存在，拒绝覆盖：{out}')
    end = pd.Timestamp(args.end)
    if end.tz is None:
        raise SystemExit('--end 必须包含时区')
    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    db = None
    if args.replay:
        source = ROOT / args.replay
        previous = json.loads((source / 'report.json').read_text(encoding='utf-8'))
        snapshot = pd.read_csv(source / 'candles.csv', float_precision='round_trip')
        snapshot['time'] = pd.to_datetime(snapshot['time'], utc=True)
        config = previous['config']
        params = config['parameters_by_symbol']
        fee, slip = config['fee'], config['slippage']
    else:
        db = sessionmaker(bind=make_engine(read_only=True), future=True)()
        state = all_state(db)
        cfg = db.query(StrategyConfig).filter_by(code=CODE).first()
        binding = db.query(StrategyBinding).filter_by(strategy_code=CODE, interval='15m', pool_code='all').first()
        if not cfg or not binding:
            raise SystemExit('未找到策略配置或 all/15m 绑定')
        base = binding_params(db, binding, state)
        fee, slip = float(base['fee_rate']), float(base['slippage'])
        params = {}
        config = {'strategy_enabled':cfg.enabled, 'binding_enabled':binding.enabled,
                  'binding_pool':binding.pool_code, 'state_execution_mode':state.get('execution_mode')}
    parts, samples, coverage = [], [], []
    for symbol in symbols:
        if args.replay:
            frame = snapshot.loc[snapshot.symbol == symbol].drop(columns='symbol').set_index('time').sort_index()
            frame = frame.loc[frame.index + STEP <= end]
            groups = frame.index.to_series().diff().ne(STEP).cumsum()
            segments = [g.copy() for _,g in frame.groupby(groups) if len(g) >= 800]
        else:
            ov = db.query(SymbolParam).filter_by(strategy_code=CODE, symbol=symbol, interval='15m').first()
            override = json.loads(ov.params) if ov else {}
            layer = {k:v for k,v in {**base, **override}.items() if k not in ('fee_rate','slippage')}
            params[symbol] = resolve_params(CODE, state, layer, interval='15m')
            segments = load(db, symbol, int(end.timestamp()*1000))
        for frame in segments:
            prepared = prepare_frame(frame, CODE, params[symbol], symbol=symbol)
            sig = evaluate_signals(prepared, CODE, params[symbol], symbol=symbol)
            parts.append((symbol, prepared, sig))
            coverage.append({'symbol':symbol,'start':str(frame.index[0]),'end':str(frame.index[-1]+STEP),'bars':len(frame)})
            for i in np.flatnonzero(sig.direction.to_numpy() != 0):
                if i < 600:
                    continue
                side = int(sig.direction.iloc[i])
                # 次根开盘入场，第16根收盘退出；双边滑点和手续费。
                if i + HORIZON < len(frame):
                    entry = float(frame.open.iloc[i+1]) * (1 + side*slip)
                    exit_price = float(frame.close.iloc[i+HORIZON]) * (1 - side*slip)
                    net = side*(exit_price-entry)/entry - fee*(1+exit_price/entry)
                    mature, label = frame.index[i+HORIZON]+STEP, int(net>0)
                else:
                    mature, label, net = pd.NaT, np.nan, np.nan
                samples.append({'symbol':symbol, 'time':frame.index[i],
                                'mature':mature, 'label':label, 'net_return':net,
                                **features(frame, i, side)})
        print(symbol, 'segments', len(segments), flush=True)
    if db is not None:
        db.close()
    if not samples:
        raise SystemExit('无可用样本')
    rows = pd.DataFrame(samples).sort_values(['time','symbol']).reset_index(drop=True)
    # 全币同一日历边界；只用边界前已成熟的标签。
    first, last = rows.time.min(), rows.mature.max()
    cut1 = first + (last-first)*0.6
    cut2 = first + (last-first)*0.8
    train = rows[(rows.time < cut1) & (rows.mature < cut1)]
    val = rows[(rows.time >= cut1) & (rows.mature < cut2)]
    test = rows[(rows.time >= cut2) & (rows.mature <= end)]
    if min(len(train),len(val),len(test)) < 30 or train.label.nunique() < 2:
        raise SystemExit(f'时间分区样本不足: {len(train)}, {len(val)}, {len(test)}')
    cols = [c for c in rows if c not in ('symbol','time','mature','label','net_return')]
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(train[cols], train.label)
    rows['probability'] = model.predict_proba(rows[cols])[:,1]
    # 特征相对方向编码；反转所有方向相关特征，比较同一时点做多与做空的预测。
    opposite = rows[cols].copy()
    direction_cols = ['side'] + [f'c{j}' for j in range(10)] + [f'body{j}' for j in range(10)]
    opposite[direction_cols] *= -1
    rows['opposite_probability'] = model.predict_proba(opposite)[:,1]
    rows['direction_agrees'] = rows['probability'] > rows['opposite_probability']
    # 不一致的方向直接拒绝；空阈值代表完全不使用模型的原始策略。
    rows['gate_probability'] = rows['probability'].where(rows['direction_agrees'], -1.0)
    probs = {s: g.set_index('time').gate_probability for s,g in rows.groupby('symbol')}
    # 固定候选分位仅使用验证样本；包括不筛选基线，选择按账户均值。
    val_probs = rows.loc[val.index, 'probability']
    candidates = [None] + [float(val_probs.quantile(q)) for q in (0.25,0.5,0.75)]
    valid_results = [path_score(parts, cut1, cut2, t, probs, params, fee, slip) for t in candidates]
    best = max(range(len(candidates)), key=lambda i: valid_results[i]['mean_return_pct'])
    chosen = candidates[best]
    baseline = path_score(parts, cut2, end, None, probs, params, fee, slip)
    filtered = path_score(parts, cut2, end, chosen, probs, params, fee, slip)
    def classification(part):
        p = rows.loc[part.index, 'probability']
        return {'samples':len(part),'positive_rate':float(part.label.mean()),
                'auc':float(roc_auc_score(part.label,p)) if part.label.nunique()==2 else None}
    out.mkdir(parents=True)
    snapshot = pd.concat([f[['open','high','low','close','volume']].assign(symbol=s).reset_index(names='time') for s,f,_ in parts], ignore_index=True)
    snapshot.to_csv(out/'candles.csv', index=False, float_format='%.17g')
    rows.to_csv(out/'samples.csv', index=False, float_format='%.17g')
    with (out/'model.pkl').open('wb') as f:
        pickle.dump({'model':model, 'features':cols, 'threshold':chosen, 'params':params}, f)
    report = {'data_source':'本地 PostgreSQL klines 原生15m；只读快照', 'end_exclusive':str(end),
              'snapshot_sha256':hashlib.sha256((out/'candles.csv').read_bytes()).hexdigest(),
              'coverage':coverage, 'config':{**config, 'parameters_by_symbol':params, 'fee':fee,'slippage':slip},
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'versions':{'python':sys.version, 'numpy':np.__version__, 'pandas':pd.__version__,
                          'sklearn':__import__('sklearn').__version__},
              'split':{'start':str(first),'train_end':str(cut1),'validation_end':str(cut2),'test_end':str(end)},
              'train':classification(train),'validation':classification(val),'test':classification(test),
              'threshold_candidates':candidates,'validation_paths':valid_results,'chosen_threshold':chosen,
              'test_baseline':baseline,'test_selected':filtered,
              'limitations':['标签为16根固定期限净收益代理，非实际策略成交收益；无资金费率','使用原生15m数据，非项目优先1m聚合的完整数据链；跨缺口分段','独立币种账户、边界强制平仓、未模拟共享资金竞争、深度及真实延迟','选阈值前未根据验证期重新训练，测试仅一次；样本和时间范围有限']}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output':str(out),'split':report['split'],'train':report['train'],'validation':report['validation'],'test':report['test'],'threshold':chosen,'baseline':baseline,'selected':filtered}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

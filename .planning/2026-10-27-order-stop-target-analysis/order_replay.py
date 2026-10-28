"""Read-only order audit and strictly bounded order-level exit sensitivity.

Historical entries, quantities and actual close time are fixed. The script only
simulates stops/targets within each actual holding period. Any alternative still
open at the actual exit inherits that exit, so it cannot measure benefits of
holding longer or changes to entry, sizing, portfolio or future orders.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.db import make_engine
from app.features.indicators import build_features
from app.market.data_service import MarketDataService
from app.models import Order, Signal, StrategyConfig
from app.state import all_state
from app.strategy.rules import resolve_params
from app.trading.risk import stop_exit, trailing_stop
from app.utils import dt_ms, interval_ms, round_price

VARIANTS = [(1.5, 2.5), (1.0, 1.5), (1.0, 2.5), (1.5, 1.5), (1.5, 4.0), (2.0, 2.5), (2.0, 3.5)]

def tick(price):
    return 0.01 if price >= 1000 else 0.0001 if price >= 1 else 0.000001

def near(a, b, price):
    return abs(float(a) - float(b)) <= tick(price) * 1.1

def calc_pnl(o, raw, slip, fee):
    side = 1 if o.side == 'LONG' else -1
    fill = raw * (1 - side * slip)
    gross = (fill - o.entry_price) * o.qty * side
    return gross - o.entry_fee - fill * o.qty * fee, round_price(fill)

def build_atrs(frame, step):
    out = pd.Series(0.0, index=frame.index)
    groups = frame.index.to_series().diff().dt.total_seconds().mul(1000).ne(step).cumsum()
    for _, part in frame.groupby(groups):
        if len(part) >= 60:
            out.loc[part.index] = build_features(part)['atr_pct'] * part['close']
    return out

def simulate_window(o, sig, bars, atrs, params, sl, tp, fee, slip):
    side = 1 if o.side == 'LONG' else -1
    step = interval_ms(o.interval)
    created = dt_ms(o.created_at)
    cursor = None
    for row in bars.itertuples():
        ts = int(row.Index.timestamp() * 1000)
        held = (ts - created) / step
        hit = stop_exit(side, row.open, row.open, row.open, sl, tp)
        if hit is None and cursor is not None and ts != cursor + step:
            return {'state':'gap', 'bar':str(row.Index)}
        if hit is None and params['max_hold_bars'] > 0 and held >= params['max_hold_bars']:
            hit = (row.open, 'time_stop')
        if hit is None:
            hit = stop_exit(side, row.open, row.high, row.low, sl, tp)
        if hit:
            pnl, fill = calc_pnl(o, hit[0], slip, fee)
            return {'state':'hit', 'reason':hit[1], 'bar':str(row.Index), 'price':fill, 'pnl':pnl}
        atr = float(atrs.loc[row.Index])
        if math.isfinite(atr) and atr > 0:
            sl = round_price(trailing_stop(side, o.entry_price, sl, row.close, atr, params))
        cursor = ts
    return {'state':'censored', 'reason':'actual_close', 'price':o.close_price, 'pnl':o.realized_pnl}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--as-of', required=True, help='ISO-8601 timestamp with timezone')
    ap.add_argument('--output-dir', type=Path, default=Path(__file__).parent)
    args=ap.parse_args()
    as_of=datetime.fromisoformat(args.as_of.replace('Z','+00:00'))
    if as_of.tzinfo is None:
        raise ValueError('--as-of must include timezone')
    engine=make_engine(read_only=True)
    audit=[]
    with Session(engine) as db:
        state=all_state(db)
        fee=float(state['fee_rate']); slip=float(state['slippage'])
        orders=db.query(Order).filter(Order.status=='CLOSED', Order.closed_at<=as_of).order_by(Order.id).all()
        service=MarketDataService(db)
        for o in orders:
            rec={'id':o.id,'symbol':o.symbol,'strategy':o.strategy,'interval':o.interval,
                 'side':o.side,'actual_pnl':o.realized_pnl,'actual_reason':o.close_reason,
                 'actual_close_price':o.close_price,'created_at':o.created_at.isoformat(),
                 'closed_at':o.closed_at.isoformat()}
            audit.append(rec)
            if o.source != 'paper' or o.close_reason not in ('stop_loss','take_profit'):
                rec['excluded']='unsupported source or actual exit'; continue
            candidates=(db.query(Signal).filter(Signal.symbol==o.symbol, Signal.interval==o.interval,
                Signal.strategy==o.strategy, Signal.direction==o.side,
                Signal.created_at<=o.created_at, Signal.created_at>=o.created_at-timedelta(minutes=1))
                .order_by(Signal.created_at.desc()).limit(10).all())
            match=[]
            side=1 if o.side=='LONG' else -1
            for sig in candidates:
                if near(round_price(sig.price*(1+side*slip)),o.entry_price,o.entry_price) and near(sig.tp,o.tp,o.tp):
                    match.append(sig)
            if len(match)!=1:
                rec['excluded']=f'plausible matching signals: {len(match)}'; continue
            sig=match[0]; step=interval_ms(o.interval); created=dt_ms(o.created_at)
            rec['signal_id_candidate']=sig.id
            rec['candidate_age_seconds']=(o.created_at-sig.created_at).total_seconds()
            if sig.ts+step>created:
                rec['excluded']='signal was not fully closed before entry'; continue
            config=db.query(StrategyConfig).filter(StrategyConfig.code==o.strategy).first()
            params=resolve_params(o.strategy,state,json.loads(config.params or '{}') if config else None,interval=o.interval)
            if params.get('ema_exit') or params.get('partial_tp_r'):
                rec['excluded']='unsupported exit behavior'; continue
            if o.qty <= 0 or not o.entry_fee or not near(o.entry_fee, o.entry_price*o.qty*fee, o.entry_fee):
                rec['excluded']='entry fee inconsistent with current rate'; continue
            try:
                frame=service.load_frame(o.symbol,o.interval,start_ms=min(sig.ts,created)-400*step,
                                         end_ms=dt_ms(o.closed_at),allow_native=False)
                if frame.empty:
                    rec['excluded']='no market data'; continue
                stamps=frame.index.as_unit('ms').asi8
                frame=frame.loc[stamps+step <= dt_ms(o.closed_at)].copy()
                if frame.empty:
                    rec['excluded']='no fully closed bars'; continue
                atrs=build_atrs(frame,step)
                signal_idx=pd.to_datetime(sig.ts,unit='ms',utc=True)
                if signal_idx not in frame.index:
                    rec['excluded']='entry signal bar unavailable'; continue
                atr=float(atrs.loc[signal_idx])
                if not math.isfinite(atr) or atr <= 0:
                    rec['excluded']='entry ATR unavailable'; continue
                original_sl=round_price(sig.price-side*max(atr*params['sl_atr_mult'],sig.price*params['min_sl_pct']))
                original_tp=round_price(sig.price+side*max(atr*params['tp_atr_mult'],sig.price*params['min_tp_pct']))
                rec['atr']=atr
                if not near(original_sl,sig.sl,sig.price) or not near(original_tp,sig.tp,sig.price):
                    rec['excluded']='entry ATR/current settings do not reconstruct saved signal SL/TP'; continue
                stamps=frame.index.as_unit('ms').asi8
                bars=frame.loc[stamps>=created]
                if bars.empty or int(bars.index[0].timestamp()*1000) != ((created+step-1)//step)*step:
                    rec['excluded']='first complete post-entry bar missing'; continue
                base=simulate_window(o,sig,bars,atrs,params,sig.sl,sig.tp,fee,slip)
                rec['base_replay']=base
                if (base['state']!='hit' or base['reason']!=o.close_reason
                    or not near(base['price'],o.close_price,o.close_price)
                    or abs(base['pnl']-o.realized_pnl)>0.02):
                    rec['excluded']='actual exit not reproduced by current rules and available bars'; continue
                results={}
                for sl_mult,tp_mult in VARIANTS:
                    sl=(sig.sl if sl_mult==params['sl_atr_mult'] else
                        round_price(sig.price-side*max(atr*sl_mult,sig.price*params['min_sl_pct'])))
                    tp=(sig.tp if tp_mult==params['tp_atr_mult'] else
                        round_price(sig.price+side*max(atr*tp_mult,sig.price*params['min_tp_pct'])))
                    results[f'{sl_mult:g}/{tp_mult:g}']=simulate_window(o,sig,bars,atrs,params,sl,tp,fee,slip)
                rec['variants']=results
            except Exception as exc:
                rec['excluded']=f'{type(exc).__name__}: {str(exc)[:150]}'
    grouped=defaultdict(lambda:{'count':0,'pnl':0.0,'wins':0})
    for r in audit:
        for name in ('all','reason:'+str(r['actual_reason']),'strategy:'+r['strategy'],'interval:'+r['interval']):
            g=grouped[name];g['count']+=1;g['pnl']+=float(r['actual_pnl'] or 0);g['wins']+=int((r['actual_pnl'] or 0)>0)
    eligible=[r for r in audit if 'variants' in r]
    def summarize(rows):
        summ={}
        for a,b in VARIANTS:
            key=f'{a:g}/{b:g}'
            valid=[r for r in rows if r['variants'][key]['state']!='gap']
            summ[key]={'orders':len(valid),'pnl':sum(float(r['variants'][key]['pnl'] or 0) for r in valid),
                       'actual_pnl_for_same_orders':sum(float(r['actual_pnl'] or 0) for r in valid),
                       'triggered':sum(r['variants'][key]['state']=='hit' for r in valid),
                       'censored_at_actual_close':sum(r['variants'][key]['state']=='censored' for r in valid)}
        return summ
    summ=summarize(eligible)
    by_interval={interval:summarize([r for r in eligible if r['interval']==interval])
                 for interval in sorted({r['interval'] for r in eligible})}
    payload={'as_of':as_of.isoformat(),'fee_rate':fee,'slippage':slip,'groups':dict(grouped),
             'matched_orders':len(eligible),'variants':summ,'variants_by_interval':by_interval,'orders':audit,
             'limitations':['Signal match is inferred, not an order foreign key.',
                'Current configuration is assumed; initial signal prices/ATR must reconstruct within one rounding tick.',
                'Only closed paper orders whose actual reason and exit price replay successfully are compared.',
                'First partial entry bar is omitted; only complete bars up to actual close are used.',
                'Alternative positions still open at actual close inherit the actual close and PnL; future paths are not evaluated.',
                'Actual entry, quantity and fees are fixed; changed sizing, later entries, portfolio and account halt are not simulated.',
                'This is a tiny in-sample sensitivity analysis, not selection or out-of-sample validation.']}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'order-replay-audit.json'
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    lines=['# 已平订单出场敏感性审计', '', f'截至 {as_of.isoformat()}；仅只读查询。',
           '', '## 实际订单', '', '| 组别 | 笔数 | 盈利笔数 | 净盈亏 USDT |', '|---|---:|---:|---:|']
    for k,v in grouped.items():
        lines.append(f"| {k} | {v['count']} | {v['wins']} | {v['pnl']:.4f} |")
    lines.extend(['', '## 限定实际持仓期的敏感性', '',
                  f'已平订单 {len(audit)} 笔；其中 {len(eligible)} 笔可基于候选信号、历史行情及当前配置复现原出场类型、价格和盈亏。',
                  '参数值是 SL/TP 的 ATR 倍数。每组均固定同一批订单的实际方向、入场价格和数量；该比较不能模拟改变止损带来的仓位变化。',
                  '', '| SL/TP | 笔数 | 反事实盈亏 USDT | 同批实际盈亏 USDT | 原平仓时间截尾笔数 |', '|---|---:|---:|---:|---:|'])
    for k,v in summ.items():
        lines.append(f"| {k} | {v['orders']} | {v['pnl']:.4f} | {v['actual_pnl_for_same_orders']:.4f} | {v['censored_at_actual_close']} |")
    for interval,vals in by_interval.items():
        lines.extend(['', f'### {interval}', '', '| SL/TP | 笔数 | 反事实盈亏 USDT | 同批实际盈亏 USDT | 截尾 |', '|---|---:|---:|---:|---:|'])
        for k,v in vals.items():
            lines.append(f"| {k} | {v['orders']} | {v['pnl']:.4f} | {v['actual_pnl_for_same_orders']:.4f} | {v['censored_at_actual_close']} |")
    lines.extend(['', '## 方法与限制', '',
        '- 在开仓前一分钟内按币种、方向、周期、策略、入场价（含当前滑点）和止盈价格推断候选信号；订单没有信号外键，此匹配不是确定证明。',
        '- 用入场前历史 K 线重新计算 ATR，以当前策略配置验证信号的原止损止盈；使用匹配信号的原始 SL/TP 和固定实际数量重放。',
        '- 只回放开仓所在部分 K 线之后的完整 K 线，包含止损优先、移动止损、当前费率及滑点；复现的平仓原因、价格和盈亏均需与实际接近，才纳入。',
        '- 回放最多到实际平仓时间；变体若尚未触发，则假设以原实际平仓价格退出（截尾），并沿用原实际盈亏；这不能评估更长持有期的收益。',
        '- 真实订单可能使用过不同参数；信号原始 SL/TP、最近信号匹配及当前参数验证只能提高可信度，无法恢复当时完整配置。',
        '- 回放不模拟按新止损调整数量、重新开仓、未来信号、共享资金或账户熔断；K线无法还原同一根内价格路径。',
        '- 这是不足 20 笔的事后敏感性分析，结果不能证明样本外盈利或作为直接调参依据。',
        '', '## 复现', '', '```powershell',
        f'.\\.venv\\Scripts\\python.exe .planning\\2026-10-27-order-stop-target-analysis\\order_replay.py --as-of {as_of.isoformat()}',
        '```', ''])
    (args.output_dir/'order-replay-report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'path':str(path),'as_of':payload['as_of'],'groups':payload['groups'],
                      'matched_orders':len(eligible),'variants':summ,'variants_by_interval':by_interval,
                      'excluded':{str(r['id']):r['excluded'] for r in audit if 'excluded' in r}},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()

"""P18-007 口径更正；只读重算，另存成对产物，保留原报告和台账。"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'backend' / 'scripts'))
from app.db import make_engine
from app.research import fit_data as fd
from app.research.fit_membership import pit_tiers_by_month, tier_disagreement
from fit_matrix_report import append_ledger, git_state


def main():
    old_path = ROOT / 'docs/indicator-fit-membership.json'
    uni_path = ROOT / 'docs/fit-universe.json'
    json_path = ROOT / 'docs/indicator-fit-membership-correction.json'
    md_path = ROOT / 'docs/indicator-fit-membership-correction.md'
    if json_path.exists() or md_path.exists():
        raise SystemExit('更正产物已存在，拒绝覆盖；新一轮必须使用新实验编号和文件名')
    old = json.loads(old_path.read_text(encoding='utf-8'))
    uni = json.loads(uni_path.read_text(encoding='utf-8'))['scopes']['A']
    end_ms = int(old['window_anchor']['end_ms'])
    ref = list(uni['common_universe'])
    fixed = dict(uni['fixed_tiers'])
    engine = make_engine(read_only=True)
    db = sessionmaker(bind=engine)()
    try:
        read_only = db.execute(text('SHOW transaction_read_only')).scalar()
        if read_only != 'on':
            raise RuntimeError('研究连接必须只读')
        db.execute(text("SET LOCAL statement_timeout = '120s'"))
        window = fd.monthly_quote_volume(db, '1h', ref, days=old['pool_window_days'], end_ms=end_ms)
        history = fd.monthly_quote_volume(db, '1h', ref, days=old['history_lookback_days'], end_ms=end_ms)
    finally:
        db.close()
        engine.dispose()
    months = sorted(window['ym'].astype(str).unique())
    short_all = pit_tiers_by_month(window.assign(partial=False))
    long_all = pit_tiers_by_month(history)
    short = tier_disagreement(fixed, {m: short_all[m] for m in months if m in short_all})
    long = tier_disagreement(fixed, {m: long_all[m] for m in months if m in long_all})
    buggy = fd.assign_fixed_tiers(window.assign(partial=False))
    corrected = fd.assign_fixed_tiers(window)
    changed = sum(fixed[s] != corrected[s] for s in fixed.keys() & corrected.keys())
    absent = {}
    for m in months:
        event = set(window.loc[window['ym'].astype(str) == m, 'symbol'].astype(str))
        comparable = set(short_all[m]['tiers']) & set(fixed)
        absent[m] = sorted(comparable - event)
    prior_short = old['lookahead_isolated']
    prior_long = old['pit']
    compare_keys = ('comparable_total', 'disagree_total', 'disagree_rate')
    comparison = {
        'strict_prior_totals_match_p18_007': all(short[k] == prior_short[k] for k in compare_keys),
        'long_prior_totals_match_p18_007': all(long[k] == prior_long[k] for k in compare_keys),
        'buggy_tiers_match_frozen': buggy == fixed,
        'corrected_changed_count_matches_p18_007': changed == len(old['corrected_recompute']['tier_delta']['changed']),
    }
    payload = {
        'experiment_id': 'P18-008', 'parent_experiment_id': 'P18-007',
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'scope_sha': old['scope_sha'], 'decision': 'BLOCKED', 'db_access': 'read_only',
        'transaction_read_only': read_only,
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_path, uni_path)},
        'window': {'end_ms': end_ms, 'semantics': 'open_time >= end_ms - days*DAY_MS and open_time <= end_ms',
                   'anchor_caveat': '来自当时读取的可变 Q 池规则时间戳取整，并非已冻结 A 轮 max(open_time)；成员相同无法证明时间戳未变'},
        'strict_prior': short, 'long_prior': long, 'comparison': comparison,
        'strict_prior_comparable_without_event_month_rows': absent,
        'corrected_tier_changed': changed,
        'limitations': ['严格前月对照排除了事件月及之后月份，包含样本月份与分档比较集合变化，不能分离未来信息效应。',
                        '参考币集是事后 common universe，未证明历史上市、退市和成员资格为 PIT。',
                        '可比分母为固定档和 prior 档的交集，不要求币在事件月有 K 线；无档计数口径不同。',
                        '窗口锚点未经独立冻结；数据库历史也可能被更新，当前重算不能证明 A 轮原始输入。',
                        '档位差异不是收益偏差；未重算收益、未执行回测或参数试验。'],
        'backtest_status': 'NOT_RUN', 'parameter_trials': 0, 'stats_recomputed': False,
    }
    lines = ['# P18-007 成员审计口径更正', '',
             f"生成：{payload['generated_at']}；实验：P18-008；结论：**BLOCKED**。", '',
             'P18-007 的“隔离未来信息”解释撤回：代码同时排除了事件月，不能把不一致率解释为纯未来信息效应或收益影响。',
             '原报告和既有台账保留，本报告更正其解释。以下为当前只读数据库、候选锚点和修复上界后的诊断。', '',
             '| 对照 | 可比币月 | 不一致 | 不一致率 |', '|---|---:|---:|---:|',
             f"| 严格前月短窗 | {short['comparable_total']} | {short['disagree_total']} | {short['disagree_rate']:.2%} |",
             f"| 严格前月长窗 | {long['comparable_total']} | {long['disagree_total']} | {long['disagree_rate']:.2%} |", '',
             f"候选锚点闭区间重算与 P18-007 总数一致性：{comparison}；修复残月判定后改档 {changed} 币。", '',
             '可比分母含 prior 有档但事件月无 K 线的币：' + '；'.join(f'{m}={len(v)}' for m,v in absent.items()) + '。', '',
             '## 限制', ''] + [f'- {item}' for item in payload['limitations']]
    lines += ['', '费用沿用每边手续费 0.0004、滑点 0.0002；本次不计算收益。', '']
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    record = {'timestamp': payload['generated_at'], 'experiment_id': 'P18-008', 'parent_experiment_id': 'P18-007',
              'supersedes_interpretation_of': 'P18-007', 'strategy_id': None, 'code_commit': git_state(),
              'stage': 'P18 membership audit correction', 'decision': 'BLOCKED',
              'dataset_version': 'local PostgreSQL klines read_only; estimated anchor',
              'universe_snapshot': 'docs/fit-universe.json A common universe, post-hoc',
              'interval': '1h', 'parameters': {'scope_sha': old['scope_sha'], 'parameter_trials': 0},
              'fee_model': '0.0004 per side', 'slippage_model': '0.0002 per side',
              'funding_model': 'NOT_MODELLED; no backtest', 'metrics': {'strict_prior': short['disagree_rate'], 'long_prior': long['disagree_rate'], 'comparison': comparison},
              'failure_reason': 'estimated anchor and post-hoc universe; strict prior removes event month and later months',
              'next_action': '冻结真实 PIT 成员及方法后重新评估；其余统计与样本外门仍阻塞',
              'artifacts': [str(md_path.relative_to(ROOT)), str(json_path.relative_to(ROOT))]}
    print(append_ledger(record, ROOT / 'research/experiments.jsonl'))
    print(json.dumps({'comparison': comparison, 'strict_prior': short['disagree_rate'], 'long_prior': long['disagree_rate'], 'changed': changed}, ensure_ascii=False))


if __name__ == '__main__':
    main()

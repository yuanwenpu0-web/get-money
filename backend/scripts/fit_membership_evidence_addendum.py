"""针对 P18-007/008 的覆盖口径补充说明；不改写原始产物或重算收益。"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'backend' / 'scripts'))
from app.research.fit_data import MONTH_PARTIAL_FILL
from fit_matrix_report import append_ledger, git_state


def main():
    source = ROOT / 'docs/indicator-fit-membership-correction.json'
    data_path = ROOT / 'docs/indicator-fit-membership-evidence-addendum.json'
    report_path = ROOT / 'docs/indicator-fit-membership-evidence-addendum.md'
    if data_path.exists() or report_path.exists():
        raise SystemExit('产物已存在；禁止覆盖既有实验')
    prior = json.loads(source.read_text(encoding='utf-8'))
    threshold = float(MONTH_PARTIAL_FILL)
    if threshold != 0.7:
        raise SystemExit('覆盖门槛已变更，不能用当前常量说明先前实验')
    short, long = prior['strict_prior'], prior['long_prior']
    payload = {
        'experiment_id': 'P18-009', 'parent_experiment_id': 'P18-008',
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'stage': 'membership audit evidence clarification', 'decision': 'BLOCKED',
        'scope_sha': prior['scope_sha'],
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_file': str(source.relative_to(ROOT)),
        'quality': {'long_prior_minimum_bar_fraction': threshold,
                    'long_prior_rule': '此前自然月每币柱数不少于该月应有柱数的70%，不保证月内无缺口或历史交易资格',
                    'short_prior_rule': '复刻 A 轮 partial 恒为 False，包含未达70%门槛的残月；仅用于口径对照'},
        'denominators': {'short_comparable': short['comparable_total'], 'short_disagree': short['disagree_total'],
                         'long_comparable': long['comparable_total'], 'long_disagree': long['disagree_total'],
                         'definition': '固定档与各自此前分档的交集；并非实际事件数，也不强制事件月存在数据',
                         'not_directly_comparable': '两路使用不同历史月份、覆盖过滤和可比分母',
                         'missing_counts': '无档计数来自事件月出现且此前无档的币，与可比分母不构成统一总体；原因未逐币确证'},
        'evidence_limits': ['实际 A 轮 max(open_time) 与原始行情输入版本尚无可靠冻结证据；池规则墙钟时间取整仅为估计锚点。',
                            '参考币集为事后 common universe；无历史上市/下架事件表及 ticker 快照，不能确认历史可交易资格。',
                            '原审计的池窗口天数不能代表实际事件占比或逐币覆盖率，亦不能证明等于各周期事件窗口。',
                            '比较不一致率并非独立未来信息效应或收益偏差；没有回测与参数试验。'],
        'backtest_status': 'NOT_RUN', 'parameter_trials': 0, 'stats_recomputed': False,
    }
    lines = ['# P18 成员审计证据补充', '',
             f"生成：{payload['generated_at']}；结论：**BLOCKED**。本文件补充 P18-007 和 P18-008 的解释，不重算数据。", '',
             '## 覆盖门槛与对照', '',
             f"- 长窗使用此前自然月中每币柱数达到应有柱数 {threshold:.0%} 的月份；这不等于整月无缺口。",
             '- 短窗为复刻原运行的溢出口径，强制所有月份 `partial=False`，因此残月也参与分档；它没有通过上述覆盖门槛。',
             '- 两路都排除事件月及之后的月份，且参考币集事后固定；差异不能作为纯粹未来信息效应或收益影响。', '',
             '| 对照 | 条件可比币月 | 不一致 |', '|---|---:|---:|',
             f"| 短窗 | {short['comparable_total']} | {short['disagree_total']} |",
             f"| 长窗 | {long['comparable_total']} | {long['disagree_total']} |", '',
             '可比分母是固定档与对应此前档的交集，并非实际事件数；无档币使用另一集合，不能简单相加。两路分母和质量过滤不同，不应直接把比例差视为时间或覆盖效应。', '',
             '## 尚未解决的证据缺口', ''] + [f'- {item}' for item in payload['evidence_limits']]
    lines += ['', '原报告保留为历史记录；本补充和 P18-008 一起限定其解释。', '']
    data_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    record = {'timestamp': payload['generated_at'], 'experiment_id': 'P18-009', 'parent_experiment_id': 'P18-008',
              'supersedes_interpretation_of': 'P18-007', 'strategy_id': None, 'code_commit': git_state(),
              'stage': payload['stage'], 'decision': payload['decision'],
              'dataset_version': 'P18-008 saved audit only; no database query',
              'universe_snapshot': 'A common universe, post-hoc', 'interval': '1h',
              'parameters': {'scope_sha': prior['scope_sha'], 'parameter_trials': 0},
              'fee_model': '0.0004 per side', 'slippage_model': '0.0002 per side',
              'funding_model': 'NOT_MODELLED; no backtest',
              'metrics': {'long_prior_month_minimum_bar_fraction': threshold, 'stats_recomputed': False},
              'failure_reason': 'historical eligibility, actual original anchor, and statistical gates not established',
              'next_action': '寻回可验证的历史行情末端、数据版本及历史合约资格；缺失时保持 BLOCKED',
              'artifacts': [str(report_path.relative_to(ROOT)), str(data_path.relative_to(ROOT))]}
    print(append_ledger(record, ROOT / 'research/experiments.jsonl'))
    print(f"P18-009 threshold={threshold:.0%}; short={short['disagree_total']}/{short['comparable_total']}; long={long['disagree_total']}/{long['comparable_total']}")


if __name__ == '__main__':
    main()

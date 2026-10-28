"""全币种 × 全策略 × 8 窗口（近一年 walk-forward）的币种分类分析。

目标：对每个币选出「该币 8 窗口均值最高的策略」，并做 holdout 验证。
输出：
 1. per-coin-classification.json — 每个币的最优策略+均值+胜率+holdout
 2. 终端打印：族群效应 Top/Worst、holdout 通过的币、路由建议

口径：
  - PF = sum(wins) / |sum(losses)|（传统口径，不是 avg_win/avg_loss）
  - holdout = 最后 1 个窗口（最靠近今天），训练 = 前 7 个窗口
  - 分类纪律：holdout 窗口 return_pct > 0.001 且 positions >= 3 才算「通过」
"""
import json, statistics as s, sys
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

DATA = ROOT / "docs" / "strong-pullback-exit-ab-8x60d.json"
d = json.load(open(DATA, encoding='utf-8'))

VARIANTS = ['partial', 'fixed', 'trend_follow', 'mtf_resonance']
# partial/fixed 都是 strong_pullback 策略（参数不同），统一按策略名归类
STRATEGY_NAME = {
    'partial': 'strong_pullback',
    'fixed': 'strong_pullback',
    'trend_follow': 'trend_follow',
    'mtf_resonance': 'mtf_resonance',
}

# ============================================================
# 1. 数据汇总：symbol × strategy → 8 个窗口的 return_pct
# ============================================================
coin_strategy_returns = defaultdict(lambda: defaultdict(list))
coin_strategy_positions = defaultdict(lambda: defaultdict(list))

for variant in VARIANTS:
    for r in d['records'][variant]:
        if 'error' not in r:
            sym = r['symbol']
            strat = STRATEGY_NAME[variant]
            coin_strategy_returns[sym][strat].append(r['return_pct'])
            coin_strategy_positions[sym][strat].append(r.get('positions', 0))

symbols = sorted(coin_strategy_returns.keys())
print(f"=== 全币种分类（{len(symbols)} 币 × 8 窗口 × 4 策略）===")
print(f"窗口: {d['window_ends']}")
print()

# ============================================================
# 2. 每币选最优策略（全 8 窗口均值最高）
# ============================================================
per_coin = []
for sym in symbols:
    strategies = coin_strategy_returns[sym]
    # 确保每个策略至少有 8 个窗口
    valid = {k: v for k, v in strategies.items() if len(v) >= 8}
    if not valid:
        continue
    # 找均值最高的
    best_strat = max(valid, key=lambda k: s.mean(valid[k]))
    best_returns = valid[best_strat]
    mean_ret = s.mean(best_returns)
    median_ret = s.median(best_returns)
    pos_wins = sum(1 for x in best_returns if x > 0)
    total_positions = sum(coin_strategy_positions[sym][best_strat])
    # PF（传统口径：sum(wins)/|sum(losses)| — 用 return_pct 近似）
    wins = [x for x in best_returns if x > 0]
    losses = [x for x in best_returns if x < 0]
    pf = sum(wins) / abs(sum(losses)) if losses else float('inf') if wins else 0.0

    per_coin.append({
        'symbol': sym,
        'best_strategy': best_strat,
        'mean_return_pct': round(mean_ret, 4),
        'median_return_pct': round(median_ret, 4),
        'win_rate_8w': pos_wins,  # 8 个窗口里赢了几个
        'positions': total_positions,
        'pf': round(pf, 3),
        'returns_8w': [round(x, 4) for x in best_returns],
    })

per_coin.sort(key=lambda x: -x['mean_return_pct'])

# ============================================================
# 3. Holdout 验证：训练 7 窗 → 选策略 → 第 8 窗 OOS 测
# ============================================================
HO_IDX = -1  # 最后一个窗口（最靠近今天）
TRAIN_IDX = slice(0, 7)  # 前 7 个窗口

holdout_results = []
for sym in symbols:
    strategies = coin_strategy_returns[sym]
    valid = {k: v for k, v in strategies.items() if len(v) >= 8}
    if not valid:
        continue
    train_means = {k: s.mean(v[TRAIN_IDX]) for k, v in valid.items()}
    best_train = max(train_means, key=train_means.get)
    ho_ret = valid[best_train][HO_IDX]
    ho_pos = coin_strategy_positions[sym][best_train][HO_IDX] if len(coin_strategy_positions[sym][best_train]) > HO_IDX else 0
    train_mean = train_means[best_train]
    # 通过判定：holdout ret > 0.001 且 positions >= 3
    passed = ho_ret > 0.001 and ho_pos >= 3
    holdout_results.append({
        'symbol': sym,
        'strategy': best_train,
        'train_mean_pct': round(train_mean, 4),
        'holdout_ret_pct': round(ho_ret, 4),
        'holdout_pos': ho_pos,
        'passed': passed,
    })

holdout_results.sort(key=lambda x: -x['train_mean_pct'])

print(f"\n=== 每币最优策略（8 窗口均值） ===")
print(f"{'Symbol':14s} {'Strategy':16s} {'Mean%':>8s} {'Med%':>8s} {'Win8w':>6s} {'Pos':>6s} {'PF':>7s} {'PassHO':>7s}")
for pc, hr in zip(per_coin, holdout_results):
    flag = "OK" if hr['passed'] else ("WARN" if hr['holdout_ret_pct'] > 0 else "FAIL")
    print(f"{pc['symbol']:14s} {pc['best_strategy']:16s} {pc['mean_return_pct']:+7.3f} {pc['median_return_pct']:+7.3f} {pc['win_rate_8w']:>2d}/8 {pc['positions']:>6d} {pc['pf']:7.3f}  {flag:5s} {hr['holdout_ret_pct']:+7.3f}%")

# ============================================================
# 4. 族群效应（CoinType × 最优策略 分布）
# ============================================================
from app.strategy.coin_router import COIN_TYPES, BLACKLIST
output_path = ROOT / "docs" / "coin-classification-8x60d.json"

type_strategy = defaultdict(lambda: defaultdict(list))
for pc in per_coin:
    ct = COIN_TYPES.get(pc['symbol'], 'Unknown')
    type_strategy[ct][pc['best_strategy']].append(pc['mean_return_pct'])

print(f"\n=== 族群效应（每类型 × 最优策略均值分布）===")
for ct in sorted(type_strategy.keys()):
    if ct == 'Unknown':
        continue
    print(f"\n【{ct}】({len(type_strategy[ct]['strong_pullback']) + len(type_strategy[ct]['trend_follow']) + len(type_strategy[ct]['mtf_resonance'])} 币)")
    for strat in ['strong_pullback', 'trend_follow', 'mtf_resonance']:
        rets = type_strategy[ct].get(strat, [])
        if rets:
            print(f"  {strat:16s}: n={len(rets):>2d} mean={s.mean(rets):+7.3f}% median={s.median(rets):+7.3f}%")

# ============================================================
# 5. Holdout 通过的币（真正有 edge 的）
# ============================================================
ho_passed = [h for h in holdout_results if h['passed']]
ho_warn = [h for h in holdout_results if h['holdout_ret_pct'] > 0 and h['holdout_pos'] < 3]
ho_fail = [h for h in holdout_results if h['holdout_ret_pct'] <= 0]

print(f"\n=== Holdout 验证（前 7 窗选策略 → 第 8 窗 OOS）===")
print(f"  ✅ 通过: {len(ho_passed)} 币（holdout ret>0 且 pos≥3）")
print(f"  ⚠️ 零交易/微正: {len(ho_warn)} 币")
print(f"  ❌ 未通过: {len(ho_fail)} 币")

if ho_passed:
    print(f"\n✅ Holdout 通过 — 确认真正 edge 的币:")
    for h in ho_passed:
        ct = COIN_TYPES.get(h['symbol'], '?')
        print(f"  {h['symbol']:14s} [{ct:6s}] {h['strategy']:16s} train={h['train_mean_pct']:+7.3f}% holdout={h['holdout_ret_pct']:+7.3f}%")

# ============================================================
# 6. 黑名单候选（8/8 窗口全负的币）
# ============================================================
all_negative = []
for pc in per_coin:
    if pc['win_rate_8w'] == 0 and pc['mean_return_pct'] < -0.1:
        all_negative.append(pc['symbol'])
print(f"\n⚠️ 建议拉黑（8/8 窗口最优策略仍全负且 mean < -0.1%）: {len(all_negative)} 币")
for sym in all_negative:
    ct = COIN_TYPES.get(sym, '?')
    print(f"  {sym:14s} [{ct:6s}]")

# ============================================================
# 7. 路由建议输出
# ============================================================
print(f"\n=== 🎯 路由建议 ===")
print(f"holdout 通过的 → 精确路由；族群均值最优且 n≥2 → 族群路由；建议拉黑 → 黑名单")

# 精确路由
print(f"\n精确路由（holdout 验证通过）:")
for h in ho_passed:
    print(f'  "{h["symbol"]}": {{"strategy": "{h["strategy"]}", "evidence": "8wf ho={h["holdout_ret_pct"]:+.2f}%"}}')

# 更新族群路由建议（族群均值最优且 n≥2）
print(f"\n族群路由建议（更新 TYPE_ROUTES）:")
for ct in sorted(type_strategy.keys()):
    if ct == 'Unknown':
        continue
    strat_means = {strat: s.mean(rets) for strat, rets in type_strategy[ct].items() if len(rets) >= 2}
    if strat_means:
        best = max(strat_means, key=strat_means.get)
        print(f'  "{ct}": "{best}"  (mean={strat_means[best]:+.3f}%, n≥2)')

# 保存完整结果
output = {
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'total_coins': len(symbols),
    'windows': d['window_ends'],
    'per_coin_classification': per_coin,
    'holdout_validation': holdout_results,
    'blacklist_candidates': all_negative,
    'ho_passed': ho_passed,
    'ho_warn': ho_warn,
    'ho_fail_count': len(ho_fail),
}
output_path.write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8'
)
print(f"\n完整结果已保存: {output_path}")

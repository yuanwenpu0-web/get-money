# P18 池级候选整体方向门复核

生成：2026-11-13T02:17:17+00:00。

结论：必要方向门已执行；完整研究仍为BLOCKED_OTHER_GATES，收益verdict为inconclusive（未新评估）。

## 冻结口径

协议在查看修订计数前冻结，协议指纹与原始输入/代码哈希见同名json。
按同周期、同列、同方向、同h、同基准匹配all池；只要求方向化整体净超额有限且严格大于0。
缺配对或非有限值记BLOCKED，整体非正记FAIL；两者的复核标签均降为exploratory。
通过该必要门仍不能说明整体统计显著或满足全部预登记。

窗口是原A轮90天筛选样本；币池、方向、成本、参数均不重选，数据库访问0、参数试验0。
成本沿用原统计名义本金往返0.12%（fee_rate=0.0004/slippage=0.0002，每边），不新增净收益。

## 结果

- 原池级candidate行：161；方向门计数：{'PASS': 120, 'FAIL': 41}。
- 降级行：41。原55列口径下逐列选中格的配对复核：{'PASS': 35, 'FAIL': 20}；降级20列。
- 原先按整体跨周期最强格看到的6个反例，不能代替本轮同坐标配对的计数。

| 周期 | PASS | FAIL | BLOCKED |
|---|---:|---:|---:|
| 5m | 2 | 2 | 0 |
| 15m | 30 | 6 | 0 |
| 30m | 23 | 17 | 0 |
| 1h | 65 | 16 | 0 |

## 原池特有选中格的降级明细

| 列 | 周期 | 池 | 方向 | h | 基准 | 池净% | 匹配整体净% | 原标签 | 复核标签 |
|---|---|---|---:|---:|---|---:|---:|---|---|
| `mtf_wavetrend_tide_wave_ripple.short_watch` | 1h | major | -1 | 24 | unconditional | 0.8894 | -0.2579 | candidate | exploratory |
| `smart_money_concepts.is_bearish_ob` | 1h | liq_q1 | -1 | 24 | unconditional | 0.7834 | -0.1923 | candidate | exploratory |
| `capital41_fusion_oscillator.wr_short` | 1h | liq_q3 | -1 | 96 | unconditional | 0.8659 | -0.3112 | candidate | exploratory |
| `capital41_fusion_scalp.wr_short` | 1h | liq_q3 | -1 | 96 | unconditional | 0.8659 | -0.3112 | candidate | exploratory |
| `cci_advanced.bear_strong` | 30m | major | -1 | 96 | cross_section | 0.7553 | -0.3813 | candidate | exploratory |
| `btc_scalping_signals.long_signal` | 1h | liq_q4 | 1 | 96 | unconditional | 3.1175 | -0.4793 | candidate | exploratory |
| `alert_engine_webhook.bear_signal` | 30m | major | -1 | 24 | cross_section | 0.3433 | -0.0248 | candidate | exploratory |
| `alert_engine_webhook.raw_bear` | 30m | major | -1 | 24 | cross_section | 0.3433 | -0.0248 | candidate | exploratory |
| `mtf_stochrsi_pair_score.short_trigger` | 30m | liq_q4 | -1 | 24 | unconditional | 0.4969 | -0.0356 | candidate | exploratory |
| `btc_scalping_signals.short_signal` | 1h | major | -1 | 24 | cross_section | 0.4127 | -0.1615 | candidate | exploratory |
| `stoch_rsi_advanced.divergence_bull` | 1h | liq_q4 | 1 | 24 | unconditional | 0.6082 | -0.1124 | candidate | exploratory |
| `gcr_strategy_scanner_tradingview.bearish_divergence` | 15m | liq_q4 | -1 | 24 | unconditional | 0.4818 | -0.3299 | candidate | exploratory |
| `live_market_scanner_v7.valid_short` | 1h | liq_q2 | -1 | 24 | cross_section | 0.4359 | -0.0498 | candidate | exploratory |
| `rsi_advanced.bear_mid_cross` | 1h | major | -1 | 24 | cross_section | 0.3994 | -0.0058 | candidate | exploratory |
| `crypto_amt_toolkit.confirmed_short` | 30m | liq_q4 | -1 | 24 | cross_section | 0.3187 | -0.0743 | candidate | exploratory |
| `tsi_advanced.bull_extreme_weak` | 1h | liq_q2 | 1 | 24 | unconditional | 0.9973 | -0.0275 | candidate | exploratory |
| `chandelier_flip_radar.bear_trap` | 30m | major | -1 | 6 | unconditional | 0.1624 | -0.0904 | candidate | exploratory |
| `capital41_duo_oscillator.bear_cross` | 5m | liq_q2 | -1 | 96 | cross_section | 0.2147 | -0.0602 | candidate | exploratory |
| `vein_reversal_zones.short_signal` | 30m | liq_q1 | -1 | 24 | unconditional | 0.3589 | -0.356 | candidate | exploratory |
| `mtf_wavetrend_tide_wave_ripple.long_watch` | 1h | major | 1 | 24 | unconditional | 0.2426 | -0.3033 | candidate | exploratory |

## 局限与启用决策

- 仅补必要的整体净超额正号门，不等同于整体显著性/稳定性通过。
- 只收紧原candidate标签，不重新筛指标/周期/池/h，不重算原统计；按原产物已舍入净值判断，0或-0保守不通过。
- 保留原A轮统计和台账；通过方向门的行仍不能交易化或声称独立OOS。
- 原池成员时点、交互和多重比较问题未在本轮修复。

未跑交易回测、未新增置信区间CI或随机基线；default_enabled/绑定不改，未启用paper/live。

## 验证

- `output/fit-pool-review-focused-pytest.log`：82 passed in 1.29s
- `output/fit-pool-review-full-pytest.log`：3 failed, 2165 passed, 5 warnings in 76.69s (0:01:16)

复现：`.venv/Scripts/python.exe backend/scripts/fit_pool_review.py`；禁止覆写已冻结协议。

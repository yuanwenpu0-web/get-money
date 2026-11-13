# P18 后续验证就绪审计

生成：2026-11-13T01:44:52+00:00；结论：**BLOCKED**。

审计与数据边界核验已执行；回测 NOT_RUN，参数试验 0，paper/live 未启动。
原始统计与台账不改写；原 candidate 和池特有数量是实现标签，不能当完整预登记发现。

## 数据实查

预登记要求 12 个自然月训练 + 3 个自然月验证。以下是全库覆盖上界，非逐币质量通过。

| 周期 | 币数 | bar数 | 全库跨度（天） | 历史门 |
|---|---:|---:|---:|---|
| 5m | 80 | 2176529 | 95.44 | BLOCKED |
| 15m | 521 | 3282355 | 316.85 | BLOCKED |
| 30m | 102 | 1008205 | 316.40 | BLOCKED |
| 1h | 535 | 4981196 | 418.38 | BLOCKED |

90 天窗口已用于选择列、周期、池和 h，不能重新切分后称为这些候选的独立 OOS。

## 方法审计

- **max_t / NOT_IMPLEMENTED**：预登记的 Westfall–Young max-T 未实施，原 candidate 仅是当前实现的标签，不能声称完整通过预登记。 证据：`docs/indicator-evaluation-plan.md:150`、`backend/app/research/fit_stats.py:382`。
- **interaction / METHOD_DEVIATION**：交互 p 是池内外逐币聚合均值的 bootstrap 比较，未做预登记的先整体交互再分格，也未校正交互 p 的多重比较。 证据：`docs/indicator-evaluation-plan.md:194`、`backend/app/research/fit_stats.py:312`。
- **membership / POSTHOC_GROUPS**：Q 池按评估窗内月成交额事后固定分档，不是事件时点可得的 PIT 成员；major/alt 统计时仍读取可变数据库池。 证据：`backend/scripts/fit_prep.py:316`、`backend/scripts/indicator_fit_matrix.py:90`。
- **pooled_direction / MISSING_GATE**：子集 candidate 未检查同列整体方向；整体无效的子集按预登记只能是复核线索。反例清单保留原始标签，不事后改统计数字。 证据：`docs/indicator-evaluation-plan.md:206`、`backend/app/research/fit_stats.py:430`。
- **interval_attribution / CONFOUNDED**：跨所有池选最强格混合了币种构成、周期和实际持有时间；最强格分布不能单独证明周期适配。 证据：`backend/scripts/fit_matrix_report.py:119`、`backend/scripts/indicator_fit_matrix.py:77`。
- **log_warning / RESOLVED_FOR_TPS_5M**：已定位RIVERUSDT的TPS未选分支log(0)，修复后80币完整输出与旧函数相同，选定缓存统计不变；不解除其余阻塞。 证据：`docs/indicator-fit-log-diagnosis.md`。

## 整体非正的池特有标签反例

以下比的是报告已选出的整体最强格；不是同 h/基准配对，也不是完整方向门复算。

| 列 | 周期 | 池 | 报告整体净超额% | 审计解释 |
|---|---|---|---:|---|
| `mtf_wavetrend_tide_wave_ripple|short_watch` | 1h | major | -0.1202 | EXPLORATORY_ONLY |
| `smart_money_concepts|is_bearish_ob` | 1h | liq_q1 | -0.0929 | EXPLORATORY_ONLY |
| `capital41_fusion_oscillator|wr_short` | 1h | liq_q3 | -0.1146 | EXPLORATORY_ONLY |
| `capital41_fusion_scalp|wr_short` | 1h | liq_q3 | -0.1146 | EXPLORATORY_ONLY |
| `btc_scalping_signals|long_signal` | 1h | liq_q4 | -0.1072 | EXPLORATORY_ONLY |
| `mtf_wavetrend_tide_wave_ripple|long_watch` | 1h | major | -0.0907 | EXPLORATORY_ONLY |

## 后续动作

1. log告警工程复核已完成；继续补齐其余统计方法与PIT门槛。
2. 冻结方法修订及新 scope；补 max-T、先整体交互的校正、PIT 分档和整体方向门。
3. 按每个训练折重新选择指标/周期/池/持有期；不得把90天全窗冠军倒灌早期折。
4. 补足预登记历史并逐币查密度，或登记新的探索性短窗方案，不能将短窗结果冒充原协议通过。
5. 冻结候选后收集未参与选择的新数据；同窗真实仓位回测只能标样本内交易化诊断。

成本固定往返 0.12%；本审计没有新增净收益数字。

## 验证

- `output/fit-readiness-focused-pytest.log`：62 passed in 1.08s
- `output/fit-readiness-full-pytest.log`：6 failed, 2147 passed, 5 warnings in 80.07s (0:01:20)
- `output/fit-readiness-isolated-pytest.log`：3 failed, 3 passed in 0.90s

全量失败中3项是代理格式问题，局部清理NO_PROXY后通过；其余是行情尾部偏移及交易入口缺少halt_status。
这些失败位于本次未修改的执行路径，不宣称全量测试通过。

复现：`.venv/Scripts/python.exe backend/scripts/fit_readiness.py`（只读数据库）。

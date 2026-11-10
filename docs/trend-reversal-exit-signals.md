# 趋势反转入场 × 其他信号卖出：过去半年受控回测

> 生成时间：2026-11-10 01:27 UTC ｜ 脚本：`backend/scripts/trend_exit_study.py` + `trend_exit_report.py` ｜ 命令见文末

## 假设与边界

- **信息源 / 可证伪假设**：买入信号固定为 `supertrend_current`（Supertrend 趋势反转）的多头方向翻转上升沿；新增信息只来自 8 条预注册卖出信号（5 条 OHLCV 量价、3 条非价格：资金费率横截面拥挤／资金费率转负／持仓量横截面拥挤）。若某卖出信号真有择时价值，则它与基线在同一入场点上的配对净收益差 CI 下界为正、FDR q < 0.10、4 段同号 ≥ 3，且跑赢「匹配触发率的随机早退」分布。
**事前终止条件**：上述任一不满足即不启用、不调阈值；判定词按手册第 3 节词表取（CI 上界为负写 `negative`、跨 0 写 `inconclusive`，不预先锁定词条）。
- **周期 / 持有期 H**：1h；所有臂共享 48 根（2 天）持仓上限并强制平仓；Gate 1 标签前瞻 24 根；相邻入场间隔 ≥ 48 根以消除重叠计数。
- **窗口起止 / 墙钟对齐**：2026-05-11 00:00 UTC → 2026-11-10 00:00 UTC（183 天，UTC 日界对齐，4392 根 1h）。样本期 BTC -7.50%（下跌市，beta 归因基准）。
- **币池 / 币数**：native 1h 全窗口池 460 币，有成交 456 币；skip 79 币（{'window_coverage_below_threshold': 37, 'no_warmup_history': 38, 'no_long_entries': 4}）。native 1h 窗口内 4264/4392 根，末端 5.3 天由 1m 聚合补齐（native 行优先，与 derived/native 一致性零差异口径同源）；15m 因仅 28 币有半年深度，不采用。
- **止损线核对（H2 冲突登记）**：已查 `references/verdicts-history.md` 与 `docs/supertrend-exit-diagnosis.md`／`docs/trend-reversal-order-exit.md`／`docs/exit-metalabel.md`——「固定入场只换出场」「不要再调止损倍数／持有期救它」是已证伪路线，按 Gate 0 本应终止。因用户明确指令，本轮执行**一次性、预注册、无网格、不改生产代码**的复核：价格类信号作为已证伪路线复述对照；唯一可能带来增量的是非价格信息源。冲突与取舍在此显式登记。
- **对照组**：① 当前参数基线（固定 TP=3R / SL=1.5ATR）；② 20 种子随机方向基线；③ 每条信号各 20 种子「匹配触发率随机早退」基线（重排触发时点，分离纯粹的早退 beta 效应）。「关闭过滤基线」会改变入场点集合、与固定入场配对 A/B 不同口径，按「受控 A/B 只改一个变量」原则不纳入，此为与标准 SOP 的显式偏差。

## 口径与参数

- **成本**：`fee_rate`=0.0004、`slippage`=0.0002、双边合计 0.12%；口径＝notional（名义本金），全程不切换，不与 margin 口径报告混比。
- **参数来源**：global_state + StrategyConfig overrides (cost keys removed)；关键项 `sl_atr_mult`=1.5、`target_r`=3.0、`supertrend_factor`=2.5、`regime_filter`=True、`cost_filter`=True。1h 无生效绑定（实查 `strategy_bindings.enabled=1` 仅 #7 `supertrend_current × all × 15m`），故不假称复现绑定参数层。
- **样本外切分**：训练/验证/测试 = 60/15/25；边界各 purge 24h，剔除 463 笔；各段 train 15863 笔 / valid 3705 笔 / test 6854 笔。冠军只在训练段选、验证段判定、测试段只读。
- **撮合**：`engine=sequential`；收盘出信号、下一根开盘成交；OHLC 触发止损止盈、跳空按开盘、同根 SL 优先、平仓当根不再开仓；信号在 bar j 收盘触发、bar j+1 开盘平。所有臂共享同一批入场点与同一段未来 K 线。
- **数据与覆盖**：native 1h + 1m 聚合尾桶；funding 8h 桶化后取上一已结算桶的横截面分位（避免结算前视）；sentiment 仅 2026-10-06 04:00 UTC → 2026-10-28 00:00 UTC；funding 覆盖 2026-03-12 00:00 UTC → 2026-10-27 12:00 UTC。

## 结果

- 总入场点 **26885** 笔（456 币）。基线每笔期望 -0.1332%，95% CI [-0.1781, -0.0889]，verdict `negative`；基线出场原因分布 {'sl': 19106, 'tp': 5833, 'time': 1935, 'tp_gap': 3, 'sl_gap': 8}（止损占绝对多数，说明 1.5ATR 结构止损先于 3R 目标触发）。
- 随机方向基线（20 种子）均值 -0.1374%、p05 -0.1639%、p95 -0.1156%；基线收益在其中的分位 0.600（未超过 0.95 —— 入场方向相对随机方向本身没有可检出的优势）。

| 卖出信号 | 类别 | 触发率 | 配对差 % | 95% CI % | FDR q | 分段同号 | 随机早退均值 % | 随机基线分位 | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `mom_fade` | price | 88.5% | -0.0186 | [-0.0341, -0.0030] | 0.1124 | 3/4 | -0.0216 | 0.600 | negative |
| `ma_break` | price | 70.2% | -0.0100 | [-0.0190, -0.0012] | 0.1124 | 3/4 | -0.0148 | 0.750 | negative |
| `vol_spike` | price | 64.8% | +0.0053 | [-0.0186, 0.0297] | 0.7890 | 1/4 | -0.0055 | 0.800 | inconclusive |
| `pullback_from_high` | price | 62.3% | +0.0021 | [-0.0066, 0.0103] | 0.7890 | 2/4 | -0.0125 | 0.950 | inconclusive |
| `st_flip` | price | 79.6% | +0.0002 | [-0.0095, 0.0095] | 0.9677 | 2/4 | -0.0196 | 1.000 | inconclusive |
| `funding_crowd` | nonprice | 29.9% | +0.0037 | [-0.0143, 0.0222] | 0.7890 | 2/4 | -0.0003 | 0.700 | inconclusive |
| `funding_neg` | nonprice | 38.0% | +0.0104 | [-0.0072, 0.0280] | 0.5153 | 2/4 | +0.0010 | 0.850 | inconclusive |
| `oi_crowd` | nonprice | 1.8% | +0.0052 | [-0.0003, 0.0102] | 0.1319 | 1/4 | +0.0008 | 1.000 | inconclusive |

- 训练段冠军：`funding_neg`（配对差 +0.0438%）；纯样本外测试段 6854 笔，配对差 -0.0490%，95% CI [-0.0808, -0.0170]，verdict `negative`。训练段正读数到测试段反号，是样本内选参噪声，不是可外推的改善。
- 三关全过的信号：0/8；随机早退分布控制了「下跌市里早平仓天然少亏」这一 beta 效应，配对差与随机早退均值之差才是择时增量。
- Gate 1 条件化 IC 四关全过 3/8 列，但其中 2 列 IC 为正、方向与「卖出」相反（短期超跌后 24h 反弹，卖了反而少赚）；方向正确的仅 1 列，且在受控 A/B 上 CI 跨 0（详见 `docs/probe-trend-exit-signals.md`）。
- 样本门槛说明：`oi_crowd` 只在 sentiment 覆盖的 23 天里有值，全样本命中率 1.8%，其测试段孤立的 `pass` 伴随全样本 CI 跨 0、FDR q=0.1319、4 段仅 1 段非零 → 判 `insufficient`／`likely_noise`，不是可用 edge（H8 小样本假精确）。

## 判定

- **verdict**：`negative` —— 测试段配对差 CI 上界为负。
- 判定对象：训练段选出的冠军 `funding_neg` 在纯样本外测试段相对基线的**配对差**（6854 笔）；其余 7 条规则全样本多为 `inconclusive`，不因冠军为负就反推「所有信号都亏」，也不把测试段孤立的 `pass` 当 edge。
- **池化 CI**：`MIN_POOLED_TRADES=30`，全样本 26885 笔、测试段 6854 笔；bootstrap 按一笔交易重采样 2000 次。
- **随机基线**：20 种子；冠军分位 0.850（需 > 0.95 才算跑赢匹配触发率的随机早退）。
- **归因**：样本期 -7.50%（下跌市），任何提前平仓都会机械性少亏；配对差最大绝对值仅 0.0186pp/笔，远小于双边成本 0.12%，且多数与随机早退均值同量级或更差。基线本身每笔期望为负，方向层没有留下可供出场规则修复的毛 edge。
- **与历史结论对照**：与 `docs/supertrend-exit-diagnosis.md`「不要再调止损倍数／持有期救它」、`docs/trend-reversal-order-exit.md`（17 条静态规则 0 过三关）、`docs/exit-metalabel.md`（26 列 0 过四关）一致；本轮把样本从 105 笔线上订单扩到 26885 笔全市场半年入场，结论未翻转。

## 局限

- 未模拟资金费率、盘口深度、强平、分批建仓；回测假定下一根开盘及时成交，与模拟盘巡检撮合有差异。
- 信号阈值固定 1σ / 0.80 分位、决策点固定第 8 根，未做网格寻优；价格类信号在 48 根持仓窗内命中率偏高，属高频门控而非稀疏事件。
- `sentiment` 仅 23 天、`funding` 末端止于 10-27，`oi_crowd` 命中率仅 1.8% → 该列判 `insufficient`（数据缺口，不是指标无效）。
- 1m 长历史仅 19 币，半年窗口依赖 native 1h；15m 深度不足未做周期敏感性。
- 口径限制：名义本金，本报告数值不得与 margin 口径报告直接相减。

## 结论与后续

- 一句话结论：**否，不可行**。8 条预注册卖出信号在 26885 笔同入场点配对 A/B 上未产生统计支持的盈利提升，主 verdict `negative`。
- **是否启用**：`default_enabled=False`；未新增 spec、未改绑定、未改任何生产参数；现行生效绑定仍只有 #7（`supertrend_current × all × 15m`）。
- 按 H2 止损线：**不再调出场阈值／止损倍数／持有期／周期，也不再扩充 OHLCV 指标族**。
- 后续只写补样本／补数据方案：补足 ≥60 天衍生品情绪与 funding 末端、把订单级下单瞬间原始 TP/SL 快照留全；下一个信息增量仍在逐笔成交顺序、盘口挂撤单、多所现货腿。

## 复现命令

```powershell
.\.venv\Scripts\python.exe backend/scripts/trend_exit_study.py --days 183 --seeds 20 --out output/trend_exit_study.json
.\.venv\Scripts\python.exe backend/scripts/trend_exit_report.py --input output/trend_exit_study.json --prefix docs/trend-reversal-exit-signals
.\.venv\Scripts\python.exe -m pytest backend/tests/test_trend_exit_common.py -q
.\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/trend-reversal-exit-signals --strict
```

## 产物

- `docs/trend-reversal-exit-signals.md` + `.json`、`docs/probe-trend-exit-signals.md` + `.json`
- `backend/scripts/trend_exit_{common,data,study,stats,report}.py`、`backend/tests/test_trend_exit_common.py`、当日工作记忆登记

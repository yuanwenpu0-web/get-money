# Kronos 下单前「能否止盈离场」离线证伪探针

> 生成时间：2026-11-10 06:55 ｜ 脚本：`backend/scripts/kronos_order_probe.py`、`backend/scripts/kronos_order_eval.py` ｜ 命令：见文末
> 版本：首版 ｜ **用户指定的一次性离线证伪诊断**：Gate 0 命中 H2 止损线，只跑 Gate 1／4／5，不进入 Gate 2、3、6，不投入实现。

## 假设与边界

- **信息源 / 可证伪假设**：信息源是下单时刻之前的 OHLCVA 历史 K 线（Kronos-small 零样本生成未来路径）。假设 H：「若 Kronos 的下单前预测含有关于『该单能否先触 TP 而非先触 SL』的增量信息，则其路径分数对真实止盈标签的 AUC 应显著高于 0.5、高于只用历史波动＋TP/SL 距离的随机游走基线、且时间分段符号稳定」。事前承诺：任一分数 `fdr_q ≥ 0.10` 或分段同号 < 3/4 或随机基线分位 ≤ 0.95，即判为无增量信息、不再调阈值。
- **周期 / 持有期 H**：15m 与 1h 两种订单周期，各自前瞻 **24 根 bar（统一 24 墙钟小时）**，即 15m=96 根、1h=24 根。标签前瞻跨度与预测跨度严格一致。live 补充队列因数据末端限制改用 24 根 bar（15m＝6 小时）。
- **窗口起止 / 墙钟对齐**：主样本 2026-10-26 ~ 2026-11-07（13 天），UTC 左闭桶、按 `open_time - open_time % step` 对齐；未来窗口从下单所在桶的**下一根完整桶**开始。样本期 BTCUSDT 1m 首尾收盘 **−6.021%**（偏熊），供 beta 归因。所有订单时间均晚于 Kronos 权重最后更新（2025-09）与论文所述预训练截止（2024-06），整段为纯样本外；截止点无法独立核验，见局限。
- **币池 / 币数**：主样本 84 个币种 / 123 笔订单；去重后全库可用 150 笔（15m 126、1h 24）。`universe.skipped` 共 **27 笔，全部为 `incomplete_future`**（订单 id 192–215、217–219，均为 `source='live'` 的 2026-11-09 订单，其 24h 前瞻窗口尚未走完）；按 H8 列入 skip 清单，不读成「无信号」。这 27 笔另用 6h 短窗口补跑，见「结果」末节。
- **止损线核对**：已查 `references/verdicts-history.md` §1 —— **命中 H2**（现货 OHLCV 价格序列不再投入新特征／参数／标签）。本课题不是往生产管线加特征，而是用户点名的**一次性离线证伪**：不新增 `StrategySpec`、不改 `resolve_params`、不碰绑定、不进 Gate 2/3/6。历史对照：P4 全频率 |IC|<0.05、`docs/ema-trend-filter.md` 通用入场门控全网格 0 `pass`。
- **对照组**：① 关闭过滤基线＝全部订单；② 随机游走基线（历史 bar 形状 bootstrap，只含 TP/SL 距离与已实现波动）；③ 20 种子随机分数与随机订单过滤；④ 正／负／半衰／20% 弱对照自校准。

## 口径与参数

- **成本**：`fee_rate=0.0004`、`slippage=0.0002`，双边合计 **0.12%**；每笔净收益按 `fee*(1+close/entry) + 2*slippage` 精确扣，口径＝**名义本金**（全程不切换，不与 margin 口径报告混比）。毛利与净利分开，判定一律用净收益。
- **参数来源**：零样本预训练 `NeoQuasar/Kronos-small` + `Kronos-Tokenizer-base`（commit `67b630e6`），**无网格、无拟合**；`lookback=128`、`samples=3`（种子 20261110）、`T=1.0`、`top_p=0.9`、过滤阈值 **0.5 事前固定**。官方 `predict_batch` 内部 `np.mean(preds, axis=1)` 会平均掉采样路径，故以 `sample_count=1` 重复 3 个种子保留逐路径。
- **样本外切分**：`train=0 / valid=0 / test=1.0`；零样本无需选参，阈值事前固定，测试段只读。`MIN_POOLED_TRADES=30`。
- **撮合 / 标签**：标签用真实未来 bar 的 triple-barrier 重放（LONG `high≥tp` / `low≤sl`，SHORT 对称），**同根 SL 优先**，窗口内未触及记 `unresolved`（6 笔，归负类）。预测路径只取**形状**（相对前一根收盘的 high/low/close 比率），锚定到 `entry_price` 后再判阈值，避免生成价格的绝对水平偏差制造假触价；OHLC 违规（336 处 `high < max(open,close)` 等）已在评分时修正。归一化统计 `x_mean/x_std` 只来自历史窗口，无全窗口泄漏。
- **数据与覆盖**：K 线为 1m 派生的 derived 口径（完整连续桶），1m 历史自 2023-11-02 起；未用 native 15m、taker、funding、情绪数据。硬因果断言：最后一根历史 bar 的 `close_time < created_at`，下单当根残余行情不计入。

## 结果

| 组 | 成交笔数 | 每笔期望 % | 95% CI % | 净收益 % | 曝光 % | 随机基线分位 | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 全部订单（关闭过滤基线） | 123 | −0.1263 | [−0.5678, 0.3086] | — | — | — | `inconclusive` |
| Kronos 过滤（`kronos_mfe > 0.5`） | 37 | −0.0028 | [−0.7368, 0.7660] | — | — | 0.6190 | `inconclusive` |
| 随机游走过滤（`rw_mfe > 0.5`） | 64 | −0.7337 | [−1.3822, −0.0423] | — | — | — | `negative` |

> 净收益 % / 曝光 % 不填：本探针是订单级分类与每笔期望判定，未构建资金曲线；不得与 margin 口径的 `realized_pnl` 直接相减。

**分类预测力（主标签＝真实路径是否先触 TP，30/123 正类）**

| 分数 | AUC | AUC 95% CI | 对 `close_reason` 标签 AUC | 触发率@0.5 | Brier | 置换 p | FDR q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `kronos_mfe`（主，最大有利偏移/TP 距离） | **0.4287** | [0.3025, 0.5545] | 0.3719 | 30.08% | 0.3028 | 0.8801 | 0.952 |
| `kronos_hit`（逐路径触 TP 比例） | 0.4896 | [0.4090, 0.5716] | 0.4823 | 8.94% | 0.2584 | 0.6167 | 0.925 |
| `kronos_end`（预测期末收益/TP 距离） | 0.5025 | [0.3853, 0.6149] | 0.5384 | 6.50% | 0.2438 | 0.4848 | 0.8726 |
| `rw_mfe`（随机游走对照） | 0.4090 | [0.3069, 0.5123] | 0.3256 | 52.03% | 0.3162 | 0.9285 | 0.952 |
| `rw_p_tp`（随机游走触 TP 频率） | 0.3966 | [0.2959, 0.5039] | 0.3083 | 10.57% | 0.2312 | 0.9520 | 0.952 |
| 正对照：未来真实路径 MFE | 0.8835 | [0.8191, 0.9360] | 0.8438 | 71.54% | 0.3881 | 0.0005 | 0.0015 |
| 负对照：随机分数 | 0.5333 | [0.4204, 0.6559] | 0.4853 | 56.10% | 0.3517 | 0.2974 | 0.669 |

- **退化列清单**：正对照触发率 71.54%、半衰对照 73.98% 超过 60% 上限，属对照组（不参与筛选，仅标注）；主分数 30.08% 与 `kronos_hit` 8.94% 均在 [0.2%, 60%] 内，未剔除。pilot 阶段 4/4 笔 `kronos_hit` 全 0，全量后 26/123 笔非零（`mean=0.103`）—— 生成路径波动普遍小于 TP 距离（LONG 平均 TP +4.515% / SL −1.421%），硬触价分数信息量有限，故主分数改用连续 MFE。
- **自校准三对照**：正 0.8835（q=0.0015，量具能识别强信号）、负 0.5333（q=0.669，不显著）、半衰 0.8645 **未明显衰减**（正对照是与标签近乎确定性的函数，50% 噪声不足以打乱排序）；追加 20% 信号弱对照 0.7075，序列 0.8835 → 0.8645 → 0.7075 → 0.5333 单调衰减。**自校准仅部分通过**，因此所有读数按更保守口径解释；主分数本身远未过 gate，该瑕疵不改变结论。
- **随机基线（H6）**：20 种子随机 AUC 均值 0.4985、p95 0.5560；`kronos_mfe` 分位 **0.0476**（要求 >0.95）。随机订单过滤（每种子 37 笔）均值 p05 −0.5003% / p95 +0.3449%，Kronos 过滤组 −0.0028% 分位 0.6190。
- **时间分段同号（H7）**：4 段 AUC 依次 0.5667 / 0.3384 / 0.5045 / 0.3077，**2/4 同号**，未达 ≥3/4；符号逐段翻转，与 EMA 门控的逐窗翻转同形。
- **分组诊断**：LONG 55 笔 AUC 0.4109、SHORT 68 笔 0.4389（两侧同向，非方向 beta 假象）；15m 99 笔 0.4222；1h 24 笔 AUC 1.0000 但**只有 1 个正类**，属 `insufficient` 的假精确，禁止引用点估计；`supertrend_current` 96 笔 0.4210，`trend_follow` 22 笔 AUC 1.0000 同样仅 1 个正类。出场结构：`stop_loss` 69、`take_profit` 27、`signal_reverse` 17、`env_switch_archived` 10。
- **补充队列：线上 live 订单（27 笔，25 币，6h 窗口，0 跳过）** —— `kronos_mfe` AUC **0.2357**，CI [0.0108, 0.5162]，置换 p 0.9810，触发率 14.81%；正类 7（路径标签）/ 9（`exchange_tp`）；每笔净收益 +0.7621%，CI [−0.2788, 1.9164]，27 < 30 → verdict **`insufficient`**（数值照给、不下结论）。该队列收益为正主要是 `exchange_tp` 占比与样本期短，不能当 edge 读数。

## 判定

- **verdict**：主样本 **`inconclusive`**；live 补充队列 **`insufficient`**。按词表，`inconclusive` 等于「不知道」，不等于「已证无效」，也不等于任何形式的正向解读。
- **池化 CI**：Kronos 过滤 37 笔 ≥ `MIN_POOLED_TRADES=30`，但每笔期望 CI [−0.7368, 0.7660] 跨 0；全部订单 CI [−0.5678, 0.3086] 亦跨 0。
- **随机基线**：主分数分位 0.0476 ≤ 0.95；收益过滤分位 0.6190 ≤ 0.95；FDR q=0.952 ≥ 0.10；分段同号 2/4。三关（净超额 CI 下界为正 ∧ q<0.10 ∧ 分段同号 ≥3/4）全不过。
- **归因**：AUC 点估计 0.4287 **低于 0.5**（分数越高反而越不易止盈），但 CI 上界 0.5545 跨 0.5、置换 p 0.8801，**不能断言统计上反向显著**。更关键的是随机游走基线 0.4090 / 0.3966 与 Kronos 同向且量级相近：预测分数主要反映「TP/SL 距离 + 已实现波动」的几何关系（TP ≈ 3.2×SL，高波动路径先触 SL），**没有超出波动率基线的增量信息**。毛 edge 本就接近 0，0.12% 成本后无转正空间。
- **与历史结论对照**：与 `references/verdicts-history.md` §1 止损线、P4 全频率 IC <0.05、`docs/ema-trend-filter.md`（门控 0 `pass`、FDR q 0.606–0.968、逐窗翻转）**一致**；无冲突。样本期 BTC −6.021%，但分类任务不做方向择时，多空两侧 AUC 同向偏低，beta 不构成解释。

## 局限

- 13 天窗口、84 币、30 个止盈路径标签、6 个未决标签，统计功效低；1h 子集仅 1 个正类，AUC 1.0 是假精确。
- 27 笔 live 订单因 24h 前瞻窗口未走完被剔除，主样本全为 paper；补充队列改用 6h 窗口，与主样本口径不同，两者数值不可直接合并比较。
- 生成路径为 3 次零样本采样，未做微调、未调温度／`top_p`／`sample_count`；OHLC 违规 336 处已修正但说明生成分布未完全满足 K 线约束。
- 下单当根残余行情被丢弃（未来窗口从下一完整桶起）；未模拟资金费率、盘口深度、强平、分批建仓、交易所保护单实际成交价与滑点分布；回测假定三重障碍可在 bar 内按 OHLC 触发，与 live 巡检撮合有差异。
- Kronos 权重最后更新 2025-09、论文称预训练截至 2024-06，但语料与评测代码未公开，**样本外截止点无法独立核验**。
- 自校准半衰对照未按预期衰减，量具仅部分通过。
- 口径限制：本报告为名义本金，不得与 margin 口径报告直接相减。

## 结论与后续

**一句话结论**：在这批线上订单上，Kronos 的下单前预测**无法判别该单能否止盈离开**——主分数 AUC 0.4287（CI 跨 0.5）、FDR q 0.952、随机基线分位 0.0476、分段 2/4 同号，且与「只用历史波动＋TP/SL 距离」的随机游走基线同向，无增量信息；每笔净收益 CI 跨 0，verdict `inconclusive`，live 补充队列 `insufficient`。

- **是否启用**：`default_enabled=False`，不进任何绑定；未新增或修改任何策略／撮合／生产代码。按 H11 与 H2，本路线到此为止，**不再调阈值、换周期、换采样数、换温度、加止损倍数救它**。
- 若仅从统计功效出发补样本，需 ≥60 天订单流且止盈正类 ≥100 笔，并用同一 24h 前瞻口径重跑；但 H2 止损线已明确，不建议再投入。
- 下一个信息增量仍只在逐笔成交顺序／盘口挂撤单／多所现货腿／衍生品情绪（P9）。

## 复现命令

```powershell
.\.venv-dml\Scripts\python.exe backend\scripts\kronos_order_probe.py --lookback 128 --samples 3 --batch-size 8 --out output/kronos_order_probe.json
.\.venv-dml\Scripts\python.exe backend\scripts\kronos_order_probe.py --source live --horizon-bars 24 --lookback 128 --samples 3 --batch-size 8 --out output/kronos_order_probe_live.json
.\.venv-dml\Scripts\python.exe backend\scripts\kronos_order_eval.py --secondary-input output/kronos_order_probe_live.json --out docs/kronos-order-exit.json
.\.venv\Scripts\python.exe "C:\Users\YUAN\.workbuddy\skills\strategy-backtest-playbook\scripts\preflight.py" --prefix docs/kronos-order-exit
```

Kronos 官方源码以 commit `67b630e6` 存于 `.workbuddy/tools/kronos/model/`（gitignore），权重由 `huggingface_hub` 自动缓存；环境清单见 `requirements-dml.txt`。

## 产物

- `docs/kronos-order-exit.md` + `docs/kronos-order-exit.json`
- `backend/scripts/kronos_order_common.py`、`kronos_order_probe.py`、`kronos_order_eval.py`
- `output/kronos_order_probe.json`、`output/kronos_order_probe_live.json`（原始逐订单分数表）
- 当日工作记忆已登记结论与报告路径
- 全量 `pytest backend/tests -q`：3 项失败为仓库基线既存（名称与 `.pytest_cache/v/cache/lastfailed` 已有记录逐条吻合：`test_market_performance.py::test_complete_tail_queries_have_a_bounded_minute_range`、`test_pool_volatility_research.py::test_live_entrypoints_reject[...]` 两条），与本轮无关；本轮未修改任何生产代码，新增脚本均为独立入口、无单测覆盖。`preflight.py --strict` 0 FAIL / 0 WARN。

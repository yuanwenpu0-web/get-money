# 已启用「策略 × 币种池」绑定：逐组合寻优与最终选型

> 生成时间：2026-11-05T14:17:22 ｜ 脚本：`backend/scripts/binding_combo_review.py`、`backend/scripts/pit_pool_check.py` ｜ 报表：`backend/scripts/binding_combo_report.py`

## 假设与边界

- **信息源**：四条绑定现有的 OHLCV 信号层，未引入任何新特征（H2 止损线下不新增特征）。
- **可证伪假设**：如果某条绑定存在正 edge，那么它在验证段（不看训练段）的每笔期望 95% CI 下界应为正，且在「同笔数、同持有期的随机入场」分布中的分位应超过 0.95。
- **终止条件**：验证段 CI 上界为负 → `negative`，停止该路线；CI 跨 0 但不优于随机 → `inconclusive`，只补样本不改参数。
- **周期 / 持有期**：A/B 为 1h（持有 120h / 48h）；C/D 为 15m（出场由 ATR 止损与目标决定）。
- **窗口**：主口径最近 180 天，稳健性口径最近 90 天；时点池对照 5 期 × 30 天。
- **止损线核对**：已读 `references/verdicts-history.md`；本课题只做「现有参数空间内的边界确认」，不新增特征、不改标签、不换周期，因此不越 H2。
- **对照组**：① 绑定当前参数；② 寻优冠军；③ 随机基线（研究型＝随机入场＋同持有期；趋势型＝方向翻转）。

## 口径与参数

- **成本**：fee 0.04%/边 + 滑点 0.02%/边，开平双边 0.12%；口径＝**名义本金**。
- **收益口径**：每笔净收益 ÷ 名义本金。通用撮合的 `return_pct` 是保证金口径，已按 `pnl/(entry×size)` 折算后与固定 200U 名义的研究池同表比较。
- **分批止盈按仓位聚合**：先按 `position_id` 合并腿再加总，避免 1/6 仓位的 1R 止盈腿与 5/6 仓位的止损腿等权（未聚合时 ETHUSDT 15m 腿级 +0.24% vs 仓位级 −0.09%）。
- **样本外切分**：训练 60% / 验证 20% / 测试 20%，按墙钟切，跨段成交清除。
- **选参规则**：训练段池化每笔期望最高者；成交笔数低于「各格中位数的 40%」或低于 30 笔的组合不参与竞争（防「开过滤后只剩十几笔」的小样本冠军）。
- **撮合**：收盘出信号、下一根开盘成交；OHLC 触发止损止盈、跳空按开盘、同根止损优先。

## 现状：四条生效绑定

| # | 策略 | 池 | 周期 | 币数 | 当前参数 |
| --- | --- | --- | --- | ---: | --- |
| A | `research_high_turnover_high_vol` | `research_high_liquidity_high_volatility` | 1h | 24 | channel_kind=0 research_hold_hours=120 |
| B | `research_low_turnover_high_vol` | `research_low_liquidity_high_volatility` | 1h | 21 | channel_kind=1 research_hold_hours=48 |
| C | `supertrend_current` | `trend_reversal_fit` | 15m | 12 | sl_atr_mult=1.5 supertrend_factor=2.5 target_r=3.0 use_adx_filter=False |
| D | `flow_imbalance_long` | `alt` | 15m | 3 | entry_quantile=0.95 flow_window=48 max_hold_bars=96 sl_atr_mult=2.0 target_r=2.0 |

四条绑定的 spec 全部 `default_enabled=False`；它们处于启用状态是历史配置，不是回测推荐（H11）。

## 组合 A · 研究·高成交额高波动（RSI或布林）

策略 `research_high_turnover_high_vol` × 池 `research_high_liquidity_high_volatility` × `1h`，可用 24 币，搜索空间 12 组。
剔除 1 币，例：BTWUSDT（历史不足，起于 2026-06-04）。

| 参数 | 训练笔数 | 训练每笔% | 验证笔数 | 验证每笔% | 验证 95% CI | 判定 | 测试每笔% |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| channel_kind=0 research_hold_hours=168 ★ | 279 | 3.76 | 78 | 9.02 | [-0.61, 21.00] | inconclusive | 14.07 |
| channel_kind=0 research_hold_hours=120 ● | 367 | 3.08 | 118 | 2.87 | [-1.00, 7.08] | inconclusive | 8.22 |
| channel_kind=1 research_hold_hours=168 | 273 | 3.02 | 82 | 6.05 | [-2.80, 17.00] | inconclusive | 15.31 |
| channel_kind=0 research_hold_hours=96 | 438 | 2.96 | 145 | 1.95 | [-1.82, 6.02] | inconclusive | 7.96 |
| channel_kind=1 research_hold_hours=120 | 349 | 2.67 | 112 | 8.90 | [-1.00, 25.38] | inconclusive | 9.27 |
| channel_kind=1 research_hold_hours=96 | 414 | 2.62 | 135 | 6.90 | [-0.87, 18.33] | inconclusive | 9.79 |
| channel_kind=1 research_hold_hours=72 | 500 | 2.01 | 164 | 1.26 | [-1.78, 4.43] | inconclusive | 6.56 |
| channel_kind=0 research_hold_hours=72 | 539 | 2.00 | 172 | 4.37 | [-0.74, 11.95] | inconclusive | 8.12 |

随机入场基线（20 种子，24 币、每种子 120 笔）：均值 7.07%，p05~p95 [1.00, 12.18]%；当前参数 2.87% 落在第 15 百分位，寻优冠军 9.02% 落在第 55 百分位。

**过程读出**：训练段每笔期望随持有期单调上升（24h 1.12% → 168h 3.76%），验证段同向；同时随机入场也能拿到 7.07%，说明这条曲线量的是「长期持有吃到币价趋势」，不是择时能力。两个通道（布林 / ATR 通道）差别在噪声内，换通道没有信息量。

## 组合 B · 研究·低成交额高波动（RSI或ATR通道）

策略 `research_low_turnover_high_vol` × 池 `research_low_liquidity_high_volatility` × `1h`，可用 21 币，搜索空间 12 组。
剔除 4 币，例：牛来USDT（历史不足，起于 2026-08-30）、CAPUSDT（历史不足，起于 2026-06-27）、PONSUSDT（历史不足，起于 2026-09-06）。

| 参数 | 训练笔数 | 训练每笔% | 验证笔数 | 验证每笔% | 验证 95% CI | 判定 | 测试每笔% |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| channel_kind=0 research_hold_hours=168 ★ | 243 | 1.61 | 72 | 15.05 | [3.70, 29.00] | positive | 0.04 |
| channel_kind=0 research_hold_hours=120 | 318 | 0.49 | 103 | 8.37 | [2.53, 16.04] | positive | -1.60 |
| channel_kind=0 research_hold_hours=72 | 462 | 0.42 | 162 | 3.42 | [0.47, 6.76] | positive | 0.26 |
| channel_kind=1 research_hold_hours=168 | 244 | 0.36 | 70 | 14.56 | [2.69, 30.11] | positive | -1.74 |
| channel_kind=0 research_hold_hours=96 | 386 | 0.31 | 128 | 5.03 | [1.41, 9.36] | positive | -1.15 |
| channel_kind=1 research_hold_hours=120 | 314 | 0.20 | 95 | 6.99 | [1.92, 12.42] | positive | -1.77 |
| channel_kind=0 research_hold_hours=24 | 814 | 0.16 | 281 | 2.27 | [0.88, 4.08] | positive | -0.22 |
| channel_kind=1 research_hold_hours=96 | 376 | 0.14 | 115 | 5.96 | [1.81, 10.57] | positive | -0.64 |
| channel_kind=1 research_hold_hours=48 ● | 566 | -0.23 | 184 | 3.38 | [1.22, 5.94] | positive | -1.34 |

随机入场基线（20 种子，21 币、每种子 189 笔）：均值 2.07%，p05~p95 [0.32, 3.40]%；当前参数 3.38% 落在第 95 百分位，寻优冠军 15.05% 落在第 90 百分位。

**过程读出**：训练段 12 个格子全在 0 附近（−0.23% ~ +1.61%），验证段却整体 +2% ~ +15%，测试段又回落到 0 附近——训练与验证不一致是「验证窗恰好是幸运 36 天」的形态，不是参数有效。网格冠军把通道从 ATR 换成布林，那等于换策略，不能靠一次寻优决定。

## 组合 C · Supertrend 趋势反转 × 主流币

策略 `supertrend_current` × 池 `trend_reversal_fit` × `15m`，可用 12 币，搜索空间 36 组。

| 参数 | 训练笔数 | 训练每笔% | 验证笔数 | 验证每笔% | 验证 95% CI | 判定 | 测试每笔% |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| sl_atr_mult=1.5 supertrend_factor=3.0 target_r=6.0 use_adx_filter=True | 16 | -0.01 | 8 | 1.76 | [-0.78, 5.55] | insufficient | 0.14 |
| sl_atr_mult=2.0 supertrend_factor=3.0 target_r=6.0 use_adx_filter=True | 16 | -0.02 | 7 | 0.49 | [-0.68, 2.08] | insufficient | 0.08 |
| sl_atr_mult=1.5 supertrend_factor=2.0 target_r=3.0 use_adx_filter=False ★ | 4341 | -0.15 | 1442 | -0.21 | [-0.26, -0.14] | negative | -0.12 |
| sl_atr_mult=2.0 supertrend_factor=2.0 target_r=6.0 use_adx_filter=False | 3511 | -0.15 | 1187 | -0.20 | [-0.28, -0.11] | negative | -0.07 |
| sl_atr_mult=1.5 supertrend_factor=2.0 target_r=4.0 use_adx_filter=False | 4170 | -0.15 | 1387 | -0.20 | [-0.27, -0.15] | negative | -0.11 |
| sl_atr_mult=2.0 supertrend_factor=2.0 target_r=4.0 use_adx_filter=False | 3621 | -0.15 | 1216 | -0.21 | [-0.29, -0.13] | negative | -0.04 |
| sl_atr_mult=1.5 supertrend_factor=2.0 target_r=6.0 use_adx_filter=False | 4001 | -0.15 | 1342 | -0.18 | [-0.24, -0.11] | negative | -0.12 |
| sl_atr_mult=1.5 supertrend_factor=3.0 target_r=3.0 use_adx_filter=False | 2660 | -0.15 | 836 | -0.19 | [-0.28, -0.10] | negative | 0.07 |
| sl_atr_mult=1.5 supertrend_factor=2.5 target_r=3.0 use_adx_filter=False ● | 3305 | -0.16 | 1101 | -0.16 | [-0.23, -0.09] | negative | -0.05 |

毛收益估算（净 + 0.12% 成本）：当前参数 -0.039%/笔，寻优冠军 -0.086%/笔——毛 edge 低于 0 说明方向本身没有预测力，不是被成本吃掉的。

**过程读出**：所有不走 ADX 过滤的密集格子（800~4300 笔）在训练段与验证段**同时为负**，且参数之间只差 0.02pp 量级——搜索空间里没有能翻正的角落。唯一为正的是 ADX 过滤后只剩 8~37 笔的格子，属于典型的小样本冠军，已被选参规则排除。

## 组合 D · 订单流净买入极端值 × 山寨

策略 `flow_imbalance_long` × 池 `alt` × `15m`，可用 3 币，搜索空间 144 组。
剔除 491 币，例：0GUSDT（1m taker 覆盖不足 90%（）、1000000MOGUSDT（1m taker 覆盖不足 90%（）、1000BONKUSDT（1m taker 覆盖不足 90%（）。

| 参数 | 训练笔数 | 训练每笔% | 验证笔数 | 验证每笔% | 验证 95% CI | 判定 | 测试每笔% |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| entry_quantile=0.98 flow_window=48 max_hold_bars=128 sl_atr_mult=3.0 target_r=2.0 ★ | 112 | 0.58 | 43 | -1.96 | [-3.28, -0.59] | negative | -0.49 |
| entry_quantile=0.98 flow_window=48 max_hold_bars=96 sl_atr_mult=3.0 target_r=2.0 | 114 | 0.56 | 43 | -1.96 | [-3.28, -0.59] | negative | -0.49 |
| entry_quantile=0.98 flow_window=48 max_hold_bars=128 sl_atr_mult=3.0 target_r=3.0 | 111 | 0.54 | 42 | -1.79 | [-3.16, -0.33] | negative | -0.44 |
| entry_quantile=0.98 flow_window=48 max_hold_bars=96 sl_atr_mult=3.0 target_r=3.0 | 113 | 0.52 | 42 | -1.79 | [-3.16, -0.33] | negative | -0.44 |
| entry_quantile=0.98 flow_window=48 max_hold_bars=64 sl_atr_mult=3.0 target_r=2.0 | 114 | 0.51 | 43 | -1.95 | [-3.23, -0.61] | negative | -0.53 |
| entry_quantile=0.98 flow_window=48 max_hold_bars=64 sl_atr_mult=3.0 target_r=3.0 | 113 | 0.47 | 42 | -1.78 | [-3.10, -0.36] | negative | -0.47 |
| entry_quantile=0.98 flow_window=48 max_hold_bars=96 sl_atr_mult=2.0 target_r=2.0 | 124 | 0.43 | 48 | -1.44 | [-2.33, -0.56] | negative | -0.53 |
| entry_quantile=0.98 flow_window=48 max_hold_bars=128 sl_atr_mult=2.0 target_r=2.0 | 124 | 0.43 | 48 | -1.44 | [-2.33, -0.56] | negative | -0.53 |
| entry_quantile=0.95 flow_window=48 max_hold_bars=96 sl_atr_mult=2.0 target_r=2.0 ● | 211 | 0.04 | 72 | -0.57 | [-1.46, 0.42] | inconclusive | -0.16 |

毛收益估算（净 + 0.12% 成本）：当前参数 -0.454%/笔，寻优冠军 -1.843%/笔——毛 edge 低于 0 说明方向本身没有预测力，不是被成本吃掉的。

**过程读出**：144 个格子全部在验证段为负或跨 0，没有一个能翻正；相对最好的一批也只是把入场分位推到 0.98（信号更稀疏、笔数更少）。3 个币、几十笔成交，任何排序都不构成证据。

## 时点池（PIT）对照：正收益是策略还是选币

研究池由「最近 168 小时成交额 + 波动率」筛出，库里存的是**当前**成员。拿这批成员回放过去 180 天等于用今天知道会涨的币去赌昨天。下表按 30 天一期往回推 5 期，每期起点用**当时之前 168 小时**的行情重建池成员，两臂参数/成本/撮合完全相同。

| 组合/参数/成员来源 | 笔数 | 每笔均值% | 95% CI | 判定 | 正收益期数 |
| --- | ---: | ---: | --- | --- | ---: |
| A best · 时点重建（可交易） | 329 | -0.970 | [-4.80, 3.47] | inconclusive | 2/5 |
| A best · 当前成员（事后选币） | 332 | 10.844 | [4.90, 18.34] | positive | 5/5 |
| A current · 时点重建（可交易） | 459 | -1.014 | [-3.49, 1.56] | inconclusive | 1/5 |
| A current · 当前成员（事后选币） | 458 | 7.100 | [2.44, 13.40] | positive | 5/5 |
| B best · 时点重建（可交易） | 314 | 0.013 | [-2.42, 2.80] | inconclusive | 3/5 |
| B best · 当前成员（事后选币） | 305 | 1.619 | [-0.78, 3.99] | inconclusive | 3/5 |
| B current · 时点重建（可交易） | 867 | 0.423 | [-0.94, 2.06] | inconclusive | 4/5 |
| B current · 当前成员（事后选币） | 835 | 0.322 | [-0.62, 1.32] | inconclusive | 3/5 |

同一期、同一参数下两臂之差就是「选币偏差」的实际价格。

## 跨组合对比

| 组合 | 180天 验证% | 180天 测试% | 180天判定 | 90天 验证% | 90天判定 | PIT 每笔% | PIT 判定 | 随机分位 |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |
| A | 2.87 | 8.22 | inconclusive | -0.65 | inconclusive | -1.014 | inconclusive | 15 |
| B | 3.38 | -1.34 | positive | -1.37 | inconclusive | 0.423 | inconclusive | 95 |
| C | -0.16 | -0.05 | negative | -0.03 | inconclusive | — | 不适用 | — |
| D | -0.57 | -0.16 | inconclusive | -0.02 | insufficient | — | 不适用 | — |

>- **读法**：`PIT 每笔%` 是唯一可交易的读数（成员按当时信息筛出）；`180天/90天 验证%` 用的是当前成员，含事后选币。两者差距越大，说明该组合的历史读数越依赖「先知道谁会涨」。

## 判定

| 组合 | 验证段判定 | 随机基线 | PIT 判定 | 综合 |
| --- | --- | --- | --- | --- |
| A | inconclusive | 分位 15% | inconclusive | 无 alpha，历史读数为选币幻觉 |
| B | positive | 分位 95% | inconclusive | 唯一在 PIT 下不亏且无选币偏差 |
| C | negative | 不适用 | 不适用 | negative，稳定亏损 |
| D | inconclusive | 不适用 | 不适用 | insufficient，宇宙仅 3 币 |

### 逐条依据

- **A（研究·高成交额高波动）**：验证段每笔 2.87% CI [-1.00, 7.08]，跨 0；同笔数随机入场均值 7.07%，本策略只落在第 15 百分位——**跑输随机**。90 天窗口验证段 -0.65%，符号翻转。时点池口径 -1.014% CI [-3.49, 1.56]，1/5 期为正；同一批期、同一参数换成当前成员则是 7.100% （5/5 期为正）——两者相差约 8.1 个百分点，这部分就是「事后选币」的价格。
- **B（研究·低成交额高波动）**：验证段 3.38% CI [1.22, 5.94] → `positive`，随机分位 95%（门槛 >95%）；时点池 0.423% CI [-0.94, 2.06]，4/5 期为正；当前成员 0.322% —— 两臂几乎相同（-0.10pp），说明它的读数**不是靠选币撑起来的**。但 90 天窗口验证段 -1.37%、测试段 -1.35%，符号翻负。
- **C（Supertrend 15m）**：验证段 -0.16% CI [-0.23, -0.09] → `negative`，1100 笔样本充足；毛收益估算为负，说明不是成本吃掉的，是方向没有预测力。网格里所有密集格子（不走 ADX 过滤）在训练段与验证段**同时为负**，唯一为正的是几个只剩十几笔的 ADX 过滤格子（`insufficient`）——典型的小样本冠军。
- **D（订单流 × 山寨）**：494 个池成员里只有 3 个同时具备 15m 所需的真实主动买入量（原生 15m 没有任何一行 taker，只能由 1m 聚合，而 1m 只覆盖约 90 个币），验证段 -0.57%、毛收益估算为负。这条绑定在当前数据层**不可验证**。

## 最终推荐

**先给硬结论：四条绑定没有一条通过 gate（`pass`），按 H11 都不该启用。**以下是「如果必须保留，留哪条、用什么参数」的相对排序。

| 排名 | 组合 | 建议 | 参数 |
| ---: | --- | --- | --- |
| 1 | B `research_low_turnover_high_vol` × `research_low_liquidity_high_volatility` × 1h | 保留（标注「未经统计支持」） | `channel_kind=1`, `research_hold_hours=48`（**不改**） |
| 2 | A `research_high_turnover_high_vol` × `research_high_liquidity_high_volatility` × 1h | 可保留，但不要把它当作有 edge | `channel_kind=0`, `research_hold_hours=120`（**不要改成 168**） |
| 3 | C `supertrend_current` × `trend_reversal_fit` × 15m | **关闭绑定** | — |
| 4 | D `flow_imbalance_long` × `alt` × 15m | **关闭绑定**（数据层不可验证） | — |

**为什么 B 排第一**：它是四条里唯一在「时点重建成员」（唯一可交易的口径）下不亏的组合——0.423%/笔、4/5 期为正、CI 跨 0；而它的当前成员读数与 PIT 读数几乎一样，说明历史成绩不是靠「先挑出会涨的币」。A 的 180 天读数看着更高，但那部分收益在 PIT 下整段消失（7.1% → -1.0%），且当前参数在验证段的随机分位只有 15%，跑输随机。

**为什么 A 的参数不要改成 168 小时**：180 天网格确实选了 168h（验证 9.02%），但那是事后选币放大出来的；换成 PIT 口径后 168h 是 -0.970%，并不优于 120h 的 -1.014%，90 天窗口的验证段也是负值。这正是「参数寻优救不了没有信号层的策略」——网格选的是选币偏差，不是 edge。

**C 与 D 不建议保留**：C 的负 edge 有 1100+ 笔样本支撑、两个窗口同号、毛收益也为负，属于应当登记止损线的 `negative`；D 的可用宇宙只有 3 个币，任何读数都不构成证据，在补齐 15m 主动买入量（或改用 1h 原生 taker）之前不应启用。

**复检触发条件**：① B 若能在 6 个月以上窗口把 PIT 口径的 CI 下界推为正（需 ≥300 笔），才谈得上 `pass`；② A 需要先解决「池成员用当前窗口筛出」这件事（改用逐期重建的池），否则它的历史读数不可用于决策；③ D 需要补齐 15m taker 覆盖（1m 回补或改用原生 1h）。

## 已执行的变更

| 时间 | 绑定 id | 组合 | 绑定 | 变更 | 依据 |
| --- | ---: | --- | --- | --- | --- |
| 2026-11-05 22:35:10+08:00 | 10 | C | supertrend_current × trend_reversal_fit × 15m | `enabled: 1 → 0` | verdict=negative，1100 笔、两窗口同号、毛收益 −0.04% < 0 |
| 2026-11-05 22:35:10+08:00 | 43 | D | flow_imbalance_long × alt × 15m | `enabled: 1 → 0` | verdict=insufficient，原生 15m 无 taker，494 个池成员里仅 3 个可回测 |

- A（#8）、B（#9）保持 `enabled=1`、参数不动：A 维持 `research_hold_hours=120`，不改成网格冠军 168h（PIT 与 90 天口径都更差）；B 维持 48h，标注「未经统计支持」。
- 停用只挡新开仓：`trading/service.py` 的持仓管理走 `open_orders(db)`，与绑定无关，`resolve_order_params` 也按 (策略, 池, 周期) 读仍存在的那行绑定，所以停用当时未平的 3 笔 supertrend 仓位（LTC/TIA/SOL）继续按原止损止盈跑完，不会被强平或失去管理。
- 副产物：周期内同步目标从 505 个币降到约 50 个（alt 池 494 成员此前每轮全量同步）。

## 局限

- 未模拟资金费率、盘口深度、强平与分批建仓；下一根开盘及时成交是乐观假设。
- A/B 的研究池成员每次维护都会整体重写，实盘的池与回测里的池不可能完全一致。
- PIT 重建放宽了「窗口内必须有完整 1m」这条（1m 只覆盖约 90 个币），成员构成与线上维护会有差异。
- 15m 的主动买入量只能由 1m 聚合得到，原生 15m 没有任何一行 taker，D 组因此只剩 3 币。
- 微盘币单笔波动可达 ±100%，bootstrap CI 在重尾分布上偏窄，读 CI 必须同时看笔数与分位。
- 本报告为名义本金口径，不得与 margin 口径报告直接相减。

## 复现命令

```bash
.venv/Scripts/python.exe backend/scripts/binding_combo_review.py --days 180 --workers 8 --out output/binding-combo-180
.venv/Scripts/python.exe backend/scripts/binding_combo_review.py --days 90 --workers 8 --out output/binding-combo-90
.venv/Scripts/python.exe backend/scripts/pit_pool_check.py --folds 5 --span 30 --out output/pit-pool-check
.venv/Scripts/python.exe backend/scripts/binding_combo_report.py
.venv/Scripts/python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/binding-combo-optimization
```

## 产物

- `docs/binding-combo-optimization.md` + `docs/binding-combo-optimization.json`
- 原始结果：`output/binding-combo-180.json`、`output/binding-combo-90.json`、`output/pit-pool-check.json`
- 单测：`backend/tests/test_combo_engine.py`


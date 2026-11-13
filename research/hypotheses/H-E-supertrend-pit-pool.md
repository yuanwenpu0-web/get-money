# P17 / H-E · Supertrend 趋势反转「缩小币种池」的 PIT 可行性判定（Gate 0 假设卡，预登记）

- 状态：`GATE0_REGISTERED` → **`CLOSED_NEGATIVE`（2026-11-12，判定见 §9）**。本卡落盘后才允许跑实验；判据 C1–C5 与终止条件在跑之前写死，跑后未改。
- 登记时间：2026-11-12T00:00Z 之后、实验启动之前；代码基线 `564d941`。
- 触发：用户指令「想要赚钱。帮我优化。看看缩小币种池能不能行，或者调整参数止盈设置低一点也行。反转里面的参数严格一点，先出方案」，随后「按你建议跑。自动跑，不要再让我批准了。给你最大权限了」。
- 授权范围：**只读回测 / 探针**。不下真实订单、不启用 live API key、不切换交易环境、不改绑定启用状态、不改 spec 默认参数。绑定 #7 维持 `enabled=0`。
- 性质：**edge 判定**（不是工程复现），Gate 1–6 全流程适用，H1–H12 全部生效。

## 1. Gate 0 查重：用户三个提议里两个已证伪，本卡只做第三个

| 用户提议 | 查重结论 | 证据 |
|---|---|---|
| **止盈设低一点** | **已证伪，且越低越差** | `docs/supertrend-optimization.json` stage1 网格 1050 组/周期 ×3 周期，`target_r ∈ {1.5,2.0,2.5,3.0,4.0,6.0}`。15m 测试段 negative 数随 r 下降**单调上升**：r1.5→158、r2.0→157、r2.5→152、r3.0→151、r4.0→145、r6.0→144；**positive 全为 0**。另 `docs/trend-reversal-fixed-tp.md` 绝对止盈 0.5/1/1.5/2/3/4/6/8% 共 24 档，CI 下界>0 的 **0 档**。`r=1.0` 被 `min_net_reward_risk=1.0` 结构性拦掉（500 币仅 109 笔），非有效读数。 |
| **反转参数严格一点** | **已证伪** | `factor` 全集 {1.0,1.5,2.0,2.5,3.0,4.0,5.0}+边界{4,5,6,8}；15m factor≤3.0 **150/150 negative**。ADX 阈值 20/30/50 全测；EMA 快慢线 17 候选 64 行 **0 pass**（`docs/ema-trend-filter.md`）。1h 仅 6 个 positive 全在 factor3.0，但 stage1 未跑随机基线（`random={}`），进 stage3 的冠军训练 +0.783% CI[+0.15,+1.43] positive → **测试 −0.0499% CI[−0.90,+0.84] inconclusive**、跑赢随机 1/16 币。1050 次检验按 95% CI 期望约 26 个假阳性，实际 6 个，**少于纯噪声预期**。 |
| **缩小币种池** | **本卡唯一课题** | 全量池已测但**选币本身未在 supertrend 口径下做过 PIT 对照**。见 §2。 |

**H2 止损线核对（必须显式登记）**：本课题的信息源仍是现货 OHLCV，**命中 H2 的字面范围**。但 H2 禁止的是「新特征／新参数／新标签／换周期／换阈值／换止损倍数」，本课题**不构造任何新特征、不改任何策略参数、不换周期、不调阈值**，唯一变量是**币池成员怎么来的**（横截面选择口径）。这与 `docs/binding-combo-optimization.json` 已做过的 `pit` vs `posthoc` 对照属同一类问题，而该对照**只在 research 池 × lab 指标上做过，从未在 supertrend 上做过**（grep 确认 `pool-research*.md`／`trend-direction-backtest.md` 均不含 supertrend）。

因此本卡按手册 §0 的冲突处理路径执行：**不视为 H2 豁免，而是把「supertrend 选币」登记为一条独立的、可证伪的、带事前终止条件的判定**。若判定非 `pass`，立即写入 `references/verdicts-history.md` 止损线，**不得换分档指标／换 N／换窗口重试**。已在 `docs/decisions/OPEN-DECISIONS.md` 登记。

## 2. 为什么这是真空白（增量信息）

- `docs/supertrend-reversal-backtest.json`：`universe kind=all`、500 币全评估 0 skip、1h、365 天（2025-10-29→2026-10-29）、**基准参数即线上 factor2.5/r3.0**：98738 笔、**−0.228%/笔 CI[−0.2534,−0.202] negative**、positive_share 0.0；多头 −0.2687% negative、空头 −0.1866% negative；**beats_random_ratio 0.0521**；账户收益中位 −8.79% vs 随机基线均值 −4.66%（**跑输抛硬币**）；499 币中 **107 币为正（21.4%）**。
- 「缩小池」＝从这 107 个里挑。**这正是事后选币。** 项目已量化过它值多少钱：`docs/binding-combo-optimization.json` 的 `pit` 段 A 组（1h/180 天/5 folds，同策略同窗口同成本，唯一差别是成员怎么来的）：

  | 选币方式 | 笔数 | 每笔 | CI | verdict | 正 fold |
  |---|---|---|---|---|---|
  | posthoc（当前成员） | 458 | **+7.1003%** | [+2.4382,+13.4036] | positive | 5/5 |
  | pit（T 时刻重建） | 459 | **−1.0136%** | [−3.4896,+1.5648] | inconclusive | 1/5 |

  **8.1pp 落差纯由「用未来信息选币」造成。** 本卡要在 supertrend 上复现这个对照，回答「存在不存在一个**事前可识别**的币子集让这套参数转正」。
- `backend/scripts/pit_pool_check.py::rebuild` 已有 PIT 重建能力，但 `POOLS` 只含两个 research 池，**不含 supertrend/all**。
- 已确认未测的其它空白（`cooldown_bars≠0`、6 个共享门控键、`tp1_r/tp2_r` 数值网格、保本止损）**不在本卡范围**，避免一次改多个变量。

## 3. 信息源与数据边界（H8，实时核表 2026-11-12）

| 周期 | 总行数 | 覆盖 | ≥365d(95%根) | ≥300d | ≥180d |
|---|---|---|---|---|---|
| 1h | 4,936,075 | 2025-09-20T15:00 → 2026-11-11T23:00 UTC | **461** | 488 | 516 |
| 15m | 2,760,834 | 2025-12-31T04:15 → 2026-11-12T00:30 UTC | **0** | 27 | **57** |

- **15m 全量池回测不可行**（0 币有 365 天）。故 15m 臂只用 57 币 × 180 天，且必须在报告里写明「57 币是按今日数据覆盖筛出的，本身含覆盖度选择成分」。
- 1h 臂 461 币 × 365 天，与既有 500 币报告窗口相差 14 天，**故本卡自带 `all` 基线臂，所有比较均为内部同口径**，不跨报告相减。
- 币种过滤：仅 `USDT` 后缀，排除 `UP/DOWN/BULL/BEAR` 杠杆代币与 `coin_universe.BLACKLIST`（5 币）。**不排除 BTC/ETH**——因为 `all` 基线臂与线上 `pool_code='all'` 都含它们；这与 `pool_maintenance.EXCLUDED`（14 币，含 BTC/ETH/稳定币对）口径不同，须在报告显式声明。

## 4. 可证伪假设

> 若「缩小币种池」能让 supertrend 趋势反转转正，则存在一个**只用 T 时刻之前信息**就能识别的币子集（按事前 168h 成交额或波动率排序取 TopN），使其在 `[T, T+span)` 上的池化每笔净期望 CI 下界为正，且该结论在 5 个 fold 上同号、并跑赢同口径随机方向基线。

**事前终止条件（预注册，任一不成立即判该臂不可行；不许换窗口／换 N／换分档指标／换参数去救）**：

| 编号 | 条件 | 门槛 |
|---|---|---|
| C1 | 池化每笔净期望 bootstrap 95% CI 下界 | **> 0** |
| C2 | 池化成交笔数 | **≥ 30** |
| C3 | 5 个 fold 的每笔均值同号为正 | **≥ 4/5** |
| C4 | 组合级随机基线分位 `percentile_rank` | **> 0.95** |
| C5 | BH-FDR（跨 6 个 PIT 臂，p = 1 − positive_share） | **q < 0.10** |

**C1–C5 是「与」关系**，全过才写 `pass`。任一不过按手册第 3 节词表写 `inconclusive` / `negative` / `insufficient`，**不得写「微正／有苗头／基本有效」**。

**必跑诊断（不参与 gate）**：
- `posthoc_N` 对照臂：按**当期已实现收益**选 TopN（作弊臂），量化 `posthoc − pit` 的偏差 pp/笔。这是本卡的核心交付之一。
- `all` 基线臂：不选币，全 eligible 宇宙。
- 逐 fold 成员重叠率（Jaccard）、逐臂多空分解、样本期基准涨跌（beta 归因）。

## 5. 预注册规格（不得事后挑选）

| 项 | 值 |
|---|---|
| 策略 | `supertrend_current`，**参数冻结**为 `resolve_params(CODE, {fee_rate, slippage}, interval)` 的输出：factor **2.5**、atr_period **14**、sl_atr_mult **1.5**、target_r **3.0**、use_adx_filter False、min_score **0.75**、heavy_score **0.90**、regime_filter True、cost_filter True、max_atr_pct 0.03、min_atr_cost_ratio 0.5、min_net_reward_risk 1.0、max_signal_age_bars 2、trailing_enabled False、cooldown_bars 0、tp1_r 1.0 / tp2_r 2.0、tp1/2/3_fraction 1/6·1/3·1/2、risk_per_trade_pct 0.005、ema_trend_fast/slow 0。**无参数网格**——本卡是选币实验，不是寻优。 |
| 成本 | `fee_rate=0.0004`、`slippage=0.0002`，双边 **0.12%**，**名义本金口径**，每币独立 10,000 账户。不得与 margin 口径混比。 |
| 窗口公共端点 | **2026-11-12T00:00Z**（UTC 零点对齐，且不晚于两周期数据末端） |
| 1h 臂 | 365 天 → 起点 **2025-11-12T00:00Z**；5 folds × **73 天** |
| 15m 臂 | 180 天 → 起点 **2026-05-16T00:00Z**；5 folds × **36 天** |
| 预热 | 每 fold 起点前 **600 根**，取自同一连续段；只计 `entry_time ≥ T_f` 的成交 |
| PIT 排序窗口 | `[T_f − 168h, T_f)` 的 **1h** K 线（两臂都用 1h 排序，保证选币口径一致） |
| 单币排序合格 | 窗口内 ≥ **160 根**有效 1h bar；末根不早于 `T_f − 2h`；`sum(quote_volume) > 0`；过滤通过 |
| 排序键 | `liquidity` = 窗口成交额降序；`volatility` = Top100 成交额内按 `std(diff(log(close)), ddof=1)` 降序。并列按 symbol 字母序（确定性） |
| 臂 | `all`、`pit_liq_{10,20,30}`、`pit_vol_{10,20,30}`、`posthoc_{10,20,30}`，共 **10 臂** |
| posthoc 排序键 | 该币在**本 fold 内**成交的 `return_pct` 均值降序（作弊，仅用于量化偏差） |
| 随机基线 | 逐币逐 fold **10 种子**，`base_seed=20261025`、步长 7919，方向 50/50 翻转，时点/仓位/风控/成本完全一致；组合级 = 各成员币同种子账户收益的等权均值，`percentile_rank(策略组合收益, 10 个组合随机样本)` |
| bootstrap | `n_boot=2000`、`seed=20261027`、`confidence=0.95`、`min_trades=30`；重采样单位 = 一笔交易（经 `aggregate_positions` 合并同仓多腿） |
| 撮合 | 复用 `app/backtest/execution.py::simulate`：信号收盘确认、次根开盘成交、跨缺口逐段独立撮合并按段复利结转、翻转根先出场不再进场 |
| 并行 | 进程池按**币**分片，每币取数一次；worker 只读 + NullPool；BLAS 单线程；并发 `min(24, CPU−4)` |
| 判定词表 | 只用 `pass`/`inconclusive`/`negative`/`insufficient`/`no_trades`；`positive` 折算成 `pass` |

## 6. 事前预期（写明以防事后改口径）

**大概率 `inconclusive` 或 `negative`。** 依据：
- 池效应实测 **0.04pp/笔** vs 周期效应 **0.27pp/笔**，两者叠加仍不转正（`references/verdicts-history.md` §4）。
- 全量池 98738 笔 negative、多空双负、跑赢随机仅 5.21%。
- 同类 PIT 对照在 research 池上给出 −1.0136% inconclusive。
- 15m `hold_H` 在 H=2..96 **两个方向毛收益全负** → 方向本身无预测力，选币改变不了这一点。

若结果真是 `pass`，必须额外做：换一段未参与的时间窗 holdout 复检 + 检查是否由 1–2 个币拉动（逐币贡献分解），才可讨论启用。**本卡不自动启用任何绑定（H11）。**

## 7. 结论边界

- 输出「存在/不存在事前可识别的盈利币子集」+「事后选币偏差值多少 pp」，**不输出启用建议**，除非 C1–C5 全过且 holdout 复检通过。
- 15m 臂的 57 币含覆盖度选择成分，结论不得外推到全量 15m 宇宙。
- 未模拟资金费率、盘口深度、强平、分批建仓；次根开盘及时成交是乐观假设。
- 微盘币单笔波动可达 ±100%，bootstrap CI 在重尾分布上偏窄，读 CI 必须同时看笔数与分位。
- `default_enabled=False` 维持不变；绑定 #7 维持 `enabled=0`。

## 8. 复现

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_pit_universe.py backend/tests/test_pit_pool_stats.py backend/tests/test_pit_pool_arms.py backend/tests/test_pit_pool_report.py backend/tests/test_pit_pool_combine.py -q
.\.venv\Scripts\python.exe backend/scripts/supertrend_pit_pool.py --interval 1h --days 365 --folds 5 --out docs/supertrend-pit-pool-1h
.\.venv\Scripts\python.exe backend/scripts/supertrend_pit_pool.py --interval 15m --days 180 --folds 5 --out docs/supertrend-pit-pool-15m
.\.venv\Scripts\python.exe backend/scripts/pit_pool_combine.py
.\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/supertrend-pit-pool
```

## 9. 判定落地（2026-11-12，跑后追加，未改任何事前判据）

**课题判定：`negative`。** 12 个 PIT 臂（1h 6 + 15m 6）**全部 negative，bootstrap 95% CI 上界全部 < 0**。

| 周期 | 窗口内有 bar | 覆盖率达标 | 实际产出 | 样本期 | `all` 基线 |
|---|---|---|---|---|---|
| 1h | 535 币 | **462 币** | 462 | 365 天 / 5 folds | n=95085 −0.2263% CI[−0.2479,−0.2050] |
| 15m | 521 币 | **30 币** | 30 | 180 天 / 5 folds | n=13563 −0.4322% CI[−0.4851,−0.3814] |

- **C4 随机方向分位：12 臂中 11 臂 = 0.000**，余下 `pit_vol_10`(1h) = 0.100。即策略账户收益**低于全部 10 个抛硬币种子**，比随机方向更差。
- **泄漏（本卡核心产出）**：`posthoc_N − pit_liq_N` = 1h **+1.86 / +1.51 / +1.33 pp/笔**（N=10/20/30）；15m +0.41 / +0.18 / −0.03 pp/笔。1h `posthoc_10` 达 +1.6239% CI[+1.3713,+1.8841]、5/5 正 fold、随机分位 1.000 —— **完全像可启用策略，但不可事前复现**。
- **成员稳定性**：1h 成交额排名相邻 fold Jaccard 中位 0.290–0.348，波动率排名 **0.026–0.091**（每 73 天几乎整批换血），远低于 P16 的 C2 门槛 0.60。波动率排名不是一个可持有的池。
- **多空分解**：做多 **7/7 臂 negative**（1h `all` −0.3064%，流动性池 −0.48%～−0.55%）；做空在流动性池点估计为正（`liq_10` +0.0972%）但 **CI 全部跨 0 → `inconclusive`**，不构成 pass。**不得据此做「只做空／砍多头」重试**（在已证伪的 OHLCV 上换标签，命中 H2）。

**与事前预期的偏差（必须登记，不得事后改口径）**：

1. §6 预期 `inconclusive`/`negative` → 实际 **`negative`**，方向一致，且比预期更确定（不是「样本不够」而是「CI 上界明确为负」）。
2. §7 写「15m 臂的 57 币」是**登记时的估计**；实测覆盖率达标只有 **30 币**（521 币中）。因此 15m 的 N=30 臂**池规模退化**（与 `all` 同构），且因排名候选另有门槛实际只凑出 **26 币**。15m 结论的外推范围比登记时设想的**更窄**，只有 N<30 的臂有区分度。
3. 达标币数在多次重跑间从 426 漂到 462：采集任务仍在增量同步。这不是 bug，报告已记 `generated_at` 与实际币数；**币数不是常量**。

**终止与后续**：预登记终止条件已触发 —— **不得换分档指标、不得换 N、不得换排名窗口重试**。止损线已写入 `.workbuddy/skills/strategy-backtest-playbook/references/verdicts-history.md` §1 延伸 + §10；`docs/decisions/OPEN-DECISIONS.md` 的 H-E 行已 `CLOSED`；台账 `research/experiments.jsonl` 追加 `P17-001`；`research/config.yaml` 追加 `supertrend_pit_pool` 块。`default_enabled=False` 未变，绑定 #7 维持 `enabled=0`，**本轮未改任何绑定启用状态与 spec 默认参数**。唯一有强先验信号的剩余路线是资金费率（非价格预测），已立项 `research/hypotheses/H-F-funding-multi-venue.md`。

**验证**：三份成对报告 preflight 均 **0 FAIL / 0 WARN / 21 PASS**；PIT 单测 **69 passed**（universe 12 / stats 15 / arms 11 / report 18 / combine 13）；全量 `pytest backend/tests -q` = **3 failed / 2035 passed / 120.41s**，3 个失败与既有基线完全一致（`test_market_performance` 1 + `test_pool_volatility_research` 2），**无回归**。本轮只新增 `backend/app/research/*` 与 `backend/scripts/*`，未改生产路径 ⇒ 无需重启服务、无需 `npm run build`。

**跑中发现并修复的实现缺陷（TDD，先红后绿）**：

1. **bootstrap CI 依赖聚合顺序**：固定 seed 下，成员集合相同但排序不同的 `pit_liq_30` / `pit_vol_30`（15m）给出不同 CI。修复：一律按 `sorted(members)` 字典序聚合。「同集合不同 CI」是实现 bug，不是统计现象。
2. **TopN 在合格过滤之前取**：不合格币会占用名额，把「固定规模」偷偷缩小。修复：先过滤再取 TopN。
3. **币数口径掩盖覆盖率筛选**：`requested == evaluated`，读者看不出 15m「521 币里只有 30 币可测」。修复：分列 present / qualified / evaluated（H8）。
4. **合并报告按插入序列臂、且丢掉分报告的「凑不满」边界**：修复为 `arm_names(sizes)` 规范序 + 透传带周期前缀的 shortfall 条目。
# H-A 市场级杠杆择时量具预置 回测／寻优报告（工程预置，非 edge 判定）

> 生成时间：2026-11-12 01:05 (Asia/Shanghai) ｜ 脚本：`backend/scripts/ha_market_timing.py` ｜ 命令：见文末
> 版本：首版 ｜ **本报告属「仅工程预置／诊断，不做 edge 判定」**：未跑任何回测、未算任何相关性或收益、未产出任何启用建议。
> 假设卡：`research/hypotheses/H-A-market-leverage-timing.md`（P13-000 已登记）｜ 台账：`P13-001`

## 假设与边界

- **信息源 / 可证伪假设**：信息源＝衍生品 `sentiment.oi_value_usd`（H2 止损线允许的四个新信息源之一，**不是**现货 OHLCV、**不是** K 线级 taker 聚合）。假设：「若全市场平均 OI 7d z-score 处于低位时开启全市场多头、高位时关闭，则相对恒定满仓基线能取得正的净差」。事前承诺的终止条件见 §判定，全部写死在 `ha_portfolio.termination_checks`，不由事后调参改写。
- **周期 / 持有期 H**：H＝24 根 1h bar（24h）。标签前瞻跨度与 H 一致；评估用**非重叠 24h 块**，不用逐小时重叠样本，避免 P12/P14 里「IC 随 H 单调上升」的重叠标签伪影。
- **窗口起止 / 墙钟对齐**：块起点按 UTC 零点对齐（`day_start = ceil(sentiment_start / DAY_MS) * DAY_MS`），当前可用块网格 2026-10-07T00:00Z 起、步长 24h。目标评估窗口＝60 个非重叠块（前 40 训练 / 后 20 样本外）。**当前尚未达到，见 §结果**。样本期基准涨跌（`market_background.cum_return` / `up_block_share`）已在量具里预留输出位，用于日后 beta 归因；本轮为 `null`。
- **币池 / 币数**：`sentiment` 515 币 ∩ `klines(interval='1h')` 535 币 ＝ **515 币**（交集即 515，无 skip）。因子取横截面均值，要求每期有效币数 ≥ `MIN_CROSS_SECTION=30`，不足则该期置 NaN 而非降门槛。成员资格按时点可得性构建：未来才上市的币不会回填影响更早时点的读数（已有单测锁定）。
- **止损线核对**：已查 `.workbuddy/skills/strategy-backtest-playbook/references/verdicts-history.md`。§1 止损线覆盖「现货价格序列 + K 线级 taker 聚合」的一切新特征／参数／标签／周期／阈值／止损倍数；本课题用的是衍生品 OI 的**市场级时序**口径，与 P9（逐币时序 IC）、P12（横截面去 beta rank IC）均不同，**未命中止损线**。同时明确：**不重跑 P12 的横截面 rank IC 判据，不靠扩窗或改 H 去救 `inconclusive`**。
- **对照组**（全部已实现并有单测，本轮**未执行**）：
  1. 恒定满仓基线 `constant_full_returns`；
  2. 匹配随机开关基线 `matched_random_switches`，200 种子，要求超过 p90；
  3. 市场级开关 ON/OFF 的受控 A/B（同币池、同成本、同块网格，只差开关）；
  4. 正对照 `perfect_foresight_net`（事后完美预知，用于验证量具能检出真信号）＋ 负对照（纯噪声必须被终止条件拒绝）。

## 口径与参数

- **成本**：`fee_rate=0.0004`、`slippage=0.0002`、双边合计 **0.12%**；口径＝**名义本金（notional）**，全程不切换为 margin，不与任何 margin 口径报告混比。成本按**精确出场名义本金**计：一段连续 ON 区间总成本 ＝ `c * (1 + exit_price/entry_price)`，**不是**固定 `2*fee` 近似（已有恒等式单测）。策略／绑定参数不得覆盖 `fee_rate` / `slippage`。
- **参数来源**：假设卡冻结值，非网格搜索。`HORIZON=24`、`ROLL_WINDOW=168`、`ROLL_MIN_PERIODS=72`、`MIN_CROSS_SECTION=30`、`MIN_SPAN_DAYS=60`、`TRAIN_DAYS=40`、`OOS_DAYS=20`、`MIN_BLOCKS=30`。阈值网格 `THRESHOLD_QUANTILES=(0.2..0.8)` **只在训练段上选**，`select_threshold(train_factor, train_ret, ...)` 的函数签名刻意收不到样本外数据（有专门的泄漏单测：扰动 OOS 后阈值必须不变）。
- **样本外切分**：40 训练 / 20 样本外，按时间顺序切，`split_train_oos` 在可用块不足 60 时直接抛 `ValueError`，不静默缩短窗口。配对差 CI 用 block bootstrap（`block=1` 与 `block=4` 各出一份，`n_boot=2000`、`seed=20261112`）。**成交笔数门槛**：非重叠块 ≥30 用的是**训练＋样本外合计 60 块**；样本外单独只有 20 块，低于 30，这一点作为局限显式登记，不当作已满足。
- **撮合**：本量具是**块级开关**口径，不是逐 bar 撮合。每块按块收益结算，开关切换时按该块名义本金收一次成本，段末强制平仓并把出场腿成本记到最后一块。未模拟盘中止损止盈、跳空、同根优先级——因为 H-A 的假设本身是「持有整块 / 空仓整块」，不含盘中出场。
- **数据与覆盖**（本轮只读实测）：

| 项 | 起 | 止 | 币数 |
| --- | --- | --- | --- |
| `sentiment` | 2026-10-06T04:00Z | 2026-11-11T16:00Z | 515 |
| `klines` 1h（native） | 2025-09-20T15:00Z | 2026-11-11T15:00Z | 535 |

  1h 是唯一有整年深度的周期；15m 有 461 币只从 2026-10 起，5m 只有 2 个 native 币，1m 约 91–94 天。本课题只用 1h ＋ sentiment，不依赖 derived 周期。标签要求 t..t+H 每根 bar 都存在（反向 rolling 完整性掩码），缺任一根则该块置 NaN，**不插值、不前视**。

## 结果

**本轮没有任何绩效结果。** `metrics = null`、`trades = 0`、`default_enabled = false`、`status = BLOCKED_BY_DATA`。下表是**数据门禁读数**，不是收益读数。

| 门禁 | 当前 | 要求 | 缺口 | 结论 |
| --- | ---: | ---: | ---: | --- |
| sentiment 跨度 | 36.50 d | ≥60 d | 23.50 d | 未过 |
| 非重叠块总数 | 35 | ≥60 | 25 | 未过 |
| 评估窗口内可用块 | 0 | 60 | 60 | 未过 |
| z-score 预热前缀块 | 3 | — | — | 需额外等待 |
| 内部空洞 `interior_gap` | false | false | — | 无（等待可修复） |

- **两个到期日不一致，取晚的那个**：按跨度算 ETA＝**2026-12-05T04:00Z**；按可用块算 ETA＝**2026-12-09T16:00Z**（`extra_days_lower_bound=28`）。差 4 天的原因是 72h 最小周期的 z-score 预热吃掉开头 3 个块。**12-05 这个「60 天 mark」出自 P15 审计（`docs/audit-p15-collection-and-ledger.md:77`）与台账 P15-000 / P15-001，属跨度口径；H-A 假设卡本身只写「≥60 个完整日」，未给具体日期。按可用块口径，实际可判定日应更正为 ~12-09。**
- **预热 vs 内部空洞必须分开**：`block_readiness` 数出前导无效块 `warmup_prefix_blocks`，并把它计入 `extra_days_lower_bound = max(0, need + warm - n)`；只有落在预热之后的无效块才标 `interior_gap=true`。前者随采集推进自然消失，后者**等下去也不会好**，必须查采集链。当前 `interior_gap=false`，说明缺口纯粹是「时间还没到」，不是数据链断了。
- **绝不静默丢块**：块级复利要求块序列连续，丢块会把收益算错。所以门禁不过就 `BLOCKED_BY_DATA` 退出，而不是「用现有的 35 块先跑一版」。

### 预置过程中由单测抓出的两个真实缺陷（工程结论，非 edge 结论）

1. **拟合马尔可夫随机基线会伪造 alpha（已修）**：原设计用拟合的两状态马尔可夫链生成随机开关。单测 `test_matched_baseline_preserves_turnover_scale` 失败（换手比 0.48 而非 1.0）。根因：信号高度持续时观测不到 OFF→ON 转移，`p_off` 被估成 1.0，链进入**吸收态**，随机基线换手系统性偏低 → 成本差被当成 alpha。**改为 `matched_random_switches`：把 ON 游程长度集合与 OFF 游程长度集合各自内部置换，再按原 pattern 铺回**，逐序列精确保持曝光、游程条数与线性转移次数。单测从「容差相等」加强为「精确相等」。`fit_markov` 降级为只报告的持续性诊断，不再充当基线。
   - 同时排除了另两种基线：iid Bernoulli（不保持游程结构，换手偏高）、循环平移 `np.roll`（保持曝光但破坏「信号—收益」的时间对齐方式与本课题的块级口径不符）。
2. **60 天跨度 ≠ 60 个可用块（已修）**：真实数据 H8 校验（只看形状／NaN 计数／对齐，不看任何相关性或收益）暴露出预热会让 60 天窗口只剩 ~56 块，`split_train_oos` 会在 23 天后直接抛异常。已加 `block_readiness` 并接进 CLI 门禁。

### 采集链保修核实（只读）

- 三个同步脚本均存在、可运行、支持 `--dry-run`；计划任务 cron 处于 enabled，下一次 2026-11-12 08:30 Asia/Shanghai。
- 逐日行数：11-08 / 11-09 / 11-10 各 12336 行（完整日），11-11 为 8722 行（截至 16:00Z 的当日部分）。**采集在推进，不是停摆**。
- P15-000 曾报 F1 CRITICAL（cron 停用），该状态已在 2026-11-11T16:19:53Z 修复，晚于 16:16:22Z 的那次审计；P15-001 已记 PASS(ops)。本轮复核确认仍为 enabled。

### 全量回归（本轮实测）

`.\.venv\Scripts\python.exe -m pytest backend/tests -q` → **6 failed / 1906 passed**，117.83s。
其中本轮新增的 51 项 H-A 单测**全绿**。6 项失败**全部先于本轮存在、与本轮改动无关**，证据：
三个失败测试文件对 `ha_common` / `ha_portfolio` / `ha_market_timing` 的引用数均为 **0**，
且 `git status backend/app/` 为**空**（本轮未改动任何生产代码，只新增脚本与测试文件）。

| 失败项 | 数量 | 直接原因 |
| --- | ---: | --- |
| `test_backfill_rate_limit.py::test_run_checkpoints_and_limits_writes_to_month[done/partial/error]` | 3 | `httpx.InvalidURL: Invalid port: ':1]'`（测试用 URL 被 httpx 解析失败，环境／版本相关） |
| `test_market_performance.py::test_complete_tail_queries_have_a_bounded_minute_range` | 1 | `lower_bounds` 比期望早 1 分钟（300000ms），断言不匹配 |
| `test_pool_volatility_research.py::test_live_entrypoints_reject[×2]` | 2 | `AttributeError: app.trading.dispatch 无属性 halt_status`（测试打桩的符号已不在生产代码里，属陈旧测试） |

**本轮不修这 6 项**：超出「H-A 量具预置 + 保采集」的授权范围，且相关文件可能正被并发会话改动。
如实登记，交由用户决定何时处理。

## 判定

- **verdict**：`insufficient`（词表取值）。理由：数据门禁未过，**本轮不做 edge 判定**，既不写 `pass` 也不写 `negative`。
- **池化 CI**：未计算。已登记的判据是「开关净收益 − 恒定满仓净收益」的配对 block bootstrap **置信区间**下界 ≤0 即终止（`ci_low` / `ci_high` 字段在 JSON 里预留为 `null`）。`MIN_POOLED_TRADES=30` 的口径按非重叠块计，训练＋样本外合计 60 块满足、**样本外单独 20 块不满足**，已登记为局限。
- **随机基线**：已实现未执行。`n_seeds=200`、`seed=20261112`，判据为净差超过匹配随机开关分布的 **p90**（`RANDOM_P90=90`）。`random_percentile` 本轮为 `null`。
- **归因**：本轮无归因读数。量具已预留 `market_background`（样本期等权全市场累计收益、上涨块占比）用于日后区分「择时能力」与「样本期 beta」；`train_rho` / `oos_rho` 均为 `null`。
- **与历史结论对照**：与 `verdicts-history.md` §9 的 P13-000（`BLOCKED_BY_DATA`）一致，本轮把「等待」变成「量具已就绪、只等数据」。**不与 P12-001/002 的 `inconclusive` 冲突**——那是横截面选币口径，本课题是市场级时序择时口径，两者被测的能力不同。P14-001 的教训（gate 6/6 全过但 6/6 稳健性红旗、79.5% 方差来自币种固定效应）已内建为对本量具的要求：块级口径天然不含币种固定效应，但仍需在真正评估时补时间分段同号与非重叠块数检查。
- **事前终止条件**（写死在代码里，`termination_checks` 已有 8 项单测）：
  1. 样本外 Spearman 变号，或 `|rho| < 0.10`；
  2. 开关净收益 ≤ 恒定满仓基线，或配对 CI 下界 ≤0；
  3. 低于匹配随机开关 p90；
  4. 非重叠块 <30，或时间四分段同号 <3/4。

## 局限

- **未模拟**：资金费率、盘口深度与冲击成本、强平、分批建仓、盘中止损止盈、跳空。块级开关假定整块持有／整块空仓，与模拟盘巡检撮合存在口径差异。
- **成本口径限制**：本报告一切数值为**名义本金**口径，**不得与 margin 口径报告直接相减或混比**。
- **样本外块数不足**：样本外仅 20 块 < 30，即使日后数据到齐，样本外单独的统计功效仍偏弱；需按假设卡补样本方案处理，不得靠改 H 或改门槛绕过。
- **数据深度**：sentiment 仅 36.5 天且只覆盖 515 币；1h 之外的周期深度不足，本课题无法在更细周期上复核。
- **阈值选择的多重比较**：7 档分位阈值在训练段上择优，本身引入选择偏差；真正评估时需报告训练段全部 7 档读数，不能只报最优档。
- **并发写入风险**：`research/experiments.jsonl` 与 `docs/` 是多会话共享写点。本轮追加前已重读尾部（现有 7 条：P12-001/002、P13-000、P14-001、P15-000/001、P16-000），选用未占用的 `P13-001`，未改动并发会话的 P14/P15/P16 产物。
- **台账读数会被并发改动**：会话内早前一次读数为 187 笔 / 净 −425.52 USDT，本轮复核为 179 笔 / 净 −423.79 USDT，说明两次读数之间台账被改动（疑似并发对账／归档，成因未核实）。**本报告只采用文末命令可复现的当前查询结果。**

## 结论与后续

- **一句话结论**：H-A 市场级杠杆择时的**量具已预置完成并通过 51 项单测**，但数据门禁未过（跨度缺 23.5 天、可用块缺 28 天），**本轮不做 edge 判定**，verdict＝`insufficient`，`status=BLOCKED_BY_DATA`。
- **是否启用**：`default_enabled = false`，无绑定、无 paper、无 live、无下单。非 `pass` 一律不启用。
- **自动化不得自启动**：CLI 默认只做就绪检查；即使数据到齐也只会输出 `READY_AWAITING_HUMAN_AUTHORIZATION`，必须显式 `--evaluate`，且按假设卡 §9 由**人工重新登记假设卡并授权**后才能进 Gate 1。
- **后续动作（按优先级）**：
  1. 保住每日 08:30 采集链，盯 `interior_gap` 是否翻 true（翻 true 说明数据链断了，比因子读数更紧迫）；
  2. ~2026-12-09 复核 `block_readiness.ready`，到齐后请用户重新登记 H-A 卡并授权；
  3. 授权后一次性跑 `--evaluate`，报告训练段全部 7 档阈值、样本外配对 CI、200 种子随机基线分位、四分段同号；
  4. 若命中任一终止条件 → 判 `negative` 并登记止损线，**不再调阈值／H／块长去救它**。

## 风险登记：负 edge 绑定仍在运行（未经统计支持）

按用户本轮明确选择「**先不动，只登记风险**」，绑定 **未做任何改动**，现如实登记其风险。

- **绑定**：`id=7`，`supertrend_current × pool=all × interval=15m`，`enabled=1`，`exit_method=strategy`，`params={}`。当前**唯一**启用的绑定（其余 6 条 `enabled=0`）。
- **回测判据（历史，非本轮产出）**：`docs/interval-exit-verdict.md:34` — 同策略 15m 单一 3R 每笔 **−0.299%**，95% CI **[−0.382, −0.209]**，7459 笔，verdict＝`negative`；同策略 1h 为 −0.016%（CI 跨 0），1611 笔，换手只有 15m 的 1/4.6。
- **实盘台账复核（本轮只读实测，2026-11-06 → 2026-11-12，剔除 10 笔 `env_switch_archived` 零盈亏归档行）**：

| 口径 | 笔数 | 毛盈亏 USDT | 手续费 USDT | 净 USDT | 每笔净 % 名义本金 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 绑定 7（可归属） | 118 | −499.07 | 132.19 | **−631.26** | **−0.3943%** | 26.27% |
| 同策略同周期但 `binding_id IS NULL`（11-02→11-03 遗留） | 51 | +359.03 | 143.13 | +215.89 | +0.1206% | 39.22% |
| 合计 `supertrend_current × all × 15m` | 179 | −140.05 | 283.75 | −423.79 | −0.1177% | 28.49% |
| 全台账 CLOSED | 282 | −1383.67 | 418.59 | −1802.27 | −0.3407% | 29.43% |

- **方向拆分（绑定 7，剔除归档）**：LONG 66 笔 −277.53、SHORT 52 笔 −221.54 —— **两个方向都亏**，不是单边行情造成的。
- **出场结构（绑定 7，含归档）**：`exchange_sl` 44 笔 −715.16、`stop_loss` 19 笔 −375.36、`signal_reverse` 28 笔 −143.34、`exchange_tp` 16 笔 +543.88、`take_profit` 1 笔 +110.31、`manual` 10 笔 +80.59、`env_switch_archived` 10 笔 0.00。止损类合计 63 笔 −1090.52，是亏损主来源。
- **一致性判断**：可归属子样本的 **−0.3943%/笔** 与回测 **−0.299%（CI[−0.382,−0.209]）** **同向、同量级**，点估计比回测 CI 下沿还略差。118 笔远小于 7459 笔，**不能据此断言两者差异显著**，但足以说明实盘没有推翻回测的负判定。
- **风险标注**：该绑定**未经统计支持**（H11）。按用户选择继续运行，**本轮不改 `enabled`、不改参数、不改出场口径、不下任何单**。
- **建议（不自动执行）**：若后续要止损，最小改动是停用绑定 7 或把周期切到 1h（换手降 4.6 倍、回测 CI 跨 0），二者都需用户明确授权后再动。

## 复现命令

```powershell
# 1) 就绪检查（默认只读、不评估、不跑回测）
.\.venv\Scripts\python.exe backend/scripts/ha_market_timing.py --out output/ha-readiness-20261112.json

# 2) 量具单测（51 项）
.\.venv\Scripts\python.exe -m pytest backend/tests/test_ha_common.py backend/tests/test_ha_portfolio.py backend/tests/test_ha_market_timing.py -q

# 3) 全量回归
.\.venv\Scripts\python.exe -m pytest backend/tests -q

# 4) 报告静态预检
.\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/ha-market-leverage-timing-harness

# 5) 风险登记复核（只读 SQL，见 docs/ha-market-leverage-timing-harness.json 的 risk_register.queries）
#    数据到齐且人工授权后才允许：
#    .\.venv\Scripts\python.exe backend/scripts/ha_market_timing.py --evaluate --seeds 200 --out docs/ha-market-leverage-timing
```

## 产物

- `docs/ha-market-leverage-timing-harness.md` + `docs/ha-market-leverage-timing-harness.json`（本报告，成对）
- `backend/scripts/ha_common.py`（126 行）、`ha_portfolio.py`（196 行）、`ha_market_timing.py`（197 行）—— 均 ≤300 行
- `backend/tests/test_ha_common.py`（19 项）、`test_ha_portfolio.py`（25 项）、`test_ha_market_timing.py`（7 项）＝ **51 项全绿**
- `output/ha-readiness-20261112.json`（CLI 就绪读数）
- `research/experiments.jsonl` 追加 `P13-001`；当日工作记忆已登记
- **未改动**：任何策略代码、任何绑定、任何参数、并发会话的 P14/P15/P16 产物
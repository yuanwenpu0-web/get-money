# 条件化止盈判别力探针（下单前／持仓中「其他指标」能否决定及时止盈或推后止盈）

> 生成时间：2026-11-10 07:43 ｜ 脚本：`backend/scripts/exit_metalabel_{probe,stats,rules,eval}.py` ｜ 命令：见文末
> 版本：首版 ｜ 课题性质：Gate 1 探针 + 预注册负对照 A/B，**未实现任何生产规则**
> 成对产物：`docs/exit-metalabel.json`

## 假设与边界

**用户问题**：能否用「其他指标」判断某笔订单该及时止盈还是推后止盈，从而获得更高盈利？

- **信息源 / 可证伪假设**：假设存在至少一个在**决策时刻可得**的指标，能以 `|Spearman IC| ≥ 0.02` ∧ 按币 block-bootstrap 同号率 `≥ 0.60` ∧ `FDR q < 0.10` ∧ 时间分段同号 `≥ 3/4` 预测「该早止盈还是晚止盈」的条件收益差。四关全过才算探针成立；**事前承诺：任一关不过即按 H1 停止，不进入实现**。
- **与上一轮的差别（Gate 0 查重）**：`docs/trend-reversal-order-exit.md` 测的是**无条件静态规则**（统一降/抬/去掉 TP、跟踪、放宽 SL），本轮测的是**条件化判别**（同一笔订单按指标选择不同动作）。两者不是同一数学对象，不属重复课题。
- **止损线核对**：已查 `references/verdicts-history.md` §1。**OHLCV 派生指标命中 H2 止损线，只作负对照与量具校准，不投入实现、不调阈值／周期／止损倍数**；订单元数据、衍生品情绪/资金费率、持仓中路径状态量才是本轮允许检验的信息源。
- **两层决策点（均严格因果）**：
  - **T0 下单时刻**：只用 `created_at` 之前 128 根 K 线，标签 `y0 = 晚止盈族净收益 − 早止盈族净收益`（正＝该推后止盈）。
  - **T1 持仓中首次浮盈 ≥ 0.7R 且尚未触发任何 barrier 的 bar**：只用 `bars[0..k]`，标签 `y1 = 继续持有净收益 − 该 bar 收盘立即平仓净收益`（正＝该推后止盈）。
- **窗口起止 / 墙钟对齐**：2026-10-26 ~ 2026-11-07，13 天；与上一轮同一订单队列、同一 `x - x % step` 1m→周期派生桶、同一固定 A/B 前瞻窗口（paper 24h / live 6h）。样本期 BTC 80,579.7 → 75,728.0，**−6.02%**（本地 PG `klines` 1m `BTCUSDT`，2026-10-26 00:00 ~ 11-07 23:59 +08:00），供 beta 归因。
- **币池 / 币数**：真实订单队列，**78 币、105 笔市场规则出场**（15m 88 / 1h 17），T1 层 54 笔有可行动决策点。`universe.skipped` = 8 笔 `sl_not_adverse`（旧策略把持仓期保护价写进 `sl`，按 H8 剔除）；另 10 笔行政类平仓排除 A/B；`manual` / `research_hold` 共 59 笔因 tp/sl/close_price 不全根本未进入查询。
- **对照组**：无条件基线（原 TP/SL）、20 种子随机臂分配基线、正/负/半衰三对照自校准、以及 LOOKAHEAD oracle 上界。

## 口径与参数

- **成本**：`fee_rate=0.0004`、`slippage=0.0002`，开平双边合计 **0.12%**；口径＝**名义本金**（全程不切换，不与 margin 口径报告混比）。T1 的 `y1` 两腿用同一成本模型，成本在差值中抵消。
- **参数来源**：无网格。指标族**事前预注册**：T0 16 列（10 个 OHLCV 负对照 + 几何/元数据 + 衍生品），T1 10 列（持仓中路径状态）。衍生品 7 列因 H8 覆盖不足被排除出主探针。
- **样本外切分**：按订单时间 60% 训练 / 40% 测试，切分点前后各 **24h purged 隔离**（标签窗口为 24h），实际 33 / 36 笔；`MIN_POOLED_TRADES = 30`。训练段只选指标与中位数分界，**测试段只读**（H5）。
- **撮合**：沿用 `order_exit_common.run_rule`，`engine=sequential`，跳空按开盘、同根 SL 优先、触发价成交；所有臂共用同一段 bar 序列与同一成本。
- **数据与覆盖**：derived（1m 聚合，完整连续桶）；`sentiment` 止于 2026-10-28 08:00、`funding_rates` 止于 2026-10-27 20:00。**新鲜度门槛 24h**：paper 新鲜情绪/资金各 **17/105**、live **0/27**。陈旧行不计为覆盖（H8）。`note` 可解析 `score`/`prob_up` 的 paper 订单有限，live 订单该字段全空。
- **标签结构**：`y0` 中 **61/105 为精确 0**（早/晚 TP 都被同一个 SL 先打掉，两臂收益相同）。这类订单「该早还是该晚」本身无答案；**未剔除**，因为按未来结果筛样本会引入选择偏差。

## 结果

### 1. Gate 1 探针：26 列，0 列过四关

| 层 | 检验列 | 最大 \|IC\| | 该列 FDR q | 该列分段同号 | 全层最小 q | 过四关列数 |
|---|---:|---:|---:|---:|---:|---:|
| T0 下单前 | 16 | `range_pos_128` +0.2206 | 0.3198 | 3/4 | **0.3198** | **0** |
| T1 持仓中 | 10 | `dist_to_tp_R` +0.2379 | 0.4673 | 4/4 | **0.4673** | **0** |

- T0 次高 `mom20` IC −0.1978（q 0.3398、分段 3/4）；T1 次高 `bar_range_R` IC +0.2316（q 0.4673、分段 3/4）。
- **四关里 |IC| 与同号率几乎全部通过，卡住的是 FDR**：26 次检验里最好的 q 仍是 0.32，远高于 0.10。这是典型的「挑出来的相关是多重比较噪声」——`range_pos_128`、`mom20`、`atr14_over_R` 本质都是入场前波动/位置的不同写法，且它们与 `y0` 的关系可由「TP≈2.69R、高波动路径先触 SL」的几何关系解释，不是新增信息。
- AUC 辅助读数同样不分离：T0 最好 0.5850，T1 最好 0.6634。

### 2. 量具自校准（不可跳过，judgement.md §3 第五步）

| 对照 | T0 | T1 |
|---|---:|---:|
| 正对照（偷看未来方向 oracle）IC | **0.6943** | **0.8638** |
| 负对照 20 种子 p95 \|IC\| | 0.0997 | 0.2029 |
| 半衰对照 IC | 0.4207（比值 0.606） | 0.6432（比值 0.745） |

正对照显著高于随机、半衰对照明显衰减 → **量具可用**，探针的 0 pass 不是量具失灵造成的。

**一个比探针本身更说明问题的对照**：把「完美知道未来窗口的最大有利偏移（MFE）」当作指标，它对 `y0` 的 IC 只有 **0.0928**（T1 对 `y1` 是 0.4225）。也就是说，**该早止盈还是晚止盈，并不由「能涨多高」决定，而由「先冲高回落」还是「慢慢走到位」的路径顺序决定**；下单时刻与浮盈时刻可得的任何指标都拿不到这个顺序。这与上一轮「止盈过早 / 止盈不及时是同一个固定 TP 在未知未来下的两面」是同一结论的更强版本。

### 3. 两条预注册条件规则（负对照性质，不是投产候选）

| 规则 | 段 | 笔数 | 配对差 % | 95% CI % | p_perm | 随机分位 | ci_verdict |
|---|---|---:|---:|---|---:|---:|---|
| `metalabel_T0`（训练段选 `mom20`，阈值 0.0118，按 IC 符号映射） | 训练 | 33 | +0.3013 | [−0.2510, +0.9790] | 0.3832 | 0.976 | `inconclusive` |
| | **purged 测试（只读）** | **36** | **+0.5309** | **[+0.1730, +1.0075]** | **0.0140** | 0.881 | `pass`（单关） |
| | 全样本 | 105 | +0.2102 | [−0.1363, +0.5900] | 0.2599 | 0.976 | `inconclusive` |
| `metalabel_T1`（`giveback_R_sofar` 阈值 0.2407，回吐大则立即平） | 全样本 | 54 | +0.1752 | [−0.4363, +0.8087] | 0.5725 | 0.643 | `inconclusive` |
| | purged 测试 | 20 | +0.6837 | [−0.1189, +1.4962] | 0.1343 | 0.405 | `insufficient` |

**测试段那个 CI 下界为正的读数必须按 `likely_noise` 处理，不能当成 edge**：它是在 16 列里训练段按 \|IC\| 选出的冠军；该特征全样本 **FDR q = 0.340**；全样本与训练段 CI 都跨 0；20 种子随机基线分位 **0.881 < 0.95**（H6 门槛）；且 Gate 1 探针 0/26 过线（H1 禁止实现）。训练段 `+0.3013` 与全样本 `+0.2102` 远小于测试段 `+0.5309`，是分段翻转而非稳定增益。

### 4. Oracle 上界（LOOKAHEAD，不可交易）

| 量 | 每笔 % |
|---|---:|
| 真实订单基线 | **−0.4841**（CI [−0.8386, −0.1202]，`negative`） |
| 每笔都选对早/晚 TP 臂 | +0.1462 |
| 每笔在浮盈决策点都选对「立刻走 / 继续拿」 | +1.7589 |

上界看着可观，但用的是已实现的未来标签。**条件化能把亏损改写成盈利的前提，是先知道那条路径会怎么走**——这正是探针测出来不存在的东西。

### 5. 衍生品子集与 live 队列（H8 数据边界）

- 新鲜（<24h）情绪/资金费率仅覆盖 **17/105** paper、**0/27** live 订单 → `insufficient`。7 列里最大 `|IC| = 0.6109`（`sent_oi_z30`）纯属 17 笔小样本 + 7 次比较的数值，**数值照给、不下结论、不进主判定**。这是数据缺口，不是「指标无效」。
- live 27 笔（币安 DEMO，6h 窗口，19 个 T1 决策点）：`y0` 均值 +0.9107%、`y1` +0.7331%，探针 0 列过 gate，判 `insufficient`（27 < 30 笔门槛），与 paper 不合并。

## 判定

- **verdict**：**`inconclusive`**（主，`metalabel_T0` purged 测试段）；**`baseline_verdict` = `negative`**；衍生品子集与 live 队列均 **`insufficient`**；测试段单规则读数登记为 **`likely_noise`**。按词表，`inconclusive` 是「不知道」，不等于「已证无效」，也不等于「亏」。
- **池化 CI**：测试段 36 笔 ≥ 30；全样本 105 笔。`MIN_POOLED_TRADES=30` 满足；但全样本 CI 跨 0。
- **三关**：测试段 CI 下界为正 ✅、`FDR q < 0.10` ❌（q=0.340）、时间分段同号 ≥3/4 ✅ → **2/3，按「与」关系不成立**。
- **随机基线**：20 种子，观测值分位 0.881（测试段）/ 0.976（全样本），测试段未过 0.95 门槛。
- **归因**：样本期 BTC −6.02%，真实每笔 −0.4841% CI 全负；毛利不足以覆盖 0.12% 双边成本（历史 `flow_imbalance_long` 同类结论）。本轮所有条件化尝试的配对差都在 ±0.7pp 内且 CI 跨 0，量级远小于路径本身的 2–5R 波动 → **瓶颈是路径顺序不可预测，不是出场阈值或指标选得不好**。
- **与历史结论对照**：与 `references/verdicts-history.md` §1 止损线、§4「不要再调止损倍数／持有期救它」、`docs/interval-exit-verdict.md`（分批止盈单调变差）、`docs/ema-trend-filter.md`（通用入场门控 64 行 0 pass、FDR q 0.606–0.968）一致。**无冲突**：本轮把「门控」从入场搬到了出场，结论同样是不增信息。

## 局限

- 样本仅 13 天、105 笔、78 币，其中 61 笔 `y0` 是精确 ties，统计功效低；`inconclusive` 部分源于样本量而非已证明无关系。
- 衍生品数据末端为 2026-10-28 / 10-27，与订单窗口后半段不相交；`oi_value_usd`（P9，唯一过 gate 的 alpha）在本队列上无法受控检验，**需等数据补齐后另立课题**。
- `sl` / `tp` 是平仓时存的值，保护单可能在持仓期间被更新；`params_snapshot` 全 NULL，无法还原下单瞬间原始 TP/SL（详见 `docs/trend-reversal-order-exit.md`）。
- OHLC 重放与交易所条件单按标记价触发存在差异，上一轮测得出场原因一致率 83.8%。
- T1 决策点只在 54 笔上存在（其余更早被 barrier 打掉），属结构性删失，不是信号筛选。
- paper 24h 与 live 6h 是不同队列，数值未合并；oracle 数字含 LOOKAHEAD，不可交易。
- 未模拟资金费率收取、盘口深度、强平、分批建仓、部分成交；名义本金口径，不得与 margin 口径报告相减。
- 单测覆盖纯因果与统计函数，但未覆盖生产撮合；本轮**未改任何策略 spec、绑定或后端代码**。

## 结论与后续

**一句话结论**：在 26 个预注册指标、两层决策时刻上，**没有找到任何能在决策时刻判别「及时止盈 vs 推后止盈」的信息**；连「完美知道未来最大有利偏移」对下单前标签的 IC 也只有 0.0928 —— 说明缺的不是指标，是路径顺序的可预测性。测试段单条规则 CI 下界为正属 `likely_noise`，不构成 edge。

- **是否启用**：**否**。`default_enabled = False`，不进任何绑定，未新增策略 spec，未改生产参数。
- 按 H1（探针不过不实现）与 H2（止损线），**不再调出场阈值、TP 倍数、周期、止损倍数或继续扩充 OHLCV 指标族**；按手册对 `inconclusive` 的要求，只写补样本方案，不写启用建议。
- 若要继续：① 补齐 `sentiment` / `funding_rates` 到订单窗口末端，再用 ≥60 天覆盖做 P9 与出场条件化的受控 A/B；② 扩订单样本到 ≥300 笔市场规则出场，降低 ties 占比后再复检 T1 层；③ 下一个信息增量仍在**逐笔成交顺序 / 盘口挂撤单 / 多所现货腿**，而非价格序列的再组合。

## 复现命令

```powershell
.\.venv\Scripts\python.exe backend/scripts/exit_metalabel_probe.py --source paper --horizon-hours 72 --ab-hours 24 --out output/exit_metalabel_paper.json
.\.venv\Scripts\python.exe backend/scripts/exit_metalabel_probe.py --source live  --horizon-hours 6  --ab-hours 6  --out output/exit_metalabel_live.json
.\.venv\Scripts\python.exe backend/scripts/exit_metalabel_eval.py --out docs/exit-metalabel.json
.\.venv\Scripts\python.exe -m pytest backend/tests/test_exit_metalabel_features.py -q
.\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/exit-metalabel --strict
```

## 产物

- `docs/exit-metalabel.md` + `docs/exit-metalabel.json`（成对）
- `backend/scripts/exit_metalabel_{features,stats,rules,probe,eval}.py`（各 < 300 行，纯离线只读）
- `backend/tests/test_exit_metalabel_features.py`（10 项单测全绿）
- 验证：`preflight.py --strict` **0 FAIL / 0 WARN**；全量 `pytest backend/tests -q` 3 项失败经收集顺序定位为仓库基线既存（`test_market_performance::test_complete_tail_queries_have_a_bounded_minute_range` 1 条 + `test_pool_volatility_research::test_live_entrypoints_reject` 2 条，与 `.pytest_cache/.../lastfailed` 吻合），**未引入新失败**
- 数据集 `output/exit_metalabel_{paper,live}.json`、当日工作记忆登记

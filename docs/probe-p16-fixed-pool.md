# H-D / P16 固定规模 PIT 币池可构造性与池规模敏感性 · 数据口径验证，非 edge 判定

> 生成时间：2026-11-11 16:35 UTC ｜ 脚本：`backend/scripts/probe_p16_fixed_pool.py`（297 行）｜ 命令见 §7
> 假设卡 `research/hypotheses/H-D-fixed-size-universe.md` ｜ 台账 `P16-000` ｜ 代码基线 `564d941`
> **性质**：Gate 0 + 只读数据口径可行性诊断。无收益估计、无回测、无成交、无策略启用。按手册 §0 例外跳过 Gate 1／Gate 4 统计判定，H3／H8／H9／H10／H12 仍生效。
> **一句话**：固定规模池**可构造但成员不稳定** —— C1／C3 在 58/59 期全过，但周均换手 26–33%；按预注册门槛只有 N=100 形式通过（C2 margin 仅 0.0129），工程判定 `operational_pass`，**不构成「N=100 最优」的证据**。

## 1. 假设与边界

- **信息源**：`klines(interval='1h')` 的 `symbol / open_time / quote_volume`。**不读 `close`、不算收益、不构造任何特征或标签**，因此不触碰 H2 止损线（H2 禁止在现货 OHLCV 价格序列与 K 线级主动成交汇总上投新特征/参数/标签；成交额与 bar 完整性属样本选择口径）。也不重跑 P12 横截面 rank IC 判据、不对 P14 的 `DO_NOT_SCALE_UP` 扩量。
- **可证伪假设**：「把策略限制在固定规模币池」若要在历史验证中可操作，则用**每期事前 168h 成交额 TopN** 逐期重建的池应满足 (a) 绝大多数时点能凑够 N 个合格币；(b) 相邻期成员不整批换血；(c) N≥30 时有效横截面宽度足以支撑 `MIN_CROSS_SECTION=30`。事前终止条件 C1／C2／C3 见 §2，**不得事后改门槛**。
- **窗口 / 墙钟对齐**：2025-09-20 15:00 ~ 2026-11-11 15:00 UTC，417 天，10,009 个 1h 网格点；重建时点从 2025-09-28 00:00 UTC 起每 7 天一期，共 **59 期**，末期 2026-11-08 00:00。事前窗口 `[T−168h, T)`，前瞻窗口 `[T, T+24h)`。**样本期基准涨跌：不适用**（本轮不读价格、无收益估计）。**样本外切分：不适用**（无策略评价）。
- **币池 / 币数**：1h 全表实时核为 535 币，按线上 `pool_maintenance.EXCLUDED`（14 个：`BTCUSDT`/`ETHUSDT`/稳定币与法币对/黄金代币）及 `UP/DOWN/BULL/BEAR` 杠杆代币后缀过滤后 **527 币**、4,861,821 行。因此本课题的 TopN 是「**排除主流与稳定币对后的成交额 TopN**」，不是全市场 TopN，与 `pit_pool_check.py::rebuild` 口径一致。扫描 N ∈ {10, 20, 30, 50, 100}。
- **数据质量（H8）**：`quote_volume` 无 NULL，**56,158 行为 0（1.15%）**；单根 0 仍计有效 bar（不读成「无信号」），窗口总和 `> 0` 才合格。无上市／下架清单，故「当时是否真可交易」无法验证，结论是**基于在库 1h bar 的下界**。
- **止损线核对（Gate 0 查重）**：`docs/binding-combo-optimization.md` 已做「当前成员 vs 时点重建」的收益对照并量化 **8.1pp/笔**事后选币偏差；`pit_pool_check.py::rebuild` 已有按 168h 重建（固定 50/100 分池）的实验能力但未落库。**本课题不重复收益对照**。只读调查确认**不存在**「池规模 N 扫描／成员稳定性／有效横截面宽度」实验，`symbol_pools` 无 `as_of` 字段、全库无 point-in-time 成员资格表，`docs/indicator-evaluation-plan.md:71` 仍把「每月成分 + 下架时间清单」列为待新增。
- **对照组**：无策略臂、无关闭过滤基线、无受控 A/B。对照是**同一口径下的 5 个 N**，以及「逐期 PIT 池」vs「当前成员静态池」的**非价格事后选币代理**（§3.3）。随机基线与 CI 见 §5，本轮不适用。

## 2. 口径与参数

- **成本**：`fee_rate=0.0004`、`slippage=0.0002`，双边 **0.12%**，**名义本金**口径（登记但 `applied=false`）；`trades=0`、无 PnL，不得与 margin 口径或每笔收益口径报告混比。
- **参数来源**：全部来自假设卡 §4 **预注册规格**，非策略默认、非绑定参数层、非网格搜索。单币合格：窗口内 ≥160 根 1h bar、末根不早于 `T−2h`、`sum(quote_volume)>0`；排序成交额降序、并列按 symbol 字母序（确定性）。门槛沿用 `pit_pool_check.rebuild` 的放宽口径。
- **判定门槛（预注册）**：C1 `eligible_count ≥ N` 期数占比 ≥0.90；C2 相邻 7 天 TopN 成员 **Jaccard 中位数 ≥0.60**；C3 仅对 N≥30，`width ≥ 30` 期数占比 ≥0.90。有效宽度要求 `[T,T+24h)` **全部 24 根** bar，属最严格口径。
- **撮合 / 样本外**：均不适用（`trades=0`，无回测、无下一根开盘成交、无腿级 `fill_status_counts`）。
- **并行**：单次 SQL + pandas 向量化聚合（10009×527 矩阵），单进程；非逐 bar `simulate`，H9 进程池不适用。

## 3. 结果

### 3.1 数据边界：首期为何只有 4 币

`eligible_count` 序列：首期 **4**，随后 403 → 423 → 430 → … → 517（峰值），末期 512。核实 DB 确认这是**回填起点不齐，不是量具缺陷**：只有 6 个币的 1h 历史从 2025-09-20 起，其余 **394 个币从 2025-09-22 才开始**，因此首期窗口里这批币只有约 144 根 bar < 160 门槛；下一期窗口即恢复到 408 币合格。该期已列入各 N 的 `failed_periods`，并在 Jaccard 诊断中作为「不可构造对」单独剔除（剔除后读数几乎不变）。

### 3.2 池规模扫描（主结果）

| N | C1 可构造 | C2 Jaccard 中位 | 剔除首期后中位 / 最小 | 周均换手 | 宽度中位 / 最小 | C3 宽度≥30 | gate |
|---:|---:|---:|---|---:|---|---:|---|
| 10 | 0.9831 | 0.5385 | 0.5385 / 0.2500 | 0.331 | 10 / 4 | 0.0000 | **FAIL**（C2） |
| 20 | 0.9831 | 0.5100 | 0.5385 / 0.3333 | 0.344 | 20 / 4 | 0.0000 | **FAIL**（C2） |
| 30 | 0.9831 | 0.5000 | 0.5000 / 0.3953 | 0.329 | 30 / 4 | 0.9831 | **FAIL**（C2） |
| 50 | 0.9831 | 0.5625 | 0.5625 / 0.4286 | 0.298 | 50 / 4 | 0.9831 | **FAIL**（C2） |
| **100** | 0.9831 | **0.6129** | 0.6129 / 0.5038 | **0.256** | 100 / 4 | 0.9831 | **PASS** |

- **C1／C3 基本全过**：58/59 期可构造（唯一失败是首期数据边界）；N≥30 时 `width≥30` 占比 0.9831。N=10/20 的宽度天然不可能 ≥30，按预注册豁免 C3，但 `supports_cross_section_30=false` —— **它们无法支撑 P12 的 `MIN_CROSS_SECTION=30` 判据**；P14 用 24 币／18 只属可行性筛查，不能当正式判据。
- **C2 是唯一分水岭**：Jaccard 中位数随 N 增大（0.50 → 0.61），周均换手随 N 减小（34.4% → 25.6%）。即**头部币的成交额排名轮换比中腰部更频繁**，这是真实读数，不是量具缺陷（剔除首期不可构造对后中位数几乎不变）。
- **N=100 只是勉强过线**：C2 margin **0.0129**，属门槛附近，`jaccard_min_constructible_pairs` 仍有 0.5038。不得据此宣称「N=100 最优」。C2 门槛 0.60 为事前设定，本轮**不得调整**；若要重设须新登记假设卡。

### 3.3 事后选币代理（当前成员池 vs 逐期 PIT 池）

取末期（当前）TopN 为「当前池」，回看每个历史 T 时这些成员在当期成交额排名的分位（1.0=最活跃）与「当期不合格」比例：

| N | 早期中位分位 | 晚期中位分位 | 早期不合格占比（中位） | 全窗口最差不合格占比 |
|---:|---:|---:|---:|---:|
| 10 | 0.9647 | 0.9902 | 0.4000 | 0.6000 |
| 30 | 0.9164 | 0.9559 | 0.1667 | 0.8667 |
| 50 | 0.8929 | 0.9264 | 0.1600 | 0.9200 |
| 100 | 0.8372 | 0.8711 | 0.1500 | **0.9600** |

**读数反直觉但重要**：当前池成员在历史各期「只要合格就排得很靠前」（分位 0.84–0.99），这恰恰是事后选币的特征 —— 它们是**后来才持续活跃**的币。最差的一期里，当前 Top100 有 **96%** 根本不在当期合格集合。N 越小，存活成员的排名分位越高，越容易掩盖「早期大量成员尚不可用」。这与 `binding-combo-optimization.md` 的 8.1pp 收益偏差方向一致，但本轮用**纯成交额口径**独立佐证，未重复任何收益对照。

## 4. 判定

- **verdict**：**`operational_pass`** —— 工程可行性判定，**不是**手册第 3 节的 edge verdict；`verdict_scope` 已声明「无收益估计、无 CI 下界判定、无启用建议」。本轮不作 `pass` / `inconclusive` 的 edge 判定。
- **决策**：`FEASIBLE_WITH_BOUNDS`；`feasible_pool_sizes=[100]`、`pool_sizes_supporting_min_cs_30=[100]`、`recommended_pool_size=100`（**按预注册 gate 的形式结果，附 §3.2 的 margin 警示**）。
- **池化 CI / 随机基线**：**不适用**。`trades=0`，`MIN_POOLED_TRADES=30` 不满足，无 `positive_share`；无多种子方向翻转基线、无分位数读数；JSON 中 `ci.applicable=false`、`random_baseline.applicable=false` 显式声明，未用零值冒充实测。
- **归因**：三条预注册条件里 C1/C3 由「1h 覆盖深度」决定并几乎全过；C2 由「成交额排名轮换」决定且对 N 敏感。**约束的可操作性取决于用户想要哪一种「固定」**：
  - 若指「**固定 N 个币、成分锁定、回放历史**」→ **数据不支持**：周均换手 26–33%，且当前成员在历史最差期 96% 不合格，用当前名单回放历史就是 `docs/binding-combo-optimization.md` 已量化的事后选币。
  - 若指「**固定规模 N、成分逐期按事前信息重建**」→ **可操作**，但须接受换手、须 N≥30 才能满足 30 币横截面判据、且不得用当前成员回放历史。
- **与历史结论对照**：与 `verdicts-history.md` §9 的 P12／P14 方法学一致（IC 达标 ≠ 可交易；样本选择口径会主导读数）；与 §2 止损线不冲突（未投任何价格特征）。本轮补上了此前**缺失**的「池规模敏感性」维度，未重复 8.1pp 收益对照。

## 5. 局限

- 无上市／下架事件表与历史 ticker 快照，无法验证「当时是否真可交易」；`eligible` 只是**在库 1h bar 的下界**，与 `indicator-evaluation-plan.md:71` 的待办一致。
- 首期（2025-09-28）仅 4–6 币合格：394 币的 1h 历史从 09-22 才起，属回填起点不齐；已列入 skip，不得当作「池不可构造」的真实样本。
- `EXCLUDED` 排除了 `BTCUSDT`/`ETHUSDT` 与稳定币／法币对，TopN 不是全市场 TopN；成交额排序不含点差、盘口深度与执行成本，只是流动性代理。
- `quote_volume=0` 单根按有效 bar 计，窗口总和 `>0` 才合格；合格门槛沿用 `pit_pool_check.rebuild` 的放宽口径（1h ≥160 根、末根不早于 `T−2h`）。
- 成员稳定性按 7 天一期测量，未覆盖日内换手；线上池维护周期可能不同。有效宽度要求 `[T,T+24h)` 全 24 根 bar，属最严格口径。
- **只验证样本选择口径**，不做策略收益／IC／净 edge 判定，无 CI、无随机基线、不重复 8.1pp 收益对照；本报告数值不得与 margin 口径或每笔收益口径报告相减。
- 本轮不改任何 `StrategySpec` / `strategy_bindings` / 参数 / 前端，不部署模拟盘、不下单、不启用 live key。

## 6. 结论与后续

- **一句话结论**：固定规模 PIT 池在 1h 数据上**可构造、宽度够**，但**成员并不固定**（周换手 26–33%）；按预注册门槛仅 N=100 形式通过且 margin 仅 0.0129，只能判「有边界地可行」，不能判「N=100 最优」。
- **是否启用**：`default_enabled=False`、无绑定、无模拟盘、无订单。工程判定非 edge `pass`，**不写任何启用建议**。
- **对后续固定池验证的约束**（建议写入 `research/config.yaml`，本轮已追加声明）：
  1. 成分必须**逐期用事前 168h 信息重建**，禁止用当前成员回放历史；
  2. 若跑 `MIN_CROSS_SECTION=30` 的横截面探针，N 须 ≥30 且 `width_ge_30_share ≥ 0.90`（N=30/50/100 均满足，N=10/20 不满足）；
  3. 假设卡须明确「固定规模」还是「固定成分」，并把周换手列为已知成本；
  4. C2 门槛 0.60 若要重设，须**新登记假设卡**，本轮不得改；
  5. 需补历史成员资格／下架清单才能把结论从「在库 bar 下界」升级为「真实可交易性」。
- **其余在途方向不变**：H-A 仍 `BLOCKED_BY_DATA`（sentiment 36.42 天 < 60 天），P12 不重跑，P14 不扩量。自动化不得自行启动回测或启用策略。

## 7. 复现命令

```powershell
.\.venv\Scripts\python.exe backend/scripts/probe_p16_fixed_pool.py --out docs/probe-p16-fixed-pool.json --force
.\.venv\Scripts\python.exe -m pytest backend/tests/test_probe_p16_fixed_pool.py -q
.\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/probe-p16-fixed-pool
```

探针默认拒绝覆盖既有输出，`--force` 才覆盖；`--step-days` 可改重建频率（默认 7）。只读 DB（`make_engine(read_only=True)`），不落盘任何中间数据。

## 8. 产物

- `docs/probe-p16-fixed-pool.md` + `docs/probe-p16-fixed-pool.json`（成对，含 `decision` / `results_by_pool_size` / `posthoc_selection_diagnostic`）
- `research/hypotheses/H-D-fixed-size-universe.md`（Gate 0 假设卡）、`research/experiments.jsonl`（`P16-000`）、`research/config.yaml`（固定池约束声明）
- `backend/scripts/probe_p16_fixed_pool.py`（297 行）、`backend/tests/test_probe_p16_fixed_pool.py`（9 项单测）
- 当日工作记忆与长期记忆已登记

## 9. 自检

- 报告预检 `preflight.py --prefix docs/probe-p16-fixed-pool`：**FAIL 0 / WARN 0 / PASS 21**。
- 新增单测 `backend/tests/test_probe_p16_fixed_pool.py`：**9 passed**；`py_compile` 通过，脚本 **297 行**（项目上限 300）。
- 全量 `pytest backend/tests -q`：**4 failed / 1895 passed / 119.31s**。3 个失败与前几轮完全相同（`dispatch.halt_status` 缺失 ×2、K 线查询下界 ×1）；**新增的第 4 个** `test_ha_portfolio.py::test_markov_baseline_preserves_turnover_scale` 来自本轮期间**并发会话新增的 `ha_*.py` 文件**，与 P16 探针无关，未改动他人未提交代码。
- 探针只读 DB、不落盘中间数据；未改后端服务／前端，无需重启 `QuantDesk-GetMoney` 或 build。

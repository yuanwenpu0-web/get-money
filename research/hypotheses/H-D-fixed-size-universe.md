# P16 / H-D · 固定规模 PIT 币池可构造性与池规模敏感性（Gate 0 假设卡，预登记）

- 状态：`GATE0_REGISTERED` → 本轮执行只读数据口径诊断，**非 edge 判定**。
- 登记时间：2026-11-11T16:20Z，代码基线 `564d941`。
- 触发：用户指令「验证思路，策略限制固定规模的币种池中」。本卡把「验证思路」具体化为**验证「把策略限制在固定规模币池」这一约束本身在历史验证里是否可操作**——它是后续任何固定池策略验证的公共前置。
- 性质：无收益估计、无 CI 下界判定、无策略回测、无成交、不改 spec／绑定／参数／前端、不启用策略。按手册 §0 例外跳过 Gate 1／Gate 4 统计判定，H3／H8／H9／H10／H12 仍生效，标题标注「数据口径验证，非 edge 判定」。

## 1. Gate 0 查重与 H2 止损线核对

- `docs/binding-combo-optimization.md` L117–195 **已做**「当前成员 vs 时点重建」的收益对照，量化事后选币偏差约 **8.1pp/笔**（当前成员 7.100% → PIT 重建 −1.014%）；`backend/scripts/pit_pool_check.py::rebuild` 已有按 168h 重建两池的实验能力（未落库、规则放宽到 1h ≥160 根）。**本课题不重复该收益对照。**
- **确认不存在的增量**（只读代码调查 + docs 检索）：无「池规模 N 扫描／成员稳定性／有效横截面宽度」实验；`docs/indicator-evaluation-plan.md:71` 明确把「每月成分 + 下架时间清单」列为必须新增；`symbol_pools` 无 `as_of`／`valid_from` 字段，全库无 point-in-time 成员资格表；运行层 `market/universe.py::resolve_universe` 与 `strategy/pools.py::resolve_members` 都是「当前时点动态全宇宙池」。
- **H2 核对**：本课题**不读 `close`，不构造任何价格/收益/技术指标/标签**，只用 `klines.open_time` 与 `klines.quote_volume` 做样本选择口径验证。成交额与 bar 完整性不是「现货 OHLCV 价格序列上的新特征」，因此不触碰 H2 止损线；也不重跑 P12 横截面 rank IC 判据，不对 P14 的 `DO_NOT_SCALE_UP` 做任何扩量。
- 与 P12／P14 的关系：P12 横截面探针要求 `MIN_CROSS_SECTION=30`；P14 用 24 币／18 仅作可行性筛查。本课题直接回答「固定 N 取多少才不会让横截面探针失效」，是两者的公共前置。

## 2. 信息源与数据边界（H8）

- 表 `klines`，`interval='1h'`，字段 `symbol / open_time / quote_volume`。
- 实时核表（2026-11-11 16:20Z）：1h 覆盖 **2025-09-20 15:00 ~ 2026-11-11 13:00 UTC**（约 417 天）、**535 币**、4,930,725 行；`quote_volume` 无 NULL，**56,134 行为 0（1.14%）**。
- 逐月 `distinct symbol`：431 → 455 → 463 → 475 → 489 → 496 → 503 → 513 → 520 → 526 → 527 → 493 → 534 → 535，**单调上升**。历史早期可交易宇宙明显小于当前，这是「用当前成员回放历史」必然漏币的直接证据。
- **零值处理**：单根 `quote_volume=0` 仍计为有效 bar（不读成「无信号」）；合格条件用窗口**总和** `> 0`，与 `pit_pool_check.py::rebuild` 一致。
- 币种过滤沿用线上规则：仅 `USDT` 后缀、排除 `pool_maintenance.EXCLUDED` 与 `UP/DOWN/BULL/BEAR` 杠杆代币。**`EXCLUDED` 含 14 个币（`BTCUSDT`/`ETHUSDT`/稳定币与法币对/黄金代币），因此本课题的 TopN 是「排除主流与稳定币对后的成交额 TopN」，不是全市场 TopN**；与 `pit_pool_check.py::rebuild` 口径一致，1h 全表 535 币过滤后为 527 币。
- **无法验证「当时是否真的可交易」**：无上市／下架事件表、无历史 ticker 快照。本课题的可构造性结论是**基于在库 1h bar 的下界**，须在局限中显式声明（与 `docs/indicator-evaluation-plan.md:71` 的待办一致）。

## 3. 可证伪假设

> 「把策略限制在固定规模币池」若要在历史验证中可操作，则用**每期事前 168h 成交额 TopN** 逐期重建的池应满足：(a) 绝大多数重建时点能凑够 N 个合格币；(b) 相邻期成员不会整批换血；(c) N≥30 时每期有效横截面宽度足以支撑 `MIN_CROSS_SECTION=30`。

事前终止条件（**预注册，任一不成立即判该 N 不可操作，不许换窗口／换阈值／换分位数去救**）：

| 编号 | 条件 | 门槛 |
|---|---|---|
| C1 | 可构造性 | `eligible_count ≥ N` 的重建时点占比 ≥ **90%** |
| C2 | 成员稳定性 | 相邻 7 天 TopN 成员 **Jaccard 中位数 ≥ 0.60** |
| C3 | 横截面宽度 | 仅对 N≥30：`width ≥ 30` 的期数占比 ≥ **90%** |

额外诊断（**不参与 gate**）：当前池成员在历史各期的成交额排名分位与「当期不合格」比例；`eligible_count` 时间趋势；各 N 的平均换手率与宽度分布；N 对上述指标的影响表。

## 4. 预注册规格（不得事后挑选）

| 项 | 值 |
|---|---|
| 唯一口径 | `klines(1h)` 的 `quote_volume` 与 bar 存在性；**不读 close** |
| 重建时点 T | 从 `max(min(open_time)+168h, 对齐 UTC 00:00)` 起，每 **7 天**一期，直到 `T+24h ≤ max(open_time)` |
| 事前窗口 | `[T−168h, T)`（与线上 `pool_maintenance.WINDOW_HOURS` 一致） |
| 单币合格条件 | 窗口内 ≥ **160 根**有效 1h bar；末根不早于 `T−2h`；`sum(quote_volume) > 0`；后缀过滤通过 |
| 排序 | 窗口成交额降序，并列按 symbol 字母序（确定性） |
| 扫描规模 | N ∈ **{10, 20, 30, 50, 100}**，全部同口径一次性算出 |
| 有效横截面宽度 | PIT TopN 成员中在 `[T, T+24h)` 具备**全部 24 根** 1h bar 的币数 |
| 事后选币代理 | 「当前池」= 最后一期 TopN；对每个历史 T，算这些成员在当期成交额排名分位，以及不在当期 eligible 集合的比例 |
| 成本 | `fee_rate=0.0004`／`slippage=0.0002`／双边 0.12%、名义本金口径**登记但不适用**（trades=0、无 PnL） |
| 随机基线／CI | **不适用**（无策略评价、无收益估计），报告显式声明而非填零 |
| 并行 | 单次 SQL + pandas 向量化聚合，单进程；非逐 bar simulate，H9 进程池不适用 |

## 5. 结论边界

- 判定为**工程可行性**（`operational_pass` / `operational_fail`），不是手册第 3 节的 edge verdict；`verdict_scope` 必须写明「数据口径验证，非 edge 判定」。
- 输出的是「哪些 N 可构造、成员有多稳、宽度够不够、当前成员池有多少事后选币成分」，**不输出任何策略收益、不给出启用建议**。
- `default_enabled=False`、`trades=0`、无绑定、无模拟盘、无订单。若 C1/C2/C3 全部通过，后续才可把「固定规模 PIT 池」写进 `research/config.yaml` 作为后续课题的约束；若任一不通过，须先补历史成员资格或显式降低 N 并标注数据边界，**不得直接拿当前成员池回放历史**。

## 6. 复现

```powershell
.\.venv\Scripts\python.exe backend/scripts/probe_p16_fixed_pool.py --out docs/probe-p16-fixed-pool.json
.\.venv\Scripts\python.exe -m pytest backend/tests/test_probe_p16_fixed_pool.py -q
.\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/probe-p16-fixed-pool
```

# P12 扩窗复核：采集链修复后 36.42 天横截面 IC 探针（Gate 1 增量）

> 生成时间：2026-11-11 22:40 ｜ 脚本：`backend/scripts/probe_p12_oi_cross_section.py`、`sync_sentiment.py`、`sync_recent_1h.py` ｜ 命令见文末
> 性质：**Gate 1 补样本复核，非 edge 判定**。承接 `docs/probe-p12-oi-cross-section.md`（21.83 天首版）。本轮先修复断裂的采集链，再以**完全相同的主判据**重跑，只扩样本、不换因子、不调阈值、不改周期。
> 版本：增量 v2 ｜ 课题编号 P12

## 1. 假设卡与事前承诺

- **信息源**：币安 `/futures/data/*` 衍生品情绪，属 H2 允许的「衍生品持仓与情绪」，非现货 OHLCV 变换。
- **可证伪假设（与首版逐字相同）**：若 P9 的 OI 负 IC 是可交易选币 alpha，则 `oi_value_z7d` × H=24 × 横截面 demean 标签的 rank IC 应满足 \|IC\|>0.02、95% CI 不跨 0、同号期≥60%、逐币同号币≥60%、置换 p<0.05。五关全过才授权实现；任一不过即终止，不以扩窗、换 H 或挑探索性组合来救它。
- **补样本的性质**：H1 未过时的正确动作是补样本（H4 词表：`inconclusive` 只能补样本）。本轮**唯一改动的变量是数据窗口**，因子构造、标签、成本、币池、随机种子、block bootstrap 参数全部冻结，属于受控补样本而非重新选参。
- **周期 / 持有期 H**：1h，主 H=24；探索性 H=8/24/48 不参与 gate。
- **窗口 / 墙钟对齐**：2026-10-06 04:00 ~ 2026-11-11 14:00 UTC，**36.42 天**、875 个整点。样本期 BTCUSDT **−13.800%**（下跌市）。
- **币池 / 币数**：`sentiment` **515 币** ∩ native 1h 535 币，对齐到 515 币；逐时点有效币数中位 514；**100 期**因不足 30 币跳过，主判据实际 **775 期**。
- **成本口径**：`fee_rate=0.0004`、`slippage=0.0002`、双边 **0.12%**，口径＝**名义本金**（不切换、不与 margin 口径混比）。本轮 IC 为毛价格关联，**未扣成本**。
- **止损线核对**：已查 `references/verdicts-history.md`；未新增现货 OHLCV 特征／参数／标签，未触碰 P4「换频率无效」与 §4「不调止损倍数」。
- **对照组**：正对照（偷看未来）、200 次逐期置换负对照、raw vs demean 标签对照、逐币时序 vs 逐期横截面对照、以及**首版 21.83 天同判据基线**。

## 2. 先修数据链：从 21.83 天到 36.42 天

首版报告 §2 实测「sentiment 停 10-28、native 1h 停 11-04，`sync_sentiment_many` 无调用者，每日自动化所调脚本已删除」。本轮按该结论修复：

| 动作 | 产物 / 命令 | 结果 |
|---|---|---|
| 新建可调度 CLI | `backend/scripts/sync_sentiment.py`（123 行） | 包一层 `MarketDataService.sync_sentiment_many`，幂等 upsert，支持 `--symbols/--workers/--limit/--dry-run/--request-interval/--out` |
| 5 币小样本验证 | `--limit 5` | 写入 2,505 行、0 错误，末尾推进到 11-11 14:00 |
| 全量补采 sentiment | `--workers 12` | **515 币 / 257,844 行 / 0 错误 / 419.73 秒**；行数 269,403 → **449,313**，跨度 21.83 → **36.42 天** |
| 补 native 1h 尾部 | 既有 `sync_recent_1h.py`（本轮仅新增 `--out` 参数，避免覆盖旧报告） | **535 币全部 ok / 90,899 行 / 201 秒**；末尾 11-04 15:00 → **11-11 13:00** |

`sync_recent_1h.py` 的改动只有两处：新增 `--out` 参数、把写报告路径改为该参数，默认仍指向原路径，向后兼容。它是独立 CLI，不属于后端服务代码，**不需要重启 `QuantDesk-GetMoney`**。

**数据质量核验**：875 个 distinct 小时、日均 12,336 行（515×24）、11-11 当日 7,710 行（15 个完整小时）；`oi_value=0` 的缺失补位共 1,553 行（0.35%）**全部在旧窗口**，新窗口缺失比例 **0.0000**；抽查 BTCUSDT 10-27 的旧值仍在且合理，幂等 upsert 未破坏历史。

**修复后的实时尾部**：sentiment 11-11 14:00、native 1h 11-11 13:00、1m 11-11 14:32；**仍停更**：funding 10-27 12:00、15m 10-25 12:30、4h 10-26 08:00、5m 10-25 11:20。P9 A/B 需要 60 天，现有 36.42 天，仍差约 24 天；但最危险的「数据窗口往后关闭」已经解除。

## 3. 结果：扩窗后主判据更弱，gate 仍 FAIL

| 指标 | 首版 21.83 天 | **本轮 36.42 天** | H1 门槛 | 本轮判定 |
|---|---:|---:|---|---|
| 有效时点 | 427 | **775** | — | 样本近翻倍 |
| 逐期横截面 IC 均值 | −0.01597 | **−0.01329** | \|IC\|>0.02 | **未过**（幅度衰减 17%） |
| 95% 置信区间（block24） | [−0.03972, +0.00971] | **[−0.03092, +0.00418]** | 不跨 0 | **未过（仍跨 0）** |
| 同号期占比 | 60.42% | **56.39%** | ≥60% | **跌破门槛** |
| 逐币时序 IC / 同号币 | −0.08173 / 65.76% | **−0.05799 / 63.88%** | ≥60% | 方向一致但减弱 |
| 置换经验 p（200 次） | 0.000 | **0.000**（零分布均值 +0.00008） | <0.05 | 过 |
| 正对照 IC | 1.0 | **1.0** | 量具正常 | 过 |
| 前半窗 / 后半窗 IC | −0.01787 / −0.01408 | **−0.01571 / −0.01089** | 同号 | 方向一致、后半更弱 |
| raw vs demean 标签 IC | 完全相同 | **完全相同**（−0.01329） | — | rank IC 恒等变换再确认 |
| **gate 合计** | 3 过 2 不过 | **2 过 3 不过** | 五关全过 | **FAIL** |

**机制归因进一步强化**：全市场平均 OI z 与全市场未来 24h 收益的时序 Spearman 从首版 **−0.23249** 变为 **−0.34803**（本轮 36.42 天）。共同时间成分更强，而券维度排序更弱 —— 与「P9 的负 IC 主要是全市场杠杆周期择时、不是跨币选币 alpha」的解释一致。注意该相关仍是两条平滑且标签窗口重叠的时序序列之间的读数，**未做非重叠样本、样本外、成本与随机基线检验，不能当作可交易择时 edge**。

**探索性组合（H7）**：17 个组合中 6 个 CI 不跨 0（多为 H=8：`oi_value_z7d` −0.02204、`oi_change_24h` −0.02172、`top_long_short` +0.01676、`taker_buy_sell` −0.00787 等），15 个 q<0.10。两个理由使它们**不得用于判定**：① 置换零分布假设各期独立，零分布极窄，经验 p 与保留自相关的 block bootstrap CI 冲突时以后者为准；② 主判据 H=24 为 −0.01329、H=48 方向上也不稳定，挑 H=8 属事后挑选。按 H2/H7 不再调 H 来救它。

## 4. 判定

- **verdict**：**`inconclusive`**（探针级）。95% CI 跨 0，按词表只能写「不知道」，不等于已证无效、也不等于小幅为正。
- **gate**：H1 **FAIL**，且从首版的 3/5 退化到 **2/5**。补样本后信号**没有变强，反而变弱、一致性跌破门槛**，这是不支持继续投入该路线作为选币特征的证据。
- **池化 CI / 随机基线**：本轮 `trades=0`，无撮合、无每笔期望 CI、无多种子随机方向基线（随机基线在本轮不适用；负对照为逐期置换零分布），不构成 H4 的净收益 edge 判定，也不满足 `MIN_POOLED_TRADES=30`。
- **与历史结论对照**：与首版方向一致，与 P9 时序读数方向一致但不复现其强度；与 P6、P4 无冲突。**P9 roadmap 的「特征 ON/OFF 受控 A/B」仍不应按原假设执行**，若要继续，须先重新登记为「市场级杠杆择时」课题并补齐非重叠样本／样本外／成本／随机基线设计；且 60 天覆盖目前只有 36.42 天。
- **是否启用**：**不启用**，`default_enabled=False`，不进绑定。本轮未改任何 spec／绑定／生产参数／前端，未部署模拟盘（status `BLOCKED`），未下任何真实订单。

## 5. 局限

- 36.42 天仍不足 P9 要求的 60 天；**单一下跌市**（BTC −13.800%），跨市场状态稳健性无法验证。
- `oi_value_z7d` 用 168 根 rolling、`min_periods=72` 近似 7 天 z-score，预热期口径与 P9 完整 7 天有偏离；100 期因此被剔除。
- 标签为固定 H 的 close-to-close 收益，不是 P9 的 triple-barrier 止盈标签；逐币 IC 数值不得与 P9 直接相减。
- IC 是毛价格关联，未扣 0.12% 双边成本、资金费率、盘口深度、滑点、强平、部分成交；名义本金口径，不得与 margin 口径报告相减。
- 置换零分布按各期独立打乱，未保留跨期自相关；FDR 在该零分布下不可靠。
- 横截面 IC 只测券维度排序，不能排除 OI 在时间维度存在择时价值；−0.34803 未做非重叠／样本外检验。
- funding、15m、4h、5m 尾部仍停更，本轮只修复了 P9/P12 直接依赖的 sentiment 与 native 1h。
- 探索性组合未参与 gate，不得事后挑选；未模拟撮合，无成交笔数门槛与池化 CI。

## 6. 结论与后续

**一句话结论**：修复采集链、把样本从 21.83 天扩到 36.42 天后，P12 主判据横截面 IC 从 −0.01597 衰减到 −0.01329、95% CI 仍跨 0、同号期占比跌破 60%，H1 两次均未过 —— **OI 信号作为跨币选币特征缺乏统计支持，不应再靠换 H 或扩窗去救**；共同杠杆周期的解释反而被更强的 −0.34803 佐证。

三条符合 H2 的后续假设及当前状态（与首版相同，但数据前置已部分缓解）：

| 假设 | 状态 | 变化 |
|---|---|---|
| **H-A 市场级杠杆择时**（非重叠样本 + 样本外 + 成本 + 随机基线） | `BLOCKED_BY_DATA` | 从 21.83 → **36.42 天**，采集链已恢复，仍需累积到 ≥60 天 |
| **H-B 逐笔成交顺序**（aggTrades 日归档可回填，不得退回 K 线级 taker 汇总） | `BLOCKED_BY_NEW_SOURCE` | 无变化 |
| **H-C 盘口挂撤单 / 多所现货腿** | `BLOCKED_BY_NEW_SOURCE` | 无变化 |

**剩余待办**：恢复每日增量自动化（sentiment + native 1h），并决定是否补 funding／15m／4h 尾部。本轮未接入任何新交易所、未启用 live API key、未创建真实订单。

## 7. 复现命令

```powershell
# 采集链修复
.\.venv\Scripts\python.exe backend/scripts/sync_sentiment.py --dry-run
.\.venv\Scripts\python.exe backend/scripts/sync_sentiment.py --workers 12 --out output/sync_sentiment_full_20261111.json
.\.venv\Scripts\python.exe backend/scripts/sync_recent_1h.py --workers 12 --out docs/sync-recent-1h-report-20261111.json

# 扩窗复核（与首版同一脚本、同一主判据，仅数据窗口变化）
.\.venv\Scripts\python.exe backend/scripts/probe_p12_oi_cross_section.py --out docs/probe-p12-oi-cross-section-extended.json --force
.\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/probe-p12-oi-cross-section-extended
.\.venv\Scripts\python.exe -m pytest backend/tests -q
```

探针只读 DB（`make_engine(read_only=True)`）；两个采集脚本用可写 `make_engine(read_only=False)` 并做幂等 upsert，重跑安全。

## 8. 产物与自检

- `docs/probe-p12-oi-cross-section-extended.md` + `.json`（成对）
- `output/sync_sentiment_full_20261111.json`（515 币采集报告）、`output/sync_sentiment_smoke.json`（5 币验证）
- `docs/sync-recent-1h-report-20261111.json`（535 币 1h 补尾报告，未覆盖 11-05 旧报告）
- `backend/scripts/sync_sentiment.py`（123 行，新增）、`backend/scripts/sync_recent_1h.py`（新增 `--out`，2 处改动）
- `backend/scripts/probe_p12_oi_cross_section.py`（仅把过时的运维备注改成中性实时核表说明，计算逻辑未变）

**交付前自检实测**：新增单测 `backend/tests/test_probe_p12_common.py` **5 passed**；两份报告预检均为 **FAIL 0 / WARN 0 / PASS 21**（首版在加了更新注记后重新预检通过）；后端全量 pytest **3 failed / 1836 passed**，三个失败是首版之前就存在、与本轮改动无关的既有失败（`test_market_performance.py::test_complete_tail_queries_have_a_bounded_minute_range`、`test_pool_volatility_research.py::test_live_entrypoints_reject` ×2，报 `app.trading.dispatch` 缺 `halt_status`）。本轮未修这些既有失败（涉及用户未提交的生产代码改动）。未改后端服务代码、未改前端 → 无需重启 `QuantDesk-GetMoney`、无需 `npm run build`。当日工作记忆与长期记忆已登记。

## 9. 附：与首版报告的状态差异

首版 `docs/probe-p12-oi-cross-section.md` §2 的「数据链断裂」表格是 **2026-11-11 22:10 的历史快照**，其中 sentiment 10-28、native 1h 11-04 的读数在本轮已被修复；funding / 15m / 4h / 5m 的停更读数仍然成立。请以本报告的 §2 实时核表结果为准，不要把首版表格当作当前状态。

---

> **免责声明**：以上内容基于公开数据和量化分析，仅供参考，不构成投资建议。市场有风险，投资需谨慎。任何投资决策应结合个人风险承受能力、资金状况和投资目标独立判断，必要时咨询持牌专业机构。过往表现不预示未来收益。

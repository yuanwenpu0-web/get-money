# P15 采集链与台账完整性审计 · 运维审计，非 edge 判定

> 生成时间：2026-11-11 16:13 UTC ｜ 方法：只读 PostgreSQL 查询 + `qoder_cron list` + `pytest`（本轮未新增脚本、未下载任何行情）
> 代码基线 `7ebb5f5` ｜ 台账 `P15-000`（并补登 `P14-001`）｜ 前序 `docs/probe-p14-trade-sequence.md`、`docs/data-pipeline-repair-20261111.md`
> **性质**：Stage 0 状态恢复 + Stage 3 数据审计 + Stage 8 复盘。**不是回测、不是净收益 edge 判定**，`trades=0`，不改变 P12/P14 既有结论。
> **一句话**：P14 证据链复核通过（13 单测全过、`DO_NOT_SCALE_UP`），但审计时发现**每日采集调度处于禁用且从未运行**，H-A 赖以到期的「每日累积」实际没有在执行；台账一度缺 `P14-001`，落账时发现另一并发会话已登记，本会话撤回重复补登。**采集已于 16:20Z 经用户授权恢复，各表尾部均推进到最近完整 UTC bar（§9）**。

## 1. 窗口 / 样本期与币池（只读计数，非本轮选币）

审计读数时刻 **2026-11-11T16:13Z**，墙钟对齐 UTC。

| 数据面 | 起点 | 末端 | 滞后 | 币数 |
|---|---|---|---|---|
| `sentiment` | 2026-10-06T04:00Z | **2026-11-11T14:00Z** | 2.22h | 515（近几期每期 514） |
| `klines` 1m | 2025-09-20T15:00Z | 2026-11-11T16:10Z | 0.05h | 535 |
| `klines` 5m | — | 2026-11-11T15:05Z | 1.13h | **2** |
| `klines` 15m | — | 2026-11-11T14:45Z | 1.47h | 521 |
| `klines` 1h（native） | — | **2026-11-11T13:00Z** | 3.22h | 535 |
| `klines` 4h | — | **2026-11-11T08:00Z** | 8.22h | 521 |
| `funding_rates` | — | 2026-11-11T15:00Z | — | — |

`sentiment` 覆盖跨度 **36.42 天**、449,313 行。skip 清单：本轮无下载、无剔除，故为空。

## 2. 成本口径（登记以保持可比，本轮未参与任何计算）

`fee_rate=0.0004`、`slippage=0.0002`，双边合计 **0.12%**，**名义本金**口径，与 P12/P14 全程一致、不切换。`applied=false`：本轮无任何收益计算，**不得与 margin 口径或每笔收益口径报告混比**。

## 3. 发现

### F1 · CRITICAL：每日采集调度禁用且从未运行

`qoder_cron list` 实测（唯一一条任务）：

| 字段 | 值 |
|---|---|
| id | `eea4dd3e-f81b-431e-9a90-e22a795c333a` |
| 名称 | 每日行情增量采集（sentiment + native 1h + 15m/4h/5m + funding） |
| 调度 | `30 8 * * *`（Asia/Shanghai）= **00:30 UTC** |
| `enabled` | **false** |
| `pauseReason` | `manual` |
| `nextRunAt` / `lastRunAt` / `lastRunStatus` | **null / null / null** |
| `consecutiveFailures` / `revision` | 0 / 1 |
| `createdAt` | 2026-11-11T15:41:31Z |

调用者核查：`sync_sentiment_many` 定义在 `backend/app/market/data_service.py:359`，**全仓库唯一调用者**是 `backend/scripts/sync_sentiment.py:86`，而该 CLI 只被上面这条已禁用调度调用。后端 APScheduler 的 `trading_cycle` 确实在跑（`job_logs` 最新 2026-11-11T15:58:35Z、509 币、约每 5 分钟一轮），但它**只同步 1m K线**，不含 sentiment / funding / native 周期同步——这正好解释「1m 新鲜、其余全部滞后」。

**影响**：`sentiment` 停在今日手工全量采集的水位 14:00Z。H-A 报告里写的「靠每日 08:30 调度累积到 60 天」**当前并不成立**。

### F2 · HIGH：台账缺 `P14-001`，而 P14 报告声称已登记

- 审计前台账只有 3 行：`P12-001`、`P12-002`、`P13-000`；`grep -c P14 research/experiments.jsonl` = **0**。
- 但 `docs/probe-p14-trade-sequence.md` §9 产物清单明确列出 `research/experiments.jsonl`（`P14-001`）。
- **影响**：机器台账与人类报告不一致。后续轮次若只读台账，会误判 P14 未跑，可能重复投入 **983 MB / 7,424 万行**归档下载。
- **本轮动作与收口**：本会话先按 `docs/probe-p14-trade-sequence.json` 的实际读数补登 `P14-001`（标 `record_kind=backfill`）；**追加时发现另一并发会话已登记 `P14-001`**（timestamp 16:06:00Z），遂**撤回自建行**，保留对方记录原样不动。最终台账 5 行、ID 唯一。
- **残留缺口**：对方那条 `P14-001` 缺 `fee_model`/`slippage_model`/`funding_model` 与 `train/validation/test_period` 字段；成本口径见成对报告，本会话**未代改他人记录**。

### F3 · MEDIUM：周期覆盖不均

只有 1m 由 `trading_cycle` 保持新鲜；**5m 仅 2 个币**（15m/4h 为 521，1h/1m 为 535），4h 滞后 8.22h、native 1h 滞后 3.22h。任何以 5m/4h 为周期的课题当前 `BLOCKED_BY_DATA`。5m 成因本轮未深查，只登记现象。

### F4 · INFO：P14 证据链复核通过

| 项 | 实测 |
|---|---|
| 单测 | `backend/tests/test_probe_p14_common.py` → **13 passed in 0.84s** |
| `verdict` / `gate_passed` | `insufficient` / `true` |
| `decision.action` / 红旗数 | **`DO_NOT_SCALE_UP`** / **6/6** |
| 主判据 | `seq_flip_dev` × H=24，mean IC **−0.06608**，312 期 |
| 置信区间（block24，2000 次） | **[−0.13233, −0.00556]**，不跨 0 |
| 缓存 | `output/probe_p14_frames.pkl`，612,965 字节，可秒级重放 |

md/json 成对一致，均为修复 `numpy.bool_` 序列化后的最终产物。**本轮不重跑探针、不重新下载归档。**

### F5 · HIGH：H-A 仍 `BLOCKED_BY_DATA`，到期日取决于采集是否恢复

- 需求：`sentiment` ≥ **60 天**且含 ≥ **20 天**样本外；当前 **36.42 天**。
- 60 天mark：**2026-12-05T04:00Z**，距审计时刻 **23.49 天**。
- 保留期口径：项目既有记录（`docs/data-pipeline-repair-20261111.md` 与调度 prompt）称 `/futures/data/*` 约 **30 天滚动保留**；**本轮未向交易所复核该保留期**。
- **影响**：若采集持续禁用，缺失日会在约 30 天后超出保留期而**永久不可回补**，60 天连续窗口将无法达成。恢复采集的最晚安全时点约在「首个缺失日 + 30 天」。

### F6 · MEDIUM：并发写同一台账导致重复登记

审计时台账 3 行、`grep -c P14` = 0；执行追加时已变 4 行——另一会话在本轮审计期间登记了 `P14-001`。本会话的 backfill 因此成为重复，已撤回自建行（`duplicate_backfill_removed=1`），最终 5 行、`experiment_id` 唯一。

**建议**：`research/experiments.jsonl` 是多会话共享写入点，无锁追加会产生重复 ID。追加前应重读末尾并校验 ID 唯一；长期并发应由单一写入者（主控角色）串行落账。

## 4. 置信区间与随机基线

- **置信区间**：不适用。本轮无收益估计，故无 CI；P14 的 block24 区间见其自身报告，本报告不重新解释。
- **随机基线**：不适用。本轮无策略臂、无交易、无仓位，未做多种子随机方向基线，因此没有「随机基线分位」类读数。

## 5. 判定

- **verdict**：`no_trades`（词表内取值）。`verdict_scope`：运维与台账审计，`trades=0`，**不做任何 edge 判定**，不加强也不推翻 P12/P14。
- **alpha 研究结论**：**`NO_IMPROVEMENT`**。本轮未检验任何新假设——P14 已判 `DO_NOT_SCALE_UP` 并登记止损线延伸，H-A 数据不足，H2 止损线仍有效，故不投新算力。这是有效结论，不是失败。
- **唯一可执行改进（运维，非 alpha）**：`RESUME_COLLECTION_REQUIRED` —— 恢复既有调度或手工跑三个 sync 脚本。**需用户显式授权**；本轮未启用调度、未运行任何采集。
- **是否启用**：`default_enabled=False`；无 `StrategySpec` / `strategy_bindings` / 参数 / 前端改动，`orders_submitted=0`，未启用 live key。
- **Stage 7 模拟盘**：`BLOCKED_BY_POLICY`（用户已把执行边界定为只到回测+风控，不接任何执行适配器）。

## 6. 局限与未模拟项

- 只读审计：未下载行情、未跑回测、**未模拟**撮合、资金费、盘口深度、强平、分批建仓；`trades=0`。
- API 约 30 天保留期沿用项目既有口径，未在本轮向交易所复核。
- 调度状态是 `qoder_cron list` 的当时快照；`enabled=false` 的成因（人为暂停或创建即禁用）无法由该接口判定。
- 5m 仅 2 币的成因未深查。
- 成本为名义本金口径，不得与 margin 口径或每笔收益口径混比。

## 7. 后续与复现

**下一步（唯一）**：取得授权 → 恢复每日采集 → 核对 `sentiment.max(ts)` 是否推进到最近完整 UTC bar。在 `sentiment` 达 60 天（最早 2026-12-05T04:00Z）前**不启动 H-A 判定、不重跑 P12/P14**。

```powershell
# 本轮实际执行的只读核验
.\.venv\Scripts\python.exe -m pytest backend/tests/test_probe_p14_common.py -q
.\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/audit-p15-collection-and-ledger
# 授权后才执行（本轮未运行）
.\.venv\Scripts\python.exe backend/scripts/sync_sentiment.py --workers 12 --out output/sync_sentiment_daily.json
```

## 8. 产物

- `docs/audit-p15-collection-and-ledger.md` + `.json`（成对）
- `research/experiments.jsonl`：追加 `P15-000`（本审计，含 F6 并发冲突）与 `P15-001`（授权后恢复采集）；`P14-001` 由并发会话登记，本会话重复 backfill 已撤回
- `output/sync_sentiment_daily.json`、`docs/sync-recent-1h-report-daily.json`、`output/sync_market_tail_daily.json`
- `.workbuddy/memory/2026-11-11.md`（append，未覆盖并发会话内容）
- 未新增脚本、未改动生产代码、未改动 `research/config.yaml`

## 9. 授权后执行与复核（16:20:16Z → 16:37:28Z，wall 1032s）

用户经选项「启用调度 + 立即补跑一次」显式授权后执行。**F1 已解除**：调度 `enabled=true`、`pauseReason=null`、`nextRunAt=2026-11-12T00:30Z`（08:30 Asia/Shanghai）。

| 数据面 | 执行前 | 执行后 | 是否到最近完整 UTC bar |
|---|---|---|---|
| `sentiment` | 14:00Z / 36.42d / 449,313 行 | **16:00Z / 36.50d / 450,325 行** | 是（滞后 0.63h） |
| native 1h | 13:00Z | **15:00Z** | 是 |
| 15m | 14:45Z | **16:15Z** | 是 |
| 4h | 08:00Z | **12:00Z** | 是（12:00–16:00 桶尚未收满） |
| 5m | 15:05Z | **16:25Z** | 是，但**仍仅 2 币** |
| 1m | 16:10Z | 16:31Z | 是（由 `trading_cycle` 维持） |
| `funding_rates` | 15:00Z | **16:00Z** | 是 |

`sentiment` 净增 **1,012 行**；报告里的 `inserted=253,836` 是**幂等重写计数**，不等于新增，勿误读。native 1h **534 ok / 1 error**、写入 2,670 行；`sync_market_tail` 15m 块写入 49,315 行。

**单币 302 共 12 条**（全部为「币安合约接口返回重定向 302，请检查代理或网络」）：

- sentiment 8：`ALT` `ARC` **`AVAX`** `AVAAI` `BAT` `BICO` `BILL` `BROCCOLI714`
- native 1h 1：`BR`
- 15m 2：**`AVAX`** `XVS`；4h 1：`APE`

**`AVAXUSDT` 在 sentiment 与 15m 重复出现且属主流币**，优先跟踪是代理问题还是合约状态问题；其余多为小市值/疑似下架币，按调度约定列出并连续跟踪。

**残留（未解决）**：

1. **5m 仍仅 2 币** —— `sync_market_tail` 按各周期既有 distinct symbol 取池，**设计上不扩池**；5m 课题仍 `BLOCKED_BY_DATA`，扩池需另行授权。
2. 12 条单币 302 未解决。
3. **H-A 仍 `BLOCKED_BY_DATA`**：36.50d < 60d，60 天 mark **2026-12-05T04:00Z**。采集恢复后到期日重新可达，但仍需每日推进且不出现永久缺日。

**边界不变**：alpha 结论 `NO_IMPROVEMENT`，`trades=0`、0 订单、0 绑定/spec/参数/前端改动、未启用 live key、Stage 7 `BLOCKED_BY_POLICY`；未改生产代码，无需重启或 build。
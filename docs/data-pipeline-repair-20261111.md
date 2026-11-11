# 行情采集链修复验收记录 · 2026-11-11

> **性质**：运维验收记录，不是 Gate 1–6 策略研究报告。本轮未提出或验证任何新因子，
> 未跑 IC 探针 / 回测 / 参数寻优，未改策略 spec、绑定、参数或前端，
> `default_enabled=False`、未部署模拟盘、未下任何订单。
> 策略结论仍以 `docs/probe-p12-oi-cross-section.md` 与
> `docs/probe-p12-oi-cross-section-extended.md` 为准。

## 1. 背景与范围

上一轮已修复 `sentiment`（每日自动化所调脚本被删、`sync_sentiment_many` 无调用者）
与 `native 1h` 尾部，并用 36.42 天窗口复核 P12（结论仍为 `inconclusive`）。
本轮处理**剩余停更周期**：实时核表确认 15m/5m 停在 2026-10-25、4h 停在 10-26、
funding 停在 10-27。代码层缺口：

- `backfill_native_15m.py` / `backfill_native_1h.py` 只往**历史起点之前**回填，不补尾部；
- `sync_recent_1h.py` 硬编码 `interval='1h'`；
- `MarketDataService.sync_funding_many` 只有定义，唯一调用者是 `POST /market/sync`，无 CLI；
- 4h 无 HTTP 采集脚本（`derive_candles.py` 是库内派生，不是采集）；
- 5m 无采集链，全库仅 BTCUSDT / ETHUSDT 2 币。

## 2. 做法

新增 `backend/scripts/sync_market_tail.py`（297 行）：把 `/fapi/v1/klines` 与
`/fapi/v1/fundingRate` 包成可调度 CLI。关键约束：

- **目标币池 = 各周期已有 `distinct symbol`**，不扩池、不改数据口径。
  因此 15m/4h 覆盖 521 币，5m 仍只维持既有 2 币，funding 507 币。
- 每币从 `max(open_time) - 2 根` 补到最后一个完整 UTC bar，留 2 根重叠保证衔接；
  缺口之前的历史**不在请求区间内**，因此不会被 upsert 触碰。
- 全部 `on_conflict_do_update` 幂等写入，重跑安全；不删除任何旧行。
- funding 窗口按「最落后币的水位」自动计算（`min(per-symbol max)`），上限 30 天。
- 资金费用 `sync_funding_many` 复用既有幂等实现，K 线用通用 upsert 绕过
  `store()` 的「仅 1m」限制，字段集与既有 native 行一致。

## 3. 两次运行

| 运行 | 报告 | 时长 | 错误 | 备注 |
|---|---|---:|---:|---|
| 首跑 | `output/sync_market_tail_full_20261111.json` | 312.76s | **66** | 15m 4 币代理 302；4h 62 币触发币安 `-1003 Too many requests`（2400 权重/分钟）；funding 窗口被全局 max 误判为 3 天，留下中间断层风险 |
| 修复后重跑 | `output/sync_market_tail_full_20261111_retry.json` | 162.78s | **0** | funding 窗口自动恢复为 18 天，补回断层 |

修复项（均在 `sync_market_tail.py` 内，未动生产代码）：

1. 默认公开请求间隔 `0.1s → 0.25s`，避开 IP 权重限流；
2. 各周期阶段之间加 3 秒冷却；
3. funding 水位由 `max(funding_time)` 全局值改为 `min(每币 max(funding_time))`，
   避免各币水位不齐时漏补；上限 30 天防止下架币把窗口无限撑大。

脚本幂等，重跑时已补齐的币返回 `fresh`，只重试上次失败的币：
15m `517 fresh + 4 ok`、4h `459 fresh + 62 ok`、5m `2 ok`、funding `507 请求 / 0 错误`。
首跑那 66 个失败币在重跑中全部补回。

## 4. 修复前后覆盖

| 数据源 | 修复前末尾 | 修复后末尾 | 行数变化 | 币数 |
|---|---|---|---:|---:|
| klines 15m | 2026-10-25 12:30 | **2026-11-11 14:45** | 1,885,554 → 2,740,515 (+854,961) | 521 |
| klines 4h | 2026-10-26 08:00 | **2026-11-11 08:00** | 1,146,043 → 1,198,812 (+52,769) | 521 |
| klines 5m | 2026-10-25 11:20 | **2026-11-11 15:05** | 34,560 → 44,442 (+9,882) | 2 |
| funding_rates | 2026-10-27 12:00 | **2026-11-11 15:00** | 1,041,678 → 1,083,786 (+42,108) | 507 |
| klines 1h | 2026-11-11 13:00 | 未变 | 4,930,725 | 535 |
| sentiment | 2026-11-11 14:00 | 未变 | 449,313（36.42 天） | 515 |

1m 随交易巡检自然增长到 15:17（48,214,850 行），不属于本轮补采。

## 5. 验收结果

- **尾部齐平**：15m 521 币、4h 521 币、5m 2 币的 `max(open_time)` 全部等于最后完整 bar，
  落后超过 1 根的币数 = 0。
- **零断层**：自 2026-10-24 起，相邻 bar 间隔大于周期步长的记录数
  15m / 4h / 5m / 1h 均为 **0**（最大间隔分别 0.25h / 4h / 0.08h / 1h）。
- **funding 断层已补**：原 10-28 → 11-07 空洞区间，507 币中仅
  `VELODROMEUSDT` 少于 20 条。该币全历史停在 2026-10-28 08:00（2,410 行），
  交易所已无新资金费记录，属**数据源停更**，不是脚本漏采；其余 506 币连续。
- **旧值未被破坏**：补采 fetch 起点为每币 `max(open_time) - 2×step`，
  缺口之前的历史 bar 完全不在请求区间内；抽样 BTCUSDT 15m 2026-10-20 的
  close/volume 仍在库且格式正常。
- **P12 判据未变**：本轮未改 `sentiment` 与 `native 1h` 数据，
  P12-001（21.83 天）与 P12-002（36.42 天）的主判据与结论保持原样，
  没有重跑探针、没有换 H / 换窗口去救 `inconclusive`。

## 6. 调度与课题登记

- 每日自动化 `62854c1d-aec5-4551-93a8-27f9b5ec5fca`（08:30、ACTIVE）用 `mode=update`
  扩展为三脚本：`sync_sentiment.py`、`sync_recent_1h.py`、`sync_market_tail.py`，
  并要求核表全部周期、写当日记忆。未新建重复自动化。
- `research/experiments.jsonl` 补登记 **P12-002**（上一轮扩窗复核，此前遗漏）
  与 **P13-000**（H-A 假设卡预登记）。
- `research/hypotheses/H-A-market-leverage-timing.md` 按 Gate 0 字段预登记
  **市场级杠杆周期择时**课题，状态 `BLOCKED_BY_DATA`：
  P9 原「攒满 60 天做逐币特征 ON/OFF 的 A/B」正式改写为**市场级仓位开关**，
  要求非重叠 24h 块、≥60 天且 ≥20 天样本外、恒定满仓/随机开关 p90/ON-OFF 对照、
  净收益扣 0.12% 往返成本。sentiment 现 36.42 天，**未启动任何探针或回测**；
  自动化不得自行跑回测或启用策略。

## 7. 口径、显著性与局限声明

本记录是运维验收，**没有成交、没有收益估计**，因此不产生每笔期望的 95% 置信区间，
也不适用多种子随机基线 / p90 分位；这些判据只属于 P12 / P13 的策略研究报告
（`docs/probe-p12-oi-cross-section*.md` 与 `research/hypotheses/H-A-market-leverage-timing.md`）。
成本口径沿用项目统一设定：**名义本金（notional）**，双边 0.12%
（fee 0.04% + slippage 0.08%）；本轮没有可扣成本的成交，故未应用。
以上为「不适用」的显式声明，不是用零值冒充实测。

## 8. 局限与未决项

- 脚本 `py_compile` 通过，行数 297（项目上限 300）；`--dry-run` 与 5 币小样本均通过。
- 成对报告预检 `preflight.py --prefix docs/data-pipeline-repair-20261111`：
  **FAIL 0 / WARN 0 / PASS 21**；CI / 成本 / 随机基线均按「本轮不适用」显式声明。
- 全量 `pytest backend/tests`：**3 failed / 1836 passed / 75.33s**，
  三个失败与前两轮完全相同且为既有问题（`test_market_performance.py` 的 K 线查询下界、
  `test_pool_volatility_research.py` 两项缺 `dispatch.halt_status`），
  与本轮新增脚本无关；未擅自修改用户未提交的生产代码。
- 未改后端服务 / 前端，**无需重启计划任务 `QuantDesk-GetMoney`、无需 `npm run build`**。
- 已知冗余：`VELODROMEUSDT` 停更会使 funding 自动窗口长期维持较大值（≤30 天），
  每日重复 upsert 约数万行，属可接受的保守冗余。
- 5m 仍只有 2 币：本轮刻意不扩池。若未来要做 5m 研究，须另行登记**数据口径变更**，
  且不得违反 H2 止损线（现货 OHLCV / K 线级主动成交汇总）。
- 代理偶发 302 可能复发；脚本幂等 + 每日调度可自愈，自动化 prompt 已要求
  列出失败币清单并连续跟踪。

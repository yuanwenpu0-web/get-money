# 币安 DEMO 直连与「以币安为准」改造（工程改造，非 edge 判定）

日期：2026-11-09 · 范围：`backend/app/trading/*`、`market/*`、`api/*`、`frontend/src/*` · 判定类型：**工程复现／执行链路改造，不做 edge 判定**

> 本轮只改执行、同步与记账口径，不新增特征、不调参数、不重跑回测、不启用/停用策略，因此
> 按 `strategy-backtest-playbook` 第 0 节例外跳过 Gate 1（IC 探针）与 Gate 4（统计判定）。
> H3（成本口径）、H8（数据边界）、H10（架构分层）、H12（成对交付／测试／重启）仍然生效。
> **本文任何读数都不是策略盈利能力结论。**

## 1. 假设卡与改造前状态

| 项 | 内容 |
| --- | --- |
| 用户诉求 | 订单直接下到币安 demo；同步收益、止盈止损与下单是否成功；后续可切正式环境，盈利以币安为准 |
| 选定环境 | 币安 U 本位合约 **Testnet**（`testnet.binancefuture.com`），后续切主网只换 `testnet` 开关 |
| 改造前 | 已有合约 REST 下单链路（HMAC 签名、交易所侧 STOP_MARKET / TAKE_PROFIT_MARKET 保护单、回滚），但默认 `execution_mode=paper` |
| 三个缺口 | ① 权益/现金/盈亏全是本地推算账本，不读币安余额，逐笔盈亏用估算费率；② 保护单挂上后不同步状态，触发后按标记价估算平仓；③ 下单失败只写后台日志，前端无法核对成败 |
| 隐蔽障碍 A | 切 `live+testnet` 会让 K 线与币种宇宙**一起**切到测试网 fapi，测试网历史稀疏 → 策略几乎不出信号，下单链路空转 |
| 隐蔽障碍 B | 切换前本地 `cash=0` 且 10 条 paper 持仓占币；若继续用本地现金算仓位，切 live 后一单都开不出来 |
| 成本口径 | paper 仍为 `fee_rate=0.0004` + `slippage=0.0002`（开平合计 0.12%）；**live 逐笔盈亏改用币安 `userTrades` 的真实成交价与手续费**，不再用估算费率，两种口径不混比 |

## 2. 改造内容

### 2.1 环境与行情源解耦

- `market/data_service.py::default_market_client` 与 `market/universe.py` 固定走**主网公开 fapi**（无需 Key），与 `testnet` 开关无关。
- 只有下单与账户通道（`trading/live.py::client_for`）随 `testnet` 切换：`demo` → 测试网域名，`real` → 主网域名。
- 依据（H8 数据边界）：合约测试网的 K 线深度与可交易对远少于主网；demo 与真实行情的价格偏差由交易所真实成交回报修正，收益以交易所账户为准，因此解耦不会造成账实不符。

### 2.2 以币安为准的收益口径

新增 `trading/exchange_gateway.py`（客户端/事件/账户读取）与 `trading/exchange_truth.py`（同步/覆盖/回填/基准）：

| 读数 | 来源 |
| --- | --- |
| 总权益 `equity` | `/fapi/v2/account.totalMarginBalance`（回落 `balance`） |
| 可用现金 `cash` | `availableBalance` |
| 未实现盈亏 | `totalUnrealizedProfit` |
| 已实现盈亏 | `walletBalance − initial_capital`（已结算进钱包，天然含手续费与资金费） |
| 手续费 / 资金费明细 | `/fapi/v1/income`（分页最多 10 页，失败标记 `income_available=false`，不影响主口径） |
| 逐笔订单 | `/fapi/v1/userTrades` 的真实成交均价、`realizedPnl`、`commission`，回填 `entry_price/close_price/entry_fee/exit_fee/realized_pnl` |
| 权益曲线 | 每周期账户快照写 `exchange_account_snapshots`；`equity_snapshots` 在 live 下用交易所权益 |

- `portfolio_snapshot()` 在 live 且 `exchange_truth_enabled=true` 时用交易所快照覆盖，并额外返回 `pnl_source=binance`、`reconcile_diff`、`exchange_env`、`ledger_realized` / `ledger_unrealized`。取不到成交明细的订单显式标 `pnl_source=estimated`，不冒充交易所口径。
- **对账口径**：live 下本地 `cash` 不再随订单增减，「本地权益」已失去可比性，`reconcile_diff` 因此定义为「本地逐笔订单盈亏合计 − 币安账户盈亏合计」（差额主要是账户级手续费、资金费与尚未回填的成交），而不是本地权益与交易所权益之差。
- live 订单不再改动本地 `cash`；`POST /settings/capital` 在 live 下拒绝，改由交易所出入金后「重置收益基准」。
- 切换模式/环境时自动 `rebase`：`initial_capital` = 币安当前钱包余额，避免 demo 与正式资金串账。

### 2.3 止盈止损双向同步

`trading/live_sync.py` 每周期对每条 live 订单：

1. 读交易所标记价写 `mark_price`（`portfolio_snapshot` 不再用本地 K 线覆盖 live 订单）；
2. `get_order` 查两个保护单，**以交易所 `stopPrice` 回写本地 `sl/tp`**，状态写进 `note.sl_status/tp_status`；
3. 保护单 `CANCELED/EXPIRED/REJECTED` 但仓位仍在 → 自动按本地价重挂（走事件留痕），失败写 error 日志告警；
4. 仓位消失 → 优先用 FILLED 保护单的 `avgPrice` 结算，再尝试 `userTrades` 回填，最后才回落标记价；
5. `modify_live` 改为「交易所全部改单成功后一次性写本地」，避免半成功导致两边不一致。

### 2.4 币安条件单接口迁移（Algo Order，实测发现的必修项）

首次 live 巡检暴露 `-4120 STOP_ORDER_SWITCH_ALGO`：币安自 **2025-12-09** 起把 U 本位合约条件单
（`STOP_MARKET` / `TAKE_PROFIT_MARKET` / `STOP` / `TAKE_PROFIT` / `TRAILING_STOP_MARKET`）
从 `POST /fapi/v1/order` 迁到 Algo 服务，旧端点只接受 LIMIT / MARKET。修复前每笔开仓都因保护单
被拒而市价回滚，白亏手续费（demo 账户 4772.727 → 4754.151）。

修复内容（`market/futures.py`、`trading/live.py`、`live_sync.py`、`live_rollback.py`）：

| 动作 | 端点 | 关键字段 |
| --- | --- | --- |
| 挂保护单 | `POST /fapi/v1/algoOrder` | `algoType=CONDITIONAL`、`type`、`triggerPrice`（原 `stopPrice`）、`closePosition=true` |
| 撤保护单 | `DELETE /fapi/v1/algoOrder` | `algoId`（原 `orderId`） |
| 查保护单 | `GET /fapi/v1/algoOrder` | `algoStatus`、`triggerPrice`、`actualPrice`（真实成交均价）、`actualOrderId` |

- `_LoggedClient.new_order` 自动识别条件单类型并路由，返回 `algoId` 归一成 `orderId`，调用点与事件留痕不用分叉；撤单/查询改用 `cancel_algo_order` / `get_algo_order`，并对迁移前的老订单保留 `protection_kind=legacy` 回落路径。
- 条件单**不支持改单**，现有 `modify_live` 本来就是「撤旧挂新」，符合新语义。
- `live_sync` 把 `TRIGGERED` / `FINISHED` 也视为已触发，用 `actualPrice` 结算而非标记价。
- demo 与正式主网都已迁移，因此这一修复对后续切正式环境同样有效。

### 2.5 下单是否成功可观测

新增 `exchange_events` 表：`env / action / symbol / status / exchange_order_id / code / message / avg_price / qty / payload / ts`。
`trading/live.py::_LoggedClient` 透明包装客户端，`new_order` / `cancel_order` 统一走 `exchange_gateway.submit/cancel`，覆盖开仓主单、maker 转市价、保护单、回滚平仓、改单、撤单。

- 状态：`filled` / `accepted` / `canceled` / `rejected`（带币安错误码）/ `failed`（网络或未知）/ `blocked`（本地资金、数量、方向闸门拦截）。
- API：`GET /api/trading/exchange`、`GET /api/trading/exchange-events`、`GET /api/settings/exchange`、`POST /api/settings/exchange/rebase`。
- 前端：仪表盘显示环境徽标、收益口径、近 100 次下单成功率与最近失败；订单页新增「币安下单回执」面板与交易所保护单状态；设置页显示账户余额、重置基准按钮、切主网二次确认。

### 2.6 正式环境防呆

- `PUT /api/settings` 在 `execution_mode=live && testnet=false` 时要求 `confirm_mainnet=true`，且 `credential_verified_env == "real"`（即主网 Key 已通过「测试连接」）。
- 前端切主网弹二次确认。切环境自动 rebase、自动清空旧环境账户缓存。
- 代码层无环境分支：`demo` 与 `real` 共用同一套下单、同步、结算逻辑。

### 2.7 第二轮巡检暴露的可用性修复

Algo 修复后真实巡检成功挂上保护单并开出 4 个 demo 持仓，同时暴露三点，已一并处理：

| 现象 | 原因 | 修复 |
| --- | --- | --- |
| `-2019 Margin is insufficient` 连续拒单 | 一个周期内连开多仓，`position_size` 反复使用周期开始时的 `availableBalance` | 每笔开仓成功后 `exchange_truth.refresh_available()` 立即刷新缓存的可用余额 |
| `-1121 Invalid symbol` | 主网宇宙里的币在 Testnet 不存在 | 记入 `exchange_unsupported_symbols` 短期跳过表（上限 300），换环境时 `rebase` 自动清空；失败事件仍保留可查 |
| `reconcile_diff` 虚高到 4240 | 用失真的本地权益做对账基准 | 改为逐笔订单盈亏 vs 币安账户盈亏（见 §2.2） |

### 2.8 收尾复核：满仓可用保证金闸门与成功率口径

最终复核发现 `/api/health` 仍硬编码 `mode=paper`（与实际 `live + demo` 不符），改为按 `app_state` 动态返回 `mode` 与 `exchange_env`；实测 `{"ok":true,"service":"quant-trading","mode":"live","exchange_env":"demo"}`。

回读又发现连续 `-2019 Margin is insufficient`：账户 `totalPositionInitialMargin=4172.78`、`availableBalance=550.90`、杠杆 1 倍、`totalOpenOrderInitialMargin=0`、无挂单占用；本地 4 个 live 持仓保证金合计约 4170，与交易所一致，**不存在孤仓**。成因是 `portfolio_snapshot()` 在币安口径下把 `equity` 覆盖为 `totalMarginBalance`（含已占用保证金），而 `position_size()` 的 `cash/(1+leverage*fee)` 只留了约 0.04% 手续费余量，算出的保证金几乎吃满 100% 可用余额，逐仓初始保证金加手续费必然被拒；满仓后每个信号还会连发 maker 与市价两次请求。

| 修复 | 位置 |
| --- | --- |
| 执行层按交易所可用余额裁剪仓位：`EXCHANGE_MARGIN_BUFFER=0.90`，`margin` 与 `qty` 同步缩放，保持 `qty = margin × leverage / entry`；**不改 `position_size` 的 equity 入参**，策略风险预算口径不变 | `trading/live.py::_exchange_margin_constraint` / `open_live` |
| 币安口径最低开仓保证金 `max(10, initial_capital × 2%)`，避免满仓后连开碎仓把可用余额吃光 | 同上，`EXCHANGE_MIN_MARGIN_PCT=0.02` |
| 可用余额不足时记 `blocked` 并安静跳过，不再发注定被拒的请求 | 同上 |
| 「主动跳过」与「交易所拒单」分离：`recent_success_rate` 分母只算真正提交的请求，另返回 `recent_blocked` / `recent_blocked_reason`；`recent_failures` 只含 `rejected` / `failed` | `api/trading.py::exchange_state`、仪表盘 |

生产环境只读核对（不下单）：`pnl_source=binance`、`available=359.6136`、`position_margin=4337.8736`、`cap(90%)=323.6522`、`min_margin=95.4545`（`initial_capital=4772.727` 的 2%）→ 闸门允许开仓，但保证金会被裁剪到 323.65，而不是按总权益算到上千。

## 3. 验证记录（2026-11-09）

| 项 | 结果 |
| --- | --- |
| 后端新增单测 `test_exchange_truth.py` | **13 passed**（事件留痕/拒单、账户覆盖、balance 回落、真实成交回填、无成交保留估算、保护单状态回写与自动重挂、保护单成交价结算、rebase 与归档、可用余额刷新、不支持交易对跳过、**满仓按可用余额裁剪仓位 / 余额耗尽安静跳过 / 相对最低保证金**） |
| 实盘链路回归 `test_live_trading.py` + `test_maker_orders.py` + `test_cycle_sync.py` | **43 passed** |
| 后端全量 pytest | 本次改动相关用例全绿；`--lf` 剩 3 项**基线既存**失败，见 §5 |
| 前端 | `npm test` **102 passed**；`npm run build` 成功（9.07s） |
| 服务重启 | `QuantDesk-GetMoney` 已 Stop + Start，`init_db()` 自动建 `exchange_events` / `exchange_account_snapshots` |
| 凭证连通性 | `POST /api/settings/credentials/test` → `ok=true, signed_ok=true, env=demo`，钱包 USDT ≈ 4772.727，另有 BTC 0.01 / USDC 5000 |
| 环境切换 | `execution_mode=live`、`testnet=true`、`initial_capital=4772.727`（按币安余额）、`cash=4772.727`、`exchange_baseline_env=demo` |
| 归档 | 切换时 10 条本地模拟 OPEN 持仓置为 `CLOSED / env_switch_archived`，`realized_pnl=0`，**记录保留、不删除、不计入币安口径收益** |
| 账户快照 | 强制同步后 `pnl_source=binance`、`equity=4772.727`、`reconcile_diff=0`、`exchange_account_snapshots` 有记录 |
| 首次 live 巡检（Algo 修复前） | 订单确实到达 Testnet（`open` / `open_maker` 事件有成交回报），但保护单被 `-4120` 拒绝并回滚：11 条事件、2 笔回滚、0 持仓；币安权益 4772.727 → 4754.151（−18.576，全部是手续费/滑点）。dashboard 正确显示 `realized_pnl=-18.5764`、`reconcile_diff=18.5764`，反证「以币安为准」已生效 |
| Algo 迁移修复 | 条件单改走 `/fapi/v1/algoOrder`；相关回归测试 **51 passed**（含 8 项新增真相源测试） |
| Algo 端到端探针（Testnet 真实） | BTCUSDT 最小仓位：`MARKET` 开仓 OK → `new_algo_order` 挂 `STOP_MARKET` / `TAKE_PROFIT_MARKET` 均返回 `algoId`、`algoStatus=NEW`、`triggerPrice` 正确 → `get_algo_order` 查询 OK → `cancel_algo_order` 撤单 OK → `reduceOnly` 平仓 OK，无残留持仓与条件单。过程中另验证 `-1111`（触发价未按 tick 取整）与 `-4509`（无持仓不能挂 closePosition 条件单）均为探针场景问题，生产代码已用 `round_to_tick` 且在开仓成功后才挂保护单 |
| 探针成本 | 上述探针产生 2 笔小额成交与手续费，已如实计入币安账户余额；非策略收益读数 |
| 修复后第二轮巡检 | `protect_sl` / `protect_tp` 均为 `accepted`（不再 `-4120`），成功开出 **4 个真实 demo 持仓**；事件表 71 条、近 100 次成功率 59.2%，失败原因逐条可见（`-1121` 无效交易对、`-2019` 保证金不足、`-5022` postOnly 被拒）；币安账户 `wallet=4713.1302`、`available=550.9011`、`unrealized=-10.5569`、`realized=-59.5968`（含手续费 −17.4857、资金费 −2.3799） |
| 口径/资金修复 | 新增 `refresh_available`（同周期连开多仓）、`exchange_unsupported_symbols` 跳过表、对账改逐笔盈亏口径；回归 **53 passed** |
| 自动巡检 | 调度器运行中；`POST /api/trading/run-once` 返回「已有交易周期在运行」（上一轮 509 币同步 `elapsed_ms≈223311`），live 周期由调度器按 60s 间隔自动触发，无需手动 |
| 收尾 health 复核 | 硬编码 `mode=paper` 改为动态读取 `app_state`，重启后实测 `mode=live`、`exchange_env=demo`，`State=Running`、8010 端口 LISTEN |
| 满仓闸门修复 | 回读发现连续 `-2019`；核对 `positionRisk` 确认本地 4 仓与币安 `totalPositionInitialMargin` 一致（无孤仓）；加可用余额裁剪、相对最低保证金、blocked 跳过与成功率口径分离；回归 **56 passed** |
| 闸门只读核对 | `available=359.6136`、`position_margin=4337.8736`、`cap=323.6522`、`min_margin=95.4545`、`would_open=true`，验证生产状态下约束生效且不会真下单 |

## 4. 窗口 / 样本期与统计口径

**窗口 / 样本期：不适用**（工程改造，非 edge 判定）。未做回测窗口切分、未做墙钟对齐、未生成成交样本，
因此没有窗口起止、样本期涨跌背景与 skip 清单可写。唯一时间口径是验证时点：2026-11-09（UTC+8），
币安 `server_time = 1794195742903`（毫秒，交易所服务器时钟），巡检日志按 UTC 记录；
凭证连通性、模式切换与账户快照均在同一日内完成。

**CI 与随机基线：不适用**。未产生任何策略收益、bootstrap 95% CI、FDR 或随机基线读数；
本报告不用于启用/停用任何策略。改造前后策略参数、绑定、币池、周期均未改动（生效绑定仍为
`supertrend_current × 15m × all`，`default_enabled` 未变更）。

## 5. 已知边界与未解决项

1. **3 项基线既存测试失败**（与本次改动无关，已在 `git show HEAD` 核实）：
   - `test_pool_volatility_research.py` 两项引用 `dispatch.halt_status`，该属性在 HEAD 版本中就不存在；
   - `test_market_performance.py::test_complete_tail_queries_have_a_bounded_minute_range` 时间窗下界差 5 分钟，相关 `data_service` 查询函数本次未改。
2. **巡检很慢**：全市场 509 币同步一轮约 223 秒，调度间隔 60s 且 `max_instances=1, coalesce=true`，实际下单机会约每 3–4 分钟一次；建议 demo 阶段把 `symbol_scope=custom` 收窄到 testnet 确实存在的交易对（如 BTCUSDT / ETHUSDT）。
3. **testnet 交易对少于主网宇宙**：主网能出信号的币未必在测试网可交易，会产出 `rejected -1121` 事件；已实现 `exchange_unsupported_symbols` 跳过表减少重复请求，但首次探测仍会产生一条失败事件（保留以便核对）。
4. **income 分页上限 10 页**（1 万条）；`userTrades` 单 symbol 单页 1000 条，极高频成交需分批。
5. `limit_maker` 仍是当前 `order_type`：maker 挂单最多轮询 10 秒后转市价，事件表会同时出现 `open_maker` 与 `open` 两条回执。
6. 交易所侧保护单触发后若 `userTrades` 暂不可查，会短暂标 `estimated`，下一周期不会自动重算（已闭单不重放）；需要在 UI 关注 `pnl_source`。
7. 未接入 WebSocket，状态靠 60s 周期轮询；正式环境建议缩短 `schedule_seconds`（下限 10s）并补资金费率/强平模型。
8. 正式环境切换后请重新核对 `FUTURES_MAIN_HOSTS` 与代理可达性，并在「测试连接」中确认 `credential_verified_env=real`。
9. `recent_success_rate` 是**事件级**成功率（近 100 条事件里排除 `blocked` 后 `filled/accepted/canceled` 的占比），历史失败事件不会抹除、读数随新事件滚动；它不是策略胜率，也不应用于任何 edge 判断。
10. 满仓时闸门只按可用余额裁剪仓位，不限制最大持仓数；若 demo/正式环境需要「最多 N 个持仓」，属于策略层风控，应另行评估后再加。

## 6. 回滚方式

- `PUT /api/settings {"execution_mode":"paper"}` 回到本地模拟（旧订单已归档，不影响历史记录）。
- `PUT /api/settings {"exchange_truth_enabled":false}` 可在 live 下临时退回本地账本口径（下单与保护单同步仍生效）。
- 新增表为纯追加，删除 `exchange_events` / `exchange_account_snapshots` 不影响主业务表。

## 7. 结论

demo 直连链路已开启并跑通：订单、Algo 保护单、平仓全部提交到币安合约 Testnet，已成功开出 4 个真实 demo 持仓；收益与逐笔盈亏以币安账户和真实成交为准，止盈止损按交易所保护单状态同步，下单成败与错误码逐条可查。首次实跑发现并修复了币安 2025-12-09 条件单迁移（`-4120`），该修复对 demo 与正式主网同时有效；第二轮巡检又修掉连开多仓保证金计算与无效交易对重复请求。收尾复核又修掉健康探针硬编码 `mode=paper` 与满仓后按总权益算仓位导致的连续 `-2019`（可用余额裁剪 + 相对最低保证金 + 主动跳过与拒单分离）。正式环境只需填入主网 Key、通过连通性验证并显式确认切换，无需改代码。

**本结论仅为工程链路结论，不代表任何策略具备正 edge。** 首次巡检产生的 −18.576 USDT 是接口迁移缺陷造成的手续费损耗，不是策略收益读数；当前生效策略仍为未经统计支持的模拟验证用途，不构成启用建议。

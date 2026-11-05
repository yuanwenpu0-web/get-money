# Supertrend 订单闭环改造 · 工程实现，非 edge 判定

> 本轮只补全「信号 → 挂单 → 成交前复核 → 持仓管理 → 退出」的状态机与专用撮合，
> **不做收益判定**。按 `strategy-backtest-playbook` §0 例外条款跳过 Gate 1（IC 探针）
> 与 Gate 4（统计判定）；H3、H8、H9、H10、H12 仍全部生效。

## 0. 手册合规说明

| 约束 | 本轮如何满足 |
| --- | --- |
| H3 成本全局统一 | 专用撮合只从回测请求读 `fee_rate` / `slippage`（默认 0.0004 / 0.0002，开平双边合计 **0.12%**），按名义本金口径；策略参数里没有成本键，`resolve_params` 仍拒绝策略级覆盖成本 |
| H8 数据边界 | 合成帧只用于时序单测；未接 DB、未声明任何样本期或币池结论，报告里不出现覆盖率读数 |
| H9 并行 | 未触碰并行代码；状态机是单帧 O(n) 顺序回放，无进程池 |
| H10 架构铁律 | 闭环模式 `signals()` 方向恒 0 早路由；新增专用 broker；API 登记条件式拒 gate；`default_enabled=False` 保持不变 |
| H12 成对交付 + 全量测试 + 后端重启 | `docs/supertrend-closed-loop.md` + 同名 `.json`；全量 `pytest backend/tests -q` = **1589 passed / 3 failed**，3 个失败为既存基线问题（见 §7）；已 `Stop-ScheduledTask` + `Start-ScheduledTask` 重启 `QuantDesk-GetMoney`，`curl --noproxy '*' 127.0.0.1:8010/api/strategies` 返回 200 且可见 `tags` 含 `closed_loop`、`cl_enabled=False`、`cl_pending_expire_bars=8` |
| 未做的项 | 窗口 / 币池 / 95% CI / 多种子随机基线 / 五态 edge 判定：本轮不产出收益结论；`preflight.py` 是关键词匹配，因本文在「不适用」语境里提到这些词而报 21 PASS，**不得据此误读为已有 CI / 随机基线 / 样本期读数**；同名 JSON 中这些字段均为 `null` 并附 `not_applicable_reason` |

统计证据状态：`insufficient`（未跑 Gate 3/4，没有任何可用于判定的成交样本）。
启用状态：`default_enabled=False`，不进绑定。

## 1. 改动清单

| 文件 | 类型 | 作用 |
| --- | --- | --- |
| `backend/app/strategy/closed_loop.py` | 新增 251 行 | 纯因果订单闭环状态机（无资金/成本概念） |
| `backend/app/backtest/closed_loop_execution.py` | 新增 85 行 | 专用 broker：事件列 → 权益曲线 + 成交列表，按全局口径扣双边成本 |
| `backend/app/strategy/specs/supertrend_current.py` | 修改 +119 行 | 扩展参数/列、旁路输出枢轴电平、早路由、describe 快照；保留全部既有命名与接口 |
| `backend/app/backtest/runner.py` | 修改 +68 行 | `run_closed_loop_backtest()` 整窗回放 + `run_backtest()` 条件早路由 |
| `backend/app/api/strategies.py` | 修改 +10 行 | H10 拒 gate：禁止把 `cl_enabled=True` 开到巡检/模拟盘 |
| `backend/tests/test_supertrend_closed_loop.py` | 新增 285 行 | 17 个用例：状态覆盖、幂等、冲突优先级、因果性、参数校验、接口兼容、成本口径 |

## 2. 状态机设计

**三个持久状态**（指标失效、趋势破坏、止盈止损都是**事件**，不新增可重入的歧义状态）：

| 状态 | 值 | 含义 |
| --- | --- | --- |
| `FLAT` | 0 | 空仓、无挂单 |
| `PENDING` | 1 | **挂单未成交**（含支撑位定价、挂单阶段已配置的 SL/TP） |
| `OPEN` | 2 | **已成交持仓**（持续做趋势判断） |

**事件**（`st_cl_event`，每根最多一个）：`submit` / `skip_no_support` / `fill` /
`cancel_indicator`（**指标失效**）/ `cancel_expired` / `close_stop_loss` /
`close_take_profit`（**固定 3R**）/ `close_take_profit_dynamic`（**动态止盈**）/
`close_trend_break`（**趋势破坏**）。

**因果时序**（与手册 §2.4 一致，禁止当根信号当根成交）：每根 bar 分三阶段——

1. **阶段 1**：执行 `t-1` 收盘做出的决定（撤单 / 趋势破位市价平仓），**早于本根任何撮合**；
2. **阶段 2**：只用本根 OHLC 撮合「`t-1` 收盘时已挂着」的单 / 已开的仓；
3. **阶段 3**：`t` 收盘用**已确认**数据复核指标与趋势，决定只对 `t+1` 生效。

因此：挂单在 bar `s` 收盘提交、`s+1` 起可成交；指标在 `s` 收盘失效 → 撤单在 `s+1`
生效，撤单发生在该根成交之前；趋势在 `t` 收盘破位 → `t+1` 开盘立即平仓。

## 3. 五类状态覆盖

| 需求 | 实现 |
| --- | --- |
| 信号触发 | 取 `st_signal` **上升沿**（`sig != sig[t-1]`）且 `indicator_ok`（方向 + ADX/EMA200 过滤 + 未被反向信号取代） |
| 支撑位判断 → 买入价 | 多头取枢轴支撑连线 `st_support`、空头取 `st_resistance`（`_setup_signals(..., levels_out=...)` 旁路输出，两参数调用行为不变）；缺枢轴时回退近 `2×pivot_length+1` 根摆动低/高点（纯因果 rolling）；须落在现价有利一侧且距离 ≤ `cl_max_entry_distance_atr` 倍 ATR，否则 `skip_no_support` 不挂单 |
| 提交挂单 + 同步配置止盈止损 | 限价 = 电平 ± `cl_support_buffer_atr`×ATR；**提交当根即算出** `planned_sl`（1.5ATR）与 `planned_tp`（3R 或动态模式下为空）；成交后按实际成交价重锚，保持 R 不漂移 |
| 动态 / 固定 3R 两模式 | `cl_tp_mode=0` 固定 `target_r`×R 限价止盈；`=1` 无固定目标，浮盈达 `cl_dynamic_activate_r`×R 后按最优价 − `cl_trailing_atr_mult`×ATR 单调 ratchet 追踪保护价（收盘更新、下一根生效） |
| 成交前持续复核 + 及时撤单 | `PENDING` 每根收盘跑 `indicator_ok`，失效即排 `cancel_indicator`；另有 `cl_pending_expire_bars`（默认 8）超时撤单，防挂单长期悬空 |
| 持仓趋势判断 + 破位平仓 | `OPEN` 每根收盘跑 `trend_broken`（Supertrend 方向翻转，或收盘跌破/升破趋势带），破位即排 `close`，下一根开盘市价平 |

## 4. 防重复与冲突优先级

- **单一平仓出口**：所有平仓都要经过同一个 `if price is not None` 分支，平完立刻
  `state=FLAT` + `closed_this_bar=True`；本根后续阶段 3 不再排任何平仓或挂单决定。
- **同根优先级**：保护价（止损 / 动态追踪）> 固定止盈 > 收盘趋势破位决定；
  趋势破位在下一根开盘执行时，若开盘已跳空越过保护价/止盈，按更保守的原因记账，**仍只平一次**。
- **防重复挂单**：`submit` 只在 `state==FLAT` 且本根无事件、无冷却、信号上升沿时发生；
  每个挂单分配单调递增 `order_id` / `cycle_id`，平仓行仍带被平订单号以便审计。
- **防重复撤单**：`cancel` 只在 `state==PENDING` 时生效，撤完立即回 `FLAT`；
  同一根不会既撤单又挂单（本根已有事件即抑制 `submit`）。
- **防重复成交**：`fill` 只在 `state==PENDING` 且 `t >= active_from` 时发生，成交后立即转 `OPEN`。

## 5. 新增参数（全部默认值保持旧行为）

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `cl_enabled` | `False` | 闭环总开关；关闭时 `signals()` 与改造前逐字节等价 |
| `cl_tp_mode` | `0` | 0 固定 3R / 1 动态追踪 |
| `cl_support_buffer_atr` | `0.10` | 挂单价相对支撑/阻力的 ATR 缓冲 |
| `cl_max_entry_distance_atr` | `3.0` | 现价到挂单价的最大 ATR 距离 |
| `cl_pending_expire_bars` | `8` | 挂单最多等待根数，0 = 不超时 |
| `cl_dynamic_activate_r` | `1.0` | 动态止盈启动浮盈 R 门槛 |
| `cl_trailing_atr_mult` | `1.5` | 追踪保护价距离（ATR 倍数） |
| `cl_cooldown_bars` | `0` | 平仓后冷却根数 |

全部进 `RANGES` / `int_keys` / `bool_keys` / `PREPARE_KEYS`，经 `resolve_params` 三层校验；
`validate()` 额外拒绝非法 `cl_tp_mode` 与「缓冲 > 最大挂单距离」。成本键未增加（H3）。

## 6. 接口兼容清单

- `CODE` / `DEFAULTS` / `RANGES` / `validate` / `lookback` / `_rma` / `_setup_signals` /
  `prepare` / `signals` / `describe` / `SPEC` 名称与签名不变；`_setup_signals` 只多一个
  **可选**第三参数 `levels_out`，原两参数调用（含既有单测）行为逐位一致。
- `signals()` 仍返回完整 `SIGNAL_COLUMNS`，默认模式下方向、sl_pct、tp_pct 全部不变；
  `prepare()` 只**新增** `st_support` / `st_resistance` / 18 个 `st_cl_*` 列，覆盖任何既有列。
- 早退判据只认原有 7 个指标列：老代码/单测手工注入 `st_*` 后只传单行时，**不会**因新增
  列缺失而重算、丢掉 setup 状态机历史（这是唯一一处曾触发回归、已修复的兼容点）。
- `grid` 未加闭环参数，避免寻优组合爆炸；`default_enabled=False` / `tags` 追加
  `closed_loop`；原 1R/2R/3R 分批止盈与 15m 绑定行为完全不动。

## 7. 验收指纹

- `pytest backend/tests/test_supertrend_closed_loop.py backend/tests/test_supertrend_current.py -q`
  = **37 passed**（17 个新增 + 20 个既有回归全绿）。
- 全量 `pytest backend/tests -q` = **1589 passed / 3 failed**。3 个失败为既存基线问题，
  与本次改动无交集、单独重跑稳定复现：
  `test_pool_volatility_research.py::test_live_entrypoints_reject`（2 个，`dispatch.py` 缺
  `halt_status` 属性，该文件本次未改且 HEAD 版本同样缺失）、
  `test_market_performance.py::test_complete_tail_queries_have_a_bounded_minute_range`
  （`data_service._candle_query` 下界断言，属行情数据层，与策略/回测闭环无关）。
- 确定性事件序列（固定 3R 合成帧）：`submit → fill → close_take_profit`，
  `planned_sl=97.1`、`planned_tp=103.1`、成交价 98.6、平仓价 103.1，全场仅 1 次平仓。
- 指标失效帧：`submit → (PENDING×3) → cancel_indicator`，零 `fill`、零重复撤单。
- 趋势破位帧：方向在 bar6 收盘翻转 → bar7 开盘 `close_trend_break`，平仓价 102.0。
- 截断因果性：完整帧与截到 30/60/85 根的 `st_cl_*` 前缀逐列 `assert_frame_equal` 通过。
- 幂等不变量（220 根随机帧）：每个 `submit`/`fill`/`cancel`/`close` 的前后状态转移合法，
  被平 `order_id` 无重复，成交数 ≥ 平仓数。

## 8. 局限与边界

- **bar 级数据固有边界**：指标在 `t` 收盘才确认，若挂单在 `t` 盘中先触价成交、同根收盘
  指标才转坏，无法「未卜先知」地撤单；本实现用「`t-1` 收盘复核 → `t` 生效」把可避免的
  滞后全部消除，剩余部分不可消除，已在 `warnings` 中显式标注。
- **闭环仅支持回测**：`cl_enabled=True` 时 `signals()` 方向恒 0，模拟盘/实盘不会产生任何
  开仓；API 拒 gate 明确报错。交易所侧挂单状态同步、撤单竞态、保护单落地不在本轮范围。
- 支撑位主用枢轴连线（需两个已确认枢轴，信号根常缺），故加近端摆动极值回退；两者都
  判不出来则不挂单，宁可空仓不盲挂。
- 未模拟：部分成交、盘口深度、maker/taker 差异、资金费率、逐笔成交顺序、多所现货腿。
- 本轮**不产生任何收益结论**，不改动历史 verdict（15m 换手、分批止盈、止损线的既有
  结论继续有效），不启用策略、不改动绑定。

## 9. 结论

订单闭环已按需求补全并通过 37 项定向单测与全量回归；统计证据 `insufficient`，
策略保持 `default_enabled=False`、不进绑定。若要判断该闭环是否有 edge，须另起课题
按 Gate 0–6 做探针、三组对照回测与 bootstrap CI，本报告的工程读数不得引用做决策。

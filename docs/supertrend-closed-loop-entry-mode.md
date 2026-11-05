# 订单闭环进场价模式（cl_entry_mode）· 工程实现，非 edge 判定

> 本轮只把闭环的「进场价怎么定」做成显式参数，**不跑收益回测、不做寻优、不产出 edge 判定**。
> 按 `strategy-backtest-playbook` §0 例外条款跳过 Gate 1（IC 探针）与 Gate 4（统计判定）；
> H3、H8、H9、H10、H12 全部生效。前序工程报告见 `docs/supertrend-closed-loop.md`。

## 0. 假设卡与手册合规

- **信息源 / 可证伪假设**：现有闭环只支持「信号 → 同向支撑/阻力回踩挂单」一种进场价。
  本轮假设「市价 / 反向回踩 / 对侧电平突破」三种价格来源可以共用同一套状态机与退出逻辑，
  且默认档逐位复现改造前读数；若定向单测出现任何旧口径漂移，即判本轮失败并回滚。
  **本假设只关乎工程时序，不关乎收益**，事前终止条件＝默认档非逐位一致。
- **H2 止损线核对**：已查 `references/verdicts-history.md`。现货 OHLCV 价格序列与
  「支撑位判定更准」两条路线均已证伪／`inconclusive`。本轮**不新增特征、不调阈值、
  不换周期、不跑网格、不比收益**，只把既有闭环里已经写死的价格来源变成可配置项，
  因此不属于「在价格序列上再投一次参数」，也不改动任何历史 verdict。
- **周期 / 持有期 H / 窗口起止 / 样本期 / 币池与币数**：不适用。未接 DB、未加载任何行情，
  全部验收使用合成帧；没有窗口、没有币池、没有 skip 清单，也没有样本期涨跌背景可写。
- **对照组**：不适用（不做收益判定）。工程对照只有「默认档 vs 显式 `cl_entry_mode=1`」
  的逐位一致性，以及同一合成帧上三档的事件序列差异。
- **CI / 随机基线**：不产出。95% 置信区间、多种子随机方向基线、FDR 均为 `null`，
  本文出现这些词只用于声明「不适用」，不构成任何统计读数。

| 约束 | 本轮如何满足 |
| --- | --- |
| H3 成本全局统一 | 三档共用一个专用 broker；成本仍只从回测请求读 `fee_rate=0.0004` / `slippage=0.0002`（双边 **0.12%**，**名义本金**口径）。市价档不额外加价差参数，突破档的「更保守成交价」是价格路径假设，不是成本覆盖；`resolve_params` 仍拒绝策略级覆盖成本 |
| H8 数据边界 | 合成帧只用于时序单测；未声明任何覆盖率、样本期或币池结论 |
| H9 并行 | 状态机仍是单帧 O(n) 顺序回放，未触碰进程池代码 |
| H10 架构铁律 | 只改既有 `StrategySpec`，参数过 `resolve_params`；闭环模式仍 `direction=0` 早路由、仍由专用 broker 回放；`default_enabled=False` 不变；API 条件式拒 gate 不变 |
| H12 成对交付 + 全量测试 + 重启 | 本报告 md＋json 成对；定向 54 passed、全量 `1630 passed / 3 deselected`（3 项为既存基线失败）、前端 74 项测试 + 构建通过；已重启 `QuantDesk-GetMoney` 并实测接口新参数生效 |
| 未做的项 | 窗口 / 币池 / 95% CI / 多种子随机基线 / FDR / 五态 edge 判定：本轮不产出收益结论。`preflight.py` 是关键词匹配，因本文在「不适用」语境里提到这些词而报 21 PASS，**不得据此误读为已有 CI / 随机基线 / 样本期读数**；同名 JSON 中这些字段均为 `null` 并附 `not_applicable_reason` |

统计证据状态：`insufficient`（未跑 Gate 3/4），启用状态：`default_enabled=False`，不进绑定。

## 1. 三档语义

「反向支撑位」在项目里有歧义（既可指「等价格反向回撤到支撑位」，也可指「取方向相反的
那条电平」），因此两读都做成显式档位，另加市价档，避免二选一时猜错语义：

| `cl_entry_mode` | 名称 | 进场价来源 | 触发方向 | 触价后成交价 |
| --- | --- | --- | --- | --- |
| `0` | 市价进场 | 不做电平判定，信号 bar `s` 收盘提交、`s+1` **开盘价**成交 | 无 | 开盘价（开盘缺失才回退该根收盘）；SL/TP 按信号收盘预配、成交后按实际价重锚 |
| `1` | 反向回踩支撑位挂单（**默认，旧口径**） | 多头取枢轴支撑 `st_support`、空头取 `st_resistance`；缺则回退近端摆动极值 | 多头 `low <= limit` / 空头 `high >= limit` | `limit`；跳空按对交易者更优的开盘价（旧逻辑逐位不变） |
| `2` | 对侧电平突破挂单 | 多头取枢轴阻力 `st_resistance`、空头取 `st_support`；缺则回退对应摆动极值 | 多头 `high >= limit` / 空头 `low <= limit` | `limit`；跳空越过触发价按**更保守**（更差）的开盘价成交，属 stop 语义 |

三档共用：静态 `cl_support_buffer_atr` 缓冲、`cl_max_entry_distance_atr` 距离上限、
`sl_atr_mult`×ATR 保护价、固定 3R 或动态追踪止盈、指标失效/超时撤单、趋势破位次根开盘平仓、
冷却与幂等守卫、以及同一套全局成本。

## 2. 改动清单

| 文件 | 类型 | 作用 |
| --- | --- | --- |
| `backend/app/strategy/closed_loop.py` | 修改（现 300 行） | 新增 `ENTRY_MARKET/PULLBACK/BREAKOUT` 常量、`st_cl_entry_mode` 审计列；`quote_limit` 改为 `quote_entry(t, side, mode)` 统一产出 `kind/touch/reference/risk`；阶段 2 分市价／回踩／突破三种撮合；阶段 3 市价单只在生效根有效，缺价即 `cancel_market_unfilled` |
| `backend/app/strategy/specs/supertrend_current.py` | 修改 | 新增 `cl_entry_mode`（默认 1）、`RANGES`/`int_keys`/`PREPARE_KEYS`/`validate`/`describe`/`description` 同步；市价档跳过「缓冲 > 最大挂单距离」这一条与电平无关的校验 |
| `backend/app/backtest/runner.py` | 修改 | 闭环回测指标增 `closed_loop_entry_mode` 与 `closed_loop_fill_modes`，warnings 写明本次进场档位 |
| `frontend/src/views/Strategies.vue` | 修改 | 策略编辑弹窗增「订单闭环 · 进场价格方式」下拉，与 `cl_entry_mode` 双向同步，并标注只在 `cl_enabled=true` 的闭环回测生效 |
| `backend/tests/test_closed_loop_entry_mode.py` | 新增 229 行 | 17 项定向单测：默认逐位一致、市价成交价/成本/不悬空/截断因果、突破多空取对侧电平与保守跳空、未触价撤单、三档幂等不变量、参数校验、describe 快照 |

专用 broker `backtest/closed_loop_execution.py` **未改**：它只消费 `st_cl_fill_price` 与
`close_*` 事件，三档共用同一套扣费路径，这是 H3 不被破坏的关键。

## 3. 因果时序与新增事件

- 三档一律「bar `s` 收盘决策 → `s+1` 生效」；阶段 1 执行旧决策、阶段 2 只用本根 OHLC
  撮合、阶段 3 用已确认数据复核并排下一根的决定。
- 市价档没有挂单等待：`submit` 当根即按信号收盘价预配 `planned_sl` / `planned_tp`，
  `s+1` 开盘成交后按实际成交价重锚，R 不漂移；`st_cl_pending_price` / `st_cl_level` 为 NaN。
- 新增事件 `skip_invalid_atr`（市价档 ATR 无效时不提交，旧档仍叫 `skip_no_support`）与
  `cancel_market_unfilled`（市价单生效根开盘、收盘都无价，下一根撤单，防止拖成僵尸挂单）。
- 突破档是顺向 stop 单：触发价必须在现价**不利**一侧且距离 ≤ `cl_max_entry_distance_atr`×ATR；
  回踩档仍要求在**有利**一侧。两者都不会在同一根既撤单又挂单、既成交又平仓。
- `st_cl_entry_mode` 在有活动挂单／持仓的行记录本轮档位（0/1/2），空仓行为 NaN，
  供 `describe()` 与 runner 的 `closed_loop_fill_modes` 审计。

## 4. 参数与校验

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `cl_entry_mode` | `1` | 0 市价 / 1 反向回踩支撑位 / 2 对侧电平突破 |

经 `resolve_params` 三层合并：`RANGES=(0,2)`、`int_keys` 拒绝小数、`validate()` 拒绝
0/1/2 之外取值；`PREPARE_KEYS` 已登记，改档会触发帧重算；`grid` 未加该键，
**不参与寻优、不做组合爆炸**。市价档忽略 `cl_support_buffer_atr` 与
`cl_max_entry_distance_atr`（因此允许二者冲突），其余两档仍执行「缓冲不能大于最大挂单距离」。

## 5. 接口兼容

- `cl_enabled=False`（线上现状）时行为完全不变；`cl_enabled=True` 且未显式配置时，
  `cl_entry_mode=1` 与改造前**逐位一致**（定向单测 `assert_frame_equal` 验证）。
- 既有 18 个 `st_cl_*` 列名与顺序不变，只在末尾追加 `st_cl_entry_mode`；
  `SIGNAL_COLUMNS`、早退判据（只认原 7 个 `st_*` 指标列）、`prepare`/`signals` 签名、
  `direction=0` 早路由、API 拒 gate、`grid`、`default_enabled=False` 全部不动。
- 前端编辑弹窗把 `cl_entry_mode` 从 JSON 文本里抽出成下拉，保存时写回；
  与三级止盈表单同样的处理路径，其他策略的编辑界面不受影响。

## 6. 验收指纹

- `pytest backend/tests/test_closed_loop_entry_mode.py backend/tests/test_supertrend_closed_loop.py backend/tests/test_supertrend_current.py -q` = **54 passed**（17 新增 + 17 闭环 + 20 既有回归）。
- 全量 `pytest backend/tests -q`：**1633 项中 1630 passed / 3 failed**；3 项为既存基线失败
  （`test_pool_volatility_research.py::test_live_entrypoints_reject` ×2 引用已移除的
  `dispatch.halt_status`；`test_market_performance.py::test_complete_tail_queries_have_a_bounded_minute_range`
  行情查询下界断言）。显式 deselect 这 3 项后重跑 = **1630 passed / 3 deselected / 0 failed**，
  证明本轮没有引入新的失败；`compileall backend/app` 通过。
- 前端 `npm test` = **74/74 pass**；`npm run build` = ✓ built，`dist` 已重建。
- 确定性指纹（合成帧，ATR=1、sl=1.5R、TP=3R、buffer=0.1ATR）：
  - 市价档：支撑位设成 1.0（回踩档在此帧 `skip_no_support`），市价档 `submit → fill`，
    成交价 `100.5`（`s+1` 开盘）、`planned_sl=98.5` / `planned_tp=104.5`、
    成交后 `sl=99.0` / `tp=105.0`；broker 记 `entry=100.5×(1+slippage)`。
  - 突破档多头：`st_cl_level=100.4`（阻力）、`pending_price=100.5`，
    开盘 100.8 高于触发价 → 按 100.8 成交；空头档 `st_cl_level=99.6`（支撑）、
    `pending_price=99.5`。
  - 未触价帧 `submit → cancel_expired`，零 `fill`；缺价帧 `submit → cancel_market_unfilled`。
  - 三档 × 200 根随机帧：每次 `submit`/`fill`/`cancel`/`close` 的状态转移合法，
    被平 `order_id` 无重复，平仓数 ≤ 成交数；市价档截断到 30/60/85 根前缀逐列 `assert_frame_equal` 通过。
- 按 H12 重启 `QuantDesk-GetMoney` 后，`curl --noproxy '*' 127.0.0.1:8010/api/strategies`
  = 200：`effective_params.cl_entry_mode=1`、`default_params.cl_entry_mode=1`、
  `cl_enabled=False`、tags 含 `closed_loop`；9 个 `cl_*` 参数全部可见。

## 7. 局限与边界

- 合成帧只验证时序与价格口径：没有真实行情、没有窗口 / 币池 / 样本期，零成交或
  任何收益读数都**不得**引用做决策；95% CI、随机基线、FDR 均不适用。
- bar 级 OHLC 无法表达盘中先后顺序：突破单与止损/止盈同根时仍按既有优先级
  （保护价 > 固定止盈 > 趋势破位）只平一次；市价档的开盘成交同样不模拟盘口深度、
  maker/taker 差异、部分成交、资金费率与多所现货腿。
- 闭环**仅回测可用**：`cl_enabled=True` 时 `signals()` 方向恒 0，模拟盘/实盘由 API
  拒 gate 阻断；前端下拉只在闭环打开后才有实际效果。
- 支撑/阻力仍用枢轴连线（需两个已确认枢轴）＋近端摆动极值回退；既有
  `docs/support-level-evaluation.md` 的 `inconclusive` 结论不变，本轮没有让任何
  支撑位定义变得更准。
- 本报告不改动历史 verdict，不启用策略、不改动任何绑定，`default_enabled=False` 保持。

## 8. 结论

**verdict：`insufficient`（未跑 Gate 3/4，工程验收通过）**。三档进场价已由
`cl_entry_mode` 显式控制，默认档与改造前逐位一致，54 项定向单测与全量回归通过，
服务重启后新参数在接口可见。`default_enabled=False`、不进绑定；若日后要判断
「市价 vs 回踩 vs 突破」哪档更好，须另起课题按 Gate 0–5 做受控 A/B（唯一变量
`cl_entry_mode`）、池化 CI 与多种子随机基线，本报告读数不得引用做启用依据。

## 9. 复现命令

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests/test_closed_loop_entry_mode.py backend/tests/test_supertrend_closed_loop.py backend/tests/test_supertrend_current.py -q
.\.venv\Scripts\python.exe -m pytest backend/tests -q
.\.venv\Scripts\python.exe -m compileall -q backend/app
cd frontend; npm test; npm run build
Stop-ScheduledTask -TaskName QuantDesk-GetMoney; Start-ScheduledTask -TaskName QuantDesk-GetMoney
curl --noproxy '*' http://127.0.0.1:8010/api/strategies
.\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/supertrend-closed-loop-entry-mode
```

## 10. 产物

- `docs/supertrend-closed-loop-entry-mode.md` + 同名 `.json`
- 单测 `backend/tests/test_closed_loop_entry_mode.py`、当日工作记忆登记

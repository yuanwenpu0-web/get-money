# Simple ZigZag Double —— TradingView Pine v5 本地移植

把用户提供的 Pine v5 策略 `Simple ZigZag Double` 1:1 移植为本地回测策略，
**出入场严格按原脚本语义**，不走项目通用的「收盘信号 → 市价单」管线。

- 报告：`docs/simple-zigzag-double.md`
- 数据：`docs/simple-zigzag-double.json`（主用例回测结果，由脚本产出）
- 状态：已完成。本次续做修复了虚拟仓位的 Pine 语义偏差，补齐回归测试、真实数据
  端到端回测与本文档。

---

## 1. 为什么需要专用撮合器

Pine 脚本用 `strategy.entry("Long", strategy.long, LOT_LONG, stop = LEVEL_LONG)`
挂的是**持久止损单**：条件成立的每根 bar 只是「创建 / 更新」同 ID 挂单，条件不成立时
挂单不撤销；价格穿越 stop 才成交，反向单成交即反手。项目的通用管线只有
`direction` + ATR 止损/止盈，无法表达这种挂单，也不能把「突破前高的一根 K 线」
简化成收盘方向信号。

因此：

| 层 | 文件 | 作用 |
|---|---|---|
| 策略 spec | `backend/app/strategy/specs/simple_zigzag_double.py` | 计算 Pine 的 ZigZag 价位、虚拟仓位、入场条件，输出 `zz_*` 列 |
| 专用撮合器 | `backend/app/backtest/zigzag_execution.py` | `simulate_zigzag`：逐 bar 回放持久挂单、反手、止损、time_end 平仓 |
| 回测分支 | `backend/app/backtest/runner.py::run_zigzag_backtest` | 按 code 分派，Pine 默认资金/成本，跳过通用数据集与选参 |
| 回测入口 | `backend/scripts/zigzag_backtest.py` | 命令行复现，产出 JSON 报告 |
| 测试 | `test_simple_zigzag_double.py` / `test_zigzag_execution.py` / `test_zigzag_runner.py` | 17 项 |

`signals()` 刻意返回 `direction=0` 且 `block_reason="stop_order_requires_dedicated_broker"`，
保证这个策略永远不会被通用市价单引擎误交易。`simulate_zigzag` 也**不套用**项目通用
风控（ATR 止损、冷却、min_hold、强平），因为原 Pine 脚本里没有这些东西。

---

## 2. Pine 逐行语义对照

### 2.1 指标：三重 EMA + ZigZag 价位

```
l  = ta.lowest(Detection)          # 近 Detection 根最低 low
h  = ta.highest(Detection)         # 近 Detection 根最高 high
ema1 = ta.ema(close, math.round(Length/3 * 3))
ema2 = ta.ema(ema1, math.round(Length/3 * 2))
ema3 = ta.ema(ema2, math.round(Length/3))
z = ema3 >= ema3[1] and ema3[1] < ema3[2] ? l
  : ema3 <  ema3[1] and ema3[1] >= ema3[2] ? h : 0.0
point := z == 0.0 ? point[1] : z
LEVEL_LONG  := point >  point[1] ? z : LEVEL_LONG[1]
LEVEL_SHORT := point <  point[1] ? z : LEVEL_SHORT[1]
```

实现要点：

- **Pine `math.round` 是 .5 向上取整**，Python `round` 是银行家舍入，实现用
  `math.floor(x + 0.5)` 对齐。`Length=5` → 三重 EMA 长度 5 / 3 / 2。
- `ta.ema` 用 `alpha = 2/(len+1)`、`adjust=False`、首值取 `source[0]`，与 pandas
  `ewm(span=n, adjust=False).mean()` 一致；历史不足 length 根时照常计算。
- `z` 只在 **ema3 拐点**取 `l`/`h`，否则为 0；`point` 用 `ffill` 持久化最近一次非零 pivot。
- **两组参数完全独立**：做多取第一组的 `LEVEL_LONG`，做空取第二组的 `LEVEL_SHORT2`
  （原脚本 `plot` 和入场条件都是这样配对，`LEVEL_SHORT` / `LEVEL_LONG2` 计算了但未被使用）。
- 语义直觉：pivot 在 bottom(`l`) / top(`h`) 间交替，`point` 上升通常对应
  bottom→top，因此 `LEVEL_LONG` 实际是**逐步抬高的顶部 pivot（前高）**，
  `LEVEL_SHORT2` 是**底部 pivot（前低）**。策略即「上破前高做多、下破前低做空」。

### 2.2 虚拟仓位 —— 本次修复的核心

```
if strict_position
    if virtual_pos == 1  and close <= LEVEL_SHORT2 and close[1] > LEVEL_SHORT2
        virtual_pos := 0
    if virtual_pos == -1 and close >= LEVEL_LONG  and close[1] < LEVEL_LONG
        virtual_pos := 0
if virtual_pos != 1  and close >= LEVEL_LONG       # 顶格，不在 strict 块内
    virtual_pos := 1
if virtual_pos != -1 and close <= LEVEL_SHORT2     # 顶格，不在 strict 块内
    virtual_pos := -1
```

**后两个方向翻转在 `if strict_position` 块之外，是无条件执行的**（Pine 缩进敏感）。
`strict_position` 只包住上面两个「虚拟出场」重置；而重置之后紧跟的无条件翻转又会
把它覆盖，所以**严格按脚本文本，这个开关对最终 `virtual_pos`、进而对出入场条件
没有实际影响**。实现保留该参数只是为了 1:1 映射源码。

半成品里四个 `if` 被一起放进了 `if strict_position:` 块，导致默认
`strict_position=false` 时 `virtual_pos` 恒为 0：

- 两个入场条件 `virtual_pos != 1` / `virtual_pos != -1` 恒真；
- 价格已经上破 `LEVEL_LONG` 之后，仍会每根 bar 用**新** pivot 去更新多单挂单价，
  而 Pine 此时 `virtual_pos=1`、多单条件为假，挂单保持旧价位不动。

这会实质改变成交价与成交时点。修复方式是把后两个 `if` 移出守卫块，并补 3 项回归测试
（默认参数下虚拟仓位跟踪、`strict` 与 `non-strict` 最终状态一致、首根即突破时
不挂多单的端到端用例）。

### 2.3 入场条件

```
long_entry_condition  = LEVEL_LONG  > 0 and needlong  and truetime and virtual_pos != 1
short_entry_condition = LEVEL_SHORT2 > 0 and needshort and truetime and virtual_pos != -1
```

- `truetime = time > timestamp_start and time < timestamp_end`（**严格**不等号），
  `time` 是 K 线开盘时间（ms），与本地 K 线索引口径一致。
- 条件为真时 `strategy.entry` **更新**同 ID 持久挂单；为假时挂单保留（不撤销）。
  挂单最早在下一根 bar 才可能成交，不会在同 bar 回溯。

### 2.4 订单与出场

```
strategy.entry("Long",  strategy.long,  LOT_LONG,  stop = LEVEL_LONG)
strategy.entry("Short", strategy.short, LOT_SHORT, stop = LEVEL_SHORT2)
strategy.cancel("Long",  when = needlong  == false)
strategy.exit("Exit Long",  "Long",  stop = LEVEL_SHORT2, when = needshort == false)
strategy.cancel("Short", when = needshort == false)
strategy.exit("Exit Short", "Short", stop = LEVEL_LONG,   when = needlong  == false)
if time > timestamp_end
    strategy.close_all()
    strategy.cancel("Long"); strategy.cancel("Short")
    virtual_pos := 0
```

- `pyramiding=0`：反向 entry 成交即**反手**（平旧仓 + 开新仓），状态记 `signal_reverse`。
- 默认 `needlong = needshort = true` → 两个 `strategy.exit` 的 `when` 恒假，
  **不创建任何止损 bracket**，多空都只靠反向 entry 反手平仓。这是原脚本的真实行为，
  不是遗漏。关闭某一方向时，另一方向才会挂上 `stop` 出场单（状态 `stop_loss`）。
- `time > timestamp_end` 的 `close_all` 在下一根开盘以市价平仓（状态 `time_end`），
  同时清空挂单并把 `virtual_pos` 归零。

### 2.5 仓位 `LOT_LONG` / `LOT_SHORT`

```
LOT_LONG := fixed_lot == false
    ? strategy.position_size != strategy.position_size[1]
        ? strategy.equity / close * lot_long / 100
        : LOT_LONG[1]
    : lot_long
```

- 序列初值为 `na`（第一根 `position_size[1]` 为 na，条件 na → 取 `LOT_LONG[1]` = na）。
  `qty=na` 时 `strategy.entry` 回落到 strategy 默认 **100% 权益**
  （`default_qty_type=strategy.percent_of_equity, default_qty_value=100`）。
- 只有**持仓发生变化**的那根 bar 才重算，用 `strategy.equity`（现金 + 已实现 + 浮动盈亏）
  除以当前收盘价；订单创建/更新时冻结 qty，成交按该 qty。
- `fixed_lot=true` 时直接把 `lot_long` / `lot_short` 当作绝对数量。

### 2.6 撮合口径（对齐 TradingView Broker Emulator）

| 规则 | Pine / TradingView | 本地实现 |
|---|---|---|
| 计算时点 | 每根 bar 收盘算指标与挂单条件 | 同 |
| 最早成交 | 下一根 bar 开盘（不在同 bar 回溯） | 同 |
| intrabar 路径 | open 离 high 近 → `open→high→low→close`；离 low 近 → `open→low→high→close` | 同 |
| bar 内穿越 stop | 按 stop 价成交（bar 内假设无 gaps） | 同 |
| bar 间 gap 穿越 | 按**下一根开盘价**成交，不按 stop 价 | 同 |
| 提交时已穿越 stop | stop 转 market 单，下一根开盘按开盘价无条件成交 | `marketable` 标记 |
| 佣金 / 滑点 | `commission_value=0.1` 即 0.1%/边，默认无滑点 | 默认 fee=0.001 / slippage=0 |

---

## 3. 参数默认值（对齐 Pine input）

| 参数 | 默认 | 说明 |
|---|---|---|
| `use_long` / `use_short` | `true` / `true` | 方向开关 |
| `long_length` / `long_detection` | `5` / `3` | 做多组三重 EMA 与 pivot 检测 |
| `short_length` / `short_detection` | `5` / `3` | 做空组（独立） |
| `fixed_lot` | `false` | 否 → 按权益百分比动态算 qty |
| `lot_long` / `lot_short` | `100` / `100` | 权益百分比，或固定数量 |
| `timestamp_start` / `timestamp_end` | 2014-01-01 / 2050-12-30 | 回测时间窗（严格开区间） |
| `strict_position` | `false` | 见 2.2，按原脚本语义对最终状态无影响 |

参数经 `rules.resolve_params` 校验（未知 key 抛错，`fee_rate`/`slippage` 为全局统一，
不允许策略级覆盖）。

---

## 4. 已知限制与口径偏差

1. **实盘暂不支持**。模拟盘已完整集成（见第 6 节）；实盘需要交易所侧止损挂单，
   在落地前 `dispatch` 层直接拒绝该策略的实盘开仓并记 warning 日志。
2. **不支持通用验证集选参**（`optimize=true` 直接抛错）、不支持通用风控过滤
   （`min_score` 分层、ATR 移动止损、保本止损对反手型语义无意义，已显式关闭）。
3. **指标从本地数据窗口首根开始计算**，没有更早历史预热；与 TradingView 对照时必须
   统一行情源、历史起点与策略属性。运行时会给出该警告。
4. **无 Bar Magnifier**（TradingView 默认），intrabar 只用 OHLC 四点推断路径；
   存在数据缺口时只提示、不臆造缺口内价格。
5. **滑点口径不同**：本地按成交价百分比，TradingView 的 `slippage` 属性按最小跳动
   点数；设置非零滑点时运行时提示。
6. `strict_position` 按 Pine 缩进语义对最终 `virtual_pos` 无影响（见 2.2）。
7. 反手策略几乎永远在场，`exposure_pct` 可能因「每笔至少 1 根 bar」的统计口径
   略超 100%（主用例 101.7%）。
8. 回测数值只用于**验证移植链路**，不是对该策略盈利能力的评价：样本内、单币、
   对 0.1%/边费用高度敏感（主用例手续费约吃掉初始权益的 16.9%）。

---

## 5. 真实数据回测验证

复现命令：

```
.\.venv\Scripts\python.exe backend/scripts/zigzag_backtest.py --symbol BTCUSDT --interval 1h --days 120 --output docs/simple-zigzag-double.json
```

### 主用例（双向，默认参数）

| 指标 | 值 |
|---|---|
| 区间 | 2026-07-01 16:00 ~ 2026-10-29 15:00 UTC |
| K 线 / 缺口 | 2880 根 / 0 处 |
| 挂单信号 bar | 2870 |
| 成交 | 88 笔，全部 `signal_reverse` |
| 净收益 / 最大回撤 | **-12.8906% / -24.4169%** |
| 买入持有 | +35.9726% |
| 胜率 / 盈亏因子 | 37.5% / 0.788 |
| 手续费 | 169238.97（初始权益 1,000,000） |
| 持仓占比 | 101.7361% |

### 单向对照（`{"use_short": false}`）

53 笔成交，全部 `stop_loss` 出场，净收益 +9.8823%。这验证了
`strategy.exit("Exit Long", "Long", stop=LEVEL_SHORT2, when=needshort==false)`
这条默认关闭、仅在关闭做空时生效的止损挂单分支确实被正确回放。

---

## 6. 模拟盘集成（2026-10-30）

策略已从「仅回测」升级为模拟盘全链路可用：启用开关、策略 × 池绑定、信号预演、
巡检开仓/持仓管理全部打通（API 移出 `BROKER_ONLY_CODES`，前端解除过滤）。

### 信号面：把挂单时序翻译成巡检信号

巡检按周期轮询、只能市价开仓，因此 `signals()` 在整帧上复刻回测撮合的挂单时序：

```
fill_long[T]  = zz_long_entry[T-1]  且 bar T 触及 long_level[T-1]
              （open/high 触价，或 T-1 收盘已越价 → marketable 按开盘成交）
fill_short[T] = zz_short_entry[T-1] 且 bar T 触及 short_level[T-1]
同 bar 双触价 → 按开盘价与两档的距离近似触达先后（与回测路径假设同源）
```

信号字段：LONG 的止损 = 当前 `zz_short_level`（`sl_pct = 1 − short_level/close`），
**无止盈（tp=0，反向通道就是退出）**，仓位 = `lot/100` 再经系统风控截断，杠杆 1。

### 执行链的三处适配

1. **无止盈开仓**：`paper.open_position` 对 `tp=0` 跳过止盈方向闸门（`stop_exit`
   本来就只在 `target>0` 时检查，天然兼容）。
2. **止损跟随**：`manage_order` 对带 `stop_follow` 标签的策略，逐周期把
   `order.sl` 跟随到最新反向通道。pivot 只单向松动（多头止损只下调、空头只上调），
   与 Pine 持续挂单的更新方向一致；保本/移动止损等通用增强对该类策略关闭。
3. **同周期反手**：反向触价在同一根 K 线上同时平旧仓开新仓。若按「本周期刚平仓」
   拦截，新仓会因后续周期触价条件消失而永久漏开——`reverse_on_stop` 标签的策略
   在 `signal_reverse`/`stop_loss` 平仓后放行同周期反向开仓。

### 真实数据冒烟（只读）

- BTCUSDT 1h 预演：FLAT（`awaiting_stop_fill`），通道 80345.0 / 79878.5，
  virtual_pos=1，reason 完整。
- 近 600 根 1h：39 次触价信号（21 多 / 18 空），与回测反手频率（88 笔/2880 根）一致。

### 模拟盘与 Pine 的已知偏差（有意为之）

- **成交价**：Pine 在触价 bar 内按挂单价成交；模拟盘在触价 bar 收盘后的巡检周期
  以市价进场（含滑点），入场/反手价最多偏差一个周期。
- **挂单生命周期**：触价 bar 之后的后续周期若再无触价，Pine 的持续挂单继续等待，
  模拟盘同样等待（信号只在触价 bar 发出），语义一致；但市价入场被系统闸门
  （熔断、资金不足、margin<10）拒绝时信号即丢失，Pine 无此约束。
- **仓位**：Pine 默认 100% 权益无杠杆；模拟盘经 `risk_per_trade_pct` 风险预算截断，
  反向通道很宽时仓位会显著缩小。

---

## 7. 测试与复现

- ZigZag 专项：**21 项**通过（指标/pivot、虚拟仓位、时间窗、参数校验、OHLC 路径、
  gap、marketable stop、bracket 同 bar 成交、time_end、LOT 重算、runner 分派、
  触价信号时序、同 bar 双触价、无效反向通道、describe）；执行链另有 3 项新用例
  （无止盈开仓、止损跟随、同周期反手）。
- 后端全量：**454 项**通过；前端：**69 项**通过；`npm run build` 成功。

```
# ZigZag 专项
.\.venv\Scripts\python.exe -m pytest backend/tests/test_simple_zigzag_double.py backend/tests/test_zigzag_execution.py backend/tests/test_zigzag_runner.py -q
# 全量
.\.venv\Scripts\python.exe -m pytest backend -q
cd frontend && npm test && npm run build
# 换币 / 换周期 / 关方向
.\.venv\Scripts\python.exe backend/scripts/zigzag_backtest.py --symbol ETHUSDT --interval 15m --days 180
.\.venv\Scripts\python.exe backend/scripts/zigzag_backtest.py --params "{\"use_short\": false}"
```

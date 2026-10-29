# Opening Candle Breakout [OCB] —— TradingView 脚本本地移植说明

源脚本：`Opening Candle Breakout [OCB] Strategy` © Swissy 2026（MPL-2.0），
策略思路来自 Zarattini & Aziz (2023)《Can Day Trading Really Be Profitable?》。

本文只讲「脚本怎么变成本地策略」；样本外的历史检验见
[`opening-candle-breakout-strategy.md`](opening-candle-breakout-strategy.md)（15m，180 天，16 币）
和 [`opening-candle-breakout-5m.md`](opening-candle-breakout-5m.md)（脚本原文会话，5m，60 天）。

## 一、脚本在做什么

一天只在**开盘第一根 K 线**上下一次注：

| 环节 | 脚本原文 | 本地实现位置 |
| --- | --- | --- |
| 入场窗 | `not na(time(tf, "0930-0935", tz))` | `features/session.py::session_flags` |
| 方向 | `close > open` / `close < open`（**严格**大小比较，十字星没方向） | `specs/opening_candle_breakout.py::prepare` |
| 实体过滤 | `abs(close-open) >= atr * minBodyATR`（0 = 不过滤） | 同上，`min_body_atr` |
| 风险距离 | `rawRisk = close-low`（多）/ `high-close`（空）<br>`risk = min(max(rawRisk, atr*0.1), atr*2.0)` | 同上，`ocb_risk` |
| 止损/目标 | `close ∓ risk`、`close ± risk*tpR`（在**信号那根的收盘**锁定） | 同上，`ocb_stop` / `ocb_target` |
| 强平 | 15:55-16:00 窗内 `strategy.close_all()` | `backtest/ocb_execution.py` |
| 每天一笔 | `tradedToday`，`newDay` 用 `dayofmonth(tz)` 比较复位 | 同上 |
| 仓位 | `budget / (risk * pointvalue)`，不足 1 张按 `allowMin` 处理 | 同上，`risk_mode` / `risk_usd` / `risk_per_trade_pct` |
| ATR | `ta.atr(14)` = Wilder RMA（**前 14 根 SMA 播种**） | `specs/.../wilder_atr` + `rma` |

## 二、订单/撮合序列（严格按 Pine 默认语义）

`calc_on_every_tick=false`、`process_orders_on_close=false`，所以：

1. **本根收盘**才下单（映射 = 信号列的判定都发生在收盘那根）；
2. 订单在**下一根开盘**成交（`fill at open`，并叠加本地滑点）；同一根里脚本写在前面的
   `close_all` 先成交，之后才轮到新开仓单；
3. 括号止损/止盈与入场一起挂出，新仓位在**它自己入场那一根**就可能被打掉；
4. 盘中路径按 TradingView 默认 OHLC 插值：`open → 离开盘更近的极值 → 另一个极值`
   （平票时先高后低，TradingView 未定义顺序）。
5. 收盘那一段严格按脚本顺序：`newDay` 复位 → 强平窗提交 `close_all` → 才轮到入场判定。

新增文件：

| 文件 | 职责 |
| --- | --- |
| `backend/app/features/session.py` | 时区换算、`time(tf, session, tz)` 的本地等价物 |
| `backend/app/strategy/specs/opening_candle_breakout.py` | 判定与中间量（因果） |
| `backend/app/backtest/ocb_execution.py` | 专用撮合（会话强平 / 风险定手数 / 每日一笔） |
| `backend/app/backtest/runner.py::run_ocb_backtest` | 回测入口自动路由 |
| `backend/scripts/ocb_backtest.py` | 批量历史检验 + 随机方向基线 |
| `backend/tests/test_opening_candle_breakout.py` | 会话/判定/撮合/路由四层 28 项测试 |

## 三、参数映射（脚本 input → 本地 key）

| 脚本 input | 本地 key | 默认 | 说明 |
| --- | --- | --- | --- |
| `sessFirst` 0930-0935 | `first_session_start_min` / `first_session_end_min` | 570 / 575 | 会话时区墙上时间的日內分钟数 |
| `sessFlat` 1555-1600 | `flat_session_start_min` / `flat_session_end_min` | 955 / 960 | 同上 |
| `tz` America/New_York | `session_tz` | 1 | 0=UTC, 1=America/New_York, 2=Europe/London, 3=Asia/Shanghai, 4=Asia/Tokyo |
| — | `tz_offset_minutes` | -300 | tzdata 缺失时的固定偏移兜底 |
| `Longs` / `Shorts` | `trade_longs` / `trade_shorts` | true | |
| `Minimum candle body` | `min_body_atr` | 0.5 | 0 = 不过滤 |
| `ATR period` | `atr_len` | 14 | |
| `Minimum stop` / `Maximum stop` | `min_stop_atr` / `max_stop_atr` | 0.10 / 2.0 | 双向夹逼 |
| `Target (R)` | `target_r` | 10.0 | |
| `Use target` | `use_target` | true | 关掉 = 只靠止损与强平 |
| `Risk mode` | `risk_mode` | 1（权益百分比） | 0 = 固定金额 `risk_usd`（1:1 复刻脚本默认 250 USD） |
| `Risk per trade` | `risk_usd` / `risk_per_trade_pct` | 250 / 0.005 | |
| `Max contracts` | `max_position_pct` | 0.25 | 见下方取舍 |
| `Allow 1 contract…` | `allow_min_position` | false | |
| — | `tp_gap_at_open` | false | 开盘跳空越过止盈价时的成交价约定 |

## 四、本地取舍（都摆在明面上）

1. **合约张数 → 碎股数量**：脚本用 `floor(budget / (risk*pointvalue))` 取整 + `maxQty` 封顶；
   加密货币可以碎股交易，所以数量保留小数并按名义上限收敛。
   `max_position_pct`（名义 ≤ 权益的这一比例）取代 `maxQty`；
   `MIN_MARGIN = 10 USDT` 取代「不足 1 张」，`allow_min_position` 对应 `allowMin`。
2. **成本模型**：脚本用 `commission_type=cash_per_contract`、`slippage=2 ticks`；
   本地统一走双向 `fee_rate` / `slippage` 比例成本（与其它策略同一口径，便于横向比较）。
3. **止盈跳空**：开盘就跳到止盈价之上时，默认按**限价**成交（保守）；
   `tp_gap_at_open=true` 切到 TradingView 的「限价单以更优价成交」口径（取开盘价）。
4. **时区以编号选择**：策略参数只接受数值/布尔（`rules.resolve_params` 的约束），
   所以时区做成 `session_tz` 枚举，`tz_choices()` 暴露候选表。
5. **不参与通用撮合**：`signals()` 输出的 `direction` 恒为 0 —— 会话强平、按风险距离定手数、
   每日一笔这三条通用撮合一条都表达不了，让它偷偷跑通用撮合等于换了一套出场规则。
   回测由 `runner.run_backtest` 自动路由到专用撮合；API 层也拒绝启用
   （实盘/模拟盘的持仓管理目前只有「按 bar 计时的时间止损」，没有会话强平）。

## 五、会话窗必须随图表周期选

会话判定的是**K 线开盘时间**是否落在 `[start, end)`（右端开区间），所以：

| 周期 | 首窗 570-575 | 强平窗（脚本原文 955-960） | 建议强平窗 |
| --- | --- | --- | --- |
| 1m / 5m | 命中 09:30 一根 | 命中 15:55 一根 | 955-960（脚本原文） |
| 15m | 命中 09:30 一根 | **一根都不命中**（15m 的开盘是 15:45 / 16:00） | 945-960 |
| 1h | **一根都不命中**（1h 的开盘是 09:00） | — | 换周期 |

15m 与 5m 两种配置都跑过，报告分别在上面两个链接里。

## 六、怎么用

```bash
# 15m，180 天，主流币，强平窗按 15m 对齐
.\.venv\Scripts\python.exe backend/scripts/ocb_backtest.py

# 脚本原文会话（5m），只跑有 5m 历史的币
.\.venv\Scripts\python.exe backend/scripts/ocb_backtest.py --interval 5m --days 60 \
    --symbols BTCUSDT,ETHUSDT --flat-start 955 --flat-end 960 --out docs/opening-candle-breakout-5m

# 换参数（例如把目标从 10R 降到 4R）
.\.venv\Scripts\python.exe backend/scripts/ocb_backtest.py --params '{"target_r": 4}'
```

单币单次回放走 API/回测入口即可（会自动识别策略并路由到专用撮合）：

```python
from app.backtest.runner import run_backtest
run_backtest(db, "BTCUSDT", "15m", "opening_candle_breakout", days=180,
             param_overrides={"flat_session_start_min": 945})
```

## 七、验证结论（摘要）

15m / 180 天 / 16 个主流币，合计 **1445 笔**：

- 每笔收益均值 **-0.1592%**，池化 95% CI **[-0.225%, -0.0934%]** → 上界 < 0，**净负贡献**；
- 出场结构：`stop_loss` 1061 / `session_close` 361 / `take_profit` 23 —— 10R 目标几乎打不到，
  73% 的仓位是先被止损扫出局的；
- 随机方向基线（同时点同成本翻方向，8 种子）账户收益均值 -10.37%，
  本策略 14/16 币跑赢它，但跑赢随机 ≠ 有正 edge，绝对值仍是负的。

处置：**默认不启用**（`StrategySpec.default_enabled=False`），也不在这个结构上继续调参数。
如果你的标的物的开盘时段波动结构与加密货币显著不同（比如真美股），
这套 port 可以直接换会话和时间区后重跑，判据不用改。

# Martin Luk 强势回调（本地适配）—— 实现与首轮真实历史读数

> 日期：2026-10-27 ｜ 策略代码 `strong_pullback` ｜ 基线默认 `15m` ｜ 只做多、1 倍杠杆
> 方法纪律：先落地可复算的因果实现 → 再接共享回测撮合 → 最后用真实历史读数说话，不把「框架跑通」当成「有 edge」

## 1. 规格到实现的映射

| 规格 | 落地位置 | 口径 |
|---|---|---|
| 4h 强势趋势 | `features/mtf.py::add_mtf_trend` + `features/pullback.py::add_pullback_context` | 4h `ema21>ema50` 且 4h `ADX_14 >= htf_adx_filter(22)`；4h 指标已整体 shift(1) 一个桶 |
| 15m 首次回调 EMA/AVWAP | `features/pullback.py` | 强势段内走满 `pullback_min_bars(8)` 根后，首次「由上方跌破 EMA9 或锚定 VWAP」；与上次触及相隔 ≥8 根才算新回调 |
| 转强确认 | 同上 | 回调后 `pullback_confirm_bars(8)` 根内出现「收阳 + 收盘高于前收 + 收复 EMA9 + 站上 AVWAP」，每次回调只确认一次 |
| 只做多、1 倍杠杆 | `strategy/rules.py::STRATEGY_PARAMS["strong_pullback"]` | `max_leverage=1`；`side` 只可能 0/+1，全帧无 −1（测试覆盖） |
| 单笔风险 0.5% | 同上 | `risk_per_trade_pct=0.005`，经 `risk_pct = 仓位×杠杆×(sl+成本)` 上限约束 |
| 名义仓位上限 35% | 同上 | `max_position_pct=0.35`，1 倍杠杆下仓位比例即名义比例 |
| EMA9 失守退出 | `backtest/execution.py::simulate`（`ema_exit`）+ `trading/order_manager.py` | 上一根收盘 < EMA9 → 本根开盘平多，reason=`ema_exit` |
| 硬止损 | `rules.py` 结构止损 | 止损价 = 回调段最低价 × (1 − `stop_buffer_pct`) 与 `sl_atr_mult×ATR` 取更宽者，下限 `min_sl_pct` |
| 10R 目标 | `rules.py` | `tp_pct = target_r(10) × sl_pct`，经 `min_tp_pct` 兜底 |
| 最长 48 小时 | `rules.py::resolve_params(interval=)` | `max_hold_bars = round(48h / 周期)`：15m→192、1h→48、4h→12、1d→2 |
| 策略管理 | `strategy/defaults.py` + `api/strategies.py` | 播种进 `StrategyConfig`，`effective_params` 直接返回合并后的参数 |
| 共享回测引擎 | `backtest/runner.py` | 与其余策略共用 `evaluate_signals` + `simulate` + 指标/基线/显著性口径 |

## 2. 实现清单（勿回退）

- `backend/app/features/pullback.py`（新增）：`add_pullback_context(frame, params)` 输出 13 列
  `pb_dir_4h / pb_adx_4h / pb_trend_ok / ema9 / avwap / avwap_dist / pb_touch / pb_first_touch /
  pb_bars_since_touch / pb_confirm / pb_depth / pb_stop_ref / pb_exit_long`。
- `backend/app/strategy/rules.py`：`STRATEGY_PARAMS["strong_pullback"]`；`RULE_DEFAULTS["ema_exit"]=False`
  作为全策略默认（只有本策略打开）；`resolve_params(..., interval=)` 做 48h→bars 换算；
  `evaluate_signals` 新增分支与 4 个专属 block reason；输出新增 `exit_long` 列（其他策略恒 0）。
- `backend/app/backtest/execution.py`：`simulate` 支持 `params["ema_exit"] + signals.exit_long` 软出场。
- `backend/app/backtest/runner.py`：`resolve_params(..., interval=interval)`；寻优网格新增
  `htf_strict` / `confirm_strict` 两个变体（仅当 params 含 `target_r` 时启用，不影响既有策略）。
- `backend/app/strategy/engine.py`：`PULLBACK_LOOKBACK=1200`（15m 下 300 小时 ≈ 75 个 4h 桶，
  否则 400 根只有 ~25 桶，4h ADX 无法预热）；整帧富化后只把末根交给规则层；`reason` 附带
  `pb_htf_dir / pb_htf_adx / ema9 / avwap / avwap_dist / bars_since_touch / pullback_depth / structure_stop`。
- `backend/app/trading/order_manager.py`：持仓回放同样按「上一根收盘失守 EMA9 → 本根开盘平仓」，
  与回测同口径（paper 与 live 都生效；live 的固定止损仍由交易所侧保护单负责）。
- `backend/app/schemas.py` / `api/settings.py` / 前端 `Backtest.vue`、`utils/format.js`、`utils/strategy.js`：
  回测策略枚举、设置校验清单、下拉项与中文文案（含新 block reason 与 `ema_exit` reason）。

## 3. 反前视设计（硬红线）

四条都只用 ≤ 当前 bar 的信息：

1. 4h 方向/ADX：桶指标整体 `shift(1)` 后才 reindex 回低周期（沿用 `add_mtf_trend` 既有语义）。
2. AVWAP：锚点 = 强势段起点（`trend_ok` False→True 的 bar），段内 `groupby.cumsum()` 累计。
3. 首次回调：`position.where(touch).groupby(segment).ffill().shift(1)` 得到「上一根为止最近一次触及位置」，
   与新位置相减判间隔；**没有用任何未来位置**。
4. 确认去重 / 止损参照：`groupby(episode).cummax()`、`groupby(episode).cummin()`，
   episode 由 `first_touch.cumsum()` 定义，均为因果累计。

验证方式（测试内固化）：把输入帧截断到确认 bar 后重算，`avwap / ema9 / pb_stop_ref / pb_depth /
pb_bars_since_touch / pb_confirm` 必须逐位一致——不一致即视为泄漏。

> 踩坑记录：`groupby.cummax()` + `shift()` 会把 dtype 升为 `object`，此时 `~series` 是位运算
> （`~True == -2`，恒真），去重会静默失效。必须 `.astype(bool)` 再取反。

## 4. 真实历史读数（首次评估，结论：无 edge 证据，样本不足）

原始数据见 `docs/strong-pullback-backtest.json`。统一口径：`algo=rule`、`days=120`、`initial=10000`、
训练/测试 75/25（测试段为最后 30 天）。

### 4.1 15m 窗口 A（截至今日，测试段为下跌行情）

| 币 | 信号 | 成交 | 胜 | 净收益 | 最大回撤 | 同期买入持有 | 出场分布 |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 10 | 8 | 2 | −0.39% | −0.40% | −5.10% | ema_exit 6 / stop_loss 2 |
| ETHUSDT | 3 | 2 | 1 | −0.01% | −0.16% | −13.87% | ema_exit 1 / stop_loss 1 |
| BNBUSDT | 2 | 2 | 0 | −0.20% | −0.20% | −5.55% | ema_exit 1 / stop_loss 1 |
| XRPUSDT | 0 | 0 | 0 | 0.00% | 0.00% | +0.75% | — |
| DOGEUSDT | 0 | 0 | 0 | 0.00% | 0.00% | −16.87% | — |
| ADAUSDT | 4 | 3 | 2 | +0.41% | −0.35% | −16.11% | ema_exit 2 / stop_loss 1 |

结论：下跌段基本空仓（DOGE 买入持有 −16.9% 时策略 0 成交），方向过滤按设计工作。

### 4.2 15m 窗口 B（截至 60 天前，测试段为上涨行情）

| 币 | 信号 | 成交 | 胜 | 净收益 | 最大回撤 | 同期买入持有 |
|---|---|---|---|---|---|---|
| BTCUSDT | 21 | 19 | 4 | +0.97% | −0.75% | +25.45% |
| ETHUSDT | 21 | 18 | 7 | +0.18% | −1.05% | +31.14% |
| BNBUSDT | 20 | 16 | 4 | −0.96% | −1.38% | +10.46% |
| DOGEUSDT | 11 | 9 | 3 | −0.09% | — | +13.47% |
| ADAUSDT | 10 | 9 | 3 | −0.00% | −0.89% | +8.56% |

### 4.3 1h 深历史对照（days=300，测试段 ~1787 根）

BTC +0.32%（15 笔，含 1 笔 take_profit）/ ETH −0.53% / BNB +0.13% / ADA −1.56% / DOGE +2.48%。

### 4.4 关键发现（首轮评估，样本不足）

1. **10R 目标几乎不可达**：15m 两个窗口共约 60 笔成交里 `take_profit` 为 0，1h/300 天里也只出现 1 次。
   出场几乎全部由 `ema_exit`（约 70%）与 `stop_loss`（约 30%）完成——策略的赔率假设（靠少数 10R
   交易撑住整体）在实测中没有被实现，胜率只有 20%~40%，于是期望值贴着 0 或略负。
2. **方向过滤是对的，收益贡献是零**：下跌窗口几乎不开仓（避免 −16% 的买入持有），上涨窗口也只吃到
   +0.97%（买入持有 +25%），说明"躲跌"成立、"跟涨"不足。
3. **15m 历史宽度是硬约束**：首轮评估时 521 币里只有 BTC/ETH 有 ≈200 天 15m。
   现已补到 Top 60 币 ≈300 天（回填补跑 21+39=60/60 全 ok），这是 walk-forward 的基础。
4. 每笔置信区间在全部标的上跨 0（样本 2~19 笔），**不能据此判断正/负 edge**——只能确认框架与撮合按设计工作。

### 4.5 出场重构 A/B 验证（2026-10-27 终判）

补完 15m 历史后，以 36 个 eligible 币（覆盖 ≥205 天、lag ≤7d）、4 个互不重叠的 120 天窗口（终点
2026-10-27 / 09-27 / 08-28 / 07-29）、同一撮合口径（fee=0.0004、slippage=0.0002、optimize=True）
对照四个变体。原始输出见 `docs/strong-pullback-exit-ab.json`。

| 变体 | 描述 | 持仓数 | 均值收益率 | 持仓级 bootstrap CI | verdict |
|---|---|---|---|---|---|
| **partial** | 默认参数，`partial_tp_r=2.0` | 2155 | −0.40% | [−0.16%, −0.03%] | **negative** |
| **fixed** | `partial_tp_r=0.0`（原版 10R 固定目标） | 2156 | −0.40% | [−0.17%, −0.02%] | **negative** |
| trend_follow | 同窗口控制组 | 12163 | −4.65% | [−0.34%, −0.25%] | **negative** |
| mtf_resonance | 同窗口控制组 | 1966 | −0.42% | [−0.28%, −0.02%] | **negative** |

**Gate 判定（未通过）：**

| 条件 | 要求 | 结果 |
|---|---|---|
| 池化持仓数 ≥ 30 | 有足够样本 | ✅ 2155+ / 2156+ / 1966+ |
| 持仓级 CI 下界 > 0 | 期望值显著为正 | ❌ 四个变体下界全负（partial −0.16、fixed −0.17） |
| partial 相对 fixed 同窗口差中位数 > 0 | 出场重构改善 | ❌ 中位数 = 0.0（144 对逐窗口差，34% 窗口 partial 更好、66% fixed 更好） |

**出场分布：**

| 变体 | ema_exit | partial_tp / take_profit | stop_loss | close_at_end |
|---|---|---|---|---|
| partial | 1547 (71.8%) | 206 (9.6%) partial_tp | 606 (28.1%) | 2 |
| fixed   | 1550 (71.9%) |   2 (0.1%) take_profit | 602 (27.9%) | 2 |

**解读：**

1. **出场结构重构本身是成功的** — fixed 的 10R take_profit 只触发 2 笔（/2156），证明确实几乎不可达；
   partial_tp 的触发率提升到 9.6%（206/2155），2R 减半让部分仓位能落袋。但这些改善**并未转化为期望值提升**。
2. **partial vs fixed 统计等价** — 逐窗口差均值 = −0.0067%、中位数 = 0.0，34% 窗口 partial 占优，
   差异不显著。EMA9 软出场仍是绝对主导（72%），两种出场结构的底层假设（等价格到盈利区间，然后退出）
   在这个入场过滤器下都没有 alpha。
3. **根因是入场，不是出场** — 四个变体 CI 全负，包括完全不同入场逻辑的 trend_follow 和 mtf_resonance。
   这指向共享的成本因子（fee + slippage），或更根本的：36 个活跃合约币里，简单的"4h 强势 + 15m 回调确认"
   入场逻辑本身没有正期望值。
4. **不保留 partial_tp_r=2.0 默认** — 已回退 strong_pullback 默认 `partial_tp_r=0.0`（原版规格），
   功能代码保留供将来入场优化后重试（rules.py / defaults.py / 测试已同步）。

## 5. 下一步（有优先级的待办）

1. **入场优化优先于出场调参** — 出场结构已证实在当前入场下不影响 edge；资源应转向：
   - 放宽 `htf_adx_filter` / 收紧 `pullback_confirm_bars` 不一定能改善（首轮样本太小），需在 36 币×4 窗口
     上做 walk-forward 网格搜索，而非凭直觉改单参数。
   - 或换思路：直接放弃 strong_pullback 入场，转向 funding 异常 / 成交量收缩 / 波动率爆发 等更有
     独立 alpha 的触发条件（已在 P11/P12 上做了探索）。
2. **P11 跨所决策（阻塞项）** — 单所 maker fee 返还（0.02%）后的 funding 期望收益 +1.43%/年 < 3% gate，
   跨所 funding 价差现货腿才是出路。**需要你决策是否开 Gate / MEXC 账户做跨所套利腿**。
3. **不要**在出场结构上继续投入时间 — partial_tp_r 功能代码保留但默认关闭，将来入场有 edge 时再调。

## 6. 测试覆盖（`backend/tests/test_strong_pullback.py`，31 项全绿）

- 参数/注册（6）：默认值（1 倍、35%、0.5%、10R、48h）、48h→bars 四周期换算、越界与未知参数拒绝、
  播种幂等、`list_strategies` 的 `effective_params`、回测请求枚举接受/拒绝。
- 特征层（10）：列齐全、AVWAP 等于段内累计 VWAP、回调前必须走满 runup、单次回调只记一次首次触及、
  两次分离回调各自计数、无收复则无确认、确认窗口收紧即失效、确认只出现一次、**截断重算一致性（反前视）**、
  `pb_exit_long` 与 close/EMA9 关系。
- 信号层（10）：确认入场、4h 趋势不足/无回调/无确认三种拦截、全帧无空头、仓位≤35% 且杠杆=1 且风险≤0.5%、
  `tp = 10×sl`、止损在回调低点下方、未富化帧安全降级。
- 撮合层（3）：共享回测按 `ema_exit` 平多、其他策略忽略 `exit_long`、持仓管理回放 `ema_exit`。
- 全仓回归：387（既有）+ 31（新增）= 418 全绿。

# P9 正交数据源（OI/多空比/taker）—— 第一个过 gate 的跨币 alpha

> 日期：2026-10-27 ｜ 依赖：P0–P8 ｜ 方法纪律：IC 探针 gate → 稳定性复测 → （数据积累后）PnL 级 A/B

## 1. 动机

P7 收敛结论：现货 OHLCV + 资金费率（线性特征 P6 / 硬过滤 P7）均无可稳健复制的方向性 edge。破局只剩两条路：① 扩大历史宽度（P8）② 接入真正正交的数据源（P9）。

P9 选定**衍生品情绪/杠杆数据**（币安 `/futures/data/*` 公开免 Key）：
- `open_interest_history`：未平仓合约量 + 名义价值（杠杆总量）
- `globalLongShortAccountRatio`：全市场多空账户比（散户情绪）
- `topLongShortPositionRatio`：大户持仓多空比（聪明钱）
- `takerlongshortRatio`：taker 主动买卖量比（订单流方向）

**API 硬限制**：这些端点仅保留最近 ~30 天 1h 历史（500 条上限 ≈ 21 天）。因此 P9 定位为**横截面 IC 探针**（514 币 × ~450 bar ≈ 23 万样本，与 P3/P6 同方法论），不做 walk-forward。

## 2. 实现（勿回退）

| 组件 | 文件 | 说明 |
|---|---|---|
| 数据表 | `app/models.py::Sentiment` | (symbol, ts, open_interest, oi_value, long_short_ratio, top_long_short, taker_buy_sell)，create_all 幂等建表 |
| 采集 | `app/market/data_service.py::sync_sentiment_many` | 每 symbol 4 请求，端点按 timestamp 外对齐合并，幂等 upsert |
| 读取 | `load_sentiment_frame` | UTC DatetimeIndex，5 列原值 |
| 特征 | `app/features/sentiment.py` | `attach_sentiment`（asof 合并，无前视）+ `SENTIMENT_FEATURES` 6 列；无数据时全 0 降级 |
| 接线 | `indicators.py` / `dataset.py` | `build_features(..., sentiment=)`；`FEATURES` 31→**37**；`build_dataset` 自动加载透传 |
| 回填 | `scripts/backfill_sentiment.py` | 全宇宙 **515 币 / 257,558 行**（2026-10-06~10-27，0 错误）|
| 探针 | `scripts/probe_sentiment.py` | 跨币 IC + 7 天 z-score 派生 + 分半窗一致性 |

## 3. 探针结果（514 币 × ~21 天 1h，triple-barrier 止盈标签）

| 特征 | IC 均值 | 中位数 | 同号% | \|IC\|>0.05 币占比 |
|---|---|---|---|---|
| **oi_value_usd** | **−0.0746** | −0.0719 | **84%** | **67%** |
| oi_value_z7d（去水平）| −0.0403 | −0.0409 | 72% | 56% |
| oi_change_24h | −0.029 | −0.0322 | 64% | 61% |
| long_short_ratio | −0.025 | −0.0211 | 59% | 60% |
| top_long_short | −0.0142 | −0.0198 | 57% | 64% |
| taker_buy_sell | +0.0051 | +0.0083 | 56% | 37% |

## 4. 稳定性复测（oi_value_usd）

- **分半窗一致性**：前 10 天 IC 均值 **−0.1213** / 后 10 天 **−0.1206**（几乎相同）；**75% 币种两半皆为负**，77% 两半同号 → 不是单一 regime 偶然。
- **去水平效应**：7 天 z-score 版 IC −0.0403、72% 同号 → 信号减半但方向稳健，说明**不是纯"币价水平共线"**，"OI 相对近期高位"本身独立携带负向信息。
- 机制可解释：杠杆持仓处于高位 → 多头拥挤 → 上行阻力大、易踩踏回调 → 做多止盈概率下降。

**对照 P6 funding_cum（+0.025 / 58% 同号 / 证伪）**：OI 信号强度、同号一致性、分半窗稳定性全面高一个量级 → **通过 P9 gate**。

## 5. 边界与下一步（关键）

- **21 天数据不足以做 PnL 级 A/B**。P6 的教训：短窗口乐观结论会被推翻。tier_backtest 最小窗口 45 天，现在连一个窗口都铺不满。
- `oi_value_usd` 已作为特征进入 `FEATURES`（LightGBM 自动消费权重）；z7d 版本暂只在探针中。
- **下一步（数据积累路径）**：
  1. **每日增量采集** sentiment（脚本幂等，跑一次即滚动前进；可挂每日自动化）；
  2. 积累至 **60–90 天** 后跑受控 A/B（`--exclude-sentiment` 式开关待加）：比较 OI 特征 ON/OFF 的组合收益/回撤；
  3. 若 A/B 确认增益 → 再考虑逆向拥挤过滤（P7 教训：过滤触发率与一致性要先验证）。

## 6. P8 同步产出：历史宽度回填

- `scripts/backfill_native_1h.py`：绕过 `store` 的 1m 限制直接 upsert `interval='1h'`（与 native 回落读取同源 fapi）；从现存最早行往前补到 `--days`（默认 400），跳过已覆盖区间。
- 回填前深度分布（521 币）：**461 币仅 30–60 天**（1001 行 ≈ 初始 recent 同步尾量）、≥300 天仅 10 币。
- 样本验证：1INCHUSDT +9546 行（42 天→400+ 天）；全量回填见 `docs/backfill-native-1h-report.json`。

## 7. 交付物

- `docs/backfill-sentiment.json` / `docs/probe-sentiment.json` — 采集与 IC 数据
- `docs/backfill-native-1h-report.json` — P8 回填报告
- 本文件 — P9 完整报告
- 回归：**387 passed**

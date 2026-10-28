# 吸收反转（Absorption Reversal）— 主流币回测报告

生成时间：2026-10-28T16:05:56+00:00　周期：15m　窗口：1×93 天（步长 93 天）

## 结论速览

- 有交易的币种 16 个，合计成交 **323 笔**
- 单币收益中位数 **-0.50%**，均值 -0.48%
- 正收益窗口占比 0%，各窗口中位数 [-0.5]
- 收益超过随机方向基线（10 种子）的币种占比 12%
- 期望值置信区间可判定的币种 2/16

### 池化判据（323 笔）

- 每笔收益均值 **-0.0794%**，95% CI [-0.2273%, +0.0811%] → **inconclusive**
- 重采样为正的比例 15.7%（>95% 才算统计上站得住）
- 随机方向基线（同参数，均值）：-0.491%

判读规则（与全项目一致）：CI 下界 > 0 才算有正 edge；跨 0 即不可判定；上界 < 0 说明这套规则在当前样本下是负贡献。

## 策略参数（脚本 1:1）

灵敏度 `high`：|delta| 分位 90.0%、实体上限 0.35×ATR、极值回看 10 根、影线比 0.15

| 参数 | 值 | 参数 | 值 |
| --- | --- | --- | --- |
| `delta_window` | 500 | `delta_pct` | 90.0 |
| `body_mult` | 0.35 | `extreme_lookback` | 10 |
| `wick_mult` | 0.15 | `dominance_pct` | 20.0 |
| `abs_gap_bars` | 10 | `conf_mode` | 1 |
| `conf_window_bars` | 3 | `sl_buffer_atr` | 0.25 |
| `target_r` | 2.0 | `be_trigger_r` | 1.0 |
| `min_sl_atr` | 0.3 | `max_sl_atr` | 3.0 |
| `max_hold_bars` | 30 | `base_position_pct` | 0.3 |
| `max_leverage` | 1 | `risk_per_trade_pct` | 0.02 |

## 逐窗口汇总

| 窗口结束 | 有交易币 | 成交 | 收益中位数 | 收益均值 | 胜率均值 | 超随机 95 分位 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-28 | 16/16 | 323 | -0.50% | -0.48% | 33.5% | 2 |

### 出场原因分布（池化）

| 出场 | 笔数 | 占比 |
| --- | --- | --- |
| stop_loss | 189 | 59% |
| time_stop | 66 | 20% |
| take_profit | 64 | 20% |
| signal_reverse | 4 | 1% |

### 信号被拦截原因（池化）

| 原因 | 根数 |
| --- | --- |
| no_absorption_signal | 133624 |
| stop_too_wide | 22 |

## 逐币种（有交易）

| 币种 | 成交 | 收益 | 胜率 | 随机基线均值 | 期望值 CI | delta 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| NEARUSDT | 22 | +5.15% | 40.9% | -2.22% | inconclusive | lower |
| SUIUSDT | 4 | +0.95% | 75.0% | +0.47% | inconclusive | lower |
| DOGEUSDT | 27 | +0.72% | 44.4% | -1.12% | inconclusive | lower |
| SOLUSDT | 5 | +0.26% | 40.0% | -0.21% | inconclusive | lower |
| LINKUSDT | 29 | +0.07% | 37.9% | -1.21% | inconclusive | lower |
| TRXUSDT | 1 | -0.05% | 0.0% | -0.04% | negative | lower |
| XRPUSDT | 6 | -0.33% | 33.3% | +0.13% | inconclusive | lower |
| BNBUSDT | 18 | -0.43% | 38.9% | -0.53% | inconclusive | lower |
| BCHUSDT | 22 | -0.57% | 27.3% | +1.26% | inconclusive | lower |
| AVAXUSDT | 28 | -0.66% | 39.3% | -0.73% | inconclusive | lower |
| ADAUSDT | 26 | -0.67% | 34.6% | +0.14% | inconclusive | lower |
| ETHUSDT | 20 | -0.68% | 30.0% | -0.35% | inconclusive | lower |
| BTCUSDT | 19 | -1.93% | 15.8% | -1.00% | negative | lower |
| LTCUSDT | 37 | -2.62% | 21.6% | -1.84% | inconclusive | lower |
| APTUSDT | 31 | -3.17% | 32.3% | +0.73% | inconclusive | lower |
| DOTUSDT | 28 | -3.69% | 25.0% | -1.33% | inconclusive | lower |

## 数据限制（必须先读）

1. **delta 来源**：脚本真正用的是 `request.security_lower_tf` 的 intrabar delta；本项目 1m 历史约 93 天，`--delta-source auto` 在有 1m 数据的区间用真实聚合 delta，其余区间降级为脚本自带的 bar shape 估计（`na(ltfDelta) ? estDelta : ltfDelta`）。两种来源的信号密度差异很大，表中 `delta 来源` 列逐币标注。
2. **信号稀疏**：吸收判定要求「巨量 + 无进展 + 极值 + 长影线 + 单边支配」同时成立，脚本默认参数下每个币每月只有个位数信号，样本量的上限由策略本身决定，不是取数问题。
3. **撮合口径**：信号收盘 → 次根开盘成交（含滑点与费率）；R 倍数目标与保本价以成交价为基准，与脚本 `position_avg_price` 一致。

耗时 18.2s。

# 吸收反转（Absorption Reversal）— 主流币回测报告

生成时间：2026-10-28T16:23:27+00:00　周期：15m　窗口：3×30 天（步长 30 天）

## 结论速览

- 有交易的币种 36 个，合计成交 **88 笔**
- 单币收益中位数 **-0.04%**，均值 +0.05%
- 正收益窗口占比 0%，各窗口中位数 [-0.03, -0.06, -0.04]
- 收益超过随机方向基线（10 种子）的币种占比 8%
- 期望值置信区间可判定的币种 19/36

### 池化判据（88 笔）

- 每笔收益均值 **+0.0704%**，95% CI [-0.2456%, +0.4278%] → **inconclusive**
- 重采样为正的比例 63.4%（>95% 才算统计上站得住）
- 随机方向基线（同参数，均值）：-0.070%

判读规则（与全项目一致）：CI 下界 > 0 才算有正 edge；跨 0 即不可判定；上界 < 0 说明这套规则在当前样本下是负贡献。

## 结论与处置

池化 CI 跨 0 → **不可判定**：既不能说有 edge，也不能说没有。这与本项目的判据一致——样本不足时的默认值是「不启用」，而不是「先跑跑看」。

处置：

1. 策略已完整移植并可随时手动开启（`StrategyConfig.absorption_reversal`，默认 enabled=0），便于对照观察与后续复检。
2. 运行范围钉在主流币（`coin_router.MAJOR_SYMBOLS`），避免量价噪声污染 c1/c5 判定。
3. **不写入** `SYMBOL_ROUTES`：没有 holdout 为正的证据，就不该在自动巡检里参与资金竞争。
4. 真正的样本瓶颈是 1m 历史（当前约 93 天）：脚本的 intrabar delta 需要它。补齐半年以上 1m 历史后重跑本脚本，样本量会提高一个数量级。

> 架构变更（2026-10）：`SYMBOL_ROUTES` / `symbol_allowed` 已随「策略 × 币种池绑定」重构删除，第 1–3 条的现行等价物是——策略 `enabled=0`，且 `absorption_reversal × major × 15m` 绑定默认 `enabled=0`。开启顺序：先开绑定，再开策略总开关。

## 策略参数（脚本 1:1）

灵敏度 `medium`：|delta| 分位 95.0%、实体上限 0.25×ATR、极值回看 20 根、影线比 0.2

| 参数 | 值 | 参数 | 值 |
| --- | --- | --- | --- |
| `delta_window` | 500 | `delta_pct` | 95.0 |
| `body_mult` | 0.25 | `extreme_lookback` | 20 |
| `wick_mult` | 0.2 | `dominance_pct` | 20.0 |
| `abs_gap_bars` | 10 | `conf_mode` | 1 |
| `conf_window_bars` | 3 | `sl_buffer_atr` | 0.25 |
| `target_r` | 2.0 | `be_trigger_r` | 1.0 |
| `min_sl_atr` | 0.3 | `max_sl_atr` | 3.0 |
| `max_hold_bars` | 30 | `base_position_pct` | 0.3 |
| `max_leverage` | 1 | `risk_per_trade_pct` | 0.02 |

## 逐窗口汇总

| 窗口结束 | 有交易币 | 成交 | 收益中位数 | 收益均值 | 胜率均值 | 超随机 95 分位 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-28 | 13/16 | 31 | -0.03% | -0.07% | 40.4% | 1 |
| 2026-09-28 | 11/15 | 23 | -0.06% | +0.11% | 32.1% | 0 |
| 2026-08-29 | 12/15 | 34 | -0.04% | +0.13% | 34.3% | 2 |

### 出场原因分布（池化）

| 出场 | 笔数 | 占比 |
| --- | --- | --- |
| stop_loss | 46 | 52% |
| time_stop | 23 | 26% |
| take_profit | 18 | 20% |
| signal_reverse | 1 | 1% |

### 信号被拦截原因（池化）

| 原因 | 根数 |
| --- | --- |
| no_absorption_signal | 101305 |
| stop_too_wide | 15 |

## 诊断附录（同一时间窗的其它口径）

| 口径 | 成交 | 收益中位数 | 池化每笔期望 | 95% CI | 判定 |
| --- | --- | --- | --- | --- | --- |
| est（15m，delta=est） | 9 | -0.28% | -0.2666% | [-1.0103%, +0.6538%] | inconclusive |
| sens-high（15m，delta=auto） | 323 | -0.50% | -0.0794% | [-0.2273%, +0.0811%] | inconclusive |
| window93（15m，delta=auto） | 106 | +0.28% | +0.2514% | [-0.0672%, +0.5854%] | inconclusive |

放宽容度（脚本 Sensitivity=High）会把信号数放大数倍，但每笔期望转负——这正说明「少而精」不是可以随便放宽的旋钮，样本密度靠放松阈值换不来 edge。

## 逐币种（有交易）

| 币种 | 成交 | 收益 | 胜率 | 随机基线均值 | 期望值 CI | delta 来源 |
| --- | --- | --- | --- | --- | --- | --- |
| APTUSDT | 9 | +1.68% | 60.0% | +0.00% | inconclusive | lower |
| NEARUSDT | 5 | +1.55% | 33.3% | +0.27% | inconclusive | lower |
| SUIUSDT | 3 | +1.29% | 100.0% | +0.20% | positive | lower |
| SOLUSDT | 2 | +0.35% | 50.0% | +0.05% | inconclusive | lower |
| XRPUSDT | 1 | +0.28% | 100.0% | +0.04% | positive | lower |
| BCHUSDT | 7 | +0.24% | 33.3% | -0.04% | inconclusive | lower |
| LINKUSDT | 8 | +0.13% | 0.0% | -0.18% | negative | lower |
| AVAXUSDT | 12 | +0.09% | 60.0% | -0.21% | positive | lower |
| DOGEUSDT | 6 | +0.03% | 66.7% | -0.05% | inconclusive | lower |
| BTCUSDT | 3 | -0.09% | 0.0% | -0.12% | negative | lower |
| ADAUSDT | 5 | -0.15% | 0.0% | -0.23% | negative | lower |
| BNBUSDT | 7 | -0.20% | 100.0% | -0.02% | positive | lower |
| LTCUSDT | 8 | -0.33% | 25.0% | -0.20% | inconclusive | lower |
| ETHUSDT | 4 | -1.03% | 0.0% | -0.01% | negative | lower |
| DOTUSDT | 8 | -2.01% | 33.3% | -0.18% | inconclusive | lower |

## 数据限制（必须先读）

1. **delta 来源**：脚本真正用的是 `request.security_lower_tf` 的 intrabar delta；本项目 1m 历史约 93 天，`--delta-source auto` 在有 1m 数据的区间用真实聚合 delta，其余区间降级为脚本自带的 bar shape 估计（`na(ltfDelta) ? estDelta : ltfDelta`）。两种来源的信号密度差异很大，表中 `delta 来源` 列逐币标注。
2. **信号稀疏**：吸收判定要求「巨量 + 无进展 + 极值 + 长影线 + 单边支配」同时成立，脚本默认参数下每个币每月只有个位数信号，样本量的上限由策略本身决定，不是取数问题。
3. **撮合口径**：信号收盘 → 次根开盘成交（含滑点与费率）；R 倍数目标与保本价以成交价为基准，与脚本 `position_avg_price` 一致。

耗时 25.3s。

# 全指标 × K 线周期 × 币种池 适配测评（轮次 A）

口径 sha `b323c0912533842e` · 统计生成 2026-11-12T20:26:45+00:00 · 报告生成 2026-11-13T01:44:44+00:00 · 方向表 885 个布尔条件列（sha `2ad8c54c75753576`）· 成本 0.12% 往返（固定，不按策略覆盖）

本报告是 `docs/indicator-evaluation-plan.md` 的 S0–S5 落地结果。md 里每个数都来自同一次运行写入的同名 `indicator-fit-matrix.json`，手改数字无效——要改结论得改口径重跑。

## 1. 结论

> 审计修正：本轮仅为探索性筛选，candidate 是当前实现标签。max-T 尚未实施；交互检验使用未校正多重比较的池内外 bootstrap，并非预登记的整体交互；Q 池是事后固定分档，major/alt 成员未完全冻结；子集标签未检查整体方向一致性。跨周期最强格还混合了币种构成和实际持有时间，不能单独证明周期适配。原始统计保留，后续验证 BLOCKED；详见 `docs/indicator-fit-readiness.md`。

- 评测量：方向表声明 885 个布尔条件列，其中 743 列进入统计（142 列因触发率退化在筛选阶段就被排除）；逐「列 × 池 × 预测长度 × 基准」出判定行，方向声明为 ambiguous 的列两个方向各计一行，合计 190,974 行（5m 47,034÷48 格 = 每格 980 行；15m 48,036÷48 格 = 每格 1001 行；30m 47,904÷48 格 = 每格 998 行；1h 48,000÷48 格 = 每格 1000 行。每格的行数 = 这一格里参评的「列 × 方向」组合数，所以会高于参评列数，多出来的是双向列，不是重复计数）。
- 说得出口（candidate/cost_fragile/exploratory/underpowered_blocks）的列：**231 / 743**；其中当前实现标记为 `candidate` 的有 96 列，均待协议复核。
- 各列在本次样本中的最强格周期分布（探索性排名）：5m 9 列、15m 28 列、30m 51 列、1h 143 列。
- 最强格落在哪个池：全样本 39 列、四周期交集 7 列、主流 46 列、山寨 17 列、Q1 最高流动 19 列、Q2 52 列、Q3 24 列、Q4 最低流动 27 列。
- 原实现列为「池特有」的有 55 列；由于事后分池、交互检验选择偏差及整体方向门缺失，该数量仅表示复核线索，不是已证实的池适配发现。

## 2. 口径（S0 冻结）

| 周期 | 币数 | 声明列数 | 进统计列数 | 判定行数 | 落盘行数 | 窗口 | bars |
|---|---|---|---|---|---|---|---|
| 5m | 80 | 885 | 715 | 47034 | 6874 | 90d | 2073600 |
| 15m | 101 | 885 | 730 | 48036 | 9287 | 90d | 872640 |
| 30m | 99 | 885 | 730 | 47904 | 11543 | 90d | 427680 |
| 1h | 119 | 885 | 737 | 48000 | 16306 | 90d | 257040 |

- 币种池规模（按周期分别统计；`—` = 这个池在该周期下没有合格币，判定分布里也不会有它）：

| 币种池 | 5m 币数 | 15m 币数 | 30m 币数 | 1h 币数 |
|---|---|---|---|---|
| 全样本 | 80 | 101 | 99 | 119 |
| 四周期交集 | 80 | 80 | 80 | 80 |
| 主流 | 16 | 16 | 16 | 16 |
| 山寨 | 60 | 79 | 78 | 96 |
| Q1 最高流动 | 20 | 20 | 20 | 20 |
| Q2 | 20 | 20 | 20 | 20 |
| Q3 | 20 | 20 | 20 | 20 |
| Q4 最低流动 | 20 | 20 | 20 | 20 |

- Q1–Q4 为本轮新建池（全样本口径 Q1 20 币、Q2 20 币、Q3 20 币、Q4 20 币），按近 4 个月月成交额四分位划分，分档稳定性中位数 0.75——「这个币属于哪一档」本身会漂，池级结论必须连同这个漂移一起读。
- 数据门：密度 ≥0.85、历史 ≥500 bar、缺口率 ≤0.02、零量占比 ≤0.05。S1 实查：
| 周期 | 送检币数 | 通过 | 被拒币数 |
|---|---|---|---|
| 5m | 80 | 80 | 0 |
| 15m | 101 | 101 | 0 |
| 30m | 99 | 99 | 0 |
| 1h | 119 | 119 | 0 |

- 出场口径：signal 在 t 收盘确认，t+1 开盘成交，t+h 收盘出场；预测长度 h ∈ [6, 24, 96] bar（各周期折算成各自的真实时间）。
- 方向口径：`excess = dir × (r − bench) − cost`，`dir` 只来自列名语义表，不用被测收益反推。来源分布：declared 495、weak 60、ambiguous 330。
- 基准不减 0 不行：同时报 `unconditional`（同币同周期全体 bar 期望）与 `cross_section`（同一时刻一篮子等权期望），90 天普涨窗口里有多少「净为正」其实是 beta，只有减法知道。

### 2.1 判定门

| 关卡 | 判据 |
|---|---|
| 1 效应 | 净均值 > 0 且 2000 次按币 block bootstrap 的 95% CI 下界 > 0 |
| 2 显著性 | BH-FDR q < 0.1（家族 = 同周期 × 同池 × 同 h × 同基准，逐周期单独校正，跨周期不共享） |
| 3 稳定性 | 窗内 4 段同号率 ≥ 0.75，币间同号率 ≥ 0.6 |
| 4 可执行 | 2× 成本后净超额仍 > 0；格子事件 ≥100、覆盖币 ≥5、触发率落在 [0.002, 0.6] |
| 5 币块下限 | `symbols_in_stat ≥ 15`：币块少时 p 打到离散下界 1/(n_boot+1)，q<0.10 是免费的，降级为 `underpowered_blocks` |
| 分层附加 | 格子须过交互检验 p < 0.1，并按 w=n/(n+30.0) 向整体收缩后才许引用 |

判定词表（强→弱）：candidate, cost_fragile, exploratory, underpowered_blocks, unstable, not_universal, likely_noise, inconclusive, no_edge, no_heterogeneity, underpowered_cell, insufficient。`candidate` 会进台账，其余都不算发现。

## 3. 周期适配：每列最强格

只列「说得出口」的列，按判定等级 → CI 下界排序。`池` 是最强格所在币种池；`整体` 是同列在 `all` 池的最强判定，用来对照池级结论是否只是整体复读。

| 指标.列 | 向 | 周期 | 池 | h | 基准 | 事件 | 毛% | 净% | CI下界% | q | 段同号 | 币同号 | 2×成本% | 交互p | 判定 | 整体判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `capital41_adx_filter.bear_direction` | 空 | 1h | 主流 | h96 | 截面 | 271 | 2.01 | 1.89 | 1.44 | 0.0006 | 0.75 | 0.94 | 1.77 | 0.000 | candidate | candidate |
| `adx_advanced.di_bear_cross` | 空 | 1h | 主流 | h96 | 截面 | 270 | 1.85 | 1.73 | 1.32 | 0.0006 | 0.75 | 0.94 | 1.61 | 0.000 | candidate | unstable |
| `williams_pct_r_advanced.trend_divergence_bear` | 空 | 1h | Q2 | h96 | 截面 | 231 | 2.35 | 2.23 | 1.11 | 0.0713 | 0.75 | 0.79 | 2.11 | 0.057 | candidate | not_universal |
| `mtf_wavetrend_tide_wave_ripple.short_signal` | 空 | 1h | Q1 最高流动 | h96 | 无条件 | 297 | 2.09 | 1.97 | 1.08 | 0.0064 | 0.75 | 0.85 | 1.85 | 0.051 | candidate | unstable |
| `mtf_wavetrend_tide_wave_ripple.trigger_bear` | 空 | 1h | Q1 最高流动 | h96 | 无条件 | 303 | 2.00 | 1.88 | 0.86 | 0.0064 | 0.75 | 0.85 | 1.76 | 0.070 | candidate | unstable |
| `support_resistance.support_breakout` | 多 | 30m | 全样本 | h96 | 无条件 | 773 | 2.17 | 2.05 | 0.81 | 0.0104 | 1.00 | 0.63 | 1.93 | — | candidate | candidate |
| `volume_flow_anomaly.upside_surge` | 多 | 1h | 全样本 | h24 | 无条件 | 1,413 | 1.59 | 1.47 | 0.75 | 0.0047 | 0.75 | 0.61 | 1.35 | — | candidate | candidate |
| `gcr_strategy_scanner_tradingview.bullish_volume_signal` | 多 | 1h | Q2 | h24 | 无条件 | 506 | 2.34 | 2.22 | 0.74 | 0.0056 | 1.00 | 0.80 | 2.10 | 0.028 | candidate | candidate |
| `roc_advanced.bull_weak` | 多 | 30m | 四周期交集 | h96 | 无条件 | 836 | 1.77 | 1.65 | 0.64 | 0.0096 | 1.00 | 0.65 | 1.53 | 0.007 | candidate | candidate |
| `capital41_duo_oscillator.bull_cross` | 多 | 15m | 全样本 | h96 | 无条件 | 1,414 | 1.38 | 1.26 | 0.57 | 0.0025 | 1.00 | 0.67 | 1.14 | — | candidate | candidate |
| `cci_advanced.bear_zero_cross` | 空 | 1h | Q4 最低流动 | h96 | 截面 | 348 | 1.68 | 1.56 | 0.56 | 0.0307 | 0.75 | 0.85 | 1.44 | 0.018 | candidate | unstable |
| `capital41_order_block_v2.bull_ob_created` | 多 | 1h | 全样本 | h24 | 无条件 | 2,224 | 1.18 | 1.06 | 0.55 | 0.0030 | 1.00 | 0.71 | 0.94 | — | candidate | candidate |
| `mtf_wavetrend_tide_wave_ripple.short_watch` | 空 | 1h | 主流 | h24 | 无条件 | 310 | 1.01 | 0.89 | 0.49 | 0.0021 | 0.75 | 0.88 | 0.77 | 0.000 | candidate | no_edge |
| `bitcoin_market_regime.is_bear` | 空 | 15m | Q4 最低流动 | h96 | 截面 | 797 | 0.95 | 0.83 | 0.47 | 0.0167 | 0.75 | 0.95 | 0.71 | 0.002 | candidate | unstable |
| `williams_vix_fix_advanced.divergence_bull` | 多 | 15m | 四周期交集 | h96 | 截面 | 1,527 | 1.34 | 1.22 | 0.46 | 0.0334 | 0.75 | 0.62 | 1.10 | 0.004 | candidate | candidate |
| `gcr_strategy_scanner_tradingview.bullish_inflection` | 多 | 30m | Q1 最高流动 | h24 | 无条件 | 284 | 1.08 | 0.96 | 0.46 | 0.0026 | 0.75 | 0.94 | 0.84 | 0.078 | candidate | underpowered_blocks |
| `sm_radar.choch_bull` | 多 | 1h | Q4 最低流动 | h24 | 无条件 | 467 | 1.59 | 1.47 | 0.43 | 0.0161 | 1.00 | 0.70 | 1.35 | 0.070 | candidate | unstable |
| `rsi_advanced.context_bearish` | 空 | 15m | Q2 | h96 | 无条件 | 768 | 1.47 | 1.35 | 0.43 | 0.0073 | 1.00 | 0.85 | 1.23 | 0.001 | candidate | inconclusive |
| `tsi_advanced.trend_divergence_bull` | 多 | 30m | Q4 最低流动 | h24 | 无条件 | 572 | 1.08 | 0.96 | 0.43 | 0.0101 | 0.75 | 0.85 | 0.84 | 0.001 | candidate | candidate |
| `mtf_structure_bias.bear_zone_entry` | 空 | 1h | 山寨 | h24 | 无条件 | 525 | 1.05 | 0.93 | 0.42 | 0.0038 | 1.00 | 0.73 | 0.81 | 0.093 | candidate | candidate |
| `chandelier_flip_radar.weak_flip_short` | 空 | 30m | Q1 最高流动 | h96 | 截面 | 227 | 1.82 | 1.70 | 0.40 | 0.0433 | 0.75 | 0.81 | 1.58 | 0.025 | candidate | inconclusive |
| `elder_ray_pressure_engine.bear_extreme` | 空 | 30m | Q2 | h96 | 无条件 | 564 | 1.35 | 1.23 | 0.38 | 0.0227 | 0.75 | 0.85 | 1.11 | 0.053 | candidate | likely_noise |
| `smooth_trend_radar.short_signal` | 空 | 15m | 主流 | h96 | 截面 | 348 | 0.60 | 0.48 | 0.38 | 0.0007 | 1.00 | 1.00 | 0.36 | 0.071 | candidate | inconclusive |
| `cvd_bias.bearish_divergence` | 空 | 1h | 全样本 | h96 | 无条件 | 1,169 | 1.70 | 1.58 | 0.37 | 0.0553 | 0.75 | 0.79 | 1.46 | — | candidate | candidate |
| `live_market_scanner_v7.ema_bearish` | 空 | 1h | 主流 | h24 | 截面 | 600 | 0.64 | 0.52 | 0.37 | 0.0009 | 0.75 | 0.94 | 0.40 | 0.007 | candidate | candidate |
| `capital41_order_block_v2.bull_impulse` | 多 | 1h | 全样本 | h24 | 无条件 | 2,810 | 0.88 | 0.76 | 0.37 | 0.0030 | 0.75 | 0.69 | 0.64 | — | candidate | candidate |
| `smart_money_concepts.is_bearish_ob` | 空 | 1h | Q1 最高流动 | h24 | 无条件 | 368 | 0.90 | 0.78 | 0.36 | 0.0039 | 1.00 | 0.85 | 0.66 | 0.000 | candidate | no_edge |
| `candle_pressure_index.spike_bull` | 多 | 1h | 全样本 | h24 | 无条件 | 1,659 | 1.04 | 0.92 | 0.35 | 0.0047 | 0.75 | 0.62 | 0.80 | — | candidate | candidate |
| `capital41_confluence_core.trend_short` | 空 | 1h | Q2 | h24 | 无条件 | 771 | 1.33 | 1.21 | 0.34 | 0.0056 | 0.75 | 0.80 | 1.09 | 0.032 | candidate | candidate |
| `capital41_fusion_oscillator.trend_short` | 空 | 1h | Q2 | h24 | 无条件 | 771 | 1.33 | 1.21 | 0.34 | 0.0056 | 0.75 | 0.80 | 1.09 | 0.032 | candidate | candidate |
| `capital41_fusion_scalp.trend_short` | 空 | 1h | Q2 | h24 | 无条件 | 771 | 1.33 | 1.21 | 0.34 | 0.0056 | 0.75 | 0.80 | 1.09 | 0.032 | candidate | candidate |
| `capital41_oscillator.trend_short` | 空 | 1h | Q2 | h24 | 无条件 | 771 | 1.33 | 1.21 | 0.34 | 0.0056 | 0.75 | 0.80 | 1.09 | 0.032 | candidate | candidate |
| `live_market_scanner_v7.bear_trend` | 空 | 1h | Q2 | h24 | 无条件 | 771 | 1.33 | 1.21 | 0.34 | 0.0056 | 0.75 | 0.80 | 1.09 | 0.032 | candidate | candidate |
| `elder_ray_pressure_engine.bearish_divergence` | 空 | 1h | 四周期交集 | h24 | 截面 | 920 | 0.89 | 0.77 | 0.31 | 0.0500 | 0.75 | 0.69 | 0.65 | 0.080 | candidate | candidate |
| `capital41_fusion_oscillator.wr_short` | 空 | 1h | Q3 | h96 | 无条件 | 377 | 0.99 | 0.87 | 0.30 | 0.0227 | 0.75 | 0.80 | 0.75 | 0.000 | candidate | no_edge |
| `capital41_fusion_scalp.wr_short` | 空 | 1h | Q3 | h96 | 无条件 | 377 | 0.99 | 0.87 | 0.30 | 0.0227 | 0.75 | 0.80 | 0.75 | 0.000 | candidate | no_edge |
| `williams_pct_r_advanced.bull_extreme_strong` | 多 | 15m | Q4 最低流动 | h96 | 无条件 | 528 | 1.61 | 1.49 | 0.30 | 0.0689 | 0.75 | 0.65 | 1.37 | 0.045 | candidate | not_universal |
| `chandelier_flip_radar.sell_signal` | 空 | 30m | Q2 | h24 | 无条件 | 535 | 1.12 | 1.00 | 0.29 | 0.0094 | 1.00 | 0.85 | 0.88 | 0.000 | candidate | inconclusive |
| `cci_advanced.bear_strong` | 空 | 30m | 主流 | h96 | 截面 | 266 | 0.88 | 0.76 | 0.29 | 0.0037 | 0.75 | 0.88 | 0.64 | 0.004 | candidate | inconclusive |
| `crypto_amt_ict_smc_companion_v1.new_bear_ifvg` | 空 | 1h | Q2 | h24 | 无条件 | 871 | 0.83 | 0.71 | 0.29 | 0.0056 | 0.75 | 0.70 | 0.59 | 0.011 | candidate | inconclusive |
| `roc_advanced.stall_bull` | 多 | 1h | Q2 | h24 | 无条件 | 500 | 1.36 | 1.24 | 0.28 | 0.0056 | 0.75 | 0.88 | 1.12 | 0.005 | candidate | candidate |
| `structure_break_risk.bear_divergence` | 空 | 1h | 四周期交集 | h24 | 无条件 | 764 | 1.01 | 0.89 | 0.28 | 0.0209 | 0.75 | 0.62 | 0.77 | 0.026 | candidate | inconclusive |
| `compression_fractal_release.release_short` | 空 | 30m | Q4 最低流动 | h24 | 截面 | 320 | 0.66 | 0.54 | 0.28 | 0.0073 | 0.75 | 0.85 | 0.42 | 0.013 | candidate | unstable |
| `mtf_stochrsi_pair_score.late_short` | 空 | 15m | Q4 最低流动 | h96 | 截面 | 720 | 0.78 | 0.66 | 0.28 | 0.0167 | 0.75 | 0.85 | 0.54 | 0.034 | candidate | inconclusive |
| `tsi_advanced.context_bearish` | 空 | 15m | Q2 | h96 | 无条件 | 912 | 0.89 | 0.77 | 0.27 | 0.0073 | 1.00 | 0.85 | 0.65 | 0.060 | candidate | unstable |
| `smc_toolkit.choch_bearish` | 空 | 1h | 主流 | h24 | 截面 | 485 | 0.63 | 0.51 | 0.27 | 0.0009 | 0.75 | 0.94 | 0.39 | 0.017 | candidate | unstable |
| `roc_advanced.div_bull` | 多 | 15m | Q4 最低流动 | h96 | 无条件 | 431 | 3.32 | 3.20 | 0.27 | 0.0913 | 0.75 | 0.60 | 3.08 | 0.059 | candidate | candidate |
| `volume_flow_anomaly.downside_surge` | 多 | 1h | Q1 最高流动 | h6 | 无条件 | 320 | 0.67 | 0.55 | 0.26 | 0.0027 | 1.00 | 0.90 | 0.43 | 0.073 | candidate | candidate |
| `adaptive_arithmetic_candles_quality_score.long_signal` | 多 | 15m | Q2 | h24 | 无条件 | 287 | 1.90 | 1.78 | 0.26 | 0.0384 | 1.00 | 0.60 | 1.66 | 0.015 | candidate | candidate |
| `tsi_advanced.bear_zero_cross` | 空 | 1h | Q2 | h24 | 截面 | 586 | 0.89 | 0.77 | 0.25 | 0.0200 | 0.75 | 0.85 | 0.65 | 0.018 | candidate | likely_noise |
| `btc_scalping_signals.long_signal` | 多 | 1h | Q4 最低流动 | h96 | 无条件 | 334 | 3.24 | 3.12 | 0.24 | 0.0752 | 0.75 | 0.85 | 3.00 | 0.000 | candidate | no_edge |
| `commodity_flowtrend.bear_mid_confirm` | 空 | 1h | 主流 | h24 | 截面 | 533 | 0.60 | 0.48 | 0.24 | 0.0017 | 0.75 | 0.81 | 0.36 | 0.042 | candidate | unstable |
| `alert_engine_webhook.bear_signal` | 空 | 30m | 主流 | h24 | 截面 | 609 | 0.46 | 0.34 | 0.24 | 0.0012 | 0.75 | 1.00 | 0.22 | 0.000 | candidate | unstable |
| `alert_engine_webhook.raw_bear` | 空 | 30m | 主流 | h24 | 截面 | 609 | 0.46 | 0.34 | 0.24 | 0.0012 | 0.75 | 1.00 | 0.22 | 0.000 | candidate | unstable |
| `capital41_regime_matrix.pressure_bull` | 多 | 15m | 全样本 | h96 | 无条件 | 2,639 | 0.72 | 0.60 | 0.24 | 0.0041 | 0.75 | 0.66 | 0.48 | — | candidate | candidate |
| `mtf_stochrsi_pair_score.short_trigger` | 空 | 30m | Q4 最低流动 | h24 | 无条件 | 373 | 0.62 | 0.50 | 0.23 | 0.0069 | 0.75 | 0.85 | 0.38 | 0.003 | candidate | inconclusive |
| `stoch_rsi_advanced.bear_strong` | 空 | 30m | 全样本 | h24 | 无条件 | 4,754 | 0.47 | 0.35 | 0.22 | 0.0023 | 0.75 | 0.78 | 0.23 | — | candidate | candidate |
| `adaptive_arithmetic_candles_quality_score.strong_bull` | 多 | 15m | Q2 | h24 | 无条件 | 359 | 1.67 | 1.55 | 0.22 | 0.0300 | 1.00 | 0.61 | 1.43 | 0.027 | candidate | candidate |
| `capital41_liquidity_void_imbalance_map.htf_bear_void` | 空 | 1h | Q4 最低流动 | h24 | 截面 | 594 | 0.81 | 0.69 | 0.21 | 0.0966 | 0.75 | 0.75 | 0.57 | 0.098 | candidate | inconclusive |
| `capital41_divergence_detector.bearish_double` | 空 | 1h | Q3 | h6 | 无条件 | 243 | 0.73 | 0.61 | 0.21 | 0.0118 | 0.75 | 0.77 | 0.49 | 0.013 | candidate | candidate |

（前 60 / 231 行；全部行见交互页与 `indicator-fit-stats-*.json`。）

### 3.1 判定分布（全量行统计，不受落盘裁剪影响）

| 周期 | 池 | candidate | cost_fragile | exploratory | underpowered_blocks | unstable | not_universal | likely_noise | inconclusive | no_edge | no_heterogeneity | underpowered_cell | insufficient |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5m | 全样本 | 4 | 2 | 4 | 0 | 1 | 1 | 0 | 97 | 5783 | 0 | 0 | 0 |
| 5m | 四周期交集 | 4 | 2 | 4 | 0 | 1 | 1 | 0 | 97 | 5783 | 0 | 0 | 0 |
| 5m | 主流 | 1 | 0 | 0 | 4 | 46 | 0 | 0 | 329 | 5428 | 0 | 0 | 0 |
| 5m | 山寨 | 1 | 5 | 5 | 0 | 1 | 2 | 0 | 123 | 5755 | 0 | 0 | 0 |
| 5m | Q1 最高流动 | 1 | 1 | 2 | 1 | 9 | 0 | 0 | 100 | 5760 | 0 | 0 | 0 |
| 5m | Q2 | 4 | 1 | 7 | 0 | 1 | 0 | 2 | 233 | 5650 | 0 | 0 | 0 |
| 5m | Q3 | 1 | 2 | 7 | 0 | 0 | 0 | 1 | 248 | 5627 | 0 | 0 | 0 |
| 5m | Q4 最低流动 | 1 | 1 | 4 | 4 | 5 | 0 | 1 | 162 | 5714 | 0 | 0 | 0 |
| 15m | 全样本 | 10 | 1 | 25 | 0 | 8 | 10 | 15 | 425 | 5524 | 0 | 0 | 0 |
| 15m | 四周期交集 | 12 | 1 | 21 | 0 | 11 | 9 | 11 | 467 | 5486 | 0 | 0 | 0 |
| 15m | 主流 | 6 | 1 | 14 | 5 | 430 | 0 | 0 | 331 | 5141 | 0 | 0 | 0 |
| 15m | 山寨 | 5 | 2 | 18 | 0 | 4 | 8 | 13 | 539 | 5435 | 0 | 0 | 0 |
| 15m | Q1 最高流动 | 5 | 0 | 7 | 1 | 27 | 0 | 15 | 617 | 5334 | 0 | 0 | 0 |
| 15m | Q2 | 15 | 0 | 12 | 1 | 3 | 2 | 22 | 1019 | 4944 | 0 | 0 | 0 |
| 15m | Q3 | 3 | 0 | 11 | 2 | 4 | 0 | 21 | 1034 | 4955 | 0 | 0 | 0 |
| 15m | Q4 最低流动 | 11 | 0 | 7 | 5 | 6 | 1 | 28 | 812 | 5124 | 0 | 0 | 0 |
| 30m | 全样本 | 13 | 0 | 27 | 1 | 17 | 27 | 20 | 922 | 4973 | 0 | 0 | 0 |
| 30m | 四周期交集 | 11 | 0 | 29 | 0 | 13 | 16 | 38 | 1007 | 4874 | 0 | 0 | 0 |
| 30m | 主流 | 23 | 3 | 21 | 8 | 661 | 0 | 12 | 722 | 4502 | 0 | 0 | 0 |
| 30m | 山寨 | 8 | 3 | 34 | 1 | 12 | 12 | 27 | 1066 | 4825 | 0 | 0 | 0 |
| 30m | Q1 最高流动 | 8 | 0 | 21 | 5 | 57 | 0 | 39 | 1161 | 4685 | 0 | 0 | 0 |
| 30m | Q2 | 9 | 0 | 10 | 7 | 9 | 0 | 63 | 1649 | 4253 | 0 | 0 | 0 |
| 30m | Q3 | 10 | 0 | 19 | 2 | 4 | 0 | 47 | 1678 | 4246 | 0 | 0 | 0 |
| 30m | Q4 最低流动 | 13 | 0 | 18 | 3 | 9 | 1 | 67 | 1435 | 4448 | 0 | 0 | 0 |
| 1h | 全样本 | 37 | 2 | 54 | 10 | 109 | 38 | 78 | 1393 | 4297 | 0 | 0 | 0 |
| 1h | 四周期交集 | 33 | 1 | 59 | 5 | 80 | 20 | 76 | 1519 | 4207 | 0 | 0 | 0 |
| 1h | 主流 | 28 | 6 | 84 | 22 | 947 | 3 | 6 | 973 | 3919 | 0 | 0 | 0 |
| 1h | 山寨 | 33 | 1 | 53 | 9 | 47 | 46 | 102 | 1572 | 4137 | 0 | 0 | 0 |
| 1h | Q1 最高流动 | 13 | 0 | 26 | 7 | 107 | 2 | 76 | 1685 | 4080 | 0 | 0 | 0 |
| 1h | Q2 | 41 | 0 | 52 | 11 | 33 | 2 | 112 | 2152 | 3591 | 0 | 0 | 0 |
| 1h | Q3 | 9 | 0 | 14 | 9 | 4 | 0 | 70 | 1993 | 3919 | 0 | 0 | 0 |
| 1h | Q4 最低流动 | 8 | 0 | 19 | 4 | 63 | 5 | 160 | 1862 | 3865 | 0 | 0 | 0 |

### 3.2 指标级汇总：每个指标适合哪个 K 线

231 个说得出口的列分布在 75 个指标上（指标按列数排序，最多列 40 行）。下表按**每一列自己的最强格**落在哪个周期计数：同一指标的不同列可以落在不同周期，`0` 表示这个指标在该周期没有一列说得出口。这是本次样本的探索性最强格计数，不能单独归因为周期适配；它统计的是**列**，同一指标多列同堆时（见 §5）算不了独立证据，读列数之前先读堆数。`最强判定` 取该指标所有列里最强的一级。

| 指标 | 列数 | 5m 列 | 15m 列 | 30m 列 | 1h 列 | 最强判定 | 最强格落点（池） |
|---|---|---|---|---|---|---|---|
| `capital41_duo_oscillator` | 7 | 1 | 2 | 1 | 3 | candidate | 全样本 2、Q2 2、主流 2、Q1 最高流动 1 |
| `gcr_strategy_scanner_tradingview` | 7 | 0 | 1 | 1 | 5 | candidate | Q2 2、Q4 最低流动 2、Q1 最高流动 1、Q3 1、主流 1 |
| `exhaustion_scanner` | 6 | 0 | 1 | 2 | 3 | exploratory | 山寨 3、四周期交集 1、Q2 1、Q4 最低流动 1 |
| `market_exhaustion` | 6 | 0 | 0 | 2 | 4 | candidate | Q1 最高流动 2、主流 2、全样本 1、Q4 最低流动 1 |
| `modern_wyckoff_state_machine_lite` | 6 | 1 | 0 | 3 | 2 | cost_fragile | 全样本 1、四周期交集 1、Q1 最高流动 1、Q3 1、Q4 最低流动 1、主流 1 |
| `roc_advanced` | 6 | 0 | 1 | 3 | 2 | candidate | Q2 2、全样本 1、四周期交集 1、Q4 最低流动 1、主流 1 |
| `smc_toolkit` | 6 | 0 | 1 | 1 | 4 | candidate | 全样本 3、Q3 1、Q4 最低流动 1、主流 1 |
| `tsi_advanced` | 6 | 0 | 1 | 2 | 3 | candidate | Q2 3、全样本 1、Q1 最高流动 1、Q4 最低流动 1 |
| `volume_flow_anomaly` | 6 | 0 | 2 | 1 | 3 | candidate | 全样本 1、山寨 1、Q1 最高流动 1、Q2 1、Q3 1、主流 1 |
| `capital41_adx_filter` | 5 | 0 | 0 | 0 | 5 | candidate | 主流 3、Q3 2 |
| `capital41_duo` | 5 | 0 | 1 | 1 | 3 | exploratory | 主流 2、全样本 1、Q1 最高流动 1、Q2 1 |
| `capital41_fusion_oscillator` | 5 | 0 | 0 | 0 | 5 | candidate | 全样本 2、Q2 2、Q3 1 |
| `crypto_amt_toolkit` | 5 | 0 | 1 | 1 | 3 | candidate | Q4 最低流动 2、Q2 1、Q3 1、主流 1 |
| `mtf_wavetrend_tide_wave_ripple` | 5 | 0 | 0 | 0 | 5 | candidate | 主流 3、Q1 最高流动 2 |
| `rsi_advanced` | 5 | 0 | 1 | 0 | 4 | candidate | Q2 3、全样本 1、主流 1 |
| `sm_radar` | 5 | 2 | 0 | 1 | 2 | candidate | 山寨 2、Q3 2、Q4 最低流动 1 |
| `smooth_trend_radar` | 5 | 0 | 1 | 2 | 2 | candidate | Q2 2、Q3 2、主流 1 |
| `storm_gauge` | 5 | 0 | 0 | 0 | 5 | exploratory | 山寨 3、Q2 1、主流 1 |
| `capital41_fusion_scalp` | 4 | 0 | 0 | 0 | 4 | candidate | 全样本 2、Q2 1、Q3 1 |
| `capital41_regime_matrix` | 4 | 0 | 2 | 0 | 2 | candidate | 全样本 2、Q2 1、主流 1 |
| `chandelier_flip_radar` | 4 | 0 | 0 | 4 | 0 | candidate | Q1 最高流动 2、Q2 1、主流 1 |
| `fib_reaction_memory` | 4 | 1 | 0 | 2 | 1 | cost_fragile | Q3 2、全样本 1、Q2 1 |
| `live_market_scanner_v7` | 4 | 0 | 0 | 0 | 4 | candidate | Q2 2、主流 2 |
| `trend_persistence_score` | 4 | 0 | 0 | 0 | 4 | exploratory | 全样本 1、山寨 1、Q4 最低流动 1、主流 1 |
| `williams_pct_r_advanced` | 4 | 0 | 1 | 0 | 3 | candidate | 全样本 1、Q2 1、Q4 最低流动 1、主流 1 |
| `zigzag_patterns` | 4 | 1 | 0 | 1 | 2 | exploratory | Q2 2、Q1 最高流动 1、Q3 1 |
| `bitcoin_market_regime` | 3 | 0 | 1 | 1 | 1 | candidate | 全样本 2、Q4 最低流动 1 |
| `capital41_divergence_detector` | 3 | 0 | 1 | 0 | 2 | candidate | Q3 3 |
| `capital41_session_vwap` | 3 | 0 | 0 | 2 | 1 | cost_fragile | 主流 2、Q4 最低流动 1 |
| `compression_fractal_release` | 3 | 0 | 0 | 1 | 2 | candidate | Q2 1、Q4 最低流动 1、主流 1 |
| `crypto_amt_ict_smc_companion_v1` | 3 | 0 | 0 | 1 | 2 | candidate | Q1 最高流动 1、Q2 1、Q4 最低流动 1 |
| `elder_ray_pressure_engine` | 3 | 0 | 0 | 2 | 1 | candidate | Q2 2、四周期交集 1 |
| `elliott_wave_radar` | 3 | 0 | 0 | 2 | 1 | exploratory | Q2 2、Q4 最低流动 1 |
| `ma_regime_bands` | 3 | 0 | 2 | 0 | 1 | candidate | Q2 2、主流 1 |
| `mfi_advanced` | 3 | 0 | 0 | 2 | 1 | candidate | 山寨 1、Q2 1、Q4 最低流动 1 |
| `multi_symbol_ema_bounce_alerts` | 3 | 0 | 0 | 0 | 3 | cost_fragile | 主流 3 |
| `predictability_regime_index` | 3 | 0 | 0 | 1 | 2 | exploratory | 全样本 1、Q2 1、主流 1 |
| `reversal_engine_score_v1` | 3 | 1 | 0 | 0 | 2 | candidate | Q3 2、Q4 最低流动 1 |
| `sr_zones_plasmazero` | 3 | 0 | 1 | 0 | 2 | cost_fragile | Q2 2、山寨 1 |
| `vein_reversal_zones` | 3 | 0 | 0 | 1 | 2 | candidate | Q1 最高流动 2、Q3 1 |

（只列前 40 / 75 个指标，按列数排序；整表在 `indicator-fit-matrix.json` 的 `indicator_rollup` 里，交互页则按列展开、可按指标名搜索。）

## 4. 币种池适配

下表保留原实现的池级统计。当前交互 p 未作跨格校正，Q 池含事后信息，且子集 candidate 未检查整体方向；不能将表中的「池特有」标签当作通过预登记的发现。

| 币种池 | 币数（分周期） | 最强格落此池的列数 | 此池 candidate 格数 | 池特有结论数 |
|---|---|---|---|---|
| 全样本 | 5m 80 / 15m 101 / 30m 99 / 1h 119 | 39 | 64 | 0 |
| 四周期交集 | 5m 80 / 15m 80 / 30m 80 / 1h 80 | 7 | 60 | 1 |
| 主流 | 5m 16 / 15m 16 / 30m 16 / 1h 16 | 46 | 58 | 12 |
| 山寨 | 5m 60 / 15m 79 / 30m 78 / 1h 96 | 17 | 47 | 0 |
| Q1 最高流动 | 5m 20 / 15m 20 / 30m 20 / 1h 20 | 19 | 27 | 7 |
| Q2 | 5m 20 / 15m 20 / 30m 20 / 1h 20 | 52 | 69 | 16 |
| Q3 | 5m 20 / 15m 20 / 30m 20 / 1h 20 | 24 | 23 | 5 |
| Q4 最低流动 | 5m 20 / 15m 20 / 30m 20 / 1h 20 | 27 | 33 | 14 |

原池特有标签明细（按判定等级 → 收缩后净超额排序，仅供复核）。`整体` 两列是同列在 `all` 池跨周期的最强格，并非同周期、同 h、同基准配对；该比较不能替代整体方向门。

| 列 | 周期 | 池 | h | 事件 | 格子净% | 收缩后净% | 整体净% | 整体判定 | 格子判定 | 交互p |
|---|---|---|---|---|---|---|---|---|---|---|
| `adx_advanced.di_bear_cross` | 1h | 主流 | h96 | 270 | 1.73 | 1.57 | 0.92 | unstable | candidate | 0.000 |
| `williams_pct_r_advanced.trend_divergence_bear` | 1h | Q2 | h96 | 231 | 2.23 | 2.09 | 1.06 | not_universal | candidate | 0.057 |
| `mtf_wavetrend_tide_wave_ripple.short_signal` | 1h | Q1 最高流动 | h96 | 297 | 1.97 | 1.89 | 1.06 | unstable | candidate | 0.051 |
| `mtf_wavetrend_tide_wave_ripple.trigger_bear` | 1h | Q1 最高流动 | h96 | 303 | 1.88 | 1.80 | 0.97 | unstable | candidate | 0.070 |
| `cci_advanced.bear_zero_cross` | 1h | Q4 最低流动 | h96 | 348 | 1.56 | 1.45 | 0.76 | unstable | candidate | 0.018 |
| `mtf_wavetrend_tide_wave_ripple.short_watch` | 1h | 主流 | h24 | 310 | 0.89 | 0.79 | -0.12 | no_edge | candidate | 0.000 |
| `bitcoin_market_regime.is_bear` | 15m | Q4 最低流动 | h96 | 797 | 0.83 | 0.81 | 0.30 | unstable | candidate | 0.002 |
| `gcr_strategy_scanner_tradingview.bullish_inflection` | 30m | Q1 最高流动 | h24 | 284 | 0.96 | 0.91 | 1.22 | underpowered_blocks | candidate | 0.078 |
| `sm_radar.choch_bull` | 1h | Q4 最低流动 | h24 | 467 | 1.47 | 1.41 | 0.64 | unstable | candidate | 0.070 |
| `rsi_advanced.context_bearish` | 15m | Q2 | h96 | 768 | 1.35 | 1.31 | 0.26 | inconclusive | candidate | 0.001 |
| `chandelier_flip_radar.weak_flip_short` | 30m | Q1 最高流动 | h96 | 227 | 1.70 | 1.53 | 0.27 | inconclusive | candidate | 0.025 |
| `elder_ray_pressure_engine.bear_extreme` | 30m | Q2 | h96 | 564 | 1.23 | 1.19 | 0.46 | likely_noise | candidate | 0.053 |
| `smooth_trend_radar.short_signal` | 15m | 主流 | h96 | 348 | 0.48 | 0.45 | 0.26 | inconclusive | candidate | 0.071 |
| `smart_money_concepts.is_bearish_ob` | 1h | Q1 最高流动 | h24 | 368 | 0.78 | 0.71 | -0.09 | no_edge | candidate | 0.000 |
| `capital41_fusion_oscillator.wr_short` | 1h | Q3 | h96 | 377 | 0.87 | 0.78 | -0.11 | no_edge | candidate | 0.000 |
| `capital41_fusion_scalp.wr_short` | 1h | Q3 | h96 | 377 | 0.87 | 0.78 | -0.11 | no_edge | candidate | 0.000 |
| `williams_pct_r_advanced.bull_extreme_strong` | 15m | Q4 最低流动 | h96 | 528 | 1.49 | 1.43 | 0.45 | not_universal | candidate | 0.045 |
| `chandelier_flip_radar.sell_signal` | 30m | Q2 | h24 | 535 | 1.00 | 0.95 | 0.40 | inconclusive | candidate | 0.000 |
| `cci_advanced.bear_strong` | 30m | 主流 | h96 | 266 | 0.76 | 0.64 | 0.05 | inconclusive | candidate | 0.004 |
| `crypto_amt_ict_smc_companion_v1.new_bear_ifvg` | 1h | Q2 | h24 | 871 | 0.71 | 0.69 | 0.56 | inconclusive | candidate | 0.011 |
| `structure_break_risk.bear_divergence` | 1h | 四周期交集 | h24 | 764 | 0.89 | 0.88 | 0.49 | inconclusive | candidate | 0.026 |
| `compression_fractal_release.release_short` | 30m | Q4 最低流动 | h24 | 320 | 0.54 | 0.51 | 1.46 | unstable | candidate | 0.013 |
| `mtf_stochrsi_pair_score.late_short` | 15m | Q4 最低流动 | h96 | 720 | 0.66 | 0.64 | 0.28 | inconclusive | candidate | 0.034 |
| `tsi_advanced.context_bearish` | 15m | Q2 | h96 | 912 | 0.77 | 0.76 | 0.34 | unstable | candidate | 0.060 |
| `smc_toolkit.choch_bearish` | 1h | 主流 | h24 | 485 | 0.51 | 0.48 | 0.34 | unstable | candidate | 0.017 |
| `tsi_advanced.bear_zero_cross` | 1h | Q2 | h24 | 586 | 0.77 | 0.74 | 0.29 | likely_noise | candidate | 0.018 |
| `btc_scalping_signals.long_signal` | 1h | Q4 最低流动 | h96 | 334 | 3.12 | 2.82 | -0.11 | no_edge | candidate | 0.000 |
| `commodity_flowtrend.bear_mid_confirm` | 1h | 主流 | h24 | 533 | 0.48 | 0.46 | 1.13 | unstable | candidate | 0.042 |
| `alert_engine_webhook.bear_signal` | 30m | 主流 | h24 | 609 | 0.34 | 0.33 | 0.27 | unstable | candidate | 0.000 |
| `alert_engine_webhook.raw_bear` | 30m | 主流 | h24 | 609 | 0.34 | 0.33 | 0.27 | unstable | candidate | 0.000 |
| `mtf_stochrsi_pair_score.short_trigger` | 30m | Q4 最低流动 | h24 | 373 | 0.50 | 0.46 | 0.10 | inconclusive | candidate | 0.003 |
| `capital41_liquidity_void_imbalance_map.htf_bear_void` | 1h | Q4 最低流动 | h24 | 594 | 0.69 | 0.67 | 0.21 | inconclusive | candidate | 0.098 |
| `btc_scalping_signals.short_signal` | 1h | 主流 | h24 | 653 | 0.41 | 0.39 | 0.94 | unstable | candidate | 0.002 |
| `crypto_amt_ict_smc_companion_v1.new_bear_fvg` | 1h | Q1 最高流动 | h24 | 882 | 0.46 | 0.45 | 0.25 | unstable | candidate | 0.054 |
| `stoch_rsi_advanced.divergence_bull` | 1h | Q4 最低流动 | h24 | 666 | 0.61 | 0.58 | 0.01 | inconclusive | candidate | 0.002 |
| `gcr_strategy_scanner_tradingview.bearish_divergence` | 15m | Q4 最低流动 | h24 | 336 | 0.48 | 0.42 | 0.22 | inconclusive | candidate | 0.000 |
| `roc_advanced.ctx_bear` | 1h | Q2 | h6 | 731 | 0.53 | 0.52 | 0.17 | unstable | candidate | 0.006 |
| `live_market_scanner_v7.valid_short` | 1h | Q2 | h24 | 1,078 | 0.44 | 0.42 | 0.56 | unstable | candidate | 0.003 |
| `capital41_divergence_detector.bear_reg_rsi` | 1h | Q3 | h6 | 297 | 0.47 | 0.44 | 0.46 | inconclusive | candidate | 0.038 |
| `rsi_advanced.bear_mid_cross` | 1h | 主流 | h24 | 678 | 0.40 | 0.38 | 0.20 | inconclusive | candidate | 0.014 |

## 5. 冗余归堆

同堆的列在**哪些币上有效**这件事高度一致——按逐币平均超额的币间 Pearson 相关衡量，r ≥ 0.8 连成一堆（并查集，传递闭包）。这不是逐事件重合率：聚合缓存里没有逐事件掩码，为了算重合率要再跑一遍几小时的事件抽取，不划算；而且币间相关会漏掉「在不同币上各说各话、但触发时刻几乎相同」的同族列——那一层留给后续小样本复核。**同堆只能算一份证据**：把 3 个同族指标当 3 个独立发现会直接放大 FDR 的欺骗性。归堆只在「说得出口」的列之间做，且按各自最强 h 分组，跨 h 的同族列不会连起来。

| 周期 | 冗余对 | 参与归堆列数 | 独立堆数 | 最大几堆规模 |
|---|---|---|---|---|
| 5m | 2 | 4 | 2 | 2列 / 2列 |
| 15m | 9 | 13 | 6 | 3列 / 2列 / 2列 / 2列 |
| 30m | 5 | 7 | 3 | 3列 / 2列 / 2列 |
| 1h | 31 | 28 | 11 | 5列 / 4列 / 3列 / 2列 |

15m 上 ≥3 列的同堆（最多列 6 堆，成员按列名排序截断）：

- 3 列：`capital41_duo_oscillator.bull_cross`. `capital41_duo_oscillator.rsi_turn_down`. `capital41_duo.rsi_turn_down`

30m 上 ≥3 列的同堆（最多列 6 堆，成员按列名排序截断）：

- 3 列：`storm_gauge.is_storm`. `storm_gauge.storm_begins`. `storm_gauge.storm_tint`

1h 上 ≥3 列的同堆（最多列 6 堆，成员按列名排序截断）：

- 5 列：`capital41_confluence_core.trend_short`. `capital41_fusion_oscillator.trend_short`. `capital41_fusion_scalp.trend_short`. `capital41_oscillator.trend_short`. `live_market_scanner_v7.bear_trend`
- 4 列：`capital41_fusion_oscillator.atr_expansion`. `capital41_fusion_oscillator.expansion_ok`. `capital41_fusion_scalp.atr_expansion`. `capital41_fusion_scalp.expansion_ok`
- 3 列：`storm_gauge.is_storm`. `storm_gauge.storm_begins`. `storm_gauge.storm_tint`

## 6. 成本与可执行性

| 周期 | 最强格数 | 2×成本后仍>0 | 3×成本后仍>0 |
|---|---|---|---|
| 5m | 16 | 5 | 3 |
| 15m | 71 | 49 | 34 |
| 30m | 137 | 109 | 88 |
| 1h | 519 | 483 | 464 |

往返成本固定 0.12%（单边手续费 0.04% + 滑点 0.02%），**不允许按策略覆盖**。`cost_fragile` = 只靠成本假设活着，一律不算发现。

## 7. 算力与运行证据

| 周期 | 币数 | 本轮计算 | 缓存命中 | 耗时 s | 失败币数 |
|---|---|---|---|---|---|
| 5m | 80 | 80 | 0 | 9573 | 0 |
| 15m | 101 | 101 | 0 | 4164 | 0 |
| 30m | 99 | 99 | 0 | 2109 | 0 |
| 1h | 119 | 119 | 0 | 1210 | 0 |

- 探针实测中位 0.04845 s/bar（1h、6 币）。算力模型是「代价 ∝ bar 数」：探针按全量 4 周期总 bar 数 3,630,960 外推 = 48.9 CPU 小时，10 worker ≈ 4.9 h、12 worker ≈ 4.1 h。本报告覆盖 5m、15m、30m、1h，其总 bar 数 3,630,960，即探针外推口径的全部；s/bar 只在单个周期上探针过，跨周期按 bar 数线性外推、没有实测第二个周期——真正的逐周期耗时看上表。
- 实测对账：四周期事件抽取墙钟合计 17,223 s ≈ 4.78 h，探针在 10 worker 下外推 4.9 h，误差 -2%（本轮 worker 数记在 `output/fit-events-A.log` 首行命令里，产物 json 不存这个字段）。「代价 ∝ bar 数」在本轮没被证伪，但它只有这一组实测——s/bar 仍只在一个周期上量过，别当周期性规律外推。
- 事件抽取跳过 55 个指标注册项（注册 156 个，其中 101 个产出布尔列；44 个一列都不出，11 个在探测帧上必然报错）。跳过它们不改变列集合——列由 dtype 决定，与币种/周期无关；`fit_events.run_interval` 对首币自检产出列数是否等于方向表的 885 列，不等就打报警。

## 8. 判负与判不出的列（REJECTED 登记）

- 512 列在任何周期、任何池、任何 h、任何基准下都没有说得出口的一格，全部按 REJECTED 登记，不做「换个参数再来一遍」。
- 判定分布（表 3.1）合计 190,974 行，全部按上面五道关逐行给出，没有「先筛掉不显著的再调参」这一步。
- 142 列在筛选阶段就被排除：触发率不在 [0.002, 0.6] 区间——恒真列等于「每根 bar 都开仓」，恒假列一个事件都没有。它们是「无法测」，不是「测过无效」，两者不许混写。
- `no_edge` = 净超额 ≤ 0（不是「没测到」）；`inconclusive` = 均值为正但 CI 下界 ≤ 0；`likely_noise` = CI 撑住了但 FDR q 没过。三者都不许写成「可能有效，待样本更多」。
- `underpowered_blocks` 是本轮新加的第 5 道门：因参与自助法的币块不足 15 个而降级，数字照给、不上候选台账。

## 9. 局限

- 窗口只有 90 天（四周期同窗可比是本轮主口径），是单一市场状态下的结论；`seg_consistency` 只覆盖窗内 4 段，跨年 regime 变化测不到。
- 事件研究只回答「信号后价格怎么走」，不回答「能不能赚钱」。`candidate` 接进生产前必须再过一遍 `runner.run_backtest`（真实仓位、分批止盈、止损、资金占用），不一致时以回测为准。
- bar 来源规则：同一轮同一周期内某币只用一个来源：native 满密度就用 native，否则派生；5m/15m/30m 由 1m 聚合、1h 由 15m 聚合。
- 横截面基准要求同一时刻 ≥5 个币可对齐，`cs_coverage` 明示样本被砍掉多少；覆盖率低的列，其截面结论本来就弱。
- 判定只用到「列名语义声明的方向」。声明为 ambiguous 的列两个方向都试，最好的一格也封顶在 `exploratory`——这是刻意的：事后挑方向不算发现。
- 校正只做到族内：BH-FDR 的家族 = 同周期 × 同池 × 同 h × 同基准，本轮共 190,974 条判定行（5m 8 池、15m 8 池、30m 8 池、1h 8 池）。跨周期、跨池这些「层」没有计入 DSR/PBO 的有效试验数——那是 S8 汇总要补的账。所以这里的 q 值回答「同一格这么多列里谁运气好」，不回答「整个项目一共试了多少次」。
- 本轮只落 S0–S5：组合筛选（S6）、真实仓位回测（S7）、台账汇总与试验数校正（S8）都没跑。候选名单要接进任何策略之前，后两步不是可选项（承接上文「事件研究只回答信号后价格怎么走」那条，已登记为未决项：`docs/decisions/OPEN-DECISIONS.md`）。
- 5m log 告警已定位并完成工程复核：trend_persistence_score 在平收盘窗口中提前求值了未选中的 log(0) 分支。重放 80 币，涉及 7 根K线；修复保留原回退值，完整输出与旧函数精确一致，选定事件统计与原缓存一致。见 `docs/indicator-fit-log-diagnosis.md`；该工程结论不解除其余方法或样本外验证阻塞。

## 10. 产物与复现

| 文件 | 内容 |
|---|---|
| `docs/indicator-fit-stats-A.json` | 判定明细：每列每池最强格 + 说得出口的列的全明细（紧凑写，分布按全量行统计）。本轮实测 31.8 MB，是原始产物不是交付物——报告与页面只读它，不重算。 |
| `docs/indicator-fit-matrix.md` | 本报告本身（与 json 同一次运行写出，md 的数都能在 json 里对上） |
| `docs/indicator-fit-matrix.json` | 本报告数字底稿 |
| `docs/indicator-fit-matrix.html` | 交互矩阵页：列 × 周期 × 池，可筛可排可下钻（由 `fit_matrix_page.py` 写出，不是本脚本） |
| `docs/indicator-directions.json` | 方向声明表（列名语义 → dir） |
| `docs/data-quality-<scope>-<interval>.json` | S1 数据门逐币检查结果 |
| `docs/fit-universe.json` | 币种池、流动性分档与分档稳定性 |

```
python backend/scripts/fit_prep.py                        # S0 冻结口径（写 scope sha）
python backend/scripts/indicator_fit_matrix.py --scope A --stage directions
python backend/scripts/indicator_fit_matrix.py --scope A --stage probe
python backend/scripts/indicator_fit_matrix.py --scope A --stage events --workers 12
python backend/scripts/indicator_fit_matrix.py --scope A --stage stats
python backend/scripts/fit_matrix_report.py --scope A     # 本 md + 同名 json
python backend/scripts/fit_matrix_page.py --scope A       # 交互页
python backend/scripts/fit_matrix_report.py --scope A --ledger \
    --verification '<测试计数>'   # 追加一条 research/experiments.jsonl 台账（数字全部取自本次产物）
python -m pytest backend/tests/test_fit_stats.py backend/tests/test_fit_events.py \
                   backend/tests/test_fit_redundancy.py backend/tests/test_fit_report.py \
                   backend/tests/test_fit_ledger.py backend/tests/test_fit_page.py -q
```


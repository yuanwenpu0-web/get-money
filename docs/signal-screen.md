# 指标信号批量筛选（1h / 400 天 / 前瞻 24 根）

## 口径

- 池化 40 个币，扫描 152 个布尔列
- 事件 = 条件上升沿，且相邻事件间隔 ≥ 24 根（前瞻窗口不重叠）
- 往返成本 0.12%（费率 0.04% + 滑点 0.02%，双边）
- 判定三关：净超额 CI 下界 > 0、FDR q < 0.10、时间分段同号 ≥ 3/4

## 结论

- 有效检验 132 个条件，净超额为正 64 个
- **通过三关的候选 0 个**
- 退化列（触发率 <0.2% 或 >60%）48 个，已剔除

## 排名（按净超额）

| 指标 | 条件列 | 事件数 | 毛% | 净% | 胜率 | t | q | CI下界% | 分段同号 | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| capital41_pdh_pdl | break_month_high | 345 | 0.9995 | 0.8795 | 0.41 | 1.192 | 0.489 | -0.5536 | 0.5 | inconclusive |
| capital41_duo | both_overbought | 642 | 0.9147 | 0.7947 | 0.47 | 1.865 | 0.350 | -0.0286 | 0.75 | inconclusive |
| capital41_duo_oscillator | both_overbought | 642 | 0.9147 | 0.7947 | 0.47 | 1.865 | 0.350 | -0.0286 | 0.75 | inconclusive |
| capital41_duo_oscillator | rsi_turn_up | 1532 | 0.5824 | 0.4624 | 0.53 | 2.432 | 0.180 | 0.1028 | 0.75 | likely_noise |
| capital41_duo | rsi_turn_up | 1532 | 0.5824 | 0.4624 | 0.53 | 2.432 | 0.180 | 0.1028 | 0.75 | likely_noise |
| capital41_duo_oscillator | bear_cross | 936 | 0.5679 | 0.4479 | 0.52 | 1.799 | 0.352 | -0.0437 | 0.75 | inconclusive |
| capital41_duo | both_oversold | 914 | 0.5261 | 0.4061 | 0.53 | 1.659 | 0.389 | -0.0287 | 0.75 | inconclusive |
| capital41_duo_oscillator | both_oversold | 914 | 0.5261 | 0.4061 | 0.53 | 1.659 | 0.389 | -0.0287 | 0.75 | inconclusive |
| candle_pressure_index | spike_bear | 2091 | 0.4825 | 0.3625 | 0.47 | 1.554 | 0.396 | -0.0532 | 0.75 | inconclusive |
| capital41_pdh_pdl | break_week_low | 1805 | 0.4726 | 0.3526 | 0.49 | 1.432 | 0.436 | -0.0742 | 0.75 | inconclusive |
| capital41_adx_filter | trending | 270 | 0.4344 | 0.3144 | 0.44 | 0.538 | 0.758 | -0.6813 | 0.5 | inconclusive |
| volume_flow_anomaly | extreme_volume | 1841 | 0.3946 | 0.2746 | 0.45 | 1.13 | 0.525 | -0.1656 | 0.5 | inconclusive |
| rsi_trading_signals | strong_sell | 2865 | 0.3857 | 0.2657 | 0.45 | 1.461 | 0.425 | -0.0713 | 0.5 | inconclusive |
| capital41_adx_filter | adx_cross_strong | 3466 | 0.3743 | 0.2543 | 0.48 | 1.707 | 0.389 | -0.0147 | 0.75 | inconclusive |
| capital41_adx_filter | strong_trend | 3490 | 0.3627 | 0.2427 | 0.48 | 1.64 | 0.389 | -0.0335 | 0.75 | inconclusive |
| rsi_trading_signals | sell_signal | 2861 | 0.3579 | 0.2379 | 0.43 | 1.259 | 0.465 | -0.1178 | 0.5 | inconclusive |
| rsi_trading_signals | overbought | 2861 | 0.3579 | 0.2379 | 0.43 | 1.259 | 0.465 | -0.1178 | 0.5 | inconclusive |
| volume_flow_anomaly | early_warning | 2256 | 0.3543 | 0.2343 | 0.48 | 1.635 | 0.389 | -0.0337 | 0.5 | inconclusive |
| capital41_duo_oscillator | bull_cross | 468 | 0.3286 | 0.2086 | 0.45 | 0.473 | 0.766 | -0.6337 | 0.5 | inconclusive |
| capital41_regime_matrix | pressure_bull | 1713 | 0.2854 | 0.1654 | 0.43 | 0.725 | 0.692 | -0.2758 | 0.5 | inconclusive |
| mtf_structure_bias | bull_zone_entry | 1191 | 0.2853 | 0.1653 | 0.45 | 0.665 | 0.718 | -0.3055 | 0.75 | inconclusive |
| capital41_duo | rsi_turn_down | 1080 | 0.2848 | 0.1648 | 0.43 | 0.51 | 0.758 | -0.4754 | 0.5 | inconclusive |
| capital41_duo_oscillator | rsi_turn_down | 1080 | 0.2848 | 0.1648 | 0.43 | 0.51 | 0.758 | -0.4754 | 0.5 | inconclusive |
| capital41_pdh_pdl | touch_week_low | 1077 | 0.2841 | 0.1641 | 0.47 | 0.857 | 0.618 | -0.1994 | 1.0 | inconclusive |
| capital41_fusion_scalp | atr_expansion | 7001 | 0.279 | 0.159 | 0.47 | 1.63 | 0.389 | -0.0229 | 0.5 | inconclusive |

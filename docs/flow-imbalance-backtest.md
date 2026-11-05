# 订单流净买入策略回测

数据：1m 聚合到 15m，主动买入字段全覆盖的主流币；切分 70% 训练 / 15% 验证 / 15% 测试。双边成本 0.04% 手续费 + 0.02% 滑点；仓位/杠杆/止损止盈由通用管线决定。

## 测试集汇总（跨币种中位数/均值）

| 指标 | 中位数 | 均值 | 币数 |
|---|---:|---:|---:|
| total_return_pct | -1.3096 | -2.0605 | 16 |
| max_drawdown_pct | -2.8487 | -3.6040 | 16 |
| sharpe | -2.5300 | -2.7544 | 16 |
| win_rate_pct | +52.3300 | +49.3269 | 16 |
| trades | +32.5000 | +32.8125 | 16 |
| avg_return_pct | -0.5509 | -0.5353 | 16 |
| expectancy_ci_low_pct | -1.5248 | -1.5257 | 16 |
| expectancy_ci_high_pct | +0.4687 | +0.5217 | 16 |
| random_mean_return_pct | -1.3380 | -1.5034 | 16 |
| random_p95_return_pct | +0.7670 | +0.3626 | 16 |
| buy_hold_return_pct | -5.0002 | +1.4840 | 16 |

## 逐币明细

| 币种 | bars | 收益% | MDD% | 胜率% | 夏普 | 笔数 | 单笔均% | 每笔CI | 随机均值 | 95%随机 | buy&hold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 34,535 | -2.54 | -3.82 | 52.94 | -3.20 | 34 | -0.8160 | [-1.9554, +0.4721] | -3.18 | -1.96 | -0.77 |
| ETHUSDT | 34,535 | -5.96 | -6.17 | 34.29 | -10.93 | 35 | -1.4796 | [-2.1933, -0.7117] | -1.90 | -1.06 | -7.73 |
| BNBUSDT | 34,447 | -4.71 | -5.02 | 45.00 | -6.41 | 40 | -1.2130 | [-2.1258, -0.2875] | -0.25 | +2.36 | -4.97 |
| SOLUSDT | 34,535 | -5.75 | -7.37 | 35.00 | -6.01 | 40 | -1.1951 | [-2.1047, -0.1689] | -2.25 | +1.32 | -8.53 |
| XRPUSDT | 34,535 | -4.93 | -5.31 | 33.33 | -6.20 | 33 | -1.5606 | [-2.7345, -0.2849] | -4.50 | -2.89 | -8.61 |
| ADAUSDT | 34,535 | -2.66 | -3.20 | 36.67 | -2.94 | 30 | -0.7317 | [-1.6615, +0.3256] | -0.71 | +2.57 | -10.95 |
| AVAXUSDT | 34,535 | -1.17 | -3.01 | 46.88 | -1.38 | 32 | -0.3514 | [-1.3880, +0.6849] | -1.25 | +0.72 | +10.19 |
| DOGEUSDT | 34,535 | +0.53 | -1.89 | 60.00 | 0.80 | 25 | +0.2403 | [-1.1515, +1.7546] | -0.23 | +0.82 | -12.48 |
| DOTUSDT | 34,535 | -0.10 | -2.48 | 56.10 | -0.09 | 41 | +0.1482 | [-0.7972, +1.1469] | -2.37 | +0.46 | -13.91 |
| LINKUSDT | 34,535 | +1.65 | -1.65 | 65.52 | 2.08 | 29 | +0.4024 | [-0.4774, +1.3358] | -2.01 | -0.55 | -5.36 |
| LTCUSDT | 34,535 | -0.74 | -2.41 | 51.72 | -0.88 | 29 | -0.3106 | [-1.3614, +0.8532] | -0.95 | +1.25 | +3.70 |
| TRXUSDT | 34,535 | -1.28 | -1.46 | 53.85 | -3.87 | 13 | -0.9428 | [-2.1009, +0.1683] | -1.63 | -0.49 | -5.03 |
| SUIUSDT | 34,535 | -1.34 | -2.69 | 53.12 | -2.12 | 32 | -0.3701 | [-1.1360, +0.4653] | -0.41 | +0.99 | +15.69 |
| NEARUSDT | 34,535 | +1.76 | -1.28 | 55.56 | 2.37 | 36 | +0.4077 | [-0.5576, +1.3649] | -1.22 | -0.34 | +59.78 |
| APTUSDT | 34,535 | +0.94 | -1.50 | 72.00 | 1.59 | 25 | +0.3446 | [-0.8576, +1.6592] | +0.23 | +0.92 | -0.51 |
| BCHUSDT | 34,535 | -6.66 | -8.38 | 37.25 | -6.88 | 51 | -1.1374 | [-1.8083, -0.4307] | -1.42 | +1.68 | +13.24 |

## 常见最优参数（各币选参分布）

- 5 币选中：`{'base_position_pct': 0.08, 'cooldown_bars': 10, 'cost_filter': True, 'entry_quantile': 0.95, 'exit_quantile': 0.5, 'flow_window': 48, 'funding_filter': False, 'heavy_score': 0.9, 'max_atr_pct': 0.03, 'max_hold_bars': 96, 'max_leverage': 10, 'max_position_pct': 0.25, 'max_signal_age_bars': 2, 'min_atr_cost_ratio': 0.5, 'min_hold_bars': 30, 'min_net_reward_risk': 1.0, 'min_score': 0.75, 'min_sl_pct': 0.0025, 'min_tp_pct': 0.005, 'min_volume_z': 0.0, 'quantile_window': 480, 'regime_filter': True, 'require_positive_flow': True, 'risk_per_trade_pct': 0.005, 'sl_atr_mult': 2.0, 'target_r': 2.0, 'tp_atr_mult': 2.5, 'trail_activation_atr': 1.5, 'trailing_atr_mult': 1.2, 'trailing_enabled': True, 'trend_ema': 0}`
- 3 币选中：`{'base_position_pct': 0.08, 'cooldown_bars': 10, 'cost_filter': True, 'entry_quantile': 0.9, 'exit_quantile': 0.5, 'flow_window': 16, 'funding_filter': False, 'heavy_score': 0.9, 'max_atr_pct': 0.03, 'max_hold_bars': 32, 'max_leverage': 10, 'max_position_pct': 0.25, 'max_signal_age_bars': 2, 'min_atr_cost_ratio': 0.5, 'min_hold_bars': 30, 'min_net_reward_risk': 1.0, 'min_score': 0.75, 'min_sl_pct': 0.0025, 'min_tp_pct': 0.005, 'min_volume_z': 0.0, 'quantile_window': 480, 'regime_filter': True, 'require_positive_flow': True, 'risk_per_trade_pct': 0.005, 'sl_atr_mult': 2.0, 'target_r': 2.0, 'tp_atr_mult': 2.5, 'trail_activation_atr': 1.5, 'trailing_atr_mult': 1.2, 'trailing_enabled': True, 'trend_ema': 0}`
- 2 币选中：`{'base_position_pct': 0.08, 'cooldown_bars': 10, 'cost_filter': True, 'entry_quantile': 0.9, 'exit_quantile': 0.5, 'flow_window': 16, 'funding_filter': False, 'heavy_score': 0.9, 'max_atr_pct': 0.03, 'max_hold_bars': 32, 'max_leverage': 10, 'max_position_pct': 0.25, 'max_signal_age_bars': 2, 'min_atr_cost_ratio': 0.5, 'min_hold_bars': 30, 'min_net_reward_risk': 1.0, 'min_score': 0.75, 'min_sl_pct': 0.0025, 'min_tp_pct': 0.005, 'min_volume_z': 0.0, 'quantile_window': 480, 'regime_filter': True, 'require_positive_flow': True, 'risk_per_trade_pct': 0.005, 'sl_atr_mult': 3.0, 'target_r': 3.0, 'tp_atr_mult': 2.5, 'trail_activation_atr': 1.5, 'trailing_atr_mult': 1.2, 'trailing_enabled': True, 'trend_ema': 0}`
- 2 币选中：`{'base_position_pct': 0.08, 'cooldown_bars': 10, 'cost_filter': True, 'entry_quantile': 0.9, 'exit_quantile': 0.5, 'flow_window': 16, 'funding_filter': False, 'heavy_score': 0.9, 'max_atr_pct': 0.03, 'max_hold_bars': 32, 'max_leverage': 10, 'max_position_pct': 0.25, 'max_signal_age_bars': 2, 'min_atr_cost_ratio': 0.5, 'min_hold_bars': 30, 'min_net_reward_risk': 1.0, 'min_score': 0.75, 'min_sl_pct': 0.0025, 'min_tp_pct': 0.005, 'min_volume_z': 0.0, 'quantile_window': 480, 'regime_filter': True, 'require_positive_flow': True, 'risk_per_trade_pct': 0.005, 'sl_atr_mult': 2.0, 'target_r': 3.0, 'tp_atr_mult': 2.5, 'trail_activation_atr': 1.5, 'trailing_atr_mult': 1.2, 'trailing_enabled': True, 'trend_ema': 0}`
- 1 币选中：`{'base_position_pct': 0.08, 'cooldown_bars': 10, 'cost_filter': True, 'entry_quantile': 0.9, 'exit_quantile': 0.5, 'flow_window': 16, 'funding_filter': False, 'heavy_score': 0.9, 'max_atr_pct': 0.03, 'max_hold_bars': 64, 'max_leverage': 10, 'max_position_pct': 0.25, 'max_signal_age_bars': 2, 'min_atr_cost_ratio': 0.5, 'min_hold_bars': 30, 'min_net_reward_risk': 1.0, 'min_score': 0.75, 'min_sl_pct': 0.0025, 'min_tp_pct': 0.005, 'min_volume_z': 0.0, 'quantile_window': 480, 'regime_filter': True, 'require_positive_flow': True, 'risk_per_trade_pct': 0.005, 'sl_atr_mult': 3.0, 'target_r': 3.0, 'tp_atr_mult': 2.5, 'trail_activation_atr': 1.5, 'trailing_atr_mult': 1.2, 'trailing_enabled': True, 'trend_ema': 0}`

## 结论

- 平均每币测试期 33 笔成交；策略平均收益 -2.06%。
- 3/16 币收益高于随机基线 95 分位；11/16 币的每笔期望 CI 跨 0（inconclusive）。
- **判定**：跨币种收益不稳定、多数币每笔期望 CI 跨 0。真实主动买入的尾部 edge 在扣费后被完全抵消，策略不具备生产价值。

## 限制

回测信号在验证/测试切分间重新评估，状态（分位阈值）严格因果滚动，探针中的全样本分位前视偏差已在本回测中消除。不同币种的最优参数差异大，存在多重比较风险；资金费率在本回测未计入（与 15m 持仓期相比影响不大，但应在正式决策前补上）。

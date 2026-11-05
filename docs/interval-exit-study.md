# 周期切换与分批止盈研究（15m vs 现周期）

数据：40 个币（15m/1h 双覆盖前 N），两个周期都取 300 天、测试段 = 各窗口最后 25%（墙钟对齐），`optimize=False`，成本 taker 0.04% + 滑点 0.02%。

## 结论速览

- `supertrend_current|single_3R`：15m 每笔 -0.299%（CI -0.382~-0.209，negative，7459 笔） / 1h 每笔 -0.016%（CI -0.220~0.195，inconclusive，1611 笔）；同币配对差中位数 -10.87pp，15m 更好的币占 20%。
- `supertrend_current|split_1_2_3_default`：15m 每笔 -0.313%（CI -0.380~-0.238，negative，7459 笔） / 1h 每笔 -0.058%（CI -0.230~0.117，inconclusive，1611 笔）；同币配对差中位数 -9.77pp，15m 更好的币占 15%。
- `supertrend_current|split_1_2_3_equal`：15m 每笔 -0.318%（CI -0.381~-0.250，negative，7459 笔） / 1h 每笔 -0.084%（CI -0.241~0.077，inconclusive，1611 笔）；同币配对差中位数 -9.35pp，15m 更好的币占 12%。
- `supertrend_current|split_half_at_1R`：15m 每笔 -0.322%（CI -0.380~-0.260，negative，7459 笔） / 1h 每笔 -0.105%（CI -0.251~0.043，inconclusive，1611 笔）；同币配对差中位数 -9.71pp，15m 更好的币占 12%。
- `absorption_reversal|single_2R`：15m 每笔 -0.848%（CI -1.648~0.096，insufficient，17 笔） / 1h 每笔 2.878%（CI -0.100~4.931，insufficient，3 笔）；同币配对差中位数 0.00pp，15m 更好的币占 8%。
- `absorption_reversal|split_0.5_1_2`：15m 每笔 -0.817%（CI -1.544~-0.082，insufficient，17 笔） / 1h 每笔 1.995%（CI 0.969~2.826，insufficient，3 笔）；同币配对差中位数 0.00pp，15m 更好的币占 5%。
- `capital41_fusion_scalp|single_atr`：15m 每笔 0.086%（CI -0.716~0.906，inconclusive，71 笔） / 1h 每笔 -1.558%（CI -3.156~0.413，insufficient，13 笔）；同币配对差中位数 0.00pp，15m 更好的币占 42%。

判读规则（全项目一致）：CI 下界 > 0 才算正 edge；CI 跨 0 是「不知道」，不是「没亏」；
上界 < 0 说明这套结构在样本内是负贡献。**同币配对差**消掉了币种涨跌的共同部分，
比直接比两列平均收益更能说明周期本身的效应。

## 逐组合明细

| 策略 / 出场结构 | 周期 | 币数 | 成交 | 每笔均值% | 95% CI | 判定 | 币均收益% | 收益中位% | 正收益币占比 | 跑赢持有 |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| supertrend_current|single_3R | 15m | 40 | 7459 | -0.299 | -0.382 ~ -0.209 | negative | -9.61 | -9.64 | 20% | 2% |
| supertrend_current|single_3R | 1h | 40 | 1611 | -0.016 | -0.220 ~ 0.195 | inconclusive | -0.11 | -0.05 | 50% | 20% |
| supertrend_current|split_1_2_3_default | 15m | 40 | 7459 | -0.313 | -0.380 ~ -0.238 | negative | -10.13 | -10.77 | 12% | 2% |
| supertrend_current|split_1_2_3_default | 1h | 40 | 1611 | -0.058 | -0.230 ~ 0.117 | inconclusive | -0.42 | -0.36 | 48% | 20% |
| supertrend_current|split_1_2_3_equal | 15m | 40 | 7459 | -0.318 | -0.381 ~ -0.250 | negative | -10.37 | -10.76 | 10% | 2% |
| supertrend_current|split_1_2_3_equal | 1h | 40 | 1611 | -0.084 | -0.241 ~ 0.077 | inconclusive | -0.60 | -0.57 | 48% | 18% |
| supertrend_current|split_half_at_1R | 15m | 40 | 7459 | -0.322 | -0.380 ~ -0.260 | negative | -10.54 | -10.93 | 5% | 2% |
| supertrend_current|split_half_at_1R | 1h | 40 | 1611 | -0.105 | -0.251 ~ 0.043 | inconclusive | -0.75 | -0.72 | 48% | 18% |
| absorption_reversal|single_2R | 15m | 40 | 17 | -0.848 | -1.648 ~ 0.096 | insufficient | -0.11 | 0.00 | 5% | 18% |
| absorption_reversal|single_2R | 1h | 40 | 3 | 2.878 | -0.100 ~ 4.931 | insufficient | 0.06 | 0.00 | 5% | 18% |
| absorption_reversal|split_0.5_1_2 | 15m | 40 | 17 | -0.817 | -1.544 ~ -0.082 | insufficient | -0.10 | 0.00 | 5% | 18% |
| absorption_reversal|split_0.5_1_2 | 1h | 40 | 3 | 1.995 | 0.969 ~ 2.826 | insufficient | 0.04 | 0.00 | 8% | 18% |
| capital41_fusion_scalp|single_atr | 15m | 40 | 71 | 0.086 | -0.716 ~ 0.906 | inconclusive | 0.04 | 0.00 | 35% | 18% |
| capital41_fusion_scalp|single_atr | 1h | 40 | 13 | -1.558 | -3.156 ~ 0.413 | insufficient | -0.07 | 0.00 | 8% | 18% |

## 出场原因分布

| 策略 / 出场结构 | 周期 | stop_loss | take_profit_* | 其它 |
| --- | --- | ---: | ---: | ---: |
| supertrend_current|single_3R | 15m | 4584 | 1570 | 1305 |
| supertrend_current|single_3R | 1h | 972 | 360 | 279 |
| supertrend_current|split_1_2_3_default | 15m | 4584 | 1570 | 1305 |
| supertrend_current|split_1_2_3_default | 1h | 972 | 360 | 279 |
| supertrend_current|split_1_2_3_equal | 15m | 4584 | 1570 | 1305 |
| supertrend_current|split_1_2_3_equal | 1h | 972 | 360 | 279 |
| supertrend_current|split_half_at_1R | 15m | 4584 | 1570 | 1305 |
| supertrend_current|split_half_at_1R | 1h | 972 | 360 | 279 |
| absorption_reversal|single_2R | 15m | 15 | 2 | 0 |
| absorption_reversal|single_2R | 1h | 1 | 2 | 0 |
| absorption_reversal|split_0.5_1_2 | 15m | 15 | 2 | 0 |
| absorption_reversal|split_0.5_1_2 | 1h | 1 | 2 | 0 |
| capital41_fusion_scalp|single_atr | 15m | 41 | 29 | 1 |
| capital41_fusion_scalp|single_atr | 1h | 10 | 3 | 0 |

## 口径与边界

- 研究策略（`research_high/low_turnover_high_vol`）不在此表：它们只做多、`direction≡0`、
  按「持有 N 小时」定时出场，走模拟盘专用通道，套不进通用撮合的止损/分批止盈。
  要测它们的 15m 版本，等于重写策略而不是换周期。
- 专用 broker 策略（`simple_zigzag_double` / `rgd_donchian_v2` / `opening_candle_breakout`）
  同样不在此表，结论见各自 `docs/*-strategy.md`。
- 15m 的 `max_hold_bars`/`cooldown_bars` 是**按 bar 计**，换周期后墙钟时长自动缩短 4 倍；
  本表沿用原参数，等于同时改了持仓时长，读数里混着这一层效应。

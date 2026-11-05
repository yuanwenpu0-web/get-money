# 研究池策略：换 15m 与「趋势反转式」分批止盈

数据：两个研究池当前成员，90 天窗口，持有期按墙钟对齐（120h / 48h）。成本 taker 0.04%/边 + 滑点 0.02%，每笔固定名义 200 USDT。

出场结构：`hold_only` = 现状（定时出场，无止损无止盈）；`hold_stop` = 加 1.5×ATR 止损；`split_1_2_3` = 再加 1R/2R/3R 按 1/6、1/3、1/2 分批止盈。

## 逐组合（同一批币，两侧可比）

| 策略 | 出场结构 | 周期 | 币数 | 成交 | 每笔均值% | 95% CI | 判定 | 币均累计% | 中位% | 正收益币 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| research_high_turnover_high_vol | hold_only | 15m | 13 | 208 | 9.702 | 3.545 ~ 16.881 | positive | 155.24 | 127.68 | 100% |
| research_high_turnover_high_vol | hold_only | 1h | 13 | 156 | 9.872 | 2.765 ~ 17.700 | positive | 118.46 | 126.80 | 92% |
| research_high_turnover_high_vol | hold_stop | 15m | 13 | 988 | 0.680 | -0.345 ~ 1.862 | inconclusive | 51.66 | 94.68 | 62% |
| research_high_turnover_high_vol | hold_stop | 1h | 13 | 346 | 3.979 | 0.603 ~ 8.471 | positive | 105.90 | 93.79 | 69% |
| research_high_turnover_high_vol | split_1_2_3 | 15m | 13 | 1872 | 0.093 | -0.140 ~ 0.321 | inconclusive | 13.34 | 7.64 | 54% |
| research_high_turnover_high_vol | split_1_2_3 | 1h | 13 | 459 | 0.976 | 0.169 ~ 1.765 | positive | 34.45 | 20.42 | 85% |
| research_low_turnover_high_vol | hold_only | 15m | 8 | 293 | 0.566 | -1.086 ~ 2.226 | inconclusive | 20.74 | 50.26 | 75% |
| research_low_turnover_high_vol | hold_only | 1h | 8 | 177 | 0.820 | -1.211 ~ 3.018 | inconclusive | 18.14 | 32.71 | 62% |
| research_low_turnover_high_vol | hold_stop | 15m | 8 | 799 | 0.292 | -0.232 ~ 0.888 | inconclusive | 29.14 | 40.32 | 75% |
| research_low_turnover_high_vol | hold_stop | 1h | 8 | 258 | -0.135 | -1.282 ~ 1.180 | inconclusive | -4.36 | -1.73 | 50% |
| research_low_turnover_high_vol | split_1_2_3 | 15m | 8 | 1191 | -0.023 | -0.212 ~ 0.177 | inconclusive | -3.40 | 5.22 | 50% |
| research_low_turnover_high_vol | split_1_2_3 | 1h | 8 | 287 | -0.360 | -1.000 ~ 0.293 | inconclusive | -12.91 | -10.55 | 38% |

## 同币配对：15m − 1h（消掉币种共同涨跌）

| 策略 | 出场结构 | 配对币数 | 中位差 pp | 15m 更好的币占比 | 差值 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| research_high_turnover_high_vol | hold_only | 13 | 43.99 | 69% | -2.47 ~ 79.52 |
| research_high_turnover_high_vol | hold_stop | 13 | -18.57 | 38% | -135.40 ~ 12.34 |
| research_high_turnover_high_vol | split_1_2_3 | 13 | -12.78 | 38% | -57.42 ~ 12.76 |
| research_low_turnover_high_vol | hold_only | 8 | 10.77 | 62% | -49.60 ~ 52.27 |
| research_low_turnover_high_vol | hold_stop | 8 | 23.85 | 88% | 0.06 ~ 70.48 |
| research_low_turnover_high_vol | split_1_2_3 | 8 | 12.09 | 62% | -33.26 ~ 54.43 |

## 覆盖：被窗口门槛剔除的币

| 策略 | 周期 | 可用币 | 剔除 | 剔除原因 |
| --- | --- | ---: | ---: | --- |
| research_high_turnover_high_vol | 15m | 13 | 12 | 15m/1h 历史短于窗口，两侧不可比 |
| research_high_turnover_high_vol | 1h | 24 | 1 | 15m/1h 历史短于窗口，两侧不可比 |
| research_low_turnover_high_vol | 15m | 8 | 17 | 15m/1h 历史短于窗口，两侧不可比 |
| research_low_turnover_high_vol | 1h | 24 | 1 | 15m/1h 历史短于窗口，两侧不可比 |

## 出场原因（同上口径）

| 策略 | 结构 | 周期 | stop_loss | take_profit | time_exit | 其它 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| research_high_turnover_high_vol | hold_only | 15m | 0 | 0 | 208 | 0 |
| research_high_turnover_high_vol | hold_only | 1h | 0 | 0 | 156 | 0 |
| research_high_turnover_high_vol | hold_stop | 15m | 904 | 0 | 84 | 0 |
| research_high_turnover_high_vol | hold_stop | 1h | 279 | 0 | 67 | 0 |
| research_high_turnover_high_vol | split_1_2_3 | 15m | 1388 | 472 | 12 | 0 |
| research_high_turnover_high_vol | split_1_2_3 | 1h | 306 | 134 | 19 | 0 |
| research_low_turnover_high_vol | hold_only | 15m | 0 | 0 | 293 | 0 |
| research_low_turnover_high_vol | hold_only | 1h | 0 | 0 | 177 | 0 |
| research_low_turnover_high_vol | hold_stop | 15m | 685 | 0 | 114 | 0 |
| research_low_turnover_high_vol | hold_stop | 1h | 192 | 0 | 66 | 0 |
| research_low_turnover_high_vol | split_1_2_3 | 15m | 882 | 298 | 11 | 0 |
| research_low_turnover_high_vol | split_1_2_3 | 1h | 199 | 65 | 23 | 0 |

## 全宇宙（两侧币集不同，只作参考，不可用于比较周期）

| 策略 | 结构 | 周期 | 币数 | 成交 | 每笔均值% | 95% CI | 判定 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| research_high_turnover_high_vol | hold_only | 15m | 13 | 208 | 9.702 | 3.545 ~ 16.881 | positive |
| research_high_turnover_high_vol | hold_only | 1h | 24 | 292 | 6.472 | 2.254 ~ 11.422 | positive |
| research_high_turnover_high_vol | hold_stop | 15m | 13 | 988 | 0.680 | -0.345 ~ 1.862 | inconclusive |
| research_high_turnover_high_vol | hold_stop | 1h | 24 | 659 | 2.980 | 0.346 ~ 6.317 | positive |
| research_high_turnover_high_vol | split_1_2_3 | 15m | 13 | 1872 | 0.093 | -0.140 ~ 0.321 | inconclusive |
| research_high_turnover_high_vol | split_1_2_3 | 1h | 24 | 843 | 0.295 | -0.193 ~ 0.794 | inconclusive |
| research_low_turnover_high_vol | hold_only | 15m | 8 | 293 | 0.566 | -1.086 ~ 2.226 | inconclusive |
| research_low_turnover_high_vol | hold_only | 1h | 24 | 523 | 0.286 | -0.843 ~ 1.468 | inconclusive |
| research_low_turnover_high_vol | hold_stop | 15m | 8 | 799 | 0.292 | -0.232 ~ 0.888 | inconclusive |
| research_low_turnover_high_vol | hold_stop | 1h | 24 | 804 | -0.483 | -1.037 ~ 0.088 | inconclusive |
| research_low_turnover_high_vol | split_1_2_3 | 15m | 8 | 1191 | -0.023 | -0.212 ~ 0.177 | inconclusive |
| research_low_turnover_high_vol | split_1_2_3 | 1h | 24 | 891 | -0.227 | -0.557 ~ 0.106 | inconclusive |

## 必须一起读的边界

- **池是事后挑的**：两个研究池按「最近 168 小时」成交额/波动率筛出（当前窗口到 2026-11-03），
  却拿来回放更早的行情 → 典型的事后选币。因此**绝对收益全部是上界，不能当预期**；
  可用的只有**同币配对差值**（同一批币、同一窗口，选币偏差在两侧相同）。
- **15m 历史是硬约束**：研究池里过半币种的 native 15m 只从 2026-10-15 起（约 1000 根），
  1m 补齐也只到 2026-07-28。所以 90 天窗口下 15m 只剩 13/25 与 8/25 个币可用，
  再拉长窗口可用币还会继续掉。这是「换 15m」目前最现实的障碍。
- 分批止盈改变了出场结构，等于改了策略本身；周期效应看同一 mode 的 15m vs 1h，
  止盈效应看同一周期的 mode 之间，两者不能混着读。
- 15m 的 ATR 约为 1h 的一半，同一个 1.5×ATR 止损在 15m 上更紧、被打掉的概率更高，
  所以 15m 的 `split_1_2_3` 与 1h 的 `split_1_2_3` 不是同一个风险预算。

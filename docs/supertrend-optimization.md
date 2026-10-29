# Supertrend 趋势反转 — 主流币周期与参数寻优

生成时间：2026-10-29T13:42:16+00:00　窗口：365 天　周期：15m, 1h, 4h

## 方法：时间切分，不是单窗口取最大值

每个连续K线段按 **70% / 30%** 切分：前段训练、后段测试。
参数选择**只看训练段**（池化每笔期望），判定**只看测试段**。

- 阶段 1 结构网格 1050 组：`supertrend_factor` × `atr_period` × `sl_atr_mult` × `target_r`
- 阶段 2 出场/过滤叠加 14 种：ema200, adx30, adx20, trail12, trail10, trail20, hold48h, hold240h, trail12_hold48h, ema200_trail12, ema200_hold48h, breakout, breakout_trail12, cooldown0
- 阶段 3 随机方向基线 20 种子

## 周期判定

| 周期 | 币数 | 可判定/总组数 | 测试段每笔期望(均值/中位) | 训练-测试相关 | negative | positive | 纯噪声期望假阳性 | 全窗口最好账户收益 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **15m** | 14 | 1050/1050 | -0.3945% / -0.4495% | -0.461 | 907 | 0 | 26 | -5.80% |
| **1h** | 16 | 1050/1050 | -0.1724% / -0.1861% | -0.341 | 450 | 6 | 26 | +2.90% |
| **4h** | 16 | 620/1050 | -0.0841% / -0.0842% | -0.090 | 26 | 2 | 16 | +4.51% |

三列读法：**训练-测试相关**为负 = 训练段选出的好参数在测试段更差（选参无意义）；**测试段 positive** 少于**纯噪声期望** = 连运气都不到纯噪声水平；**全窗口最好账户收益**是 1050 组里事后最好的那组，属于样本内上限。

## 同一窗口的买入持有

| 周期 | 币数 | 训练段买入持有(中位) | 测试段买入持有(中位) | 测试段 positive 组合 |
| --- | --- | --- | --- | --- |
| 15m | 14 | -42.18% | +27.52% | 0 组 |
| 1h | 16 | -57.46% | +23.65% | 6 组 |
| 4h | 16 | -57.25% | +23.47% | 2 组 |

这张表解释了两个周期的符号为什么会反：**训练段是熊市、测试段是牛市**。
趋势策略在熊市里做多必亏、在牛市里做多必赚 —— 于是「哪组参数更好」实际测的是「这组参数有多偏多」，而不是它有没有信息量。这也是训练段与测试段相关系数为负的机制：偏多在熊市排最后、在牛市排最前，排名整个翻转。

## 15m

训练段最优结构参数：

- 参数 `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0`，训练段 -0.0531% [-0.2601, +0.1630] inconclusive（4076 笔）
- 同一组在测试段：-0.3710% [-0.6748, -0.0696] negative（1744 笔）
- 账户收益中位：训练 -2.92% → 测试 -4.25%

样本内→样本外传导：训练段与测试段每笔期望相关 **-0.461**（Spearman -0.395，n=1050）；测试段 positive 0 组，纯噪声下应有约 26 组。

### 收益最高的 5 组（全窗口账户收益中位，样本内）

这一列是「如果一年前就选中它」的结果，**含训练段**，所以天然偏高。右边两列是同一组在各段的真实表现：

| # | 组合 | 全窗口账户中位 | 训练段每笔期望 | 测试段每笔期望 | 测试段判定 |
| --- | --- | --- | --- | --- | --- |
| 1 | `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | -5.80% | -0.0531% | -0.3710% | negative |
| 2 | `supertrend_factor5.0_atr_period21_sl_atr_mult3.0_target_r6.0` | -7.52% | -0.4931% | -0.3397% | inconclusive |
| 3 | `supertrend_factor5.0_atr_period21_sl_atr_mult3.0_target_r1.5` | -7.74% | -0.4962% | -0.0918% | inconclusive |
| 4 | `supertrend_factor5.0_atr_period21_sl_atr_mult3.0_target_r3.0` | -8.04% | -0.5981% | +0.0100% | inconclusive |
| 5 | `supertrend_factor5.0_atr_period30_sl_atr_mult3.0_target_r2.0` | -8.12% | -0.5269% | +0.1525% | inconclusive |

### 参数敏感性（测试段池化每笔期望的边际均值）

| 维度 | 取值 | 组合数 | 边际每笔期望 | 训练段边际 |
| --- | --- | --- | --- | --- |
| `supertrend_factor` | 1.0 | 150 | -0.5007% | -0.4689% |
| `supertrend_factor` | 1.5 | 150 | -0.4945% | -0.4665% |
| `supertrend_factor` | 2.0 | 150 | -0.5050% | -0.4663% |
| `supertrend_factor` | 2.5 | 150 | -0.4577% | -0.4407% |
| `supertrend_factor` | 3.0 | 150 | -0.3788% | -0.4394% |
| `supertrend_factor` | 4.0 | 150 | -0.1871% | -0.5849% |
| `supertrend_factor` | 5.0 | 150 | -0.2379% | -0.5416% |
| `atr_period` | 7 | 210 | -0.4180% | -0.5202% |
| `atr_period` | 10 | 210 | -0.4134% | -0.5023% |
| `atr_period` | 14 | 210 | -0.3906% | -0.4801% |
| `atr_period` | 21 | 210 | -0.3861% | -0.4718% |
| `atr_period` | 30 | 210 | -0.3644% | -0.4602% |
| `sl_atr_mult` | 0.75 | 210 | -0.4298% | -0.4850% |
| `sl_atr_mult` | 1.0 | 210 | -0.4102% | -0.4813% |
| `sl_atr_mult` | 1.5 | 210 | -0.3924% | -0.4919% |
| `sl_atr_mult` | 2.0 | 210 | -0.3850% | -0.4915% |
| `sl_atr_mult` | 3.0 | 210 | -0.3551% | -0.4848% |
| `target_r` | 1.5 | 175 | -0.4023% | -0.4628% |
| `target_r` | 2.0 | 175 | -0.4119% | -0.4885% |
| `target_r` | 2.5 | 175 | -0.3989% | -0.4949% |
| `target_r` | 3.0 | 175 | -0.3893% | -0.4961% |
| `target_r` | 4.0 | 175 | -0.3805% | -0.4971% |
| `target_r` | 6.0 | 175 | -0.3842% | -0.4821% |

### 出场/过滤叠加效果（训练段前 8 名上）

| 基准结构 | 叠加 | 测试段每笔期望 | 测试段 CI | 判定 | 测试段账户中位 |
| --- | --- | --- | --- | --- | --- |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | adx30 | +0.6694% | [-0.1669, +1.5054] | inconclusive | +1.71% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | adx30 | +0.6542% | [-0.2720, +1.6095] | inconclusive | +1.10% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | adx30 | +0.6079% | [-0.2806, +1.5282] | inconclusive | +0.74% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | adx30 | +0.5640% | [-0.2286, +1.3510] | inconclusive | +1.66% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | adx30 | +0.4475% | [-0.2840, +1.1813] | inconclusive | +1.92% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | adx30 | +0.4137% | [-0.2838, +1.1091] | inconclusive | +1.00% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | adx30 | +0.0503% | [-0.6012, +0.8005] | inconclusive | -0.39% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | adx20 | -0.1232% | [-0.5117, +0.2950] | inconclusive | +0.15% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | ema200 | -0.1740% | [-0.7989, +0.4712] | inconclusive | -2.01% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | adx20 | -0.1880% | [-0.6172, +0.2612] | inconclusive | -1.41% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | adx20 | -0.2032% | [-0.5998, +0.2214] | inconclusive | -2.16% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | ema200 | -0.2545% | [-0.8140, +0.3429] | inconclusive | -2.22% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | adx20 | -0.2798% | [-0.7162, +0.1375] | inconclusive | -1.46% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | adx20 | -0.2875% | [-0.6370, +0.0699] | inconclusive | -3.08% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | adx30 | -0.2908% | [-1.1815, +0.6692] | inconclusive | -0.80% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | ema200_hold48h | -0.2936% | [-0.7763, +0.2067] | inconclusive | -1.98% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | trail12 | -0.2991% | [-0.4767, -0.1191] | negative | -3.96% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | trail12_hold48h | -0.2991% | [-0.4767, -0.1191] | negative | -3.96% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | trail20 | -0.3072% | [-0.5092, -0.1018] | negative | -4.56% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | ema200 | -0.3134% | [-0.8362, +0.2148] | inconclusive | -1.41% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | trail12 | -0.3141% | [-0.4905, -0.1347] | negative | -4.45% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | trail12_hold48h | -0.3141% | [-0.4905, -0.1347] | negative | -4.45% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | ema200_hold48h | -0.3245% | [-0.8964, +0.2485] | inconclusive | -2.71% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | ema200_hold48h | -0.3408% | [-0.8357, +0.1724] | inconclusive | -2.59% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | hold48h | -0.3457% | [-0.6149, -0.0563] | negative | -4.57% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | hold240h | -0.3457% | [-0.6149, -0.0563] | negative | -4.57% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | cooldown0 | -0.3457% | [-0.6149, -0.0563] | negative | -4.57% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | breakout | -0.3548% | [-0.4975, -0.2088] | negative | -11.13% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | ema200_trail12 | -0.3568% | [-0.5968, -0.1360] | negative | -2.62% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | ema200_trail12 | -0.3570% | [-0.5679, -0.1468] | negative | -5.42% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | trail20 | -0.3578% | [-0.5591, -0.1528] | negative | -5.26% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | ema200 | -0.3586% | [-0.9378, +0.2148] | inconclusive | -3.88% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | hold48h | -0.3689% | [-0.6160, -0.1187] | negative | -4.94% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | hold240h | -0.3689% | [-0.6160, -0.1187] | negative | -4.94% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | cooldown0 | -0.3689% | [-0.6160, -0.1187] | negative | -4.94% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | ema200 | -0.3700% | [-0.9856, +0.2727] | inconclusive | -0.72% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | trail10 | -0.3738% | [-0.5231, -0.2244] | negative | -5.53% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | hold240h | -0.3739% | [-0.6767, -0.0722] | negative | -4.25% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | cooldown0 | -0.3739% | [-0.6767, -0.0722] | negative | -4.25% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | hold48h | -0.3746% | [-0.6780, -0.0731] | negative | -4.25% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | ema200_hold48h | -0.3770% | [-0.8795, +0.1354] | inconclusive | -4.35% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | hold48h | -0.3853% | [-0.6738, -0.0960] | negative | -5.58% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | ema200_trail12 | -0.3865% | [-0.6283, -0.1687] | negative | -3.53% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | hold240h | -0.3872% | [-0.6747, -0.0992] | negative | -5.58% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | cooldown0 | -0.3872% | [-0.6747, -0.0992] | negative | -5.58% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | trail10 | -0.3880% | [-0.5359, -0.2407] | negative | -5.94% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | adx20 | -0.3886% | [-0.7398, -0.0458] | negative | -4.84% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r4.0` | breakout_trail12 | -0.3978% | [-0.5054, -0.2899] | negative | -14.55% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | breakout | -0.4019% | [-0.5419, -0.2627] | negative | -11.62% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | trail20 | -0.4022% | [-0.6038, -0.1996] | negative | -6.40% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | trail12 | -0.4052% | [-0.5684, -0.2333] | negative | -7.68% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | trail12_hold48h | -0.4052% | [-0.5684, -0.2333] | negative | -7.68% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | ema200_trail12 | -0.4065% | [-0.6080, -0.1954] | negative | -4.41% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | breakout_trail12 | -0.4086% | [-0.5178, -0.2997] | negative | -14.75% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | breakout | -0.4098% | [-0.5414, -0.2671] | negative | -11.77% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | hold48h | -0.4109% | [-0.6574, -0.1706] | negative | -8.39% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | hold240h | -0.4109% | [-0.6574, -0.1706] | negative | -8.39% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | cooldown0 | -0.4109% | [-0.6574, -0.1706] | negative | -8.39% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | breakout_trail12 | -0.4193% | [-0.5281, -0.3094] | negative | -13.86% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | trail12 | -0.4216% | [-0.5825, -0.2484] | negative | -8.57% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | trail12_hold48h | -0.4216% | [-0.5825, -0.2484] | negative | -8.57% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | breakout_trail12 | -0.4247% | [-0.5240, -0.3241] | negative | -13.75% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | trail20 | -0.4275% | [-0.6371, -0.2251] | negative | -6.79% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | trail20 | -0.4281% | [-0.6253, -0.2165] | negative | -8.10% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | adx20 | -0.4289% | [-0.8971, +0.0271] | inconclusive | -3.51% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | trail12 | -0.4331% | [-0.5801, -0.2830] | negative | -9.36% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | trail12_hold48h | -0.4331% | [-0.5801, -0.2830] | negative | -9.36% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | breakout_trail12 | -0.4342% | [-0.5326, -0.3306] | negative | -17.54% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | trail10 | -0.4373% | [-0.5857, -0.2773] | negative | -9.57% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | breakout | -0.4397% | [-0.5852, -0.2920] | negative | -11.14% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | breakout | -0.4420% | [-0.5675, -0.3091] | negative | -15.47% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | ema200_trail12 | -0.4421% | [-0.6631, -0.2220] | negative | -6.36% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | breakout_trail12 | -0.4491% | [-0.5601, -0.3441] | negative | -15.74% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | trail10 | -0.4505% | [-0.6010, -0.2951] | negative | -9.66% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | trail20 | -0.4540% | [-0.6253, -0.2813] | negative | -10.79% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | ema200_trail12 | -0.4662% | [-0.6861, -0.2529] | negative | -5.88% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | ema200_hold48h | -0.4681% | [-0.9268, -0.0106] | negative | -3.69% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | breakout_trail12 | -0.4699% | [-0.5700, -0.3689] | negative | -16.82% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | ema200_trail12 | -0.4724% | [-0.6938, -0.2722] | negative | -6.08% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | ema200_hold48h | -0.4735% | [-1.0168, +0.1004] | inconclusive | -0.72% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | breakout | -0.4771% | [-0.6127, -0.3559] | negative | -15.63% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | breakout_trail12 | -0.4792% | [-0.5812, -0.3792] | negative | -18.31% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | adx20 | -0.4802% | [-0.8382, -0.1390] | negative | -4.99% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r4.0` | ema200 | -0.4832% | [-0.9873, +0.0419] | inconclusive | -4.40% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | breakout | -0.4834% | [-0.6280, -0.3416] | negative | -13.44% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | breakout | -0.4852% | [-0.6164, -0.3581] | negative | -16.13% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | trail12 | -0.4858% | [-0.6610, -0.3203] | negative | -9.82% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | trail12_hold48h | -0.4867% | [-0.6594, -0.3246] | negative | -9.82% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | trail12 | -0.4881% | [-0.6352, -0.3435] | negative | -11.27% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | trail12_hold48h | -0.4881% | [-0.6352, -0.3435] | negative | -11.27% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | trail12 | -0.4895% | [-0.6424, -0.3518] | negative | -10.95% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | trail12_hold48h | -0.4895% | [-0.6424, -0.3518] | negative | -10.95% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | hold48h | -0.4918% | [-0.7883, -0.1830] | negative | -8.15% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | hold240h | -0.4938% | [-0.7900, -0.1849] | negative | -8.15% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | cooldown0 | -0.4938% | [-0.7900, -0.1849] | negative | -8.15% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | hold48h | -0.4968% | [-0.7401, -0.2637] | negative | -9.60% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | hold240h | -0.4971% | [-0.7291, -0.2627] | negative | -9.60% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | cooldown0 | -0.4971% | [-0.7291, -0.2627] | negative | -9.60% |
| `supertrend_factor2.5_atr_period30_sl_atr_mult3.0_target_r6.0` | trail10 | -0.5009% | [-0.6252, -0.3735] | negative | -12.51% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | trail10 | -0.5028% | [-0.6371, -0.3638] | negative | -12.29% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | trail10 | -0.5056% | [-0.6554, -0.3476] | negative | -10.13% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | ema200 | -0.5253% | [-1.0878, +0.0752] | inconclusive | -3.81% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | ema200_trail12 | -0.5270% | [-0.7622, -0.2983] | negative | -6.32% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | ema200 | -0.5305% | [-1.1509, +0.0687] | inconclusive | -4.27% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | trail10 | -0.5324% | [-0.6572, -0.4071] | negative | -12.04% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | trail20 | -0.5421% | [-0.7163, -0.3721] | negative | -10.62% |
| `supertrend_factor2.5_atr_period14_sl_atr_mult3.0_target_r6.0` | ema200_hold48h | -0.5436% | [-1.0147, -0.0425] | negative | -5.28% |
| `supertrend_factor2.5_atr_period21_sl_atr_mult3.0_target_r6.0` | trail20 | -0.5522% | [-0.7260, -0.3835] | negative | -12.34% |
| `supertrend_factor3.0_atr_period21_sl_atr_mult3.0_target_r6.0` | ema200_hold48h | -0.5778% | [-1.1271, -0.0542] | negative | -3.03% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | hold240h | -0.5950% | [-0.8705, -0.2839] | negative | -6.93% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | cooldown0 | -0.5950% | [-0.8705, -0.2839] | negative | -6.93% |
| `supertrend_factor3.0_atr_period14_sl_atr_mult3.0_target_r6.0` | hold48h | -0.5967% | [-0.8712, -0.2858] | negative | -6.93% |

### 最终候选（含随机基线）

| 组合 | 测试段成交 | 每笔期望 | CI | 判定 | 做多 | 做空 | 账户中位 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0__hold240h` | 1744 | -0.3739% | [-0.6767, -0.0722] | negative | -0.0083% | -0.6781% | -4.25% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0__cooldown0` | 1744 | -0.3739% | [-0.6767, -0.0722] | negative | -0.0083% | -0.6781% | -4.25% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0__hold48h` | 1744 | -0.3746% | [-0.6780, -0.0731] | negative | -0.0083% | -0.6794% | -4.25% |

## 1h

训练段最优结构参数：

- 参数 `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0`，训练段 +0.3518% [-0.0381, +0.7413] inconclusive（848 笔）
- 同一组在测试段：-0.1772% [-0.6358, +0.3494] inconclusive（421 笔）
- 账户收益中位：训练 +1.49% → 测试 -0.58%

样本内→样本外传导：训练段与测试段每笔期望相关 **-0.341**（Spearman -0.335，n=1050）；测试段 positive 6 组，纯噪声下应有约 26 组。

### 收益最高的 5 组（全窗口账户收益中位，样本内）

这一列是「如果一年前就选中它」的结果，**含训练段**，所以天然偏高。右边两列是同一组在各段的真实表现：

| # | 组合 | 全窗口账户中位 | 训练段每笔期望 | 测试段每笔期望 | 测试段判定 |
| --- | --- | --- | --- | --- | --- |
| 1 | `supertrend_factor5.0_atr_period30_sl_atr_mult1.5_target_r6.0` | +2.90% | +0.1137% | -0.2452% | inconclusive |
| 2 | `supertrend_factor5.0_atr_period14_sl_atr_mult1.5_target_r6.0` | +2.61% | +0.1155% | -0.0722% | inconclusive |
| 3 | `supertrend_factor5.0_atr_period30_sl_atr_mult1.0_target_r6.0` | +1.85% | +0.1206% | +0.0107% | inconclusive |
| 4 | `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | +1.70% | +0.1569% | -0.2298% | inconclusive |
| 5 | `supertrend_factor5.0_atr_period14_sl_atr_mult2.0_target_r6.0` | +1.18% | -0.0297% | +0.0010% | inconclusive |

### 参数敏感性（测试段池化每笔期望的边际均值）

| 维度 | 取值 | 组合数 | 边际每笔期望 | 训练段边际 |
| --- | --- | --- | --- | --- |
| `supertrend_factor` | 1.0 | 150 | -0.3086% | -0.1650% |
| `supertrend_factor` | 1.5 | 150 | -0.2631% | -0.2425% |
| `supertrend_factor` | 2.0 | 150 | -0.1604% | -0.2474% |
| `supertrend_factor` | 2.5 | 150 | -0.0878% | -0.2808% |
| `supertrend_factor` | 3.0 | 150 | +0.0463% | -0.2836% |
| `supertrend_factor` | 4.0 | 150 | -0.1765% | -0.2047% |
| `supertrend_factor` | 5.0 | 150 | -0.2567% | -0.0900% |
| `atr_period` | 7 | 210 | -0.2013% | -0.2572% |
| `atr_period` | 10 | 210 | -0.1806% | -0.2435% |
| `atr_period` | 14 | 210 | -0.1585% | -0.2086% |
| `atr_period` | 21 | 210 | -0.1701% | -0.1855% |
| `atr_period` | 30 | 210 | -0.1516% | -0.1865% |
| `sl_atr_mult` | 0.75 | 210 | -0.1987% | -0.1863% |
| `sl_atr_mult` | 1.0 | 210 | -0.1587% | -0.1930% |
| `sl_atr_mult` | 1.5 | 210 | -0.1277% | -0.2098% |
| `sl_atr_mult` | 2.0 | 210 | -0.1334% | -0.2288% |
| `sl_atr_mult` | 3.0 | 210 | -0.2435% | -0.2634% |
| `target_r` | 1.5 | 175 | -0.1686% | -0.2333% |
| `target_r` | 2.0 | 175 | -0.1820% | -0.2409% |
| `target_r` | 2.5 | 175 | -0.1836% | -0.2294% |
| `target_r` | 3.0 | 175 | -0.1854% | -0.2134% |
| `target_r` | 4.0 | 175 | -0.1714% | -0.1995% |
| `target_r` | 6.0 | 175 | -0.1435% | -0.1812% |

### 出场/过滤叠加效果（训练段前 8 名上）

| 基准结构 | 叠加 | 测试段每笔期望 | 测试段 CI | 判定 | 测试段账户中位 |
| --- | --- | --- | --- | --- | --- |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | ema200_hold48h | +0.3297% | [-0.4232, +1.0674] | inconclusive | +0.62% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | ema200 | +0.1113% | [-0.3358, +0.5621] | inconclusive | +1.22% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | ema200_hold48h | +0.1008% | [-0.4285, +0.6717] | inconclusive | +0.33% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | ema200_trail12 | +0.0877% | [-0.2417, +0.4369] | inconclusive | +0.89% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | ema200_hold48h | +0.0851% | [-0.3970, +0.5954] | inconclusive | -1.10% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | adx20 | +0.0657% | [-0.3479, +0.5155] | inconclusive | -0.60% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | adx20 | +0.0592% | [-0.9375, +1.2303] | inconclusive | -0.71% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | ema200_hold48h | +0.0554% | [-0.3523, +0.4669] | inconclusive | +1.13% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | hold48h | +0.0393% | [-0.6228, +0.7357] | inconclusive | +0.43% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | ema200_hold48h | +0.0221% | [-0.5049, +0.5709] | inconclusive | -0.27% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | ema200 | +0.0123% | [-1.1479, +1.2215] | inconclusive | +0.40% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | hold48h | +0.0101% | [-0.4035, +0.4658] | inconclusive | +0.15% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | trail20 | +0.0067% | [-0.3233, +0.3353] | inconclusive | +0.51% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | ema200_hold48h | -0.0020% | [-0.6638, +0.7685] | inconclusive | +0.04% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | ema200_trail12 | -0.0108% | [-0.3850, +0.3813] | inconclusive | -0.37% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | hold240h | -0.0364% | [-0.3885, +0.3192] | inconclusive | -0.54% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | cooldown0 | -0.0364% | [-0.3885, +0.3192] | inconclusive | -0.54% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | cooldown0 | -0.0457% | [-0.9284, +0.8526] | inconclusive | +0.05% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | ema200 | -0.0499% | [-0.9031, +0.8433] | inconclusive | +0.27% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | ema200 | -0.0555% | [-0.6726, +0.5650] | inconclusive | -0.99% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | ema200_hold48h | -0.0560% | [-0.6779, +0.5444] | inconclusive | -0.45% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | hold48h | -0.0591% | [-0.5212, +0.4364] | inconclusive | -0.20% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | breakout | -0.0708% | [-0.3717, +0.2351] | inconclusive | -1.51% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | adx20 | -0.0744% | [-0.6386, +0.5623] | inconclusive | -1.99% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | hold240h | -0.0768% | [-1.0129, +0.8172] | inconclusive | -0.04% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | hold48h | -0.0822% | [-0.6498, +0.5140] | inconclusive | -0.70% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | trail12 | -0.0859% | [-0.3652, +0.1966] | inconclusive | -0.35% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | trail12_hold48h | -0.0859% | [-0.3652, +0.1966] | inconclusive | -0.35% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | hold48h | -0.0865% | [-0.4382, +0.2558] | inconclusive | +0.05% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | breakout | -0.0931% | [-0.2941, +0.1115] | inconclusive | -1.37% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | breakout | -0.0947% | [-0.3086, +0.1148] | inconclusive | -1.21% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | breakout_trail12 | -0.1017% | [-0.2935, +0.1018] | inconclusive | -0.82% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | breakout | -0.1077% | [-0.2488, +0.0394] | inconclusive | -4.23% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | breakout | -0.1120% | [-0.3146, +0.0887] | inconclusive | -2.86% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | trail20 | -0.1221% | [-0.4885, +0.2463] | inconclusive | -1.61% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | ema200_trail12 | -0.1221% | [-0.4782, +0.2468] | inconclusive | -0.57% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | breakout_trail12 | -0.1221% | [-0.3178, +0.0760] | inconclusive | -1.04% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | breakout_trail12 | -0.1225% | [-0.2600, +0.0221] | inconclusive | -3.09% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | trail20 | -0.1260% | [-0.6472, +0.4514] | inconclusive | -0.71% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | trail10 | -0.1273% | [-0.3823, +0.1309] | inconclusive | -1.29% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | trail20 | -0.1295% | [-0.6970, +0.4756] | inconclusive | -1.66% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | hold48h | -0.1368% | [-0.5918, +0.3261] | inconclusive | +0.14% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | breakout | -0.1404% | [-0.4010, +0.1220] | inconclusive | -0.86% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | ema200_hold48h | -0.1406% | [-0.7313, +0.4959] | inconclusive | -0.59% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.0_target_r6.0` | breakout_trail12 | -0.1459% | [-0.2537, -0.0310] | negative | -4.96% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | trail20 | -0.1541% | [-0.5277, +0.2550] | inconclusive | -1.33% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | breakout | -0.1566% | [-0.4280, +0.1386] | inconclusive | -1.71% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | hold240h | -0.1651% | [-0.6259, +0.3627] | inconclusive | -0.57% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | breakout_trail12 | -0.1697% | [-0.3101, -0.0296] | negative | -3.09% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | breakout_trail12 | -0.1750% | [-0.3347, -0.0098] | negative | -2.86% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | cooldown0 | -0.1819% | [-0.6424, +0.3462] | inconclusive | -0.57% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | ema200 | -0.1843% | [-0.8063, +0.4867] | inconclusive | -0.56% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | ema200_trail12 | -0.1845% | [-0.5662, +0.2176] | inconclusive | -0.97% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | trail20 | -0.1878% | [-0.5903, +0.2273] | inconclusive | -1.78% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | breakout_trail12 | -0.1891% | [-0.3868, +0.0213] | inconclusive | -1.57% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | breakout | -0.1969% | [-0.4439, +0.0805] | inconclusive | -1.19% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | trail10 | -0.1972% | [-0.4683, +0.0875] | inconclusive | -1.45% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | ema200 | -0.2023% | [-0.8048, +0.3879] | inconclusive | -1.21% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | breakout_trail12 | -0.2228% | [-0.4152, -0.0228] | negative | -2.04% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | trail12_hold48h | -0.2240% | [-0.5356, +0.0924] | inconclusive | -1.91% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | trail12 | -0.2255% | [-0.5360, +0.0924] | inconclusive | -1.91% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | trail12_hold48h | -0.2308% | [-0.5575, +0.0904] | inconclusive | -1.55% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | trail12 | -0.2316% | [-0.5555, +0.0865] | inconclusive | -1.55% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | trail10 | -0.2326% | [-0.5176, +0.0448] | inconclusive | -1.34% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | cooldown0 | -0.2339% | [-0.7708, +0.2694] | inconclusive | -1.84% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | hold48h | -0.2343% | [-0.7633, +0.2939] | inconclusive | -1.51% |
| `supertrend_factor5.0_atr_period21_sl_atr_mult1.5_target_r6.0` | hold240h | -0.2361% | [-0.7786, +0.2918] | inconclusive | -1.84% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | trail10 | -0.2460% | [-0.5336, +0.0615] | inconclusive | -1.24% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | hold240h | -0.2633% | [-0.8066, +0.2575] | inconclusive | -0.76% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | cooldown0 | -0.2714% | [-0.7896, +0.2674] | inconclusive | -0.76% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | ema200_trail12 | -0.2882% | [-0.7345, +0.1649] | inconclusive | -1.39% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | ema200_trail12 | -0.2888% | [-0.7356, +0.1724] | inconclusive | -1.44% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | trail20 | -0.2889% | [-0.7644, +0.1995] | inconclusive | -1.09% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | trail12_hold48h | -0.2911% | [-0.6239, +0.0459] | inconclusive | -1.70% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | hold48h | -0.2990% | [-0.8219, +0.2603] | inconclusive | -0.99% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | trail12_hold48h | -0.3045% | [-0.7058, +0.0896] | inconclusive | -1.52% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult1.5_target_r6.0` | adx20 | -0.3074% | [-0.8651, +0.2563] | inconclusive | -1.67% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | trail12 | -0.3122% | [-0.6552, +0.0281] | inconclusive | -1.97% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | adx20 | -0.3246% | [-1.1268, +0.5271] | inconclusive | -0.87% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | trail10 | -0.3259% | [-0.6831, +0.0249] | inconclusive | -0.92% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | trail10 | -0.3327% | [-0.6866, +0.0157] | inconclusive | -0.90% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | trail12_hold48h | -0.3601% | [-0.7399, +0.0137] | inconclusive | -1.52% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | trail20 | -0.3654% | [-0.8465, +0.1522] | inconclusive | -1.69% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | trail10 | -0.4020% | [-0.7517, -0.0418] | negative | -1.37% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | trail12_hold48h | -0.4067% | [-0.7792, -0.0384] | negative | -2.00% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | trail10 | -0.4074% | [-0.7556, -0.0515] | negative | -1.37% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | ema200_trail12 | -0.4128% | [-0.8830, +0.0892] | inconclusive | -0.98% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | trail12_hold48h | -0.4145% | [-0.7998, -0.0376] | negative | -2.29% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | ema200_trail12 | -0.4359% | [-0.8931, +0.0537] | inconclusive | -0.98% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult2.0_target_r4.0` | adx20 | -0.4382% | [-1.0356, +0.1955] | inconclusive | -1.61% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0` | trail12 | -0.4900% | [-0.8788, -0.1085] | negative | -2.68% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | trail12 | -0.4910% | [-0.8926, -0.0676] | negative | -1.42% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | trail12 | -0.4953% | [-0.8840, -0.1172] | negative | -2.56% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | trail12 | -0.5157% | [-0.9061, -0.1058] | negative | -1.74% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | hold240h | -0.5389% | [-1.2136, +0.1781] | inconclusive | -1.53% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0` | cooldown0 | -0.5513% | [-1.2092, +0.1470] | inconclusive | -1.26% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | hold240h | -0.5814% | [-1.4006, +0.2262] | inconclusive | -1.31% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | cooldown0 | -0.5875% | [-1.3768, +0.2934] | inconclusive | -1.31% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | adx20 | -0.6760% | [-1.6413, +0.3565] | inconclusive | -1.34% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | ema200 | -0.8422% | [-1.6870, +0.0300] | inconclusive | -1.70% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | hold240h | -0.8750% | [-1.5175, -0.2096] | negative | -2.12% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | cooldown0 | -0.8816% | [-1.4977, -0.2336] | negative | -2.12% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r3.0` | adx20 | -0.9691% | [-1.6760, -0.1321] | negative | -1.64% |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0` | ema200 | -1.2234% | [-2.1830, -0.2738] | negative | -1.78% |

### 最终候选（含随机基线）

| 组合 | 测试段成交 | 每笔期望 | CI | 判定 | 做多 | 做空 | 账户中位 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0__ema200` | 254 | -1.2234% | [-2.1830, -0.2738] | negative | -0.8304% | -1.6042% | -1.78% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0__ema200` | 279 | -0.0499% | [-0.9031, +0.8433] | inconclusive | -0.1731% | +0.0640% | +0.27% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0__ema200` | 255 | +0.0123% | [-1.1479, +1.2215] | inconclusive | +0.6869% | -0.6677% | +0.40% |

## 4h

训练段最优结构参数：

- 参数 `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r6.0`，训练段 +0.2525% [-0.4867, +0.9690] inconclusive（448 笔）
- 同一组在测试段：-0.8611% [-1.5884, -0.0631] negative（207 笔）
- 账户收益中位：训练 +0.27% → 测试 -0.78%

样本内→样本外传导：训练段与测试段每笔期望相关 **-0.090**（Spearman -0.109，n=1050）；测试段 positive 2 组，纯噪声下应有约 16 组。

### 收益最高的 5 组（全窗口账户收益中位，样本内）

这一列是「如果一年前就选中它」的结果，**含训练段**，所以天然偏高。右边两列是同一组在各段的真实表现：

| # | 组合 | 全窗口账户中位 | 训练段每笔期望 | 测试段每笔期望 | 测试段判定 |
| --- | --- | --- | --- | --- | --- |
| 1 | `supertrend_factor1.5_atr_period30_sl_atr_mult0.75_target_r2.5` | +4.51% | +0.1429% | +0.0190% | inconclusive |
| 2 | `supertrend_factor1.5_atr_period30_sl_atr_mult0.75_target_r3.0` | +4.09% | +0.0614% | +0.0405% | inconclusive |
| 3 | `supertrend_factor1.5_atr_period14_sl_atr_mult0.75_target_r2.5` | +3.17% | +0.0355% | +0.0616% | inconclusive |
| 4 | `supertrend_factor5.0_atr_period30_sl_atr_mult3.0_target_r6.0` | +3.15% | +1.2532% | +0.9554% | insufficient |
| 5 | `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | +3.12% | +0.1544% | +0.2683% | inconclusive |

### 参数敏感性（测试段池化每笔期望的边际均值）

| 维度 | 取值 | 组合数 | 边际每笔期望 | 训练段边际 |
| --- | --- | --- | --- | --- |
| `supertrend_factor` | 1.0 | 150 | -0.0123% | -0.2161% |
| `supertrend_factor` | 1.5 | 150 | -0.0359% | -0.0655% |
| `supertrend_factor` | 2.0 | 150 | -0.1662% | -0.1753% |
| `supertrend_factor` | 2.5 | 113 | -0.0619% | -0.3471% |
| `supertrend_factor` | 3.0 | 57 | -0.2276% | -0.3391% |
| `atr_period` | 7 | 125 | +0.0051% | -0.2248% |
| `atr_period` | 10 | 124 | -0.0430% | -0.2148% |
| `atr_period` | 14 | 125 | -0.0811% | -0.1829% |
| `atr_period` | 21 | 122 | -0.1514% | -0.2032% |
| `atr_period` | 30 | 124 | -0.1517% | -0.1992% |
| `sl_atr_mult` | 0.75 | 144 | -0.0496% | -0.1100% |
| `sl_atr_mult` | 1.0 | 142 | -0.1130% | -0.1822% |
| `sl_atr_mult` | 1.5 | 131 | -0.1475% | -0.2777% |
| `sl_atr_mult` | 2.0 | 113 | -0.0978% | -0.3054% |
| `sl_atr_mult` | 3.0 | 90 | +0.0161% | -0.1609% |
| `target_r` | 1.5 | 103 | -0.1573% | -0.2052% |
| `target_r` | 2.0 | 110 | -0.1637% | -0.2256% |
| `target_r` | 2.5 | 106 | -0.1241% | -0.2116% |
| `target_r` | 3.0 | 105 | -0.0945% | -0.2229% |
| `target_r` | 4.0 | 102 | -0.0240% | -0.2122% |
| `target_r` | 6.0 | 94 | +0.0810% | -0.1453% |

### 出场/过滤叠加效果（训练段前 8 名上）

| 基准结构 | 叠加 | 测试段每笔期望 | 测试段 CI | 判定 | 测试段账户中位 |
| --- | --- | --- | --- | --- | --- |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | breakout | +1.5367% | [+0.5490, +2.6277] | positive | +1.28% |
| `supertrend_factor2.0_atr_period21_sl_atr_mult3.0_target_r6.0` | breakout | +0.9566% | [+0.1730, +1.8261] | positive | +1.30% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r6.0` | cooldown0 | +0.7830% | [-0.0760, +1.7192] | inconclusive | +1.54% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r6.0` | breakout | +0.7000% | [-0.0717, +1.4781] | inconclusive | +1.27% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0` | cooldown0 | +0.6103% | [-0.1873, +1.4021] | inconclusive | +1.56% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r6.0` | hold240h | +0.5882% | [-0.2797, +1.3741] | inconclusive | +1.13% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0` | hold240h | +0.5873% | [-0.1484, +1.3395] | inconclusive | +1.29% |
| `supertrend_factor2.0_atr_period21_sl_atr_mult3.0_target_r6.0` | trail20 | +0.5482% | [-0.3217, +1.4861] | inconclusive | +1.27% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r6.0` | breakout | +0.5376% | [-0.0973, +1.2162] | inconclusive | +1.34% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r6.0` | ema200_trail12 | +0.5216% | [-0.2039, +1.2779] | inconclusive | +1.10% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r6.0` | trail20 | +0.4799% | [-0.4517, +1.4199] | inconclusive | +1.45% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | adx20 | +0.4747% | [-0.3708, +1.4214] | inconclusive | +0.91% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r6.0` | trail20 | +0.4624% | [-0.1949, +1.1760] | inconclusive | +0.83% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r6.0` | ema200_hold48h | +0.4530% | [-0.1179, +1.0436] | inconclusive | +0.92% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0` | ema200_trail12 | +0.4452% | [-0.2670, +1.1900] | inconclusive | +1.10% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r3.0` | breakout | +0.4408% | [-0.1938, +1.0661] | inconclusive | +0.48% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | breakout_trail12 | +0.4359% | [-0.1592, +1.0639] | inconclusive | +0.40% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | breakout | +0.4306% | [-0.1329, +0.9691] | inconclusive | +1.88% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0` | ema200_hold48h | +0.4273% | [-0.1432, +1.0017] | inconclusive | +0.92% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | ema200_hold48h | +0.3901% | [-0.1186, +0.9786] | inconclusive | +1.44% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | hold48h | +0.3629% | [-0.0650, +0.8034] | inconclusive | +1.53% |
| `supertrend_factor2.0_atr_period21_sl_atr_mult3.0_target_r6.0` | breakout_trail12 | +0.3525% | [-0.1493, +0.8621] | inconclusive | +0.70% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r6.0` | hold48h | +0.3219% | [-0.1533, +0.8027] | inconclusive | +1.08% |
| `supertrend_factor3.0_atr_period30_sl_atr_mult3.0_target_r6.0` | trail12_hold48h | +0.3165% | [-0.4373, +1.1357] | inconclusive | +0.54% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r6.0` | breakout_trail12 | +0.3157% | [-0.1157, +0.7585] | inconclusive | +0.72% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0` | breakout | +0.3157% | [-0.2232, +0.8960] | inconclusive | +0.68% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | hold240h | +0.3151% | [-0.3258, +1.0214] | inconclusive | +1.00% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0` | trail20 | +0.3070% | [-0.3090, +0.9368] | inconclusive | +0.46% |
| `supertrend_factor2.0_atr_period21_sl_atr_mult3.0_target_r6.0` | hold48h | +0.3031% | [-0.2450, +0.8938] | inconclusive | +0.60% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r6.0` | hold48h | +0.2881% | [-0.3072, +0.9230] | inconclusive | +0.45% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0` | hold48h | +0.2871% | [-0.1763, +0.7484] | inconclusive | +1.08% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | cooldown0 | +0.2683% | [-0.3845, +0.9257] | inconclusive | +0.97% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | ema200_hold48h | +0.2524% | [-0.0421, +0.5634] | inconclusive | +0.59% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r6.0` | trail12 | +0.2325% | [-0.2749, +0.7880] | inconclusive | +0.68% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0` | breakout_trail12 | +0.2203% | [-0.1917, +0.5939] | inconclusive | +0.47% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | breakout_trail12 | +0.2071% | [-0.1601, +0.5903] | inconclusive | +0.68% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | trail12 | +0.1936% | [-0.2450, +0.6495] | inconclusive | +1.40% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | ema200_trail12 | +0.1803% | [-0.1228, +0.5034] | inconclusive | +1.22% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | trail20 | +0.1800% | [-0.3354, +0.7455] | inconclusive | -0.04% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | ema200 | +0.1753% | [-0.1351, +0.5008] | inconclusive | +1.16% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | trail12_hold48h | +0.1676% | [-0.1939, +0.5523] | inconclusive | +0.90% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0` | trail12_hold48h | +0.1398% | [-0.2641, +0.5594] | inconclusive | +0.69% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | ema200_trail12 | +0.1378% | [-0.4052, +0.7605] | inconclusive | +0.50% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r6.0` | trail12_hold48h | +0.1346% | [-0.2668, +0.5448] | inconclusive | +0.69% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r6.0` | breakout_trail12 | +0.1144% | [-0.3901, +0.6457] | inconclusive | +0.32% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0` | trail12 | +0.0973% | [-0.3708, +0.5844] | inconclusive | +0.44% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r3.0` | hold48h | +0.0930% | [-0.4107, +0.6292] | inconclusive | +0.15% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r3.0` | trail20 | +0.0680% | [-0.6547, +0.8269] | inconclusive | +0.18% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0` | trail10 | +0.0630% | [-0.3293, +0.4640] | inconclusive | +0.78% |
| `supertrend_factor2.0_atr_period21_sl_atr_mult3.0_target_r6.0` | hold240h | +0.0492% | [-0.8943, +1.0937] | inconclusive | +0.46% |
| `supertrend_factor2.0_atr_period21_sl_atr_mult3.0_target_r6.0` | trail12 | +0.0480% | [-0.6076, +0.7122] | inconclusive | +0.15% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r6.0` | trail10 | +0.0326% | [-0.3996, +0.5028] | inconclusive | +0.04% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r3.0` | breakout_trail12 | -0.0151% | [-0.4392, +0.4112] | inconclusive | +0.07% |
| `supertrend_factor2.0_atr_period21_sl_atr_mult3.0_target_r6.0` | trail12_hold48h | -0.0245% | [-0.4754, +0.4528] | inconclusive | -0.02% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0` | trail10 | -0.0246% | [-0.4241, +0.4131] | inconclusive | +0.06% |
| `supertrend_factor2.0_atr_period21_sl_atr_mult3.0_target_r6.0` | trail10 | -0.0277% | [-0.6216, +0.6159] | inconclusive | -0.06% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | trail12 | -0.0287% | [-0.2658, +0.2008] | inconclusive | -0.18% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | hold48h | -0.0320% | [-0.2448, +0.1821] | inconclusive | +0.64% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | trail12_hold48h | -0.0385% | [-0.2545, +0.1768] | inconclusive | +0.62% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | breakout_trail12 | -0.0413% | [-0.2204, +0.1330] | inconclusive | -0.65% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | breakout | -0.0565% | [-0.2313, +0.1291] | inconclusive | -0.55% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | trail10 | -0.0728% | [-0.2924, +0.1420] | inconclusive | -0.25% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | hold240h | -0.0823% | [-0.3018, +0.1534] | inconclusive | -0.64% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | cooldown0 | -0.0823% | [-0.3018, +0.1534] | inconclusive | -0.64% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | trail20 | -0.0876% | [-0.3161, +0.1467] | inconclusive | -0.65% |
| `supertrend_factor1.5_atr_period30_sl_atr_mult1.0_target_r2.0` | adx20 | -0.1091% | [-0.4074, +0.1813] | inconclusive | -0.26% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r6.0` | trail12 | -0.1894% | [-0.8076, +0.4503] | inconclusive | +0.32% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r6.0` | trail12_hold48h | -0.1917% | [-0.6416, +0.2499] | inconclusive | -0.29% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r6.0` | trail10 | -0.1919% | [-0.7634, +0.4120] | inconclusive | -0.38% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r3.0` | trail12_hold48h | -0.2267% | [-0.6599, +0.2129] | inconclusive | -0.23% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r3.0` | trail10 | -0.3088% | [-0.8333, +0.2602] | inconclusive | -0.38% |
| `supertrend_factor2.0_atr_period21_sl_atr_mult3.0_target_r6.0` | cooldown0 | -0.3424% | [-1.3329, +0.7924] | inconclusive | -0.60% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r3.0` | hold240h | -0.3429% | [-1.1606, +0.5203] | inconclusive | +0.10% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r3.0` | trail12 | -0.3463% | [-0.9060, +0.2301] | inconclusive | -0.53% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r6.0` | hold240h | -0.3758% | [-1.2339, +0.5551] | inconclusive | +0.33% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r3.0` | cooldown0 | -0.4875% | [-1.3234, +0.3475] | inconclusive | -0.42% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r6.0` | cooldown0 | -0.8611% | [-1.5884, -0.0631] | negative | -0.78% |

### 最终候选（含随机基线）

| 组合 | 测试段成交 | 每笔期望 | CI | 判定 | 做多 | 做空 | 账户中位 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0__ema200` | 103 | +0.2663% | [-1.4788, +2.1467] | insufficient | +1.8552% | -1.7815% | +0.12% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0__ema200` | 140 | +0.0647% | [-1.1999, +1.5152] | insufficient | +1.0913% | -1.4303% | +0.18% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r3.0__ema200` | 114 | +0.3869% | [-1.1805, +2.1073] | insufficient | +2.0133% | -2.2006% | +0.33% |

## 数据覆盖上限（为什么没有「两年复核」）

| 周期 | 币数 | 最早K线 | 可用天数(中位) |
| --- | --- | --- | --- |
| 15m | 14 | 2025-12-31 | 302 天 |
| 1h | 16 | 2025-10-29 | 365 天 |
| 4h | 16 | 2025-10-29 | 365 天 |

按主流池逐币统计最长连续段：1h/4h 上限约 404 天、15m 约 302 天。因此不存在「两年窗口复核」，365 天窗口对 15m 实际只覆盖约 302 天。

## 边界扩展：最优解在默认网格之外

默认网格的四个维度边际是**单调**的（factor 越大、止损越宽、目标越远、ATR 越慢 → 每笔期望越好），最优点全部顶在网格边界上。于是往同一方向再推一档（factor 4/5/6/8 × atr 21/30/44 × sl 3/4/5 × target 6/8/10（108 组）），验证这是真趋势还是随机游走的端点：

| 周期 | 扩展组数 | 扩展后最好全窗口收益 | 该组训练段 | 该组测试段 | 测试段判定 | 相对默认网格最好值 |
| --- | --- | --- | --- | --- | --- | --- |
| 15m | 108 | +4.42% | +0.7941% | +0.5167% | inconclusive | +10.21pp |
| 1h | 108 | +2.80% | +0.2553% | +0.5715% | inconclusive | -0.10pp |
| 4h | 108 | +3.15% | +1.2532% | +0.9554% | insufficient | -1.36pp |

扩展只在 **15m** 上有实质改善（−5.80% → +4.42%）：15m 成本占比最高，减少交易频次的边际收益最大。1h / 4h 的最佳值反而略降，说明在交易本来就稀疏的周期上继续拉宽只是把样本变少、不是把策略变好。扩展后的 15m 最佳组合训练段 +0.79%、测试段 +0.52%，两段同号——但测试段 CI 仍跨 0，且随机方向基线（同组参数、20 种子）账户均值就有 +0.72%，14 个币里只有 2 个跑赢自己的随机分布。

## 最高收益组合的交易画像

参数：`supertrend_factor8.0_atr_period30_sl_atr_mult5.0_target_r10.0`（15m，全窗口 302 天）

| 指标 | 值 |
| --- | --- |
| 账户收益中位（302 天） | +4.42% |
| 为正币种 | 9/14 |
| 胜率中位 | 33.0% |
| 最大回撤中位 | -10.12% |
| 持仓暴露中位 | 53.5% |
| 平均持仓根数中位 | 165 根（≈41 小时） |
| 每币成交中位 | 90 笔 |

逐币：

| 币种 | 收益 | 成交 | 胜率 | 最大回撤 | 暴露 | 均持仓(根) |
| --- | --- | --- | --- | --- | --- | --- |
| NEARUSDT | +22.08% | 79 | 36.7% | -4.54% | 51.7% | 189.8 |
| SUIUSDT | +13.46% | 83 | 36.1% | -6.67% | 55.4% | 193.8 |
| DOTUSDT | +11.55% | 82 | 31.7% | -7.75% | 54.5% | 192.8 |
| LTCUSDT | +8.71% | 86 | 33.7% | -10.67% | 53.4% | 180.2 |
| BCHUSDT | +7.25% | 97 | 28.9% | -9.11% | 50.3% | 150.7 |
| XRPUSDT | +6.40% | 88 | 33.0% | -6.26% | 49.9% | 164.4 |
| ETHUSDT | +4.72% | 98 | 35.7% | -5.88% | 55.3% | 163.9 |
| BTCUSDT | +4.12% | 97 | 34.0% | -9.57% | 56.9% | 170.2 |
| SOLUSDT | +0.34% | 97 | 33.0% | -12.66% | 55.2% | 165.3 |
| AVAXUSDT | -1.23% | 96 | 34.4% | -12.29% | 53.6% | 162.2 |
| ADAUSDT | -1.71% | 88 | 26.1% | -13.38% | 53.9% | 177.9 |
| APTUSDT | -3.47% | 84 | 29.8% | -13.10% | 44.1% | 152.3 |
| DOGEUSDT | -6.66% | 91 | 33.0% | -14.36% | 51.6% | 164.6 |
| LINKUSDT | -7.15% | 95 | 29.5% | -12.93% | 49.2% | 150.2 |

这组参数已经不是「频繁翻转」的 Supertrend，而是一个**慢速趋势跟随**：band 宽到 8×ATR、止损 5×ATR，平均持仓约 41 小时，一半时间在场内。胜率只有 33%，靠 10R 的尾部盈利拉正期望。收益中位 +4.42% / 回撤中位 -10.1% → 收益回撤比约 0.44（按 0.5%/笔 的风险预算）。要放大到可观的年化只能加风险预算，回撤会同比例放大 —— 而它的统计显著性还没过线。

## 判读规则

- **positive** = 池化 CI 下界 > 0；**negative** = 上界 < 0；跨 0 = **inconclusive**。
- 池化成交少于 200 笔记 **insufficient**，不构成任何结论（少笔数的 bootstrap 会给假精确区间）。
- 训练段与测试段结论不一致时，一律以测试段为准；训练段好看但测试段翻负 = 拟合噪声。
- 本报告累计检验 3 个周期 × 1050 组结构参数 + 叠加层。95% 双侧区间下，若真 edge 为零，单侧「CI 下界 > 0」的假阳性期望是 2.5%；实测 positive 组数若低于这个期望，说明连运气都没站在策略这边。单个 positive 不足以支撑启用，必须同时满足：测试段 positive + 超随机 + 逐币一致 + 长窗口复现。

## 口径

- 信号收盘确认 → 次根开盘成交；双边费率与滑点；ATR 止损 / R 倍数止盈 / 反向翻转出场。
- 跨缺口逐段独立撮合，账户权益按段复利结转；段长 < 800 根丢弃。
- 训练段与测试段在切点处各自独立撮合，切点上的持仓被截断（影响可忽略但存在）。
- 池化成交超过 20000 笔时 bootstrap 确定性降采样，报告里标 `sampled`。

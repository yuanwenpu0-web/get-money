# Supertrend 寻优 — 长窗口稳健性复核

生成时间：2026-10-29T14:10:56+00:00　窗口：**365 天**　来源：`docs/supertrend-optimization-boundary.json`

候选组合按**训练段**前 3 名取（与主报告同一选择口径），换上更长窗口后重新做 70/30 切分。

## 15m（14 币）

| 组合 | 训练段每笔期望 | 测试段每笔期望 | 测试段 CI | 判定 | 测试段账户中位 |
| --- | --- | --- | --- | --- | --- |
| `supertrend_factor8.0_atr_period21_sl_atr_mult5.0_target_r10.0` | +0.9425% | +0.6959% | [-0.9349, +2.4919] | inconclusive | +1.22% |
| `supertrend_factor8.0_atr_period30_sl_atr_mult5.0_target_r10.0` | +0.7941% | +0.5167% | [-0.9682, +2.2353] | inconclusive | +0.99% |
| `supertrend_factor8.0_atr_period21_sl_atr_mult5.0_target_r8.0` | +0.7408% | +0.1622% | [-1.2543, +1.6624] | inconclusive | +0.31% |

## 1h（16 币）

| 组合 | 训练段每笔期望 | 测试段每笔期望 | 测试段 CI | 判定 | 测试段账户中位 |
| --- | --- | --- | --- | --- | --- |
| `supertrend_factor5.0_atr_period30_sl_atr_mult5.0_target_r6.0` | +0.6824% | -0.8824% | [-2.0496, +0.4729] | inconclusive | -0.72% |
| `supertrend_factor8.0_atr_period21_sl_atr_mult4.0_target_r10.0` | +0.6664% | -0.7094% | [-2.6337, +1.4817] | insufficient | -0.42% |
| `supertrend_factor5.0_atr_period44_sl_atr_mult5.0_target_r6.0` | +0.6538% | -0.2465% | [-1.5695, +1.1675] | inconclusive | -0.19% |

## 4h（16 币）

| 组合 | 训练段每笔期望 | 测试段每笔期望 | 测试段 CI | 判定 | 测试段账户中位 |
| --- | --- | --- | --- | --- | --- |
| `supertrend_factor8.0_atr_period21_sl_atr_mult4.0_target_r10.0` | +3.4193% | +1.0617% | [-3.0144, +5.9786] | insufficient | -0.36% |
| `supertrend_factor8.0_atr_period44_sl_atr_mult3.0_target_r10.0` | +3.3961% | -0.2465% | [-3.2879, +3.6380] | insufficient | -0.50% |
| `supertrend_factor8.0_atr_period44_sl_atr_mult3.0_target_r8.0` | +3.3403% | -0.2054% | [-3.3721, +3.7498] | insufficient | -0.50% |

## 读法

长窗口上仍然 positive 才算稳健；窗口变长后期望变差、CI 变宽跨 0，说明原窗口的「最优」是拟合。注意长窗口包含更多不同市场状态，样本内（训练段）也会随之变化，因此这里只看**测试段**。

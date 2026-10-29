# Supertrend 寻优 — 长窗口稳健性复核

生成时间：2026-10-29T14:05:37+00:00　窗口：**730 天**　来源：`docs/supertrend-optimization.json`

候选组合按**训练段**前 3 名取（与主报告同一选择口径），换上更长窗口后重新做 70/30 切分。

## 1h（15 币）

| 组合 | 训练段每笔期望 | 测试段每笔期望 | 测试段 CI | 判定 | 测试段账户中位 |
| --- | --- | --- | --- | --- | --- |
| `supertrend_factor5.0_atr_period7_sl_atr_mult3.0_target_r6.0__ema200` | +1.1729% | -1.1398% | [-2.0393, -0.0943] | negative | -1.64% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r3.0__ema200` | +0.8699% | -0.2229% | [-1.0331, +0.6006] | inconclusive | -0.86% |
| `supertrend_factor5.0_atr_period14_sl_atr_mult3.0_target_r6.0__ema200` | +0.9280% | -0.3001% | [-1.3121, +0.8127] | inconclusive | -0.38% |

## 4h（15 币）

| 组合 | 训练段每笔期望 | 测试段每笔期望 | 测试段 CI | 判定 | 测试段账户中位 |
| --- | --- | --- | --- | --- | --- |
| `supertrend_factor1.5_atr_period14_sl_atr_mult3.0_target_r4.0__ema200` | +1.2236% | -0.1369% | [-1.8021, +1.7183] | insufficient | -0.28% |
| `supertrend_factor1.5_atr_period14_sl_atr_mult2.0_target_r6.0__ema200` | +1.5197% | -0.1565% | [-1.4358, +1.1852] | insufficient | -0.20% |
| `supertrend_factor2.0_atr_period30_sl_atr_mult3.0_target_r3.0__ema200` | +1.4774% | +0.1512% | [-1.4857, +1.9793] | insufficient | +0.55% |

## 读法

长窗口上仍然 positive 才算稳健；窗口变长后期望变差、CI 变宽跨 0，说明原窗口的「最优」是拟合。注意长窗口包含更多不同市场状态，样本内（训练段）也会随之变化，因此这里只看**测试段**。

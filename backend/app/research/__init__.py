"""指标信号实验室：可重复的入场信号搜索、组合与样本外验证。

只做研究，不接触实盘下单路径。模块划分：

- `lab_indicators`  指标原语（全部因果计算，t 时刻只用 ≤t 的数据）
- `lab_signals`     入场信号生成器（长期/短期两组布尔数组）
- `lab_grid`        搜索空间清单（每个指标的参数区间与步长）
- `lab_sim`         以 rsi50_cross 统一出场的撮合器
- `lab_stats`       评价指标、显著性、随机基线
- `lab_experiment`  单指标遍历 / 组合搜索 / 滚动验证的编排
"""

from __future__ import annotations

__all__ = [
    "lab_indicators",
    "lab_signals",
    "lab_grid",
    "lab_sim",
    "lab_stats",
    "lab_experiment",
]

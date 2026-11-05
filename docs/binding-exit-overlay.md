# 绑定层出场覆盖：支撑线指标 + 止盈止损方法

> 工程实现说明，**非 edge 判定**。本文不产出任何策略有效性结论；`support_r` 目前
> 的定位是「可配置、可回测、默认可关」的试验通道，不是已验证配置。
> 生成日期：2026-11-05 ｜ 相关代码：`backend/app/strategy/{exits,support_levels}.py`

## 1. 目标与边界

`strategy_bindings` 原来只表达「哪套参数 × 哪些币 × 什么周期」。本轮把它扩成
「哪套参数 × 哪些币 × 什么周期 × **怎么止损止盈**」，于是同一策略可以同时跑：

| 绑定 | 参数 | 出场 |
| --- | --- | --- |
| `absorption_reversal × research_* × 1h` | 脚本默认 | 策略自算 |
| `absorption_reversal × research_* × 1h` | 收紧 delta | ATR 2× 止损 + 2.5R 止盈 |
| 同上（另一条池） | 同上 | 枢轴池支撑位下方 0.25ATR 止损 + 2R 止盈 |

边界（违反即报错，不会静默降级）：

- **专用撮合策略**（`opening_candle_breakout`、`rgd_donchian_v2`、`simple_zigzag_double`）
  的止损止盈由 broker 按价格路径自推，覆盖层不生效 → 绑定校验直接 400。
- **研究流程策略**（`research_high/low_turnover_high_vol`，`tags=("research",)`）的
  `signals()` 方向恒为 0、由专用模拟流程按时长开仓；覆盖层会把「本来不该开仓」的行
  变成可执行信号 → 同样拒绝。
- `supertrend_current` 在 `cl_enabled=true`（订单闭环）时出场由闭环状态机决定 →
  拒绝；关闭闭环后可用。
- 成本（`fee_rate`/`slippage`）仍然全局统一，出场配置不允许覆盖成本（H3）。

## 2. 数据模型

`strategy_bindings` 新增三列（`db.py::_ADDED_COLUMNS` 幂等补列，不重建表）：

| 列 | 取值 | 说明 |
| --- | --- | --- |
| `exit_method` | `strategy` / `atr_r` / `support_r` | 默认 `strategy`，与改造前完全等价 |
| `support_indicator` | `support_levels.METHODS` 的 key | `support_r` 必填 |
| `exit_params` | JSON 对象 | 方法参数，缺省键由注册表补全 |

解析时以保留键 `exit` 注入有效参数：
`{"method": "support_r", "support": "pivot_pool", "params": {...}}`。
这样回测（`binding_overrides` → `runner`）、模拟盘/巡检（`active_bindings` → `engine`）、
订单管理（`resolve_order_params`）读到的都是同一份配置，不存在「回测一套、实盘一套」。

## 3. 出场方法

| code | 止损 | 止盈 | 参数（默认 / 区间） |
| --- | --- | --- | --- |
| `strategy` | 策略自算 | 策略自算 | — |
| `atr_r` | `atr_mult × ATR%` | `target_r × 止损距离` | `atr_mult`(1.5, 0.2–10)、`target_r`(2.0, 0.2–20) |
| `support_r` | 支撑位下方 `buffer_atr × ATR` | `target_r × 实际止损距离` | 同上 + `buffer_atr`(0.25, 0–3)、`max_sl_atr`(3.0, 0.5–10) |

`support_r` 的回退规则：水平缺失、水平站错边（多头时不低于现价）、或距离超过
`max_sl_atr × ATR` 时，该根回退 ATR 止损，并在信号帧 `exit_fallback=1` 留痕。
做空用镜像的「最近上方阻力」；来源无阻力列时同样回退。

## 4. 生效链路

```
binding 三列 ─┬─ binding_params / validate_binding / binding_overrides / resolve_order_params
              └─ params["exit"] ── resolve_params（校验 + 补默认）
                                     │
evaluate_signals ─── spec.prepare → spec.signals → 统一信号帧
                                     │
                     exits.overlay_distances（新 sl/tp + 水平价 + 回退标记）
                                     │
                     signals.finalize_signal_frame（净盈亏比、成本过滤、杠杆、仓位、风险截断）
```

关键点：**换止损必须重算仓位**。`finalize_signal_frame` 从 `assemble_signal_frame`
里原样抽出，覆盖路径用新距离重新走一遍成本过滤与 `risk_per_trade_pct` 截断，
不会出现「止损改了、仓位没改」。上一轮打在 `block_reason` 上的 `cost_filter` /
`invalid_distance` 会先清空，否则新距离达标也会被旧标签挡住。

实时巡检原本只把末行切片喂给 `evaluate_signals`；启用覆盖后才改为整帧评估后取末行
（支撑位需要枢轴确认与滚动水平池，单行切片会让 `support_r` 全部回退）。默认
`method=strategy` 时仍是末行评估，性能与行为都不变。

## 5. 支撑线来源（11 种，与研究评估同源）

定义已上移到 `strategy/support_levels.py`，`research/support_level_eval.py` 改为转出，
保证「研究量具」与「实盘量具」是同一套：

`pivot_pool`、`mtf_pivot_1h`、`mtf_pivot_4h`、`plasmazero`、`sr_zones_wu`、
`sr_zones_mtf`、`edge_registry`、`pdh_pdl`、`pivot_map`、`trendline`、`donchian20`。

因果性：枢轴类用 Pine `ta.pivothigh(left, right)`（值在枢轴后 `right` 根才出现）或
上一完整周期水平，`level[t]` 只依赖 `t` 及之前的数据；信号是「第 t 根收盘决策、
第 t+1 根开盘成交」，因此 `level[t]` 在该时点已知。单测
`test_overlay_distance_is_causal` 用「截断末尾不影响历史行」做守卫。

## 6. 未经统计支持的声明

`docs/support-level-evaluation.{md,json}` 的结论是：11 种支撑位定义在触线做多净收益、
破位率、反弹幅度三项上都不优于随机价位（verdict `inconclusive`，随机基线分位≈0.86，
`likely_noise`）。因此：

- `support_r` 在 API/UI 上统一标注「未经统计支持」，`exit.verified=false`；
- 默认值保持 `strategy`，不改变任何在跑绑定；
- 要把它变成默认或写进报告，必须按手册 Gate 3–4 出**受控 A/B**：
  同一绑定、同一币池、同一窗口，唯一变量 `exit.method`，净收益 + bootstrap CI +
  多种子随机基线，`CI 下界 > 0` 才算 `pass`。

## 7. 验收

- 单测：`backend/tests/test_binding_exit.py`（15 项）覆盖注册表校验、专用流程拦截、
  ATR/支撑位距离、回退、最小距离下限、因果性、绑定贯通与 API 往返。
- 回归：`pytest backend/tests/test_bindings.py backend/tests/test_support_level_eval.py`
  全绿；`research/support_level_eval.py` 改为转出后口径未变。
- 前端：`npm run build` 通过；绑定表格新增「出场」列，编辑弹窗新增方法/支撑源/参数。
- 部署：后端改动需重启计划任务 `QuantDesk-GetMoney`（新列由 `init_db` 补），
  前端只需 `npm run build`。

## 8. 后续（未做）

1. 用 `runner` 走一条受控 A/B，比较 11 种支撑源 × 3 档 buffer 与 `atr_r` 基线；
2. `optimizer` 目前只对 `spec.grid` 里的键寻优，出场参数尚未进入搜索空间；
3. 若 A/B 未过 gate，`support_r` 保持「可配置但不启用」，并把结论登记到
   `docs/decisions/` 与本手册的止损线清单。

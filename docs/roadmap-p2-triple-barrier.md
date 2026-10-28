# P2 验证报告：triple-barrier 三分类标签

> 路线图中 P2 的目标是把 `future_return > 0` 纯方向二分类，换成与实盘 sl/tp/max_hold
> 出场对齐的 triple-barrier 三分类标签（+1 止盈 / -1 止损 / 0 横盘）。
> 本报告记录实现与实测结论。

## 1. 为什么做这一刀

原标签 `label = (future_return(horizon) > 0)` 有两个根本性错配：

1. **无视路径**：某 bar 之后先暴跌打掉止损、最后又涨回——二分类仍标 +1（"涨了"），
   模型学到"该做多"，但实盘这单必亏在止损上。
2. **与出场脱钩**：实盘出场是 `sl = atr*sl_atr_mult` / `tp = atr*tp_atr_mult` / `max_hold_bars`
   时间屏障（见 `execution.simulate` + `rules.evaluate_signals` 行 275-276），标签却用固定 horizon 后的净收益。

triple-barrier 的标签 = 实盘出场的真实结果，模型学到的"该做多"= "做多能止盈出场"。

## 2. 实现（最小侵入、向后兼容）

- `backend/app/features/indicators.py`：新增 `_triple_barrier_label()`（路径感知扫描，H 封顶 300 控成本）；
  `build_features` 增加 `barrier: dict | None` 参数，传则生成三分类标签，不传保持原二分类。
- `backend/app/ml/dataset.py`：`build_dataset` 透传 `barrier`。
- `backend/app/backtest/runner.py`：`run_backtest` 增加 `triple_barrier: bool = False`；
  当开启且策略为 `ml_ensemble` 时，用 resolve 后的 `sl/tp/max_hold` 构造 barrier 传入；
  `predict` 在三分类下取 `class==1`（止盈）列的概率作"做多概率"，二分类/Mock 自动回退 `[:, 1]`。
- `backend/scripts/edge_diagnostic.py`：增加 `--triple-barrier` 开关，可同窗口对比两种标签。

默认行为完全不变（barrier=None → 二分类），既有 387 个测试全绿。

## 3. 标签分布（400 天，sl=1.5 / tp=2.5 / H=120）

| 币种 | 行数 | +1 止盈 | -1 止损 | 0 横盘 |
|---|---:|---:|---:|---:|
| BTCUSDT | 9546 | 0.390 | 0.609 | 0.001 |
| ETHUSDT | 9546 | 0.385 | 0.614 | 0.002 |
| SOLUSDT | 9546 | 0.364 | 0.635 | 0.000 |
| DOGEUSDT | 9546 | 0.342 | 0.655 | 0.003 |

分布健康，无退化（不是全 0 或全 -1）。横盘类极少是因为 H=120 内绝大多数 bar 都会触到 sl 或 tp。

## 4. 二分类 vs 三分类（同窗口、同币种、真实成本）

| 币种 | 模式 | 成交 | 净% | 随机% | 跑赢随机 | 95% CI |
|---|---|---:|---:|---:|---|---|
| BTCUSDT | 二分类 | 61 | −6.36 | −6.20 | 否 | [−1.03, +0.05] 跨0 |
| BTCUSDT | 三分类 | 58 | **−4.38** | −1.01 | 否 | [−0.98, +0.17] 跨0 |
| ETHUSDT | 二分类 | 55 | +0.62 | −7.73 | 是 | [−0.81, +0.41] 跨0 |
| ETHUSDT | 三分类 | 61 | +0.01 | −0.07 | 是 | [−0.61, +0.64] 跨0 |
| DOGEUSDT | 二分类 | 59 | +0.62 | −7.53 | 是 | [−0.50, +0.74] 跨0 |
| DOGEUSDT | 三分类 | 53 | −2.71 | −5.72 | 是 | [−0.88, +0.36] 跨0 |

## 5. 结论（gate 未翻转）

- **标签更正确，但单靠它翻不了盘**。BTC 净收益 −6.36% → −4.38% 改善明显；
  ETH/DOGE 三分类略退于二分类。三者 bootstrap CI **仍跨 0**，无法判定存在正 edge。
- 根因是 P1 已诊断出的事实：**400 天窗口整体无正 edge + 成本吸收率高**（费用吸收率 0.8~2.5）。
  标签只是"学习目标"，模型要有足够信息才能区分 +1/−1——而当前 26 个特征全是单币种时序，
  缺量价结构、形态、横截面相对强弱（见路线图 P3）。
- `ml_ensemble` 信号里 `ml_weight=0.6`、开多门槛 `min_score` 偏高，模型概率只是"强度"，
  标签改进被规则分支稀释，进一步压低了 P2 的边际收益。

## 6. 下一步

- **保留 `--triple-barrier` 开关，默认关**（不破坏既有回测基线）。它是 P3/P4 的"正确学习目标地基"，
  等特征与横截面补上后组合验证再决定是否默认开启。
- 真正的收益杠杆在 **P3（量价结构 + 横截面强弱）**：让模型有信息区分止盈 vs 止损。
- 已知 TODO：`significance.information_coefficient` 在三分类下语义错位（止盈概率 vs future_return 相关性弱，
  报告显示 IC=None）。应改为"止盈概率 vs 是否止盈标签"，但不阻塞主结论。

复现：
```powershell
.\.venv\Scripts\python.exe backend/scripts/edge_diagnostic.py --days 400 --symbols BTCUSDT ETHUSDT DOGEUSDT
.\.venv\Scripts\python.exe backend/scripts/edge_diagnostic.py --days 400 --symbols BTCUSDT ETHUSDT DOGEUSDT --triple-barrier
```

# Regime-Gated Donchian v2 [RGD] — 本地移植交付报告

> 交付日期：2026-10-31　主周期 15m / HTF 4h　状态：**已完整实现，默认关闭（`default_enabled=False`）**
> 溯源基线：`data/rgd-v2-source.pine`（Pine v6，237 行）
> 规格来源：`docs/rgd-porting-contract.md`（逐行 1:1 映射）+ `docs/rgd-spec.md`（范围与门禁）
> 回测数据：`docs/rgd-backtest.md` / `.json` / `-G1..G4.json`（99 币 × 180 天，2026-05-03 → 2026-10-30）

---

## 1. 结论

策略已按 Pine 脚本**逐行 1:1 移植**并可在本地回测库运行，出入场时序、状态机、R 倍数管理全部对齐。全市场 180 天检验：

- **合约多空（G1/G3）：负贡献且统计显著**，池化每笔期望 −0.1665%，95% CI [−0.2150%, −0.1161%] 全负，重采样为正比例 0.0%。
- **现货仅多（G2/G4）：不可判定**，池化每笔期望 −0.0267%，CI [−0.1086%, +0.0593%] 跨 0；但账户收益中位数 −5.10%、超随机占比仅 21%，**没有任何正 edge 证据**。
- **亏损集中在做空**：做空 −0.3024%（CI [−0.3585%, −0.2415%] 全负），做多 −0.0267%（跨 0）。
- 处置：保持 `default_enabled=False`，**不进实盘绑定、不参与自动巡检资金竞争、不做参数微调挽救**（样本内调优大概率是噪声，与项目 P0–P10 的既有结论一致）。

## 2. 交付清单

| 类型 | 文件 | 行数 | 职责 |
|---|---|---|---|
| 新增 | `backend/app/features/rgd_indicators.py` | 176 | Wilder ATR / percentrank / rolling Pearson / tension 状态机 / state machine |
| 新增 | `backend/app/features/rgd.py` | 280 | 通道·效率·张力·相关量能·防守·入场条件·入场价位 |
| 新增 | `backend/app/features/rgd_htf.py` | 70 | `attach_htf_levels` 因果 join |
| 新增 | `backend/app/strategy/specs/rgd_donchian_v2.py` | 180 | spec 注册（32 参数 1:1、`direction=0` 强制专用管线） |
| 新增 | `backend/app/backtest/rgd_orders.py` | 122 | 现金账户 / 开平仓 / 部分平 / 盯市原语 |
| 新增 | `backend/app/backtest/rgd_execution.py` | 212 | `simulate_rgd` 逐 bar 时序编排 |
| 新增 | `backend/scripts/rgd_backtest.py` | 222 | 4 组矩阵编排入口 |
| 新增 | `backend/scripts/rgd_backtest_worker.py` | 约 200 | 取数 + HTF attach + 切段 + 单币 4 组 |
| 新增 | `backend/scripts/rgd_baselines.py` | 124 | `randomize_rgd` / 池化 bootstrap / 组汇总 |
| 新增 | `backend/scripts/rgd_backtest_report.py` | 约 180 | markdown 渲染（仓位级 + 腿级两张表） |
| 新增 | `backend/tests/test_rgd_features.py` | 355 | A1–A11 / A20 / A21 / A23 / A25 / A27 / A28 |
| 新增 | `backend/tests/test_rgd_execution.py` | 367 | A12–A19 / A22 / A24 / A26 / A29 / A30 |
| 修改 | `backend/app/strategy/specs/__init__.py` | +2 | 导入即注册 |
| 修改 | `backend/app/backtest/runner.py` | +约 80 | `RGD` 常量 + `run_rgd_backtest` + 早路由 |
| 修改 | `backend/tests/test_strategy.py` | +1 | 内置策略集合同步 `rgd_donchian_v2` |
| 文档 | `docs/rgd-porting-contract.md` / `rgd-spec.md` / 本文件 | — | 契约 / Spec / 交付报告 |

全部新增源文件 ≤300 行（最大 280）。

## 3. 移植严格性

### 3.1 逐行映射与裁决

12 个脚本分节全部落到具体函数，细节见契约 §3。16 条歧义裁决中 12 条严格对齐、4 条为显式偏差并已登记 OPEN-DECISIONS：

| 类别 | 内容 |
|---|---|
| 严格对齐 | 自适应 Donchian 通道（`high[1]` 前置窗口）、`ta.percentrank`（前 length 根不含当前 / `<=` / 分母 length）、Wilder ATR（SMA 播种，不用 ewm）、`tensionStep` 显式状态机 + `nz` 自愈、`ta.correlation` 仿射不变、acceptance 加权 `defL/defS` 完整公式、state 0/1/−1 转移优先级、入场四条件与 `prevState` 门控、R 倍数定仓 |
| 已登记偏差 | `syminfo.mintick` 用 `1e-8` 护栏（无每币 tick 数据）、EMA/corr 预热 na 语义、TP1 成交后不 re-arm、同根 TP1 与 stop 双触时 stop 优先 |

### 3.2 关键时序（已逐条测试）

1. **信号收盘下市价单、次根开盘成交**；反向 entry 反手（数量 = 旧 + 新，两腿费用分别计），同向 entry 在 `pyramiding=0` 下忽略。
2. **入场 bar 裸奔**：成交当根不挂 exit，止损最早下一根生效。
3. **TP1 限价部分平当前仓 50%**，此后剩余仓仅由 trailing 管理。
4. **保本滞后一根**：本根 `high/low` 触及目标只把 `stopLvl` 提到 `entryRef`，新 stop 下一根才可成交。
5. **TP1 双状态**：`tp1_seen`（Pine `tp1Hit`，价格曾触，只驱动保本）与 `tp1_filled`（真实部分成交，只有它撤单）。混用会在价格触达但订单未成交时永久撤掉 TP1，导致 50% 部分平永不发生——这是实现期发现并修复的 P0。
6. **trailing 单调**：多头 `max` 不降、空头 `min` 不升。
7. **不强制收盘平仓**：末根未平仓按盯市计入净值并写入 `equity.attrs["open_position"]`。
8. **HTF 不重绘**：`searchsorted(htf_open + step <= base_open, side="right") - 1`，同一 4h 桶内 16 根 15m 的 hh/ll 唯一；未完成桶剔除。

## 4. 回测结果（99 币 × 180 天 × 4 组）

| 组 | 方向 | fee | slippage | 仓位 | 腿 | TP1 腿 | TP1 触发率 | 每笔期望 | 95% CI | 判定 | 账户收益中位 | 超随机 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **G1** | 多空 | 0.0004 | 0.0002 | 18497 | 22356 | 3860 | 20.9% | **−0.1665%** | [−0.2150%, −0.1161%] | **negative** | −16.72% | 13% |
| G2 | 仅多 | 0.0004 | 0.0002 | 9118 | 11336 | 2219 | 24.3% | −0.0267% | [−0.1086%, +0.0593%] | inconclusive | −5.10% | 21% |
| G3 | 多空 | 0.0005 | 2 ticks 近似 | 18497 | 22356 | 3860 | 20.9% | −0.1781% | [−0.2272%, −0.1279%] | negative | −17.31% | 12% |
| G4 | 仅多 | 0.0005 | 2 ticks 近似 | 9118 | 11336 | 2219 | 24.3% | −0.0374% | [−0.1194%, +0.0484%] | inconclusive | −5.28% | 21% |

- **多空分解（G1）**：做多 −0.0267%（inconclusive，9118 笔）、**做空 −0.3024%（negative，9380 笔）**。
- **出场结构**：仓位级最终出场几乎全是 `stop_loss`（18496/18497）；腿级看 TP1 实际触发 3860 腿、覆盖 20.9% 的仓位。原因是 `aggregate_positions` 合并同仓多腿时用后一腿覆盖前一腿，仓位级会掩盖已发生的 TP1。报告同时给两套口径。
- **随机基线**：G1 超随机占比 13%（远低于 95% 门槛），重采样为正比例 0.0%。
- 正收益币种 14/99，平均最大回撤 −26.57%，平均持仓占比 29.3%，单币账户收益均值 −15.16%。

## 5. 质量门禁

| 门禁 | 结果 |
|---|---|
| 全量回归 | RGD 相关全绿（特征 17 + 撮合 13 + 注册 19）；全量 638 passed / 1 failed，唯一失败在 `tests/test_catalog_port_w3d.py`（catalog 并行工作线，失败用例在 `roc_advanced` / `flowtrend` 间波动），与 RGD 零文件交集 |
| RGD 专项测试 | 30 passed（特征 17 + 撮合 13） |
| 单文件行数 | 全部 ≤300（最大 280） |
| HTF 静默零成交 | 已杜绝：整帧 attach → 切段 → 每段 assert `rgd_htf_ok`，缺列 raise |
| 随机基线失效 | 已杜绝：`randomize_rgd` 循环位移整组决策列，非 `randomize_directions` no-op |
| 通用管线误接 | 安全失败：`direction≡0` + `block_reason`，误调通用 `simulate` 零成交 |
| 数据可复现 | 两轮 99 币回测每笔期望一致（−0.1665%） |

## 6. 已知限制与偏差（全部登记 `docs/decisions/OPEN-DECISIONS.md`）

| 编号 | 内容 | 影响 |
|---|---|---|
| OPEN-01 | `syminfo.mintick` 固定 `1e-8` 护栏 | 仅除零保护；真实 tick 远大于此值，结果不变 |
| OPEN-02 | EMA/corr/sum 预热 na 语义未与 TV 逐根对照 | `ready` 由 `rank_len=100` 主导，决策无影响 |
| OPEN-03 | `closedtrades` 是否对部分平 bump | 已证等价（`coolOk` 仅门控 pullback 且要求 flat） |
| OPEN-04 | `strategy.equity` 持仓期费用归属 | 影响定仓 q 的极小量（≤费量级） |
| OPEN-05 | TP1 成交后 TV 是否 re-arm | 采用一次性 latch（符合脚本 `tp1Hit` 意图） |
| OPEN-06 | 同根 TP1/stop 优先级 | 保守 stop 优先，已测试锁定 |
| OPEN-07 | `slippage=2 ticks` 每币换算 | 用 `round_price` 精度约定代理，**近似非逐位** |
| OPEN-08 | 15m 历史仅 298 天（57 币 ≥180 天） | 报告已列 skipped，零成交≠无信号 |
| OPEN-09 | 1d 主周期 → `1W` 不在白名单 | 显式 raise，不静默降级 |
| OPEN-10 | 实盘持仓管理未迁移 | 见第 7 节 |

## 7. 实盘边界（Out-of-Scope）

TP1 限价挂单 + 动态 trailing 的持仓管理**只在回测 broker 中实现**，未迁移到 `trading/order_manager`。因此本策略不进实盘绑定。若将来要实盘化，需按 `docs/rgd-porting-contract.md` §5 的 broker 状态机在订单层重建（含 TP1 部分平、保本上移、逐 bar trailing），并单独立项。

## 8. 复现命令

```bash
cd D:/UGit/get-money/backend
# RGD 专项测试
"D:/UGit/get-money/.venv/Scripts/python.exe" -m pytest tests/test_rgd_features.py tests/test_rgd_execution.py -q
# 正式回测（99 币 × 180 天 × 4 组 × 10 种子，约 110s）
"D:/UGit/get-money/.venv/Scripts/python.exe" scripts/rgd_backtest.py --interval 15m --days 180 --universe all --seeds 10
# 全量回归
"D:/UGit/get-money/.venv/Scripts/python.exe" -m pytest tests -q
```

---

> **免责声明**：以上内容基于公开数据和量化分析，仅供参考，不构成投资建议。市场有风险，投资需谨慎。任何投资决策应结合个人风险承受能力、资金状况和投资目标独立判断，必要时咨询持牌专业机构。过往表现不预示未来收益。

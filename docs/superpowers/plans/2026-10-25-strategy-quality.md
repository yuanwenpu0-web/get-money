# 策略质量与收益验证 Implementation Plan

> **For agentic workers:** Implement the checked tasks in this session. The user's request authorizes implementation; no additional approval or delegation is needed.

**Goal:** 改善扣费后的策略表现，降低无效交易，并提供可复现的样本外对比。

**Architecture:** 把信号、过滤与仓位计算抽成模拟交易和回测共用的纯函数。统一使用逐根 K 线撮合，训练/验证/测试按时间隔离，参数只在验证段选择，测试段只报告结果。

**Tech Stack:** Python、pandas、SQLAlchemy、scikit-learn、FastAPI、Vue 3。

**Spec:** 用户要求“完善项目，策略部分优化提高收益”；按收益与回撤平衡执行，不承诺样本外必然盈利。

## Global Constraints

- 保留现有策略代码、API 与本地模拟交易方式。
- 不改用户现有订单、资金、数据库中的策略参数或模型文件。
- 双边手续费/滑点、杠杆、保证金、止损止盈必须参与评估。
- 搜索参数不得读取最终测试集；不自动发布验证集选出的参数。

## Review Focus

- 未收盘/过期行情、缺失指标不得触发新仓。
- 预测跨度跨过训练边界时必须清除对应样本。
- 跳空和同根止盈止损按保守顺序撮合，不能用当根收盘信号在当根开盘成交。
- 零交易、全盈利、首笔亏损与采样尾点必须如实呈现。
- 已开仓订单不得使用开仓前 K 线重复触发止损，止损后须冷却。

### Task 1: 共享策略与风险规则

**Files:** `backend/app/strategy/rules.py`、`engine.py`、`defaults.py`、`config.py`、`backend/tests/test_strategy.py`

**Interfaces:** `resolve_params(code, state, overrides)`；`evaluate_signals(data, code, params, prob_up=None)` 返回逐行方向、评分、拒绝原因和风险仓位。

- [x] 写测试：规则/模拟信号一致、震荡趋势过滤、强趋势回归过滤、手续费门槛、缺失值、单笔风险上限。
- [x] 运行测试确认失败，实施共享规则与模拟端集成，再运行测试。

### Task 2: 可信回测与参数验证

**Files:** `backend/app/backtest/{runner,execution}.py`、`backend/app/ml/{dataset,trainer}.py`、`backend/app/features/indicators.py`、`backend/tests/test_backtest.py`

**Interfaces:** `simulate(data, signals, params, initial, fee, slippage, interval)`；`purged_split(data, split, horizon)`；`run_backtest(..., optimize=False)`。

- [x] 写测试：下一根开盘成交、OHLC 止损/跳空、费用核对、风险/冷却、标签清除、测试段隔离、最后权益点、无亏损盈亏比。
- [x] 实现统一撮合、数据质量诊断、独立验证段参数选择与同口径基线。
- [x] 运行测试和 BTC/ETH 真实本地历史数据对比，保存所有结果及局限。

### Task 3: 模拟持仓与产品集成

**Files:** `backend/app/trading/{paper,order_manager,service}.py`、`backend/app/api/{strategies,backtests,settings}.py`、`schemas.py`、`frontend/src/views/{Backtest,Strategies,Settings}.vue`、`backend/tests/test_trading.py`、`README.md`

- [x] 写测试：重复巡检不重放开仓前行情、同策略信号管理、成本后风险限制、参数校验。
- [x] 展示过滤原因、风险参数、验证/测试区间、基线和优化对比；修复对应模拟执行边界。
- [x] 运行完整 pytest、Python 编译与前端构建；记录验证结果。

## 验证结果

- 34 项 pytest 通过，Python 编译通过；修改范围内 Ruff F/I 检查通过。
- Vue/Vite 生产构建成功；仍有原有的大 bundle 提示。
- Playwright 在隔离的只读数据库预览服务中验证回测提交、报告展开、过滤解释、过期行情预演及风险设置显示；没有执行巡检或资金操作。
- BTC/ETH × 三策略 × 30/40 天共 12 组结果保存在 `docs/strategy-benchmark.md` 和同名 JSON。大多数组合仍负收益；不得将改善等同于稳定盈利。
- 回测与模拟盘共用策略、仓位和保护退出函数；模拟盘仍存在巡检延迟、开仓所在部分K线等执行近似，已在 README 说明。

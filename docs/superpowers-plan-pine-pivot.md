# Pine 策略改造实施计划

**Goal:** 将项目入口转为 Pine 源码分析、逐 K 线对照和本地回测，保留独立执行的演进路径。

**Architecture:** 新建受限且拒绝不支持语义的 Pine 执行路径，与旧策略评分链隔离。复用行情模块；禁止旧调度器启动和误导性实盘入口。交易接入需使用同一策略语义且验证后才开放。

**Tech Stack:** Python、FastAPI、pandas、Vue。

**Spec:** 本次用户需求（聊天中提供）。

## Global Constraints

- 不调用交易所真实下单接口；不对未经对照的策略宣称一致。
- 保留已有行情和策略数据；依赖脱钩后清理旧代码与文档。

## Review Focus

- 未支持 Pine 语法必须报错。
- 缺失、重复或错位 K 线不能被视为对照通过。
- 订单按收盘信号和下一根开盘撮合。
- 旧配置不能在启动时自动触发旧交易。
- 实盘尚未接入时不能显示为可用。

### Task 1: 转换与验证
**Files:** `backend/app/pine/`、对应测试。
- [x] 实现有限子集及拒绝路径，逐 K 线对照并运行测试。

### Task 2: 回测及 API
**Files:** `backend/app/pine/backtest.py`、`backend/app/api/pine.py`、对应测试。
- [x] 实现收盘信号次开成交回测及 API；验证手续费、反手与数据错误。

### Task 3: 本地模拟
**Files:** `backend/app/pine/paper.py`、`backend/app/pine/paper_runner.py`。
- [x] 持久化模拟账户并显式控制公开行情轮询；注明历史回放与实时成交的差别。

### Task 4: 入口与文档
**Files:** `backend/app/main.py`、`frontend/src/`、`README.md`。
- [x] 移除旧自动调度入口，提供 Pine 工作台和真实能力说明。
- [x] 测试 API、数据库启动、前端构建及新流程；真实 TradingView 样本尚未提供，币安网络探测超时。

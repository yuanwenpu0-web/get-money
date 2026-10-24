# Quant Desk · 虚拟币量化交易系统

本地运行的 BTC / ETH 量化交易系统：从已有项目导入 1 分钟K线，计算技术指标与 ML 方向概率，按评分自动分配仓位、推荐杠杆、给出止损止盈，通过定时任务持续巡检持仓，并在前端查看订单、收益与系统状态。当前所有订单均为**本地模拟撮合**，不触达真实资金。

## 1. 功能

| 模块 | 能力 |
| --- | --- |
| 行情 | 从 `D:\UGit\nofx-new` 的 PostgreSQL `market_klines` 只读导入 BTC/ETH 各周期K线，落本地 SQLite；也可切换币安公共接口 |
| 指标 | TA-Lib 为主，pandas-ta-classic 与纯 pandas 逐级回退；RSI、MACD、ATR、SuperTrend、ADX、布林带、EMA、量能 Z 等 25 维特征 |
| 模型 | LightGBM / XGBoost 二分类，预测未来 N 根K线方向；按时间前 80% 训练、后 20% 评估；无模型时规则回退 |
| 策略 | ML 趋势共振、SuperTrend 趋势跟随、RSI 均值回归；评分 = ML 概率 + 指标共振；评分决定开仓与否、仓位占比、是否重仓 |
| 风控 | ATR 止损止盈、最小止损止盈距离、移动止损、趋势中扩展止盈、最大持仓K线数、最小持仓冷却、最大杠杆 |
| 回测 | vectorbt `from_signals`，计入手续费/滑点，包含 ATR 止损止盈、信号冷却、评分仓位；输出收益、回撤、夏普、胜率、盈亏比、手续费、买入持有基准 |
| 交易 | 本地模拟开仓/平仓/反手，保证金与权益核算，盈利自动累加总资金，支持追加/提取/重置资金 |
| 巡检 | APScheduler 每 N 秒同步最新K线、重算信号、检查止损止盈、移动止损、信号反转、超时平仓 |
| 前端 | Vue 3 + Element Plus + ECharts 深色界面：总览、订单、策略、回测、模型、行情、设置 7 个页面 |

## 2. 目录

```
get-money/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI 路由
│   │   ├── backtest/       # vectorbt 回测与回退撮合
│   │   ├── features/       # 指标与特征工程
│   │   ├── market/         # nofx 导入、币安客户端、K线存储
│   │   ├── ml/             # 数据集、训练、预测
│   │   ├── strategy/       # 策略引擎与评分
│   │   ├── trading/        # 资金、模拟撮合、订单管理、调度
│   │   └── main.py
├── frontend/               # Vue 3 前端
├── data/                   # 运行后生成：trading.db、models/、fernet.key
├── requirements.txt
└── README.md
```

## 3. 存储方案

- **K线 / 订单 / 模型元数据 / 权益快照：SQLite + WAL**（`data/trading.db`）。
  - 无需额外服务、单文件可备份；`(symbol, interval, open_time)` 唯一索引保证幂等增量写入；2 个币种 1 分钟线量级约 5 万行/40 天，SQLite 足够快。
- **模型文件：`data/models/*.pkl`**，数据库只存路径与指标。
- **API Secret：Fernet 加密**，密钥保存在 `data/fernet.key`，接口不返回明文。
- **nofx 数据源：只读导入**，不会写入原项目数据库。

## 4. 启动

### 4.1 后端

```powershell
cd D:\UGit\get-money
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

- 首次启动会自动从 nofx 导入 BTC/ETH 1m 历史K线（约 19 秒），随后后台自动训练缺失模型。
- 首次启动完成后，后续启动只做增量导入，速度很快。
- 如需改数据源连接：

```powershell
$env:NOFX_DATABASE_URL = "postgresql://postgres:admin@127.0.0.1:5432/nofx_lite"
$env:NOFX_SYMBOL_PREFIX = "BINANCE_"
```

### 4.2 前端

生产构建（后端直接托管）：

```powershell
cd D:\UGit\get-money\frontend
npm install
npm run build
# 浏览器打开 http://127.0.0.1:8010
```

开发模式（热更新，Vite 已配置代理到 8010）：

```powershell
cd D:\UGit\get-money\frontend
npm run dev
# http://127.0.0.1:5173
```

后端端口 8010（8000 可能被其他服务占用）；如需更改，修改 `vite.config.js` 代理目标与启动命令。

## 5. 使用流程

1. **行情**：确认 nofx 数据源在线，选择币种/周期导入；本地数据表可看到行数与起止时间。
2. **模型**：选择币种、周期、LightGBM/XGBoost、预测跨度与天数，训练后可在列表对比 AUC/准确率。
3. **策略**：启用策略、编辑 JSON 参数、预演信号；查看方向、评分、上涨概率、推荐杠杆、仓位占比、止损止盈。
4. **回测**：选择策略/模型/天数/手续费/滑点/评分阈值，跑 vectorbt 回测；查看权益曲线与成交明细。
5. **设置**：配置币安 API（仅校验，当前模拟）、资金调整、风控参数、定时巡检开关。
6. **总览 / 订单**：查看权益曲线、当前持仓、未实现/已实现盈亏、胜率、系统日志；订单页可改止损止盈、平仓、反手。

## 6. 策略与仓位规则

- 方向原始强度 `signed ∈ [-1, 1]`：
  - `ML 趋势共振`：`0.6 × (2×prob_up−1) + 0.4 × (0.5×SuperTrend + 0.3×EMA + 0.2×MACD)`
  - `SuperTrend 趋势跟随`：SuperTrend 方向 + EMA 排列 + MACD，ADX 低时减半
  - `RSI 均值回归`：RSI < 30 做多、> 70 做空，强度按偏离程度线性放大
- 评分 `score = 0.5 + 0.5 × |signed|`，达到 `min_score` 才开仓。
- 仓位占比：评分在 `min_score` 与 `heavy_score` 之间线性从基础仓位升到上限的 70%；达到 `heavy_score` 直接取单仓上限（重仓）。
- 杠杆：`0.015 / ATR占比` 后按评分置信度缩放，限制在 `[1, max_leverage]`；止损距离越窄、波动越低，杠杆越高。
- 止损止盈：`max(ATR × 倍数, 价格 × 最小距离)`；持仓盈利超过 0.8 ATR 后移动止损；强信号时扩展止盈。
- 巡检：反向信号需满足最小持仓K线数才反手；超过最大持仓K线数强制平仓。

## 7. 回测口径

- 训练集为前 75%，测试集为后 25%，测试集只用于样本外评估。
- 手续费与滑点默认单边 0.04% / 0.02%，双边计入。
- vectorbt 多空分别建组合再合并权益，避免反转仓位与百分比仓位不兼容。
- 排序指标包含：总收益、最大回撤、夏普（按日收益年化）、胜率、盈亏比、手续费、买入持有收益、信号K线数。
- 结果会明确标记 `engine=vectorbt`；若 vectorbt 不可用则回退到内置顺序撮合，并在指标里标记 `engine=fallback`。

## 8. 主要 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/settings` | 系统设置、数据源状态、依赖状态、调度状态 |
| PUT | `/api/settings` | 保存交易对、周期、风控、调度参数 |
| POST | `/api/settings/capital` | 追加 / 提取 / 重置总资金 |
| GET/PUT | `/api/settings/credentials` | 读取掩码 / 加密保存币安密钥 |
| POST | `/api/settings/credentials/test` | 测试币安连接与签名 |
| GET | `/api/market/klines` | 本地K线（ECharts 格式） |
| GET | `/api/market/sources` | 本地 / nofx / 币安数据源状态 |
| POST | `/api/market/sync` | 从 nofx 或币安导入K线 |
| POST | `/api/market/preview` | 预演信号与指标快照 |
| GET/PUT | `/api/strategies`、`/api/strategies/{code}` | 策略列表 / 启停与参数 |
| POST | `/api/strategies/run-once` | 立即执行一次巡检 |
| GET/POST | `/api/models`、`/api/models/train` | 模型列表 / 训练 |
| POST | `/api/backtests` | 创建回测（后台线程执行） |
| GET | `/api/backtests/{id}` | 回测结果、权益曲线、成交 |
| GET | `/api/trading/dashboard` | 总览：权益、持仓、信号、日志 |
| GET | `/api/trading/orders` | 订单列表与汇总 |
| PATCH | `/api/trading/orders/{id}` | 修改止损止盈 |
| POST | `/api/trading/orders/{id}/close` | 平仓 |
| POST | `/api/trading/orders/{id}/reverse` | 反手 |
| GET | `/api/trading/equity` | 权益快照 |
| POST | `/api/trading/run-once` | 手动巡检 |

## 9. 验证

```powershell
cd D:\UGit\get-money\backend
..\.venv\Scripts\python.exe -m compileall app

# 后端健康
Invoke-RestMethod http://127.0.0.1:8010/api/health

# 前端构建
cd ..\frontend
npm run build
```

## 10. 已知边界

- **仅模拟盘**：币安签名接口（账户/下单/撤单）已封装，但未接入真实下单流程。
- 1 分钟级别噪声大、手续费占比高；默认策略与模型仅用于流程验证，不保证盈利，回测为负也可能出现。
- 模型准确率受限于样本与市场状态，需定期重新训练；系统会在缺失模型时自动训练，但不会自动滚动更新。
- 未实现资金费率、盘口深度、逐仓/全仓强平、部分成交、真实滑点模型；杠杆为简化线性放大。
- 不构成投资建议。

## 11. 风险提示

加密货币合约交易风险极高，可能导致全部本金损失。请先在模拟盘长期验证，再考虑任何实盘接入；实盘前需补充交易所精度、最小下单量、限价/市价策略、异常重试、风控熔断与审计日志。

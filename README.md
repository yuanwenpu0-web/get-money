# Quant Desk · 虚拟币量化交易系统

本地运行的 BTC / ETH 量化交易系统：从已有项目导入 1 分钟K线，计算技术指标与 ML 方向概率，按评分自动分配仓位、推荐杠杆、给出止损止盈，通过定时任务持续巡检持仓，并在前端查看订单、收益与系统状态。当前所有订单均为**本地模拟撮合**，不触达真实资金。

## 1. 功能

| 模块 | 能力 |
| --- | --- |
| 行情 | 从 `D:\UGit\nofx-new` 的 PostgreSQL `market_klines` 只读导入 BTC/ETH 各周期K线，落本地 SQLite；也可切换币安公共接口 |
| 指标 | TA-Lib 为主，pandas-ta-classic 与纯 pandas 逐级回退；RSI、MACD、ATR、SuperTrend、ADX、布林带、EMA、量能 Z 等 25 维特征 |
| 模型 | LightGBM / XGBoost 二分类，预测未来 N 根K线方向；按时间前 80% 训练、后 20% 评估；无模型时规则回退 |
| 策略 | ML 趋势共振、SuperTrend 趋势跟随、RSI 均值回归；评分 = ML 概率 + 指标共振；评分决定开仓与否、仓位占比、是否重仓 |
| 风控 | ATR 止损止盈、移动止损、含成本的单笔风险预算、持仓时限、平仓冷却、过期行情保护、最大杠杆 |
| 回测 | 统一顺序撮合；隔离训练/验证/测试、验证集选参、同口径基线对比，计入杠杆、双边成本、风险预算、止损止盈与冷却 |
| 交易 | 本地模拟开仓/平仓/反手，保证金与权益核算，盈利自动累加总资金，支持追加/提取/重置资金 |
| 巡检 | APScheduler 每 N 秒同步最新K线、重算信号、检查止损止盈、移动止损、信号反转、超时平仓 |
| 前端 | Vue 3 + Element Plus + ECharts 深色界面：总览、订单、策略、回测、模型、行情、设置 7 个页面 |

## 2. 目录

```
get-money/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI 路由
│   │   ├── backtest/       # 顺序撮合、独立验证与样本外对比
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
4. **回测**：选择策略/模型/天数/手续费/滑点/评分阈值，运行顺序回测，可启用验证集选参；查看收益对比、权益曲线、过滤原因与成交明细。
5. **设置**：配置币安 API（仅校验，当前模拟）、资金调整、风控参数、定时巡检开关。
6. **总览 / 订单**：查看权益曲线、当前持仓、未实现/已实现盈亏、胜率、系统日志；订单页可改止损止盈、平仓、反手。

## 6. 策略与仓位规则

模拟盘和回测共用 `strategy/rules.py` 计算信号、过滤原因和仓位。策略 JSON 覆盖全局风控配置；手续费与滑点统一由全局设置或回测请求指定，策略不能覆盖撮合成本。策略编辑器显示所有可编辑的生效参数，删除某字段可恢复继承。

- **ML 趋势共振**：模型方向概率与 SuperTrend/EMA/MACD 加权；须满足 ADX、快慢趋势同向和模型方向优势。`require_model=true` 时没有模型不能开仓。
- **趋势跟随**：ADX 不足或快慢趋势矛盾时暂停开仓，减少震荡市场追单。
- **均值回归**：RSI 超买超卖同时由布林带位置确认；ADX 高于 `max_range_adx` 时避免逆势入场。
- **成本过滤**：ATR 必须达到双边成本的 `min_atr_cost_ratio` 倍，扣费后的目标盈利/止损损失必须达到 `min_net_reward_risk`。评分是信号强度，不是校准后的胜率。
- **风险仓位**：评分给出保证金上限，再按止损距离、杠杆和双边费用/滑点缩小仓位。默认 `risk_per_trade_pct=0.005`，即常规止损预算为权益的 0.5%；实际成交时重新检查可用资金与风险。跳空可能突破预算。
- **退出与冷却**：固定目标止盈；盈利达到 `trail_activation_atr` 后移动止损。`trailing_enabled` 同时控制回测和模拟盘。最小持仓仅约束反向信号退出，不阻止保护性止损；最大持仓超时退出；默认平仓后冷却 10 根K线。
- **模拟巡检**：持仓只接收原策略信号；按时间处理尚未检查的已收盘K线，不重复使用开仓前或已处理行情。过期行情、重复信号不能开新仓。开仓所在的部分K线无法重建开仓后的路径，保守地从下一根完整K线检查。

关键参数：`risk_per_trade_pct`、`cooldown_bars`、`regime_filter`、`cost_filter`、`min_atr_cost_ratio`、`min_net_reward_risk`、`max_atr_pct`、`min_model_edge`、`trailing_enabled`、`trail_activation_atr`。错误字段、非有限数值和越界值会在策略保存时被拒绝。旧 `backtest_trailing` 写入接口兼容映射到 `trailing_enabled`，历史保存值不再用于切换回测引擎。

## 7. 回测口径

- **不选参**：前 75% 训练、后 25% 样本外测试；**启用选参**：前 60% 训练、中间 15% 验证、后 25% 测试。边界移除至少一个预测跨度的样本，防止训练标签跨入后续区间。
- 仅 ML 策略训练模型；规则策略不训练无用模型。验证选参使用固定小集合，按净收益减去 0.5 倍回撤幅度排序，至少要求 10 笔验证成交；不足时保留当前配置。测试数据不参与选择，结果不会自动写回模拟盘。
- 所有回测标记 `engine=sequential`。收盘计算信号，下一根开盘成交；多空共享一个资金账户；使用评分仓位、杠杆、风险上限、手续费与滑点。
- OHLC 触发止损止盈；开盘跳空止损按开盘价成交；无法判断同根路径时止损优先；移动止损在收盘更新、下一根生效；平仓当根不重新开仓。
- 基线关闭行情和成本过滤，保留相同模型、风险限额和撮合规则，以比较过滤及选参效果。它与旧 vectorbt 回测不属于相同口径，不应直接比较收益。
- 数据检查排除未收盘及无效K线；数据缺口后重置指标预热。缺口中持仓在下一可见开盘平仓，并在报告提示缺失根数与滞后时长。不能据此推断缺失行情中的真实盈亏。
- 首笔损失计入回撤；无亏损时盈利因子返回 `null`，另用 `profit_factor_infinite` 标记无穷大；曲线降采样包含最后权益点。
- 回测的下一根开盘是假定及时执行。模拟盘仍按巡检时的信号价格加滑点撮合，调度延迟及开仓所在部分K线会造成差异；真实成交、资金费率和强平尚未模拟。

本次完整比较见 [策略验证报告](docs/strategy-benchmark.md)。它记录全部 12 组结果与数据质量；负收益即亏损，优于基线不代表已稳定盈利。

复现（数据库只读，不改订单、资金和模型）：

```powershell
.\.venv\Scripts\python.exe backend/scripts/benchmark_strategies.py --as-of "2026-10-25T06:00:00+00:00" --days 30 40
```

时间切分参考 [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) 的 gap 机制；前视偏差检查参考 [Freqtrade 官方说明](https://www.freqtrade.io/en/stable/lookahead-analysis/)。

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
..\.venv\Scripts\python.exe -m pytest tests -q

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

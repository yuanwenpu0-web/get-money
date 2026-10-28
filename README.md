# Quant Desk · 虚拟币量化交易系统（吸收反转单策略版）

本地运行的币安 U 本位合约交易系统：从 fapi 同步 1 分钟K线并在本地聚合其他周期，计算技术指标与订单流吸收特征，由**唯一策略「吸收反转 Absorption Reversal」**给出方向、评分、结构止损与目标，按评分分配仓位，通过定时巡检管理持仓。交易执行支持**模拟盘与币安 U 本位合约（测试网 / 主网）**（`execution_mode` 可切换，需自行承担 API Key 安全责任）。

> 本仓库已按「只保留能稳定执行的策略」收敛：`absorption_reversal` 之外的策略、ML 训练/预测、TradingView 研究端口、资金费率收割与早期探针脚本均已删除。策略默认 **`enabled=0`**，原因见第 6 节与 [吸收反转策略报告](docs/absorption-reversal-strategy.md)。

## 1. 功能

| 模块 | 能力 |
| --- | --- |
| 行情 | 币安 U 本位永续全量（`symbol_scope=all`，可切 `custom`）；1 小时内存缓存 + 失败回退；按 24h 成交额过滤低流动性合约（`min_quote_volume_usd`，0=不过滤）；只落 1m K线，5m/15m/1h/4h/1d 由本地按 UTC 边界聚合；支持 HTTP/SOCKS5 代理（`proxy_url`） |
| 指标 | TA-Lib 为主，pandas-ta-classic 与纯 pandas 逐级回退；RSI、MACD、ATR、SuperTrend、ADX、布林带、EMA、量能 Z 等基础特征 |
| 订单流 | `features/absorption.py`：`|delta|` 滚动最近秩百分位、无价格进展、摆动极值、拒绝影线、单边支配度五条件 + 3 根内收盘穿越实体的确认状态机；优先用 lower-tf intrabar delta，缺数据回退脚本自带的 estDelta 估计 |
| 资金费 | fapi 历史资金费入库（8 小时 1 条，幂等增量），asof 合并为K线特征（当前费率 / 年化 / z-score / 24h 累计），可选拥挤过滤 |
| 策略 | 仅 `absorption_reversal`；参数与 Pine 脚本 1:1（灵敏度档位、`legacy_gap_logic` 可复现脚本运算符优先级 bug） |
| 风控 | 结构止损（极值 ∓ 0.25 ATR）与 `min/max_sl_atr` 夹逼、2R 固定目标、1R 保本、30 根时间止损、含成本的单笔风险预算、平仓冷却、过期行情保护、最大杠杆 |
| 回测 | 顺序撮合；样本内/外切分（`train_horizon` 隔离跨度）、可选验证集选参、同口径基线对比、随机方向基线、bootstrap 期望值置信区间 |
| 交易 | 模拟盘与合约（测试网/主网）：开仓（杠杆、步长/最小名义价值校验、市价或 `limit_maker` postOnly）、交易所侧止损止盈、撤改单、reduceOnly 平仓、状态周期同步、反手；账户级熔断；本地保证金与权益核算、追加/提取/重置资金 |
| 巡检 | APScheduler 每 N 秒同步最新K线（默认最多 8 路并发、共享连接池）、单轮内复用指标帧、检查止损止盈/保本/时间止损/信号反转；周期日志记录币种数与各阶段耗时 |
| 指标目录 | 读取 `data/pine-collection`、`data/pine-crypto-collection` 的第三方 Pine 指标索引，支持搜索与源码查看（只读资料，不参与计算） |
| 前端 | Vue 3 + Element Plus + ECharts：总览、订单、策略、回测、行情、指标目录、设置；按需加载、独立空态与重试；订单/回测历史/成交/本地数据列表服务端分页（默认 20） |

## 2. 目录

```
get-money/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI 路由（backtests / indicators / market / settings / strategies / trading）
│   │   ├── backtest/       # 顺序撮合、指标、显著性检验
│   │   ├── features/       # 基础指标 + absorption 订单流特征
│   │   ├── market/         # 币安客户端、K线存储、metrics 观测
│   │   ├── ml/             # 数据集构建与 purged 切分（仅回测用，无训练/预测）
│   │   ├── strategy/       # coin_universe（主流币白名单）、rules、engine、defaults
│   │   ├── trading/        # 资金、模拟撮合、订单管理、调度、实盘同步
│   │   └── main.py
│   ├── scripts/            # absorption_backtest + 数据回补/迁移/校验脚本
│   └── tests/              # 后端测试
├── frontend/               # Vue 3 前端
├── docs/                   # 回测与研究报告（md + json）
├── data/                   # 运行后生成：trading.db、fernet.key
├── requirements.txt
└── README.md
```

## 3. 存储方案

- **数据库：PostgreSQL（推荐）或 SQLite，通过 `DATABASE_URL` 切换。**

  ```bash
  DATABASE_URL=postgresql+psycopg2://postgres:<密码>@127.0.0.1:5432/get_money
  ```

  - 未设置时回退到 `data/trading.db`；方言在启动时校验（仅支持 `sqlite` / `postgresql`）。
  - `app/db.py::make_engine()` 是唯一连接工厂：SQLite 加 `check_same_thread` 与 PRAGMA；PostgreSQL 走连接池、`pool_pre_ping` 与超时。回测脚本传 `read_only=True`，由数据库层拦截误写入。

- **从 SQLite 迁移到 PostgreSQL**（一次性）：

  ```bash
  createdb get_money
  ./.venv/Scripts/python.exe backend/scripts/migrate_sqlite_to_postgres.py
  ```

- **统一 1m 来源**：只下载写入 1m，其余周期在库内按 UTC 边界聚合。
  - 桶边界用 `open_time - open_time % step` 对齐（SQLite 整除与 PG NUMERIC 除法口径不同，混用会让桶整体错位）。
  - 缺失任意分钟的已结束周期被排除；连续当前周期仅供行情展示与浮盈估值，策略与回测仍过滤未收盘周期。
  - 增量巡检刷新共享 1m 来源，同币种多周期持仓只同步一次。
  - 补齐分钟历史：`.\.venv\Scripts\python.exe backend/scripts/backfill_minutes.py --days 40 --workers 4`。
- **Pine 指标资料**：只读 `data/pine-collection/`、`data/pine-crypto-collection/` 的 `index.json` 与源码，不入库；`data/` 被 Git 忽略。
- **API Secret：Fernet 加密**，密钥 `data/fernet.key`，接口不返回明文。

## 4. 启动

### Windows 后台运行

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start-background.ps1
```

通过任务计划程序 `QuantDesk-GetMoney` 运行，访问 http://127.0.0.1:8010；日志 `output/server.log`。停止：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\stop-background.ps1
```

### 后端

```powershell
cd D:\UGit\get-money
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

启动时自动建表、补齐默认状态与策略行（老库中已下线策略的配置行会被强制停用，且不再出现在策略列表/巡检中）。

### 前端

```powershell
cd D:\UGit\get-money\frontend
npm install
npm run build      # 后端直接托管，访问 http://127.0.0.1:8010
npm run dev        # 开发热更新，http://127.0.0.1:5173（代理到 8010）
```

## 5. 使用流程

1. **行情**：默认 U 本位永续全量，可在设置页切自定义列表；选择币种/周期同步历史，本地数据表显示行数与起止时间；`funding=true` 同时拉资金费历史。
2. **策略**：只有「吸收反转」一张卡；可启停、编辑 JSON 参数、预演信号（方向、评分、上涨概率、推荐杠杆、仓位占比、止损止盈、吸收快照）。
3. **回测**：选择币种/周期/天数/手续费/滑点/评分阈值，可启用验证集选参；查看收益对比、权益曲线、过滤原因、显著性区间与成交明细。
4. **设置**：币安 API（测试网校验测试网余额，主网校验合约账户）、执行模式、代理、资金调整、风控参数、定时巡检、流动性门槛、熔断参数、主单类型。
5. **总览 / 订单**：权益曲线、持仓、已实现/未实现盈亏、胜率、日志；订单页可改止损止盈、平仓、反手。
6. **指标目录**：搜索并查看 Pine 指标说明与源码（只读）。

## 6. 策略与仓位规则

模拟盘与回测共用 `strategy/rules.py`；手续费与滑点只由全局设置或回测请求指定，策略不能覆盖撮合成本。

- **吸收反转（`absorption_reversal`）**：放量单边主动成交（`|delta|` 位于滚动百分位之上）却推不出价格、在摆动极值留下拒绝影线 → 进攻方被困。**买方被吸收做空、卖方被吸收做多**。
  - 五条件：`|delta|` 最近秩百分位、无价格进展（实体 < `body_mult × ATR`）、摆动极值（`extreme_lookback`）、拒绝影线（`wick_mult`）、单边支配度（`dominance_pct`）。
  - 确认：`conf_mode=1` 要求后续 `conf_window_bars` 根内收盘穿越吸收 bar 实体；期间打穿吸收极值或超时作废。
  - 出场：结构止损 `极值 ∓ sl_buffer_atr × ATR`（`min_sl_atr`/`max_sl_atr` 夹逼）、`target_r=2R`、浮盈 1R 移保本、`max_hold_bars=30` 时间止损。
  - 仓位：脚本 `percent_of_equity=30`、`max_leverage=1`，双向交易；`min_score=0.75` 分层（默认全部通过，评分只用于仓位与分层）。
  - 白名单：仅在 `coin_universe.MAJOR_SYMBOLS`（BTC/ETH/SOL/BNB/XRP/DOGE/ADA/LINK/AVAX/LTC/DOT/SUI/TRX/NEAR/APT/BCH）上运行，与「资金路由」解耦。
  - **默认停用**：15m 主流币池化检验的每笔期望 95% 置信区间跨 0（不可判定），放宽灵敏度后转负。详见 `docs/absorption-reversal-strategy.md`。
- **风险仓位**：评分给出保证金上限，再按止损距离、杠杆与双边费用/滑点缩小；默认 `risk_per_trade_pct=0.02`（单笔预算 2% 权益），成交时复查可用资金。跳空可能突破预算。
- **巡检**：持仓只接收原策略信号；只处理尚未检查的已收盘K线，不复用开仓前/已处理行情；过期与重复信号不能开新仓。

关键参数：`delta_window`、`delta_pct`、`body_mult`、`extreme_lookback`、`wick_mult`、`dominance_pct`、`abs_gap_bars`、`legacy_gap_logic`、`conf_mode`、`conf_window_bars`、`direction_mode`、`trend_mode`、`session_start_hour`/`session_end_hour`、`sl_buffer_atr`、`target_r`、`be_trigger_r`、`min_sl_atr`/`max_sl_atr`、`max_hold_bars`、`cooldown_bars`、`risk_per_trade_pct`、`max_leverage`、`cost_filter`、`min_atr_cost_ratio`、`min_net_reward_risk`、`max_atr_pct`。未知字段、非有限数值与越界值在保存时被拒绝。

## 7. 回测口径

- **不选参**：前 75% / 后 25%；**启用选参**：前 60% 训练、中间 15% 验证、后 25% 测试，边界按 `train_horizon` 隔离。
- 验证选参使用固定小集合（评分门槛 0.60/0.65/0.70 与更严格的成本门槛），按净收益 + 0.5×回撤排序，成交笔数门槛 `max(30, 验证K线数 // 50)`；不足则保留当前参数。测试段不参与选择。
- 所有回测标记 `engine=sequential`：收盘出信号、下一根开盘成交；多空共享一个资金账户；计入评分仓位、杠杆、风险上限、双边成本与滑点。
- OHLC 触发止损止盈；跳空按开盘价；同根路径不明时止损优先；保本止损收盘更新、下一根生效；平仓当根不再开仓。
- **基线**：关闭 `regime_filter` / `cost_filter`，其余（成本、风控、撮合）完全相同。
- **随机入场基线**（`random_baseline`）：在基线信号上按固定种子对每个非零方向 50/50 翻转，入场时点、仓位、成本、风控完全一致；策略收益未超过它即说明方向无信息量。
- **显著性**：`bootstrap_expectancy_ci` 给出每笔期望的 95% 置信区间，跨 0 时报告标注 `inconclusive`——不得据此下结论。
- **曝光时间**：`exposure_pct`（持仓 bar 占比）与 `avg_bars_held` 三组都会给出，零成交为 0。
- 数据缺口后重置指标预热；缺口中持仓在下一可见开盘平仓并在报告提示。回测假定下一根开盘及时成交，与模拟盘巡检撮合存在差异；未模拟资金费率、盘口深度与强平。

复现（数据库只读）：

```powershell
.\.venv\Scripts\python.exe backend/scripts/absorption_backtest.py --interval 15m --days 30 --windows 3 --sensitivity medium --out docs/absorption-reversal-window93
```

报告：`docs/absorption-reversal-strategy.md`（Medium × 3 窗 × 30 天，88 笔，池化期望 +0.0704%，CI [−0.2456%, +0.4278%] → inconclusive）、`absorption-reversal-sens-high.*`（323 笔，期望 −0.0794%）。

## 8. 主要 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/settings` | 设置、数据源状态、依赖状态、调度状态 |
| PUT | `/api/settings` | 保存交易对、周期、风控、调度参数 |
| POST | `/api/settings/capital` | 追加 / 提取 / 重置总资金 |
| GET/PUT | `/api/settings/credentials` | 读取掩码 / 加密保存币安密钥 |
| POST | `/api/settings/credentials/test` | 测试币安连接与签名 |
| GET | `/api/market/klines` | 本地K线（ECharts 格式） |
| GET | `/api/market/symbols` | 可交易币种列表 |
| GET | `/api/market/funding` | 最近资金费记录 + 当前费率/年化/z-score/24h 累计 |
| GET | `/api/market/sources`、`/api/market/stats` | 本地数据分组分页 |
| POST | `/api/market/sync` | 从币安同步K线（可含资金费） |
| POST | `/api/market/preview` | 预演信号与指标快照 |
| GET | `/api/indicators`、`/api/indicators/{id}` | Pine 指标目录与详情 |
| GET/PUT | `/api/strategies`、`/api/strategies/{code}` | 策略列表 / 启停与参数 |
| POST | `/api/strategies/run-once` | 立即执行一次巡检 |
| GET | `/api/backtests` | 回测历史分页列表 |
| POST | `/api/backtests` | 创建回测（后台线程执行） |
| GET | `/api/backtests/{id}` | 结果与权益曲线（前端传 `include_trades=false`） |
| GET | `/api/backtests/{id}/trades` | 已保存成交分页（最近 500 笔范围） |
| GET | `/api/trading/dashboard` | 总览：权益、持仓、信号、日志 |
| GET | `/api/trading/orders` | 订单分页与筛选汇总 |
| PATCH | `/api/trading/orders/{id}` | 修改止损止盈 |
| POST | `/api/trading/orders/{id}/close` | 平仓 |
| POST | `/api/trading/orders/{id}/reverse` | 反手 |
| GET | `/api/trading/equity` | 权益快照 |
| POST | `/api/trading/run-once` | 手动巡检 |

列表统一支持 `page`（从 1）与 `page_size`（默认 20，上限 100），响应附带 `total`、`page`、`page_size`、`pages`；筛选或条数变化回到第一页，空页自动回到有效页。设置读取用 `include_stats=false`、总览用 `include_market=false` 省去未展示的统计。

## 9. 验证

```powershell
cd D:\UGit\get-money\backend
..\.venv\Scripts\python.exe -m compileall app
..\.venv\Scripts\python.exe -m pytest tests -q

cd ..\frontend
npm test
npm run build
```

## 10. 已知边界

- **实盘仅限小资金/测试网验证**：本地账本与交易所是两套账，实盘盈亏以本地口径展示，可能与交易所对账有出入。
- **行情源**：K线一律走 fapi 公共接口；paper 固定合约主网，live+测试网走测试网。历史 paper 期现货K线与合约价格存在口径差异，切换后建议清空 `klines` 重新同步。
- **数据深度**：1m 历史仅约 90 余天；lower-tf intrabar delta 依赖 1m，历史不足时自动降级为 estDelta 估计（信号密度与质量都会变化）。
- **实盘简化**：未模拟资金费率、盘口深度、强平与分批建仓；市价单以返回均价入账；`limit_maker` 挂 postOnly 最多轮询约 10 秒，超时撤单后剩余转市价；交易所侧止损止盈触发后本地按标记价结算，与真实成交价有偏差；保护单挂失败会市价回滚开仓并写日志。
- **策略未经足够长期验证**：当前读数不支持「有正 edge」的结论，默认关闭；开启前请自行补足样本与独立复检。
- 不构成投资建议。

## 11. 风险提示

加密货币合约交易风险极高，可能导致全部本金损失。请先在模拟盘与合约测试网长期验证；接入真实资金前需自行补充风控熔断、资金费率与强平模型、对账与审计流程，并妥善保管 API Key（仅授予合约交易权限、禁用提币）。

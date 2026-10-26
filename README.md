# Quant Desk · 虚拟币量化交易系统

本地运行的 BTC / ETH 量化交易系统：从币安合约接口同步 1 分钟K线，并在本地生成其他周期，计算技术指标与 ML 方向概率，按评分自动分配仓位、推荐杠杆、给出止损止盈，通过定时任务持续巡检持仓，并在前端查看订单、收益与系统状态。交易执行**支持模拟盘与币安 U 本位合约测试网实盘**（`execution_mode` 可切换，实盘默认指向测试网，需自行承担 API Key 安全责任）。

## 1. 功能

| 模块 | 能力 |
| --- | --- |
| 行情 | 默认覆盖币安 U 本位永续合约全量（`symbol_scope=all`，可切 `custom` 自定义列表），1 小时内存缓存 + 失败回退；按 24h 成交额过滤低流动性合约（`min_quote_volume_usd`，默认 3000 万，0=不过滤，ticker 拉取失败时不过滤）；只通过 fapi 合约公共接口获取 1m K线，5m/15m/1h/4h/1d 均由本地 1m 原始数据聚合（模拟盘固定合约主网，实盘按测试网设置选择同源行情），缓存到本地 SQLite；支持配置 HTTP/SOCKS5 代理（设置页 `proxy_url`，留空直连） |
| 指标 | TA-Lib 为主，pandas-ta-classic 与纯 pandas 逐级回退；RSI、MACD、ATR、SuperTrend、ADX、布林带、EMA、量能 Z 等 25 维特征 |
| 资金费 | 从 fapi 拉取历史资金费入库（8 小时 1 条，幂等增量）；asof 合并为K线特征：当前费率/年化/z-score/24h 累计；提供"资金费率极值"独立策略，并为所有策略提供可选的资金费拥挤过滤器 |
| 模型 | LightGBM / XGBoost 二分类，预测未来 N 根K线方向；按时间前 80% 训练、后 20% 评估；无模型时规则回退 |
| 策略 | ML 趋势共振、SuperTrend 趋势跟随、RSI 均值回归；评分 = ML 概率 + 指标共振；评分决定开仓与否、仓位占比、是否重仓 |
| 风控 | ATR 止损止盈、移动止损、含成本的单笔风险预算、持仓时限、平仓冷却、过期行情保护、最大杠杆 |
| 回测 | 统一顺序撮合；隔离训练/验证/测试、验证集选参、同口径基线对比，计入杠杆、双边成本、风险预算、止损止盈与冷却 |
| 交易 | 支持模拟盘与币安 U 本位合约测试网实盘：开仓（杠杆设置、步长/最小名义价值校验、市价单或 limit_maker postOnly 限价单，`order_type` 可切）、交易所侧止损止盈保护单、撤单改单、reduceOnly 平仓、交易所状态周期同步、反手；账户级熔断（总回撤或连续亏损达到设置线即冻结开仓，`risk_halt` 需人工恢复）；保证金与权益本地核算，盈利自动累加总资金，支持追加/提取/重置资金 |
| 巡检 | APScheduler 每 N 秒同步最新K线（全量宇宙时 8 线程并发、每线程独立 DB session）、重算信号、检查止损止盈、移动止损、信号反转、超时平仓；周期日志记录币种数与耗时，单轮耗时随币种数增长 |
| 前端 | Vue 3 + Element Plus + ECharts 深色界面：总览、订单、策略、回测、模型、行情、设置 7 个页面 |

## 2. 目录

```
get-money/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI 路由
│   │   ├── backtest/       # 顺序撮合、独立验证与样本外对比
│   │   ├── features/       # 指标与特征工程
│   │   ├── market/         # 币安客户端、K线存储
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

- **数据库：PostgreSQL（推荐）或 SQLite，二者通过环境变量切换。**
  - 连接串统一由环境变量 `DATABASE_URL` 决定，写在项目根目录 `.env`（已在 `.gitignore` 中，不入库）：

    ```bash
    DATABASE_URL=postgresql+psycopg2://postgres:<密码>@127.0.0.1:5432/get_money
    ```

  - 不设置时回退到本地 SQLite 文件 `data/trading.db`，因此既有部署无需改动即可继续运行。
  - 方言校验在启动时快速失败（当前支持 `sqlite` / `postgresql`），配置写错会立刻报错而不是运行到一半才炸。
  - `app/db.py::make_engine()` 是唯一的连接工厂，按方言分别处理：SQLite 加 `check_same_thread` 与 PRAGMA；PostgreSQL 走连接池、`pool_pre_ping` 与连接超时。回测类脚本传 `read_only=True`，由数据库层面拦截误写入。

- **从 SQLite 迁移到 PostgreSQL**（一次性，本机 1118 万行约 3 分钟）：

  ```bash
  createdb get_money                     # 或 psql -c "CREATE DATABASE get_money;"
  ./.venv/Scripts/python.exe backend/scripts/migrate_sqlite_to_postgres.py
  ```

  脚本会自动建表、丢二级索引后用 COPY 批量装载、重建索引并修正所有自增序列，最后逐表核对行数。加 `--truncate` 可覆盖已迁移的表。

- **统一 1m 来源**：仅下载和写入 1m K线；读取其他周期时在数据库内按 UTC 边界聚合。开盘/收盘取首尾，最高/最低取极值，成交量、成交额、成交笔数累加。
  - 桶边界用 `open_time - open_time % step` 对齐而非除法截断：SQLite 的整数除法会截断，PostgreSQL 遇到 NUMERIC 会转十进制除法再四舍五入，两者混用会让桶整体错位。
  - 缺失任意分钟的已结束周期会被排除；连续的当前周期可用于行情展示和浮盈估值，策略、模型及回测仍过滤未收盘周期。
  - 旧的直接下载高周期记录保留在库中，但不参与新查询。行情页的本地数据统计仅展示 1m 原始数据。
  - 增量巡检刷新共享 1m 来源，同一币种有多个周期持仓时只同步一次。切换长周期后，如历史不足，需同步足够天数的 1m 数据以满足指标预热。
  - 可补齐当前交易范围及已有持仓的分钟历史：` .\.venv\Scripts\python.exe backend/scripts/backfill_minutes.py --days 40 --workers 4`（从项目根目录执行）。历史窗口越长，请求量和本地存储量越大。
- **模型文件：`data/models/*.pkl`**，数据库只存路径与指标。
- **API Secret：Fernet 加密**，密钥保存在 `data/fernet.key`，接口不返回明文。

## 4. 启动

### Windows 后台运行

完成依赖安装和前端构建后，在项目根目录执行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start-background.ps1
```

通过 Windows 任务计划程序运行 `QuantDesk-GetMoney`，访问 http://127.0.0.1:8010。关闭 ChatGPT 或启动终端后仍会继续运行，没有运行时长限制。日志保存在 `output/server.log`。该任务按需启动，电脑重启或用户注销后需重新执行启动命令。

停止后台项目：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\stop-background.ps1
```

### 4.1 后端

```powershell
cd D:\UGit\get-money
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

- 启动后在行情页同步币安历史K线，再到模型页训练模型。已有模型会在后续启动时继续使用。

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

1. **行情**：默认币种范围为 U 本位永续全量（约数百个）；可在设置页切到自定义列表（`symbol_scope=custom`）。选择币种/周期，通过币安公共接口同步；本地数据表可看到行数与起止时间。全量同步为同步请求，耗时随币种数线性增长。资金费历史通过同步请求的 `funding=true` 拉取（天数取 `days`），行情页可查看最近记录与当前费率/年化/z-score。
2. **模型**：选择币种、周期、LightGBM/XGBoost、预测跨度与天数，训练后可在列表对比 AUC/准确率。
3. **策略**：启用策略、编辑 JSON 参数、预演信号；查看方向、评分、上涨概率、推荐杠杆、仓位占比、止损止盈。
4. **回测**：选择策略/模型/天数/手续费/滑点/评分阈值，运行顺序回测，可启用验证集选参；查看收益对比、权益曲线、过滤原因与成交明细。
5. **设置**：配置币安 API（测试网校验合约测试网余额，主网校验合约账户）、执行模式（paper / live，切换 live 需先配置凭证）、代理地址（`proxy_url`，支持 http/https/socks5/socks5h，K线同步与交易所 REST 调用均经该代理，留空直连）、资金调整、风控参数、定时巡检开关、流动性过滤门槛（`min_quote_volume_usd`）、账户熔断参数（`max_drawdown_halt_pct` / `consecutive_loss_halt` / `risk_halt` 手动恢复）、实盘主单类型（`order_type`）。
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
- 仅 ML 策略训练模型；规则策略不训练无用模型。验证选参使用固定小集合，按净收益减去 0.5 倍回撤幅度排序，成交笔数门槛为 max(10, 验证集K线数 // 50)（2880 根验证集 → 57 笔，防小样本幸运儿）；不足时保留当前配置，零成交变体安全不报错。测试数据不参与选择，结果不会自动写回模拟盘。
- 所有回测标记 `engine=sequential`。收盘计算信号，下一根开盘成交；多空共享一个资金账户；使用评分仓位、杠杆、风险上限、手续费与滑点。
- OHLC 触发止损止盈；开盘跳空止损按开盘价成交；无法判断同根路径时止损优先；移动止损在收盘更新、下一根生效；平仓当根不重新开仓。
- 基线关闭行情和成本过滤，保留相同模型、风险限额和撮合规则，以比较过滤及选参效果。它与旧 vectorbt 回测不属于相同口径，不应直接比较收益。
- **随机入场基线**（`random_baseline`，证伪工具）：在基线信号上按固定种子 20261025 对每个非零方向独立 50/50 翻转（LONG 与 SHORT 对调，空方向不变），其余全部相同——入场时点、仓位、杠杆、评分、手续费、滑点与风控完全一致；由于止损止盈按翻转后方向重新推导，绝对距离天然镜像。若策略收益不显著高于该组，说明方向判断没有信息量，应淘汰；此时回测报告会给出相应提示。同一种子结果完全可复现。
- **曝光时间**：`exposure_pct` 为持仓 bar 数占总 bar 数的比例，`avg_bars_held` 为平均每笔持仓 bar 数；策略、基线与随机基线三组都会给出。零成交时为 0。该指标反映资金实际在场时间，可与收益合并判断（低曝光低收益不等于策略无效，高曝光需核对成本与回撤）。
- 数据检查排除未收盘及无效K线；数据缺口后重置指标预热。缺口中持仓在下一可见开盘平仓，并在报告提示缺失根数与滞后时长。不能据此推断缺失行情中的真实盈亏。
- 首笔损失计入回撤；无亏损时盈利因子返回 `null`，另用 `profit_factor_infinite` 标记无穷大；曲线降采样包含最后权益点。
- 回测的下一根开盘是假定及时执行。模拟盘仍按巡检时的信号价格加滑点撮合，调度延迟及开仓所在部分K线会造成差异；真实成交、资金费率和强平尚未模拟。

本次完整比较见 [策略验证报告](docs/strategy-benchmark.md)。它记录全部 12 组结果与数据质量；负收益即亏损，优于基线不代表已稳定盈利。

复现（数据库只读，不改订单、资金和模型）：

```powershell
.\.venv\Scripts\python.exe backend/scripts/benchmark_strategies.py --as-of "2026-10-25T06:00:00+00:00" --days 30 40
```

### 组合回测（多币种等权）

`backtest/portfolio.py` 在单币种回测之上做组合层封装（不修改 `run_backtest` 行为）：每个 (币种, 策略) 为一条腿，资金等权分配 `初始资金 / (币种数 × 策略数)`（单策略时即每币种等权）；各腿权益曲线按 bar 时间 outer join + 前向填充对齐，腿首个 bar 之前按该腿初始资金填充（资金尚未投入），组合权益 = 各腿权益之和。组合级指标（收益/回撤/夏普/曝光/成交/费用）与单币种共用同一 `metrics_from_equity` 口径；同时输出单币种明细（收益/回撤/成交/期望值/贡献占比）与分散度（收益横截面标准差、正收益占比、Top3/Bottom3 贡献）。等权下组合收益恒等于各币种收益均值，可用于校验"同时运行 vs 顺序运行"无差异；单币种数据不足等失败会被跳过并记入 `errors`，不阻塞整体。注意：组合层允许同一币种的多策略腿同时存在（各占一份资金），实盘同一币种同时只持一个仓位（评分择优），解读组合容量时需考虑该差异。复现：

```powershell
.\.venv\Scripts\python.exe backend/scripts/run_portfolio_validation.py
```

本次组合验证见 [组合验证报告](docs/portfolio-validation.md)。

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
| GET | `/api/market/symbols` | 可交易币种列表（全量宇宙或自定义） |
| GET | `/api/market/funding` | 最近资金费记录 + 当前费率/年化/z-score/24h 累计 |
| GET | `/api/market/sources` | 本地 / 币安数据源状态 |
| POST | `/api/market/sync` | 从币安同步K线 |
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

- **实盘仅限测试网验证**：实盘链路对接币安 U 本位合约（测试网 / 主网可切），本地账本与交易所真实资金是两套账；实盘盈亏仍以本地口径展示，可能与交易所对账存在出入。
- **行情数据源**：K线一律走币安 fapi 合约公共接口（公开无需签名），与 U 本位永续宇宙同源；paper 模式固定合约主网（价格≈真实行情），live+测试网走合约测试网。历史上 paper 阶段积累的现货K线与合约价格存在口径差异，切换后建议清空 `klines` 表并重新同步。
- **实盘简化处理**：未实现资金费率、盘口深度、强平与分批建仓；市价单以返回均价入账。`order_type=limit_maker` 时以标记价挂 postOnly 限价单（费率 0.04% → 0.02%），最多轮询约 10 秒：成交按成交均价入账，超时未成交撤单后仅对剩余数量转市价保证成交，限价单被交易所拒绝则直接转市价。交易所侧止损止盈触发后，本地按标记价格结算（`exchange_sl` / `exchange_tp` / `exchange_closed`），与真实成交价可能有小幅偏差。保护单挂失败会立即市价回滚开仓并写日志，但网络异常期间的极端情况仍需人工核对交易所仓位。
- 1 分钟级别噪声大、手续费占比高；默认策略与模型仅用于流程验证，不保证盈利，回测为负也可能出现。
- 模型准确率受限于样本与市场状态，需定期重新训练；系统会在缺失模型时自动训练，但不会自动滚动更新。
- 未实现资金费率、盘口深度、逐仓/全仓强平、部分成交、真实滑点模型；杠杆为简化线性放大。
- 不构成投资建议。

## 11. 风险提示

加密货币合约交易风险极高，可能导致全部本金损失。请先在模拟盘与合约测试网长期验证，再考虑任何真实资金接入；切换真实资金前需自行补充风控熔断、资金费率与强平模型、对账与审计流程，并妥善保管 API Key（仅授予合约交易权限、禁用提币）。

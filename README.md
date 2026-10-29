# Quant Desk · 虚拟币量化交易系统

本地运行的币安 U 本位合约交易系统：从 fapi 同步 1 分钟K线并在本地聚合其他周期，计算技术指标与订单流吸收特征，由策略给出方向、评分、结构止损与目标，按评分分配仓位，通过定时巡检管理持仓。交易执行支持**模拟盘与币安 U 本位合约（测试网 / 主网）**（`execution_mode` 可切换，需自行承担 API Key 安全责任）。

> 架构以「**策略 × 币种池绑定**」为中心：策略是可替换的插件（`strategy/specs/`），跑哪些币、用什么周期、用哪套参数由绑定决定，同一个策略可以在「主流 15m」「山寨 5m」「山寨 1m」上各跑一套互不干扰的参数。详见第 6 节。
>
> 本仓库已按「只保留能稳定执行的策略」收敛：`absorption_reversal` 之外的策略、ML 训练/预测、TradingView 研究端口、资金费率收割与早期探针脚本均已删除。策略默认 **`enabled=0`**、两条种子绑定也默认关闭，原因见第 6 节与 [吸收反转策略报告](docs/absorption-reversal-strategy.md)。

## 1. 功能

| 模块 | 能力 |
| --- | --- |
| 行情 | 币安 U 本位永续全量（`symbol_scope=all`，可切 `custom`）；1 小时内存缓存 + 失败回退；按 24h 成交额过滤低流动性合约（`min_quote_volume_usd`，0=不过滤）；只落 1m K线，5m/15m/1h/4h/1d 由本地按 UTC 边界聚合；支持 HTTP/SOCKS5 代理（`proxy_url`） |
| 指标 | TA-Lib 为主，pandas-ta-classic 与纯 pandas 逐级回退；RSI、MACD、ATR、SuperTrend、ADX、布林带、EMA、量能 Z 等基础特征 |
| 订单流 | `features/absorption.py`：`|delta|` 滚动最近秩百分位、无价格进展、摆动极值、拒绝影线、单边支配度五条件 + 3 根内收盘穿越实体的确认状态机；优先用 lower-tf intrabar delta，缺数据回退脚本自带的 estDelta 估计 |
| 资金费 | fapi 历史资金费入库（8 小时 1 条，幂等增量），asof 合并为K线特征（当前费率 / 年化 / z-score / 24h 累计），可选拥挤过滤 |
| 策略 | 注册表 + spec 插件：`strategy/specs/*` 只实现 `lookback/prepare/signals/describe`，参数默认值、范围、整数键、布尔键随 spec 声明；信号拼装（成本过滤、ATR 定杠杆、评分分层仓位、风险预算截断）由 `strategy/signals.py` 统一完成 |
| 池与绑定 | `symbol_pools` 定义「跑哪些币」（静态列表或规则），`strategy_bindings` 定义「策略 × 池 × 周期 + 参数覆盖」，`symbol_params` 再给单个币种打参数补丁；参数四层合并：全局设置 < 策略默认 < 绑定 < 币种；同币多池各自评估、按评分只开一仓 |
| 风控 | 结构止损（极值 ∓ 0.25 ATR）与 `min/max_sl_atr` 夹逼、2R 固定目标、1R 保本、30 根时间止损、含成本的单笔风险预算、平仓冷却、过期行情保护、最大杠杆 |
| 回测 | 顺序撮合；样本内/外切分（`train_horizon` 隔离跨度）、可选验证集选参、同口径基线对比、随机方向基线、bootstrap 期望值置信区间；可指定「策略 × 池」绑定，参数层与实盘完全同源 |
| 参数寻优 | 一键对全部有足够历史的币种做参数网格搜索：**周期本身也是搜索维度**（多选 15m/1h/4h，每币再选最优周期）；训练段选参、验证段只做判定（每笔期望 95% CI，池化须 ≥30 笔才下结论）；进程池按「(币种, 周期)」分片吃满 CPU（默认留 4 核给巡检），单任务独占、可取消、可看实时进度与 ETA；结果可一键写入币种级覆盖，或按参数聚类生成新池 + 绑定 |
| 交易 | 模拟盘与合约（测试网/主网）：开仓（杠杆、步长/最小名义价值校验、市价或 `limit_maker` postOnly）、交易所侧止损止盈、撤改单、reduceOnly 平仓、状态周期同步、反手；账户级熔断；本地保证金与权益核算、追加/提取/重置资金 |
| 巡检 | 由**生效绑定**驱动：绑定启用 + 策略总开关启用 + 池启用 → 展开成 `(symbol, binding)` 逐对评估，每对带自己的参数与周期；APScheduler 每 N 秒同步最新K线（默认最多 8 路并发、共享连接池）、单轮内复用指标帧、检查止损止盈/保本/时间止损/信号反转；周期日志记录绑定数、币种数与各阶段耗时 |
| 指标目录 | 读取 `data/pine-collection`、`data/pine-crypto-collection` 的第三方 Pine 指标索引，支持搜索与源码查看（只读资料，不参与计算） |
| 前端 | Vue 3 + Element Plus + ECharts：总览、订单、策略、寻优、回测、行情、指标目录、设置；按需加载、独立空态与重试；订单/回测历史/成交/本地数据列表服务端分页（默认 20） |

## 2. 目录

```
get-money/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI 路由（backtests / bindings / optimizations / pools / symbol_params / ...）
│   │   ├── backtest/       # 顺序撮合、指标、显著性检验、optimizer（全币种参数寻优）
│   │   ├── features/       # 基础指标 + absorption 订单流特征
│   │   ├── market/         # 币安客户端、K线存储、metrics 观测
│   │   ├── ml/             # 数据集构建与 purged 切分（仅回测用，无训练/预测）
│   │   ├── strategy/       # registry（策略注册表）、signals（公共拼装）、pools、bindings、symbol_params、
│   │   │                   # rules（参数分派）、engine、specs/absorption_reversal、coin_universe、defaults
│   │   ├── trading/        # 资金、模拟撮合、订单管理、调度、实盘同步
│   │   └── main.py
│   ├── scripts/            # absorption_backtest / optimize_all / derive_candles + 数据回补/迁移/校验脚本
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
2. **策略**：上半部分是「策略 × 池 × 周期」绑定矩阵（新增/删除绑定、启停、优先级、参数 JSON 覆盖、成员数预览、新建币种池）；下半部分是策略卡（启停、编辑默认参数）与信号预演（可指定用哪条绑定的参数层）。
3. **回测**：选择币种/周期/天数/手续费/滑点/评分阈值；「参数层」下拉可直接选一条「策略 × 池」绑定，回测参数与实盘巡检完全同源（不选则用策略默认参数）；可启用验证集选参；查看收益对比、权益曲线、过滤原因、显著性区间与成交明细。
4. **寻优**：选策略/周期（可多选，周期即搜索维度）/范围（全部、某个池或手填币种）→ 表单下方即时显示「每个周期各有多少币可扫」→ 开始。进度条给出已完成数、ETA 与判定分布；跑完后默认按「每币最佳周期」一行展示（可切到「周期明细」对比同币各周期），列出训练/验证收益、成交笔数、每笔 CI 与最优参数，可「仅统计通过 / 验证段为正 / 全部」+「写币种覆盖 / 生成池+绑定」一键应用；页面底部还能查/清空全部币种级覆盖。
5. **设置**：币安 API（测试网校验测试网余额，主网校验合约账户）、执行模式、代理、资金调整、风控参数、定时巡检、流动性门槛、熔断参数、主单类型。
6. **总览 / 订单**：权益曲线、持仓、已实现/未实现盈亏、胜率、日志；订单页可改止损止盈、平仓、反手。
7. **指标目录**：搜索并查看 Pine 指标说明与源码（只读）。

## 6. 策略、币种池与绑定

```
StrategySpec（注册表）        SymbolPool（池）            StrategyBinding（绑定）
  默认参数 / 范围              static 手写列表              (策略, 池, 周期) 唯一
  lookback / prepare          rule   规则动态解析           enabled / priority
  signals（方向与评分）        最终统一减 BLACKLIST         params（参数覆盖层）
        └───────────────┬───────────────┴───────────────┬────────────┘
                        ▼                               ▼
              engine.evaluate(symbol, interval,      巡检 / 回测 / 预演
              strategy, params=绑定参数, pool=池)
```

模拟盘与回测共用同一套参数与信号拼装；手续费与滑点只由全局设置或回测请求指定，策略与绑定都不能覆盖撮合成本。

### 6.1 策略注册表：新增策略只实现一个 spec

`strategy/registry.py` 用 `StrategySpec` 描述一个策略，`strategy/specs/<code>.py` 里 `register()` 即接入，不需要改 `rules.py`、`defaults.py`、API 或巡检：

| 字段 | 作用 |
| --- | --- |
| `code` / `name` / `description` / `tags` | 标识与展示；`tags` 供前端分组 |
| `defaults` | 策略默认参数（第二层） |
| `ranges` / `bool_keys` / `int_keys` | 保存与绑定时的取值校验（必须覆盖 `signals` 用到的全部数值键） |
| `lookback(params, interval)` | 需要多少根K线才能出信号；巡检与回测据此拉数据 |
| `prepare(data, ctx)` | 策略专属特征（如 lower-tf intrabar delta） |
| `signals(data, params)` | **唯一必须写死的部分**：返回 `signed`（方向）、`score`、`prob_up`、`sl_pct`、`tp_pct`、`reason` |
| `describe(data, params)` | 预演面板展示的观测快照 |
| `default_enabled` | 新建策略行时的初始启用状态 |

成本过滤、ATR 倒数定杠杆、评分分层仓位、单笔风险预算截断、方向清零统一在 `strategy/signals.py::assemble_signal_frame` 完成，因此所有策略自动共享同一套风控与撮合口径。拦截原因用 `block()` 按 Pine 的判定顺序先到先得，保证与脚本一致。

### 6.2 币种池：跑哪些币

`symbol_pools` 一张表，`kind` 二选一：

| kind | 含义 | 关键字段 |
| --- | --- | --- |
| `static` | 手写列表 | `symbols`（去空白、大写、去重） |
| `rule` | 规则动态解析 | `rule.base`(`all`/`major`/`static`)、`rule.min_quote_volume_usd`、`rule.exclude_major`、`rule.max_symbols`（按 24h 成交额取 TopN）、`rule.exclude` |

- 规则池的成员 = 解析结果 **− 全局黑名单**（`coin_universe.BLACKLIST`）**∩** 当前 `symbol_scope`（自定义币种时自动收窄）。
- 成交量取不到时「宁贵不漏」：保留缺额币种而不是静默丢弃，避免池子因一次接口抖动突然变空。
- 成员数是实时解析的（成交量走 TTL 缓存），前端绑定向导会先给「成员数预览」再让你调参数。
- 种子池：`major`（主流 16 币）、`alt`（全量剔除主流）、`all`（全量）；只在表为空时写入，删掉不会被复活。

### 6.3 绑定：四层参数与同币多池

一条绑定 = `(策略, 池, 周期)` + 一套参数覆盖，唯一约束 `uq_binding_scope`。同一个策略可以在多个池上跑不同周期，例如：

| 策略 | 池 | 周期 | 意图 |
| --- | --- | --- | --- |
| `absorption_reversal` | `major` | 15m | 主流币，脚本默认灵敏度 |
| `absorption_reversal` | `alt` | 5m | 山寨币，更宽的 delta 门槛 + 更紧的止损 |
| `absorption_reversal` | `alt` | 1m | 短线，只做多 |

**参数四层合并**（`rules.resolve_params`）：全局设置 < 策略默认 < 本条绑定 < **币种级覆盖**（`symbol_params`）。`fee_rate` / `slippage` 由全局强制统一，任何一层都不能覆盖。未知字段、非有限数值、越界值在保存时直接 400，不会等到巡检才炸。

**币种级覆盖**解决「同一个池里 BTC 和小币不该用同一套阈值」：寻优按币种算出不同最优参数后，把差异写进覆盖表即可，绑定本身不用拆。覆盖按 `(策略, 币种, 周期)` 唯一，删除即回到绑定参数。写入时会按「绑定层 + 覆盖」合并后的整体校验——只拿覆盖单独校验会漏掉跨字段约束（例如 `max_hold_bars` 小于全局 `min_hold_bars`）。

**生效条件**：绑定启用 **且** 策略总开关启用 **且** 池启用。

**同币多池**：BTC 既在 `major` 又在 `all` 时，两条绑定各自独立评估，产生的信号按评分排序后**只开一仓**，不做互斥校验——这样池的组合可以随意调整而不会互相卡死。

**订单可追溯**：`signals.pool` / `orders.pool` 记录来源绑定，持仓管理用 `resolve_order_params(策略, 周期, 池, symbol)` 精确回取「绑定 + 该币种覆盖」的参数（老订单无池时回落策略配置，行为等价）。

**缓存正确性**：`engine` 的缓存键包含参数指纹，同币不同池的参数不会互相污染特征。

### 6.4 吸收反转（`absorption_reversal`）

放量单边主动成交（`|delta|` 位于滚动百分位之上）却推不出价格、在摆动极值留下拒绝影线 → 进攻方被困。**买方被吸收做空、卖方被吸收做多**。

- 五条件：`|delta|` 最近秩百分位、无价格进展（实体 < `body_mult × ATR`）、摆动极值（`extreme_lookback`）、拒绝影线（`wick_mult`）、单边支配度（`dominance_pct`）。
- 确认：`conf_mode=1` 要求后续 `conf_window_bars` 根内收盘穿越吸收 bar 实体；期间打穿吸收极值或超时作废。
- 出场：结构止损 `极值 ∓ sl_buffer_atr × ATR`（`min_sl_atr`/`max_sl_atr` 夹逼）、`target_r=2R`、浮盈 1R 移保本、`max_hold_bars=30` 时间止损。
- 仓位：脚本 `percent_of_equity=30`、`max_leverage=1`，双向交易；`min_score=0.75` 分层（默认全部通过，评分只用于仓位与分层）。
- 运行范围：由绑定决定（种子 `major` 池即原 `MAJOR_SYMBOLS` 那 16 个主流币），不再有写死的白名单。
- **默认停用**：15m 主流币池化检验的每笔期望 95% 置信区间跨 0（不可判定），放宽灵敏度后转负。详见 `docs/absorption-reversal-strategy.md`。

### 6.5 风险仓位与巡检

- **风险仓位**：评分给出保证金上限，再按止损距离、杠杆与双边费用/滑点缩小；默认 `risk_per_trade_pct=0.02`（单笔预算 2% 权益），成交时复查可用资金。跳空可能突破预算。
- **巡检**：`trading/service.py::run_cycle` 先取生效绑定并展开成 `(symbol, binding)`，逐对评估；持仓只接收原策略信号；只处理尚未检查的已收盘K线，不复用开仓前/已处理行情；过期与重复信号不能开新仓。summary 返回 `bindings` 明细与逐条错误（单条绑定坏掉不影响其它绑定）。

关键参数：`delta_window`、`delta_pct`、`body_mult`、`extreme_lookback`、`wick_mult`、`dominance_pct`、`abs_gap_bars`、`legacy_gap_logic`、`conf_mode`、`conf_window_bars`、`direction_mode`、`trend_mode`、`session_start_hour`/`session_end_hour`、`sl_buffer_atr`、`target_r`、`be_trigger_r`、`min_sl_atr`/`max_sl_atr`、`max_hold_bars`、`cooldown_bars`、`risk_per_trade_pct`、`max_leverage`、`cost_filter`、`min_atr_cost_ratio`、`min_net_reward_risk`、`max_atr_pct`。未知字段、非有限数值与越界值在保存时被拒绝。

## 7. 回测口径

- **参数层与实盘同源**：请求带 `pool` 且存在 `(策略, 池, 周期)` 绑定时，回测直接用该绑定的参数覆盖层，与巡检完全一致；不带或没有对应绑定则回落策略默认。前端「参数层」下拉就是这么用的。
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

## 8. 参数寻优：一键找每个币种自己的参数

入口：前端「寻优」页，或命令行 `backend/scripts/optimize_all.py`。两者共用 `backtest/optimizer.py`，
区别只是网页版把结果写库供后续一键应用，命令行适合在服务器上批量跑并导出 CSV。

### 8.1 为什么不能「在整段历史上挑收益最高」

520 个币种 × 972 种参数组合里，一定有某个币的某个组合看起来很美——那是噪声，不是 edge。
所以选参和判定被拆成两段：

| 阶段 | 数据 | 作用 |
| --- | --- | --- |
| 训练段（默认前 70%） | 只用于**选**参数 | 按 `净收益 + 0.5×最大回撤` 排序，成交笔数不足门槛的组合直接出局 |
| 验证段（后 30%） | 只用于**判**定 | 最优组合在验证段重跑，给净收益与每笔期望的 bootstrap 95% CI |

判定取值（`optimizer._verdict`，顺序即优先级）：

| verdict | 含义 |
| --- | --- |
| `pass` | 验证段 CI 下界 > 0，统计上站得住 |
| `inconclusive` | 验证段 CI 跨 0——「不知道」，不等于「亏」 |
| `negative` | 验证段 CI 上界 < 0 |
| `insufficient` | 验证段成交笔数不足（低于 `max(4, 门槛//2)`），不判定 |
| `no_trades` | 训练段没有任何组合达到成交门槛 |

此外整轮会给一次**验证段池化 CI**（所有币种的最优组合在验证段的成交合起来算）：单币几十笔
太少时，池化是唯一能回答「这套参数体系整体有没有 edge」的量。

### 8.2 算力怎么用

- **按 (币种, 周期) 分片**：`simulate` 是纯 Python 逐 bar，线程被 GIL 锁死；一个 worker 拿到
  一个 (币, 周期) 就把整张网格跑完，数据帧只建一次、指标只预热一次。分片到周期这一层
  还有个好处：1h 的 365 天和 4h 的 365 天 bar 数差 4 倍，混在一个 worker 里会让尾部
  只剩一个大任务在跑。
- **默认并发 = CPU−4**（上限 24）：跑满会把行情同步和巡检挤到无 CPU，回测读数也会因争抢失真。
- **单任务独占**：同时只允许一个寻优任务，第二个请求返回 409。
- **BLAS 单线程**：进程池里每个进程默认开 `cpu_count` 个 OpenBLAS 线程，实测 20 进程直接
  `Memory allocation failed` 并崩池；`optimizer` 在 import 时就把 `OPENBLAS_NUM_THREADS` 等
  置 1（纯 Python 循环用不到多线程 BLAS）。
- **崩了不整轮失败**：`BrokenProcessPool` 会把已完成的结果保留、剩余任务降并发重试一次。
- **prepared 帧缓存**：`spec.prepare_keys` 声明哪些参数影响 `prepare()`，网格里只改出场参数
  （`target_r` / `max_hold_bars`）时复用同一份已富化的帧。
- **重启清理**：寻优跑在后台线程里，进程退出后库里的任务仍挂着 `running`。
  启动时 `reap_stale_runs()` 把它们标成 `failed`（"进程重启，任务中断"），否则界面永远显示运行中。

实测（24 核 / 32GB）：

| 任务 | 规模 | 耗时 | 说明 |
| --- | --- | --- | --- |
| 15m × 90 天 × 24 组 | 59 币 | 23 秒 | CPU 峰值 92%、均值 66%、零失败 |
| **15m+1h+4h × 365 天 × 32 组** | **1055 个 (币,周期) 任务** | **322 秒** | 1053 完成 / 2 跳过 / 0 失败 |

### 8.3 周期也是搜索维度

同一个策略在 15m 和 4h 上的最优参数不是同一套，甚至同一个币在 15m 上不可交易、
在 4h 上有 edge。所以：

- 表单/命令行可以多选周期，任务单位是 **(币种, 周期)**，每个组合在自己周期上模拟；
- 每个周期用自己的**基准参数与网格**（绑定按 `(策略, 周期)` 唯一，混用会让「改善幅度」失去参照）；
- 结果分两层：默认「每币最佳周期」一行，切到「周期明细」看同币在各周期的对比。

币级最优的排序规则（`optimizer.pick_best_per_symbol`）是**先看结论强度，再看分数**：

| 优先级 | 判定 | 理由 |
| --- | --- | --- |
| 1 | `pass` | 验证段 CI 下界 > 0 |
| 2 | `inconclusive` | 不知道，但至少不是负的 |
| 3 | `negative` | 确认为负 |
| 4 | `insufficient` / `no_trades` | 样本不足或无成交 |

只按分数排会出问题：`no_trades` 的分数是 0，而微亏的配置是负数，于是「根本没交易」
会排在「小额亏损」前面——那是没结论，不是最好。

### 8.4 delta 源：整段降级是错的

同一份 15m 帧，est delta 下 90 天只有 **1** 个信号，lower-tf delta 下有 **5~34** 个。
寻优若跑在 est 上，优化的根本不是实盘那个策略。

但这里有个代价很大的坑，本轮修掉了：**1m 只覆盖约 94 天**，而窗口可以是 365 天。
早先的实现是「覆盖 < 90% 就整段降级 est」，理由是「半截 NaN 会让滚动百分位算出假阈值」。
真正的问题是 `bucket_delta` 把**没有 lower-tf 数据的桶填成了 0**：

| 周期 | 窗口 | 帧根数 | 非零 delta | 判定 |
| --- | --- | --- | --- | --- |
| 1h | 365 天 | 8711 | 2244 | 前 270 天 delta 恒 0 → 信号永不触发 |
| 4h | 365 天 | 2141 | 561 | 同上，全窗口 0 个信号 |

而 `coverage = lower.notna().mean()` 在填 0 之后永远是 1.0，保护完全失效，
整轮寻优得到 **8/8 `no_trades`**。修法两处：

1. `bucket_delta` 对没有数据的桶返回 **NaN**（对齐 Pine 的 `array.sum()` 空数组给 `na`）；
2. `add_absorption_context` 按脚本原语义**逐 bar 兜底**：`na(ltfDelta) ? estDelta : ltfDelta`。

结果里的 `delta` 列因此有三档：

| 值 | 含义 |
| --- | --- |
| `lower` | lower-tf 覆盖 ≥ 90%，整段真实买卖差 |
| `mixed` | 只覆盖一部分，其余逐根用 bar shape 估计兜底（悬停可看覆盖率） |
| `est` | 完全没有 lower-tf 数据 |

### 8.5 数据边界：4h 靠 1h 派生

「全币种 + 一年」不是一个愿望，是被数据卡死的硬约束：

| 周期 | 原始行覆盖 | 365 天可用币 | 长窗口瓶颈 |
| --- | --- | --- | --- |
| `1m` | 535 币 / 94 天 | 70（90 天） | 只有 94 天，且 454 币是 10 月才开始 |
| `5m` | **仅 2 币** | — | 只能由 1m 聚合，实际 94 天 |
| `15m` | 521 币，但 461 币 2026-10 才有 | **57** | 长历史只覆盖部分币 |
| `1h` | 411 币从 2025-09 起 | **499** | 唯一原生覆盖满一年 |
| `4h` | **0 行** | **499** | 全靠派生 |

4h 一条都没有，但它完全可以从 1h 聚合出来。`scripts/derive_candles.py` 做这件事：

```bash
# 1h -> 4h，全币种（幂等，可反复执行）
python backend/scripts/derive_candles.py
# 只统计不写库
python backend/scripts/derive_candles.py --dry-run
```

派生行写进 `klines` 的 `interval='4h'`，此后对回测而言它就是普通 native 行
（`NATIVE_FALLBACK_INTERVALS` 已含 4h），取数链路零改动。实测 **521 币 / 114.6 万根 /
401 天**，high/low 逐桶与 1h 比对零差异；4h 的 365 天从 94 天补齐到 2191 根。

桶内不足 75% 根数的整桶丢弃（避免用 3 根 1h 拼出 volume 偏小、high/low 可能漏极值的 4h 桶），
未闭合的最后一桶也丢弃。

### 8.6 本次全量结果

`15m + 1h + 4h × 365 天 × 32 组 × 1055 任务`，322 秒：

| 周期 | 任务 | 有候选 | 验证段池化 | 每笔期望 | 95% CI | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| 15m | 57 | 54 | 594 笔 | −0.1927% | [−0.3176%, −0.0538%] | **negative** |
| 1h | 498 | 107 | 126 笔 | −0.0551% | [−0.4163%, +0.3373%] | inconclusive |
| 4h | 498 | 11 | **4 笔** | +4.98% | [+0.86%, +10.48%] | **insufficient** |
| 合计 | 1053 | 172 | 724 笔 | **−0.1402%** | **[−0.2738%, −0.0060%]** | **negative** |

判定分布：`no_trades 935 / insufficient 77 / inconclusive 30 / negative 11 / pass 0`。
币级最佳周期：`15m 51 / 1h 417 / 4h 30`。

三个必须点破的点：

1. **池化从 inconclusive 变成 negative。** 上一轮 15m×90 天的池化是
   −0.2777%/笔、CI 上界 +0.0065%（跨 0），本轮把窗口拉到一年、币种扩到全市场后
   CI 上界变成 −0.0060%——**之前的「不确定」是小样本在掩盖负 edge**。
2. **4h 那 4 笔不能看。** n=4 的 bootstrap 会给出「CI 下界 +0.86%」这种假精确读数。
   池化因此加了 `MIN_POOLED_TRADES = 30` 门槛，不足就判 `insufficient`（数值照给、不下结论）。
3. **1h/4h 绝大多数是 `no_trades`（925/996）。** 不是参数不好，是这两个周期上吸收形态
   太稀疏（BTC 一年 1h 只有 5 次命中、4h 是 0 次）。判定门槛 `min_trades` 过不去，
   就不该硬选一套参数出来。

结论与 `docs/absorption-reversal-strategy.md` 一致且更硬：**参数寻优救不了一个没有
信号层的策略**；周期维度也救不了——换周期只是把「信号太少」和「信号为负」对调。
这套基础设施的价值在于下次接入真正有 edge 的策略时，能立刻算出「每个币该用什么参数、在哪个周期」。

### 8.7 一键应用

| 模式 | 落点 | 生效时机 | 风险 |
| --- | --- | --- | --- |
| `symbol` | 写 `symbol_params`（币种级覆盖） | 下一轮巡检立即按新参数评估 | 绑定若已启用，等于直接换线上参数 |
| `pool` | 按参数指纹聚类，建静态池 + 绑定（`enabled=0`） | 需到「策略」页手动启用 | 不会自动上线，人工确认后才跑 |

- 应用前会先在**币级别选一次最佳周期**，每币只落那一套参数：同一币同时写 4 个周期的覆盖，
  等于把「选周期」的结论又丢回给巡检去碰运气。
- 因为覆盖表的键是 `(策略, 币种, 周期)`，写入按周期分组进行；`pool` 模式生成的每条绑定
  也带自己的 `interval`。
- 筛选门槛：`pass`（最严）/ `positive`（验证段为正）/ `all`。
- `pool` 模式会清掉这批币的币种级覆盖，否则旧覆盖会盖住新绑定的参数。
- 应用会在 `optimization_results.applied_mode / applied_at` 留痕（逐周期精确标记，
  被挑掉的周期不会被误标成「已应用」）。

## 9. 主要 API

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
| POST | `/api/market/preview` | 预演信号与指标快照（可带 `pool` 指定参数层） |
| GET | `/api/indicators`、`/api/indicators/{id}` | Pine 指标目录与详情 |
| GET/PUT | `/api/strategies`、`/api/strategies/{code}` | 策略列表（含 `specs` 元信息）/ 启停与默认参数 |
| GET | `/api/strategies/{code}/bindings` | 该策略的全部绑定 |
| POST | `/api/strategies/run-once` | 立即执行一次巡检 |
| GET/POST | `/api/pools` | 币种池列表（含实时成员数）/ 新建 |
| PUT/DELETE | `/api/pools/{code}` | 改池定义 / 删池（被绑定引用时 400） |
| GET | `/api/pools/{code}/members` | 成员预览（调参数前先看会跑多少币） |
| GET/POST | `/api/bindings` | 绑定列表（`?with_members=true` 带成员明细）/ 新建 |
| PUT/DELETE | `/api/bindings/{id}` | 改周期/启停/优先级/参数覆盖（全量校验）/ 删除 |
| GET | `/api/bindings/{id}/params` | 该绑定合并三层后的完整生效参数 |
| GET | `/api/optimizations/options` | 策略搜索空间、池列表、默认并发、是否已有任务在跑 |
| GET | `/api/optimizations/universe` | 预估扫描范围（`?intervals=15m,1h&days=` → 逐周期币种数，`total_tasks` 为任务总数） |
| POST | `/api/optimizations` | 创建寻优任务（`intervals` 多周期；返回 id，后台线程 + 进程池执行；已有任务时 409） |
| GET | `/api/optimizations`、`/api/optimizations/{id}` | 任务列表 / 详情（进度、ETA、判定分布、池化 CI、逐周期池化） |
| POST | `/api/optimizations/{id}/cancel` | 请求取消（等已派发的币种跑完） |
| DELETE | `/api/optimizations/{id}` | 删除任务与结果（运行中 400） |
| GET | `/api/optimizations/{id}/results` | 结果分页（`?best_only=true` 每币最佳周期 / `false` 周期明细；`&verdict=&q=&sort=score\|symbol\|interval`） |
| POST | `/api/optimizations/{id}/apply` | 一键应用：`mode=symbol` 写币种覆盖 / `mode=pool` 聚类建池+绑定（按周期分组） |
| GET/PUT/DELETE | `/api/symbol-params` | 币种级参数覆盖的查询 / 批量写入 / 批量删除 |
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

## 10. 验证

```powershell
cd D:\UGit\get-money\backend
..\.venv\Scripts\python.exe -m compileall app
..\.venv\Scripts\python.exe -m pytest tests -q

cd ..\frontend
npm test
npm run build
```

当前状态：后端 `367 passed`，前端 `67 passed`，构建通过；多周期寻优全链路冒烟 14 项（逐周期扫描范围 → 创建 → 进度 → 每币最佳/周期明细 → 应用写覆盖 → 清理）全部通过。

## 11. 已知边界

- **实盘仅限小资金/测试网验证**：本地账本与交易所是两套账，实盘盈亏以本地口径展示，可能与交易所对账有出入。
- **行情源**：K线一律走 fapi 公共接口；paper 固定合约主网，live+测试网走测试网。历史 paper 期现货K线与合约价格存在口径差异，切换后建议清空 `klines` 重新同步。
- **数据深度（各周期差异很大，直接决定能跑哪些币）**：

  | 周期 | 原始行 | 365 天可用币 | 说明 |
  | --- | --- | --- | --- |
  | `1m` | 535 币 / 约 94 天 | 70（仅 90 天） | 454 个币是 2026-10 才开始有数据 |
  | `5m` | **仅 2 币** | — | 实际由 1m 聚合，上限就是 94 天 |
  | `15m` | 521 币，但 461 币只有 2026-10 之后 | 57 | 长历史只覆盖部分币 |
  | `1h` | 411 币从 2025-09 起 | 499 | 唯一原生覆盖满一年 |
  | `4h` | **0 行（靠 1h 派生）** | 499 | `derive_candles.py` 补齐 |

  lower-tf intrabar delta 依赖 1m，1m 短于窗口时结果里 `delta` 标为 `mixed`（缺的段逐根用 estDelta 兜底）。
- **实盘简化**：未模拟资金费率、盘口深度、强平与分批建仓；市价单以返回均价入账；`limit_maker` 挂 postOnly 最多轮询约 10 秒，超时撤单后剩余转市价；交易所侧止损止盈触发后本地按标记价结算，与真实成交价有偏差；保护单挂失败会市价回滚开仓并写日志。
- **老库升级**：启动时自动建 `symbol_pools` / `strategy_bindings` / `symbol_params` / `optimization_runs` / `optimization_results`，给 `signals` / `orders` 补 `pool` 列，并把 `optimization_results` 的唯一键扩到 `(run_id, symbol, interval)`（PG 改约束；SQLite 不支持改约束，直接重建该表——本地回退库的寻优结果可重跑）。均为幂等操作。
- **寻优的样本天花板**：判定不成立有两种，必须区分开——
  1. **信号太稀疏**：1h/4h 上吸收形态一年只有几次（BTC 1h 365 天命中 5 次、4h 0 次），
     绝大多数币落进 `no_trades`，此时"没有最优参数"才是正确答案；
  2. **样本够但读数为负**：15m 一年 594 笔、CI [−0.32%, −0.05%]，判 `negative`。
  池化判定另设 `MIN_POOLED_TRADES = 30` 门槛，不足只给数值不下结论。
- **策略未经足够长期验证**：当前读数不支持「有正 edge」的结论，默认关闭；开启前请自行补足样本与独立复检。
- 不构成投资建议。

## 12. 风险提示

加密货币合约交易风险极高，可能导致全部本金损失。请先在模拟盘与合约测试网长期验证；接入真实资金前需自行补充风控熔断、资金费率与强平模型、对账与审计流程，并妥善保管 API Key（仅授予合约交易权限、禁用提币）。

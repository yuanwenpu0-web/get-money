# Spec - Regime-Gated Donchian v2 [RGD] 本地策略移植 v1.0

> 生成日期：2026-10-30 · 状态：已确认（用户 Phase 0 四决策锁定）
> 基于：`docs/rgd-porting-contract.md`（高见远逐行移植契约，876 行）+ 用户 4 决策 + 总监一致性检查
> 溯源基线：`data/rgd-v2-source.pine`（Pine v6，逐行 1:1，勿改）
> 本 Spec 是**范围与门禁契约**；逐表达式映射、broker 伪代码、16 条裁决以 `docs/rgd-porting-contract.md` 为唯一权威（living-spec），Spec 不复制其全文。

---

## 1. 产品定义

- **一句话**：把 TradingView Pine v6 策略 `Regime-Gated Donchian v2 [RGD]` 严格移植为 `get-money` 本地回测库可运行的策略，出入场逐行对齐脚本。
- **目标用户**：项目量化研究者（本地回测/复检），非实盘交易（本次）。
- **核心问题**：脚本的「自适应 Donchian 突破 + HTF 流动性张力 + 价时相关 + 量能接受度 + ER 门控」入场，及「TP1 限价平 50% + 成交后保本 + 接受度加权动态 trailing」出场，通用引擎无法表达，必须专用 broker。

## 2. MVP 范围（锁定——不在此列表的功能一律不做）

| 优先级 | 功能 | 验收标准摘要 |
|---|---|---|
| P0 | `features/rgd.py` 纯因果特征（Wilder ATR / percentrank / rolling Pearson / tensionStep 状态机 / 通道 / ER / 防守 / state 机 / 入场价位） | 契约 §3 映射、A1–A11 |
| P0 | `features/rgd_htf.py` `attach_htf_levels` 因果 join | 契约 §4.2、A9/A10/A28 |
| P0 | `specs/rgd_donchian_v2.py` 注册 StrategySpec（32 参数 1:1，direction=0 强制专用管线） | 契约 §6、A21/A23/A25 |
| P0 | `backtest/rgd_execution.py` `simulate_rgd` 专用 broker（persistent pending + OHLC 路径 + TP1 部分平 + 保本 + trailing + 反手/trend flip） | 契约 §5、A12–A19/A26 |
| P0 | `scripts/rgd_backtest.py` 4 组矩阵回测（含 HTF 拼帧、随机基线、bootstrap CI、段间复利） | 契约 §7、A22/A24 |
| P0 | `tests/test_rgd_features.py` + `tests/test_rgd_execution.py`（A1–A30） | 契约 §10 |
| P1 | `docs/rgd-backtest.{md,json}` + `-G1..G4.json` 报告 | 契约 §7.6 |

## 3. 明确不做（Out-of-Scope——锁定）

| 不做 | 原因 | 何时考虑 |
|---|---|---|
| 实盘持仓管理（TP1 限价挂单 + 动态 trailing 迁移到 `trading/order_manager`） | 用户 D4 锁定只交付回测；`default_enabled=False` live 不调用 | 用户立项实盘化后 |
| 1d 派生 / 周月线 | `autoTF(1d)="1W"` 项目周期白名单无周线；D1 禁派生 | 引入周线或改 `auto_htf=False+htf_in` |
| UI / API / DB 新表 / OpenAPI / 图标库落地 | 纯后端策略库，无 REST/前端/新表 | 接 UI 时锁定单一 SVG 图标库 |
| 独立 ADR 文件 | 契约 §8 裁决以 MADR 字段内联充当决策记录 | — |
| 通用 `simulate` 路径 | RGD `direction≡0`，必须专用 broker | — |
| 补齐 15m 历史 / 拉 exchangeInfo tickSize | 本次交付边界禁止拉新数据/落库 | OPEN-07/08 |

## 4. 技术架构（锁定 · 版本锚定）

| 层 | 技术/版本 | 锁定原因 |
|---|---|---|
| 语言 | Python **3.12.14**（`.venv/Scripts/python.exe`，实测 2026-10-29） | 项目现役 |
| 数值 | pandas / numpy（项目已装） | `.ewm(adjust=False)`、`sliding_window_view`、`np.searchsorted` |
| 测试 | pytest（项目已装），`backend/` 下 `..\.venv\Scripts\python.exe -m pytest tests -q` | 现役 |
| 注册 | `StrategySpec` 注册表（`registry.py:46-71`）+ `rules.resolve_params`（三层合并，fee/slip 硬禁止策略级覆盖） | 现行架构（旧 `SYMBOL_ROUTES`/`coin_router`/分支式 rules.py 已删） |
| 撮合 | `simulate_rgd(data, signals, params, initial, fee, slippage, interval)` | 对齐 `simulate_zigzag`/`simulate_ocb` 签名 |
| 数据 | `load_frame(symbol, "15m"/"4h", start_ms, end_ms)` = 1m 聚合 + native 补齐 | 15m native 298.3 天 / 4h native 400.7 天，PG `get_money` |
| 部署 | 无（本地脚本） | — |

**周期锁定**：主 15m（900s）+ HTF 4h（14_400_000ms），`autoHTF=true→autoTF(15m)="240"→"4h"`（D1）。`INTERVAL_MS` 见 `utils.py:3-16`。

## 5. API 端点清单（N/A）

无 REST 端点。规格接口 = 契约 §6 参数表（DEFAULTS/RANGES/INT_KEYS/BOOL_KEYS/PREPARE_KEYS）+ §6.5 `signals` 列契约（SIGNAL_COLUMNS 11 列 neutral + 21 自定义 `rgd_*` 列）+ §5 broker 状态机（pending/position 结构）。

## 6. 数据库表清单（N/A）

无新表、无写入、只读查询 `klines`。4h HTF 帧内存拼入 15m 帧，不落库。

## 7. 页面清单（N/A）

纯后端，无页面。

## 8. 设计 Token（N/A）

无 UI。图标库：条件性登记 `lucide-react`（若将来接 UI，锁定单一 SVG 描边库、16/20/24px），本次**不落地**。禁止 emoji 图标 / 紫粉渐变 / 硬编码颜色——本任务无 UI 故 N/A，代码注释与文档不得出现 emoji 作功能图标。

## 9. 验收标准（EARS，锁定——QA 唯一依据）

> 完整 A1–A27 见契约 §10；以下为门禁必过项 + 总监追加 A28–A30。

| 编号 | EARS 验收标准 | 优先级 |
|---|---|---|
| AC-01 | While 输入截断到 `frame.iloc[:k]`，`prepare` 在第 k 根全部 `rgd_*` 列**必须**与整帧逐位 `==`（含 tension/state 状态机） | P0 |
| AC-02 | While 同一 4h 桶内 16 根 15m bar，`attach_htf_levels` 的 `hh/ll` **必须**唯一值=1（不重绘）；未完成桶剔除 | P0 |
| AC-03 | If `syminfo.mintick` 无每币数据，系统**必须**用 `MINTICK=1e-8` guard（R5），不得拉新数据 | P0 |
| AC-04 | While `ta.percentrank(atr/close, rank_len)`，系统**必须**用「前 length 根不含当前 / `<=` / 分母=length」定义（R3），4 手工值 `[105→80,97→0,102→60,108→100]` 逐位对 | P0 |
| AC-05 | While 有持仓，`stop_lvl` **必须**单调（多 max / 空 min）不回退；TP1 成交后**必须**保本 `max(stop_lvl, entryRef)` 且生效滞后一根（R12） | P0 |
| AC-06 | If 信号 bar 成交，系统**必须**让该成交 bar 裸奔（当根无 exit，N8/R12），不得顺手挂止损 | P0 |
| AC-07 | If 同根盘中 `high>=tp1Lvl 且 low<=stopLvl`（open 未穿 tp1Lvl），系统**必须** stop 优先全平（R14，边界见裁决 3） | P0 |
| AC-08 | While `strategy.entry` 反向，系统**必须**反手（平旧+开新，数量=旧+新）；同向且 `pyramiding=0` **必须**忽略 | P0 |
| AC-09 | While `signals()`，`direction` **必须**恒 0 且 `block_reason="stop_order_requires_dedicated_broker"`；误接通用 simulate **必须**零成交（安全失败） | P0 |
| AC-10 | While 未 attach 的裸 OHLCV 帧进入完整链路，系统**必须**零成交**且**降级可观测（`frame.attrs["rgd_htf_degraded"]=True` 或脚本 assert，禁止静默，A28） | P0 |
| AC-11 | If OHLC 平票（`abs(o-h)==abs(o-l)`），系统**必须**先低后高（以 `ocb_execution.py:139`/`zigzag_execution.py:139` **代码**为准，其注释文字写反，A29） | P0 |
| AC-12 | If 开盘跳空穿 TP1 且同根触 stop，系统**必须**先以 open 部分平 tp1_pct、剩余仓按 stop 全平（两段成交，R14 仅约束盘中路径，A30） | P0 |
| AC-13 | While 回测 4 组矩阵 G1–G4 **必须**各产出 `docs/rgd-backtest-G*.json`；G1 主报告；G3/G4 标注「slippage 近似」 | P0 |
| AC-14 | While 随机基线，系统**必须**用 `randomize_rgd(shift)`（不得用 `randomize_directions`，对 `direction≡0` 是 no-op 静默失效，A24） | P0 |
| AC-15 | While `resolve_params`，`fee_rate`/`slippage` 作为 override **必须**抛错；默认参数**必须**脚本 1:1 | P0 |
| AC-16 | While 数据末根未平仓，**必须**写入 `equity.attrs["open_position"]`，**不得** `close_at_end` 强平 | P0 |
| AC-17 | While 新增每个源文件，`wc -l` **必须** ≤300（超阈值按契约 §11 拆分） | P1 |

## 10. 边界与约束

- **成本两组**：主 G1 `fee=0.0004/slip=0.0002`（项目全局）；对照 G3 `fee=0.0005/slip≈2ticks`（脚本原样，R15 每币标量代理 `2*tick_size(median_close)/median_close`，`tick_size` 用 `utils.round_price` decimals 约定：≥1000→1e-2、≥1→1e-4、else→1e-6）。
- **方向**：`dir_mode` 0/1/2 完整实现；4 组矩阵取 G1(0)/G2(1)/G3(0)/G4(1)；`dir_mode=2` 实现但不入矩阵。
- **初始资金** 10000 / 币独立账户，段间复利，`pyramiding=0`、1× 无杠杆。
- **样本**：15m 仅 57 币 ≥180 天、BNB 10.4 天；报告**必须**显式列 `universe.skipped`/`symbols_with_trades`，样本不足（池化 <30 笔）默认「不启用」，不把零成交读成无信号。
- **不强制收盘平仓**（Pine 结束持仓保留）。

## 11. 内嵌已知坑（从项目记忆 + 门禁拉取）

| 坑 | 根因 | 修法 |
|---|---|---|
| 静默零成交（高危） | prepare 幂等短路 + 无 HTF 列降级 `htf_ok=False` 全列；若 attach 放 prepare 之后 → 全 False → 零成交无报错 | **attach_htf_levels 必须在 load_segments 内、切段前对整个 frame 做一次**（本次 htf_in 恒 4h 安全）→ segments 自带 `rgd_htf_*`；改 `htf_in/auto_htf` 必须重建 segments；降级须可观测（AC-10） |
| OHLC 平票注释矛盾 | `ocb_execution.py:139` 注释「先高后低」但代码 `(low,high)` 先低后高 | 以**代码**为准先低后高，勿被注释误导（AC-11） |
| R14 表述过宽 | 「同根双触 stop 全平」与「开盘跳空穿 limit 既成部分平」冲突 | R14 仅限盘中路径（open 未穿 tp1Lvl）；跳空穿 limit 先部分平、余仓 stop（AC-12） |
| `gap` 笔误 | 契约 §5.4 第(2)步 `close_at(S if not gap else o)` 变量未定义 | 改 `close_at(S)`（开盘跳空已在第(1)步 return） |
| dtype=object | `groupby().cummax()+shift()` → object，`~` 位运算恒真 | bool 门控列 `&`/`~` 前 `.astype(bool)` |
| shift 方向 | `shift(-1).rolling(H)` 是过去窗口 | 全用 `high[1]/low[1]`（`shift(+1)`），不写反向 rolling |
| 周期桶对齐 | PG NUMERIC 四舍五入错位 | `x - x % step`（`data_service.py:463`） |
| 并行/复现 | BLAS 子进程 OOM；`as_completed` 顺序随机致 bootstrap ±0.02pp 漂移 | numpy import 前设 BLAS=1；worker `read_only+NullPool`；结果**按 symbol 排序**后再池化 |
| 未收盘桶 | 半截 OHLCV 造未来函数 | base `(index+base_step)<=as_of`；HTF `closed` 掩码 |
| 随机基线 no-op | `randomize_directions` 只翻 `direction`，RGD 恒 0 | `randomize_rgd(shift)` 循环位移 `rgd_*` 决策列（AC-14） |
| prepare_keys 漏列 | 缓存帧复用致静默错单 | 29 个 prepare 影响参数全集（含 `dir_mode`，G1/G2 仅此不同） |

## 12. 端到端验证步骤（锁定）

```bash
cd D:/UGit/get-money/backend
# 单元（核心成功流 + 关键边界流）
..\.venv\Scripts\python.exe -m pytest tests/test_rgd_features.py tests/test_rgd_execution.py -q
# 冒烟（4 组都产出、随机基线生效、报告可渲染）
..\.venv\Scripts\python.exe scripts/rgd_backtest.py --interval 15m --days 180 --universe major --limit 8 --seeds 5
# 正式（主报告 G1 + 三组对照）
..\.venv\Scripts\python.exe scripts/rgd_backtest.py --interval 15m --days 180 --universe all --seeds 10
# 全量回归
..\.venv\Scripts\python.exe -m pytest tests -q
```

**完成定义**：AC-01–AC-17 全绿 + 4 组冒烟均产出 `docs/rgd-backtest-G*.json` + 全量 `pytest tests -q` 回归 0 + 单文件 `wc -l` ≤300。任一不满足即退回。

## 13. 变更记录

| 日期 | 变更 | 原因 | 影响 |
|---|---|---|---|
| 2026-10-30 | v1.0 初版 | 契约完成 + 用户 4 决策 + 总监门禁（4 处瑕疵裁决：HTF attach 衔接 / 平票注释 / R14 边界 / gap 笔误 → AC-10/11/12 + A28–A30） | 全范围 |
| 2026-10-31 | v1.1 实现期口径修正（Phase 3 实测） | ① Pine `tp1Hit` 只是「价格曾触」的保本 latch，**不等于** `strategy.exit(qty_percent=50)` 已成交；单一 bool 兼做「保本」与「撤单」会在价格触达但订单未成交时永久撤掉 TP1，导致 50% 部分平永不发生 → 必须拆 `tp1_seen` / `tp1_filled`（AC-05 落地口径）。② `trades.py:27` 的 `aggregate_positions` 合并同仓位多腿时用**后一腿 status 覆盖前一腿**，仓位级 `status_counts` 天然看不到 `take_profit_1`；实测 BTCUSDT 分腿 238 stop_loss + 45 take_profit_1 → 报告必须同时给「仓位级最终出场」与「腿级成交」两张表并披露 TP1 触发率（AC-13 落地口径）。 | 出场统计口径 + 回测报告结构 |

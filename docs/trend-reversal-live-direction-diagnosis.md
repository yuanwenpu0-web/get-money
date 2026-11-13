# 趋势反转 live 订单：方向选择与中途盈利回吐的工程诊断（工程诊断，非 edge 判定）

> 生成时间：2026-11-12 00:55 UTC ｜ **修复落地：2026-11-12 08:0x UTC+8** ｜ 脚本：`backend/scripts/diagnose_trend_reversal_live.py`、`backend/scripts/reanchor_open_protections.py` ｜ 命令：见文末
> 性质：**工程正确性诊断 + 正确性修复**。按手册 §0 例外条款跳过 Gate 1／Gate 4 的 edge 判定；H3／H8／H10／H12 仍生效。
> **本轮已改生产代码**：`backend/app/trading/live.py` 开仓路径改为锚定实际成交价（用户 2026-11-12 授权「一起改」）。**策略、绑定、参数、出场口径一行未动，未下任何新单，未切换执行环境（仍 DEMO/testnet）。**
> §1–§5 的数字是 **91 笔冻结快照**；§4.1／§4.2 已用**权威信号行**在 **101 笔**上复核并更正，见各节「复核更正」。

## 假设与边界

- **信息源 / 可证伪假设**：只用真实 live 订单 + 本地 1m K 线重放，不引入新信息源。待检假设 H_dir：「方向选择有明显问题」＝订单 side 与信号 bar 方向不一致，或 PnL 记账反号。待检假设 H_exec：下单链路存在使实际风险偏离预算的工程缺陷。终止条件：若方向一致且记账自洽，则 H_dir 证伪，不得以「反号」解释亏损。
- **周期 / 持有期**：15m，绑定 #7 `supertrend_current × all × 15m`。实际持仓中位 172.5 分钟（≈11.5 根），最长 1278 分钟。
- **窗口起止 / 墙钟对齐**：2026-11-09 11:57:47Z ~ 2026-11-11 22:37:02Z（**2.44 天**），按 `orders.created_at` UTC 排序。同期 BTCUSDT 74,823.4 → 74,215.2（**−0.81%**，本地 PG 1m），近似横盘略跌。
- **币池 / 币数**：`all` 池，**69 币 91 笔**已平仓 live 订单，**重放成功 91/91，skip 清单为空**。报告生成时另有 7 笔 OPEN 未纳入（无平仓价，无法算 MFE 终点）。
- **执行环境（已核表，非记忆）**：`app_state.execution_mode="live"`、`app_state.testnet=true` ⇒ 走 `exchange_gateway.env_of()` 的 **DEMO＝币安合约 Testnet**，**不是主网真钱**。`strategy_bindings` 中 `supertrend_current` 仅 #7（`all`×15m）`enabled=1`、#10（`trend_reversal_fit`×15m）`enabled=0`；91 笔全部 `interval='15m'`，与 #7 一致。
- **快照时效**：本报告是 **91 笔的冻结快照**（窗口末 22:37:02Z）。2026-11-12 01:0x 复核时已 92 笔 / −225.23 USDT、OPEN 6 笔（新增 id 289 `0GUSDT` `signal_reverse` −2.12）。**下文所有数字均指 91 笔快照，勿当实时读数。**
- **止损线核对（Gate 0）**：已查 `references/verdicts-history.md`。**命中 H2**（现货 OHLCV 不再投入新特征／参数／标签／换周期／换止损倍数）与 §4（不换 15m、不加分批止盈、不要再调止损倍数／持有期）。`docs/supertrend-inverse-check.md` 已做过方向整列取反对照。因此本轮**不是新 alpha 投入**，只做工程正确性核查与现象量化。
- **对照组**：方向反号对照沿用 `docs/supertrend-inverse-check.md`（原方向 15m −0.4577%／反向 −0.4611%，**两者均 negative**）；出场规则族对照沿用 `docs/trend-reversal-order-exit.md`（17 条规则 `three_gates_pass=0`）。本轮**不重跑**这两族。

## 口径与参数

- **成本**：用订单库真实 `entry_fee + exit_fee`，双边合计均值 **0.0838%**（名义本金），低于手册回测口径 0.12%（`fee_rate=0.0004` + `slippage=0.0002`）。**口径＝名义本金**（`notional = qty × entry_price`），全程不切换，不与 margin 口径混比。
- **参数来源**：逐笔读 `orders.params_snapshot`（**现已有值**，修正 11-10 报告中「全 NULL」的旧边界）。关键值：`sl_atr_mult=1.5`、`target_r=3.0`、`tp1/2/3_fraction=1/6,1/3,1/2`、`trailing_enabled=false`、`max_hold_bars=0`、`use_adx_filter=false`、`use_ema200_filter=false`、`cl_enabled=false`、`risk_per_trade_pct=0.005`、`max_signal_age_bars=2`。
- **撮合（重放）**：复刻 `engine._prepare_features` 取数与「只留已收盘桶」过滤（`bar_open + step <= as_of`），把 `now` 换成订单 `created_at`；`build_features` → `spec.prepare` → `spec.signals`，取最后一根有方向的 bar 为信号 bar。**已确认 engine.py:91-95 排除未收盘桶，不存在盘中重绘。**
- **MFE/MAE**：持仓期 1m 真实 OHLC，`R_fill = side×(entry−sl)`（实际成交价口径），纯因果，不含平仓后行情。
- **样本外切分**：不适用（工程诊断，不选参、不判定 edge）。

## 结果

### 1. H_dir 证伪：方向没有反号

| 检查项 | 读数 |
|---|---|
| 订单 side 与重放信号 bar 方向一致 | **90 / 91** |
| 唯一不一致 | id 275 `1000RATSUSDT`（重放 `bars_since_signal=61`、latency 920 分钟，属重构异常，非反号证据） |
| PnL 记账对账 `realized_pnl` vs `side×(close−entry)×qty − 手续费` | 残差最大 **0.747 USDT**，91 笔中 11 笔 >0.01（部分成交/舍入） |
| LONG 均价变动 / SHORT 均价变动 | +0.1423% / −0.1306%（各自有利方向，记账自洽） |

**结论：「方向选择反了」不成立。** 且方向整列取反在 2026-11-04 已测：原方向与反向**都是 negative**，反向不是解药。

### 2. 盈亏的真实分解（USDT 加权，完整划分）

| 项 | USDT |
|---|---:|
| 毛利（价差 `side×(close−entry)×qty`） | **−124.92** |
| 手续费（`entry_fee+exit_fee`） | **−101.04** |
| 毛利 − 手续费（本地重建） | −225.95 |
| **`orders.realized_pnl` 合计（权威）** | **−223.10** |
| 重建残差（权威 − 重建） | **+2.85** |

残差来自部分成交与价格/数量舍入（逐笔最大 0.747 USDT，11/91 笔 >0.01）。**下文一律以 `realized_pnl` 为权威口径**，毛利/手续费仅用于分解。手续费占净亏损 **45.3%**。但**等权每笔**口径是另一个故事：

| 口径 | 每笔 % | 95% CI | verdict |
|---|---:|---|---|
| 毛利（名义本金） | +0.1369 | — | — |
| 手续费 | −0.0838 | — | — |
| **净收益（等权每笔）** | **+0.0564** | **[−0.3766, +0.5397]** | **`inconclusive`** |

**等权每笔 `inconclusive`（点估计 +0.0564%，CI 跨 0）、USDT 加权 −223.10** ⇒ 两个口径不能互相替代：等权口径**不构成任何方向性结论**，加权亏损集中在仓位分布上。名义本金与 `margin` 相关 0.654、与 `entry_dev_pct` 相关 0.320（弱，91 笔不足以定因）。

### 2b. 为什么「总是亏损」：生效绑定跑在已被判 `negative` 的周期上

这是**已有判定 + 现行配置**的直接对照，不需要新统计：

| 项 | 读数 | 来源 |
|---|---|---|
| 唯一生效绑定 | #7 `supertrend_current × all × **15m**` `enabled=1`（2026-11-06 01:08 用户手工启用） | 本轮查 `strategy_bindings` |
| 15m 样本外判定 | 每笔 **−0.299%**，CI **[−0.382, −0.209]** → **`negative`** | `verdicts-history.md` §4、`docs/interval-exit-verdict.md` |
| 1h 样本外判定 | 每笔 −0.016%，CI 跨 0 → `inconclusive` | 同上 |
| 换手倍数 | 15m 成交 **7459** vs 1h **1611**（**4.6×**） | 同上 |
| 本轮 live 实证 | 91 笔全为 15m；手续费占净亏 **45.3%** | 本报告 §2 |

`verdicts-history.md` §6 的历史快照记的是「`supertrend_current` 的 15m 绑定 `enabled=0`」，**现已变为 1**。按 **H11**，该绑定属**未经统计支持**、且与既有 `negative` 判定相反。手册 §4 已写明「**不换 15m**」「不要再调止损倍数／持有期救它」。

⇒ **「总是亏损」的第一因不是方向、不是止盈规则，而是生效周期本身已被样本外判负，且 4.6× 换手把成本放大到吃掉近一半净亏。**

**用户已有显式决策（2026-11-12，经 AskUserQuestion，本轮延续）**：负 edge 绑定「**先不动，只登记风险**」——绑定 #7 保持 `enabled=1` 继续运行，标注「未经统计支持」（H11）。**本轮不改 `enabled`／参数／出场口径，未下任何单**，也不把「停用 #7」当作新建议提出。

### 2c. 绑定级台账复核，并更正一处并发会话的手续费重复扣减

本报告 §2 的 91 笔已核实 **全部 `binding_id=7`**（无 NULL 遗留、无 `env_switch_archived`），归因干净。按绑定口径重查当前库（`close_reason<>'env_switch_archived'`）：

| 口径 | 笔数 | 净 USDT | 名义本金合计 | 净/名义本金 | 胜率 |
|---|---:|---:|---:|---:|---:|
| 绑定 7 `live` | 92 | −225.23 | — | — | 29.35% |
| 绑定 7 `paper` | 26 | −273.85 | — | — | 15.38% |
| **绑定 7 合计** | **118** | **−499.07** | **160,097.20** | **−0.3117%** | **26.27%** |

**更正**：并发会话在 `memory/workbuddy-daily-2026-11-12.md` 登记的「绑定 7 118 笔 / **−631.26** / **−0.3943%**」把手续费扣了两次。算术证据（同一批 118 笔，名义本金完全一致）：

```
sum(entry_fee + exit_fee)              = 132.19
sum(realized_pnl)                      = -499.07
-499.07 - 132.19                       = -631.26   ← 与并发读数逐分吻合
-631.26 / 160097.20 * 100              = -0.3943%  ← 与其每笔读数逐位吻合
```

代码依据 [backend/app/trading/paper.py:201](file:///D:/UGit/get-money/backend/app/trading/paper.py#L201)：`net = gross - order.entry_fee - exit_fee`，[:215](file:///D:/UGit/get-money/backend/app/trading/paper.py#L215) `order.realized_pnl = round(net, 6)` ⇒ **`realized_pnl` 已是扣双边费的净额，不得再减一次费**。本轮 91 笔独立对账也证实这点（`Σ价差 − Σ费 = −225.95` vs `Σrealized_pnl = −223.10`，残差仅 +2.85 来自舍入/部分成交；若 `realized_pnl` 是毛利，残差应约 +101）。

**更正后的结论方向不变、量级更贴合回测**：−0.3117%/笔落在回测 15m `negative` 的 CI **[−0.382, −0.209] 之内**，而不是并发会话所写的「比 CI 下沿还略差」。118 ≪ 7459，**仍不声称差异显著**。

> 另：本轮 91 笔快照与当前库逐笔比对，`realized_pnl` **0 笔发生变化**，`pnl_source` 仍 62 estimated / 29 binance。并发会话提到的「台账读数被并发改动」在本样本上未复现，其差异已完全由上述重复扣费解释。

### 3. 按持仓期最大浮盈分群（**事后定义，描述性，不可交易**）

| 群 | 定义 | 笔数 | 合计 USDT | 均 MFE% | 均 MFE(R) | 均净%（名义本金，扣费） | 均持仓分钟 |
|---|---|---:|---:|---:|---:|---:|---:|
| A 从未浮盈 | MFE ≤ 0 | 21 | **−321.32** | −0.378 | −0.445 | −1.1839 | 23.4 |
| B 曾浮盈终亏 | MFE>0 且净<0 | **40** | **−588.74** | **+1.298** | **1.091** | −1.0825 | 266.9 |
| C 浮盈保住 | MFE>0 且净≥0 | 30 | +686.95 | +3.741 | 2.876 | +2.4432 | 378.8 |

三群合计 −223.10。**B 群是最大单一亏损源**。A 群平均 23 分钟直接打死，说明「方向错」的体感主要来自这批从未浮盈的单。

> **口径修正留痕**：本表「均净%」一度误填 `net_pct`（该字段实为**未扣费**的价差收益 −1.104／−1.001／+2.523）。现已统一为 `net_nom_pct`（名义本金、扣双边费），与 MFE% 同基准，符合 H3「不混用口径」。

> 分群按已实现结果定义，存在选择偏差；各群 CI（A [−1.4032,−0.9673]、B [−1.2790,−0.9245]、C [+1.6896,+3.2899]）**仅为描述性，不是 verdict**，与 `docs/trend-reversal-order-exit.md` 已指出的「事后分组在下单前不可得」一致。

### 4. 新发现的工程缺陷（本轮增量，此前报告未覆盖）

#### 4.1 风险预算被突破（确定性缺陷，无需统计显著性）

`live.py:198` 用 `entry = signal.price`（信号 bar 收盘价）调 `position_size(..., entry, signal.sl, ...)` 定仓位；`live.py:330-338 / 375-376` 的保护单与落库 `sl/tp` **直接用 `signal.sl/signal.tp` 原值**，而实际成交在 `fill_price`。二者偏离时：

- 实际单位风险 `side×(fill−sl)/fill` ≠ 预算所用 `side×(sig_close−sl)/sig_close`；
- 实际权益风险 = `0.5% × 风险膨胀倍数`。

| 读数 | 值（91 笔重放快照） |
|---|---|
| 风险膨胀倍数 中位 / 最大 | 1.000 / **2.862** |
| >1.5 倍 / >2 倍 笔数 | **6 / 2** |
| 预算权益风险 / 实际最大 | 0.5% / **1.43%** |
| 成交后真实 R:R（名义 3.0）中位 / 最小 | 3.000 / **0.416** |
| R:R <2 / <1.5 笔数 | **7 / 4** |

**复核更正（2026-11-12 08:0x，改用权威信号行，101 笔）**：上表的锚点来自「按 `created_at` 重放 kline」，而**引擎是按本地已同步的 bar 发信号的**——重放会多取到下单时还没入库的那一根。实证：订单 287 `APRUSDT` 创建于 22:48:07，重放取到 22:30 bar 收盘 0.03587，但 `signals` 表里它所依据的是 22:46:11 那条（id 3851732，ts=22:15 bar，**price 0.03572 / sl 0.036962 / tp 0.031994**），与落库 sl/tp 逐位吻合。改用「下单前最后一条同向 `signals` 行」后：

| 读数 | 值（101 笔，权威信号行） |
|---|---|
| 信号行匹配率 | **101 / 101**（skip 清单为空） |
| 成交价恰等于信号价（maker 原价成交） | **63 / 101** |
| 追价成交（entry ≠ signal_price） | **38 / 101** |
| 风险膨胀倍数 中位 / p90 / 最大 | 1.0000 / 1.2462 / **2.8143** |
| >1.5 倍 / >2 倍 笔数 | **6 / 2**（与快照一致） |
| 成交后真实 R:R 中位 / 最小 | 3.0000 / **0.4162** |
| 全部订单 `order_type` | `limit_maker`（101/101） |

⇒ **缺陷结论不变、量级微调**（最大 2.862 → **2.8143**）：63 笔 maker 原价成交时旧代码恰好正确，**风险突破只发生在 38 笔追价单里**，其中 6 笔超预算 1.5 倍、2 笔超 2 倍。最严重 id 248 `1INCHUSDT` 追价 1.68% → 膨胀 **2.8143×**、R:R **0.4162**。

最严重 6 笔（膨胀 >1.5）**全部以 `signal_reverse` 离场**，合计 −86.79 USDT，平均 R:R 仅 1.176。典型：id 249 INJUSDT SHORT，信号价 7.950、成交 7.734（追低 2.72%），止损 8.090 → 实际风险 4.60%、目标仅 2.65%，**R:R 0.576**，亏 −40.98。

> 注意：回测 `execution.py:209-211` 同样把 stop/target 锚在 `previous.close`，但成交在 `row.open`（≈信号收盘价），所以**回测里这个偏差不显现**。这是**回测与实盘的口径分叉**，不是策略设计意图。

#### 4.2 信号→成交延迟 ≈ 0.76 根 bar

| 读数 | 值 |
|---|---|
| 延迟 中位 / p75 / 均值（剔除 1 笔异常） | **11.44 / 15.18 / 12.08 分钟** |
| 占 15m bar 比例（中位） | **0.76** |
| 回测假设 | 下一根**开盘**成交（延迟 0） |

根因（`job_logs.trading_cycle`）：509 币每轮 **274–325 秒**，其中 `sync_ms` 188–216 秒、`signal_ms` 36–70 秒，轮次几乎首尾相接。

**复核更正：延迟归属拆解（101 笔，权威信号行）**。此前把 11.44 分钟整体归给「下单链路」，实际它由两段构成，且**大头在数据同步而不是下单**：

| 段 | 中位 | p75 | p90 | 最大 |
|---|---:|---:|---:|---:|
| bar 收盘 → `signals` 行落库（同步+调度） | **9.671** | 13.107 | 16.185 | 20.787 |
| `signals` 行落库 → `orders.created_at`（下单） | **1.867** | 2.229 | 2.722 | 4.737 |
| 合计 bar 收盘 → 下单 | **11.402** | 15.176 | 18.526 | 22.186 |

单位分钟。⇒ 优化优先级应是**kline 同步/bar 收盘对齐触发**（占 85%），而不是下单代码。

**但延迟的平均直接成本是负的**：若按回测口径在下一根开盘成交，合计反而少赚 **−95.79 USDT**（即晚成交平均略好），仅 26/91 笔延迟成本为正。**所以不能把亏损归因于「延迟本身」**；延迟的真实危害是制造偏离离散度（std 0.476%）并触发 4.1 的风险突破。

#### 4.3 PnL 可信度与残留条件单

- **62/91 笔 `pnl_source=estimated`**：凡 `exchange_sl`/`exchange_tp` 触发平仓都对不上交易所真实成交，`close_price` 恰等于 `sl`/`tp`，即**假设按触发价成交、未含滑点**。真实亏损很可能**大于** −223.10。
- id 279 ARCUSDT：`exchange_sl` 平仓后 `tp_status` 仍为 `NEW`（其余 43 笔同场景均为 `EXPIRED`），疑似**孤儿止盈条件单**。`closePosition=true` 的残留单在同币后续开仓时可能误平。
- `manual`/`signal_reverse` 共 31 笔 note 里 `sl_status/tp_status` 仍为 `NEW`（`close_live` 已撤单但未回写），属**记账陈旧**，非裸奔。

### 5. 中途盈利回吐的规模（回答第二问）

| 读数 | 值 |
|---|---|
| B 群笔数 / 占比 | **40 / 91 = 44.0%** |
| 平均最大浮盈 | **+1.298%（1.091R）** |
| 平均最终净收益（名义本金，扣费） | **−1.0825%** |
| 平均回吐（MFE% − 净%） | **2.380 pp**（合计 95.21 pp） |
| MFE ≥0.5R / ≥1R / ≥1.5R | **29 / 18 / 8** 笔 |
| MFE 出现时刻占持仓比例 中位 | **0.338** |
| 极端例 | id 229 0GUSDT LONG，MFE **5.444R**，最终 `exchange_sl` −30.28 |

放宽到**全部 91 笔**：MFE ≥0.5R 却最终净亏的有 **30 笔，合计 −348.41 USDT**。B 群按平仓原因：`exchange_sl` 24（−398.45）、`signal_reverse` 15（−177.83）、`manual` 1（−12.45）。

> 回吐口径已统一为「MFE% − **扣费后**净%」，两者同为名义本金基准。此前一版用未扣费的 `net_pct` 算出 2.299pp／91.96pp，低估了约 0.08pp/笔（即手续费）。

### 6. 已实施的工程修复（用户授权「一起改」，2026-11-12）

#### 6.1 开仓路径：保护单锚定实际成交价

新增纯函数 [backend/app/trading/live.py](file:///D:/UGit/get-money/backend/app/trading/live.py) `reanchor_exits(fill_price, signal_price, sl, tp)`：

```
offset = fill_price - signal_price
return sl + offset, tp + offset
```

**是纯平移，与方向无关**（推导：`sl_new = fill − side·(side·(P−sl)) = fill + sl − P`，因 `side²=1`）。因此**绝对价格距离不变** ⇒ `sl_atr_mult×ATR` 与 `target_r` 在成交后仍成立，且 `qty×距离` 恰等于开仓前按信号价算出的风险预算。

`_place_live` 改动三处：① 保护单触发价改用 `reanchor_exits(fill_price, entry, signal.sl, signal.tp)` 后按 tick 取整；② 落库 `order.sl/tp` 同步改为取整后的新水平；③ **取整后方向守卫**——若 `(fill−sl)×side ≤ 0` 或 `(tp−fill)×side ≤ 0`（粗 tick 把止损推到成交价另一侧），记录失败原因并 `rollback_open` 平掉刚开的仓，不留脏单。

**`position_size(...)` 仍用信号价，未改**：sizing 必须发生在下单之前，无法预知成交价；改为绝对距离锚定后，实际风险 USDT 自动等于预算，无需回头改仓位。

TDD 留痕：先写 [backend/tests/test_live_fill_anchor.py](file:///D:/UGit/get-money/backend/tests/test_live_fill_anchor.py)（7 例）并**看它失败**（含「不同成交价下风险 USDT 应恒定」一例，RED 时极差 105.793），再实现至 GREEN。同时更正 `test_live_trading.py` 中**断言旧错误行为**的 2 处（触发价 99/103 → 99.5/103.5）。

#### 6.2 存量 OPEN 仓位：已验证为**空操作**，未向交易所发任何改单

新增 [backend/scripts/reanchor_open_protections.py](file:///D:/UGit/get-money/backend/scripts/reanchor_open_protections.py)（191 行，默认只读 dry-run，`--apply` 才撤旧挂新），配 [backend/tests/test_reanchor_open_protections.py](file:///D:/UGit/get-money/backend/tests/test_reanchor_open_protections.py)（**20 例**：15 例纯规划函数 + 5 例 apply 路径，用既有 `FakeFuturesClient`，不联网）。

> **TDD 顺序自我纠正**：`apply_plans`（真正会撤单/挂单的那条路径）一度先写实现后补测试。按 TDD 铁律**已删除该实现**，先看到 5 例 RED（`ImportError: cannot import name 'apply_plans'`）再重新实现至 GREEN。覆盖：正常换单并回写水平、订单已平仓则零交易所调用、现价穿过新水平则零交易所调用、挂新单失败后挂回旧水平、兜底也失败时显式报告 `restored_old_levels=False`。

信号价**取自 `signals` 表「下单前最后一条同向信号」**，而非按 `created_at` 重放 kline（理由见 §4.1 复核更正）。并直接校验落库 `sl/tp` 是否仍等于信号水平按 tick 取整的结果——被手工/保本移过就跳过，不盲目平移。

dry-run 结果（2026-11-12 08:0x，4 笔 OPEN）：

| 订单 | 币种 | 方向 | 成交价 | 信号价 | 信号 sl / tp | 结论 |
|---|---|---|---:|---:|---|---|
| 284 | KERNELUSDT | SHORT | 0.04492 | 0.04492 | 0.045552 / 0.043024 | `already_anchored` |
| 286 | AIOUSDT | LONG | 0.04289 | 0.04289 | 0.041892 / 0.045884 | `already_anchored` |
| 287 | APRUSDT | SHORT | 0.03572 | 0.03572 | 0.036962 / 0.031994 | `already_anchored` |
| 296 | STBLUSDT | LONG | 0.02137 | 0.02137 | 0.021166 / 0.021982 | `already_anchored` |

**4 笔全部 `limit_maker` 且成交价恰等于信号价**，保护单已在预算距离上 ⇒ `plans=0`，**无需重挂，未执行 `--apply`，未产生任何撤单/挂单请求**。这与 §4.1 的 63/101 一致：maker 原价成交时旧代码本就正确。

> 存量集合是**移动靶**：授权时是 6 笔（280/281/284/286/287/288），复核时 280/281/288 等已平仓、另新增 290/292/293/294/295/296。脚本按**当前库状态**查询，不沿用旧 id 列表。
> 安全设计：现价已越过新止损/新止盈的仓位一律跳过（`would_stop_out` / `would_take_profit`），**不主动把浮亏变成已实现亏损**；`--apply` 前再取一次现价复核；`modify_live` 先撤后挂，挂新单失败时尽力挂回旧水平并显式报告 `restored_old_levels`。

#### 6.3 服务重启（H12）

后端改动已重启计划任务 `QuantDesk-GetMoney`（`schtasks /end` → `/run`，2026-11-12 07:55:29 本地）。重启后 `/api/health` 返回 `{"ok":true,"mode":"live","exchange_env":"demo"}`，**执行环境未变**。重启前 07:55:04 开的 STBLUSDT（id 296）走的是旧代码，已纳入 §6.2 复核（结论 `already_anchored`）。

#### 6.4 重启后的线上复核（含**尚未被触发的路径**）

重启后新开的 2 笔走的是新代码，逐笔对权威信号行核验：

| 订单 | 币种 | tick | 信号价 | 成交价 | 追价? | 信号距离 | 成交距离 | 膨胀 | R:R |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| 297 | AIAUSDT | 1e-05 | 0.06307 | 0.06307 | 否 | 0.000770 | 0.000770 | **1.000000** | **3.000000** |
| 298 | ATUSDT | 1e-04 | 0.0889 | 0.0889 | 否 | 0.001387 | 0.001400 | **1.009373** | **3.000000** |

298 的 1.0094 倍**纯粹来自 tick 取整**（0.001387 → 止损取整到 0.0875），不是追价。最终 dry-run 覆盖全部 **6 笔 OPEN（284/286/287/296/297/298）→ `plans=0`，全部 `already_anchored`**。

> **诚实边界**：这 2 笔都是 maker 原价成交，**「追价成交」这条路径重启后尚未在线上出现过**，因此线上只验证了「不劣化」，追价场景的正确性目前**仅由单测覆盖**（`test_risk_usdt_and_reward_risk_are_independent_of_fill_price` 用 4 个不同成交价断言风险 USDT 恒定、R:R 恒为 3）。不得据此声称追价缺陷已在线上实证消除。

## 判定

- **verdict**：等权每笔净收益 **`inconclusive`**（+0.0564%，CI [−0.3766, +0.5397] 跨 0，91 笔 ≥ `MIN_POOLED_TRADES=30`）。`inconclusive` ＝不知道，**不等于亏，也不等于有效**。
- **H_dir**：**证伪**。方向未反号（90/91），记账自洽（残差 ≤0.747 USDT）。
- **H_exec**：**成立，且已修复**。风险预算突破是确定性缺陷（权威信号行复核 101 笔：最大 **2.8143** 倍、R:R 最低 **0.4162**、6 笔 >1.5×／2 笔 >2×，全部落在 38 笔追价单内），不依赖统计显著性。修复见 §6，**属正确性修复，不构成 edge 改善的证据**。
- **随机基线**：`NOT_RUN`。本轮为工程诊断，不做 edge 判定；方向反号对照沿用既有报告（原/反向均 negative）。
- **归因**：方向本身无 alpha 的结论**沿用既有 negative 判定**（15m −0.299% CI 全负、原/反向均 negative），本轮不重新判定。手续费占净亏损 45.3%。USDT 加权与等权口径的差来自仓位分布，与追价弱相关（0.32），**未做因果识别**。
- **与历史结论对照**：与 `verdicts-history.md` §1 H2、§4 一致；与 `docs/trend-reversal-order-exit.md`「两类抱怨是同一固定 TP 的两面、事前不可区分」一致。**增量**在于：11-10 报告 live 队列仅 27 笔、`params_snapshot` 全 NULL、6h 窗口；本轮 91 笔、参数快照齐全、精确重放信号 bar，首次量化了**风险预算突破**与**延迟口径分叉**。

## 局限

- 事后分群按已实现结果定义，CI 仅描述性，**不得当作可交易 verdict**。
- 62/91 笔 PnL 为 `estimated`，交易所条件单真实成交价与手续费不可得；未模拟资金费率、盘口深度、强平、部分成交。
- 样本期仅 **2.44 天**、单一市场状态（BTC −0.81%），91 笔不足以做跨市场稳健性与时间分段。
- id 275 重放 `bars_since_signal=61` 属重构异常，方向一致性按 90/91 计，未单独定因。
- 追价偏离与亏损的关联是观察性的，未做 FDR、未做因果识别，**不得读作「修好追价就能转正」**。
- 名义本金口径，不得与 margin 口径报告直接相减。
- `orders.sl` 可被移动止损回写；本轮 91 笔 `trailing_enabled=false` 且 `sl` 均在不利方向，未受此影响。

## 结论与后续

**一句话结论**：方向没有反号（90/91 一致，且反号早已证伪）；**「总是亏损」的第一因是唯一生效绑定 #7 跑在样本外已判 `negative` 的 15m 上**——绑定级复核 **118 笔 / −499.07 USDT / −0.3117% 每笔**，正落在回测 CI [−0.382, −0.209] 内，4.6× 换手让手续费吃掉 45% 净亏；91 笔 live 快照的亏损结构是「21 笔从未浮盈 −321 + 40 笔曾浮盈回吐 −589 + 30 笔保住 +687」；**可修的是工程缺陷，不是方向，也不是止盈规则**——其中**风险预算突破已修复并重启生效**（权威信号行复核 101 笔：63 笔 maker 原价成交本就正确，38 笔追价单里 6 笔超预算 1.5×、2 笔超 2×、最大 2.8143×、R:R 最低 0.4162；存量 4 笔 OPEN 重挂为已验证空操作），延迟与 PnL 估算仍待处理。附带更正了并发会话台账的一处**手续费重复扣减**（−631.26 → −499.07，见 §2c），以及本轮自身的一处**锚点取错 bar**（重放 kline → 权威信号行，见 §4.1）。

- **是否启用**：`default_enabled=False` 不变。绑定 #7 现 `enabled=1`，**未经统计支持且与既有 `negative` 判定相反**（H11）。**按用户 2026-11-12 显式决策「负 edge 绑定先不动，只登记风险」，本轮保持 `enabled=1`，未改绑定／参数／出场口径，未下单。** 风险已登记于 §2b／§2c。
- **第二问的答复（能否避免中途盈利回吐）**：**以现有信息不能**。`docs/trend-reversal-order-exit.md` 已证：4 条候选规则在 `late` 类全改善（+0.42~+1.15pp）、在 `early` 类全恶化（−1.26~−1.58pp）；训练冠军 `trail_0.5R` 测试段 +0.3285pp、CI [−0.2396,+0.8826]、q 0.354；全样本 17 条规则 `three_gates_pass=0`。**回吐与过早止盈是同一固定 TP 的两面，下单前不可区分**。按 H2／H5，**不再调止损倍数、持有期、TP 阈值或加追踪/保本止损来救它**。
- **工程修复：已实施（用户 2026-11-12 授权「一起改」）**。**仓位 sizing 口径不变、保护单与落库 `sl/tp` 改为锚定实际成交价 `fill_price`**，保持 1.5×ATR 风险与 3R 盈亏比的**设计意图**在成交后仍然成立，并加取整后方向守卫＋回滚。这是正确性修复（消除回测/实盘口径分叉与风险预算突破），**不是参数寻优、不创造 alpha**。存量 4 笔 OPEN 经 dry-run 复核**全部已在预算距离上（maker 原价成交），重挂为空操作，未向交易所发任何改单**。详见 §6。
- **修复的效果边界（不得夸大）**：它只保证「实际风险 = 预算风险、实际 R:R = 3」，**不改变 15m 已被判 `negative` 的事实**，也不保证扭亏。§4.2 已证延迟的平均直接成本为负（−95.79 USDT），因此**「修好追价就能转正」没有证据**。
- **可选的次级工程项**（同样需授权）：① 缩短 `trading_cycle`（sync 188–216s 是大头）或按 bar 收盘对齐触发，降低延迟离散度；② 修复 `exchange_sl/tp` 平仓的真实成交回填，消除 62 笔 `estimated`；③ 核查 id 279 孤儿 TP 条件单并补平仓后状态回写。
- **不建议**：反号方向、**为救 15m 而去搜索新周期**、加/调分批止盈、加追踪或保本止损、调 `sl_atr_mult`／`target_r`——全部命中 H2 止损线且已有 negative／inconclusive 判定。（停用已判负的 15m 绑定属 H11 合规动作，与「换周期寻优」是两件事。）

## 交付自检（H12）

- `preflight.py --prefix docs/trend-reversal-live-direction-diagnosis`：**FAIL 0 / WARN 0 / PASS 21**（诊断快照阶段）。
- **修复阶段单测**：`test_live_fill_anchor.py`（7）+ `test_reanchor_open_protections.py`（20）+ `test_live_trading.py` + `test_maker_orders.py` → 全绿；均**先看到 RED 再实现**（fill-anchor RED 时风险极差 105.793；migration 纯函数 RED 为 `ModuleNotFoundError`/`ImportError`；migration apply 路径 RED 为 `ImportError: cannot import name 'apply_plans'`）。
- **全量 `pytest backend/tests -q`（修复后）：3 failed / 1936 passed / 117.26s**。三个失败均为既有、与本轮无关：`test_market_performance.py::test_complete_tail_queries_have_a_bounded_minute_range`、`test_pool_volatility_research.py::test_live_entrypoints_reject` ×2（`app.trading.dispatch` 缺 `halt_status`）。1936 = 修复前 1916 + 本轮新增 20，**无回归**。
- **环境坑（会影响基线判读）**：直接跑会多出 `test_backfill_rate_limit.py::test_run_checkpoints_and_limits_writes_to_month` ×3 失败，根因是**本机 `NO_PROXY` 环境变量值畸形**，httpx 构造 Client 时抛 `InvalidURL: Invalid port: ':1]'`，与代码无关。`env -u NO_PROXY` 后该文件 9 passed，全量回到上述 3 failed 基线。
- **本轮改了后端服务代码**（`live.py`）⇒ **已重启计划任务 `QuantDesk-GetMoney`**（07:55:29 本地，`/api/health` 复核 `exchange_env=demo`）。**未改前端** ⇒ 无需 `npm run build`。新增脚本均 ≤300 行。
- **未下单、未启用 live key、未切换执行环境**（现仍为 DEMO/testnet）；存量重挂为 dry-run 空操作，**未向交易所发任何撤单/挂单请求**。诊断与复核全部走 `make_engine(read_only=True)`。

## 复现命令

```bash
cd D:/UGit/get-money
# 诊断（只读）
PYTHONPATH=backend .venv/Scripts/python.exe backend/scripts/diagnose_trend_reversal_live.py --out output/trend_reversal_live_diag.json
# 存量保护单重挂：默认只读 dry-run；确认 plans 后才加 --apply
.venv/Scripts/python.exe backend/scripts/reanchor_open_protections.py
# 单测 + 全量（本机 NO_PROXY 畸形，必须 unset，否则多 3 个假失败）
env -u NO_PROXY -u no_proxy .venv/Scripts/python.exe -m pytest backend/tests/test_live_fill_anchor.py backend/tests/test_reanchor_open_protections.py backend/tests/test_live_trading.py backend/tests/test_maker_orders.py -q
env -u NO_PROXY -u no_proxy .venv/Scripts/python.exe -m pytest backend/tests -q
PYTHONPATH=backend .venv/Scripts/python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/trend-reversal-live-direction-diagnosis
```

产物：`output/trend_reversal_live_diag.json`（逐笔 91 条重放 + MFE 原始记录）、`output/tr_diag_final.json`（合并延迟/风险/成本）、`output/reanchor_open_protections.json`（存量 dry-run 逐笔判定）、`output/reanchor_closed_audit.json`（101 笔权威信号行复核）、本报告 md + json。
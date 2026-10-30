# RGD v2 Pine→本地 移植契约（Implementation Contract）

> 状态：**Phase 1/2 交付（可直接据以实现）** · 作者：高见远（首席架构师） · 日期：2026-10-29
> 溯源基线：`data/rgd-v2-source.pine`（Pine Script v6，`//@version=6`，共 237 行，shorttitle `RGD2`）。
> 本契约对基线**逐行 1:1** 映射；任何无法严格复刻的点在第 8 节显式裁决并登记 `docs/decisions/OPEN-DECISIONS.md`，不做静默简化。
> 用户最高约束原文："把这个策略改成本地策略。**出入场都需要严格按照上面的脚本**。"

---

## 0. 依据、门禁与团队规则

### 0.1 知识库依据（本契约引用并遵循）

| 知识库 | 路径（插件根 `.../mvp-dev-expert-team/references/`） | 在本契约的落点 |
|---|---|---|
| 规格即契约 | `01-standards/spec-as-contract.md` | 点名文件/接口、out-of-scope、版本锚定、内嵌坑、端到端验证（§6/§9/§10/§11） |
| 上下文工程 | `01-standards/context-engineering.md` | 决策留痕、改动文件清单外置、即时检索压实（§8/§11） |
| 生成式代码失效模式 | `01-standards/generated-code-failure-modes.md` | 沉默逻辑错误防线：影响图 + 变异加固（§10 验收点） |
| MVP 技术选型矩阵 | `architecture/mvp-stack.md` | 不引入新依赖、用团队已会栈（§0.3） |
| 代码组织规范 | `01-standards/code-organization.md` | 单一职责、单文件 ≤300 行、按资源分包（§11） |
| 开发成本参考 | `cost-models/development-costs.md` | 规模/隐性成本（§7.6） |
| 悬而未决登记册 | `01-standards/open-decisions-register.md` | 三类 slug、七字段、append-only、就地关闭（§8.2） |

### 0.2 团队 P0 绝对规则（大湾区靓仔制定）

1. **禁止 emoji 作功能图标**：本任务为回测库/后端，**无 UI、无图标**，规则不适用（N/A）。若后续接 UI，须锁定**单一** SVG 图标库，不混用。选型理由（条件性）：若前端沿用 React 生态则选 `lucide-react`（tree-shakable、无运行时依赖、社区维护活跃，优于手工散落 SVG）；本任务不涉及，故**只登记不落地**。
2. **禁止紫色→粉色渐变**：无 UI，N/A；本契约不出现任何配色方案。
3. **禁止 AI 模板味/空洞占位**：本文档每节都有可执行内容，无 Lorem ipsum、无"待补充"、无硬编码颜色。

### 0.3 技术选型（依据 mvp-stack：学习成本/生态成熟度/部署成本/团队熟悉度 高权重，扩展性低权重）

沿用项目**现有且已验证**的栈，**零新增第三方依赖**：Python `3.12.14`（`.venv/Scripts/python.exe`）、`pandas`、`numpy`、`SQLAlchemy`、`pytest`。理由：
- 专用 broker 是纯 Python 逐 bar 状态机，不需要新框架；HTF 因果对齐用 `numpy.searchsorted`（与 `features/absorption.py::bucket_delta` 同源原语），不引入 MTF 库。
- 回测脚本范式直接继承 `scripts/supertrend_backtest.py`（进程池 + NullPool + 段间复利），不重写并行层。
- 无 DB 新表、无 API、无前端 → 无部署/迁移成本。

### 0.4 版本锚定（spec-as-contract §2）

| 组件 | 版本/位置 | 备注 |
|---|---|---|
| Python | 3.12.14 | `backend/../.venv/Scripts/python.exe` |
| pandas / numpy | 项目已装（见 `backend/requirements*.txt`） | 使用 `.ewm(adjust=False)`、`sliding_window_view`、`searchsorted` |
| pytest | 项目已装 | `在 backend/ 下 ..\.venv\Scripts\python.exe -m pytest tests -q` |
| 周期白名单 | `config.py:81` = `1m/5m/15m/1h/4h/1d` | `utils.py:3-16` `INTERVAL_MS` |
| 撮合签名 | `simulate_rgd(data, signals, params, initial, fee, slippage, interval)` | 与 `simulate_zigzag`/`simulate_ocb` 一致 |

---

## 1. 版本与溯源

- 源文件：`D:/UGit/get-money/data/rgd-v2-source.pine`，`//@version=6`，策略声明见 L5-7：
  `initial_capital=10000`、`default_qty_type=strategy.percent_of_equity`、`default_qty_value=100`、`commission_type=strategy.commission.percent`、`commission_value=0.05`、`slippage=2`。
- 脚本 12 个分节（按注释横幅）：① Inputs(L9-56) ② Helpers(L58-67) ③ Volatility regime(L69-72) ④ Adaptive breakout channel(L74-82) ⑤ Path efficiency(L84-87) ⑥ HTF liquidity tension(L89-99) ⑦ Price-time correlation & volume(L101-108) ⑧ Acceptance-weighted trailing defense(L110-121) ⑨ State(L123-155) ⑩ Trade management(L157-174) ⑪ Entries(L176-204) ⑫ Visuals(L206-236)。
- **属性默认值（Pine v6，脚本未显式覆盖，必须按默认）**：`pyramiding=0`、`calc_on_every_tick=false`、`calc_on_order_fills=false`、`process_orders_on_close=false`、`close_entries_rule="FIFO"`、`margin_long/margin_short=100%`（即 1× 无杠杆）、`commission_value=0.05%`、`slippage=2 ticks`。这些默认值决定了 §4/§5 的时序与成本，不可改。
- **本契约声明"逐行 1:1"的含义**：§3 逐表达式给出 Pine→Python 映射；`prepare()`/`signals()`/`simulate_rgd()` 三者的组合必须与脚本在同一 OHLCV 输入上产生同序、同价、同量的成交（在 §8 已裁决的边界内）。

---

## 2. 已锁定决策（用户拍板，不得更改）与 Out-of-Scope

### 2.1 四条锁定决策

| # | 决策 | 契约落点 |
|---|---|---|
| D1 | 主周期 **15m + HTF 4h**，`autoHTF=true` → `autoTF()` 对 15m 返回 `"240"` → 项目 interval `"4h"`（`interval_ms=14_400_000`）。4h native 已有 400.7 天/521 币；15m native 298.3 天/521 币（仅 57 币 ≥180 天，BNBUSDT 仅 10.4 天）。**HTF 在回测脚本内从已加载的 4h native 帧因果拼进 15m 帧，不落库、不派生 1d。** | §4 / §7.2 |
| D2 | **完整实现三种 `dirMode`**（默认 `"Long & Short"` 忠实脚本）；回测同时给「合约多空」与「现货仅多（Long Only）」两组读数。 | §6.2 / §7.1 |
| D3 | **成本两组都跑**：主报告 `fee_rate=0.0004 / slippage=0.0002`（项目全局）；对照 `commission 0.05%=0.0005 + slippage 2 ticks`（脚本原样，近似换算见 §8 裁决 R15）。 | §7.1 / §8 R15 |
| D4 | **交付边界**：spec + 专用 broker + 回测脚本 + 逐 bar 对齐测试 + docs 报告完整交付；`default_enabled=False` 不进实盘绑定；实盘持仓管理（TP1 限价挂单 + 动态 trailing 状态机）明确 Out-of-Scope / 后续 TODO。 | §6.6 / §2.2 |

### 2.2 Out-of-Scope（本次不做，杜绝镀金）

1. **实盘持仓管理**：`trading/order_manager.py`/`live.py` 的 TP1 限价挂单、动态 trailing、多挂单状态机——不改动。理由：D4 明确排除；且 `default_enabled=False` 故 live 路径不会调用本策略。
2. **1d 派生 / 周月线**：不派生 1d，不引入 `1W`/`1M`（`autoTF()` 对 1d 返回 `1W`，项目无此周期，见 §8 R1）。
3. **UI / API / DB 新表 / OpenAPI**：本任务无 REST 端点、无前端、无新表；`openapi.yaml` 不适用，契约以 §6 参数表 + §6.5 signals 列契约 + §5 broker 状态机为唯一依据（等同 spec-as-contract 的"点名接口"）。
4. **独立 ADR 文件**：受团队"只改两个文件"约束，不建 `docs/decisions/ADR-*.md`；§8「歧义裁决清单」以 MADR 字段（背景/决策/后果/关联）内联充当决策记录。
5. **通用 `simulate` 路径**：RGD 的 `signals()` 强制 `direction=0` + `block_reason="stop_order_requires_dedicated_broker"`，必须走 `simulate_rgd`，不得回退通用管线（§6.5）。

---

## 3. Pine → Python 逐行映射表（12 分节）

约定：`严格` = 逐位/逐 bar 可对齐；`偏差` = 已登记 OPEN 或 §8 裁决，附理由。`fN` 表示 `features/rgd.py` 的函数，`spec` 表示 `strategy/specs/rgd_donchian_v2.py`，`broker` 表示 `backtest/rgd_execution.py`。所有无前缀列均以 `rgd_` 命名（prepare 输出见 §6.4）。

### ① Inputs（L9-56）

| # | Pine（行号·表达式） | 本地实现（文件/函数） | 口径说明 | 严格/偏差 |
|---|---|---|---|---|
| I1 | L17 `dirMode` string options `["Long & Short","Long Only","Short Only"]` | `spec.DEFAULTS["dir_mode"]=0`，`INT_KEYS` | 项目 `resolve_params`(`rules.py:118-130`) 拒收字符串 → 用 int 枚举 `0/1/2`。`allowL=(dir_mode!=2)`、`allowS=(dir_mode!=1)`（Pine L138-139 `allowL=dirMode!="Short Only"`、`allowS=dirMode!="Long Only"`）。**语义等价，仅编码适配**。 | 严格（编码适配） |
| I2 | L31 `htfIn=input.timeframe("240")` + L32 `autoHTF=true` | `spec.DEFAULTS["auto_htf"]=True`、`["htf_in"]=1`（枚举 0→1h/1→4h/2→1d） | `auto_htf=True` 时由 §4 的 `autoTF()` 表决定 HTF，`htf_in` 仅 `False` 时生效；脚本默认 `"240"`→枚举 1。 | 严格（编码适配） |
| I3 | L18-52 其余 29 个 input（除 I1 `dirMode` / I2 `htfIn`·`autoHTF`） | `spec.DEFAULTS` / `spec.RANGES`（§6.2/§6.3） | 全部 1:1 默认值；`RANGES` 取脚本 `minval/maxval`，无 max 者给保守上限。脚本业务参数合计 32 个（I1 1 + I2 2 + I3 29）。`riskPct`(L48) 命名 `risk_pct`，**与项目 `risk_per_trade_pct` 区分**（后者 inert）。`cooldown`(L19) 命名 `cooldown`，**与项目 `cooldown_bars` 区分**（后者在 DEFAULTS 置 0 中和）。 | 严格 |
| I4 | L54-56 `colBull/colBear/showTbl` | 不实现（Visuals，§3⑫） | 纯样式，不影响任何成交。 | 严格（省略无影响） |

### ② Helpers（L58-67）

| # | Pine（行号·表达式） | 本地实现 | 口径说明 | 严格/偏差 |
|---|---|---|---|---|
| H1 | L59-61 `autoTF()` | `fN.auto_tf(base_interval) -> str`（§4.1 完整表） | `timeframe.in_seconds(period)` → 项目 `INTERVAL_MS`。`s<900→"60"`、`900≤s<3600→"240"`、`3600≤s<86400→"1D"`、`86400≤s<604800→"1W"`、`else→"1M"`。15m(900)→"240"→"4h"（D1）。 | 严格 |
| H2 | L63-67 `tensionStep(prev,tgt,b,r,cap)` | `fN.tension_step(...)` + `fN.tension_series(...)` 显式 `for i in range(n)` | 见 §3⑥ T4 逐条：`sameDir = prev==0 or tgt==0 or sign(prev)==sign(tgt)`；`building = sameDir and abs(tgt)>abs(prev)`；`a = 2/(b+1)` if building else `2/(r+1)`；`clamp(prev+a*(tgt-prev), -cap, cap)`。`math.sign(0)=0` → Python `_sign(x)=(x>0)-(x<0)`（返回 0 for 0）。`prev=nz(tension[1],0.0)` 首根 0。 | 严格 |

### ③ Volatility regime（L69-72）

| # | Pine | 本地实现 | 口径说明 | 严格/偏差 |
|---|---|---|---|---|
| V1 | L70 `atr = ta.atr(atrLen)` | `fN.wilder_atr(high,low,close,atr_len)` | **必须用 Wilder RMA（对齐 `supertrend_current.py:82-101 _rma`：「按前 length 个有效值的简单均值初始化」），不得用 `indicators.py:99-109 _atr` 的 `ewm(alpha=1/L,adjust=False)`（后者首根即出值、seed 不同）**。`tr = max(high-low, abs(high-prev_close), abs(low-prev_close))` 首根用 `high-low`（Pine `ta.tr(true)` handle_na）。Pine `ta.atr` 前 `atrLen-1` 根为 na，第 `atrLen-1` 根 = SMA(TR[0..L-1])，之后 `rma=(rma*(L-1)+tr)/L`。`ready` 含 `not na(atr)`，预热由 `rankLen` 主导，见 §8 R6。 | 严格（选型已裁决） |
| V2 | L71 `volRank = ta.percentrank(atr/close, rankLen)` | `fN.percentrank_preceding(src, length)`（§4.3 向量化） | **官方定义（已核验 Pineify v6 参考 + 手工例，`current=97`→0% 判定当前 bar 不入比较集）**：`volRank[i] = 100 * |{ j∈[i-length, i-1] : src[j] <= src[i] }| / length`。即**比较集 = 当前 bar 之前的 `length` 根，不含当前**；`<=`（相等计数）；分母 = `length`（不是 `length+1`/`length-1`）；窗口内任一 na → na；首个有效 bar = `length`（0-based）。源序列 = `atr/close`（不是 `atr_pct` 全局列，口径同值但独立计算）。**不得用 `rolling.rank(pct=True)`（含当前 bar 且用平均并列秩），不得用 `features.absorption.percentile_nearest_rank`（那返回分位数值而非秩）**。 | 严格（定义已核验） |
| V3 | L72 `highVol = volRank > rankCut` | `fN`: `vol_rank > rank_cut` bool | **严格 `>`**（非 `>=`）；`rankCut=50.0` 是 0-100 百分位数值（非 0.5）。 | 严格 |

### ④ Adaptive breakout channel（L74-82）

| # | Pine | 本地实现 | 口径说明 | 严格/偏差 |
|---|---|---|---|---|
| C1 | L75-78 `upFast/loFast/upSlow/loSlow = ta.highest(high[1],N)/ta.lowest(low[1],N)` | `high.shift(1).rolling(N,min_periods=N).max()` / `low.shift(1).rolling(...).min()` | `high[1]` 先 shift 再 rolling：bar i 的窗口 = `[i-1 .. i-N]`，**不含 bar i**。`min_periods=N`（Pine 预热期返回 na）。 | 严格 |
| C2 | L79-80 `entUp/entLo = highVol ? upSlow/loSlow : upFast/loFast` | `np.where(high_vol, up_slow, up_fast)` | 逐 bar 切换（highVol 逐 bar 变），`np.where` 保留 nan。 | 严格 |
| C3 | L81-82 `mid=(entUp+entLo)*0.5`、`chanW=max(entUp-entLo, syminfo.mintick)` | `fN`: `(ent_up+ent_lo)*0.5`、`np.maximum(ent_up-ent_lo, MINTICK)` | `MINTICK` 裁决见 §8 R5 = `1e-8` 固定极小正数（仅作除零/退化保护，非真实 tick）。 | 严格（R5 裁决） |

### ⑤ Path efficiency（L84-87）

| # | Pine | 本地实现 | 口径说明 | 严格/偏差 |
|---|---|---|---|---|
| E1 | L85 `noise = math.sum(abs(ta.change(close)), erLen)` | `close.diff().abs().rolling(er_len,min_periods=er_len).sum()` | 含当前 bar 的最近 `erLen` 根 `|close-close[1]|` 之和；`min_periods=er_len`（预热 na）。 | 严格 |
| E2 | L86 `er = ta.ema(noise>mintick ? abs(close-close[erLen])/noise : 0.0, 3)` | `raw=np.where(noise>MINTICK, (close-close.shift(er_len)).abs()/noise, 0.0)`；`er=Series(raw).ewm(span=3,adjust=False).mean()` | 效率比分子 `abs(close[i]-close[i-erLen])`、分母 `noise`（最近 erLen 根位移和）。Pine `ta.ema` seed=首值、`span=3`（alpha=2/4）。预热 na 处理见 §8 R6（决策无影响：EMA(3) 约 10 bar 内收敛，远早于 `ready`）。 | 严格（R6 已裁决 immaterial） |
| E3 | L87 `erOk = er >= erMin` | `er >= er_min`（0.30） | na→False（保守）。 | 严格 |

### ⑥ HTF liquidity tension（L89-99）—— 最关键因果点

| # | Pine | 本地实现 | 口径说明 | 严格/偏差 |
|---|---|---|---|---|
| T1 | L90 `htf = autoHTF ? autoTF() : htfIn` | §4.1 `auto_tf()` + §4.2 脚本侧 HTF 帧加载 | D1：15m→`"240"`→项目 `"4h"`。 | 严格 |
| T2 | L91 `[hh,ll] = request.security(syminfo.tickerid, htf, [high[1],low[1]], lookahead=barmerge.lookahead_on)` | `features/rgd_htf.py::attach_htf_levels(base_frame, htf_frame, htf_interval, as_of)`（§4.2） | **本地语义（团队已定）**：在 base bar 开盘时刻 `t`，只能看到满足 `htf_open + htf_step <= t` 的**最后一根已完成** HTF bar 的 `high/low`。实现：`pos = np.searchsorted(htf_stamps, base_stamps - htf_step, side="right") - 1`（对齐 `bucket_delta:145` 原语）；`pos<0 → na`。`lookahead_on + [1]` 的标准无重绘模式：HH/LL 在**同一 HTF 期间内跨 base bar 恒定**，仅在 HTF 边界更新。**未完成桶剔除**：`htf_frame = htf_frame.loc[htf_stamps + step <= as_of]`。 | 严格 |
| T3 | L92-95 `hr=hh-ll`、`hOk=not na(hh) and not na(ll) and hr>0`、`hm=(hh+ll)/2`、`np = hOk ? 2*(close-hm)/max(hr,mintick) : na` | `fN` 逐 bar 向量化 | **`np` 用 base bar 的 `close`（当前 15m 收盘），但 `hh/ll/hm/hr` 用上一根已完成 4h bar**。这是脚本刻意的混合口径，不得统一到 HTF close。`hOk=false → np=na`。`max(hr,mintick)`：因 `hOk` 已要求 `hr>0`，此处 guard 不触发。 | 严格 |
| T4 | L96 `tg = max(-maxTen, min(maxTen, not na(np) ? np*er : 0.0))` | `fN.tension_series(...)` | `er` 是 **base 周期** ER（⑤ E2），不是 HTF ER。`hOk=false → np=na → tg=0`（不是 na）。`tg` clamp ±8.25。 | 严格 |
| T5 | L97-98 `var tension=0.0; tension := tensionStep(nz(tension[1],0.0), tg, buildLen, relLen, maxTen)` | `fN.tension_series(tg_arr, ...)` 显式 `for i in range(n)` 状态机 | **必须显式循环**（自引用 `tension[1]` 无法向量化）。首根 `prev=0.0`。`nz(tension[1],0.0)`：即使某根 tg=na 致 tension=na，下一根 `prev` 由 `nz` 归 0 **自愈**。`building` 用 `a=2/(buildLen+1)=2/7≈0.2857`（快建），`release` 用 `a=2/(relLen+1)=2/21≈0.0952`（慢放）。 | 严格 |
| T6 | L99 `rg = tension>=tenTh ? 1 : tension<=-tenTh ? -1 : 0` | `fN`: `np.where(tension>=ten_th,1,np.where(tension<=-ten_th,-1,0))`，na→0 | `tenTh=0.20`。tension 为 na 时 `rg=0`（Pine 嵌套三元 na 条件取 else 分支等价）。 | 严格 |

### ⑦ Price-time correlation & volume（L101-108）

| # | Pine | 本地实现 | 口径说明 | 严格/偏差 |
|---|---|---|---|---|
| P1 | L102 `corr = nz(ta.correlation(close, bar_index, corrLen), 0.0)` | `fN.rolling_pearson(close, idx, corr_len)`（§4.4） | **Pearson 滚动相关**，`min_periods=corr_len`（默认档 30）。`bar_index` 用 `np.arange(n)`（局部 0-based）：**Pearson 对任一变量仿射变换不变**，`bar_index_local = bar_index_abs - offset`（斜率 1），逐段连续 → 与 Pine 绝对 bar_index **严格等价**。`nz(na,0)` → 0。length<2（预热）→ na → 0。 | 严格（仿射不变已证） |
| P2 | L103-104 `hasVol = not na(volume) and volume>0`；`relVol = hasVol ? volume/ta.sma(volume,20) : 1.0` | `fN`: `volume.rolling(20,min_periods=1).mean()` | `ta.sma(volume,20)` 预热用可用bar（Pine sma 语义）；无成交量 → `relVol=1.0`（bypass）。 | 严格 |
| P3 | L105 `volOk = not useVol or not hasVol or relVol>=volMin` | `fN`: na `relVol` → False | `volMin=1.0`。na 保守取 False（预热期无成交）。 | 严格 |
| P4 | L107-108 `bullOk=(not useTen or rg==1) and (not useCorr or corr>=corrMin)`；`bearOk=(not useTen or rg==-1) and (not useCorr or corr<=-corrMin)` | `fN` 逐 bar bool | `corrMin=0.25`。`useTen/useCorr` 默认 True。 | 严格 |

### ⑧ Acceptance-weighted trailing defense（L110-121）

| # | Pine | 本地实现 | 口径说明 | 严格/偏差 |
|---|---|---|---|---|
| A1 | L111-114 `fH/fL/sH/sL = ta.highest/lowest(high[1]/low[1], defFast/defSlow)` | 同 C1 模式 `shift(1).rolling(N,min_periods=N)` | `defFast=8`、`defSlow=21`。 | 严格 |
| A2 | L116 `pos = clamp((close-entLo)/chanW, 0, 1)` | `np.clip((close-ent_lo)/chan_w,0,1)` | 用**当前** close 与当前 `entLo/chanW`。 | 严格 |
| A3 | L117 `acceptance = ta.ema(pos, accLen)` | `Series(pos).ewm(span=6,adjust=False).mean()` | `accLen=6`。 | 严格 |
| A4 | L118-119 `bullQ=clamp((acceptance-0.35)/0.55,0,1)`；`bearQ=clamp((0.65-acceptance)/0.55,0,1)` | `np.clip(...)` | 硬编码 0.35/0.55/0.65，不得改。 | 严格 |
| A5 | L120 `defL = max(fL,sL)*(1-maxDepth*bullQ) + min(fL,sL)*(maxDepth*bullQ)` | `fN` 精确公式 | **精确写法**：`bullQ=0 → defL=max(fL,sL)`（更紧）；`bullQ=1 → defL=(1-maxDepth)*max(fL,sL)+maxDepth*min(fL,sL)`（向 min 移动＝更深；仅 `maxDepth=1` 时才 `=min`）。团队提示的"defL=min 更深"是 maxDepth→1 的简写，实现必须用完整加权式。 | 严格 |
| A6 | L121 `defS = min(fH,sH)*(1-maxDepth*bearQ) + max(fH,sH)*(maxDepth*bearQ)` | `fN` 精确公式 | 对称：`bearQ=0 → defS=min(fH,sH)`（更紧）；`bearQ=1 → 向 max 移动（更深）`。 | 严格 |

### ⑨ State（L123-155）

| # | Pine | 本地实现 | 口径说明 | 严格/偏差 |
|---|---|---|---|---|
| S1 | L124 `var state=0`（+L125-130 `stopLvl/tp1Lvl/entryRef/tp1Hit/lastCT/lastExit`） | `state` → `fN.state_machine`（**prepare 纯状态机**）；其余 → **broker** 状态字典（§5.2） | **关键拆分**：`state` 的转移只依赖 `prevState + longBrk/shortBrk + close/defL/defS`（全为纯 OHLCV/HTF 派生，不含成交态）⇒ **纯因果，放 `prepare`**，满足整帧/单 bar 等价；`stopLvl/tp1Lvl/entryRef/tp1Hit/lastCT/lastExit` 是**成交依赖**，只能在 broker（TP1 部分平 + 保本无法在 prepare 虚拟复现，见 §5.1）。 | 严格（架构裁决 §5.1） |
| S2 | L132-134 `if strategy.closedtrades>lastCT: lastCT:=strategy.closedtrades; lastExit:=bar_index` | `broker`: 镜像 `closed_count`（**全平**计数）；`if closed_count>last_ct: last_ct=closed_count; last_exit=i` | 语义等价 `strategy.closedtrades`（Pine 按全平计数，TP1 部分平不 bump）。**等价已证**：`coolOk` 仅门控 pullback，pullback 要求 `flat`；任何部分平的 bump 必早于其后的全平 bump，故 pullback 发生时 `lastExit` 恒等于「仓位归零那根」——不依赖部分平是否 bump（§8 R10，登记 OPEN-03 备 TV 核验）。 | 严格（等价已证） |
| S3 | L136 `ready = not na(entUp/entLo/defL/defS/er/atr)` | `fN`: `ready` bool 列 | **6 项**，不含 `corr/relVol/tension`（它们已 `nz` 或由 `volOk/bullOk/bearOk` 内部处理）。 | 严格 |
| S4 | L137 `flat = strategy.position_size==0` | `broker`: `position is None` | 仅 pullback 用。 | 严格 |
| S5 | L138-139 `allowL=dirMode!="Short Only"`；`allowS=dirMode!="Long Only"` | `fN`: `allow_long=(dir_mode!=2)`、`allow_short=(dir_mode!=1)` | 见 I1。 | 严格 |
| S6 | L140 `coolOk = bar_index - lastExit > cooldown` | `broker`: `i - last_exit > cooldown` | **严格 `>`**（非 `>=`）：`cooldown=5` 意即平仓那根之后至少隔 5 根，第 6 根才允许（`i-lastExit>=6`）。`lastExit` 初值 `-9999` ⇒ 开头恒 True。仅门控 pullback，不门控 breakout。 | 严格 |
| S7 | L141 `prevState = state` | `fN.state_machine`: 在 bar i 捕获 `state[i-1]` | 转移前值，供 `longBrk/shortBrk` 的 `!=1/!=-1` 与 `longPull/shortPull` 的 `==1/==-1` 门控。 | 严格 |
| S8 | L143 `longBrk = ready and close>entUp+brkBuf*atr and erOk and bullOk and volOk and prevState!=1` | `fN`: **纯条件，prepare 内完整计算**（只依赖 `prev_state` + 纯特征，无成交态） | 逐条：`ready`、`close > entUp + 0.10*atr`、`erOk`、`bullOk`、`volOk`、`prevState!=1`。`brkBuf=0.10`。 | 严格 |
| S9 | L144 `shortBrk = ready and close<entLo-brkBuf*atr and erOk and bearOk and volOk and prevState!=-1` | 同 S8 | `close < entLo - 0.10*atr`、`bearOk`、`prevState!=-1`。与 longBrk **互斥**（close 不能同时 >entUp 且 <entLo）。 | 严格 |
| S10 | L145 `longPull = usePull and ready and prevState==1 and flat and coolOk and bullOk and erOk and low<=mid and close>mid and close>open` | `fN.long_pull_core`（去掉 `flat`/`coolOk`）+ `broker` AND | core 含 7 项：`usePull and ready and prevState==1 and bullOk and erOk and low<=mid and close>mid and close>open`；`flat`/`coolOk` 成交依赖由 broker 补。 | 严格（拆分） |
| S11 | L146 `shortPull = usePull and ready and prevState==-1 and flat and coolOk and bearOk and erOk and high>=mid and close<mid and close<open` | `fN.short_pull_core` + `broker` AND | 对称：`high>=mid and close<mid and close<open`。与 longPull 互斥（prevState 不能同时 1 与 -1）。 | 严格（拆分） |
| S12 | L148-155 转移 | `fN.state_machine` **纯循环** | 顺序：`longBrk→1` else `shortBrk→-1` else `prevState==1 and close<defL→0` else `prevState==-1 and close>defS→0` else 保持。`else if` 里的 `state` 未被前面赋值（分支互斥）⇒ 等价于读 `prevState`。**`state` 完全纯因果 ⇒ 放 prepare**（§5.1）。 | 严格 |

> **⑨ 结论**：`state/prevState/longBrk/shortBrk/ready/allowL/allowS/longPull_core/shortPull_core` 全为**纯 OHLCV/HTF 派生**，归 `prepare`；`flat/coolOk/lastExit` 及交易管理态归 `broker`。这条拆分是整个移植能同时满足「严格 1:1」与「整帧/单 bar 等价」的枢纽。

### ⑩ Trade management（L157-174）—— 专用 broker 核心

| # | Pine | 本地实现 | 口径说明 | 严格/偏差 |
|---|---|---|---|---|
| X1 | L158-165 多头管理 | `broker.manage_position(side=+1)` | `if position_size>0: if high>=tp1Lvl: tp1Hit:=true; stopLvl:=max(stopLvl,defL); if tp1Hit: stopLvl:=max(stopLvl,entryRef); strategy.exit("L-TP1","Long",qty_percent=tp1Pct,limit=tp1Lvl,stop=stopLvl); strategy.exit("L-Trail","Long",stop=stopLvl)` | 严格 |
| X2 | L167-174 空头管理 | `broker.manage_position(side=-1)` | 对称：`low<=tp1Lvl` 置 `tp1Hit`；`stopLvl:=min(stopLvl,defS)`；`if tp1Hit: min(stopLvl,entryRef)`；`S-TP1`/`S-Trail`。 | 严格 |
| X3 | `stopLvl` 单调 ratchet | `broker` | 多头 `max` 单调不降、空头 `min` 单调不升；`defL/defS` 取**本 bar** 值；保本 `max(stopLvl,entryRef)`（多）/`min`（空）。 | 严格 |
| X4 | `tp1Hit` 时序（团队点名裁决） | `broker` + 挂单 | 脚本在 bar N 收盘用 `high[N]>=tp1Lvl` 置位 `tp1Hit`，同根算出的保本 `stopLvl` 随 `strategy.exit` 重挂，**成交最早在 bar N+1**（`calc_on_every_tick=false` + `process_orders_on_close=false`）。⇒ broker 必须在 bar N 收盘更新状态、把新 stop 挂到 N+1 才生效；本 bar 内不得用新 stop。见 §5.4 / §8 R12。 | 严格（R12） |
| X5 | `qty_percent=tp1Pct` 语义 | `broker.partial_close(frac=0.5)` | Pine `qty_percent` = **当前持仓** 的百分比（默认 50，占当前仓 50%）；TP1 限价成交平掉当前持仓 50%，剩余 50% 继续 trailing。裁决：**TP1 每次持仓仅触发一次**（`tp1Hit` latch，§8 R13），`L-TP1` 在 TP1 成交后从 pending 移除，剩余仓只由 `L-Trail` 管理。 | 严格（R13 裁决） |
| X6 | 同仓多挂单 | `broker.pending` | `L-TP1`（带 `limit` + `stop` 的 bracket）与 `L-Trail`（仅 `stop`）是**同一持仓的两个 exit order**，stop 价相同 = `stopLvl`。stop 触及时两者同时触发 → 净效果 **剩余仓位全平在 stopLvl**（L-TP1 的 50% + L-Trail 的 100% 合并 = 全部，Pine 按可用仓位封顶）。QA 断言：stop 触发后仓位为 0。 | 严格 |
| X7 | 同根 TP1 与 stop 优先级（团队点名裁决） | `broker` | **保守：stop 优先**（参照 `risk.py:8-20 stop_exit` 先判 stop 再判 target）。**R14 仅管辖盘中路径**：`open` 未穿 `tp1Lvl` 且某 bar 同时 `high>=tp1Lvl` 且 `low<=stopLvl` → stop 全平，不给 TP1 盘中部分成交机会。**开盘跳空穿 `tp1Lvl`** 的既成部分平（`tp1_fill(o)` @ open，更优价）不受 R14 约束 → 其后同根 `low<=stopLvl` 剩余仓按 stop 全平（两段成交，A30）。跳空穿 stop 仍按 open 全平。登记 §8 R14。 | 偏差（保守裁决 R14，边界见 A30） |
| X8 | 跳空穿越 | `broker` | 开盘价已穿越 stop（多头 `open<=stopLvl`）→ 按 **open** 全平；开盘已穿越 limit（多头 `open>=tp1Lvl`）→ 按 **open** 部分平 `tp1_pct%`（比 limit 更优，既成事实、不受 R14 约束），剩余仓继续参与本根 stop 判定。与 `stop_exit` / `zigzag_execution.process_at_open` 一致。 | 严格 |

### ⑪ Entries（L176-204）

| # | Pine | 本地实现 | 口径说明 | 严格/偏差 |
|---|---|---|---|---|
| N1 | L177-181 多头初始止损与仓位 | `fN` 纯价位 `rgd_stp_long/rgd_rd_long/rgd_tp1_long` + `broker` 判 `rd>0` | `rawStop=max(defL, close-maxStopATR*atr)`(2.5)；`stp=min(rawStop, close-minStopATR*atr)`(0.5)；`rd=close-stp`；`if rd>0` 才继续。`stp` 落在 `[close-2.5ATR, close-0.5ATR]` 且 `<=defL`（defL 更紧时取 defL，除非比 minStop 还紧则取 close-0.5ATR）。 | 严格 |
| N2 | L182 `q = min(strategy.equity*riskPct/100/rd, strategy.equity/close)` | `broker`（需 equity，§5.6） | `riskPct=1.0`；`q` 是**币数量（units）** 非名义；上限 `equity/close` = 100% equity 名义（`default_qty_value=100` + `margin=100%` ⇒ 1× 无杠杆）。首项 = 风险预算法（止损时亏 equity×1%）。 | 严格 |
| N3 | L183-187 多头状态赋值 + 下单 | `broker` | `stopLvl:=stp`；`entryRef:=close`（**信号 bar 收盘价，非成交价**）；`tp1Lvl:=close+tp1R*rd`(2.0)；`tp1Hit:=false`；`strategy.entry("Long",strategy.long,qty=q)` → 市价单 **bar N+1 开盘成交**（N 收盘下单）。 | 严格 |
| N4 | L189-199 空头对称 | `broker` | `rawStop=min(defS, close+2.5ATR)`；`stp=max(rawStop, close+0.5ATR)`；`rd=stp-close`；`q` 同式；`tp1Lvl=close-2.0*rd`；`strategy.entry("Short",qty=q)`。 | 严格 |
| N5 | L201-204 Trend flip | `broker` | `if state==-1 and not allowS and position_size>0: strategy.close("Long",comment="Trend flip")`；`if state==1 and not allowL and position_size<0: strategy.close("Short",...)`。**仅 Long Only / Short Only 模式生效**（双向模式 `allowL/allowS` 恒 True 不触发）。市价平仓单 N 收盘挂、N+1 开盘成交。与双向模式的**反手**（`strategy.entry` 反向）语义不同：flip 是**纯平仓**，反手是**平旧+开新**。 | 严格 |
| N6 | `strategy.equity` 口径（团队点名裁决） | `broker.equity(i)`（§5.6 / §8 R11） | Pine `equity = initial_capital + netprofit(已平仓, 已扣费) + openprofit(持仓盯市, 毛)`。本地：`cash`（已实现，含已平仓各腿净 PnL）+ `(close_i - entry_fill)*qty*side`（持仓毛浮动）。用于 N2 的 `q` 与逐 bar 盯市净值。 | 严格（R11） |
| N7 | `strategy.entry` 反转 / `pyramiding=0`（已核验官方） | `broker` | 反方向 `strategy.entry` **反手**：先平旧仓再开新仓，成交数量 = 旧仓数量 + 新 q（官方 remark：「An entry in the opposite direction reverses the open position by default. The transaction size can therefore be larger than the requested resulting position.」），最终持仓 = 新 q。同方向 `strategy.entry` 且 `pyramiding=0` → **忽略**（不加仓）。局部费用各腿分别计，等价。 | 严格 |
| N8 | 入场 bar 裸奔（团队点名裁决） | `broker` | 信号 bar N 收盘 `position_size==0` ⇒ 管理块不挂 exit；N+1 开盘成交后 N+1 收盘才首次提交 exit ⇒ **N+1 整根无止损保护**。broker 必须复现（不得在成交瞬间即挂止损）。见 §5.4 / §8 R12。 | 严格（R12） |

### ⑫ Visuals（L206-236）

| # | Pine | 本地实现 | 口径说明 | 严格/偏差 |
|---|---|---|---|---|
| Z1 | L206-236 `plot/fill/plotshape/barcolor/table` | 不实现 | 纯展示，零成交影响；`activeDef`(L208) 只是 `state==1?defL:state==-1?defS:na` 的展示别名，无额外逻辑（`defL/defS/state` 已在 ⑧⑨ 计算）。Out-of-Scope §2.2。 | 严格（省略无影响） |

---

## 4. 数据层方案（15m 主 + 4h HTF 因果 join）

### 4.1 `autoTF()` 完整映射表（D1 锁定 15m→4h）

| 主周期 | `timeframe.in_seconds` | Pine `autoTF()` 返回 | 项目 HTF interval | `interval_ms` | 可行性 |
|---|---|---|---|---|---|
| 1m | 60 | `"60"` | `1h` | 3_600_000 | 可行（未测） |
| 5m | 300 | `"60"` | `1h` | 3_600_000 | 可行（未测） |
| **15m** | **900** | **`"240"`** | **`4h`** | **14_400_000** | **D1 锁定（本次）** |
| 1h | 3600 | `"1D"` | `1d` | 86_400_000 | 可行（未测） |
| 4h | 14400 | `"1D"` | `1d` | 86_400_000 | 可行（未测） |
| 1d | 86400 | `"1W"` | （无） | 604_800_000 | **不支持** → `validate`/脚本 raise（§8 R1） |

> `"1M"` = 日历月（无固定 ms），仅 base ≥1W 可达，不在项目白名单 → 恒不支持。
> **D1 铁律**：HTF 帧由 **`load_frame(symbol,"4h",start_ms,end_ms)` 的 4h native** 直接取得，**不落库、不派生 1d**。`auto_htf=False` 时 `htf_in` 枚举 0/1/2 → `1h/4h/1d`（均在白名单）。

### 4.2 HTF 因果 join 伪代码（`features/rgd_htf.py::attach_htf_levels`）

```python
import numpy as np
import pandas as pd
from ..utils import interval_ms

def attach_htf_levels(base: pd.DataFrame, htf: pd.DataFrame,
                      htf_interval: str, as_of: int) -> pd.DataFrame:
    """把【上一根已完成】HTF bar 的 high/low 因果拼进 base(15m) 帧。
    等价 request.security(tickerid, htf, [high[1],low[1]], lookahead=barmerge.lookahead_on)。
    返回 base 的 copy，新增 rgd_htf_hh / rgd_htf_ll / rgd_htf_ok 三列。
    纯因果：base bar i 只依赖 htf_open + step <= t_i 的桶。
    """
    out = base.copy()
    step = interval_ms(htf_interval)                 # 4h = 14_400_000
    htf_stamps = htf.index.as_unit("ms").asi8        # HTF open_time（升序）
    base_stamps = out.index.as_unit("ms").asi8        # base open_time t

    # ① 剔除未完成桶：只保留 open + step <= as_of 的【已收盘】HTF bar
    closed = (htf_stamps + step) <= as_of
    htf_stamps = htf_stamps[closed]
    hh_src = pd.to_numeric(htf["high"], errors="coerce").to_numpy()[closed]
    ll_src = pd.to_numeric(htf["low"],  errors="coerce").to_numpy()[closed]

    if len(htf_stamps) == 0:                          # 防御：空 HTF 帧 → 全 na
        out["rgd_htf_hh"] = np.nan
        out["rgd_htf_ll"] = np.nan
        out["rgd_htf_ok"] = False
        return out

    # ② searchsorted 对齐：最后一根满足 htf_open + step <= t 的 HTF bar
    #    等价于 side="right" 于 (t - step)：pos = 最大 j 使 htf_stamps[j] <= t - step
    pos = np.searchsorted(htf_stamps, base_stamps - step, side="right") - 1
    valid = pos >= 0
    pos = np.clip(pos, 0, len(htf_stamps) - 1)
    hh = np.where(valid, hh_src[pos], np.nan)
    ll = np.where(valid, ll_src[pos], np.nan)

    out["rgd_htf_hh"] = hh
    out["rgd_htf_ll"] = ll
    out["rgd_htf_ok"] = valid & np.isfinite(hh) & np.isfinite(ll) & ((hh - ll) > 0)
    return out
```

**因果红线（团队点名：`request.security(...,[high[1],low[1]], lookahead_on)` 的本地语义）**：
- 在 base bar 开盘时刻 `t`，只能看到 `htf_open + htf_step <= t` 的最后一根**已完成** HTF bar。`side="right"` 于 `t-step` 精确保证这点（严禁用 `side="left"`，严禁用 `t` 而非 `t-step`，严禁用未完成桶）。
- **不重绘性**：同一 4h 桶内的 16 根 15m bar 取到**同一个** `pos` ⇒ `hh/ll` 完全相同；只有 `base_stamps` 跨过 `htf_open+step` 时 `pos` 才 +1。QA A9 断言此唯一值。
- **未完成桶剔除**：base 帧同样须 `frame.loc[(frame.index + base_step) <= as_of]`（沿用 `supertrend_backtest.py:140`）；HTF 帧用 `closed` 掩码。
- **HTF 帧起点**：`htf_start = base_start - 2*step`（多取 2 个 4h 桶），否则首根 base bar `pos<0 → hOk=false`（虽可自愈但不必要丢预热）。HTF 覆盖 400.7 天 > base 298.3 天，**HTF 从不瓶颈**。
- `syminfo.tickerid` → 同一 symbol，`load_frame` 直取，无跨品种。
- `syminfo.mintick` → §8 R5 的 `1e-8`。

### 4.3 `ta.percentrank` 本地实现（已核验定义，向量化）

```python
def percentrank_preceding(src: np.ndarray, length: int) -> np.ndarray:
    """Pine ta.percentrank(source, length)：
        100 * |{ j in [i-length, i-1] : src[j] <= src[i] }| / length
    比较集 = 当前 bar 之前的 length 根（【不含当前】）；<= 含相等；窗口含 na 则 na；
    首个有效位置 = length（0-based）。返回 0..100 的 float。
    依据：Pineify v6 参考 + 手工例 length=5, prev=[98,100,102,104,108]：
          cur=105 -> 4/5=80, cur=97 -> 0, cur=102 -> 3/5=60, cur=108 -> 5/5=100。
    """
    n = src.size
    out = np.full(n, np.nan, dtype=float)
    if n < length + 1 or length <= 0:
        return out
    win = np.lib.stride_tricks.sliding_window_view(src, length + 1)  # win[k]=src[k:k+length+1]
    prev = win[:, :-1]          # src[k .. k+length-1]  (length 根, 不含当前)
    cur  = win[:, -1:]          # src[k+length]         (当前值)
    finite = np.isfinite(prev).all(axis=1) & np.isfinite(cur[:, 0])
    leq = (prev <= cur).sum(axis=1)                 # 相等计入
    vals = np.where(finite, 100.0 * leq / float(length), np.nan)
    out[length:] = vals                            # out[i] <- k = i-length
    return out
```

> **三个不许**：① 不许用 `rolling().rank(pct=True)`（含当前 bar 且用平均并列秩，与 `<=` 定义不符）；② 不许用 `features.absorption.percentile_nearest_rank`（那是 NIST nearest-rank，返回**分位数值**而非秩，是另一个函数）；③ 不许把分母写成 `length+1` 或 `length-1`（官方 `length=5 → /5`）。QA A3 逐一核对 4 个手工值。

### 4.4 `ta.correlation(close, bar_index, corrLen)`

```python
def rolling_pearson(x: pd.Series, y: pd.Series, length: int) -> pd.Series:
    return x.rolling(length, min_periods=length).corr(y)   # length<2 -> na -> nz(...,0)
```

- **仿射不变性证明**：`corr(x, a*y+b) = sign(a)·corr(x,y)`。`bar_index_local = bar_index_abs - offset`（斜率 a=1，b=-offset）⇒ **完全相等**。逐段（segments 已按 gap 切分，段内连续）成立，故用 `np.arange(n)` 与 Pine 绝对 `bar_index` 严格等价（§8 R7）。
- 预热 `length<2` 与零方差 → na → `nz(...,0)` → `bullOk/bearOk` 因 `corr>=0.25`/`<=-0.25` 不成立而 False（保守，无成交）。

### 4.5 预热根数 `lookback(params)`

```python
def lookback(params) -> int:
    return max(600, 4 * max(int(params["rank_len"]), int(params["brk_slow"]),
                            int(params["def_slow"]), int(params["corr_len"]),
                            int(params["atr_len"]), int(params["er_len"]), 20))
```

- 默认 = `max(600, 4*100) = 600`。`percentrank` 需 `rank_len+1=101` 根出首值；`brk_slow=34 / def_slow=21 / corr_len=30 / atr=14 / er=10+ema3 / sma(vol,20)` 均 ≪ 600；tension 慢放 α≈0.095（时间常数 ~10 bar）600 根足够收敛。
- 与项目惯例一致（`supertrend_current.lookback` 亦 `max(600, 4*...)`）。
- 脚本侧 `MIN_SEGMENT_BARS = max(min_bars, lookback + 200)`；HTF 预热另需 ~37 个 4h 桶（600 bar 15m = 150h），由 `htf_start = base_start - 2*step` + 窗口内 4h 数据保证。

---

## 5. 专用 broker 详细设计（`backtest/rgd_execution.py::simulate_rgd`）

签名与 `simulate_zigzag`/`simulate_ocb` 对齐：

```python
def simulate_rgd(data, signals, params, initial, fee, slippage, interval):
    """返回 (逐 bar 盯市权益 Series, 成交记录 list[dict])；equity.attrs['open_position'] 记录末根未平仓。"""
```

### 5.1 架构裁决：为什么状态分两处（依据 generated-code-failure-modes「沉默逻辑错误」防线）

| 层 | 承载 | 理由 |
|---|---|---|
| `prepare(frame, params, ctx)` | 全部纯 OHLCV/HTF 派生列 + **`rgd_state`/`rgd_prev_state`（纯状态机）** | 必须幂等、纯因果，`ctx.service is None` 可降级；`engine.evaluate` 单取 `data.iloc[-1:]` 与整帧等价（`registry.PrepareContext:27-36`、`rules.evaluate_signals:156-174`） |
| `signals(data, params)` | **纯投影**：`SIGNAL_COLUMNS`（neutral）+ 自定义列；不做成交态判断 | 只输出「core 条件 + 纯价位」，不含 `flat/coolOk/lastExit` |
| `simulate_rgd(...)` | 唯一持有成交态：`position / pending / lastExit / lastCT / tp1Hit / stopLvl / entryRef / tp1Lvl / closed_count` | **TP1 部分平 + 保本 + trailing 是成交依赖**，无法在 prepare 里虚拟复现（不像 `simple_zigzag_double.virtual_pos` 纯价格穿越可虚拟）；若塞进 prepare 会与 broker 形成**双状态机必然漂移**（沉默逻辑错误，编译/冒烟都过、结果错） |

- **等价性界定**：`prepare`/`signals` 的纯列满足 `frame` 与 `frame.iloc[-1:]` 逐位一致；成交态只在整帧 broker 存在。live 绑定 Out-of-Scope（D4 / `default_enabled=False`），故**不要求 broker 的单 bar 等价**——这是对 spec-as-contract「两种调用等价」的显式范围界定，不是漏洞。
- `signals()` **强制 `direction=0` + `block_reason="stop_order_requires_dedicated_broker"`**（照抄 `simple_zigzag_double.py:119-132`），确保通用 `execution.simulate` 即使被误调也只会产生「零成交」的可见安全失败，而非错误成交。

### 5.2 状态与 pending 结构

```python
state = {
  "cash": float,                  # 已实现资金（含已平仓各腿净 PnL，已扣费）
  "position": None | {
      "side": +1|-1, "qty": float, "entry": float, "entry_fee": float,
      "time": stamp, "index": int, "initial_qty": float,
      "stop_lvl": float, "entry_ref": float, "tp1_lvl": float, "tp1_hit": bool,
  },
  "pending": {},                  # order_id -> order；持久挂单，成交/取消才移除
  "last_ct": 0,                   # strategy.closedtrades 镜像（全平计数）
  "last_exit": -9999,             # bar_index
  "closed_count": 0,              # 全平次数
  "market_next": [],              # 上一根收盘提交的市价单，本根开盘成交
}

order = {                          # pending 中的挂单
  "id": str,                      # "L-TP1" / "L-Trail" / "S-TP1" / "S-Trail" / "Long" / "Short" / "CloseLong" / "CloseShort"
  "kind": "entry"|"exit_tp1"|"exit_trail"|"close",
  "from_entry": "Long"|"Short",  # 绑定的入场 ID；反手平旧仓时整组取消
  "side": +1|-1,                  # 成交方向：开多=+1 开空=-1 平多=-1 平空=+1
  "limit": float|None,            # TP1 限价（exit_tp1）
  "stop": float|None,             # 止损（exit_tp1 bracket + exit_trail 共用 stopLvl）
  "qty": float|None,              # 市价单数量（entry/close）
  "qty_percent": float|None,      # TP1 = tp1_pct（占【当前持仓】）
  "armed_bar": int,               # 提交所在 bar N；仅在 i > N 时生效（成交最早 N+1）
}
```

**不变量**：
- `stop_lvl` **单调**：多头 `max`（不降）、空头 `min`（不升），永不回退。
- `tp1_hit` **latch**：一旦 True 不再 False，直到新仓 `strategy.entry` 重置。
- 每 bar 收盘 `strategy.exit` **重挂**（同 id 覆盖参数），新 stop 生效于下一根。
- **TP1 一次性**：TP1 成交后 `exit_tp1` 从 pending 移除，剩余仓仅 `exit_trail`（§8 R13）。
- 反手/翻转平掉旧仓时，**取消该 `from_entry` 的全部 pending**。

### 5.3 逐 bar 处理顺序（伪代码）

```python
for i, row in enumerate(rows):            # row: open/high/low/close, stamp
    sig = signals.iloc[i]

    # ① 开盘：上一根收盘提交的市价单在此成交（Pine 市价单 next open）
    process_market_orders_at_open(row.open, stamp)     # entry / close / reversal

    # ② 盘中：按 OHLC 路径撮合【已生效】挂单（armed_bar < i）
    process_intrabar(row.open, row.high, row.low, stamp)

    # ③ 收盘：脚本执行（Pine 每 bar 仅一次，calc_on_every_tick=false）
    if state.closed_count > state.last_ct:             # L132-134
        state.last_ct = state.closed_count
        state.last_exit = i
    flat = (state.position is None)                    # L137
    cool_ok = (i - state.last_exit) > params["cooldown"]   # L140
    long_brk   = bool(sig.rgd_long_break)              # 纯，已含 prevState!=1
    short_brk  = bool(sig.rgd_short_break)
    long_pull  = bool(sig.rgd_long_pull_core)  and flat and cool_ok
    short_pull = bool(sig.rgd_short_pull_core) and flat and cool_ok
    allow_l, allow_s = bool(sig.rgd_allow_long), bool(sig.rgd_allow_short)

    # ③-a 交易管理（L158-174）：用【已有持仓】，重挂 exit → 生效 N+1
    manage_position(sig, row.high, row.low, stamp, i)

    # ③-b 入场（L177-199）：市价单 → 成交 N+1 open
    if (long_brk or long_pull) and allow_l:  submit_long_entry(sig, row.close, i)
    if (short_brk or short_pull) and allow_s: submit_short_entry(sig, row.close, i)

    # ③-c Trend flip（L201-204）：市价平仓 → 成交 N+1 open
    if sig.rgd_state == -1 and not allow_s and pos_long:  submit_close("Long", i)
    if sig.rgd_state ==  1 and not allow_l and pos_short: submit_close("Short", i)

    # ④ 盯市净值
    values.append(mark(row.close))
```

**顺序铁律**：③-a 管理在 ③-b 入场**之前**（Pine 源码 L157 管理 → L176 入场 → L201 flip）。因此**信号 bar 新开的仓当根不挂 exit**（收盘脚本视角 `position_size` 仍是旧值/0）⇒ **入场 bar 裸奔**（N8）。这是最易被实现者"顺手修掉"从而破坏 1:1 的地方，QA A17 必须守住。

### 5.4 OHLC 路径撮合（对齐 `zigzag_execution.py:139` / `risk.py:8-20`）

```python
# 路径：离开盘更近的极值先走；平票 -> 先低后高
# 【以代码行为为准】zigzag_execution.py:139 与 ocb_execution.py:140 均为
#   `abs(o-h)<abs(o-l) ? (h,l) : (l,h)`，平票走 else -> (low,high) = 先低后高。
# 注意：ocb_execution.py:139 的【注释文字】写「距离相等时先高后低」，与其 :140
#   代码矛盾 —— 勿被注释误导，两个 broker 实际都是先低后高（A29 断言守）。
path = [o, h, l, c] if abs(o - h) < abs(o - l) else [o, l, h, c]

def process_intrabar(o, h, l, stamp):
    p = state.position
    if p is None:
        return
    S, T = p["stop_lvl"], p["tp1_lvl"]
    side = p["side"]
    # (1) 开盘跳空，stop 优先于 limit：
    if S > 0 and (o - S) * side <= 0:                 # 多头 o<=S / 空头 o>=S
        close_at(o, stamp, "stop_loss"); cancel_all(p["from_entry"]); return
    tp1_at_open = False                               # 开盘是否已穿 limit 的既成部分成交
    if T > 0 and (o - T) * side >= 0:                 # 多头 o>=T / 空头 o<=T
        tp1_fill(o, stamp)                            # 既成事实：以 open（更优价）部分平 tp1_pct%
        tp1_at_open = True                            # 不受 R14 约束；剩余仓继续，本根仍可能被 stop 打掉
    # (2) 同根 stop 触及 -> 剩余仓按 stop 全平
    #     R14「stop 优先、不给 TP1 部分成交」仅适用【盘中路径】：即 open 未穿 tp1Lvl
    #     的情形（tp1_at_open=False）；此时即便 high>=T 也不做 TP1 部分成交。
    #     若 (1) 已在 open 部分成交（tp1_at_open=True），那是既成事实，不受 R14 约束，
    #     本步只把【剩余 (1-tp1_pct)%】按 stop 全平 —— 共两段成交，见 A30。
    if S > 0 and ((l <= S) if side == 1 else (h >= S)):
        # open 已在 stop 内侧（开盘穿 stop 已在 (1) return），故成交价就是 S，无 gap 变量
        close_at(S, stamp, "stop_loss")               # 盘中双触且未开盘穿 T：不给 TP1（R14）
        cancel_all(p["from_entry"]); return
    # (3) 仅 TP1 盘中触及（open 未穿 T、盘中未触 stop）-> 部分平 tp1_pct%，移 exit_tp1，余仓 exit_trail 继续
    if T > 0 and ((h >= T) if side == 1 else (l <= T)) and not p["tp1_hit"] and not tp1_at_open:
        tp1_fill(T, stamp)
    # (4) 其余沿 path 逐段做价格穿越判定（bracket 的 stop/limit 同 stopLvl）
```

- **stop 优先**（R14，**仅盘中路径**）：`open` 未穿越 `tp1Lvl` 且某 bar `high>=tp1Lvl` 与 `low<=stopLvl` 同现 → 先按 stop 全平，**不给 TP1 部分成交机会**。**开盘跳空穿 `tp1Lvl` 的既成部分成交不受 R14 约束**：`tp1_fill(o)` 已按 open 部分平 `tp1_pct%`，其后同根 `low<=stopLvl` 则剩余 `1-tp1_pct%` 按 stop 全平（两段成交，A30）。
- **跳空**：`open<=stopLvl`(多)/`open>=stopLvl`(空) → `open` 全平；`open>=tp1Lvl`(多)/`open<=tp1Lvl`(空) → `open` 部分平 `tp1_pct%`（更优价）。
- **bracket 合并**（X6）：`exit_tp1.stop` 与 `exit_trail.stop` 同为 `stopLvl`，stop 触发时 50%+100% 合并 = 全平；TP1 触发时仅 `exit_tp1` 部分平 50%，`exit_trail` 留存。
- **市价反手**（N7）：反向 `entry` 且持仓反向 → 先 `close_at(open,"signal_reverse")` 全平（成交数量 = 旧 qty + 新 q，两腿费用分别计），再开新仓 `qty=新 q`，取消旧仓全部 pending；同向 `entry` 且 `pyramiding=0` → 忽略。

### 5.5 TP1 / 保本 / trailing / 反手 时序图（ASCII，多头）

```
时间 →  bar N-1 close        bar N open          bar N intrabar       bar N close              bar N+1 ...
        ─────────────────────────────────────────────────────────────────────────────────────────────────
信号    longBrk=True         市价单成交          无已生效 exit         manage: 首次挂          exit 生效
        (prevState!=1)      @open*(1+slip)      (入场 bar 裸奔 N8)    L-TP1(limit=T,stop=stp)  stop@stp/T@open
        strategy.entry      position=long q     ← 不触发任何 exit      L-Trail(stop=stp)
        ("Long",qty=q)                                                 armed_bar=N
状态    stopLvl:=stp         —                   —                    tp1Hit = high[N]>=T ?   tp1Hit latch
        entryRef:=close[N-1] (信号 bar 收盘价)                          (本根 high 置位,           stopLvl:=max(stp,defL[N])
        tp1Lvl :=close[N-1]+2R*rd                                      保本生效于 N+1 R12)      if tp1Hit: max(..,entryRef)
        tp1Hit :=false                                                                          重挂 -> 生效 N+2
        ─────────────────────────────────────────────────────────────────────────────────────────────────
        ↑ 信号 bar            ↑ 成交 bar          ↑ 裸奔                 ↑ 首次挂 exit            ↑ 保本滞后一根
```

- 空头对称：全部 `max→min`、`high→low`、`>=→<=`、符号取反。
- `trend flip`（N5）在 Long Only 模式：`state==-1 and not allowS and pos_long` → `strategy.close("Long")` 市价平仓单 N 收盘挂、N+1 开盘成交；**不反手开空**。

### 5.6 `equity` 口径（§8 R11）

```python
def mark(close_i):
    p = state.position
    return state.cash + ((close_i - p["entry"]) * p["qty"] * p["side"] if p else 0.0)
# 用于 N2 的 q 与逐 bar values：equity_i = mark(close_i)
```

- `cash` = `initial + Σ(已平仓各腿 net)`，每腿 `net = gross − 该腿分摊的 entry_fee − exit_fee`。
- `openprofit` 用**毛**浮动（不扣未发生的平仓费），与 Pine `strategy.equity = initial_capital + netprofit + openprofit` 一致。
- `q = min(equity_i * risk_pct/100 / rd, equity_i / close_i)`：在**信号 bar N 收盘**的 equity 上计算（N 收盘下单，N+1 开盘成交）。
- 保证金 1× 无杠杆：`margin = qty*fill`，`entry_fee = margin*fee`；affordability `qty = min(qty, cash/(fill*(1+fee)))`（沿用 `ocb_execution.py:102-104`，保守夹逼，偏差 ≤ 滑点+费量级，§8 R11）。

### 5.7 成本、初始资金、收尾

- 开仓 `fill = price*(1+side*slippage)` 收 `abs(fill*qty)*fee`；平仓 `price*(1-side*slippage)` 收 fee；一个来回 = 2 fee + 2 滑点（与 `execution.py:70-73` 同口径）。
- 两组：主 `fee=0.0004 / slip=0.0002`；对照 `fee=0.0005 / slip≈2ticks`（R15）。初始资金 **10000**（`initial_capital=10000`），每币独立账户，段间复利（继承 `supertrend_backtest.py:199-201`）。
- **不强制收盘平仓**：Pine 结束时持仓保留（盯市计入净值），`equity.attrs["open_position"]` 记录未平仓（对齐 `zigzag_execution.py:173-179`）；与通用 `simulate` 的 `close_at_end`（`execution.py:301-303`）**不同**，须显式实现。

---

## 6. spec 契约（`backend/app/strategy/specs/rgd_donchian_v2.py`）

### 6.1 标识

```python
CODE = "rgd_donchian_v2"
NAME = "Regime-Gated Donchian v2 [RGD]"
```

### 6.2 DEFAULTS（脚本 1:1 默认值 + 项目必需中和键）

```python
DEFAULTS = {
    # —— 脚本 1:1（Pine L17-52）——
    "dir_mode": 0,          # 0=Long & Short 1=Long Only 2=Short Only（I1 编码适配）
    "use_pull": True,       # L18
    "cooldown": 5,          # L19（≠ 项目 cooldown_bars）
    "brk_fast": 14, "brk_slow": 34, "atr_len": 14, "rank_len": 100,
    "rank_cut": 50.0, "er_len": 10, "er_min": 0.30, "brk_buf": 0.10,
    "use_ten": True, "htf_in": 1, "auto_htf": True,   # htf_in: 0=1h 1=4h 2=1d
    "build_len": 6, "rel_len": 20, "max_ten": 8.25, "ten_th": 0.20,
    "use_corr": True, "corr_len": 30, "corr_min": 0.25,
    "use_vol": True, "vol_min": 1.0,
    "def_fast": 8, "def_slow": 21, "acc_len": 6, "max_depth": 0.70,
    "risk_pct": 1.0, "max_stop_atr": 2.5, "min_stop_atr": 0.5,
    "tp1_r": 2.0, "tp1_pct": 50.0,
    # —— 项目必需但 RGD 不用（中和，避免通用管线误触发）——
    "cost_filter": False, "regime_filter": False, "funding_filter": False,
    "trailing_enabled": False,          # 防误用 risk.trailing_stop
    "max_hold_bars": 0, "min_hold_bars": 0, "cooldown_bars": 0,
    "max_signal_age_bars": 2, "min_atr_cost_ratio": 0.5,
    "min_net_reward_risk": 1.0, "max_atr_pct": 0.03, "trail_activation_atr": 1.5,
    # [注意] fee_rate / slippage 严禁写入 DEFAULTS 或 overrides（rules.py:105-109 硬禁止）；
    #   只能走 state 或回测请求 --fee/--slippage。
}
```

### 6.3 RANGES / INT_KEYS / BOOL_KEYS

```python
RANGES = {                      # 取脚本 minval/maxval，无 max 者给保守上限
    "dir_mode": (0, 2), "cooldown": (0, 10000),
    "brk_fast": (2, 500), "brk_slow": (3, 500), "atr_len": (2, 500),
    "rank_len": (20, 2000), "rank_cut": (10, 90), "er_len": (2, 500),
    "er_min": (0.05, 0.95), "brk_buf": (0.0, 20.0),
    "htf_in": (0, 2), "build_len": (1, 100), "rel_len": (1, 100),
    "max_ten": (0.25, 10.0), "ten_th": (0.05, 2.0),
    "corr_len": (5, 500), "corr_min": (0.0, 0.95), "vol_min": (0.1, 100.0),
    "def_fast": (2, 500), "def_slow": (3, 500), "acc_len": (1, 500),
    "max_depth": (0.0, 1.0),
    "risk_pct": (0.1, 10.0), "max_stop_atr": (0.5, 20.0), "min_stop_atr": (0.1, 20.0),
    "tp1_r": (0.5, 50.0), "tp1_pct": (10.0, 90.0),
}
INT_KEYS = ("dir_mode", "cooldown", "brk_fast", "brk_slow", "atr_len", "rank_len",
            "er_len", "htf_in", "build_len", "rel_len", "corr_len",
            "def_fast", "def_slow", "acc_len")   # cooldown 无 _bars 后缀，须显式声明
BOOL_KEYS = ("use_pull", "use_ten", "auto_htf", "use_corr", "use_vol")
```

> `resolve_params` 会把 `SHARED_KEYS`（`rules.py:38-53`）+ `RULE_DEFAULTS`（`:23-35`）+ `DEFAULTS` 三层合并，并校验每个数值 key 在 `SHARED_RANGES ∪ spec.ranges` 内（`:115-133`）。`DEFAULTS` 中和键覆盖 `RULE_DEFAULTS` 同名项；`SHARED_KEYS` 里的 `fee_rate/slippage` 由回测请求注入。**所有 DEFAULTS key 必须出现在 `RANGES` 或 `BOOL_KEYS` 或 `SHARED_RANGES`**，否则 `resolve_params` 抛 `KeyError`。

### 6.4 `lookback` / `validate` / `prepare` 输出列

```python
def lookback(params): return max(600, 4*max(rank_len, brk_slow, def_slow,
                                            corr_len, atr_len, er_len, 20))   # §4.5

def validate(params):
    if params["min_stop_atr"] > params["max_stop_atr"]:
        raise ValueError("min_stop_atr 不能大于 max_stop_atr（stp 语义会退化）")
    if params["brk_fast"] >= params["brk_slow"]:
        raise ValueError("brk_fast 必须小于 brk_slow（快/慢通道失去意义）")
    if params["def_fast"] >= params["def_slow"]:
        raise ValueError("def_fast 必须小于 def_slow")
    if params["dir_mode"] not in (0, 1, 2) or params["htf_in"] not in (0, 1, 2):
        raise ValueError("dir_mode / htf_in 枚举越界")
    # 1d 主周期 -> autoTF "1W" 不支持：由回测脚本/prepare 按 interval 校验（§8 R1）
```

**`prepare` 输出列（全部 `rgd_` 前缀，幂等：列齐则直接返回 frame，照抄 `supertrend_current.prepare:213-214`）**：

| 分组 | 列名 |
|---|---|
| 波动/通道 | `rgd_atr, rgd_vol_rank, rgd_high_vol, rgd_up_fast, rgd_lo_fast, rgd_up_slow, rgd_lo_slow, rgd_ent_up, rgd_ent_lo, rgd_mid, rgd_chan_w` |
| 效率 | `rgd_noise, rgd_er, rgd_er_ok` |
| HTF tension | `rgd_htf_hh, rgd_htf_ll, rgd_htf_ok, rgd_hr, rgd_hm, rgd_np, rgd_tg, rgd_tension, rgd_rg` |
| 相关/量 | `rgd_corr, rgd_rel_vol, rgd_vol_ok, rgd_bull_ok, rgd_bear_ok` |
| 防守 | `rgd_f_h, rgd_f_l, rgd_s_h, rgd_s_l, rgd_pos, rgd_acceptance, rgd_bull_q, rgd_bear_q, rgd_def_long, rgd_def_short` |
| 状态机（纯） | `rgd_ready, rgd_allow_long, rgd_allow_short, rgd_state, rgd_prev_state, rgd_long_break, rgd_short_break, rgd_long_pull_core, rgd_short_pull_core` |
| 入场价位（纯） | `rgd_stp_long, rgd_rd_long, rgd_tp1_long, rgd_stp_short, rgd_rd_short, rgd_tp1_short` |

> HTF 列来自脚本预注入（`attach_htf_levels`）；若 `prepare` 被直接以未注入帧调用（`ctx.service is None` 且无 HTF 列）→ **降级为 `rgd_htf_ok=False` 全列**（`np=na → tg=0 → rg=0 → bullOk/bearOk=False → 无成交`），与 `absorption_reversal` 的 lower-tf 降级路径同纪律（`registry.PrepareContext` 注释）。**但回测链路禁止触发此降级**：`rgd_backtest.py::load_segments` 必须先整帧 attach 再切段并 `assert` HTF 列存在，否则四组静默零成交——见 **§7.3.1 铁律** + QA **A28**。此降级仅为 live `engine.evaluate` 单根直调保留的兜底，不是回测的合法路径。
> `atr_pct = rgd_atr/close` 一并写出（供 `describe`/调试，非 broker 依赖）。

### 6.5 `signals` 输出列（SIGNAL_COLUMNS + 自定义）

```python
def signals(data, params) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)
    # SIGNAL_COLUMNS 11 列 neutral（照抄 simple_zigzag_double.signals）
    result["signed"] = 0; result["score"] = 0.5; result["prob_up"] = 0.5
    result["direction"] = 0
    result["block_reason"] = "stop_order_requires_dedicated_broker"
    for k in ("position_pct","leverage","sl_pct","tp_pct","risk_pct","net_reward_risk"):
        result[k] = 0.0
    # 自定义列：broker 消费子集
    for k in ("rgd_long_break","rgd_short_break","rgd_long_pull_core","rgd_short_pull_core",
              "rgd_allow_long","rgd_allow_short","rgd_state",
              "rgd_def_long","rgd_def_short",
              "rgd_stp_long","rgd_rd_long","rgd_tp1_long",
              "rgd_stp_short","rgd_rd_short","rgd_tp1_short"):
        result[k] = data[k]
    # 诊断/QA/describe 子集
    for k in ("rgd_ready","rgd_atr","rgd_er","rgd_vol_rank","rgd_high_vol",
              "rgd_tension","rgd_rg","rgd_corr","rgd_acceptance","rgd_bull_q","rgd_bear_q",
              "rgd_ent_up","rgd_ent_lo","rgd_mid","rgd_htf_hh","rgd_htf_ll","rgd_htf_ok",
              "rgd_prev_state"):
        result[k] = data[k]
    return result
```

- **`direction=0` 强制专用 broker**（团队裁定，勿推翻）：`runner.run_backtest` 用 `strategy == "rgd_donchian_v2"` 早路由到 `run_rgd_backtest`；通用 `simulate` 不接。`block_reason` 是双保险。
- `describe(row)` 返回末根观测快照（tension/er/volRank/state/def 等，纯展示，对齐 `supertrend_current.describe:306-314`）。

### 6.6 `register` 参数

`prepare_keys` 是回测脚本的**富化帧缓存键**（`supertrend_backtest.py:183`：`cache_key = tuple(params[key] for key in task["prepare_keys"])`）。它必须是**所有会改变 `prepare` 输出的参数的超集**——漏列一个参数，则两组仅该参数不同的 combo 会复用同一份过期帧 → 静默错单（generated-code-failure-modes 最致命模式）。**宁可多列（只少命中缓存，结果仍对），绝不可少列（结果静默错）**。脚本业务参数共 32 个，其中 **3 个是纯 broker 参数、不改 `prepare` 输出**（`cooldown`→`coolOk`、`risk_pct`→定仓 `q`、`tp1_pct`→TP1 部分平比例），故排除；其余 **29 个全部纳入**：

```python
# 29 个 prepare 影响参数（32 业务参数 - 3 纯 broker 参数）
# 排除：cooldown(S6 coolOk) / risk_pct(N2 定仓 q) / tp1_pct(X5 TP1 部分平比例)
# dir_mode 必须在列：G1/G2 仅此不同，却改 rgd_allow_short -> 漏列即 Long Only 复用
#   Long&Short 帧，静默做空（多空/仅多矩阵的核心正确性依赖）
PREPARE_KEYS = (
    "dir_mode", "use_pull", "brk_fast", "brk_slow", "atr_len", "rank_len",
    "rank_cut", "er_len", "er_min", "brk_buf", "use_ten", "htf_in",
    "auto_htf", "build_len", "rel_len", "max_ten", "ten_th", "use_corr",
    "corr_len", "corr_min", "use_vol", "vol_min", "def_fast", "def_slow",
    "acc_len", "max_depth", "max_stop_atr", "min_stop_atr", "tp1_r",
)
SPEC = register(StrategySpec(
    code=CODE, name=NAME,
    description="Regime-Gated Donchian v2：自适应 Donchian 突破 + HTF 流动性张力"
                "regime + 价时相关 + 量能接受度 + ER 门控，R 倍数管理与风险定仓。"
                "15m 主 / 4h HTF，需专用 broker（TP1 限价部分平 + 保本 + 动态 trailing）。",
    defaults=DEFAULTS, ranges=RANGES,
    lookback=lookback, prepare=prepare, signals=signals, describe=describe,
    validate=validate, bool_keys=BOOL_KEYS, int_keys=INT_KEYS,
    grid={},                      # 不支持通用寻优（D1/D4 只做固定参数检验）
    prepare_keys=PREPARE_KEYS,
    default_enabled=False,        # D4：不进实盘绑定、不参与自动巡检资金竞争
    tags=("tradingview", "donchian", "mtf", "dedicated_broker"),
))
```

- 注册：`specs/__init__.py` 加 `from . import rgd_donchian_v2  # noqa: F401` + `__all__` 一行（导入即注册，`registry.register:77-83`）。

---

## 7. 回测脚本方案（`backend/scripts/rgd_backtest.py`）

### 7.1 4 组矩阵（D2 × D3）

| 组 | `dir_mode` | `fee` | `slippage` | 用途 |
|---|---|---|---|---|
| **G1 主报告** | `0` Long & Short | `0.0004` | `0.0002` | 合约多空，项目全局成本（**主报告取此组**） |
| G2 | `1` Long Only | `0.0004` | `0.0002` | 现货仅多对照（项目全局成本） |
| G3 | `0` Long & Short | `0.0005` | `2ticks≈R15` | TV 对齐对照（脚本原样成本） |
| G4 | `1` Long Only | `0.0005` | `2ticks≈R15` | TV 对齐·仅多 |

- `dir_mode=2`（Short Only）**完整实现但不入 4 组矩阵**（D2 只要合约多空 + 现货仅多两组读数；留待手动绑定）。
- 主报告结论以 **G1** 为准；G2 与之并列给「多空 vs 仅多」差；G3/G4 只作 TV 方向性对齐对照，须标注「slippage 近似，非逐位复制」（R15）。

### 7.2 universe / days

- 主周期固定 `--interval 15m`（D1）；`--days` 建议 `180` 与 `298`（尽用 15m 覆盖）双档；`--universe all` + 覆盖度过滤（`symbol_universe(db, interval, days)`）；`--min-bars = lookback+200 = 800`。
- **覆盖度事实**：15m 仅 57 币 ≥180 天、BNBUSDT 仅 10.4 天 → 大量币因段长不足被 skip。报告必须显式列 `universe.skipped` 与 `symbols_with_trades`（继承 `supertrend_backtest.py:505-511`），不得把"零成交"读成"无信号"。
- HTF 帧 `start = base_start - 2*4h_step`、`end = as_of`、同 symbol，从 4h native 取（D1，不落库、不派生 1d）。
- 其它周期 autoTF 见 §4.1；`1d → 1W` 不支持 → 脚本 raise（R1）。

### 7.3 segment 切分 / 复利 / 聚合

- `load_segments`（照抄 `supertrend_backtest.py:125-158`）：`diffs.ne(step).cumsum()` 按缺口切连续段；`len >= min_bars` 才纳入。
- 段间复利 `factor *= equity.iloc[-1]/cash`（`:199-201`）；缺口处仓已平、现金结转。
- `aggregate_positions`（`trades.py:6-30`）把 **TP1 部分平 + L-Trail 剩余平** 合并成一笔（同 `position_id = entry_time`），`return_pct = 总 net / 总 margin`，避免小 fill 被高估。
- `metrics_from_equity`（`metrics.py`）出账户读数；`exposure_metrics`（`execution.py:30-50`）出持仓占比。

### 7.3.1 HTF attach 与 segments / prepare 衔接铁律（**最高危：沉默零成交**）

> 这是 generated-code-failure-modes「沉默逻辑错误」的头号风险：编译通过、冒烟通过、四组全跑完、`pytest` 全绿、JSON 报告照常产出，**但所有币零成交**。三处契约事实叠加会静默触发，必须写死顺序。

**三处事实的致命叠加**：

1. §6.4：`rgd_htf_hh/ll/ok` 由脚本「预注入 `attach_htf_levels`」，`prepare` 本身**不加载 HTF**。
2. §5.1 / §6.4：`prepare` 收到无 HTF 列的帧时**降级为 `rgd_htf_ok=False` 全列** → `np=na → tg=0 → rg=0` → `bullOk/bearOk=False` → `longBrk/shortBrk=False` → **永不入场**，且**不抛任何异常**。
3. §7.3 沿用 `supertrend_backtest.py:164-190` 的 `evaluate_symbol`：`segments` 在 combo 循环**外** `load_segments` 加载一次；`prepared` 缓存键 `cache_key=tuple(params[k] for k in prepare_keys)` 只缓存 `frames=[prepare_frame(segment,...)]`。
4. §6.4 `prepare` **幂等短路**：所需 `rgd_*` 列齐则直接 `return frame`（照抄 `supertrend_current.prepare:213-214`），不会重算 HTF 派生列。

⇒ 只要 HTF attach 漏做、或放在 `prepare` 之后，`prepare` 走降级分支或幂等短路，四组全部零成交且无任何告警。

**铁律（唯一正确顺序，贝洛奇必须照此实现 `rgd_backtest.py::load_segments`）**：

```python
def load_segments(symbol, interval, days, end_ms, min_bars, as_of):
    # ① 对【整个 frame】load 一次并 attach 一次，再切段（本次 4 组 auto_htf=True/htf_in=4h 恒定，安全）
    full = load_frame(symbol, "15m", start, end)                          # 主周期原始 OHLCV
    htf = load_frame(symbol, "4h", start - 2*interval_ms("4h"), as_of)    # HTF native，§7.2
    full = attach_htf_levels(full, htf, "4h", as_of)                      # ★ 整帧一次，产出 rgd_htf_*
    # ② 再按缺口切段（照抄 supertrend_backtest.py:125-158 的 inline groupby）；segments 自带 rgd_htf_*
    diffs = full.index.to_series().diff()
    segments = [seg for _, seg in full.groupby(diffs.ne(interval_ms("15m")).cumsum())
                if len(seg) >= min_bars]
    # ③ 断言：每个 segment 必带 HTF 列且至少一根有效（禁止静默降级）
    for seg in segments:
        assert {"rgd_htf_hh", "rgd_htf_ll", "rgd_htf_ok"} <= set(seg.columns) and seg["rgd_htf_ok"].any(), \
            f"{symbol}: HTF 列缺失或全 False，拒绝静默零成交"
    return segments

# combo 循环内只做 prepare/signals/simulate（照抄 supertrend_backtest.py:178-203）
for combo in combos:                                   # segments 在循环外只 load 一次
    params = {**base, **combo}
    cache_key = tuple(params[k] for k in PREPARE_KEYS)
    if cache_key not in prepared:
        prepared[cache_key] = [prepare_frame(seg, CODE, params) for seg in segments]  # 消费已注入列
    ...
```

- **attach 必须在 `load_segments` 内、对整个 frame 做一次，再切段**：不能在切段后逐段 attach（会重复 join 且易漏段），更**绝不能放到 `prepare` 之后**（`prepare` 幂等短路列齐直接 return，attach 白做 → 静默零成交）。
- **`htf_in`/`auto_htf` 虽在 `PREPARE_KEYS`（§6.6），但改它们必须重建 segments**：`segments` 在 combo 循环外固定，`prepare_keys` 的 `cache_key` 只能区分 prepare 的入参，**无法重新加载不同周期的 HTF 帧**。本次 4 组矩阵 `auto_htf=True`/`htf_in=4h` 恒定、不触发；一旦将来放开 HTF 周期做网格，**必须把 `htf_in`/`auto_htf` 提到 combo 外层，按每个 HTF 周期各 `load_segments` 一次**，否则换周期复用旧 HTF 列 = 静默错周期。
- **禁止静默降级**：`load_segments` 内显式 `assert` 每个 segment 含 `rgd_htf_hh/ll/ok` 三列且 `rgd_htf_ok.any()`（或等价警告写入 `warnings`）；缺列直接 `raise` 带 symbol 的错误，**不允许把「零成交」当成「无信号」交付**（呼应 §7.2 覆盖度告警）。QA A28 守住。

### 7.4 随机方向基线（**关键陷阱，必读**）

- **禁止直接复用 `execution.randomize_directions`**：它只翻 `direction` 列（`execution.py:15-27`），而 RGD 的 `direction ≡ 0` ⇒ **no-op 静默失效**（generated-code-failure-modes「沉默逻辑错误」高发区：测试照跑、基线恒等于策略、结论全错）。
- 必须实现 `randomize_rgd(signals, seed, mode)`：
  - **`shift`（主用）**：段内对所有 `rgd_*` 决策列做**循环位移**（circular shift）随机 offset → 保留信号数与结构、去相关时序。
  - **`flip`（可选）**：交换 `rgd_long_break↔rgd_short_break`、`rgd_long_pull_core↔rgd_short_pull_core`，并镜像价位 `stp' = 2*close − stp`、`tp1' = 2*close − tp1`、交换 `def_long↔def_short`（`rd` 不变）。
- 主用 `shift`（时序证伪），`n_seeds=10`（`base_seed=20261025`，种子步长 7919，继承 `:459`）；`beats_random_ratio` = 账户收益 > 该币随机分布 p95 的币占比（`:337-348`）。

### 7.5 统计显著性

- `bootstrap_expectancy_ci(trades, min_trades=30)`（`significance.py:28-77`），池化 + 多空分解（`:274-292`）；`MIN_POOLED_TRADES=30`（4 笔会出假精确 `[0.86%,10.48%]`，见 `significance.py:43-45`）。
- 判读：CI 下界 > 0 才算正 edge；跨 0 不可判定；上界 < 0 负贡献。样本不足默认「不启用」。

### 7.6 输出与成本（cost-models/development-costs）

- `docs/rgd-backtest.md` + `docs/rgd-backtest.json`（主 G1），4 组各存 `docs/rgd-backtest-G{1..4}.json`（或单 JSON 内 `combos` 4 组，继承 `:497-526`）。
- **隐性成本**：无新增云服务/API/DB；进程池 CPU 时间（521 币 × 4 组，15m×298 天 ≈ 28k bar/币，纯 Python 逐 bar × 4，中等耗时）→ `--processes` 默认 `CPU-2`、`--limit 8` 冒烟、`--seeds 5` 冒烟；数据缺口/样本不足/滑点近似须在 `warnings` 显式列出（继承 `:65-69/338-365`）。
- **并行铁律**（踩过的坑，照抄 `:35-42/68-79/111-119/484-487`）：BLAS 线程数在 numpy 前设 1；worker 只读 + NullPool；`as_completed` 结果按 symbol 排序后再池化，保证固定种子 bootstrap 逐位可复现。

---

## 8. 歧义裁决清单 + OPEN-DECISIONS 登记

### 8.1 裁决表（每条：裁决 + 理由 + 关联）

| ID | 争议点 | 裁决 | 理由 | 严格/偏差 |
|---|---|---|---|---|
| R1 | `autoTF()` 对 1d 返回 `"1W"`，项目无周线 | **不支持**：`validate`/脚本对 base=1d 或 `htf_in` 解析到不支持周期时 raise | 白名单 `config.py:81` 仅 6 个周期；D1 禁派生 1d；15m→4h 已覆盖本次需求 | 严格（范围界定） |
| R2 | `dirMode`/`htfIn` 是 string，项目 `resolve_params` 拒收 | int 枚举编码 `0/1/2`（I1/I2） | `rules.py:118-130` 只接受数值/bool；枚举与字符串一一对应，语义等价 | 严格（编码适配） |
| R3 | `ta.percentrank` 定义（是否含当前 bar / `<` vs `<=` / 分母） | **比较集 = 前 `length` 根、不含当前；`<=`；分母 = `length`** | Pineify v6 手工例 `length=5, prev=[98,100,102,104,108], cur=97 → 0%` 证伪「含当前」；`cur=102 → 60%` 证实 `<=`；`100*4/5=80` 证实分母 | 严格（已核验） |
| R4 | `ta.atr` 用 Wilder RMA 还是 pandas ewm | **Wilder RMA**（`supertrend_current._rma:82-101`），不用 `indicators._atr` 的 ewm | Pine `ta.atr=ta.rma(ta.tr,len)` 以 SMA 初始化；项目 `_rma` 已对齐；ewm seed 不同 | 严格（选型已定） |
| R5 | `syminfo.mintick` 本地值 | **`1e-8` 固定极小正数**（仅除零/退化保护） | 三处使用（`chanW/noise/np`）都是 guard，真实值远大于 1e-8 时结果不变；项目无每币 tick 数据 | 偏差（登记 OPEN-01） |
| R6 | `ta.ema/ta.correlation/math.sum` 预热期 na 语义 | 用 `min_periods` 对齐；na→保守 False | EMA(3) ~10 bar 内收敛，`ready` 由 `rank_len=100` 主导 ⇒ 预热差异**决策无影响** | 偏差（登记 OPEN-02） |
| R7 | `bar_index` 绝对 vs 局部 | 局部 `np.arange(n)` | Pearson 对仿射变换不变，段内连续 ⇒ 逐位等价 | 严格（已证） |
| R8 | `defL/defS` 混合公式 | **完整加权式**（A5）：`defL=(1−d·bullQ)·max(fL,sL)+d·bullQ·min(fL,sL)` | 团队提示"bullQ=1→defL=min"是 `maxDepth→1` 简写，实现不可省权重 | 严格 |
| R9 | `state` 放 prepare 还是 broker | **放 prepare**（纯状态机） | 转移只依赖 `prevState+longBrk/shortBrk+close/defL/defS`，无成交态 ⇒ 满足整帧/单 bar 等价 | 严格 |
| R10 | `strategy.closedtrades` 是否对部分平仓 bump | broker 镜像 `closed_count`（**全平**计数） | 已证：`coolOk` 仅门控 pullback，`flat` 必在全平后，最后 bump 恒为全平那根 ⇒ 与部分平是否 bump 无关 | 严格（等价已证，OPEN-03 备核验） |
| R11 | `strategy.equity` 口径 | `cash + 毛浮动盯市`；affordability 夹逼 | Pine `equity = initial + netprofit + openprofit`；夹逼偏差 ≤ 滑点+费量级 | 严格（OPEN-04 备核验） |
| R12 | Pine 时序（`strategy.exit` 重挂生效 / `tp1Hit` 保本滞后 / 入场 bar 裸奔） | **bar N 收盘算 + 重挂，成交最早 N+1；tp1Hit 在 N 置位、保本生效 N+1；信号 bar 成交当根无 exit** | `calc_on_every_tick=false` + `process_orders_on_close=false` 默认；源码顺序 L157 管理→L176 入场→L201 flip | 严格（团队点名） |
| R13 | `strategy.exit("L-TP1")` 成交后是否 re-arm | **TP1 每次持仓仅触发一次**（`tp1Hit` latch），移除 `exit_tp1`，余仓仅 `exit_trail` | 符合脚本 `tp1Hit` 一次性 latch 语义；避免重复部分平导致仓位被无限切分 | 偏差（保守，OPEN-05 备 TV replay） |
| R14 | **盘中**同根 TP1 与 stop 都触及（`open` 未穿 `tp1Lvl`）的优先级 | **stop 优先**（全平，不给 TP1 部分成交）。**边界**：仅适用盘中路径；`open` 跳空穿 `tp1Lvl` 的既成部分平 `tp1_pct%`@open 不受本裁决约束，其后同根 `low<=stopLvl` 则剩余 `1-tp1_pct%` 按 stop 全平（两段成交，A30） | 参照 `risk.py:8-20 stop_exit` 先 stop 后 target；保守、避免高估。开盘穿 limit 是 Pine broker 的既成成交（更优价），stop-first 无法也不应覆盖 | 偏差（团队点名保守，OPEN-06 备 TV parity，边界 A30） |
| R15 | `slippage=2 ticks` 本地换算 | **每币标量** `2 * tick_size(median_close) / median_close`，`tick_size` 用项目 `utils.round_price` decimals 约定作代理（`>=1000→1e-2`、`>=1→1e-4`、`else→1e-6`） | 无每币 mintick 数据；代理使 BTC(60000)→`≈3.3e-7`、低价币(0.5)→`4e-4`，方向性合理；保持 broker 标量签名 | 偏差（近似，OPEN-07 待真实 tickSize） |
| R16 | 跨缺口 `bar_index` 重置 | 段内重置不影响 | gap 切段后每段连续，`bar_index` 局部化仍仿射等价（R7） | 严格 |

### 8.2 登记到 `docs/decisions/OPEN-DECISIONS.md` 的悬而未决项（append-only）

> 依据 `open-decisions-register.md`：三类固定 slug、七字段、`Resolves when` 必须可判定、只追加 + 就地关闭。现有文件是 8 列表格且无 category 列 → **保持既有表结构**，把 category slug 写进 `Source` 字段（形如 `design-decision-to-evaluate · RGD §8 R5`），不重写既有行。本次追加 **10 条**（slug + 条数见下）。

| # | slug | 标题 | Current Leaning | Blocked By | Resolves When |
|---|---|---|---|---|---|
| OPEN-01 | `design-decision-to-evaluate` | `syminfo.mintick` 本地值 = 1e-8 guard | 保持 1e-8（仅退化保护） | 项目无每币 tick 数据 | 接入交易所 `priceFilter.tickSize` 后按真实值复算 `chanW/noise/np`，读数无实质变化即可关闭 |
| OPEN-02 | `design-decision-to-evaluate` | EMA/corr/sum 预热 na 语义 | `min_periods` + na→False | 需 TV 逐根对照预热段 | 用 TV 同行情导出前 200 根 `er/corr/atr` 逐位比对；若首笔信号 bar 相同即关闭 |
| OPEN-03 | `design-decision-to-evaluate` | `closedtrades` 是否对 TP1 部分平 bump | broker 全平计数（已证等价） | 需 TV 确认 `closedtrades` 精确定义 | TV 上构造「TP1 部分平后 pullback」场景，确认 `lastExit` 取值与全平那根一致 |
| OPEN-04 | `design-decision-to-evaluate` | `strategy.equity` 持仓期是否已扣 entry fee | `cash + 毛浮动` + affordability 夹逼 | 需 TV 导出 equity 序列 | TV 导出 equity 与本地 `mark(close)` 逐根比对，差异 ≤ 费量级即关闭 |
| OPEN-05 | `design-decision-to-evaluate` | `strategy.exit("L-TP1")` 成交后是否 re-arm | TP1 一次性（不 re-arm） | 需 TV replay 同 id 重挂行为 | TV 构造「TP1 成交后再触 tp1Lvl」场景，确认是否二次部分平；若否则关闭并保留 latch 实现 |
| OPEN-06 | `design-decision-to-evaluate` | **盘中**同根 TP1 与 stop 优先级（`open` 跳空穿 `tp1Lvl` 的既成部分成交除外，见 R14/A30） | 盘中 stop 优先（保守）；开盘穿 limit 既成部分成交不受约束 | 需 TV OHLC 路径逐根对照 | TV 构造盘中同根双触 bar，确认是 stop 全平还是 TP1 先部分平；另构造开盘穿 TP1 bar 确认既成部分成交；与本地一致或登记已知偏差 |
| OPEN-07 | `waiting-on-external-condition` | `slippage=2 ticks` 真实 tick 换算 | `round_price` decimals 代理（每币标量） | 缺每币 `tickSize`（本次不落库、不拉 exchangeInfo） | 接入 `priceFilter.tickSize` 元数据后重跑 G3/G4，与主报告差异可解释即关闭 |
| OPEN-08 | `existing-design-boundary` | 15m 数据覆盖薄（57 币≥180d / BNB 10.4d） | 报告显式列 skipped / symbols_with_trades；样本不足默认不启用 | 15m 历史深度不足 | 补齐全 universe ≥365 天 15m 后重跑，池化笔数达门槛且 CI 可判定 |
| OPEN-09 | `existing-design-boundary` | 1d 主周期 → `autoTF "1W"` 不支持 / 不派生 1d | raise 明确报错 | 项目周期白名单无周线 | 若将来引入周线或用户改 `auto_htf=False + htf_in∈{1h,4h,1d}` 可支持 1d 主周期 |
| OPEN-10 | `existing-design-boundary` | 实盘持仓管理（TP1 限价挂单 + 动态 trailing 状态机）Out-of-Scope | `default_enabled=False`，不进 live 绑定 | D4 用户锁定交付边界 | 用户明确要求实盘化并立项后，按 §5 broker 状态机迁移到 `order_manager` |

---

## 9. 因果性红线与已知坑（内嵌进 spec-as-contract §2「内嵌已知坑」）

1. **截断重算逐位一致**：`prepare` 对 `frame.iloc[:k]` 与整帧在第 k 根的**每一列**（含 `rgd_tension`/`rgd_state` 状态机）必须逐位 `==`（参照 `test_absorption_reversal.py:398-409`）。`tension`/`state` 用显式 `for i in range(n)`，只读 `<=i` 的值 ⇒ 天然满足。**禁止用 `.shift(-1)` 或任何前向引用**。
2. **HTF 不重绘**：同一 4h 桶内 16 根 15m bar 的 `hh/ll` 必须**完全相同**；`np.searchsorted(htf_stamps, base_stamps - step, side="right") - 1` 不得改成 `side="left"`、不得并入未完成桶、不得 ffill 到未来桶。
3. **dtype=object 坑（项目记忆）**：`groupby().cummax() + shift()` 会产出 `dtype=object`，`~series` 位运算**恒真** → 所有 bool 门控列在 `&`/`~` 前必须 `.astype(bool)`；`np.where` 返回 object 时 `.astype(float/bool)`。信号帧 `block()` 已用 `.astype(bool)`（照抄 `signals.py:48`）。
4. **shift 方向坑**：`shift(-1).rolling(H)` 是过去窗口；本策略全用 `high[1]/low[1]`（`shift(+1)`，过去）安全。**不得**为求「未来窗口」写反向 rolling（本策略不需要）。
5. **周期桶对齐**：用 `x - x % step`（`data_service.py:463`），勿用 cast 除法（PG NUMERIC 四舍五入会整体错位）。
6. **PG 方言**：`_insert_fn` 按 bind dialect 分流（`data_service.py:62-69`）；本任务**只读**，无写入，不受影响。
7. **`SIGNAL_COLUMNS` 11 列齐全**：`rules.evaluate_signals:171-173` 缺列即 raise；`signals()` 必须补齐（`signals.py:21-33`）。
8. **`fee_rate/slippage` 硬禁止策略级覆盖**（`rules.py:105-109`）：绝不进 `DEFAULTS`/`overrides`，只走 state / 回测请求 `--fee/--slippage`。
9. **并行坑**（`supertrend_backtest.py:35-42/68-79/111-119/484-487`）：BLAS 线程数必须在 numpy import 前设 1；worker `read_only=True, null_pool=True`；`as_completed` 结果**按 symbol 排序**后再池化，否则固定种子 bootstrap 每跑漂移 ±0.02pp。
10. **未收盘桶**：base 帧 `frame.loc[(index + base_step) <= as_of]`；HTF 帧 `closed` 掩码；末根半截 OHLCV 会造未来函数。
11. **入场 bar 裸奔**：不得"顺手"在成交瞬间挂止损（N8/R12），这是 1:1 的一部分。
12. **交叉验证**：`simulate_rgd` 的 OHLC 路径顺序必须与 `zigzag_execution.py:139` / `ocb_execution.py:140` 完全一致（近极值先走，平票先低后高）。**以代码行为为准**：`ocb_execution.py:139` 注释文字写「先高后低」，与其 `:140` 代码（平票 `else→(low,high)`）矛盾，勿被注释误导（A29 断言守）。

---

## 10. 下游验收点清单（QA 严过关·可执行断言）

> 依据 `test-discipline` + `generated-code-failure-modes`：火力压到**未被覆盖的关键行为**（沉默逻辑错误），核心逻辑做**变异加固**。测试范式：`tests/test_zigzag_execution.py::replay`（手搓 OHLCV）+ `tests/test_absorption_reversal.py`（随机游走帧 + 截断重算 + 参数校验）。

| ID | 断言 | 覆盖失效模式 |
|---|---|---|
| A1 | `auto_tf()` 对 6 个周期返回 §4.1 表值；15m→"240"→"4h"；1d→"1W"→raise | 幻觉签名/映射错 |
| A2 | `tension_step` 状态机：首根 `prev=0`；`_sign(0)=0`；`sameDir` 三分支；`building` 用 `2/(b+1)`、`release` 用 `2/(r+1)`；clamp ±cap；`nz(tension[1],0)` 对 na 自愈（构造 tg 单根 na 后下一根恢复） | 沉默逻辑错误 |
| A3 | `percentrank_preceding` 4 个手工值 `[105→80, 97→0, 102→60, 108→100]`；窗口含 na→na；首有效位=`length`；**当前 bar 不入比较集**（用「全窗口相等」证伪含当前：应 100 还是 0 的判别） | 定义错（最高危） |
| A4 | `wilder_atr` 与 `supertrend_current._rma` 逐根一致；与 `indicators._atr`(ewm) 在预热段**不同**（防止误用） | 选型回退 |
| A5 | `noise/er/erOk` 手算对照；`close[er_len]` 位移正确；`er_min=0.30` 门控 | off-by-one |
| A6 | `defL`：bullQ=0→`max(fL,sL)`；bullQ=1→`(1−0.7)*max+0.7*min`；`defS` 对称；`pos/acceptance/bullQ/bearQ` 手算 | 公式简化错 |
| A7 | `state` 转移优先级（longBrk>shortBrk>多头破 defL>空头破 defS）；`prevState` 门控 `!=1/!=-1` 与 `==1/==-1`；4 条件互斥；截断重算逐位一致 | 状态机漂移 |
| A8 | `ready` 精确 6 项 `not na(...)`，任一 na 则该根不入场 | 过早入场 |
| A9 | HTF join：同一 4h 桶内 16 根 hh/ll 唯一值数=1；未完成桶剔除；首根 `pos<0→hOk=false`；截断重算逐位一致 | 未来函数/重绘 |
| A10 | `np` 用 base `close`（改 base close 会改 np 但**不改** hh/ll）；`hOk=false→np=na,tg=0` | 混合口径错 |
| A11 | `corr` 仿射不变：`bar_index` 从 0 与从 12345 起结果逐位相同 | bar_index 语义 |
| A12 | TP1 部分平仓：恰好平当前仓 50%，剩余 50% 继续；`aggregate_positions` 合并两腿为一笔 | TP1 语义（X5） |
| A13 | 保本滞后：构造 N 触 `tp1Lvl`、N+1 跌破 `entryRef` 但 > 旧 `stopLvl` → 断言 N 根不被新保本 stop 平、N+1 根才生效（R12） | 时序（团队点名） |
| A14 | trailing 单调：多头 `stopLvl` 序列非降、空头非升；保本 `max(stopLvl,entryRef)` 生效后不回落 | ratchet 回退 |
| A15 | **盘中**（`open` 未穿 `tp1Lvl`）同根 stop+TP1 双触 → **stop 全平**，TP1 无部分成交（R14）；开盘穿 `tp1Lvl` 的两段成交另见 A30 | 优先级（R14 盘中边界） |
| A16 | 跳空：`open` 穿越 stop→`open` 成交；穿越 limit→`open` 成交 | 跳空 |
| A17 | 入场 bar 裸奔：信号 bar 成交后**当根无 exit**（构造该根直接反向穿越 tp1Lvl/stopLvl，断言不成交） | 时序（团队点名） |
| A18 | Trend flip：仅 Long Only / Short Only 触发；双向模式同场景走**反手**（平旧+开新） | flip vs 反手 |
| A19 | `pyramiding=0`：同向 `entry` 忽略；反向 `entry` 反手（数量=旧+新，两腿费用分别计） | 反转 |
| A20 | 因果截断：`prepare(frame.iloc[:k])` 与整帧在第 k 根全部 `rgd_*` 列逐位一致 | 未来函数 |
| A21 | 参数校验：`fee_rate/slippage` 作为 override 抛错；`int_keys` 非整抛错；`bool_keys` 收数字抛错；`validate` 交叉约束抛错；`resolve_params` 默认值=脚本 1:1 | 参数层 |
| A22 | 两组成本：G1(`0.0004/0.0002`) 与 G3(`0.0005/≈2ticks`) 各跑一遍且读数不同；一个来回=2 fee+2 滑点 | 成本口径 |
| A23 | `signals()` `direction≡0` 且 `block_reason="stop_order_requires_dedicated_broker"`；误接通用 `simulate` 时零成交（安全失败） | 误路由 |
| A24 | 随机基线：断言 `randomize_directions` 对 RGD 是 no-op（回归护栏）；`randomize_rgd(shift)` 确实改变成交流 | 沉默逻辑错误（基线失效） |
| A25 | `register`：`default_enabled=False`、`tags` 含 `dedicated_broker`、`seed_strategies` 后 `enabled=0`；`specs/__init__` 导入即注册 | 交付边界 D4 |
| A26 | 收尾：数据末根未平仓写入 `equity.attrs["open_position"]`；**无** `close_at_end` 强制平仓 | 收尾语义 |
| A27 | 代码组织门禁：新增源文件 `wc -l` 最大 ≤300（超限已拆分） | 巨型文件 |
| A28 | **HTF attach 衔接（§7.3.1）**：① 未 attach 的裸 OHLCV 帧喂完整链路 → 断言脚本侧 `assert` 缺 `rgd_htf_*` 时抛错（或 `warnings` 显式可观测），**不得静默零成交**；② 正确顺序 `load_frame→attach(整帧一次)→切段→prepare` 后，构造已知会入场的行情 → 断言成交流非空；③ 断言 attach 在切段**之前**、且 `prepare` 之后无第二次 attach（幂等短路，attach 后置即失效） | 沉默零成交（最高危） |
| A29 | **平票 bar 撮合顺序（瑕疵 2）**：构造 `abs(open-high)==abs(open-low)` 且 stop/TP1 分别在 high/low 反向的 bar → 断言路径 `[o,l,h,c]`（先低后高）；对照 `zigzag_execution.py:139`/`ocb_execution.py:140` 代码，勿信 ocb `:139` 注释「先高后低」 | OHLC 路径/注释误导 |
| A30 | **开盘跳空穿 TP1 + 同根触 stop（R14 边界）**：构造 bar `open>=tp1Lvl`（多）且 `low<=stopLvl` → 断言**两段成交**：先 `tp1_fill(open)` 部分平 `tp1_pct%`，再剩余 `1-tp1_pct%` 按 `stopLvl` 全平；两段合计=原仓、`tp1_hit` latch 生效、`exit_tp1` 已移除。对照 A15（盘中双触→stop 全平、不给 TP1）确认两者不同分支 | R14 边界（开盘 vs 盘中） |

---

## 11. 实现清单（新增/修改文件逐一列出，标明改动范围）

> 依据 `code-organization.md`：单一职责、单文件 ≤300 行、按资源分包、入口只装配。贝洛奇（实现）按此清单落地；严过关（QA）按 §10 验收。

### 11.1 新增

| # | 文件 | 职责 | 改动范围 |
|---|---|---|---|
| 1 | `backend/app/features/rgd.py` | 纯因果特征：`wilder_atr` / `percentrank_preceding` / `rolling_pearson` / `tension_step`+`tension_series` / 通道 / ER / 防守 / `state_machine` / 核心入场条件 / 入场价位 | 新增，≤300 行；若超阈值拆 `rgd_indicators.py`（指标原语） |
| 2 | `backend/app/features/rgd_htf.py` | `attach_htf_levels`（§4.2 HTF 因果 join） | 新增，~60 行 |
| 3 | `backend/app/strategy/specs/rgd_donchian_v2.py` | spec：`DEFAULTS/RANGES/INT_KEYS/BOOL_KEYS/validate/lookback/prepare/signals/describe/register`（§6） | 新增，≤300 行 |
| 4 | `backend/app/backtest/rgd_execution.py` | `simulate_rgd` 专用 broker（§5） | 新增，≤300 行；若超阈值拆 `rgd_orders.py`（`close_at/partial_close/fill/process_segment` 原语） |
| 5 | `backend/scripts/rgd_backtest.py` | 4 组矩阵主脚本（继承 `supertrend_backtest.py`） | 新增；建议拆 `rgd_backtest_report.py`（markdown 渲染）+ `rgd_baselines.py`（`randomize_rgd`/池化）以守 300 行门 |
| 6 | `backend/tests/test_rgd_execution.py` | broker replay：A12–A19/A26（手搓 OHLCV + 信号列，对齐 `test_zigzag_execution.py`） | 新增 |
| 7 | `backend/tests/test_rgd_features.py` | 特征/状态机：A1–A11/A20–A21/A23/A25（对齐 `test_absorption_reversal.py`） | 新增 |
| 8 | `docs/rgd-backtest.md` + `docs/rgd-backtest.json`（+ `-G1..G4.json`） | 回测报告（脚本产出） | 由 #5 生成 |

### 11.2 修改

| # | 文件 | 改动范围 |
|---|---|---|
| 9 | `backend/app/strategy/specs/__init__.py` | 加 1 行 `from . import rgd_donchian_v2  # noqa: F401` + `__all__` 追加（导入即注册） |
| 10 | `backend/app/backtest/runner.py` | 加常量 `RGD = "rgd_donchian_v2"`；新增 `run_rgd_backtest(...)`（单币单窗单成本组、**不做 train/test split**，对齐 `run_zigzag_backtest:38-81`）；`run_backtest` 在 `is_zigzag`/OCB 分支附近加 `if strategy == RGD: return run_rgd_backtest(...)`（约 `:257-291`） |
| 11 | `docs/decisions/OPEN-DECISIONS.md` | **append 10 行**（§8.2），不重写既有行 |
| 12 | `docs/rgd-porting-contract.md` | 本契约（已产出） |

### 11.3 显式不改动（防止范围蔓延）

`execution.py` / `risk.py` / `data_service.py` / `config.py` / `registry.py` / `rules.py` / `signals.py` / `trading/*`（`order_manager`/`live`/`paper`）/ 任何 DB 模型 / 任何 API 路由 / 任何前端。理由：D4 交付边界 + 专用 broker 已自洽，通用管线一行不动。

### 11.4 端到端验证命令（spec-as-contract §4「收尾即验收」）

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

**完成定义**：A1–A30 全绿 + 4 组冒烟均产出 `docs/rgd-backtest-G*.json` + 全量 `pytest tests -q` 回归率为 0 + 单文件门禁 `wc -l` ≤300。任一不满足即退回，不算 Phase 2 通过。

---

## 附：关键事实的代码锚点（供实现/QA 即时检索，不做全文常驻）

| 事实 | 锚点 |
|---|---|
| `StrategySpec` 字段 / `PrepareContext` / `register` | `backend/app/strategy/registry.py:27-36 / 46-71 / 77-83` |
| 三层参数合并 / `fee_rate·slippage` 硬禁止 / int·bool 校验 | `backend/app/strategy/rules.py:91-141`（禁止在 `:105-109`） |
| `SIGNAL_COLUMNS` 11 列 / `block().astype(bool)` / `assemble_signal_frame` | `backend/app/strategy/signals.py:21-33 / 46-49 / 52-132` |
| Wilder RMA 模板 / 纯状态机模板 / `prepare` 幂等 | `backend/app/strategy/specs/supertrend_current.py:82-101 / 104-206 / 213-214` |
| `direction=0` 强制专用 broker 模板 | `backend/app/strategy/specs/simple_zigzag_double.py:119-132` |
| `searchsorted` 因果对齐原语 | `backend/app/features/absorption.py:145` |
| OHLC 路径顺序 / pending / `process_at_open` / `open_position` | `backend/app/backtest/zigzag_execution.py:139 / 31-38 / 94-100 / 173-179` |
| stop 优先于 target / 跳空按 open | `backend/app/trading/risk.py:8-20` |
| 成本口径（开+平，2 fee+2 滑点） | `backend/app/backtest/execution.py:70-73 / 95-126` |
| `randomize_directions` 只翻 direction（RGD 会 no-op） | `backend/app/backtest/execution.py:15-27` |
| 段切分 / 复利 / `aggregate_positions` / bootstrap / 并行铁律 | `backend/scripts/supertrend_backtest.py:125-158 / 199-201 / :90 / :274-292 / :35-42` |
| `load_frame`（1m 聚合 + native 补齐） | `backend/app/market/data_service.py:593-622`（native `:530-553`、merged `:555-591`） |
| 周期白名单 / 全局成本默认 / `INTERVAL_MS` / `round_price` | `backend/app/config.py:81 / 84-85 / backend/app/utils.py:3-16 / 37-44` |
| 现有 OPEN-DECISIONS 表结构 | `docs/decisions/OPEN-DECISIONS.md`（8 列，2 行既有） |

---

*本契约为可寻址产物（living-spec）：实现中若发现规格与代码现实冲突，先改本契约再改代码；§8 裁决被 TV 证伪时就地翻 `RESOLVED` 并补 `Resolution`（`open-decisions-register.md` §4）。*

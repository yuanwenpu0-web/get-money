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
| I3 | L18-52 其余 22 个 input | `spec.DEFAULTS` / `spec.RANGES`（§6.2/§6.3） | 全部 1:1 默认值；`RANGES` 取脚本 `minval/maxval`，无 max 者给保守上限。`riskPct`(L48) 命名 `risk_pct`，**与项目 `risk_per_trade_pct` 区分**（后者 inert）。`cooldown`(L19) 命名 `cooldown`，**与项目 `cooldown_bars` 区分**（后者在 DEFAULTS 置 0 中和）。 | 严格 |
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
| S1 | L124-130 `var state=0; stopLvl/tp1Lvl/entryRef=na; tp1Hit=false; lastCT=0; lastExit=-9999` | **broker** 状态字典（§5.2） | 这些是**成交依赖**状态，不能放 `prepare`/`signals`（单 bar 调用不等价，见 §5.1 架构裁决）。`signals()` 只产出「core 条件 + 纯价位」。 | 严格（架构裁决 §5.1） |
| S2 | L132-134 `if strategy.closedtrades>lastCT: lastCT:=...; lastExit:=bar_index` | `broker`: 本 bar 有**平仓成交**（含 TP1 部分平/止损/trailing/翻转/反手）则 `lastCT+=1; lastExit=i` | 等价于「bar i 收盘时 closedtrades 增量」。裁决：因 `coolOk` 只门控 pullback，而 pullback 要求 `flat`，故 `lastExit` 的有效值恒等于「仓位归零那根」，部分平仓是否 bump 不影响任何决策（§8 R10）。 | 严格（等价已证） |
| S3 | L136 `ready = not na(entUp/entLo/defL/defS/er/atr)` | `fN`: `ready` bool 列 | **6 项**，不含 `corr/relVol/tension`（它们已 `nz` 或由 `volOk/bullOk/bearOk` 内部处理）。 | 严格 |
| S4 | L137 `flat = strategy.position_size==0` | `broker`: `position is None` | 仅 pullback 用。 | 严格 |
| S5 | L138-139 `allowL=dirMode!="Short Only"`；`allowS=dirMode!="Long Only"` | `fN`: `allow_long=(dir_mode!=2)`、`allow_short=(dir_mode!=1)` | 见 I1。 | 严格 |
| S6
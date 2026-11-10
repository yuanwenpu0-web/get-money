# EMA 快慢线方向过滤 · 全策略通用入场门控 回测／寻优报告

> 生成时间：2026-11-07 04:20 ｜ 脚本：`backend/scripts/ema_filter_optimize.py`、`backend/scripts/ema_filter_ab.py` ｜ 复现命令见文末
> 版本：首版。按用户明确要求「给策略列表所有策略加 EMA 方向过滤并选出最佳参数组合」执行，走完整 Gate 0–6 做 edge 判定。

## 假设与边界

- **信息源 / 可证伪假设**：EMA(fast) 与 EMA(slow) 的相对位置是**收盘价序列的衍生**。假设 H：把它作为通用入场门控（多头要求 fast>slow、空头要求 fast<slow），在**不改变策略自身方向判断**的前提下，能提高净收益每笔期望，使池化 bootstrap 95% CI 下界为正。事前终止条件：若全网格寻优 0 `pass` 且受控 A/B 配对差 CI 跨 0、FDR q≥0.10，则判「不存在统计上可支持的最佳参数组合」，不启用。
- **周期 / 持有期 H**：1h。本轮不新增特征、不新增标签，持有期沿用各策略 spec 自身定义（supertrend 持有型、flow_imbalance 固定持有），故 H 与既有回测口径一致。
- **窗口起止 / 墙钟对齐**：4 个**不重叠** 90 天窗口，每窗 2159 根 1h（UTC）——
  W1 `2026-08-08 ~ 2026-11-06`、W2 `2026-05-10 ~ 2026-08-08`、W3 `2026-02-09 ~ 2026-05-10`、W4 `2025-11-11 ~ 2026-02-09`。
  寻优窗口＝最近 180 天（≈W1+W2）；**W3+W4 是未被选参读取过的样本外**。
  样本期基准（BTCUSDT）：W1 **+23.72%**、W2 **−19.53%**、W3 **+14.72%**、W4 **−33.67%** —— 牛熊交替，不是单边行情，读数不能归因于 beta 方向。
- **币池 / 币数**：`major` 池 **16 币**（BTC/ETH/SOL/BNB/XRP/DOGE/ADA/LINK/AVAX/LTC/DOT/SUI/TRX/NEAR/APT/BCH，均为 USDT）。`universe.skipped = []`：BNBUSDT 的 1h native 只有 210 天，但 1m 有 1096 天，`load_frame` 从 1m 聚合派生 1h（derived 回测口径），4 个窗口均有完整 2159 根，无需剔除；W1 末段因 1m 数据止于 2026-11-02 少约 2 天。
- **策略适用性（9 个 spec 全覆盖）**：EMA 门控做在通用层 `rules.evaluate_signals`，9 个策略全部接入；但只有 2 个在 1h 上产生可判定样本：
  | 类别 | 策略 | 说明 |
  | --- | --- | --- |
  | 可判定 | `supertrend_current`、`flow_imbalance_long` | 进入寻优与 A/B |
  | 1h 上 `no_trades` | `absorption_reversal`、`capital41_fusion_scalp` | 信号太稀疏，不是参数不好 |
  | 专用 broker / `direction≡0` | `opening_candle_breakout`、`rgd_donchian_v2`、`simple_zigzag_double` | 通用门控对专用撮合路径无样本 |
  | 研究池 / `direction≡0` | `research_high_turnover_high_vol`、`research_low_turnover_high_vol` | 同上 |
- **止损线核对**：已查 `references/verdicts-history.md` §1 ——「现货 OHLCV 价格序列 + K 线级主动成交汇总均已证伪，不再投入新特征、新参数、新标签」。本课题**命中**该止损线。因用户明确要求，本轮仍按手册走完整 Gate 做判定，结论作为止损线的又一次可复现实测印证，而非新特征投入。
- **对照组**：关闭过滤基线 `(0,0)` / 全网格选出的开启组合 / 20 种子方向翻转随机基线 / 同币配对差（受控 A/B 只改 EMA 两键）。

## 口径与参数

- **成本**：`fee_rate=0.0004`、`slippage=0.0002`，开平双边合计 **0.12%**；口径＝**名义本金**，全程不切换；从 `app_state` 读取，臂级禁止覆盖（H3）。本报告数值不得与 margin 口径报告直接相减。
- **参数来源**：策略默认参数（`resolve_params` 三层合并），只搜 EMA 两键：`fast ∈ [9,12,20,30,50]`、`slow ∈ [26,50,100,200]`。5×4=20 档，剔除 `fast>=slow` 的 3 档，再由 `build_grid` 补入关闭基线 `(0,0)`，共 **17 个合法候选**；策略自身参数全程固定。
- **样本外切分**：寻优 `train_ratio=0.7` / 验证段 0.3，`purged_split` 隔离，寻优 `--min-trades 3`；A/B 为固定参数受控对比，W3+W4 为独立样本外；池化门槛 `MIN_POOLED_TRADES=30`。
- **撮合**：`engine=sequential`，收盘出信号、下一根开盘成交；OHLC 触发止损止盈、跳空按开盘、同根止损优先、平仓当根不再开仓。EMA 在**含预热的完整帧**上计算 signals，再裁到窗口段撮合，避免慢线在窗口起点重新播种。
- **并行与聚合**：按 (策略, 币, 窗口) 分 128 个进程分片，每片只建一次帧（H9）；逐片先 `aggregate_positions` 再取 `return_pct`，避免跨币 `position_id` 撞车；池化 bootstrap `n_boot=2000`、`seed=20261027`。
- **数据与覆盖**：derived（1m 聚合，与回测标准口径一致）；1m 深度 1096 天（2023-11-02 起）；`funding` 8h 桶化、`taker_buy_volume` 由 `data_service` 补齐，本轮未改数据口径。

## 结果

### 全窗口池化每笔期望（4 窗口 × 16 币，净收益，95% CI）

| 策略 | 臂 | 成交笔数 | 每笔期望 % | 95% CI % | verdict |
| --- | --- | ---: | ---: | ---: | --- |
| supertrend_current | 关闭过滤 (0,0) | 3689 | −0.2106 | [−0.3031, −0.1101] | `negative` |
| supertrend_current | 12/50 | 1490 | −0.1868 | [−0.3410, −0.0231] | `negative` |
| supertrend_current | 50/100 | 1569 | −0.1498 | [−0.3090, 0.0120] | `inconclusive` |
| supertrend_current | 20/26 | 1478 | −0.1952 | [−0.3492, −0.0318] | `negative` |
| flow_imbalance_long | 关闭过滤 (0,0) | 825 | −0.4338 | [−0.6391, −0.2091] | `negative` |
| flow_imbalance_long | 12/50 | 715 | −0.3685 | [−0.5938, −0.1462] | `negative` |
| flow_imbalance_long | 50/100 | 557 | −0.2917 | [−0.5469, −0.0169] | `negative` |
| flow_imbalance_long | 20/26 | 732 | −0.3879 | [−0.6053, −0.1584] | `negative` |

**8 个（策略×臂）组合：6 个 `negative`、2 个 `inconclusive`、0 个 `pass`。**

EMA 门控唯一稳定的作用是**削减换手**：supertrend 成交从 3689 降到约 1512（−58%），flow_imbalance 从 825 降到约 668（−19%）。点估计略有抬升，但 CI 全部未推正 —— 属于「少做几笔少亏一点」，不是 alpha。

### 同币配对差（实验臂 − 关闭过滤，64 对，只改 EMA 两键）

| 比较 | 配对数 | 差值 % | 95% CI % | BH-FDR q | 正差值窗口 | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| supertrend × 12/50 | 64 | +0.0065 | [−0.1223, 0.1321] | 0.9680 | 1/4 | `inconclusive` |
| supertrend × 50/100 | 64 | +0.0395 | [−0.0923, 0.1799] | 0.8280 | 1/4 | `inconclusive` |
| supertrend × 20/26 | 64 | +0.0030 | [−0.1209, 0.1210] | 0.9680 | 1/4 | `inconclusive` |
| flow × 12/50 | 64 | +0.0810 | [−0.0570, 0.2276] | 0.6740 | 3/4 | `inconclusive` |
| flow × 50/100 | 64 | +0.2265 | [−0.0503, 0.5248] | 0.6060 | 3/4 | `inconclusive` |
| flow × 20/26 | 64 | +0.0605 | [−0.0622, 0.1860] | 0.6740 | 3/4 | `inconclusive` |

6 个比较**点估计全为正**，但 **CI 全部跨 0、FDR q 最小 0.6060（阈值 0.10）**。按手册表述纪律，只能写 `inconclusive`，不得写成「改善」或「有苗头」。W4 单窗贡献了绝大部分正差值（supertrend 三个臂 W4 分别 +0.51/+0.35/+0.48），而 W1–W3 全为负 —— 分段同号 1/4，未达 3/4。

### 纯样本外 W3+W4（寻优未读取）

| 策略 | 关闭过滤 | 12/50 | 50/100 | 20/26 |
| --- | --- | --- | --- | --- |
| supertrend_current | −0.2072 `[−0.3376,−0.0699]` `negative` | +0.0024 `[−0.2131,0.2388]` `inconclusive` | −0.0646 `[−0.2902,0.1520]` `inconclusive` | −0.0442 `[−0.2606,0.1893]` `inconclusive` |
| flow_imbalance_long | −0.5205 `[−0.7876,−0.2410]` `negative` | −0.3603 `[−0.6546,−0.0687]` `negative` | −0.4321 `[−0.7568,−0.1145]` `negative` | −0.4514 `[−0.7352,−0.1560]` `negative` |

0 个 `pass`；最好的一臂仍是 `inconclusive`。

### 全网格寻优与选参稳定性

- 16 币 × 1h × 180 天 × 4 策略 = **64 行，`pass` 0 行**。分布：supertrend 15 `inconclusive` + 1 `negative`；flow 7 `insufficient` + 5 `inconclusive` + 2 `negative` + 2 `no_trades`；另两个策略各 16 `no_trades`。
- **选参不稳定**：首轮只抽 12/25 档时，最常选中 `12/50`（8 次）；扩到全网格 17 档后，最常选中变成 `50/100`（6 次）、`50/200`（5 次），而 `12/50` 只剩 **1 次**。32 个有交易结果分散在 **12 种组合**上 —— 这是噪声选参的典型形态，不是一个稳定最优解。

### 多种子随机基线（20 种子，方向 50/50 翻转，时点/仓位/成本/风控一致）

关闭过滤臂的收益在随机分布中的分位数：supertrend W1 **0.70** / W2 **0.35** / W3 **1.00** / W4 **0.00**；flow W1 **0.25** / W2 **0.50** / W3 **0.20** / W4 **0.00**。逐窗翻转，没有任何一臂稳定超过 0.95 门槛；W3 的 1.00 与 W4 的 0.00 恰好说明方向判断没有稳定信息量（H6）。

## 判定

- **verdict**：`negative`（绝对收益口径：8 组中 6 组 CI 上界为负、2 组跨 0、0 组下界为正）。受控 A/B 这一问题的次级判定为 `inconclusive`（6 个配对差 CI 全跨 0），**不等于「有改善」**。
- **池化 CI**：全窗口 supertrend 关闭基线 3689 笔、flow 825 笔，均远超 `MIN_POOLED_TRADES=30`；最好的一臂 supertrend×50/100 CI `[−0.3090, 0.0120]`，`ci_low_pct < 0`，`positive_share` 未达 0.95。
- **随机基线**：`n_seeds=20`，分位数逐窗 0.00–1.00，无稳定超过 0.95。
- **FDR / 分段同号**：BH q 0.6060–0.9680 全部 ≥0.10；正差值窗口 supertrend 1/4、flow 3/4，均未与 q 同时过关。
- **归因**：关闭过滤基线本身就是 `negative`（supertrend −0.2106%、flow −0.4338%），说明这两条策略在 major 池 1h 上本就没有正 edge；EMA 门控只是过滤掉约一半信号，让剩余样本的点估计略微上移，但没有把 CI 下界推正。毛 edge 与成本关系：双边 0.12% 已扣，读数仍为负，不是「被成本吃掉」而是方向本身无预测力。周期与池未变，故无周期/池归因项。
- **与历史结论对照**：与 `verdicts-history.md` §1「现货 OHLCV 价格序列已全线证伪」、以及 `docs/indicator-evaluation-current.md` 中 `ema_cross` 家族 168 组检验 0 过线（训练盈亏 −52.111，倒数第三差）**完全一致**。本次把它作为「入场门控」而非「信号源」再验一次，结论仍是 0 `pass` —— 换了用法也翻不了盘。无冲突项。

## 局限

- W1/W2 与寻优窗口重叠，只有 W3/W4 是未被选参读取的样本外；重叠只会让读数偏乐观，不改变 0 `pass` 的结论。
- 未模拟资金费率、盘口深度、强平、分批建仓；撮合假定下一根开盘及时成交，与模拟盘巡检撮合有差异。
- 9 个 spec 里只有 2 个在 1h 上产生可判定样本；其余 7 个或信号太稀疏，或走专用 broker / `direction≡0`，本轮无法对它们给出统计结论（`no_trades` 不等于「过滤无效」）。
- EMA 是收盘价序列的衍生，落在 H2 止损线禁区；本轮为按用户明确要求的受控证伪，不构成新特征投入。
- BNBUSDT 1h native 仅 210 天，靠 1m 聚合派生补齐；W1 末段 1m 数据止于 2026-11-02，少约 2 天。
- 本报告为**名义本金**口径，不得与 margin 口径报告直接相减。

## 结论与后续

**统计上不存在「最佳参数组合」**。全网格 17 个候选、4 个策略、64 行寻优 0 `pass`；受控 A/B 8 组绝对收益 0 `pass`、6 个配对差全部 `inconclusive`、FDR q 全部 ≥0.60；扩大候选集后最常选中组合即从 `12/50` 跳到 `50/100`，选参本身就是噪声。

若仅作**工程参考**（严禁据此启用）：`fast=50 / slow=100` 是全网格最常被选中、且 A/B 中 supertrend 点估计最高（−0.1498%）的一档，但它的 CI `[−0.3090, 0.0120]` 跨 0、verdict 为 `inconclusive`，**未经统计支持**。

- **是否启用**：`default_enabled=False`，**不进任何绑定**；`ema_trend_fast` / `ema_trend_slow` 保持默认 `(0,0)`，全局关闭，既有 9 个策略行为完全不变。
- **止损线登记**：EMA 快慢线方向过滤（价格序列衍生）全线证伪 —— 不再为它调周期、调阈值、换币池或换出场。这是 H2 止损线的又一次实测印证。
- **若仍想推进**：唯一还有信息增量的方向在价格序列之外（衍生品情绪 `oi_value_usd`、资金费率横截面、逐笔成交顺序／盘口挂撤单、多所现货腿），且必须先过 IC 探针（|IC|>0.02 且同号 ≥60%）再谈融合。

## 复现命令

```powershell
# 全网格寻优（17 个合法候选）
.\.venv\Scripts\python.exe backend/scripts/ema_filter_optimize.py --pool major --interval 1h --days 180 --combos 20 --workers 8 --min-trades 3 --out output/ema-filter-major-1h-full
# 受控 A/B 池化判定
.\.venv\Scripts\python.exe backend/scripts/ema_filter_ab.py --workers 8 --out output/ema-filter-ab
# 单测与报告预检
.\.venv\Scripts\python.exe -m pytest backend/tests/test_ema_trend_filter.py -q
.\.venv\Scripts\python.exe .workbuddy\skills\strategy-backtest-playbook\scripts\preflight.py --prefix docs/ema-trend-filter
```

## 产物

- `docs/ema-trend-filter.md` + `docs/ema-trend-filter.json`（本报告成对文件）
- 原始读数：`output/ema-filter-major-1h.csv`（首轮抽样）、`output/ema-filter-major-1h-full.csv` / `.json`（全网格）、`output/ema-filter-ab.json`（128 分片明细）
- 代码：`backend/app/strategy/rules.py`（通用 EMA 门控 + 参数校验）、`backend/app/backtest/optimizer.py`（寻优同口径接入）、`backend/scripts/ema_filter_optimize.py`、`backend/scripts/ema_filter_ab.py`、`backend/tests/test_ema_trend_filter.py`（19 个单测）
- 结论登记至当日工作记忆与项目长期记忆（含「1m 历史深度 1096 天」的数据边界更新）；本轮无未决项，不新增 `OPEN-DECISIONS.md` 条目。

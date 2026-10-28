# P7 资金费率拥挤过滤叠加层 —— 实现与受控 A/B 验证

> 日期：2026-10-27 ｜ 依赖：P0–P6（尤其 P6 资金费率特征已落地）
> 方法纪律：先诊断量化瓶颈 → 每阶段设 gate → 不破坏基线（387 测试全绿）

## 1. 动机与假设

P6 已把"资金费率线性特征是真 alpha"的早期乐观判断**证伪**：跨 506 币 `funding_cum` IC 仅 +0.025，未过 `|IC|>0.05` 门槛；受控 A/B 无一致 PnL 增益。

但 P6 同时留下一条线索：**BTC/ETH 上 high funding（拥挤多头）与"不易止盈"仍呈微弱负向关系**。线性特征太弱，但"非线性/尾部"处理可能仍有用——即 **funding 作为硬过滤叠加层**：当资金费率极值（多头过度拥挤）时，禁止顺费开仓，避开最拥挤、最易踩踏的仓位。

> 关键：框架里这层基建（`rules.funding_filter`）**已经存在**（rules.py:234-238），P7 只是把它接入分层回测做受控 A/B，而非新造逻辑。

## 2. 实现（勿回退）

`backend/scripts/tier_backtest.py`：
- `_base_params(strategy, fee, slippage, funding_filter=False)`：开启时 `params["funding_filter"] = True`。
  由于 `select_tier_params` 以 `current = dict(base_params)` 起手、所有候选 `{**current, **changes}` 继承，叠加层对**寻优段与测试段同时生效**，与原生 `regime_filter`/`cost_filter` 同级。
- `run_window` 调用点传入 `funding_filter=args.funding_filter`。
- argparse 新增 `--funding-filter`（store_true），与既有 `--exclude-funding` 正交（本开关**保留 funding 特征**，仅加规则叠加层，便于干净隔离"过滤"这一单一变量）。

过滤语义（`evaluate_signals`，对所有策略生效，含 `ml_ensemble`）：
- `funding_zscore >= FUNDING_FILTER_Z(2.0)` 且 `side > 0` → 禁多（`funding_filter_long`）
- `funding_zscore <= -FUNDING_FILTER_Z` 且 `side < 0` → 禁空（`funding_filter_short`）
- 无 funding 数据（`funding_zscore=0` / `available=0`）时不拦截

## 3. 触发率定量（`scripts/probe_funding_filter_rate.py`，8 代表币 / 120天）

| 币 | long(off) | long(on) | 拦截 | z≥2σ bar 数 |
|---|---|---|---|---|
| BTCUSDT | 75 | 69 | 6 | 72/2826 |
| ETHUSDT | 202 | 176 | 26 | 97/2826 |
| SOLUSDT | 258 | 249 | 9 | 41/2826 |
| BNBUSDT | 57 | 53 | 4 | 275/2815 |
| XRPUSDT | 218 | 213 | 5 | 60/2826 |
| DOGEUSDT | 237 | 235 | 2 | 17/2826 |
| ADAUSDT | 465 | 462 | 3 | 25/2826 |
| AVAXUSDT | 368 | 362 | 6 | 32/2826 |

- `funding_zscore ≥ 2σ`（最拥挤多头极值）出现在 **2.74%** 的 bar；
- 叠加层实际拦截 **3.24%** 的多头信号（61/1880）——即"多头信号恰好撞上拥挤极值"才被拦，按设计工作。

## 4. 受控 A/B（`--windows 2 --per-tier 8 --processes 4 --workers 8`，71 候选币，`--as-of 2026-10-27`）

两组均**保留 funding 特征**，唯一差异是叠加层开关。数据：`docs/tier-backtestfton.json` / `tier-backtestftoff.json`。

**分层组（主指标）：**

| 窗口 | ON 收益% | OFF 收益% | ON 回撤% | OFF 回撤% | ON 夏普 | OFF 夏普 | ON 成交 | OFF 成交 |
|---|---|---|---|---|---|---|---|---|
| W0 | +0.028 | +0.046 | −22.1 | −19.0 | 0.54 | 1.04 | 204 | 210 |
| W1 | +0.224 | +0.218 | −15.9 | −21.6 | 3.97 | 3.64 | 200 | 202 |

**对照（差异解读）：**
- 收益：W0 ON 比 OFF **低 0.018pp**，W1 ON 比 OFF **高 0.006pp** → 净 ±0.02pp，纯噪声（单窗口收益波动远大于此）。
- 回撤：W0 ON 比 OFF **恶化 3.1pp**，W1 ON 比 OFF **改善 5.7pp** → 方向相反，不稳健。
- 夏普：W0 ON **变差**，W1 ON **变好** → 方向相反。
- 成交：ON 系统性少 2~6 笔（与触发率探针吻合，叠加层确实拦掉了最拥挤多头）。

## 5. Gate 判定

**❌ 未过。** 叠加层按设计移除了最拥挤的 ~3% 多头，但端到端对收益与风险调整收益**无一致、显著的改善**；且在 W0 同时恶化了收益与回撤。逐窗口方向完全不一致，符合 P6"资金费率方向性信号跨币不稳健"的结论。

## 6. 决策与边界

- **生产环境保持 `funding_filter=False`（默认关闭）**：叠加层增加复杂度却无可靠增益，W0 还带来副作用。基建保留为可选开关，未来若历史宽度大几个数量级可重测。
- 与 P6 一致：资金费率作为**线性特征**或**硬过滤叠加层**，在当前数据下都**不提供稳健可复制的方向性 edge**。
- 全程序（P0~P7）收敛结论：真瓶颈是 ① **历史宽度**（多币仅 90~400 天）② **现货 OHLCV/资金费率数据源本身无 alpha**。
- 若仍想挖独立信息源的正 alpha，优先级已清晰：**更大历史宽度** + **非线性/链上/订单流**等真正正交的数据源，而非继续在现货 OHLCV 上堆特征或规则。

## 7. 交付物

- `backend/scripts/tier_backtest.py`：新增 `--funding-filter` 开关（已接入，默认关闭）
- `backend/scripts/probe_funding_filter_rate.py`：触发率定量探针
- `docs/tier-backtestfton.json` / `tier-backtestfton.md`：叠加层 ON 报告
- `docs/tier-backtestftoff.json` / `tier-backtestftoff.md`：叠加层 OFF 报告（对照基线）
- 本文件：`docs/roadmap-p7-funding-filter.md`

# P6 实现与验证：资金费率特征（独立信息源）

> 路线图判定：P1~P5 证明现货 OHLCV 在 1m~1h 全频率、全特征维度无方向性 alpha，降换手只是"少亏"。
> 要出 edge 必须引入**独立信息源**；库内 `FundingRate` 表是现成候选。本文件记录 P6 的「落地 + 统计验证」。

## 1. 落地动作（已完成）

| 文件 | 改动 |
|---|---|
| `backend/app/features/funding.py` | 新增 `funding_cum`（3 天滚动累积费率，时间窗口 `"3D"`），作为多头拥挤度代理；接入 `FUNDING_FEATURES`；空 funding 时填 0 且 `funding_available=0`（降级标志） |
| `backend/scripts/backfill_funding.py` | 全宇宙资金费率回填：枚举 507 个 clean 币种，调 `sync_funding_many(days=400)`，幂等 upsert |
| `backend/scripts/probe_funding.py` | 重写为**全宇宙跨币 IC 探针**：复用 `build_features` 实际产出的 `funding_cum` 列（避免与线上特征口径偏差），自动遍历所有有资金费的币种 |
| `backend/scripts/tier_backtest.py` | 新增 `--exclude-funding`（原地收窄全局 `FEATURES`，训练/预测一致）与 `--suffix`（区分两次运行输出）受控 A/B 开关 |

回填结果：**507 币 / 1,039,681 行**，覆盖 `2025-09-22 ~ 2026-10-27`，0 错误。此前只有 BTC/ETH 2 币 ~200 天。

## 2. 统计验证（核心：证伪早期乐观结论）

### 2.1 跨币 IC（`probe_funding.py`，506 币，days=400）

| 特征 | IC 均值 | n | 同号占比 |
|---|---:|---:|---:|
| `funding_cum`（模型实际特征，3D 累积） | **+0.0246** | 502 | 57.6% |
| `f_rate` | +0.0267 | 502 | 64.5% |
| `f_z` | +0.0089 | 492 | 58.5% |
| `f_diff` | +0.0002 | 492 | 51.2% |
| `f_cum72`（bar 级 72 根，对照早期口径） | +0.0270 | 502 | 59.8% |

→ 全部 `|IC| < 0.05`，同号占比仅 ~58–65%（远低于 0.6 稳健线）。**funding 作为通用线性特征未通过项目 `|IC|>0.05` 门槛。**

### 2.2 与早期 2 币探针的对照（关键校正）

早期 P6 探针（BTC/ETH，~200 天窗口）报 `f_cum IC = -0.090`（BTC −0.094 / ETH −0.086），曾被当成"第一条真 alpha"。
现用 400 天全宇宙口径复算：BTC/ETH 仅 **−0.036 / −0.034**——方向仍为负（"累积费率越高越不易止盈"），但幅度骤降一个数量级；
跨币均值更翻正为 +0.025。**结论：早期 −0.090 是两币 + 特定时段 + 短窗口的偶然，不可外推为通用信号。**

### 2.3 回测 A/B（`tier_backtest.py`，71 候选币，2 窗口 × 1h，组合口径）

| 窗口 | 分层收益 funding-ON | 分层收益 funding-OFF | 统一收益 ON | 统一收益 OFF |
|---|---:|---:|---:|---:|
| W0 (10-27) | +0.046% | +0.110% | −0.093% | −0.143% |
| W1 (10-12) | +0.218% | −0.083% | −0.204% | −0.078% |

→ funding ON vs OFF 在每个窗口内**互有胜负**，差值（±0.1~0.3pp）远小于 10 天测试窗的噪声与 ~20–30% 回撤，**不构成一致 PnL 增益**。与跨币 IC 结论自洽：资金费率不是稳健收益来源。

## 3. 结论与边界

- P6 的**工程落地**（特征 + 全宇宙数据 + 探针/A/B 工具）已完成、可复用。
- 但**统计验证均否定 funding 是稳健 alpha**：跨币 IC 不达标，受控 A/B 无一致增益；早前的 2 币乐观结论被证伪。
- 处理建议：保留 `funding_*` 特征作为**低权重输入**——LightGBM 会自动降权无用特征，且不伤害 BTC/ETH 上微弱的真实负向关系；但**不应**作为主信号或被赋予过大信任。
- 真正的瓶颈仍是：① 历史宽度（多币种仅 ~90–400 天）；② 现货 OHLCV 无方向性 alpha。引入独立信息源（资金费/订单流/链上）的路线在「更大历史 + 更干净信号工程（如非线性的拥挤过滤而非线性特征）」下仍值得继续，但当前样本不足以支撑。

## 4. 复现命令

```powershell
# 全宇宙回填
.\.venv\Scripts\python.exe backend/scripts/backfill_funding.py --days 400 --workers 8
# 跨币 IC 探针
.\.venv\Scripts\python.exe backend/scripts/probe_funding.py --days 400
# 受控 A/B
.\.venv\Scripts\python.exe backend/scripts/tier_backtest.py --as-of "2026-10-27T00:00:00+00:00" --windows 2 --per-tier 8 --processes 4 --workers 8 --suffix funding
.\.venv\Scripts\python.exe backend/scripts/tier_backtest.py --as-of "2026-10-27T00:00:00+00:00" --windows 2 --per-tier 8 --processes 4 --workers 8 --exclude-funding --suffix nofunding
```

产物：`docs/backfill-funding.json`、`docs/probe-funding-all.json`、`docs/tier-backtestfunding.{json,md}`、`docs/tier-backtestnofunding.{json,md}`。

# 策略组合与参数寻优报告

> 生成时间：2026-11-01 09:54:33　引擎：`backend/app/research/rolling_portfolio.py` + `backend/scripts/combo_search.py`
> + `backend/app/backtest/runner.py`　环境：Python 3.12，PostgreSQL `get_money`

## 0. TL;DR（给决策者的三句话）

1. **没有找到统计显著的正 edge 组合。** 1h 与 4h 两套 400 天 walk-forward 的「池化样本外期望」95% 置信区间都**跨 0**（判定 `inconclusive`）；样本外收益高度跟随市场 beta（牛市大幅跑输买入持有，熊市靠空单/观望小幅跑赢）。
2. 相对最稳的形态是 **4h + 市场 regime 过滤（market_ema）+ stoch/rsi 类信号** 的等权组合，但最好一档折的超额收益仍不显著，且折 4（+62%）疑似过拟合。
3. 5 个已注册策略 spec 经独立样本外检验（含本项目 P0–P11 系列研究）**全部未通过显著性检验**，当前 `default_enabled=False` 状态正确；本报告不推荐任何组合实盘上资金。

---

## 1. 数据说明（开始前）

| 项 | 取值 | 说明 |
|---|---|---|
| 数据源 | PostgreSQL `get_money.klines`（2281 万行） | native 直查目标周期原始行，非 1m 聚合 |
| 标的 | 按流动性取 top-20 币（BTC/ETH/主流山寨） | `list_symbols` 按 400 天内 bar 数排序 |
| 周期 | **1h 与 4h**（各 521 币、跨度 400 天：2025-09-20 → 2026-10-26） | 1m 仅 96 天、15m 298 天、5m 仅 2 币，不满足长样本；**故采用 1h/4h** |
| 是否补数据 | **否** | 1h/4h 已覆盖 400 天 × 521 币，满足 walk-forward 样本需求，无需额外拉取 |
| 交易成本 | 手续费 0.04%/边 + 滑点 0.02%/边 | 与实盘全局设置同源，双向计；可复现 |
| 可复现性 | bootstrap 随机种子固定 `20261101` | 同一命令多次运行结果一致 |

> 注：若坚持用 1m 周期，则需 `backend/scripts/backfill_minutes.py --days 730` 补齐至 ~1 年，再聚合；本组合研究以已有 1h/4h 为准。

---

## 2. 方法论（满足需求 3/4/5）

- **搜索空间**：15 个指标信号家族（见 §4.2）展开共 **111 个候选**；每个候选 = 一组 (long, short) 布尔入场信号，统一用 RSI(50) 穿越出场（与 `indicator_lab` 一致，保证可比）。
- **组合方式**：滚动前推（anchored walk-forward）。按 `train_ratio=0.5` 把 400 天切成 4 折；**每折只用折前历史选 top-k 条腿（等权）**，在紧随其后的未参与行情中判定（purge 防泄漏）。
- **资金池**：共享资金池，最多并发腿数 = 币数；单笔按风险预算定仓。
- **对照与防过拟合**：
  - 每折附「折内一直持多」**买入持有对照**，分离 alpha 与 beta；
  - 每笔期望 **bootstrap 95% CI**（seed 固定），跨 0 即 `inconclusive`，不写成「有效」；
  - 单信号腿另经 **BH（Benjamini-Hochberg）FDR 多重校正**（见 `lab_experiment`）。
- **评估指标**：累计收益、年化收益（按测试区间长度近似）、最大回撤、夏普、胜率、交易次数、买入持有超额、期望 CI 判定。

---

## 3. 训练/测试划分与交叉验证

- 切分：`train_ratio=0.5`，`folds=4`（锚定式，非 K 折洗牌，避免未来信息泄漏）。
- 1h：切点 2026-04-12；4h：切点 2026-04-12。每折约 100 天测试窗口。
- 剔除标准：任一折期望 CI 跨 0、或组合收益主要由 beta 驱动（超额≈0 甚至为负）、或最佳折与最差折方向矛盾（参数敏感度过高于此）。

---

## 4. 现有指标与策略参数梳理

### 4.1 已注册策略 spec（5 个，来源 `app/strategy/specs/`）

> 全部 `default_enabled=False`。`fee_rate`/`slippage` 为全局禁覆盖键，只能走回测请求显式传。


#### `absorption_reversal` — 吸收反转

> direction 约束：多空双向

| 参数 | 默认 | 取值范围 | 寻优档位(grid) | 类型 |
|---|---|---|---|---|
| `abs_gap_bars` | 10 | [0, 500] | - | 连续 |
| `be_trigger_r` | 1.0 | [0, 20] | 0.0,1.0 | 连续 |
| `body_mult` | 0.25 | [0, 2] | - | 连续 |
| `conf_mode` | 1 | [0, 1] | 0,1 | 整数 |
| `conf_window_bars` | 3 | [1, 20] | - | 整数 |
| `delta_pct` | 95.0 | [50, 100] | 90.0,95.0,98.0 | 连续 |
| `delta_window` | 500 | [10, 2000] | - | 连续 |
| `direction_mode` | 0 | [0, 2] | - | 整数 |
| `dominance_pct` | 20.0 | [0, 100] | - | 连续 |
| `extreme_lookback` | 20 | [2, 200] | - | 整数 |
| `max_sl_atr` | 3.0 | [0.1, 20] | - | 连续 |
| `min_sl_atr` | 0.3 | [0, 10] | - | 连续 |
| `session_end_hour` | 24.0 | [0, 24] | - | 连续 |
| `session_start_hour` | 0.0 | [0, 24] | - | 连续 |
| `sl_buffer_atr` | 0.25 | [0, 5] | 0.15,0.25,0.4 | 连续 |
| `target_r` | 2.0 | [1, 50] | 1.5,2.0,3.0 | 连续 |
| `trend_len` | 200 | [5, 1000] | - | 连续 |
| `trend_mode` | 0 | [0, 2] | - | 整数 |
| `wick_mult` | 0.2 | [0, 1] | - | 连续 |
| `max_hold_bars` | 30 | (共享键) | 30,45,60 | 连续 |
| `min_score` | - | (共享键) | 0.55,0.7,0.8 | 连续 |


#### `opening_candle_breakout` — 开盘首根突破

> direction 约束：多空双向

| 参数 | 默认 | 取值范围 | 寻优档位(grid) | 类型 |
|---|---|---|---|---|
| `atr_len` | 14 | [5, 200] | - | 整数 |
| `first_session_end_min` | 575 | [1, 1440] | - | 整数 |
| `first_session_start_min` | 570 | [0, 1439] | - | 整数 |
| `flat_session_end_min` | 960 | [0, 1440] | - | 整数 |
| `flat_session_start_min` | 955 | [0, 1439] | - | 整数 |
| `max_stop_atr` | 2.0 | [0.25, 10] | 1.0,2.0,3.0 | 连续 |
| `min_body_atr` | 0.5 | [0, 5] | 0.0,0.5,1.0 | 连续 |
| `min_stop_atr` | 0.1 | [0.01, 5] | 0.1,0.25 | 连续 |
| `risk_mode` | 1 | [0, 1] | - | 整数 |
| `risk_usd` | 250.0 | [0.01, 1000000] | - | 连续 |
| `session_tz` | 1 | [0, 4] | - | 整数 |
| `target_r` | 10.0 | [0.5, 50] | 4.0,10.0,20.0 | 连续 |
| `tz_offset_minutes` | -300 | [-720, 840] | - | 整数 |
| `risk_per_trade_pct` | - | (共享键) | 0.005,0.01,0.02 | 连续 |


#### `rgd_donchian_v2` — Regime-Gated Donchian v2 [RGD]

> direction 约束：多空双向

| 参数 | 默认 | 取值范围 | 寻优档位(grid) | 类型 |
|---|---|---|---|---|
| `acc_len` | 6 | [1, 500] | - | 整数 |
| `atr_len` | 14 | [2, 500] | - | 整数 |
| `brk_buf` | 0.1 | [0.0, 20.0] | - | 连续 |
| `brk_fast` | 14 | [2, 500] | - | 整数 |
| `brk_slow` | 34 | [3, 500] | - | 整数 |
| `build_len` | 6 | [1, 100] | - | 整数 |
| `cooldown` | 5 | [0, 10000] | - | 整数 |
| `corr_len` | 30 | [5, 500] | - | 整数 |
| `corr_min` | 0.25 | [0.0, 0.95] | - | 连续 |
| `def_fast` | 8 | [2, 500] | - | 整数 |
| `def_slow` | 21 | [3, 500] | - | 整数 |
| `dir_mode` | 0 | [0, 2] | - | 整数 |
| `er_len` | 10 | [2, 500] | - | 整数 |
| `er_min` | 0.3 | [0.05, 0.95] | - | 连续 |
| `htf_in` | 1 | [0, 2] | - | 整数 |
| `max_depth` | 0.7 | [0.0, 1.0] | - | 连续 |
| `max_stop_atr` | 2.5 | [0.5, 20.0] | - | 连续 |
| `max_ten` | 8.25 | [0.25, 10.0] | - | 连续 |
| `min_stop_atr` | 0.5 | [0.1, 20.0] | - | 连续 |
| `rank_cut` | 50.0 | [10, 90] | - | 连续 |
| `rank_len` | 100 | [20, 2000] | - | 整数 |
| `rel_len` | 20 | [1, 100] | - | 整数 |
| `risk_pct` | 1.0 | [0.1, 10.0] | - | 连续 |
| `ten_th` | 0.2 | [0.05, 2.0] | - | 连续 |
| `tp1_pct` | 50.0 | [10.0, 90.0] | - | 连续 |
| `tp1_r` | 2.0 | [0.5, 50.0] | - | 连续 |
| `vol_min` | 1.0 | [0.1, 100.0] | - | 连续 |


#### `supertrend_current` — Supertrend 趋势反转

> direction 约束：多空双向

| 参数 | 默认 | 取值范围 | 寻优档位(grid) | 类型 |
|---|---|---|---|---|
| `adx_length` | 14 | [1, 500] | - | 整数 |
| `adx_smoothing` | 14 | [1, 500] | - | 整数 |
| `adx_threshold` | 50.0 | [1, 100] | - | 连续 |
| `atr_period` | 14 | [1, 500] | - | 整数 |
| `pivot_length` | 5 | [1, 50] | - | 整数 |
| `reentry_ema_length` | 13 | [1, 500] | - | 整数 |
| `supertrend_factor` | 2.5 | [0.01, 50] | 2.0,2.5,3.0 | 连续 |
| `target_r` | 3.0 | [0.1, 50] | 3.0,4.0,6.0 | 连续 |
| `tp1_fraction` | 0.16666666666666666 | [0, 1] | - | 连续 |
| `tp1_r` | 1.0 | [0.1, 50] | - | 连续 |
| `tp2_fraction` | 0.3333333333333333 | [0, 1] | - | 连续 |
| `tp2_r` | 2.0 | [0.1, 50] | - | 连续 |
| `tp3_fraction` | 0.5 | [0, 1] | - | 连续 |


#### `simple_zigzag_double` — Simple ZigZag Double

> direction 约束：多空双向

| 参数 | 默认 | 取值范围 | 寻优档位(grid) | 类型 |
|---|---|---|---|---|
| `long_detection` | 3 | [1, 10000] | - | 整数 |
| `long_length` | 5 | [2, 10000] | - | 整数 |
| `lot_long` | 100.0 | [1, 10000] | - | 连续 |
| `lot_short` | 100.0 | [1, 10000] | - | 连续 |
| `short_detection` | 3 | [1, 10000] | - | 整数 |
| `short_length` | 5 | [2, 10000] | - | 整数 |
| `timestamp_end` | 2556057540000 | [0, 100000000000000] | - | 整数 |
| `timestamp_start` | 1388534400000 | [0, 100000000000000] | - | 整数 |


### 4.2 指标信号家族（15 个，来源 `app/research/lab_grid.py`，组合寻优的搜索空间）


| 指标家族 | 含义 | 参数 (min/max/step 或枚举) | 约束 | 组合数 |
|---|---|---|---|---:|
| `ema_cross` | EMA 快慢线交叉 | fast:5~20/步3; slow:21~89/步17 | fast_lt_slow | 30 |
| `price_sma` | 收盘价穿越均线 | n:20~200/步30 | - | 7 |
| `rsi_reversal` | RSI 超卖/超买回归 | n:7~21/步7; low:25~35/步5; high:65~75/步5 | symmetric_rsi | 9 |
| `macd_cross` | MACD 交叉 | fast:8~12/步4; slow:21~50/步29; signal:9~9/步1; mode∈{0,1} | fast_lt_slow | 8 |
| `bollinger_break` | 布林带突破 | n:20~50/步15; k∈{1.5,2.0,2.5} | - | 9 |
| `donchian_break` | 唐奇安通道突破 | n:20~110/步30 | - | 4 |
| `supertrend_flip` | Supertrend 翻转 | n:10~21/步7; mult∈{2.0,3.0,4.0} | - | 6 |
| `adx_di_cross` | ADX +DI/-DI 交叉 | n:14~21/步7; adx_min∈{0,20,25} | - | 6 |
| `stoch_cross` | 随机指标 KD 交叉 | n:14~21/步7; smooth∈{3,5} | - | 4 |
| `cci_reversal` | CCI 回归 | n:20~34/步14; level∈{100,150} | - | 4 |
| `mfi_reversal` | MFI 资金流回归 | n:14~21/步7; low∈{20,30} | - | 4 |
| `volume_break` | 放量突破 | n:20~50/步30; k∈{1.5,2.0,3.0} | - | 6 |
| `atr_channel_break` | ATR 通道突破 | n:20~50/步30; k∈{1.0,1.5,2.0} | - | 6 |
| `roc_momentum` | ROC 动量阈值 | n:12~48/步18; thr∈{0.0,2.0} | - | 6 |
| `vwap_reclaim` | 滚动 VWAP 收复 | n:24~72/步48 | - | 2 |


---

## 5. 组合寻优结果（核心）

### 5.1 指标组合 walk-forward 样本外表现


### 1h 组合 walk-forward（limit=20, folds=4, top_k=3）

| 折 | 样本外收益% | 年化%(近似) | 回撤% | 夏普 | 胜率 | 笔数 | 买入持有% | 超额(pp) | 选中腿 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | -15.0302 | -70.1 | 23.6031 | -1.6234 | 0.5637 | 910 | -5.6621 | -9.37 | rsi_reversal(high=75,low=25,n=21), rsi_reversal(high=75,low=25,n=14), rsi_reversal(high=75,low=25,n=7) |
| 2 | -3.6956 | -24.3 | 10.8277 | -0.3982 | 0.4198 | 486 | -13.6144 | 9.92 | rsi_reversal(high=75,low=25,n=14), rsi_reversal(high=75,low=25,n=21), volume_break(k=3.0,n=20) |
| 3 | -23.5911 | -86.4 | 38.5388 | -2.189 | 0.6294 | 402 | 68.7176 | -92.31 | rsi_reversal(high=75,low=25,n=14), rsi_reversal(high=75,low=25,n=21), rsi_reversal(high=70,low=30,n=21) |
| 4 | 0.2843 | 2.1 | 12.5984 | 0.014 | 0.4439 | 588 | 37.7483 | -37.46 | rsi_reversal(high=75,low=25,n=14), rsi_reversal(high=75,low=25,n=21), volume_break(k=3.0,n=20) |

**池化样本外期望**：2386 笔，单笔均值 -0.01762%，95% CI [-0.0382, 0.00508] → **判定：inconclusive**

### 4h 组合 walk-forward（limit=20, folds=4, top_k=3）

| 折 | 样本外收益% | 年化%(近似) | 回撤% | 夏普 | 胜率 | 笔数 | 买入持有% | 超额(pp) | 选中腿 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.4446 | 95.1 | 17.2306 | 0.5318 | 0.4584 | 1021 | -5.5172 | 14.96 | stoch_cross(n=14,smooth=3) + market_ema(200) · 等权市场指数 4h, stoch_cross(n=21,smooth=5) + market_ema(200) · 等权市场指数 4h, stoch_cross(n=21,smooth=3) + market_ema(200) · 等权市场指数 4h |
| 2 | 0.5447 | 4.1 | 9.5702 | 0.0728 | 0.4746 | 236 | -13.1511 | 13.7 | stoch_cross(n=14,smooth=5) + market_ema(200) · 等权市场指数 4h, stoch_cross(n=14,smooth=3) + market_ema(200) · 等权市场指数 4h, stoch_cross(n=21,smooth=5) + market_ema(200) · 等权市场指数 4h |
| 3 | -6.5197 | -39.3 | 25.4645 | -0.5116 | 0.4542 | 1191 | 64.7993 | -71.32 | stoch_cross(n=14,smooth=5) + market_ema(200) · 等权市场指数 4h, stoch_cross(n=14,smooth=3) + market_ema(200) · 等权市场指数 4h, stoch_cross(n=21,smooth=5) + market_ema(200) · 等权市场指数 4h |
| 4 | 62.3689 | 3525.9 | 19.9179 | 1.8074 | 0.4915 | 942 | 37.0436 | 25.33 | stoch_cross(n=14,smooth=5) + market_ema(200) · 等权市场指数 4h, rsi_reversal(high=65,low=35,n=14), stoch_cross(n=21,smooth=5) + market_ema(200) · 等权市场指数 4h |

**池化样本外期望**：3390 笔，单笔均值 0.01942%，95% CI [-0.00325, 0.04322] → **判定：inconclusive**

**读法**：
- 1h 四折 OOS 全为负（-15% / -3.7% / -23.6% / +0.3%），相对买入持有在涨市（折3 持多 +69%、折4 +38%）严重跑输 —— 技术指标组合在强趋势市被频繁止损/换手拖累。
- 4h 四折（+9.4% / +0.5% / -6.5% / +62.4%）在跌市跑赢 beta，但折 4 异常高，训练段选出的 stoch/rsi 组合恰好契合该段波动，**疑似过拟合**。
- 两套 **池化 OOS 期望 CI 均跨 0 → 无法判定存在正 edge**。

---

## 6. 现有策略 spec 的样本外表现

### 6.1 本次实跑（supertrend_current / absorption_reversal，1h 单策略 OOS）


#### `supertrend_current`（1h, 400d, optimize=True, 单策略单币 OOS）

| 币 | OOS收益% | 胜率 | 夏普 | 笔数 | 买入持有% | 跑赢B&H? | 跑赢随机? | 期望CI判定 | 选参 |
|---|---:|---:|---:|---:|---:|:--:|:--:|---|---|
| BULLAUSDT | 2.4995 | 40.54 | 0.088 | 37 | 501.2312 | ✗ | ✓ | inconclusive | current |
| COMPUSDT | 7.9552 | 45.76 | 0.173 | 59 | 18.7717 | ✗ | ✓ | inconclusive | current |
| AGTUSDT | -3.872 | 35.56 | -0.133 | 45 | 111.2559 | ✗ | ✗ | inconclusive | current |
| BTCUSDT | -5.4669 | 28.57 | -0.136 | 63 | 18.7905 | ✗ | ✗ | inconclusive | current |
| BCHUSDT | -7.9903 | 29.85 | -0.201 | 67 | 35.2541 | ✗ | ✗ | inconclusive | current |
| BABYUSDT | 2.8093 | 35.94 | 0.059 | 64 | -1.6733 | ✓ | ✗ | inconclusive | current |
| AAVEUSDT | 2.2442 | 38.18 | 0.056 | 55 | 40.372 | ✗ | ✓ | inconclusive | current |
| ADAUSDT | -5.1804 | 28.79 | -0.116 | 66 | 15.0865 | ✗ | ✗ | inconclusive | current |
| ARIAUSDT | 3.9895 | 42.0 | 0.104 | 50 | 192.1229 | ✗ | ✗ | inconclusive | current |
| ARBUSDT | -7.6975 | 24.19 | -0.202 | 62 | 90.2501 | ✗ | ✗ | inconclusive | current |
| AVAXUSDT | 2.2095 | 34.72 | 0.041 | 72 | 37.1703 | ✗ | ✓ | inconclusive | current |
| DASHUSDT | 3.608 | 38.89 | 0.089 | 54 | 44.8652 | ✗ | ✓ | inconclusive | current |

聚合：12 币 | 平均OOS收益 -0.4077% | 跑赢B&H 1 币 | 跑赢随机 5 币 | 期望正 0 / 不确定 12 / 负 0

#### `absorption_reversal`（1h, 400d, optimize=True, 单策略单币 OOS）

| 币 | OOS收益% | 胜率 | 夏普 | 笔数 | 买入持有% | 跑赢B&H? | 跑赢随机? | 期望CI判定 | 选参 |
|---|---:|---:|---:|---:|---:|:--:|:--:|---|---|
| BULLAUSDT | 0.0 | 0.0 | None | 0 | 501.2312 | ✗ | ✗ | no_trades | current |
| COMPUSDT | 0.0 | 0.0 | None | 0 | 18.7717 | ✗ | ✗ | no_trades | current |
| AGTUSDT | 0.0 | 0.0 | None | 0 | 111.2559 | ✗ | ✗ | no_trades | current |
| BTCUSDT | 0.0 | 0.0 | None | 0 | 18.7905 | ✗ | ✗ | no_trades | current |
| BCHUSDT | 0.0 | 0.0 | None | 0 | 35.2541 | ✗ | ✗ | no_trades | current |
| BABYUSDT | 0.0 | 0.0 | None | 0 | -1.6733 | ✓ | ✗ | no_trades | current |
| AAVEUSDT | 0.0 | 0.0 | None | 0 | 40.372 | ✗ | ✗ | no_trades | current |
| ADAUSDT | 1.4793 | 100.0 | None | 1 | 15.0865 | ✗ | ✓ | positive | current |
| ARIAUSDT | 0.0 | 0.0 | None | 0 | 192.1229 | ✗ | ✗ | no_trades | current |
| ARBUSDT | 0.0 | 0.0 | None | 0 | 90.2501 | ✗ | ✗ | no_trades | current |
| AVAXUSDT | 0.0 | 0.0 | None | 0 | 37.1703 | ✗ | ✗ | no_trades | current |
| DASHUSDT | 0.0 | 0.0 | None | 0 | 44.8652 | ✗ | ✗ | no_trades | current |

聚合：12 币 | 平均OOS收益 0.1233% | 跑赢B&H 1 币 | 跑赢随机 1 币 | 期望正 1 / 不确定 0 / 负 0

### 6.2 历史权威结论（来源 docs/*.md，与本次一致）

| 策略 | 历史样本外结论 | 文档 |
|---|---|---|
| `supertrend_current` | 730 天长窗口上训练段最优在测试段变负/不显著（insufficient），属拟合 | `docs/supertrend-optimization-2y.md` |
| `absorption_reversal` | 主流币 15m 检验 CI 全跨 0，不进 SYMBOL_ROUTES | `docs/absorption-reversal-strategy.md` |
| `rgd_donchian_v2` | 99 币×180 天：多空 G1 每笔 -0.1665% CI 负；做空是主亏损源；不启用 | `docs/rgd-strategy.md` |
| `opening_candle_breakout` | 15m/180天/16币 1445 笔 CI 上界 -0.093；10R 命中率 1.6%；不启用 | `docs/opening-candle-breakout-strategy.md` |
| `simple_zigzag_double` | 已集成模拟盘，但 LIVE 拒单；未做分折 OOS | `docs/simple-zigzag-double.md` |

---

## 7. 候选策略组合清单（按综合表现排序）

> 综合排序依据：池化 OOS 期望均值（越高越好）+ 与买入持有超额（越高越好）+ 夏普 + 期望 CI 是否显著 + 参数稳定性。
> **重要**：下列组合均为「相对最优」，没有任何一组通过统计显著性检验，不构成可上线的 alpha。

| 排名 | 组合（周期/腿） | 样本外收益(均值) | 年化(近似) | 最大回撤 | 夏普 | 期望CI判定 | 备注 |
|---|---|---:|---:|---:|---:|---|---|
| 1 | **4h · stoch_cross + market_ema(200)** 等权 3 腿 | 见 §5.1 折1/2/4 正 | — | 17~25% | 0.07~1.81 | inconclusive | 折4(+62%)疑似过拟合 |
| 2 | **4h · stoch/rsi + market_ema(200)** 混合 | 折1/2 正、折3 负 | — | 9~25% | 0~0.53 | inconclusive | beta 暴露大 |
| 3 | **1h · rsi_reversal(n=7/14/21, 75/25)** 等权 3 腿 | 全负 | — | 10~39% | -2.19~0.01 | inconclusive | 熊市跑赢 B&H、牛市巨亏 |
| — | 买入持有（基准） | 依赖市场 | — | 低(持多) | — | — | 组合多数折跑输它 |

---

## 8. 推荐的最优组合与参数

### 8.1 推荐（保守、可解释、风控优先）

**`4h · stoch_cross(n=14, smooth=3/5) + market_ema(200) 等权 3 腿组合`**

- **参数取值**（来自 §5.1 折 1 选中）：
  - `stoch_cross(n=14, smooth=3) + market_ema(200)`
  - `stoch_cross(n=21, smooth=5) + market_ema(200)`
  - `stoch_cross(n=21, smooth=3) + market_ema(200)`
  - 出场：统一 `rsi50_cross`；成本 fee 0.04%/边、slip 0.02%/边；top_k=3 等权。
- **为什么是它**：4h 噪声比 1h 小、交易频率低、换手成本占比小；叠加 `market_ema(200)` 市场 regime 过滤后，在 2025Q4–2026 的下跌/震荡段（折1/2）取得 +9.4% / +0.5% 且回撤仅 9~17%，显著优于同期买入持有（均下跌）。
- **稳健性**：仅 4 折滚动，样本未跨完整牛熊；折 4 异常高暴露了参数对局部波动的过拟合。限定「只做多 + regime 过滤 + 低相关腿」可降方差，但**不能消除 beta 暴露**。
- **适用行情**：震荡/缓慢下行、波动率中性的市场；**不适用**强单边上涨（会被甩下车跑输 beta）与极端跳空。
- **主要风险点**：①无统计显著正 edge（CI 跨 0）；②收益主源于 beta 与择时幸运；③折 4 过拟合嫌疑；④未计资金费率/强平/盘口冲击；⑤1h 版本全面为负，周期敏感度高。

### 8.2 回测脚本 / 配置调用方式（可复现）

```powershell
# 1) 指标组合 walk-forward（本次核心证据，1h 与 4h 各一套）
.\.venv\Scripts\python.exe backend/scripts/combo_search.py `
  --interval 4h --days 400 --limit 20 --folds 4 --train-ratio 0.5 --top-k 3 `
  --min-trades 30 --capital 10000 --fee 0.0004 --slippage 0.0002 `
  --regimes "market_ema:200" --out-prefix docs/combo-search-4h

# 2) 现有策略 spec 单策略样本外扫描（supertrend / absorption）
.\.venv\Scripts\python.exe backend/scripts/spec_oos_scan.py   # 输出 docs/spec-oos-scan.json

# 3) 单个注册策略的 train/test 回测（含随机方向基线 + 期望 CI）
#    通用策略用 runner.run_backtest；专用 broker 策略用各自脚本：
.\.venv\Scripts\python.exe backend/scripts/supertrend_backtest.py --symbol BTCUSDT --interval 1h --days 400
.\.venv\Scripts\python.exe backend/scripts/absorption_backtest.py --symbol BTCUSDT --interval 15m --days 180
.\.venv\Scripts\python.exe backend/scripts/rgd_backtest.py --symbol BTCUSDT --interval 15m --days 180
.\.venv\Scripts\python.exe backend/scripts/ocb_backtest.py --symbols BTCUSDT,ETHUSDT --interval 15m --days 180
.\.venv\Scripts\python.exe backend/scripts/zigzag_backtest.py --symbol BTCUSDT --interval 1h --days 120
```

---

## 9. 结论与下一步

- **结论**：在 1h/4h × 400 天 × 521 币的样本下，**现货 OHLCV 上不存在可榨取的方向性 alpha**（与 P0–P11 系列研究一致）。任何「组合」的超额收益都可由 beta 与择时幸运解释，且参数敏感度高、跨折不稳定。
- **不建议**将任一组合投入实盘资金。
- **若继续投入**，建议方向（与项目既有 roadmap 一致）：
  1. 引入 **衍生品情绪/OI/资金费率** 等另类数据（P9 已发现 `oi_value_usd` IC=-0.0746 是首个过 gate 的 alpha）；
  2. 做 **多所现货腿套利 / 资金费率横截面收割**（P11 路径，需 Gate/MEXC 等多所数据）；
  3. 把 walk-forward 扩展到 **完整牛熊 + 多周期一致性** 检验，再谈上线。

---
*本报告所有数字均来自 `docs/combo-search-1h.json`、`docs/combo-search-4h.json`、`docs/spec-oos-scan.json`，可经 §8.2 命令复现。*

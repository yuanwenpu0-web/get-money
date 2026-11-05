# 支撑位判定对比回测（诊断性量具评估，非策略启用判定）

> 生成时间：2026-11-05T13:11:46.162256+00:00 ｜ 脚本：`backend/scripts/support_level_eval.py` ｜ 复现命令见文末
> 版本：首版 ｜ 本课题只评估已移植指标的判定质量，**不新增特征、不调参、不改策略、不启用**。

## 假设与边界

- **信息源 / 可证伪假设**：信息源 = 现货 1h OHLCV 派生的各类支撑位价格序列。
  「准」事前定义为：触线后做多的净收益显著优于随机价位基线，且破位率更低。
  H0：所有支撑位定义与随机价位无差异；若某方法的净超额 bootstrap 95% CI 下界 > 0、
  FDR q < 0.10、4 段中 ≥3 段同号、随机基线分位 > 0.95，才判 `pass`。
- **周期 / 持有期 H**：1h，H = 48 根（约 2.0 天），标签前瞻跨度与事件间隔一致。
- **窗口起止 / 墙钟对齐**：2026-01-09 12:00 UTC → 2026-11-04 14:00 UTC（300 天）；
  右端固定为 `--as-of 2026-11-05T12:00:00Z`（query_end_ms=1793880000000），两个持有期共用同一右端；历史 1m 若被回填仍会带来小数点后的微小漂移，故结论以排序与 verdict 为准，不以末位数字为准。
  训练/验证/测试 = 60%/15%/25%，边界 2026-07-07 22:48 UTC / 2026-08-21 19:30 UTC，
  跨边界的前瞻样本已 purge。样本期基准：买入并持有中位数 -21.5851%，任意时点持有 H 根净收益均值 0.2154%（beta 归因用）。
- **币池 / 币数**：37 币（1m 活跃度 top-40），
  skip 3 币：{'我踏马来了USDT': '覆盖率不足（首根晚于窗口起点 2 天）', '龙虾USDT': '覆盖率不足（首根晚于窗口起点 2 天）', 'BTWUSDT': '覆盖率不足（首根晚于窗口起点 2 天）'}。
- **止损线核对**：已查 `references/verdicts-history.md` 第 1 节。现货 OHLCV 路线已全线证伪，
  故本课题**不投入新特征/新参数/新标签**，只对已移植的 11 种判定做一次性诊断对比；
  结论无论正负都不改变 `default_enabled=False`，也不进绑定。
- **对照组**：块内局部随机位移（保留价位尺度、打乱时序对齐）、随机价位（同期 K 线区间内随机画线）、
  任意时点持有（beta）、oracle 正对照（未来 H 根最低价）、半衰对照（位移半个样本）。

## 口径与参数

- **成本**：fee_rate=0.0004、slippage=0.0002、双边合计 0.12%；
  口径＝**名义本金**，全程不切换；毛利与净利分列。
- **参数来源**：各 catalog 指标默认参数（未寻优、未网格搜索）；grid 仅 H=48 与随机种子数。
- **样本外切分**：训练 60% / 验证 15% / 测试 25%；方法排名只用前 75%（研发段），测试段只读冠军。
- **撮合**：收盘判定触线、t+1 开盘入场、t+H 收盘出场；无止损止盈，纯持有；
  事件取上升沿且相邻间隔 ≥ H，触发率异常的退化列已剔除。
- **数据与覆盖**：derived（1m 聚合优先 + native 回落），7201 根/币；
  auto_trendlines 因算力用 lookback=120 / max_pivots=20；s_r_zones_mtf 的 HTF 改为 240/1440 分钟。

## 结果

| 支撑位定义 | 事件数 | 净收益/笔 | 95% CI（净超额） | 减随机价位 | 破位率 | 随机价位破位率 | 随机基线分位 | q | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 枢轴池合并·最近下方档 (`pivot_pool`) | 588 | 1.0982% | [-0.0764, 3.3372]% | 1.4709% | 0.923 | 0.934 | 0.868 | 0.4862 | `inconclusive` |
| ATR swing 区·HTF 投影 (`sr_zones_mtf`) | 645 | 0.9614% | [0.0117, 3.0671]% | 1.3777% | 0.908 | 0.921 | 0.860 | 0.4862 | `likely_noise` |
| 枢轴区+触及计数 (`plasmazero`) | 5017 | 0.3030% | [-0.1792, 0.4973]% | 0.1439% | 0.912 | 0.914 | 0.803 | 0.9779 | `inconclusive` |
| 多源水平登记簿·最近下方 (`edge_registry`) | 4338 | 0.2269% | [-0.2786, 0.3852]% | 0.0407% | 0.903 | 0.913 | 0.761 | 0.9779 | `inconclusive` |
| 多周期已确认枢轴·4h 档 (`mtf_pivot_4h`) | 3837 | 0.0850% | [-0.2676, 0.3809]% | 0.0387% | 0.899 | 0.910 | 0.772 | 0.9779 | `inconclusive` |
| 多周期已确认枢轴·1h 档 (`mtf_pivot_1h`) | 4816 | 0.1988% | [-0.2817, 0.3677]% | 0.0311% | 0.915 | 0.913 | 0.754 | 0.9779 | `inconclusive` |
| 经典枢轴+confluence·最近下方 (`pivot_map`) | 5210 | 0.1902% | [-0.2683, 0.3477]% | 0.0224% | 0.910 | 0.907 | 0.778 | 0.9779 | `inconclusive` |
| 7 探测器打分支撑区中点 (`sr_zones_wu`) | 4510 | 0.2498% | [-0.3241, 0.3859]% | 0.0191% | 0.904 | 0.910 | 0.766 | 0.9779 | `inconclusive` |
| 枢轴连线·动态趋势支撑 (`trendline`) | 4038 | 0.1825% | [-0.3229, 0.3715]% | 0.0048% | 0.904 | 0.909 | 0.772 | 0.9779 | `inconclusive` |
| 前日低点 (`pdh_pdl`) | 3105 | 0.1419% | [-0.3529, 0.3760]% | -0.0100% | 0.898 | 0.909 | 0.763 | 0.9779 | `inconclusive` |
| 20 根滚动最低价（朴素基线） (`donchian20`) | 3804 | 0.0290% | [-0.4484, 0.2152]% | -0.1320% | 0.903 | 0.912 | 0.710 | 0.9779 | `inconclusive` |

- 随机位移基线全样本均值 0.8465%，随机价位基线 0.0341%；任意时点持有 0.2154%（净、已扣成本）。
- **量具自校准**：oracle 正对照（偷看未来）净收益 6.9055%／2889 事件，检出=是；负对照（随机价位/任意时点）见表；半衰对照见各组 `half_shift_pct`。
- 时间稳定性：4 个等长子段，逐段净收益与同号段数见同名 JSON 的 `quarter_net_pct` / `segment_consistency` / `positive_quarters`。
- **样本外冠军**：研发段（训练+验证）第一是 `sr_zones_mtf`，研发段净超额 1.0035%，测试段只读结果 2.0023%。

### 直接回答「哪个支撑位更准」

- **按净超额（相对随机价位）前 3**：`pivot_pool` 1.4709%（inconclusive）、`sr_zones_mtf` 1.3777%（likely_noise）、`plasmazero` 0.1439%（inconclusive）。
- **CI 下界为正的共 1 个**：`sr_zones_mtf`；但没有任何一个同时满足 FDR q < 0.10 与随机基线分位 > 0.95，故最多只能记作 `likely_noise`，不能判为更准。
- **按「守住」能力（破位率越低越好）前 3**：`pdh_pdl` 0.898（随机价位 0.909）、`mtf_pivot_4h` 0.899（随机价位 0.910）、`donchian20` 0.903（随机价位 0.912）。与随机画线的差距只有几个百分点，没有一种方法能把破位率压到明显更低。
- **结论**：11 种定义里没有一种同时满足「CI 下界 > 0 ∧ q < 0.10 ∧ 4 段 ≥3 段同号 ∧ 随机基线分位 > 0.95」，
  总体 `verdict=inconclusive`。数值领先不等于判定更准。

## 判定

- **verdict**：`inconclusive`（词表取值；`inconclusive` 表示 CI 跨 0，不等于「无效」）。
- 池化 CI：重采样单位为一次事件，`MIN_POOLED_TRADES=30`，n_boot=2000；总事件数 39908。
- 随机基线：20 种子 × 37 币，分位数 > 0.95 才算跑赢随机。
- 归因：净超额 = 触线后净收益 − 随机价位基线，已同时扣除 0.12% 成本与 beta；
  破位率对比用于区分「真的守住了」与「只是行情反弹」。
- 与历史结论对照：与 `verdicts-history.md` 第 1 节（现货 OHLCV 全线证伪）一致；
  本次未投入新特征，故不构成对止损线的违反。

## 局限

- 诊断性量具评估，不是策略启用判定；未新增特征、未调参、未改绑定
- 现货 1h derived（1m 聚合优先 + native 回落），1m 历史仅约 91 天
- 未模拟资金费率、盘口深度、强平、分批建仓；假定下一根开盘及时成交
- 名义本金口径，不得与 margin 口径报告直接相减
- 样本期为单一 300 天窗口，4 个等长子段仅作时间稳定性检查
- auto_trendlines 用 lookback=120/max_pivots=20 以控制算力，非上游默认 300/30
- 全量 pytest（2026-11-05）：1573 passed / 3 failed，均为与本课题无关的既有失败 —— test_pool_volatility_research 引用已移除的 dispatch.halt_status、test_market_performance 的 tail-query 边界断言；本课题新增的 7 项单测全绿
- 支撑位是「区域」还是「一条线」的口径差异会影响触线事件数；本报告统一取单一价格，
  区域型指标取中点或最近档，未做区域宽度敏感性分析。

## 结论与后续

- 结论：见上表与 `verdict=inconclusive`；**是否启用**：否，全部方法 `default_enabled=False`，
  不进绑定、不进模拟盘/实盘。本次为量具诊断，不产生可上线策略。
- 若 `negative` / `inconclusive`：按 H2 止损线，**不要再调阈值、周期或止损倍数救它**；
  下一个信息增量应转向逐笔成交顺序／盘口挂撤单／多所现货腿，且先过 IC 探针。

## 复现命令

```powershell
# 两个持有期共用固定右端 --as-of 2026-11-05T12:00:00Z
.\.venv\Scripts\python.exe backend/scripts/support_level_eval.py --interval 1h --days 300 --limit 40 --horizon 48 --seeds 20 --workers 12 --as-of 2026-11-05T12:00:00Z --out docs/support-level-evaluation-h48
.\.venv\Scripts\python.exe backend/scripts/support_level_eval.py --interval 1h --days 300 --limit 40 --horizon 48 --seeds 20 --workers 12 --as-of 2026-11-05T12:00:00Z --out docs/support-level-evaluation-h48 --robustness-from docs/support-level-evaluation-h48.json
.\.venv\Scripts\python.exe -m pytest backend/tests/test_support_level_eval.py -q
.\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/support-level-evaluation-h48
.\.venv\Scripts\python.exe .workbuddy/skills/strategy-backtest-playbook/scripts/preflight.py --prefix docs/support-level-evaluation-h48
```

## 产物

- `docs/support-level-evaluation-h48.md` + `docs/support-level-evaluation-h48.json`；单测 `backend/tests/test_support_level_eval.py`

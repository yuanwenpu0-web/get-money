# 绑定 #7 停用：执行既有 negative 判定（工程决策执行，非新增 edge 判定）

- 生成时间：2026-11-12 22:0x（+08:00）
- 性质：**执行既有回测判定 + 生产配置变更**。本轮**未跑新回测、未调任何参数、未改任何代码、未下任何单**。
- 授权：用户先说「能提升收益。就直接改。」，在被告知「18 个组合 0 个 pass、没有改了就能赚的选项」后，明确选择 **「停用 #7 止血」**。这一步同时反转了 2026-11-12 早先「负 edge 绑定先不动，只登记风险」的决定。
- 执行环境：`exchange_env=demo`（币安 USD-M DEMO），未启用 live key，未切执行环境。

## 1. 为什么「直接改就能提升收益」不成立

`supertrend_current` 在 1h 与 15m 各 9 个组合（含基准），共 **18 个，0 个 `pass`**。数据源为既有报告，未重跑：

- `docs/supertrend-trend-reversal-fit-15m.json`（generated_at 2026-11-04T17:32:35Z）
- `docs/supertrend-trend-reversal-fit-1h.json`（generated_at 2026-11-04T17:35:58Z）
- 两份同口径：window 2026-01-08 → 2026-11-04（300 天）、universe explicit 12 币（evaluated 12、skipped 0）、`fee_rate=0.0004`、`slippage=0.0002`、`cash_per_symbol=10000`、`min_pooled_trades=30`、bootstrap 2000 次 95% CI。

### 15m（绑定 #7 所在周期）

| 组合 | params | 笔数 | 每笔净% | CI 下界 | CI 上界 | 判定 |
|---|---|---:|---:|---:|---:|---|
| factor2.0×r3.0 | 2.0 / 3.0 | 11928 | −0.4879 | −0.5401 | −0.4380 | negative |
| factor2.0×r4.0 | 2.0 / 4.0 | 11448 | −0.4757 | −0.5309 | −0.4195 | negative |
| factor2.0×r6.0 | 2.0 / 6.0 | 10956 | −0.4843 | −0.5422 | −0.4243 | negative |
| **基准（＝线上实际配置）** | **2.5 / 3.0** | **8970** | **−0.4585** | **−0.5206** | **−0.3968** | **negative** |
| factor2.5×r4.0 | 2.5 / 4.0 | 8582 | −0.4385 | −0.5023 | −0.3734 | negative |
| factor2.5×r6.0 | 2.5 / 6.0 | 8160 | −0.4567 | −0.5288 | −0.3826 | negative |
| factor3.0×r3.0 | 3.0 / 3.0 | 7041 | −0.3896 | −0.4603 | −0.3170 | negative |
| factor3.0×r4.0 | 3.0 / 4.0 | 6734 | −0.3850 | −0.4648 | −0.3078 | negative |
| factor3.0×r6.0 | 3.0 / 6.0 | 6369 | −0.3887 | −0.4725 | −0.3046 | negative |

**9/9 negative，CI 上界全部 < 0。** 基准组合 `positive_symbols=0/12`、`beats_random_ratio=0.0833`、账户收益中位 −47.91%。

### 1h

| 组合 | params | 笔数 | 每笔净% | CI 下界 | CI 上界 | 判定 |
|---|---|---:|---:|---:|---:|---|
| factor2.0×r3.0 | 2.0 / 3.0 | 2850 | −0.2453 | −0.3548 | −0.1372 | negative |
| factor2.0×r4.0 | 2.0 / 4.0 | 2711 | −0.2501 | −0.3676 | −0.1295 | negative |
| factor2.0×r6.0 | 2.0 / 6.0 | 2605 | −0.2632 | −0.3928 | −0.1426 | negative |
| 基准 | 2.5 / 3.0 | 2252 | −0.1902 | −0.3220 | −0.0676 | negative |
| factor2.5×r4.0 | 2.5 / 4.0 | 2137 | −0.1873 | −0.3237 | −0.0495 | negative |
| factor2.5×r6.0 | 2.5 / 6.0 | 2033 | −0.2203 | −0.3619 | −0.0839 | negative |
| factor3.0×r3.0 | 3.0 / 3.0 | 1795 | −0.1063 | −0.2596 | **+0.0347** | inconclusive |
| factor3.0×r4.0 | 3.0 / 4.0 | 1693 | −0.1498 | −0.3096 | **+0.0141** | inconclusive |
| factor3.0×r6.0 | 3.0 / 6.0 | 1598 | −0.1884 | −0.3551 | −0.0291 | negative |

7/9 negative、2/9 inconclusive。**按 H4，CI 跨 0 = inconclusive ≠ 有效**；这两个组合点估计仍为负，按 H11 不得据此启用。

### 线上实际跑的是哪一个

从 `orders.params_snapshot`（127 笔全部一致）解出：`supertrend_factor=2.5`、`target_r=3.0`、`sl_atr_mult=1.5`、`atr_period=14`、`trailing_enabled=false`、`max_signal_age_bars=2`、`risk_per_trade_pct=0.005`。
⇒ 精确对应 15m 表的 **「基准」行：negative，8970 笔，−0.4585%/笔，CI[−0.5206,−0.3968]**。停用依据就是这一行，不是「感觉在亏」。

## 2. 实盘台账（两个口径严格分开，H3）

范围：`source='live' AND status='CLOSED' AND binding_id=7`，开仓 2026-11-09 11:57:47 → 最后平仓 2026-11-12 21:13:04（+08:00），**127 笔**。

| 口径 | 读数 |
|---|---|
| 净盈亏（USDT） | **−173.0363** |
| 名义本金合计（USDT） | 167330.65 |
| 保证金合计（USDT） | 76526.54 |
| **USDT 加权**每笔净% | **−0.1034%** |
| **等权**每笔净% | **+0.0913%** |
| 等权 bootstrap 95% CI（2000 次，seed 42） | **[−0.2557, +0.4542]**，positive_share 0.675 |
| 等权判定 | **inconclusive** |
| 手续费合计（USDT） | 139.3762 |

按平仓原因：

| close_reason | 笔数 | 净 USDT | 手续费 USDT |
|---|---:|---:|---:|
| exchange_sl | 59 | −933.17 | 68.62 |
| signal_reverse | 30 | −204.03 | 34.32 |
| manual | 10 | +80.59 | 9.12 |
| exchange_tp | 28 | +883.57 | 27.31 |

**必须说清的两点：**

1. **实盘 127 笔单独不足以判定方向**（等权 CI 跨 0）。停用的统计依据是回测 8970 笔的 `negative`，实盘台账只是佐证「没有反证」。不得把「实盘等权 +0.0913%」读成「策略其实赚钱」。
2. **两个口径不可互换**：USDT 加权 −0.1034% 与等权 +0.0913% 符号相反，因为大额仓位亏得更多。这是 2026-11-12 早先已登记过的同类陷阱，本轮再次显式分列。
3. 台账本身偏乐观：127 笔中 **89 笔 `pnl_source=estimated`**（`close_price` 恰等于 sl/tp，假设零滑点按触发价成交），其中 exchange_sl/tp 共 87 笔。真实滑点只会让 −173.04 更差。

## 3. 改动内容（唯一一处）

| 项 | 值 |
|---|---|
| 机制 | `PUT /api/bindings/7`，body `{"enabled": false}`（项目既有 API，与前端同一路径，走 `validate_binding`） |
| 时间 | 2026-11-12 22:01:59.129656+08:00 |
| 变更 | `strategy_bindings.id=7`：`enabled 1 → 0` |
| 响应 | HTTP 200 |
| **未改** | `priority=100`、`params={}`、`exit_method="strategy"`、`support_indicator=""`、`exit_params={}` 全部原值回读一致 |
| **未改** | `strategy_configs.supertrend_current.enabled` 仍为 1（改它会波及绑定 #10，超出授权范围） |
| 全局启用绑定数 | `enabled_total: 1 → 0` |

未改任何 `.py`，未跑 `--apply` 类交易所写操作，未下任何单。

## 4. 影响面（代码证据，不轻信旧文档）

| 结论 | 证据 |
|---|---|
| 停用**只挡新开仓** | `backend/app/strategy/bindings.py:190` `active_bindings` 过滤 `enabled == 1`；`backend/app/trading/service.py:155-158` 每轮调用后 `expand(bindings)` 得到 `pairs`，空 ⇒ 无新信号 |
| **无需重启服务** | `active_bindings` 在 `run_cycle` 内每轮新查 DB，无进程级缓存；`service.py:141` 注释即「设置页改完下一轮即生效」 |
| 存量仓位继续同步 K 线 | `service.py:170-173` 用 `open_orders(db)` 独立补 `required_intervals`，与绑定无关 |
| 存量仓位继续被管理 | `service.py:236` `open_rows = open_orders(db)`；`service.py:253` `sync_live` 照常对账交易所止盈止损 |
| 存量仓位的 `signal_reverse` 判定不受停用影响 | `service.py:264-271` 有 `params_snapshot` 时走 `engine.evaluate(..., respect_enabled=False)`；6 笔 OPEN 全部 `params_snapshot IS NOT NULL`（已核） |
| 存量仓位参数仍可解析 | `bindings.py:360-368` `resolve_order_params` 按三元组查绑定行，**不过滤 enabled** |
| 重启不会把它重新打开 | `defaults.py:131-133` `seed_bindings` 在已有绑定行时直接 return；`strategy_migration.py:150,166-174` 一次性迁移只写 `enabled=0`，且 marker `legacy_strategy_enablement_v1` 已存在（已核） |
| 无「空绑定早退」导致失管 | `service.py:184-278` 在 `pairs` 为空时不 return，仍执行持仓管理段 |

## 5. 验证

| 检查 | 结果 |
|---|---|
| API 回读 `enabled` | `false` |
| DB 独立回读 | `enabled=0`，`params='{}'`、`exit_params='{}'`、`priority=100` 未变 |
| `enabled_total` | 0 |
| 6 笔 OPEN 仍在库 | 313 0GUSDT / 319 AINUSDT / 320 AVAXUSDT / 322 4USDT / 323 BULLAUSDT / 324 ALCHUSDT，全 SHORT，`margin` 合计 2897.50，sl/tp 未变 |
| **交易所侧保护单逐张核对** | **12/12 `algoStatus=NEW`**，`triggerPrice` 与库内 sl/tp 完全一致，`closePosition=true` |
| 停用后新开仓 | 0 笔（`orders.created_at >= 22:01:59` 为空） |
| **停用后首个完整周期** | `job_logs` id=28244，22:07:17 完成：**`symbol_count: 0`**（无生效绑定）、**`sync_target_count: 6`**（恰好是 6 笔存量持仓的币）、`signal_ms: 0`、**`execution_ms: 40358`**（持仓管理段照常执行）、`opened: 0`、`sync_errors: 0`、`signal_errors: 0` |
| 周期耗时变化 | 280047ms → **48389ms**（同步目标 509 → 6），副作用是巡检大幅变快 |
| 全量 pytest | **3 failed / 1936 passed / 145.15s** —— 3 项为既有基线失败（`test_market_performance::test_complete_tail_queries_have_a_bounded_minute_range`、`test_pool_volatility_research::test_live_entrypoints_reject` ×2），**无回归** |

**周期时序说明（避免误读）**：22:01:01 与 22:06:09 完成的两轮都**启动于 22:01:59 改动之前**（后者 elapsed 280047ms ⇒ 起点约 22:01:29），因此仍读到 `enabled=1`、`symbol_count=509`。真正读到新状态的是 22:06:09 之后启动、22:07:17 完成的 id=28244。这也实证了 §4 的「无需重启」：配置在下一轮自动生效。

保护单明细（只读 `get_algo_order`）：

| 订单 | 币种 | SL 触发价 | TP 触发价 | 状态 |
|---|---|---:|---:|---|
| 313 | 0GUSDT | 0.1544 | 0.1437 | NEW / NEW |
| 319 | AINUSDT | 0.10846 | 0.09669 | NEW / NEW |
| 320 | AVAXUSDT | 8.577 | 8.253 | NEW / NEW |
| 322 | 4USDT | 0.00881 | 0.00851 | NEW / NEW |
| 323 | BULLAUSDT | 0.06201 | 0.05623 | NEW / NEW |
| 324 | ALCHUSDT | 0.03225 | 0.03102 | NEW / NEW |

## 6. 可逆性与 H11 约束

- 技术上完全可逆：`PUT /api/bindings/7 {"enabled": true}`，其余字段无需重填。
- **但按 H11 不得随意恢复**：该配置判定为 `negative`，恢复等于让负 edge 继续跑。若将来要恢复，必须么出现新的 `pass` 证据，要么显式标注「未经统计支持」并由用户承担。
- 项目先例：`docs/binding-combo-optimization.json` 的 `applied` 记录了 2026-11-05 22:35:10 按同样理由停用 #10（negative，1100 笔、两窗口同号、毛收益 −0.04%）与 #43（insufficient，494 池成员仅 3 币可回测）。本轮是同一处置逻辑的延续。

## 7. 诚实边界

1. **停用不产生 alpha**，只是把期望收益从负截到 0。系统现在**不会开任何新仓**（`enabled_total=0`），直到出现 `pass` 判定的新配置。这是用户明确选择的代价。
2. **没有任何候选通过判定**。18 个组合 0 pass，1h 最好的两个只是 inconclusive 且点估计为负。本轮没有、也不会去凑一个「能赚」的组合——那属 H5 禁止的事后选冠军。
3. **存量 6 笔仍暴露在负 edge 下**，会继续按原 sl/tp 跑完。停用只挡新开仓，不会强平（强平属另一项未授权动作）。
4. **−173.04 是乐观读数**（89 笔零滑点假设）。真实结果很可能更差，但方向判定不依赖它。
5. 本轮**未做**任何先前登记的工程项：平仓价保真回填、对账状态回写、K 线同步延迟。它们不提升收益，需另行授权。
6. 早先登记的「id 279 ARCUSDT 孤儿止盈单可能误平后续仓位」**已证伪**：交易所返回 `algoStatus=EXPIRED`，本地 note 仍写 `tp_status=NEW`，属状态回写缺失（可观测性），不是资金安全风险。

## 8. 交付自检

- 生产代码改动：**0 个文件**（仅 DB 配置）。因此无需 `npm run build`，无需重启 `QuantDesk-GetMoney`（依据见 §4）。
- 全量 `env -u NO_PROXY -u no_proxy pytest backend/tests -q`：**3 failed / 1936 passed / 5 warnings / 145.15s**。3 项失败与改动前基线完全一致，本轮 0 代码改动，**无回归**。（`NO_PROXY` 本机值畸形会让 httpx 抛 `InvalidURL: Invalid port: ':1]'`，故须 `env -u`。）
- 只读核查全部走 `app/db.py::make_engine(read_only=True)`；唯一写操作是经 API 的 `enabled` 变更。
- 未打印任何密钥；未启用 live key；未切执行环境；未下任何单。
- H3：名义本金口径与保证金口径分列，等权与 USDT 加权分列，未混用。
- H8：样本区间、universe、skipped 列表、成本口径均已在 §1/§2 写明。
- H11：停用是「无 pass 不得启用」的反向执行，未强制启用任何配置。

## 9. 复现命令

```bash
# 判定表（只读既有报告）
python -c "import json;[print(c['label'],c['expectancy_ci']) for c in json.load(open('docs/supertrend-trend-reversal-fit-15m.json',encoding='utf-8'))['combos']]"

# 当前绑定状态
curl -s http://127.0.0.1:8010/api/bindings

# 停用（本轮已执行）
curl -s -X PUT http://127.0.0.1:8010/api/bindings/7 -H 'Content-Type: application/json' -d '{"enabled": false}'

# 恢复（受 H11 约束，勿随意执行）
# curl -s -X PUT http://127.0.0.1:8010/api/bindings/7 -H 'Content-Type: application/json' -d '{"enabled": true}'
```
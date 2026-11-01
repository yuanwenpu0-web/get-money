# 最佳策略组合与参数搜索（结论：当前指标库没有可上线的正 edge）

> 生成时间：2026-11-01；代码：`backend/app/research/rolling_portfolio.py`、
> `backend/scripts/combo_search.py`、`backend/scripts/correlation_check.py`

## 1. 这次回答的问题

「在有指标、有策略、有 K 线的情况下，找出最佳策略组合与参数」。
做法是**不做历史最优排名**，而是把候选放进**滚动样本外 + 共享资金池**里检验：
每折只用折前历史选规则，在紧随其后的未参与行情中判定。

## 2. 口径（与既有报告的关键差异）

| 项 | 取值 | 说明 |
|---|---|---|
| 数据 | 1h native，10 币，94720 根 bar | 库里 4h 为 **0 行**、15m 仅 2 币够 200 天，多份历史报告声称的「4h/99 币」**当前无法复现** |
| 成本 | fee 0.04%/边 + 滑点 0.02%/边 | 与实盘全局设置同源 |
| 资金 | 总资金 10000，最多 5 条腿并发 | **新增**：单笔按 ATR 风险预算定仓（0.5% × 资金 / (2×ATR%)） |
| 出场 | 统一 rsi50_cross | 与 `indicator_lab` 一致，保证可比 |
| 判定 | 每笔期望 bootstrap 95% CI | CI 跨 0 即 `inconclusive`，不写成「有效」 |

## 3. 结果

只做多（long）口径，4 折滚动：

| 折 | 样本外收益 | 回撤 | 同期买入持有 | 超额 |
|---|---:|---:|---:|---:|
| 1 | +1.64% | 9.60% | −27.34% | +28.98pp |
| 2 | −11.35% | 27.99% | −19.97% | +8.62pp |
| 3 | +7.66% | 1.82% | +204.29% | −196.63pp |
| 4 | +58.64% | 7.99% | +116.57% | −57.93pp |

- **样本外每笔期望 +0.125%，95% CI [−0.0099%, +0.2978%] → `inconclusive`（451 笔）**
- 多空对称口径明显更差：4 折为 −24.4% / −17.6% / −30.3% / +10.2%，回撤 32%~60%
- 冗余度检查：Top3 入场数之和 1213，去重后 1065，**冗余仅 12.2%** → 组合确有分散，不是同一笔交易重复计数

## 4. 结论

1. **没有找到「赚大钱」的组合**：最好一档的每笔期望 CI 仍跨 0，统计上无法判定为正。
2. **只做多显著优于多空对称**，这与 `docs/indicator-lab-long.md` 的方向一致；
   但在牛市段（折 3/4）大幅跑输买入持有，**正收益主要来自 beta 而非选股能力**。
3. **参数选择不稳定**：4 折都在 `rsi_reversal` / `mfi_reversal` / `volume_break`
   之间漂移，且训练段最优在样本外常转负——这是过拟合的典型特征，不是「最佳参数」。
4. 与项目既有结论自洽：P0–P10、`signal-screen`（132 个条件 0 个过关）、
   `optimization-all-symbols`（pass 0）都指向同一事实——
   **1h 现货 OHLCV 上不存在可榨取的方向性 alpha**。

## 5. 复现

```powershell
# 只做多 + 风险预算 + 5 腿并发，4 折滚动
.\.venv\Scripts\python.exe backend/scripts/combo_search.py --interval 1h --days 400 --limit 10 `
  --folds 4 --train-ratio 0.5 --top-k 3 --side long --max-concurrent 5 `
  --risk-per-trade 0.005 --atr-mult 2.0 --out-prefix docs/combo-search-long

# 候选冗余度
.\.venv\Scripts\python.exe backend/scripts/correlation_check.py --interval 1h --days 400 --limit 10 --side long
```

## 6. 限制

- 10 个币、约 13 个月，跨不过完整牛熊；结论只在样本期内成立
- 未计永续资金费率、强平、盘口冲击
- 风险预算按信号期 ATR 中位数静态估算，未按逐笔实际止损动态缩放
- 买入持有对照为「折内一直持多」，与策略曝光时段不完全一致，仅作量级参照

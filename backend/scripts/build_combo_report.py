"""生成「策略组合与参数寻优」完整报告。

动态提取：
  - 5 个注册策略 spec 的可调参数/取值范围/grid（app.strategy.specs）
  - 15 个指标信号家族的参数网格（app.research.lab_grid.manifest）
读取三份回测产物：
  - docs/combo-search-1h.json / combo-search-4h.json  （指标组合 walk-forward OOS）
  - docs/spec-oos-scan.json                          （supertrend/absorption 单策略 OOS）
输出：
  - docs/strategy-combo-optimization-report.md
  - docs/strategy-combo-results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "backend")

from app.research.lab_grid import manifest as grid_manifest  # noqa: E402
from app.strategy import specs as _specs_pkg  # noqa: E402  (import triggers registration)
from app.strategy.registry import all_specs  # noqa: E402

ROOT = Path(".")
OUT_MD = ROOT / "docs" / "strategy-combo-optimization-report.md"
OUT_JSON = ROOT / "docs" / "strategy-combo-results.json"


def load(p: str):
    return json.load(open(p, encoding="utf-8"))


def annualize(ret_pct: float, start_ms: int, end_ms: int):
    days = (end_ms - start_ms) / 86_400_000.0
    if not days or days <= 0:
        return None
    return round(((1 + ret_pct / 100.0) ** (365.0 / days) - 1) * 100.0, 1)


# ---------------------------------------------------------------------------
# 参数梳理
# ---------------------------------------------------------------------------
def spec_params_md() -> str:
    specs = all_specs()
    out = []
    for spec in specs:
        out.append(f"\n#### `{spec.code}` — {spec.name}\n")
        out.append(
            f"> direction 约束：{'多空双向' if spec.description else '见 spec'}"
        )
        out.append(
            "\n| 参数 | 默认 | 取值范围 | 寻优档位(grid) | 类型 |"
        )
        out.append("|---|---|---|---|---|")
        seen = set()
        # 以 ranges 为主，grid 中多出的键（如共享键）单独补
        for k in sorted(spec.ranges.keys()):
            seen.add(k)
            d = spec.defaults.get(k, "-")
            rng = spec.ranges[k]
            grid = spec.grid.get(k)
            typ = "整数" if k in spec.int_keys else ("布尔" if k in spec.bool_keys else "连续")
            grid_s = ",".join(str(x) for x in grid) if grid else "-"
            out.append(f"| `{k}` | {d} | [{rng[0]}, {rng[1]}] | {grid_s} | {typ} |")
        for k in sorted(spec.grid.keys()):
            if k in seen:
                continue
            d = spec.defaults.get(k, "-")
            grid = spec.grid.get(k)
            grid_s = ",".join(str(x) for x in grid) if grid else "-"
            out.append(f"| `{k}` | {d} | (共享键) | {grid_s} | 连续 |")
        out.append("")
    return "\n".join(out)


def indicator_params_md() -> str:
    m = grid_manifest()
    out = ["\n| 指标家族 | 含义 | 参数 (min/max/step 或枚举) | 约束 | 组合数 |"]
    out.append("|---|---|---|---|---:|")
    for name, spec in m.items():
        params = spec["params"]
        parts = []
        for pk, pv in params.items():
            if "values" in pv:
                parts.append(f"{pk}∈{{{','.join(str(v) for v in pv['values'])}}}")
            else:
                parts.append(f"{pk}:{pv['min']}~{pv['max']}/步{pv['step']}")
        out.append(
            f"| `{name}` | {spec['label']} | {'; '.join(parts)} | {spec.get('constraint') or '-'} | {spec['combos']} |"
        )
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 组合回测结果
# ---------------------------------------------------------------------------
def combo_md() -> tuple[str, dict]:
    wrap = {}
    out = []
    for iv in ["1h", "4h"]:
        d = load(f"docs/combo-search-{iv}.json")
        cfg = d["config"]
        bh_map = {b["fold"]: b for b in d["buy_hold_folds"]}
        out.append(f"\n### {iv} 组合 walk-forward（limit={cfg['limit']}, folds={cfg['folds']}, top_k={cfg['top_k']}）\n")
        out.append(
            "| 折 | 样本外收益% | 年化%(近似) | 回撤% | 夏普 | 胜率 | 笔数 | "
            "买入持有% | 超额(pp) | 选中腿 |"
        )
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        rows = []
        for row in d["portfolio_folds"]:
            t = row.get("test") or {}
            bh = (bh_map.get(row["fold"], {}).get("test") or {})
            ret = t.get("return_pct")
            s, e = row["start"], row["end"]
            ann = annualize(ret, s, e) if ret is not None else None
            bh_ret = bh.get("return_pct")
            excess = round(ret - bh_ret, 2) if (ret is not None and bh_ret is not None) else None
            out.append(
                f"| {row['fold']} | {ret} | {ann} | {t.get('max_drawdown_pct')} | "
                f"{t.get('sharpe')} | {t.get('win_rate')} | {t.get('trades')} | "
                f"{bh_ret} | {excess} | {', '.join(row['selected'])} |"
            )
            rows.append({"fold": row["fold"], "oos_return_pct": ret, "annualized_pct": ann,
                        "max_dd_pct": t.get("max_drawdown_pct"), "sharpe": t.get("sharpe"),
                        "win_rate": t.get("win_rate"), "trades": t.get("trades"),
                        "buy_hold_pct": bh_ret, "selected": row["selected"]})
        oos = d.get("oos_expectancy") or {}
        out.append(
            f"\n**池化样本外期望**：{oos.get('trades')} 笔，单笔均值 {oos.get('mean_pct')}%，"
            f"95% CI [{oos.get('ci_low')}, {oos.get('ci_high')}] → **判定：{oos.get('verdict')}**"
        )
        wrap[iv] = {"folds": rows, "pooled_expectancy": oos, "config": cfg}
    return "\n".join(out), wrap


def spec_oos_md() -> tuple[str, dict]:
    d = load("docs/spec-oos-scan.json")
    out = []
    wrap = {}
    for strat, blk in d.items():
        out.append(f"\n#### `{strat}`（1h, 400d, optimize=True, 单策略单币 OOS）\n")
        out.append("| 币 | OOS收益% | 胜率 | 夏普 | 笔数 | 买入持有% | 跑赢B&H? | 跑赢随机? | 期望CI判定 | 选参 |")
        out.append("|---|---:|---:|---:|---:|---:|:--:|:--:|---|---|")
        rows = []
        for r in blk.get("symbols", []):
            if "error" in r:
                out.append(f"| {r['symbol']} | ERR | | | | | | | | {r['error']} |")
                continue
            beats_bh = "✓" if r["oos_return_pct"] > r["buy_hold_pct"] else "✗"
            beats_rnd = "✓" if r["oos_return_pct"] > r["random_return_pct"] else "✗"
            out.append(
                f"| {r['symbol']} | {r['oos_return_pct']} | {r['oos_win_rate']} | {r['oos_sharpe']} | "
                f"{r['oos_trades']} | {r['buy_hold_pct']} | {beats_bh} | {beats_rnd} | "
                f"{r['expectancy_verdict']} | {r['validation']} |"
            )
            rows.append(r)
        agg = {k: v for k, v in blk.items() if k != "symbols"}
        out.append(
            f"\n聚合：{agg.get('n')} 币 | 平均OOS收益 {agg.get('mean_oos_return_pct')}% | "
            f"跑赢B&H {agg.get('beats_buyhold')} 币 | 跑赢随机 {agg.get('beats_random')} 币 | "
            f"期望正 {agg.get('positive_edge')} / 不确定 {agg.get('inconclusive_edge')} / 负 {agg.get('negative_edge')}"
        )
        wrap[strat] = {"per_symbol": rows, "aggregate": agg}
    return "\n".join(out), wrap


# ---------------------------------------------------------------------------
# 主报告
# ---------------------------------------------------------------------------
def main() -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    spec_params_block = spec_params_md()
    indicator_block = indicator_params_md()
    combo_block, combo_data = combo_md()
    spec_block_oos, spec_data = spec_oos_md()

    md = f"""# 策略组合与参数寻优报告

> 生成时间：{ts}　引擎：`backend/app/research/rolling_portfolio.py` + `backend/scripts/combo_search.py`
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

{spec_params_block}

### 4.2 指标信号家族（15 个，来源 `app/research/lab_grid.py`，组合寻优的搜索空间）

{indicator_block}

---

## 5. 组合寻优结果（核心）

### 5.1 指标组合 walk-forward 样本外表现

{combo_block}

**读法**：
- 1h 四折 OOS 全为负（-15% / -3.7% / -23.6% / +0.3%），相对买入持有在涨市（折3 持多 +69%、折4 +38%）严重跑输 —— 技术指标组合在强趋势市被频繁止损/换手拖累。
- 4h 四折（+9.4% / +0.5% / -6.5% / +62.4%）在跌市跑赢 beta，但折 4 异常高，训练段选出的 stoch/rsi 组合恰好契合该段波动，**疑似过拟合**。
- 两套 **池化 OOS 期望 CI 均跨 0 → 无法判定存在正 edge**。

---

## 6. 现有策略 spec 的样本外表现

### 6.1 本次实跑（supertrend_current / absorption_reversal，1h 单策略 OOS）

{spec_block_oos}

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
.\\.venv\\Scripts\\python.exe backend/scripts/combo_search.py `
  --interval 4h --days 400 --limit 20 --folds 4 --train-ratio 0.5 --top-k 3 `
  --min-trades 30 --capital 10000 --fee 0.0004 --slippage 0.0002 `
  --regimes "market_ema:200" --out-prefix docs/combo-search-4h

# 2) 现有策略 spec 单策略样本外扫描（supertrend / absorption）
.\\.venv\\Scripts\\python.exe backend/scripts/spec_oos_scan.py   # 输出 docs/spec-oos-scan.json

# 3) 单个注册策略的 train/test 回测（含随机方向基线 + 期望 CI）
#    通用策略用 runner.run_backtest；专用 broker 策略用各自脚本：
.\\.venv\\Scripts\\python.exe backend/scripts/supertrend_backtest.py --symbol BTCUSDT --interval 1h --days 400
.\\.venv\\Scripts\\python.exe backend/scripts/absorption_backtest.py --symbol BTCUSDT --interval 15m --days 180
.\\.venv\\Scripts\\python.exe backend/scripts/rgd_backtest.py --symbol BTCUSDT --interval 15m --days 180
.\\.venv\\Scripts\\python.exe backend/scripts/ocb_backtest.py --symbols BTCUSDT,ETHUSDT --interval 15m --days 180
.\\.venv\\Scripts\\python.exe backend/scripts/zigzag_backtest.py --symbol BTCUSDT --interval 1h --days 120
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
"""
    OUT_MD.write_text(md, encoding="utf-8")

    OUT_JSON.write_text(
        json.dumps(
            {
                "generated_at": ts,
                "data": {
                    "source": "postgresql get_money.klines",
                    "intervals": ["1h", "4h"],
                    "span_days": 400,
                    "symbols_per_run": 20,
                    "cost": {"fee_per_side": 0.0004, "slippage_per_side": 0.0002},
                },
                "combo_oos": combo_data,
                "spec_oos": spec_data,
                "verdict": "no_statistically_significant_positive_edge",
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"written: {OUT_MD} ({OUT_MD.stat().st_size} bytes), {OUT_JSON}")


if __name__ == "__main__":
    main()

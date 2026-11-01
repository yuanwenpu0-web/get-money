"""币种分类维度的判别力检验（研究脚本，只读，不改任何策略状态）。

问题：哪些「币种分类维度」真的对量化有帮助？

判断标准不是「维度看起来合理」，而是**用训练期算出的指标，能否预测样本外的
真实结果**。这里把「结果」拆成三类，分别检验，因为它们的用途完全不同：

| 用途 | 预测目标 | 用在哪 |
| --- | --- | --- |
| 收益 | 样本外每笔净收益中位数 | 选币、选策略族 |
| 风险 | 样本外每笔收益的**离散度**与最大回撤 | 仓位大小、止损距离 |
| 尾部 | 样本外最差 5% 单笔亏损 | 爆仓风险、单币风险预算 |

一个维度可能完全预测不了收益，却非常准确地预测风险 —— 后者同样有价值。

流程严格分两段，防前视：
1. 训练期（每个窗口的前 60%）：只用这段算画像指标（`coin_taxonomy.profile`）。
2. 样本外（该窗口后 40%）：固定参数跑 15 个策略构造器，得到每币净收益序列。

再用三种相关口径层层剥离混杂（`ic` 原始 → `pic` 控流动性 → `picv` 再控波动率），
并做 **4 个滚动窗口** 重复，只有跨窗口方向一致的维度才算稳定，
避免把一次切分的运气当成规律。

`picv` 是判定口径。波动率类指标预测「收益离散度」的原始偏 IC 能到 0.5，
但那是定义式恒等（波动率预测波动率），剥离后归零的会被单独标为「伪发现」。

口径：
- 每笔固定名义 100 USDT，fee 0.04%、滑点 0.02%（双边 0.12%）。
- 出场统一 RSI(14) 穿 50，最长持有 240 根。
- 策略参数写死（不寻优），避免参数搜索把噪声拟合进去。

运行：`.venv/Scripts/python.exe backend/scripts/coin_taxonomy_study.py`
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import make_engine
from app.research import lab_signals as signals
from app.research import lab_sim as sim
from app.research.coin_taxonomy import cross_section, dimension_keys, profile

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "docs" / "coin-taxonomy.json"
OUT_MD = ROOT / "docs" / "coin-taxonomy.md"

SPLIT_FRACTION = 0.60
WARMUP_BARS = 240
MIN_TRAIN_BARS = 800
STAKE = 100.0
N_BOOTSTRAP = 2000
N_RANDOM_BASELINE = 200
SEED = 20261101

# 滚动窗口：每个窗口内再按 60/40 切训练/样本外。窗口长度 180 天、步长 60 天。
WINDOW_DAYS = 180
WINDOW_STEP_DAYS = 60

# 五个预测目标。风险类目标必须配 `picv_`（再控波动率）才能区分「真的另有信息」与
# 「波动率预测波动率」这种定义式恒等。
TARGETS = ("return", "sharpe", "risk_std", "risk_worst5", "risk_mdd")
TARGET_TITLES = {
    "return": "每笔净收益中位数",
    "sharpe": "每笔收益/风险（mean÷std）",
    "risk_std": "每笔收益离散度",
    "risk_worst5": "最差 5% 单笔",
    "risk_mdd": "已实现权益最大回撤",
}
# 二次偏相关的控制变量：小时实现波动。用它代表「这个币有多波动」。
VOL_CONTROL_KEY = "vol_hourly_std"

# 固定参数，不参与搜索。
STRATEGY_PARAMS: dict[str, dict] = {
    "ema_cross": {"fast": 12, "slow": 26},
    "price_sma": {"n": 50},
    "rsi_reversal": {"n": 14, "high": 70, "low": 30},
    "macd_cross": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger_break": {"n": 20, "k": 2.0},
    "donchian_break": {"n": 20},
    "supertrend_flip": {"n": 10, "mult": 3.0},
    "adx_di_cross": {"n": 14, "adx_min": 20},
    "stoch_cross": {"n": 14, "smooth": 3},
    "cci_reversal": {"n": 20, "level": 100},
    "mfi_reversal": {"n": 14, "low": 20},
    "volume_break": {"n": 20, "k": 2.0},
    "atr_channel_break": {"n": 20, "k": 2.0},
    "roc_momentum": {"n": 24, "thr": 0.02},
    "vwap_reclaim": {"n": 20},
}

STRATEGY_FAMILY = {
    "ema_cross": "trend", "price_sma": "trend", "macd_cross": "trend",
    "donchian_break": "trend", "supertrend_flip": "trend", "adx_di_cross": "trend",
    "roc_momentum": "trend", "volume_break": "trend", "atr_channel_break": "trend",
    "vwap_reclaim": "trend",
    "rsi_reversal": "reversion", "bollinger_break": "reversion",
    "stoch_cross": "reversion", "cci_reversal": "reversion", "mfi_reversal": "reversion",
}

DIMENSION_TITLES = {
    "liq": "流动性", "vol": "波动率", "trd": "趋势/均值回归倾向",
    "tail": "尾部与跳跃", "life": "生命周期与微观结构", "hist": "历史表现",
}

QUERY = text(
    """
    select symbol, open_time, open, high, low, close, volume, quote_volume, trades
    from klines
    where interval = '1h'
    order by symbol, open_time
    """
)


# ---------------------------------------------------------------- 数据加载


def load_frames() -> tuple[dict[str, pd.DataFrame], int, int]:
    """一次全量加载，按币种切帧；返回 (帧字典, 全局起点, 全局终点) 毫秒。"""
    engine = make_engine(read_only=True)
    with engine.connect() as conn:
        raw = pd.read_sql(QUERY, conn)
    start_ms = int(raw["open_time"].min())
    end_ms = int(raw["open_time"].max())
    frames: dict[str, pd.DataFrame] = {}
    for symbol, group in raw.groupby("symbol", sort=False):
        frame = group.drop(columns=["symbol", "open_time"]).copy()
        frame.index = pd.to_datetime(group["open_time"].to_numpy(), unit="ms", utc=True)
        frames[str(symbol)] = frame.sort_index()
    return frames, start_ms, end_ms


def rolling_windows(start_ms: int, end_ms: int) -> list[dict]:
    """生成滚动窗口；窗口必须完整落在数据范围内。返回空列表表示只能用全样本。"""
    span_days = (end_ms - start_ms) / 86_400_000
    if span_days < WINDOW_DAYS + WINDOW_STEP_DAYS:
        return []
    windows = []
    offset = 0
    while True:
        win_start = start_ms + offset * 86_400_000
        win_end = win_start + WINDOW_DAYS * 86_400_000
        if win_end > end_ms:
            break
        split = int(win_start + (win_end - win_start) * SPLIT_FRACTION)
        windows.append({
            "label": f"W{len(windows)}",
            "start": win_start,
            "split": split,
            "end": win_end,
        })
        offset += WINDOW_STEP_DAYS
    return windows


# ---------------------------------------------------------------- 指标与收益


def train_profiles(frames: dict[str, pd.DataFrame], start_ms: int, split_ms: int) -> dict[str, dict]:
    """训练期画像：严格只用 [start, split) 的 bar。"""
    lo = pd.to_datetime(start_ms, unit="ms", utc=True)
    hi = pd.to_datetime(split_ms, unit="ms", utc=True)
    out: dict[str, dict[str, float]] = {}
    for symbol, frame in frames.items():
        train = frame[(frame.index >= lo) & (frame.index < hi)]
        if len(train) < MIN_TRAIN_BARS:
            continue
        row = profile(train)
        if row:
            out[symbol] = row
    return out


def oos_returns(
    frames: dict[str, pd.DataFrame], split_ms: int, end_ms: int
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict]]:
    """样本外每币每策略的逐笔净收益（%）。

    返回 (逐笔收益, 汇总指标)。汇总里除了均值还带风险口径：
    `std`（每笔收益离散度）、`worst5`（最差 5% 单笔）、`mdd`（已实现权益最大回撤）。
    """
    cfg = sim.SimConfig(stake=STAKE)
    lo = pd.to_datetime(split_ms, unit="ms", utc=True)
    hi = pd.to_datetime(end_ms, unit="ms", utc=True)

    per_trade: dict[str, dict[str, list[float]]] = {}
    summary: dict[str, dict] = {}

    for symbol, frame in frames.items():
        pre = frame[frame.index < lo]
        post = frame[(frame.index >= lo) & (frame.index < hi)]
        if len(post) < 100 or len(pre) < 1:
            continue
        merged = pd.concat([pre.tail(WARMUP_BARS), post])
        exits = sim.exit_signals(merged["close"].to_numpy(dtype=float))
        trades_out: dict[str, list[float]] = {}
        for name, builder in signals.BUILDERS.items():
            try:
                sig = builder(merged, STRATEGY_PARAMS[name])
            except (KeyError, ValueError):
                continue
            trades = [t for t in sim.simulate(symbol, merged, sig, exits, cfg) if t.entry_ts >= split_ms]
            if not trades:
                continue
            values = [float(t.net_pct) * 100.0 for t in trades]
            trades_out[name] = values
        if not trades_out:
            continue

        all_values = np.array([v for vals in trades_out.values() for v in vals], dtype=float)
        per_trade[symbol] = trades_out
        summary[symbol] = {
            "trades": int(all_values.size),
            "mean": float(np.mean(all_values)),
            "std": float(np.std(all_values, ddof=1)) if all_values.size > 1 else float("nan"),
            "worst5": float(np.quantile(all_values, 0.05)),
            "mdd": _realized_mdd(trades_out),
            "win": float(np.mean(all_values > 0)),
        }
    return per_trade, summary


def _realized_mdd(trades_out: dict[str, list[float]]) -> float:
    """把所有策略的逐笔收益按序累计成权益曲线，取最大回撤（正数 %）。

    这是「已实现」口径，不含浮亏；只用来做币种间风险相对排序，不代表组合回撤。
    """
    flat = np.concatenate([np.asarray(v, dtype=float) for v in trades_out.values()])
    if flat.size < 2:
        return float("nan")
    equity = np.cumsum(flat)
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


# ---------------------------------------------------------------- 统计工具


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 8:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """控制变量后的偏 Spearman：x、y 各对 control 做秩回归取残差，再算 Spearman。"""
    return partial_spearman_multi(x, y, [control])


def partial_spearman_multi(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    """同时控制多个变量的偏 Spearman。controls 里任一含 NaN 的样本会被剔除。"""
    mask = np.isfinite(x) & np.isfinite(y)
    for control in controls:
        mask &= np.isfinite(control)
    if mask.sum() < 12:
        return float("nan")
    xs, ys = stats.rankdata(x[mask]), stats.rankdata(y[mask])
    columns = [np.ones(int(mask.sum()))]
    for control in controls:
        columns.append(stats.rankdata(control[mask]))
    design = np.column_stack(columns)
    rx = xs - design @ np.linalg.lstsq(design, xs, rcond=None)[0]
    ry = ys - design @ np.linalg.lstsq(design, ys, rcond=None)[0]
    if np.std(rx) <= 0 or np.std(ry) <= 0:
        return float("nan")
    return float(stats.spearmanr(rx, ry)[0])


def bucket_stats(values: pd.Series, target: pd.Series, rng: np.random.Generator) -> dict:
    """三分档统计：每档均值、档内样本数、高档减低档差与其 bootstrap CI。"""
    ranks = values.rank(pct=True, method="average")
    buckets = pd.Series(
        np.where(ranks <= 1 / 3, 0, np.where(ranks <= 2 / 3, 1, 2)),
        index=values.index, dtype="float",
    ).where(values.notna())

    bucket_mean, bucket_n = {}, {}
    for level in (0, 1, 2):
        members = target[buckets == level].to_numpy(dtype=float)
        finite = members[np.isfinite(members)]
        bucket_mean[f"b{level}"] = float(finite.mean()) if finite.size else float("nan")
        bucket_n[f"b{level}"] = int(finite.size)

    low = target[buckets == 0].to_numpy(dtype=float)
    high = target[buckets == 2].to_numpy(dtype=float)
    low, high = low[np.isfinite(low)], high[np.isfinite(high)]
    spread = float(high.mean() - low.mean()) if low.size and high.size else float("nan")

    if low.size >= 4 and high.size >= 4:
        li = rng.integers(0, low.size, size=(N_BOOTSTRAP, low.size))
        hi = rng.integers(0, high.size, size=(N_BOOTSTRAP, high.size))
        diffs = high[hi].mean(axis=1) - low[li].mean(axis=1)
        ci = (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)))
    else:
        ci = (float("nan"), float("nan"))

    return {"bucket_mean": bucket_mean, "bucket_n": bucket_n, "spread": spread, "spread_ci": ci}


def random_baseline(target: pd.Series, rng: np.random.Generator, repeats: int = N_RANDOM_BASELINE) -> float:
    """零分布：随机三分组后 |档差| 的 95 分位。真实档差超过它才算「比随便分组强」。"""
    arr = target.dropna().to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 12:
        return float("nan")
    third = arr.size // 3
    spreads = []
    for _ in range(repeats):
        perm = rng.permutation(arr)
        spreads.append(float(perm[-third:].mean() - perm[:third].mean()))
    return float(np.quantile(np.abs(spreads), 0.95))


# ---------------------------------------------------------------- 单窗口分析


def analyze_window(
    profiles: dict[str, dict],
    per_trade: dict[str, dict[str, list[float]]],
    summary: dict[str, dict],
) -> dict:
    """对单个窗口做全部维度的判别力检验。"""
    rng = np.random.default_rng(SEED)
    symbols = sorted(set(profiles) & set(summary))

    # 组合收益：跨策略取中位数（抗单个极端策略）
    combo = pd.Series({s: float(np.nanmedian(
        [np.mean(v) if v else np.nan for v in per_trade[s].values()]
    )) for s in symbols}, dtype=float)

    families = {}
    for family in ("trend", "reversion"):
        families[family] = pd.Series({
            s: float(np.nanmean([np.mean(v) for n, v in per_trade[s].items()
                                 if STRATEGY_FAMILY.get(n) == family] or [np.nan]))
            for s in symbols
        }, dtype=float)

    # 五个预测目标：收益、风险调整收益、离散度、尾部、回撤
    targets = {
        "return": combo,
        "sharpe": pd.Series({s: summary[s]["mean"] / summary[s]["std"]
                             if summary[s]["std"] and np.isfinite(summary[s]["std"]) and summary[s]["std"] > 0
                             else np.nan for s in symbols}, dtype=float),
        "risk_std": pd.Series({s: summary[s]["std"] for s in symbols}, dtype=float),
        "risk_worst5": pd.Series({s: summary[s]["worst5"] for s in symbols}, dtype=float),
        "risk_mdd": pd.Series({s: summary[s]["mdd"] for s in symbols}, dtype=float),
    }

    log_liq = pd.Series({
        s: np.log10(max(profiles[s].get("liq_qv_median_usd", np.nan), 1e-9)) for s in symbols
    }, dtype=float)
    # 波动率控制项：代表「这个币天生有多波动」，用来剥离定义式恒等
    vol_ctrl = pd.Series({
        s: profiles[s].get(VOL_CONTROL_KEY, np.nan) for s in symbols
    }, dtype=float)

    out: dict[str, dict] = {}
    for dimension in DIMENSION_TITLES:
        for key in dimension_keys(profiles, dimension):
            metric = cross_section(profiles, key).reindex(symbols)
            if metric.notna().sum() < 12 or metric.dropna().nunique() < 5:
                continue
            entry: dict = {"coverage": int(metric.notna().sum())}
            mv = metric.to_numpy(dtype=float)

            for target_name, target in targets.items():
                tv = target.reindex(symbols).to_numpy(dtype=float)
                ic, p = spearman(mv, tv)
                entry[f"ic_{target_name}"] = ic
                entry[f"p_{target_name}"] = p
                entry[f"pic_{target_name}"] = partial_spearman(mv, tv, log_liq.to_numpy(dtype=float))
                # 双重偏相关：同时控流动性 + 波动率。这个才是「除波动外是否还有信息」
                entry[f"picv_{target_name}"] = partial_spearman_multi(
                    mv, tv, [log_liq.to_numpy(dtype=float), vol_ctrl.to_numpy(dtype=float)]
                )

            # 每个目标各自算档差、CI、随机零分布 —— 否则「能预测风险」会被误判成无用
            for target_name in TARGETS:
                entry[f"buckets_{target_name}"] = bucket_stats(metric, targets[target_name], rng)
            entry["random_95"] = random_baseline(targets["return"], rng)
            spread_return = entry["buckets_return"]["spread"]
            entry["beats_random"] = bool(
                np.isfinite(spread_return) and np.isfinite(entry["random_95"])
                and abs(spread_return) > entry["random_95"]
            )
            # 风险目标的超随机判定
            entry["beats_random_risk"] = False
            risk_spread = entry["buckets_risk_std"]["spread"]
            if np.isfinite(risk_spread):
                entry["beats_random_risk"] = bool(abs(risk_spread) > random_baseline(targets["risk_std"], rng))

            diff_family = (families["trend"] - families["reversion"]).reindex(symbols)
            entry["ic_family"] = spearman(metric.to_numpy(dtype=float), diff_family.to_numpy(dtype=float))[0]

            per_strategy = {}
            all_strategies = sorted({n for s in symbols for n in per_trade[s]})
            for name in all_strategies:
                tgt = pd.Series(
                    {s: float(np.mean(per_trade[s][name])) if name in per_trade[s] else np.nan
                     for s in symbols},
                    dtype=float,
                )
                rho, _ = spearman(metric.to_numpy(dtype=float), tgt.to_numpy(dtype=float))
                if np.isfinite(rho):
                    per_strategy[name] = rho
            entry["per_strategy_ic"] = per_strategy
            entry["ic_consistent"] = float(np.mean([v > 0 for v in per_strategy.values()])) if per_strategy else float("nan")
            out[key] = entry

    return {
        "symbols": len(symbols),
        "metrics": out,
        "combo_target": {s: float(v) for s, v in combo.items() if np.isfinite(v)},
        "summary": summary,
        "log_liq": {s: float(v) for s, v in log_liq.items()},
    }


def aggregate_windows(window_results: list[dict]) -> dict:
    """跨窗口聚合：每个指标在每个口径上给出中位数、符号一致率、通过率。"""
    all_keys: set[str] = set()
    for result in window_results:
        all_keys |= set(result["metrics"])

    aggregated: dict[str, dict] = {}
    for key in sorted(all_keys):
        rows = [r["metrics"][key] for r in window_results if key in r["metrics"]]
        if not rows:
            continue
        entry: dict = {"windows": len(rows)}
        fields = [f"{prefix}_{target}" for prefix in ("ic", "pic", "picv") for target in TARGETS]
        fields += ["ic_family"]
        for field in fields:
            values = [r.get(field) for r in rows]
            finite = [v for v in values if v is not None and np.isfinite(v)]
            entry[field] = float(np.median(finite)) if finite else float("nan")
            entry[f"{field}_sign_consistency"] = (
                float(max(np.mean([v > 0 for v in finite]), np.mean([v < 0 for v in finite]))
                ) if finite else float("nan")
            )

        # 每个目标各自聚合档差、CI、超随机率
        for target_name in TARGETS:
            buckets = [r.get(f"buckets_{target_name}") for r in rows]
            buckets = [b for b in buckets if isinstance(b, dict)]
            if not buckets:
                continue
            spreads = [b["spread"] for b in buckets if np.isfinite(b.get("spread") or np.nan)]
            entry[f"spread_{target_name}"] = float(np.median(spreads)) if spreads else float("nan")
            lows = [b["spread_ci"][0] for b in buckets
                    if np.isfinite((b.get("spread_ci") or (np.nan, np.nan))[0])]
            highs = [b["spread_ci"][1] for b in buckets
                     if np.isfinite((b.get("spread_ci") or (np.nan, np.nan))[1])]
            entry[f"spread_ci_{target_name}"] = (
                float(np.median(lows)) if lows else float("nan"),
                float(np.median(highs)) if highs else float("nan"),
            )

        # 兼容旧字段名：spread / spread_ci 仍指收益口径
        entry["spread"] = entry.get("spread_return", float("nan"))
        entry["spread_ci"] = entry.get("spread_ci_return", (float("nan"), float("nan")))
        entry["beats_random_rate"] = float(np.mean([bool(r.get("beats_random")) for r in rows]))
        entry["beats_random_risk_rate"] = float(np.mean([bool(r.get("beats_random_risk")) for r in rows]))
        entry["ic_consistent"] = float(np.nanmedian([r.get("ic_consistent") for r in rows]))
        entry["coverage"] = int(np.median([r["coverage"] for r in rows]))
        aggregated[key] = entry
    return aggregated


# ---------------------------------------------------------------- 报告


def write_report(result: dict) -> None:
    lines: list[str] = []
    add = lines.append
    meta = result["meta"]

    add("# 币种分类维度：判别力检验")
    add("")
    add(f"- 生成时间：{meta['generated_at']}")
    add(f"- 数据源：klines native 1h（{meta['train_start']} ~ {meta['test_end']}）")
    add(f"- 每笔名义 {STAKE:.0f} USDT，双边成本 0.12%；15 个策略构造器，参数写死不寻优")
    add(f"- 滚动窗口 {meta['window_count']} 个（每个窗口长 {WINDOW_DAYS} 天、步长 {WINDOW_STEP_DAYS} 天，"
        f"窗口内按 {int(SPLIT_FRACTION*100)}/{int((1-SPLIT_FRACTION)*100)} 切训练/样本外）")
    add(f"- 参与币种：{meta['symbol_range']}（各窗口因上市时间不同略有差异）")
    add("")
    add("## 0. 怎么读这张表")
    add("")
    add("五个预测目标分开看，因为用途完全不同：")
    add("")
    add("| 列 | 预测什么 | 用在哪 |")
    add("| --- | --- | --- |")
    add("| `收益IC` | 样本外每笔净收益中位数 | 选币、选策略族 |")
    add("| `夏普IC` | 每笔收益 ÷ 风险（mean÷std） | 单位风险回报，选币首选口径 |")
    add("| `风险IC` | 样本外每笔收益的离散度（std） | 仓位大小、风险预算 |")
    add("| `尾部IC` | 样本外最差 5% 单笔亏损 | 止损距离、爆仓风险 |")
    add("| `回撤IC` | 已实现权益最大回撤 | 单币风险预算上限 |")
    add("")
    add("三种相关口径，**层层剥离混杂**：")
    add("")
    add("| 前缀 | 含义 | 为什么需要 |")
    add("| --- | --- | --- |")
    add("| `ic` | 原始 Spearman | 基线，混入了所有共同因素 |")
    add("| `pic` | 控制 log 流动性 | 不控的话所有维度都退化成「低流动性币波动大」 |")
    add(f"| `picv` | 同时控 log 流动性 + `{VOL_CONTROL_KEY}` | **判定用这个**：剥离波动率后仍有信息才算真发现 |")
    add("")
    add("`picv` 是核心。波动率类指标预测「收益离散度」的原始偏 IC 能到 0.5，"
        "但那是**定义式恒等**（波动率预测波动率），不是发现。第 5 节单独列出了这批伪发现。")
    add("")
    add("| 判定 | 门槛（全部用 `picv`） |")
    add("| --- | --- |")
    add("| 可用·收益 | `\\|picv 收益\\| ≥ 0.10` 且 符号一致率 ≥ 0.80 且 收益超随机率 ≥ 0.67 |")
    add("| 可用·风险 | `\\|picv 风险/尾部\\| ≥ 0.10` 且 符号一致率 ≥ 0.80 且 风险超随机率 ≥ 0.67 |")
    add("| 参考 | `max(\\|picv\\|) ≥ 0.08` |")
    add("| 无用 | 其余 |")
    add("")

    verdicts = result["verdicts"]
    sections = (
        ("usable_return", "1. 可用：能预测收益"),
        ("usable_risk", "2. 可用：能预测风险（不能预测收益）"),
        ("reference", "3. 仅供参考"),
        ("useless", "4. 判定无用"),
        ("vol_identity", "5. 伪发现：原始很强、控波动率后消失"),
    )
    for tag, title in sections:
        items = verdicts.get(tag) or []
        add(f"## {title}")
        add("")
        if not items:
            add("（无）")
            add("")
            continue
        if tag == "vol_identity":
            add("这些指标的原始风险偏 IC 很高，但控制波动率后基本归零 —— "
                "它们只是「波动率」的另一种写法，**不能当作独立维度**。"
                "注意「塌陷目标」列：只在那个口径上是替身，换个口径可能仍有独立信息。")
            add("")
            add("| 指标 | 维度 | 塌陷目标 | 原始风险IC | 控波后风险IC | 是否仍可用于其它口径 |")
            add("| --- | --- | --- | ---: | ---: | --- |")
            for item in items:
                targets = item.get("identity_targets") or []
                target_cn = {"risk_std": "离散度", "risk_worst5": "最差5%"}
                shown = "/".join(target_cn.get(t, t) for t in targets) or "—"
                raw = max((_abs(item.get(f"pic_{t}")) for t in targets), default=0.0)
                after = max((_abs(item.get(f"picv_{t}")) for t in targets), default=0.0)
                survived = [t for t in ("risk_std", "risk_worst5")
                            if t not in targets and _abs(item.get(f"picv_{t}")) >= 0.08]
                note = ("是：" + "/".join(target_cn.get(t, t) for t in survived)) if survived else "否"
                add(f"| `{item['key']}` | {DIMENSION_TITLES[item['dimension']]} "
                    f"| {shown} | {_f(raw)} | {_f(after)} | {note} |")
            add("")
            continue
        add("| 指标 | 维度 | 收益IC | 夏普IC | 风险IC | 尾部IC | 回撤IC | 通过口径 | 档差%(风险) | 超随机率 |")
        add("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |")
        target_cn_map = {"risk_std": "离散度", "risk_worst5": "最差5%"}
        for item in items:
            target = item.get("risk_target")
            shown = target_cn_map.get(target or "", "—")
            spread = item.get(f"spread_{target}") if target else None
            add(f"| `{item['key']}` | {DIMENSION_TITLES[item['dimension']]} "
                f"| {_f(item.get('picv_return'))} | {_f(item.get('picv_sharpe'))} "
                f"| {_f(item.get('picv_risk_std'))} | {_f(item.get('picv_risk_worst5'))} "
                f"| {_f(item.get('picv_risk_mdd'))} | {shown} "
                f"| {_f(spread, 2)} | {_f(item.get('beats_random_risk_rate'), 2)} |")
        add("")
        if tag == "reference":
            flagged = [it for it in items if it.get("sign_conflicts") or it.get("degenerate_bins")]
            if flagged:
                add("其中以下指标**数值过门槛但被质量闸门拦下**，不可直接用于分档：")
                add("")
                add("| 指标 | 拦截原因 | 说明 |")
                add("| --- | --- | --- |")
                for it in flagged:
                    reasons = []
                    if it.get("sign_conflicts"):
                        reasons.append("符号冲突")
                    if it.get("degenerate_bins"):
                        reasons.append("分档退化")
                    detail = []
                    if it.get("sign_conflicts"):
                        detail.append("原始档差与控波后偏相关方向相反")
                    if it.get("degenerate_bins"):
                        detail.append("存在空档，指标有大量并列值")
                    add(f"| `{it['key']}` | {'/'.join(reasons)} | {'；'.join(detail)} |")
                add("")

    add("## 6. 逐维度明细（跨窗口中位数，全部为 `picv` 口径）")
    add("")
    for dimension, title in DIMENSION_TITLES.items():
        rows = {k: v for k, v in result["aggregate"].items() if k.split("_", 1)[0] == dimension}
        if not rows:
            continue
        add(f"### {dimension} — {title}")
        add("")
        add("| 指标 | 覆盖 | 收益IC | 夏普IC | 风险IC(原) | 风险IC(控波) | 尾部IC | 回撤IC | 风险档差% | 风险档差CI | 超随机率 |")
        add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |")
        ranked = sorted(rows.items(), key=lambda kv: -max(_abs(kv[1].get("picv_return")),
                                                          _abs(kv[1].get("picv_risk_std"))))
        for key, e in ranked:
            ci = e.get("spread_ci_risk_std") or (float("nan"), float("nan"))
            add(f"| `{key}` | {e['coverage']} | {_f(e.get('picv_return'))} | {_f(e.get('picv_sharpe'))} "
                f"| {_f(e.get('pic_risk_std'))} | {_f(e.get('picv_risk_std'))} "
                f"| {_f(e.get('picv_risk_worst5'))} | {_f(e.get('picv_risk_mdd'))} "
                f"| {_f(e.get('spread_risk_std'), 2)} "
                f"| {_f(ci[0], 2)}~{_f(ci[1], 2)} | {_f(e.get('beats_random_risk_rate'), 2)} |")
        add("")

    add("## 7. 维度间重复（|ρ| ≥ 0.80）")
    add("")
    pairs = result["high_corr_pairs"]
    if pairs:
        add("这些指标在横截面上几乎是同一个东西，分档时任选其一即可，不要重复计入。")
        add("")
        add("| 指标对 | Spearman |")
        add("| --- | ---: |")
        for pair, rho in sorted(pairs.items(), key=lambda kv: -abs(kv[1]))[:24]:
            add(f"| `{pair}` | {rho:.2f} |")
    else:
        add("无 |ρ| ≥ 0.80 的指标对。")
    add("")
    add("## 8. 逐窗口结果")
    add("")
    add("| 窗口 | 训练期 | 样本外 | 币种 | 组合收益中位% |")
    add("| --- | --- | --- | ---: | ---: |")
    for w in result["windows_meta"]:
        add(f"| {w['label']} | {w['train_start'][:10]}~{w['split'][:10]} "
            f"| {w['split'][:10]}~{w['end'][:10]} | {w['symbols']} | {_f(w['combo_median'], 2)} |")
    add("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def _f(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if np.isfinite(number) else "—"


def _abs(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return abs(number) if np.isfinite(number) else 0.0


def classify(aggregate: dict) -> dict:
    """按门槛把指标分成 可用 / 参考 / 无用，并单独标出「定义式恒等」的伪发现。

    判定用 `picv_`（同时控流动性 + 波动率）而不是 `pic_`。原因：
    波动率类指标预测「收益离散度」的偏 IC 高达 0.5，这是定义式恒等（波动率预测波动率），
    不是发现。只有剥离波动率后仍有信息，才算这个维度真的额外有用。

    关键：**伪发现必须按目标逐项判断**。同一个指标可能在「离散度」上是波动率替身，
    却在「最差 5% 尾部」上仍有独立信息（例如 `tail_worst_day`：
    std 口径 −0.46→−0.07 塌掉，worst5 口径 +0.57→+0.10 存活）。
    对整个指标一律贴「伪发现」标签会误杀这类专精指标。
    """
    buckets: dict[str, list[dict]] = {
        "usable_return": [], "usable_risk": [], "reference": [], "useless": [],
    }
    vol_identity: list[dict] = []
    # 每个风险目标各自判断是否「塌成波动率替身」
    risk_targets = ("risk_std", "risk_worst5")

    for key, entry in aggregate.items():
        item = dict(entry, key=key, dimension=key.split("_", 1)[0])

        # --- 收益路径
        ret_pic = _abs(entry.get("picv_return"))
        ret_cons = entry.get("picv_return_sign_consistency") or 0
        ret_rand = entry.get("beats_random_rate") or 0
        ret_ok = ret_pic >= 0.10 and ret_cons >= 0.80 and ret_rand >= 0.67

        # --- 风险路径：逐个目标看，取通过的那一个
        risk_ok_target = None
        for target in risk_targets:
            picv = _abs(entry.get(f"picv_{target}"))
            cons = entry.get(f"picv_{target}_sign_consistency") or 0
            rand = entry.get("beats_random_risk_rate") or 0
            if picv >= 0.10 and cons >= 0.80 and rand >= 0.67:
                if risk_ok_target is None or picv > _abs(entry.get(f"picv_{risk_ok_target}")):
                    risk_ok_target = target
        item["risk_target"] = risk_ok_target

        # --- 伪发现：原始很强，控波动率后塌掉。按目标分别记录
        identity_targets = [
            target for target in risk_targets
            if _abs(entry.get(f"pic_{target}")) >= 0.25 and _abs(entry.get(f"picv_{target}")) < 0.08
        ]
        item["identity_targets"] = identity_targets

        # --- 质量闸门（两个，任一命中就不能推荐使用）
        # 闸门 1：符号冲突。原始档差与控波后偏相关方向相反 ⇒ 结论不可解释。
        #   例：vol_max_bar_absret 原始档差 −2.33（高值更差），picv 却是 +0.173。
        #   根因是控制项与指标本身高度相关（|ρ|≈0.9），残差被翻向。这类不能用。
        sign_conflicts: list[str] = []
        for target in risk_targets:
            spread = entry.get(f"spread_{target}")
            picv = entry.get(f"picv_{target}")
            if np.isfinite(spread or np.nan) and np.isfinite(picv or np.nan) \
                    and spread != 0 and picv != 0 and np.sign(spread) != np.sign(picv):
                sign_conflicts.append(target)
        item["sign_conflicts"] = sign_conflicts

        # 闸门 2：分档退化。任一档样本数为 0 ⇒ 该指标有大量并列值，三分档切不开。
        degenerate = sum(
            1 for target in risk_targets
            if any((entry.get(f"buckets_{target}") or {}).get("bucket_n", {}).get(f"b{n}", 1) == 0
                   for n in (0, 1, 2))
        ) > 0
        item["degenerate_bins"] = bool(degenerate)

        if ret_ok:
            buckets["usable_return"].append(item)
        elif risk_ok_target is not None and risk_ok_target not in identity_targets \
                and not sign_conflicts and not degenerate:
            buckets["usable_risk"].append(item)
        elif risk_ok_target is not None:
            # 通过数值门槛但被质量闸门拦下 ⇒ 归入「参考」，并标注原因
            buckets["reference"].append(item)
        elif max(ret_pic, _abs(entry.get("picv_risk_std")),
                 _abs(entry.get("picv_risk_worst5"))) >= 0.08:
            buckets["reference"].append(item)
        else:
            buckets["useless"].append(item)

        # 只要存在任一目标塌成波动率替身，就记录到伪发现清单（供文档警示）
        if identity_targets and item not in vol_identity \
                and not (ret_ok and not identity_targets):
            vol_identity.append(item)

    def _rank(it: dict) -> float:
        return -max(_abs(it.get("picv_return")), _abs(it.get("picv_risk_std")),
                    _abs(it.get("picv_risk_worst5")))

    for tag in buckets:
        buckets[tag].sort(key=_rank)
    # 伪发现清单去重（同一指标只列一次）
    seen: set[str] = set()
    deduped = []
    for it in sorted(vol_identity, key=lambda x: -_abs(x.get("pic_risk_std"))):
        if it["key"] not in seen:
            seen.add(it["key"])
            deduped.append(it)
    buckets["vol_identity"] = deduped
    return buckets


# ---------------------------------------------------------------- 主流程


def main() -> int:
    started = time.time()
    frames, start_ms, end_ms = load_frames()

    windows = rolling_windows(start_ms, end_ms)
    if not windows:
        split = int(start_ms + (end_ms - start_ms) * SPLIT_FRACTION)
        windows = [{"label": "ALL", "start": start_ms, "split": split, "end": end_ms}]

    window_results: list[dict] = []
    windows_meta: list[dict] = []
    for window in windows:
        profiles = train_profiles(frames, window["start"], window["split"])
        per_trade, summary = oos_returns(frames, window["split"], window["end"])
        shared = sorted(set(profiles) & set(summary))
        if len(shared) < 12:
            print(f"{window['label']}: 可用币种仅 {len(shared)}，跳过")
            continue
        result = analyze_window(profiles, per_trade, summary)
        window_results.append(result)
        combo_median = float(np.nanmedian(list(result["combo_target"].values())))
        windows_meta.append({
            "label": window["label"],
            "train_start": pd.to_datetime(window["start"], unit="ms", utc=True).isoformat(),
            "split": pd.to_datetime(window["split"], unit="ms", utc=True).isoformat(),
            "end": pd.to_datetime(window["end"], unit="ms", utc=True).isoformat(),
            "symbols": result["symbols"],
            "combo_median": combo_median,
        })
        print(f"{window['label']}: {result['symbols']} 币，组合收益中位 {combo_median:+.2f}%")

    if not window_results:
        print("没有可用窗口")
        return 1

    aggregate = aggregate_windows(window_results)
    verdicts = classify(aggregate)

    # 维度间重复：用最后一个窗口的画像算（跨窗口结论一致，取最新即可）
    last = window_results[-1]
    profiles_last = train_profiles(frames, windows[-1]["start"], windows[-1]["split"])
    symbols_last = sorted(set(profiles_last) & set(last["summary"]))
    pairs = {}
    metrics = {k: cross_section(profiles_last, k).reindex(symbols_last) for k in aggregate}
    names = sorted(metrics)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair = pd.concat([metrics[a], metrics[b]], axis=1).dropna()
            if len(pair) < 12:
                continue
            rho = float(stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])[0])
            if np.isfinite(rho) and abs(rho) >= 0.80:
                pairs[f"{a}|{b}"] = rho

    symbol_counts = [m["symbols"] for m in windows_meta]
    result = {
        "meta": {
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "train_start": windows_meta[0]["train_start"],
            "test_end": windows_meta[-1]["end"],
            "window_count": len(windows_meta),
            "symbol_range": f"{min(symbol_counts)}~{max(symbol_counts)}",
            "split_fraction": SPLIT_FRACTION,
            "stake": STAKE,
            "elapsed_sec": round(time.time() - started, 1),
        },
        "windows_meta": windows_meta,
        "aggregate": aggregate,
        "verdicts": verdicts,
        "high_corr_pairs": pairs,
        "per_window": [{"label": m["label"], "metrics": r["metrics"]} for m, r in zip(windows_meta, window_results)],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    write_report(result)

    print(f"\n窗口 {len(windows_meta)}，耗时 {result['meta']['elapsed_sec']}s")
    for tag, items in verdicts.items():
        print(f"  {tag:14s} {len(items):2d} 个: " + ", ".join(i["key"] for i in items[:8]))
    print(f"写出 {OUT_JSON.name} / {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

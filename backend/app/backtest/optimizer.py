"""全币种参数寻优：网格搜索 × 样本外验证 × 进程池并行。

为什么单独成模块
----------------
`runner.run_backtest` 是「单币种、单套参数、带完整报告」的口径，一次调用只为看一个
币的结论。而「每个币种找出自己的最优参数」要同时满足三件事，不能用它的循环实现：

1. **算力要打满**：`simulate` 是纯 Python 逐 bar，线程被 GIL 锁死，只有进程池有意义。
   分片单位是「币种」——一个 worker 拿到一个币就把它整张网格跑完，这样数据帧只
   建一次、指标只预热一次，通信开销摊到几百个组合上可以忽略。
2. **选择必须样本外**：直接在整段历史上挑收益最高的组合，520 个币种里必然挑到运气。
   这里把区间切成训练段 / 验证段，训练段选参、验证段只做判定，验证段不参与选择。
3. **结论要能被推翻**：验证段除了收益，还要给每笔期望的 bootstrap 置信区间。
   CI 跨 0 就是「不知道」，必须和「亏损」区分开。

并行铁律（踩过的坑）
--------------------
- `simulate` 纯 Python → 线程零收益，必须进程池。
- worker 内部自建**只读、NullPool** 引擎：默认连接池（5+10）在 20 个进程下会打满
  PostgreSQL 的 max_connections，而 worker 只做顺序查询，池化没有收益。
- 进程池全局建一次，不要每个币重建（Windows spawn 的 import 成本极高）。
- prepare 的输出按 `spec.prepare_keys` 缓存：滚动百分位是每币最贵的一步，
  网格里出场参数（target_r / max_hold_bars）变化不影响它，可以复用。

周期也是搜索维度
----------------
同一个策略在 15m 和 4h 上的最优参数不是同一套，甚至同一个币在 15m 上不可交易、
在 4h 上有 edge。所以任务单位是 **(币种, 周期)** 而不是币种：每个组合都在它自己
的周期上模拟，`simulate` 的持仓时长、年化换算、成本摊销都随周期变。任务粒度变细
还有个副作用是负载更均衡——1h 的 365 天和 4h 的 365 天 bar 数差 4 倍，混在一个
worker 里会让尾部只剩一个大任务在跑。

选完每个 (币, 周期) 的最优组合后，还要在币级别再选一次「最佳周期」，见
`pick_best_per_symbol`。

delta 源
--------
`delta_source="auto"` 时有 1m 数据就用 lower-tf delta（与实盘口径一致），覆盖不足
90% 或全零则降级 est。两条路径的读数不可直接比较，所以结果的 `delta_source`
字段必须一起看：同一轮里混用两种源，池化 CI 会被口径差异污染。
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Callable

# 必须在 numpy/BLAS 于**子进程**里初始化之前设置：进程池里 20 个进程各自默认开
# cpu_count 个 OpenBLAS 线程，每个线程预留栈缓冲，实测直接 "Memory allocation
# still failed after 10 retries" 并杀掉整个池。这里算的是逐 bar 的纯 Python 循环，
# 用不到多线程 BLAS，单线程反而更快（没有线程切换与缓存争抢）。
# 父进程的 numpy 早已 import，改不到它；关键是让 spawn 出来的子进程继承这些变量。
for _threads_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_threads_var, "1")

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from ..config import SUPPORTED_INTERVALS
from ..db import make_engine
from ..ml.dataset import build_dataset
from ..state import all_state
from ..strategy.registry import get_spec
from ..strategy.rules import evaluate_signals, prepare_frame, resolve_params
from ..utils import interval_ms, now_ms
from .execution import simulate
from .metrics import metrics_from_equity
from .significance import bootstrap_expectancy_ci
from .trades import aggregate_positions

# 默认组合数：972 种全组合里抽 64 个。实测 64 组已覆盖各参数的边缘分布，
# 再多是线性增加的算力换越来越小的信息量。
DEFAULT_COMBOS = 64
MAX_COMBOS = 512
MAX_WORKERS = 24
# 保留 4 核给 FastAPI / 调度器 / 数据库：寻优跑满 24 核会把巡检挤到没 CPU
RESERVED_CORES = 4
# 池化判定的最小成交笔数：单币判定有 min_trades 门槛，池化层也要有。
# 实测 4h × 365 天全市场只有 4 笔成交，n=4 的 bootstrap 会给出「CI 下界 +0.86%」
# 的假精确读数，必须降级成 insufficient 而不是当成找到了 edge。
MIN_POOLED_TRADES = 30
# worker 内 prepared 帧缓存上限：一份 15m×180天 帧约 4MB，8 份 ≈ 32MB/进程
PREPARE_CACHE_LIMIT = 8

_SESSION_FACTORY = None


def default_workers() -> int:
    cpu = os.cpu_count() or 2
    return max(1, min(MAX_WORKERS, cpu - RESERVED_CORES))


# ------------------------------------------------------------------ 搜索空间


def grid_size(spec, custom: dict | None = None) -> int:
    space = custom or spec.grid
    total = 1
    for values in space.values():
        total *= max(1, len(values))
    return total


def build_grid(
    spec,
    base_params: dict,
    n: int = DEFAULT_COMBOS,
    seed: int = 7,
    custom: dict | None = None,
) -> list[dict]:
    """生成候选参数组合（只含搜索空间里的键）。

    `n` 小于全组合数时按**均匀下标抽样**而不是逐键独立随机：逐键随机会让某些
    组合反复出现、另一些永不出现，均匀下标能保证每个参数档位都被覆盖到。
    基准参数（当前线上参数）一定在候选集里，作为「不改动」的对照。
    """
    space = custom or spec.grid
    if not space:
        raise ValueError(f"{spec.code} 没有声明参数搜索空间（spec.grid）")
    keys = list(space)
    values = [list(space[key]) for key in keys]
    for key, vals in zip(keys, values):
        if not vals:
            raise ValueError(f"搜索空间 {key} 的候选值不能为空")
    total = grid_size(spec, space)
    n = max(1, min(int(n), MAX_COMBOS, max(1, total)))

    def decode(index: int) -> dict:
        combo = {}
        for key, vals in zip(keys, values):
            index, rest = divmod(index, len(vals))
            combo[key] = vals[rest]
        return combo

    if total <= n:
        combos = [decode(index) for index in range(total)]
    else:
        rng = random.Random(seed)
        combos = [decode(index) for index in rng.sample(range(total), n)]

    base_combo = {key: base_params[key] for key in keys if key in base_params}
    if base_combo and base_combo not in combos:
        combos[0] = base_combo
    # 丢掉违反参数约束的组合：最典型的是 max_hold_bars < min_hold_bars。
    # 不在这里过滤的话，worker 会照着一组「写不进线上」的参数跑出漂亮的回测，
    # 而 apply 时才被 resolve_params 拒绝——用户会以为寻优结果不可用。
    legal = []
    for combo in combos:
        layer = {
            key: value
            for key, value in {**base_params, **combo}.items()
            if key not in ("fee_rate", "slippage")  # 撮合成本不允许作为策略覆盖
        }
        try:
            resolve_params(spec.code, {}, layer)
        except ValueError:
            continue
        legal.append(combo)
    return legal or [base_combo]


def diff_params(combo: dict, base: dict) -> dict:
    """只保留与基准不同的键，写入覆盖表时保持可读。"""
    return {key: value for key, value in combo.items() if base.get(key) != value}


# ------------------------------------------------------------------ worker 侧


def _session() -> Session:
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        engine = make_engine(read_only=True, null_pool=True)
        _SESSION_FACTORY = sessionmaker(
            bind=engine, autoflush=False, expire_on_commit=False, future=True
        )
    return _SESSION_FACTORY()


def _monthly_positive_ratio(equity: pd.Series) -> float:
    """按自然月统计正收益月占比：从同一条权益曲线算，不额外回测。"""
    if len(equity) < 2:
        return 0.0
    monthly = equity.resample("1ME").last().dropna()
    if len(monthly) < 2:
        return 0.0
    changes = monthly.pct_change().dropna()
    if changes.empty:
        return 0.0
    return round(float((changes > 0).mean()), 4)


def _objective(metrics: dict, kind: str) -> float:
    """选参目标。默认与 runner 一致：净收益 - 0.5×回撤幅度。"""
    if not metrics or not metrics.get("total_trades"):
        return -math.inf
    if kind == "sharpe":
        return float(metrics.get("sharpe") or 0.0)
    return float(metrics.get("total_return_pct") or 0.0) + 0.5 * float(
        metrics.get("max_drawdown_pct") or 0.0
    )


def _verdict(valid_metrics: dict, ci: dict, min_trades: int) -> str:
    """验证段判定。

    顺序很重要：先看有没有成交，再看样本够不够，最后才谈置信区间。
    成交 3 笔跑出来的「CI 下界 > 0」不算通过——bootstrap 在小样本上会假精确。
    """
    trades = int(valid_metrics.get("total_trades") or 0)
    if trades == 0:
        return "no_trades"
    if trades < max(4, min_trades // 2):
        return "insufficient"
    if ci.get("verdict") == "positive":
        return "pass"
    if ci.get("verdict") == "negative":
        return "negative"
    return "inconclusive"


def _attach_lower_delta(db: Session, symbol: str, frame: pd.DataFrame, mode: str) -> str:
    """把 1m 聚合出的 intrabar delta 附到帧上，返回实际使用的 delta 源。

    这不是可选优化：同一份 15m 帧，est delta 下 90 天只有 1 个信号，lower delta
    下有 5~34 个——寻优如果跑在 est 上，优化的压根不是实盘那个策略。

    返回值有三档：
    - `lower`：lower-tf 覆盖 >=90%，整段都是真实买卖差
    - `mixed`：只覆盖一部分（1m 历史比窗口短）。缺的那段由 `add_absorption_context`
      逐根用 est 兜底，这是脚本 `na(ltfDelta) ? estDelta` 的原语义
    - `est`：完全没有 lower-tf 数据

    早先这里对部分覆盖直接整段降级 est，理由是「半截 NaN 会让滚动百分位算出假阈值」。
    但整段降级反而更糟：长窗口下等于放弃了那 94 天的真实口径；而 bucket_delta 曾经
    把空桶填 0，覆盖率永远显示 100%，保护完全失效。
    """
    if mode == "est":
        return "est"
    try:
        from ..features.absorption import load_lower_tf_delta
        from ..market.data_service import MarketDataService

        lower = load_lower_tf_delta(MarketDataService(db), symbol, frame.index)
    except Exception:
        return "est"
    if lower is None or not len(lower):
        return "est"
    coverage = float(lower.notna().mean())
    if coverage <= 0:
        return "est"
    frame["intrabar_delta"] = lower
    frame.attrs["delta_coverage"] = round(coverage, 4)
    return "lower" if coverage >= 0.9 else "mixed"


def _finish(payload: dict, task: dict) -> dict:
    """给 worker 结果补上周期标识：上层要靠它区分同一个币的不同周期结果。"""
    payload["interval"] = task.get("interval", "")
    payload["key"] = task.get("key") or payload.get("symbol", "")
    return payload


def evaluate_symbol_task(task: dict) -> dict:
    """进程池 worker：单个 (币种, 周期) 跑完整张网格，返回摘要（不含成交明细）。

    不返回 equity / trades：500 个币 × 几个周期 × 几百笔成交会把结果表撑爆，
    而选参只需要指标。要看具体成交时用单币回测复现。
    """
    symbol = task["symbol"]
    strategy = task["strategy"]
    interval = task["interval"]
    started = time.perf_counter()
    try:
        spec = get_spec(strategy)
        base = task["base_params"]
        step = interval_ms(interval)
        days = int(task["days"])
        # 多取一段预热再裁掉：滚动百分位需要 lookback 根历史，否则训练段开头
        # 几百根算出来的阈值是失真的（比不预热更危险，因为它看着像正常读数）
        warmup_days = spec.lookback(base) * step / 86_400_000
        db = _session()
        try:
            frame = build_dataset(
                db,
                symbol,
                interval,
                horizon=5,
                days=days + math.ceil(warmup_days) + 1,
                with_label=False,
            )
            cutoff = pd.Timestamp(now_ms() - days * 86_400_000, unit="ms", tz="UTC")
            frame = frame[frame.index >= cutoff]
            delta_source = _attach_lower_delta(db, symbol, frame, task.get("delta_source", "auto"))
            delta_coverage = float(frame.attrs.get("delta_coverage") or 0.0)
        finally:
            db.close()
        if len(frame) < int(task["min_bars"]):
            return _finish({
                "symbol": symbol,
                "status": "skipped",
                "error": f"可用K线仅 {len(frame)} 根（门槛 {task['min_bars']}）",
                "rows": len(frame),
                "delta_source": delta_source,
                "delta_coverage": delta_coverage,
                "elapsed": round(time.perf_counter() - started, 2),
            }, task)

        split = int(len(frame) * float(task["train_ratio"]))
        train, valid = frame.iloc[:split], frame.iloc[split:]
        if len(train) < 300 or len(valid) < 100:
            return _finish({
                "symbol": symbol,
                "status": "skipped",
                "error": f"切分后样本不足（训练 {len(train)} / 验证 {len(valid)}）",
                "rows": len(frame),
                "delta_source": delta_source,
                "delta_coverage": delta_coverage,
                "elapsed": round(time.perf_counter() - started, 2),
            }, task)

        initial, fee, slippage = task["initial"], task["fee"], task["slippage"]
        min_trades = int(task["min_trades"])
        objective_kind = task.get("objective", "return_dd")
        prepared_keys = tuple(spec.prepare_keys)
        cache: dict[tuple, pd.DataFrame] = {}
        best: dict | None = None
        evaluated = eligible = 0

        for combo in task["grid"]:
            params = {**base, **combo}
            evaluated += 1
            signature = tuple(params.get(key) for key in prepared_keys)
            prepared = cache.get(signature)
            if prepared is None:
                prepared = prepare_frame(train, strategy, params)
                if len(cache) >= PREPARE_CACHE_LIMIT:
                    cache.clear()
                cache[signature] = prepared
            signals = spec.signals(prepared, params)
            equity, trades = simulate(
                prepared, signals, params, initial, fee, slippage, interval
            )
            metrics = metrics_from_equity(equity, trades, initial, interval)
            if int(metrics.get("total_trades") or 0) < min_trades:
                continue
            eligible += 1
            score = _objective(metrics, objective_kind)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "combo": combo,
                    "params": params,
                    "metrics": metrics,
                    "equity": equity,
                    "trades": trades,
                }

        if best is None:
            return _finish({
                "symbol": symbol,
                "status": "ok",
                "rows": len(frame),
                "delta_source": delta_source,
                "delta_coverage": delta_coverage,
                "train_bars": len(train),
                "valid_bars": len(valid),
                "grid_size": len(task["grid"]),
                "evaluated": evaluated,
                "eligible": 0,
                "best_combo": {},
                "train": {},
                "valid": {},
                "ci": {"trades": 0, "verdict": "no_trades"},
                "verdict": "no_trades",
                "score": 0.0,
                "elapsed": round(time.perf_counter() - started, 2),
            }, task)

        valid_signals = evaluate_signals(valid, strategy, best["params"])
        valid_equity, valid_trades = simulate(
            valid, valid_signals, best["params"], initial, fee, slippage, interval
        )
        valid_metrics = metrics_from_equity(valid_equity, valid_trades, initial, interval)
        ci = bootstrap_expectancy_ci(valid_trades)
        train_metrics = dict(best["metrics"])
        train_metrics["positive_months"] = _monthly_positive_ratio(best["equity"])
        return _finish({
            "symbol": symbol,
            "status": "ok",
            "rows": len(frame),
            "delta_source": delta_source,
            "delta_coverage": delta_coverage,
            "train_bars": len(train),
            "valid_bars": len(valid),
            "grid_size": len(task["grid"]),
            "evaluated": evaluated,
            "eligible": eligible,
            "best_combo": diff_params(best["combo"], base),
            "train": train_metrics,
            "valid": valid_metrics,
            "ci": ci,
            # 池化判定的原料：单币样本太少时，把所有币的验证段成交合起来
            # 才能回答「这套策略到底有没有 edge」。只带收益百分比，不带明细。
            "valid_returns": [round(float(t.get("return_pct", 0.0)), 4) for t in aggregate_positions(valid_trades)[:300]],
            "verdict": _verdict(valid_metrics, ci, min_trades),
            "score": round(best["score"], 4),
            "elapsed": round(time.perf_counter() - started, 2),
        }, task)
    except Exception as exc:  # 单币失败不中断整轮
        return _finish({
            "symbol": symbol,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}"[:500],
            "elapsed": round(time.perf_counter() - started, 2),
        }, task)


# ------------------------------------------------------------------ 主进程侧


@dataclass
class OptimizationRequest:
    strategy: str
    interval: str = "15m"
    # 周期作为搜索维度：给了 intervals 就跑遍这些周期，每个 (币, 周期) 各出一行结果。
    # `interval` 保留为主周期（表单默认值、单周期调用的入口）。
    intervals: list[str] | None = None
    days: int = 180
    train_ratio: float = 0.7
    combos: int = DEFAULT_COMBOS
    workers: int = 0
    min_trades: int = 5
    objective: str = "return_dd"
    # auto = 有 1m 数据就用 lower-tf delta（与实盘口径一致），覆盖不足降级 est
    delta_source: str = "auto"
    custom_grid: dict | None = None
    seed: int = 7
    initial: float = 0.0
    fee: float = -1.0
    slippage: float = -1.0

    @property
    def interval_list(self) -> list[str]:
        """要扫的周期列表（去重并保持顺序）。"""
        raw = self.intervals or [self.interval]
        seen: list[str] = []
        for item in raw:
            text_ = str(item or "").strip()
            if text_ and text_ not in seen:
                seen.append(text_)
        return seen or [self.interval]

    def validate(self) -> None:
        intervals = self.interval_list
        if not intervals:
            raise ValueError("至少需要一个周期")
        for item in intervals:
            if item not in SUPPORTED_INTERVALS:
                raise ValueError(f"不支持的周期: {item}")
        if not 30 <= int(self.days) <= 720:
            raise ValueError("回测天数允许 30-720")
        if not 0.5 <= float(self.train_ratio) <= 0.85:
            raise ValueError("训练占比允许 0.5-0.85")
        if not 4 <= int(self.combos) <= MAX_COMBOS:
            raise ValueError(f"组合数允许 4-{MAX_COMBOS}")
        if not 1 <= int(self.min_trades) <= 500:
            raise ValueError("成交门槛允许 1-500")
        if self.objective not in ("return_dd", "sharpe"):
            raise ValueError("未知的选参目标")
        if self.delta_source not in ("auto", "lower", "est"):
            raise ValueError("未知的 delta 源")
        if not 1 <= int(self.workers or 1) <= MAX_WORKERS:
            raise ValueError(f"并发进程允许 1-{MAX_WORKERS}")


def _base_params(db: Session, strategy: str, interval: str, state: dict) -> dict:
    """基准 = 该策略在线上最常用的那套参数（绑定 → 策略配置 → 默认）。

    「最常用」取启用优先、优先级数字小的那条绑定：寻优结果最终要写回覆盖表，
    基准选错会让「改善幅度」这一列失去参照。
    """
    from ..models import StrategyBinding, StrategyConfig

    row = (
        db.query(StrategyBinding)
        .filter(
            StrategyBinding.strategy_code == strategy,
            StrategyBinding.interval == interval,
        )
        .order_by(StrategyBinding.enabled.desc(), StrategyBinding.priority.asc())
        .first()
    )
    layer: dict = {}
    target = row.params if row is not None else None
    if target is None:
        config = db.query(StrategyConfig).filter(StrategyConfig.code == strategy).first()
        target = config.params if config is not None else None
    if target:
        try:
            parsed = json.loads(target)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            layer = parsed
    return resolve_params(strategy, state, layer or None, interval=interval)


def prepare_request(
    db: Session, strategy: str, interval: str | None = None, **kwargs
) -> tuple[OptimizationRequest, dict, dict[str, list[dict]]]:
    """把请求补齐成可直接派发的形态：逐周期的基准参数、成本、候选网格。

    返回的第三个元素是 `{周期: 组合列表}`。每个周期的基准参数不同（绑定按
    `(策略, 周期)` 唯一），所以网格也必须逐周期独立抽样——否则 4h 的基准会被
    15m 的绑定参数污染，「改善幅度」这一列就失去参照。
    """
    spec = get_spec(strategy)
    if interval and not kwargs.get("intervals"):
        kwargs["intervals"] = [interval]
    request = OptimizationRequest(strategy=strategy, interval=interval or "15m", **kwargs)
    request.validate()
    state = all_state(db)
    if request.initial <= 0:
        request.initial = float(state.get("initial_capital", 10000.0))
    if request.fee < 0:
        request.fee = float(state["fee_rate"])
    if request.slippage < 0:
        request.slippage = float(state["slippage"])

    intervals = request.interval_list
    per_interval: dict[str, dict] = {}
    grids: dict[str, list[dict]] = {}
    for item in intervals:
        base = _base_params(db, strategy, item, state)
        grids[item] = build_grid(
            spec, base, n=request.combos, seed=request.seed, custom=request.custom_grid
        )
        space = request.custom_grid or spec.grid
        per_interval[item] = {
            "base_params": base,
            "grid_size": grid_size(spec, request.custom_grid),
            "combos": len(grids[item]),
            "grid_keys": list(space),
        }

    primary = intervals[0]
    meta = {
        "intervals": intervals,
        "base_params": per_interval[primary]["base_params"],
        # 结果表里「寻优前」的对照只列被搜索过的键，存整套 40 多个参数没有信息量
        "grid_keys": per_interval[primary]["grid_keys"],
        "fee": request.fee,
        "slippage": request.slippage,
        "initial": request.initial,
        "grid_size": per_interval[primary]["grid_size"],
        "combos": len(grids[primary]),
        "per_interval": per_interval,
        "objective": request.objective,
        "min_trades": request.min_trades,
        "train_ratio": request.train_ratio,
        "days": request.days,
        "delta_source": request.delta_source,
    }
    return request, meta, grids


def symbol_universe(db: Session, interval: str, days: int) -> list[str]:
    """有足够历史覆盖的币种。

    按原始K线计数判断而不是逐个 build_dataset —— 对 500+ 币种来说，
    前者是一条 SQL，后者是 500 次指标预热。

    计数口径要覆盖回测的两条取数路径：`interval` 自身的 native 行 + 1m 聚合行
    （1m 只覆盖 94 天，但它是短周期唯一来源）。只数 native 行会把「1m 聚合够用」
    的币误判成不可用——实测 15m 90 天：只数 native 是 31 币，合并口径是 70+。
    两条路径在重叠时段会被重复计入，所以门槛本身是宽松的粗筛；真正的硬门槛在
    `evaluate_symbol_task` 里（`min_bars` + 训练/验证段最小长度）。
    """
    step = interval_ms(interval)
    needed = max(200, int(days * 86_400_000 / step * 0.5))
    since = now_ms() - int(days) * 86_400_000
    if interval == "1m":
        rows = db.execute(
            text(
                "select symbol from klines where interval = '1m' and open_time >= :since "
                "group by symbol having count(*) >= :needed order by symbol"
            ),
            {"since": since, "needed": needed},
        ).all()
        return [row[0] for row in rows]
    rows = db.execute(
        text(
            """
            select symbol from (
              select symbol, sum(w) as bars from (
                select symbol, 1.0 as w from klines
                 where interval = :iv and open_time >= :since
                union all
                select symbol, :m1_weight as w from klines
                 where interval = '1m' and open_time >= :since
              ) u group by symbol
            ) t where bars >= :needed order by symbol
            """
        ),
        {
            "iv": interval,
            "since": since,
            "needed": needed,
            # 一根 1m 只能折算成 1/step_minutes 根目标周期
            "m1_weight": round(60_000 / step, 6),
        },
    ).all()
    return [row[0] for row in rows]


def symbol_universes(
    db: Session, intervals: list[str], days: int
) -> dict[str, list[str]]:
    """逐周期算可用币种（同一币在不同周期的覆盖完全不同）。"""
    return {item: symbol_universe(db, item, days) for item in intervals}


def _run_round(
    tasks: list[dict],
    workers: int,
    *,
    total: int,
    started: float,
    counts: dict,
    verdicts: dict,
    completed: set[str],
    on_result: Callable[[dict], None] | None,
    on_progress: Callable[[dict], None] | None,
    cancel: threading.Event | None,
) -> bool:
    """跑一批任务，返回是否被取消；池崩溃时把 BrokenProcessPool 抛给调用方。"""
    cancelled = False
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(evaluate_symbol_task, task): task["key"] for task in tasks}
        for future in as_completed(futures):
            if cancel is not None and cancel.is_set():
                cancelled = True
                for pending in futures:
                    pending.cancel()
                break
            try:
                payload = future.result()
            except Exception as exc:
                # 池崩溃必须冒泡到上层重建，其它异常降级成单任务失败
                if isinstance(exc, BrokenProcessPool):
                    raise
                payload = {
                    "key": futures[future],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            key = payload.get("key") or futures[future]
            completed.add(key)
            status = payload.get("status")
            if status == "error":
                counts["failed"] += 1
            elif status == "skipped":
                counts["skipped"] += 1
            else:
                counts["done"] += 1
                verdict_key = payload.get("verdict") or ""
                if verdict_key:
                    verdicts[verdict_key] = verdicts.get(verdict_key, 0) + 1
            if on_result is not None:
                on_result(payload)
            if on_progress is not None:
                done = counts["done"] + counts["failed"] + counts["skipped"]
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                on_progress(
                    {
                        "total": total,
                        "done": done,
                        "failed": counts["failed"],
                        "skipped": counts["skipped"],
                        "elapsed": round(elapsed, 1),
                        "eta": round((total - done) / rate, 1) if rate > 0 else None,
                        "current": payload.get("key", ""),
                        "verdicts": dict(verdicts),
                    }
                )
    finally:
        try:
            pool.shutdown(wait=not cancelled, cancel_futures=True)
        except Exception:
            pass  # 池已崩溃时 shutdown 自己也可能抛
    return cancelled


VERDICT_RANK = {
    "pass": 4,
    "inconclusive": 3,
    "negative": 2,
    "insufficient": 1,
    "no_trades": 0,
}


def pick_best_per_symbol(items: list[dict]) -> dict[str, dict]:
    """在每个币的多个周期结果里选一个「最佳配置」。

    排序不看单一指标，而是先看结论强度再看分数：
    1. 判定等级（pass 优于 inconclusive 优于 negative 优于样本不足/无成交）——
       验证段没成交的周期即使训练段分数再高也不能用；
    2. 同等级比 score（验证段净收益 + 0.5×最大回撤）；
    3. 再比验证段成交笔数，样本多的更可信。

    没有成交的周期分数都是 0，如果只按 score 排，会出现「4h 无成交」和「1h 微亏
    且 -0.3 分」之间的胡乱排序，所以等级必须先于分数。
    """
    best: dict[str, dict] = {}
    for item in items:
        symbol = item.get("symbol")
        if not symbol:
            continue
        current = best.get(symbol)
        if current is None or _rank(item) > _rank(current):
            best[symbol] = item
    return best


def _rank(item: dict) -> tuple:
    valid = item.get("valid") or {}
    return (
        VERDICT_RANK.get(item.get("verdict") or "", 0),
        float(item.get("score") or 0.0),
        int(valid.get("total_trades") or 0),
    )


def build_tasks(
    request: OptimizationRequest,
    meta: dict,
    grids: dict[str, list[dict]],
    symbols: dict[str, list[str]],
) -> list[dict]:
    """展开成 (币种, 周期) 任务列表。

    每个任务带自己的 `base_params` 与 `grid`：同一策略在不同周期上生效的线上参数
    本来就不同，混用会让「相对基准改善了多少」失去意义。
    """
    tasks: list[dict] = []
    for interval in request.interval_list:
        iv_symbols = symbols.get(interval) or []
        if not iv_symbols:
            continue
        step = interval_ms(interval)
        min_bars = max(400, int(request.days * 86_400_000 / step * 0.4))
        grid = grids.get(interval) or []
        base = (meta.get("per_interval") or {}).get(interval, {}).get(
            "base_params", meta.get("base_params") or {}
        )
        for symbol in iv_symbols:
            tasks.append(
                {
                    "key": f"{symbol}@{interval}",
                    "symbol": symbol,
                    "interval": interval,
                    "strategy": request.strategy,
                    "days": request.days,
                    "train_ratio": request.train_ratio,
                    "min_trades": request.min_trades,
                    "objective": request.objective,
                    "base_params": base,
                    "initial": request.initial,
                    "fee": request.fee,
                    "slippage": request.slippage,
                    "delta_source": request.delta_source,
                    "min_bars": min_bars,
                    "grid": grid,
                }
            )
    return tasks


def run_optimization(
    *,
    request: OptimizationRequest,
    meta: dict,
    grids: dict[str, list[dict]],
    symbols: dict[str, list[str]],
    on_progress: Callable[[dict], None] | None = None,
    on_result: Callable[[dict], None] | None = None,
    cancel: threading.Event | None = None,
    workers: int | None = None,
) -> dict:
    """跑完整轮寻优，返回汇总。

    `symbols` 是 `{周期: 币种列表}`；同一币种在不同周期的覆盖不同，必须逐周期给。
    `on_result` 每完成一个 (币种, 周期) 回调一次（写库/推进度）。
    `cancel` 被置位后不再收集新结果，进程池等已派发的任务跑完再退出——
    进程池没有安全的强杀手段，硬杀会留下僵尸进程和半写的连接。

    进程池崩溃（BLAS 申请不到内存、worker 被系统杀掉）时**不整轮失败**：
    已完成的任务结果保留，剩下的降并发重试一次（并发减半，压力通常来自
    同时 import numpy 的瞬时内存峰值）。第二次仍崩才把剩余任务标记失败。
    """
    tasks = build_tasks(request, meta, grids, symbols)
    total = len(tasks)
    if total == 0:
        return {
            "done": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0,
            "elapsed": 0.0,
            "workers": 0,
            "verdicts": {},
            "cancelled": False,
            "error": "",
        }
    base_workers = min(int(workers or request.workers or default_workers()), MAX_WORKERS, total)

    started = time.perf_counter()
    counts = {"done": 0, "failed": 0, "skipped": 0}
    verdicts: dict[str, int] = {}
    completed: set[str] = set()
    cancelled = False
    last_error = ""
    pending = tasks
    for attempt in range(2):
        if not pending:
            break
        workers_try = base_workers if attempt == 0 else max(1, base_workers // 2)
        try:
            cancelled = _run_round(
                pending,
                workers_try,
                total=total,
                started=started,
                counts=counts,
                verdicts=verdicts,
                completed=completed,
                on_result=on_result,
                on_progress=on_progress,
                cancel=cancel,
            )
            pending = []
        except BrokenProcessPool as exc:
            last_error = f"进程池崩溃：{exc}"
            pending = [task for task in pending if task["key"] not in completed]
        if cancelled:
            break

    if pending:
        counts["failed"] += len(pending)
        for task in pending:
            if on_result is not None:
                on_result(
                    _finish(
                        {
                            "symbol": task["symbol"],
                            "status": "error",
                            "error": last_error or "进程池不可用",
                        },
                        task,
                    )
                )
    return {
        "done": counts["done"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "total": total,
        "elapsed": round(time.perf_counter() - started, 1),
        "workers": base_workers,
        "verdicts": verdicts,
        "cancelled": cancelled,
        "error": last_error,
    }

"""S3 事件抽取：全指标 × 四周期 × 多 horizon 的事件聚合（可断点续跑）。

设计上只解决一个问题：**怎么在 156 个指标 / 近 900 个布尔条件列 / 4 个周期 / 90 个币
的量级下，把逐事件数据压成可以反复做统计的聚合量**。逐事件存原始收益会产出上亿行，
既跑不完也没法看，所以按评估手册的要求把每个 (指标, 列, horizon, 币) 压成一组充分聚合量：

| 存什么 | 为什么够用 |
|---|---|
| n / Σr / Σx / Σx² | 均值、标准误、t、正态近似 CI |
| 分段 Σx（4 段，按全局时间四分位） | 手册第 4 关「时间分段同号」 |
| 高于/低于基准的笔数与各自和 | 方向翻转（做多↔做空）后仍能算胜率与盈亏比 |
| Σcs / n_cs / Σr_cs | cross_section 超额只对**该时刻能对齐 ≥5 个币**的事件计算，否则分子分母口径不一致 |
| hit_bars / bars | 退化列（恒真或几乎不触发）必须先剔，否则排名被垃圾列占满 |

**按币 block bootstrap 不需要原始序列**：重抽的单位是币，统计量是币内事件均值，
per-symbol 聚合量就是它的最小充分表示。

三条硬性口径（写在这里是为了别偷偷改掉）：
1. 事件 = 布尔列的**上升沿**，且相邻事件间隔 ≥ horizon，前瞻窗口互不重叠；
2. 入场 `t+1` 开盘价，出场 `t+h` 收盘价；`bench` 用同币同周期同口径的**全体 bar** 期望收益；
3. `calculate()` 每个 (指标, 币) 只跑一次。老脚本 `signal_screen.py` 在 `candidate_columns`
   和主循环里各跑一遍，白付一倍算力。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .lab_sim import to_ms

DAY_MS = 86_400_000
HORIZONS = (6, 24, 96)
SEGMENTS = 4
MIN_CS_SYMBOLS = 5  # 某时刻可对齐的币少于这个数，横截面基准不可比

FIELDS = (
    "n", "sum_r", "sum_x", "sum_sq", "n_above", "sum_above", "n_below", "sum_below",
    "sum_cs", "n_cs", "sum_r_cs", "hit_bars",
    "seg_n0", "seg_n1", "seg_n2", "seg_n3",
    "seg_x0", "seg_x1", "seg_x2", "seg_x3",
)
NF = len(FIELDS)
FI = {name: i for i, name in enumerate(FIELDS)}

_STATE: dict = {}


@dataclass
class CacheLayout:
    """缓存寻址：root/<scope哈希>/<周期>/<币>.npz。哈希进路径，换口径不会误用旧缓存。"""

    root: Path
    scope_sha: str
    interval: str

    @property
    def dir(self) -> Path:
        p = Path(self.root) / self.scope_sha / self.interval
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def csbench(self) -> Path:
        return self.dir / "csbench.npz"

    def symbol_file(self, symbol: str) -> Path:
        return self.dir / f"{symbol}.npz"


def rising_edges(mask: np.ndarray, min_gap: int) -> np.ndarray:
    """上升沿 + 最小间隔去重。min_gap=horizon 保证前瞻窗口不重叠。"""
    mask = np.asarray(mask, dtype=bool)
    prev = np.concatenate(([False], mask[:-1]))
    edges = np.flatnonzero(mask & ~prev)
    if edges.size == 0:
        return edges
    kept = [int(edges[0])]
    for pos in edges[1:]:
        if pos - kept[-1] >= min_gap:
            kept.append(int(pos))
    return np.asarray(kept, dtype=int)


def forward_returns(open_: np.ndarray, close: np.ndarray, horizon: int) -> np.ndarray:
    """t 收盘决策 → t+1 开盘进场 → t+h 收盘出场。尾部不够位置的置 nan。"""
    open_ = np.asarray(open_, dtype=float)
    close = np.asarray(close, dtype=float)
    n = close.size
    out = np.full(n, np.nan)
    if n <= horizon + 1:
        return out
    idx = np.arange(0, n - horizon)
    entry = open_[idx + 1]
    exit_ = close[idx + horizon]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[idx] = exit_ / entry - 1.0
    bad = ~np.isfinite(entry) | (entry <= 0) | ~np.isfinite(exit_)
    out[idx[bad]] = np.nan  # 掩码是 idx 长度，必须映射回原下标，否则尾部错位甚至越界
    return out


def bool_column_positions(out: pd.DataFrame) -> list[tuple[int, str]]:
    """按**位置**取布尔列，避免重名列在 `out[cols]` 上炸掉。"""
    hits = []
    for j, name in enumerate(out.columns):
        # 按位置取列：重名列（Pine 移植里真实存在）用 out[name] 会返回 DataFrame，
        # 于是 dtype 判断直接炸掉
        series = out.iloc[:, j]
        if series.dtype == bool:
            hits.append((j, str(name)))
            continue
        if series.dtype == object:
            values = series.dropna().unique()
            if len(values) and all(isinstance(v, (bool, np.bool_)) for v in values):
                hits.append((j, str(name)))
    return hits


def accumulate(pos: np.ndarray, r: np.ndarray, x: np.ndarray, cs: np.ndarray,
               seg: np.ndarray, hit_bars: int) -> np.ndarray:
    """把某个 (列, horizon, 币) 的事件压成一行聚合量。"""
    row = np.zeros(NF, dtype=np.float64)
    row[FI["hit_bars"]] = hit_bars
    if pos.size == 0:
        return row
    row[FI["n"]] = pos.size
    row[FI["sum_r"]] = float(np.nansum(r))
    row[FI["sum_x"]] = float(np.nansum(x))
    row[FI["sum_sq"]] = float(np.nansum(x * x))
    above, below = x > 0, x < 0
    row[FI["n_above"]] = int(above.sum())
    row[FI["sum_above"]] = float(x[above].sum())
    row[FI["n_below"]] = int(below.sum())
    row[FI["sum_below"]] = float(x[below].sum())
    cs_ok = np.isfinite(cs)  # 只有该时刻能对齐篮子的事件才进 cross_section 口径
    row[FI["sum_cs"]] = float(cs[cs_ok].sum()) if cs_ok.any() else 0.0
    row[FI["n_cs"]] = int(cs_ok.sum())
    row[FI["sum_r_cs"]] = float(r[cs_ok].sum()) if cs_ok.any() else 0.0
    for s in range(SEGMENTS):
        sel = seg == s
        row[FI[f"seg_n{s}"]] = int(sel.sum())
        row[FI[f"seg_x{s}"]] = float(x[sel].sum())
    return row


# --------------------------------------------------------------------------- 父进程侧


def build_cs_bench(frames: dict[str, pd.DataFrame], horizons=HORIZONS) -> dict[str, np.ndarray]:
    """同一时刻全池等权篮子的前瞻收益，作为 cross_section 基准。

    逐币先在**自己的时间栅格**上算 r_h（避免用并集栅格 shift 时把缺口当成下一根），
    再按 open_time 散列到全局索引上取均值。
    """
    if not frames:
        return {}
    index = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
    payload: dict[str, np.ndarray] = {"time_ms": to_ms(index).astype("int64")}
    for h in horizons:
        mat = np.full((len(index), len(frames)), np.nan)
        for j, (symbol, frame) in enumerate(frames.items()):
            r = forward_returns(frame["open"].to_numpy(float), frame["close"].to_numpy(float), h)
            where = np.searchsorted(payload["time_ms"], to_ms(frame.index), side="left")
            ok = where < len(index)
            mat[where[ok], j] = r[ok]
        count = np.isfinite(mat).sum(axis=1)
        total = np.nansum(mat, axis=1)
        mean = np.where(count >= MIN_CS_SYMBOLS, total / np.maximum(count, 1), np.nan)
        payload[f"h{h}"] = mean.astype("float64")
    return payload


def _init_worker(cs_path: str, seg_edges, cfg: dict):
    data = np.load(cs_path)
    _STATE["time_ms"] = data["time_ms"]
    _STATE["cs"] = {int(key[1:]): data[key] for key in data.files if key.startswith("h")}
    _STATE["seg_edges"] = np.asarray(seg_edges, dtype="int64")
    _STATE["cfg"] = cfg


def _cs_at(times_ms: np.ndarray, h: int) -> np.ndarray:
    table = _STATE["cs"][h]
    idx = np.searchsorted(_STATE["time_ms"], times_ms)
    idx = np.clip(idx, 0, len(table) - 1)
    match = _STATE["time_ms"][idx] == times_ms
    return np.where(match, table[idx], np.nan)


def aggregate_symbol(args) -> dict:
    """单个 (周期, 币) 的全指标事件聚合，在子进程里执行。"""
    from sqlalchemy.orm import sessionmaker

    from app.catalog_indicators import list_indicators
    from app.db import make_engine
    from app.market.data_service import MarketDataService
    from app.research import fit_directions as fdirections

    symbol, interval, start_ms, end_ms, horizons = args
    if "svc" not in _STATE:
        engine = make_engine(read_only=True)
        _STATE["svc"] = MarketDataService(sessionmaker(bind=engine)())
    svc = _STATE["svc"]
    # 方向表已经枚举过一遍：注册表里 55/156 个指标根本不产出布尔列（或依赖外部
    # 序列而必然报错），它们占掉 12.6% 的 calculate 时间却一列都不用。跳过它们不
    # 改变任何口径——列集合由 dtype 决定，与币种/周期无关，探测帧已确认稳定。
    keep = _STATE["cfg"].get("keep_names")
    started = time.time()
    frame = svc.load_frame(symbol, interval, limit=0, start_ms=start_ms, end_ms=end_ms)
    if frame.empty or len(frame) < max(horizons) * 3:
        return {"symbol": symbol, "status": "short", "bars": int(len(frame)), "cols": {}}
    open_ = frame["open"].to_numpy(float)
    close = frame["close"].to_numpy(float)
    times_ms = to_ms(frame.index).astype("int64")
    seg_edges = _STATE["seg_edges"]
    r_table = {h: forward_returns(open_, close, h) for h in horizons}
    bench = {h: float(np.nanmean(r_table[h])) for h in horizons}

    cols_out: dict[str, dict] = {}
    timings: dict[str, float] = {}
    skipped = 0
    for ind in list_indicators():
        if keep is not None and ind.name not in keep:
            skipped += 1
            continue
        t0 = time.time()
        try:
            out = ind.calculate(frame)
        except Exception:  # noqa: BLE001 — 依赖外部序列的指标按老口径直接跳过，不伪装成 0 事件
            timings[ind.name] = round(time.time() - t0, 3)
            continue
        timings[ind.name] = round(time.time() - t0, 3)
        positions = bool_column_positions(out)
        for j, col in positions:
            mask = out.iloc[:, j].to_numpy(dtype=bool)
            hit = int(mask.sum())
            agg = np.zeros((len(horizons), NF), dtype=np.float64)
            for hi, h in enumerate(horizons):
                r = r_table[h]
                pos = rising_edges(mask, h)
                if pos.size:
                    pos = pos[np.isfinite(r[pos])]
                x_all = r[pos] - bench[h] if pos.size else np.asarray([], dtype=float)
                cs = _cs_at(times_ms[pos], h) if pos.size else np.asarray([], dtype=float)
                seg = (np.searchsorted(seg_edges, times_ms[pos], side="right") if pos.size
                       else np.asarray([], dtype=int))
                seg = np.clip(seg, 0, SEGMENTS - 1)
                agg[hi] = accumulate(pos, r[pos] if pos.size else r[:0], x_all, cs, seg, hit)
            info = fdirections.direction_of(col, ind.name)
            cols_out[f"{ind.name}|{col}"] = {
                "direction": info["direction"],
                "direction_source": info["source"],
                "matched": info["matched"],
                "agg": agg,
            }
    return {
        "symbol": symbol,
        "status": "ok",
        "bars": int(len(frame)),
        "first_ms": int(times_ms[0]),
        "last_ms": int(times_ms[-1]),
        "bench": {str(h): bench[h] for h in horizons},
        "elapsed": round(time.time() - started, 1),
        "timings": timings,
        "skipped": skipped,
        "cols": cols_out,
    }


def save_symbol(path: Path, res: dict, horizons=HORIZONS) -> None:
    """按序号存数组（`a0..aN`），列名放进 meta。

    不直接用 `指标|列名` 当 npz 条目名：列名里出现 `/`、`\\` 或 `.` 会让 zip 条目变得
    含混（甚至和 `__meta__` 冲突），而序号 + meta 映射既能保证一致又不受字符集影响。
    """
    keys = sorted(res["cols"])
    arrays = {f"a{i}": res["cols"][k]["agg"].astype("float32") for i, k in enumerate(keys)}
    meta = {
        "symbol": res["symbol"],
        "bars": res["bars"],
        "nh": len(horizons),
        "horizons": list(horizons),
        "first_ms": res.get("first_ms"),
        "last_ms": res.get("last_ms"),
        "bench": res.get("bench"),
        "elapsed": res.get("elapsed"),
        "timings": res.get("timings"),
        "keys": keys,
        "columns": {
            k: {"direction": res["cols"][k]["direction"],
                 "direction_source": res["cols"][k]["direction_source"],
                 "matched": res["cols"][k]["matched"]}
            for k in keys
        },
    }
    np.savez_compressed(path, __meta__=np.array([json.dumps(meta, ensure_ascii=False)]), **arrays)


def run_interval(db, svc, interval: str, symbols: list[str], days: int, layout: CacheLayout,
                 workers: int = 10, horizons=HORIZONS, resume: bool = True,
                 log=print, progress_every: int = 2,
                 keep_names: set[str] | None = None,
                 expect_keys: int | None = None) -> dict:
    """先建 cross_section 基准（需要各币 open/close），再按币并行抽事件。"""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * DAY_MS
    todo = [s for s in symbols if not (resume and layout.symbol_file(s).exists())]

    if not layout.csbench.exists():
        t0 = time.time()
        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = svc.load_frame(symbol, interval, limit=0, start_ms=start_ms, end_ms=end_ms)
            if frame.empty or len(frame) < max(horizons) * 3:
                continue
            frames[symbol] = frame[["open", "close"]]
        payload = build_cs_bench(frames, horizons)
        np.savez_compressed(layout.csbench, **payload)
        log(f"[csbench] {interval}: {len(frames)} 币 / {len(payload['time_ms'])} 时刻，"
            f"{time.time() - t0:.0f}s")
        del frames, payload

    seg_edges = np.linspace(start_ms, end_ms, SEGMENTS + 1, dtype="int64")[1:-1]
    log(f"[events ] {interval}: {len(symbols)} 币，待算 {len(todo)}，缓存命中 {len(symbols) - len(todo)}")
    started = time.time()
    done = 0
    failures: list[str] = []
    if todo:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(layout.csbench), seg_edges, {"keep_names": keep_names}),
        ) as pool:
            futures = {
                pool.submit(aggregate_symbol, (s, interval, start_ms, end_ms, horizons)): s
                for s in todo
            }
            for fut in as_completed(futures):
                symbol = futures[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{symbol}:{type(exc).__name__}")
                    log(f"  !! {symbol} 失败 {type(exc).__name__}: {exc}")
                    continue
                if res.get("status") != "ok" or not res["cols"]:
                    failures.append(f"{symbol}:{res.get('status')}")
                    continue
                save_symbol(layout.symbol_file(symbol), res, horizons)
                done += 1
                if done == 1 and expect_keys is not None and len(res["cols"]) != expect_keys:
                    log(f"  !! {symbol} 产出 {len(res['cols'])} 列，方向表期望 {expect_keys} 列"
                        f"——按名字跳过指标不成立，检查 keep_names")
                if done % progress_every == 0 or done == len(todo):
                    rate = done / max(time.time() - started, 1e-6)
                    eta = (len(todo) - done) / max(rate, 1e-6) / 60
                    log(f"  [{done}/{len(todo)}] {symbol} {res['elapsed']}s "
                        f"| 累计 {time.time() - started:.0f}s，剩约 {eta:.0f} 分钟")
    return {
        "interval": interval,
        "symbols": len(symbols),
        "computed": done,
        "cached": len(symbols) - len(todo),
        "failures": failures,
        "seconds": round(time.time() - started, 1),
    }


def load_cached(layout: CacheLayout, symbols: list[str], horizons=HORIZONS):
    """把逐币 npz 拼成 `fit_stats.Evaluator` 要的张量。

    返回 `(keys, syms, matrix, covered, meta, sym_meta)`：
    - `matrix` 形状 (列, 币, horizon, 字段)；未覆盖的位置是 0；
    - `covered[k, s]` 表示这一列在这个币上**存在**（不是「事件数为 0」）。
      区分这两件事很重要：某个指标只在部分币上返回布尔列，把缺失当 0 会把池化均值
      系统性拉向 0，也会让触发率的分母虚高。
    - 缺缓存文件的币直接跳过，`sym_meta` 带回每个币的 bar 数与该币的无条件基准。
    """
    blocks: dict[str, dict[str, np.ndarray]] = {}
    key_meta: dict[str, dict] = {}
    sym_meta: dict[str, dict] = {}
    order: list[str] = []
    for symbol in symbols:
        path = layout.symbol_file(symbol)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            sm = json.loads(str(data["__meta__"][0]))
            if int(sm.get("nh", -1)) != len(horizons) or list(sm.get("horizons", [])) != list(horizons):
                continue  # 口径变了，旧缓存不可用（scope 哈希也变了，这里只是双保险）
            col_keys = sm.get("keys") or []
            blocks[symbol] = {
                col_keys[i]: data[f"a{i}"]
                for i in range(len(col_keys))
                if f"a{i}" in data.files and i < len(col_keys)
            }
            order.append(symbol)
            sym_meta[symbol] = {"bars": sm["bars"], "bench": sm.get("bench"),
                                "elapsed": sm.get("elapsed"), "timings": sm.get("timings")}
            for key, col in (sm.get("columns") or {}).items():
                key_meta.setdefault(key, dict(col))
    keys = sorted(key_meta)
    syms = [s for s in order if s in blocks]
    matrix = np.zeros((len(keys), len(syms), len(horizons), NF), dtype=np.float64)
    covered = np.zeros((len(keys), len(syms)), dtype=bool)
    key_pos = {k: i for i, k in enumerate(keys)}
    for j, symbol in enumerate(syms):
        for key, arr in blocks[symbol].items():
            i = key_pos.get(key)
            if i is None or arr.shape != (len(horizons), NF):
                continue
            covered[i, j] = True
            matrix[i, j] = arr
    return keys, syms, matrix, covered, key_meta, sym_meta

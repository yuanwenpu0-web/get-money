"""S3–S4 统计：五道关判定、池分层、交互检验、收缩估计。

相比 `backend/scripts/signal_screen.py` 的三关，这里补掉了评估手册点名的四个缺口：

1. **方向对齐**：`excess = dir × (r − bench) − cost`。看跌列（`strong_sell`/`bear_cross`）
   在老口径下永远算不出正 edge，因为它们的收益本来就是负的。`dir` 只来自列名语义
   （`fit_directions`），不用被测收益反推。
2. **基准不是 0**：同时报 `unconditional`（同币同周期全体 bar 期望）与
   `cross_section`（同一时刻一篮子等权期望）。400 天普涨窗口里 64/132「净为正」有多少是
   beta，不减基准无法区分。
3. **按币 block bootstrap**：重抽单位是币。事件级 iid 重抽在同币强自相关下会把 CI 做得
   过窄（p 值虚低），这是手册第 1 关的硬要求。因为 `fit_events` 已把每个
   (列, horizon, 币) 压成 (n, Σstat) 充分统计量，重抽币只需两次矩阵乘。
4. **分层不许挑格子**：格子结论必须过交互检验（cell 与其余样本之差的 bootstrap p），
   并按 `n/(n+k)` 向池化均值收缩；事件 <100 或覆盖币 <5 的格子只报样本量不报结论。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fit_events import FI, SEGMENTS

ROUND_TRIP = 0.0012
BENCH_KEYS = ("unconditional", "cross_section")


@dataclass
class GateConfig:
    n_boot: int = 2000
    seed: int = 20261112
    q_alpha: float = 0.10
    interaction_alpha: float = 0.10
    seg_sign_min: float = 0.75
    sym_sign_min: float = 0.60
    cell_events_min: int = 100
    cell_symbols_min: int = 5
    pooled_events_min: int = 30
    sym_events_min: int = 10
    shrink_k: float = 30.0
    min_hit: float = 0.002
    max_hit: float = 0.60
    cost_stress: tuple[float, ...] = (1.0, 2.0, 3.0)
    # 币块自助法的 p 值有下界 1/(n_boot+1)：只要参与统计的 k 个币「同号且重抽不翻负」，
    # p 就打到地板。全局零假设下这件事的概率约 2^(1-k)——k=6 时 3%，一个 900 检验的家族
    # 里会有 ~28 行白白打到地板，BH 再把它们几乎全判成发现。6 币探针实测出 116 个
    # candidate，就是这个离散性造出来的，不是 edge。
    # 取 15：2^(1-15)=6e-5，乘以家族规模 ~900 后期望假地板 <0.1 行，BH 的 FDR 声明才成立。
    # 这条只压 `candidate`（会写进台账、可能被接进生产的那一级）；币数不足但方向已声明的
    # 行仍保留 exploratory/cost_fragile 等标签，币种池层面的差异照看得到。
    blocks_for_candidate: int = 15


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def resample_counts(n_symbols: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """cnt[b, s] = 第 b 次重抽中币 s 出现的次数（有放回抽 n_symbols 个币）。"""
    if n_symbols <= 0:
        return np.zeros((n_boot, 0), dtype=np.float32)
    idx = rng.integers(0, n_symbols, size=(n_boot, n_symbols))
    cnt = np.zeros((n_boot, n_symbols), dtype=np.float32)
    rows = np.repeat(np.arange(n_boot, dtype=np.int64), n_symbols)
    np.add.at(cnt, (rows, idx.ravel()), 1.0)
    return cnt


def boot_p(values: np.ndarray) -> float:
    """双侧置换式 p：看重抽分布跨过 0 的程度。至少给 1/(B+1) 的下界，别报 0。

    用 ≤0 / ≥0 而不是 <0 / >0：格子与池子分布完全相同时重抽差值可能**恒等于 0**，
    旧的严格不等式会把 p 压到下界 1/(B+1)，等于「一点差异都没有」被判成显著。
    """
    b = values.size
    if b == 0:
        return float("nan")
    frac_le = float((values <= 0).mean())
    frac_ge = float((values >= 0).mean())
    p = min(2.0 * min(frac_le, frac_ge), 1.0)
    return max(p, 1.0 / (b + 1.0))


class Evaluator:
    """把一个周期内缓存的聚合量变成判定表。

    输入张量契约见 `fit_events.load_cached`：`matrix` 是 (列, 币, horizon, 字段)，
    `covered` 是 (列, 币) 布尔。scope 通过币下标切换，不重算缓存。
    """

    def __init__(self, keys: list[str], symbols: list[str], matrix: np.ndarray,
                 covered: np.ndarray, meta: dict[str, dict], bars: dict[str, int],
                 cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        self.keys = keys
        self.symbols = symbols
        self.matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        self.covered = covered.astype(bool)
        self.meta = meta
        self.bars = bars
        self.sym_pos = {s: i for i, s in enumerate(symbols)}
        self._rng = np.random.default_rng(self.cfg.seed)
        self._cnt: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ 内部工具
    def _counts(self, n_sel: int) -> np.ndarray:
        if n_sel not in self._cnt:
            self._cnt[n_sel] = resample_counts(n_sel, self.cfg.n_boot, self._rng)
        return self._cnt[n_sel]

    @staticmethod
    def _slice(M: np.ndarray, C: np.ndarray, field: str) -> np.ndarray:
        """取某字段的 (列, horizon, 币) 切片，未覆盖的格子强制为 0。

        `M` 的契约是 (列, 币, horizon, 字段)（见 `fit_events.load_cached`），先按字段
        取轴再 transpose，是为了让下面的向量化都统一成 (列, horizon, 币)。
        """
        out = M[:, :, :, FI[field]].transpose(0, 2, 1)
        return np.where(C[:, None, :], out, 0.0)

    # ------------------------------------------------------------------ 主入口
    def evaluate(self, horizons: list[int], scopes: dict[str, list[str]],
                 atr_pct: dict[str, float] | None = None) -> list[dict]:
        """对每个 (scope, horizon, 列, 方向, 基准) 组合出一行判定所需的全部统计量。

        只在该列**有事件的币**上做池化：缓存矩阵里未覆盖的位置是 0，直接求和会把
        「这个指标在这个币上根本不存在」当成「没有超额收益」，那会系统性地把均值拉向 0。
        触发率的分母同样只算覆盖币的 bar 数。
        """
        cfg = self.cfg
        atr_pct = atr_pct or {}
        rows: list[dict] = []
        bars_all = np.array([float(self.bars.get(s, 0)) for s in self.symbols])
        for scope, sel in scopes.items():
            cols = [self.sym_pos[s] for s in sel if s in self.sym_pos]
            if len(cols) < 2:
                continue
            M = self.matrix[:, cols, :]
            C = self.covered[:, cols]
            cnt = self._counts(len(cols))
            sel_syms = [self.symbols[c] for c in cols]
            bars_vec = bars_all[cols]
            n_i = self._slice(M, C, "n")
            n_cs = self._slice(M, C, "n_cs")
            sum_x = self._slice(M, C, "sum_x")
            sum_r_cs = self._slice(M, C, "sum_r_cs")
            sum_cs = self._slice(M, C, "sum_cs")
            sum_sq = self._slice(M, C, "sum_sq")
            hit = self._slice(M, C, "hit_bars")
            n_above = self._slice(M, C, "n_above")
            sum_above = self._slice(M, C, "sum_above")
            n_below = self._slice(M, C, "n_below")
            sum_below = self._slice(M, C, "sum_below")
            seg_n = np.stack(
                [self._slice(M, C, f"seg_n{s}") for s in range(SEGMENTS)], axis=-1
            )
            seg_x = np.stack(
                [self._slice(M, C, f"seg_x{s}") for s in range(SEGMENTS)], axis=-1
            )
            # 交互检验用的「池外样本」：同周期里不属于本 scope 的币
            in_scope = set(sel)
            rest_syms = [s for s in self.symbols if s not in in_scope]
            rest = None
            if rest_syms and scope != "all":
                rest_cols = [self.sym_pos[s] for s in rest_syms]
                Mr = self.matrix[:, rest_cols, :]
                Cr = self.covered[:, rest_cols]
                rest = {
                    "n": self._slice(Mr, Cr, "n"),
                    "n_cs": self._slice(Mr, Cr, "n_cs"),
                    "sum_x": self._slice(Mr, Cr, "sum_x"),
                    "sum_r_cs": self._slice(Mr, Cr, "sum_r_cs"),
                    "sum_cs": self._slice(Mr, Cr, "sum_cs"),
                    "cnt": self._counts(len(rest_cols)),
                    "syms": rest_syms,
                }

            for hi, h in enumerate(horizons):
                n_kh = n_i[:, hi, :]
                bars_cov = C @ bars_vec
                hit_rate = np.where(
                    bars_cov > 0, hit[:, hi, :].sum(axis=1) / np.maximum(bars_cov, 1.0), np.nan
                )
                n_events = n_kh.sum(axis=1)
                syms_with = (n_kh > 0).sum(axis=1)
                for k, key in enumerate(self.keys):
                    if not C[k].any() or n_events[k] < cfg.pooled_events_min:
                        continue
                    if not (cfg.min_hit <= float(hit_rate[k]) <= cfg.max_hit):
                        continue
                    info = self.meta.get(key, {})
                    raw_dir = int(info.get("direction", 0))
                    variants = [1, -1] if raw_dir == 0 else [raw_dir]
                    indicator, _, column = key.partition("|")
                    for d in variants:
                        for bench in BENCH_KEYS:
                            row = self._row(
                                key=key, indicator=indicator, column=column, scope=scope,
                                horizon=h, hi=hi, k=k, direction=d, bench=bench,
                                cnt=cnt,
                                per_sym={
                                    "n": n_i[k, hi], "n_cs": n_cs[k, hi],
                                    "sum_x": sum_x[k, hi], "sum_sq": sum_sq[k, hi],
                                    "sum_r_cs": sum_r_cs[k, hi], "sum_cs": sum_cs[k, hi],
                                    "n_above": n_above[k, hi], "sum_above": sum_above[k, hi],
                                    "n_below": n_below[k, hi], "sum_below": sum_below[k, hi],
                                    "seg_n": seg_n[k, hi], "seg_x": seg_x[k, hi],
                                },
                                raw_dir=raw_dir, info=info,
                                syms_with=int(syms_with[k]), covered=int(C[k].sum()),
                                hit_rate=float(hit_rate[k]), sel_syms=sel_syms,
                                rest=rest, atr_pct=atr_pct,
                            )
                            if row:
                                rows.append(row)
        return rows

    def _row(self, *, key, indicator, column, scope, horizon, hi, k, direction, bench,
             cnt, per_sym, raw_dir, info, syms_with, covered, hit_rate,
             sel_syms, rest, atr_pct) -> dict:
        cfg = self.cfg
        # 口径分叉只发生在这里：unconditional 用全部事件，cross_section 只用「该时刻
        # 能对齐 ≥MIN_CS_SYMBOLS 个币」的那部分事件（分母换成 n_cs，分子换成 Σr_cs−Σcs），
        # 否则分子分母覆盖的事件集合不一致，超额收益就是假的。
        n_i = np.asarray(per_sym["n"] if bench == "unconditional" else per_sym["n_cs"], dtype=float)
        if bench == "unconditional":
            stat_sym = direction * np.asarray(per_sym["sum_x"], dtype=float)
        else:
            stat_sym = direction * (
                np.asarray(per_sym["sum_r_cs"], dtype=float)
                - np.asarray(per_sym["sum_cs"], dtype=float)
            )
        stat_sym = np.nan_to_num(stat_sym)
        n = float(n_i.sum())
        if n < cfg.pooled_events_min:
            return {}
        mean_gross = float(stat_sym.sum() / n)
        mean_net = mean_gross - ROUND_TRIP
        with np.errstate(divide="ignore", invalid="ignore"):
            den = cnt @ n_i
            # 一次重抽可能一个有事件的币都没抽到：那不属于样本，置 NaN 丢掉，
            # 不能当成「收益 0、净 −成本」的观察值参与 CI 与 p。
            reps = np.where(den > 0, (cnt @ stat_sym) / np.maximum(den, 1e-9), np.nan)
        reps_net = reps - ROUND_TRIP
        good = np.isfinite(reps_net)
        if int(good.sum()) < 100:
            return {}
        lo, hi_ci = np.nanpercentile(reps_net[good], [2.5, 97.5])
        p = boot_p(reps_net[good])
        # sum_sq 只按 unconditional 事件缓存，所以 cross_section 行的 se_iid 是近似
        # （分子分母的事件集合略有差异）。判定链条只看 block bootstrap 的 CI。
        var_iid = max(float(per_sym["sum_sq"].sum()) / n - mean_gross**2, 0.0) / n
        se_iid = math.sqrt(var_iid)

        # 胜率/盈亏比按「是否跑赢无条件基准」计（x>0 / x<0），不是绝对盈亏。
        # 缓存里没有按 cs 可用性切分的above/below，所以 cross_section 行不报这两个量。
        n_all = float(np.asarray(per_sym["n"], dtype=float).sum())
        if direction > 0:
            win_n, loss_n = float(per_sym["n_above"].sum()), float(per_sym["n_below"].sum())
            win_sum, loss_sum = float(per_sym["sum_above"].sum()), float(per_sym["sum_below"].sum())
        else:
            win_n, loss_n = float(per_sym["n_below"].sum()), float(per_sym["n_above"].sum())
            win_sum, loss_sum = -float(per_sym["sum_below"].sum()), -float(per_sym["sum_above"].sum())
        win_rate = win_n / n_all if n_all else float("nan")
        avg_win = win_sum / win_n if win_n > 0 else float("nan")
        avg_loss = abs(loss_sum) / loss_n if loss_n > 0 else float("nan")
        payoff = (avg_win / avg_loss
                  if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss > 0
                  else float("nan"))

        with np.errstate(divide="ignore", invalid="ignore"):
            sym_mean = np.where(
                n_i >= cfg.sym_events_min,
                stat_sym / np.maximum(n_i, 1.0),
                np.nan,
            )
        usable = np.isfinite(sym_mean)
        overall_sign = float(np.sign(mean_net))
        sym_sign = float((np.sign(sym_mean[usable]) == overall_sign).mean()) if usable.any() else float("nan")
        n_syms_usable = int(usable.sum())

        # 分段同号：缓存里只有 unconditional 基准的分段 Σx，两种基准都用它做时间稳定性
        # 判断（报告里要写明这个口径，不是笔误）。有效段不足 2 个时写 None，由 verdict
        # 按「测不到稳定性 = 不通过」处理，而不是当成满分。
        seg_n_tot = np.asarray(per_sym["seg_n"], dtype=float).sum(axis=0)
        seg_x_tot = np.asarray(per_sym["seg_x"], dtype=float).sum(axis=0)
        seg_same = seg_valid = 0
        for s in range(SEGMENTS):
            sn = float(seg_n_tot[s])
            if sn <= 0:
                continue
            seg_valid += 1
            seg_mean = direction * float(seg_x_tot[s]) / sn - ROUND_TRIP
            if np.sign(seg_mean) == overall_sign:
                seg_same += 1
        seg_cons = seg_same / seg_valid if seg_valid >= 2 else float("nan")

        stress = {f"x{m:g}": round((mean_gross - ROUND_TRIP * m) * 100, 4) for m in cfg.cost_stress}
        atr = _mean_atr(atr_pct, sel_syms)
        cost_ratio = ROUND_TRIP / (atr * math.sqrt(horizon)) if atr else float("nan")

        interaction = None
        if rest is not None:
            r_n = np.asarray(rest["n"][k, hi, :], dtype=float)
            r_cs = np.asarray(rest["n_cs"][k, hi, :], dtype=float)
            r_den = r_n if bench == "unconditional" else r_cs
            if bench == "unconditional":
                stat_r = direction * np.asarray(rest["sum_x"][k, hi, :], dtype=float)
            else:
                stat_r = direction * (
                    np.asarray(rest["sum_r_cs"][k, hi, :], dtype=float)
                    - np.asarray(rest["sum_cs"][k, hi, :], dtype=float)
                )
            stat_r = np.nan_to_num(stat_r)
            nr_total = float(r_den.sum())
            if nr_total >= cfg.pooled_events_min:
                with np.errstate(divide="ignore", invalid="ignore"):
                    da, db = cnt @ n_i, rest["cnt"] @ r_den
                    a = np.where(da > 0, (cnt @ stat_sym) / np.maximum(da, 1e-9), np.nan)
                    b = np.where(db > 0, (rest["cnt"] @ stat_r) / np.maximum(db, 1e-9), np.nan)
                # 只有「这一抽里两组都至少有一个有事件的币」才构成一次可比观测；
                # 拿 0 补位会把 p 做得虚高，所以直接丢掉。
                diff = a - b
                diff = diff[np.isfinite(diff)]
                if diff.size >= 100:
                    interaction = {
                        "diff_pct": round((mean_gross - float(stat_r.sum() / nr_total)) * 100, 4),
                        "p": boot_p(diff),
                        "rest_events": int(nr_total),
                        "rest_symbols": int((r_den > 0).sum()),
                    }

        return {
            "indicator": indicator,
            "column": column,
            "key": key,
            "scope": scope,
            "horizon_bars": horizon,
            "direction": direction,
            "direction_source": info.get("direction_source") if raw_dir else "ambiguous",
            "direction_matched": info.get("matched", []),
            "benchmark": bench,
            "events": int(n),
            "events_all_bars": int(n_all),
            "cs_coverage": round(n / n_all, 4) if n_all else None,
            "symbols_with_events": syms_with,
            "symbols_covered": covered,
            "symbols_in_stat": n_syms_usable,
            "hit_rate": round(hit_rate, 5),
            "mean_gross_pct": round(mean_gross * 100, 4),
            "mean_net_pct": round(mean_net * 100, 4),
            "ci_low_pct": round(float(lo) * 100, 4),
            "ci_high_pct": round(float(hi_ci) * 100, 4),
            "p_bootstrap": round(float(p), 6),
            "se_iid_pct": round(se_iid * 100, 4),
            "win_rate": round(win_rate, 4) if bench == "unconditional" else None,
            "payoff_ratio": round(payoff, 3) if np.isfinite(payoff) else None,
            "seg_consistency": round(seg_cons, 3) if np.isfinite(seg_cons) else None,
            "seg_valid": seg_valid,
            "symbol_sign_rate": round(sym_sign, 3) if np.isfinite(sym_sign) else None,
            "cost_stress_pct": stress,
            "cost_ratio": round(cost_ratio, 3) if np.isfinite(cost_ratio) else None,
            "interaction": interaction,
        }


def _mean_atr(atr_pct: dict[str, float], syms: list[str]) -> float:
    vals = [atr_pct[s] for s in syms if s in atr_pct]
    return float(np.mean(vals)) if vals else 0.0


def apply_fdr(rows: list[dict]) -> list[dict]:
    """在 (scope, horizon, benchmark) 家族内做 BH 校正，再打判定。

    调用方按周期分别调用 `stage_stats`，所以家族实际是 (周期, 池, h, 基准)——跨周期的
    同一列不构成同一家族的两次试验，报告 §2.1 与 §9 都按这个口径写。
    """
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["scope"], r["horizon_bars"], r["benchmark"])].append(r)
    for _, group in groups.items():
        q = bh_qvalues(np.array([r["p_bootstrap"] for r in group], dtype=float))
        for r, qq in zip(group, q):
            r["q_value"] = round(float(qq), 5)
            r["verdict"] = verdict(r)
    return rows


def verdict(row: dict, cell: bool = False) -> str:
    """五道关 + 方向可信度上限（报告 §2.1 的表就是这个函数的次序）。

    1 效应（净均值>0 且 CI 下界>0）→ 2 FDR q → 3 分段/币间同号 → 4 2× 成本 →
    5 币块下限 `symbols_in_stat ≥ blocks_for_candidate`。第 5 道在代码里是最后一道：
    前四道全过但币块不足 15 个，降级 `underpowered_blocks`——p 值会打到离散下界，
    那时 q<0.10 是免费的，不构成 FDR 成立（推导见 `GateConfig`）。

    `cell=True` 时额外要求格子门槛（事件数、覆盖币数、交互显著），这就是手册 5.3 的
    「不允许把『整体无效但某个子集有效』当正面结论」。
    """
    cfg = GateConfig()
    if row["mean_net_pct"] <= 0:
        return "no_edge"
    if row.get("ci_low_pct") is None or row["ci_low_pct"] <= 0:
        return "inconclusive"
    if row.get("q_value") is None or row["q_value"] >= cfg.q_alpha:
        return "likely_noise"
    if row.get("seg_valid", SEGMENTS) < 2:
        # 事件全挤在同一段时间：一致性无从谈起，第 4 关直接不过。写成 None 会被
        # 下一条判据放过，等于「测不到就等于通过」。
        return "unstable"
    if row.get("seg_consistency") is not None and row["seg_consistency"] < cfg.seg_sign_min:
        return "unstable"
    if row.get("symbol_sign_rate") is not None and row["symbol_sign_rate"] < cfg.sym_sign_min:
        return "not_universal"
    stress = row.get("cost_stress_pct") or {}
    if stress.get("x2", 0.0) <= 0:
        return "cost_fragile"
    if cell:
        if row["events"] < cfg.cell_events_min or row["symbols_with_events"] < cfg.cell_symbols_min:
            return "underpowered_cell"
        inter = row.get("interaction")
        if not inter or inter["p"] >= cfg.interaction_alpha:
            return "no_heterogeneity"
    if row.get("direction_source") in ("ambiguous", "weak"):
        # 方向是猜的（弱线索）或两向都试，就不配叫「候选」——只登记为待验证假设
        return "exploratory"
    if row.get("symbols_in_stat", 0) < cfg.blocks_for_candidate:
        # 自助法只有这么几个币块，p 值打到离散下界、BH 的 FDR 声明也不成立（见 GateConfig
        # 里 blocks_for_candidate 的推导）。数字照抄，标签不上「候选」。
        return "underpowered_blocks"
    return "candidate"


# 判定分级：越小越强。矩阵页的排序、配色、报告里「最能用的格子」都按这个次序，
# 所以定义放在判定器旁边，避免页面和统计各写一份再漂移。
VERDICT_RANK = {
    "candidate": 0,
    "cost_fragile": 1,
    "exploratory": 2,
    "underpowered_blocks": 3,
    "unstable": 4,
    "not_universal": 5,
    "likely_noise": 6,
    "inconclusive": 7,
    "no_edge": 8,
    "no_heterogeneity": 9,
    "underpowered_cell": 10,
    "insufficient": 11,
}
# 「说得出口」的级别：报告与矩阵页给明细，其余只在格子里留个判定名。
# underpowered_blocks 也在内——四道关全过，只是币块少，作为线索值得看。
PROMOTABLE = ("candidate", "cost_fragile", "exploratory", "underpowered_blocks")

# 周期的显示次序（从快到慢）。`sorted()` 给出的是 15m/1h/30m/5m 这种字典序，矩阵列一乱，
# 「这列适合哪个 K 线」就得靠读者自己重排。报告与矩阵页共用这里，别各写一份。
INTERVAL_ORDER = ["5m", "15m", "30m", "1h"]


def intervals_sorted(ivs) -> list[str]:
    seen = set(ivs)
    known = [iv for iv in INTERVAL_ORDER if iv in seen]
    return known + sorted(seen - set(INTERVAL_ORDER))


def level(row: dict) -> int:
    return VERDICT_RANK.get(row.get("verdict_cell") or row.get("verdict"), 99)


def best_cell(rows: list[dict]) -> dict | None:
    """同一个 (周期, 列, 池) 下，从 3 个预测长度 × 2 个基准里挑「最能用」的一条。

    先看判定等级，再看置信下界——`candidate` 必须 CI 下界>0，所以同级别里 ci_low 越大
    越可信。并列时取事件数多的（样本更足）。判定用 `verdict_cell`（格子口径），没有
    就退回池化口径——这样同一个函数既能喂池化表也能喂格子表。
    """
    if not rows:
        return None
    return min(
        rows,
        key=lambda r: (
            level(r),
            -(r.get("ci_low_pct") if r.get("ci_low_pct") is not None else -9e9),
            -int(r.get("events") or 0),
        ),
    )


def shrink(cell_row: dict, pooled_row: dict, k: float | None = None) -> dict:
    """把格子均值向池化均值收缩，w = n/(n+k)。事件数不够的格子会被拉回池化水平。"""
    kk = k if k is not None else GateConfig().shrink_k
    n = float(cell_row["events"])
    w = n / (n + kk)
    shrunk = w * cell_row["mean_net_pct"] + (1 - w) * pooled_row["mean_net_pct"]
    out = dict(cell_row)
    out["shrunk_net_pct"] = round(shrunk, 4)
    out["shrink_weight"] = round(w, 3)
    out["delta_vs_pooled_pct"] = round(cell_row["mean_net_pct"] - pooled_row["mean_net_pct"], 4)
    out["verdict_cell"] = verdict(cell_row, cell=True)
    return out

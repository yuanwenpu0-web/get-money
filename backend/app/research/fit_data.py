"""全指标测评的数据层：周期覆盖、质量门、事件时点的币种池分档。

为什么要单独一层
----------------
`docs/indicator-evaluation-plan.md` 的第 2.2 节把「数据质量门不通过就不许进评估」写成
硬性要求，第 5.1 节又要求「分组用事件时点的属性，不是评估末期的属性」。这两条在现有脚本里
都没有实现：`signal_screen.py` 只用 `count(*) > days*12` 粗筛币种，池化 40 个币就完事了。
本模块负责三件事：

1. 按周期实测覆盖（币种数、密度、跨度），给出可用于评估的币种集；
2. 跑质量门（步长、缺口、重复、OHLC 逻辑、零成交、极端插针、上市时长），逐币给出剔除原因；
3. 按**自然月**给币种做流动性分档（Q1 最高…Q4 最低），供「指标 × 池」分层使用，
   保证分层信息不是未来函数。

周期窗口是按数据现实定的，不是按对称好看定的：1m 只有约 90 天满覆盖（5m 只能派生到这里），
15m/4h 有 300+ 天，1h 有 400+ 天。跨周期比较时，评估侧还会再取币种交集，避免
「5m 看起来更好」其实只是换了批币。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

DAY_MS = 86_400_000

# 每个周期的评估窗口（天）。见模块 docstring：由本地数据覆盖决定。
WINDOW_DAYS = {"5m": 90, "15m": 300, "30m": 300, "1h": 400}
# 每周期最多参与评估的币种数：控制算力，同时保证池内格子样本量。
MAX_SYMBOLS = 90
# 密度门槛：**按该币自身可用跨度**算，实际 bar 数 / 跨度内应有 bar 数。
# 为什么不按整窗口算：同一批币上市时间不同，用「窗口应有根数」当分母会把
# 「上市 60 天的新币」判成密度 0.66 而剔除，于是四个周期的币种交集被新上市币
# 拖小（本轮 1h 有 38 个锚定币因此被误杀），跨周期比较的样本就不是同一批币了。
# 数据完整性看 own-span 密度，「覆盖够不够长」是另一道门：见 MIN_WINDOW_COVERAGE。
MIN_DENSITY = 0.85
# 窗口覆盖门槛：该币在本轮窗口内至少要有这么长的一段连续历史（首根到末根跨度 / 窗口）。
# 0.5 的取舍：再低就出现「5m 有 90 天、1h 只有 20 天」的错配，同一指标在不同周期上
# 比较的其实是不同时间段；再高则把新上市币几乎全清出去，Q4 档和跨周期交集都会塌。
MIN_WINDOW_COVERAGE = 0.5
# 历史长度门槛：不足这么多根就没有指标暖机与统计样本（5m 的 500 根≈42 小时，
# 与 MIN_WINDOW_COVERAGE 联合生效：短命币由覆盖率拦，长窗口周期由根数拦）。
MIN_HISTORY_BARS = 500


# 自然月「算完整」的最低柱数占比。窗口首尾月往往只盖到十几天，成交额天然偏低，
# 混进分档中位数会把币压到错误的档（尤其是窗口内才上市的币）。
MONTH_PARTIAL_FILL = 0.7


def _step_ms(interval: str) -> int:
    from ..utils import interval_ms

    return interval_ms(interval)


def month_expected_bars(ym: pd.Series, interval: str) -> pd.Series:
    """每个自然月在该周期下应有的柱数。

    坑（2026-11-13 修）：pandas 3.x 的 `.dt.days_in_month` 返回 **int32**，
    `30 * 86_400_000 = 2.59e9` 超过 int32 上限 2.147e9，乘完直接翻负
    （30 天 → -1702967296，//3600000 → -474）。于是 `bars < 0.7 * 负数` 恒为 False，
    `partial` 标记全库失效、残月从不被剔除，A 轮 Q 档中位数因此混进了窗口首尾残月。
    必须先升到 int64 再乘。单测覆盖 28/29/30/31 天 × 四个周期。
    """
    ym = pd.Series(ym)
    days_in_month = pd.to_datetime(ym.astype(str) + "-01", utc=True).dt.days_in_month
    return (days_in_month.astype("int64") * DAY_MS // _step_ms(interval)).astype("int64")


def mark_partial_months(frame: pd.DataFrame, interval: str,
                        min_fill: float = MONTH_PARTIAL_FILL) -> pd.DataFrame:
    """给月度成交额帧补 `expected_bars` / `partial`，返回副本（不改调用方的帧）。"""
    out = frame.copy()
    out["expected_bars"] = month_expected_bars(out["ym"], interval)
    out["partial"] = out["bars"] < min_fill * out["expected_bars"]
    return out


@dataclass
class SymbolQuality:
    """单币单周期的质量门结果。`reject` 非空即表示该币该周期不进评估。"""

    symbol: str
    bars: int
    expected: int
    density: float
    first_ms: int
    last_ms: int
    gaps: int = 0
    duplicates: int = 0
    ohlc_bad: int = 0
    zero_vol: int = 0
    pins: int = 0
    own_expected: int = 0
    win_coverage: float = 0.0
    reject: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bars": self.bars,
            "expected_bars": self.expected,
            "own_expected_bars": self.own_expected,
            "density": round(self.density, 4),
            "win_coverage": round(self.win_coverage, 4),
            "first_ms": self.first_ms,
            "last_ms": self.last_ms,
            "gaps": self.gaps,
            "duplicates": self.duplicates,
            "ohlc_bad": self.ohlc_bad,
            "zero_vol_bars": self.zero_vol,
            "extreme_pins": self.pins,
            "reject": self.reject,
        }


def coverage_rows(db: Session, interval: str, days: int, symbols: Iterable[str] | None = None) -> pd.DataFrame:
    """按币统计窗口内的覆盖：柱数、首尾时间。窗口以该周期最新一根为终点。"""
    step = _step_ms(interval)
    hi = db.execute(
        text("select max(open_time) from klines where interval=:iv"), {"iv": interval}
    ).scalar()
    if hi is None:
        return pd.DataFrame()
    lo = hi - days * DAY_MS
    params: dict = {"iv": interval, "lo": int(lo)}
    sql = (
        "select symbol, count(*) n, min(open_time) a, max(open_time) b, "
        "sum(coalesce(quote_volume, 0)) quote_volume "
        "from klines where interval=:iv and open_time>=:lo"
    )
    stmt_sql = sql
    if symbols:
        wanted = list(symbols)
        stmt_sql += " and symbol in :syms"
        params["syms"] = wanted
    stmt_sql += " group by symbol"
    stmt = text(stmt_sql)
    if symbols:
        stmt = stmt.bindparams(bindparam("syms", expanding=True))
    rows = db.execute(stmt, params).mappings().all()
    frame = pd.DataFrame([dict(r) for r in rows])
    if frame.empty:
        return frame
    frame["expected"] = int(days * DAY_MS / step) + 1
    # own-span：分母换成「该币自己首末根之间应有的根数」，新上市币不会因为
    # 「上市前没有行情」被判成数据不完整；窗口覆盖单独成一个字段，别混进密度。
    frame["own_expected"] = np.maximum((frame["b"] - frame["a"]) / step + 1, 1.0)
    frame["density"] = frame["n"] / frame["own_expected"]
    span_hi = np.minimum(frame["b"], int(hi))
    span_lo = np.maximum(frame["a"], int(lo))
    frame["win_coverage"] = np.clip((span_hi - span_lo) / (days * DAY_MS), 0.0, 1.0)
    return frame.sort_values("n", ascending=False).reset_index(drop=True)


def select_candidates(db: Session, interval: str, source_interval: str | None = None,
                      days: int | None = None, limit: int = MAX_SYMBOLS) -> list[str]:
    """挑该周期的候选币种；需要派生的周期按其上游周期挑（上游还没落库）。

    排序按**窗口内成交额**而不是柱数：按柱数取前 N 会挑到一批最老、流动性最高的币，
    Q4（低流动性档）直接空掉，「指标适合哪种池」就无从回答。密度不达标的币先剔除，
    再按成交额降序取前 N，剩下的自然覆盖从高流动性到中尾。
    """
    iv = source_interval or interval
    win = days or WINDOW_DAYS.get(interval, 90)
    frame = coverage_rows(db, iv, win)
    if frame.empty:
        return []
    frame = frame[frame["density"] >= MIN_DENSITY]
    frame = frame[frame["n"] >= MIN_HISTORY_BARS]
    frame = frame[frame["win_coverage"] >= MIN_WINDOW_COVERAGE]
    return frame.sort_values("quote_volume", ascending=False).head(limit)["symbol"].tolist()


def quality_gate(db: Session, interval: str, symbols: list[str], days: int | None = None) -> list[SymbolQuality]:
    """跑 2.2 节质量门。逐币一次聚合查询，避免把上百万行拉进 Python。

    插针「标记不删」：`|ret| > 20×ATR` 的 bar 只计数，不作为剔除理由——真实行情里
    插针本身就是指标要吃的事件，删掉等于改了标的。
    """
    step = _step_ms(interval)
    win = days or WINDOW_DAYS.get(interval, 90)
    hi = db.execute(
        text("select max(open_time) from klines where interval=:iv"), {"iv": interval}
    ).scalar()
    if hi is None:
        # 这个周期库里一行都没有（典型：30m 尚未派生）。全部按 no_bars 记账而不是抛错，
        # 让上层能把「这个周期还没数据」作为结论写出来。
        expected = int(win * DAY_MS / step) + 1
        return [
            SymbolQuality(symbol=s, bars=0, expected=expected, density=0.0,
                          first_ms=0, last_ms=0, reject=["no_bars"])
            for s in symbols
        ]
    lo = int(hi - win * DAY_MS)
    out: list[SymbolQuality] = []
    for symbol in symbols:
        row = db.execute(
            text(
                """
                select count(*) n, min(open_time) a, max(open_time) b,
                       count(distinct open_time) d,
                       sum(case when low > least(open, close) + 1e-12
                                  or greatest(open, close) > high + 1e-12 then 1 else 0 end) bad_ohlc,
                       sum(case when volume = 0 or volume is null then 1 else 0 end) zv,
                       sum(case when quote_volume = 0 or quote_volume is null then 1 else 0 end) qzv
                  from klines
                 where symbol=:s and interval=:iv and open_time>=:lo
                """
            ),
            {"s": symbol, "iv": interval, "lo": lo},
        ).mappings().fetchone()
        bars = int(row["n"] or 0)
        expected = int(win * DAY_MS / step) + 1
        first_ms, last_ms = int(row["a"] or 0), int(row["b"] or 0)
        # own-span 密度 + 窗口覆盖：见模块顶部两个门槛的注释
        own_expected = int((last_ms - first_ms) / step) + 1 if last_ms >= first_ms else 0
        density = bars / own_expected if own_expected > 0 else 0.0
        win_coverage = (min(last_ms, hi) - max(first_ms, lo)) / (win * DAY_MS)
        if bars == 0 or row["a"] is None:
            # 该周期在这个币上一根都没有（典型场景：30m 还没派生就跑了门）
            out.append(
                SymbolQuality(
                    symbol=symbol, bars=0, expected=expected, density=0.0,
                    first_ms=0, last_ms=0, reject=["no_bars"],
                )
            )
            continue
        dup = max(0, bars - int(row["d"] or 0))
        # 缺口：按时间序做 lag，统计跨度不等于 step 的次数
        gaps = int(
            db.execute(
                text(
                    """
                    select count(*) from (
                      select open_time - lag(open_time) over (order by open_time) delta
                        from klines where symbol=:s and interval=:iv and open_time>=:lo
                    ) t where t.delta is not null and t.delta <> :step
                    """
                ),
                {"s": symbol, "iv": interval, "lo": lo, "step": step},
            ).scalar()
            or 0
        )
        # 极端插针：|close/open-1| > 20×ATR%（用 14 根滚动 ATR 的窗口近似）
        pins = int(
            db.execute(
                text(
                    """
                    select count(*) from (
                      select abs(close / nullif(open,0) - 1) r,
                             avg((high - low) / nullif(close,0)) over (order by open_time
                                  rows between 14 preceding and 1 preceding) atr_pct
                        from klines where symbol=:s and interval=:iv and open_time>=:lo
                    ) t where t.atr_pct is not null and t.r > 20 * t.atr_pct
                    """
                ),
                {"s": symbol, "iv": interval, "lo": lo},
            ).scalar()
            or 0
        )
        q = SymbolQuality(
            symbol=symbol,
            bars=bars,
            expected=expected,
            density=density,
            first_ms=int(row["a"]),
            last_ms=int(row["b"]),
            gaps=gaps,
            duplicates=dup,
            ohlc_bad=int(row["bad_ohlc"] or 0),
            zero_vol=int(row["zv"] or row["qzv"] or 0),
            pins=pins,
            own_expected=own_expected,
            win_coverage=win_coverage,
        )
        if density < MIN_DENSITY:
            q.reject.append(f"density<{MIN_DENSITY}")
        if dup:
            q.reject.append("duplicate_bars")
        if q.ohlc_bad:
            q.reject.append("ohlc_logic")
        if gaps / max(own_expected, 1) > 0.02:
            q.reject.append("gap_rate>2%")
        if q.zero_vol / max(bars, 1) > 0.05:
            q.reject.append("zero_volume>5%")
        if bars < MIN_HISTORY_BARS:
            q.reject.append(f"insufficient_history<{MIN_HISTORY_BARS}bars")
        if win_coverage < MIN_WINDOW_COVERAGE:
            q.reject.append(f"low_window_coverage<{MIN_WINDOW_COVERAGE}")
        out.append(q)
    return out


def monthly_quote_volume(db: Session, interval: str, symbols: list[str],
                         days: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
    """按 UTC 自然月统计每币成交额——事件时点的属性，用于流动性分档。

    返回列：symbol, ym(YYYY-MM), quote_volume, bars, expected_bars, partial。

    窗口按 K 线开盘时间取闭区间 [end_ms - days, end_ms]；未指定末端时使用
    当前 max(open_time)。指定末端可避免后续新增数据进入窗口，但不证明末端本身已冻结。
    """
    win = days or WINDOW_DAYS.get(interval, 90)
    hi = end_ms if end_ms is not None else db.execute(
        text("select max(open_time) from klines where interval=:iv"), {"iv": interval}
    ).scalar()
    lo = int(hi - win * DAY_MS)
    sql = text(
        """
        select symbol,
               to_char(to_timestamp(open_time / 1000) at time zone 'UTC', 'YYYY-MM') ym,
               sum(coalesce(quote_volume, 0)) quote_volume,
               count(*) bars
          from klines
         where interval=:iv and open_time>=:lo and open_time<=:hi and symbol in :syms
         group by symbol, ym
        """
    ).bindparams(bindparam("syms", expanding=True))
    rows = db.execute(sql, {"iv": interval, "lo": lo, "hi": int(hi), "syms": list(symbols)}).mappings().all()
    frame = pd.DataFrame([dict(r) for r in rows])
    if frame.empty:
        return frame
    # 每个自然月的应有柱数按该月真实天数算；跨月首尾（窗口没盖全的月）标 partial，
    # 否则「月初那几天才上市的币」会因为样本短而排到错误的档位。
    # 别名不能叫 `month`：PostgreSQL 里它是日期字段关键字，裸别名会报语法错误。
    # 应有柱数与 partial 判定统一走 mark_partial_months：那里修掉了 int32 溢出，
    # 之前内联写法让 partial 恒为 False，残月从未被剔除。
    return mark_partial_months(frame, interval)


def assign_quartiles(frame: pd.DataFrame, groups: int = 4) -> pd.Series:
    """逐月在成交额上做分位分档，返回每组标签（Q1=成交额最高）。

    月份内样本太少（<groups×3 个币）时不强行切档，返回 `NA`，避免把噪声当分层。
    """
    labels = pd.Series("NA", index=frame.index, dtype=object)
    for month, part in frame.groupby("ym"):
        if len(part) < groups * 3:
            continue
        ranks = part["quote_volume"].rank(ascending=False, method="first") - 1
        bins = (ranks * groups / len(part)).astype(int).clip(0, groups - 1)
        labels.loc[part.index] = [f"Q{int(b) + 1}" for b in bins]
    return labels


def assign_fixed_tiers(frame: pd.DataFrame, groups: int = 4) -> dict[str, str]:
    """按「窗口内成交额中位数」给每个币定一个固定档：Q1 最高…Q4 最低。

    为什么不用逐月档位：事件缓存按币聚合，统计阶段拿不到「这个事件发生在哪个月」，
    所以池成员必须固定。逐月档位仍然会写进 universe 文件，并用 `tier_stability` 量化
    这个折算带来的失真（档位稳定率）。

    两个必须注意的口径：
    - 成交额取**每月中位数**而不是总和：新上市币只有两三个月的数据，求和会让它排到中间档，
      中位数则按它实际交易时的量级定档；
    - 分档只在**传进来的币种集内部**切分。把 500 个币的档位拿来给 73 个币的评估集用，
      四分位会切在评估集外面，出现「四个池只有 1 个非空」这种废结果。
    """
    if frame.empty:
        return {}
    frame = frame[~frame["partial"]] if "partial" in frame.columns else frame
    if frame.empty:
        return {}
    med = frame.groupby("symbol")["quote_volume"].median()
    vol = frame.groupby("symbol")["quote_volume"].sum()
    order = med.sort_values(ascending=False).index.tolist()
    if not order:
        return {}
    per = max(1, -(-len(order) // groups))
    out: dict[str, str] = {}
    for rank in range(groups):
        for symbol in order[rank * per : (rank + 1) * per]:
            out[str(symbol)] = f"Q{rank + 1}"
    # 只给真的有成交额的币定档（全 0 的币排在末尾，会挤进 Q4 变成假样本）
    for symbol in [s for s in out if float(vol.get(s, 0.0) or 0.0) <= 0.0]:
        out.pop(symbol, None)
    return out


def pool_members_from_frame(frame: pd.DataFrame, groups: int = 4,
                            include_partial: bool = False) -> dict[str, dict[str, str]]:
    """从 `monthly_quote_volume` 的结果算逐月档位，避免重复查库。"""
    if frame.empty:
        return {}
    if not include_partial:
        frame = frame[~frame["partial"]]
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["group"] = assign_quartiles(frame, groups)
    out: dict[str, dict[str, str]] = {}
    for month, part in frame.groupby("ym"):
        mapping = {r.symbol: r.group for r in part.itertuples() if r.group != "NA"}
        if mapping:
            out[str(month)] = mapping
    return out


def pool_members_by_month(db: Session, interval: str, symbols: list[str],
                          groups: int = 4, include_partial: bool = False,
                          days: int | None = None) -> dict[str, dict[str, str]]:
    """返回 {month: {symbol: 'Q1'..'Q4'}}，供档位稳定率诊断。"""
    return pool_members_from_frame(
        monthly_quote_volume(db, interval, symbols, days=days), groups, include_partial
    )



def tier_stability(bins: dict[str, dict[str, str]], assigned: dict[str, str]) -> dict[str, float]:
    """每个币在其出现过的月份里与固定档位一致的比例（<0.5 说明该币档位不可信）。"""
    out: dict[str, float] = {}
    for symbol, tier in assigned.items():
        months = [g.get(symbol) for g in bins.values() if symbol in g]
        if months:
            out[symbol] = round(sum(1 for g in months if g == tier) / len(months), 3)
    return out


def month_of(ms: int) -> str:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").strftime("%Y-%m")

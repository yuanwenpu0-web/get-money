"""HTF 因果 join：把「上一根已完成」HTF bar 的 high/low 拼进 base 帧。

等价 Pine `request.security(syminfo.tickerid, htf, [high[1], low[1]], lookahead=barmerge.lookahead_on)`。

本地语义（契约 §4.2，团队已定）：在 base bar 开盘时刻 `t`，只能看到满足
`htf_open + htf_step <= t` 的最后一根**已完成** HTF bar。因此

    pos = searchsorted(htf_stamps, t - htf_step, side="right") - 1

而不是 `side="left"`、也不是拿 `t` 直接比——那两种写法都会把当前未完成的 HTF 桶
泄漏进 base 帧，形成未来函数（契约 §9 因果红线 2）。

不重绘性：同一 HTF 桶内的所有 base bar 取到同一个 `pos`，`hh/ll` 完全相同，
只有 base 时间跨过 `htf_open + htf_step` 时 `pos` 才 +1。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import interval_ms

HTF_COLUMNS = ("rgd_htf_hh", "rgd_htf_ll", "rgd_htf_ok")


def attach_htf_levels(
    base: pd.DataFrame, htf: pd.DataFrame, htf_interval: str, as_of: int
) -> pd.DataFrame:
    """返回 `base` 的副本，新增 `rgd_htf_hh / rgd_htf_ll / rgd_htf_ok` 三列。

    `as_of` 是回测窗口的「现在」：只有 `open + step <= as_of` 的 HTF 桶算已收盘。
    HTF 帧为空或缺列时仍返回三列（全 na / False），让下游「降级可观测」而不是
    在 join 处静默产出一个没有 HTF 列的帧。
    """
    if not isinstance(base.index, pd.DatetimeIndex):
        raise ValueError("attach_htf_levels 需要 DatetimeIndex 的 base 帧")
    out = base.copy()
    step = interval_ms(htf_interval)
    base_stamps = out.index.as_unit("ms").asi8

    if htf is None or htf.empty or not {"high", "low"}.issubset(htf.columns):
        out["rgd_htf_hh"] = np.nan
        out["rgd_htf_ll"] = np.nan
        out["rgd_htf_ok"] = False
        return out

    htf_sorted = htf.loc[~htf.index.duplicated(keep="last")].sort_index()
    htf_stamps = htf_sorted.index.as_unit("ms").asi8
    # 未完成桶剔除：末根 HTF 的 OHLCV 是半截快照，用它等于读取未来数据。
    closed = (htf_stamps + step) <= int(as_of)
    htf_stamps = htf_stamps[closed]
    hh_src = pd.to_numeric(htf_sorted["high"], errors="coerce").to_numpy(dtype=float)[closed]
    ll_src = pd.to_numeric(htf_sorted["low"], errors="coerce").to_numpy(dtype=float)[closed]

    if htf_stamps.size == 0:
        out["rgd_htf_hh"] = np.nan
        out["rgd_htf_ll"] = np.nan
        out["rgd_htf_ok"] = False
        return out

    pos = np.searchsorted(htf_stamps, base_stamps - step, side="right") - 1
    valid = pos >= 0
    pos = np.clip(pos, 0, htf_stamps.size - 1)
    hh = np.where(valid, hh_src[pos], np.nan)
    ll = np.where(valid, ll_src[pos], np.nan)

    out["rgd_htf_hh"] = hh
    out["rgd_htf_ll"] = ll
    out["rgd_htf_ok"] = valid & np.isfinite(hh) & np.isfinite(ll) & ((hh - ll) > 0)
    return out

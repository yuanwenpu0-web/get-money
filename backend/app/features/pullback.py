"""强势回调（strong pullback）特征：4h 强势段 → 15m 首次回调 → 转强确认。

本地适配自 Martin Luk 的「强势趋势回调」思路，全部离线可算，不新增数据源：

1. **4h 强势趋势**：复用 `mtf.add_mtf_trend` 的 4h 方向（ema21/ema50）与 ADX_14，
   桶内指标已整体 shift(1)，任意 15m bar 拿到的 4h 值只依赖 ≤ 该 bar 的数据。
2. **锚定 VWAP（AVWAP）**：锚点 = 当前 4h 强势段起点（`trend_ok` 由 False→True 的 bar），
   段内累计 Σ(典型价×量)/Σ量。锚点只由过去决定，因此 AVWAP 因果。
3. **首次回调**：强势段内走满 `pullback_min_bars` 根后，第一次「由上方跌破 15m EMA9 或
   AVWAP」的 bar 记为 `pb_first_touch`。连续几根擦 EMA9 视为同一次回调，与之相隔
   `pullback_min_bars` 根以上才算新回调；`pb_bars_since_touch` 是距本次回调首次触及的 bar 数。
4. **转强确认**：首次回调后 `pullback_confirm_bars` 根内，出现「收阳 + 收盘上行 +
   收复 EMA9 + 站上 AVWAP」的 bar → `pb_confirm=1`；一次回调只确认一次，不重复发信号。
5. **结构止损参照**：首次回调的最低价 `pb_stop_ref`（硬止损放它下方），
   以及回调深度 `pb_depth`（段内最高到回调低点的回撤比例，用于评分）。
6. **EMA9 失守**：`pb_exit_long = close < ema9`，供共享回测/持仓管理做软出场。

所有列都以 `pb_` 前缀（`ema9`/`avwap` 例外，二者是价格参照），
`evaluate_signals` 只读单行，因此回测与实盘都先对整帧调用本模块。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .mtf import add_mtf_trend

# 4h = 240 分钟；本策略只用 4h 上下文，1d 不参与
TREND_TF_MINUTES = (240,)

# `evaluate_signals` 判断「帧是否已富化」用的列清单（缺一列就整帧重算）
PULLBACK_COLUMNS = (
    "pb_dir_4h",
    "pb_adx_4h",
    "pb_trend_ok",
    "ema9",
    "avwap",
    "avwap_dist",
    "pb_touch",
    "pb_first_touch",
    "pb_bars_since_touch",
    "pb_confirm",
    "pb_depth",
    "pb_stop_ref",
    "pb_exit_long",
)


def _num(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").astype(float)


def add_pullback_context(
    frame: pd.DataFrame, params: dict | None = None
) -> pd.DataFrame:
    """给 15m 帧附加强势回调所需的全部派生列（幂等、纯因果）。

    params 只消费 `htf_adx_filter`（4h 趋势强度门槛）与 `pullback_confirm_bars`
    （回调后允许的确认窗口），缺失时用策略默认值。
    """
    params = params or {}
    htf_adx = float(params.get("htf_adx_filter", 22.0))
    confirm_bars = max(1, int(params.get("pullback_confirm_bars", 8)))
    min_bars = max(0, int(params.get("pullback_min_bars", 8)))
    data = frame.copy()
    if data.empty:
        for col in PULLBACK_COLUMNS:
            data[col] = pd.Series(dtype=float)
        return data

    if "mtf_dir_4h" not in data.columns or "mtf_adx_4h" not in data.columns:
        data = add_mtf_trend(data, TREND_TF_MINUTES)

    close = _num(data, "close")
    open_ = _num(data, "open")
    high = _num(data, "high")
    low = _num(data, "low")
    volume = _num(data, "volume", 0.0).fillna(0.0).clip(lower=0.0)

    dir4 = _num(data, "mtf_dir_4h", 0.0).fillna(0.0)
    adx4 = _num(data, "mtf_adx_4h")
    # NaN 的 4h ADX（预热不足）一律视为不达标
    trend_ok = (dir4 > 0) & adx4.ge(htf_adx)

    ema9 = close.ewm(span=9, adjust=False).mean()

    # 强势段编号：trend_ok 每次翻转都开新段，AVWAP/首次回调都按段重置
    segment = trend_ok.ne(trend_ok.shift(1).fillna(False)).cumsum()

    typical = (high + low + close) / 3.0
    if float(volume.sum()) <= 0:  # 无量数据（部分导入源）退化为等权均值
        volume = pd.Series(1.0, index=data.index)
    cum_pv = (typical * volume).groupby(segment).cumsum()
    cum_v = volume.groupby(segment).cumsum().replace(0, np.nan)
    avwap = (cum_pv / cum_v).ffill()

    # 回调触及：此前站在 EMA9/AVWAP 上方，本根低点跌破其一（首次回调的定义基础）。
    # runup_ok 要求强势段已走满 pullback_min_bars 根，避免把「趋势刚起点」当成回调。
    runup_ok = data.groupby(segment).cumcount().ge(min_bars)
    was_above = close.shift(1).ge(ema9.shift(1))
    touch = (
        trend_ok
        & runup_ok
        & was_above.fillna(False)
        & (low.le(ema9) | (avwap.gt(0) & low.le(avwap)))
    )

    # 「首次回调」= 与上一次触及相隔 min_gap 根以上的第一次触及：
    # 相邻几根连续擦 EMA9 属于同一次回调，隔开的一次才是新回调。
    # 只用到 t-1 为止的触及位置（ffill + shift），严格因果。
    position = pd.Series(np.arange(len(data), dtype=float), index=data.index)
    previous_touch = (
        position.where(touch).groupby(segment).ffill().shift(1)
    )
    gap_ok = previous_touch.isna() | position.sub(previous_touch).ge(min_bars)
    first_touch = touch & gap_ok
    # 当前回调的起点位置（本次回调的首次触及），stop_ref / 确认窗口都按它算
    episode_pos = position.where(first_touch).groupby(segment).ffill()
    bars_since = position - episode_pos
    episode = first_touch.cumsum()

    # 回调段的最高点与最低点（都只到当前 bar 为止）：run_high 取段内最高，
    # stop_ref 取本次回调开始后的最低价，硬止损放它下方。
    run_high = high.groupby(segment).cummax()
    stop_ref = (
        low.where(bars_since.notna()).groupby(episode).cummin().groupby(episode).ffill()
    )
    depth = (run_high - stop_ref) / run_high.replace(0, np.nan)

    # 转强确认：确认窗口内出现「收阳 + 收盘高于前收 + 收复 EMA9 + 站上 AVWAP」
    in_window = bars_since.between(1, confirm_bars)
    reclaim = close.gt(ema9)
    bull_bar = close.gt(open_) & close.gt(close.shift(1))
    above_avwap = close.ge(avwap)
    confirm_raw = trend_ok & in_window & reclaim & bull_bar & above_avwap
    # 一次回调只给一次入场：确认窗口内若已确认过，后续 bar 不再重复发信号
    # 注意 shift 会把 dtype 升为 object，必须 astype(bool) 再取反（object 上的 ~ 是位运算）
    confirmed_before = (
        confirm_raw.groupby(episode)
        .cummax()
        .groupby(episode)
        .shift(1)
        .fillna(False)
        .astype(bool)
    )
    confirm = confirm_raw & ~confirmed_before

    data["pb_dir_4h"] = dir4
    data["pb_adx_4h"] = adx4
    data["pb_trend_ok"] = trend_ok.astype(int)
    data["ema9"] = ema9
    data["avwap"] = avwap
    data["avwap_dist"] = close / avwap.replace(0, np.nan) - 1.0
    data["pb_touch"] = touch.astype(int)
    data["pb_first_touch"] = first_touch.astype(int)
    data["pb_bars_since_touch"] = bars_since
    data["pb_confirm"] = confirm.astype(int)
    data["pb_depth"] = depth
    data["pb_stop_ref"] = stop_ref
    data["pb_exit_long"] = close.lt(ema9).astype(int)
    return data

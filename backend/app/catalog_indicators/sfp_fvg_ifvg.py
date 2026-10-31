"""SFP + FVG / IFVG  (adSFP 5.17) — catalog port (chart timeframe core).

Source: data/pine-crypto-collection/amadeo987/SK_SFP_Najblizsze_Poziomy.pine

Computational core ported on the chart timeframe:
  * Quality-swing memory (pivots gated by a >=1.25 ATR reaction) and the
    active-liquidity radar (nearest pools above/below price).
  * SFP detection: wick sweep of a confirmed swing level + full-body close back
    behind it = RAW setup; a later close beyond the opposite SFP candle extreme
    (structure break) within the confirmation window = CONF.
  * FVG / IFVG book (chart timeframe): 3-bar imbalance creation, fill and
    inversion lifecycle, nearest above/below zones.
  * Dual auto-anchored VWAP (fast/slow) and EMA 200/100/50/custom.
  * PVP naked POC (daily/weekly/monthly) from an OHLCV volume-profile
    approximation.

Not ported: watermark, day/month separators, multiple custom-timeframe FVG
books (request.security), and all drawing/box/label rendering.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ._pine import atr, boolean, choice, ema, highest, integer, lowest, number, ohlcv, pivothigh_strict, pivotlow_strict, sma

_MINTICK = 1e-8


def _add_or_merge(ceny, bary, jak, price, bar, quality, merge_thr):
    idx = -1
    for i in range(len(ceny)):
        if abs(price - ceny[i]) <= merge_thr:
            idx = i
            break
    if idx >= 0:
        if quality > jak[idx] or (quality == jak[idx] and bar > bary[idx]):
            ceny[idx] = price
            bary[idx] = bar
            jak[idx] = quality
    else:
        ceny.append(price)
        bary.append(bar)
        jak.append(quality)
    while len(ceny) > 100:
        ceny.pop(0); bary.pop(0); jak.pop(0)


def _anchored_vwap(typical, volume, keys):
    n = len(typical)
    out = np.full(n, np.nan)
    cum_pv = cum_v = 0.0
    last = None
    for i in range(n):
        if keys[i] != last:
            cum_pv = cum_v = 0.0
            last = keys[i]
        vv = volume[i] if not math.isnan(volume[i]) else 0.0
        cum_pv += typical[i] * vv
        cum_v += vv
        out[i] = cum_pv / cum_v if cum_v > 0 else np.nan
    return out


def _pvp_poc(highs, lows, vols, closes, rows):
    if not highs:
        return float("nan")
    lo, hi = min(lows), max(highs)
    rng = hi - lo
    if rng <= _MINTICK:
        return closes[-1]
    bin_size = rng / rows
    bins = max(1, int(math.ceil(rng / bin_size)))
    hist = np.zeros(bins)
    for bh, bl, bc, bv in zip(highs, lows, closes, vols):
        br = bh - bl
        if br <= _MINTICK:
            k = min(bins - 1, max(0, int((bc - lo) / bin_size)))
            hist[k] += bv
        else:
            f = min(bins - 1, max(0, int((bl - lo) / bin_size)))
            lst = min(bins - 1, max(0, int((bh - lo) / bin_size)))
            for k in range(f, lst + 1):
                bin_low = lo + k * bin_size
                bin_high = bin_low + bin_size
                overlap = max(0.0, min(bh, bin_high) - max(bl, bin_low))
                hist[k] += bv * overlap / br
    poc = int(np.argmax(hist))
    return lo + (poc + 0.5) * bin_size


def sfp_fvg_ifvg(
    frame: pd.DataFrame,
    *,
    swiece_swingu: int = 10,
    min_reakcja_atr: float = 1.25,
    laczenie_atr: float = 0.50,
    okres_atr: int = 14,
    maks_wiek: int = 400,
    maks_odleglosc_atr: float = 15.0,
    min_sweep_atr: float = 0.10,
    max_sweep_atr: float = 1.00,
    min_zakres_setup_atr: float = 1.20,
    okno_potwierdzenia: int = 3,
    bufor_uniewaznienia_atr: float = 0.10,
    filtr_wolumenu: bool = False,
    okres_wolumenu: int = 20,
    mnoznik_wolumenu: float = 1.20,
    min_wielkosc_fvg_atr: float = 0.12,
    maks_wiek_stref_fvg: int = 300,
    maks_wiek_stref_ifvg: int = 72,
    ema_fast_len: int = 50,
    ema_mid_len: int = 100,
    ema_slow_len: int = 200,
    ema_custom_len: int = 21,
    wlacz_fvg: bool = True,
    pvp_rows: int = 24,
) -> pd.DataFrame:
    swiece_swingu = integer("swiece_swingu", swiece_swingu, 2, 100)
    okres_atr = integer("okres_atr", okres_atr, 2, 200)
    maks_wiek = integer("maks_wiek", maks_wiek, 10, 5000)
    okres_wolumenu = integer("okres_wolumenu", okres_wolumenu, 2, 300)
    okno_potwierdzenia = integer("okno_potwierdzenia", okno_potwierdzenia, 1, 50)
    maks_wiek_stref_fvg = integer("maks_wiek_stref_fvg", maks_wiek_stref_fvg, 5, 2000)
    maks_wiek_stref_ifvg = integer("maks_wiek_stref_ifvg", maks_wiek_stref_ifvg, 5, 2000)
    min_reakcja_atr = number("min_reakcja_atr", min_reakcja_atr, 0.1)
    laczenie_atr = number("laczenie_atr", laczenie_atr, 0.01)
    maks_odleglosc_atr = number("maks_odleglosc_atr", maks_odleglosc_atr, 0.1)
    min_sweep_atr = number("min_sweep_atr", min_sweep_atr, 0.0)
    max_sweep_atr = number("max_sweep_atr", max_sweep_atr, 0.0)
    min_zakres_setup_atr = number("min_zakres_setup_atr", min_zakres_setup_atr, 0.0)
    bufor_uniewaznienia_atr = number("bufor_uniewaznienia_atr", bufor_uniewaznienia_atr, 0.0)
    mnoznik_wolumenu = number("mnoznik_wolumenu", mnoznik_wolumenu, 0.0)
    min_wielkosc_fvg_atr = number("min_wielkosc_fvg_atr", min_wielkosc_fvg_atr, 0.0)
    ema_fast_len = integer("ema_fast_len", ema_fast_len, 1, 2000)
    ema_mid_len = integer("ema_mid_len", ema_mid_len, 1, 2000)
    ema_slow_len = integer("ema_slow_len", ema_slow_len, 1, 2000)
    ema_custom_len = integer("ema_custom_len", ema_custom_len, 1, 2000)
    pvp_rows = integer("pvp_rows", pvp_rows, 5, 200)
    filtr_wolumenu = boolean("filtr_wolumenu", filtr_wolumenu)
    wlacz_fvg = boolean("wlacz_fvg", wlacz_fvg)

    o, h, l, c, vol = ohlcv(frame, "open", "high", "low", "close", "volume")
    o = o.to_numpy(dtype=float); h = h.to_numpy(dtype=float); l = l.to_numpy(dtype=float)
    c = c.to_numpy(dtype=float); vol = vol.to_numpy(dtype=float)
    idx = frame.index
    n = len(frame)

    atr_a = np.maximum(atr(frame, okres_atr).to_numpy(dtype=float), _MINTICK)
    ph = pivothigh_strict(frame["high"], swiece_swingu, swiece_swingu).to_numpy(dtype=float)
    pl = pivotlow_strict(frame["low"], swiece_swingu, swiece_swingu).to_numpy(dtype=float)
    lowest_after_high = lowest(frame["low"], swiece_swingu + 1).to_numpy(dtype=float)
    highest_after_low = highest(frame["high"], swiece_swingu + 1).to_numpy(dtype=float)
    vol_avg = sma(pd.Series(vol), okres_wolumenu).to_numpy(dtype=float)

    ema50 = ema(frame["close"], ema_fast_len).to_numpy(dtype=float)
    ema100 = ema(frame["close"], ema_mid_len).to_numpy(dtype=float)
    ema200 = ema(frame["close"], ema_slow_len).to_numpy(dtype=float)
    ema_custom = ema(frame["close"], ema_custom_len).to_numpy(dtype=float)

    typical = (h + l + c) / 3.0
    if isinstance(idx, pd.DatetimeIndex):
        dates = idx.normalize()
        iso = idx.isocalendar()
        day_keys = list(dates)
        week_keys = [(r.year, r.week) for r in iso.itertuples()]
        month_keys = [(d.year, d.month) for d in dates]
        quarter_keys = [(d.year, (d.month - 1) // 3) for d in dates]
        year_keys = [d.year for d in dates]
        deltas = np.diff(idx.values).astype("timedelta64[s]").astype(float) if n > 1 else np.array([])
        med = float(np.median(deltas)) if len(deltas) else 86400.0
    else:
        day_keys = list(range(n)); week_keys = day_keys; month_keys = day_keys
        quarter_keys = day_keys; year_keys = day_keys; med = 86400.0

    if med < 86400:
        vwap_fast = _anchored_vwap(typical, vol, day_keys)
        vwap_slow = _anchored_vwap(typical, vol, week_keys)
    elif med <= 90000:
        vwap_fast = _anchored_vwap(typical, vol, month_keys)
        vwap_slow = _anchored_vwap(typical, vol, quarter_keys)
    else:
        vwap_fast = _anchored_vwap(typical, vol, year_keys)
        vwap_slow = _anchored_vwap(typical, vol, [0] * n)

    # ---- SFP state ----
    ceny_high: list = []; bary_high: list = []; jak_high: list = []
    ceny_low: list = []; bary_low: list = []; jak_low: list = []
    oczekuje_bear = False
    ob_level = ob_high = ob_low = ob_atr = ob_wyb = None
    ob_bar = None
    oczekuje_bull = False
    ou_level = ou_high = ou_low = ou_atr = ou_wyb = None
    ou_bar = None

    # ---- FVG book ----
    fvg = []  # dicts: top,bottom,direction,inverted,start_bar,inversion_bar

    # ---- PVP ----
    pvp_levels = {"D": [], "W": [], "M": []}
    pvp_samples = {"D": [], "W": [], "M": []}
    pvp_last = {"D": None, "W": None, "M": None}

    out = {k: np.full(n, np.nan) for k in [
        "ema_fast", "ema_mid", "ema_slow", "ema_custom", "vwap_fast", "vwap_slow",
        "sfp_high_level", "sfp_low_level", "fvg_above_top", "fvg_above_bottom",
        "fvg_below_top", "fvg_below_bottom", "pvp_daily", "pvp_weekly", "pvp_monthly"]}
    flags = {k: np.zeros(n, dtype=np.int64) for k in [
        "sfp_bear_raw", "sfp_bull_raw", "sfp_bear_conf", "sfp_bull_conf",
        "sfp_pending_bear", "sfp_pending_bull", "sfp_fvg_bull_new", "sfp_fvg_bear_new",
        "sfp_ifvg_bull_new", "sfp_ifvg_bear_new"]}

    def current_period(i):
        if isinstance(idx, pd.DatetimeIndex):
            d = idx[i]
        else:
            d = i
        return (d.year, d.isocalendar()[1]) if isinstance(d, pd.Timestamp) else (i, i), d

    for i in range(n):
        atr_i = atr_a[i]

        # ---- quality swings ----
        if not math.isnan(ph[i]) and i >= swiece_swingu:
            atr_sw = atr_a[i - swiece_swingu] if i >= swiece_swingu else atr_i
            if atr_sw > 0:
                reaction = (ph[i] - lowest_after_high[i]) / atr_sw
                if reaction >= min_reakcja_atr:
                    merge = max(_MINTICK, atr_sw * laczenie_atr)
                    _add_or_merge(ceny_high, bary_high, jak_high, ph[i], i - swiece_swingu, reaction, merge)
        if not math.isnan(pl[i]) and i >= swiece_swingu:
            atr_sw = atr_a[i - swiece_swingu] if i >= swiece_swingu else atr_i
            if atr_sw > 0:
                reaction = (highest_after_low[i] - pl[i]) / atr_sw
                if reaction >= min_reakcja_atr:
                    merge = max(_MINTICK, atr_sw * laczenie_atr)
                    _add_or_merge(ceny_low, bary_low, jak_low, pl[i], i - swiece_swingu, reaction, merge)

        # ---- radar levels (nearest above/below) ----
        ref = c[i]
        limit = atr_i * maks_odleglosc_atr
        best_hi = None; best_hi_d = 1e20
        for pz, pb in zip(ceny_high, bary_high):
            if i > pb and i - pb <= maks_wiek and abs(ref - pz) <= limit and pz > ref:
                d = pz - ref
                if d < best_hi_d:
                    best_hi_d = d; best_hi = pz
        best_lo = None; best_lo_d = 1e20
        for pz, pb in zip(ceny_low, bary_low):
            if i > pb and i - pb <= maks_wiek and abs(ref - pz) <= limit and pz < ref:
                d = ref - pz
                if d < best_lo_d:
                    best_lo_d = d; best_lo = pz

        # ---- setups (RAW) ----
        vol_ok = (not filtr_wolumenu) or (not math.isnan(vol[i]) and not math.isnan(vol_avg[i]) and vol[i] >= vol_avg[i] * mnoznik_wolumenu)
        min_sweep = atr_i * min_sweep_atr
        max_sweep = 1e20 if max_sweep_atr == 0.0 else atr_i * max_sweep_atr
        bear_setup = False; bear_level = None
        bull_setup = False; bull_level = None
        for j in range(len(ceny_high) - 1, -1, -1):
            pz = ceny_high[j]; pb = bary_high[j]
            if i - pb > maks_wiek or h[i] >= pz:
                if h[i] >= pz:
                    wyb = h[i] - pz
                    zakres_ok = min_zakres_setup_atr == 0.0 or (h[i] - l[i]) >= atr_i * min_zakres_setup_atr
                    setup = (c[i] < pz and o[i] < pz and wyb >= min_sweep and wyb <= max_sweep and vol_ok and zakres_ok)
                    if setup and (not bear_setup or pz < bear_level):
                        bear_setup = True; bear_level = pz
                ceny_high.pop(j); bary_high.pop(j); jak_high.pop(j)
        for j in range(len(ceny_low) - 1, -1, -1):
            pz = ceny_low[j]; pb = bary_low[j]
            if i - pb > maks_wiek or l[i] <= pz:
                if l[i] <= pz:
                    wyb = pz - l[i]
                    zakres_ok = min_zakres_setup_atr == 0.0 or (h[i] - l[i]) >= atr_i * min_zakres_setup_atr
                    setup = (c[i] > pz and o[i] > pz and wyb >= min_sweep and wyb <= max_sweep and vol_ok and zakres_ok)
                    if setup and (not bull_setup or pz > bull_level):
                        bull_setup = True; bull_level = pz
                ceny_low.pop(j); bary_low.pop(j); jak_low.pop(j)

        # ---- confirmation of previously pending setups ----
        bear_conf = bull_conf = False
        if oczekuje_bear and ob_bar is not None and i > ob_bar:
            invalid = h[i] > ob_high + ob_atr * bufor_uniewaznienia_atr
            expired = i - ob_bar > okno_potwierdzenia
            confirmed = c[i] < ob_low
            if invalid or expired:
                oczekuje_bear = False
            elif confirmed:
                bear_conf = True
                oczekuje_bear = False
        if oczekuje_bull and ou_bar is not None and i > ou_bar:
            invalid = l[i] < ou_low - ou_atr * bufor_uniewaznienia_atr
            expired = i - ou_bar > okno_potwierdzenia
            confirmed = c[i] > ou_high
            if invalid or expired:
                oczekuje_bull = False
            elif confirmed:
                bull_conf = True
                oczekuje_bull = False

        # ---- enqueue new setups ----
        if bear_setup:
            oczekuje_bear = True
            ob_level, ob_high, ob_low, ob_atr, ob_bar = bear_level, h[i], l[i], atr_i, i
        if bull_setup:
            oczekuje_bull = True
            ou_level, ou_high, ou_low, ou_atr, ou_bar = bull_level, h[i], l[i], atr_i, i

        # ---- FVG / IFVG (chart timeframe: source = previous closed bar) ----
        new_bull_fvg = new_bear_fvg = new_bull_ifvg = new_bear_ifvg = False
        if wlacz_fvg and i >= 3:
            src_high, src_low, src_close = h[i - 1], l[i - 1], c[i - 1]
            first_high, first_low = h[i - 3], l[i - 3]
            src_atr = atr_a[i - 1]
            for j in range(len(fvg) - 1, -1, -1):
                z = fvg[j]
                age_start = z["inversion_bar"] if z["inverted"] and z["inversion_bar"] is not None else z["start_bar"]
                max_age = maks_wiek_stref_ifvg if z["inverted"] else maks_wiek_stref_fvg
                remove = (i - age_start) > max_age
                if not remove and not z["inverted"]:
                    inversion = (z["direction"] == 1 and src_close < z["bottom"]) or (z["direction"] == -1 and src_close > z["top"])
                    filled = (z["direction"] == 1 and src_low <= z["bottom"]) or (z["direction"] == -1 and src_high >= z["top"])
                    if inversion:
                        nd = -z["direction"]
                        inverted_filled = (nd == 1 and src_low <= z["bottom"]) or (nd == -1 and src_high >= z["top"])
                        if inverted_filled:
                            remove = True
                        else:
                            z["direction"] = nd; z["inverted"] = True; z["inversion_bar"] = i
                            new_bull_ifvg = new_bull_ifvg or nd == 1
                            new_bear_ifvg = new_bear_ifvg or nd == -1
                    elif filled:
                        remove = True
                elif not remove:
                    filled_ifvg = (z["direction"] == 1 and src_low <= z["bottom"]) or (z["direction"] == -1 and src_high >= z["top"])
                    if filled_ifvg:
                        remove = True
                if remove:
                    fvg.pop(j)
            bull_candidate = src_atr > 0 and src_low > first_high and src_low - first_high >= src_atr * min_wielkosc_fvg_atr
            bear_candidate = src_atr > 0 and src_high < first_low and first_low - src_high >= src_atr * min_wielkosc_fvg_atr
            if bull_candidate or bear_candidate:
                bull = bull_candidate
                top = src_low if bull else first_low
                bottom = first_high if bull else src_high
                fvg.append({"top": top, "bottom": bottom, "direction": 1 if bull else -1,
                            "inverted": False, "start_bar": i, "inversion_bar": None})
                new_bull_fvg = bull
                new_bear_fvg = not bull
            while len(fvg) > 60:
                fvg.pop(0)

        # nearest zones
        above_top = above_bottom = below_top = below_bottom = np.nan
        best_above = None; best_above_d = 1e20
        best_below = None; best_below_d = 1e20
        for z in fvg:
            live_filled = (z["direction"] == 1 and l[i] <= z["bottom"]) or (z["direction"] == -1 and h[i] >= z["top"])
            if live_filled:
                continue
            if z["bottom"] > c[i]:
                d = z["bottom"] - c[i]
                if d < best_above_d:
                    best_above_d = d; best_above = z
            if z["top"] < c[i]:
                d = c[i] - z["top"]
                if d < best_below_d:
                    best_below_d = d; best_below = z
        if best_above is not None:
            above_top, above_bottom = best_above["top"], best_above["bottom"]
        if best_below is not None:
            below_top, below_bottom = best_below["top"], best_below["bottom"]

        # ---- PVP naked POC (daily / weekly / monthly) ----
        if isinstance(idx, pd.DatetimeIndex):
            d = idx[i]
            keys = {"D": d.date(), "W": (d.year, d.isocalendar()[1]), "M": (d.year, d.month)}
        else:
            keys = {"D": i, "W": i, "M": i}
        for tf in ("D", "W", "M"):
            k = keys[tf]
            if pvp_last[tf] is not None and k != pvp_last[tf]:
                hist = pvp_samples[tf]
                if hist:
                    poc = _pvp_poc([x[0] for x in hist], [x[1] for x in hist],
                                   [x[2] for x in hist], [x[3] for x in hist], pvp_rows)
                    if not math.isnan(poc):
                        pvp_levels[tf].append(poc)
                pvp_samples[tf] = []
            pvp_last[tf] = k
            pvp_samples[tf].append((h[i], l[i], vol[i] if not math.isnan(vol[i]) else 0.0, c[i]))
            # touch removal (naked)
            keep = []
            for lv in pvp_levels[tf]:
                if not (l[i] <= lv <= h[i]):
                    keep.append(lv)
            pvp_levels[tf] = keep

        def nearest_level(levels):
            if not levels:
                return np.nan
            return min(levels, key=lambda x: abs(x - c[i]))

        # ---- record ----
        out["ema_fast"][i] = ema50[i]; out["ema_mid"][i] = ema100[i]
        out["ema_slow"][i] = ema200[i]; out["ema_custom"][i] = ema_custom[i]
        out["vwap_fast"][i] = vwap_fast[i]; out["vwap_slow"][i] = vwap_slow[i]
        out["sfp_high_level"][i] = best_hi if best_hi is not None else np.nan
        out["sfp_low_level"][i] = best_lo if best_lo is not None else np.nan
        out["fvg_above_top"][i] = above_top; out["fvg_above_bottom"][i] = above_bottom
        out["fvg_below_top"][i] = below_top; out["fvg_below_bottom"][i] = below_bottom
        out["pvp_daily"][i] = nearest_level(pvp_levels["D"])
        out["pvp_weekly"][i] = nearest_level(pvp_levels["W"])
        out["pvp_monthly"][i] = nearest_level(pvp_levels["M"])
        flags["sfp_bear_raw"][i] = 1 if bear_setup else 0
        flags["sfp_bull_raw"][i] = 1 if bull_setup else 0
        flags["sfp_bear_conf"][i] = 1 if bear_conf else 0
        flags["sfp_bull_conf"][i] = 1 if bull_conf else 0
        flags["sfp_pending_bear"][i] = 1 if oczekuje_bear else 0
        flags["sfp_pending_bull"][i] = 1 if oczekuje_bull else 0
        flags["sfp_fvg_bull_new"][i] = 1 if new_bull_fvg else 0
        flags["sfp_fvg_bear_new"][i] = 1 if new_bear_fvg else 0
        flags["sfp_ifvg_bull_new"][i] = 1 if new_bull_ifvg else 0
        flags["sfp_ifvg_bear_new"][i] = 1 if new_bear_ifvg else 0

    df = pd.DataFrame(index=idx)
    df["sfp_ema_fast"] = out["ema_fast"]
    df["sfp_ema_mid"] = out["ema_mid"]
    df["sfp_ema_slow"] = out["ema_slow"]
    df["sfp_ema_custom"] = out["ema_custom"]
    df["sfp_vwap_fast"] = out["vwap_fast"]
    df["sfp_vwap_slow"] = out["vwap_slow"]
    df["sfp_high_level"] = out["sfp_high_level"]
    df["sfp_low_level"] = out["sfp_low_level"]
    df["sfp_fvg_above_top"] = out["fvg_above_top"]
    df["sfp_fvg_above_bottom"] = out["fvg_above_bottom"]
    df["sfp_fvg_below_top"] = out["fvg_below_top"]
    df["sfp_fvg_below_bottom"] = out["fvg_below_bottom"]
    df["sfp_pvp_daily"] = out["pvp_daily"]
    df["sfp_pvp_weekly"] = out["pvp_weekly"]
    df["sfp_pvp_monthly"] = out["pvp_monthly"]
    for k, v in flags.items():
        df[k] = pd.Series(v, index=idx).astype("Int64")
    return df

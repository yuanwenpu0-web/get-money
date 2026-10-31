"""搜索空间清单：每个指标的参数区间、步长、约束与规则说明。

这里是唯一一处定义「可以搜什么」的地方。报告、配置快照与测试都从这里取，
所以改动搜索空间只改本文件，不会出现脚本与文档口径不一致。
参数一律用 `min / max / step` 表达（等价于闭区间等差取值），约束用命名谓词，
保证整份清单可以直接 json 序列化进实验配置。
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from typing import Callable

GRID: dict[str, dict] = {
    "ema_cross": {
        "label": "EMA 快慢线交叉",
        "rule": "快线(EMA fast) 上穿慢线(EMA slow) 做多，下穿做空",
        "params": {"fast": {"min": 5, "max": 20, "step": 3}, "slow": {"min": 21, "max": 89, "step": 17}},
        "constraint": "fast_lt_slow",
    },
    "price_sma": {
        "label": "收盘价穿越均线",
        "rule": "收盘上穿 SMA(n) 做多，下穿做空",
        "params": {"n": {"min": 20, "max": 200, "step": 30}},
    },
    "rsi_reversal": {
        "label": "RSI 超卖/超买回归",
        "rule": "RSI(n) 上穿超卖线做多，下穿超买线做空",
        "params": {
            "n": {"min": 7, "max": 21, "step": 7},
            "low": {"min": 25, "max": 35, "step": 5},
            "high": {"min": 65, "max": 75, "step": 5},
        },
        "constraint": "symmetric_rsi",
    },
    "macd_cross": {
        "label": "MACD 交叉",
        "rule": "DIF 上穿 DEA（或零轴）做多，下穿做空",
        "params": {
            "fast": {"min": 8, "max": 12, "step": 4},
            "slow": {"min": 21, "max": 50, "step": 29},
            "signal": {"min": 9, "max": 9, "step": 1},
            "mode": {"values": [0, 1]},
        },
        "constraint": "fast_lt_slow",
    },
    "bollinger_break": {
        "label": "布林带突破",
        "rule": "收盘上穿上轨做多，下穿下轨做空",
        "params": {"n": {"min": 20, "max": 50, "step": 15}, "k": {"values": [1.5, 2.0, 2.5]}},
    },
    "donchian_break": {
        "label": "唐奇安通道突破",
        "rule": "收盘创 n 根新高做多，创 n 根新低做空（通道已 shift 1）",
        "params": {"n": {"min": 20, "max": 110, "step": 30}},
    },
    "supertrend_flip": {
        "label": "Supertrend 翻转",
        "rule": "趋势方向由空转多做多，由多转空做空",
        "params": {"n": {"min": 10, "max": 21, "step": 7}, "mult": {"values": [2.0, 3.0, 4.0]}},
    },
    "adx_di_cross": {
        "label": "ADX +DI/-DI 交叉",
        "rule": "+DI 上穿 -DI 且 ADX ≥ 门槛做多，反向做空",
        "params": {"n": {"min": 14, "max": 21, "step": 7}, "adx_min": {"values": [0, 20, 25]}},
    },
    "stoch_cross": {
        "label": "随机指标 KD 交叉",
        "rule": "K 上穿 D 且 K<50 做多，K 下穿 D 且 K>50 做空",
        "params": {"n": {"min": 14, "max": 21, "step": 7}, "smooth": {"values": [3, 5]}},
    },
    "cci_reversal": {
        "label": "CCI 回归",
        "rule": "CCI(n) 上穿 -level 做多，下穿 +level 做空",
        "params": {"n": {"min": 20, "max": 34, "step": 14}, "level": {"values": [100, 150]}},
    },
    "mfi_reversal": {
        "label": "MFI 资金流回归",
        "rule": "MFI(n) 上穿超卖线做多，下穿对称超买线做空",
        "params": {"n": {"min": 14, "max": 21, "step": 7}, "low": {"values": [20, 30]}},
    },
    "volume_break": {
        "label": "放量突破",
        "rule": "成交量 > k×均量 且收盘创 n 根新高做多，新低做空",
        "params": {"n": {"min": 20, "max": 50, "step": 30}, "k": {"values": [1.5, 2.0, 3.0]}},
    },
    "atr_channel_break": {
        "label": "ATR 通道突破",
        "rule": "收盘上穿 SMA(n)+k×ATR(n) 做多，下穿下轨做空",
        "params": {"n": {"min": 20, "max": 50, "step": 30}, "k": {"values": [1.0, 1.5, 2.0]}},
    },
    "roc_momentum": {
        "label": "ROC 动量阈值",
        "rule": "ROC(n) 上穿 +thr 做多，下穿 -thr 做空",
        "params": {"n": {"min": 12, "max": 48, "step": 18}, "thr": {"values": [0.0, 2.0]}},
    },
    "vwap_reclaim": {
        "label": "滚动 VWAP 收复",
        "rule": "收盘上穿 n 根滚动 VWAP 做多，下穿做空",
        "params": {"n": {"min": 24, "max": 72, "step": 48}},
    },
}

CONSTRAINTS: dict[str, Callable[[dict], bool]] = {
    "fast_lt_slow": lambda p: p["fast"] < p["slow"],
    "symmetric_rsi": lambda p: p["low"] < 50 < p["high"] and (p["low"] + p["high"] == 100),
}


@dataclass(frozen=True, slots=True)
class Candidate:
    family: str
    params: dict
    key: str

    @property
    def label(self) -> str:
        return describe(self.family, self.params)


def _values(spec: dict) -> list:
    if "values" in spec:
        return list(spec["values"])
    low, high, step = spec["min"], spec["max"], spec["step"]
    out = []
    value = low
    while value <= high + 1e-9:
        out.append(value)
        value += step
    if all(isinstance(v, int) for v in out):
        return [int(v) for v in out]
    return [round(v, 6) for v in out]


def describe(family: str, params: dict) -> str:
    body = ",".join(f"{k}={params[k]}" for k in sorted(params))
    return f"{family}({body})"


def expand(family: str) -> list[dict]:
    spec = GRID[family]
    names = list(spec["params"])
    combos = [
        dict(zip(names, values))
        for values in itertools.product(*(_values(spec["params"][name]) for name in names))
    ]
    predicate = CONSTRAINTS.get(spec.get("constraint", ""))
    if predicate is not None:
        combos = [row for row in combos if predicate(row)]
    return combos


def candidates(families: list[str] | None = None) -> list[Candidate]:
    out: list[Candidate] = []
    for family in families or list(GRID):
        for params in expand(family):
            key = family + "|" + json.dumps(params, sort_keys=True, separators=(",", ":"))
            out.append(Candidate(family, params, key))
    return out


def manifest() -> dict:
    """可直接写入实验配置的搜索空间快照（含每个指标展开后的组合数）。"""
    return {
        name: {
            "label": spec["label"],
            "rule": spec["rule"],
            "params": spec["params"],
            "constraint": spec.get("constraint"),
            "combos": len(expand(name)),
        }
        for name, spec in GRID.items()
    }

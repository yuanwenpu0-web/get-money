"""指标条件列的枚举与**方向声明**。

为什么必须有这一层
------------------
`backend/scripts/signal_screen.py:100` 的老口径只算多头前瞻收益（`forward_return` 返回
`close[t+h]/close[t]-1`），于是 `strong_sell`、`bear_cross` 这类看跌列**永远不可能被判成正
edge**——它们携带的信息本来是「信号后下跌」。评估手册把这条列为三个决定性缺口之首
（`docs/indicator-evaluation-plan.md` 第 0.1 节；误区表第 2 条「只测多头 → 看跌指标全部被误杀」），
并明确写了「方向靠猜是这套评估最容易死的地方」。

所以方向不能靠看收益符号反推（那是用被测样本决定假设）。本模块只按**列名与指标名的语义**判方向，
完全不接触任何收益率数据，规则是三档：

| 档位 | 依据 | 允许的结论 |
|---|---|---|
| `declared` | 命中强方向词（bull/bear/long/short/buy/sell/cross_up/...） | 可以进 candidate/PROMOTE |
| `weak` | 只命中语境词（oversold/support/above/...），语义依赖策略流派 | 可以报 candidate，但标 `direction_weak` |
| `ambiguous` | 多空词都命中或都不命中 | 两个方向都进检验，按两重自由度计入 FDR，判定上限 `exploratory` |

命中的具体词根会随产物输出（`matched`），便于人工复核词表，而不是悄悄放宽判据。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

STRONG_BULL = (
    "bull", "bullish", "buy", "long", "uptrend", "breakout", "breakout_up",
    "cross_up", "golden", "accumulation", "gap_up", "surge", "rise",
)
STRONG_BEAR = (
    "bear", "bearish", "sell", "short", "downtrend", "breakdown", "break_down",
    "cross_down", "death", "distribution", "gap_down", "dump", "fall",
)
# 语境词：多数 Pine 脚本里 oversold 是「超卖买入」、support 是「支撑看多」，
# 但这属于流派约定而不是词义本身，只能算弱证据。
WEAK_BULL = (
    "oversold", "reclaim", "spring", "sos", "lps", "ascending", "support",
    "demand", "above", "push_up", "up_break", "overhead_clear", "break_up",
)
WEAK_BEAR = (
    "overbought", "reject", "sow", "lpsy", "utad", "descending", "resistance",
    "supply", "below", "push_down", "down_break", "break_down_zone",
)
# 这些上下文里的 up/down/high/low/buy/sell 不表方向（HTF、摆动点、主动买卖量、档位高度）
NEGATE_PATTERNS = (
    r"higher[_ ]?time", r"lower[_ ]?time", r"buy[_ ]?(volume|quote)",
    r"sell[_ ]?(volume|quote)", r"taker[_ ]?buy", r"taker[_ ]?sell",
    r"high[_ ]?(vol|volatility|vwap|width|bar|band|line)",
    r"low[_ ]?(vol|volatility|vwap|width|bar|band|line)",
    r"swing[_ ]?(high|low)", r"pivot[_ ]?(high|low)",
    r"recent[_ ]?(high|low)", r"period[_ ]?(high|low)",
    r"day[_ ]?(high|low)", r"week[_ ]?(high|low)", r"month[_ ]?(high|low)",
    r"highest", r"lowest", r"high[_ ]or[_ ]low",
)

_SPLIT = re.compile(r"[^a-z0-9]+")


def _normalize(name: str) -> str:
    text = " ".join(t for t in _SPLIT.split(str(name).lower()) if t)
    for pat in NEGATE_PATTERNS:
        text = re.sub(pat, " ", text)
    # 下划线粘连写法（breakoutup / longsignal）拆不出词根时，用子串兜底
    return text


def _hits(text: str, tokens) -> list[str]:
    padded = f" {text} "
    found = [tok for tok in tokens if f" {tok} " in padded]
    if not found:
        glued = text.replace(" ", "")
        found = [tok for tok in tokens if len(tok) >= 5 and tok in glued]
    return found


def direction_of(column: str, indicator: str = "") -> dict:
    """返回 {direction, source, matched}。判定顺序：列名 -> 指标名。

    列名不带方向词时才看指标名（有些指标只有 `xxx_bull` / `xxx_bear` 两个入口，
    列名却叫 `signal`）。
    """
    text = _normalize(column)
    bull, bear = _hits(text, STRONG_BULL), _hits(text, STRONG_BEAR)
    if bull and not bear:
        return {"direction": 1, "source": "declared", "matched": bull}
    if bear and not bull:
        return {"direction": -1, "source": "declared", "matched": bear}
    if not (bull or bear):
        wb, wbr = _hits(text, WEAK_BULL), _hits(text, WEAK_BEAR)
        if wb and not wbr:
            return {"direction": 1, "source": "weak", "matched": wb}
        if wbr and not wb:
            return {"direction": -1, "source": "weak", "matched": wbr}
        it = _normalize(indicator)
        bull, bear = _hits(it, STRONG_BULL), _hits(it, STRONG_BEAR)
        if bull and not bear:
            return {"direction": 1, "source": "declared", "matched": bull, "from": "indicator"}
        if bear and not bull:
            return {"direction": -1, "source": "declared", "matched": bear, "from": "indicator"}
        wb, wbr = _hits(it, WEAK_BULL), _hits(it, WEAK_BEAR)
        if wb and not wbr:
            return {"direction": 1, "source": "weak", "matched": wb, "from": "indicator"}
        if wbr and not wb:
            return {"direction": -1, "source": "weak", "matched": wbr, "from": "indicator"}
    return {"direction": 0, "source": "ambiguous", "matched": bull + bear}


def build_column_index(sample_columns: dict[str, list[str]]) -> dict:
    """把 {指标: [布尔列名]} 展开成带方向与稳定序号的全局列表。

    序号按 (indicator, column) 字典序固定，事件聚合数组按它寻址；探测阶段换币种样本
    只要列集合单调增加，老序号不变，缓存仍可复用。
    """
    entries = []
    for ind in sorted(sample_columns or {}):
        for col in sorted(set(sample_columns[ind])):
            info = direction_of(col, ind)
            entries.append(
                {
                    "index": len(entries),
                    "indicator": ind,
                    "column": col,
                    "key": f"{ind}|{col}",
                    **info,
                }
            )
    return {
        "count": len(entries),
        "by_source": {
            k: sum(1 for e in entries if e["source"] == k) for k in ("declared", "weak", "ambiguous")
        },
        "by_direction": {
            "bull": sum(1 for e in entries if e["direction"] == 1),
            "bear": sum(1 for e in entries if e["direction"] == -1),
            "none": sum(1 for e in entries if e["direction"] == 0),
        },
        "columns": entries,
    }


def index_lookup(index: dict) -> dict[str, dict]:
    return {e["key"]: e for e in index["columns"]}


def write_index(index: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

"""币种名单基础数据：主流币白名单与全局黑名单。

历史版本还负责「策略 → 允许交易的币种」映射（`symbol_allowed`），那是单策略时代的
运行范围门控。引入「策略 × 币种池」绑定后，运行范围由池（`strategy/pools.py`）表达，
本模块只保留两份基础名单供池规则引用：

- `MAJOR_SYMBOLS`：长期成交额居前、分钟数据连续、深度充足的白名单（池 `base=major`）
- `BLACKLIST`：回测反复为负、不参与任何策略的币种（任何池成员都会被剔除）
"""
from __future__ import annotations

# 吸收反转依赖 volume 与影线结构：中小币种的量价噪声会让 c1（|delta| 进入滚动
# 95 分位）与 c5（单边支配度）失真，因此主流池钉在这些高流动性合约上。
MAJOR_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT", "SUIUSDT",
    "TRXUSDT", "NEARUSDT", "APTUSDT", "BCHUSDT",
)

# 拉黑来源：2026-10-27 walk-forward 8×60d，最优策略 mean ≤ -0.1% 且 8/8 窗口无正收益。
BLACKLIST: set[str] = {
    "PAXGUSDT",   # Commodity, strong_pullback 0/8 窗口全负 mean=-0.19%
    "XAUTUSDT",   # Commodity, strong_pullback 0/8 窗口全负 mean=-0.19%
    "BANKUSDT",   # Alt, strong_pullback 0/8 窗口全负 mean=-0.27%
    "KATUSDT",    # Alt, strong_pullback 0/8 窗口全负 mean=-0.35%
    "IRYSUSDT",   # Alt, strong_pullback 0/8 窗口全负 mean=-0.50%
}

MAJOR_SET: frozenset[str] = frozenset(MAJOR_SYMBOLS)


def is_blacklisted(symbol: str) -> bool:
    """黑名单判定：命中则任何策略都不该在该币上产生信号。"""
    return symbol.upper() in BLACKLIST

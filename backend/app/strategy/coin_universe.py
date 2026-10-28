"""策略级币种宇宙：限定「某个策略只在哪些币上运行」+ 拉黑清单。

历史上这里还承担「每个币选最优策略」的路由职责（symbol→strategy 查表）。仓库收敛到
单一策略（吸收反转）后，没有第二个策略可路由，那层查表连同路由表一起删除；
保留下来的只有运行范围约束与黑名单——它们直接决定信号能不能产生。

- MAJOR_SYMBOLS / STRATEGY_UNIVERSE：策略 → 允许交易的币种白名单。
- BLACKLIST：历史回测反复为负、不参与任何策略的币种。
"""
from __future__ import annotations

# 吸收反转依赖 volume 与影线结构：中小币种的量价噪声会让 c1（|delta| 进入滚动
# 95 分位）与 c5（单边支配度）失真，因此把运行范围钉在高流动性主流币上。
# 主流币 = 长期成交额居前、分钟数据连续、且在 U 本位永续上深度充足的白名单。
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

STRATEGY_UNIVERSE: dict[str, frozenset[str]] = {
    "absorption_reversal": frozenset(MAJOR_SYMBOLS),
}


def symbol_allowed(code: str, symbol: str) -> bool:
    """策略 → 币种的运行范围检查：未登记宇宙的策略不限制（返回 True）。"""
    universe = STRATEGY_UNIVERSE.get(code)
    return universe is None or symbol.upper() in universe


def is_blacklisted(symbol: str) -> bool:
    """黑名单判定：命中则任何策略都不该在该币上产生信号。"""
    return symbol.upper() in BLACKLIST
